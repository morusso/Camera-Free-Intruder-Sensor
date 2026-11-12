/*
 * Wi-Fi + SNTP + MQTT over TLS (mutual authentication).
 *
 * Topics (id = CONFIG_CSI_NODE_ID):
 *   sensors/<id>/event    JSON, QoS 1: alarm_start, alarm_end, calib_done, calib_failed, link_lost, link_restored
 *   sensors/<id>/status   JSON, QoS 1, retained: state of the device; online:0 is the last will
 *                         (flags are 0/1, not true/false: Telegraf's JSON parser drops booleans)
 *   sensors/<id>/metrics  JSON, QoS 0, every CONFIG_CSI_MQTT_METRICS_INTERVAL_S seconds
 *   sensors/<id>/cmd      subscribed: "calibrate" or "calibrate:<seconds>"
 * Only classes and metrics leave the device, never raw CSI.
 */
#include "sdkconfig.h"

#ifdef CONFIG_CSI_RX_MQTT

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_netif_sntp.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "mqtt_client.h"

#include "node.h"
#include "node_mqtt.h"

#define TAG "node_mqtt"
#define ID CONFIG_CSI_NODE_ID
#define T_EVENT "sensors/" ID "/event"
#define T_STATUS "sensors/" ID "/status"
#define T_METRICS "sensors/" ID "/metrics"
#define T_CMD "sensors/" ID "/cmd"

extern const char ca_pem_start[] asm("_binary_ca_pem_start");
extern const char client_crt_start[] asm("_binary_client_crt_start");
extern const char client_key_start[] asm("_binary_client_key_start");

static esp_mqtt_client_handle_t s_client;
static volatile int s_mqtt_up;

static const char *EVENT_NAMES[] = {"alarm_start", "alarm_end", "calib_done", "calib_failed", "link_lost", "link_restored"};

static void publish_status(void)
{
    if (!s_mqtt_up) {
        return;
    }
    node_stats_t s;
    detector_get_stats(&s);
    wifi_ap_record_t ap;
    const int rssi = esp_wifi_sta_get_ap_info(&ap) == ESP_OK ? ap.rssi : 0;
    char buf[256];
    int n = snprintf(buf, sizeof(buf),
                     "{\"v\":1,\"device\":\"%s\",\"link\":\"%s\",\"online\":1,\"state\":\"%s\",\"calib_pct\":%u,"
                     "\"alarm\":%d,\"csi_link\":%d,\"wifi_rssi\":%d,\"uptime_s\":%lu}",
                     ID, CONFIG_CSI_NODE_LINK_ID, detector_state_name(s.state), s.calib_pct,
                     s.alarm_active ? 1 : 0, s.link_up ? 1 : 0, rssi,
                     (unsigned long)(esp_timer_get_time() / 1000000));
    esp_mqtt_client_publish(s_client, T_STATUS, buf, n, 1, 1);
}

static void publish_event(const node_event_t *ev)
{
    char buf[256];
    int n = snprintf(buf, sizeof(buf),
                     "{\"v\":1,\"device\":\"%s\",\"link\":\"%s\",\"event\":\"%s\",\"ts_ms\":%lld,\"uptime_ms\":%lu,"
                     "\"class\":\"%s\",\"confidence\":%.3f,\"p_person\":%.3f}",
                     ID, CONFIG_CSI_NODE_LINK_ID, EVENT_NAMES[ev->kind], (long long)ev->ts_ms,
                     (unsigned long)ev->uptime_ms, detector_class_name(ev->cls), ev->confidence, ev->p_person);
    /* QoS 1: the broker must acknowledge. The client library queues messages while the connection is down. */
    esp_mqtt_client_enqueue(s_client, T_EVENT, buf, n, 1, 0, true);
}

static void publish_metrics(void)
{
    if (!s_mqtt_up) {
        return;
    }
    node_stats_t s;
    detector_get_stats(&s);
    char buf[384];
    int n = snprintf(buf, sizeof(buf),
                     "{\"v\":1,\"device\":\"%s\",\"link\":\"%s\",\"frames\":%lu,\"windows\":%lu,\"alarms\":%lu,"
                     "\"gaps\":%lu,\"queue_drops\":%lu,\"infer_us_avg\":%lu,\"infer_us_max\":%lu,"
                     "\"p_person\":%.3f,\"energy\":%.4f,\"energy_avg\":%.4f,\"heap_free\":%lu,\"heap_min\":%lu}",
                     ID, CONFIG_CSI_NODE_LINK_ID, (unsigned long)s.frames, (unsigned long)s.windows,
                     (unsigned long)s.alarms, (unsigned long)s.gaps, (unsigned long)s.queue_drops,
                     (unsigned long)s.infer_us_avg, (unsigned long)s.infer_us_max, s.p_person_last, s.energy_last,
                     s.energy_avg, (unsigned long)esp_get_free_heap_size(),
                     (unsigned long)esp_get_minimum_free_heap_size());
    esp_mqtt_client_publish(s_client, T_METRICS, buf, n, 0, 0);
}

static void handle_command(const char *data, int len)
{
    char cmd[32];
    if (len <= 0 || len >= (int)sizeof(cmd)) {
        return;
    }
    memcpy(cmd, data, len);
    cmd[len] = 0;
    if (strncmp(cmd, "calibrate", 9) == 0) {
        uint32_t secs = 0;
        if (cmd[9] == ':') {
            secs = (uint32_t)atoi(cmd + 10);
            if (secs < 10 || secs > 3600) {
                ESP_LOGW(TAG, "calibration length out of range (10..3600 s): ignored");
                return;
            }
        } else if (cmd[9] != 0) {
            return;
        }
        ESP_LOGW(TAG, "calibration requested over MQTT");
        detector_request_calibration(secs);
    } else {
        ESP_LOGW(TAG, "unknown command ignored");
    }
}

static void mqtt_event_handler(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    esp_mqtt_event_handle_t e = data;
    switch ((esp_mqtt_event_id_t)id) {
    case MQTT_EVENT_CONNECTED:
        s_mqtt_up = 1;
        esp_mqtt_client_subscribe(s_client, T_CMD, 1);
        publish_status();
        ESP_LOGW(TAG, "MQTT connected");
        break;
    case MQTT_EVENT_DISCONNECTED:
        s_mqtt_up = 0;
        break;
    case MQTT_EVENT_DATA:
        handle_command(e->data, e->data_len);
        break;
    case MQTT_EVENT_ERROR:
        ESP_LOGW(TAG, "MQTT error (type %d)", e->error_handle ? e->error_handle->error_type : -1);
        break;
    default:
        break;
    }
}

static void wifi_event_handler(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();  /* keep trying; events are queued by the MQTT client meanwhile */
    }
}

static void node_task(void *arg)
{
    TickType_t last_metrics = xTaskGetTickCount(), last_status = last_metrics;
    for (;;) {
        node_event_t ev;
        if (detector_poll_event(&ev, 500)) {
            publish_event(&ev);
            publish_status();
        }
        const TickType_t now = xTaskGetTickCount();
        if (now - last_metrics >= pdMS_TO_TICKS(CONFIG_CSI_MQTT_METRICS_INTERVAL_S * 1000)) {
            last_metrics = now;
            publish_metrics();
        }
        if (now - last_status >= pdMS_TO_TICKS(30000)) {  /* the state can change without an event (calibration progress) */
            last_status = now;
            publish_status();
        }
    }
}

void node_mqtt_start(void)
{
    /* wifi_init() in rx_main.c has already started Wi-Fi in STA mode; add the connection here. */
    wifi_config_t wc = {0};
    strlcpy((char *)wc.sta.ssid, CONFIG_CSI_WIFI_SSID, sizeof(wc.sta.ssid));
    strlcpy((char *)wc.sta.password, CONFIG_CSI_WIFI_PASSWORD, sizeof(wc.sta.password));
    wc.sta.threshold.authmode = strlen(CONFIG_CSI_WIFI_PASSWORD) ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_connect());

    esp_sntp_config_t sntp = ESP_NETIF_SNTP_DEFAULT_CONFIG(CONFIG_CSI_SNTP_SERVER);
    ESP_ERROR_CHECK(esp_netif_sntp_init(&sntp));

    const esp_mqtt_client_config_t cfg = {
        .broker.address.uri = CONFIG_CSI_MQTT_URI,
        .broker.verification.certificate = ca_pem_start,
        .credentials.client_id = ID,
        .credentials.authentication.certificate = client_crt_start,
        .credentials.authentication.key = client_key_start,
        .session.last_will = {.topic = T_STATUS,
                              .msg = "{\"v\":1,\"device\":\"" ID "\",\"online\":0}",
                              .qos = 1,
                              .retain = 1},
        .session.keepalive = 30,
        .buffer.size = 1024,
    };
    s_client = esp_mqtt_client_init(&cfg);
    ESP_ERROR_CHECK(s_client ? ESP_OK : ESP_ERR_NO_MEM);
    ESP_ERROR_CHECK(esp_mqtt_client_register_event(s_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL));
    ESP_ERROR_CHECK(esp_mqtt_client_start(s_client));
    xTaskCreatePinnedToCore(node_task, "node_mqtt", 6144, NULL, 4, NULL, 1);
}

#endif /* CONFIG_CSI_RX_MQTT */
