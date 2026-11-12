/*
 * CSI receiver.
 *
 * Core 0 (Wi-Fi): the CSI callback only copies the data into a queue.
 * Core 1: the output task formats each frame as one text line (docs/format.md)
 *         and writes it to the console UART (CONFIG_CSI_RX_STREAM_UART); the detector task runs
 *         preprocessing, the model and the alarm logic (CONFIG_CSI_RX_DETECTOR, detector_task.c);
 *         with CONFIG_CSI_RX_MQTT the events go to the broker (node_mqtt.c).
 */
#include <stdio.h>
#include <string.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#include "csi_link.h"
#include "csi_model.h"
#include "node.h"
#include "node_mqtt.h"
#include "preprocess_params.h"

/* A bool Kconfig option that is "n" is not defined at all, so test it with #ifdef. */
#ifdef CONFIG_CSI_RX_USE_HTLTF
#define USE_HTLTF 1
#else
#define USE_HTLTF 0
#endif

#define CSI_MAX_LEN 384 /* LLTF + HT-LTF + STBC HT-LTF2 for HT20 */
#define OUT_TASK_PRIORITY 5
#define LINE_BUF_SIZE (160 + CSI_MAX_LEN * 5)

static const char *TAG = "csi_rx";
static const uint8_t TX_MAC[6] = CSI_LINK_TX_MAC;

typedef struct {
    uint32_t rx_idx;
    uint32_t tx_seq;
    uint32_t timestamp;
    uint16_t sig_len;
    uint16_t len;
    int8_t rssi;
    int8_t noise_floor;
    uint8_t rate;
    uint8_t channel;
    uint8_t rx_format;
    uint8_t first_word;
    uint8_t tx_seq_valid;
    int8_t csi[CSI_MAX_LEN];
} csi_item_t;

#ifdef CONFIG_CSI_RX_STREAM_UART
static QueueHandle_t s_queue;
#endif
/* Written only by the Wi-Fi task, read by the stats printer: plain 32-bit accesses. */
static volatile uint32_t s_rx_idx;
static volatile uint32_t s_queue_drops;
static volatile uint32_t s_oversize;

/* Runs in the Wi-Fi task. Must be fast and must never block. */
static void csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf || memcmp(info->mac, TX_MAC, 6) != 0) {
        return;
    }
    /* rx_idx counts every frame from our TX, including ones dropped below,
     * so drops show up as gaps in rx_idx. */
    const uint32_t idx = s_rx_idx++;

    if (info->len > CSI_MAX_LEN) {
        s_oversize++;
        return;
    }

#ifdef CONFIG_CSI_RX_DETECTOR
    detector_feed(idx, (const int8_t *)info->buf, info->len);
#endif
#ifdef CONFIG_CSI_RX_STREAM_UART
    csi_item_t item;
    item.rx_idx = idx;
    item.tx_seq = 0;
    item.tx_seq_valid = info->payload ? csi_link_find_seq(info->payload, info->payload_len, &item.tx_seq) : 0;
    item.timestamp = info->rx_ctrl.timestamp;
    item.sig_len = info->rx_ctrl.sig_len;
    item.len = info->len;
    item.rssi = info->rx_ctrl.rssi;
    item.noise_floor = info->rx_ctrl.noise_floor;
    item.rate = info->rx_ctrl.rate;
    item.channel = info->rx_ctrl.channel;
    item.rx_format = info->rx_ctrl.sig_mode;
    item.first_word = info->first_word_invalid ? 1 : 0;
    memcpy(item.csi, info->buf, info->len);

    if (xQueueSend(s_queue, &item, 0) != pdTRUE) {
        s_queue_drops++;
    }
#endif
}

#ifdef CONFIG_CSI_RX_STREAM_UART
static void output_task(void *arg)
{
    static char line[LINE_BUF_SIZE];
    csi_item_t item;

    for (;;) {
        if (xQueueReceive(s_queue, &item, portMAX_DELAY) == pdTRUE) {
            int n = snprintf(line, sizeof(line), "CSI_DATA,%lu,", (unsigned long)item.rx_idx);
            if (item.tx_seq_valid) {
                n += snprintf(line + n, sizeof(line) - n, "%lu,", (unsigned long)item.tx_seq);
            } else {
                n += snprintf(line + n, sizeof(line) - n, "-1,");
            }
            n += snprintf(line + n, sizeof(line) - n, MACSTR ",%d,%d,%d,%d,%lu,%u,%u,%u,%u,\"[",
                          MAC2STR(TX_MAC), item.rssi, item.rate, item.noise_floor, item.channel,
                          (unsigned long)item.timestamp, item.sig_len, item.rx_format, item.len,
                          item.first_word);
            for (int i = 0; i < item.len; i++) {
                n += snprintf(line + n, sizeof(line) - n, i ? ",%d" : "%d", item.csi[i]);
            }
            n += snprintf(line + n, sizeof(line) - n, "]\"\n");
            fwrite(line, 1, n, stdout);
        }
    }
}
#endif

static void stats_task(void *arg)
{
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(CONFIG_CSI_RX_STATS_INTERVAL_S * 1000));
        printf("# STATS frames=%lu queue_drops=%lu oversize=%lu free_heap=%lu min_heap=%lu stack_free=%u\n",
               (unsigned long)s_rx_idx, (unsigned long)s_queue_drops, (unsigned long)s_oversize,
               (unsigned long)esp_get_free_heap_size(), (unsigned long)esp_get_minimum_free_heap_size(),
               (unsigned)uxTaskGetStackHighWaterMark(NULL));
#ifdef CONFIG_CSI_RX_DETECTOR
        node_stats_t d;
        detector_get_stats(&d);
        printf("# DET state=%s calib=%u%% link=%u alarm=%u frames=%lu windows=%lu alarms=%lu gaps=%lu "
               "drops=%lu infer_us_avg=%lu infer_us_max=%lu p_person=%.2f energy=%.3f\n",
               detector_state_name(d.state), d.calib_pct, d.link_up, d.alarm_active, (unsigned long)d.frames,
               (unsigned long)d.windows, (unsigned long)d.alarms, (unsigned long)d.gaps,
               (unsigned long)d.queue_drops, (unsigned long)d.infer_us_avg, (unsigned long)d.infer_us_max,
               d.p_person_last, d.energy_last);
#endif
    }
}

static void wifi_init(void)
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_CSI_LINK_CHANNEL, WIFI_SECOND_CHAN_NONE));
}

static void csi_init(void)
{
    /* Promiscuous mode lets the radio pass the broadcast ESP-NOW frames to the CSI engine. */
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = USE_HTLTF,
        .stbc_htltf2_en = USE_HTLTF,
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = false,
        .shift = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

/* Boot check of the classifier: golden output, arena use and the time of one inference (steps 4.2/4.3). */
static void model_selftest(void)
{
    esp_err_t err = csi_model_init();
    if (err != ESP_OK) {
        printf("# MODEL init failed: %s\n", esp_err_to_name(err));
        return;
    }
    const int diff = csi_model_selftest();

    static float window[128 * CSI_PP_N_KEEP];  /* zeros: the empty-room baseline */
    float probs[8] = {0};
    uint32_t us = 0;
    if (csi_model_input_len() != sizeof(window) / sizeof(float) || csi_model_n_classes() > 8) {
        printf("# MODEL unexpected shape: input=%u classes=%u\n", (unsigned)csi_model_input_len(),
               (unsigned)csi_model_n_classes());
        return;
    }
    csi_model_infer(window, probs, &us);
    printf("# MODEL ok: input=%u classes=%u arena_used=%u golden_max_diff=%d infer_us=%lu free_heap=%lu\n",
           (unsigned)csi_model_input_len(), (unsigned)csi_model_n_classes(), (unsigned)csi_model_arena_used(),
           diff, (unsigned long)us, (unsigned long)esp_get_free_heap_size());
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

#ifdef CONFIG_CSI_RX_STREAM_UART
    s_queue = xQueueCreate(CONFIG_CSI_RX_QUEUE_LEN, sizeof(csi_item_t));
    ESP_ERROR_CHECK(s_queue ? ESP_OK : ESP_ERR_NO_MEM);

    /* Core 1: formatting and serial output, away from the Wi-Fi task on core 0. */
    xTaskCreatePinnedToCore(output_task, "csi_out", 6144, NULL, OUT_TASK_PRIORITY, NULL, 1);
#endif
    xTaskCreatePinnedToCore(stats_task, "stats", 3072, NULL, 1, NULL, 1);

    printf("# CSI_FMT,1,rx_idx,tx_seq,mac,rssi,rate,noise_floor,channel,local_timestamp,"
           "sig_len,rx_format,len,first_word,data\n");
    printf("# CSI RX: channel=%d htltf=%d tx_mac=" MACSTR "\n",
           CONFIG_CSI_LINK_CHANNEL, USE_HTLTF, MAC2STR(TX_MAC));

    model_selftest();

#ifdef CONFIG_CSI_RX_DETECTOR
    detector_start();
#endif
    wifi_init();
    csi_init();
#ifdef CONFIG_CSI_RX_MQTT
    node_mqtt_start();
#endif
    ESP_LOGW(TAG, "CSI capture started");
}
