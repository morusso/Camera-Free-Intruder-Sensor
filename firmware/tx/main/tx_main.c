/*
 * CSI transmitter: sends tiny ESP-NOW broadcast frames at a fixed rate.
 * The receiver measures CSI on every one of them.
 */
#include <string.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#include "csi_link.h"

#define TX_FREQUENCY_HZ CONFIG_CSI_TX_FREQUENCY_HZ
#define TX_PERIOD_TICKS (CONFIG_FREERTOS_HZ / TX_FREQUENCY_HZ)
#define TX_TASK_PRIORITY 10
#define STATS_INTERVAL_S 10

_Static_assert(CONFIG_FREERTOS_HZ % CONFIG_CSI_TX_FREQUENCY_HZ == 0,
               "CSI_TX_FREQUENCY_HZ must divide FREERTOS_HZ evenly");

static const char *TAG = "csi_tx";
static const uint8_t TX_MAC[6] = CSI_LINK_TX_MAC;

static void wifi_init(void)
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20));
    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, TX_MAC));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));  /* power save adds jitter */
    ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_CSI_LINK_CHANNEL, WIFI_SECOND_CHAN_NONE));
}

static void esp_now_setup(const uint8_t *peer_addr)
{
    ESP_ERROR_CHECK(esp_now_init());

    esp_now_peer_info_t peer = {
        .channel = CONFIG_CSI_LINK_CHANNEL,
        .ifidx = WIFI_IF_STA,
        .encrypt = false,
    };
    memcpy(peer.peer_addr, peer_addr, ESP_NOW_ETH_ALEN);
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));

    esp_now_rate_config_t rate = {
        .phymode = WIFI_PHY_MODE_HT20,
        .rate = WIFI_PHY_RATE_MCS0_LGI,
        .ersu = false,
        .dcm = false,
    };
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer_addr, &rate));
}

static void tx_task(void *arg)
{
    static const uint8_t broadcast[ESP_NOW_ETH_ALEN] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    csi_link_payload_t payload = {.magic = CSI_LINK_MAGIC, .seq = 0};
    uint32_t send_errors = 0;
    uint32_t sent_in_interval = 0;
    TickType_t last_wake = xTaskGetTickCount();

    for (;;) {
        if (esp_now_send(broadcast, (const uint8_t *)&payload, sizeof(payload)) != ESP_OK) {
            send_errors++;
        }
        payload.seq++;
        sent_in_interval++;

        if (sent_in_interval >= STATS_INTERVAL_S * TX_FREQUENCY_HZ) {
            ESP_LOGI(TAG, "seq=%lu send_errors=%lu free_heap=%lu stack_free=%u",
                     (unsigned long)payload.seq, (unsigned long)send_errors,
                     (unsigned long)esp_get_free_heap_size(),
                     (unsigned)uxTaskGetStackHighWaterMark(NULL));
            sent_in_interval = 0;
        }
        /* Absolute delay: no drift, unlike a plain delay after each send. */
        vTaskDelayUntil(&last_wake, TX_PERIOD_TICKS);
    }
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    wifi_init();
    static const uint8_t broadcast[ESP_NOW_ETH_ALEN] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    esp_now_setup(broadcast);

    ESP_LOGI(TAG, "CSI TX: channel=%d rate=%d Hz mac=" MACSTR,
             CONFIG_CSI_LINK_CHANNEL, TX_FREQUENCY_HZ, MAC2STR(TX_MAC));

    xTaskCreate(tx_task, "csi_tx", 4096, NULL, TX_TASK_PRIORITY, NULL);
}
