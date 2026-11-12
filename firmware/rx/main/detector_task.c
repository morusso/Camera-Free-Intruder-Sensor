/*
 * Detector task (core 1): frames from the CSI callback -> csi_detector (preprocess, model, alarm) -> events.
 *
 * Also owns the calibration: the baseline is stored in NVS, loaded at boot, and learned again on a button hold
 * (2 s), an MQTT command or detector_request_calibration().
 */
#include "sdkconfig.h"

#ifdef CONFIG_CSI_RX_DETECTOR

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "driver/gpio.h"
#include "esp_crc.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "nvs.h"

#include "csi_detector.h"
#include "csi_model.h"
#include "node.h"

#define TAG "detector"
#define FRAME_QUEUE_LEN 64
#define EVENT_QUEUE_LEN 16
#define GAP_RESET_FRAMES 10      /* more lost frames than this in a row restarts the stream */
#define BUTTON_HOLD_MS 2000
#define NVS_NS "csi"
#define NVS_KEY "baseline"
#define BASELINE_MAGIC 0x43534231u  /* "CSB1" */

typedef struct {
    uint32_t rx_idx;
    int8_t csi[CSI_PP_N_VALUES];
} frame_t;

typedef struct {
    uint32_t magic;
    uint32_t n_keep;
    csi_pp_baseline_t bl;
    uint32_t crc;
} stored_baseline_t;

static const char *CLASS_NAMES[] = {"empty", "static_person", "walking", "interference"};
static const char *STATE_NAMES[] = {"uncalibrated", "calibrating", "running"};

static QueueHandle_t s_frames;
static QueueHandle_t s_events;
static TaskHandle_t s_task;
static csi_det_t s_det;           /* about 55 KB, kept out of the stack */
static node_stats_t s_stats;
static volatile uint32_t s_calib_request_s;   /* 0 = none; the Kconfig default is applied by the requester */
static volatile uint32_t s_queue_drops;
static uint64_t s_infer_us_sum;

const char *detector_class_name(uint8_t cls) { return cls < 4 ? CLASS_NAMES[cls] : "unknown"; }
const char *detector_state_name(uint8_t state) { return state < 3 ? STATE_NAMES[state] : "unknown"; }

static int model_infer(void *ctx, const float *window, float *probs)
{
    (void)ctx;
    uint32_t us = 0;
    if (csi_model_infer(window, probs, &us) != ESP_OK) {
        return -1;
    }
    s_stats.infer_us_max = us > s_stats.infer_us_max ? us : s_stats.infer_us_max;
    s_infer_us_sum += us;
    return (int)csi_model_n_classes();
}

static int64_t now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    /* Before SNTP the clock starts at 1970 + uptime; report 0 in that case. */
    return tv.tv_sec < 1700000000 ? 0 : (int64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static void emit(node_event_kind_t kind, const csi_det_result_t *r)
{
    node_event_t ev = {.kind = kind, .ts_ms = now_ms(), .uptime_ms = (uint32_t)(esp_timer_get_time() / 1000)};
    if (r && r->n_classes) {
        uint8_t best = 0;
        for (uint8_t i = 1; i < r->n_classes; i++) {
            best = r->probs[i] > r->probs[best] ? i : best;
        }
        ev.cls = best;
        ev.confidence = r->probs[best];
        ev.p_person = r->p_person;
    }
#ifdef CONFIG_CSI_RX_MQTT  /* without MQTT nobody reads the queue; the console line below is the output */
    if (xQueueSend(s_events, &ev, 0) != pdTRUE) {
        ESP_LOGW(TAG, "event queue full, dropping event %d", (int)kind);
    }
#endif
    printf("# EVENT kind=%d ts_ms=%lld class=%s conf=%.2f p_person=%.2f\n", (int)kind, (long long)ev.ts_ms,
           detector_class_name(ev.cls), ev.confidence, ev.p_person);
}

/* ---- baseline in NVS --------------------------------------------------------------------------------------- */

static uint32_t baseline_crc(const stored_baseline_t *s)
{
    return esp_crc32_le(0, (const uint8_t *)s, offsetof(stored_baseline_t, crc));
}

static int baseline_load(csi_pp_baseline_t *bl)
{
    nvs_handle_t h;
    if (nvs_open(NVS_NS, NVS_READONLY, &h) != ESP_OK) {
        return -1;
    }
    stored_baseline_t s;
    size_t len = sizeof(s);
    int ok = nvs_get_blob(h, NVS_KEY, &s, &len) == ESP_OK && len == sizeof(s) && s.magic == BASELINE_MAGIC &&
             s.n_keep == CSI_PP_N_KEEP && s.crc == baseline_crc(&s);
    nvs_close(h);
    if (ok) {
        *bl = s.bl;
    }
    return ok ? 0 : -1;
}

static void baseline_save(const csi_pp_baseline_t *bl)
{
    stored_baseline_t s = {.magic = BASELINE_MAGIC, .n_keep = CSI_PP_N_KEEP, .bl = *bl};
    s.crc = baseline_crc(&s);
    nvs_handle_t h;
    if (nvs_open(NVS_NS, NVS_READWRITE, &h) != ESP_OK) {
        ESP_LOGE(TAG, "cannot open NVS to store the baseline");
        return;
    }
    if (nvs_set_blob(h, NVS_KEY, &s, sizeof(s)) != ESP_OK || nvs_commit(h) != ESP_OK) {
        ESP_LOGE(TAG, "cannot store the baseline");
    }
    nvs_close(h);
}

/* ---- LED and button ---------------------------------------------------------------------------------------- */

static void io_init(void)
{
#if CONFIG_CSI_DET_BUTTON_GPIO >= 0
    gpio_config_t b = {.pin_bit_mask = 1ULL << CONFIG_CSI_DET_BUTTON_GPIO, .mode = GPIO_MODE_INPUT,
                       .pull_up_en = GPIO_PULLUP_ENABLE};
    gpio_config(&b);
#endif
#if CONFIG_CSI_DET_LED_GPIO >= 0
    gpio_config_t l = {.pin_bit_mask = 1ULL << CONFIG_CSI_DET_LED_GPIO, .mode = GPIO_MODE_OUTPUT};
    gpio_config(&l);
#endif
}

static void led_update(const csi_det_t *d, uint32_t now_ms_u)
{
#if CONFIG_CSI_DET_LED_GPIO >= 0
    int on = 0;
    if (d->state == CSI_DET_CALIBRATING) {
        on = 1;
    } else if (d->state == CSI_DET_UNCALIBRATED) {
        on = (now_ms_u / 150) & 1;
    } else if (csi_alarm_active(&d->alarm)) {
        on = (now_ms_u / 50) & 1;
    }
    gpio_set_level(CONFIG_CSI_DET_LED_GPIO, on);
#else
    (void)d;
    (void)now_ms_u;
#endif
}

/* Returns 1 once when the button has been held for BUTTON_HOLD_MS. */
static int button_held(uint32_t now_ms_u)
{
#if CONFIG_CSI_DET_BUTTON_GPIO >= 0
    static uint32_t since;
    static int fired;
    if (gpio_get_level(CONFIG_CSI_DET_BUTTON_GPIO) == 0) {
        if (!since) {
            since = now_ms_u ? now_ms_u : 1;
        }
        if (!fired && now_ms_u - since >= BUTTON_HOLD_MS) {
            fired = 1;
            return 1;
        }
    } else {
        since = 0;
        fired = 0;
    }
#else
    (void)now_ms_u;
#endif
    return 0;
}

/* ---- API --------------------------------------------------------------------------------------------------- */

int detector_feed(uint32_t rx_idx, const int8_t *csi, uint16_t len)
{
    if (!s_frames || len < CSI_PP_N_VALUES) {
        return -1;
    }
    frame_t f;
    f.rx_idx = rx_idx;
    memcpy(f.csi, csi, CSI_PP_N_VALUES);
    if (xQueueSend(s_frames, &f, 0) != pdTRUE) {
        s_queue_drops++;
        return -1;
    }
    return 0;
}

void detector_request_calibration(uint32_t seconds)
{
    s_calib_request_s = seconds ? seconds : CONFIG_CSI_DET_CALIB_SECONDS;
    if (s_task) {
        xTaskNotifyGive(s_task);
    }
}

void detector_get_stats(node_stats_t *out)
{
    *out = s_stats;
    out->queue_drops = s_queue_drops;
}

int detector_poll_event(node_event_t *ev, uint32_t timeout_ms)
{
    return s_events && xQueueReceive(s_events, ev, pdMS_TO_TICKS(timeout_ms)) == pdTRUE;
}

/* ---- the task ---------------------------------------------------------------------------------------------- */

static void publish_stats(const csi_det_t *d, const csi_det_result_t *r, int link_up)
{
    s_stats.state = d->state;
    s_stats.calib_pct = (uint8_t)csi_det_calib_percent(d);
    s_stats.alarm_active = (uint8_t)csi_alarm_active(&d->alarm);
    s_stats.link_up = (uint8_t)link_up;
    if (r && r->kind == CSI_DET_WINDOW && !r->infer_error) {
        s_stats.windows++;
        s_stats.infer_us_avg = (uint32_t)(s_infer_us_sum / s_stats.windows);
        s_stats.p_person_last = r->p_person;
        s_stats.energy_last = r->energy;
        s_stats.energy_avg = s_stats.windows == 1 ? r->energy : 0.98f * s_stats.energy_avg + 0.02f * r->energy;
    }
}

static void detector_task(void *arg)
{
    (void)arg;
    csi_det_result_t r;
    frame_t f;
    int link_up = 0, have_prev = 0;
    uint32_t prev_idx = 0, last_frame_ms = 0;

    csi_pp_baseline_t bl;
    if (baseline_load(&bl) == 0) {
        csi_det_set_baseline(&s_det, &bl);
        ESP_LOGW(TAG, "baseline loaded from NVS");
    } else {
        ESP_LOGW(TAG, "no baseline: hold the button for 2 s (or send the calibrate command) with an empty room");
    }

    for (;;) {
        const uint32_t t = (uint32_t)(esp_timer_get_time() / 1000);
        led_update(&s_det, t);

        const uint32_t req = s_calib_request_s;
        if (req || button_held(t)) {
            s_calib_request_s = 0;
            const uint32_t secs = req ? req : CONFIG_CSI_DET_CALIB_SECONDS;
            csi_det_calibrate_begin(&s_det, secs * CONFIG_CSI_DET_SAMPLE_RATE_HZ);
            ESP_LOGW(TAG, "calibrating for %lu s: keep the room empty", (unsigned long)secs);
        }

        if (xQueueReceive(s_frames, &f, pdMS_TO_TICKS(100)) != pdTRUE) {
            if (link_up && t - last_frame_ms > CONFIG_CSI_DET_LINK_TIMEOUT_MS) {
                link_up = 0;
                csi_det_reset_stream(&s_det);
                have_prev = 0;
                emit(NODE_EV_LINK_LOST, NULL);
            }
            publish_stats(&s_det, NULL, link_up);
            continue;
        }
        last_frame_ms = t;
        if (!link_up) {
            link_up = 1;
            emit(NODE_EV_LINK_RESTORED, NULL);
        }
        if (have_prev && f.rx_idx - prev_idx > GAP_RESET_FRAMES) {
            /* Too many frames missing: the filter state and the window no longer describe a continuous signal. */
            s_stats.gaps++;
            csi_det_reset_stream(&s_det);
        }
        have_prev = 1;
        prev_idx = f.rx_idx;
        s_stats.frames++;

        csi_det_push(&s_det, f.csi, &r);
        switch (r.kind) {
        case CSI_DET_CALIB_DONE:
            baseline_save(&s_det.baseline);
            emit(NODE_EV_CALIB_DONE, NULL);
            break;
        case CSI_DET_CALIB_FAILED:
            emit(NODE_EV_CALIB_FAILED, NULL);
            break;
        case CSI_DET_WINDOW:
            if (r.alarm_event == CSI_ALARM_START) {
                s_stats.alarms++;
                emit(NODE_EV_ALARM_START, &r);
            } else if (r.alarm_event == CSI_ALARM_END) {
                emit(NODE_EV_ALARM_END, &r);
            }
            break;
        default:
            break;
        }
        publish_stats(&s_det, &r, link_up);
    }
}

void detector_start(void)
{
    s_frames = xQueueCreate(FRAME_QUEUE_LEN, sizeof(frame_t));
    s_events = xQueueCreate(EVENT_QUEUE_LEN, sizeof(node_event_t));
    configASSERT(s_frames && s_events);

    const csi_det_cfg_t cfg = {
        .alarm = {.on_threshold = CONFIG_CSI_DET_ALARM_ON_PCT / 100.0f,
                  .off_threshold = CONFIG_CSI_DET_ALARM_OFF_PCT / 100.0f,
                  .n = CONFIG_CSI_DET_ALARM_N,
                  .k_on = CONFIG_CSI_DET_ALARM_K_ON,
                  .k_off = CONFIG_CSI_DET_ALARM_K_OFF,
                  .min_hold_windows = CONFIG_CSI_DET_ALARM_MIN_HOLD},
        .stride = CONFIG_CSI_DET_STRIDE,
        .n_person = 2,
        .person_idx = {1, 2},  /* static_person, walking: order of ml/dataset.py */
        .max_calib_std_ratio = CONFIG_CSI_DET_MAX_CALIB_STD_X100 / 100.0f,
        .infer = model_infer,
    };
    if (csi_det_init(&s_det, &cfg) != 0) {
        ESP_LOGE(TAG, "bad detector configuration (check the alarm n / k / threshold values)");
        return;
    }
    io_init();
    xTaskCreatePinnedToCore(detector_task, "detector", 8192, NULL, 6, &s_task, 1);
}

#endif /* CONFIG_CSI_RX_DETECTOR */
