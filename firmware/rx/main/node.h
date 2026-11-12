/* Types shared by the detector task and the MQTT client (rx/main). */
#pragma once

#include <stdint.h>

typedef enum {
    NODE_EV_ALARM_START,
    NODE_EV_ALARM_END,
    NODE_EV_CALIB_DONE,
    NODE_EV_CALIB_FAILED,
    NODE_EV_LINK_LOST,
    NODE_EV_LINK_RESTORED,
} node_event_kind_t;

typedef struct {
    node_event_kind_t kind;
    int64_t ts_ms;        /* unix time in ms, 0 while the clock is not synchronized */
    uint32_t uptime_ms;
    uint8_t cls;          /* index of the most likely class (alarm events) */
    float confidence;     /* its probability */
    float p_person;
} node_event_t;

typedef struct {
    uint8_t state;        /* csi_det_state_t */
    uint8_t calib_pct;
    uint8_t alarm_active;
    uint8_t link_up;
    uint32_t frames;      /* frames fed to the detector */
    uint32_t windows;     /* windows evaluated */
    uint32_t alarms;      /* alarm_start events */
    uint32_t infer_us_avg;
    uint32_t infer_us_max;
    uint32_t gaps;        /* stream resets because of lost frames */
    uint32_t queue_drops; /* frames dropped before the detector */
    float p_person_last;
    float energy_last;    /* feature for drift monitoring */
    float energy_avg;     /* exponential average of the above */
} node_stats_t;

/* Detector API (detector_task.c). */
void detector_start(void);
/* Called from the Wi-Fi task: never blocks. Returns 0 if the frame was queued. */
int detector_feed(uint32_t rx_idx, const int8_t *csi, uint16_t len);
/* Thread-safe: asks the detector task to calibrate for `seconds` (0 = Kconfig default). */
void detector_request_calibration(uint32_t seconds);
void detector_get_stats(node_stats_t *out);
const char *detector_class_name(uint8_t cls);
const char *detector_state_name(uint8_t state);
/* Pops one event; returns 1 if there was one. */
int detector_poll_event(node_event_t *ev, uint32_t timeout_ms);
