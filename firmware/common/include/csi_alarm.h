/*
 * Alarm decision on top of the classifier (step 4.5): k-of-n smoothing, hysteresis and a minimum event length.
 *
 * Feed one P(person) per window (P(static_person) + P(walking)) to csi_alarm_update().
 *
 *   - A window is "positive" if p > on_threshold while idle, or p > off_threshold while the alarm is active.
 *     With off_threshold < on_threshold a person near the threshold does not make the alarm flicker.
 *   - The alarm turns on when at least k_on of the last n windows are positive (same rule as k_of_n in
 *     ml/evaluate.py).
 *   - It turns off when fewer than k_off of the last n windows are positive, but not before min_hold_windows
 *     windows have passed since it turned on.
 *
 * With on_threshold == off_threshold, k_off == k_on and min_hold_windows == 0 the output equals k_of_n(), which
 * is what the host test checks. Plain C, no heap, no ESP-IDF dependencies.
 */
#pragma once

#include <stdint.h>

#define CSI_ALARM_MAX_N 16

typedef struct {
    float on_threshold;
    float off_threshold;
    uint8_t n;                 /* 1..CSI_ALARM_MAX_N */
    uint8_t k_on;              /* 1..n */
    uint8_t k_off;             /* 1..n; alarm drops when positives < k_off */
    uint16_t min_hold_windows; /* minimum length of an alarm, in windows */
} csi_alarm_cfg_t;

typedef enum {
    CSI_ALARM_NONE = 0,
    CSI_ALARM_START = 1,
    CSI_ALARM_END = 2,
} csi_alarm_event_t;

typedef struct {
    csi_alarm_cfg_t cfg;
    uint16_t history;  /* bit i = positive decision i windows ago */
    uint32_t seen;     /* windows fed so far (saturating), the history is not full before n */
    uint16_t held;     /* windows since the alarm started */
    uint8_t active;
} csi_alarm_t;

/* Returns 0 on success, -1 if the configuration is out of range (the state is then left unusable). */
int csi_alarm_init(csi_alarm_t *a, const csi_alarm_cfg_t *cfg);

/* Clears the history and the alarm (use after calibration or a link outage). */
void csi_alarm_reset(csi_alarm_t *a);

/* Feeds one window. Returns the transition that happened, if any. */
csi_alarm_event_t csi_alarm_update(csi_alarm_t *a, float p_person);

static inline int csi_alarm_active(const csi_alarm_t *a) { return a->active; }
