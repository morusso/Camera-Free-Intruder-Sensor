/*
 * Detector pipeline (steps 4.1, 4.4, 4.5): CSI frames in, window decisions and alarm events out.
 *
 *   frame -> csi_preprocess (amplitude, Hampel, low-pass) -> [calibration: empty-room baseline]
 *         -> normalize -> ring of WIN frames -> every STRIDE frames: infer() -> P(person) -> csi_alarm
 *
 * Plain C with no ESP-IDF dependencies and no heap: the caller owns the csi_det_t (about 55 KB, mostly the window
 * ring and the linearized window), so it is unit-tested on the host with a fake model (ml/test_detector_c.py).
 *
 * States: UNCALIBRATED (no baseline: frames are cleaned but nothing is evaluated), CALIBRATING (the baseline is
 * accumulated from n frames), RUNNING. A baseline can be loaded with csi_det_set_baseline() (from NVS).
 * Calibrating clears the alarm. Run it only when the room is empty.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "csi_alarm.h"
#include "csi_preprocess.h"

#define CSI_DET_WIN 128
#define CSI_DET_MAX_CLASSES 8

typedef enum {
    CSI_DET_UNCALIBRATED = 0,
    CSI_DET_CALIBRATING = 1,
    CSI_DET_RUNNING = 2,
} csi_det_state_t;

typedef enum {
    CSI_DET_NOTHING = 0,
    CSI_DET_WINDOW = 1,        /* a window was evaluated: see probs / p_person / alarm_event */
    CSI_DET_CALIB_DONE = 2,    /* baseline ready, state is RUNNING now */
    CSI_DET_CALIB_FAILED = 3,  /* the calibration data was unusable, state is UNCALIBRATED */
} csi_det_kind_t;

/* Runs the model on one window of CSI_DET_WIN * CSI_PP_N_KEEP floats (time-major). Returns the number of classes
 * written to probs, or <= 0 on error. */
typedef int (*csi_det_infer_fn)(void *ctx, const float *window, float *probs);

typedef struct {
    csi_alarm_cfg_t alarm;
    uint16_t stride;                     /* frames between evaluations, e.g. 32 */
    uint8_t n_person;                    /* classes that count as a person */
    uint8_t person_idx[CSI_DET_MAX_CLASSES];
    float max_calib_std_ratio;           /* reject a calibration whose mean std is above this (0 = no check) */
    csi_det_infer_fn infer;
    void *infer_ctx;
} csi_det_cfg_t;

typedef struct {
    csi_det_kind_t kind;
    uint32_t window_idx;                 /* evaluations since the last reset */
    uint8_t n_classes;
    float probs[CSI_DET_MAX_CLASSES];
    float p_person;
    float energy;                        /* mean x^2 of the normalized window: a cheap feature for drift monitoring */
    csi_alarm_event_t alarm_event;
    int alarm_active;
    int infer_error;                     /* the model returned an error, the window was skipped */
} csi_det_result_t;

typedef struct {
    csi_det_cfg_t cfg;
    csi_det_state_t state;
    csi_pp_t pp;
    csi_pp_baseline_t baseline;
    csi_pp_baseline_acc_t acc;
    uint32_t calib_target;
    csi_alarm_t alarm;
    uint32_t filled;                     /* frames in the ring, saturates at CSI_DET_WIN */
    uint32_t head;                       /* next ring slot to write */
    uint32_t since_eval;
    uint32_t window_idx;
    float ring[CSI_DET_WIN][CSI_PP_N_KEEP];
    float window[CSI_DET_WIN * CSI_PP_N_KEEP];
} csi_det_t;

/* 0 on success, -1 on a bad configuration. */
int csi_det_init(csi_det_t *d, const csi_det_cfg_t *cfg);

/* Starts the empty-room calibration over the next n_frames cleaned frames. Valid in any state; the alarm is cleared. */
void csi_det_calibrate_begin(csi_det_t *d, uint32_t n_frames);
void csi_det_calibrate_cancel(csi_det_t *d);

/* Loads a stored baseline and goes to RUNNING. */
void csi_det_set_baseline(csi_det_t *d, const csi_pp_baseline_t *bl);

/* Clears the ring, the alarm and the filter (after a gap in the frames); keeps the baseline. */
void csi_det_reset_stream(csi_det_t *d);

/* Pushes one frame of CSI_PP_N_VALUES int8 values. Fills `out` always (kind CSI_DET_NOTHING if nothing happened). */
void csi_det_push(csi_det_t *d, const int8_t *csi, csi_det_result_t *out);

/* Progress of the calibration, 0..100. */
int csi_det_calib_percent(const csi_det_t *d);
