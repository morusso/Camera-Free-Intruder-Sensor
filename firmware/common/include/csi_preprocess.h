/*
 * Streaming CSI preprocessing, a C port of ml/preprocess.py (steps 1-4).
 *
 *   int8 CSI frame -> amplitude of the kept subcarriers -> Hampel filter -> causal low-pass -> normalize
 *
 * Feed frames with csi_pp_push(). The Hampel window is centered, so the output lags the input by
 * CSI_PP_HAMPEL_HALF frames: the first CSI_PP_HAMPEL_HALF pushes return 0. Call csi_pp_flush() after the
 * last frame to drain the rest (the Python version pads the end by repeating the last sample; so does flush).
 *
 * Plain C with no ESP-IDF dependencies, so it is unit-tested on the host (ml/test_preprocess_c.py).
 * State is about 2.2 KB; no heap allocation.
 */
#pragma once

#include <stdint.h>
#include "preprocess_params.h"

typedef struct {
    float mean[CSI_PP_N_KEEP];
    float std[CSI_PP_N_KEEP];   /* already floored at CSI_PP_NORM_MIN_STD */
} csi_pp_baseline_t;

typedef struct {
    float ring[CSI_PP_HAMPEL_WIN][CSI_PP_N_KEEP];  /* last amplitudes */
    float last[CSI_PP_N_KEEP];                     /* newest amplitude, repeated by flush */
    float z[CSI_PP_N_SECTIONS][2][CSI_PP_N_KEEP];  /* low-pass state (transposed direct form II) */
    uint32_t n_in;                                 /* real frames pushed */
    uint32_t n_pushed;                             /* including the padding pushed by flush */
} csi_pp_t;

void csi_pp_init(csi_pp_t *pp);

/*
 * Push one frame of CSI_PP_N_VALUES int8 values. Returns 1 and writes CSI_PP_N_KEEP amplitudes to `out`
 * when an output is ready, 0 otherwise. Returns -1 if the frame is NULL.
 */
int csi_pp_push(csi_pp_t *pp, const int8_t *csi, float *out);

/* After the last frame: returns 1 and writes the next pending output, 0 when everything has been drained. */
int csi_pp_flush(csi_pp_t *pp, float *out);

/* x = (x - mean) / std, in place. */
void csi_pp_normalize(float *x, const csi_pp_baseline_t *bl);

/*
 * Empty-room baseline from a stream of cleaned frames (Welford, so no frame buffer is needed; this is what the
 * calibration of step 4.4 will use). Population std, like numpy.
 */
typedef struct {
    double mean[CSI_PP_N_KEEP];
    double m2[CSI_PP_N_KEEP];
    uint32_t n;
} csi_pp_baseline_acc_t;

void csi_pp_baseline_acc_init(csi_pp_baseline_acc_t *acc);
void csi_pp_baseline_acc_add(csi_pp_baseline_acc_t *acc, const float *frame);
/* Returns 0 on success, -1 if no frames were added. std is floored at CSI_PP_NORM_MIN_STD. */
int csi_pp_baseline_acc_finish(const csi_pp_baseline_acc_t *acc, csi_pp_baseline_t *bl);
