#include "csi_detector.h"

#include <string.h>

static void clear_window_state(csi_det_t *d)
{
    d->filled = 0;
    d->head = 0;
    d->since_eval = d->cfg.stride - 1;  /* evaluate as soon as the ring is full, like stream_probs */
    d->window_idx = 0;
    csi_alarm_reset(&d->alarm);
}

int csi_det_init(csi_det_t *d, const csi_det_cfg_t *cfg)
{
    if (!d || !cfg || !cfg->infer || cfg->stride < 1 || cfg->stride > CSI_DET_WIN || cfg->n_person < 1 ||
        cfg->n_person > CSI_DET_MAX_CLASSES) {
        return -1;
    }
    for (int i = 0; i < cfg->n_person; i++) {
        if (cfg->person_idx[i] >= CSI_DET_MAX_CLASSES) {
            return -1;
        }
    }
    memset(d, 0, sizeof(*d));
    d->cfg = *cfg;
    if (csi_alarm_init(&d->alarm, &cfg->alarm) != 0) {
        return -1;
    }
    csi_pp_init(&d->pp);
    d->state = CSI_DET_UNCALIBRATED;
    return 0;
}

void csi_det_calibrate_begin(csi_det_t *d, uint32_t n_frames)
{
    if (n_frames == 0) {
        n_frames = 1;
    }
    csi_pp_baseline_acc_init(&d->acc);
    d->calib_target = n_frames;
    d->state = CSI_DET_CALIBRATING;
    clear_window_state(d);
}

void csi_det_calibrate_cancel(csi_det_t *d)
{
    if (d->state == CSI_DET_CALIBRATING) {
        d->state = d->baseline.std[0] > 0.0f ? CSI_DET_RUNNING : CSI_DET_UNCALIBRATED;
        clear_window_state(d);
    }
}

void csi_det_set_baseline(csi_det_t *d, const csi_pp_baseline_t *bl)
{
    d->baseline = *bl;
    d->state = CSI_DET_RUNNING;
    clear_window_state(d);
}

void csi_det_reset_stream(csi_det_t *d)
{
    csi_pp_init(&d->pp);
    clear_window_state(d);
}

int csi_det_calib_percent(const csi_det_t *d)
{
    if (d->state != CSI_DET_CALIBRATING) {
        return d->state == CSI_DET_RUNNING ? 100 : 0;
    }
    return (int)((uint64_t)d->acc.n * 100u / d->calib_target);
}

static void evaluate(csi_det_t *d, csi_det_result_t *out)
{
    /* Linearize the ring: oldest frame first. head points at the oldest one once the ring is full. */
    const size_t row = CSI_PP_N_KEEP * sizeof(float);
    const uint32_t first = CSI_DET_WIN - d->head;
    memcpy(d->window, &d->ring[d->head][0], first * row);
    memcpy(d->window + first * CSI_PP_N_KEEP, &d->ring[0][0], d->head * row);

    float probs[CSI_DET_MAX_CLASSES] = {0};
    const int n = d->cfg.infer(d->cfg.infer_ctx, d->window, probs);
    out->kind = CSI_DET_WINDOW;
    out->window_idx = d->window_idx++;
    if (n <= 0 || n > CSI_DET_MAX_CLASSES) {
        out->infer_error = 1;
        out->alarm_active = csi_alarm_active(&d->alarm);
        return;
    }
    double e = 0.0;
    for (int i = 0; i < CSI_DET_WIN * CSI_PP_N_KEEP; i++) {
        e += (double)d->window[i] * d->window[i];
    }
    out->energy = (float)(e / (CSI_DET_WIN * CSI_PP_N_KEEP));
    out->n_classes = (uint8_t)n;
    memcpy(out->probs, probs, sizeof(probs));
    float p = 0.0f;
    for (int i = 0; i < d->cfg.n_person; i++) {
        if (d->cfg.person_idx[i] < n) {
            p += probs[d->cfg.person_idx[i]];
        }
    }
    out->p_person = p > 1.0f ? 1.0f : p;
    out->alarm_event = csi_alarm_update(&d->alarm, out->p_person);
    out->alarm_active = csi_alarm_active(&d->alarm);
}

void csi_det_push(csi_det_t *d, const int8_t *csi, csi_det_result_t *out)
{
    memset(out, 0, sizeof(*out));
    float x[CSI_PP_N_KEEP];
    if (csi_pp_push(&d->pp, csi, x) != 1) {
        out->alarm_active = csi_alarm_active(&d->alarm);
        return;
    }

    if (d->state == CSI_DET_CALIBRATING) {
        csi_pp_baseline_acc_add(&d->acc, x);
        if (d->acc.n >= d->calib_target) {
            csi_pp_baseline_t bl;
            int ok = csi_pp_baseline_acc_finish(&d->acc, &bl) == 0;
            if (ok && d->cfg.max_calib_std_ratio > 0.0f) {
                /* A room with someone in it (or a loud fan) has a much larger spread than the usual noise. */
                float mean_std = 0.0f;
                for (int i = 0; i < CSI_PP_N_KEEP; i++) {
                    mean_std += bl.std[i];
                }
                ok = (mean_std / CSI_PP_N_KEEP) <= d->cfg.max_calib_std_ratio;
            }
            if (ok) {
                d->baseline = bl;
                d->state = CSI_DET_RUNNING;
                out->kind = CSI_DET_CALIB_DONE;
            } else {
                d->state = CSI_DET_UNCALIBRATED;
                out->kind = CSI_DET_CALIB_FAILED;
            }
            clear_window_state(d);
        }
        return;
    }
    if (d->state != CSI_DET_RUNNING) {
        return;
    }

    csi_pp_normalize(x, &d->baseline);
    memcpy(&d->ring[d->head][0], x, sizeof(x));
    d->head = (d->head + 1) % CSI_DET_WIN;
    if (d->filled < CSI_DET_WIN) {
        d->filled++;
    }
    if (d->filled < CSI_DET_WIN) {
        return;
    }
    if (++d->since_eval >= d->cfg.stride) {
        d->since_eval = 0;
        evaluate(d, out);
    }
}
