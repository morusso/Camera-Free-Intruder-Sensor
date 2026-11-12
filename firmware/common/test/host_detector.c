/*
 * Host harness for csi_detector.c, driven by ml/test_detector_c.py. The "model" is a fake: P(person) = clamp(mean(x^2)/4).
 *
 *   host_detector base  <csi.bin> <baseline.bin> <out.bin>   run with a stored baseline (mean[N], std[N] float32)
 *   host_detector calib <csi.bin> <n_frames> <out.bin>       calibrate on the first n_frames cleaned frames, then run
 *
 * Output: one 8-byte record per non-empty result: float p_person, u8 kind, u8 alarm_event, u8 alarm_active, u8 0.
 * Alarm: on 0.5, off 0.5, n 5, k_on 3, k_off 3, no hold; stride 32.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "csi_detector.h"

static int fake_infer(void *ctx, const float *w, float *probs)
{
    (void)ctx;
    double s = 0;
    for (int i = 0; i < CSI_DET_WIN * CSI_PP_N_KEEP; i++) {
        s += (double)w[i] * w[i];
    }
    float p = (float)(s / (CSI_DET_WIN * CSI_PP_N_KEEP) / 4.0);
    p = p > 1.0f ? 1.0f : p;
    probs[0] = 1.0f - p;
    probs[1] = p / 2;
    probs[2] = p / 2;
    probs[3] = 0;
    return 4;
}

static void *read_file(const char *path, size_t *size)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    *size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc(*size ? *size : 1);
    if (fread(buf, 1, *size, f) != *size) {
        exit(2);
    }
    fclose(f);
    return buf;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        return 2;
    }
    size_t size;
    const int8_t *csi = read_file(argv[2], &size);
    const size_t frames = size / CSI_PP_N_VALUES;

    static csi_det_t det;
    csi_det_cfg_t cfg = {
        .alarm = {.on_threshold = 0.5f, .off_threshold = 0.5f, .n = 5, .k_on = 3, .k_off = 3, .min_hold_windows = 0},
        .stride = 32,
        .n_person = 2,
        .person_idx = {1, 2},
        .infer = fake_infer,
    };
    if (csi_det_init(&det, &cfg) != 0) {
        return 3;
    }
    if (strcmp(argv[1], "base") == 0) {
        size_t bsize;
        const float *b = read_file(argv[3], &bsize);
        csi_pp_baseline_t bl;
        memcpy(bl.mean, b, sizeof(bl.mean));
        memcpy(bl.std, b + CSI_PP_N_KEEP, sizeof(bl.std));
        csi_det_set_baseline(&det, &bl);
    } else {
        csi_det_calibrate_begin(&det, (uint32_t)atoi(argv[3]));
    }
    FILE *out = fopen(argv[4], "wb");
    csi_det_result_t r;
    for (size_t i = 0; i < frames; i++) {
        csi_det_push(&det, csi + i * CSI_PP_N_VALUES, &r);
        if (r.kind != CSI_DET_NOTHING) {
            const float p = r.p_person;
            const uint8_t rec[4] = {(uint8_t)r.kind, (uint8_t)r.alarm_event, (uint8_t)r.alarm_active, 0};
            fwrite(&p, sizeof p, 1, out);
            fwrite(rec, 1, 4, out);
        }
    }
    if (strcmp(argv[1], "calib") == 0) {
        /* the baseline that was learned, for comparison with ml/preprocess.fit_baseline */
        fwrite(det.baseline.mean, sizeof(float), CSI_PP_N_KEEP, out);
        fwrite(det.baseline.std, sizeof(float), CSI_PP_N_KEEP, out);
    }
    fclose(out);
    return 0;
}
