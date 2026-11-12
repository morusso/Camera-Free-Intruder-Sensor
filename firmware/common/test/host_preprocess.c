/*
 * Host harness for csi_preprocess.c, driven by ml/test_preprocess_c.py.
 *
 *   host_preprocess clean  <csi.bin> <out.bin>              int8 frames -> cleaned amplitudes (float32)
 *   host_preprocess norm   <csi.bin> <baseline.bin> <out>   same, then normalized; baseline = mean[N], std[N] float32
 *   host_preprocess fit    <csi.bin> <out.bin>              fit a baseline from the cleaned frames: mean[N], std[N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "csi_preprocess.h"

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
        perror("fread");
        exit(2);
    }
    fclose(f);
    return buf;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: see header comment\n");
        return 2;
    }
    const char *mode = argv[1];
    size_t size;
    const int8_t *csi = read_file(argv[2], &size);
    const size_t frames = size / CSI_PP_N_VALUES;

    csi_pp_baseline_t bl;
    const char *out_path = argv[3];
    int normalize = 0;
    if (strcmp(mode, "norm") == 0) {
        if (argc < 5) {
            return 2;
        }
        size_t bsize;
        void *b = read_file(argv[3], &bsize);
        if (bsize != sizeof(bl)) {
            fprintf(stderr, "baseline must be %zu bytes\n", sizeof(bl));
            return 2;
        }
        memcpy(&bl, b, sizeof(bl));
        out_path = argv[4];
        normalize = 1;
    }

    csi_pp_t *pp = malloc(sizeof(*pp));
    csi_pp_init(pp);
    csi_pp_baseline_acc_t acc;
    csi_pp_baseline_acc_init(&acc);
    float *out = malloc(sizeof(float) * CSI_PP_N_KEEP * (frames ? frames : 1));
    size_t n_out = 0;
    float row[CSI_PP_N_KEEP];

    for (size_t t = 0; t <= frames; t++) {
        int got;
        while (1) {
            got = t < frames ? csi_pp_push(pp, csi + t * CSI_PP_N_VALUES, row) : csi_pp_flush(pp, row);
            if (got > 0) {
                if (normalize) {
                    csi_pp_normalize(row, &bl);
                }
                csi_pp_baseline_acc_add(&acc, row);
                memcpy(out + n_out * CSI_PP_N_KEEP, row, sizeof(row));
                n_out++;
            }
            if (t < frames || got <= 0) {
                break;  /* flush is called until it has nothing left */
            }
        }
    }
    if (n_out != frames) {
        fprintf(stderr, "expected %zu outputs, got %zu\n", frames, n_out);
        return 1;
    }

    FILE *f = fopen(out_path, "wb");
    if (strcmp(mode, "fit") == 0) {
        csi_pp_baseline_t fit;
        csi_pp_baseline_acc_finish(&acc, &fit);
        fwrite(&fit, sizeof(fit), 1, f);
    } else {
        fwrite(out, sizeof(float) * CSI_PP_N_KEEP, n_out, f);
    }
    fclose(f);
    return 0;
}
