#include "csi_preprocess.h"

#include <math.h>
#include <string.h>

#define N CSI_PP_N_KEEP
#define W CSI_PP_HAMPEL_WIN
#define H CSI_PP_HAMPEL_HALF
#define MAD_TO_SIGMA 1.4826f

static const uint16_t KEEP[N] = CSI_PP_KEEP_INIT;
static const float SOS[CSI_PP_N_SECTIONS][6] = CSI_PP_SOS_INIT;
static const float ZI[CSI_PP_N_SECTIONS][2] = CSI_PP_ZI_INIT;

void csi_pp_init(csi_pp_t *pp)
{
    memset(pp, 0, sizeof(*pp));
}

static void sort_small(float *v, int n)
{
    for (int i = 1; i < n; i++) {
        float x = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > x) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = x;
    }
}

/* Hampel on the center sample of the window, for subcarrier k. */
static float hampel_center(const csi_pp_t *pp, int center_slot, int k)
{
    float tmp[W];
    for (int i = 0; i < W; i++) {
        tmp[i] = pp->ring[i][k];
    }
    sort_small(tmp, W);
    const float med = tmp[H];

    const float x = pp->ring[center_slot][k];
    for (int i = 0; i < W; i++) {
        tmp[i] = fabsf(pp->ring[i][k] - med);
    }
    sort_small(tmp, W);
    float scale = tmp[H] * MAD_TO_SIGMA;
    if (scale < CSI_PP_HAMPEL_MIN_SCALE) {
        scale = CSI_PP_HAMPEL_MIN_SCALE;
    }
    return fabsf(x - med) > CSI_PP_HAMPEL_NSIGMA * scale ? med : x;
}

static int push_amp(csi_pp_t *pp, const float *amp, float *out)
{
    if (pp->n_pushed == 0) {
        /* Left edge: the Python version repeats the first sample. */
        for (int i = 0; i < W; i++) {
            memcpy(pp->ring[i], amp, sizeof(float) * N);
        }
    }
    const int newest = (int)(pp->n_pushed % W);
    memcpy(pp->ring[newest], amp, sizeof(float) * N);
    pp->n_pushed++;
    if (pp->n_pushed <= H) {
        return 0;
    }

    const int center = (newest - H + W) % W;
    const int first = (pp->n_pushed == H + 1);
    for (int k = 0; k < N; k++) {
        const float x0 = hampel_center(pp, center, k);
        float v = x0;
        for (int s = 0; s < CSI_PP_N_SECTIONS; s++) {
            const float *c = SOS[s];
            float *z0 = &pp->z[s][0][k];
            float *z1 = &pp->z[s][1][k];
            if (first) {
                /* No start-up bump: begin at the steady state for a constant input x0. ZI already includes the
                 * gain of the preceding sections, so it is scaled by the filter input, not this section's. */
                *z0 = ZI[s][0] * x0;
                *z1 = ZI[s][1] * x0;
            }
            const float y = c[0] * v + *z0;
            *z0 = c[1] * v - c[4] * y + *z1;
            *z1 = c[2] * v - c[5] * y;
            v = y;
        }
        out[k] = v;
    }
    return 1;
}

int csi_pp_push(csi_pp_t *pp, const int8_t *csi, float *out)
{
    if (csi == NULL) {
        return -1;
    }
    float amp[N];
    for (int k = 0; k < N; k++) {
        const float im = (float)csi[2 * KEEP[k]];
        const float re = (float)csi[2 * KEEP[k] + 1];
        amp[k] = sqrtf(im * im + re * re);
    }
    memcpy(pp->last, amp, sizeof(amp));
    pp->n_in++;
    return push_amp(pp, amp, out);
}

int csi_pp_flush(csi_pp_t *pp, float *out)
{
    /* Padding pushes that do not yet yield an output (only when fewer than H frames were pushed) are looped over. */
    while (pp->n_in > 0 && pp->n_pushed < pp->n_in + H) {
        if (push_amp(pp, pp->last, out)) {
            return 1;
        }
    }
    return 0;
}

void csi_pp_normalize(float *x, const csi_pp_baseline_t *bl)
{
    for (int k = 0; k < N; k++) {
        x[k] = (x[k] - bl->mean[k]) / bl->std[k];
    }
}

void csi_pp_baseline_acc_init(csi_pp_baseline_acc_t *acc)
{
    memset(acc, 0, sizeof(*acc));
}

void csi_pp_baseline_acc_add(csi_pp_baseline_acc_t *acc, const float *frame)
{
    acc->n++;
    for (int k = 0; k < N; k++) {
        const double d = (double)frame[k] - acc->mean[k];
        acc->mean[k] += d / acc->n;
        acc->m2[k] += d * ((double)frame[k] - acc->mean[k]);
    }
}

int csi_pp_baseline_acc_finish(const csi_pp_baseline_acc_t *acc, csi_pp_baseline_t *bl)
{
    if (acc->n == 0) {
        return -1;
    }
    for (int k = 0; k < N; k++) {
        float sd = (float)sqrt(acc->m2[k] / acc->n);
        bl->mean[k] = (float)acc->mean[k];
        bl->std[k] = sd < CSI_PP_NORM_MIN_STD ? CSI_PP_NORM_MIN_STD : sd;
    }
    return 0;
}
