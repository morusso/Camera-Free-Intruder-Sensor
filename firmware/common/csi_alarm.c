#include "csi_alarm.h"

static int popcount16(uint16_t v)
{
    int c = 0;
    for (; v; v &= (uint16_t)(v - 1)) {
        c++;
    }
    return c;
}

int csi_alarm_init(csi_alarm_t *a, const csi_alarm_cfg_t *cfg)
{
    if (!a || !cfg || cfg->n < 1 || cfg->n > CSI_ALARM_MAX_N || cfg->k_on < 1 || cfg->k_on > cfg->n ||
        cfg->k_off < 1 || cfg->k_off > cfg->n || cfg->off_threshold > cfg->on_threshold) {
        return -1;
    }
    a->cfg = *cfg;
    csi_alarm_reset(a);
    return 0;
}

void csi_alarm_reset(csi_alarm_t *a)
{
    a->history = 0;
    a->seen = 0;
    a->held = 0;
    a->active = 0;
}

csi_alarm_event_t csi_alarm_update(csi_alarm_t *a, float p_person)
{
    const float thr = a->active ? a->cfg.off_threshold : a->cfg.on_threshold;
    const uint16_t mask = (uint16_t)((a->cfg.n == 16) ? 0xFFFFu : ((1u << a->cfg.n) - 1u));

    a->history = (uint16_t)(((a->history << 1) | (p_person > thr ? 1u : 0u)) & mask);
    if (a->seen < 0xFFFFFFFFu) {
        a->seen++;
    }
    const int positives = popcount16(a->history);

    if (!a->active) {
        if (positives >= a->cfg.k_on) {
            a->active = 1;
            a->held = 0;
            return CSI_ALARM_START;
        }
        return CSI_ALARM_NONE;
    }
    if (a->held < 0xFFFF) {
        a->held++;
    }
    if (positives < a->cfg.k_off && a->held >= a->cfg.min_hold_windows) {
        a->active = 0;
        return CSI_ALARM_END;
    }
    return CSI_ALARM_NONE;
}
