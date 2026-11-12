/* Host test driver for csi_alarm: host_alarm in.f32 on off n k_on k_off min_hold out.u8
 * out has two bytes per window: active, event. */
#include <stdio.h>
#include <stdlib.h>

#include "csi_alarm.h"

int main(int argc, char **argv)
{
    if (argc != 9) {
        fprintf(stderr, "usage: %s in.f32 on off n k_on k_off min_hold out.u8\n", argv[0]);
        return 2;
    }
    csi_alarm_cfg_t cfg = {.on_threshold = (float)atof(argv[2]),
                           .off_threshold = (float)atof(argv[3]),
                           .n = (uint8_t)atoi(argv[4]),
                           .k_on = (uint8_t)atoi(argv[5]),
                           .k_off = (uint8_t)atoi(argv[6]),
                           .min_hold_windows = (uint16_t)atoi(argv[7])};
    csi_alarm_t a;
    if (csi_alarm_init(&a, &cfg) != 0) {
        return 3;
    }
    FILE *in = fopen(argv[1], "rb");
    FILE *out = fopen(argv[8], "wb");
    if (!in || !out) {
        return 4;
    }
    float p;
    while (fread(&p, sizeof p, 1, in) == 1) {
        const int ev = csi_alarm_update(&a, p);
        fputc(a.active, out);
        fputc(ev, out);
    }
    fclose(in);
    fclose(out);
    return 0;
}
