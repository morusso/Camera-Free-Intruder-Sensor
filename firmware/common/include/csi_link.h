/*
 * Definitions shared by the CSI transmitter (TX) and receiver (RX).
 * Keep this file in sync with docs/format.md.
 */
#pragma once

#include <stdint.h>

/* Fixed MAC of the transmitter, so the receiver can filter frames by sender. */
#define CSI_LINK_TX_MAC {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00}

/* Marks our frames; lets the receiver ignore foreign ESP-NOW traffic. */
#define CSI_LINK_MAGIC 0xC51A

/* ESP-NOW payload sent by the TX. Little endian, no padding. */
typedef struct __attribute__((packed)) {
    uint16_t magic;  /* CSI_LINK_MAGIC */
    uint32_t seq;    /* TX sequence number, wraps at 2^32 (becomes tx_seq in the serial line) */
} csi_link_payload_t;

/*
 * Find our payload inside a raw received frame and read the sequence number.
 *
 * The offset of the ESP-NOW payload inside the raw frame depends on the frame
 * layout, so instead of hard-coding it we search for CSI_LINK_MAGIC in the
 * first bytes. Returns 1 and sets *seq on success, 0 if not found.
 */
static inline int csi_link_find_seq(const uint8_t *frame, uint16_t frame_len, uint32_t *seq)
{
    const uint16_t need = (uint16_t)sizeof(csi_link_payload_t);
    const uint16_t scan = frame_len < 96 ? frame_len : 96;
    for (uint16_t i = 0; (uint16_t)(i + need) <= scan; i++) {
        if (frame[i] == (CSI_LINK_MAGIC & 0xff) && frame[i + 1] == (CSI_LINK_MAGIC >> 8)) {
            *seq = (uint32_t)frame[i + 2] | ((uint32_t)frame[i + 3] << 8) |
                   ((uint32_t)frame[i + 4] << 16) | ((uint32_t)frame[i + 5] << 24);
            return 1;
        }
    }
    return 0;
}
