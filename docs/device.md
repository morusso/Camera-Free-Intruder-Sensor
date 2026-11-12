# RX device: detector, calibration, MQTT

Status: **builds, the logic is tested on the host, never run on a board.** Every number below marked "unverified"
has to be confirmed on hardware (steps 4.3 and 4.8).

## Data path (firmware/rx)

```
Wi-Fi task (core 0)  csi_rx_cb: copy the frame into a queue, never blocks
detector task (core 1)  csi_detector (firmware/common):
    preprocess (amplitude, Hampel, low-pass) -> normalize with the empty-room baseline
    -> ring of 128 frames -> every 32 frames: TFLite Micro model -> P(person) = P(static_person) + P(walking)
    -> csi_alarm: threshold with hysteresis, k of n windows, minimum alarm length
    -> events (alarm_start, alarm_end, calib_*, link_lost, link_restored)
MQTT task (core 1)  node_mqtt: events, status, metrics over TLS (only with CONFIG_CSI_RX_MQTT)
```

Memory: the detector state is about 55 KB static (window ring + linearized window), the tensor arena 80 KB
(CSI_MODEL_ARENA_BYTES). Free heap and the minimum since boot are in `# STATS` and in the metrics.

## Build variants (`idf.py menuconfig` -> "CSI receiver")

| Option | Default | Meaning |
|---|---|---|
| `CSI_RX_STREAM_UART` | y | print every frame as `CSI_DATA` (needed for data collection, turn off when deployed) |
| `CSI_RX_DETECTOR` | y | run the detector |
| `CSI_RX_MQTT` | n | join Wi-Fi and publish over MQTT/TLS; needs `firmware/rx/certs/{ca.pem,client.crt,client.key}` |

Alarm parameters (`CSI_DET_ALARM_*`) have placeholder defaults (on 60%, off 40%, 3 of 5, hold 5 windows). Tune them
with `ml/evaluate.py events` on real recordings (same rule, same names) before trusting any number.

## Calibration (step 4.4)

The model works on deviations from the empty-room statistics, so **a new device does nothing until it is calibrated**
(LED blinks fast if `CSI_DET_LED_GPIO` is set; `# DET state=uncalibrated`).

1. Leave the room. Everyone, including you.
2. Start: hold the BOOT button (GPIO0) for 2 s, or publish `calibrate` (or `calibrate:<seconds>`, 10-3600) to
   `sensors/<id>/cmd`. The default length is `CSI_DET_CALIB_SECONDS` = 180 s. Any alarm is cleared.
3. The result is stored in NVS (`csi/baseline`, with a CRC and the subcarrier count) and loaded at the next boot.
   `calib_done` or `calib_failed` is reported as an event.

`CSI_DET_MAX_CALIB_STD_X100` rejects a calibration whose mean amplitude spread is too high (person present, loud fan).
It is off (0) until a value is read from real empty-room data. A device that loses its link (`link_lost`, no frames for
`CSI_DET_LINK_TIMEOUT_MS`) restarts the filter and the window; the baseline stays.

## MQTT (step 4.6)

Wi-Fi limitation: the radio has one channel. With MQTT on, the access point must be on `CONFIG_CSI_LINK_CHANNEL`,
and the TX must use the same channel. If that is not possible, use Plan B of the project (send features, not raw CSI).

| Topic | Direction | Content |
|---|---|---|
| `sensors/<id>/event` | device -> broker, QoS 1 | `{"v":1,"device","link","event","ts_ms","uptime_ms","class","confidence","p_person"}` |
| `sensors/<id>/status` | QoS 1, retained | state, calibration progress, alarm, CSI link, RSSI, uptime; last will `{"online":0}` |
| `sensors/<id>/metrics` | QoS 0, every 60 s | frames, windows, alarms, gaps, queue drops, inference time, heap, `p_person`, `energy` |
| `sensors/<id>/cmd` | broker -> device | `calibrate`, `calibrate:<seconds>` |

`ts_ms` is 0 until SNTP has synchronized the clock; the backend stores arrival time. Flags in `status` are 0/1
because Telegraf's JSON parser drops booleans. Raw CSI never leaves the device.

## Console lines (all start with `#`, `tools/csi_reader.py` ignores them)

- `# MODEL ok: ...` boot self-test: arena use, golden difference, inference time (step 4.3, read it on the board).
- `# DET state=... windows=... infer_us_avg=... p_person=... energy=...` every `CSI_RX_STATS_INTERVAL_S`.
- `# EVENT kind=...` for every event (kind: 0 alarm_start, 1 alarm_end, 2 calib_done, 3 calib_failed, 4 link_lost,
  5 link_restored). This is the only output when MQTT is off.

## Not done (needs hardware or data)

- 4.3 profiling numbers, 4.8 integration tests (alarm in Grafana < 2 s, 24 h false alarms, 24 h heap).
- The model in the image is an untrained placeholder: regenerate `model_data.c` with `ml/tflite_to_c.py` from a model
  trained on real data.
- The addressable RGB LED of the DevKitM-1 is not driven (needs `led_strip`).
