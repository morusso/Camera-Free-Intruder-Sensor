# Camera-free intruder sensor (Wi-Fi CSI, ESP32-S3)

> **Status: early stage, not ready to run yet.**
> The firmware has not been run on real boards, and the tools and models have only been checked on
> synthetic data. There are no results on real recordings yet.

## Goals and assumptions

The sensor detects the presence and movement of people in a room from changes in the Wi-Fi radio channel
(CSI, Channel State Information), without cameras and without recording any images. It is meant as a
physical security component (server room, warehouse after hours).

- **Hardware:** a pair of ESP32-S3 boards (transmitter TX and receiver RX), ESP-NOW at a fixed rate (start: 100 Hz), 2.4 GHz band, HT20.
- **Classes:** `empty` (nobody there), `static_person` (a person standing or sitting), `walking` (a person moving), `interference` (fan, air conditioning, doors, etc.).
- **Model:** a small 1D-CNN on windows of 128 samples × 52 subcarriers, int8 quantization, with inference on the device as the goal (TFLite Micro, tensor arena ≤ 80 KB).
- **Hardest part:** a model trained in one room does not work in another, or after furniture is moved. So the plan includes per-installation calibration (empty-room statistics) and drift monitoring.
- **Success criteria:** detect a person entering in < 2 s, with at most about 1 false alarm per day in an empty room.

The full step-by-step plan (Stages 0–6) is in the project description. Working documentation is in `docs/`.

## What is done

Everything below passes its unit tests, but **without real data and without hardware**.

| Area | State |
|---|---|
| Environment (ESP-IDF 5.5, Python `.venv`, directory layout) | done |
| `firmware/tx`, `firmware/rx`, `firmware/common` | builds for esp32s3, **not tested on boards** |
| `tools/csi_reader.py` (reading, validation, loss counting, `.npz` output) | done, tested on synthetic data |
| `tools/live_heatmap.py` (live view, `--demo` mode) | done, the live window has not been checked |
| `tools/jitter_report.py` (jitter and loss, Stage 1 criteria) | done |
| `tools/labeler`, `tools/align_labels.py` (labeling from a phone, attaching labels to frames) | done, the phone page has not been checked |
| `docs/format.md`, `docs/collection-protocol.md` | done (format to be confirmed on hardware) |
| `ml/preprocess.py`, `ml/baseline.py`, `ml/dataset.py` | done |
| `ml/models.py`, `ml/train.py`, `ml/evaluate.py` (CNN, GRU, autoencoder) | done |
| `ml/export_tflite.py`, `ml/arena.py` (int8, arena estimate) | done, CNN arena about 13 KB (lower bound) |
| `firmware/common` C preprocessing, alarm logic, detector pipeline | done, host parity tests against `ml/` (fake model) |
| `firmware/rx`: TFLite Micro model, detector task, NVS calibration, button/LED, MQTT over mTLS | builds (also with MQTT), **never run on a board**; the model is an untrained placeholder; see `docs/device.md` |
| `backend/` Mosquitto + Telegraf + InfluxDB + Grafana, `tools/provision_pki.sh` | run with real containers and simulated messages, see `docs/backend.md` |

## What is left to do

1. **Hardware (Stage 1):** connect the boards, pick a channel, measure jitter and loss, confirm the data format (including the number of CSI values and the order of the pairs), look at the heatmap of an empty room versus movement.
2. **Data (Stage 2):** collect and label data following `docs/collection-protocol.md` (at least about 3 h per class, over several days), compute the baseline and the false alarm rate per day.
3. **Models (Stage 3):** train on real data, evaluate on a test set from a different day and after moving furniture, check the quality drop after quantization, simulate per-installation calibration (step 3.5).
4. **Device (Stage 4):** code for 4.1-4.7 exists; what is left is hardware: profiling (4.3), flashing and checking calibration/MQTT on boards, the integration test (4.8), and a model trained on real data.
5. **Multiple links and drift (Stage 5):** fuse decisions from several links, recalibration, a 7-day test.
6. **Hardening (Stage 6):** Secure Boot, flash encryption, signed OTA, mTLS, threat model, privacy.

## Repository layout

```
firmware/   tx, rx, common (ESP-IDF)
tools/      CSI reader, live view, labeling, label alignment, jitter report
ml/         preprocessing, baseline, dataset, models, training, evaluation, TFLite export
backend/    docker-compose: Mosquitto (mTLS), Telegraf, InfluxDB, Grafana
docs/       data format, data collection protocol, device (detector, calibration, MQTT), backend
data/       data (not in git)
```

## Running the tests

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tools
.venv/bin/python -m unittest discover -s ml
```

The firmware needs ESP-IDF v5.5 (`. ~/esp/esp-idf/export.sh`, then `idf.py build` in `firmware/tx` or `firmware/rx`).
