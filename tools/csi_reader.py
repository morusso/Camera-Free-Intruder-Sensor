#!/usr/bin/env python3
"""Read the CSI stream from the receiver (RX) and save it to a file.

The line format is described in docs/format.md.

Examples:
    python tools/csi_reader.py --port /dev/cu.usbmodem1101 --out data/raw/test --duration 60
    python tools/csi_reader.py --replay example.txt --out data/raw/test
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import numpy as np
import typer

PREFIX = "CSI_DATA,"
N_FIELDS = 14  # CSI_DATA + 12 fields + data list
TS_WRAP = 2 ** 32  # local_timestamp is a uint32 in microseconds
SEQ_WRAP = 2 ** 32
MAC_RE = re.compile(r"^[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5}$")


@dataclass
class Frame:
    rx_idx: int
    tx_seq: int
    mac: str
    rssi: int
    rate: int
    noise_floor: int
    channel: int
    local_timestamp: int
    sig_len: int
    rx_format: int
    first_word: int
    csi: np.ndarray  # int8, length = `len` field from the line


def parse_line(line: str) -> Frame:
    """Turn one line into a Frame. Raises ValueError with the reason on failure."""
    line = line.strip()
    if not line.startswith(PREFIX):
        raise ValueError("not_data")
    # The number list is cut out separately because it contains commas.
    q1, q2 = line.find('"'), line.rfind('"')
    if q1 < 0 or q2 <= q1:
        raise ValueError("bad_field_count")
    head = line[:q1].rstrip(",").split(",")
    body = line[q1 + 1:q2].strip()
    if len(head) != N_FIELDS - 1:
        raise ValueError("bad_field_count")
    if not (body.startswith("[") and body.endswith("]")):
        raise ValueError("bad_data_list")
    try:
        (_, rx_idx, tx_seq, mac, rssi, rate, noise, channel, ts, sig_len, rx_fmt, n, first_word) = head
        rx_idx, tx_seq, rssi, rate, noise = int(rx_idx), int(tx_seq), int(rssi), int(rate), int(noise)
        channel, ts, sig_len, rx_fmt = int(channel), int(ts), int(sig_len), int(rx_fmt)
        n, first_word = int(n), int(first_word)
        values = [int(x) for x in body[1:-1].split(",") if x.strip() != ""]
    except ValueError:
        raise ValueError("bad_number")
    if not MAC_RE.match(mac):
        raise ValueError("bad_mac")
    if len(values) != n:
        raise ValueError("len_mismatch")
    if values and (min(values) < -128 or max(values) > 127):
        raise ValueError("out_of_range")
    return Frame(rx_idx, tx_seq, mac.lower(), rssi, rate, noise, channel, ts,
                 sig_len, rx_fmt, first_word, np.array(values, dtype=np.int8))


def unwrap_timestamps(ts: np.ndarray) -> np.ndarray:
    """Remove the uint32 wraparound (every ~71.6 min); returns monotonically increasing int64."""
    ts = ts.astype(np.int64)
    if len(ts) < 2:
        return ts
    wraps = np.concatenate([[0], np.cumsum(np.diff(ts) < -TS_WRAP // 2)])
    return ts + wraps * TS_WRAP


def count_missing(seq: np.ndarray) -> int:
    """Count numbers missing from an increasing sequence (handles uint32 wraparound).

    Repeats and backward jumps (e.g. a transmitter restart) are not counted as losses.
    """
    if len(seq) < 2:
        return 0
    step = np.diff(seq.astype(np.int64)) % SEQ_WRAP
    # step 1 = no loss; step above half the range = backward jump, ignored
    ok = (step >= 1) & (step < SEQ_WRAP // 2)
    return int(np.sum(step[ok] - 1))


class Collector:
    def __init__(self) -> None:
        self.pc_time: List[float] = []
        self.frames: List[Frame] = []
        self.bad = {}  # reason -> count
        self.ref_len: Optional[int] = None

    def feed(self, line: str, pc_time: Optional[float] = None) -> None:
        try:
            f = parse_line(line)
        except ValueError as e:
            reason = str(e)
            if reason != "not_data" and line.strip():
                self.bad[reason] = self.bad.get(reason, 0) + 1
            return
        if self.ref_len is None:
            self.ref_len = len(f.csi)
        elif len(f.csi) != self.ref_len:
            self.bad["csi_len_changed"] = self.bad.get("csi_len_changed", 0) + 1
            return
        self.pc_time.append(time.time() if pc_time is None else pc_time)
        self.frames.append(f)

    def arrays(self) -> dict:
        fr = self.frames
        col = lambda name, dt: np.array([getattr(f, name) for f in fr], dtype=dt)  # noqa: E731
        return {
            "pc_time": np.array(self.pc_time, dtype=np.float64),
            "rx_idx": col("rx_idx", np.int64),
            "tx_seq": col("tx_seq", np.int64),
            "rssi": col("rssi", np.int16),
            "rate": col("rate", np.int16),
            "noise_floor": col("noise_floor", np.int16),
            "channel": col("channel", np.int16),
            "dev_ts_us": unwrap_timestamps(col("local_timestamp", np.int64)),
            "first_word": col("first_word", np.int8),
            "mac": np.array([f.mac for f in fr]),
            "csi": np.stack([f.csi for f in fr]) if fr else np.zeros((0, 0), np.int8),
        }

    def stats(self) -> dict:
        a = self.arrays()
        n = len(self.frames)
        lost_radio = count_missing(a["tx_seq"]) if n else 0
        lost_usb = count_missing(a["rx_idx"]) if n else 0
        return {
            "frames_ok": n,
            "frames_bad": sum(self.bad.values()),
            "bad_reasons": dict(self.bad),
            "lost_radio_tx_seq": lost_radio,
            "lost_usb_rx_idx": lost_usb,
            "received_pct": round(100.0 * n / (n + lost_radio), 2) if n else 0.0,
        }

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"csi_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(path, **self.arrays())
        return path


def lines_from_serial(port: str, baud: int, duration: Optional[float]) -> Iterator[str]:
    import serial  # imported here so --replay works without pyserial

    end = time.time() + duration if duration else None
    with serial.Serial(port, baud, timeout=1) as ser:
        ser.reset_input_buffer()
        while end is None or time.time() < end:
            raw = ser.readline()
            if raw:
                yield raw.decode("ascii", errors="replace")


def lines_from_file(path: str) -> Iterable[str]:
    with open(path, encoding="ascii", errors="replace") as fh:
        yield from fh


app = typer.Typer(add_completion=False, help="Record the CSI stream from the receiver (RX).")


@app.command()
def record(
    out: Path = typer.Option(..., "--out", help="Output directory (created if missing)."),
    port: Optional[str] = typer.Option(None, help="Serial port of the receiver, e.g. /dev/cu.usbmodem1101."),
    replay: Optional[Path] = typer.Option(
        None, exists=True, dir_okay=False, help="Text file with CSI_DATA lines (for testing without hardware)."),
    baud: int = typer.Option(921600, min=1200, help="Serial baud rate."),
    duration: Optional[float] = typer.Option(None, min=0.1, help="Recording time in seconds (default: until Ctrl+C)."),
) -> None:
    """Read CSI lines from a serial port or a file and save them as .npz."""
    if (port is None) == (replay is None):
        raise typer.BadParameter("give exactly one of --port or --replay")

    lines = lines_from_file(str(replay)) if replay else lines_from_serial(port, baud, duration)
    c = Collector()
    last_report = time.time()
    try:
        for line in lines:
            c.feed(line)
            if port and time.time() - last_report > 5:
                s = c.stats()
                typer.echo(f"ok={s['frames_ok']} bad={s['frames_bad']} lost_radio={s['lost_radio_tx_seq']}", err=True)
                last_report = time.time()
    except KeyboardInterrupt:
        pass

    if not c.frames:
        typer.echo("No valid frames, nothing saved.", err=True)
        typer.echo(str(c.stats()))
        raise typer.Exit(1)
    path = c.save(out)
    typer.echo(f"Saved {path}")
    for k, v in c.stats().items():
        typer.echo(f"  {k}: {v}")


if __name__ == "__main__":
    app()
