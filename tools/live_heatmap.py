#!/usr/bin/env python3
"""Live view of the CSI stream: a heatmap (subcarrier x time) and one subcarrier's amplitude.

Examples:
    python tools/live_heatmap.py --port /dev/cu.usbmodem1101
    python tools/live_heatmap.py --replay recording.txt
    python tools/live_heatmap.py --demo                      # synthetic data, no hardware
    python tools/live_heatmap.py --demo --save demo.png --duration 30
"""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Iterator, Optional, Tuple

import numpy as np
import typer

import csi_reader

TS_WRAP = 2 ** 32
N_DEMO_SUBCARRIERS = 64
DEMO_RATE_HZ = 100


def amplitudes(csi: np.ndarray) -> np.ndarray:
    """CSI int8 values (imag, real, imag, real, ...) -> amplitude per subcarrier."""
    n = len(csi) // 2 * 2
    pairs = csi[:n].astype(np.float32).reshape(-1, 2)
    return np.hypot(pairs[:, 0], pairs[:, 1])


class ClockUnwrapper:
    """Turns the wrapping uint32 microsecond timestamp into seconds since the first frame."""

    def __init__(self) -> None:
        self._first: Optional[int] = None
        self._last = 0
        self._offset = 0

    def feed(self, ts_us: int) -> float:
        if self._first is None:
            self._first = ts_us
        elif ts_us < self._last - TS_WRAP // 2:
            self._offset += TS_WRAP
        self._last = ts_us
        return (ts_us + self._offset - self._first) / 1e6


class Window:
    """Keeps the last `window_s` seconds of (time, amplitudes)."""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self.t: Deque[float] = deque()
        self.a: Deque[np.ndarray] = deque()
        self.n_sub: Optional[int] = None

    def add(self, t: float, amp: np.ndarray) -> None:
        if self.n_sub is None:
            self.n_sub = len(amp)
        elif len(amp) != self.n_sub:
            return
        self.t.append(t)
        self.a.append(amp)
        while self.t and self.t[-1] - self.t[0] > self.window_s:
            self.t.popleft()
            self.a.popleft()

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (times, matrix) with matrix shape (n_subcarriers, n_frames)."""
        if not self.t:
            return np.zeros(0), np.zeros((0, 0))
        return np.array(self.t), np.stack(self.a).T


def demo_source(duration_s: float, seed: int = 0) -> Iterator[Tuple[float, np.ndarray]]:
    """Synthetic stream: empty room, then a person walking (middle third), then empty again."""
    rng = np.random.default_rng(seed)
    base = 40 + 15 * np.sin(np.linspace(0, 3 * np.pi, N_DEMO_SUBCARRIERS))
    phase = rng.uniform(0, 2 * np.pi, N_DEMO_SUBCARRIERS)
    for k in range(int(duration_s * DEMO_RATE_HZ)):
        t = k / DEMO_RATE_HZ
        walking = duration_s / 3 <= t < 2 * duration_s / 3
        amp = base + rng.normal(0, 0.5, N_DEMO_SUBCARRIERS)
        if walking:
            amp = amp + 12 * np.sin(2 * np.pi * 1.5 * t + phase) + rng.normal(0, 1.5, N_DEMO_SUBCARRIERS)
        yield t, amp.astype(np.float32)


def stream_source(lines) -> Iterator[Tuple[float, np.ndarray]]:
    clock = ClockUnwrapper()
    for line in lines:
        try:
            frame = csi_reader.parse_line(line)
        except ValueError:
            continue
        yield clock.feed(frame.local_timestamp), amplitudes(frame.csi)


def paced(source: Iterator[Tuple[float, np.ndarray]], speed: float) -> Iterator[Tuple[float, np.ndarray]]:
    """Slows a pre-recorded source down to real time (times `speed`)."""
    start = time.time()
    for t, amp in source:
        delay = t / speed - (time.time() - start)
        if delay > 0:
            time.sleep(delay)
        yield t, amp


def draw(fig, axes, window: Window, subcarrier: int) -> None:
    times, matrix = window.snapshot()
    ax_map, ax_line = axes
    ax_map.clear()
    ax_line.clear()
    if matrix.size == 0:
        return
    t0, t1 = times[0], max(times[-1], times[0] + 1e-3)
    lo, hi = np.percentile(matrix, [2, 98])
    ax_map.imshow(matrix, aspect="auto", origin="lower", extent=[t0, t1, 0, matrix.shape[0]],
                  vmin=lo, vmax=max(hi, lo + 1e-3), cmap="viridis", interpolation="nearest")
    ax_map.set_ylabel("subcarrier")
    ax_map.set_title("CSI amplitude")
    sc = min(subcarrier, matrix.shape[0] - 1)
    ax_line.plot(times, matrix[sc], linewidth=0.8)
    ax_line.set_xlim(t0, t1)
    ax_line.set_xlabel("time [s]")
    ax_line.set_ylabel(f"amplitude, subcarrier {sc}")
    ax_line.grid(alpha=0.3)


app = typer.Typer(add_completion=False, help="Live CSI heatmap and single-subcarrier amplitude plot.")


@app.command()
def main(
    port: Optional[str] = typer.Option(None, help="Serial port of the receiver."),
    replay: Optional[Path] = typer.Option(None, exists=True, dir_okay=False, help="Text file with CSI_DATA lines."),
    demo: bool = typer.Option(False, "--demo", help="Use synthetic data (no hardware needed)."),
    baud: int = typer.Option(921600, min=1200),
    window_s: float = typer.Option(10.0, "--window", min=1.0, help="Seconds shown on screen."),
    subcarrier: int = typer.Option(20, min=0, help="Subcarrier shown in the lower plot."),
    speed: float = typer.Option(1.0, min=0.1, help="Playback speed for --replay and --demo."),
    save: Optional[Path] = typer.Option(None, help="Do not open a window; save a PNG of the whole run."),
    duration: float = typer.Option(30.0, min=1.0, help="Length of the run for --demo / --save."),
) -> None:
    if sum([port is not None, replay is not None, demo]) != 1:
        raise typer.BadParameter("give exactly one of --port, --replay or --demo")

    if save:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if demo:
        source = demo_source(duration)
    elif replay:
        source = stream_source(csi_reader.lines_from_file(str(replay)))
    else:
        source = stream_source(csi_reader.lines_from_serial(port, baud, None))

    # With --save the whole run is drawn at once, so the window must cover it.
    window = Window(duration if save else window_s)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]})
    fig.tight_layout()

    if save:
        for t, amp in source:
            window.add(t, amp)
        draw(fig, axes, window, subcarrier)
        fig.tight_layout()
        fig.savefig(save, dpi=120)
        typer.echo(f"Saved {save}")
        return

    lock = threading.Lock()

    def reader() -> None:
        src = source if port else paced(source, speed)
        for t, amp in src:
            with lock:
                window.add(t, amp)

    threading.Thread(target=reader, daemon=True).start()

    def update(_frame):
        with lock:
            draw(fig, axes, window, subcarrier)

    anim = FuncAnimation(fig, update, interval=100, cache_frame_data=False)  # noqa: F841 (keep a reference)
    plt.show()


if __name__ == "__main__":
    app()
