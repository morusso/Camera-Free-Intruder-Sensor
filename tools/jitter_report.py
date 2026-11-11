#!/usr/bin/env python3
"""Sampling quality report for a recording made with csi_reader.py (step 1.7).

Reports the distribution of gaps between received frames, the jitter, and the
loss rate, and checks the Stage 1 acceptance criteria.

Examples:
    python tools/jitter_report.py data/raw/test/csi_20261111_124015.npz
    python tools/jitter_report.py rec.npz --markdown --plot hist.png
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import typer

import csi_reader

MIN_RECEIVED_PCT = 95.0
MAX_P99_JITTER_MS = 20.0
PERCENTILES = (50, 95, 99)


def pct_dict(values: np.ndarray) -> Dict[str, float]:
    out = {f"p{p}": float(np.percentile(values, p)) for p in PERCENTILES}
    out["max"] = float(values.max())
    return out


def compute_report(data, nominal_hz: Optional[float] = None) -> dict:
    """Build the report from the arrays stored by csi_reader (dev_ts_us, tx_seq, rx_idx)."""
    ts = np.asarray(data["dev_ts_us"], dtype=np.int64)
    if len(ts) < 3:
        raise ValueError("need at least 3 frames")

    seq = np.asarray(data["tx_seq"], dtype=np.int64)
    gaps_ms = np.diff(ts) / 1000.0
    adjacent = (np.diff(seq) % csi_reader.SEQ_WRAP) == 1
    if not adjacent.any():
        adjacent = np.ones(len(gaps_ms), bool)
    valid = gaps_ms > 0
    gaps_ms = gaps_ms[valid]
    adjacent = adjacent[valid]
    nominal_ms = 1000.0 / nominal_hz if nominal_hz else float(np.median(gaps_ms[adjacent]))
    jitter_ms = np.abs(gaps_ms[adjacent] - nominal_ms)

    lost_radio = csi_reader.count_missing(np.asarray(data["tx_seq"]))
    lost_usb = csi_reader.count_missing(np.asarray(data["rx_idx"]))
    received = len(ts)
    received_pct = 100.0 * received / (received + lost_radio)

    report = {
        "frames": received,
        "duration_s": float((ts[-1] - ts[0]) / 1e6),
        "nominal_ms": nominal_ms,
        "gap_ms": pct_dict(gaps_ms),
        "jitter_ms": pct_dict(jitter_ms),
        "lost_radio": lost_radio,
        "lost_usb": lost_usb,
        "received_pct": received_pct,
        "gaps_ms": gaps_ms,
    }
    report["pass_received"] = received_pct >= MIN_RECEIVED_PCT
    report["pass_jitter"] = report["jitter_ms"]["p99"] < MAX_P99_JITTER_MS
    return report


def format_text(r: dict) -> str:
    g, j = r["gap_ms"], r["jitter_ms"]
    mark = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    return "\n".join([
        f"frames received : {r['frames']} in {r['duration_s']:.1f} s",
        f"nominal gap     : {r['nominal_ms']:.2f} ms",
        f"gap    [ms]     : p50 {g['p50']:.2f}  p95 {g['p95']:.2f}  p99 {g['p99']:.2f}  max {g['max']:.2f}",
        f"jitter [ms]     : p50 {j['p50']:.2f}  p95 {j['p95']:.2f}  p99 {j['p99']:.2f}  max {j['max']:.2f}",
        f"lost over radio : {r['lost_radio']} (tx_seq gaps)",
        f"lost over USB   : {r['lost_usb']} (rx_idx gaps)",
        f"received        : {r['received_pct']:.2f} %   [{mark(r['pass_received'])}: >= {MIN_RECEIVED_PCT:.0f} %]",
        f"p99 jitter      : {j['p99']:.2f} ms   [{mark(r['pass_jitter'])}: < {MAX_P99_JITTER_MS:.0f} ms]",
    ])


def format_markdown(r: dict) -> str:
    g, j = r["gap_ms"], r["jitter_ms"]
    return "\n".join([
        "| Metric | p50 | p95 | p99 | max |",
        "|---|---|---|---|---|",
        f"| Gap between frames [ms] | {g['p50']:.2f} | {g['p95']:.2f} | {g['p99']:.2f} | {g['max']:.2f} |",
        f"| Jitter, \\|gap - {r['nominal_ms']:.2f}\\|, adjacent frames only [ms] | {j['p50']:.2f} | {j['p95']:.2f} | {j['p99']:.2f} | {j['max']:.2f} |",
        "",
        f"- Frames received: {r['frames']} in {r['duration_s']:.1f} s",
        f"- Lost over radio (tx_seq gaps): {r['lost_radio']}",
        f"- Lost over USB (rx_idx gaps): {r['lost_usb']}",
        f"- Received: {r['received_pct']:.2f} % (criterion: >= {MIN_RECEIVED_PCT:.0f} %)",
        f"- p99 jitter: {j['p99']:.2f} ms (criterion: < {MAX_P99_JITTER_MS:.0f} ms)",
    ])


def plot_histogram(r: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gaps = r["gaps_ms"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gaps, bins=np.linspace(0, min(gaps.max(), 5 * r["nominal_ms"]), 80))
    ax.set_yscale("log")
    ax.axvline(r["gap_ms"]["p99"], color="r", linestyle="--", label=f"p99 = {r['gap_ms']['p99']:.1f} ms")
    ax.set_xlabel("gap between frames [ms]")
    ax.set_ylabel("count (log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


app = typer.Typer(add_completion=False, help="Sampling jitter and loss report for a CSI recording.")


@app.command()
def main(
    recording: Path = typer.Argument(..., exists=True, dir_okay=False, help=".npz file from csi_reader.py"),
    nominal_hz: Optional[float] = typer.Option(
        None, min=0.1, help="Expected frame rate. Default: the median gap is used as nominal."),
    markdown: bool = typer.Option(False, "--markdown", help="Print a Markdown table for docs/."),
    plot: Optional[Path] = typer.Option(None, help="Save a histogram of the gaps as PNG."),
) -> None:
    try:
        report = compute_report(np.load(recording), nominal_hz)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(format_markdown(report) if markdown else format_text(report))
    if plot:
        plot_histogram(report, plot)
        typer.echo(f"Saved {plot}", err=True)
    if not (report["pass_received"] and report["pass_jitter"]):
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
