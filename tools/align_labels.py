#!/usr/bin/env python3
"""Attach labels from the labeling page to the recorded CSI frames (step 2.3).

Reads a session folder with csi_*.npz (from csi_reader.py) and labels.csv (from
labeler/server.py) and writes aligned.npz: all CSI arrays plus
    label      int8, class id, or -1 where no label was active
    uncertain  bool, True within +-margin seconds of a class change
and the class_names array that explains the ids.

Use only frames with  (label >= 0) & ~uncertain  for training.

Example:
    python tools/align_labels.py data/raw/2026-11-12_office-walk
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import typer

CLASS_NAMES = ["empty", "static_person", "walking", "interference"]
# Button on the phone -> class (see docs/collection-protocol.md)
BUTTON_TO_CLASS = {
    "empty": "empty",
    "standing": "static_person",
    "walking": "walking",
    "entering": "walking",
    "leaving": "walking",
    "interference": "interference",
}
ADJACENT_S = 0.001  # an end and a start closer than this count as the same instant


@dataclass
class Interval:
    button: str
    t0: float
    t1: float

    @property
    def cls(self) -> int:
        return CLASS_NAMES.index(BUTTON_TO_CLASS[self.button])


def load_intervals(path: Path, last_time: float) -> Tuple[List[Interval], List[str]]:
    """Turn the start/end log into intervals. Returns (intervals, warnings)."""
    intervals: List[Interval] = []
    warnings: List[str] = []
    open_label, open_t = None, 0.0
    with open(path, newline="", encoding="utf-8") as fh:
        for n, row in enumerate(csv.DictReader(fh), start=2):
            label, event, t = row["label"], row["event"], float(row["unix_time"])
            if label not in BUTTON_TO_CLASS:
                raise ValueError(f"{path}:{n}: unknown label '{label}'")
            if event == "start":
                if open_label is not None:
                    warnings.append(f"line {n}: '{label}' started while '{open_label}' was still open; closing it")
                    intervals.append(Interval(open_label, open_t, t))
                open_label, open_t = label, t
            elif event == "end":
                if open_label != label:
                    warnings.append(f"line {n}: end of '{label}' without a matching start, ignored")
                    continue
                intervals.append(Interval(label, open_t, t))
                open_label = None
            else:
                raise ValueError(f"{path}:{n}: unknown event '{event}'")
    if open_label is not None:
        warnings.append(f"'{open_label}' was never ended; closing it at the end of the recording")
        intervals.append(Interval(open_label, open_t, max(last_time, open_t)))
    intervals.sort(key=lambda i: i.t0)
    for a, b in zip(intervals, intervals[1:]):
        if b.t0 < a.t1 - ADJACENT_S:
            raise ValueError(f"labels overlap: '{a.button}' and '{b.button}' around t={b.t0:.3f}")
    return intervals, warnings


def class_change_times(intervals: List[Interval]) -> np.ndarray:
    """Moments where the class changes, or a labeled stretch begins or ends."""
    times = []
    for i, iv in enumerate(intervals):
        prev = intervals[i - 1] if i else None
        nxt = intervals[i + 1] if i + 1 < len(intervals) else None
        if prev is None or iv.t0 - prev.t1 > ADJACENT_S or prev.cls != iv.cls:
            times.append(iv.t0)
        if nxt is None or nxt.t0 - iv.t1 > ADJACENT_S:
            times.append(iv.t1)
    return np.array(sorted(times), dtype=np.float64)


def align(pc_time: np.ndarray, intervals: List[Interval], margin_s: float) -> Tuple[np.ndarray, np.ndarray]:
    label = np.full(len(pc_time), -1, dtype=np.int8)
    for iv in intervals:
        label[(pc_time >= iv.t0) & (pc_time < iv.t1)] = iv.cls

    uncertain = np.zeros(len(pc_time), dtype=bool)
    changes = class_change_times(intervals)
    if len(changes):
        pos = np.searchsorted(changes, pc_time)
        left = np.abs(pc_time - changes[np.clip(pos - 1, 0, len(changes) - 1)])
        right = np.abs(pc_time - changes[np.clip(pos, 0, len(changes) - 1)])
        uncertain = np.minimum(left, right) <= margin_s
    uncertain &= label >= 0  # unlabeled frames are already excluded by label == -1
    return label, uncertain


def load_recordings(session: Path) -> dict:
    files = sorted(session.glob("csi_*.npz"))
    if not files:
        raise ValueError(f"no csi_*.npz files in {session}")
    parts = [dict(np.load(f)) for f in files]
    keys = set(parts[0])
    widths = {p["csi"].shape[1] for p in parts}
    if len(widths) != 1:
        raise ValueError(f"recordings have different CSI lengths: {sorted(widths)}")
    merged = {k: np.concatenate([p[k] for p in parts]) for k in keys if all(k in p for p in parts)}
    order = np.argsort(merged["pc_time"], kind="stable")
    return {k: v[order] for k, v in merged.items()}


def summarize(data: dict, intervals: List[Interval]) -> str:
    label, unc = data["label"], data["uncertain"]
    t = data["pc_time"]
    rate = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0
    lines = [f"frames: {len(t)} over {t[-1] - t[0]:.0f} s (~{rate:.0f} per second)"]
    for i, name in enumerate(CLASS_NAMES):
        total = int(np.sum(label == i))
        good = int(np.sum((label == i) & ~unc))
        lines.append(f"  {name:14s} {total:7d} frames, {good:7d} usable (~{good / rate if rate else 0:.0f} s)")
    lines.append(f"  {'unlabeled':14s} {int(np.sum(label < 0)):7d} frames")
    lines.append(f"  {'uncertain':14s} {int(np.sum(unc)):7d} frames (near a class change)")
    return "\n".join(lines)


app = typer.Typer(add_completion=False, help="Attach class labels to recorded CSI frames.")


@app.command()
def main(
    session: Path = typer.Argument(..., exists=True, file_okay=False, help="Session folder."),
    margin: float = typer.Option(2.0, min=0.0, help="Seconds around a class change marked uncertain."),
    offset: float = typer.Option(
        0.0, help="Seconds added to all label times (use if the labels are consistently early or late)."),
    out: Path = typer.Option(None, help="Output file (default: <session>/aligned.npz)."),
) -> None:
    labels_csv = session / "labels.csv"
    if not labels_csv.exists():
        typer.echo(f"Error: {labels_csv} not found", err=True)
        raise typer.Exit(1)
    try:
        data = load_recordings(session)
        intervals, warnings = load_intervals(labels_csv, float(data["pc_time"][-1]))
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    for iv in intervals:
        iv.t0 += offset
        iv.t1 += offset
    for w in warnings:
        typer.echo(f"Warning: {w}", err=True)

    t_lo, t_hi = float(data["pc_time"][0]), float(data["pc_time"][-1])
    for iv in intervals:
        if iv.t1 < t_lo or iv.t0 > t_hi:
            typer.echo(f"Warning: '{iv.button}' ({iv.t0:.0f}-{iv.t1:.0f}) is outside the recording "
                       f"({t_lo:.0f}-{t_hi:.0f}). Is the computer clock right?", err=True)

    data["label"], data["uncertain"] = align(data["pc_time"], intervals, margin)
    data["class_names"] = np.array(CLASS_NAMES)
    target = out or session / "aligned.npz"
    np.savez_compressed(target, **data)
    typer.echo(summarize(data, intervals))
    typer.echo(f"Saved {target}")


if __name__ == "__main__":
    app()
