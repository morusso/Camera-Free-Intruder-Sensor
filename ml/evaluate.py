#!/usr/bin/env python3
"""Evaluate a trained classifier on test sessions (step 3.4).

Two levels:
  windows  confusion matrix and F1 per class on clean windows (one class, nothing uncertain)
  events   the model runs over the whole continuous recording, decisions are smoothed (k of n
           windows) and counted as alarm events. This gives false alarms per day in the empty
           room and the delay between a person entering and the first alarm.

Delay is measured on the continuous stream because the +-2 s around every class change are
"uncertain" and therefore missing from the window level, and that is exactly where an entry happens.
The label time is when the button was pressed, so it includes the operator's reaction time.

Example:
    python ml/evaluate.py ml/runs/20261112_cnn_s0 --test day4/aligned.npz moved-shelf/aligned.npz
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import typer

import dataset as ds
import metrics
import preprocess as pp

PERSON = ("static_person", "walking")
TARGET_LATENCY_S = 2.0


def k_of_n(flags: np.ndarray, k: int, n: int) -> np.ndarray:
    """Causal smoothing: True where at least k of the last n flags are True."""
    c = np.concatenate([[0], np.cumsum(flags.astype(np.int64))])
    idx = np.arange(1, len(flags) + 1)
    return (c[idx] - c[np.maximum(idx - n, 0)]) >= k


def rising_edges(mask: np.ndarray) -> int:
    return int(np.sum(mask & ~np.concatenate([[False], mask[:-1]])))


def entry_frames(label: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    """Frame index where the label changes from empty straight to a person class."""
    empty = class_names.index("empty")
    person = [class_names.index(c) for c in PERSON]
    return np.flatnonzero((label[:-1] == empty) & np.isin(label[1:], person)) + 1


def detection_latencies(alarm: np.ndarray, starts: np.ndarray, win: int, entries: np.ndarray,
                        sample_rate: float, max_s: float) -> List[Optional[float]]:
    """For every entry: seconds until the first alarm on a window ending at or after it (None = missed)."""
    ends = starts + win - 1
    out: List[Optional[float]] = []
    for t in entries:
        ok = alarm & (ends >= t) & (ends <= t + max_s * sample_rate)
        out.append(float((ends[np.argmax(ok)] - t) / sample_rate) if ok.any() else None)
    return out


def stream_probs(model, x: np.ndarray, win: int, step: int, batch: int = 256):
    starts = np.arange(0, len(x) - win + 1, step)
    probs = [model.predict_on_batch(np.stack([x[i:i + win] for i in starts[s:s + batch]]))
             for s in range(0, len(starts), batch)]
    return starts, (np.concatenate(probs) if probs else np.zeros((0, 0)))


def evaluate_session(model, path: Path, cfg: pp.PreprocessConfig, baseline: Optional[pp.Baseline],
                     person_prob: float, k: int, n: int, win: int, step: int, max_latency_s: float) -> Dict:
    data = np.load(path)
    names = [str(c) for c in data["class_names"]]
    x = pp.preprocess(data["csi"], cfg, baseline)
    starts, probs = stream_probs(model, x, win, step)
    cls = ds.window_classes(data["label"], data["uncertain"], starts, win, names)

    person_idx = [names.index(c) for c in PERSON]
    alarm = k_of_n(probs[:, person_idx].sum(axis=1) > person_prob, k, n)
    step_s = step / cfg.sample_rate_hz
    clean = cls != ""
    return {
        "y_true": np.array([names.index(c) for c in cls[clean]], dtype=np.int64),
        "probs": probs[clean],
        "empty_s": float(np.sum(cls == "empty") * step_s),
        "interference_s": float(np.sum(cls == "interference") * step_s),
        "empty_false_alarms": rising_edges(alarm & (cls == "empty")),
        "interference_alarms": rising_edges(alarm & (cls == "interference")),
        "latencies": detection_latencies(alarm, starts, win, entry_frames(data["label"], names),
                                         cfg.sample_rate_hz, max_latency_s),
        "class_names": names,
    }


def aggregate(parts: List[Dict], settings: Dict) -> Dict:
    names = parts[0]["class_names"]
    y = np.concatenate([p["y_true"] for p in parts])
    probs = np.concatenate([p["probs"] for p in parts])
    empty_h = sum(p["empty_s"] for p in parts) / 3600
    interf_h = sum(p["interference_s"] for p in parts) / 3600
    lat = [v for p in parts for v in p["latencies"]]
    found = np.array([v for v in lat if v is not None])
    false_alarms = sum(p["empty_false_alarms"] for p in parts)
    return {
        "settings": settings,
        "windows": metrics.classification_metrics(y, probs, names),
        "events": {
            "empty_hours": empty_h,
            "false_alarms": false_alarms,
            "false_alarms_per_day": false_alarms / empty_h * 24 if empty_h else None,
            "interference_hours": interf_h,
            "interference_alarms": sum(p["interference_alarms"] for p in parts),
            "interference_alarms_per_hour": sum(p["interference_alarms"] for p in parts) / interf_h if interf_h else None,
            "entries": len(lat),
            "entries_missed": len(lat) - len(found),
            "latency_median_s": float(np.median(found)) if len(found) else None,
            "latency_p95_s": float(np.percentile(found, 95)) if len(found) else None,
            "latency_max_s": float(found.max()) if len(found) else None,
            "entries_within_target": int(np.sum(found <= TARGET_LATENCY_S)) if len(found) else 0,
        },
    }


def format_report(r: Dict) -> str:
    w, e, s = r["windows"], r["events"], r["settings"]
    num = lambda v, fmt="{:.2f}": "n/a" if v is None else fmt.format(v)  # noqa: E731
    lines = [
        f"alarm = person probability > {s['person_prob']}, smoothed {s['k']} of {s['n']} windows",
        "",
        f"windows: accuracy {w['accuracy']:.3f}, macro F1 {w['macro_f1']:.3f}",
        "  F1 per class: " + ", ".join(f"{k} {v:.3f}" for k, v in w["f1_per_class"].items()),
        "  confusion matrix (rows = true, columns = predicted):",
    ]
    lines += ["    " + " ".join(f"{v:6d}" for v in row) for row in w["confusion_matrix"]]
    lines += [
        "",
        f"false alarms, empty room : {e['false_alarms']} in {e['empty_hours']:.2f} h"
        f" = {num(e['false_alarms_per_day'], '{:.1f}')} per day",
        f"alarms during interference: {e['interference_alarms']} in {e['interference_hours']:.2f} h"
        f" = {num(e['interference_alarms_per_hour'], '{:.1f}')} per hour",
        f"entries detected         : {e['entries'] - e['entries_missed']} of {e['entries']}"
        f" (missed {e['entries_missed']})",
        f"detection delay [s]      : median {num(e['latency_median_s'])}, p95 {num(e['latency_p95_s'])},"
        f" max {num(e['latency_max_s'])}",
        f"within {TARGET_LATENCY_S:.0f} s target        : {e['entries_within_target']} of {e['entries']}",
    ]
    return "\n".join(lines)


def load_run(run_dir: Path):
    from tensorflow import keras
    summary = json.loads((run_dir / "params.json").read_text())
    if summary["config"]["model"] == "autoencoder":
        raise ValueError("evaluating the autoencoder needs a threshold; it is handled in the hybrid step")
    model = keras.models.load_model(run_dir / "model.keras")
    cfg = pp.PreprocessConfig.load(run_dir / "preprocess_config.json")
    bl_path = run_dir / "baseline.json"
    return model, cfg, pp.Baseline.load(bl_path) if bl_path.exists() else None


def evaluate_paths(model, cfg, baseline, paths: Sequence[Path], person_prob=0.5, k=3, n=5,
                   win=ds.WIN, step=ds.STEP, max_latency_s=10.0) -> Dict:
    parts = [evaluate_session(model, p, cfg, baseline, person_prob, k, n, win, step, max_latency_s) for p in paths]
    return aggregate(parts, {"person_prob": person_prob, "k": k, "n": n, "win": win, "step": step,
                             "max_latency_s": max_latency_s, "test_files": [str(p) for p in paths]})


app = typer.Typer(add_completion=False, help="Evaluate a trained classifier on test sessions.")


@app.command()
def main(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Run folder from train.py."),
    test: List[Path] = typer.Option(..., exists=True, dir_okay=False, help="aligned.npz test sessions (repeat)."),
    person_prob: float = typer.Option(0.5, min=0.0, max=1.0, help="Alarm threshold on P(person)."),
    k: int = typer.Option(3, min=1), n: int = typer.Option(5, min=1),
    max_latency: float = typer.Option(10.0, min=0.5, help="Later than this an entry counts as missed (s)."),
) -> None:
    if k > n:
        raise typer.BadParameter("k must not exceed n")
    try:
        model, cfg, bl = load_run(run_dir)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    result = evaluate_paths(model, cfg, bl, test, person_prob, k, n, max_latency_s=max_latency)
    typer.echo(format_report(result))
    (run_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    typer.echo(f"\nSaved {run_dir / 'evaluation.json'}")


if __name__ == "__main__":
    app()
