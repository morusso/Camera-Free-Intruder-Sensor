#!/usr/bin/env python3
"""Baseline presence detector without ML (step 2.7). The model has to beat this.

Feature: variance of the (cleaned) amplitude over a 1-2 s window, averaged over subcarriers.
Decision: feature > threshold. The threshold is chosen on the validation sessions.

Task: person present (static_person or walking) versus empty room. Interference windows
are reported separately, because they are what fools a simple variance detector.

Example:
    python ml/baseline.py --val data/raw/day1/aligned.npz --test data/raw/day3/aligned.npz --roc roc.png
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import typer
from sklearn.metrics import roc_auc_score, roc_curve

import preprocess as pp
from dataset import window_classes

Session = Tuple[np.ndarray, np.ndarray]  # (score per window, class name per window; "" = not usable)


def window_scores(x: np.ndarray, win: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    """Variance over time in each window, averaged over subcarriers. Returns (window starts, scores)."""
    if len(x) < win:
        return np.zeros(0, int), np.zeros(0)
    x = x.astype(np.float64)
    pad = np.zeros((1, x.shape[1]))
    s1, s2 = np.vstack([pad, np.cumsum(x, 0)]), np.vstack([pad, np.cumsum(x * x, 0)])
    starts = np.arange(0, len(x) - win + 1, step)
    m1 = (s1[starts + win] - s1[starts]) / win
    m2 = (s2[starts + win] - s2[starts]) / win
    return starts, np.maximum(m2 - m1 * m1, 0).mean(axis=1)


def load_session(path: Path, cfg: pp.PreprocessConfig, win_s: float, step_s: float) -> Session:
    data = np.load(path)
    x = pp.clean(data["csi"], cfg)  # filters need the whole continuous recording
    win, step = int(win_s * cfg.sample_rate_hz), int(step_s * cfg.sample_rate_hz)
    starts, score = window_scores(x, win, step)
    classes = window_classes(data["label"], data["uncertain"], starts, win, [str(n) for n in data["class_names"]])
    return score, classes


def choose_threshold(sessions: List[Session]) -> float:
    """Threshold with the best sensitivity + specificity on empty vs person windows."""
    score = np.concatenate([s for s, _ in sessions])
    cls = np.concatenate([c for _, c in sessions])
    keep = np.isin(cls, ["empty", "static_person", "walking"])
    y = cls[keep] != "empty"
    if y.all() or not y.any():
        raise ValueError("the validation sessions need both empty and person windows")
    fpr, tpr, thr = roc_curve(y, score[keep])
    return float(thr[np.argmax(tpr - fpr)])


def false_alarm_events(score: np.ndarray, cls: np.ndarray, thr: float) -> int:
    """Number of separate runs of positive windows inside an empty stretch."""
    pos = (score > thr) & (cls == "empty")
    return int(np.sum(pos & ~np.concatenate([[False], pos[:-1]])))


def evaluate(sessions: List[Session], thr: float, step_s: float) -> Dict[str, float]:
    score = np.concatenate([s for s, _ in sessions])
    cls = np.concatenate([c for _, c in sessions])
    rate = lambda name: float(np.mean(score[cls == name] > thr)) if np.any(cls == name) else float("nan")  # noqa: E731
    person = np.isin(cls, ["static_person", "walking"])
    empty = cls == "empty"
    empty_hours = int(empty.sum()) * step_s / 3600
    events = sum(false_alarm_events(s, c, thr) for s, c in sessions)
    keep = person | empty
    res = {
        "threshold": thr,
        "windows_empty": int(empty.sum()),
        "windows_person": int(person.sum()),
        "empty_hours": empty_hours,
        "sensitivity": float(np.mean(score[person] > thr)) if person.any() else float("nan"),
        "sensitivity_static_person": rate("static_person"),
        "sensitivity_walking": rate("walking"),
        "specificity_empty": 1.0 - rate("empty"),
        "false_alarm_events": events,
        "false_alarms_per_day": events / empty_hours * 24 if empty_hours else float("nan"),
        "false_positive_rate_interference": rate("interference"),
        "auc": float(roc_auc_score(person[keep], score[keep])) if person.any() and empty.any() else float("nan"),
    }
    return res


def format_report(r: Dict[str, float]) -> str:
    pct = lambda v: f"{100 * v:.1f} %"  # noqa: E731
    return "\n".join([
        f"threshold                       : {r['threshold']:.3f}",
        f"windows (empty / person)        : {r['windows_empty']} / {r['windows_person']}",
        f"empty-room time evaluated       : {r['empty_hours']:.2f} h",
        f"sensitivity (person detected)   : {pct(r['sensitivity'])}"
        f"   [static {pct(r['sensitivity_static_person'])}, walking {pct(r['sensitivity_walking'])}]",
        f"specificity (empty stays quiet) : {pct(r['specificity_empty'])}",
        f"false alarms in empty room      : {r['false_alarm_events']} events"
        f" = {r['false_alarms_per_day']:.1f} per day",
        f"interference triggers alarm     : {pct(r['false_positive_rate_interference'])} of windows",
        f"AUC (empty vs person)           : {r['auc']:.3f}",
    ])


def plot_roc(sessions: List[Session], thr: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    score = np.concatenate([s for s, _ in sessions])
    cls = np.concatenate([c for _, c in sessions])
    keep = np.isin(cls, ["empty", "static_person", "walking"])
    fpr, tpr, _ = roc_curve(cls[keep] != "empty", score[keep])
    op = np.mean(score[keep & (cls != "empty")] > thr), np.mean(score[cls == "empty"] > thr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "k:", linewidth=0.8)
    ax.plot(op[1], op[0], "ro", label="chosen threshold")
    ax.set_xlabel("false positive rate (empty windows)")
    ax.set_ylabel("sensitivity (person windows)")
    ax.set_title("Baseline ROC (variance detector)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


app = typer.Typer(add_completion=False, help="Variance-threshold baseline for presence detection.")


@app.command()
def main(
    val: List[Path] = typer.Option(..., exists=True, dir_okay=False, help="aligned.npz for choosing the threshold (repeat)."),
    test: List[Path] = typer.Option(..., exists=True, dir_okay=False, help="aligned.npz for the final numbers (repeat)."),
    window: float = typer.Option(1.5, min=0.5, max=5.0, help="Window length in seconds."),
    step: float = typer.Option(0.5, min=0.1, help="Step between windows in seconds."),
    config: Path = typer.Option(pp.CONFIG_PATH, exists=True, help="Preprocessing config."),
    roc: Optional[Path] = typer.Option(None, help="Save the test ROC curve as PNG."),
) -> None:
    cfg = pp.PreprocessConfig.load(config)
    val_s = [load_session(p, cfg, window, step) for p in val]
    test_s = [load_session(p, cfg, window, step) for p in test]
    try:
        thr = choose_threshold(val_s)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo("Validation (threshold chosen here):")
    typer.echo(format_report(evaluate(val_s, thr, step)))
    typer.echo("\nTest (numbers to beat):")
    typer.echo(format_report(evaluate(test_s, thr, step)))
    if roc:
        plot_roc(test_s, thr, roc)
        typer.echo(f"\nSaved {roc}")


if __name__ == "__main__":
    app()
