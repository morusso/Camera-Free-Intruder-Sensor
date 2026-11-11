#!/usr/bin/env python3
"""Train and track one model (step 3.3).

Every run gets its own folder under ml/runs/ with params.json, history.csv, summary.json and the
best model, plus one line in ml/runs/experiments.csv so runs can be compared.

Example:
    python ml/train.py --model cnn --train day1/aligned.npz day2/aligned.npz \\
        --val day3/aligned.npz --baseline baseline.json --seed 0

Split by session or day (see docs/collection-protocol.md). The autoencoder is trained on the
`empty` windows only; its quality is the AUC of the reconstruction error (empty vs everything else).
"""
from __future__ import annotations

import csv
import inspect
import json
import platform
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import typer

import dataset as ds
import metrics
import models
import preprocess as pp

RUNS_DIR = Path(__file__).with_name("runs")
EXPERIMENT_COLUMNS = ["run_id", "model", "seed", "epochs_run", "best_epoch", "lr", "batch_size", "augment",
                      "class_weights", "n_train", "n_val", "val_macro_f1", "val_accuracy", "val_loss", "val_auc"]


@dataclass
class TrainConfig:
    model: str = "cnn"
    seed: int = 0
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    patience: int = 8
    augment: bool = True
    class_weights: bool = False
    augment_config: ds.AugmentConfig = field(default_factory=ds.AugmentConfig)


def _set_determinism(seed: int) -> None:
    import tensorflow as tf
    from tensorflow import keras
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def _predict(model, data: ds.WindowDataset, batch_size: int = 256):
    """Predictions and labels over a whole dataset, in chunks so memory stays small."""
    outs, ys = [], []
    for x, y in data.iter_batches(batch_size):
        outs.append(model.predict(x, verbose=0))
        ys.append(y)
    return np.concatenate(outs), np.concatenate(ys)


def _ae_scores(model, data: ds.WindowDataset, batch_size: int = 256):
    scores, ys = [], []
    for x, y in data.iter_batches(batch_size):
        scores.append(models.anomaly_scores(model, x))
        ys.append(y)
    return np.concatenate(scores), np.concatenate(ys)


def run_training(cfg: TrainConfig, train: ds.WindowDataset, val: ds.WindowDataset, out_dir: Path,
                 extra_params: Optional[Dict] = None) -> Dict:
    """Train, keep the best epoch, write everything to out_dir. Returns the summary dict."""
    from tensorflow import keras

    _set_determinism(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    is_ae = cfg.model == "autoencoder"
    names = train.class_names
    if is_ae:
        train, val_empty = train.subset(["empty"]), val.subset(["empty"])
    if len(train) == 0 or len(val) == 0:
        raise ValueError("training and validation sets must not be empty")

    builder = models.BUILDERS[cfg.model]
    model = builder() if is_ae else builder(n_classes=len(names))
    model.compile(keras.optimizers.Adam(cfg.lr), "mse" if is_ae else "sparse_categorical_crossentropy")

    weights = None
    if cfg.class_weights and not is_ae:
        weights = {i: float(w) for i, w in enumerate(train.class_weights())}

    params = {
        "config": {**asdict(cfg), "augment_config": asdict(cfg.augment_config)},
        "class_names": names,
        "train_class_counts": {n: int(c) for n, c in zip(names, train.class_counts())},
        "val_class_counts": {n: int(c) for n, c in zip(names, val.class_counts())},
        "model_parameters": int(model.count_params()),
        "model_builder_defaults": {k: v.default for k, v in inspect.signature(builder).parameters.items()
                                   if v.default is not inspect.Parameter.empty},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "tensorflow": __import__("tensorflow").__version__},
        **(extra_params or {}),
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=2, default=str) + "\n")

    rng = np.random.default_rng(cfg.seed)
    aug = cfg.augment_config if cfg.augment else None
    history: List[Dict] = []
    best_score, best_epoch, best_weights, stale = -np.inf, -1, model.get_weights(), 0

    for epoch in range(1, cfg.epochs + 1):
        losses, sizes = [], []
        for x, y in train.iter_batches(cfg.batch_size, shuffle=True, augment=aug, rng=rng):
            if is_ae:
                loss = model.train_on_batch(x, x)
            else:
                loss = model.train_on_batch(x, y, class_weight=weights)
            losses.append(float(loss))
            sizes.append(len(x))
        row = {"epoch": epoch, "train_loss": float(np.average(losses, weights=sizes))}

        if is_ae:
            scores, _ = _ae_scores(model, val_empty)
            row["val_loss"] = float(scores.mean())
            score = -row["val_loss"]
        else:
            probs, y_val = _predict(model, val)
            m = metrics.classification_metrics(y_val, probs, names)
            row.update(val_loss=m["loss"], val_accuracy=m["accuracy"], val_macro_f1=m["macro_f1"])
            score = m["macro_f1"]
        history.append(row)
        typer.echo("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

        if score > best_score + 1e-6:
            best_score, best_epoch, best_weights, stale = score, epoch, model.get_weights(), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                typer.echo(f"No improvement for {cfg.patience} epochs, stopping.")
                break

    model.set_weights(best_weights)
    model.save(out_dir / "model.keras")
    with open(out_dir / "history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    summary: Dict = {"best_epoch": best_epoch, "epochs_run": len(history)}
    if is_ae:
        from sklearn.metrics import roc_auc_score
        scores, y_val = _ae_scores(model, val)
        not_empty = y_val != names.index("empty")
        summary["val_auc"] = float(roc_auc_score(not_empty, scores)) if not_empty.any() and not not_empty.all() else None
        summary["val_loss"] = history[best_epoch - 1]["val_loss"]
    else:
        probs, y_val = _predict(model, val)
        summary["val"] = metrics.classification_metrics(y_val, probs, names)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    _log_experiment(out_dir.parent / "experiments.csv", out_dir.name, cfg, summary, len(train), len(val))
    return summary


def _log_experiment(path: Path, run_id: str, cfg: TrainConfig, summary: Dict, n_train: int, n_val: int) -> None:
    val = summary.get("val", {})
    row = {"run_id": run_id, "model": cfg.model, "seed": cfg.seed, "epochs_run": summary["epochs_run"],
           "best_epoch": summary["best_epoch"], "lr": cfg.lr, "batch_size": cfg.batch_size, "augment": cfg.augment,
           "class_weights": cfg.class_weights, "n_train": n_train, "n_val": n_val,
           "val_macro_f1": val.get("macro_f1", ""), "val_accuracy": val.get("accuracy", ""),
           "val_loss": val.get("loss", summary.get("val_loss", "")), "val_auc": summary.get("val_auc", "") or ""}
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=EXPERIMENT_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


app = typer.Typer(add_completion=False, help="Train one model and record the run.")


@app.command()
def main(
    model: str = typer.Option("cnn", help="cnn, gru or autoencoder."),
    train: List[Path] = typer.Option(..., exists=True, dir_okay=False, help="aligned.npz for training (repeat)."),
    val: List[Path] = typer.Option(..., exists=True, dir_okay=False, help="aligned.npz for validation (repeat)."),
    baseline: Optional[Path] = typer.Option(None, exists=True, help="baseline.json (empty-room statistics)."),
    preprocess_config: Path = typer.Option(pp.CONFIG_PATH, exists=True),
    seed: int = typer.Option(0),
    epochs: int = typer.Option(40, min=1),
    batch_size: int = typer.Option(64, min=1),
    lr: float = typer.Option(1e-3, min=0.0),
    patience: int = typer.Option(8, min=1),
    augment: bool = typer.Option(True, "--augment/--no-augment"),
    class_weights: bool = typer.Option(False, "--class-weights/--no-class-weights"),
    runs_dir: Path = typer.Option(RUNS_DIR, help="Where run folders are created."),
) -> None:
    if model not in models.BUILDERS:
        raise typer.BadParameter(f"model must be one of {sorted(models.BUILDERS)}")
    pcfg = pp.PreprocessConfig.load(preprocess_config)
    bl = pp.Baseline.load(baseline) if baseline else None
    train_ds = ds.load_dataset(train, pcfg, bl)
    val_ds = ds.load_dataset(val, pcfg, bl)

    cfg = TrainConfig(model, seed, epochs, batch_size, lr, patience, augment, class_weights)
    out_dir = runs_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{model}_s{seed}"
    try:
        run_training(cfg, train_ds, val_ds, out_dir, {"train_files": [str(p) for p in train],
                                                      "val_files": [str(p) for p in val]})
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    # The model is only usable together with exactly this preprocessing and baseline.
    shutil.copy(preprocess_config, out_dir / "preprocess_config.json")
    if baseline:
        shutil.copy(baseline, out_dir / "baseline.json")
    typer.echo(f"Run saved to {out_dir}")


if __name__ == "__main__":
    app()
