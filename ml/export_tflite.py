#!/usr/bin/env python3
"""Quantize a trained classifier to int8 TFLite and compare it with the float model (step 3.6).

The representative data (used to pick the int8 scales) are training windows. Input and output of the
TFLite model are int8; the scale and zero point are saved in quantization.json, because the firmware
needs them to convert the normalized window to int8 and to read the class scores.

Target: macro F1 drops by less than 2 percentage points.

Example:
    python ml/export_tflite.py ml/runs/20261112_cnn_s0 --train day1/aligned.npz --test day4/aligned.npz
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import typer

import dataset as ds
import metrics
import preprocess as pp

MAX_DROP_PP = 2.0


def convert_int8(model, representative: np.ndarray) -> bytes:
    """Full-integer quantization with int8 input and output."""
    import tensorflow as tf

    def gen():
        for w in representative:
            yield [w[None].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


class Int8Model:
    """Runs a quantized TFLite classifier on float windows and returns float class probabilities."""

    def __init__(self, tflite_bytes: bytes) -> None:
        import tensorflow as tf
        self.interp = tf.lite.Interpreter(model_content=tflite_bytes)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.in_scale, self.in_zero = self.inp["quantization"]
        self.out_scale, self.out_zero = self.out["quantization"]

    def quantize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.round(x / self.in_scale + self.in_zero), -128, 127).astype(np.int8)

    def saturation(self, x: np.ndarray) -> float:
        """Share of input values that do not fit into int8 and get clipped."""
        raw = np.round(x / self.in_scale + self.in_zero)
        return float(np.mean((raw < -128) | (raw > 127)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = []
        for w in self.quantize(x):
            self.interp.set_tensor(self.inp["index"], w[None])
            self.interp.invoke()
            out.append(self.interp.get_tensor(self.out["index"])[0])
        return (np.array(out, dtype=np.float32) - self.out_zero) * self.out_scale

    def quantization_info(self) -> Dict:
        return {"input": {"scale": float(self.in_scale), "zero_point": int(self.in_zero),
                          "shape": [int(s) for s in self.inp["shape"]]},
                "output": {"scale": float(self.out_scale), "zero_point": int(self.out_zero),
                           "shape": [int(s) for s in self.out["shape"]]}}


def sample_indices(n_total: int, n_max: int, seed: int) -> np.ndarray:
    if n_total <= n_max:
        return np.arange(n_total)
    return np.sort(np.random.default_rng(seed).choice(n_total, n_max, replace=False))


def compare(model, q: Int8Model, data: ds.WindowDataset, max_windows: int, seed: int = 0) -> Dict:
    idx = sample_indices(len(data), max_windows, seed)
    x, y = data.batch(idx)
    pf = model.predict(x, batch_size=256, verbose=0)
    pq = q.predict(x)
    mf = metrics.classification_metrics(y, pf, data.class_names)
    mq = metrics.classification_metrics(y, pq, data.class_names)
    return {
        "windows": int(len(idx)),
        "float": {k: mf[k] for k in ("accuracy", "macro_f1")},
        "int8": {k: mq[k] for k in ("accuracy", "macro_f1")},
        "macro_f1_drop_pp": 100.0 * (mf["macro_f1"] - mq["macro_f1"]),
        "accuracy_drop_pp": 100.0 * (mf["accuracy"] - mq["accuracy"]),
        "same_class_as_float": float(np.mean(pf.argmax(1) == pq.argmax(1))),
        "input_saturation": q.saturation(x),
        "int8_confusion_matrix": mq["confusion_matrix"],
    }


def format_report(size_bytes: int, info: Dict, r: Dict) -> str:
    ok = r["macro_f1_drop_pp"] < MAX_DROP_PP
    return "\n".join([
        f"model_int8.tflite : {size_bytes / 1024:.1f} KB",
        f"input  quantization: scale {info['input']['scale']:.5f}, zero point {info['input']['zero_point']}",
        f"output quantization: scale {info['output']['scale']:.5f}, zero point {info['output']['zero_point']}",
        f"compared on {r['windows']} windows",
        f"  float : accuracy {r['float']['accuracy']:.3f}, macro F1 {r['float']['macro_f1']:.3f}",
        f"  int8  : accuracy {r['int8']['accuracy']:.3f}, macro F1 {r['int8']['macro_f1']:.3f}",
        f"  macro F1 drop {r['macro_f1_drop_pp']:.2f} pp   [{'PASS' if ok else 'FAIL'}: < {MAX_DROP_PP:.0f} pp]",
        f"  same class as float model: {100 * r['same_class_as_float']:.1f} %",
        f"  input values clipped by int8: {100 * r['input_saturation']:.2f} %",
    ])


app = typer.Typer(add_completion=False, help="Quantize a trained classifier to int8 TFLite and compare.")


@app.command()
def main(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Run folder from train.py."),
    train: List[Path] = typer.Option(..., exists=True, dir_okay=False,
                                     help="aligned.npz whose windows pick the int8 scales (repeat)."),
    test: List[Path] = typer.Option(..., exists=True, dir_okay=False,
                                    help="aligned.npz for the float vs int8 comparison (repeat)."),
    n_representative: int = typer.Option(500, min=50, help="Windows used to calibrate the scales."),
    max_windows: int = typer.Option(5000, min=100, help="Cap on the windows used for the comparison."),
    seed: int = typer.Option(0),
) -> None:
    from tensorflow import keras

    params = json.loads((run_dir / "params.json").read_text())
    if params["config"]["model"] == "autoencoder":
        typer.echo("Error: only classifiers are supported here.", err=True)
        raise typer.Exit(1)
    model = keras.models.load_model(run_dir / "model.keras")
    cfg = pp.PreprocessConfig.load(run_dir / "preprocess_config.json")
    bl_path = run_dir / "baseline.json"
    bl: Optional[pp.Baseline] = pp.Baseline.load(bl_path) if bl_path.exists() else None

    train_ds = ds.load_dataset(train, cfg, bl)
    rep, _ = train_ds.batch(sample_indices(len(train_ds), n_representative, seed))
    tflite = convert_int8(model, rep)
    (run_dir / "model_int8.tflite").write_bytes(tflite)

    q = Int8Model(tflite)
    result = compare(model, q, ds.load_dataset(test, cfg, bl), max_windows, seed)
    info = q.quantization_info()
    (run_dir / "quantization.json").write_text(json.dumps({**info, "comparison": result}, indent=2) + "\n")
    typer.echo(format_report(len(tflite), info, result))
    if result["macro_f1_drop_pp"] >= MAX_DROP_PP:
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
