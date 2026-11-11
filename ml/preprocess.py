#!/usr/bin/env python3
"""Offline preprocessing of CSI amplitudes. The firmware (step 4.1) must do the same.

Pipeline, per link:
    1. amplitude from the (imag, real) pairs, keep only the usable subcarriers
    2. Hampel filter along time (removes single-sample spikes)
    3. causal low-pass filter along time (human motion is mostly below 10-20 Hz)
    4. normalize with the statistics of the empty room (a baseline fitted per installation)

All parameters live in preprocess_config.json, so Python and C read the same numbers.

Example (fit the baseline from the empty-room frames of an aligned session):
    python ml/preprocess.py baseline data/raw/2026-11-12_office-empty/aligned.npz --out baseline.json
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

CONFIG_PATH = Path(__file__).with_name("preprocess_config.json")
MAD_TO_SIGMA = 1.4826


@dataclass
class PreprocessConfig:
    n_values: int                 # expected number of int8 values per frame (128 = LLTF only)
    keep_subcarriers: List[int]   # indices into the 64 subcarriers of the LLTF
    hampel_half_window: int       # window is 2*half+1 samples
    hampel_n_sigma: float
    hampel_min_scale: float       # floor for the spread, so a flat stretch does not flag everything
    sample_rate_hz: float
    lowpass_cutoff_hz: float
    lowpass_order: int
    norm_min_std: float           # floor for the baseline std, so a quiet subcarrier does not explode

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "PreprocessConfig":
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


@dataclass
class Baseline:
    mean: np.ndarray
    std: np.ndarray
    n_frames: int

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(
            {"mean": self.mean.tolist(), "std": self.std.tolist(), "n_frames": self.n_frames}, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Baseline":
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["mean"], np.float32), np.array(d["std"], np.float32), d["n_frames"])


def amplitude(csi: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """int8 CSI [T, n_values] (imag, real, ...) -> amplitude [T, n_kept_subcarriers], float32."""
    if csi.ndim != 2 or csi.shape[1] != cfg.n_values:
        raise ValueError(f"expected frames of {cfg.n_values} values, got shape {csi.shape}")
    pairs = csi.astype(np.float32).reshape(len(csi), -1, 2)
    amp = np.hypot(pairs[..., 0], pairs[..., 1])
    return amp[:, cfg.keep_subcarriers]


def hampel(x: np.ndarray, half_window: int, n_sigma: float, min_scale: float) -> np.ndarray:
    """Replace samples that differ from the local median by more than n_sigma robust sigmas.

    Works along axis 0. The window is centered (so the output lags by half_window samples
    in a streaming implementation); the edges are padded by repeating the edge sample.
    """
    padded = np.pad(x, ((half_window, half_window), (0, 0)), mode="edge")
    win = sliding_window_view(padded, 2 * half_window + 1, axis=0)  # [T, S, W]
    med = np.median(win, axis=-1)
    scale = np.maximum(np.median(np.abs(win - med[..., None]), axis=-1) * MAD_TO_SIGMA, min_scale)
    return np.where(np.abs(x - med) > n_sigma * scale, med, x).astype(np.float32)


def lowpass(x: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Causal Butterworth low-pass along axis 0, started from the first sample (no start-up bump)."""
    sos = signal.butter(cfg.lowpass_order, cfg.lowpass_cutoff_hz, fs=cfg.sample_rate_hz, output="sos")
    zi = signal.sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
    y, _ = signal.sosfilt(sos, x, axis=0, zi=zi)
    return y.astype(np.float32)


def clean(csi: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Steps 1-3: amplitude, Hampel, low-pass. Result is in amplitude units, not yet normalized."""
    x = amplitude(csi, cfg)
    x = hampel(x, cfg.hampel_half_window, cfg.hampel_n_sigma, cfg.hampel_min_scale)
    return lowpass(x, cfg)


def fit_baseline(empty_csi: np.ndarray, cfg: PreprocessConfig) -> Baseline:
    """Statistics of the empty room, from a continuous recording with nobody in it."""
    x = clean(empty_csi, cfg)
    return Baseline(x.mean(axis=0), np.maximum(x.std(axis=0), cfg.norm_min_std), len(x))


def normalize(x: np.ndarray, baseline: Baseline) -> np.ndarray:
    """Deviation from the empty room, in units of the empty room's own spread."""
    return ((x - baseline.mean) / baseline.std).astype(np.float32)


def preprocess(csi: np.ndarray, cfg: PreprocessConfig, baseline: Optional[Baseline] = None) -> np.ndarray:
    x = clean(csi, cfg)
    return normalize(x, baseline) if baseline is not None else x


app = typer.Typer(add_completion=False, help="CSI preprocessing tools.")


@app.command()
def baseline(
    aligned: Path = typer.Argument(..., exists=True, dir_okay=False, help="aligned.npz from align_labels.py"),
    out: Path = typer.Option(..., help="Where to write the baseline JSON."),
    config: Path = typer.Option(CONFIG_PATH, exists=True, help="Preprocessing config."),
) -> None:
    """Fit the empty-room baseline from the frames labeled `empty` (and not uncertain)."""
    data = np.load(aligned)
    names = list(data["class_names"])
    mask = (data["label"] == names.index("empty")) & ~data["uncertain"]
    if mask.sum() < 1000:
        typer.echo(f"Error: only {int(mask.sum())} usable empty frames; record a longer empty stretch.", err=True)
        raise typer.Exit(1)

    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    starts, ends = edges[::2], edges[1::2]
    best = np.argmax(ends - starts)
    cfg = PreprocessConfig.load(config)
    bl = fit_baseline(data["csi"][starts[best]:ends[best]], cfg)
    bl.save(out)
    typer.echo(f"Baseline from {bl.n_frames} frames ({bl.n_frames / cfg.sample_rate_hz:.0f} s) saved to {out}")


if __name__ == "__main__":
    app()
