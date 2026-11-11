"""Windowed dataset for training (step 3.1). Plain numpy, no deep learning framework needed.

A window is `win` consecutive frames (default 128) of `S` preprocessed subcarriers (52):
shape [win, S] (time first). Batches have shape [B, win, S]; for PyTorch Conv1d use
x.transpose(0, 2, 1) to get [B, S, win].

Windows are cut only where the whole window has one class and no uncertain frame.
Memory stays small: the dataset keeps one cleaned array per session and cuts the windows
when a batch is requested.

Split by session or by day, never randomly by window (see docs/collection-protocol.md):
build one dataset from the train sessions, another from validation, another from test.

Example:
    cfg = pp.PreprocessConfig.load()
    bl = pp.Baseline.load("baseline.json")
    train = load_dataset(train_paths, cfg, bl)
    for x, y in train.iter_batches(64, shuffle=True, augment=AugmentConfig(), rng=np.random.default_rng(0)):
        ...
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

import preprocess as pp

WIN = 128
STEP = 32


def window_classes(label: np.ndarray, uncertain: np.ndarray, starts: np.ndarray, win: int,
                   class_names: List[str]) -> np.ndarray:
    """Class name of every window, or "" when the window mixes labels, is unlabeled or uncertain."""
    out = np.full(len(starts), "", dtype=object)
    bad = np.concatenate([[0], np.cumsum(uncertain.astype(np.int64))])
    clean = (bad[starts + win] - bad[starts]) == 0
    for i, name in enumerate(class_names):
        c = np.concatenate([[0], np.cumsum(label == i)])
        out[clean & ((c[starts + win] - c[starts]) == win)] = name
    return out


@dataclass
class AugmentConfig:
    """Each augmentation is applied to a window with probability `p`. Values are in normalized units."""
    p: float = 0.5
    noise_std: float = 0.1
    scale_range: Tuple[float, float] = (0.9, 1.1)
    max_shift: int = 8
    zero_fraction: float = 0.1


def jittered_start(start: int, session_len: int, win: int, rng: np.random.Generator, cfg: AugmentConfig) -> int:
    if cfg.max_shift <= 0 or rng.random() >= cfg.p:
        return start
    return int(np.clip(start + rng.integers(-cfg.max_shift, cfg.max_shift + 1), 0, session_len - win))


def augment(x: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Noise, amplitude scaling and subcarrier dropout on one window [win, S]. Time shift is done by the dataset."""
    x = x.copy()
    if cfg.noise_std > 0 and rng.random() < cfg.p:
        x += rng.normal(0, cfg.noise_std, x.shape).astype(np.float32)
    if rng.random() < cfg.p:
        x *= np.float32(rng.uniform(*cfg.scale_range))
    if cfg.zero_fraction > 0 and rng.random() < cfg.p:
        x[:, rng.random(x.shape[1]) < cfg.zero_fraction] = 0.0
    return x


class WindowDataset:
    def __init__(self, sessions: List[np.ndarray], session_idx: np.ndarray, starts: np.ndarray,
                 labels: np.ndarray, class_names: List[str], win: int = WIN) -> None:
        self.sessions = sessions
        self.session_idx = session_idx
        self.starts = starts
        self.labels = labels
        self.class_names = class_names
        self.win = win

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def n_subcarriers(self) -> int:
        return self.sessions[0].shape[1]

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=len(self.class_names))

    def class_weights(self) -> np.ndarray:
        """Inverse-frequency weights (mean 1 over present classes); 0 for classes with no windows."""
        counts = self.class_counts().astype(np.float64)
        present = counts > 0
        w = np.zeros_like(counts)
        w[present] = len(self) / (present.sum() * counts[present])
        return w.astype(np.float32)

    def subset(self, classes: Sequence[str]) -> "WindowDataset":
        """Keep only windows of the given classes (e.g. ["empty"] for the autoencoder)."""
        ids = [self.class_names.index(c) for c in classes]
        keep = np.isin(self.labels, ids)
        return WindowDataset(self.sessions, self.session_idx[keep], self.starts[keep], self.labels[keep],
                             self.class_names, self.win)

    def get(self, i: int, rng: Optional[np.random.Generator] = None,
            aug: Optional[AugmentConfig] = None) -> np.ndarray:
        x = self.sessions[self.session_idx[i]]
        start = int(self.starts[i])
        if aug is not None and rng is not None:
            start = jittered_start(start, len(x), self.win, rng, aug)
        w = x[start:start + self.win]
        return augment(w, rng, aug) if aug is not None and rng is not None else w

    def batch(self, idx: Sequence[int], rng: Optional[np.random.Generator] = None,
              aug: Optional[AugmentConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
        x = np.stack([self.get(i, rng, aug) for i in idx]).astype(np.float32)
        return x, self.labels[list(idx)].astype(np.int64)

    def iter_batches(self, batch_size: int, shuffle: bool = False, augment: Optional[AugmentConfig] = None,
                     rng: Optional[np.random.Generator] = None,
                     drop_last: bool = False) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if (shuffle or augment is not None) and rng is None:
            raise ValueError("pass an rng (np.random.default_rng(seed)) for shuffling or augmentation")
        order = rng.permutation(len(self)) if shuffle else np.arange(len(self))
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            if drop_last and len(idx) < batch_size:
                return
            yield self.batch(idx, rng, augment)


def load_dataset(paths: Sequence[Path], cfg: pp.PreprocessConfig, baseline: Optional[pp.Baseline] = None,
                 win: int = WIN, step: int = STEP) -> WindowDataset:
    """Build a dataset from aligned.npz files (output of tools/align_labels.py)."""
    sessions, s_idx, starts, labels = [], [], [], []
    class_names: Optional[List[str]] = None
    for n, path in enumerate(paths):
        data = np.load(path)
        names = [str(c) for c in data["class_names"]]
        if class_names is None:
            class_names = names
        elif names != class_names:
            raise ValueError(f"{path}: class names differ from the first session")
        x = pp.preprocess(data["csi"], cfg, baseline)
        st = np.arange(0, len(x) - win + 1, step)
        cls = window_classes(data["label"], data["uncertain"], st, win, names)
        ok = cls != ""
        sessions.append(x)
        s_idx.append(np.full(int(ok.sum()), n, np.int32))
        starts.append(st[ok])
        labels.append(np.array([names.index(c) for c in cls[ok]], dtype=np.int8))
    if class_names is None:
        raise ValueError("no sessions given")
    return WindowDataset(sessions, np.concatenate(s_idx), np.concatenate(starts), np.concatenate(labels),
                         class_names, win)
