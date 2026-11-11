"""Classification metrics shared by training and evaluation."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: List[str]) -> Dict:
    """Accuracy, cross-entropy, per-class F1 and macro F1 (over classes that occur), confusion matrix.

    Rows of the confusion matrix are the true class, columns the predicted class.
    """
    k = len(class_names)
    pred = probs.argmax(axis=1)
    f1 = f1_score(y_true, pred, labels=list(range(k)), average=None, zero_division=0)
    present = np.bincount(y_true, minlength=k) > 0
    p = np.clip(probs[np.arange(len(y_true)), y_true], 1e-7, 1.0)
    return {
        "accuracy": float(np.mean(pred == y_true)),
        "loss": float(-np.mean(np.log(p))),
        "macro_f1": float(f1[present].mean()) if present.any() else float("nan"),
        "f1_per_class": {name: float(f) for name, f in zip(class_names, f1)},
        "confusion_matrix": confusion_matrix(y_true, pred, labels=list(range(k))).tolist(),
    }
