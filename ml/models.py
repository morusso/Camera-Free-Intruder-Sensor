"""Models to compare (step 3.2). All take windows of shape [win, n_subcarriers] (time first).

    build_cnn          1D-CNN, convolution over time, subcarriers as channels. Main candidate.
    build_gru          GRU on a 4x time-subsampled window. Alternative.
    build_autoencoder  trained on the empty room only; the reconstruction error is the anomaly score.

Everything is kept small because the model must run on the ESP32-S3 (int8, tensor arena <= 80 KB).
Only layers that convert cleanly to TFLite are used (Conv1D, BatchNorm, ReLU, pooling, Dense, UpSampling1D).
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TensorFlow start-up messages

import numpy as np  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers, regularizers  # noqa: E402

N_CLASSES = 4
WIN = 128
N_SUB = 52


def build_cnn(n_classes: int = N_CLASSES, win: int = WIN, n_sub: int = N_SUB,
              filters=(32, 48, 64), kernels=(5, 5, 3), dropout: float = 0.2, l2: float = 1e-4) -> keras.Model:
    reg = regularizers.l2(l2)
    x = inp = keras.Input((win, n_sub), name="window")
    for f, k in zip(filters, kernels):
        x = layers.Conv1D(f, k, strides=2, padding="same", use_bias=False, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", kernel_regularizer=reg, name="class")(x)
    return keras.Model(inp, out, name="cnn")


def build_gru(n_classes: int = N_CLASSES, win: int = WIN, n_sub: int = N_SUB,
              units: int = 48, subsample: int = 4, dropout: float = 0.2) -> keras.Model:
    x = inp = keras.Input((win, n_sub), name="window")
    x = layers.AveragePooling1D(subsample)(x)  # motion is mostly < 10 Hz, 25 Hz is plenty
    x = layers.GRU(units)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", name="class")(x)
    return keras.Model(inp, out, name="gru")


def build_autoencoder(win: int = WIN, n_sub: int = N_SUB, filters=(32, 16, 8)) -> keras.Model:
    """Encoder: three stride-2 convolutions (128 -> 16 time steps). Decoder mirrors it with upsampling."""
    x = inp = keras.Input((win, n_sub), name="window")
    for f in filters:
        x = layers.Conv1D(f, 5, strides=2, padding="same", activation="relu")(x)
    for f in reversed(filters[:-1]):
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(f, 5, padding="same", activation="relu")(x)
    x = layers.UpSampling1D(2)(x)
    out = layers.Conv1D(n_sub, 5, padding="same", name="reconstruction")(x)
    return keras.Model(inp, out, name="autoencoder")


def anomaly_scores(autoencoder: keras.Model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Mean squared reconstruction error per window. High = does not look like the empty room."""
    rec = autoencoder.predict(x, batch_size=batch_size, verbose=0)
    return np.mean((x - rec) ** 2, axis=(1, 2))


def peak_activation_bytes(model: keras.Model) -> int:
    """Rough tensor arena estimate for an int8 model: the largest (input + output) of any single layer.

    Real TFLite Micro planning can differ (it reuses buffers, but adds scratch space), so treat this
    as a sanity check, not as the final number (that is measured in step 3.7).
    """
    def size(shape) -> int:
        return int(np.prod([d for d in shape if d is not None]))

    peak = 0
    for layer in model.layers:
        ins = layer.input if isinstance(layer.input, list) else [layer.input]
        peak = max(peak, sum(size(t.shape) for t in ins) + size(layer.output.shape))
    return peak


def summary_line(model: keras.Model) -> str:
    return (f"{model.name}: {model.count_params()} parameters (~{model.count_params() / 1024:.1f} KB as int8), "
            f"activations peak ~{peak_activation_bytes(model) / 1024:.1f} KB as int8")


BUILDERS = {"cnn": build_cnn, "gru": build_gru, "autoencoder": build_autoencoder}


if __name__ == "__main__":
    for name, build in BUILDERS.items():
        print(summary_line(build()))
