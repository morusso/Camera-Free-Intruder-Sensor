"""Tests for models. Run from the repo root: python -m unittest discover -s ml"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import models as m  # noqa: E402

ARENA_BUDGET = 80 * 1024  # bytes, step 3.7


class Shapes(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(0).normal(0, 1, (6, m.WIN, m.N_SUB)).astype(np.float32)

    def test_classifiers_output_probabilities(self):
        for build in (m.build_cnn, m.build_gru):
            with self.subTest(build.__name__):
                y = build().predict(self.x, verbose=0)
                self.assertEqual(y.shape, (6, m.N_CLASSES))
                np.testing.assert_allclose(y.sum(axis=1), 1.0, atol=1e-5)

    def test_autoencoder_reconstructs_input_shape(self):
        self.assertEqual(m.build_autoencoder().predict(self.x, verbose=0).shape, self.x.shape)

    def test_custom_sizes(self):
        y = m.build_cnn(n_classes=3, win=64, n_sub=30).predict(np.zeros((2, 64, 30), np.float32), verbose=0)
        self.assertEqual(y.shape, (2, 3))


class Budget(unittest.TestCase):
    def test_cnn_fits_the_device_budget(self):
        model = m.build_cnn()
        self.assertLess(model.count_params(), 30_000)
        self.assertLess(m.peak_activation_bytes(model), ARENA_BUDGET)

    def test_autoencoder_fits_the_device_budget(self):
        model = m.build_autoencoder()
        self.assertLess(model.count_params(), 60_000)
        self.assertLess(m.peak_activation_bytes(model), ARENA_BUDGET)

    def test_peak_estimate_for_a_known_layer(self):
        from tensorflow import keras
        from tensorflow.keras import layers
        inp = keras.Input((10, 4))
        model = keras.Model(inp, layers.Dense(6)(inp))
        self.assertEqual(m.peak_activation_bytes(model), 10 * 4 + 10 * 6)


class Training(unittest.TestCase):
    def test_one_step_gives_finite_loss(self):
        x = np.random.default_rng(1).normal(0, 1, (8, m.WIN, m.N_SUB)).astype(np.float32)
        y = np.arange(8) % m.N_CLASSES
        cnn = m.build_cnn()
        cnn.compile("adam", "sparse_categorical_crossentropy")
        self.assertTrue(np.isfinite(cnn.train_on_batch(x, y)))
        ae = m.build_autoencoder()
        ae.compile("adam", "mse")
        self.assertTrue(np.isfinite(ae.train_on_batch(x, x)))

    def test_anomaly_score_is_per_window(self):
        ae = m.build_autoencoder()
        x = np.random.default_rng(2).normal(0, 1, (5, m.WIN, m.N_SUB)).astype(np.float32)
        s = m.anomaly_scores(ae, x)
        self.assertEqual(s.shape, (5,))
        self.assertTrue((s >= 0).all())


if __name__ == "__main__":
    unittest.main()
