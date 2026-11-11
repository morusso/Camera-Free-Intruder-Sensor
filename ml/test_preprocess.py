"""Tests for preprocess. Run from the repo root: python -m unittest discover -s ml"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent))
import preprocess as pp

CFG = pp.PreprocessConfig.load()
FS = CFG.sample_rate_hz


def fake_csi(amp, rng=None):
    """Build int8 (imag, real) pairs from a [T, 64] amplitude array (phase 0 -> all in 'real')."""
    t, n = amp.shape
    out = np.zeros((t, n, 2), np.float32)
    out[..., 1] = amp
    return np.clip(np.round(out), -128, 127).astype(np.int8).reshape(t, -1)


class Config(unittest.TestCase):
    def test_default_config_matches_plan(self):
        self.assertEqual(len(CFG.keep_subcarriers), 52)
        self.assertEqual(len(set(CFG.keep_subcarriers)), 52)
        self.assertTrue(all(0 <= i < CFG.n_values // 2 for i in CFG.keep_subcarriers))

        self.assertTrue({0, *range(27, 38)}.isdisjoint(CFG.keep_subcarriers))

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            CFG.save(Path(d) / "c.json")
            self.assertEqual(pp.PreprocessConfig.load(Path(d) / "c.json"), CFG)


class Amplitude(unittest.TestCase):
    def test_shape_and_values(self):
        csi = np.zeros((5, 128), np.int8)
        csi[:, 2 * 10] = 3   # subcarrier 10: imag 3
        csi[:, 2 * 10 + 1] = 4  # real 4
        amp = pp.amplitude(csi, CFG)
        self.assertEqual(amp.shape, (5, 52))
        self.assertAlmostEqual(float(amp[0, CFG.keep_subcarriers.index(10)]), 5.0)

    def test_wrong_width_rejected(self):
        with self.assertRaises(ValueError):
            pp.amplitude(np.zeros((5, 384), np.int8), CFG)


class Hampel(unittest.TestCase):
    def test_spike_removed_smooth_signal_kept(self):
        t = np.arange(300) / FS
        x = (50 + 5 * np.sin(2 * np.pi * 2 * t))[:, None].astype(np.float32)
        spiked = x.copy()
        spiked[100] += 80
        out = pp.hampel(spiked, CFG.hampel_half_window, CFG.hampel_n_sigma, CFG.hampel_min_scale)
        self.assertLess(abs(out[100, 0] - x[100, 0]), 2.0)
        mask = np.ones(len(x), bool)
        mask[100] = False
        np.testing.assert_allclose(out[mask], spiked[mask], atol=1e-4)

    def test_flat_signal_is_not_changed(self):
        x = np.full((50, 3), 20.0, np.float32)
        np.testing.assert_array_equal(pp.hampel(x, 3, 3.0, 0.5), x)


class Lowpass(unittest.TestCase):
    def gain(self, freq):
        t = np.arange(1000) / FS
        x = (10 * np.sin(2 * np.pi * freq * t))[:, None].astype(np.float32) + 50
        y = pp.lowpass(x, CFG)
        return (y[300:].max() - y[300:].min()) / (x[300:].max() - x[300:].min())

    def test_passband_and_stopband(self):
        self.assertGreater(self.gain(5), 0.95)
        self.assertLess(self.gain(40), 0.1)

    def test_no_startup_transient(self):
        y = pp.lowpass(np.full((100, 2), 37.0, np.float32), CFG)
        np.testing.assert_allclose(y, 37.0, atol=1e-3)

    def test_is_causal(self):
        rng = np.random.default_rng(0)
        x = rng.normal(50, 3, (200, 4)).astype(np.float32)
        changed = x.copy()
        changed[150:] += 100
        np.testing.assert_allclose(pp.lowpass(x, CFG)[:150], pp.lowpass(changed, CFG)[:150], atol=1e-4)


class Baseline(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        base = 40 + 10 * np.sin(np.linspace(0, 3, 64))
        self.empty = fake_csi(base + rng.normal(0, 1.0, (3000, 64)))
        t = (np.arange(1000) / FS)[:, None]
        moving = base + rng.normal(0, 1.0, (1000, 64)) + 12 * np.sin(2 * np.pi * 1.5 * t)
        self.moving = fake_csi(moving)
        self.bl = pp.fit_baseline(self.empty, CFG)

    def test_empty_room_is_near_zero(self):
        z = pp.preprocess(self.empty, CFG, self.bl)
        self.assertLess(abs(float(z.mean())), 0.05)
        self.assertLess(float(z.std()), 1.2)

    def test_motion_shows_as_large_deviation(self):
        z = pp.preprocess(self.moving, CFG, self.bl)
        self.assertGreater(float(z.std()), 3 * float(pp.preprocess(self.empty, CFG, self.bl).std()))

    def test_std_floor(self):
        flat = fake_csi(np.full((500, 64), 40.0))
        self.assertTrue((pp.fit_baseline(flat, CFG).std >= CFG.norm_min_std).all())

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            self.bl.save(Path(d) / "b.json")
            back = pp.Baseline.load(Path(d) / "b.json")
        np.testing.assert_allclose(back.mean, self.bl.mean, rtol=1e-5)
        self.assertEqual(back.n_frames, self.bl.n_frames)


class Cli(unittest.TestCase):
    def test_baseline_uses_longest_empty_run(self):
        rng = np.random.default_rng(2)
        csi = fake_csi(40 + rng.normal(0, 1.0, (4000, 64)))
        label = np.full(4000, -1, np.int8)
        label[200:700] = 0
        label[1000:3500] = 0
        data = dict(csi=csi, label=label, uncertain=np.zeros(4000, bool), class_names=np.array(["empty"]))
        with tempfile.TemporaryDirectory() as d:
            np.savez(Path(d) / "aligned.npz", **data)
            res = CliRunner().invoke(pp.app, [str(Path(d) / "aligned.npz"), "--out", str(Path(d) / "b.json")])
            self.assertEqual(res.exit_code, 0, res.output)
            self.assertEqual(pp.Baseline.load(Path(d) / "b.json").n_frames, 2500)

    def test_too_little_empty_data(self):
        data = dict(csi=np.zeros((500, 128), np.int8), label=np.zeros(500, np.int8),
                    uncertain=np.zeros(500, bool), class_names=np.array(["empty"]))
        with tempfile.TemporaryDirectory() as d:
            np.savez(Path(d) / "aligned.npz", **data)
            res = CliRunner().invoke(pp.app, [str(Path(d) / "aligned.npz"), "--out", str(Path(d) / "b.json")])
            self.assertEqual(res.exit_code, 1)


if __name__ == "__main__":
    unittest.main()
