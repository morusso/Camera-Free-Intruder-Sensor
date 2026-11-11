"""Tests for baseline. Run from the repo root: python -m unittest discover -s ml"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent))
import baseline as bl  # noqa: E402
import preprocess as pp  # noqa: E402

CFG = pp.PreprocessConfig.load()
NAMES = ["empty", "static_person", "walking", "interference"]
FS = int(CFG.sample_rate_hz)


def make_session(seed, segments, noise=1.0, walk_amp=12.0, fan_amp=0.0):
    """segments: list of (class name, seconds). Returns the arrays align_labels.py would produce."""
    rng = np.random.default_rng(seed)
    base = 40 + 10 * np.sin(np.linspace(0, 3, 64))
    amp, label = [], []
    t_total = 0
    for name, sec in segments:
        n = sec * FS
        t = (t_total + np.arange(n))[:, None] / FS
        a = base + rng.normal(0, noise, (n, 64))
        if name == "walking":
            a = a + walk_amp * np.sin(2 * np.pi * 1.5 * t + np.arange(64))
        elif name == "static_person":
            a = a + 3.0 + rng.normal(0, 2.0, (n, 64))
        elif name == "interference":
            a = a + fan_amp * np.sin(2 * np.pi * 8 * t + np.arange(64))
        amp.append(a)
        label.append(np.full(n, NAMES.index(name), np.int8))
        t_total += n
    amp = np.vstack(amp)
    csi = np.zeros((len(amp), 64, 2), np.float32)
    csi[..., 1] = amp
    csi = np.clip(np.round(csi), -128, 127).astype(np.int8).reshape(len(amp), -1)
    return dict(csi=csi, label=np.concatenate(label), uncertain=np.zeros(len(amp), bool),
                class_names=np.array(NAMES))


def save(d, name, data):
    p = Path(d) / name
    np.savez(p, **data)
    return p


SEGMENTS = [("empty", 60), ("walking", 60), ("empty", 60), ("static_person", 60), ("interference", 60)]


class Windows(unittest.TestCase):
    def test_scores_match_numpy_variance(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 3, (500, 4))
        starts, score = bl.window_scores(x, 100, 50)
        expect = [x[s:s + 100].var(axis=0).mean() for s in starts]
        np.testing.assert_allclose(score, expect, rtol=1e-6)

    def test_too_short(self):
        self.assertEqual(len(bl.window_scores(np.zeros((10, 2)), 100, 50)[1]), 0)

    def test_window_classes_drop_mixed_and_uncertain(self):
        label = np.array([0] * 100 + [2] * 100, np.int8)
        unc = np.zeros(200, bool)
        unc[20] = True
        starts = np.array([0, 10, 50, 90, 100])
        cls = bl.window_classes(label, unc, starts, 50, NAMES)
        self.assertEqual(list(cls), ["", "", "empty", "", "walking"])
        # window at 0 and 10 contain the uncertain frame, window at 90 mixes two classes


class Detector(unittest.TestCase):
    def sessions(self, **kw):
        with tempfile.TemporaryDirectory() as d:
            return [bl.load_session(save(d, f"s{i}.npz", make_session(i, SEGMENTS, **kw)), CFG, 1.5, 0.5)
                    for i in range(2)]

    def test_detects_walking_and_stays_quiet_when_empty(self):
        val = self.sessions()
        thr = bl.choose_threshold(val)
        res = bl.evaluate(val, thr, 0.5)
        self.assertGreater(res["sensitivity_walking"], 0.95)
        self.assertGreater(res["specificity_empty"], 0.95)
        self.assertGreater(res["auc"], 0.9)

    def test_interference_is_reported_separately(self):
        val = self.sessions(fan_amp=10.0)  # a fan that looks like motion to a variance detector
        res = bl.evaluate(val, bl.choose_threshold(val), 0.5)
        self.assertGreater(res["false_positive_rate_interference"], 0.9)

    def test_false_alarm_events_count_runs(self):
        score = np.array([0, 5, 5, 0, 5, 0, 0, 5], float)
        cls = np.array(["empty"] * 8, dtype=object)
        self.assertEqual(bl.false_alarm_events(score, cls, 1.0), 3)
        cls[4] = "walking"  # that detection is a true one, not a false alarm
        self.assertEqual(bl.false_alarm_events(score, cls, 1.0), 2)

    def test_per_day_scaling(self):
        s = (np.array([5.0] + [0.0] * 7199), np.array(["empty"] * 7200, dtype=object))
        res = bl.evaluate([s], 1.0, 0.5)  # 7200 windows * 0.5 s = 1 h with one false alarm
        self.assertAlmostEqual(res["empty_hours"], 1.0)
        self.assertAlmostEqual(res["false_alarms_per_day"], 24.0)

    def test_needs_both_classes_for_threshold(self):
        only_empty = (np.array([1.0, 2.0]), np.array(["empty", "empty"], dtype=object))
        with self.assertRaises(ValueError):
            bl.choose_threshold([only_empty])


class Cli(unittest.TestCase):
    def test_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            v = save(d, "val.npz", make_session(1, SEGMENTS))
            t = save(d, "test.npz", make_session(2, SEGMENTS))
            res = CliRunner().invoke(bl.app, ["--val", str(v), "--test", str(t), "--roc", str(Path(d) / "roc.png")])
            self.assertEqual(res.exit_code, 0, res.output)
            self.assertIn("false alarms in empty room", res.output)
            self.assertTrue((Path(d) / "roc.png").exists())


if __name__ == "__main__":
    unittest.main()
