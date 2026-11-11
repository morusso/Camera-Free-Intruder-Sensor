"""Tests for evaluate. Run from the repo root: python -m unittest discover -s ml"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent))
import dataset as ds  # noqa: E402
import evaluate as ev  # noqa: E402
import preprocess as pp  # noqa: E402
import train as tr  # noqa: E402
from test_baseline import NAMES, make_session, save  # noqa: E402

CFG = pp.PreprocessConfig.load()
FS = int(CFG.sample_rate_hz)


class Smoothing(unittest.TestCase):
    def test_k_of_n(self):
        flags = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0], bool)
        # n=5, k=3: True at index 4 and 5 (three of the last five), False at 6 (only two left)
        self.assertEqual(list(ev.k_of_n(flags, 3, 5)), [0, 0, 0, 0, 1, 1, 0, 0, 0])

    def test_single_blip_is_ignored(self):
        flags = np.zeros(20, bool)
        flags[8] = True
        self.assertFalse(ev.k_of_n(flags, 3, 5).any())

    def test_rising_edges(self):
        self.assertEqual(ev.rising_edges(np.array([1, 1, 0, 1, 0, 0, 1], bool)), 3)


class Latency(unittest.TestCase):
    def setUp(self):
        self.starts = np.arange(0, 2000, 32)

    def test_delay_is_measured_from_the_entry(self):
        alarm = self.starts + 127 >= 1000 + 64  # alarm from the first window that ends 64 frames after the entry
        lat = ev.detection_latencies(alarm, self.starts, 128, np.array([1000]), 100.0, 10.0)
        self.assertEqual(len(lat), 1)
        self.assertGreaterEqual(lat[0], 0.64)
        self.assertLess(lat[0], 0.64 + 0.32 + 1e-9)

    def test_missed_when_too_late_or_never(self):
        alarm = self.starts + 127 >= 1900
        self.assertEqual(ev.detection_latencies(alarm, self.starts, 128, np.array([1000]), 100.0, 5.0), [None])
        self.assertEqual(ev.detection_latencies(np.zeros(len(self.starts), bool), self.starts, 128,
                                                np.array([1000]), 100.0, 5.0), [None])

    def test_entry_frames_only_from_empty_to_person(self):
        label = np.array([0, 0, 2, 2, 0, 3, 0, -1, 1, 1], np.int8)
        # empty -> walking at 2; empty -> interference and empty -> unlabeled do not count
        self.assertEqual(list(ev.entry_frames(label, NAMES)), [2])


SEGS = [("empty", 60), ("walking", 60), ("empty", 60), ("static_person", 60), ("empty", 60), ("interference", 60)]


def make_aligned(seed, fan=6.0):
    """Synthetic session with uncertain frames +-2 s around class changes, like align_labels.py."""
    s = make_session(seed, SEGS, fan_amp=fan)
    change = np.flatnonzero(np.diff(s["label"]) != 0) + 1
    unc = np.zeros(len(s["label"]), bool)
    for c in change:
        unc[max(0, c - 2 * FS):c + 2 * FS] = True
    s["uncertain"] = unc
    return s


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        train_s, val_s = make_aligned(1), make_aligned(2)
        cls.test_p = save(d, "test.npz", make_aligned(3))
        bl = pp.fit_baseline(train_s["csi"][train_s["label"] == 0], CFG)
        train = ds.load_dataset([save(d, "train.npz", train_s)], CFG, bl)
        val = ds.load_dataset([save(d, "val.npz", val_s)], CFG, bl)
        cls.run_dir = d / "runs" / "cnn"
        tr.run_training(tr.TrainConfig(model="cnn", epochs=8, batch_size=64), train, val, cls.run_dir)
        shutil.copy(pp.CONFIG_PATH, cls.run_dir / "preprocess_config.json")
        bl.save(cls.run_dir / "baseline.json")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_results_on_easy_synthetic_data(self):
        model, cfg, bl = ev.load_run(self.run_dir)
        r = ev.evaluate_paths(model, cfg, bl, [self.test_p])
        self.assertGreater(r["windows"]["macro_f1"], 0.9)
        e = r["events"]
        self.assertEqual(e["entries"], 2)
        self.assertEqual(e["entries_missed"], 0)
        self.assertLess(e["latency_max_s"], 3.0)
        self.assertEqual(e["false_alarms"], 0)
        self.assertGreater(e["empty_hours"], 0)

    def test_cli_writes_evaluation_json(self):
        res = CliRunner().invoke(ev.app, [str(self.run_dir), "--test", str(self.test_p)])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("detection delay", res.output)
        data = json.loads((self.run_dir / "evaluation.json").read_text())
        self.assertIn("confusion_matrix", data["windows"])

    def test_k_larger_than_n_is_rejected(self):
        res = CliRunner().invoke(ev.app, [str(self.run_dir), "--test", str(self.test_p), "--k", "6", "--n", "5"])
        self.assertNotEqual(res.exit_code, 0)

    def test_autoencoder_runs_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "ae"
            run.mkdir()
            (run / "params.json").write_text(json.dumps({"config": {"model": "autoencoder"}}))
            with self.assertRaises(ValueError):
                ev.load_run(run)


if __name__ == "__main__":
    unittest.main()
