"""Tests for train and metrics. Run from the repo root: python -m unittest discover -s ml"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent))
import dataset as ds  # noqa: E402
import metrics  # noqa: E402
import preprocess as pp  # noqa: E402
import train as tr  # noqa: E402
from test_baseline import make_session, save  # noqa: E402

CFG = pp.PreprocessConfig.load()
SEGS = [("empty", 60), ("walking", 60), ("static_person", 60), ("interference", 60)]


def make_data(d, fan=6.0):
    train_p = [save(d, "a.npz", make_session(1, SEGS, fan_amp=fan))]
    val_p = [save(d, "b.npz", make_session(2, SEGS, fan_amp=fan))]
    s = np.load(train_p[0])
    bl = pp.fit_baseline(s["csi"][s["label"] == 0], CFG)
    return ds.load_dataset(train_p, CFG, bl), ds.load_dataset(val_p, CFG, bl), train_p, val_p, bl


class Metrics(unittest.TestCase):
    def test_perfect_and_confused(self):
        y = np.array([0, 0, 1, 1, 2])
        probs = np.eye(3)[[0, 0, 1, 0, 2]]  # one class-1 window predicted as class 0
        m = metrics.classification_metrics(y, probs, ["a", "b", "c"])
        self.assertAlmostEqual(m["accuracy"], 0.8)
        self.assertEqual(m["confusion_matrix"], [[2, 0, 0], [1, 1, 0], [0, 0, 1]])
        self.assertAlmostEqual(m["f1_per_class"]["c"], 1.0)

    def test_macro_f1_ignores_absent_classes(self):
        y = np.array([0, 0, 1, 1])
        m = metrics.classification_metrics(y, np.eye(4)[y], ["a", "b", "c", "d"])
        self.assertAlmostEqual(m["macro_f1"], 1.0)


class Training(unittest.TestCase):
    def run_cnn(self, d, name, seed=0, **kw):
        train, val, *_ = make_data(d)
        # Fewer than ~6 epochs is too few steps for the BatchNorm running averages to settle.
        cfg = tr.TrainConfig(model="cnn", seed=seed, epochs=6, batch_size=64, **kw)
        return tr.run_training(cfg, train, val, Path(d) / "runs" / name), Path(d) / "runs" / name

    def test_run_writes_all_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            summary, out = self.run_cnn(d, "r1")
            for f in ("params.json", "history.csv", "summary.json", "model.keras"):
                self.assertTrue((out / f).exists(), f)
            params = json.loads((out / "params.json").read_text())
            self.assertEqual(params["config"]["seed"], 0)
            self.assertEqual(set(params["train_class_counts"]), set(params["class_names"]))
            rows = list(csv.DictReader(open(out / "history.csv")))
            self.assertEqual(len(rows), summary["epochs_run"])
            self.assertGreater(summary["val"]["accuracy"], 0.8)
            exp = list(csv.DictReader(open(out.parent / "experiments.csv")))
            self.assertEqual(exp[0]["run_id"], "r1")

    def test_same_seed_same_result(self):
        with tempfile.TemporaryDirectory() as d:
            a, _ = self.run_cnn(d, "a", seed=5)
            b, _ = self.run_cnn(d, "b", seed=5)
            self.assertAlmostEqual(a["val"]["loss"], b["val"]["loss"], places=4)

    def test_experiments_log_accumulates(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_cnn(d, "a")
            self.run_cnn(d, "b", class_weights=True)
            rows = list(csv.DictReader(open(Path(d) / "runs" / "experiments.csv")))
            self.assertEqual([r["run_id"] for r in rows], ["a", "b"])
            self.assertEqual(rows[1]["class_weights"], "True")

    def test_autoencoder_separates_empty_from_the_rest(self):
        with tempfile.TemporaryDirectory() as d:
            train, val, *_ = make_data(d)
            cfg = tr.TrainConfig(model="autoencoder", epochs=4, batch_size=64, augment=False)
            summary = tr.run_training(cfg, train, val, Path(d) / "runs" / "ae")
            self.assertGreater(summary["val_auc"], 0.9)

    def test_empty_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            train, val, *_ = make_data(d)
            only_walking = train.subset(["walking"]).subset([])
            with self.assertRaises(ValueError):
                tr.run_training(tr.TrainConfig(epochs=1), only_walking, val, Path(d) / "runs" / "x")


class Cli(unittest.TestCase):
    def test_cli_copies_preprocessing_next_to_the_model(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, train_p, val_p, bl = make_data(d)
            bl.save(Path(d) / "baseline.json")
            res = CliRunner().invoke(tr.app, [
                "--model", "cnn", "--train", str(train_p[0]), "--val", str(val_p[0]),
                "--baseline", str(Path(d) / "baseline.json"), "--epochs", "2",
                "--runs-dir", str(Path(d) / "runs")])
            self.assertEqual(res.exit_code, 0, res.output)
            run = next((Path(d) / "runs").glob("*_cnn_s0"))
            self.assertTrue((run / "baseline.json").exists())
            self.assertTrue((run / "preprocess_config.json").exists())

    def test_unknown_model(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, train_p, val_p, _ = make_data(d)
            res = CliRunner().invoke(tr.app, ["--model", "nope", "--train", str(train_p[0]), "--val", str(val_p[0])])
            self.assertNotEqual(res.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
