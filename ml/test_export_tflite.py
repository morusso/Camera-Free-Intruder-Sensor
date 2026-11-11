"""Tests for export_tflite. Run from the repo root: python -m unittest discover -s ml"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tensorflow import keras
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent))
import dataset as ds  # noqa: E402
import export_tflite as ex  # noqa: E402
import preprocess as pp  # noqa: E402
import train as tr  # noqa: E402
from test_baseline import SEGMENTS, make_session, save  # noqa: E402

CFG = pp.PreprocessConfig.load()
SEGS = [("empty", 60), ("walking", 60), ("static_person", 60), ("interference", 60)]


class Quantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        a, b, c = (make_session(s, SEGS, fan_amp=6.0) for s in (1, 2, 3))
        cls.paths = {n: save(d, f"{n}.npz", s) for n, s in zip(("train", "val", "test"), (a, b, c))}
        cls.bl = pp.fit_baseline(a["csi"][a["label"] == 0], CFG)
        cls.train = ds.load_dataset([cls.paths["train"]], CFG, cls.bl)
        cls.val = ds.load_dataset([cls.paths["val"]], CFG, cls.bl)
        cls.test = ds.load_dataset([cls.paths["test"]], CFG, cls.bl)
        cls.run_dir = d / "runs" / "cnn"
        tr.run_training(tr.TrainConfig(model="cnn", epochs=8, batch_size=64), cls.train, cls.val, cls.run_dir)
        shutil.copy(pp.CONFIG_PATH, cls.run_dir / "preprocess_config.json")
        cls.bl.save(cls.run_dir / "baseline.json")
        cls.model = keras.models.load_model(cls.run_dir / "model.keras")
        cls.rep, _ = cls.train.batch(ex.sample_indices(len(cls.train), 200, 0))
        cls.tflite = ex.convert_int8(cls.model, cls.rep)
        cls.q = ex.Int8Model(cls.tflite)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_model_is_int8_and_small(self):
        self.assertEqual(self.q.inp["dtype"], np.int8)
        self.assertEqual(self.q.out["dtype"], np.int8)
        self.assertEqual(list(self.q.inp["shape"]), [1, 128, 52])
        self.assertLess(len(self.tflite), 60 * 1024)

    def test_quantize_roundtrip_is_close(self):
        x = self.rep[:2]
        back = (self.q.quantize(x).astype(np.float32) - self.q.in_zero) * self.q.in_scale
        self.assertLess(float(np.abs(back - x).max()), 1.5 * self.q.in_scale)

    def test_probabilities_are_sane(self):
        p = self.q.predict(self.rep[:20])
        self.assertEqual(p.shape, (20, 4))
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=0.1)

    def test_int8_matches_float(self):
        r = ex.compare(self.model, self.q, self.test, max_windows=400)
        self.assertLess(abs(r["macro_f1_drop_pp"]), ex.MAX_DROP_PP)
        self.assertGreater(r["same_class_as_float"], 0.95)
        self.assertLess(r["input_saturation"], 0.01)

    def test_saturation_detects_out_of_range_input(self):
        wide = self.rep[:4] * 50
        self.assertGreater(self.q.saturation(wide), 0.1)

    def test_sample_indices(self):
        self.assertEqual(list(ex.sample_indices(5, 10, 0)), [0, 1, 2, 3, 4])
        idx = ex.sample_indices(1000, 100, 0)
        self.assertEqual(len(set(idx)), 100)
        np.testing.assert_array_equal(idx, ex.sample_indices(1000, 100, 0))

    def test_cli_writes_model_and_quantization_info(self):
        res = CliRunner().invoke(ex.app, [str(self.run_dir), "--train", str(self.paths["train"]),
                                          "--test", str(self.paths["test"]), "--n-representative", "100"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertTrue((self.run_dir / "model_int8.tflite").exists())
        info = json.loads((self.run_dir / "quantization.json").read_text())
        self.assertIn("scale", info["input"])
        self.assertIn("PASS", res.output)


if __name__ == "__main__":
    unittest.main()
