"""Checks the C detector pipeline (firmware/common/csi_detector.c) against ml/preprocess.py + ml/evaluate.py.
The model is replaced by a fake whose output depends only on the window, so everything else is compared.
Run from the repo root: python -m unittest discover -s ml
"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as ev
import preprocess as pp
from test_preprocess_c import realistic_csi

COMMON = Path(__file__).resolve().parents[1] / "firmware" / "common"
CFG = pp.PreprocessConfig.load()
N_KEEP = len(CFG.keep_subcarriers)
WIN, STRIDE = 128, 32
DTYPE = np.dtype([("p", "<f4"), ("kind", "u1"), ("event", "u1"), ("active", "u1"), ("pad", "u1")])
WINDOW, CALIB_DONE, CALIB_FAILED = 1, 2, 3


def fake_p(x, starts):
    return np.array([min(1.0, float(np.mean(x[s:s + WIN] ** 2) / 4.0)) for s in starts], np.float32)


@unittest.skipUnless(shutil.which("cc"), "no C compiler")
class DetectorC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.exe = Path(cls.tmp.name) / "host_detector"
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(COMMON / "include"),
                        str(COMMON / "csi_preprocess.c"), str(COMMON / "csi_alarm.c"), str(COMMON / "csi_detector.c"),
                        str(COMMON / "test" / "host_detector.c"), "-lm", "-o", str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_c(self, mode, csi, arg):
        d = Path(self.tmp.name)
        csi.tofile(d / "csi.bin")
        subprocess.run([str(self.exe), mode, str(d / "csi.bin"), str(arg), str(d / "out.bin")], check=True)
        return (d / "out.bin").read_bytes()

    def scene(self):
        """An empty room followed by a louder stretch (someone moving), as int8 frames."""
        quiet = realistic_csi(1500, seed=10)
        busy = realistic_csi(1500, seed=11).astype(np.int16)
        busy = np.clip(busy * 2 + np.random.default_rng(1).integers(-20, 20, busy.shape), -128, 127).astype(np.int8)
        return np.concatenate([quiet, busy])

    def test_windows_and_alarm_match_python(self):
        csi = self.scene()
        bl = pp.fit_baseline(realistic_csi(2000, seed=12), CFG)
        d = Path(self.tmp.name)
        (d / "bl.bin").write_bytes(bl.mean.astype(np.float32).tobytes() + bl.std.astype(np.float32).tobytes())
        rec = np.frombuffer(self.run_c("base", csi, d / "bl.bin"), DTYPE)

        x = pp.preprocess(csi, CFG, bl)
        # The streaming output lags by hampel_half frames, so it has len(csi) - half cleaned frames.
        x = x[:len(csi) - CFG.hampel_half_window]
        starts = np.arange(0, len(x) - WIN + 1, STRIDE)
        want_p = fake_p(x, starts)
        self.assertEqual(len(rec), len(starts))
        np.testing.assert_allclose(rec["p"], want_p, atol=2e-3)
        # decisions: compare with k_of_n on the C probabilities themselves (avoids flips at the threshold)
        np.testing.assert_array_equal(rec["active"].astype(bool), ev.k_of_n(rec["p"] > 0.5, 3, 5))
        self.assertTrue(rec["active"].any() and not rec["active"].all(), "scene should have both states")

    def test_calibration_matches_fit_and_runs(self):
        quiet = realistic_csi(1500, seed=13)
        csi = np.concatenate([quiet, self.scene()])
        n = 1000
        raw = self.run_c("calib", csi, n)
        rec = np.frombuffer(raw[:-2 * N_KEEP * 4], DTYPE)
        mean, std = np.frombuffer(raw[-2 * N_KEEP * 4:], "<f4").reshape(2, N_KEEP)
        self.assertEqual(int(np.sum(rec["kind"] == CALIB_DONE)), 1)
        cleaned = pp.clean(csi, CFG)[:n]
        np.testing.assert_allclose(mean, cleaned.mean(axis=0), atol=2e-3)
        np.testing.assert_allclose(std, np.maximum(cleaned.std(axis=0), CFG.norm_min_std), atol=2e-3)
        windows = rec[rec["kind"] == WINDOW]
        self.assertGreater(len(windows), 10)
        self.assertEqual(np.flatnonzero(rec["kind"] == CALIB_DONE)[0], 0, "calibration result comes first")

    def test_nothing_is_evaluated_without_a_baseline_or_before_the_ring_is_full(self):
        d = Path(self.tmp.name)
        short = realistic_csi(WIN + 2, seed=14)  # cleaned frames: WIN - 1 (lag), never a full window
        (d / "bl.bin").write_bytes(np.r_[np.zeros(N_KEEP), np.ones(N_KEEP)].astype("<f4").tobytes())
        self.assertEqual(self.run_c("base", short, d / "bl.bin"), b"")
        # calibration longer than the data: never finishes, never evaluates
        raw = self.run_c("calib", realistic_csi(300, seed=15), 5000)
        self.assertEqual(len(raw), 2 * N_KEEP * 4)


if __name__ == "__main__":
    unittest.main()
