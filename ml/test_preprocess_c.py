"""Checks that the C port (firmware/common/csi_preprocess.c) matches ml/preprocess.py.
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
import gen_preprocess_header as gen
import preprocess as pp

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "firmware" / "common"
CFG = pp.PreprocessConfig.load()
N_KEEP = len(CFG.keep_subcarriers)
TOL = 2e-3  # float32 C vs float64 scipy, after dividing by the baseline std


class HeaderUpToDate(unittest.TestCase):
    def test_generated_header_matches_config(self):
        self.assertEqual(gen.HEADER_PATH.read_text(), gen.render(CFG),
                         "preprocess_params.h is stale: run python ml/gen_preprocess_header.py")


def realistic_csi(t, seed):
    """Smooth spectrum + slow motion + noise + a few spikes, as int8 (imag, real) pairs."""
    rng = np.random.default_rng(seed)
    base = 30 + 10 * np.sin(np.linspace(0, 3, 64))
    time = (np.arange(t) / CFG.sample_rate_hz)[:, None]
    amp = base + 6 * np.sin(2 * np.pi * 1.2 * time + np.linspace(0, 2, 64)) + rng.normal(0, 1.5, (t, 64))
    spikes = rng.random((t, 64)) < 0.01
    amp = np.where(spikes, amp + rng.choice([-40, 40], (t, 64)), amp)
    phase = rng.uniform(-np.pi, np.pi, (1, 64))
    csi = np.stack([amp * np.sin(phase), amp * np.cos(phase)], axis=-1)
    return np.clip(np.round(csi), -128, 127).astype(np.int8).reshape(t, -1)


@unittest.skipUnless(shutil.which("cc"), "no C compiler")
class CMatchesPython(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.exe = Path(cls.tmp.name) / "host_preprocess"
        subprocess.run(
            ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(COMMON / "include"),
             str(COMMON / "csi_preprocess.c"), str(COMMON / "test" / "host_preprocess.c"), "-lm", "-o", str(cls.exe)],
            check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_c(self, mode, csi, *extra):
        d = Path(self.tmp.name)
        (d / "csi.bin").write_bytes(csi.tobytes())
        out = d / "out.bin"
        subprocess.run([str(self.exe), mode, str(d / "csi.bin"), *map(str, extra), str(out)], check=True)
        return np.fromfile(out, np.float32)

    def test_clean_matches_python(self):
        csi = realistic_csi(1500, seed=1)
        got = self.run_c("clean", csi).reshape(-1, N_KEEP)
        want = pp.preprocess(csi, CFG)
        self.assertEqual(got.shape, want.shape)
        np.testing.assert_allclose(got, want, atol=TOL)

    def test_normalized_matches_python(self):
        empty = realistic_csi(2000, seed=2)
        bl = pp.fit_baseline(empty, CFG)
        bl_path = Path(self.tmp.name) / "bl.bin"
        bl_path.write_bytes(bl.mean.astype(np.float32).tobytes() + bl.std.astype(np.float32).tobytes())
        csi = realistic_csi(1000, seed=3)
        got = self.run_c("norm", csi, bl_path).reshape(-1, N_KEEP)
        np.testing.assert_allclose(got, pp.preprocess(csi, CFG, bl), atol=TOL)

    def test_sequences_shorter_than_the_hampel_window(self):
        for t in (1, 2, 3, 4, 7, 8):
            csi = realistic_csi(t, seed=t)
            got = self.run_c("clean", csi).reshape(-1, N_KEEP)
            np.testing.assert_allclose(got, pp.preprocess(csi, CFG), atol=TOL, err_msg=f"T={t}")

    def test_baseline_fit_matches_python(self):
        csi = realistic_csi(3000, seed=4)
        got = self.run_c("fit", csi)
        want = pp.fit_baseline(csi, CFG)
        np.testing.assert_allclose(got[:N_KEEP], want.mean, atol=1e-3)
        np.testing.assert_allclose(got[N_KEEP:], want.std, atol=1e-3)

    def test_lagged_output_is_causal(self):
        """A change at frame k must not alter outputs earlier than k - hampel_half (streaming = no look-ahead)."""
        a = realistic_csi(400, seed=5)
        b = a.copy()
        b[300:] = realistic_csi(100, seed=6)
        ya = self.run_c("clean", a).reshape(-1, N_KEEP)
        yb = self.run_c("clean", b).reshape(-1, N_KEEP)
        np.testing.assert_array_equal(ya[:300 - CFG.hampel_half_window], yb[:300 - CFG.hampel_half_window])


if __name__ == "__main__":
    unittest.main()
