"""Tests for jitter_report. Run from the repo root: python -m unittest discover -s tools"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent))
import jitter_report as jr


def synthetic(n=3000, hz=100, jitter_ms=1.0, lost_every=0, seed=0):
    """Frames every 1/hz s with Gaussian timing noise; optionally drop every Nth frame."""
    rng = np.random.default_rng(seed)
    seq = np.arange(n)
    ts = (seq * 1e6 / hz + rng.normal(0, jitter_ms * 1000, n)).astype(np.int64)
    ts = np.maximum.accumulate(ts)
    keep = np.ones(n, bool)
    if lost_every:
        keep[::lost_every] = False
        keep[0] = True
    seq, ts = seq[keep], ts[keep]
    return {"dev_ts_us": ts, "tx_seq": seq, "rx_idx": np.arange(len(seq))}


class Report(unittest.TestCase):
    def test_clean_link_passes(self):
        r = jr.compute_report(synthetic())
        self.assertAlmostEqual(r["nominal_ms"], 10.0, delta=0.2)
        self.assertTrue(r["pass_received"] and r["pass_jitter"])
        self.assertEqual(r["lost_radio"], 0)

    def test_loss_is_counted_and_fails(self):
        r = jr.compute_report(synthetic(lost_every=5))  # ~20 % of frames missing
        self.assertGreater(r["lost_radio"], 500)
        self.assertFalse(r["pass_received"])
        self.assertAlmostEqual(r["received_pct"], 80.0, delta=1.0)

    def test_loss_does_not_inflate_jitter(self):
        clean = jr.compute_report(synthetic(jitter_ms=0.5, seed=1))
        lossy = jr.compute_report(synthetic(jitter_ms=0.5, lost_every=10, seed=1))
        self.assertAlmostEqual(lossy["jitter_ms"]["p99"], clean["jitter_ms"]["p99"], delta=1.0)
        self.assertGreater(lossy["gap_ms"]["max"], 15.0)  # the gaps over losses are still visible

    def test_large_jitter_fails(self):
        r = jr.compute_report(synthetic(jitter_ms=15.0))
        self.assertFalse(r["pass_jitter"])

    def test_explicit_nominal(self):
        r = jr.compute_report(synthetic(hz=50), nominal_hz=100)
        self.assertEqual(r["nominal_ms"], 10.0)
        self.assertGreater(r["jitter_ms"]["p50"], 9.0)

    def test_too_few_frames(self):
        with self.assertRaises(ValueError):
            jr.compute_report({"dev_ts_us": np.array([1, 2]), "tx_seq": np.array([0, 1]), "rx_idx": np.array([0, 1])})


class Cli(unittest.TestCase):
    def test_markdown_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            good, bad = Path(d) / "good.npz", Path(d) / "bad.npz"
            np.savez(good, **synthetic())
            np.savez(bad, **synthetic(lost_every=5))
            res = CliRunner().invoke(jr.app, [str(good), "--markdown", "--plot", str(Path(d) / "h.png")])
            self.assertEqual(res.exit_code, 0, res.output)
            self.assertIn("| Metric |", res.output)
            self.assertTrue((Path(d) / "h.png").exists())
            self.assertEqual(CliRunner().invoke(jr.app, [str(bad)]).exit_code, 2)


if __name__ == "__main__":
    unittest.main()
