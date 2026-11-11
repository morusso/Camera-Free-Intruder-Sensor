"""Tests for the pure parts of live_heatmap. Run from the repo root: python -m unittest discover -s tools"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent))
import live_heatmap as lh  # noqa: E402


class Amplitudes(unittest.TestCase):
    def test_pairs(self):
        amp = lh.amplitudes(np.array([3, 4, 0, -5, 6, 8], dtype=np.int8))
        np.testing.assert_allclose(amp, [5, 5, 10])

    def test_odd_length_drops_last(self):
        self.assertEqual(len(lh.amplitudes(np.array([1, 1, 7], dtype=np.int8))), 1)


class Clock(unittest.TestCase):
    def test_starts_at_zero_and_unwraps(self):
        c = lh.ClockUnwrapper()
        self.assertEqual(c.feed(2 ** 32 - 1_000_000), 0.0)
        self.assertAlmostEqual(c.feed(1_000_000), 2.0)  # wrapped: +2 s in total
        self.assertAlmostEqual(c.feed(1_500_000), 2.5)


class WindowTest(unittest.TestCase):
    def test_trims_old_frames(self):
        w = lh.Window(1.0)
        for k in range(30):
            w.add(k * 0.1, np.ones(4, np.float32))
        times, m = w.snapshot()
        self.assertLessEqual(times[-1] - times[0], 1.0 + 1e-9)
        self.assertEqual(m.shape, (4, len(times)))

    def test_ignores_wrong_width(self):
        w = lh.Window(5.0)
        w.add(0.0, np.ones(4, np.float32))
        w.add(0.1, np.ones(6, np.float32))
        self.assertEqual(w.snapshot()[1].shape, (4, 1))

    def test_empty(self):
        self.assertEqual(lh.Window(1.0).snapshot()[1].size, 0)


class Demo(unittest.TestCase):
    def test_walking_has_larger_variance_than_empty(self):
        frames = list(lh.demo_source(30.0))
        t = np.array([f[0] for f in frames])
        a = np.stack([f[1] for f in frames])
        var_empty = a[t < 9].var(axis=0).mean()
        var_walk = a[(t >= 11) & (t < 19)].var(axis=0).mean()
        self.assertGreater(var_walk, 20 * var_empty)

    def test_save_png(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "demo.png"
            result = CliRunner().invoke(lh.app, ["--demo", "--save", str(out), "--duration", "5"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertGreater(out.stat().st_size, 1000)

    def test_needs_exactly_one_source(self):
        self.assertNotEqual(CliRunner().invoke(lh.app, []).exit_code, 0)


if __name__ == "__main__":
    unittest.main()
