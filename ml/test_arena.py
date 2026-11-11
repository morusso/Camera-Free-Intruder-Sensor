"""Tests for arena. Run from the repo root: python -m unittest discover -s ml"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent))
import arena as ar  # noqa: E402
import export_tflite as ex  # noqa: E402
import models  # noqa: E402

B = ar.Buffer


def overlap(a, b):
    return not (a.last < b.first or b.last < a.first)


class Planner(unittest.TestCase):
    def test_chain_reuses_memory(self):
        # A -> B -> C: A and C are never alive together, so they can share
        bufs = [B("a", 160, 0, 1), B("b", 160, 1, 2), B("c", 160, 2, 3)]
        offsets, total = ar.plan(bufs)
        self.assertEqual(total, 320)
        self.assertEqual(offsets["a"], offsets["c"])

    def test_alive_together_never_overlap_in_memory(self):
        rng = np.random.default_rng(0)
        bufs = []
        for i in range(40):
            first = int(rng.integers(0, 20))
            bufs.append(B(f"t{i}", int(rng.integers(1, 3000)), first, first + int(rng.integers(0, 8))))
        offsets, total = ar.plan(bufs)
        for i, a in enumerate(bufs):
            self.assertEqual(offsets[a.name] % ar.ALIGN, 0)
            for b in bufs[i + 1:]:
                if overlap(a, b):
                    a_end = offsets[a.name] + ar.aligned(a.size)
                    b_end = offsets[b.name] + ar.aligned(b.size)
                    self.assertTrue(a_end <= offsets[b.name] or b_end <= offsets[a.name], (a, b))
        peak_alone = max(ar.aligned(b.size) for b in bufs)
        self.assertGreaterEqual(total, peak_alone)
        self.assertLessEqual(total, sum(ar.aligned(b.size) for b in bufs))

    def test_empty(self):
        self.assertEqual(ar.plan([]), ({}, 0))

    def test_alignment(self):
        self.assertEqual((ar.aligned(1), ar.aligned(16), ar.aligned(17)), (16, 16, 32))


class RealModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rep = np.random.default_rng(0).normal(0, 1, (60, models.WIN, models.N_SUB)).astype(np.float32)
        cls.cnn = ex.convert_int8(models.build_cnn(), rep)

    def test_cnn_fits_the_budget(self):
        buffers, ops = ar.buffers_from_tflite(self.cnn)
        _, arena = ar.plan(buffers)
        self.assertLess(arena, 20 * 1024)
        # the arena cannot be smaller than the input plus the first layer's output
        self.assertGreater(arena, 128 * 52)
        self.assertIn("CONV_2D", ops)
        self.assertIn("SOFTMAX", ops)

    def test_graph_is_not_changed_by_a_delegate(self):
        _, ops = ar.buffers_from_tflite(self.cnn)
        self.assertNotIn("DELEGATE", ops)

    def test_sharing_saves_memory(self):
        buffers, _ = ar.buffers_from_tflite(self.cnn)
        _, arena = ar.plan(buffers)
        self.assertLess(arena, sum(ar.aligned(b.size) for b in buffers))

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.tflite"
            p.write_bytes(self.cnn)
            res = CliRunner().invoke(ar.app, [str(p)])
            self.assertEqual(res.exit_code, 0, res.output)
            self.assertIn("planned tensor arena", res.output)
            self.assertIn("PASS", res.output)
            tight = CliRunner().invoke(ar.app, [str(p), "--reserve-kb", "100"])
            self.assertEqual(tight.exit_code, 2)


if __name__ == "__main__":
    unittest.main()
