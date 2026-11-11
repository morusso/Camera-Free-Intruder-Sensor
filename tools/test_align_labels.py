"""Tests for align_labels. Run from the repo root: python -m unittest discover -s tools"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent))
import align_labels as al  # noqa: E402

T0 = 1_000_000.0
EMPTY, STATIC, WALKING, INTERFERENCE = 0, 1, 2, 3


def write_labels(path, rows):
    """rows: (seconds from T0, label, event)"""
    lines = ["unix_time,iso_time,label,event,note"]
    lines += [f"{T0 + t:.3f},x,{label},{event}," for t, label, event in rows]
    path.write_text("\n".join(lines) + "\n")


def write_recording(path, seconds=100, hz=100, width=8):
    n = seconds * hz
    np.savez(path, pc_time=T0 + np.arange(n) / hz, csi=np.zeros((n, width), np.int8),
             tx_seq=np.arange(n), rx_idx=np.arange(n), dev_ts_us=np.arange(n) * 10000)


class Intervals(unittest.TestCase):
    def load(self, rows, last=T0 + 100):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "labels.csv"
            write_labels(p, rows)
            return al.load_intervals(p, last)

    def test_basic(self):
        iv, w = self.load([(10, "empty", "start"), (20, "empty", "end"), (20, "standing", "start"),
                           (30, "standing", "end")])
        self.assertEqual([(i.button, i.t1 - i.t0) for i in iv], [("empty", 10), ("standing", 10)])
        self.assertEqual(w, [])

    def test_unclosed_label_is_closed_at_end_of_recording(self):
        iv, w = self.load([(10, "walking", "start")], last=T0 + 50)
        self.assertEqual(iv[0].t1, T0 + 50)
        self.assertEqual(len(w), 1)

    def test_overlap_is_an_error(self):
        with self.assertRaises(ValueError):
            self.load([(10, "empty", "start"), (30, "empty", "end"), (20, "walking", "start"),
                       (40, "walking", "end")])

    def test_unknown_label_is_an_error(self):
        with self.assertRaises(ValueError):
            self.load([(10, "dancing", "start")])


class Align(unittest.TestCase):
    def setUp(self):
        self.t = T0 + np.arange(0, 100, 0.01)

    def ivs(self, *spec):
        return [al.Interval(b, T0 + a, T0 + c) for b, a, c in spec]

    def test_labels_and_gaps(self):
        label, _ = al.align(self.t, self.ivs(("empty", 10, 20), ("walking", 30, 40)), 2.0)
        at = lambda s: label[int(s * 100)]  # noqa: E731
        self.assertEqual((at(5), at(15), at(25), at(35)), (-1, EMPTY, -1, WALKING))

    def test_uncertain_window_around_class_change(self):
        label, unc = al.align(self.t, self.ivs(("empty", 10, 20), ("standing", 20, 30)), 2.0)
        at = lambda s: bool(unc[int(s * 100)])  # noqa: E731
        self.assertTrue(at(18.5) and at(21.5))   # within 2 s of the change at 20 s
        self.assertFalse(at(17.5) or at(22.5))
        self.assertTrue(at(10.5) and at(29.5))   # edges of the labeled stretch
        self.assertFalse(at(15))

    def test_same_class_buttons_do_not_split(self):
        # "entering" then "walking" are both class walking: no change in the middle
        label, unc = al.align(self.t, self.ivs(("entering", 10, 15), ("walking", 15, 40)), 2.0)
        self.assertEqual(label[int(15 * 100)], WALKING)
        self.assertFalse(unc[int(15.5 * 100)])

    def test_unlabeled_frames_are_not_marked_uncertain(self):
        _, unc = al.align(self.t, self.ivs(("empty", 10, 20)), 2.0)
        self.assertFalse(unc[int(5 * 100)])

    def test_no_intervals(self):
        label, unc = al.align(self.t, [], 2.0)
        self.assertTrue((label == -1).all() and not unc.any())


class Cli(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            s = Path(d)
            write_recording(s / "csi_20261112_100000.npz")
            write_labels(s / "labels.csv", [
                (10, "empty", "start"), (40, "empty", "end"),
                (40, "entering", "start"), (45, "entering", "end"),
                (45, "standing", "start"), (80, "standing", "end"),
            ])
            res = CliRunner().invoke(al.app, [str(s)])
            self.assertEqual(res.exit_code, 0, res.output)
            data = np.load(s / "aligned.npz")
            self.assertEqual(len(data["label"]), len(data["pc_time"]))
            self.assertEqual(list(data["class_names"]), al.CLASS_NAMES)
            usable = (data["label"] >= 0) & ~data["uncertain"]
            self.assertGreater(usable.sum(), 0)
            self.assertIn("static_person", res.output)

    def test_offset_shifts_labels(self):
        with tempfile.TemporaryDirectory() as d:
            s = Path(d)
            write_recording(s / "csi_a.npz")
            write_labels(s / "labels.csv", [(10, "empty", "start"), (20, "empty", "end")])
            CliRunner().invoke(al.app, [str(s), "--offset", "5"])
            data = np.load(s / "aligned.npz")
            first = data["pc_time"][data["label"] == EMPTY][0]
            self.assertAlmostEqual(first - T0, 15.0, places=1)

    def test_missing_labels_file(self):
        with tempfile.TemporaryDirectory() as d:
            write_recording(Path(d) / "csi_a.npz")
            self.assertEqual(CliRunner().invoke(al.app, [d]).exit_code, 1)


if __name__ == "__main__":
    unittest.main()
