"""Checks the C alarm logic (firmware/common/csi_alarm.c) against k_of_n in ml/evaluate.py, and its hysteresis.
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

COMMON = Path(__file__).resolve().parents[1] / "firmware" / "common"
NONE, START, END = 0, 1, 2


@unittest.skipUnless(shutil.which("cc"), "no C compiler")
class AlarmC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.exe = Path(cls.tmp.name) / "host_alarm"
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(COMMON / "include"),
                        str(COMMON / "csi_alarm.c"), str(COMMON / "test" / "host_alarm.c"), "-o", str(cls.exe)],
                       check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_c(self, p, on, off, n, k_on, k_off, hold=0):
        d = Path(self.tmp.name)
        np.asarray(p, np.float32).tofile(d / "p.bin")
        subprocess.run([str(self.exe), str(d / "p.bin"), str(on), str(off), str(n), str(k_on), str(k_off),
                        str(hold), str(d / "o.bin")], check=True)
        o = np.fromfile(d / "o.bin", np.uint8).reshape(-1, 2)
        return o[:, 0].astype(bool), o[:, 1]

    def test_matches_k_of_n(self):
        rng = np.random.default_rng(0)
        p = np.clip(rng.normal(0.4, 0.3, 5000), 0, 1).astype(np.float32)
        for n, k in [(5, 3), (1, 1), (8, 2), (16, 9)]:
            active, _ = self.run_c(p, 0.5, 0.5, n, k, k)
            np.testing.assert_array_equal(active, ev.k_of_n(p > 0.5, k, n), err_msg=f"n={n} k={k}")

    def test_events_are_transitions(self):
        p = np.r_[np.zeros(10), np.ones(10), np.zeros(10)]
        active, events = self.run_c(p, 0.5, 0.5, 5, 3, 3)
        self.assertEqual(list(np.flatnonzero(events == START)), [12])
        self.assertEqual(list(np.flatnonzero(events == END)), [22])
        self.assertTrue(np.all(active[12:22]) and not active[11] and not active[22])

    def test_hysteresis_keeps_alarm_on_between_thresholds(self):
        p = np.r_[np.ones(5), np.full(30, 0.4)]  # 0.4 is under on=0.6 but over off=0.3
        active, _ = self.run_c(p, 0.6, 0.3, 5, 3, 1)
        self.assertTrue(active[2:].all())
        flat, _ = self.run_c(p, 0.6, 0.6, 5, 3, 1)
        self.assertFalse(flat[-1])

    def test_min_hold(self):
        p = np.r_[np.ones(3), np.zeros(30)]
        short, _ = self.run_c(p, 0.5, 0.5, 3, 3, 1, hold=0)
        held, events = self.run_c(p, 0.5, 0.5, 3, 3, 1, hold=10)
        self.assertLess(short.sum(), held.sum())
        self.assertEqual(held.sum(), 10)  # START at window 2, END once held >= 10
        self.assertEqual(int(np.sum(events == END)), 1)

    def test_rejects_bad_config(self):
        d = Path(self.tmp.name)
        np.zeros(3, np.float32).tofile(d / "p.bin")
        for args in [(0.5, 0.5, 0, 1, 1), (0.5, 0.5, 17, 1, 1), (0.5, 0.5, 5, 6, 1), (0.5, 0.5, 5, 3, 0),
                     (0.4, 0.5, 5, 3, 3)]:
            r = subprocess.run([str(self.exe), str(d / "p.bin"), *map(str, args), "0", str(d / "o.bin")])
            self.assertEqual(r.returncode, 3, args)


if __name__ == "__main__":
    unittest.main()
