"""Parser tests. Run from the repo root: python -m unittest discover -s tools"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent))
import csi_reader as cr  # noqa: E402


def make_line(rx_idx=0, tx_seq=0, ts=1000, n=8, mac="aa:bb:cc:dd:ee:ff"):
    vals = ",".join(str((i % 20) - 10) for i in range(n))
    return f'CSI_DATA,{rx_idx},{tx_seq},{mac},-48,11,-92,6,{ts},64,1,{n},0,"[{vals}]"'


class ParseLine(unittest.TestCase):
    def test_valid(self):
        f = cr.parse_line(make_line(5, 7, 123, n=4))
        self.assertEqual((f.rx_idx, f.tx_seq, f.local_timestamp, len(f.csi)), (5, 7, 123, 4))
        self.assertEqual(f.csi.dtype, np.int8)

    def test_errors(self):
        cases = {
            "not_data": "I (123) wifi: something",
            "bad_field_count": make_line().replace(",64,1,", ",64,"),
            "len_mismatch": make_line(n=8).replace(",8,0,", ",9,0,"),
            "out_of_range": make_line(n=2).replace("[-10,-9]", "[-10,300]"),
            "bad_mac": make_line(mac="zz"),
            "bad_number": make_line().replace("-48", "abc"),
        }
        for reason, line in cases.items():
            with self.subTest(reason):
                with self.assertRaises(ValueError) as cm:
                    cr.parse_line(line)
                self.assertEqual(str(cm.exception), reason)


class Helpers(unittest.TestCase):
    def test_unwrap(self):
        ts = np.array([2 ** 32 - 10, 5, 15])
        self.assertEqual(list(cr.unwrap_timestamps(ts)), [2 ** 32 - 10, 2 ** 32 + 5, 2 ** 32 + 15])

    def test_missing(self):
        self.assertEqual(cr.count_missing(np.array([1, 2, 3, 4])), 0)
        self.assertEqual(cr.count_missing(np.array([1, 2, 5, 6])), 2)
        self.assertEqual(cr.count_missing(np.array([2 ** 32 - 1, 0, 1])), 0)  # wraparound
        self.assertEqual(cr.count_missing(np.array([10, 11, 0, 1])), 0)  # transmitter restart


class CollectorTest(unittest.TestCase):
    def test_end_to_end(self):
        c = cr.Collector()
        c.feed("# CSI_FMT,1,...")
        for i in (0, 1, 2, 4, 5):  # tx_seq 3 lost over the air
            c.feed(make_line(rx_idx=i if i < 3 else i - 1, tx_seq=i, ts=1000 + i * 10000))
        c.feed("CSI_DATA,garbage")
        c.feed(make_line(rx_idx=9, tx_seq=6, n=10))  # changed CSI length
        s = c.stats()
        self.assertEqual(s["frames_ok"], 5)
        self.assertEqual(s["lost_radio_tx_seq"], 1)
        self.assertEqual(s["lost_usb_rx_idx"], 0)
        self.assertEqual(s["bad_reasons"], {"bad_field_count": 1, "csi_len_changed": 1})
        with tempfile.TemporaryDirectory() as d:
            path = c.save(Path(d))
            data = np.load(path)
            self.assertEqual(data["csi"].shape, (5, 8))
            self.assertEqual(len(data["pc_time"]), 5)

    def test_cli_replay(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "in.txt"
            src.write_text("\n".join(make_line(i, i, 1000 + i) for i in range(3)) + "\n")
            result = CliRunner().invoke(cr.app, ["--replay", str(src), "--out", str(Path(d) / "out")])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(len(list((Path(d) / "out").glob("*.npz"))), 1)

    def test_cli_needs_exactly_one_source(self):
        with tempfile.TemporaryDirectory() as d:
            result = CliRunner().invoke(cr.app, ["--out", d])
            self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
