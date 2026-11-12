"""Tests for tflite_to_c. Run from the repo root: python -m unittest discover -s ml"""
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import export_tflite as ex
import models
import tflite_to_c as t2c

CPP = Path(__file__).resolve().parents[1] / "firmware" / "rx" / "components" / "csi_model" / "csi_model.cpp"


def snake(name):
    """ExpandDims -> EXPAND_DIMS, Conv2D -> CONV_2D (TFLite operator names)."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[a-z])(?=\d)", "_", name).upper()


class Tflite2C(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = models.build_cnn(filters=(4, 4, 4))
        rep = np.random.default_rng(0).normal(0, 1, (20, 128, 52)).astype(np.float32)
        cls.tflite = ex.convert_int8(model, rep)

    def test_supported_ops_match_firmware_resolver(self):
        registered = [snake(n) for n in re.findall(r"s_resolver\.Add(\w+)\(\)", CPP.read_text())]
        self.assertEqual(sorted(registered), sorted(t2c.SUPPORTED_OPS))
        self.assertIn(f"MicroMutableOpResolver<{len(registered)}>", CPP.read_text())

    def test_cnn_uses_only_supported_ops(self):
        self.assertFalse(set(t2c.model_ops(self.tflite)) - set(t2c.SUPPORTED_OPS))

    def test_unsupported_op_is_rejected(self):
        with mock.patch.object(t2c, "model_ops", return_value=["CONV_2D", "TANH"]):
            with self.assertRaisesRegex(ValueError, "TANH"):
                t2c.render(self.tflite, "m.tflite")

    def test_render_contains_whole_model_aligned(self):
        text = t2c.render(self.tflite, "m.tflite")
        self.assertIn(f"csi_model_data_len = {len(self.tflite)};", text)
        self.assertIn("aligned(16)", text)
        body = re.search(r"csi_model_data\[\][^{]*\{(.*?)\n\};", text, re.S).group(1)
        data = bytes(int(h, 16) for h in re.findall(r"0x([0-9a-f]{2})", body))
        self.assertEqual(data, self.tflite)

    def test_golden_output_is_reproducible_and_int8(self):
        a, b = t2c.golden_output(self.tflite), t2c.golden_output(self.tflite)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.dtype, np.int8)
        self.assertEqual(len(a), models.N_CLASSES)

    def test_golden_input_formula(self):
        x = t2c.golden_input(300)
        self.assertEqual(int(x[0]), 11 - 127)
        self.assertEqual(int(x[1]), 48 - 127)
        self.assertTrue(x.min() >= -127 and x.max() <= 127)


if __name__ == "__main__":
    unittest.main()
