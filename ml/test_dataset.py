"""Tests for dataset. Run from the repo root: python -m unittest discover -s ml"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import dataset as ds  # noqa: E402
import preprocess as pp  # noqa: E402
from test_baseline import NAMES, SEGMENTS, make_session, save  # noqa: E402

CFG = pp.PreprocessConfig.load()


def build(d, seeds=(1,), segments=SEGMENTS, baseline=None, **kw):
    paths = [save(d, f"s{s}.npz", make_session(s, segments)) for s in seeds]
    return ds.load_dataset(paths, CFG, baseline, **kw)


class Loading(unittest.TestCase):
    def test_window_count_and_shape(self):
        with tempfile.TemporaryDirectory() as d:
            data = build(d)
        self.assertEqual(data.n_subcarriers, 52)
        # 5 segments x 60 s x 100 Hz = 30000 frames; windows of 128 at step 32 never cross a segment boundary
        per_segment = (6000 - 128) // 32 + 1
        self.assertAlmostEqual(len(data), 5 * per_segment, delta=10)
        x, y = data.batch(range(4))
        self.assertEqual(x.shape, (4, 128, 52))
        self.assertEqual((x.dtype, y.dtype), (np.float32, np.int64))

    def test_windows_never_mix_classes(self):
        with tempfile.TemporaryDirectory() as d:
            sess = make_session(1, SEGMENTS)
            path = save(d, "s.npz", sess)
            data = ds.load_dataset([path], CFG)
            for i in range(0, len(data), 37):
                start = int(data.starts[i])
                labels = sess["label"][start:start + 128]
                self.assertTrue((labels == labels[0]).all())
                self.assertEqual(int(labels[0]), int(data.labels[i]))

    def test_unlabeled_and_uncertain_are_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            sess = make_session(1, [("empty", 60), ("walking", 60)])
            sess["label"][:1000] = -1       # first 10 s unlabeled
            sess["uncertain"][5000:5500] = True
            data = ds.load_dataset([save(d, "s.npz", sess)], CFG)
            for s in data.starts:
                self.assertGreaterEqual(s, 1000)
                self.assertFalse(sess["uncertain"][s:s + 128].any())

    def test_normalized_with_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            sess = make_session(1, SEGMENTS)
            bl = pp.fit_baseline(sess["csi"][sess["label"] == 0][:3000], CFG)
            data = ds.load_dataset([save(d, "s.npz", sess)], CFG, bl)
            empty = data.subset(["empty"])
            x, _ = empty.batch(range(0, len(empty), 10))
            self.assertLess(abs(float(x.mean())), 0.2)
            walking, _ = data.subset(["walking"]).batch(range(0, 50))
            self.assertGreater(float(walking.std()), 3 * float(x.std()))

    def test_mismatched_class_names_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a = make_session(1, SEGMENTS)
            b = make_session(2, SEGMENTS)
            b["class_names"] = np.array(["a", "b", "c", "d"])
            with self.assertRaises(ValueError):
                ds.load_dataset([save(d, "a.npz", a), save(d, "b.npz", b)], CFG)


class Classes(unittest.TestCase):
    def test_subset_counts_and_weights(self):
        with tempfile.TemporaryDirectory() as d:
            data = build(d)
        counts = data.class_counts()
        self.assertEqual(counts.sum(), len(data))
        only = data.subset(["empty"])
        self.assertTrue((only.labels == NAMES.index("empty")).all())
        self.assertEqual(len(only), counts[NAMES.index("empty")])
        w = data.class_weights()
        self.assertAlmostEqual(float((w * counts).sum()) / len(data), 1.0, places=5)

    def test_weights_zero_for_missing_class(self):
        with tempfile.TemporaryDirectory() as d:
            data = build(d, segments=[("empty", 60), ("walking", 60)])
        w = data.class_weights()
        self.assertEqual(w[NAMES.index("static_person")], 0.0)
        self.assertGreater(w[NAMES.index("empty")], 0.0)


class Batches(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = build(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_iterates_everything_once(self):
        total = sum(len(y) for _, y in self.data.iter_batches(64))
        self.assertEqual(total, len(self.data))

    def test_drop_last(self):
        sizes = [len(y) for _, y in self.data.iter_batches(64, drop_last=True)]
        self.assertTrue(all(s == 64 for s in sizes))

    def test_shuffle_needs_rng_and_is_reproducible(self):
        with self.assertRaises(ValueError):
            next(self.data.iter_batches(8, shuffle=True))
        a = next(self.data.iter_batches(16, shuffle=True, rng=np.random.default_rng(3)))
        b = next(self.data.iter_batches(16, shuffle=True, rng=np.random.default_rng(3)))
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


class Augmentation(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(0).normal(0, 1, (128, 52)).astype(np.float32)

    def test_input_is_not_modified(self):
        before = self.x.copy()
        ds.augment(self.x, np.random.default_rng(0), ds.AugmentConfig(p=1.0))
        np.testing.assert_array_equal(self.x, before)

    def test_disabled_returns_same_values(self):
        cfg = ds.AugmentConfig(p=0.0)
        np.testing.assert_array_equal(ds.augment(self.x, np.random.default_rng(0), cfg), self.x)

    def test_noise_scale_and_dropout(self):
        rng = np.random.default_rng(1)
        noisy = ds.augment(self.x, rng, ds.AugmentConfig(p=1.0, noise_std=0.5, scale_range=(1.0, 1.0), zero_fraction=0))
        self.assertAlmostEqual(float((noisy - self.x).std()), 0.5, delta=0.05)
        scaled = ds.augment(self.x, rng, ds.AugmentConfig(p=1.0, noise_std=0, scale_range=(2.0, 2.0), zero_fraction=0))
        np.testing.assert_allclose(scaled, 2 * self.x, rtol=1e-6)
        dropped = ds.augment(self.x, rng, ds.AugmentConfig(p=1.0, noise_std=0, scale_range=(1.0, 1.0), zero_fraction=0.3))
        zero_cols = (dropped == 0).all(axis=0)
        self.assertGreater(zero_cols.sum(), 3)
        self.assertLess(zero_cols.sum(), 30)

    def test_shift_stays_inside_the_session(self):
        cfg = ds.AugmentConfig(p=1.0, max_shift=8)
        rng = np.random.default_rng(0)
        for start in (0, 5, 300, 372, 380):
            for _ in range(50):
                s = ds.jittered_start(start, 500, 128, rng, cfg)
                self.assertTrue(0 <= s <= 500 - 128)
                self.assertLessEqual(abs(s - start), 8 + max(0, start - 372))

    def test_augmented_batch_has_same_shape_and_labels(self):
        with tempfile.TemporaryDirectory() as d:
            data = build(d)
        rng = np.random.default_rng(0)
        x, y = next(data.iter_batches(32, shuffle=True, augment=ds.AugmentConfig(p=1.0), rng=rng))
        self.assertEqual(x.shape, (32, 128, 52))
        self.assertEqual(y.shape, (32,))


if __name__ == "__main__":
    unittest.main()
