from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from ultralytics.models.yolo.lane.dataset import LaneRobotDataset
from ultralytics.models.yolo.lane.protocol import (
    MANUAL_ROW_ANCHORS,
    MANUAL_X_BINS,
    MANUAL_Y_END,
    MANUAL_Y_START,
    manual_y_anchors,
)


class TestTaskNavManualProtocol(unittest.TestCase):
    @staticmethod
    def _load_manual_script():
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "manual_fix_56anchors_v11_class1_789_update.py"
        spec = importlib.util.spec_from_file_location("tasknav_manual_annotation", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _line(xs=None, ys=None, task_id=0):
        xs = np.full(MANUAL_ROW_ANCHORS, 0.6, dtype=np.float64) if xs is None else np.asarray(xs)
        ys = manual_y_anchors(dtype=np.float64) if ys is None else np.asarray(ys)
        fields = [str(task_id)]
        for x, y in zip(xs, ys):
            fields.extend(("-1" if x == -1 else f"{x:.6f}", f"{y:.6f}"))
        return " ".join(fields) + "\n"

    def _make_dataset(self, label_text=None, *, write_label=True, y_start=MANUAL_Y_START, y_end=MANUAL_Y_END):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        root = Path(tempdir.name)
        image_dir = root / "images" / "train"
        label_dir = root / "labels" / "train"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        Image.new("RGB", (32, 24), color=(20, 30, 40)).save(image_dir / "sample.jpg")
        if write_label:
            content = label_text if label_text is not None else self._line()
            (label_dir / "sample.txt").write_text(content, encoding="utf-8")

        args = SimpleNamespace(
            imgsz=64,
            lane_row_anchors=MANUAL_ROW_ANCHORS,
            lane_x_grids=160,
            lane_num_lanes=4,
            lane_y_start=y_start,
            lane_y_end=y_end,
            lane_strict_labels=True,
        )
        data = {
            "path": str(root),
            "train_labels": "labels/train",
            "row_anchors": MANUAL_ROW_ANCHORS,
            "x_grids": 160,
            "num_lanes": 4,
            "y_start": y_start,
            "y_end": y_end,
            "strict_labels": True,
        }
        dataset = LaneRobotDataset(image_dir, args, data, mode="train")
        return dataset

    def test_training_protocol_matches_manual_script(self):
        manual = self._load_manual_script()
        self.assertEqual(MANUAL_ROW_ANCHORS, manual.NUM_Y_ANCHORS)
        self.assertEqual(MANUAL_X_BINS, manual.NUM_X_BINS)
        self.assertAlmostEqual(MANUAL_Y_START, 1.0)
        self.assertAlmostEqual(MANUAL_Y_END, 1.0 - manual.CROP_RATIO)
        np.testing.assert_allclose(manual_y_anchors(np.float64), manual.make_y_anchor_norms(), rtol=0.0, atol=0.0)

    def test_dataset_maps_normalized_x_and_absent_rows(self):
        xs = np.full(MANUAL_ROW_ANCHORS, 0.6, dtype=np.float64)
        xs[0] = -1.0
        sample = self._make_dataset(self._line(xs=xs))[0]
        self.assertEqual(tuple(sample["lane"].shape), (56, 4))
        self.assertEqual(int(sample["lane"][0, 0]), 160)
        self.assertEqual(float(sample["lane_x"][0, 0]), -1.0)
        self.assertAlmostEqual(float(sample["lane_x"][1, 0]), 0.6 * 159, places=4)
        self.assertEqual(int(sample["lane"][1, 0]), round(0.6 * 159))
        np.testing.assert_allclose(sample["lane_y"].numpy(), manual_y_anchors(), rtol=0.0, atol=0.0)

    def test_preflight_accepts_valid_and_empty_manual_labels(self):
        dataset = self._make_dataset(self._line())
        stats = dataset.preflight_validate_labels()
        self.assertEqual(stats, {"images": 1, "labels": 1, "annotations": 1, "empty_labels": 0})

        empty_dataset = self._make_dataset("")
        empty_stats = empty_dataset.preflight_validate_labels()
        self.assertEqual(empty_stats, {"images": 1, "labels": 1, "annotations": 0, "empty_labels": 1})

    def test_preflight_rejects_bad_label_before_getitem(self):
        dataset = self._make_dataset("0 0.5 1.0\n")
        with self.assertRaisesRegex(ValueError, "expected 113"):
            dataset.preflight_validate_labels()

    def test_preflight_rejects_missing_label(self):
        dataset = self._make_dataset(write_label=False)
        with self.assertRaisesRegex(FileNotFoundError, "preflight"):
            dataset.preflight_validate_labels()

    def test_dataset_rejects_wrong_point_count(self):
        dataset = self._make_dataset("0 0.5 1.0\n")
        with self.assertRaisesRegex(ValueError, "expected 113"):
            _ = dataset[0]

    def test_dataset_rejects_wrong_y_order(self):
        ys = manual_y_anchors(dtype=np.float64)[::-1]
        dataset = self._make_dataset(self._line(ys=ys))
        with self.assertRaisesRegex(ValueError, "bottom-to-top order"):
            _ = dataset[0]

    def test_dataset_rejects_non_manual_negative_x(self):
        xs = np.full(MANUAL_ROW_ANCHORS, 0.6, dtype=np.float64)
        xs[10] = -2.0
        dataset = self._make_dataset(self._line(xs=xs))
        with self.assertRaisesRegex(ValueError, "expected -1"):
            _ = dataset[0]

    def test_dataset_rejects_missing_label(self):
        dataset = self._make_dataset(write_label=False)
        with self.assertRaises(FileNotFoundError):
            _ = dataset[0]

    def test_dataset_rejects_geometry_config_drift(self):
        with self.assertRaisesRegex(ValueError, "requires y_start"):
            self._make_dataset(y_start=1.0 / 3.0, y_end=1.0)


if __name__ == "__main__":
    unittest.main()
