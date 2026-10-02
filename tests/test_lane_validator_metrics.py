from __future__ import annotations

import unittest

import torch

from ultralytics.models.yolo.lane.val import LaneRobotMetrics


class TestLaneRobotMetrics(unittest.TestCase):
    def test_matched_location_and_existence_metrics_are_separate(self):
        metrics = LaneRobotMetrics()
        target_x = torch.tensor([[[10.0], [20.0], [-1.0], [-1.0]]])
        pred_x = torch.tensor([[[11.0], [-1.0], [30.0], [-1.0]]])

        metrics.update(pred_x, target_x, image_width=160, x_grids=160)
        stats = metrics.compute()

        self.assertAlmostEqual(stats["metrics/lane_matched_mae"], 1.0)
        self.assertAlmostEqual(stats["metrics/lane_matched_mae_px"], 1.0)
        self.assertAlmostEqual(stats["metrics/lane_acc_valid_tol1"], 0.5)
        self.assertAlmostEqual(stats["metrics/lane_acc_valid_tol3"], 0.5)
        self.assertAlmostEqual(stats["metrics/lane_acc_valid_tol5"], 0.5)
        self.assertAlmostEqual(stats["metrics/lane_miss_rate"], 0.5)
        self.assertAlmostEqual(stats["metrics/lane_exist_precision"], 0.5)
        self.assertAlmostEqual(stats["metrics/lane_exist_recall"], 0.5)
        self.assertAlmostEqual(stats["metrics/lane_exist_f1"], 0.5)
        self.assertAlmostEqual(stats["metrics/lane_exist_acc"], 0.5)

    def test_all_misses_do_not_use_minus_one_as_a_coordinate(self):
        metrics = LaneRobotMetrics()
        target_x = torch.tensor([[[0.0], [100.0]]])
        pred_x = torch.full_like(target_x, -1.0)

        metrics.update(pred_x, target_x, image_width=640, x_grids=160)
        stats = metrics.compute()

        self.assertEqual(metrics.matched_total, 0)
        self.assertAlmostEqual(stats["metrics/lane_matched_mae"], 0.0)
        self.assertAlmostEqual(stats["metrics/lane_miss_rate"], 1.0)
        self.assertAlmostEqual(stats["metrics/lane_exist_recall"], 0.0)
        self.assertAlmostEqual(stats["metrics/lane_acc_valid_tol5"], 0.0)


if __name__ == "__main__":
    unittest.main()
