from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from ultralytics.nn.modules.head import SingleLaneRobotV2Branch
from ultralytics.utils.loss import LaneRobotLoss


class _DummyLaneHead(nn.Module):
    def __init__(self, x_grids=160, row_anchors=6, num_lanes=4):
        super().__init__()
        self.x_grids = x_grids
        self.row_anchors = row_anchors
        self.num_lanes = num_lanes


class _DummyLaneModel(nn.Module):
    def __init__(self, args, x_grids=160, row_anchors=6, num_lanes=4):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.model = nn.ModuleList([_DummyLaneHead(x_grids, row_anchors, num_lanes)])
        self.args = args


class TestLaneLossGeometry(unittest.TestCase):
    x_grids = 160
    rows = 6
    lanes = 4

    @classmethod
    def _criterion(cls, **overrides):
        values = {
            "lane_ce": 1.0,
            "lane_loc": 2.0,
            "lane_exist": 1.0,
            "lane_smooth": 0.03,
            "lane_curv": 0.02,
            "lane_offset": 3.0,
            "lane_soft_label": True,
            "lane_soft_sigma": 1.2,
            "lane_label_smoothing": 0.0,
            "lane_softargmax_topk": 5,
            "lane_task_weights": [1.0] * cls.lanes,
        }
        values.update(overrides)
        model = _DummyLaneModel(SimpleNamespace(**values), cls.x_grids, cls.rows, cls.lanes)
        return LaneRobotLoss(model)

    @classmethod
    def _peaked_logits(cls, target_class):
        logits = torch.full((1, cls.x_grids + 1, cls.rows, 1), -80.0)
        logits.scatter_(1, target_class.unsqueeze(1), 80.0)
        return logits

    def test_lane_loc_backward_reaches_cls_logits_and_offset(self):
        criterion = self._criterion()
        logits = torch.randn(1, self.x_grids + 1, self.rows, 1, requires_grad=True)
        offset = torch.zeros(1, 1, self.rows, 1, requires_grad=True)
        target_x = torch.tensor([20.2, 32.4, 48.1, 65.3, 87.2, 101.4]).view(1, self.rows, 1)
        target = target_x.round().long()
        valid = torch.ones_like(target, dtype=torch.bool)

        _, components = criterion._compute_single_task_loss(logits, offset, target, target_x, valid)
        components[1].backward()

        self.assertGreater(float(logits.grad[:, : self.x_grids].abs().sum()), 0.0)
        self.assertGreater(float(offset.grad.abs().sum()), 0.0)

    def test_gt_relative_geometry_accepts_a_perfect_curve_and_penalizes_a_line(self):
        criterion = self._criterion()
        target_x = torch.tensor([20.2, 22.2, 26.2, 32.2, 40.2, 50.2]).view(1, self.rows, 1)
        target = target_x.round().long()
        valid = torch.ones_like(target, dtype=torch.bool)

        curved_logits = self._peaked_logits(target)
        curved_offset = (target_x - target.float()).unsqueeze(1)
        _, curved = criterion._compute_single_task_loss(
            curved_logits, curved_offset, target, target_x, valid
        )

        straight_target = torch.full_like(target, 35)
        straight_logits = self._peaked_logits(straight_target)
        straight_offset = torch.zeros(1, 1, self.rows, 1)
        _, straight = criterion._compute_single_task_loss(
            straight_logits, straight_offset, target, target_x, valid
        )

        self.assertLess(float(curved[3]), 1e-10)
        self.assertLess(float(curved[4]), 1e-10)
        self.assertGreater(float(straight[3]), float(curved[3]) + 1e-5)
        self.assertGreater(float(straight[4]), float(curved[4]) + 1e-5)

    def test_invalid_rows_do_not_enter_pair_or_triplet_differences(self):
        criterion = self._criterion()
        target_x = torch.tensor([10.0, 12.0, -1.0, 40.0, 42.0, -1.0]).view(1, self.rows, 1)
        valid = target_x >= 0
        target = torch.where(valid, target_x.long(), torch.full_like(target_x.long(), self.x_grids))
        prediction_class = torch.tensor([10, 12, 159, 40, 42, 0]).view(1, self.rows, 1)
        logits = self._peaked_logits(prediction_class)
        offset = torch.zeros(1, 1, self.rows, 1)

        _, components = criterion._compute_single_task_loss(logits, offset, target, target_x, valid)

        self.assertLess(float(components[3]), 1e-10)
        self.assertLess(float(components[4]), 1e-10)

    def test_nearest_grid_signed_offset_regression(self):
        target_x = torch.tensor([100.8, 100.2], dtype=torch.float64)
        target_class = target_x.round()
        signed_offset = target_x - target_class

        torch.testing.assert_close(target_class, torch.tensor([101.0, 100.0], dtype=torch.float64))
        torch.testing.assert_close(
            signed_offset, torch.tensor([-0.2, 0.2], dtype=torch.float64), atol=1e-12, rtol=0.0
        )

    def test_four_task_random_loss_and_components_are_finite(self):
        torch.manual_seed(0)
        criterion = self._criterion(lane_label_smoothing=0.1)
        logits = torch.randn(2, self.x_grids + 1, self.rows, self.lanes, requires_grad=True)
        offset = (torch.rand(2, 1, self.rows, self.lanes) - 0.5).requires_grad_()
        target = torch.randint(0, self.x_grids, (2, self.rows, self.lanes))
        target[:, 2, :] = self.x_grids
        residual = torch.rand_like(target.float()) - 0.5
        target_x = torch.where(target == self.x_grids, torch.full_like(residual, -1.0), target.float() + residual)

        total, components = criterion({"cls": logits, "offset": offset}, {"lane": target, "lane_x": target_x})
        total.backward()

        self.assertTrue(bool(torch.isfinite(total)))
        self.assertTrue(bool(torch.isfinite(components).all()))
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(offset.grad)

    def test_hard_and_soft_label_smoothing_change_ce(self):
        torch.manual_seed(1)
        logits = torch.randn(1, self.x_grids + 1, self.rows, 1)
        target = torch.tensor([10, 20, 30, 40, 50, self.x_grids]).view(1, self.rows, 1)
        target_x = torch.where(target == self.x_grids, torch.full_like(target.float(), -1.0), target.float())
        valid = target != self.x_grids
        offset = torch.zeros(1, 1, self.rows, 1)

        hard_zero = self._criterion(lane_soft_label=False, lane_label_smoothing=0.0)
        hard_smooth = self._criterion(lane_soft_label=False, lane_label_smoothing=0.1)
        _, hard_zero_items = hard_zero._compute_single_task_loss(logits, offset, target, target_x, valid)
        _, hard_smooth_items = hard_smooth._compute_single_task_loss(logits, offset, target, target_x, valid)
        self.assertNotAlmostEqual(float(hard_zero_items[0]), float(hard_smooth_items[0]), places=6)

        soft_zero = self._criterion(lane_soft_label=True, lane_label_smoothing=0.0)
        soft_smooth = self._criterion(lane_soft_label=True, lane_label_smoothing=0.1)
        _, soft_zero_items = soft_zero._compute_single_task_loss(logits, offset, target, target_x, valid)
        _, soft_smooth_items = soft_smooth._compute_single_task_loss(logits, offset, target, target_x, valid)
        self.assertNotAlmostEqual(float(soft_zero_items[0]), float(soft_smooth_items[0]), places=6)

    def test_soft_visible_target_normalization_and_zero_smoothing_compatibility(self):
        criterion = self._criterion(lane_label_smoothing=0.0)
        logits = torch.zeros(1, self.x_grids + 1, self.rows, 1)
        target = torch.tensor([10, 20, 30, 40, 50, 60]).view(1, self.rows, 1)
        soft = criterion._soft_visible_targets(logits, target)

        grid = torch.arange(self.x_grids, dtype=logits.dtype).view(1, -1, 1, 1)
        expected = torch.exp(-0.5 * ((grid - target.float().unsqueeze(1)) / criterion.soft_sigma) ** 2)
        expected = expected / expected.sum(dim=1, keepdim=True)

        torch.testing.assert_close(soft[:, : self.x_grids], expected)
        torch.testing.assert_close(soft.sum(dim=1), torch.ones(1, self.rows, 1))
        self.assertEqual(float(soft[:, self.x_grids].abs().sum()), 0.0)

    def test_label_smoothing_rejects_values_outside_half_open_unit_interval(self):
        for value in (-0.01, 1.0, 1.2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self._criterion(lane_label_smoothing=value)

    def test_pool_ablation_branches_keep_output_shapes_and_backward(self):
        for feat_h, feat_w in ((8, 10), (10, 10), (16, 16)):
            with self.subTest(pool=(feat_h, feat_w)):
                branch = SingleLaneRobotV2Branch(
                    c1=4,
                    x_grids=self.x_grids,
                    row_anchors=56,
                    reduce_channels=2,
                    hidden_dim=16,
                    feat_h=feat_h,
                    feat_w=feat_w,
                )
                features = torch.randn(1, 4, 20, 20, requires_grad=True)
                cls_logits, offset = branch(features)
                self.assertEqual(tuple(cls_logits.shape), (1, 161, 56))
                self.assertEqual(tuple(offset.shape), (1, 1, 56))
                self.assertEqual(branch.flatten_dim, 2 * feat_h * feat_w)
                (cls_logits.mean() + offset.mean()).backward()
                self.assertIsNotNone(features.grad)


if __name__ == "__main__":
    unittest.main()
