# Ultralytics 🚀 AGPL-3.0 License

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.lane.dataset import LaneRobotDataset
from ultralytics.models.yolo.lane.plotting import decode_lane, save_lane_grid
from ultralytics.utils import LOGGER


def get_lane_head(model):
    """Return the LaneRobot/LaneRobotV2/LaneRobotV2Independent head from a wrapped, fused, EMA, or raw model object."""
    from ultralytics.nn.modules.head import LaneRobot, LaneRobotV2, LaneRobotV2Independent

    seen = set()

    def walk(obj):
        if obj is None:
            return None
        oid = id(obj)
        if oid in seen:
            return None
        seen.add(oid)
        if isinstance(obj, (LaneRobot, LaneRobotV2, LaneRobotV2Independent)):
            return obj
        seq = None
        if isinstance(obj, (list, tuple, torch.nn.ModuleList, torch.nn.Sequential)):
            seq = obj
        elif hasattr(obj, "model") and getattr(obj, "model") is not obj:
            found = walk(getattr(obj, "model"))
            if found is not None:
                return found
        if seq is not None:
            for item in reversed(seq):
                found = walk(item)
                if found is not None:
                    return found
        if isinstance(obj, torch.nn.Module):
            for m in obj.modules():
                if isinstance(m, (LaneRobot, LaneRobotV2, LaneRobotV2Independent)):
                    return m
        return None

    head = walk(model)
    if head is None:
        raise AttributeError("Could not locate LaneRobot/LaneRobotV2/LaneRobotV2Independent head on validator model.")
    return head


class LaneRobotMetrics:
    """Lane metrics with location quality separated from existence failures."""

    def __init__(self):
        self.keys = [
            "metrics/lane_matched_mae",
            "metrics/lane_matched_mae_px",
            "metrics/lane_acc_valid_tol1",
            "metrics/lane_acc_valid_tol3",
            "metrics/lane_acc_valid_tol5",
            "metrics/lane_miss_rate",
            "metrics/lane_exist_precision",
            "metrics/lane_exist_recall",
            "metrics/lane_exist_f1",
            "metrics/lane_exist_acc",
        ]
        self.speed = None
        self.save_dir = None
        self.reset()

    def reset(self):
        """Reset accumulated counts and published values for a new validation run."""
        self.matched_mae_sum = 0.0
        self.matched_mae_px_sum = 0.0
        self.matched_total = 0
        self.valid_total = 0
        self.tol1 = 0
        self.tol3 = 0
        self.tol5 = 0
        self.exist_tp = 0
        self.exist_fp = 0
        self.exist_fn = 0
        self.exist_tn = 0
        self.lane_matched_mae = 0.0
        self.lane_matched_mae_px = 0.0
        self.lane_acc_valid_tol1 = 0.0
        self.lane_acc_valid_tol3 = 0.0
        self.lane_acc_valid_tol5 = 0.0
        self.lane_miss_rate = 0.0
        self.lane_exist_precision = 0.0
        self.lane_exist_recall = 0.0
        self.lane_exist_f1 = 0.0
        self.lane_exist_acc = 0.0
        self.fitness = 0.0

    @staticmethod
    def _divide(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    def update(self, pred_x, target_x, image_width, x_grids):
        """Accumulate one batch without treating the no-lane sentinel as a coordinate."""
        valid = target_x >= 0
        pred_valid = pred_x >= 0
        matched = valid & pred_valid

        batch_valid = int(valid.sum().item())
        batch_matched = int(matched.sum().item())
        self.valid_total += batch_valid
        self.matched_total += batch_matched

        if batch_matched:
            matched_err = (pred_x[matched] - target_x[matched]).abs()
            self.matched_mae_sum += float(matched_err.sum().item())
            pixel_scale = max(float(image_width) - 1.0, 1.0) / max(int(x_grids) - 1, 1)
            self.matched_mae_px_sum += float((matched_err * pixel_scale).sum().item())
            # The denominator remains all valid GT rows, so a missed row fails every location tolerance.
            self.tol1 += int((matched_err <= 1).sum().item())
            self.tol3 += int((matched_err <= 3).sum().item())
            self.tol5 += int((matched_err <= 5).sum().item())

        self.exist_tp += int((valid & pred_valid).sum().item())
        self.exist_fp += int((~valid & pred_valid).sum().item())
        self.exist_fn += int((valid & ~pred_valid).sum().item())
        self.exist_tn += int((~valid & ~pred_valid).sum().item())

    def compute(self):
        """Publish aggregate metrics and return the trainer-compatible result dictionary."""
        self.lane_matched_mae = self._divide(self.matched_mae_sum, self.matched_total)
        self.lane_matched_mae_px = self._divide(self.matched_mae_px_sum, self.matched_total)
        self.lane_acc_valid_tol1 = self._divide(self.tol1, self.valid_total)
        self.lane_acc_valid_tol3 = self._divide(self.tol3, self.valid_total)
        self.lane_acc_valid_tol5 = self._divide(self.tol5, self.valid_total)
        self.lane_miss_rate = self._divide(self.exist_fn, self.exist_tp + self.exist_fn)
        self.lane_exist_precision = self._divide(self.exist_tp, self.exist_tp + self.exist_fp)
        self.lane_exist_recall = self._divide(self.exist_tp, self.exist_tp + self.exist_fn)
        self.lane_exist_f1 = self._divide(
            2 * self.lane_exist_precision * self.lane_exist_recall,
            self.lane_exist_precision + self.lane_exist_recall,
        )
        self.lane_exist_acc = self._divide(
            self.exist_tp + self.exist_tn,
            self.exist_tp + self.exist_fp + self.exist_fn + self.exist_tn,
        )
        self.fitness = float(
            self.lane_acc_valid_tol3
            + 0.5 * self.lane_acc_valid_tol5
            - 0.003 * self.lane_matched_mae
            + 0.05 * self.lane_exist_f1
        )
        return self.results_dict

    @property
    def results_dict(self):
        return {
            "metrics/lane_matched_mae": self.lane_matched_mae,
            "metrics/lane_matched_mae_px": self.lane_matched_mae_px,
            "metrics/lane_acc_valid_tol1": self.lane_acc_valid_tol1,
            "metrics/lane_acc_valid_tol3": self.lane_acc_valid_tol3,
            "metrics/lane_acc_valid_tol5": self.lane_acc_valid_tol5,
            "metrics/lane_miss_rate": self.lane_miss_rate,
            "metrics/lane_exist_precision": self.lane_exist_precision,
            "metrics/lane_exist_recall": self.lane_exist_recall,
            "metrics/lane_exist_f1": self.lane_exist_f1,
            "metrics/lane_exist_acc": self.lane_exist_acc,
            "fitness": self.fitness,
        }

    def mean_results(self):
        return [self.results_dict[k] for k in self.keys]


class LaneRobotValidator(BaseValidator):
    """Validator for row-anchor lane classification."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self.args.task = "lane"
        self.metrics = LaneRobotMetrics()

    def build_dataset(self, img_path):
        return LaneRobotDataset(img_path, self.args, self.data, mode=self.args.split or "val")

    def get_dataloader(self, dataset_path, batch_size):
        dataset = self.build_dataset(dataset_path)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=LaneRobotDataset.collate_fn,
            drop_last=False,
        )

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda").float() / 255.0
        batch["lane"] = batch["lane"].to(self.device, non_blocking=self.device.type == "cuda").long()
        if "lane_x" in batch:
            batch["lane_x"] = batch["lane_x"].to(self.device, non_blocking=self.device.type == "cuda").float()
        if "lane_y" in batch:
            batch["lane_y"] = batch["lane_y"].to(self.device, non_blocking=self.device.type == "cuda").float()
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def init_metrics(self, model):
        head = get_lane_head(model)
        self.x_grids = int(head.x_grids)
        self.row_anchors = int(head.row_anchors)
        self.num_lanes = int(head.num_lanes)
        self.no_lane_idx = self.x_grids
        self.metrics.reset()

    def _split_preds(self, preds):
        if isinstance(preds, dict):
            return preds["cls"], preds.get("offset", None)
        if isinstance(preds, (list, tuple)) and preds and isinstance(preds[0], dict):
            return preds[0]["cls"], preds[0].get("offset", None)
        return preds, None

    def update_metrics(self, preds, batch):
        logits, offset = self._split_preds(preds)
        pred_xy = decode_lane(
            preds,
            no_lane_idx=self.no_lane_idx,
            topk=int(getattr(self.args, "lane_softargmax_topk", 5)),
            exist_thr=float(getattr(self.args, "lane_exist_thr", 0.5)),
            post_smooth=False,
        )
        pred_x = torch.as_tensor(pred_xy, device=logits.device, dtype=torch.float32)
        target_x = batch.get("lane_x", None)
        if target_x is None:
            target_x = batch["lane"].float()
            target_x = torch.where(batch["lane"] == self.no_lane_idx, torch.full_like(target_x, -1.0), target_x)
        self.metrics.update(pred_x, target_x, image_width=batch["img"].shape[-1], x_grids=self.x_grids)

    def get_stats(self):
        return self.metrics.compute()

    def finalize_metrics(self):
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def print_results(self):
        stats = self.get_stats()
        LOGGER.info(
            f"Lane matched_MAE={stats['metrics/lane_matched_mae']:.3f}, "
            f"matched_MAE_px={stats['metrics/lane_matched_mae_px']:.2f}, "
            f"Acc@1={stats['metrics/lane_acc_valid_tol1']:.4f}, "
            f"Acc@3={stats['metrics/lane_acc_valid_tol3']:.4f}, "
            f"Acc@5={stats['metrics/lane_acc_valid_tol5']:.4f}, "
            f"Miss={stats['metrics/lane_miss_rate']:.4f}, "
            f"Exist(P/R/F1)={stats['metrics/lane_exist_precision']:.4f}/"
            f"{stats['metrics/lane_exist_recall']:.4f}/{stats['metrics/lane_exist_f1']:.4f}, "
            f"ExistAcc={stats['metrics/lane_exist_acc']:.4f}"
        )

    def get_desc(self):
        return ("%22s" + "%11s" * 10) % (
            "lane",
            "match_mae",
            "mae_px",
            "acc@1",
            "acc@3",
            "acc@5",
            "miss",
            "exist_p",
            "exist_r",
            "exist_f1",
            "exist_acc",
        )

    @property
    def metric_keys(self):
        return self.metrics.keys

    def plot_val_samples(self, batch, ni):
        tgt = batch.get("lane_x", batch["lane"].float()).detach().cpu().numpy().copy()
        tgt[tgt >= self.x_grids] = -1
        save_lane_grid(
            batch["img"].detach().cpu(),
            tgt,
            tgt,
            x_grids=self.x_grids,
            row_anchors=self.row_anchors,
            save_path=self.save_dir / f"val_batch{ni}_labels.jpg",
            row_y=batch.get("lane_y"),
            y_start=float(getattr(self.args, "lane_y_start", 1.0)),
            y_end=float(getattr(self.args, "lane_y_end", 0.3333333333)),
        )

    def plot_predictions(self, batch, preds, ni):
        pred_xy = decode_lane(
            preds,
            no_lane_idx=self.no_lane_idx,
            topk=int(getattr(self.args, "lane_softargmax_topk", 5)),
            exist_thr=float(getattr(self.args, "lane_exist_thr", 0.5)),
            post_smooth=bool(getattr(self.args, "lane_post_smooth", True)),
            poly_degree=int(getattr(self.args, "lane_poly_degree", 2)),
            poly_blend=float(getattr(self.args, "lane_poly_blend", 0.5)),
        )
        tgt = batch.get("lane_x", batch["lane"].float()).detach().cpu().numpy().copy()
        tgt[tgt >= self.x_grids] = -1
        save_lane_grid(
            batch["img"].detach().cpu(),
            pred_xy,
            tgt,
            x_grids=self.x_grids,
            row_anchors=self.row_anchors,
            save_path=self.save_dir / f"val_batch{ni}_pred.jpg",
            row_y=batch.get("lane_y"),
            y_start=float(getattr(self.args, "lane_y_start", 1.0)),
            y_end=float(getattr(self.args, "lane_y_end", 0.3333333333)),
        )
