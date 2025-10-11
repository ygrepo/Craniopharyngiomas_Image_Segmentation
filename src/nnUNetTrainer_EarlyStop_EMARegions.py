# nnUNetTrainer_EarlyStop_EMARegions.py
from __future__ import annotations
import os, json, shutil
from typing import Dict, List, Sequence, Optional

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.evaluation.evaluate_predictions import (
    compute_metrics_on_folder,
    labels_to_list_of_regions,
)
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from nnunetv2.utilities.helpers import empty_cache, dummy_context


# ----------------------------
# Utility: per-class Dice (soft or hard)
# ----------------------------
def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # labels: (B, 1, D, H, W) or (B, D, H, W)
    if labels.ndim == 5 and labels.size(1) == 1:
        labels = labels[:, 0]
    return (
        F.one_hot(labels.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    )


def dice_per_class_from_logits(
    logits: torch.Tensor,  # (B, C, D, H, W)
    labels: torch.Tensor,  # (B, 1, D, H, W) or (B, D, H, W)
    eps: float = 1e-6,
    hard: bool = True,
) -> torch.Tensor:
    """
    Returns Dice per class averaged over batch: shape (C,).
    """
    C = logits.shape[1]
    if hard:
        pred = torch.argmax(logits, dim=1)  # (B, D, H, W)
        pred_1h = one_hot(pred, C)  # (B, C, D, H, W)
    else:
        pred_1h = F.softmax(logits, dim=1)  # soft probabilities

    gt_1h = one_hot(labels, C).to(logits.dtype)  # (B, C, D, H, W)

    intersect = (pred_1h * gt_1h).sum(dim=(0, 2, 3, 4))  # (C,)
    denom = pred_1h.sum(dim=(0, 2, 3, 4)) + gt_1h.sum(dim=(0, 2, 3, 4))  # (C,)
    dice = (2.0 * intersect + eps) / (denom + eps)  # (C,)
    return dice


class nnUNetTrainer_EarlyStop_EMARegions(nnUNetTrainer):
    # ---------- lifecycle ----------
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
        monitor_classes: Sequence[int] = (),
        reducer: str = "mean",
        ema_alpha: float = 0.1,
        weight_decay: float = 3e-5,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # If empty, default to all non-background classes (assuming 0 is background)
        self.monitor_classes = list(monitor_classes)  # e.g., [1,2,3]
        self.reducer = reducer.lower()
        assert self.reducer in ("mean", "min")

        self.ema_alpha = float(ema_alpha)
        self._ema_value: Optional[float] = None  # scalar EMA over monitored classes

        self.weight_decay = weight_decay

        # buffers for epoch-wise aggregation
        self._val_dice_sums: Optional[torch.Tensor] = None  # (C,)
        self._val_dice_counts: int = 0
        self._num_classes: Optional[int] = None

        self.print_to_log_file(
            f"[earlystop] monitor classes={self.monitor_classes}, reducer={self.reducer}, "
            f"alpha={self.ema_alpha}, weight_decay={self.weight_decay}"
        )

    # -------------
    # Validation epoch end (aggregate + EMA + log)
    # -------------
    def on_validation_epoch_start(self):
        self._val_dice_sums = None
        self._val_dice_counts = 0

    # ---------- hooks ----------
    def validation_step(self, batch: dict) -> dict:
        super().validation_step(batch)
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            logits = self.network(data)  # (B, C, D, H, W)

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # compute per-class Dice for this batch
        y = (
            target[0] if isinstance(target, list) else target
        )  # (B, 1, D, H, W) or (B, D, H, W)
        dice_c = dice_per_class_from_logits(logits, y, hard=True)  # (C,)
        dice_c = dice_per_class_from_logits(logits, y, hard=True)  # (C,)
        C = dice_c.numel()
        if self._num_classes is None:
            self._num_classes = C
        if self._val_dice_sums is None:
            self._val_dice_sums = torch.zeros(
                C, dtype=dice_c.dtype, device=dice_c.device
            )

        self._val_dice_sums += dice_c
        self._val_dice_counts += 1

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._val_dice_sums = None
        self._val_dice_counts = 0

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        
        if self._val_dice_sums is None or self._val_dice_counts == 0:
            # nothing to aggregate
            return

        mean_dice_per_class = self._val_dice_sums / float(self._val_dice_counts)  # (C,)
        # Decide which classes to monitor
        if not self.monitor_classes:
            # default: all non-background (1..C-1) if C>1
            classes = list(range(1, self._num_classes or len(mean_dice_per_class)))
        else:
            classes = self.monitor_classes

        # gather monitored values present in range
        selected = []
        for c in classes:
            if 0 <= c < mean_dice_per_class.numel():
                selected.append(mean_dice_per_class[c].item())

        # reducer
        if not selected:
            monitored_now = 0.0
        else:
            monitored_now = (
                sum(selected) / len(selected)
                if self.reducer == "mean"
                else min(selected)
            )

        # EMA update
        if self._ema_value is None:
            self._ema_value = float(monitored_now)
        else:
            self._ema_value = float(
                self.ema_alpha * monitored_now
                + (1.0 - self.ema_alpha) * self._ema_value
            )

        # Log everything Lightning can monitor/save
        # Raw per-class dice (optional: as scalars on the bar)
        for c_idx, val in enumerate(mean_dice_per_class.tolist()):
            self.log(
                f"val_dice_c{c_idx}",
                val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        # Aggregates
        self.log(
            "val_dice_monitored_now",
            monitored_now,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "ema_regions",
            self._ema_value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
