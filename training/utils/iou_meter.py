# Drop this file at: training/utils/iou_meter.py
#
# Implements the meter interface expected by training/trainer.py:
#   - reset()
#   - update(...)                 -> called once per batch, per phase (train/val)
#   - compute_synced() -> Dict     -> called once per epoch, all-reduced across ranks
#   - is_better(new, old) -> bool  -> OPTIONAL. Only meters that define this are
#                                     candidates for `checkpoint.save_best_meters`.
#
# Trainer._step() calls:
#     meter.update(find_stages=outputs, find_metadatas=batch.metadata)
# which does NOT include the ground-truth masks, so `update()` alone can't compute
# a real (prediction vs. GT) IoU. This meter instead expects to be called with an
# extra `targets=` kwarg, which training/val_trainer.py (also provided) adds.

from typing import Dict, List

import torch

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized


class IoUMeter:
    """
    Tracks mean IoU between the model's final predicted mask (per object, per
    frame) and the ground-truth mask, for one dataset key (e.g. "all").

    Config (Hydra):
        _target_: training.utils.iou_meter.IoUMeter
    """

    def __init__(self, name: str = "iou"):
        self.name = name
        self.device = None
        self.reset()

    def reset(self):
        self._intersection = 0.0
        self._union = 0.0
        self._num_masks = 0

    @torch.no_grad()
    def update(self, find_stages=None, find_metadatas=None, targets=None, **kwargs):
        """
        Args:
            find_stages: the model's raw output (`outs_batch`), a list with one
                dict per frame, each dict containing
                "multistep_pred_multimasks_high_res": List[Tensor[N, M, H, W]]
                (same structure fed into training/loss_fns.py).
            targets: the ground-truth masks batch (same `targets_batch` that is
                passed to the loss, shape [num_frames, N, H, W] or [N, H, W]
                depending on how your subclass forwards it). Required to
                compute a real IoU — if it's not provided, this update is a
                no-op (so mis-wiring fails silently-but-visibly: iou stays 0).
        """
        if find_stages is None or targets is None:
            return

        outs_batch = find_stages
        targets_batch = targets
        if not isinstance(outs_batch, (list, tuple)):
            outs_batch = [outs_batch]
        if not isinstance(targets_batch, (list, tuple)):
            targets_batch = [targets_batch]

        for outs, target_masks in zip(outs_batch, targets_batch):
            # Use the last refinement step's prediction.
            pred_masks = outs["multistep_pred_multimasks_high_res"][-1]  # [N, M, H, W]
            if pred_masks.dim() == 4:
                # If multiple mask candidates (M>1) were predicted, take the
                # first channel (matches SAM2's default single-mask-per-click
                # eval convention; set multimask_output=False during val if
                # you want this to always be M=1).
                pred_masks = pred_masks[:, 0]  # [N, H, W]

            target_masks = target_masks.to(pred_masks.device)
            if target_masks.dim() == 4:
                target_masks = target_masks[:, 0]

            pred = pred_masks.sigmoid() > 0.5
            gt = target_masks > 0.5

            intersection = (pred & gt).flatten(1).sum(-1).float()
            union = (pred | gt).flatten(1).sum(-1).float()

            self._intersection += intersection.sum().item()
            self._union += union.sum().item()
            self._num_masks += pred.shape[0]

            if self.device is None:
                self.device = pred_masks.device

    def compute_synced(self) -> Dict[str, float]:
        stats = torch.tensor(
            [self._intersection, self._union, self._num_masks],
            dtype=torch.float64,
            device=self.device if self.device is not None else "cpu",
        )
        if is_dist_avail_and_initialized() and get_world_size() > 1:
            torch.distributed.all_reduce(stats)

        intersection, union, num_masks = stats.tolist()
        mean_iou = intersection / union if union > 0 else 0.0
        return {self.name: mean_iou}

    # Higher IoU is better. This is what makes `checkpoint.save_best_meters`
    # pick this meter up.
    def is_better(self, new_value: float, old_value: float) -> bool:
        return new_value > old_value

    def __str__(self):
        iou = self._intersection / self._union if self._union > 0 else 0.0
        return f"{self.name}: {iou:.4f}"