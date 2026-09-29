# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.



# This source code is licensed under the license found in the

# LICENSE file in the root directory of this source tree.



from collections import defaultdict

from typing import Dict, List



import torch

import torch.distributed

import torch.nn as nn

import torch.nn.functional as F



from training.trainer import CORE_LOSS_KEY



from training.utils.distributed import get_world_size, is_dist_avail_and_initialized





def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):

    """

    Compute the DICE loss, similar to generalized IOU for masks

    Args:

        inputs: A float tensor of arbitrary shape.

                The predictions for each example.

        targets: A float tensor with the same shape as inputs. Stores the binary

                 classification label for each element in inputs

                (0 for the negative class and 1 for the positive class).

        num_objects: Number of objects in the batch

        loss_on_multimask: True if multimask prediction is enabled

    Returns:

        Dice loss tensor

    """

    inputs = inputs.sigmoid()

    if loss_on_multimask:

        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks

        assert inputs.dim() == 4 and targets.dim() == 4

        # flatten spatial dimension while keeping multimask channel dimension

        inputs = inputs.flatten(2)

        targets = targets.flatten(2)

        numerator = 2 * (inputs * targets).sum(-1)

    else:

        inputs = inputs.flatten(1)

        numerator = 2 * (inputs * targets).sum(1)

    denominator = inputs.sum(-1) + targets.sum(-1)

    loss = 1 - (numerator + 1) / (denominator + 1)

    if loss_on_multimask:

        return loss / num_objects

    return loss.sum() / num_objects





def sigmoid_focal_loss(

    inputs,

    targets,

    num_objects,

    alpha: float = 0.25,

    gamma: float = 2,

    loss_on_multimask=False,

):

    """

    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:

        inputs: A float tensor of arbitrary shape.

                The predictions for each example.

        targets: A float tensor with the same shape as inputs. Stores the binary

                 classification label for each element in inputs

                (0 for the negative class and 1 for the positive class).

        num_objects: Number of objects in the batch

        alpha: (optional) Weighting factor in range (0,1) to balance

                positive vs negative examples. Default = -1 (no weighting).

        gamma: Exponent of the modulating factor (1 - p_t) to

               balance easy vs hard examples.

        loss_on_multimask: True if multimask prediction is enabled

    Returns:

        focal loss tensor

    """

    prob = inputs.sigmoid()

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    p_t = prob * targets + (1 - prob) * (1 - targets)

    loss = ce_loss * ((1 - p_t) ** gamma)



    if alpha >= 0:

        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

        loss = alpha_t * loss



    if loss_on_multimask:

        # loss is [N, M, H, W] where M corresponds to multiple predicted masks

        assert loss.dim() == 4

        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims

    return loss.mean(1).sum() / num_objects





def iou_loss(

    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False

):

    """

    Args:

        inputs: A float tensor of arbitrary shape.

                The predictions for each example.

        targets: A float tensor with the same shape as inputs. Stores the binary

                 classification label for each element in inputs

                (0 for the negative class and 1 for the positive class).

        pred_ious: A float tensor containing the predicted IoUs scores per mask

        num_objects: Number of objects in the batch

        loss_on_multimask: True if multimask prediction is enabled

        use_l1_loss: Whether to use L1 loss is used instead of MSE loss

    Returns:

        IoU loss tensor

    """

    assert inputs.dim() == 4 and targets.dim() == 4

    pred_mask = inputs.flatten(2) > 0

    gt_mask = targets.flatten(2) > 0

    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()

    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()

    actual_ious = area_i / torch.clamp(area_u, min=1.0)



    if use_l1_loss:

        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")

    else:

        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")

    if loss_on_multimask:

        return loss / num_objects

    return loss.sum() / num_objects





class MultiStepMultiMasksAndIous(nn.Module):
    """Balanced existence BCE, region-balanced mask BCE, positive Dice and IoU.

    API-compatible constructor. Focal parameters remain accepted for existing
    YAMLs but are unused by this class: BCE replaces focal loss deliberately.
    Raw, ungated logits must reach this loss. No runtime weight controller.
    Frames and correction steps retain the original summed-loss convention.
    """

    def __init__(self, weight_dict, focal_alpha=0.25, focal_gamma=2,
                 supervise_all_iou=False, iou_use_l1_loss=False,
                 pred_obj_scores=False, focal_gamma_obj_score=0.0,
                 focal_alpha_obj_score=-1):
        super().__init__()
        self.weight_dict = dict(weight_dict)
        for key in ('loss_mask', 'loss_dice', 'loss_iou'):
            if key not in self.weight_dict:
                raise ValueError(f'Missing loss weight: {key}')
        self.weight_dict.setdefault('loss_class', 0.0)
        self.pred_obj_scores = pred_obj_scores
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        if targets_batch.ndim != 4 or len(outs_batch) != len(targets_batch):
            raise ValueError('Expected targets [T,N,H,W] and T output dictionaries.')
        # Use GLOBAL counts before clamping: rare groups with fewer examples
        # than ranks must still get their full contribution after DDP averaging.
        positive = (targets_batch > 0).flatten(2).any(-1)
        counts = torch.stack((positive.sum(1), (~positive).sum(1)), dim=1).float()
        world_size = 1
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(counts)
            world_size = get_world_size()
        count_rows = counts.tolist()
        losses = defaultdict(int)
        for out, target, (npos, nneg) in zip(outs_batch, targets_batch, count_rows):
            groups = int(npos > 0) + int(nneg > 0)
            group_weight = 1.0 / max(groups, 1)
            pos_scale = world_size / max(npos, 1.0)
            neg_scale = world_size / max(nneg, 1.0)
            masks = out['multistep_pred_multimasks_high_res']
            ious = out['multistep_pred_ious']
            scores = out['multistep_object_score_logits']
            if not masks or not (len(masks) == len(ious) == len(scores)):
                raise ValueError('Prediction-step lists must be nonempty and aligned.')
            for z, iou, score in zip(masks, ious, scores):
                self._update_losses(losses, z, target, iou, score,
                                    pos_scale, neg_scale, group_weight)
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return dict(losses)

    def _update_losses(self, losses, src_masks, targets, ious, object_logits,
                       pos_scale, neg_scale, group_weight):
        z = src_masks.float()
        if z.ndim != 4 or z.shape[1] < 1:
            raise ValueError('Expected mask logits [N,M,H,W], M >= 1.')
        if targets.shape != (z.shape[0], *z.shape[-2:]):
            raise ValueError('GT and prediction shapes do not match.')
        if ious.shape != z.shape[:2]:
            raise ValueError('Expected IoU predictions [N,M].')
        y = (targets > 0).unsqueeze(1).expand_as(z).float()
        pixel_count = z.shape[-2] * z.shape[-1]
        if pixel_count == 0:
            raise ValueError('Masks must have nonzero spatial size.')
        fg_count = y.flatten(2).sum(-1)
        bg_count = pixel_count - fg_count
        positive = (fg_count[:, 0] > 0).float()
        negative = 1.0 - positive

        # Positive masks: equal foreground/background region coefficients.
        # Empty/full masks: use the only available region at full weight.
        bce = F.binary_cross_entropy_with_logits(z, y, reduction='none')
        fg_mean = (bce * y).flatten(2).sum(-1) / fg_count.clamp_min(1)
        bg_mean = (bce * (1-y)).flatten(2).sum(-1) / bg_count.clamp_min(1)
        regions = (fg_count > 0).float() + (bg_count > 0).float()
        mask_each = (fg_mean + bg_mean) / regions.clamp_min(1)
        dice_each = dice_loss(z, y, 1.0, loss_on_multimask=True)
        iou_each = iou_loss(z, y, ious.float(), 1.0,
                            loss_on_multimask=True,
                            use_l1_loss=self.iou_use_l1_loss)
        # Select by mask accuracy, never by the predicted object score.
        combo = (self.weight_dict['loss_mask'] * mask_each
                 + self.weight_dict['loss_dice'] * dice_each)
        best = combo.detach().argmin(1, keepdim=True)
        positive_mask = mask_each.gather(1, best).squeeze(1)
        positive_dice = dice_each.gather(1, best).squeeze(1)
        positive_iou = (iou_each.mean(1) if self.supervise_all_iou
                        else iou_each.gather(1, best).squeeze(1))

        mask_pos = (positive_mask * positive).sum() * pos_scale
        mask_neg = (mask_each.mean(1) * negative).sum() * neg_scale
        iou_pos = (positive_iou * positive).sum() * pos_scale
        iou_neg = (iou_each.mean(1) * negative).sum() * neg_scale
        losses['loss_mask'] += group_weight * (mask_pos + mask_neg)
        losses['loss_iou'] += group_weight * (iou_pos + iou_neg)
        # Dice is a positive-only shape objective, with its own outer weight.
        losses['loss_dice'] += (positive_dice * positive).sum() * pos_scale

        if self.pred_obj_scores:
            if object_logits.numel() != z.shape[0]:
                raise ValueError('Expected one object score per query.')
            class_each = F.binary_cross_entropy_with_logits(
                object_logits.float().reshape(-1), positive, reduction='none')
            class_pos = (class_each * positive).sum() * pos_scale
            class_neg = (class_each * negative).sum() * neg_scale
        else:
            class_pos = z.sum() * 0.0
            class_neg = z.sum() * 0.0
        losses['loss_class'] += group_weight * (class_pos + class_neg)
        # Detached, unweighted group means for logging; not extra objectives.
        for name, value in (('mask_positive', mask_pos), ('mask_empty', mask_neg),
                            ('class_positive', class_pos), ('class_empty', class_neg),
                            ('iou_positive', iou_pos), ('iou_empty', iou_neg)):
            losses['debug_' + name] += value.detach()

    def reduce_loss(self, losses):
        total = 0.0
        for key, weight in self.weight_dict.items():
            if key.startswith('debug_') or key not in losses:
                raise ValueError(f'Unsupported objective: {key}')
            if weight:
                total = total + weight * losses[key]
        return total
