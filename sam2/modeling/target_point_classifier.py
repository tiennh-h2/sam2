"""Target-class point proposals for single-class SAM 2 segmentation."""

import torch
import torch.nn.functional as F
from torch import nn


class TargetPointClassifier(nn.Module):
    def __init__(self, channels: int, hidden_channels: int | None = None):
        super().__init__()
        hidden = hidden_channels or channels * 2

        self.head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),

            # Spatial context at low computational cost.
            nn.Conv2d(
                hidden, hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
            ),
            nn.GELU(),

            # Mix information across channels.
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),

            nn.Conv2d(hidden, 1, kernel_size=1),
        )

        for layer in self.head:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


def union_masks_per_image(masks: torch.Tensor, image_indices: torch.Tensor, num_images: int):
    """Union all annotated target instances on each image, leaving empty images empty."""
    result = torch.zeros((num_images, *masks.shape[-2:]), device=masks.device, dtype=torch.float32)
    if masks.shape[0]:
        result.index_add_(0, image_indices.long(), masks.float())
    return result > 0


def sample_negative_grid_points(target_union, grid_size=32, per_image=32):
    """Sample off-target locations, returned as image indices and positive-click xy."""
    batch, height, width = target_union.shape
    y = ((torch.arange(grid_size, device=target_union.device) + 0.5) * height / grid_size).long().clamp(max=height - 1)
    x = ((torch.arange(grid_size, device=target_union.device) + 0.5) * width / grid_size).long().clamp(max=width - 1)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    locations = torch.stack((gx.flatten(), gy.flatten()), 1)
    indices, selected = [], []
    for image_id in range(batch):
        eligible = locations[~target_union[image_id, locations[:, 1], locations[:, 0]]]
        if len(eligible) == 0:
            continue
        chosen = eligible[torch.randperm(len(eligible), device=eligible.device)[:per_image]]
        selected.append(chosen.float() + 0.5)
        indices.append(torch.full((len(chosen),), image_id, dtype=torch.long, device=eligible.device))
    if not indices:
        return (torch.empty(0, dtype=torch.long, device=target_union.device),
                torch.empty(0, 2, device=target_union.device))
    return torch.cat(indices), torch.cat(selected)


def sample_point_logits(logits: torch.Tensor, points: torch.Tensor, image_hw):
    """Sample a single image's score map at pixel-center coordinates (x, y)."""
    height, width = image_hw
    grid = points.to(device=logits.device, dtype=logits.dtype).clone()
    grid[:, 0] = 2 * grid[:, 0] / width - 1
    grid[:, 1] = 2 * grid[:, 1] / height - 1
    return F.grid_sample(logits, grid.view(1, -1, 1, 2), align_corners=False).flatten()


def select_target_points(logits, points, image_hw, threshold=0.5, min_points=1):
    """Boolean keep mask in the original point order; optionally retain fallback points."""
    if not 0 <= threshold <= 1 or min_points < 0:
        raise ValueError("threshold must be in [0, 1] and min_points must be nonnegative")
    if len(points) == 0:
        return torch.zeros(0, device=logits.device, dtype=torch.bool)
    scores = sample_point_logits(logits, points, image_hw)
    keep = scores.sigmoid() >= threshold
    if keep.sum() < min_points:
        keep[scores.topk(min(min_points, len(points))).indices] = True
    return keep


def point_classification_loss(logits, target_union, grid_size=32, boundary_ignore=2, gamma=2.0):
    """Equal positive/negative contribution when both groups occur in an image."""
    if grid_size < 1 or boundary_ignore < 0:
        raise ValueError("grid_size must be positive and boundary_ignore nonnegative")
    batch, _, height, width = logits.shape
    if target_union.ndim != 3 or target_union.shape[0] != batch:
        raise ValueError("target_union must have shape [batch, image_height, image_width]")
    target_union = target_union.float().unsqueeze(1)
    coords = (torch.arange(grid_size, device=logits.device, dtype=logits.dtype) + 0.5) / grid_size * 2 - 1
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack((gx, gy), -1)[None].expand(batch, -1, -1, -1)
    pred = F.grid_sample(logits, grid, mode="bilinear", align_corners=False).flatten(1)
    label = F.grid_sample(target_union, grid, mode="nearest", align_corners=False).flatten(1)
    valid = torch.ones_like(label, dtype=torch.bool)
    if boundary_ignore:
        k = 2 * boundary_ignore + 1
        dilated = F.max_pool2d(target_union, k, stride=1, padding=boundary_ignore)
        eroded = 1 - F.max_pool2d(1 - target_union, k, stride=1, padding=boundary_ignore)
        boundary = (dilated != eroded).float()
        valid = F.grid_sample(boundary, grid, mode="nearest", align_corners=False).flatten(1) == 0
    ce = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
    pt = pred.sigmoid() * label + (1 - pred.sigmoid()) * (1 - label)
    focal = ce * (1 - pt).pow(gamma)
    positive = valid & (label > 0)
    negative = valid & (label == 0)
    pos_count = positive.sum(1)
    neg_count = negative.sum(1)
    pos_loss = (focal * positive).sum(1) / pos_count.clamp(min=1)
    neg_loss = (focal * negative).sum(1) / neg_count.clamp(min=1)
    return ((pos_loss + neg_loss) / ((pos_count > 0).float() + (neg_count > 0).float()).clamp(min=1)).mean()


def negative_decoder_losses(mask_logits, predicted_ious, object_score_logits):
    """A positive point on a non-target region should yield no target instance."""
    zeros = torch.zeros_like(mask_logits)
    ce = F.binary_cross_entropy_with_logits(mask_logits, zeros, reduction="none")
    mask_loss = (ce * mask_logits.sigmoid().square()).mean()
    iou_loss = predicted_ious.square().mean()
    class_loss = F.binary_cross_entropy_with_logits(
        object_score_logits, torch.zeros_like(object_score_logits)
    )
    return mask_loss, iou_loss, class_loss
