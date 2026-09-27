"""Instance-preserving grid prompts for GridNegativeSAM2Train.

The public function returns ONLY (points, labels), with one output row per
existing target, in unchanged order. The historical function name is retained
for callers; it does not reassign or create instances.

Nonempty targets prefer points on a shared cell-centered grid and fall back to
random foreground pixels if the grid misses the target. Empty targets receive
placeholder label -1: GridNegativeSAM2Train replaces those placeholders with
background coordinates and label 1 before decoding. Do not use this function
alone for empty targets without that replacement.
"""

from numbers import Integral
from typing import Optional, Tuple

import torch


@torch.no_grad()
def sample_grid_points_and_assign_instances(
    gt_masks: torch.Tensor,
    points_per_side: int = 32,
    *,
    num_pt: int = 1,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample grid-preferred points while preserving target count and order.

    Args:
        gt_masks: Boolean [N,1,H,W] or [N,H,W]. Each row is a fixed target;
            rows may belong to different images of the same spatial shape.
        points_per_side: Positive integer; grid contains this many squared
            candidate points. Coordinates stay in the supplied mask space.
        num_pt: Nonnegative number of points PER target. Sampling is with
            replacement, so repeated coordinates are possible.
        generator: Optional torch generator on the same device as gt_masks.

    Returns:
        points: float32 [N,num_pt,2], in (x,y) order.
        labels: int32 [N,num_pt]; 1 for foreground, -1 for empty placeholders.

    Grid membership uses floor(x/y), but returned grid coordinates retain their
    fractional values. Fallback foreground coordinates are integer pixels.
    Overlapping masks are sampled independently without changing their targets.
    This is an INITIAL prompt sampler, not a prediction-error sampler.

    There is no max_points/include_background option: target creation and
    negative-target capping belong to the existing dataset/model wrapper.
    """
    if gt_masks.dtype != torch.bool:
        raise TypeError("gt_masks must have dtype torch.bool")
    if gt_masks.ndim == 4 and gt_masks.shape[1] == 1:
        masks = gt_masks[:, 0]
    elif gt_masks.ndim == 3:
        masks = gt_masks
    else:
        raise ValueError("gt_masks must have shape [N,1,H,W] or [N,H,W]")
    if isinstance(points_per_side, bool) or not isinstance(points_per_side, Integral) or points_per_side < 1:
        raise ValueError("points_per_side must be a positive integer")
    if isinstance(num_pt, bool) or not isinstance(num_pt, Integral) or num_pt < 0:
        raise ValueError("num_pt must be a nonnegative integer")
    n, height, width = masks.shape
    if height < 1 or width < 1:
        raise ValueError("Mask height and width must be positive")
    device = masks.device
    points = torch.zeros((n, num_pt, 2), dtype=torch.float32, device=device)
    labels = torch.full((n, num_pt), -1, dtype=torch.int32, device=device)
    if n == 0 or num_pt == 0:
        return points, labels

    side = int(points_per_side)
    unit = (torch.arange(side, dtype=torch.float32, device=device) + 0.5) / side
    grid = torch.stack([
        (unit * width).repeat(side),
        (unit * height).repeat_interleave(side),
    ], dim=-1)
    xs = grid[:, 0].long().clamp(0, width - 1)
    ys = grid[:, 1].long().clamp(0, height - 1)
    inside = masks[:, ys, xs]  # [N, side**2], not [N,num_pt,H,W]
    has_grid = inside.any(dim=1)

    # Equal positive weights sample uniformly among eligible grid points.
    grid_rows = torch.where(has_grid)[0]
    if grid_rows.numel():
        selected = torch.multinomial(
            inside[grid_rows].float(), num_pt, replacement=True, generator=generator,
        )
        points[grid_rows] = grid[selected]
        labels[grid_rows] = 1

    # Preserve small/thin targets missed by the grid; leave empty rows deferred.
    for row in torch.where(~has_grid)[0].tolist():
        foreground_yx = torch.nonzero(masks[row], as_tuple=False)
        if foreground_yx.shape[0] == 0:
            continue
        selected = torch.randint(
            foreground_yx.shape[0], (num_pt,), device=device, generator=generator,
        )
        points[row] = foreground_yx[selected][:, [1, 0]].to(torch.float32)
        labels[row] = 1
    return points, labels
