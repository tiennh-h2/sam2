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
    image_indices: torch.Tensor,
    num_pt: int = 1,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if gt_masks.dtype != torch.bool:
        raise TypeError("gt_masks must have dtype torch.bool")

    if gt_masks.ndim == 4 and gt_masks.shape[1] == 1:
        masks = gt_masks[:, 0]
    elif gt_masks.ndim == 3:
        masks = gt_masks
    else:
        raise ValueError("gt_masks must have shape [N,1,H,W] or [N,H,W]")

    if (
        isinstance(points_per_side, bool)
        or not isinstance(points_per_side, Integral)
        or points_per_side < 1
    ):
        raise ValueError("points_per_side must be a positive integer")
    if (
        isinstance(num_pt, bool)
        or not isinstance(num_pt, Integral)
        or num_pt < 0
    ):
        raise ValueError("num_pt must be a nonnegative integer")

    n, height, width = masks.shape
    if height < 1 or width < 1:
        raise ValueError("Mask height and width must be positive")
    if image_indices.shape != (n,):
        raise ValueError("image_indices must have shape [N]")

    device = masks.device
    image_indices = image_indices.to(device)
    points = torch.zeros((n, num_pt, 2), dtype=torch.float32, device=device)
    labels = torch.full((n, num_pt), -1, dtype=torch.int32, device=device)
    if n == 0 or num_pt == 0:
        return points, labels

    side = int(points_per_side)
    unit = (
        torch.arange(side, dtype=torch.float32, device=device) + 0.5
    ) / side
    grid = torch.stack(
        [
            (unit * width).repeat(side),
            (unit * height).repeat_interleave(side),
        ],
        dim=-1,
    )

    xs = grid[:, 0].long().clamp(0, width - 1)
    ys = grid[:, 1].long().clamp(0, height - 1)
    inside = masks[:, ys, xs]  # [N, side**2]
    empty = ~masks.flatten(1).any(dim=1)

    # Nonempty rows: sample grid points inside their own mask.
    positive_rows = torch.where(~empty & inside.any(dim=1))[0]
    if positive_rows.numel():
        selected = torch.multinomial(
            inside[positive_rows].float(),
            num_pt,
            replacement=True,
            generator=generator,
        )
        points[positive_rows] = grid[selected]
        labels[positive_rows] = 1

    # Preserve the existing fallback for nonempty masks missed by the grid.
    for row in torch.where(~empty & ~inside.any(dim=1))[0].tolist():
        foreground_yx = torch.nonzero(masks[row], as_tuple=False)
        selected = torch.randint(
            len(foreground_yx),
            (num_pt,),
            device=device,
            generator=generator,
        )
        points[row] = foreground_yx[selected][:, [1, 0]].float()
        labels[row] = 1

    # Empty rows: sample only grid points outside all masks in their image.
    for image_idx in torch.unique(image_indices[empty]):
        in_image = image_indices == image_idx
        empty_rows = torch.where(in_image & empty)[0]
        occupied = masks[in_image].any(dim=0)
        available = torch.where(~occupied[ys, xs])[0]

        if available.numel() == 0:
            raise ValueError(
                f"No background grid point for image {image_idx.item()}"
            )

        selected = torch.randint(
            len(available),
            (len(empty_rows), num_pt),
            device=device,
            generator=generator,
        )
        points[empty_rows] = grid[available[selected]]
        labels[empty_rows] = 1

    return points, labels
