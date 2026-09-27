"""SAM 2 image training with grid-preferred positives and capped empty targets.

Place grid_point_sampling.py beside this file in training/model/.
Initial positive prompts preserve the existing target count and ordering.
Foreground and background use the same cell-centered grid, with spacing
width/grid_points_per_side and height/grid_points_per_side. If that grid
misses the eligible region, sampling falls back to a random eligible pixel.
Keep the base get_next_point interface returning (points, labels); use uniform
or center for corrections, not grid-first target reassignment.

If sample_all_correct_region=True and correction clicks are enabled, apply
sam2_empty_target_sampler.patch to sam2_utils.py as well.
"""

import math
from numbers import Integral

import torch

from training.model.sam2 import SAM2Train
from .grid_point_sampling import sample_grid_points_and_assign_instances


class GridNegativeSAM2Train(SAM2Train):
    def __init__(
        self,
        *args,
        max_ratio_of_negative_objs=1,
        grid_points_per_side=32,
        **kwargs,
    ):
        if (
            isinstance(grid_points_per_side, bool)
            or not isinstance(grid_points_per_side, Integral)
            or grid_points_per_side < 1
        ):
            raise ValueError("grid_points_per_side must be a positive integer")
        if not math.isfinite(max_ratio_of_negative_objs) or max_ratio_of_negative_objs < 0:
            raise ValueError("max_ratio_of_negative_objs must be finite and nonnegative")
        if kwargs.get("prob_to_use_box_input_for_train", 0) != 0:
            raise ValueError("set prob_to_use_box_input_for_train=0 for empty targets")
        super().__init__(*args, **kwargs)
        self.max_ratio_of_negative_objs = max_ratio_of_negative_objs
        self.grid_points_per_side = grid_points_per_side

    def forward(self, input):
        if self.training:
            self._limit_negative_objects(input)
        return super().forward(input)

    def _limit_negative_objects(self, input):
        """Filter targets in place so the trainer's loss sees the same objects.

        The SAM 2 collator stores [frames, objects, ...] tensors, with one
        object ordering shared across frames. This implementation is for the
        single-frame image dataset; video targets need track-level sampling.
        """
        if input.num_frames != 1:
            raise ValueError("grid negative object sampling requires one frame per sample")

        masks = input.masks[0]  # [objects, height, width]
        image_indices = input.obj_to_frame_idx[0, :, 1]
        positive = masks.flatten(1).any(dim=1)
        keep = torch.zeros_like(positive)

        for image_idx in torch.unique(image_indices).tolist():
            in_image = image_indices == image_idx
            positives = torch.where(in_image & positive)[0]
            negatives = torch.where(in_image & ~positive)[0]
            keep[positives] = True
            limit = math.floor(self.max_ratio_of_negative_objs * len(positives))
            if len(negatives) > limit:
                chosen = self.rng.choice(len(negatives), size=limit, replace=False)
                negatives = negatives[torch.as_tensor(chosen, device=negatives.device)]
            keep[negatives] = True

        if bool(keep.all()):
            return
        # Mutate the same batch object the SAM 2 trainer passes to its loss.
        # Its metadata has the same [frames, objects, ...] alignment.
        input.masks = input.masks[:, keep]
        input.obj_to_frame_idx = input.obj_to_frame_idx[:, keep]
        input.metadata.unique_objects_identifier = (
            input.metadata.unique_objects_identifier[:, keep]
        )
        input.metadata.frame_orig_size = input.metadata.frame_orig_size[:, keep]

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        out = super().prepare_prompt_inputs(backbone_out, input, start_frame_idx)
        if not self.training:
            return out

        for t, prompts in out["point_inputs_per_frame"].items():
            masks = input.masks[t]
            empty = ~masks.flatten(1).any(dim=1)
            # Sample each fixed target independently, even across images.
            # This changes only the initial foreground click, not targets
            # or object-to-image mappings. Background rows are set below.
            points, labels = sample_grid_points_and_assign_instances(
                masks[~empty],
                points_per_side=self.grid_points_per_side,
            )
            prompts["point_coords"][~empty, :1] = points
            prompts["point_labels"][~empty, :1] = labels
            image_indices = input.obj_to_frame_idx[t, :, 1]
            for obj_idx in torch.where(empty)[0].tolist():
                occupied = masks[image_indices == image_indices[obj_idx]].any(dim=0)
                x, y = self._background_point(occupied)
                prompts["point_coords"][obj_idx, 0] = torch.tensor(
                    [x, y],
                    dtype=prompts["point_coords"].dtype,
                    device=prompts["point_coords"].device,
                )
                # Positive query prompt on background, with an empty target.
                prompts["point_labels"][obj_idx, 0] = 1
        return out

    def _background_point(self, occupied):
        height, width = occupied.shape
        side = self.grid_points_per_side
        unit = (torch.arange(side, dtype=torch.float32, device=occupied.device) + 0.5) / side
        grid = torch.stack([
            (unit * width).repeat(side),
            (unit * height).repeat_interleave(side),
        ], dim=-1)
        # Preserve fractional prompt coordinates; floor only for mask lookup.
        xs = grid[:, 0].long().clamp(0, width - 1)
        ys = grid[:, 1].long().clamp(0, height - 1)
        candidates = grid[~occupied[ys, xs]]
        if candidates.numel() == 0:
            candidates = torch.nonzero(~occupied, as_tuple=False)[:, [1, 0]]
        if candidates.numel() == 0:
            raise ValueError("empty query target has no available background pixel")
        idx = int(self.rng.integers(len(candidates)))
        return candidates[idx].tolist()
