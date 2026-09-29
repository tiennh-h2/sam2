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
from pathlib import Path

import numpy as np
import torch

from training.model.sam2 import SAM2Train
from .grid_point_sampling import sample_grid_points_and_assign_instances


class GridNegativeSAM2Train(SAM2Train):
    def __init__(
        self,
        *args,
        max_ratio_of_negative_objs=1,
        grid_points_per_side=32,
        debug_dir=None, # "/home/tien.nguyen/workspace/project/sam2/.debug",
        debug_every_n_batches=1,
        debug_max_images=1000,
        debug_image_mean=(0.485, 0.456, 0.406),
        debug_image_std=(0.229, 0.224, 0.225),
        **kwargs,
    ):
        if (
            isinstance(grid_points_per_side, bool)
            or not isinstance(grid_points_per_side, Integral)
            or grid_points_per_side < 1
        ):
            raise ValueError("grid_points_per_side must be a positive integer")
        if (
            not math.isfinite(max_ratio_of_negative_objs)
            or max_ratio_of_negative_objs < 0
        ):
            raise ValueError(
                "max_ratio_of_negative_objs must be finite and nonnegative"
            )
        if kwargs.get("prob_to_use_box_input_for_train", 0) != 0:
            raise ValueError(
                "set prob_to_use_box_input_for_train=0 for empty targets"
            )
        if debug_every_n_batches < 1:
            raise ValueError("debug_every_n_batches must be positive")
        if debug_max_images < 0:
            raise ValueError("debug_max_images must be nonnegative")

        super().__init__(*args, **kwargs)
        self.max_ratio_of_negative_objs = max_ratio_of_negative_objs
        self.grid_points_per_side = grid_points_per_side

        self.debug_dir = Path(debug_dir) if debug_dir is not None else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        self.debug_every_n_batches = debug_every_n_batches
        self.debug_max_images = debug_max_images
        self.debug_image_mean = debug_image_mean
        self.debug_image_std = debug_image_std
        self._debug_batch_count = 0
        self._debug_images_saved = 0

    def prepare_prompt_inputs(
        self, backbone_out, input, start_frame_idx=0
    ):
        out = super().prepare_prompt_inputs(
            backbone_out, input, start_frame_idx
        )
        if not self.training:
            return out

        self._debug_batch_count += 1
        save_debug = (
            self.debug_dir is not None
            and self._debug_images_saved < self.debug_max_images
            and self._debug_batch_count % self.debug_every_n_batches == 0
        )

        for t, prompts in out["point_inputs_per_frame"].items():
            masks = input.masks[t]
            points, labels = sample_grid_points_and_assign_instances(
                input.masks[t],
                points_per_side=self.grid_points_per_side,
                image_indices=input.obj_to_frame_idx[t, :, 1],
            )
            if masks.ndim == 4:
                masks = masks[:, 0]

            image_indices = input.flat_obj_to_img_idx[t]
            empty = ~masks.flatten(1).any(dim=1)

            for row in torch.where(empty)[0].tolist():
                same_image = image_indices == image_indices[row]
                occupied = masks[same_image].any(dim=0)

                x, y = points[row, 0]
                px = int(x.item())
                py = int(y.item())

                if occupied[py, px]:
                    raise AssertionError(
                        f"Empty target {row} sampled inside another GT mask "
                        f"at ({x.item():.1f}, {y.item():.1f})"
                    )
            prompts["point_coords"][:, :1] = points
            prompts["point_labels"][:, :1] = labels

            self._limit_sampler_objects(input, out)

            if save_debug:
                print(f"Saving debug image for batch {self._debug_batch_count}, frame {t}")
                self._save_prompt_debug(input, t, prompts)

        return out

    def _limit_sampler_objects(self, input, out):
        if input.num_frames != 1:
            raise ValueError("Object capping requires single-frame images")

        cap = self.max_num_sampler_objects
        if cap is None:
            return
        if cap < 1:
            raise ValueError("max_num_sampler_objects must be positive")

        n = input.masks.shape[1]
        if n <= cap:
            return

        selected = self.rng.choice(n, size=cap, replace=False)
        keep = torch.zeros(n, dtype=torch.bool, device=input.masks.device)
        keep[torch.as_tensor(selected, device=keep.device)] = True

        input.masks = input.masks[:, keep]
        input.obj_to_frame_idx = input.obj_to_frame_idx[:, keep]
        input.metadata.unique_objects_identifier = (
            input.metadata.unique_objects_identifier[:, keep]
        )
        input.metadata.frame_orig_size = (
            input.metadata.frame_orig_size[:, keep]
        )

        for t, masks in out["gt_masks_per_frame"].items():
            out["gt_masks_per_frame"][t] = masks[keep]

        for t, masks in out["mask_inputs_per_frame"].items():
            out["mask_inputs_per_frame"][t] = masks[keep]

        for prompts in out["point_inputs_per_frame"].values():
            prompts["point_coords"] = prompts["point_coords"][keep]
            prompts["point_labels"] = prompts["point_labels"][keep]

    @torch.no_grad()
    def _save_prompt_debug(self, input, t, prompts):
        from PIL import Image, ImageDraw

        masks = input.masks[t]
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        elif masks.ndim != 3:
            raise ValueError("Expected masks of shape [N,H,W] or [N,1,H,W]")

        image_indices = input.flat_obj_to_img_idx[t].detach().cpu()
        masks = masks.detach().cpu().bool()
        coords = prompts["point_coords"].detach().cpu()
        labels = prompts["point_labels"].detach().cpu()

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available()
            and torch.distributed.is_initialized()
            else 0
        )
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        for row in range(len(masks)):
            if self._debug_images_saved >= self.debug_max_images:
                break

            image_idx = int(image_indices[row])
            image_tensor = input.flat_img_batch[image_idx].detach().float().cpu()

            mean = torch.tensor(self.debug_image_mean)[:, None, None]
            std = torch.tensor(self.debug_image_std)[:, None, None]
            image_array = (
                (image_tensor * std + mean)
                .clamp(0, 1)
                .permute(1, 2, 0)
                .numpy()
            )
            image_array = (image_array * 255).round().astype(np.uint8)
            image = Image.fromarray(image_array, mode="RGB")

            width, height = image.size
            if tuple(masks[row].shape) != (height, width):
                raise ValueError("Image and target mask dimensions differ")

            is_empty = not bool(masks[row].any())

            # Overlay only this instance's target mask.
            if not is_empty:
                mask_image = Image.fromarray(
                    masks[row].numpy().astype(np.uint8) * 255
                )
                tinted = Image.blend(
                    image,
                    Image.new("RGB", image.size, (0, 220, 220)),
                    alpha=0.35,
                )
                image.paste(tinted, (0, 0), mask_image)

            draw = ImageDraw.Draw(image)

            # Faint dots show the complete candidate grid.
            side = self.grid_points_per_side
            for gy in range(side):
                y = (gy + 0.5) * height / side
                for gx in range(side):
                    x = (gx + 0.5) * width / side
                    draw.ellipse(
                        (x - 1, y - 1, x + 1, y + 1),
                        fill=(120, 120, 120),
                    )

            point_color = (255, 40, 40) if is_empty else (0, 255, 40)
            for point, label in zip(coords[row], labels[row]):
                if int(label) < 0:
                    continue
                x, y = point.tolist()
                draw.ellipse(
                    (x - 6, y - 6, x + 6, y + 6),
                    fill=point_color,
                    outline=(0, 0, 0),
                    width=2,
                )

            target_type = "empty" if is_empty else "instance"
            draw.text(
                (10, 10),
                f"image={image_idx}  target={row}  {target_type}",
                fill=(255, 255, 0),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )

            filename = (
                f"rank{rank}_batch{self._debug_batch_count:06d}"
                f"_frame{t}_image{input.img_names[image_idx]}"
                f"_target{row:04d}_{target_type}.png"
            )
            image.save(self.debug_dir / filename)
            self._debug_images_saved += 1

    # def forward(self, input):
    #     if self.training:
    #         self._limit_negative_objects(input)
    #     return super().forward(input)

    # def _limit_negative_objects(self, input):
    #     """Filter targets in place so the trainer's loss sees the same objects.

    #     The SAM 2 collator stores [frames, objects, ...] tensors, with one
    #     object ordering shared across frames. This implementation is for the
    #     single-frame image dataset; video targets need track-level sampling.
    #     """
    #     if input.num_frames != 1:
    #         raise ValueError("grid negative object sampling requires one frame per sample")

    #     masks = input.masks[0]  # [objects, height, width]
    #     image_indices = input.obj_to_frame_idx[0, :, 1]
    #     positive = masks.flatten(1).any(dim=1)
    #     keep = torch.zeros_like(positive)

    #     for image_idx in torch.unique(image_indices).tolist():
    #         in_image = image_indices == image_idx
    #         positives = torch.where(in_image & positive)[0]
    #         negatives = torch.where(in_image & ~positive)[0]
    #         keep[positives] = True
    #         limit = math.floor(self.max_ratio_of_negative_objs * len(positives))
    #         if len(negatives) > limit:
    #             chosen = self.rng.choice(len(negatives), size=limit, replace=False)
    #             negatives = negatives[torch.as_tensor(chosen, device=negatives.device)]
    #         keep[negatives] = True

    #     if bool(keep.all()):
    #         return
    #     # Mutate the same batch object the SAM 2 trainer passes to its loss.
    #     # Its metadata has the same [frames, objects, ...] alignment.
    #     input.masks = input.masks[:, keep]
    #     input.obj_to_frame_idx = input.obj_to_frame_idx[:, keep]
    #     input.metadata.unique_objects_identifier = (
    #         input.metadata.unique_objects_identifier[:, keep]
    #     )
    #     input.metadata.frame_orig_size = input.metadata.frame_orig_size[:, keep]

    # def _background_point(self, occupied):
    #     height, width = occupied.shape
    #     side = self.grid_points_per_side
    #     unit = (torch.arange(side, dtype=torch.float32, device=occupied.device) + 0.5) / side
    #     grid = torch.stack([
    #         (unit * width).repeat(side),
    #         (unit * height).repeat_interleave(side),
    #     ], dim=-1)
    #     # Preserve fractional prompt coordinates; floor only for mask lookup.
    #     xs = grid[:, 0].long().clamp(0, width - 1)
    #     ys = grid[:, 1].long().clamp(0, height - 1)
    #     candidates = grid[~occupied[ys, xs]]
    #     if candidates.numel() == 0:
    #         candidates = torch.nonzero(~occupied, as_tuple=False)[:, [1, 0]]
    #     if candidates.numel() == 0:
    #         raise ValueError("empty query target has no available background pixel")
    #     idx = int(self.rng.integers(len(candidates)))
    #     return candidates[idx].tolist()
