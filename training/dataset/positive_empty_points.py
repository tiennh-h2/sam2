"""CPU, post-augmentation sampling of independent positive point prompts."""

import math
import random
from dataclasses import replace

import torch

from training.utils.data_utils import Object, VideoDatapoint


class PositiveEmptyPointSampler:
    """One candidate per visible instance, plus label-1/empty-target candidates.

    The ratio is relative to all visible instances BEFORE final random selection.
    Call after geometric transforms, with ALL instances still present. Only the
    final selection is collated and transferred to the GPU. Uses the data worker's
    seeded Python RNG; no per-call seeding or HxW random float tensors.
    """

    def __init__(self, max_points=10, min_empty_ratio=0.10, max_empty_ratio=0.25):
        if not isinstance(max_points, int) or max_points < 1:
            raise ValueError("max_points must be a positive integer")
        if not 0 <= min_empty_ratio <= max_empty_ratio <= 1:
            raise ValueError("Require 0 <= min_empty_ratio <= max_empty_ratio <= 1")
        self.max_points = max_points
        self.min_empty_ratio = min_empty_ratio
        self.max_empty_ratio = max_empty_ratio

    def __call__(self, datapoint: VideoDatapoint) -> VideoDatapoint:
        if len(datapoint.frames) != 1:
            raise ValueError("PositiveEmptyPointSampler supports single-frame images only")
        frame = datapoint.frames[0]
        if not frame.objects:
            raise ValueError("No visible instances after augmentation")
        union = torch.zeros_like(frame.objects[0].segment, dtype=torch.bool)
        if union.device.type != "cpu" or union.ndim != 2:
            raise ValueError("Point sampling requires 2D CPU masks")

        # Preserve every instance in the union, including those not selected below.
        for obj in frame.objects:
            if obj.segment.shape != union.shape:
                raise ValueError("All instance masks must have the same shape")
            union |= obj.segment.bool()

        width = union.shape[1]
        candidates = []
        for obj in frame.objects:
            pixels = obj.segment.flatten().nonzero().flatten()
            if pixels.numel() == 0:
                continue  # A geometric transform may have removed this instance.
            pixel = int(pixels[random.randrange(pixels.numel())])
            candidates.append((obj, (pixel % width, pixel // width)))
        num_instances = len(candidates)
        if num_instances == 0:
            raise ValueError("No visible instances after augmentation")

        low = math.ceil(self.min_empty_ratio * num_instances)
        high = math.floor(self.max_empty_ratio * num_instances)
        if low <= high and high > 0:
            background_pixels = (~union).flatten().nonzero().flatten()
            # Establish feasibility before the draw, so a scarce background
            # cannot force a count below the requested minimum or bias the draw.
            high = min(high, background_pixels.numel())
            num_empty = random.randint(low, high) if low <= high else 0
            for index in random.sample(range(background_pixels.numel()), num_empty):
                pixel = int(background_pixels[index])
                candidates.append((None, (pixel % width, pixel // width)))

        # A point is a separate decoder sample, not an additional click on an object.
        selected = random.sample(candidates, min(self.max_points, len(candidates)))
        next_empty_id = min(0, min(obj.object_id for obj in frame.objects)) - 1
        objects = []
        for obj, coords in selected:
            if obj is None:
                obj = Object(
                    object_id=next_empty_id,
                    frame_index=frame.objects[0].frame_index,
                    segment=torch.zeros_like(union, dtype=torch.uint8),
                )
                next_empty_id -= 1
            objects.append(replace(obj, point_coords=coords, point_label=1))
        frame.objects = objects
        return datapoint
