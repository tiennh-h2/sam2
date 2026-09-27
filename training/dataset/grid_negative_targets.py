"""Add a background-query training example to a one-frame SAM 2 image sample.

Place this transform LAST, after all image and mask geometry transforms.
It adds an empty *query target*, not a new annotation in the source dataset.
"""

import random

import torch

from training.utils.data_utils import Object


class AddGridNegativeTarget:
    def __init__(self, probability=0.5, max_ratio_of_negative_objs=1):
        if not 0 <= probability <= 1:
            raise ValueError("probability must lie in [0, 1]")
        if max_ratio_of_negative_objs < 0:
            raise ValueError("max_ratio_of_negative_objs must be nonnegative")
        self.probability = probability
        self.max_ratio_of_negative_objs = max_ratio_of_negative_objs

    def __call__(self, datapoint, **kwargs):
        if len(datapoint.frames) != 1:
            raise ValueError("Only one-frame image samples are supported")

        frame = datapoint.frames[0]
        if not frame.objects or random.random() >= self.probability:
            return datapoint

        positives = [obj for obj in frame.objects if obj.segment.bool().any()]
        negatives = [obj for obj in frame.objects if not obj.segment.bool().any()]
        if not positives:
            return datapoint

        occupied = torch.stack(
            [obj.segment.bool() for obj in positives]
        ).any(dim=0)
        if bool(occupied.all()):
            return datapoint

        target_negatives = len(positives) * self.max_ratio_of_negative_objs
        num_to_add = max(0, target_negatives - len(negatives))
        used_ids = {obj.object_id for obj in frame.objects}
        next_id = -1

        for _ in range(num_to_add):
            while next_id in used_ids:
                next_id -= 1
            frame.objects.append(
                Object(
                    object_id=next_id,
                    frame_index=positives[0].frame_index,
                    segment=torch.zeros_like(positives[0].segment),
                )
            )
            used_ids.add(next_id)

        return datapoint