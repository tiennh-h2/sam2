"""Use SAM 2's trained object-presence head in automatic mask generation.

Install AFTER build_sam2 loads the checkpoint, BEFORE constructing
SAM2AutomaticMaskGenerator. Assumes the generator uses its normal
predicted-IoU filtering (pred_iou_thresh > 0) and use_m2m=False.
"""

import torch
from torch import nn


class ObjectScoreForwardFilter:
    def __init__(self, min_object_score: float = 0.5):
        if not 0.0 < min_object_score < 1.0:
            raise ValueError("min_object_score must be strictly between 0 and 1")
        self.min_object_score = min_object_score

    def __call__(self, _decoder, _inputs, output):
        masks, pred_ious, tokens, object_logits = output
        if object_logits is None:
            raise RuntimeError("SAM 2 checkpoint has no object-presence score")

        keep = object_logits.sigmoid().reshape(-1, 1) >= self.min_object_score
        if keep.shape[0] != masks.shape[0]:
            raise RuntimeError("object score batch size does not match mask batch size")
        # The stock image predictor discards object_logits. Encode rejection in
        # both the mask logits and IoU so the automatic generator drops it.
        masks = torch.where(keep[:, :, None, None], masks, -32.0)
        pred_ious = torch.where(keep, pred_ious, -1.0)
        return masks, pred_ious, tokens, object_logits


def enable_object_score_filter(model: nn.Module, min_object_score: float = 0.5):
    if not getattr(model, "pred_obj_scores", False):
        raise ValueError("model.pred_obj_scores is disabled in the inference config")
    if getattr(model, "_object_score_filter_handle", None) is not None:
        raise ValueError("object score filter is already installed")
    # Preserve the decoder module itself: SAM2Base.forward_image directly uses
    # its conv_s0 and conv_s1 attributes before any prompt is decoded.
    model._object_score_filter_handle = model.sam_mask_decoder.register_forward_hook(
        ObjectScoreForwardFilter(min_object_score)
    )
    return model
