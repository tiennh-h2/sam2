"""Run SAM2 point-grid mask generation on overlapping SAHI image slices.

This uses the tile-location patch, so it can no longer use
SAM2AutomaticMaskGenerator: per the patch's README, "the stock automatic mask
generator [is] not wired for this extension." Instead this script drives
TileLocationImagePredictor directly with a point grid, replicating the
filter/NMS steps the automatic generator used to do internally.

Every prediction needs a normalized [x0,y0,x1,y1] tile location. Since these
tiles are cut in-memory from the full image (not loaded from files named
`..._orig_wW_hH__xX_yY.png`), the location is passed explicitly via
`location=` using the slice bbox from get_slice_bboxes() and the full image's
own width/height as the parent ROI.

Example:
    python sam2_sahi_slice_inference.py \
        --slice-size 2048 --overlap-ratio 0.2 \
        --test-dir /path/to/test --out-dir outputs --save-labels

Each output image is a slice with its predicted masks overlaid. The optional
label PNG stores 0 for background and 1..N for the masks within that slice.
Masks in adjacent slices are not merged.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from sahi.slicing import get_slice_bboxes
from sam2.build_sam import build_sam2
from sam2.tile_location_predictor import TileLocationImagePredictor
from sam2.utils.amg import batch_iterator, batched_mask_to_box, build_point_grid, calculate_stability_score
from sam2_object_score_filter import enable_object_score_filter


# Point to a checkpoint/config produced by tile-location training, e.g. via
#   python sam2_tile_location/make_config.py --repo . \
#       --input <your current training yaml> \
#       --output sam2/configs/sam2.1_training/sam2_fbm_tile_location.yaml \
#       --inference-output sam2/configs/sam2.1/sam2_fbm_tile_location.yaml
# The checkpoint path below is illustrative; use the path your trainer prints.
DEFAULT_CHECKPOINT = (
    "/home/tien.nguyen/workspace/project/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_FBM_finetune_sahi_tile_locations.yaml/checkpoints/checkpoint.pt"
)
DEFAULT_TEST_DIR = (
    "/home/tien.nguyen/workspace/project/common/data/fbm/commercial/"
    "specialty_segmentation_dataset_for_sam3_20260916/test/"
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+_with_tiles_pos.yaml")
    parser.add_argument("--test-dir", type=Path, default=Path(DEFAULT_TEST_DIR))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_neg_samples_test_with_tiles/"))
    parser.add_argument("--slice-size", type=int, default=2048)
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--stability-score-offset", type=float, default=1.0)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument(
        "--save-labels", action="store_true",
        help="Also save a per-slice uint16 PNG with one integer ID per mask.",
    )
    args = parser.parse_args()
    if args.slice_size <= 0:
        parser.error("--slice-size must be positive")
    if not 0 <= args.overlap_ratio < 1:
        parser.error("--overlap-ratio must be in [0, 1)")
    if args.points_per_batch <= 0:
        parser.error("--points-per-batch must be positive")
    if args.points_per_side <= 0:
        parser.error("--points-per-side must be positive")
    return args


def generate_tile_masks(
    predictor,
    tile_rgb,
    points_per_side,
    points_per_batch,
    pred_iou_thresh,
    stability_score_thresh,
    stability_score_offset,
    box_nms_thresh,
    mask_threshold=0.0,
):
    """Point-grid mask generation for one tile, mirroring what
    SAM2AutomaticMaskGenerator used to do, but through the tile-location
    predictor (which only supports point prompts one tile at a time)."""
    height, width = tile_rgb.shape[:2]
    unit_points = build_point_grid(points_per_side)
    points = unit_points * np.array([width, height], dtype=np.float32)

    kept_masks, kept_boxes, kept_scores = [], [], []
    for (points_batch,) in batch_iterator(points_per_batch, points):
        coords = points_batch[:, None, :].astype(np.float32)  # [N,1,2]
        labels = np.ones((coords.shape[0], 1), dtype=np.int32)
        with torch.inference_mode():
            masks, iou_preds, _ = predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=True,
                return_logits=True,
            )
        masks_t = torch.from_numpy(masks).reshape(-1, height, width)
        iou_t = torch.from_numpy(iou_preds).reshape(-1)

        keep = iou_t > pred_iou_thresh
        masks_t, iou_t = masks_t[keep], iou_t[keep]
        if masks_t.numel() == 0:
            continue

        stability = calculate_stability_score(masks_t, mask_threshold, stability_score_offset)
        keep = stability > stability_score_thresh
        masks_t, iou_t = masks_t[keep], iou_t[keep]
        if masks_t.numel() == 0:
            continue

        binary = masks_t > mask_threshold
        boxes = batched_mask_to_box(binary).float()
        kept_masks.append(binary)
        kept_boxes.append(boxes)
        kept_scores.append(iou_t)

    if not kept_masks:
        return []

    all_masks = torch.cat(kept_masks)
    all_boxes = torch.cat(kept_boxes)
    all_scores = torch.cat(kept_scores)
    keep_idx = torchvision.ops.nms(all_boxes, all_scores, iou_threshold=box_nms_thresh)

    anns = []
    for i in keep_idx.tolist():
        mask = all_masks[i].numpy()
        anns.append({"segmentation": mask, "area": int(mask.sum())})
    return anns


def render_anns_overlay(image_bgr, anns, rng, borders=True, save_labels=False):
    height, width = image_bgr.shape[:2]
    colors = np.zeros_like(image_bgr)
    covered = np.zeros((height, width), dtype=bool)
    border_mask = np.zeros((height, width), dtype=np.uint8)
    labels = np.zeros((height, width), dtype=np.uint16) if save_labels else None

    # Small masks drawn last remain visible in overlapping areas.
    for index, ann in enumerate(sorted(anns, key=lambda x: x["area"], reverse=True), 1):
        mask = np.asarray(ann["segmentation"], dtype=bool)
        colors[mask] = rng.integers(0, 256, size=3, dtype=np.uint8)
        covered[mask] = True
        if labels is not None:
            if index > np.iinfo(np.uint16).max:
                raise ValueError("More than 65535 masks in one slice; cannot save uint16 labels")
            labels[mask] = index
        if borders:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(border_mask, contours, -1, 255, thickness=1)

    result = image_bgr.copy()
    result[covered] = (
        (image_bgr[covered].astype(np.uint16) + colors[covered].astype(np.uint16)) // 2
    ).astype(np.uint8)
    if borders:
        edge = border_mask != 0
        result[edge] = (
            (image_bgr[edge].astype(np.uint16) * 3
             + np.array([255, 0, 0], dtype=np.uint16) * 2) // 5
        ).astype(np.uint8)
    return result, labels


def save_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not save {path}")


def main():
    args = parse_args()
    if not args.test_dir.is_dir():
        raise NotADirectoryError(args.test_dir)

    sam2 = build_sam2(
        args.model_cfg, args.checkpoint, device="cuda", apply_postprocessing=False
    )
    enable_object_score_filter(sam2, min_object_score=0.5)

    # TileLocationImagePredictor requires a model built with
    # use_tile_location: true (see the patch's example_inference.yaml) and
    # refuses set_image()/set_image_batch() so location can't be silently
    # omitted; use set_tile(..., location=...) per tile instead.
    predictor = TileLocationImagePredictor(sam2)
    rng = np.random.default_rng(3)

    for image_path in sorted(args.test_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        height, width = image.shape[:2]
        boxes = get_slice_bboxes(
            image_height=height,
            image_width=width,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_height_ratio=args.overlap_ratio,
            overlap_width_ratio=args.overlap_ratio,
        )
        # Include the input suffix to keep e.g. plan.jpg and plan.png separate.
        image_output_dir = args.out_dir / image_path.name
        for tile_index, (x1, y1, x2, y2) in enumerate(boxes):
            tile_bgr = image[y1:y2, x1:x2]
            tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)

            # Location is relative to the full (unsliced) image, which is the
            # parent ROI here; x2/y2 from get_slice_bboxes are exclusive,
            # matching what set_tile(location=...) expects.
            location = [x1 / width, y1 / height, x2 / width, y2 / height]
            predictor.set_tile(tile_rgb, location=location)

            anns = generate_tile_masks(
                predictor,
                tile_rgb,
                points_per_side=args.points_per_side,
                points_per_batch=args.points_per_batch,
                pred_iou_thresh=args.pred_iou_thresh,
                stability_score_thresh=args.stability_score_thresh,
                stability_score_offset=args.stability_score_offset,
                box_nms_thresh=args.box_nms_thresh,
            )

            overlay, labels = render_anns_overlay(
                tile_bgr, anns, rng, save_labels=args.save_labels
            )
            tile_name = f"tile_{tile_index:04d}_x{x1}_y{y1}_x{x2}_y{y2}.png"
            (image_output_dir / "overlay").mkdir(parents=True, exist_ok=True)
            save_png(image_output_dir / "overlay" / tile_name, overlay)
            if labels is not None:
                save_png(image_output_dir / "labels" / tile_name, labels)
            print(f"{image_path.name}: tile {tile_index + 1}/{len(boxes)}, {len(anns)} masks")


if __name__ == "__main__":
    main()