"""Inspect SAM2 grid predictions before automatic-mask-generator filtering.

Use the exact training tile image for an overfit check, e.g.:
  python sam2_grid_probe.py --checkpoint /path/to/checkpoint.pt \
      --image /path/to/exact_training_tile.png --out-dir probe_out \
      --point 736 851 --point 1824 851 --probe-only

Point coordinates are LOCAL to the supplied image/tile. Set --pred-iou-thresh
and --stability-score-thresh to 0 to inspect permissively filtered full-grid
predictions as well. With --probe-only, skip the full 32x32 grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sahi.slicing import get_slice_bboxes
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--model-cfg', default='configs/sam2.1/sam2.1_hiera_b+.yaml')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path, help='Exact training tile (recommended)')
    source.add_argument('--test-dir', type=Path, help='Full images to slice with SAHI')
    p.add_argument('--out-dir', type=Path, default=Path('sam2_grid_probe'))
    p.add_argument('--point', nargs=2, type=int, action='append', metavar=('X', 'Y'),
                   help='Grid point LOCAL to the image or selected SAHI tile; repeatable')
    p.add_argument('--probe-only', action='store_true',
                   help='Skip full-grid generation and save only raw point predictions')
    p.add_argument('--probe-tile-index', type=int, default=0,
                   help='SAHI tile index to probe when --test-dir is used')
    p.add_argument('--points-per-side', type=int, default=32)
    p.add_argument('--points-per-batch', type=int, default=64)
    p.add_argument('--pred-iou-thresh', type=float, default=0.8)
    p.add_argument('--stability-score-thresh', type=float, default=0.95)
    p.add_argument('--slice-size', type=int, default=2048)
    p.add_argument('--overlap-ratio', type=float, default=0.2)
    args = p.parse_args()
    if args.probe_only and not args.point:
        p.error('--probe-only requires at least one --point X Y')
    if args.image is not None and not args.image.is_file():
        p.error(f'Image does not exist: {args.image}')
    if args.test_dir is not None and not args.test_dir.is_dir():
        p.error(f'Directory does not exist: {args.test_dir}')
    if not args.checkpoint.is_file():
        p.error(f'Checkpoint does not exist: {args.checkpoint}')
    if args.points_per_side <= 0 or args.points_per_batch <= 0 or args.slice_size <= 0:
        p.error('Grid, batch and slice sizes must be positive')
    if not (0 <= args.overlap_ratio < 1):
        p.error('--overlap-ratio must be in [0, 1)')
    if not (0 <= args.pred_iou_thresh <= 1 and 0 <= args.stability_score_thresh <= 1):
        p.error('Score thresholds must be in [0, 1]')
    if args.probe_tile_index < 0:
        p.error('--probe-tile-index must be nonnegative')
    return args


def write_png(path: Path, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), data):
        raise OSError(f'Failed to save {path}')


@torch.inference_mode()
def probe(model, tile_rgb: np.ndarray, points: list[list[int]], output: Path):
    """Save all candidate masks and decoder object scores without AMG filters."""
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(tile_rgb)
    tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    h, w = tile_rgb.shape[:2]
    records = []
    captured = []

    def capture_score(_module, _inputs, decoder_output):
        # Image predictor returns masks/IoU but normally drops object logits.
        captured.append(decoder_output[3].detach().float().cpu().flatten().tolist())

    handle = model.sam_mask_decoder.register_forward_hook(capture_score)
    try:
        for x, y in points:
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError(f'Probe ({x},{y}) is outside tile of size {w}x{h}')
            captured.clear()
            logits, ious, _ = predictor.predict(
                point_coords=np.array([[x, y]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
                return_logits=True,
            )
            if len(captured) != 1 or len(captured[0]) != 1:
                raise RuntimeError(f'Expected one object score per point; got {captured}')
            object_logit = captured[0][0]
            object_prob = float(torch.sigmoid(torch.tensor(object_logit)))
            row = {'point_xy': [x, y], 'object_logit': object_logit,
                   'object_probability': object_prob, 'candidates': []}
            for candidate, (logit, iou) in enumerate(zip(logits, ious)):
                mask = logit > 0
                pixels = int(mask.sum())
                viz = tile_bgr.copy()
                if pixels:
                    green = np.zeros_like(viz)
                    green[:] = (0, 200, 0)
                    blended = cv2.addWeighted(viz, 0.55, green, 0.45, 0)
                    viz[mask] = blended[mask]
                cv2.circle(viz, (x, y), 7, (0, 0, 255), -1)
                name = f'point_x{x}_y{y}_candidate{candidate}'
                write_png(output / f'{name}_overlay.png', viz)
                write_png(output / f'{name}_mask.png', mask.astype(np.uint8) * 255)
                row['candidates'].append({
                    'candidate': candidate, 'predicted_iou': float(iou),
                    'mask_area_pixels': pixels,
                    'contains_prompt': bool(mask[y, x]),
                    'overlay': f'{name}_overlay.png', 'mask': f'{name}_mask.png',
                })
            records.append(row)
            print(json.dumps(row))
    finally:
        handle.remove()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'raw_point_predictions.json').write_text(json.dumps(records, indent=2))


@torch.inference_mode()
def generate_overlay(generator, tile_rgb: np.ndarray, output: Path):
    anns = generator.generate(tile_rgb)
    bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    covered = np.zeros(bgr.shape[:2], dtype=bool)
    colored = np.zeros_like(bgr)
    rng = np.random.default_rng(3)
    for ann in sorted(anns, key=lambda a: a['area'], reverse=True):
        mask = np.asarray(ann['segmentation'], dtype=bool)
        colored[mask] = rng.integers(0, 256, size=3, dtype=np.uint8)
        covered[mask] = True
    overlay = bgr.copy()
    overlay[covered] = ((bgr[covered].astype(np.uint16)
                         + colored[covered].astype(np.uint16)) // 2).astype(np.uint8)
    write_png(output / 'filtered_grid_overlay.png', overlay)
    print(f'Filtered full-grid masks: {len(anns)} at {output}')


def main():
    args = parse_args()
    model = build_sam2(args.model_cfg, str(args.checkpoint),
                       device='cuda', apply_postprocessing=False)
    # Do not install object-score filter: the probe must see raw decoder output.
    generator = None
    if not args.probe_only:
        generator = SAM2AutomaticMaskGenerator(
            model, points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            output_mode='binary_mask', use_m2m=False,
        )
    images = [args.image] if args.image is not None else [
        p for p in sorted(args.test_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUFFIXES
    ]
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f'Skipping unreadable image: {image_path}')
            continue
        h, w = image.shape[:2]
        boxes = [(0, 0, w, h)] if args.image is not None else get_slice_bboxes(
            image_height=h, image_width=w,
            slice_height=args.slice_size, slice_width=args.slice_size,
            overlap_height_ratio=args.overlap_ratio,
            overlap_width_ratio=args.overlap_ratio,
        )
        if args.point and args.probe_tile_index >= len(boxes):
            raise ValueError(f'Tile {args.probe_tile_index} not found in {image_path}')
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if args.probe_only and i != args.probe_tile_index:
                continue
            tile_rgb = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            out = args.out_dir / image_path.name / f'tile_{i:04d}_x{x1}_y{y1}'
            if args.point and i == args.probe_tile_index:
                probe(model, tile_rgb, args.point, out)
            if generator is not None:
                generate_overlay(generator, tile_rgb, out)
            if args.probe_only:
                break


if __name__ == '__main__':
    main()
