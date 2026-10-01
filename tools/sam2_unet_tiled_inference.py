#!/usr/bin/env python3
"""SAM2-UNet gated grid inference with tile and whole-image test-set evaluation.

Evaluation (optional): --eval-annotations TEST_COCO.json OR --eval-mask-dir TEST_MASKS
Re-score existing final labels/scores without loading models: add --eval-only.
--filename-contains RCP restricts to RCP images; default processes ALL images.

mDICE/mIOU: mean per-image foreground Dice/IoU after full-image stitching.
Both GT and prediction empty scores 1. Foreground/background are not class-averaged.
AP/AP50/AP75/AP90 and AR: COCO instance masks on saved painted labels and scores.
P0.5/R0.5/F1_0.5, P0.75/R0.75/F1_0.75, P0.9/R0.9/F1_0.9 use mask IoU thresholds.
F1 is an alias for F1_0.5. P/R/F1 counts are pooled across each evaluation set.
--eval-score-threshold affects P/R/F1 only (default 0 = all saved predictions).
AP/AR are null for an item/set with no non-crowd GT. Both empty P/R/F1 = 1.
Tile GT is clipped to the tile; nonempty fragments remain instances. Tile metrics
count overlapping views separately. Image metrics count each stitched image once.
Tile AP evaluates visible saved label instances, not overlapping raw AMG proposals.
High inference filtering thresholds truncate the AP curve; --eval-only cannot
recover discarded masks. For a broader curve rerun with --pred-iou-threshold 0.

Outputs: evaluation/images/{metrics.json,per_image.csv,per_image.json,worst_first.csv}
         evaluation/tiles/{metrics.json,per_tile.csv,per_tile.json,worst_first.csv}
Each row includes Dice/IoU, AP/AR, P/R/F1 at all thresholds, and TP/FP/FN counts.
Tile rows also contain source-image names and x0/y0/x1/y1. Worst-first order is
ascending F1_0.5, then IoU. Both levels export COCO GT/predictions and run settings.
Ground truth is used only for evaluation; the original UNet-gated inference is retained.
Requires pycocotools for evaluation. See --help for all options.
"""
import argparse
import copy
import contextlib
import io
import colorsys
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


@dataclass
class TilePrediction:
    mask: np.ndarray  # binary mask in tile coordinates
    box: tuple[int, int, int, int]  # tile's (x0, y0, x1, y1) in image coordinates
    score: float


def _starts(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, step))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def iter_tile_boxes(height: int, width: int, tile_size: int = 2048,
                    overlap: float = 0.2):
    if height < 1 or width < 1 or tile_size < 1 or not 0 <= overlap < 1:
        raise ValueError("height, width and tile_size must be positive; overlap in [0, 1)")
    step = max(1, round(tile_size * (1 - overlap)))
    for y0 in _starts(height, tile_size, step):
        for x0 in _starts(width, tile_size, step):
            yield (x0, y0, min(x0 + tile_size, width), min(y0 + tile_size, height))


def _tile_overlap_iou(a: TilePrediction, b: TilePrediction) -> tuple[float, int]:
    ax0, ay0, ax1, ay1 = a.box
    bx0, by0, bx1, by1 = b.box
    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if x0 >= x1 or y0 >= y1 or a.box == b.box:
        return 0.0, 0
    ma = a.mask[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0]
    mb = b.mask[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
    intersection = np.count_nonzero(ma & mb)
    union = np.count_nonzero(ma | mb)
    return intersection / union if union else 0.0, intersection


def merge_tile_predictions(predictions: list[TilePrediction], image_hw: tuple[int, int],
                           overlap_iou: float = 0.6, min_overlap_pixels: int = 64):
    """Merge predictions agreeing in tile overlap, then paint higher scores first."""
    if not 0 <= overlap_iou <= 1 or min_overlap_pixels < 1:
        raise ValueError("overlap_iou must be in [0, 1]; min_overlap_pixels positive")
    height, width = image_hw
    parent = list(range(len(predictions)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(predictions):
        x0, y0, x1, y1 = a.box
        if a.mask.shape != (y1 - y0, x1 - x0) or not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
            raise ValueError("Tile mask shape or box is inconsistent with the image")
        for j in range(i):
            iou, intersection = _tile_overlap_iou(a, predictions[j])
            if intersection >= min_overlap_pixels and iou >= overlap_iou:
                parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(predictions)):
        groups.setdefault(root(i), []).append(i)
    ranked = sorted(groups.values(), key=lambda ids: max(predictions[i].score for i in ids), reverse=True)
    if len(ranked) > 65535:
        raise ValueError("More than 65535 instances; use a larger label format")
    labels = np.zeros((height, width), dtype=np.uint16)
    scores = []
    for ids in ranked:
        instance_id = len(scores) + 1
        painted = False
        for i in ids:
            prediction = predictions[i]
            x0, y0, x1, y1 = prediction.box
            window = labels[y0:y1, x0:x1]
            available = prediction.mask & (window == 0)
            painted |= bool(available.any())
            window[available] = instance_id
        if painted:
            scores.append(max(predictions[i].score for i in ids))
    return labels, scores


def make_overlay(image: np.ndarray, labels: np.ndarray, alpha: float = 0.45):
    palette = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
    for instance_id in range(1, len(palette)):
        r, g, b = colorsys.hsv_to_rgb((instance_id * 0.61803398875) % 1, 0.75, 1)
        palette[instance_id] = np.array([r, g, b]) * 255
    selected = labels != 0
    result = image.copy()
    result[selected] = (
        (1 - alpha) * result[selected].astype(np.float32)
        + alpha * palette[labels[selected]].astype(np.float32)
    ).astype(np.uint8)
    return result


def make_grid(points_per_side):
    """Cell-center grid in normalized (x, y) coordinates, as used by AMG."""
    if points_per_side < 1:
        raise ValueError('points_per_side must be positive')
    axis = (np.arange(points_per_side, dtype=np.float32) + 0.5) / points_per_side
    x, y = np.meshgrid(axis, axis)
    return np.stack((x.ravel(), y.ravel()), axis=1)


def filter_grid(grid, probability, threshold):
    """Keep grid points whose original-image pixel has foreground probability >= threshold."""
    grid = np.asarray(grid, dtype=np.float32)
    probability = np.asarray(probability)
    if probability.ndim != 2 or not probability.size:
        raise ValueError('probability must be a nonempty HxW array')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('probability must contain finite values in [0, 1]')
    if grid.ndim != 2 or grid.shape[1] != 2 or not np.isfinite(grid).all():
        raise ValueError('grid must be a finite Nx2 array')
    if np.any((grid < 0) | (grid > 1)) or not 0 <= threshold <= 1:
        raise ValueError('grid coordinates and threshold must be in [0, 1]')
    h, w = probability.shape
    x = np.clip((grid[:, 0] * w).astype(np.int64), 0, w - 1)
    y = np.clip((grid[:, 1] * h).astype(np.int64), 0, h - 1)
    keep = probability[y, x] >= threshold
    return grid[keep], keep


def unet_worker(manifest_path):
    """Run only the bundled SAM2-UNet package in this process."""
    manifest = json.loads(Path(manifest_path).read_text())
    repo = Path(manifest['unet_repo'])
    sys.path.insert(0, str(repo))
    # Imports happen after choosing the repository, before any official SAM2 imports.
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from sam2 import build_sam as bundled_build
    # SAM2UNet() calls build_sam2 with a hardcoded default CUDA device.
    # Override that default for CPU or an explicitly selected GPU.
    original_build = bundled_build.build_sam2
    def build_on_device(*args, **kwargs):
        kwargs['device'] = manifest['device']
        return original_build(*args, **kwargs)
    bundled_build.build_sam2 = build_on_device
    from SAM2UNet import SAM2UNet
    model = SAM2UNet()
    checkpoint = torch.load(manifest['unet_checkpoint'], map_location='cpu', weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a SAM2-UNet state_dict checkpoint')
    # Support checkpoints saved with DDP/DataParallel, without silently ignoring missing weights.
    checkpoint = {k.removeprefix('module.'): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)
    model.to(manifest['device']).eval()
    transform = transforms.Compose([
        transforms.Resize((manifest['unet_size'], manifest['unet_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.inference_mode():
        for index, item in enumerate(manifest['items'], 1):
            with Image.open(item['image']) as source:
                rgb = source.convert('RGB')
            tensor = transform(rgb).unsqueeze(0).to(manifest['device'])
            # First output is the main mask; remaining outputs are auxiliary supervision heads.
            logits, _, _ = model(tensor)
            logits = F.interpolate(logits.float(), size=(rgb.height, rgb.width),
                                   mode='bilinear', align_corners=False)
            probability = logits.sigmoid()[0, 0].cpu().numpy()
            # Deliberately do NOT min-max normalize each image: an empty image
            # must not acquire foreground just because its highest logit is rescaled to 1.
            np.save(item['probability'], probability.astype(np.float32))
            print(f'UNet {index}/{len(manifest["items"])}: {item["image"]}', flush=True)


def save_gate_debug(rgb, probability, grid, keep, out, threshold, save_probability=True):
    out.mkdir(parents=True, exist_ok=True)
    if save_probability:
        np.save(out / 'unet_probability.npy', probability)
    Image.fromarray((probability * 255).round().astype(np.uint8)).save(out / 'unet_probability.png')
    Image.fromarray(((probability >= threshold) * 255).astype(np.uint8)).save(out / 'foreground.png')
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    h, w = probability.shape
    pixels = grid * np.array([w, h], dtype=np.float32)
    radius = max(2, min(h, w) // 256)
    for (x, y), selected in zip(pixels, keep):
        color = (0, 220, 0) if selected else (240, 60, 60)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    canvas.save(out / 'grid_points.png')
    (out / 'points.json').write_text(json.dumps({
        'threshold': threshold, 'total_points': len(grid), 'kept_points': int(keep.sum()),
        'kept_normalized_xy': grid[keep].tolist(), 'kept_pixel_xy': pixels[keep].tolist(),
    }, indent=2))


def cached_probability_valid(item):
    """An existing folder and readable, matching probability map form a UNet cache."""
    if not Path(item['folder']).is_dir() or not Path(item['probability']).is_file():
        return False
    try:
        probability = np.load(item['probability'], mmap_mode='r', allow_pickle=False)
        x0, y0, x1, y1 = item['box']
        return (probability.shape == (y1-y0, x1-x0)
                and np.issubdtype(probability.dtype, np.floating)
                and bool(np.isfinite(probability).all())
                and bool(((probability >= 0) & (probability <= 1)).all()))
    except (OSError, ValueError, EOFError, TypeError):
        return False


def select_unet_items(items, skip_existing):
    """Skip cached UNet work; all tiles still participate in SAM2 and merging."""
    return [item for item in items if not (skip_existing and cached_probability_valid(item))]


def save_tile_instances(tile, annotations, folder, box, dump_masks=False):
    """Always save one overlay per instance, preserving overlapping raw masks."""
    mask_dir = folder / 'instances'
    mask_dir.mkdir(parents=True, exist_ok=True)
    # Remove only files this function owns, so a rerun with fewer masks cannot
    # leave obsolete instance previews behind.
    for path in mask_dir.iterdir():
        if path.is_file() and re.fullmatch(r'\d{5}(?:_overlay)?\.png', path.name):
            path.unlink()
    labels = np.zeros(tile.shape[:2], dtype=np.uint16)
    predictions, records = [], []
    for ann in sorted(annotations, key=lambda ann: ann['predicted_iou'], reverse=True):
        mask = np.asarray(ann['segmentation'], dtype=bool)
        if mask.shape != labels.shape:
            raise ValueError(f'Prediction shape does not match tile: {folder}, {box}')
        if not mask.any():
            continue
        instance_id = len(records) + 1
        if instance_id > 65535:
            raise ValueError('More than 65535 instances in a tile')
        labels[mask & (labels == 0)] = instance_id
        predictions.append(TilePrediction(mask, box, float(ann['predicted_iou'])))
        overlay_file = f'{instance_id:05d}_overlay.png'
        Image.fromarray(make_overlay(tile, mask.astype(np.uint16))).save(mask_dir / overlay_file)
        record = {'instance_id': instance_id, 'predicted_iou': float(ann['predicted_iou']),
                  'area': int(mask.sum()), 'point_coords': ann['point_coords'],
                  'overlay_file': f'instances/{overlay_file}'}
        if dump_masks:
            mask_file = f'{instance_id:05d}.png'
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / mask_file)
            record['mask_file'] = f'instances/{mask_file}'
        records.append(record)
    return labels, predictions, records


def prepare_tiles(paths, image_root, output_root, tile_size, overlap, skip_existing=False):
    """Dump lossless tiles for the isolated UNet stage and record their offsets."""
    items = []
    for path in paths:
        with Image.open(path) as source:
            image = np.asarray(source.convert('RGB'))
        h, w = image.shape[:2]
        boxes = list(iter_tile_boxes(h, w, tile_size, overlap))
        tile_root = output_root / 'tiles' / path.relative_to(image_root)
        for index, box in enumerate(boxes, 1):
            x0, y0, x1, y1 = box
            name = f'tile_{index:04d}_x{x0}_y{y0}_w{x1-x0}_h{y1-y0}'
            folder = tile_root / name
            folder.mkdir(parents=True, exist_ok=True)
            tile_path = folder / 'tile.png'
            if not (skip_existing and tile_path.is_file()):
                Image.fromarray(image[y0:y1, x0:x1]).save(tile_path)
            items.append({
                'source': str(path), 'image': str(tile_path), 'box': list(box),
                'probability': str(folder / 'unet_probability.npy'),
                'folder': str(folder), 'overlay': str(tile_root / f'{name}_overlay.png'),
                'index': index, 'tile_count': len(boxes),
            })
    return items


def foreground_metrics(predicted, target, valid=None):
    """Foreground-only Dice/IoU over one complete image, not per instance/tile."""
    predicted, target = np.asarray(predicted, bool), np.asarray(target, bool)
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError('Prediction and GT must have the same HxW shape')
    if valid is not None:
        if not valid.any():
            return dict(DICE=None, IOU=None, intersection=0, pred_pixels=0, gt_pixels=0, union=0)
        predicted, target = predicted & valid, target & valid
    intersection = int(np.count_nonzero(predicted & target))
    p, g = int(predicted.sum()), int(target.sum())
    union = p + g - intersection
    return dict(DICE=2 * intersection / (p + g) if p + g else 1.0,
                IOU=intersection / union if union else 1.0,
                intersection=intersection, pred_pixels=p, gt_pixels=g, union=union)


def discover_images(args):
    """COCO images are the test manifest when supplied; otherwise scan image_dir."""
    if args.eval_annotations:
        data = json.loads(args.eval_annotations.read_text())
        root = args.img_dir.resolve()
        paths = []
        for im in data['images']:
            p = (root / im['file_name']).resolve()
            if not p.is_relative_to(root):
                raise ValueError(f'COCO file_name must be relative to --img-dir: {im["file_name"]}')
            paths.append(p)
    else:
        paths = [p for p in args.img_dir.rglob('*') if p.is_file() and
                 p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}]
    paths = sorted(p for p in paths if not args.filename_contains or args.filename_contains in p.name)
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate image paths in evaluation manifest')
    if not paths:
        raise ValueError(f'No selected images found in {args.img_dir}')
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    print(f'Selected {len(paths)} images; filename filter: {args.filename_contains or "ALL"}', flush=True)
    return paths


def instance_metrics(evaluator, score_threshold):
    precision = evaluator.eval['precision'][:, :, :, 0, -1]
    recall = evaluator.eval['recall'][:, :, 0, -1]
    def mean_defined(a):
        a = np.asarray(a)
        a = a[a >= 0]
        return float(a.mean()) if a.size else None
    out = dict(AP=mean_defined(precision), AP50=mean_defined(precision[0]),
               AP75=mean_defined(precision[5]), AP90=mean_defined(precision[8]),
               AR=mean_defined(recall))
    for threshold in (0.5, 0.75, 0.9):
        index = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, threshold))[0])
        tp = fp = fn = 0
        for r in evaluator.evalImgs:
            if r is None or list(r['aRng']) != list(evaluator.params.areaRng[0]):
                continue
            active = (np.asarray(r['dtScores']) >= score_threshold) & ~r['dtIgnore'][index].astype(bool)
            matched = r['dtMatches'][index] > 0
            image_tp = int((active & matched).sum())
            tp += image_tp
            fp += int((active & ~matched).sum())
            fn += int((~r['gtIgnore'].astype(bool)).sum()) - image_tp
        suffix = str(threshold)
        out['TP' + suffix], out['FP' + suffix], out['FN' + suffix] = tp, fp, fn
        out['P' + suffix] = tp / (tp + fp) if tp + fp else float(fn == 0)
        out['R' + suffix] = tp / (tp + fn) if tp + fn else 1.0
        out['F1_' + suffix] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    out['F1'] = out['F1_0.5']
    return out


def crop_ground_truth(annotations, boxes, mask_utils):
    """Decode each full GT mask once, then retain nonempty clipped tile instances."""
    result = [[] for _ in boxes]
    for rle, iscrowd in annotations:
        mask = mask_utils.decode(rle)
        h, w = mask.shape
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError('Tile box outside ground-truth image bounds')
            crop = mask[y0:y1, x0:x1]
            if crop.any():
                encoded = mask_utils.encode(np.asfortranarray(crop, dtype=np.uint8))
                encoded['counts'] = encoded['counts'].decode('ascii')
                result[i].append((encoded, iscrowd))
    return result


class FullImageEvaluator:
    """Stream GT preparation per image; evaluate final painted labels and scores.

    Only pycocotools is required in addition to NumPy/Pillow. No model/GPU imports.
    AP uses final postprocessing, including the existing overlap merge, painting,
    and score filtering. Semantic foreground means include empty images (both
    empty -> 1). COCO crowd regions are excluded from semantic pixel metrics.
    """
    def __init__(self, paths, image_root, *, annotation_file=None, mask_dir=None,
                 category_id=0, background_label=0, max_dets=100, score_threshold=0.0,
                 precomputed_gt=False):
        try:
            from pycocotools import mask as mask_utils
            from pycocotools.coco import COCO
        except ImportError as exc:
            raise ImportError('Evaluation needs pycocotools: pip install pycocotools') from exc
        if not precomputed_gt and bool(annotation_file) == bool(mask_dir):
            raise ValueError('Set exactly one of --eval-annotations or --eval-mask-dir')
        if max_dets < 1:
            raise ValueError('--eval-max-dets must be positive')
        self.score_threshold = score_threshold
        self.mask_utils, self.COCO = mask_utils, COCO
        self.category_id, self.background_label, self.max_dets = category_id, background_label, max_dets
        self.root = Path(image_root)
        self.paths = list(paths)
        self.ids = {p.relative_to(self.root).as_posix(): i for i, p in enumerate(paths, 1)}
        self.source = COCO(str(annotation_file)) if annotation_file else None
        self.gt_masks = Path(mask_dir) if mask_dir else None
        self.metadata = {}
        if self.source:
            if category_id not in self.source.cats:
                raise ValueError(f'category_id {category_id} not found in COCO categories')
            for im in self.source.dataset['images']:
                name = Path(im['file_name']).as_posix()
                if name in self.metadata:
                    raise ValueError(f'Duplicate COCO image filename: {name}')
                self.metadata[name] = im
            for name in self.ids:
                if name not in self.metadata:
                    raise ValueError(f'No COCO image entry for {name}')
        elif not precomputed_gt:
            seen = set()
            for name in self.ids:
                label_path = self.gt_masks / Path(name).with_suffix('.png')
                if not label_path.is_file():
                    raise FileNotFoundError(label_path)
                if label_path in seen:
                    raise ValueError(f'Two images map to the same GT label: {label_path}')
                seen.add(label_path)
        self.gt = dict(info={}, images=[], annotations=[],
                       categories=[dict(id=category_id, name='target')])
        self.predictions, self.rows, self.seen = [], [], set()

    def encode(self, mask):
        rle = self.mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
        rle['counts'] = rle['counts'].decode('ascii')
        return rle

    def read_ground_truth(self, path):
        name = Path(path).relative_to(self.root).as_posix()
        with Image.open(path) as image:
            w, h = image.size
        gt_union, crowd = np.zeros((h, w), bool), np.zeros((h, w), bool)
        anns = []
        if self.source:
            info = self.metadata[name]
            if (info['height'], info['width']) != (h, w):
                raise ValueError(f'COCO/image shape mismatch: {name}')
            for a in self.source.loadAnns(self.source.getAnnIds(imgIds=[info['id']], catIds=[self.category_id])):
                mask = self.source.annToMask(a).astype(bool)
                if mask.shape != (h, w):
                    raise ValueError(f'COCO mask shape mismatch: {name}')
                anns.append((self.encode(mask), int(a.get('iscrowd', 0))))
                if a.get('iscrowd', 0):
                    crowd |= mask
                else:
                    gt_union |= mask
        else:
            label_path = self.gt_masks / Path(name).with_suffix('.png')
            with Image.open(label_path) as image:
                target = np.asarray(image)
            if target.shape != (h, w) or not np.issubdtype(target.dtype, np.integer):
                raise ValueError(f'GT label shape/type mismatch: {label_path}; expected HxW instance IDs')
            gt_union = target != self.background_label
            for instance_id in np.unique(target):
                if instance_id != self.background_label:
                    anns.append((self.encode(target == instance_id), 0))
        return anns, gt_union, crowd

    def add(self, path, labels, scores, gt_annotations=None, row_metadata=None):
        name = Path(path).relative_to(self.root).as_posix()
        if name not in self.ids or name in self.seen:
            raise ValueError(f'Unknown or repeated evaluation image: {name}')
        with Image.open(path) as image:
            w, h = image.size
        labels = np.asarray(labels)
        if labels.shape != (h, w) or not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
            raise ValueError(f'Prediction label shape/type invalid for {name}; expected integer {(h, w)}')
        ids = np.unique(labels[labels != 0]).tolist()
        if any(i > len(scores) for i in ids) or not np.isfinite(scores).all():
            raise ValueError(f'Instance IDs must map to finite scores[instance_id - 1]: {name}')
        image_id = self.ids[name]
        if gt_annotations is None:
            anns, gt_union, crowd = self.read_ground_truth(path)
        else:
            anns = gt_annotations
            gt_union, crowd = np.zeros((h, w), bool), np.zeros((h, w), bool)
            for rle, iscrowd in anns:
                mask = self.mask_utils.decode(rle).astype(bool)
                if mask.shape != (h, w):
                    raise ValueError(f'Clipped GT shape mismatch for {name}')
                if iscrowd:
                    crowd |= mask
                else:
                    gt_union |= mask
        self.gt['images'].append(dict(id=image_id, file_name=name, width=w, height=h))
        for rle, iscrowd in anns:
            self.gt['annotations'].append(dict(
                id=len(self.gt['annotations']) + 1, image_id=image_id,
                category_id=self.category_id, segmentation=rle, iscrowd=iscrowd,
                area=float(self.mask_utils.area(rle)), bbox=self.mask_utils.toBbox(rle).tolist()))
        for instance_id in ids:
            self.predictions.append(dict(image_id=image_id, category_id=self.category_id,
                                         segmentation=self.encode(labels == instance_id),
                                         score=float(scores[instance_id - 1])))
        # A known non-crowd instance remains valid even if it overlaps a crowd region.
        metrics = foreground_metrics(labels > 0, gt_union, (~crowd) | gt_union)
        row = dict(image=name, image_id=image_id, **metrics,
                   gt_instances=sum(1 for _, c in anns if not c), pred_instances=len(ids))
        row.update(row_metadata or {})
        self.rows.append(row)
        self.seen.add(name)
        print(f'Eval {len(self.rows)}/{len(self.paths)}: {name}: '
              f'Dice={metrics["DICE"]}, IoU={metrics["IOU"]}, '
              f'GT={row["gt_instances"]}, predictions={len(ids)}', flush=True)

    def finish(self, output_dir, metadata=None):
        import csv
        from pycocotools.cocoeval import COCOeval
        if self.seen != set(self.ids):
            raise ValueError('Missing image predictions: refusing to report a partial test set')
        gt = self.COCO()
        gt.dataset = self.gt
        gt.createIndex()
        if self.predictions:
            dt = gt.loadRes(copy.deepcopy(self.predictions))
        else:
            dt = self.COCO()
            dt.dataset = dict(images=self.gt['images'], categories=self.gt['categories'], annotations=[])
            dt.createIndex()
        evaluator = COCOeval(gt, dt, iouType='segm')
        evaluator.params.imgIds = [im['id'] for im in self.gt['images']]
        evaluator.params.catIds = [self.category_id]
        evaluator.params.maxDets = sorted({min(1, self.max_dets), min(10, self.max_dets), self.max_dets})
        evaluator.evaluate()
        evaluator.accumulate()
        # COCO orders evalImgs by category, area, then image. Reuse its matching
        # records to compute each image's AP/AR without matching masks twice.
        ordered_ids = list(evaluator.params.imgIds)
        count = len(ordered_ids)
        row_by_id = {r['image_id']: r for r in self.rows}
        for offset, image_id in enumerate(ordered_ids):
            local = COCOeval(gt, dt, iouType='segm')
            local.params = copy.deepcopy(evaluator.params)
            local.params.imgIds = [image_id]
            local._paramsEval = copy.deepcopy(local.params)
            local.evalImgs = [evaluator.evalImgs[area * count + offset]
                              for area in range(len(evaluator.params.areaRng))]
            with contextlib.redirect_stdout(io.StringIO()):
                local.accumulate()
            row_by_id[image_id].update(instance_metrics(local, self.score_threshold))
        precision = evaluator.eval['precision'][:, :, :, 0, -1]
        recall = evaluator.eval['recall'][:, :, 0, -1]

        def defined_mean(values):
            values = np.asarray(values)
            valid = values[values >= 0]
            return float(valid.mean()) if valid.size else None

        def image_mean(key, nonempty=False):
            values = [r[key] for r in self.rows if r[key] is not None
                      and (not nonempty or r['gt_pixels'] > 0)]
            return float(np.mean(values)) if values else None

        intersection = sum(r['intersection'] for r in self.rows)
        total = sum(r['pred_pixels'] + r['gt_pixels'] for r in self.rows)
        union = sum(r['union'] for r in self.rows)
        metrics = dict(
            mDICE=image_mean('DICE'), mIOU=image_mean('IOU'),
            mDICE_nonempty_gt=image_mean('DICE', True), mIOU_nonempty_gt=image_mean('IOU', True),
            micro_DICE=2 * intersection / total if total else 1.0,
            micro_IOU=intersection / union if union else 1.0,
            AP=defined_mean(precision), AP50=defined_mean(precision[0]),
            AP75=defined_mean(precision[5]), AR=defined_mean(recall),
            images=len(self.rows), empty_gt_images=sum(r['gt_instances'] == 0 for r in self.rows),
            gt_instances=sum(r['gt_instances'] for r in self.rows),
            predicted_instances=len(self.predictions), AP_max_dets=self.max_dets,
            images_exceeding_AP_max_dets=sum(r['pred_instances'] > self.max_dets for r in self.rows),
            semantic_average='mean of per-image foreground scores; both empty = 1',
            AP_scope='final stitched, painted instances after pipeline filtering',
            filename_manifest=[r['image'] for r in self.rows])
        metrics.update(instance_metrics(evaluator, self.score_threshold))
        metrics['PR_score_threshold'] = self.score_threshold
        metrics['empty_instance_policy'] = 'P/R/F1=1 when both empty; AP/AR=null without GT'
        if metrics['images_exceeding_AP_max_dets']:
            print('Some images exceed the AP prediction cap; increase --eval-max-dets if needed.', flush=True)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for filename, value in [('metrics.json', metrics), ('per_image.json', self.rows),
                                ('predictions.coco.json', self.predictions),
                                ('ground_truth.coco.json', self.gt),
                                ('run_config.json', metadata or {})]:
            p = output / filename
            temp = p.with_suffix(p.suffix + '.tmp')
            temp.write_text(json.dumps(value, indent=2, allow_nan=False))
            temp.replace(p)
        with (output / 'per_image.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        with (output / 'worst_first.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(sorted(self.rows, key=lambda r: (r['F1_0.5'], r['IOU'] if r['IOU'] is not None else 1)))
        print(json.dumps({k: v for k, v in metrics.items() if k != 'filename_manifest'}, indent=2), flush=True)
        print(f'Evaluation saved to {output}', flush=True)
        return metrics


def saved_tile_manifest(args, paths):
    summary_path = args.output_dir / 'summary.json'
    if not summary_path.is_file():
        raise FileNotFoundError(f'Tile evaluation needs {summary_path}; use --eval-level image for image-only results')
    summary = json.loads(summary_path.read_text())
    by_source = {}
    for row in summary:
        by_source.setdefault(str(Path(row['image']).resolve()), []).append(row)
    result = []
    for path in paths:
        rows = by_source.get(str(path.resolve()), [])
        if not rows:
            raise ValueError(f'No saved tile manifest for {path}; the output may cover only an image subset')
        for row in rows:
            x0, y0, x1, y1 = row['box']
            name = f'tile_{row["tile_index"]:04d}_x{x0}_y{y0}_w{x1-x0}_h{y1-y0}'
            folder = args.output_dir / 'tiles' / path.relative_to(args.img_dir) / name
            for filename in ('tile.png', 'labels.png', 'instances.json'):
                if not (folder / filename).is_file():
                    raise FileNotFoundError(folder / filename)
            result.append(dict(source=path, folder=folder, image=folder / 'tile.png',
                               box=(x0, y0, x1, y1), tile_index=row['tile_index']))
    if len({r['image'] for r in result}) != len(result):
        raise ValueError('Duplicate tile entries in summary.json')
    return result


def evaluate_saved_outputs(args, paths, evaluator):
    include_images = args.eval_level in ('both', 'image')
    include_tiles = args.eval_level in ('both', 'tile')
    tile_evaluator = None
    tiles_by_source = {}
    if include_tiles:
        manifest = saved_tile_manifest(args, paths)
        tile_evaluator = FullImageEvaluator(
            [item['image'] for item in manifest], args.output_dir / 'tiles', precomputed_gt=True,
            category_id=args.eval_category_id, max_dets=args.eval_max_dets,
            score_threshold=args.eval_score_threshold)
        for item in manifest:
            tiles_by_source.setdefault(item['source'], []).append(item)
    print('Evaluating saved final tile/image outputs; ground truth does not affect inference.', flush=True)
    for path in paths:
        annotations, _, _ = evaluator.read_ground_truth(path)
        if include_tiles:
            items = tiles_by_source[path]
            cropped = crop_ground_truth(annotations, [item['box'] for item in items], evaluator.mask_utils)
            for item, gt in zip(items, cropped):
                with Image.open(item['folder'] / 'labels.png') as image:
                    tile_labels = np.asarray(image)
                records = json.loads((item['folder'] / 'instances.json').read_text())
                if sorted(r['instance_id'] for r in records) != list(range(1, len(records) + 1)):
                    raise ValueError(f'Invalid tile instance IDs: {item["folder"]}')
                scores = [r['predicted_iou'] for r in sorted(records, key=lambda r: r['instance_id'])]
                x0, y0, x1, y1 = item['box']
                tile_evaluator.add(item['image'], tile_labels, scores, gt_annotations=gt,
                                   row_metadata=dict(source_image=path.relative_to(args.img_dir).as_posix(),
                                                     tile_index=item['tile_index'], x0=x0, y0=y0, x1=x1, y1=y1))
        if include_images:
            relative = path.relative_to(args.img_dir)
            relative = relative.with_name(relative.name + '.png')
            label_path = args.output_dir / 'labels' / relative
            score_path = args.output_dir / 'scores' / relative.with_suffix('.json')
            with Image.open(label_path) as image:
                labels = np.asarray(image)
            record = json.loads(score_path.read_text())
            scores = record['scores']
            if record.get('instance_count', len(scores)) != len(scores):
                raise ValueError(f'Invalid instance_count in {score_path}')
            evaluator.add(path, labels, scores, gt_annotations=annotations)
    output = args.output_dir / 'evaluation'
    metadata = evaluation_metadata(args)
    results = {}
    if include_images:
        results['images'] = evaluator.finish(output / 'images', metadata)
    if include_tiles:
        results['tiles'] = tile_evaluator.finish(output / 'tiles', metadata)
        for name in ('per_image.csv', 'per_image.json'):
            (output / 'tiles' / name).rename(output / 'tiles' / name.replace('per_image', 'per_tile'))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    return results


def evaluation_metadata(args):
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata['note'] = ('In eval-only mode model/inference CLI defaults do not establish how cached '
                        'predictions were generated. See saved inference_config.json if available.')
    saved = args.output_dir / 'inference_config.json'
    if saved.is_file():
        metadata['saved_inference_config'] = json.loads(saved.read_text())
    return metadata


def run(args):
    paths = discover_images(args)
    evaluator = None
    if args.eval_annotations or args.eval_mask_dir:
        evaluator = FullImageEvaluator(
            paths, args.img_dir, annotation_file=args.eval_annotations,
            mask_dir=args.eval_mask_dir, category_id=args.eval_category_id,
            background_label=args.eval_background_label, max_dets=args.eval_max_dets,
            score_threshold=args.eval_score_threshold)
    if args.eval_only:
        return evaluate_saved_outputs(args, paths, evaluator)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'inference_config.json').write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    items = prepare_tiles(paths, args.img_dir, args.output_dir, args.tile_size, args.overlap,
                          skip_existing=args.skip_existing)
    pending = select_unet_items(items, args.skip_existing)
    print(f'UNet: {len(pending)} tiles to compute, {len(items)-len(pending)} cached tiles reused', flush=True)
    if pending:
        with tempfile.TemporaryDirectory(prefix='unet_tiles_') as temporary:
            manifest = Path(temporary) / 'manifest.json'
            manifest.write_text(json.dumps({
                'unet_repo': str(args.unet_repo), 'unet_checkpoint': str(args.unet_checkpoint),
                'device': args.device, 'unet_size': args.unet_size, 'items': pending,
            }))
            environment = os.environ.copy()
            environment['PYTHONPATH'] = str(args.unet_repo)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--_unet-worker', str(manifest)],
                           cwd=temporary, env=environment, check=True)
    # Import local SAM2 only after the UNet subprocess has released GPU memory.
    sys.path.insert(0, str(Path.cwd()))
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(Path.cwd() / 'sam2'):
        raise RuntimeError('Run from your official/custom SAM2 repository root')
    grid = make_grid(args.points_per_side)
    generator = None
    by_source = {}
    for item in items:
        by_source.setdefault(item['source'], []).append(item)
    summary = []
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as source:
                image = np.asarray(source.convert('RGB'))
            h, w = image.shape[:2]
            predictions = []
            for item in by_source[str(path)]:
                box = tuple(item['box'])
                x0, y0, x1, y1 = box
                tile = np.ascontiguousarray(image[y0:y1, x0:x1])
                probability = np.load(item['probability'])
                if probability.shape != tile.shape[:2]:
                    raise ValueError(f'UNet output shape does not match tile: {item["image"]}')
                retained, keep = filter_grid(grid, probability, args.foreground_threshold)
                folder = Path(item['folder'])
                save_gate_debug(tile, probability, grid, keep, folder, args.foreground_threshold,
                                save_probability=not args.skip_existing)
                anns = []
                if len(retained) >= args.min_points:
                    if generator is None:
                        overrides = ['++model.target_point_classifier=true'] if args.target_point_classifier else []
                        model = build_sam2(args.config, ckpt_path=str(args.checkpoint), device=args.device,
                                           hydra_overrides_extra=overrides)
                        kwargs = dict(
                            points_per_side=None, point_grids=[retained],
                            points_per_batch=args.points_per_batch,
                            pred_iou_thresh=args.pred_iou_threshold,
                            stability_score_thresh=args.stability_threshold,
                            box_nms_thresh=args.box_nms_threshold,
                            output_mode='binary_mask', crop_n_layers=0,
                        )
                        parameters = inspect.signature(SAM2AutomaticMaskGenerator.__init__).parameters
                        # Your custom AMG has a classifier-based point filter.
                        # A zero threshold accepts every sigmoid probability,
                        # leaving UNet as the only foreground point gate.
                        if 'target_point_threshold' in parameters:
                            kwargs['target_point_threshold'] = 0.0
                        if 'target_point_min_points' in parameters:
                            kwargs['target_point_min_points'] = 1
                        generator = SAM2AutomaticMaskGenerator(model, **kwargs)
                    generator.point_grids = [retained]  # MUST refresh for each tile.
                    anns = generator.generate(tile)
                tile_labels, tile_predictions, records = save_tile_instances(
                    tile, anns, folder, box, dump_masks=args.dump_instance_masks)
                predictions.extend(tile_predictions)
                Image.fromarray(tile_labels).save(folder / 'labels.png')
                Image.fromarray(make_overlay(tile, tile_labels)).save(item['overlay'])
                (folder / 'instances.json').write_text(json.dumps(records, indent=2))
                row = {'image': str(path), 'tile_index': item['index'], 'box': list(box),
                       'total_points': len(grid), 'retained_points': len(retained), 'masks': len(anns)}
                summary.append(row)
                print(f'{path.name}: tile {item["index"]}/{item["tile_count"]}, '
                      f'points {len(retained)}/{len(grid)}, {len(anns)} masks', flush=True)
            # Preserve the full-image overlap merge used in your uploaded script.
            labels, scores = merge_tile_predictions(predictions, (h, w),
                                                    args.merge_overlap_iou, args.min_overlap_pixels)
            relative = path.relative_to(args.img_dir)
            relative = relative.with_name(relative.name + '.png')
            label_path = args.output_dir / 'labels' / relative
            overlay_path = args.output_dir / 'overlays' / relative
            scores_path = args.output_dir / 'scores' / relative.with_suffix('.json')
            for output in (label_path, overlay_path, scores_path):
                output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(labels).save(label_path)
            Image.fromarray(make_overlay(image, labels)).save(overlay_path)
            scores_path.write_text(json.dumps({'instance_count': len(scores), 'scores': scores}, indent=2))
            print(f'{path.name}: {len(scores)} merged instances -> {label_path}', flush=True)

    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    if evaluator is not None:
        evaluate_saved_outputs(args, paths, evaluator)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', default="/data1/workspace/tien.nguyen/project/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_FBM_no_backout_finetune_sahi_positive_points_with_eval.yaml/checkpoints/checkpoint_best_ap.pt", type=Path, help='Your SAM2 checkpoint')
    parser.add_argument('--unet-repo', default="/data1/workspace/tien.nguyen/project/org-ais-models/primus/segmentation/SAM2-UNet/src/models", type=Path, help='Separate SAM2-UNet checkout, e.g. ../SAM2-UNet')
    parser.add_argument('--unet-checkpoint', default="/data1/workspace/tien.nguyen/project/sam2/SAM2-UNet_epoch-182_loss-0.616_iou-0.762_score-0.794.pth", type=Path, help='Target-trained SAM2-UNet state_dict')
    parser.add_argument('--unet-size', type=int, default=1024, help='Match your UNet training resolution')
    parser.add_argument('--img-dir', default="/data1/workspace/ai_shared_workspace/young_team_results/sam3_v0930/sam3_v260929_specialty_only_no_backout_full/test/", type=Path)
    parser.add_argument('--output-dir', default="outputs_with_sam2_unet_latest_checkpoint_20260110_0900_no_backout", type=Path, help='Output directory; use --skip-existing to resume')
    parser.add_argument('--config', default='configs/sam2.1/sam2.1_hiera_b+.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tile-size', type=int, default=2048)
    parser.add_argument('--overlap', type=float, default=0.2)
    parser.add_argument('--points-per-side', type=int, default=32)
    parser.add_argument('--points-per-batch', type=int, default=64)
    parser.add_argument('--foreground-threshold', '--target-threshold', dest='foreground_threshold', type=float,
                        default=0.5, help='UNet probability threshold; --target-threshold is a compatibility alias')
    parser.add_argument('--min-points', type=int, default=1, help='Skip SAM2 if fewer UNet-gated grid points remain')
    parser.add_argument('--target-point-classifier', action='store_true',
                        help='Instantiate the custom classifier head to match your SAM2 training checkpoint')
    parser.add_argument('--pred-iou-threshold', type=float, default=0.8)
    parser.add_argument('--stability-threshold', type=float, default=0.9)
    parser.add_argument('--box-nms-threshold', type=float, default=0.7)
    parser.add_argument('--merge-overlap-iou', type=float, default=0.6)
    parser.add_argument('--min-overlap-pixels', type=int, default=64)
    parser.add_argument('--dump-instance-masks', action='store_true',
                        help='Also save binary masks; individual instance overlays are always saved')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Reuse existing tile folders and valid UNet .npy files; rerun SAM2 and debug outputs')
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument('--eval-annotations', type=Path, help='Test COCO JSON: polygons or RLE', default="/data1/workspace/ai_shared_workspace/young_team_results/sam3_v0930/sam3_v260929_specialty_only_no_backout_full/test/_annotations.coco.json")
    evaluation.add_argument('--eval-mask-dir', type=Path, help='Mirrored GT instance-ID PNG root')
    parser.add_argument('--eval-only', action='store_true', help='Score saved final labels/scores; do not load models')
    parser.add_argument('--eval-level', choices=('both', 'image', 'tile'), default='both')
    parser.add_argument('--eval-score-threshold', type=float, default=0.0,
                        help='Score threshold for P/R/F1 only; AP uses all saved predictions')
    parser.add_argument('--eval-category-id', type=int, default=0, help='Single target category ID')
    parser.add_argument('--eval-background-label', type=int, default=0)
    parser.add_argument('--eval-max-dets', type=int, default=100, help='Per-image AP cap; use e.g. 300 for dense drawings')
    parser.add_argument('--filename-contains', default='', help='Optional filename substring, e.g. RCP; default ALL')
    args = parser.parse_args()
    for name in ('eval_annotations', 'eval_mask_dir'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    for name in ('checkpoint', 'unet_repo', 'unet_checkpoint', 'img_dir', 'output_dir'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not 0 <= args.eval_score_threshold <= 1:
        parser.error('--eval-score-threshold must be in [0, 1]')
    if args.eval_max_dets < 1:
        parser.error('--eval-max-dets must be positive')
    if args.eval_annotations and not args.eval_annotations.is_file():
        parser.error('--eval-annotations must be a file')
    if args.eval_mask_dir and not args.eval_mask_dir.is_dir():
        parser.error('--eval-mask-dir must be a directory')
    if args.eval_only:
        if not (args.eval_annotations or args.eval_mask_dir):
            parser.error('--eval-only requires --eval-annotations or --eval-mask-dir')
        if not args.img_dir.is_dir() or not args.output_dir.is_dir():
            parser.error('--img-dir and --output-dir must exist')
        run(args)
        return
    if not (Path.cwd() / 'sam2' / 'build_sam.py').is_file():
        parser.error('Run from your SAM2 repository root')
    if not (args.unet_repo / 'SAM2UNet.py').is_file() or args.unet_repo == Path.cwd():
        parser.error('--unet-repo must point to a separate SAM2-UNet checkout')
    if not args.checkpoint.is_file() or not args.unet_checkpoint.is_file() or not args.img_dir.is_dir():
        parser.error('Checkpoints must be files and --img-dir must be a directory')
    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            parser.error('--output-dir must be a directory')
        if any(args.output_dir.iterdir()) and not args.skip_existing:
            parser.error('For an existing output directory, add --skip-existing')
    if min(args.tile_size, args.points_per_side, args.points_per_batch, args.min_points, args.min_overlap_pixels) < 1:
        parser.error('Tile, grid, batch, and minimum counts must be positive')
    if not 0 <= args.overlap < 1:
        parser.error('--overlap must be in [0, 1)')
    if args.unet_size < 32 or args.unet_size % 32:
        parser.error('--unet-size must be a positive multiple of 32')
    for name in ('foreground_threshold', 'pred_iou_threshold', 'stability_threshold', 'box_nms_threshold', 'merge_overlap_iou'):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f'--{name.replace("_", "-")} must be in [0, 1]')
    run(args)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--_unet-worker':
        unet_worker(sys.argv[2])
    else:
        main()