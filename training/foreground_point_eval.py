"""Single-class instance evaluation with supplied or GT-sampled foreground points.

Torch/SAM2 imports are deliberately lazy so metric/data tests run on CPU without
installing the model. Coordinates are original-image pixels, not normalized.
"""
from contextlib import nullcontext
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def encode_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('ascii')
    return rle


def _image_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Image path must be relative to image_dir: {name}')
    return path


def load_dataset(image_dir, annotation_file=None, mask_dir=None,
                 category_id=0, background_label=0):
    """Read COCO polygons/RLE or mirrored single-channel instance-ID PNGs.

    Layout: image_dir/task/00000.jpg -> mask_dir/task/00000.png.
    Empty images are retained; missing label files fail rather than becoming empty.
    """
    if bool(annotation_file) == bool(mask_dir):
        raise ValueError('Set exactly one of annotation_file or mask_dir')
    if annotation_file:
        coco = COCO(str(annotation_file))
        if {c['id'] for c in coco.dataset['categories']} != {category_id}:
            raise ValueError('This evaluator requires exactly the configured single category')
        gt = deepcopy(coco.dataset)
        gt.setdefault('info', {})
        if len({im['id'] for im in gt['images']}) != len(gt['images']):
            raise ValueError('Duplicate image IDs')
        if len({a['id'] for a in gt['annotations']}) != len(gt['annotations']):
            raise ValueError('Duplicate annotation IDs')
        for ann in gt['annotations']:
            if ann['category_id'] != category_id:
                raise ValueError('Annotation category does not match category_id')
            ann['segmentation'] = coco.annToRLE(ann)
            if isinstance(ann['segmentation']['counts'], bytes):
                ann['segmentation']['counts'] = ann['segmentation']['counts'].decode('ascii')
            ann['area'] = float(mask_utils.area(ann['segmentation']))
            ann['bbox'] = mask_utils.toBbox(ann['segmentation']).tolist()
            ann.setdefault('iscrowd', 0)
    else:
        gt = {'info': {}, 'images': [], 'annotations': [],
              'categories': [{'id': category_id, 'name': 'target'}]}
        exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
        files = sorted(p for p in Path(image_dir).rglob('*') if p.suffix.lower() in exts)
        label_paths = set()
        for image_id, path in enumerate(files, 1):
            rel = path.relative_to(image_dir)
            label_path = Path(mask_dir) / rel.with_suffix('.png')
            if label_path in label_paths:
                raise ValueError(f'Two images map to the same label file: {label_path}')
            label_paths.add(label_path)
            with Image.open(path) as image:
                w, h = image.size
            with Image.open(label_path) as image:
                label = np.asarray(image)
            if label.ndim != 2 or label.shape != (h, w):
                raise ValueError(f'Expected HxW instance-ID labels: {label_path}')
            gt['images'].append(dict(id=image_id, file_name=rel.as_posix(), width=w, height=h))
            for instance_id in np.unique(label):
                if instance_id == background_label:
                    continue
                rle = encode_mask(label == instance_id)
                gt['annotations'].append(dict(
                    id=len(gt['annotations']) + 1, image_id=image_id,
                    category_id=category_id, segmentation=rle, iscrowd=0,
                    area=float(mask_utils.area(rle)), bbox=mask_utils.toBbox(rle).tolist()))
    if not gt['images']:
        raise ValueError('No validation images found')
    names = [im['file_name'] for im in gt['images']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate image file_name values')
    for im in gt['images']:
        if not _image_path(image_dir, im['file_name']).is_file():
            raise FileNotFoundError(_image_path(image_dir, im['file_name']))
    return gt


def load_points(points_file, images):
    """JSON object {relative_image_name: [[x, y], ...]}; [] explicitly means none."""
    with open(points_file) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError('points_file must be a JSON object keyed by relative image name')
    result = {}
    for im in images:
        name = im['file_name']
        if name not in data:
            raise ValueError(f'Missing points entry for {name}; use [] for no points')
        value = data[name]
        arr = np.asarray(value, dtype=float)
        if isinstance(value, list) and len(value) == 0:
            result[im['id']] = []
            continue
        if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
            raise ValueError(f'Expected finite Nx2 point coordinates for {name}')
        if ((arr < 0).any() or (arr[:, 0] >= im['width']).any()
                or (arr[:, 1] >= im['height']).any()):
            raise ValueError(f'Point out of bounds for {name}')
        result[im['id']] = arr.tolist()
    return result


def sample_gt_points(gt, points_per_instance=1, seed=42):
    """Uniform foreground pixels per non-crowd instance, fixed across epochs.

    Sample without replacement, capped at the instance's pixel count. A separate
    RNG per annotation keeps prompts stable when selecting an image subset.
    """
    if not isinstance(points_per_instance, int) or points_per_instance < 1:
        raise ValueError('points_per_instance must be a positive integer')
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('points_seed must be a nonnegative integer')
    result = {im['id']: [] for im in gt['images']}
    for ann in sorted(gt['annotations'], key=lambda a: (a['image_id'], a['id'])):
        if ann.get('iscrowd', 0) or ann['image_id'] not in result:
            continue
        mask = mask_utils.decode(ann['segmentation'])
        ys, xs = np.nonzero(mask)
        if not len(xs):
            raise ValueError(f"Cannot sample a positive point from empty GT annotation {ann['id']}")
        rng = np.random.default_rng(np.random.SeedSequence([seed, ann['image_id'], ann['id']]))
        indices = rng.choice(len(xs), min(points_per_instance, len(xs)), replace=False)
        result[ann['image_id']].extend([[float(xs[i]), float(ys[i])] for i in indices])
    return result


def mask_nms(predictions, threshold):
    """Greedy mask-IoU suppression; never unions distinct predictions.

    Pass None to disable. Run on one image at a time.
    """
    if threshold is None:
        return sorted(predictions, key=lambda p: -p['score'])
    if not 0 <= threshold <= 1:
        raise ValueError('nms_iou must be in [0, 1] or null')
    remaining = sorted(predictions, key=lambda p: -p['score'])
    kept = []
    while remaining:
        best, remaining = remaining[0], remaining[1:]
        kept.append(best)
        if remaining:
            ious = mask_utils.iou([p['segmentation'] for p in remaining],
                                  [best['segmentation']], [0])[:, 0]
            remaining = [p for p, iou in zip(remaining, ious) if iou <= threshold]
    return kept


def evaluate_predictions(gt, predictions, score_threshold=0.5, max_dets=100):
    """COCO mask AP plus score-thresholded, one-to-one instance metrics.

    AP sees all scores. Operational counts use the same per-image max_dets cap.
    Undefined AP (no non-crowd GT) is None, never a fabricated zero/perfect score.
    """
    if max_dets < 1:
        raise ValueError('max_dets must be positive')
    coco_gt = COCO()
    coco_gt.dataset = deepcopy(gt)
    coco_gt.dataset.setdefault('info', {})
    # COCOeval reserves zero in match arrays for "unmatched". Some exports
    # number annotations from zero, so normalize IDs without changing masks.
    for annotation_id, ann in enumerate(coco_gt.dataset['annotations'], 1):
        ann['id'] = annotation_id
    coco_gt.createIndex()
    if predictions:
        coco_dt = coco_gt.loadRes(deepcopy(predictions))
    else:
        coco_dt = COCO()
        coco_dt.dataset = dict(images=deepcopy(gt['images']), annotations=[],
                               categories=deepcopy(gt['categories']))
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, iouType='segm')
    evaluator.params.imgIds = [im['id'] for im in gt['images']]
    evaluator.params.catIds = [c['id'] for c in gt['categories']]
    evaluator.params.maxDets = sorted(set([1, 10, int(max_dets)]))
    # Keep requested cap last, including when it is smaller than 10.
    evaluator.params.maxDets = [v for v in evaluator.params.maxDets if v <= max_dets]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval['precision'][:, :, :, 0, -1]
    recall = evaluator.eval['recall'][:, :, 0, -1]

    def mean_defined(values):
        valid = values[values >= 0]
        return float(valid.mean()) if valid.size else None

    out = {'AP': mean_defined(precision), 'AP50': mean_defined(precision[0]),
           'AP75': mean_defined(precision[5]), 'AR': mean_defined(recall)}
    tp = fp = fn = empty_images = empty_with_fp = empty_predictions = 0
    all_area = evaluator.params.areaRng[0]
    for record in evaluator.evalImgs:
        if record is None or list(record['aRng']) != list(all_area):
            continue
        n_gt = int((~record['gtIgnore'].astype(bool)).sum())
        active = np.asarray(record['dtScores']) >= score_threshold
        active &= ~record['dtIgnore'][0].astype(bool)
        matches = record['dtMatches'][0] > 0
        image_tp = int((active & matches).sum())
        image_fp = int((active & ~matches).sum())
        tp += image_tp
        fp += image_fp
        fn += n_gt - image_tp
        if n_gt == 0:
            empty_images += 1
            empty_with_fp += int(image_fp > 0)
            empty_predictions += image_fp
    # COCO returns None for images with neither GT nor predictions.
    evaluated = {r['image_id'] for r in evaluator.evalImgs
                 if r is not None and list(r['aRng']) == list(all_area)}
    empty_images += len(set(evaluator.params.imgIds) - evaluated)
    out.update(TP=tp, FP=fp, FN=fn,
               precision_50=tp / (tp + fp) if tp + fp else 0.0,
               recall_50=tp / (tp + fn) if tp + fn else 0.0,
               F1_50=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
               empty_images=empty_images,
               empty_fp_rate=empty_with_fp / empty_images if empty_images else None,
               empty_predictions_per_image=empty_predictions / empty_images if empty_images else None,
               images=len(gt['images']), predictions=len(predictions), max_dets=max_dets,
               score_threshold=score_threshold)
    return out


def choose_best_metrics(best, metrics, save_best_f1=False):
    updated, names = dict(best), []
    selections = [('AP', 'checkpoint_best_ap')]
    if save_best_f1:
        selections.append(('F1_50', 'checkpoint_best_f1'))
    for metric, name in selections:
        value = metrics.get(metric)
        key = f'point_eval/{metric}'
        if value is not None and math.isfinite(value) and value > updated.get(key, -math.inf):
            updated[key] = value
            names.append(name)
    return updated, names


class ForegroundPointEvaluator:
    def __init__(self, *, image_dir, points_file=None, annotation_file=None, mask_dir=None,
                 category_id=0, background_label=0, points_per_batch=32,
                 multimask_output=True, nms_iou=0.7, score_threshold=0.5,
                 max_dets=100, amp_dtype='bfloat16', num_images=None, sample_seed=42,
                 points_per_instance=1, points_seed=42):
        self.image_dir = Path(image_dir)
        self.category_id = category_id
        if points_per_batch < 1:
            raise ValueError('points_per_batch must be positive')
        if amp_dtype not in ('bfloat16', 'float16', 'float32'):
            raise ValueError('amp_dtype must be bfloat16, float16 or float32')
        self.points_per_batch = points_per_batch
        self.multimask_output = multimask_output
        self.nms_iou = nms_iou
        self.score_threshold = score_threshold
        self.max_dets = max_dets
        self.amp_dtype = amp_dtype
        self.gt = load_dataset(image_dir, annotation_file, mask_dir, category_id, background_label)
        if num_images is not None:
            if num_images < 1:
                raise ValueError('num_images must be positive or null')
            images = sorted(self.gt['images'], key=lambda im: im['id'])
            rng = np.random.default_rng(sample_seed)
            indices = sorted(rng.choice(len(images), min(num_images, len(images)), replace=False))
            self.gt['images'] = [images[i] for i in indices]
            ids = {im['id'] for im in self.gt['images']}
            self.gt['annotations'] = [a for a in self.gt['annotations'] if a['image_id'] in ids]
        self.point_source = 'supplied' if points_file else 'gt_random'
        self.points = (load_points(points_file, self.gt['images']) if points_file
                       else sample_gt_points(self.gt, points_per_instance, points_seed))

    def run(self, model, output_dir=None, epoch=None, config=None):
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = next(model.parameters()).device
        was_training = model.training
        predictions = []
        model.eval()
        predictor = None
        amp = (torch.autocast('cuda', dtype=getattr(torch, self.amp_dtype))
               if device.type == 'cuda' and self.amp_dtype != 'float32' else nullcontext())
        try:
            predictor = SAM2ImagePredictor(model)
            with torch.inference_mode(), amp:
                for index, im in enumerate(self.gt['images'], 1):
                    points = self.points[im['id']]
                    image_predictions = []
                    with Image.open(_image_path(self.image_dir, im['file_name'])) as image:
                        if image.size != (im['width'], im['height']):
                            raise ValueError(f"Image/annotation size mismatch: {im['file_name']}")
                        rgb = np.asarray(image.convert('RGB'))
                    if points:
                        predictor.set_image(rgb)
                    for start in range(0, len(points), self.points_per_batch):
                        batch = np.asarray(points[start:start + self.points_per_batch], dtype=np.float32)
                        # B independent prompts, each with one positive point; not one
                        # multi-point prompt that would encourage merging all instances.
                        masks, scores, _ = predictor.predict(
                            point_coords=batch[:, None, :],
                            point_labels=np.ones((len(batch), 1), dtype=np.int32),
                            multimask_output=self.multimask_output, return_logits=False)
                        if masks.ndim == 3:  # public API squeezes batch dimension for B=1
                            masks, scores = masks[None], scores[None]
                        if not np.isfinite(scores).all():
                            raise ValueError('Non-finite predicted IoU scores')
                        for candidate_masks, candidate_scores in zip(masks, scores):
                            best = int(np.argmax(candidate_scores))
                            mask = candidate_masks[best].astype(bool)
                            if not mask.any():
                                continue
                            image_predictions.append(dict(
                                image_id=im['id'], category_id=self.category_id,
                                segmentation=encode_mask(mask), score=float(candidate_scores[best])))
                    predictions.extend(mask_nms(image_predictions, self.nms_iou))
                    predictor.reset_predictor()
                    print(f"Point eval {index}/{len(self.gt['images'])}: {im['file_name']}, "
                          f'{len(points)} points, {len(image_predictions)} nonempty masks', flush=True)
        finally:
            if predictor is not None:
                predictor.reset_predictor()
            model.train(was_training)
        metrics = evaluate_predictions(self.gt, predictions, self.score_threshold, self.max_dets)
        metrics['supplied_points'] = sum(map(len, self.points.values()))
        metrics['point_source'] = self.point_source
        metrics['epoch'] = epoch
        if output_dir:
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            for name, value in [('metrics.json', metrics), ('predictions.coco.json', predictions),
                                ('evaluation_config.json', config or {}),
                                ('images.json', self.gt['images']),
                                ('points.json', {im['file_name']: self.points[im['id']]
                                                 for im in self.gt['images']})]:
                path = output / name
                temp = path.with_suffix(path.suffix + '.tmp')
                temp.write_text(json.dumps(value, indent=2, allow_nan=False))
                temp.replace(path)
        return metrics
