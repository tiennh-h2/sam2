"""Single-class mask evaluation using official pycocotools; no Detectron2."""
import contextlib
import copy
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def encode_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def improves(value, previous):
    return math.isfinite(value) and value >= 0 and (previous is None or value > previous)


def _make_coco(dataset):
    coco = COCO()
    coco.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def build_png_coco(image_dir, mask_dir, category_id=0, background_label=0):
    """Recursively pair images with relative-path PNG instance label maps.

    One positive integer label = one instance, including disconnected pieces.
    Every image requires a mask file; all-background masks represent negatives.
    RGB color masks and binary semantic masks with multiple objects are not
    instance-label maps. Palette PNG indices and uint16 PNGs are preserved.
    """
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    paths = sorted(p for p in image_dir.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'})
    if not paths:
        raise ValueError(f'No validation images under {image_dir}')
    images, annotations, seen_masks = [], [], set()
    for image_id, path in enumerate(paths):
        relative = path.relative_to(image_dir)
        mask_path = mask_dir / relative.with_suffix('.png')
        if mask_path in seen_masks:
            raise ValueError(f'Multiple images map to the same mask: {mask_path}')
        seen_masks.add(mask_path)
        with Image.open(path) as image:
            width, height = image.size
        with Image.open(mask_path) as image:
            labels = np.asarray(image)
        if labels.ndim != 2 or labels.shape != (height, width):
            raise ValueError(f'Expected {height}x{width} integer label map: {mask_path}')
        if not np.issubdtype(labels.dtype, np.integer) and labels.dtype != np.bool_:
            raise ValueError(f'Mask must contain integer instance IDs: {mask_path}')
        images.append(dict(id=image_id, file_name=relative.as_posix(), width=width, height=height))
        for label in np.unique(labels):
            if label == background_label:
                continue
            rle = encode_mask(labels == label)
            annotations.append(dict(id=len(annotations) + 1, image_id=image_id,
                                    category_id=int(category_id), segmentation=rle,
                                    area=float(mask_utils.area(rle)),
                                    bbox=mask_utils.toBbox(rle).tolist(), iscrowd=0))
    return _make_coco(dict(info={}, images=images, annotations=annotations,
                          categories=[dict(id=int(category_id), name='specialty')]))


def load_coco(annotation_file):
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(annotation_file))
    coco.dataset.setdefault('info', {})
    if len(coco.dataset['annotations']) != len(coco.anns):
        raise ValueError('Duplicate COCO annotation IDs')
    if len(coco.dataset['images']) != len(coco.imgs):
        raise ValueError('Duplicate COCO image IDs')
    # Reassign IDs from 1: COCOeval uses 0 as the unmatched sentinel.
    annotations = []
    for ann in coco.dataset['annotations']:
        ann = copy.deepcopy(ann)
        ann['segmentation'] = coco.annToRLE(ann)
        if isinstance(ann['segmentation']['counts'], bytes):
            ann['segmentation']['counts'] = ann['segmentation']['counts'].decode('ascii')
        ann['id'] = len(annotations) + 1
        ann.setdefault('iscrowd', 0)
        ann['area'] = float(mask_utils.area(ann['segmentation']))
        ann['bbox'] = mask_utils.toBbox(ann['segmentation']).tolist()
        annotations.append(ann)
    coco.dataset['annotations'] = annotations
    return _make_coco(coco.dataset)


def evaluate_predictions(coco_gt, predictions, max_dets=100, img_ids=None):
    """Return COCO mask metrics in percent; -1 means undefined/no GT.

    Empty predictions yield AP=0 if GT exists. Images with no predictions and
    negative images are included. Deliberately remove detection bboxes so mask
    area (not bounding-box area) determines detection area-range filtering.
    """
    if int(max_dets) != max_dets or max_dets < 10:
        raise ValueError('max_dets must be an integer >= 10')
    results = [{k: p[k] for k in ('image_id', 'category_id', 'segmentation', 'score')} for p in predictions]
    with contextlib.redirect_stdout(io.StringIO()):
        if results:
            coco_dt = coco_gt.loadRes(results)
        else:
            dataset = copy.deepcopy(coco_gt.dataset)
            dataset['annotations'] = []
            coco_dt = _make_coco(dataset)
        evaluator = COCOeval(coco_gt, coco_dt, 'segm')
        evaluator.params.imgIds = sorted(img_ids) if img_ids is not None else sorted(coco_gt.imgs)
        evaluator.params.maxDets = [1, 10, int(max_dets)]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval['precision']  # IoU, recall, category, area, maxDet
    recall = evaluator.eval['recall']
    def mean_valid(values):
        valid = values[values > -1]
        return float(valid.mean() * 100) if valid.size else -1.0
    metrics = {}
    for name, iou, area in [('AP', None, 0), ('AP50', .50, 0), ('AP75', .75, 0),
                            ('APs', None, 1), ('APm', None, 2), ('APl', None, 3)]:
        values = precision[:, :, :, area, -1]
        if iou is not None:
            values = values[np.isclose(evaluator.params.iouThrs, iou)]
        metrics[name] = mean_valid(values)
    for index, count in enumerate(evaluator.params.maxDets):
        metrics[f'AR{count}'] = mean_valid(recall[:, :, 0, index])
    return metrics
