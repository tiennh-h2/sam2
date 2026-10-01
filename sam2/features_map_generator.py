#!/usr/bin/env python3
"""UNet-selected grid points -> SAM2 instances -> feature clustering.
Run from the official SAM2 repository root. UNet uses a separate subprocess
so its bundled sam2 package cannot collide with the official SAM2 package.
Clustering assigns visualization groups; it does not merge instance masks.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw

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

class SAM2FeatureMaskGenerator:
    def __init__(self, model, points_per_side=32, min_points=1,
                 foreground_threshold=0.5, **kwargs):
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch = torch
        self.grid = make_grid(points_per_side)
        self.min_points = min_points
        self.foreground_threshold = foreground_threshold
        # Custom SAM2 checkpoints may contain a target-point classifier.
        # UNet already filters points, so disable an additional AMG point gate.
        parameters = inspect.signature(SAM2AutomaticMaskGenerator.__init__).parameters
        if 'target_point_threshold' in parameters:
            kwargs['target_point_threshold'] = 0.0
        if 'target_point_min_points' in parameters:
            kwargs['target_point_min_points'] = 1
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model, points_per_side=None, point_grids=[self.grid],
            crop_n_layers=0, output_mode='binary_mask', **kwargs)
        self.predictor = SAM2ImagePredictor(model)

    def generate(self, image, probability, debug_dir=None):
        import torch.nn.functional as F
        if probability.shape != image.shape[:2]:
            raise ValueError('UNet probability map must match the tile dimensions')
        retained, keep = filter_grid(self.grid, probability, self.foreground_threshold)
        if debug_dir is not None:
            save_gate_debug(image, probability, self.grid, keep, debug_dir,
                            self.foreground_threshold, save_probability=False)
        print(f'UNet selected {len(retained)}/{len(self.grid)} grid points', flush=True)
        # Never pass an empty grid to AMG or fall back to the unfiltered grid.
        if len(retained) < self.min_points:
            return []
        self.mask_generator.point_grids = [retained]
        with self.torch.inference_mode():
            masks = self.mask_generator.generate(image)
            if not masks:
                return []
            self.predictor.set_image(image)
            features = self.predictor.get_image_embedding().squeeze(0)
            channels, height, width = features.shape
            # Float32 avoids accumulation overflow with half-precision embeddings.
            feature_flat = features.float().reshape(channels, -1)
            # Pool in chunks instead of holding every full-resolution mask on GPU.
            for start in range(0, len(masks), 16):
                batch = masks[start:start + 16]
                array = np.stack([m['segmentation'] for m in batch])
                tensor = self.torch.from_numpy(array).to(features.device).float().unsqueeze(1)
                resized = F.interpolate(tensor, size=(height, width), mode='nearest').squeeze(1)
                flat = resized.reshape(len(batch), -1)
                # Very small masks can vanish under nearest-neighbor downsampling.
                vanished = flat.sum(1) == 0
                if vanished.any():
                    resized[vanished] = F.interpolate(tensor[vanished], size=(height, width),
                                                     mode='area').squeeze(1)
                    flat = resized.reshape(len(batch), -1)
                pooled = (flat @ feature_flat.T) / flat.sum(1, keepdim=True).clamp_min(1e-6)
                for record, embedding in zip(batch, pooled.cpu().numpy()):
                    record['embedding'] = embedding
            return masks


def prepare_unet_tiles(image_path, image, output_dir, tile_size, overlap):
    h, w = image.shape[:2]
    items = []
    for index, box in enumerate(iter_tile_boxes(h, w, tile_size, overlap), 1):
        x0, y0, x1, y1 = box
        folder = output_dir / 'tiles' / f'tile_{index:04d}_W{w}_H{h}_x{x0}_y{y0}_w{x1-x0}_h{y1-y0}'
        folder.mkdir(parents=True, exist_ok=True)
        tile_path = folder / 'tile.png'
        Image.fromarray(image[y0:y1, x0:x1]).save(tile_path)
        items.append({'image': str(tile_path), 'probability': str(folder / 'unet_probability.npy'),
                      'folder': str(folder), 'box': list(box)})
    return items


def run_unet(args, items):
    manifest = {'unet_repo': str(args.unet_repo), 'unet_checkpoint': str(args.unet_checkpoint),
                'unet_size': args.unet_size, 'device': args.device, 'items': items}
    manifest_path = args.output_dir / 'unet_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    env = os.environ.copy()
    env['PYTHONPATH'] = str(args.unet_repo)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--_unet-worker',
                    str(manifest_path)], cwd=str(args.unet_repo), env=env, check=True)


def process_image_tiled(image, generator, items):
    # Store masks in tile coordinates to avoid one whole-image canvas per mask.
    instances = []
    for index, item in enumerate(items, 1):
        x0, y0, x1, y1 = item['box']
        tile = np.ascontiguousarray(image[y0:y1, x0:x1])
        probability = np.load(item['probability'], allow_pickle=False)
        masks = generator.generate(tile, probability, Path(item['folder']))
        for mask in masks:
            instances.append({'segmentation': mask['segmentation'],
                              'embedding': mask['embedding'], 'tile_box': item['box'],
                              'area': int(mask['area']),
                              'predicted_iou': float(mask['predicted_iou']),
                              'bbox': [x0 + mask['bbox'][0], y0 + mask['bbox'][1],
                                       mask['bbox'][2], mask['bbox'][3]]})
        print(f'Tile {index}/{len(items)}: {len(masks)} masks', flush=True)
    return instances


def save_clustered_overlay(image, instances, n_clusters, output_dir):
    import cv2
    from sklearn.cluster import KMeans
    import matplotlib
    overlay = image.copy().astype(np.float32)
    labels = np.empty(0, dtype=np.int32)
    embeddings = np.empty((0, 0), dtype=np.float32)
    actual_clusters = 0
    if instances:
        embeddings = np.stack([m['embedding'] for m in instances])
        # Duplicate embeddings cannot form distinct clusters.
        actual_clusters = min(n_clusters, len(np.unique(embeddings, axis=0)))
        labels = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10).fit_predict(embeddings)
        cmap = matplotlib.colormaps['tab10' if actual_clusters <= 10 else 'gist_rainbow']
        positions = np.arange(actual_clusters) if actual_clusters <= 10 else np.linspace(0, 1, actual_clusters)
        colors = (cmap(positions)[:, :3] * 255).astype(np.uint8)
        for instance, label in zip(instances, labels):
            x0, y0, x1, y1 = instance['tile_box']
            local = overlay[y0:y1, x0:x1]
            mask = np.asarray(instance['segmentation'], dtype=bool)
            color = colors[label]
            local[mask] = 0.5 * local[mask] + 0.5 * color
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(local, contours, -1, color.tolist(), 2)
    # Always write an output, including when every tile is rejected by UNet.
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_dir / 'output_grouped_instances.jpg')
    np.save(output_dir / 'instance_embeddings.npy', embeddings)
    records = []
    for index, (instance, label) in enumerate(zip(instances, labels), 1):
        records.append({'instance_id': index, 'cluster_id': int(label),
                        'tile_box': instance['tile_box'], 'bbox': instance['bbox'],
                        'area': instance['area'], 'predicted_iou': instance['predicted_iou']})
    (output_dir / 'instances.json').write_text(json.dumps({
        'instance_count': len(instances), 'cluster_count': actual_clusters, 'instances': records}, indent=2))
    print(f'Saved {len(instances)} instances in {actual_clusters} clusters to {output_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=Path("/data1/workspace/tien.nguyen/project/common/data/fbm/commercial/specialty_roi_crop_dataset_20260916/test/roi_images/task5559__3001_COM_Ceiling_2-RCP_003__img2__roi-ann172__roi-id-1.png"), help='Input image file')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs_unet_feature_clusters'))
    parser.add_argument('--config', default='configs/sam2.1/sam2.1_hiera_b+.yaml')
    parser.add_argument('--checkpoint', type=Path, default="/data1/workspace/tien.nguyen/project/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_FBM_no_backout_finetune_sahi_positive_points_with_eval.yaml/checkpoints/checkpoint_best_ap.pt")
    parser.add_argument('--unet-repo', type=Path, default="/data1/workspace/tien.nguyen/project/org-ais-models/primus/segmentation/SAM2-UNet/src/models",
                        help='Folder containing SAM2UNet.py and its bundled sam2 package')
    parser.add_argument('--unet-checkpoint', type=Path, default="/data1/workspace/tien.nguyen/project/sam2/SAM2-UNet_epoch-182_loss-0.616_iou-0.762_score-0.794.pth",)
    parser.add_argument('--unet-size', type=int, default=1024)
    parser.add_argument('--foreground-threshold', '--target-threshold', type=float, default=0.5,
                        help='Keep grid points with UNet foreground probability >= this threshold')
    parser.add_argument('--min-points', type=int, default=1)
    parser.add_argument('--points-per-side', type=int, default=32)
    parser.add_argument('--points-per-batch', type=int, default=64)
    parser.add_argument('--pred-iou-threshold', type=float, default=0.8)
    parser.add_argument('--stability-threshold', type=float, default=0.9)
    parser.add_argument('--box-nms-threshold', type=float, default=0.7)
    parser.add_argument('--tile-size', type=int, default=2048)
    parser.add_argument('--overlap', type=float, default=0.2)
    parser.add_argument('--n-clusters', type=int, default=8)
    parser.add_argument('--target-point-classifier', action='store_true',
                        help='Build the custom SAM2 classifier head if present in your checkpoint')
    parser.add_argument('--device', default=None, help='Default: cuda if available, otherwise cpu')
    args = parser.parse_args()
    for name in ('image', 'output_dir', 'checkpoint', 'unet_repo', 'unet_checkpoint'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for name in ('image', 'checkpoint', 'unet_checkpoint'):
        if not getattr(args, name).is_file():
            parser.error(f'--{name.replace("_", "-")} must be an existing file')
    if not (args.unet_repo / 'SAM2UNet.py').is_file():
        parser.error('--unet-repo must contain SAM2UNet.py')
    if min(args.min_points, args.points_per_side, args.points_per_batch, args.tile_size, args.n_clusters) < 1:
        parser.error('Point counts, tile size and n-clusters must be positive')
    if args.unet_size < 32 or args.unet_size % 32:
        parser.error('--unet-size must be a positive multiple of 32')
    if not 0 <= args.overlap < 1:
        parser.error('--overlap must be in [0, 1)')
    for name in ('foreground_threshold', 'pred_iou_threshold', 'stability_threshold', 'box_nms_threshold'):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f'--{name.replace("_", "-")} must be in [0, 1]')
    # Put the official repository first; script may live outside that checkout.
    if not (Path.cwd() / 'sam2' / 'build_sam.py').is_file():
        parser.error('Run from your official SAM2 repository root')
    sys.path.insert(0, str(Path.cwd()))
    import torch
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(args.image) as source:
        image = np.asarray(source.convert('RGB'))
    items = prepare_unet_tiles(args.image, image, args.output_dir, args.tile_size, args.overlap)
    run_unet(args, items)
    # UNet worker exits and frees its GPU weights before the SAM2 stage starts.
    from sam2.build_sam import build_sam2
    overrides = ['++model.target_point_classifier=true'] if args.target_point_classifier else []
    model = build_sam2(args.config, ckpt_path=str(args.checkpoint), device=args.device,
                       hydra_overrides_extra=overrides)
    generator = SAM2FeatureMaskGenerator(
        model, points_per_side=args.points_per_side, min_points=args.min_points,
        foreground_threshold=args.foreground_threshold, points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_threshold, stability_score_thresh=args.stability_threshold,
        box_nms_thresh=args.box_nms_threshold)
    instances = process_image_tiled(image, generator, items)
    save_clustered_overlay(image, instances, args.n_clusters, args.output_dir)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--_unet-worker":
        unet_worker(sys.argv[2])
    else:
        main()