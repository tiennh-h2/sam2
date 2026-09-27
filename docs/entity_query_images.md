# Image-only EntitySAM-style queries on SAM 2

This extension adapts [EntitySAM's](https://github.com/ymq2017/entitysam) learned-query idea to SAM 2 image features. It is a separately trained query decoder; it does not load EntitySAM weights or change SAM 2's point/box APIs. Video tracking and merging the identities of instances across image slices are not part of this extension.

From the SAM 2 repository root, with SAM 2 and SciPy installed (`pip install -e . scipy`), train on folders such as `DATA/train/example.png` and `DATA/train/_annotations.coco.json`:

```bash
python -m training.entity_query_images \
  --data-root /path/to/DATA \
  --model-cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
  --sam-checkpoint /path/to/sam2.1_hiera_base_plus.pt \
  --output /path/to/entity_query.pt --epochs 20 --num-queries 50
```

The COCO file must list one category and contain polygon segmentations. Images with no annotations are valid background examples. A polygon that becomes empty after resize is ignored. Increase `--num-queries` when a training image has more valid instances than queries. By default the SAM 2 backbone is frozen; `--finetune-backbone` trains it and saves its weights in the query checkpoint.

Run tiled prediction:

```bash
python -m tools.entity_query_slices \
  --query-checkpoint /path/to/entity_query.pt \
  --sam-checkpoint /path/to/sam2.1_hiera_base_plus.pt \
  --img-dir /path/to/DATA/test --out-dir /path/to/slice_outputs \
  --tile-size 1024 --overlap 256 --score-threshold 0.5
```

For every tile the script writes `image.png`, `overlay.png`, individual binary `mask_*.png` files, and `predictions.json` containing source image, slice coordinates, query indices, scores and mask filenames. Masks have the tile's original dimensions. The same query index on different tiles does not indicate the same instance.
