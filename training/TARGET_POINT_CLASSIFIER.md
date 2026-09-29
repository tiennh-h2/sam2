# Single-class grid point classification

The optional `target_point_classifier` predicts whether a positive grid point
falls on the **one annotated target class**. Pixels outside target instance
masks, including other visible object types, are negative examples. It runs
from the image encoder's highest-resolution FPN output before mask decoding.

## Train

Start from `sam2/configs/sam2.1_training/sam2.1_hiera_b+_target_points.yaml`.
Set the dataset paths and checkpoint path for your data. The included data
section is the upstream PNGRawDataset example; convert your single-class COCO
annotations to its format or replace the data section with your image dataset.
Every image must have exhaustive annotations for the chosen target class.
Set `scratch.max_num_objects` at or above the maximum target instance count;
the upstream sampler otherwise omits target masks and labels real targets
as negatives for the classifier.
The standard collator cannot stack an image with zero object records, so an
empty image must have an empty-mask object record (or use a collator that
supports zero objects). Other classes must NOT be supplied as target masks.

The model head is enabled by `target_point_classifier: true`. The criterion
adds `loss_target_point` to the original SAM2 loss. It samples 32 by 32
locations, unions all target masks on each image, ignores labels near mask
boundaries, and balances foreground and negative locations within each image.
The training model also samples up to 16 grid locations outside the target
masks per image and sends them to SAM2 as **positive** clicks. Its separate
negative losses teach the decoder to predict an absent object, empty mask,
and zero IoU for these clicks. These negative losses are normalized by the
number of sampled negatives, independently of the original positive losses.
`ignore_missing_keys` initializes the new head randomly when loading the
original SAM2.1 checkpoint. Keep the head's trained weights at inference.

This trains on off-target grid points, which include visible non-target objects
and blank background. If your annotation format also records the other object
classes, a sampler that preferentially picks those object interiors would give
stronger hard-negative coverage than uniform off-target grid sampling.

## Inference

Load a checkpoint from this training run into a SAM2.1 base-plus model with
the new head enabled, then turn on point selection in the mask generator:

```python
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

model = build_sam2(
    "configs/sam2.1/sam2.1_hiera_b+.yaml",
    ckpt_path="/path/to/training_checkpoint.pt",
    hydra_overrides_extra=["++model.target_point_classifier=true"],
)
generator = SAM2AutomaticMaskGenerator(
    model,
    points_per_side=32,
    target_point_threshold=0.5,
    target_point_min_points=1,
)
masks = generator.generate(rgb_image)
```

Tune the threshold using **instance recall** on held-out images: an object
missed by the point classifier cannot be recovered by the decoder. The
minimum-point fallback ensures every crop retains at least one prompt;
it can still yield false proposals. Continue to use the generator's predicted
IoU, stability, and NMS filters.
