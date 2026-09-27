# Prompt-free entity queries for SAM 2 images

## Purpose

Discover instances of the single foreground class in FBM image slices without points or boxes at inference. Train from COCO polygon instance masks, including valid images with zero instances. Save per-slice predictions for visual inspection.

## Architecture

Load an existing SAM 2 image checkpoint and use its image encoder and feature pyramid. A separate trainable head cross-attends a fixed number of learned queries to the lowest-resolution image features; each query predicts an objectness logit and a mask via a shared pixel projection. This adapts EntitySAM's query-based entity discovery and communication to still images. It deliberately omits temporal association, DINOv2 semantic encoding, and changes to SAM 2's interactive prompt decoder. Freeze the pretrained SAM 2 backbone by default; optional backbone fine-tuning is explicit.

This is a new head initialized randomly. The stock checkpoint cannot directly perform prompt-free inference. Store the new head's weights, query count, SAM 2 config and checkpoint metadata in a separate checkpoint.

## Data and optimization

Read COCO polygon instances for one class. Resize image and mask together to SAM 2's square input, using the SAM 2 image normalization. Exclude zero-area instance masks from matching. Keep an image with no instances: every query learns background. Match queries to target instances per image using Hungarian assignment over soft mask BCE and Dice costs. Train matched masks with BCE and Dice, and all query objectness logits with binary cross entropy (background down-weighted). If there are more instances than queries, fail with a helpful message instead of silently dropping targets.

## Inference

Load the trained head together with the exact SAM 2 config and backbone checkpoint. Split each image into overlapping fixed-size slices; resize each slice to the encoder input, predict instance masks, reject low-objectness or empty masks, and save masks and colored overlays per slice. Record slice coordinates and scores as JSON; do not assume queries have stable identities across slices. Output masks in each slice's original resolution. Full-image identity merging is a separate problem and is outside this first image-only integration.

## Verification

Exercise matching and loss on empty and populated targets, verify foreground/background gradients, and test polygon rasterization and slice coordinates. Compile the Python files; run a training/inference smoke test when PyTorch, its SAM 2 dependencies and a pretrained checkpoint are present.
