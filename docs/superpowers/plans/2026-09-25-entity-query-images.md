# Entity Query Images Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and run prompt-free single-class instance discovery on image slices using SAM 2 features.

**Architecture:** Frozen SAM 2 backbone with a separate learned query decoder and mask/objectness head; Hungarian mask matching. COCO polygons supply image training data, and tiled inference saves per-slice outputs.

**Tech Stack:** Python, PyTorch, SAM 2, SciPy, Pillow.

**Spec:** `docs/superpowers/specs/2026-09-25-entity-query-images-design.md`

## Global Constraints

- Input is single-class COCO polygon segmentation, including zero-instance images.
- Existing interactive SAM 2 APIs remain compatible.
- Trained query weights are required for prompt-free inference.
- Slice identities are local; this milestone saves per-slice results.

## Review Focus

- Empty images yield finite objectness loss and no mask-match loss.
- Degenerate polygons are excluded from matching.
- Excess targets raise an actionable error.
- Right and bottom border tiles keep their actual dimensions.
- Background queries receive gradient without overwhelming positive queries.

### Task 1: Query model and matching

**Files:** Create `sam2/modeling/entity_query.py`; test `tests/test_entity_query.py`.

**Interfaces:** `EntityQueryModel(sam_model, num_queries=50, freeze_backbone=True)` returns `{'pred_masks': BxQxHxW, 'pred_logits': BxQ}`; `entity_query_loss(outputs, targets)` returns a dict of differentiable losses.

- [ ] Write tests for empty targets, matching, excess targets, forward shapes, and gradients; confirm they fail with a missing module.
- [ ] Implement image feature extraction, query attention, mask/objectness projection, matching and losses.
- [ ] Re-run focused tests; inspect class balance and target-mask resizing.

### Task 2: COCO training entry point

**Files:** Create `training/entity_query_images.py`; test `tests/test_entity_query_data.py`.

**Interfaces:** `CocoInstanceDataset(root, split, image_size)`, `collate_instances`, `train` CLI.

- [ ] Write and run a failing dataset test for polygons, empty images and invalid polygons.
- [ ] Implement dataset, initialization, training loop and checkpoint metadata.
- [ ] Re-run tests and check an example training invocation.

### Task 3: Slice inference and usage

**Files:** Create `tools/entity_query_slices.py`, `docs/entity_query_images.md`; test `tests/test_entity_query_slices.py`.

**Interfaces:** `tile_boxes(width, height, size, overlap)` yields `(x0,y0,x1,y1)`; CLI takes SAM 2 backbone, trained head, image folder and output folder.

- [ ] Write and run a failing boundary/overlap test.
- [ ] Implement slice prediction, PNG/JSON dump and examples.
- [ ] Re-run focused tests, Python compile, and smoke inference if dependencies and weights are available.
