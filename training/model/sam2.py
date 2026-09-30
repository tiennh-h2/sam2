# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import colorsys
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
import torch
import torch.distributed
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.target_point_classifier import (
    sample_negative_grid_points,
    union_masks_per_image,
)
from sam2.modeling.sam2_utils import (
    get_1d_sine_pe,
    get_next_point,
    sample_box_points,
    select_closest_cond_frames,
)
from sam2.utils.misc import concat_points
from training.utils.data_utils import BatchedVideoDatapoint
class SAM2Train(SAM2Base):
    def __init__(
        self,
        image_encoder,
        memory_attention=None,
        memory_encoder=None,
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        # if it is greater than 1, we interactive point sampling in the 1st frame and other randomly selected frames
        num_frames_to_correct_for_train=1,  # default: only iteratively sample on first frame
        num_frames_to_correct_for_eval=1,  # default: only iteratively sample on first frame
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        # how many frames to use as initial conditioning frames (for both point input and mask input; the first frame is always used as an initial conditioning frame)
        # - if `rand_init_cond_frames` below is True, we randomly sample 1~num_init_cond_frames initial conditioning frames
        # - otherwise we sample a fixed number of num_init_cond_frames initial conditioning frames
        # note: for point input, we sample correction points on all such initial conditioning frames, and we require that `num_frames_to_correct` >= `num_init_cond_frames`;
        # these are initial conditioning frames because as we track the video, more conditioning frames might be added
        # when a frame receives correction clicks under point input if `add_all_frames_to_correct_as_cond=True`
        num_init_cond_frames_for_train=1,  # default: only use the first frame as initial conditioning frame
        num_init_cond_frames_for_eval=1,  # default: only use the first frame as initial conditioning frame
        rand_init_cond_frames_for_train=True,  # default: random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
        rand_init_cond_frames_for_eval=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        # how many additional correction points to sample (on each frame selected to be corrected)
        # note that the first frame receives an initial input click (in addition to any correction clicks)
        num_correction_pt_per_frame=7,
        # method for point sampling during evaluation
        # "uniform" (sample uniformly from error region) or "center" (use the point with the largest distance to error region boundary)
        # default to "center" to be consistent with evaluation in the SAM paper
        pt_sampling_for_eval="center",
        # During training, we optionally allow sampling the correction points from GT regions
        # instead of the prediction error regions with a small probability. This might allow the
        # model to overfit less to the error regions in training datasets
        prob_to_sample_from_gt_for_train=0.0,
        use_act_ckpt_iterative_pt_sampling=False,
        # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
        # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
        forward_backbone_per_frame_for_eval=False,
        freeze_image_encoder=False,
        negative_points_per_image=0,
        negative_grid_size=32,
        max_positive_objects_per_image=0,
        debug_dir="/home/tien.nguyen/workspace/project/sam2/.debug",
        debug_every_n_batches=1,
        debug_max_images=100,
        debug_rank0_only=True,
        debug_image_mean=(0.485, 0.456, 0.406),
        debug_image_std=(0.229, 0.224, 0.225),
        **kwargs,
    ):
        super().__init__(image_encoder, memory_attention, memory_encoder, **kwargs)
        if debug_every_n_batches < 1 or debug_max_images < 0:
            raise ValueError("debug_every_n_batches >= 1 and debug_max_images >= 0 required")
        if len(debug_image_mean) != 3 or len(debug_image_std) != 3:
            raise ValueError("Debug image mean/std must contain three RGB values")
        self.debug_dir = debug_dir
        self.debug_every_n_batches = debug_every_n_batches
        self.debug_max_images = debug_max_images  # 0 means unlimited
        self.debug_rank0_only = debug_rank0_only
        self.debug_image_mean = tuple(debug_image_mean)
        self.debug_image_std = tuple(debug_image_std)
        self._debug_batch_idx = 0
        self._debug_images_written = 0
        self._debug_run_id = uuid4().hex[:12]
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval
        self.negative_points_per_image = negative_points_per_image
        self.negative_grid_size = negative_grid_size
        if max_positive_objects_per_image < 0:
            raise ValueError("max_positive_objects_per_image must be non-negative")
        self.max_positive_objects_per_image = max_positive_objects_per_image
        if negative_points_per_image < 0 or negative_grid_size < 1:
            raise ValueError("Invalid negative point sampling configuration")
        # Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info(
                f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}"
            )
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval
        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        # Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)
        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
    def forward(self, input: BatchedVideoDatapoint):
        if self.training or not self.forward_backbone_per_frame_for_eval or self.target_point_classifier is not None:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(input.flat_img_batch)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        # The full union still defines background when only some objects are decoded.
        full_target_union = torch.stack([
            union_masks_per_image(
                input.masks[t], input.obj_to_frame_idx[t, :, 1], input.num_videos
            )
            for t in range(input.num_frames)
        ])
        if self.training and self.max_positive_objects_per_image:
            self._sample_training_objects(input)
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        # Capture the actual selected targets and initial prompts before decoding.
        debug_paths = self._dump_training_inputs(input, backbone_out, full_target_union)
        previous_stages_out = self.forward_tracking(
            backbone_out, input, full_target_union=full_target_union
        )
        if self.training and self.negative_points_per_image:
            if input.num_frames != 1:
                raise ValueError("Non-target positive-click training currently supports single-frame images")
            image_ids, negative_points = sample_negative_grid_points(
                full_target_union[0], self.negative_grid_size, self.negative_points_per_image
            )
            if debug_paths:
                self._dump_background_points(debug_paths, image_ids, negative_points)
            if len(image_ids):
                image_features = backbone_out["backbone_fpn"][-1][image_ids]
                if self.directly_add_no_mem_embed:
                    image_features = image_features + self.no_mem_embed.view(1, -1, 1, 1)
                high_res = (
                    [level[image_ids] for level in backbone_out["backbone_fpn"][:-1]]
                    if self.use_high_res_features_in_sam else None
                )
                _, high_masks, ious, _, _, _, object_scores = self._forward_sam_heads(
                    backbone_features=image_features,
                    point_inputs={
                        "point_coords": negative_points[:, None],
                        "point_labels": torch.ones(
                            len(image_ids), 1, dtype=torch.int32, device=image_ids.device
                        ),
                    },
                    high_res_features=high_res,
                    multimask_output=False,
                )
                previous_stages_out[0]["negative_point_predictions"] = (
                    high_masks, ious, object_scores
                )
        return previous_stages_out

    @staticmethod
    def _debug_color(object_index):
        return tuple(int(255 * c) for c in colorsys.hsv_to_rgb(
            (object_index * 0.61803398875) % 1.0, 0.75, 1.0
        ))

    @staticmethod
    def _debug_draw_points(image, coords, labels, color, prefix):
        """Draw SAM labels: 1=circle, 0=cross, 2/3=box corners, -1=padding."""
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        corners = {}
        for point_idx, (xy, label) in enumerate(zip(coords, labels)):
            label = int(label)
            if label == -1 or not np.isfinite(xy).all():
                continue
            x, y = map(float, xy)
            r = 5
            if label == 0:
                draw.line((x-r, y-r, x+r, y+r), fill=color, width=3)
                draw.line((x-r, y+r, x+r, y-r), fill=color, width=3)
            elif label in (2, 3):
                corners[label] = (x, y)
                draw.rectangle((x-r, y-r, x+r, y+r), outline=color, width=2)
            else:
                draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline='black', width=2)
            draw.text((x+7, y+2), f"{prefix}:{point_idx} L{label}",
                      fill=color, stroke_width=1, stroke_fill='black')
        if 2 in corners and 3 in corners:
            a, b = corners[2], corners[3]
            draw.rectangle((min(a[0], b[0]), min(a[1], b[1]),
                            max(a[0], b[0]), max(a[1], b[1])), outline=color, width=2)

    @staticmethod
    def _debug_save_image(folder, rgb, masks, object_ids, coords, labels, union, metadata):
        """CPU-only rendering; preserve overlapping instances in separate PNGs."""
        from PIL import Image
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(folder / 'image.png')
        Image.fromarray(union.astype(np.uint8) * 255).save(folder / 'full_target_union.png')
        overlay = rgb.astype(np.float32).copy()
        point_view = Image.fromarray(rgb)
        objects = []
        for local_idx, object_id in enumerate(object_ids):
            object_id = int(object_id)
            mask = masks[local_idx].astype(bool)
            color = SAM2Train._debug_color(object_id)
            overlay[mask] = overlay[mask] * 0.55 + np.asarray(color) * 0.45
            instance = rgb.astype(np.float32).copy()
            instance[mask] = instance[mask] * 0.55 + np.asarray(color) * 0.45
            instance = Image.fromarray(instance.astype(np.uint8))
            points = []
            if coords is not None:
                SAM2Train._debug_draw_points(instance, coords[local_idx], labels[local_idx], color, str(object_id))
                SAM2Train._debug_draw_points(point_view, coords[local_idx], labels[local_idx], color, str(object_id))
                for xy, label in zip(coords[local_idx], labels[local_idx]):
                    finite = bool(np.isfinite(xy).all())
                    x, y = map(float, xy)
                    inside = finite and 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]
                    points.append({'xy': [x, y] if finite else None, 'label': int(label),
                                   'in_bounds': bool(inside),
                                   'inside_own_mask': bool(mask[int(y), int(x)]) if inside else False})
            instance.save(folder / f'instance_{object_id:04d}_overlay.png')
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f'instance_{object_id:04d}_mask.png')
            objects.append({'object_row': object_id, 'color_rgb': list(color),
                            'mask_area': int(mask.sum()), 'empty_target': not bool(mask.any()),
                            'points': points})
        Image.fromarray(overlay.astype(np.uint8)).save(folder / 'masks_overlay.png')
        combined = Image.fromarray(overlay.astype(np.uint8))
        if coords is not None:
            for i, object_id in enumerate(object_ids):
                SAM2Train._debug_draw_points(combined, coords[i], labels[i],
                                             SAM2Train._debug_color(int(object_id)), str(int(object_id)))
        combined.save(folder / 'points_masks_overlay.png')
        point_view.save(folder / 'points.png')
        metadata['objects'] = objects
        metadata['legend'] = 'circle=SAM label 1; cross=0; square/rectangle=2,3; padding -1 hidden'
        (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    @torch.no_grad()
    def _dump_training_inputs(self, input, backbone_out, full_target_union):
        paths = {}
        if not self.training or not self.debug_dir:
            return paths
        batch_idx = self._debug_batch_idx
        self._debug_batch_idx += 1
        rank = (torch.distributed.get_rank() if torch.distributed.is_available()
                and torch.distributed.is_initialized() else 0)
        if (self.debug_rank0_only and rank != 0) or batch_idx % self.debug_every_n_batches:
            return paths
        if self.debug_max_images and self._debug_images_written >= self.debug_max_images:
            return paths
        root = Path(self.debug_dir) / f'run_{self._debug_run_id}' / f'rank_{rank:03d}'
        # Use the same flat image mapping used by forward_tracking (not t * B + b).
        for t in range(input.num_frames):
            frame_ids = input.flat_obj_to_img_idx[t].detach().cpu().numpy()
            video_ids = input.obj_to_frame_idx[t, :, 1].detach().cpu().numpy()
            prompt = backbone_out['point_inputs_per_frame'].get(t)
            for video_id in range(input.num_videos):
                if self.debug_max_images and self._debug_images_written >= self.debug_max_images:
                    return paths
                object_ids = np.flatnonzero(video_ids == video_id)
                flat_id = int(frame_ids[object_ids[0]]) if len(object_ids) else video_id * input.num_frames + t
                tensor = input.flat_img_batch[flat_id].detach().float().cpu().numpy()
                rgb_float = tensor.transpose(1, 2, 0)
                rgb_float = rgb_float * np.asarray(self.debug_image_std) + np.asarray(self.debug_image_mean)
                rgb = (rgb_float.clip(0, 1) * 255).round().astype(np.uint8)
                index = torch.as_tensor(object_ids, device=input.masks.device, dtype=torch.long)
                masks = input.masks[t].index_select(0, index).detach().cpu().numpy()
                union = full_target_union[t, video_id].detach().cpu().numpy().astype(bool)
                union = union.reshape(union.shape[-2:])
                if masks.shape[-2:] != rgb.shape[:2] or union.shape != rgb.shape[:2]:
                    raise ValueError('Debug dump: image and target mask sizes differ')
                coords = labels = None
                if prompt is not None:
                    pi = index.to(prompt['point_coords'].device)
                    coords = prompt['point_coords'].index_select(0, pi).detach().float().cpu().numpy()
                    labels = prompt['point_labels'].index_select(0, pi).detach().cpu().numpy()
                folder = root / f'batch_{batch_idx:07d}_frame_{t:03d}_image_{video_id:03d}'
                identifiers = input.metadata.unique_objects_identifier[t].index_select(
                    0, index.to(input.metadata.unique_objects_identifier.device)
                ).detach().cpu().tolist()
                self._debug_save_image(folder, rgb, masks, object_ids, coords, labels, union, {
                    'batch_index': batch_idx, 'frame_index': t, 'video_index': video_id,
                    'flat_image_index': flat_id, 'unique_objects_identifier': identifiers,
                    'image_mean': list(self.debug_image_mean), 'image_std': list(self.debug_image_std),
                    'normalized_image_min': float(tensor.min()), 'normalized_image_max': float(tensor.max()),
                    'selected_object_count': len(object_ids),
                    'prompt_type': 'point_or_box' if prompt is not None else (
                        'mask' if t in backbone_out['mask_inputs_per_frame'] else 'none'),
                    'note': 'Augmented model input; selected GT masks and initial prompts, before decoder. '
                            'Full target union includes objects omitted by sampling. '
                            'Later prediction-dependent correction clicks are not included.',
                })
                paths[flat_id] = folder
                self._debug_images_written += 1
        return paths

    @torch.no_grad()
    def _dump_background_points(self, paths, image_ids, negative_points):
        """These are label-1 clicks on background, each trained against an empty mask."""
        from PIL import Image
        ids = image_ids.detach().cpu().numpy()
        points = negative_points.detach().float().cpu().numpy()
        for flat_id, folder in paths.items():
            selected = points[ids == flat_id]
            with Image.open(folder / 'image.png') as image:
                view = image.convert('RGB')
            with Image.open(folder / 'points_masks_overlay.png') as image:
                combined = image.convert('RGB')
            with Image.open(folder / 'full_target_union.png') as image:
                union = np.asarray(image) > 0
            records = []
            for i, xy in enumerate(selected):
                for image in (view, combined):
                    self._debug_draw_points(image, [xy], [1], (255, 60, 60), f'BG{i}')
                x, y = map(float, xy)
                inside = bool(np.isfinite(xy).all() and 0 <= x < union.shape[1] and 0 <= y < union.shape[0])
                records.append({'xy': [x, y] if np.isfinite(xy).all() else None,
                                'label': 1, 'target_mask': 'empty', 'in_bounds': inside,
                                'inside_full_target_union': bool(union[int(y), int(x)]) if inside else False})
            view.save(folder / 'background_points.png')
            combined.save(folder / 'all_points_masks_overlay.png')
            (folder / 'background_points.json').write_text(json.dumps(records, indent=2), encoding='utf-8')

    def _sample_training_objects(self, input: BatchedVideoDatapoint):
        """Limit per-image decoder work, keeping all object-aligned fields in sync."""
        if input.num_frames > 1 and not torch.equal(
            input.obj_to_frame_idx[:, :, 1],
            input.obj_to_frame_idx[0, :, 1].expand(input.num_frames, -1),
        ):
            raise ValueError("Object order differs between frames")
        video_ids = input.obj_to_frame_idx[0, :, 1]
        selected = []
        for video_id in range(input.num_videos):
            candidates = torch.nonzero(video_ids == video_id).flatten()
            if len(candidates) > self.max_positive_objects_per_image:
                positions = self.rng.choice(
                    len(candidates), self.max_positive_objects_per_image,
                    replace=False,
                )
                candidates = candidates[torch.as_tensor(positions, device=candidates.device)]
            selected.append(candidates)
        keep = torch.cat(selected).sort().values
        # Trainer._step reads batch.masks after model(batch), so update that batch.
        input.masks = input.masks[:, keep]
        input.obj_to_frame_idx = input.obj_to_frame_idx[:, keep]
        input.metadata.unique_objects_identifier = (
            input.metadata.unique_objects_identifier[:, keep]
        )
        input.metadata.frame_orig_size = input.metadata.frame_orig_size[:, keep]
    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """Compute the image backbone features on the fly for the given img_ids."""
        # Only forward backbone on unique image ids to avoid repetitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None
        # Compute the image features on those unique image ids
        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]
        return image, vision_feats, vision_pos_embeds, feat_sizes
    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts. Optionally, we allow tracking from
        a custom `start_frame_idx` to the end of the video (for evaluation purposes).
        """
        # Load the ground-truth masks on all frames (so that we can later
        # sample correction points from them)
        # gt_masks_per_frame = {
        #     stage_id: targets.segments.unsqueeze(1)  # [B, 1, H_im, W_im]
        #     for stage_id, targets in enumerate(input.find_targets)
        # }
        gt_masks_per_frame = {
            stage_id: masks.unsqueeze(1)  # [B, 1, H_im, W_im]
            for stage_id, masks in enumerate(input.masks)
        }
        # gt_masks_per_frame = input.masks.unsqueeze(2) # [T,B,1,H_im,W_im] keep everything in tensor form
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = input.num_frames
        backbone_out["num_frames"] = num_frames
        # Randomly decide whether to use point inputs or mask inputs
        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        if num_frames == 1:
            # here we handle a special case for mixing video + SAM on image training,
            # where we force using point input for the SAM task on static images
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        assert num_init_cond_frames >= 1
        # (here `self.rng.random()` returns value in range 0.0 <= X < 1.0)
        use_pt_input = self.rng.random() < prob_to_use_pt_input
        if rand_init_cond_frames and num_init_cond_frames > 1:
            # randomly select 1 to `num_init_cond_frames` frames as initial conditioning frames
            num_init_cond_frames = self.rng.integers(
                1, num_init_cond_frames, endpoint=True
            )
        if (
            use_pt_input
            and rand_frames_to_correct
            and num_frames_to_correct > num_init_cond_frames
        ):
            # randomly select `num_init_cond_frames` to `num_frames_to_correct` frames to sample
            # correction clicks (only for the case of point input)
            num_frames_to_correct = self.rng.integers(
                num_init_cond_frames, num_frames_to_correct, endpoint=True
            )
        backbone_out["use_pt_input"] = use_pt_input
        # Sample initial conditioning frames
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]  # starting frame
        else:
            # starting frame + randomly selected remaining frames (without replacement)
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        # Prepare mask or point inputs on initial conditioning frames
        backbone_out["mask_inputs_per_frame"] = {}  # {frame_idx: <input_masks>}
        backbone_out["point_inputs_per_frame"] = {}  # {frame_idx: <input_points>}
        for t in init_cond_frames:
            if not use_pt_input:
                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
            else:
                # During training # P(box) = prob_to_use_pt_input * prob_to_use_box_input
                use_box_input = self.rng.random() < prob_to_use_box_input
                if use_box_input:
                    points, labels = sample_box_points(
                        gt_masks_per_frame[t],
                    )
                else:
                    # (here we only sample **one initial point** on initial conditioning frames from the
                    # ground-truth mask; we may sample more correction points on the fly)
                    points, labels = get_next_point(
                        gt_masks=gt_masks_per_frame[t],
                        pred_masks=None,
                        method=(
                            "uniform" if self.training else self.pt_sampling_for_eval
                        ),
                    )
                point_inputs = {"point_coords": points, "point_labels": labels}
                backbone_out["point_inputs_per_frame"][t] = point_inputs
        # Sample frames where we will add correction clicks on the fly
        # based on the error between prediction and ground-truth masks
        if not use_pt_input:
            # no correction points will be sampled when using mask inputs
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            assert num_frames_to_correct > num_init_cond_frames
            # initial cond frame + randomly selected remaining frames (without replacement)
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = (
                init_cond_frames
                + self.rng.choice(
                    backbone_out["frames_not_in_init_cond"], extra_num, replace=False
                ).tolist()
            )
        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt
        return backbone_out
    def forward_tracking(
        self, backbone_out, input: BatchedVideoDatapoint, return_dict=False,
        full_target_union=None,
    ):
        """Forward video tracking on each frame (and sample correction clicks)."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)
        # Starting the stage loop
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        # first process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        for stage_id in processing_order:
            # Get the image features for the current frames
            # img_ids = input.find_inputs[stage_id].img_ids
            img_ids = input.flat_obj_to_img_idx[stage_id]
            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if they are already computed).
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                # Otherwise, compute the image features on the fly for the given img_ids
                # (this might be used for evaluation on long videos to avoid backbone OOM).
                (
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids
                )
            # Get output masks based on this frame's prompts and previous memory
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
            )
            if "target_point_logits" in backbone_out:
                frame_image_ids = torch.arange(
                    input.num_videos, device=input.masks.device
                ) * input.num_frames + stage_id
                current_out["target_point_logits"] = backbone_out[
                    "target_point_logits"
                ][frame_image_ids]
                current_out["target_point_union"] = (
                    full_target_union[stage_id] if full_target_union is not None
                    else union_masks_per_image(
                        input.masks[stage_id],
                        input.obj_to_frame_idx[stage_id, :, 1],
                        input.num_videos,
                    )
                )
            # Append the output, depending on whether it's a conditioning frame
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out
        if return_dict:
            return output_dict
        # turn `output_dict` into a list for loss function
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]
        return all_frame_outputs
    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        run_mem_encoder=True,  # Whether to run the memory encoder on the predicted masks.
        prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
        frames_to_add_correction_pt=None,
        gt_masks=None,
    ):
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )
        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs
        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]
        # Optionally, sample correction points iteratively to correct the mask
        if frame_idx in frames_to_add_correction_pt and self.num_correction_pt_per_frame > 0:
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame,
                point_inputs,
                gt_masks,
                high_res_features,
                pix_feat,
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                object_score_logits,
                current_out,
            )
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = final_sam_outputs
        # Use the final prediction (after all correction steps for output and eval)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )
        return current_out
    def _iter_correct_pt_sampling(
        self,
        is_init_cond_frame,
        point_inputs,
        gt_masks,
        high_res_features,
        pix_feat_with_mem,
        low_res_multimasks,
        high_res_multimasks,
        ious,
        low_res_masks,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        assert gt_masks is not None
        all_pred_masks = [low_res_masks]
        all_pred_high_res_masks = [high_res_masks]
        all_pred_multimasks = [low_res_multimasks]
        all_pred_high_res_multimasks = [high_res_multimasks]
        all_pred_ious = [ious]
        all_point_inputs = [point_inputs]
        all_object_score_logits = [object_score_logits]
        for _ in range(self.num_correction_pt_per_frame):
            # sample a new point from the error between prediction and ground-truth
            # (with a small probability, directly sample from GT masks instead of errors)
            if self.training and self.prob_to_sample_from_gt_for_train > 0:
                sample_from_gt = (
                    self.rng.random() < self.prob_to_sample_from_gt_for_train
                )
            else:
                sample_from_gt = False
            # if `pred_for_new_pt` is None, only GT masks will be used for point sampling
            pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)
            new_points, new_labels = get_next_point(
                gt_masks=gt_masks,
                pred_masks=pred_for_new_pt,
                method="uniform" if self.training else self.pt_sampling_for_eval,
            )
            point_inputs = concat_points(point_inputs, new_points, new_labels)
            # Feed the mask logits of the previous SAM outputs in the next SAM decoder step.
            # For tracking, this means that when the user adds a correction click, we also feed
            # the tracking output mask logits along with the click as input to the SAM decoder.
            mask_inputs = low_res_masks
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            if self.use_act_ckpt_iterative_pt_sampling and not multimask_output:
                sam_outputs = torch.utils.checkpoint.checkpoint(
                    self._forward_sam_heads,
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    use_reentrant=False,
                )
            else:
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                )
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                _,
                object_score_logits,
            ) = sam_outputs
            all_pred_masks.append(low_res_masks)
            all_pred_high_res_masks.append(high_res_masks)
            all_pred_multimasks.append(low_res_multimasks)
            all_pred_high_res_multimasks.append(high_res_multimasks)
            all_pred_ious.append(ious)
            all_point_inputs.append(point_inputs)
            all_object_score_logits.append(object_score_logits)
        # Concatenate the masks along channel (to compute losses on all of them,
        # using `MultiStepIteractiveMasks`)
        current_out["multistep_pred_masks"] = torch.cat(all_pred_masks, dim=1)
        current_out["multistep_pred_masks_high_res"] = torch.cat(
            all_pred_high_res_masks, dim=1
        )
        current_out["multistep_pred_multimasks"] = all_pred_multimasks
        current_out["multistep_pred_multimasks_high_res"] = all_pred_high_res_multimasks
        current_out["multistep_pred_ious"] = all_pred_ious
        current_out["multistep_point_inputs"] = all_point_inputs
        current_out["multistep_object_score_logits"] = all_object_score_logits
        return point_inputs, sam_outputs