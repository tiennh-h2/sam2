"""SAM2 Trainer extension: evaluate grid inference and retain only best AP.

Keep trainer.mode=train_only. The inherited epoch-end save call triggers this
hook on ALL ranks. Explicit meter saves are suppressed. No upstream edits.
"""
import contextlib
import json
import logging
import os
from pathlib import Path
import random
import traceback

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from training.trainer import Trainer, unwrap_ddp_if_wrapped
from training.coco_instance_eval import build_png_coco, load_coco, evaluate_predictions, improves


class BestCOCOTrainer(Trainer):
    def __init__(self, *, instance_eval, **kwargs):
        self.instance_eval_conf = OmegaConf.to_container(
            OmegaConf.create(instance_eval), resolve=True)
        cfg = self.instance_eval_conf
        if kwargs.get('mode', 'train') != 'train_only':
            raise ValueError('BestCOCOTrainer requires mode: train_only')
        if int(cfg.get('every_n_epochs', 1)) < 1:
            raise ValueError('every_n_epochs must be >= 1')
        if cfg.get('metric', 'AP') not in {'AP', 'AP50', 'AP75'}:
            raise ValueError('metric must be AP, AP50, or AP75')
        if not cfg.get('annotation_file') and not cfg.get('mask_dir'):
            raise ValueError('Provide annotation_file or mask_dir')
        self._coco_gt = None
        self._evaluation_state = None
        self._metric_key = 'InstanceSegBest/' + cfg.get('metric', 'AP')
        super().__init__(**kwargs)
        self._ground_truth()  # Fail on missing validation files before training.

    def _ground_truth(self):
        if self._coco_gt is None:
            cfg = self.instance_eval_conf
            if cfg.get('annotation_file'):
                self._coco_gt = load_coco(cfg['annotation_file'])
            else:
                self._coco_gt = build_png_coco(
                    cfg['image_dir'], cfg['mask_dir'],
                    category_id=cfg.get('category_id', 0),
                    background_label=cfg.get('background_label', 0))
            categories = self._coco_gt.getCatIds()
            if len(categories) != 1:
                raise ValueError('This grid evaluator requires exactly one COCO category')
            if not self._coco_gt.imgs or not any(not a.get('iscrowd', 0) and a['area'] > 0 for a in self._coco_gt.anns.values()):
                raise ValueError('Validation requires images and at least one non-crowd instance')
            self._category_id = int(categories[0])
            self._eval_image_ids = self._select_eval_images()
        return self._coco_gt

    def _select_eval_images(self):
        cfg = self.instance_eval_conf
        all_ids = sorted(self._coco_gt.imgs)
        n = cfg.get('num_images')
        if not n or n >= len(all_ids):
            return all_ids
        # Fixed, config-driven seed so every rank samples the identical subset
        # without touching the global random/np.random state.
        seed = int(cfg.get('sample_seed', 0))
        sampled = random.Random(seed).sample(all_ids, int(n))
        return sorted(sampled)

    def _predict_local_images(self):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        cfg = self.instance_eval_conf
        coco = self._ground_truth()
        model = unwrap_ddp_if_wrapped(self.model)
        # Restore mixed train/eval submodule flags, even after an exception.
        modes = [(module, module.training) for module in model.modules()]
        py_rng, np_rng = random.getstate(), np.random.get_state()
        device = next(model.parameters()).device
        devices = [device.index] if device.type == 'cuda' else []
        amp_dtype = cfg.get('amp_dtype', 'bfloat16')
        if amp_dtype not in {'bfloat16', 'float16', 'float32'}:
            raise ValueError('amp_dtype must be bfloat16, float16, or float32')
        amp = (torch.autocast('cuda', dtype=getattr(torch, amp_dtype))
               if device.type == 'cuda' and amp_dtype != 'float32' else contextlib.nullcontext())
        options = dict(cfg.get('generator', {}))
        if 'output_mode' in options:
            raise ValueError('Do not set generator.output_mode; evaluator uses coco_rle')
        predictions = []
        generator = None
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices), torch.inference_mode(), amp:
                generator = SAM2AutomaticMaskGenerator(model, output_mode='coco_rle', **options)
                image_ids = self._eval_image_ids[self.distributed_rank::dist.get_world_size()]
                for index, image_id in enumerate(image_ids):
                    record = coco.imgs[image_id]
                    with Image.open(Path(cfg['image_dir']) / record['file_name']) as image:
                        image = np.asarray(image.convert('RGB'))
                    if image.shape[:2] != (record['height'], record['width']):
                        raise ValueError(f"Image/annotation size mismatch: {record['file_name']}")
                    instances = generator.generate(image)
                    for instance in instances:
                        score = float(instance['predicted_iou'])
                        if instance['area'] <= 0 or not np.isfinite(score):
                            continue
                        predictions.append(dict(image_id=int(image_id), category_id=self._category_id,
                                                segmentation=instance['segmentation'], score=score))
                    if (index + 1) % 20 == 0:
                        logging.info('COCO eval rank %s: %s/%s images', self.distributed_rank, index + 1, len(image_ids))
        finally:
            if generator is not None:
                generator.predictor.reset_predictor()
            for module, was_training in modes:
                module.training = was_training
            random.setstate(py_rng)
            np.random.set_state(np_rng)
        return predictions

    def _evaluate_distributed(self):
        # All ranks execute direct, unwrapped inference on disjoint images.
        # Object collectives carry compact RLEs, not full-resolution mask arrays.
        packet = None
        try:
            packet = dict(predictions=self._predict_local_images(), error=None)
        except Exception:
            packet = dict(predictions=[], error=traceback.format_exc())
        packets = [None] * dist.get_world_size() if self.distributed_rank == 0 else None
        dist.gather_object(packet, packets, dst=0)
        result = [None]
        if self.distributed_rank == 0:
            try:
                errors = [f"rank {rank}: {p['error']}" for rank, p in enumerate(packets) if p['error']]
                if errors:
                    raise RuntimeError('\n'.join(errors))
                predictions = [p for packet in packets for p in packet['predictions']]
                metrics = evaluate_predictions(self._coco_gt, predictions,
                                               max_dets=int(self.instance_eval_conf.get('max_dets', 100)),
                                               img_ids=self._eval_image_ids)
                result[0] = dict(metrics=metrics, error=None)
            except Exception:
                result[0] = dict(metrics=None, error=traceback.format_exc())
        dist.broadcast_object_list(result, src=0)
        if result[0]['error']:
            raise RuntimeError('Instance validation failed:\n' + result[0]['error'])
        return result[0]['metrics']

    def save_checkpoint(self, epoch, checkpoint_names=None):
        # Ignore any upstream per-meter save requests: a single selection policy.
        if checkpoint_names is not None:
            return
        cfg = self.instance_eval_conf
        if epoch % int(cfg.get('every_n_epochs', 1)) and epoch != self.max_epochs:
            return
        metrics = self._evaluate_distributed()
        metric = cfg.get('metric', 'AP')
        previous = self.best_meter_values.get(self._metric_key)
        improved = improves(metrics[metric], previous)
        self._evaluation_state = dict(epoch=int(epoch), metric=metric, metrics=metrics,
                                      improved=improved, settings=cfg)
        if improved:
            self.best_meter_values[self._metric_key] = metrics[metric]
            self.best_meter_values['InstanceSegBest/epoch'] = int(epoch)
        # Rank zero writes both metrics and the checkpoint; broadcast errors so
        # other ranks cannot proceed into training after a failed disk write.
        status = [None]
        if self.distributed_rank == 0:
            try:
                self.logger.log_dict({f'InstanceSeg/{k}': v for k, v in metrics.items()}, int(epoch))
                log_path = Path(self.logging_conf.log_dir) / 'instance_eval.jsonl'
                with log_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(self._evaluation_state) + '\n')
                if improved:
                    # Explicit name suppresses save_freq/save_list copies. The
                    # native checkpoint name preserves automatic resume support.
                    super().save_checkpoint(epoch, checkpoint_names=['checkpoint'])
                logging.info('Epoch %s: %s=%.4f; best=%.4f; saved=%s', epoch, metric,
                             metrics[metric], self.best_meter_values.get(self._metric_key, -1), improved)
            except Exception:
                status[0] = traceback.format_exc()
        dist.broadcast_object_list(status, src=0)
        if status[0]:
            raise RuntimeError('Best-checkpoint save failed:\n' + status[0])

    def _save_checkpoint(self, checkpoint, checkpoint_path):
        checkpoint['instance_evaluation'] = self._evaluation_state
        # Current upstream save_checkpoint does not include this optional state.
        dataset = self.train_dataset
        if dataset is not None and hasattr(dataset, 'get_checkpoint_state'):
            checkpoint['train_dataset'] = dataset.get_checkpoint_state()
        # Local filesystem atomic replacement: no delete-before-rename gap.
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        try:
            with temporary.open('wb') as handle:
                torch.save(checkpoint, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
