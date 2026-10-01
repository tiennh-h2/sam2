"""Add given-point evaluation to the existing SAM2 Trainer without changing it."""
import hashlib
import json
import logging
from pathlib import Path
import traceback

import torch.distributed as dist
from omegaconf import OmegaConf

from training.trainer import Trainer
from training.foreground_point_eval import ForegroundPointEvaluator, choose_best_metrics


class BestPointTrainer(Trainer):
    def __init__(self, *, instance_eval, **kwargs):
        self.point_eval_config = (OmegaConf.to_container(instance_eval, resolve=True)
                                  if OmegaConf.is_config(instance_eval) else dict(instance_eval))
        self.every_n_epochs = int(self.point_eval_config.get('every_n_epochs', 1))
        if self.every_n_epochs < 1:
            raise ValueError('instance_eval.every_n_epochs must be positive')
        self.save_best_f1 = bool(self.point_eval_config.get('save_best_f1', False))
        self.latest_point_metrics = None
        self.point_evaluator = None
        super().__init__(**kwargs)
        if self.mode == 'val':
            raise ValueError('Use tools/evaluate_foreground_points.py for standalone evaluation')
        # Build the fixed validation manifest before the first training epoch.
        packet = [None]
        if self.distributed_rank == 0:
            try:
                conf = {k: v for k, v in self.point_eval_config.items()
                        if k not in ('every_n_epochs', 'save_best_f1')}
                self.point_evaluator = ForegroundPointEvaluator(**conf)
                payload = {'config': conf, 'ground_truth': self.point_evaluator.gt,
                           'points': self.point_evaluator.points}
                signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                if not any(not a.get('iscrowd', 0) for a in self.point_evaluator.gt['annotations']):
                    raise ValueError('Checkpoint selection requires at least one non-crowd GT instance')
                old = self.best_meter_values.get('point_eval/config_sha256')
                if old is not None and old != signature:
                    raise ValueError('Evaluation settings/data/points changed on resume. '
                                     'Start a new experiment to keep checkpoint comparisons valid.')
                packet[0] = {'signature': signature}
            except Exception:
                packet[0] = {'error': traceback.format_exc()}
        self._broadcast_packet(packet)
        if 'error' in packet[0]:
            raise RuntimeError(packet[0]['error'])
        self.best_meter_values['point_eval/config_sha256'] = packet[0]['signature']

    @staticmethod
    def _broadcast_packet(packet):
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast_object_list(packet, src=0)

    def save_checkpoint(self, epoch, checkpoint_names=None):
        # The upstream trainer calls this with no names once per completed epoch.
        # Explicit meter checkpoints must not recursively trigger evaluation.
        if checkpoint_names is not None:
            return super().save_checkpoint(epoch, checkpoint_names)
        names = ['checkpoint', 'checkpoint_last']  # checkpoint.pt is upstream auto-resume
        if ((self.checkpoint_conf.save_freq > 0 and epoch % self.checkpoint_conf.save_freq == 0)
                or epoch in self.checkpoint_conf.save_list):
            names.append(f'checkpoint_{epoch}')
        if epoch % self.every_n_epochs == 0 or epoch == self.max_epochs:
            packet = [None]
            if self.distributed_rank == 0:
                try:
                    model = self.model.module if hasattr(self.model, 'module') else self.model
                    metrics = self.point_evaluator.run(
                        model,
                        output_dir=Path(self.logging_conf.log_dir) / 'point_eval' / f'epoch_{epoch:04d}',
                        epoch=epoch, config=self.point_eval_config)
                    packet[0] = {'metrics': metrics}
                except Exception:
                    packet[0] = {'error': traceback.format_exc()}
            self._broadcast_packet(packet)
            if 'error' in packet[0]:
                # Preserve the trained epoch for recovery, but never mark it best.
                super().save_checkpoint(epoch, names)
                raise RuntimeError('Point evaluation failed; latest checkpoint saved.\n' + packet[0]['error'])
            metrics = packet[0]['metrics']
            self.latest_point_metrics = metrics
            self.best_meter_values, best_names = choose_best_metrics(
                self.best_meter_values, metrics, self.save_best_f1)
            for name in best_names:
                self.best_meter_values[f'point_eval/{name}_epoch'] = epoch
            names.extend(best_names)
            if self.distributed_rank == 0:
                scalars = {f'PointEval/{k}': v for k, v in metrics.items()
                           if isinstance(v, (int, float))}
                self.logger.log_dict(scalars, epoch)
                with open(Path(self.logging_conf.log_dir) / 'point_eval_history.jsonl', 'a') as f:
                    f.write(json.dumps(metrics, allow_nan=False) + '\n')
                logging.info('Point-conditioned evaluation: %s', metrics)
        # Saving occurs AFTER best values have been updated, including in last.pt.
        return super().save_checkpoint(epoch, list(dict.fromkeys(names)))

    def _save_checkpoint(self, checkpoint, checkpoint_path):
        checkpoint = dict(checkpoint)
        checkpoint['point_eval_config'] = self.point_eval_config
        checkpoint['point_eval_metrics'] = self.latest_point_metrics
        return super()._save_checkpoint(checkpoint, checkpoint_path)
