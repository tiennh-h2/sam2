#!/usr/bin/env python3
"""Run from the SAM2 repository root: python tools/evaluate_foreground_points.py --help."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAM2 masks from given foreground points')
    parser.add_argument('--config', required=True, help='Hydra config name, e.g. configs/sam2.1/sam2.1_hiera_b+.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--eval-config', required=True, help='YAML containing evaluator settings, or instance_eval: settings')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', default='point_eval_output')
    args = parser.parse_args()

    import torch
    import sam2  # initializes Hydra's sam2 config module
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from training.utils.train_utils import register_omegaconf_resolvers
    from training.foreground_point_eval import ForegroundPointEvaluator

    register_omegaconf_resolvers()
    cfg = compose(config_name=args.config)
    # Preserve the user's actual architecture, including custom heads. Do not
    # silently discard checkpoint keys or substitute the base SAM2 model.
    model_cfg = cfg.trainer.model if 'trainer' in cfg else cfg.model
    model = instantiate(model_cfg, _recursive_=True).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
    conf = OmegaConf.load(args.eval_config)
    if 'instance_eval' in conf:
        conf = conf.instance_eval
    conf = OmegaConf.to_container(conf, resolve=True)
    evaluator = ForegroundPointEvaluator(**{k: v for k, v in conf.items()
                                          if k not in ('every_n_epochs', 'save_best_f1')})
    metrics = evaluator.run(model, output_dir=args.output_dir, config=conf)
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
