"""Normalized parent-ROI tile bounds and a learned SAM 2 prompt token."""
import re
from pathlib import Path
import torch
from torch import nn

_PATTERN = re.compile(r'__orig_w(\d+)_h(\d+)__x(\d+)_y(\d+)$')


def parse_tile_location(filename, tile_width, tile_height):
    """All dimensions are pixels BEFORE resizing/padding; offsets are ROI-local."""
    match = _PATTERN.search(Path(filename).stem)
    if match is None:
        raise ValueError(f'Missing __orig_wW_hH__xX_yY suffix: {filename}')
    width, height, x, y = map(int, match.groups())
    if min(width, height, tile_width, tile_height) <= 0:
        raise ValueError(f'Nonpositive image/crop dimensions: {filename}')
    if x + tile_width > width or y + tile_height > height:
        raise ValueError(f'Tile exceeds parent ROI bounds: {filename}; crop={tile_width}x{tile_height}')
    return [x / width, y / height, (x + tile_width) / width, (y + tile_height) / height]


class TileLocationEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        # Small initial token, but not an identity-equivalent decoder change.
        nn.init.normal_(self.net[-1].weight, std=0.001)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, location):
        return self.net(location * 2.0 - 1.0).unsqueeze(1)


def append_tile_location(encoder, sparse_embeddings, location):
    if encoder is None:
        return sparse_embeddings
    if location is None:
        raise ValueError('Tile-location model requires ROI-normalized bounds during training AND inference')
    parameter = next(encoder.parameters())
    location = torch.as_tensor(location, device=sparse_embeddings.device, dtype=parameter.dtype)
    if location.ndim == 1:
        location = location.unsqueeze(0)
    if location.ndim != 2 or location.shape[1] != 4:
        raise ValueError('tile_location must have shape [4] or [N,4]')
    if not bool(torch.isfinite(location).all()) or bool(((location < 0) | (location > 1)).any()):
        raise ValueError('tile_location must contain finite normalized bounds in [0,1]')
    if bool((location[:,2:] <= location[:,:2]).any()):
        raise ValueError('tile_location must have positive width and height')
    batch = sparse_embeddings.shape[0]
    if location.shape[0] == 1:
        location = location.expand(batch, -1)
    if location.shape[0] != batch:
        raise ValueError('tile_location batch does not match prompt batch')
    token = encoder(location).to(dtype=sparse_embeddings.dtype)
    return torch.cat((sparse_embeddings, token), dim=1)


def initialize_tile_location_model(state_dict, model, **kwargs):
    """Warm-start only: allow absent NEW location weights, reject other mismatches."""
    if kwargs.get('checkpoint_kernels'):
        for kernel in kwargs['checkpoint_kernels']:
            state_dict = kernel(state_dict=state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad = [key for key in missing if not key.startswith('tile_location_encoder.')]
    if bad or unexpected:
        raise RuntimeError(f'Checkpoint mismatch: missing={bad}, unexpected={unexpected}')
    if missing:
        import logging
        logging.warning('Initializing new tile-location encoder; fine-tuning is required.')
    return model
