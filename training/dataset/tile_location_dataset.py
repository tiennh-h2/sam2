"""Metadata is captured before image transforms, then collated as a tensor."""
from pathlib import Path
import torch
from training.dataset.vos_dataset import VOSDataset
from training.utils.data_utils import collate_fn
from sam2.tile_location import parse_tile_location

# Only transforms preserving the crop's geometry are supported in this version.
_ALLOWED = {'ComposeAPI','RandomResizeAPI','ColorJitter','RandomGrayscale','ToTensorAPI','NormalizeAPI','AddGridNegativeTarget'}


def validate_transforms(transforms):
    for transform in transforms:
        name = type(transform).__name__
        if name not in _ALLOWED:
            raise ValueError(f'{name} is not supported with tile locations; use resize/photometric transforms only')
        if name == 'ComposeAPI':
            validate_transforms(transform.transforms)


class TileLocationVOSDataset(VOSDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        validate_transforms(self._transforms)

    def construct(self, video, sampled_frms_and_objs, segment_loader):
        data = super().construct(video, sampled_frms_and_objs, segment_loader)
        if len(data.frames) != 1:
            raise ValueError('Tile location extension supports single-frame image training only')
        path = Path(sampled_frms_and_objs.frames[0].image_path)
        # PNG raw datasets commonly use <tile_name>/00000.jpg.
        name = path.name if '__orig_w' in path.stem else path.parent.name
        width, height = data.frames[0].data.size
        data.tile_locations = [parse_tile_location(name, width, height)]
        return data


def collate_tile_location(batch, dict_key):
    result = collate_fn(batch, dict_key)
    result.tile_locations = torch.tensor([sample.tile_locations for sample in batch], dtype=torch.float32).transpose(0, 1).contiguous()
    return result