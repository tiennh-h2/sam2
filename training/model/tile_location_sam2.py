"""Adds explicit per-object metadata after the existing grid-negative sampler."""
from training.model.grid_negative_sam2 import GridNegativeSAM2Train


class TileLocationGridNegativeSAM2Train(GridNegativeSAM2Train):
    def __init__(self, *args, **kwargs):
        if not kwargs.get('use_tile_location', False):
            raise ValueError('Set use_tile_location: true')
        if kwargs.get('num_correction_pt_per_frame', 0) != 0:
            raise ValueError('This image/grid extension requires num_correction_pt_per_frame: 0')
        kwargs['num_correction_pt_per_frame'] = 0
        for phase in ('train', 'eval'):
            if kwargs.get(f'prob_to_use_pt_input_for_{phase}', 0) != 1:
                raise ValueError(f'Set prob_to_use_pt_input_for_{phase}: 1.0')
            if kwargs.get(f'prob_to_use_box_input_for_{phase}', 0) != 0:
                raise ValueError(f'Set prob_to_use_box_input_for_{phase}: 0.0')
        super().__init__(*args, **kwargs)

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        if input.num_frames != 1 or input.tile_locations is None:
            raise ValueError('Single-frame batch and tile_locations are required')
        out = super().prepare_prompt_inputs(backbone_out, input, start_frame_idx)
        out['frames_to_add_correction_pt'] = []
        for t, prompts in out['point_inputs_per_frame'].items():
            # Includes empty targets added by the existing negative sampler.
            image_indices = input.obj_to_frame_idx[t, :, 1].long()
            prompts['tile_location'] = input.tile_locations[t].index_select(0, image_indices)
        return out
