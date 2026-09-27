"""Use set_tile(), then the normal predict() API with tile-local grid points."""
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.tile_location import parse_tile_location


class TileLocationImagePredictor(SAM2ImagePredictor):
    def __init__(self, sam_model, **kwargs):
        super().__init__(sam_model, **kwargs)
        if getattr(sam_model, 'tile_location_encoder', None) is None:
            raise ValueError('Load a model configured with use_tile_location: true')
        size = self.model.image_size
        self._bb_feat_sizes = [(size // stride, size // stride) for stride in (4, 8, 16)]

    def reset_predictor(self):
        super().reset_predictor()
        self._tile_location = None

    def set_image(self, image):
        raise ValueError('Use set_tile(image, filename=...) so location cannot be silently omitted')

    def set_image_batch(self, image_list):
        raise NotImplementedError('Use one tile at a time; point prompts can still be batched')

    def set_tile(self, image, *, filename=None, location=None):
        self.reset_predictor()
        if (filename is None) == (location is None):
            raise ValueError('Pass exactly one of filename or location')
        if filename is not None:
            if isinstance(image, np.ndarray):
                height, width = image.shape[:2]
            else:
                width, height = image.size
            location = parse_tile_location(filename, width, height)
        location = np.asarray(location, dtype=np.float32)
        if location.shape != (4,) or not np.isfinite(location).all() or (location < 0).any() or (location > 1).any() or (location[2:] <= location[:2]).any():
            raise ValueError('Expected valid normalized bounds [x0,y0,x1,y1]')
        super().set_image(image)
        self._tile_location = location

    def _predict(self, *args, **kwargs):
        if getattr(self, '_tile_location', None) is None:
            raise ValueError('Call set_tile before predicting')
        kwargs['tile_location'] = self._tile_location
        return super()._predict(*args, **kwargs)
