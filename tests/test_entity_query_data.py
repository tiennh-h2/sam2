import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from training.entity_query_images import CocoInstanceDataset


class CocoInstanceDatasetTests(unittest.TestCase):
    def test_polygon_and_empty_image(self):
        with tempfile.TemporaryDirectory() as root:
            split = Path(root) / "train"
            split.mkdir()
            for image_id in (1, 2):
                Image.new("RGB", (12, 8), "white").save(split / f"{image_id}.png")
            coco = {
                "images": [{"id": i, "file_name": f"{i}.png"} for i in (1, 2)],
                "categories": [{"id": 1, "name": "pattern"}],
                "annotations": [
                    {"image_id": 1, "segmentation": [[1, 1, 6, 1, 6, 6, 1, 6]]},
                    {"image_id": 1, "segmentation": [[2, 2, 2, 2, 2, 2]]},
                ],
            }
            (split / "_annotations.coco.json").write_text(json.dumps(coco))
            ds = CocoInstanceDataset(root, "train", image_size=32)
            _, mask = ds[0]
            self.assertEqual(mask.shape, (1, 32, 32))
            self.assertGreater(mask.sum().item(), 0)
            _, mask = ds[1]
            self.assertEqual(mask.shape, (0, 32, 32))
