import unittest

from tools.entity_query_slices import tile_boxes


class TilingTests(unittest.TestCase):
    def test_border_and_overlap(self):
        boxes = list(tile_boxes(2300, 700, size=1024, overlap=256))
        self.assertEqual(boxes[0], (0, 0, 1024, 700))
        self.assertEqual(boxes[-1], (1276, 0, 2300, 700))
        self.assertEqual(len(boxes), len(set(boxes)))

    def test_small_image(self):
        self.assertEqual(list(tile_boxes(30, 20, size=1024, overlap=256)), [(0, 0, 30, 20)])
