import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO


def convert(split_dir: Path, output_dir: Path, split: str) -> None:
    coco = COCO(str(split_dir / "annotations.coco.json"))
    image_root = output_dir / "images" / split
    mask_root = output_dir / "masks" / split

    converted = 0
    for image_id in coco.getImgIds():
        info = coco.loadImgs(image_id)[0]
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
        annotations = [
            ann for ann in annotations
            if ann.get("iscrowd", 0) == 0
        ]
        if not annotations:
            continue  # The SAM 2 training sampler needs a visible object.

        source = split_dir / info["file_name"]
        if not source.is_file():
            source = split_dir / Path(info["file_name"]).name
        if not source.is_file():
            raise FileNotFoundError(f"Image {image_id}: {info['file_name']}")

        with Image.open(source) as im:
            rgb = im.convert("RGB")
            width, height = rgb.size

        if (width, height) != (info["width"], info["height"]):
            raise ValueError(f"COCO dimensions disagree with {source}")

        # A palettised PNG has at most 255 non-background IDs.
        if len(annotations) > 255:
            raise ValueError(f"Image {image_id} has over 255 instances")

        instance_mask = np.zeros((height, width), dtype=np.uint8)
        next_id = 1

        # Draw small objects last so they remain visible where polygons overlap.
        for ann in sorted(annotations, key=lambda a: a["area"], reverse=True):
            binary = coco.annToMask(ann).astype(bool)
            if not binary.any():
                continue
            instance_mask[binary] = next_id
            next_id += 1

        if next_id == 1:
            continue

        image_dir = image_root / str(image_id)
        annotation_dir = mask_root / str(image_id)
        image_dir.mkdir(parents=True, exist_ok=True)
        annotation_dir.mkdir(parents=True, exist_ok=True)

        rgb.save(image_dir / "00000.jpg", quality=95)
        Image.fromarray(instance_mask, mode="P").save(
            annotation_dir / "00000.png"
        )
        converted += 1

    print(f"{split}: converted {converted} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()
    convert(args.split_dir, args.output_dir, args.split)