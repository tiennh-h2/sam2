#!/usr/bin/env python3
"""
convert_to_sam2_vos.py

Converts an image/label-mask dataset of the form:

    <root>/<split>/images/*.png
    <root>/<split>/labels/*.png

into the layout expected by SAM2's training `PNGRawDataset` (VOSRawDataset /
PalettisedPNGSegmentLoader), where every training sample is treated as a
1-frame "video":

    <out>/images/<video_name>/00000.jpg
    <out>/masks/<video_name>/00000.png   # palette-mode PNG, pixel value = object id
    <out>/train.txt                            # list of video_names (== file_list_txt)

`video_name` is the original filename stem, so one image = one video.
`00000` is required to be purely numeric because PNGRawDataset.get_video()
parses the frame id with `int(os.path.basename(fpath).split(".")[0])`.

Then in your SAM2 training config:

    dataset = PNGRawDataset(
        img_folder="<out>/images",
        gt_folder="<out>/masks",
        file_list_txt="<out>/train.txt",   # optional, defaults to all dirs in img_folder
        is_palette=True,
    )

Object-id assumptions (same as before, override with --connected-components
if your labels are semantic/class masks rather than instance masks):
  - Each distinct non-zero pixel value in labels/*.png is a separate object.
  - RGB-encoded label maps are auto-flattened to indexed ids.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from scipy import ndimage
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def davis_palette() -> bytes:
    """Standard bit-interleaved colormap (DAVIS/PASCAL-VOC style) so masks are
    also human-viewable, and distinct object ids never collide in color."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        lab = i
        r = g = b = 0
        for j in range(8):
            r |= ((lab >> 0) & 1) << (7 - j)
            g |= ((lab >> 1) & 1) << (7 - j)
            b |= ((lab >> 2) & 1) << (7 - j)
            lab >>= 3
        palette[i] = (r, g, b)
    return palette.flatten().tobytes()


DAVIS_PALETTE = davis_palette()


def find_pairs(images_dir: Path, labels_dir: Path):
    image_files = {p.stem: p for p in images_dir.glob("*.png")}
    label_files = {p.stem: p for p in labels_dir.glob("*.png")}

    stems = sorted(set(image_files) & set(label_files))
    missing_labels = sorted(set(image_files) - set(label_files))
    missing_images = sorted(set(label_files) - set(image_files))

    if missing_labels:
        print(f"  [warn] {len(missing_labels)} images have no matching label, skipping "
              f"(e.g. {missing_labels[:3]})", file=sys.stderr)
    if missing_images:
        print(f"  [warn] {len(missing_images)} labels have no matching image, skipping "
              f"(e.g. {missing_images[:3]})", file=sys.stderr)

    return [(stem, image_files[stem], label_files[stem]) for stem in stems]


def load_label_as_indexed(label_path: Path) -> np.ndarray:
    """Load a label PNG and return an int32 array where each distinct object
    has a distinct value and 0 is background, regardless of source encoding
    (single-channel indexed/greyscale, or RGB-encoded classes)."""
    label = Image.open(label_path)
    arr = np.array(label)

    if arr.ndim == 3:
        flat = arr.reshape(-1, arr.shape[-1])
        colors, inverse = np.unique(flat, axis=0, return_inverse=True)
        arr = inverse.reshape(arr.shape[:2]).astype(np.int32)
        bg_idx = np.argmax(np.bincount(inverse))
        if bg_idx != 0:
            remap = np.arange(len(colors))
            remap[0], remap[bg_idx] = remap[bg_idx], remap[0]
            arr = remap[arr]
    return arr.astype(np.int32)


def remap_mask(mask: np.ndarray, connected_components: bool) -> np.ndarray:
    """Compact an arbitrary-valued label mask to ids 0 (background), 1..N (objects)."""
    if connected_components:
        if not HAVE_SCIPY:
            raise RuntimeError(
                "--connected-components requires scipy (pip install scipy --break-system-packages)"
            )
        out = np.zeros_like(mask, dtype=np.int32)
        next_id = 1
        for class_id in np.unique(mask):
            if class_id == 0:
                continue
            labeled, n = ndimage.label(mask == class_id)
            for comp_id in range(1, n + 1):
                out[labeled == comp_id] = next_id
                next_id += 1
        return out
    else:
        out = np.zeros_like(mask, dtype=np.int32)
        next_id = 1
        for val in np.unique(mask):
            if val == 0:
                continue
            out[mask == val] = next_id
            next_id += 1
        return out


def convert_split(images_dir: Path, labels_dir: Path, out_root: Path,
                   connected_components: bool, min_area: int):
    pairs = find_pairs(images_dir, labels_dir)
    if not pairs:
        print("  [warn] no matched image/label pairs found", file=sys.stderr)
        return []

    img_root = out_root / "images"
    ann_root = out_root / "masks"

    video_names = []
    skipped_empty = 0

    for stem, image_path, label_path in pairs:
        raw = load_label_as_indexed(label_path)
        remapped = remap_mask(raw, connected_components)

        if min_area > 1:
            for obj_id in np.unique(remapped):
                if obj_id == 0:
                    continue
                if (remapped == obj_id).sum() < min_area:
                    remapped[remapped == obj_id] = 0

        if remapped.max() == 0:
            skipped_empty += 1
            continue  # PalettisedPNGSegmentLoader needs at least one object per video

        n_objects = int(remapped.max())
        if n_objects > 255:
            print(f"  [warn] {stem}: {n_objects} objects exceeds palette-PNG limit of 255, "
                  f"clipping extra ids to 255", file=sys.stderr)
            remapped = np.clip(remapped, 0, 255)

        video_dir_img = img_root / stem
        video_dir_ann = ann_root / stem
        video_dir_img.mkdir(parents=True, exist_ok=True)
        video_dir_ann.mkdir(parents=True, exist_ok=True)

        Image.open(image_path).convert("RGB").save(video_dir_img / "00000.jpg", quality=95)

        mask_img = Image.fromarray(remapped.astype(np.uint8), mode="P")
        mask_img.putpalette(DAVIS_PALETTE)
        mask_img.save(video_dir_ann / "00000.png")

        video_names.append(stem)

    if skipped_empty:
        print(f"  [info] skipped {skipped_empty} frames with no foreground objects", file=sys.stderr)

    return video_names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                     help="Dataset root containing <split>/images and <split>/labels")
    ap.add_argument("--out", required=True, type=Path, help="Output directory for the SAM2 VOS-format dataset")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                     help="Split subfolders to convert (default: train). Each split gets its own "
                          "images/masks/train.txt under <out>/<split>/")
    ap.add_argument("--connected-components", action="store_true",
                     help="Split each label value into connected components (use if labels are "
                          "semantic/class masks rather than instance masks)")
    ap.add_argument("--min-area", type=int, default=1, help="Drop objects smaller than this many pixels")
    args = ap.parse_args()

    for split in args.splits:
        images_dir = args.root / split / "images"
        labels_dir = args.root / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            print(f"[error] expected {images_dir} and {labels_dir} to exist, skipping split '{split}'",
                  file=sys.stderr)
            continue

        out_split = args.out / split
        out_split.mkdir(parents=True, exist_ok=True)

        print(f"Converting split '{split}' ...")
        video_names = convert_split(images_dir, labels_dir, out_split, args.connected_components, args.min_area)

        list_path = out_split / "train.txt"
        with open(list_path, "w") as f:
            f.write("\n".join(video_names) + "\n")

        print(f"  wrote {len(video_names)} single-frame videos -> {out_split}")
        print(f"  img_folder = {out_split / 'images'}")
        print(f"  gt_folder  = {out_split / 'masks'}")
        print(f"  file_list_txt = {list_path}")

    print(f"\nDone. SAM2 VOS-format dataset written to: {args.out}")


if __name__ == "__main__":
    main()