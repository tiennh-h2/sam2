#!/usr/bin/env python3
"""
Build a SAM2-style copy of a flat segmentation dataset in a NEW folder.
The source dataset is never modified.

Source: <root>/<split>/images/<name>.png   and  <root>/<split>/labels/<name>.png
Output: <out-root>/<split>/images/<name>/00000.jpg
        <out-root>/<split>/labels/<name>/00000.png

Images are converted to JPEG (SAM2's video loader only reads .jpg/.jpeg).
Masks stay PNG (optionally binarized to 0/1 with --binarize-masks).

Usage:
    python to_sam2_layout.py --dry-run
    python to_sam2_layout.py --workers 16
    python to_sam2_layout.py --binarize-masks          # 0/255 masks -> 0/1
    python to_sam2_layout.py --out-root /some/other/path
"""
import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # large tiles; disable decompression-bomb guard

DEFAULT_ROOT = (
    "/data1/workspace/tien.nguyen/project/common/data/fbm/commercial/"
    "specialty_segmentation_dataset_tile_filtered_dataset_20260916"
)


def process_one(job):
    """Convert/copy one file. job = (src, dst, role, opts). Returns error str or None."""
    src, dst, role, opts = job
    src, dst = Path(src), Path(dst)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if role == "image":
            if src.suffix.lower() in (".jpg", ".jpeg"):
                shutil.copy2(src, dst)
            else:
                with Image.open(src) as im:
                    im.convert("RGB").save(dst, "JPEG", quality=opts["quality"])
        else:  # mask
            if opts["binarize"]:
                with Image.open(src) as im:
                    if im.mode not in ("P", "L", "I", "I;16"):
                        im = im.convert("L")
                    arr = (np.array(im) > 0).astype(np.uint8)
                Image.fromarray(arr, mode="L").save(dst, "PNG")
            else:
                shutil.copy2(src, dst)
        return None
    except Exception as e:  # keep going, report at the end
        return f"{src}: {e}"


def collect_jobs(root, out_root, splits, image_sub, mask_sub, exts, frame_name,
                 quality, binarize):
    opts = {"quality": quality, "binarize": binarize}
    jobs, skipped = [], 0
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            print(f"[warn] missing split dir: {split_dir}")
            continue
        stems = {}
        for sub, role, out_ext in ((image_sub, "image", ".jpg"),
                                   (mask_sub, "mask", ".png")):
            d = split_dir / sub
            if not d.is_dir():
                print(f"[warn] missing dir: {d}")
                continue
            stems[role] = set()
            for f in sorted(d.iterdir()):
                if not f.is_file() or f.suffix.lower() not in exts:
                    continue
                stems[role].add(f.stem)
                dst = out_root / split / sub / f.stem / f"{frame_name}{out_ext}"
                if dst.exists():
                    skipped += 1
                    continue
                jobs.append((str(f), str(dst), role, opts))
        if len(stems) == 2:
            oi, om = stems["image"] - stems["mask"], stems["mask"] - stems["image"]
            if oi or om:
                print(f"[warn] {split}: {len(oi)} images without mask, "
                      f"{len(om)} masks without image")
                for n in sorted(oi)[:5]:
                    print(f"   image only: {n}")
                for n in sorted(om)[:5]:
                    print(f"   mask only:  {n}")
    return jobs, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT, help="source dataset root")
    ap.add_argument("--out-root", default=None,
                    help="output root (default: <root>_sam2)")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--image-subdir", default="images")
    ap.add_argument("--label-subdir", default="labels")
    ap.add_argument("--exts", nargs="+", default=[".png", ".jpg", ".jpeg"])
    ap.add_argument("--frame-name", default="00000")
    ap.add_argument("--jpg-quality", type=int, default=95)
    ap.add_argument("--binarize-masks", action="store_true",
                    help="write masks as 0/1 (any value >0 becomes 1)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root) if args.out_root else Path(str(root).rstrip("/") + "_sam2")
    if out_root.resolve() == root.resolve():
        raise SystemExit("out-root must differ from root")
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.exts}

    print(f"source : {root}\noutput : {out_root}\n")

    jobs, skipped = collect_jobs(root, out_root, args.splits, args.image_subdir,
                                 args.label_subdir, exts, args.frame_name,
                                 args.jpg_quality, args.binarize_masks)
    print(f"{len(jobs)} files to write, {skipped} already exist (skipped)")

    if args.dry_run:
        for src, dst, role, _ in jobs[:10]:
            print(f"[dry-run] {src} -> {dst}")
        if len(jobs) > 10:
            print(f"... and {len(jobs) - 10} more")
        return

    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, err in enumerate(ex.map(process_one, jobs, chunksize=16), 1):
            if err:
                errors.append(err)
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}")

    print(f"\nDone. written={len(jobs) - len(errors)} errors={len(errors)}")
    for e in errors[:20]:
        print(f"[error] {e}")


if __name__ == "__main__":
    main()