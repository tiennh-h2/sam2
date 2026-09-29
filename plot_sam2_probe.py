"""Combine the saved candidates from sam2_grid_probe.py into one image.

Example:
    python plot_sam2_probe.py --probe-dir sam2_grid_probe/my_image.png/tile_0000_x0_y0

The probe directory must contain raw_point_predictions.json and the candidate
overlay PNGs. The script reads saved results; it does not load SAM2 or a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create_contact_sheet(probe_dir: Path, output: Path, panel_width: int = 1400):
    predictions = probe_dir / "raw_point_predictions.json"
    if not predictions.is_file():
        raise FileNotFoundError(f"Missing {predictions}")
    records = json.loads(predictions.read_text())
    if not records:
        raise ValueError(f"No points in {predictions}")

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 23) if Path(font_path).is_file() else ImageFont.load_default()
    header_height = 91
    margin = 12
    panels = []
    for record in records:
        candidates = sorted(record["candidates"], key=lambda c: c["candidate"])
        if not candidates:
            raise ValueError(f"No candidates for point {record['point_xy']}")
        best = max(candidates, key=lambda c: c["predicted_iou"])["candidate"]
        row = []
        for candidate in candidates:
            overlay_path = probe_dir / candidate["overlay"]
            if not overlay_path.is_file():
                raise FileNotFoundError(f"Missing {overlay_path}")
            with Image.open(overlay_path) as opened:
                image = opened.convert("RGB")
            height = round(image.height * panel_width / image.width)
            image = image.resize((panel_width, height), Image.Resampling.LANCZOS)
            panel = Image.new("RGB", (panel_width, height + header_height), "white")
            panel.paste(image, (0, header_height))
            x, y = record["point_xy"]
            label = f"Point ({x}, {y})  |  candidate {candidate['candidate']}"
            if candidate["candidate"] == best:
                label += "  [HIGHEST PREDICTED IOU]"
            details = (
                f"pred IoU: {candidate['predicted_iou']:.3f}  |  "
                f"area: {candidate['mask_area_pixels']:,} px  |  "
                f"object P: {record['object_probability']:.3f}"
            )
            draw = ImageDraw.Draw(panel)
            draw.text((18, 10), label, font=font, fill=(20, 20, 20))
            draw.text((18, 52), details, font=font, fill=(20, 20, 20))
            row.append(panel)
        panels.append(row)

    columns = max(map(len, panels))
    total_width = columns * panel_width + (columns + 1) * margin
    total_height = sum(max(panel.height for panel in row) for row in panels)
    total_height += (len(panels) + 1) * margin
    sheet = Image.new("RGB", (total_width, total_height), (230, 230, 230))
    top = margin
    for row in panels:
        row_height = max(panel.height for panel in row)
        for column, panel in enumerate(row):
            left = margin + column * (panel_width + margin)
            sheet.paste(panel, (left, top))
        top += row_height + margin

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(f"Saved {output} ({total_width}x{total_height})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path,
                        help="Folder with raw_point_predictions.json and overlay PNGs")
    parser.add_argument("--output", type=Path,
                        help="Output PNG; defaults to <probe-dir>/all_candidates.png")
    parser.add_argument("--panel-width", type=int, default=1400,
                        help="Width in pixels of each mask panel (default: 1400)")
    args = parser.parse_args()
    if args.panel_width < 400:
        parser.error("--panel-width must be at least 400 pixels")
    create_contact_sheet(args.probe_dir,
                         args.output or args.probe_dir / "all_candidates.png",
                         args.panel_width)


if __name__ == "__main__":
    main()
