#!/usr/bin/env python3
"""Visualize {split}/{images,labels}/{task}/{frame} without modifying data.
Dependencies: pip install numpy pillow scipy
Outputs: output/{split}/{task}/{frame}_layout.jpg and summary.csv.
Stored nonzero values are INSTANCE IDs, not class IDs. Disconnected regions
with the same ID keep the same color; no automatic instance splitting occurs.
"""
import argparse
import colorsys
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
DEFAULT_ROOT = '/data1/workspace/tien.nguyen/project/common/data/fbm/commercial/specialty_segmentation_dataset_for_sam2_20260916_sahi'


def load_ids(path):
    with Image.open(path) as im:
        mode = im.mode
        a = np.array(im)
    if a.ndim == 3:
        if a.shape[2] not in (3, 4) or not np.all(a[..., :3] == a[..., :1]):
            raise ValueError('RGB channels differ: color-coded labels need an explicit color-to-ID mapping')
        if a.shape[2] == 4 and not np.all(a[..., 3] == 255):
            raise ValueError('Non-opaque RGBA label: resolve alpha before interpreting IDs')
        a = a[..., 0]  # identical to L for RGB grayscale; preserves exact IDs
    if a.ndim != 2 or a.dtype.kind not in 'biu' or np.any(a < 0):
        raise ValueError('Expected a nonnegative integer 2D label map')
    return a, mode  # P retains palette indices; I/I;16 retain high IDs


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def color(v):
    return tuple(round(255*c) for c in colorsys.hsv_to_rgb((int(v)*0.61803398875)%1, .70, 1.0))


def badge(draw, x, y, text, f, width, height):
    box = draw.textbbox((0, 0), text, font=f)
    tw, th = box[2]-box[0]+10, box[3]-box[1]+8
    x = max(0, min(int(x)-tw//2, width-tw))
    y = max(0, min(int(y)-th//2, height-th))
    draw.rounded_rectangle((x,y,x+tw,y+th), radius=3, fill='black', outline='white')
    draw.text((x+5-box[0], y+4-box[1]), text, font=f, fill='white')


def process(job):
    split, relative, label_path, image_path, out_root, panel_width, alpha = job
    row = dict(split=split, label=label_path, image=image_path or '', status='error',
               instances='', ids='', areas='', components='', output='', error='')
    try:
        if not image_path:
            raise FileNotFoundError('No image with matching relative path/stem')
        labels, mode = load_ids(label_path)
        with Image.open(image_path) as im:
            original = im.convert('RGB')
        h,w = labels.shape
        if original.size != (w,h):
            raise ValueError(f'Image {original.size} != label {(w,h)}; refusing to resize misaligned data')
        values, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        palette = np.array([(0,0,0) if v == 0 else color(v) for v in values], dtype=np.uint8)
        rgb = palette[inverse.reshape(labels.shape)]
        pixels = np.array(original)
        foreground = labels != 0
        overlay = pixels.copy()
        overlay[foreground] = np.rint((1-alpha)*pixels[foreground]+alpha*rgb[foreground]).astype(np.uint8)
        pw = min(w, panel_width)
        ph = max(1, round(h*pw/w))
        panels = [original.resize((pw,ph), Image.Resampling.LANCZOS),
                  Image.fromarray(rgb).resize((pw,ph), Image.Resampling.NEAREST),
                  Image.fromarray(overlay).resize((pw,ph), Image.Resampling.LANCZOS)]
        draws = [ImageDraw.Draw(p) for p in panels]
        ids, areas, components = [], [], []
        # Place one label on the deepest interior pixel of every connected piece.
        for v, area in zip(values, counts):
            if v == 0:
                continue
            ids.append(int(v)); areas.append(int(area))
            cc, n = ndimage.label(labels == v, structure=np.ones((3,3),dtype=np.uint8))
            components.append(int(n))
            for i, slices in enumerate(ndimage.find_objects(cc), 1):
                if slices is None:
                    continue
                piece = cc[slices] == i
                distance = ndimage.distance_transform_edt(np.pad(piece,1))[1:-1,1:-1]
                yy,xx = np.unravel_index(distance.argmax(), distance.shape)
                x=(xx+slices[1].start+.5)*pw/w
                y=(yy+slices[0].start+.5)*ph/h
                for draw in draws[1:]:
                    badge(draw,x,y,str(int(v)),font(16),pw,ph)
        gap=16; margin=20; title_h=68; legend_cols=max(1,(3*pw+2*gap)//220)
        legend_rows=max(1,(len(ids)+legend_cols-1)//legend_cols)
        canvas=Image.new('RGB',(3*pw+2*gap+2*margin, ph+title_h+margin+50+28*legend_rows),'#eeeeee')
        d=ImageDraw.Draw(canvas)
        title=f'{split}/{relative} | {len(ids)} instance IDs | label mode {mode} | {w} x {h}'
        # Fit long task names without cutting off the heading.
        f=font(18)
        while d.textlength(title,font=f)>canvas.width-2*margin and getattr(f,'size',10)>10:
            f=font(f.size-1)
        d.text((margin,8),title,font=f,fill='black')
        for i,name in enumerate(['Original image','Instance IDs (0 = background)','Overlay + IDs']):
            x=margin+i*(pw+gap)
            d.text((x,42),name,font=font(17),fill='black')
            canvas.paste(panels[i],(x,title_h))
        y0=title_h+ph+14
        d.text((margin,y0),'Stored IDs preserved; repeated ID = same instance, even across disconnected regions.',font=font(13),fill='black')
        if not ids:
            d.text((margin,y0+28),'Background only (no nonzero IDs)',font=font(15),fill='black')
        for k,(v,area,n) in enumerate(zip(ids,areas,components)):
            x=margin+(k%legend_cols)*220; y=y0+28+(k//legend_cols)*28
            d.rectangle((x,y+3,x+14,y+17),fill=color(v))
            d.text((x+20,y),f'ID {v}: {area}px / {n} parts',font=font(13),fill='black')
        destination=Path(out_root)/split/Path(relative).parent/(Path(relative).stem+'_layout.jpg')
        destination.parent.mkdir(parents=True,exist_ok=True)
        canvas.save(destination,quality=95)
        row.update(status='ok',instances=len(ids),ids=';'.join(map(str,ids)),
                   areas=';'.join(map(str,areas)),components=';'.join(map(str,components)),output=str(destination))
    except Exception as exc:
        row['error']=f'{type(exc).__name__}: {exc}'
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',type=Path,default=Path(DEFAULT_ROOT))
    p.add_argument('--output-dir',type=Path,default=Path('dataset_instance_viz'))
    p.add_argument('--splits',nargs='+',default=['train','valid'])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--panel-width',type=int,default=900)
    p.add_argument('--alpha',type=float,default=.45)
    args=p.parse_args()
    if args.workers<1 or args.panel_width<200 or not 0<=args.alpha<=1:
        p.error('workers >= 1, panel-width >= 200, and alpha in [0,1] required')
    jobs=[]
    for split in args.splits:
        base=args.root; image_root=base/'images'/split; label_root=base/'masks'/split
        if not image_root.is_dir() or not label_root.is_dir():
            p.error(f'Missing images or labels directory in {base}')
        by_stem={}
        for path in sorted(image_root.rglob('*')):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                key=path.relative_to(image_root).with_suffix('').as_posix()
                by_stem.setdefault(key,[]).append(path)
        for label_path in sorted(label_root.rglob('*.png')):
            rel=label_path.relative_to(label_root)
            matches=by_stem.get(rel.with_suffix('').as_posix(),[])
            exact=image_root/rel
            image_path=exact if exact.is_file() else (matches[0] if len(matches)==1 else None)
            if len(matches)>1 and not exact.is_file():
                p.error(f'Ambiguous image match for {label_path}: {matches}')
            jobs.append((split,str(rel),str(label_path),str(image_path) if image_path else None,
                         str(args.output_dir),args.panel_width,args.alpha))
    if not jobs:
        p.error('No label PNGs found')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    executor=ProcessPoolExecutor(args.workers) if args.workers>1 else None
    rows=executor.map(process,jobs,chunksize=1) if executor else map(process,jobs)
    failed=0
    try:
        with (args.output_dir/'summary.csv').open('w',newline='') as f:
            writer=None
            for i,row in enumerate(rows,1):
                if writer is None:
                    writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                writer.writerow(row)
                if row['status']!='ok':
                    failed+=1;print(f"ERROR: {row['label']}: {row['error']}",flush=True)
                if i%25==0 or i==len(jobs):
                    print(f'{i}/{len(jobs)} processed; {failed} errors',flush=True)
    finally:
        if executor:
            executor.shutdown()
    print(f'Layouts and summary: {args.output_dir.resolve()}')
    raise SystemExit(1 if failed else 0)


if __name__=='__main__':
    main()
