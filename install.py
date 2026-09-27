#!/usr/bin/env python3
"""Install into an existing SAM 2 checkout; validate all anchors before writes."""
import argparse
import ast
from pathlib import Path

MARKER = '# SAM2_TILE_LOCATION_V1'


def change(text, old, new, path):
    if text.count(old) != 1:
        raise RuntimeError(f'{path}: expected exactly one matching anchor; source differs. No files changed.')
    return text.replace(old, new, 1)


def install(repo, source):
    if not (repo / 'training/model/grid_negative_sam2.py').is_file():
        raise FileNotFoundError('Install your existing grid-negative training patch first')
    edits = {}
    specs = {
        'sam2/modeling/sam2_base.py': [
            ('        compile_image_encoder: bool = False,\n', '        compile_image_encoder: bool = False,\n        use_tile_location: bool = False,\n'),
            ('        self._build_sam_heads()\n', '        self._build_sam_heads()\n        from sam2.tile_location import TileLocationEncoder\n        self.tile_location_encoder = TileLocationEncoder(self.sam_prompt_embed_dim) if use_tile_location else None\n'),
            ('            masks=sam_mask_prompt,\n        )\n', '            masks=sam_mask_prompt,\n        )\n        from sam2.tile_location import append_tile_location\n        sparse_embeddings = append_tile_location(\n            self.tile_location_encoder, sparse_embeddings,\n            None if point_inputs is None else point_inputs.get("tile_location"),\n        )\n'),
        ],
        'training/utils/data_utils.py': [
            ('    dict_key: str\n', '    dict_key: str\n    tile_locations: Optional[torch.Tensor] = None  # [T,B,4], parent-ROI normalized bounds\n'),
        ],
        'sam2/sam2_image_predictor.py': [
            ('        img_idx: int = -1,\n', '        img_idx: int = -1,\n        tile_location=None,\n'),
            ('            masks=mask_input,\n        )\n', '            masks=mask_input,\n        )\n        from sam2.tile_location import append_tile_location\n        sparse_embeddings = append_tile_location(\n            self.model.tile_location_encoder, sparse_embeddings, tile_location\n        )\n'),
        ],
    }
    for rel, replacements in specs.items():
        path = repo / rel
        original = path.read_text()
        if MARKER in original:
            continue
        updated = original
        for old, new in replacements:
            updated = change(updated, old, new, rel)
        updated += '\n' + MARKER + '\n'
        ast.parse(updated)
        if path.with_suffix(path.suffix + '.before_tile_location').exists():
            raise RuntimeError(f'Backup already exists for {path}; inspect before reinstalling')
        edits[path] = (original, updated)
    for folder in ('sam2', 'training'):
        for src in (source / folder).rglob('*.py'):
            target = repo / src.relative_to(source)
            text = src.read_text()
            ast.parse(text)
            if target.exists() and target.read_text() != text:
                raise RuntimeError(f'Refusing to overwrite different extension file: {target}')
            if not target.exists():
                edits[target] = (None, text)
    for path, (original, updated) in edits.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if original is not None:
            path.with_suffix(path.suffix + '.before_tile_location').write_text(original)
        path.write_text(updated)
    print(f'Installed {len(edits)} files into {repo}. Existing grid-negative code preserved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    args = parser.parse_args()
    install(args.repo.resolve(), Path(__file__).resolve().parent)
