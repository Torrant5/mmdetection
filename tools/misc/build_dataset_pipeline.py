#!/usr/bin/env python3
"""
Build custom train/test splits from a COCO-style dataset and materialize
images via symlink/copy/move. Supports an optional 2-class reduction
(person, sports_ball).

Outputs under <dataset_root>/splits/<variant>/{train,test}:
  - instances_train.json / instances_test.json
  - image files materialized in each split dir (by symlink/copy/move)

Example (80 classes, images variant):
  python tools/misc/build_dataset_pipeline.py \
    data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200/instances_default.json \
    data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200 \
    --variant images --num-images 200 --train-ratio 0.8 --copy-mode symlink

Example (2 classes: person/sports_ball):
  python tools/misc/build_dataset_pipeline.py \
    data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200/instances_default.json \
    data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200 \
    --variant images --images-2class --num-images 200 --train-ratio 0.8 --copy-mode symlink
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def subset_first_n_images(data: dict, n: int | None) -> dict:
    if n is None or n < 0:
        return copy.deepcopy(data)
    out = copy.deepcopy(data)
    images = out.get('images', [])
    keep_images = images[:max(0, n)]
    kept_ids: Set[int] = {int(img['id']) for img in keep_images}
    out['images'] = copy.deepcopy(keep_images)
    out['annotations'] = [
        copy.deepcopy(a) for a in out.get('annotations', []) if int(a.get('image_id')) in kept_ids
    ]
    return out


def reduce_to_two_classes_person_ball(data: dict) -> dict:
    # Keep only COCO category ids {1: person, 33: sports ball}
    keep = {1, 33}
    out = copy.deepcopy(data)
    # Remap to contiguous ids {1, 2}
    id_map: Dict[int, int] = {1: 1, 33: 2}
    classes = [
        {'id': 1, 'name': 'person', 'supercategory': 'person'},
        {'id': 2, 'name': 'sports_ball', 'supercategory': 'sports'},
    ]
    out['categories'] = classes
    new_anns: List[dict] = []
    for ann in out.get('annotations', []):
        cid = int(ann.get('category_id'))
        if cid in keep:
            na = copy.deepcopy(ann)
            na['category_id'] = id_map[cid]
            new_anns.append(na)
    out['annotations'] = new_anns
    return out


def split_coco_annotations(data: dict, train_ratio: float = 0.8, seed: int = 42, shuffle: bool = True) -> tuple[dict, dict]:
    import random
    image_ids = [int(img['id']) for img in data.get('images', [])]
    if shuffle:
        random.seed(seed)
        random.shuffle(image_ids)
    split_idx = int(len(image_ids) * train_ratio)
    train_ids = set(image_ids[:split_idx])
    test_ids = set(image_ids[split_idx:])

    def subset(ids: Set[int]) -> dict:
        imgs = [img for img in data.get('images', []) if int(img['id']) in ids]
        anns = [ann for ann in data.get('annotations', []) if int(ann['image_id']) in ids]
        return {
            'licenses': data.get('licenses', []),
            'info': data.get('info', {}),
            'categories': data.get('categories', []),
            'images': imgs,
            'annotations': anns,
        }

    return subset(train_ids), subset(test_ids)


def write_split_annotations_and_materialize(
    train_data: dict,
    test_data: dict,
    *,
    source_dir: Path,
    output_dir: Path,
    mode: str = 'symlink',  # symlink|copy|move
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    def to_basename(d: dict) -> dict:
        d2 = copy.deepcopy(d)
        for img in d2.get('images', []):
            img['file_name'] = os.path.basename(img.get('file_name', ''))
        return d2

    save_json(to_basename(train_data), train_dir / 'instances_train.json')
    save_json(to_basename(test_data), test_dir / 'instances_test.json')

    def materialize(images: List[dict], dst_dir: Path) -> tuple[int, int]:
        ok = 0
        missing = 0
        for img in images:
            name = os.path.basename(img.get('file_name', ''))
            src = source_dir / name
            dst = dst_dir / name
            if not src.exists():
                print(f'WARN: missing {src}')
                missing += 1
                continue
            if os.path.lexists(dst):
                if not os.path.exists(dst):
                    try:
                        dst.unlink()
                    except Exception:
                        pass
                else:
                    ok += 1
                    continue
            if mode == 'symlink':
                try:
                    dst.symlink_to(src.resolve())
                except FileExistsError:
                    pass
            elif mode == 'copy':
                import shutil
                shutil.copy2(src, dst)
            elif mode == 'move':
                import shutil
                shutil.move(str(src), str(dst))
            else:
                raise ValueError(f'Unknown mode: {mode}')
            ok += 1
        return ok, missing

    ok_tr, miss_tr = materialize(train_data.get('images', []), train_dir)
    ok_te, miss_te = materialize(test_data.get('images', []), test_dir)
    print(f"Split '{output_dir.name}': train ok/missing = {ok_tr}/{miss_tr}, test ok/missing = {ok_te}/{miss_te}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build train/test splits and materialize images')
    p.add_argument('input_json', type=Path, help='Path to original instances JSON (e.g., instances_default.json)')
    p.add_argument('dataset_root', type=Path, help='Dataset root dir that contains images_mode0')
    p.add_argument('--num-images', type=int, default=200, help='Number of images to keep (default: 200)')
    p.add_argument('--variant', choices=['images'], default='images', help='Variant to build (default: images)')
    p.add_argument('--images-2class', action='store_true', help='Reduce to 2 classes: person, sports_ball')
    p.add_argument('--train-ratio', type=float, default=0.8, help='Train split ratio (default: 0.8)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for shuffling (default: 42)')
    p.add_argument('--no-shuffle', action='store_true', help='Disable shuffling before split')
    p.add_argument('--copy-mode', choices=['symlink', 'copy', 'move'], default='symlink', help='Materialization mode')
    p.add_argument('--images-source', type=str, default='images_mode0', help='Relative image source directory name')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.input_json)

    # Optional 2-class reduction
    if args.images_2class:
        data = reduce_to_two_classes_person_ball(data)

    # Optional subset of first N images
    data = subset_first_n_images(data, args.num_images)

    # Split
    train_data, test_data = split_coco_annotations(
        data,
        train_ratio=args.train_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    # Materialize
    variant_dirname = 'images2' if args.images_2class else 'images'
    output_dir = args.dataset_root / 'splits' / variant_dirname
    source_dir = args.dataset_root / args.images_source
    write_split_annotations_and_materialize(
        train_data,
        test_data,
        source_dir=source_dir,
        output_dir=output_dir,
        mode=args.copy_mode,
    )


if __name__ == '__main__':
    main()

