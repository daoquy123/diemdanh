"""Gộp hai thư mục ImageFolder (train riêng / val riêng) rồi split 6:2:2 theo từng người.

Dùng khi bạn có ``train/<id>/*.jpg`` và ``val/<id>/*.jpg`` (không trùng id),
muốn hợp nhất thành một bộ và chia **mỗi identity** thành train/val/test 60%/20%/20%
để fine-tune và đánh giá test cuối (ảnh test không tham gia train).

Output:
  * ``--out`` — một root ImageFolder ``<out>/<id>/`` chứa mọi ảnh của id đó
  * ``data/splits/<name>_splits.json`` — đường dẫn tương đối từ ``--out`` (train/val/test)

Fine-tune ví dụ::

    python -m scripts.finetune_recognition \\
        --config configs/recognition/facenet.yaml \\
        --custom-data data/processed/merged_faces \\
        --splits-file data/splits/merged_faces_splits.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tqdm import tqdm

from scripts.prepare_data import _write_split_artifacts, make_splits
from src.utils import get_logger, load_config

logger = get_logger("logs/merge_train_val_split.log")

_VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _merge_roots(train_src: Path, val_src: Path, dst: Path, clean: bool) -> None:
    if not train_src.is_dir():
        raise FileNotFoundError(f"Train root not found: {train_src}")
    if not val_src.is_dir():
        raise FileNotFoundError(f"Val root not found: {val_src}")

    if clean and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    def copy_branch(src_root: Path, tag: str) -> None:
        class_dirs = sorted(p for p in src_root.iterdir() if p.is_dir())
        for class_dir in tqdm(class_dirs, desc=f"Merge ({tag})"):
            out_class = dst / class_dir.name
            out_class.mkdir(parents=True, exist_ok=True)
            for img in class_dir.iterdir():
                if img.suffix.lower() not in _VALID_EXT:
                    continue
                dest = out_class / img.name
                if dest.exists():
                    stem, suf = img.stem, img.suffix
                    k = 1
                    while dest.exists():
                        dest = out_class / f"{stem}_{tag}_{k}{suf}"
                        k += 1
                shutil.copy2(img, dest)

    copy_branch(train_src, "train")
    copy_branch(val_src, "val")
    logger.success(f"Merged ImageFolder -> {dst.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge train/val folders then 60/20/20 split per identity.")
    parser.add_argument("--train-src", type=Path, default=Path("train"), help="ImageFolder train branch")
    parser.add_argument("--val-src", type=Path, default=Path("val"), help="ImageFolder val branch")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Merged output root (default: from configs/data/merged_faces.yaml processed_root)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="merged_faces",
        help="Prefix for data/splits/<name>_splits.json",
    )
    parser.add_argument("--config", type=str, default="configs/data/merged_faces.yaml", help="YAML for ratios / min_images_per_id")
    parser.add_argument("--clean", action="store_true", help="Remove --out before merge")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = Path(args.out) if args.out else Path(cfg["processed_root"])

    val_ratio = float(cfg.get("val_ratio", 0.2))
    test_ratio = float(cfg.get("test_ratio", 0.2))
    train_ratio = 1.0 - val_ratio - test_ratio
    min_per_id = int(cfg.get("min_images_per_id", 3))

    _merge_roots(args.train_src.resolve(), args.val_src.resolve(), out_root.resolve(), clean=args.clean)

    splits = make_splits(
        out_root,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_per_id=min_per_id,
        seed=int(cfg.get("split_seed", 42)),
    )

    split_dir = Path("data/splits")
    split_dir.mkdir(parents=True, exist_ok=True)
    out_json = split_dir / f"{args.dataset_name}_splits.json"
    out_json.write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.success(
        f"Splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} -> {out_json}"
    )
    _write_split_artifacts(args.dataset_name, splits, out_root, val_ratio, test_ratio, train_ratio)
    logger.success(f"Diagram + summary -> data/splits/{args.dataset_name}_pipeline.md")


if __name__ == "__main__":
    main()
