"""Prepare datasets: download LFW, align faces, persist train/val/test splits.

Custom dataset default: **train : val : test = 6 : 2 : 2** per identity (see
``configs/data/custom.yaml``). After ``--dataset custom`` you also get
``data/splits/custom_pipeline.md`` (Mermaid diagram) + ``*_split_summary.json``.

Usage:
    python -m scripts.prepare_data --dataset lfw
    python -m scripts.prepare_data --dataset custom --root data/raw/custom
    python -m scripts.prepare_data --dataset casia --root data/raw/casia-webface
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.alignment import align_face
from src.data import download_lfw
from src.detection import build_detector
from src.utils import get_logger, load_config

logger = get_logger("logs/prepare_data.log")


def _iter_image_paths(root: Path):
    valid = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in class_dir.iterdir():
            if img.suffix.lower() in valid:
                yield class_dir.name, img


def align_dataset(
    src_root: Path,
    dst_root: Path,
    detector_cfg_path: str,
    image_size: int = 112,
) -> None:
    """Align every image in ``src_root`` and write to ``dst_root`` mirroring
    the per-identity layout. Faces with no detection are skipped.
    """
    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_root}")
    detector_cfg = load_config(detector_cfg_path)
    detector = build_detector(detector_cfg)

    n_total, n_aligned, n_skipped = 0, 0, 0
    items = list(_iter_image_paths(src_root))
    for identity, src_path in tqdm(items, desc=f"Aligning {src_root.name}"):
        n_total += 1
        out_dir = dst_root / identity
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / src_path.name
        if out_path.exists():
            n_aligned += 1
            continue
        bgr = cv2.imread(str(src_path))
        if bgr is None:
            n_skipped += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = detector.detect(rgb)
        if not faces:
            n_skipped += 1
            continue
        # Pick the highest-confidence face (single-face dataset assumption)
        best = max(faces, key=lambda f: f.score)
        crop = align_face(rgb, best.landmarks, output_size=image_size)
        cv2.imwrite(str(out_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        n_aligned += 1
    logger.success(
        f"Done {src_root.name}: total={n_total} aligned={n_aligned} skipped={n_skipped}"
    )


def make_splits(
    aligned_root: Path,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    min_per_id: int = 2,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Per-identity stratified split into train / val / test.

    Default ratios **train : val : test = 6 : 2 : 2** (``val_ratio=test_ratio=0.2``).

    For each identity, images are shuffled with ``seed``, then counts are
    ``n_test = round(n * test_ratio)``, ``n_val = round(n * val_ratio)``,
    ``n_train = n - n_val - n_test`` so that the three partitions sum to ``n``.
    """
    train_ratio = 1.0 - val_ratio - test_ratio
    if train_ratio <= 0:
        raise ValueError(f"val_ratio + test_ratio must be < 1, got {val_ratio=} {test_ratio=}")

    rng = np.random.default_rng(seed)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for class_dir in sorted(p for p in aligned_root.iterdir() if p.is_dir()):
        imgs = [p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}]
        if len(imgs) < min_per_id:
            continue
        rng.shuffle(imgs)
        n = len(imgs)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        n_train = n - n_val - n_test
        # Safety: keep at least one sample in train when n is tiny
        if n_train < 1:
            n_train = 1
            remainder = n - n_train
            n_val = remainder // 2
            n_test = remainder - n_val
        # Fix any rounding drift (should already sum to n)
        while n_train + n_val + n_test > n:
            if n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                n_train -= 1
        while n_train + n_val + n_test < n:
            n_train += 1

        train = imgs[:n_train]
        val = imgs[n_train : n_train + n_val]
        test = imgs[n_train + n_val :]
        for img in train:
            splits["train"].append(str(img.relative_to(aligned_root)).replace("\\", "/"))
        for img in val:
            splits["val"].append(str(img.relative_to(aligned_root)).replace("\\", "/"))
        for img in test:
            splits["test"].append(str(img.relative_to(aligned_root)).replace("\\", "/"))
    return splits


def _write_split_artifacts(
    dataset_name: str,
    splits: dict[str, list[str]],
    aligned_root: Path,
    val_ratio: float,
    test_ratio: float,
    train_ratio: float,
) -> None:
    """Persist a Mermaid diagram + JSON summary for reports / thesis."""
    split_dir = Path("data/splits")
    split_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": dataset_name,
        "aligned_root": str(aligned_root).replace("\\", "/"),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "counts": {k: len(v) for k, v in splits.items()},
    }
    (split_dir / f"{dataset_name}_split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        f"# Luồng dữ liệu — `{dataset_name}` (train : val : test = {train_ratio:.0%} : {val_ratio:.0%} : {test_ratio:.0%})",
        "",
        "## Sơ đồ pipeline",
        "",
        "```mermaid",
        "flowchart LR",
        "  RAW[(Ảnh thô / webcam<br/>data/raw/...)] --> ALIGN[Align mặt 112×112<br/>Detection + 5-point warp]",
        "  ALIGN --> PROC[(Đã xử lý<br/>data/processed/...)]",
        "  PROC --> TRAIN[Train 60%]",
        "  PROC --> VAL[Val 20%]",
        "  PROC --> TEST[Test 20%]",
        "  TRAIN --> FT[fine-tune / train model]",
        "  VAL --> METRIC[Đánh giá metric<br/>loss, acc, confusion]",
        "  TEST --> FINAL[Đánh giá cuối<br/>không tham gia train]",
        "```",
        "",
        "## Số lượng ảnh theo split",
        "",
        "| Split | Số ảnh |",
        "|-------|--------:|",
    ]
    for k in ("train", "val", "test"):
        md.append(f"| {k} | {len(splits.get(k, []))} |")
    md.append("")
    md.append(f"- File danh sách: `data/splits/{dataset_name}_splits.json`")
    md.append(f"- Tóm tắt JSON: `data/splits/{dataset_name}_split_summary.json`")
    (split_dir / f"{dataset_name}_pipeline.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["lfw", "casia", "custom"], required=True)
    parser.add_argument("--root", default=None, help="Override raw root (defaults from config)")
    parser.add_argument("--detector", default="configs/detection/retinaface.yaml")
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Val fraction per identity (default: from configs/data/<dataset>.yaml)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="Test fraction per identity (default: from configs/data/<dataset>.yaml)",
    )
    parser.add_argument("--skip-align", action="store_true")
    args = parser.parse_args()

    cfg = load_config(f"configs/data/{args.dataset}.yaml")
    raw_root = Path(args.root) if args.root else Path(cfg["root"])
    processed_root = Path(cfg["processed_root"])

    val_ratio = float(args.val_ratio) if args.val_ratio is not None else float(cfg.get("val_ratio", 0.2))
    test_ratio = float(args.test_ratio) if args.test_ratio is not None else float(cfg.get("test_ratio", 0.2))
    train_ratio = 1.0 - val_ratio - test_ratio
    if train_ratio <= 0:
        raise ValueError("val_ratio + test_ratio must be < 1")

    if args.dataset == "lfw":
        if cfg.get("auto_download", False) and not raw_root.exists():
            logger.info("Auto-downloading LFW…")
            download_lfw(raw_root)
        elif not raw_root.exists():
            raise FileNotFoundError(
                f"LFW not found at {raw_root}. Pass --root or set auto_download=true."
            )
    elif not raw_root.exists():
        raise FileNotFoundError(f"Dataset not found at {raw_root}.")

    if not args.skip_align:
        align_dataset(raw_root, processed_root, args.detector, image_size=args.image_size)

    if args.dataset != "lfw":
        splits = make_splits(
            processed_root,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            min_per_id=int(cfg.get("min_images_per_id", 2)),
        )
        out_path = Path(f"data/splits/{args.dataset}_splits.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2, ensure_ascii=False)
        logger.success(
            f"Splits: train={len(splits['train'])} val={len(splits['val'])} "
            f"test={len(splits['test'])} -> {out_path}"
        )
        _write_split_artifacts(args.dataset, splits, processed_root, val_ratio, test_ratio, train_ratio)
        logger.success(f"Diagram + summary -> data/splits/{args.dataset}_pipeline.md")


if __name__ == "__main__":
    main()
