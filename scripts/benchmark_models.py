"""Benchmark multiple recognition configs side by side and produce a markdown
results table — exactly the kind of comparison every supervisor wants to see.

Each entry pairs a config file with an optional weights checkpoint::

    python -m scripts.benchmark_models \\
        --models arcface_r50:weights/arcface_r50/best.pth \\
                 facenet:weights/facenet/best.pth \\
                 mobilefacenet:weights/mobilefacenet/best.pth
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import LFWPairs, build_eval_transform
from src.metrics import verification_metrics
from src.recognition import EmbeddingExtractor
from src.utils import get_logger, load_config, select_device

logger = get_logger("logs/benchmark_models.log")


@torch.no_grad()
def _eval_one(
    name: str,
    cfg_path: str,
    weights: str | None,
    lfw_root: Path,
    pairs_file: Path,
    device: torch.device,
) -> dict:
    cfg = load_config(cfg_path)
    eval_tf = build_eval_transform(
        image_size=int(cfg["data"]["image_size"]),
        mean=tuple(cfg["data"]["mean"]),
        std=tuple(cfg["data"]["std"]),
    )
    dataset = LFWPairs(root=lfw_root, pairs_file=pairs_file, transform=eval_tf)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    extractor = EmbeddingExtractor.from_config(cfg, weights=weights)

    e_a, e_b, labels = [], [], []
    fps_samples = []
    for img_a, img_b, lbl in tqdm(loader, desc=name):
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        t0 = time.perf_counter()
        fa = extractor.model(img_a, normalize=True).cpu().numpy()
        fb = extractor.model(img_b, normalize=True).cpu().numpy()
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        # Both forward passes embedded `2 * batch` faces in `dt` seconds:
        fps_samples.append((2 * img_a.size(0)) / max(dt, 1e-6))
        e_a.append(fa); e_b.append(fb); labels.append(lbl.numpy())

    e_a = np.concatenate(e_a); e_b = np.concatenate(e_b); labels = np.concatenate(labels)
    result = verification_metrics(e_a, e_b, labels, n_folds=10)
    return {
        "model": name,
        "config": cfg["name"],
        "accuracy": result.accuracy,
        "fold_acc_mean": result.fold_accuracy_mean,
        "fold_acc_std": result.fold_accuracy_std,
        "tar_at_far_1e3": result.tar_at_far_1e3,
        "tar_at_far_1e4": result.tar_at_far_1e4,
        "roc_auc": result.roc_auc,
        "eer": result.eer,
        "throughput_fps": float(np.mean(fps_samples)) if fps_samples else 0.0,
        "params_M": sum(p.numel() for p in extractor.model.parameters()) / 1e6,
    }


def _parse_pair(spec: str) -> tuple[str, Path, str | None]:
    """Parse ``name:path/to/weights`` or ``configfile.yaml`` (no weights)."""
    if ":" in spec:
        name, weights = spec.split(":", 1)
    else:
        name, weights = spec, None
    cfg_path = Path(f"configs/recognition/{name}.yaml")
    if not cfg_path.exists():
        cfg_path = Path(name)  # let user pass full path too
    return name, cfg_path, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "arcface_r50:weights/arcface_r50/best.pth",
            "facenet:weights/facenet/best.pth",
            "mobilefacenet:weights/mobilefacenet/best.pth",
        ],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="reports/benchmark_models")
    args = parser.parse_args()

    device = select_device(args.device)
    lfw_cfg = load_config("configs/data/lfw.yaml")
    lfw_root = Path(lfw_cfg["processed_root"]) if Path(lfw_cfg["processed_root"]).exists() else Path(lfw_cfg["root"])
    pairs_file = Path(lfw_cfg["pairs_file"])

    rows = []
    for spec in args.models:
        name, cfg_path, weights = _parse_pair(spec)
        if not cfg_path.exists():
            logger.warning(f"Skip {name}: config not found at {cfg_path}")
            continue
        logger.info(f"=== Benchmarking {name} ({cfg_path}) ===")
        try:
            row = _eval_one(name, str(cfg_path), weights, lfw_root, pairs_file, device)
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Benchmark failed for {name}: {e}")

    if not rows:
        logger.error("No models benchmarked.")
        return

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))

    md = ["# Recognition model benchmark on LFW", ""]
    md.append("| Model | Acc | 10-fold Acc | TAR@FAR=1e-3 | TAR@FAR=1e-4 | AUC | EER | FPS | Params (M) |")
    md.append("|------|-----|-------------|--------------|--------------|-----|-----|-----|-----------|")
    for r in rows:
        md.append(
            f"| {r['model']} | {r['accuracy']:.4f} | "
            f"{r['fold_acc_mean']:.4f} ± {r['fold_acc_std']:.4f} | "
            f"{r['tar_at_far_1e3']:.4f} | {r['tar_at_far_1e4']:.4f} | "
            f"{r['roc_auc']:.4f} | {r['eer']:.4f} | "
            f"{r['throughput_fps']:.1f} | {r['params_M']:.2f} |"
        )
    (out_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    logger.success(f"Done. Results in {out_dir}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
