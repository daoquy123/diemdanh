"""Evaluate a recognition model on the LFW verification protocol.

Outputs a markdown summary + ROC plot + score-distribution histogram.

Example:
    python -m scripts.eval_lfw \\
        --config configs/recognition/arcface_r50.yaml \\
        --weights weights/arcface_r50/best.pth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import LFWPairs, build_eval_transform
from src.metrics import (
    compute_roc_curve,
    plot_roc_curve,
    plot_score_histogram,
    verification_metrics,
)
from src.recognition import EmbeddingExtractor
from src.utils import get_logger, load_config, select_device

logger = get_logger("logs/eval_lfw.log")


@torch.no_grad()
def _embed_pairs(extractor: EmbeddingExtractor, loader: DataLoader, device: torch.device):
    e_a, e_b, labs = [], [], []
    for img_a, img_b, labels in tqdm(loader, desc="LFW"):
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        fa = extractor.model(img_a, normalize=True).cpu().numpy()
        fb = extractor.model(img_b, normalize=True).cpu().numpy()
        e_a.append(fa)
        e_b.append(fb)
        labs.append(labels.numpy())
    return np.concatenate(e_a), np.concatenate(e_b), np.concatenate(labs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--lfw-data", default=None, help="Override LFW root (else from configs/data/lfw.yaml)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="reports/eval_lfw")
    args = parser.parse_args()

    cfg = load_config(args.config)
    lfw_cfg = load_config("configs/data/lfw.yaml")
    device = select_device(args.device)

    # Use the *processed* (aligned) LFW if it exists, fall back to raw + warn.
    aligned_root = Path(lfw_cfg["processed_root"])
    raw_root = Path(args.lfw_data) if args.lfw_data else Path(lfw_cfg["root"])
    if aligned_root.exists():
        logger.info(f"Using aligned LFW at {aligned_root}")
        root = aligned_root
    else:
        logger.warning(
            f"Aligned LFW not found ({aligned_root}). Using raw {raw_root} — "
            "metrics will be lower because faces aren't aligned. Run "
            "`python -m scripts.prepare_data --dataset lfw` first."
        )
        root = raw_root

    pairs_file = Path(lfw_cfg["pairs_file"])

    eval_tf = build_eval_transform(
        image_size=int(cfg["data"]["image_size"]),
        mean=tuple(cfg["data"]["mean"]),
        std=tuple(cfg["data"]["std"]),
    )
    dataset = LFWPairs(root=root, pairs_file=pairs_file, transform=eval_tf)
    logger.info(f"LFW pairs: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    extractor = EmbeddingExtractor.from_config(cfg, weights=args.weights)
    e_a, e_b, labels = _embed_pairs(extractor, loader, device)
    result = verification_metrics(e_a, e_b, labels, n_folds=10)
    scores = (e_a * e_b).sum(axis=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = compute_roc_curve(scores, labels)
    plot_roc_curve(fpr, tpr, result.roc_auc, out_dir / f"{cfg['name']}_roc.png", label=cfg["name"])
    plot_score_histogram(scores, labels, out_dir / f"{cfg['name']}_scores.png", threshold=result.best_threshold)

    logger.success(
        f"\nLFW results [{cfg['name']}]\n"
        f"  acc            = {result.accuracy:.4f} (thr={result.best_threshold:.4f})\n"
        f"  10-fold acc    = {result.fold_accuracy_mean:.4f} ± {result.fold_accuracy_std:.4f}\n"
        f"  TAR@FAR=1e-3   = {result.tar_at_far_1e3:.4f}\n"
        f"  TAR@FAR=1e-4   = {result.tar_at_far_1e4:.4f}\n"
        f"  ROC AUC        = {result.roc_auc:.4f}\n"
        f"  EER            = {result.eer:.4f}\n"
    )

    json_path = out_dir / f"{cfg['name']}_metrics.json"
    json_path.write_text(json.dumps(result.__dict__, indent=2))
    logger.info(f"Metrics saved to {json_path}")


if __name__ == "__main__":
    main()
