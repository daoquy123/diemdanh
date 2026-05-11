"""Lighting-robustness benchmark.

Synthesise multiple lighting conditions (low-light, over-exposure, side-light,
backlight) on LFW, evaluate verification accuracy under each, and produce a
table + bar chart of the *delta* vs the normal-light baseline.

Example:
    python -m scripts.benchmark_lighting \\
        --config configs/recognition/arcface_r50.yaml \\
        --weights weights/arcface_r50/best.pth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import LFWPairs, lighting_transform
from src.metrics import verification_metrics
from src.recognition import EmbeddingExtractor
from src.utils import get_logger, load_config, select_device

logger = get_logger("logs/benchmark_lighting.log")

CONDITIONS = ["normal", "low_light", "over_exposure", "side_light", "backlight"]


@torch.no_grad()
def _eval_condition(extractor: EmbeddingExtractor, lfw_root: Path, pairs_file: Path, condition: str, device: torch.device, image_size: int):
    transform = lighting_transform(condition, image_size=image_size)
    dataset = LFWPairs(root=lfw_root, pairs_file=pairs_file, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    e_a, e_b, lbls = [], [], []
    for ia, ib, lbl in tqdm(loader, desc=condition):
        ia, ib = ia.to(device), ib.to(device)
        fa = extractor.model(ia, normalize=True).cpu().numpy()
        fb = extractor.model(ib, normalize=True).cpu().numpy()
        e_a.append(fa); e_b.append(fb); lbls.append(lbl.numpy())
    e_a, e_b, lbls = np.concatenate(e_a), np.concatenate(e_b), np.concatenate(lbls)
    return verification_metrics(e_a, e_b, lbls, n_folds=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="reports/benchmark_lighting")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = select_device(args.device)
    lfw_cfg = load_config("configs/data/lfw.yaml")
    lfw_root = Path(lfw_cfg["processed_root"]) if Path(lfw_cfg["processed_root"]).exists() else Path(lfw_cfg["root"])
    pairs_file = Path(lfw_cfg["pairs_file"])

    extractor = EmbeddingExtractor.from_config(cfg, weights=args.weights)

    rows = []
    for cond in CONDITIONS:
        result = _eval_condition(
            extractor=extractor,
            lfw_root=lfw_root,
            pairs_file=pairs_file,
            condition=cond,
            device=device,
            image_size=int(cfg["data"]["image_size"]),
        )
        rows.append({
            "condition": cond,
            "accuracy": result.accuracy,
            "tar_at_far_1e3": result.tar_at_far_1e3,
            "roc_auc": result.roc_auc,
            "eer": result.eer,
        })

    df = pd.DataFrame(rows)
    baseline_acc = float(df.loc[df["condition"] == "normal", "accuracy"].iloc[0])
    df["delta_vs_baseline"] = df["accuracy"] - baseline_acc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{cfg['name']}_lighting.csv", index=False)
    (out_dir / f"{cfg['name']}_lighting.json").write_text(json.dumps(rows, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:blue" if c == "normal" else "tab:orange" for c in df["condition"]]
    ax.bar(df["condition"], df["accuracy"], color=colors)
    ax.axhline(baseline_acc, color="gray", linestyle="--", linewidth=1, label="baseline")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(max(0.0, baseline_acc - 0.2), min(1.0, baseline_acc + 0.05))
    ax.set_title(f"Lighting robustness — {cfg['name']}")
    for x, y in zip(df["condition"], df["accuracy"]):
        ax.text(x, y + 0.005, f"{y:.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig.savefig(out_dir / f"{cfg['name']}_lighting.png", dpi=150)
    plt.close(fig)

    md = [f"# Lighting robustness — {cfg['name']}", "", "| Condition | Accuracy | TAR@FAR=1e-3 | AUC | EER | Δ vs baseline |", "|-----------|----------|--------------|-----|-----|---------------|"]
    for r in rows:
        md.append(
            f"| {r['condition']} | {r['accuracy']:.4f} | {r['tar_at_far_1e3']:.4f} | "
            f"{r['roc_auc']:.4f} | {r['eer']:.4f} | "
            f"{r['accuracy'] - baseline_acc:+.4f} |"
        )
    (out_dir / f"{cfg['name']}_lighting.md").write_text("\n".join(md), encoding="utf-8")
    logger.success(f"Lighting benchmark done -> {out_dir}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
