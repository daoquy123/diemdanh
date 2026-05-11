"""Closed-set identification report on a custom test set.

Generates Top-1 / Top-5 accuracy, macro Precision/Recall/F1, confusion matrix
and t-SNE visualisation — exactly the metrics block your supervisor wants.

Example:
    python -m scripts.identification_report \\
        --recognition configs/recognition/arcface_r50.yaml \\
        --weights weights/arcface_r50/finetuned_custom/best.pth \\
        --gallery-data data/processed/custom \\
        --query-data   data/processed/custom_test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.alignment import align_face
from src.detection import build_detector
from src.metrics import (
    identification_report,
    plot_confusion_matrix,
    plot_tsne_embeddings,
)
from src.recognition import EmbeddingExtractor
from src.utils import get_logger, load_config

logger = get_logger("logs/identification_report.log")


def _scan(root: Path):
    valid = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for c in classes:
        for img in (root / c).iterdir():
            if img.suffix.lower() in valid:
                paths.append(str(img))
                labels.append(cls_to_idx[c])
    return paths, np.array(labels), classes


def _embed_set(extractor: EmbeddingExtractor, detector, paths: list[str], align_size: int) -> np.ndarray:
    crops, ok_idx = [], []
    for i, p in enumerate(tqdm(paths, desc="embed")):
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = detector.detect(rgb)
        if not faces:
            # If detector misses, treat the image itself as already aligned
            crops.append(cv2.resize(rgb, (align_size, align_size)))
            ok_idx.append(i)
            continue
        best = max(faces, key=lambda f: f.score)
        crops.append(align_face(rgb, best.landmarks, output_size=align_size))
        ok_idx.append(i)
    return extractor.encode(crops), np.array(ok_idx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection", default="configs/detection/retinaface.yaml")
    parser.add_argument("--recognition", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--gallery-data", required=True)
    parser.add_argument("--query-data", required=True)
    parser.add_argument("--threshold", type=float, default=None, help="If set, predictions below the threshold are flagged 'unknown'")
    parser.add_argument("--out-dir", default="reports/identification")
    args = parser.parse_args()

    det_cfg = load_config(args.detection)
    rec_cfg = load_config(args.recognition)
    detector = build_detector(det_cfg)
    extractor = EmbeddingExtractor.from_config(rec_cfg, weights=args.weights)
    align_size = int(det_cfg.get("align_size", 112))

    g_paths, g_labels, classes = _scan(Path(args.gallery_data))
    q_paths, q_labels, q_classes = _scan(Path(args.query_data))
    if classes != q_classes:
        # Allow a subset of classes in the query
        unknown_q = [c for c in q_classes if c not in classes]
        if unknown_q:
            logger.warning(f"Query classes not in gallery (will appear as out-of-set): {unknown_q}")

    g_emb, _ = _embed_set(extractor, detector, g_paths, align_size)
    q_emb, _ = _embed_set(extractor, detector, q_paths, align_size)

    result = identification_report(
        query_embeddings=q_emb,
        query_labels=np.array([classes.index(q_classes[l]) if q_classes[l] in classes else -1 for l in q_labels]),
        gallery_embeddings=g_emb,
        gallery_labels=g_labels,
        threshold=args.threshold,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = rec_cfg["name"]
    plot_confusion_matrix(
        result.confusion_matrix,
        class_names=classes + (["unknown"] if -1 in np.unique(np.array([classes.index(q_classes[l]) if q_classes[l] in classes else -1 for l in q_labels])) else []),
        out_path=out_dir / f"{name}_confusion.png",
        title=f"Confusion matrix — {name}",
    )
    plot_tsne_embeddings(g_emb, g_labels, out_dir / f"{name}_gallery_tsne.png", label_names=classes)

    summary = {
        "model": name,
        "top1": result.top1_accuracy,
        "top5": result.top5_accuracy,
        "macro_precision": result.macro_precision,
        "macro_recall": result.macro_recall,
        "macro_f1": result.macro_f1,
    }
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"{name}_classification_report.txt").write_text(result.classification_report)

    logger.success(
        f"\nIdentification report [{name}]\n"
        f"  Top-1     = {result.top1_accuracy:.4f}\n"
        f"  Top-5     = {result.top5_accuracy:.4f}\n"
        f"  Precision = {result.macro_precision:.4f}\n"
        f"  Recall    = {result.macro_recall:.4f}\n"
        f"  F1        = {result.macro_f1:.4f}\n"
    )


if __name__ == "__main__":
    main()
