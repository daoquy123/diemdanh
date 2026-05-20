"""Enroll a folder of student photos into the FAISS gallery.

Expects an ImageFolder layout (``root/<student_name>/<img>.jpg``). For each
image: detect → align → embed → add to FAISS index. After enrollment, also
generates a t-SNE visualization of the gallery so you can sanity-check that
identities cluster.

Example:
    python -m scripts.enroll_faces \\
        --detection configs/detection/retinaface.yaml \\
        --recognition configs/recognition/arcface_r50.yaml \\
        --weights weights/arcface_r50/finetuned_custom/best.pth \\
        --data data/raw/custom \\
        --gallery embeddings_db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.alignment import align_face
from src.db import FaissEmbeddingStore
from src.detection import build_detector
from src.metrics import plot_tsne_embeddings
from src.recognition import EmbeddingExtractor
from src.utils import get_logger, load_config

logger = get_logger("logs/enroll_faces.log")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection", default="configs/detection/retinaface.yaml")
    parser.add_argument("--recognition", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--data", required=True, help="ImageFolder root: <data>/<student_name>/*.jpg")
    parser.add_argument("--gallery", default="embeddings_db")
    parser.add_argument("--reset", action="store_true", help="Wipe the gallery before enrolling")
    parser.add_argument(
        "--with-tsne",
        action="store_true",
        help="Plot gallery t-SNE after enroll (slow; often hangs on Windows — gallery is already saved)",
    )
    args = parser.parse_args()

    det_cfg = load_config(args.detection)
    rec_cfg = load_config(args.recognition)
    detector = build_detector(det_cfg)
    embedder = EmbeddingExtractor.from_config(rec_cfg, weights=args.weights)

    store_path = Path(args.gallery)
    if args.reset and store_path.exists():
        for f in store_path.iterdir():
            f.unlink()
    gallery = FaissEmbeddingStore(
        embedding_dim=int(rec_cfg["model"]["embedding_dim"]),
        root=store_path,
    )

    align_size = int(det_cfg.get("align_size", 112))
    data_root = Path(args.data)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    enrolled_emb: list[np.ndarray] = []
    enrolled_names: list[str] = []

    for student_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        name = student_dir.name
        crops, paths = [], []
        for img_path in student_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            faces = detector.detect(rgb)
            if not faces:
                logger.warning(f"No face in {img_path}, skipping")
                continue
            best = max(faces, key=lambda f: f.score)
            crops.append(align_face(rgb, best.landmarks, output_size=align_size))
            paths.append(str(img_path))

        if not crops:
            logger.warning(f"Skipped {name}: no detectable faces")
            continue

        embeddings = embedder.encode(crops)
        gallery.add(embeddings, [name] * len(crops), paths)
        enrolled_emb.extend(list(embeddings))
        enrolled_names.extend([name] * len(crops))
        logger.info(f"Enrolled {name}: +{len(crops)} embeddings")

    gallery.save()
    logger.success(
        f"Gallery now contains {len(gallery)} embeddings across "
        f"{len(gallery.unique_identities)} identities → {store_path}"
    )

    if not args.with_tsne:
        if sys.platform == "win32":
            logger.info("t-SNE skipped (Windows). Gallery OK. Add --with-tsne only if you need the plot.")
        return

    try:
        if len(enrolled_emb) >= 4 and len(set(enrolled_names)) >= 2:
            unique = sorted(set(enrolled_names))
            id_to_lbl = {n: i for i, n in enumerate(unique)}
            labels_arr = np.array([id_to_lbl[n] for n in enrolled_names])
            embs_arr = np.stack(enrolled_emb).astype(np.float32)
            out_png = Path("reports/figures/gallery_tsne.png")
            out_png.parent.mkdir(parents=True, exist_ok=True)
            plot_tsne_embeddings(embs_arr, labels_arr, out_png, label_names=unique)
            logger.info(f"t-SNE saved to {out_png}")
    except KeyboardInterrupt:
        logger.warning("t-SNE interrupted — embeddings_db is already saved.")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"t-SNE skipped: {e}")


if __name__ == "__main__":
    main()
