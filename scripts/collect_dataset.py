"""Tiny helper to collect a custom student dataset from your webcam.

Captures N frames per identity, saving them under ``data/raw/custom/<name>/``.
The user's own face is detected and only frames with a single face above the
threshold are kept — saves tedious manual filtering.

Example:
    python -m scripts.collect_dataset --name quy
    python -m scripts.collect_dataset --name quy --n 120
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.detection import build_detector
from src.utils import get_logger, load_config

logger = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Identity (folder name)")
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of frames to keep per identity (default 100 ≈ 60%% train after 60/20/20 split)",
    )
    parser.add_argument("--out", default="data/raw/custom")
    parser.add_argument("--detector", default="configs/detection/retinaface.yaml")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--every", type=int, default=3, help="Save every N-th detected frame (helps diversity)")
    args = parser.parse_args()

    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector(load_config(args.detector))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    saved = 0
    frame_idx = 0
    last_faces = []  # cache so UI keeps drawing bbox between detections
    logger.info("Quay trái - phải - cúi - ngẩng - đeo kính / khẩu trang nhẹ. ESC để thoát sớm.")
    while saved < args.n:
        ok, frame = cap.read()
        if not ok:
            break
        # Run detection on every Nth frame only (CPU-friendly). Display + save
        # follow the same cadence so we never save without a fresh detection.
        run_detect = frame_idx % max(1, args.every) == 0
        if run_detect:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_faces = detector.detect(rgb)

        annotated = frame.copy()
        for f in last_faces:
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{saved}/{args.n}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collect - press ESC to stop", annotated)

        if run_detect and len(last_faces) == 1:
            path = out_dir / f"{args.name}_{saved:04d}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
        frame_idx += 1

        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.success(f"Saved {saved} images to {out_dir}")


if __name__ == "__main__":
    main()
