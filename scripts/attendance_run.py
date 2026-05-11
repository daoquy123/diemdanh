"""CLI runner: open a webcam, run the full pipeline, log attendance.

Press 'q' to quit. This script is convenient for live demos when you don't
want to launch the Streamlit app.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from src.pipeline import AttendancePipeline
from src.utils import get_logger

logger = get_logger()


def _annotate(frame_rgb: np.ndarray, results) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for r in results:
        x1, y1, x2, y2 = (int(v) for v in r.bbox)
        is_unknown = r.name == "Unknown"
        spoof_flag = r.is_real is False
        color = (0, 0, 255) if is_unknown or spoof_flag else (0, 200, 0)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)

        label = f"{r.name} {r.similarity:.2f}"
        if r.real_prob is not None:
            tag = "real" if r.is_real else "SPOOF"
            label += f" | {tag} {r.real_prob:.2f}"
        cv2.rectangle(bgr, (x1, y1 - 22), (x1 + 8 * len(label), y1), color, -1)
        cv2.putText(bgr, label, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return bgr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection", default="configs/detection/retinaface.yaml")
    parser.add_argument("--recognition", required=True)
    parser.add_argument("--rec-weights", default=None)
    parser.add_argument("--antispoof", default=None)
    parser.add_argument("--antispoof-weights", default=None)
    parser.add_argument("--gallery", default="embeddings_db")
    parser.add_argument("--db", default="attendance.db")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cooldown-min", type=int, default=5)
    args = parser.parse_args()

    pipeline = AttendancePipeline.from_configs(
        detection_cfg=args.detection,
        recognition_cfg=args.recognition,
        recognition_weights=args.rec_weights,
        antispoof_cfg=args.antispoof,
        antispoof_weights=args.antispoof_weights,
        gallery_root=args.gallery,
        db_path=args.db,
        threshold=args.threshold,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    last_log_t = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pipeline.recognize_image(rgb)

        # Only call DB at ~2 Hz to limit churn
        now = time.time()
        if now - last_log_t > 0.5:
            logged = pipeline.log_results(results, source="webcam", cooldown_minutes=args.cooldown_min)
            for name in logged:
                logger.info(f"Attendance ✓ {name}")
            last_log_t = now

        annotated = _annotate(rgb, results)
        cv2.imshow("Attendance — press q to quit", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
