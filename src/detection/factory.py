"""Detector factory — keeps training scripts / app code decoupled from
specific implementations.
"""
from __future__ import annotations

from omegaconf import DictConfig

from .base import BaseDetector


def build_detector(cfg: DictConfig | dict) -> BaseDetector:
    """Instantiate a detector from a config dict.

    Supported ``cfg.name`` values:

    * ``retinaface`` — insightface SCRFD with 5-point landmarks
    * ``yolov8_face`` — ultralytics YOLOv8-face (with optional landmark fallback)
    """
    name = str(cfg.get("name", "retinaface")).lower()
    device = cfg.get("device", "cuda")

    if name == "retinaface":
        from .retinaface import RetinaFaceDetector

        allowed_modules = cfg.get("allowed_modules", ["detection"])
        if allowed_modules is not None:
            allowed_modules = list(allowed_modules)

        return RetinaFaceDetector(
            model_pack=cfg.get("model_pack", "buffalo_s"),
            det_size=tuple(cfg.get("det_size", (320, 320))),
            det_thresh=float(cfg.get("det_thresh", 0.5)),
            device=device,
            allowed_modules=allowed_modules,
        )

    if name in {"yolov8_face", "yolov8-face", "yolov8"}:
        from .yolov8_face import YOLOv8FaceDetector

        return YOLOv8FaceDetector(
            weights=cfg.get("weights", "yolov8n-face.pt"),
            det_size=tuple(cfg.get("det_size", (640, 640))),
            det_thresh=float(cfg.get("det_thresh", 0.4)),
            iou_thresh=float(cfg.get("iou_thresh", 0.5)),
            device=device,
            landmark_fallback=str(cfg.get("landmark_fallback", "insightface")) != "none",
        )

    raise ValueError(f"Unknown detector: {name!r}")
