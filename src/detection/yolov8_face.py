"""YOLOv8-face detector via ultralytics.

YOLOv8-face checkpoints (``yolov8n-face.pt`` / ``yolov8s-face.pt`` / etc.)
predict 5 landmark keypoints in addition to the bbox, so we get alignment
keypoints "for free". When the chosen weights do **not** carry keypoints, we
fall back to RetinaFace / SCRFD just for the landmark step.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from .base import BaseDetector, DetectedFace

# Mirrors of the akanametov/yolov8-face release weights — small, well-known.
_YOLO_FACE_URLS = {
    "yolov8n-face.pt": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
    "yolov8s-face.pt": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8s-face.pt",
}


def _ensure_weights(weights: str | Path, weights_dir: Path) -> Path:
    """Download YOLOv8-face weights if missing. Accepts a name or a full path."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights = str(weights)
    if Path(weights).exists():
        return Path(weights)
    url = _YOLO_FACE_URLS.get(weights)
    if url is None:
        raise FileNotFoundError(
            f"Cannot find YOLOv8-face weights at '{weights}' and no known URL. "
            f"Provide a full path or use one of {list(_YOLO_FACE_URLS.keys())}."
        )
    target = weights_dir / weights
    if not target.exists():
        urllib.request.urlretrieve(url, str(target))
    return target


class YOLOv8FaceDetector(BaseDetector):
    """Adapter for the YOLOv8-face checkpoints (ultralytics runtime)."""

    def __init__(
        self,
        weights: str = "yolov8n-face.pt",
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.4,
        iou_thresh: float = 0.5,
        device: str = "cuda",
        weights_dir: Path | str = "weights/yolov8_face",
        landmark_fallback: bool = True,
    ):
        super().__init__(det_size=det_size, det_thresh=det_thresh)
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for YOLOv8FaceDetector. Install with: "
                "pip install ultralytics"
            ) from e

        weight_path = _ensure_weights(weights, Path(weights_dir))
        self.model = YOLO(str(weight_path))
        self.iou_thresh = iou_thresh
        self.device = device

        self._fallback_lm = None
        if landmark_fallback:
            try:
                from .retinaface import RetinaFaceDetector  # avoid cycle
                self._fallback_lm = RetinaFaceDetector(
                    model_pack="buffalo_s",
                    det_size=det_size,
                    det_thresh=det_thresh,
                    device=device,
                    allowed_modules=["detection"],  # avoid loading 4 extra models
                )
            except Exception:
                # If insightface isn't available, we can still detect — alignment
                # will fall back to a simple center-crop downstream.
                self._fallback_lm = None

    def detect(self, image_rgb: np.ndarray) -> list[DetectedFace]:
        results = self.model.predict(
            source=image_rgb,
            imgsz=self.det_size[0],
            conf=self.det_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))

        # Some YOLOv8-face weights expose keypoints; keep them when present.
        kps_all: np.ndarray | None = None
        if hasattr(res, "keypoints") and res.keypoints is not None and res.keypoints.xy is not None:
            kps_all = res.keypoints.xy.cpu().numpy()  # (N, 5, 2)

        faces: list[DetectedFace] = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if kps_all is not None and i < len(kps_all):
                landmarks = kps_all[i].astype(np.float32)
            else:
                landmarks = np.zeros((0, 2), dtype=np.float32)
            faces.append(DetectedFace(bbox=box.astype(np.float32), score=float(score), landmarks=landmarks))

        # If any face is missing landmarks, query the fallback detector for
        # those bboxes specifically (cheap when there are 1-3 faces).
        if self._fallback_lm is not None and any(not f.has_landmarks for f in faces):
            fallback = self._fallback_lm.detect(image_rgb)
            for face in faces:
                if face.has_landmarks:
                    continue
                best = self._best_iou(face.bbox, fallback)
                if best is not None and best.has_landmarks:
                    face.landmarks = best.landmarks

        return faces

    @staticmethod
    def _best_iou(bbox: np.ndarray, candidates: list[DetectedFace]) -> DetectedFace | None:
        if not candidates:
            return None
        x1, y1, x2, y2 = bbox
        a_area = max(0.0, (x2 - x1) * (y2 - y1))
        best, best_iou = None, 0.0
        for c in candidates:
            cx1, cy1, cx2, cy2 = c.bbox
            ix1, iy1 = max(x1, cx1), max(y1, cy1)
            ix2, iy2 = min(x2, cx2), min(y2, cy2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            c_area = max(0.0, (cx2 - cx1) * (cy2 - cy1))
            union = a_area + c_area - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best, best_iou = c, iou
        return best if best_iou > 0.3 else None
