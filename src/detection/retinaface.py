"""RetinaFace / SCRFD detector via the ``insightface`` runtime.

We use ``insightface.app.FaceAnalysis`` because:

* it bundles a pretrained SCRFD detector (RetinaFace family) **with 5-point
  landmarks**, which we need for alignment,
* it works on CPU via onnxruntime out of the box, and
* the same FaceAnalysis app exposes a pretrained ArcFace model — convenient
  baseline for ablations.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .base import BaseDetector, DetectedFace


class RetinaFaceDetector(BaseDetector):
    """Adapter over ``insightface.app.FaceAnalysis``."""

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        device: str = "cuda",
        providers: list[str] | None = None,
        allowed_modules: list[str] | None = None,
    ):
        super().__init__(det_size=det_size, det_thresh=det_thresh)
        # Lazy import: insightface pulls onnxruntime which is heavy.
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "insightface is required for RetinaFaceDetector. "
                "Install with: pip install insightface onnxruntime"
            ) from e

        if providers is None:
            providers = self._auto_providers(device)

        # IMPORTANT: by default insightface FaceAnalysis loads 5 ONNX models per
        # frame (detection + 2D/3D landmarks + gender-age + recognition). We
        # only need the detector (which already includes 5-point keypoints), so
        # we explicitly restrict loaded modules — this is a 4-5x speed-up on
        # CPU and avoids running a redundant ArcFace pass.
        if allowed_modules is None:
            allowed_modules = ["detection"]

        self.app: Any = FaceAnalysis(
            name=model_pack,
            providers=providers,
            allowed_modules=allowed_modules,
        )
        ctx_id = 0 if device.startswith("cuda") else -1
        self.app.prepare(ctx_id=ctx_id, det_size=tuple(det_size), det_thresh=det_thresh)
        self.model_pack = model_pack

    @staticmethod
    def _auto_providers(device: str) -> list[str]:
        if device.startswith("cuda"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def detect(self, image_rgb: np.ndarray) -> list[DetectedFace]:
        # insightface expects BGR
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        faces = self.app.get(bgr)
        out: list[DetectedFace] = []
        for f in faces:
            bbox = np.asarray(f.bbox, dtype=np.float32)  # (x1,y1,x2,y2)
            score = float(getattr(f, "det_score", 1.0))
            kps = getattr(f, "kps", None)
            landmarks = (
                np.asarray(kps, dtype=np.float32).reshape(-1, 2)
                if kps is not None
                else np.zeros((0, 2), dtype=np.float32)
            )
            out.append(DetectedFace(bbox=bbox, score=score, landmarks=landmarks))
        return out
