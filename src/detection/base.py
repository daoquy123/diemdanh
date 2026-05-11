"""Common interface for face detectors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class DetectedFace:
    """Single detected face.

    Attributes:
        bbox:      (x1, y1, x2, y2) in pixel coordinates of the original image.
        score:     detector confidence in [0, 1].
        landmarks: (5, 2) array of (x, y) — left eye, right eye, nose,
                   left mouth corner, right mouth corner. May be empty.
        crop:      aligned face crop (HxWx3 uint8 RGB) — populated downstream.
    """

    bbox: np.ndarray
    score: float
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    crop: np.ndarray | None = None

    @property
    def has_landmarks(self) -> bool:
        return self.landmarks is not None and self.landmarks.size >= 10


class BaseDetector:
    """Abstract face detector.

    Concrete detectors must implement :meth:`detect` returning a list of
    :class:`DetectedFace`.
    """

    def __init__(self, det_size: tuple[int, int] = (640, 640), det_thresh: float = 0.5):
        self.det_size = tuple(det_size)
        self.det_thresh = det_thresh

    def detect(self, image_rgb: np.ndarray) -> list[DetectedFace]:  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, image_rgb: np.ndarray) -> list[DetectedFace]:
        return self.detect(image_rgb)

    @staticmethod
    def filter_top_k(faces: Sequence[DetectedFace], k: int) -> list[DetectedFace]:
        return sorted(faces, key=lambda f: f.score, reverse=True)[:k]
