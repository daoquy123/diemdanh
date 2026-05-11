"""Face alignment via 5-point similarity transform.

This is the standard ArcFace / InsightFace alignment used in nearly every
modern face recognition pipeline. Given the 5 landmarks (left eye, right eye,
nose, left mouth, right mouth) of a detected face, we compute the similarity
transform that maps them to canonical positions on a 112x112 canvas, and warp
the original image accordingly.

The canonical landmark template is from:
    Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage import transform as sk_tf

# Canonical 5-landmark template for 112x112 ArcFace input (RGB).
REFERENCE_LANDMARKS_112: np.ndarray = np.array(
    [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # left mouth corner
        [70.7299, 92.2041],   # right mouth corner
    ],
    dtype=np.float32,
)


def _scale_template(target_size: int) -> np.ndarray:
    """Scale the 112x112 template if a different output crop is requested."""
    return REFERENCE_LANDMARKS_112 * (target_size / 112.0)


def align_face(
    image_rgb: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 112,
) -> np.ndarray:
    """Warp ``image_rgb`` so that ``landmarks`` align with the canonical template.

    Args:
        image_rgb:   ``HxWx3`` uint8 RGB image (the original frame).
        landmarks:   ``(5, 2)`` numpy array of (x, y) landmark coordinates.
        output_size: side of the output square crop (default 112 for ArcFace).

    Returns:
        ``output_size x output_size x 3`` uint8 RGB aligned face crop.

    Falls back to a tight bbox center-crop if landmarks are missing.
    """
    if landmarks is None or landmarks.size < 10:
        return _center_crop_fallback(image_rgb, output_size)

    src = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)[:5]
    dst = _scale_template(output_size)

    tform = sk_tf.SimilarityTransform()
    if not tform.estimate(src, dst):
        return _center_crop_fallback(image_rgb, output_size)
    M = tform.params[0:2, :]

    aligned = cv2.warpAffine(
        image_rgb,
        M,
        (output_size, output_size),
        borderValue=0.0,
        flags=cv2.INTER_LINEAR,
    )
    return aligned


def _center_crop_fallback(image_rgb: np.ndarray, size: int) -> np.ndarray:
    """Fallback when landmarks are not available."""
    h, w = image_rgb.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = image_rgb[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
