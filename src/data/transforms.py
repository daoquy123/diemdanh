"""Albumentations-based train / eval / lighting transforms.

The training augmentations follow the recipe used in modern face papers
(ArcFace, CosFace) plus the "tăng khả năng tổng quát hóa" extras the user asked
for: brightness, blur, rotation, noise.
"""
from __future__ import annotations

from typing import Sequence

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


_DEFAULT_MEAN: Sequence[float] = (0.5, 0.5, 0.5)
_DEFAULT_STD: Sequence[float] = (0.5, 0.5, 0.5)


def _gauss_noise(var_limit: tuple[float, float], p: float) -> A.BasicTransform:
    """Albumentations 1.x used ``var_limit``; 2.x uses ``std_range`` (fraction of 255 for uint8)."""
    try:
        major = int(str(A.__version__).split(".", maxsplit=1)[0])
    except ValueError:
        major = 1
    if major >= 2:
        lo, hi = float(var_limit[0]), float(var_limit[1])
        std_lo = max(1e-4, (lo**0.5) / 255.0)
        std_hi = min(1.0, max(std_lo + 1e-4, (hi**0.5) / 255.0))
        return A.GaussNoise(std_range=(std_lo, std_hi), p=p)
    return A.GaussNoise(var_limit=var_limit, p=p)


def build_train_transform(
    image_size: int = 112,
    mean: Sequence[float] = _DEFAULT_MEAN,
    std: Sequence[float] = _DEFAULT_STD,
    *,
    horizontal_flip: float = 0.5,
    color_jitter: dict | None = None,
    blur: float = 0.1,
    rotation: int = 5,
    noise: float = 0.05,
) -> A.Compose:
    """Build the training augmentation pipeline.

    The defaults match what we use in face recognition literature: a mild
    geometric jitter, photometric jitter and occasional blur/noise so the
    embedding learns to be invariant to capture conditions.
    """
    color_jitter = color_jitter or {"brightness": 0.2, "contrast": 0.2, "saturation": 0.1}
    pipeline: list[A.BasicTransform] = [
        A.Resize(image_size, image_size, interpolation=1),
        A.HorizontalFlip(p=horizontal_flip),
        A.Affine(
            rotate=(-rotation, rotation),
            translate_percent=(0.0, 0.05),
            scale=(0.95, 1.05),
            p=0.5,
        ),
        A.ColorJitter(
            brightness=color_jitter.get("brightness", 0.2),
            contrast=color_jitter.get("contrast", 0.2),
            saturation=color_jitter.get("saturation", 0.1),
            hue=color_jitter.get("hue", 0.0),
            p=0.5,
        ),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=5),
            ],
            p=blur,
        ),
        _gauss_noise((5.0, 30.0), p=noise),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    return A.Compose(pipeline)


def build_eval_transform(
    image_size: int = 112,
    mean: Sequence[float] = _DEFAULT_MEAN,
    std: Sequence[float] = _DEFAULT_STD,
) -> A.Compose:
    """Deterministic eval pipeline: resize → normalize → tensor."""
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def lighting_transform(condition: str, image_size: int = 112) -> A.Compose:
    """Synthetic lighting condition transforms used by the lighting benchmark.

    Conditions:
        normal         — identity
        low_light      — gamma > 1 (darken) + slight blur (sensor noise)
        over_exposure  — gamma < 1 (brighten) + clip
        side_light     — directional brightness gradient (random LR)
        backlight      — vignette (dark border)
    """
    base = [A.Resize(image_size, image_size, interpolation=1)]
    if condition == "normal":
        ops = []
    elif condition == "low_light":
        ops = [
            A.RandomGamma(gamma_limit=(180, 260), p=1.0),
            _gauss_noise((10.0, 25.0), p=1.0),
        ]
    elif condition == "over_exposure":
        ops = [
            A.RandomGamma(gamma_limit=(40, 60), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(0.2, 0.4), contrast_limit=0.0, p=1.0),
        ]
    elif condition == "side_light":
        ops = [_SideLight(p=1.0)]
    elif condition == "backlight":
        ops = [_Vignette(p=1.0)]
    else:
        raise ValueError(f"Unknown lighting condition: {condition}")

    return A.Compose(
        base
        + ops
        + [A.Normalize(mean=_DEFAULT_MEAN, std=_DEFAULT_STD), ToTensorV2()]
    )


class _SideLight(A.ImageOnlyTransform):
    """Apply a horizontal brightness gradient to simulate side lighting."""

    def __init__(self, strength: float = 0.6, p: float = 1.0):
        super().__init__(p=p)
        self.strength = strength

    def apply(self, img: np.ndarray, **kwargs) -> np.ndarray:
        h, w = img.shape[:2]
        gradient = np.linspace(-self.strength, self.strength, w, dtype=np.float32)
        if np.random.rand() < 0.5:
            gradient = gradient[::-1]
        gradient = np.tile(gradient, (h, 1))
        if img.ndim == 3:
            gradient = np.stack([gradient] * img.shape[2], axis=-1)
        out = img.astype(np.float32) / 255.0
        out = np.clip(out + gradient * 0.5, 0.0, 1.0)
        return (out * 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("strength",)


class _Vignette(A.ImageOnlyTransform):
    """Radial darkening to simulate backlight conditions."""

    def __init__(self, strength: float = 0.7, p: float = 1.0):
        super().__init__(p=p)
        self.strength = strength

    def apply(self, img: np.ndarray, **kwargs) -> np.ndarray:
        h, w = img.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        dist /= dist.max()
        mask = 1.0 - self.strength * dist  # darker on edges
        if img.ndim == 3:
            mask = mask[..., None]
        out = img.astype(np.float32) * mask
        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("strength",)
