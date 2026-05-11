"""Wrapper to run MiniFASNetV2 on aligned face crops at inference."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..utils.device import select_device
from .model import MiniFASNetV2, build_antispoof_model


class AntiSpoofPredictor:
    def __init__(
        self,
        model: MiniFASNetV2,
        image_size: int = 80,
        device: str | torch.device = "cuda",
        threshold: float = 0.5,
    ):
        self.model = model.eval()
        self.image_size = image_size
        self.device = device if isinstance(device, torch.device) else select_device(device)
        self.threshold = threshold
        self.model.to(self.device)
        self._mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)

    @classmethod
    def from_config(cls, cfg, weights: str | Path | None = None) -> "AntiSpoofPredictor":
        model = build_antispoof_model(cfg)
        if weights is not None and Path(weights).exists():
            state = torch.load(weights, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
        return cls(
            model,
            image_size=int(cfg["data"]["image_size"]),
            device=cfg.get("device", "cuda"),
            threshold=float(cfg.get("inference", {}).get("spoof_threshold", 0.5)),
        )

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        if crop.shape[0] != self.image_size or crop.shape[1] != self.image_size:
            crop = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        arr = crop.astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return (tensor - self._mean) / self._std

    @torch.no_grad()
    def predict(self, crop: np.ndarray) -> tuple[bool, float]:
        """Return ``(is_real, real_prob)``."""
        logits = self.model(self._preprocess(crop))
        probs = F.softmax(logits, dim=1)[0]
        # Class 0 = real, class 1 = spoof (consistent with training script)
        real_prob = float(probs[0].cpu().item())
        return real_prob >= self.threshold, real_prob
