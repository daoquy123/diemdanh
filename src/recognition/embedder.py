"""High-level helper to extract embeddings from face crops at inference.

Wraps a ``FaceEmbedder`` plus the appropriate normalization, batching and
device handling used by the attendance pipeline and the benchmark scripts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch

from ..utils.device import select_device
from .factory import FaceEmbedder, build_recognition_model


class EmbeddingExtractor:
    """Run a recognition model on aligned face crops and return embeddings."""

    def __init__(
        self,
        model: FaceEmbedder,
        image_size: int = 112,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
        device: str | torch.device = "cuda",
        batch_size: int = 64,
    ):
        self.model = model.eval()
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.device = device if isinstance(device, torch.device) else select_device(device)
        self.batch_size = batch_size
        self.model.to(self.device)
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)

    @classmethod
    def from_config(cls, cfg, weights: str | Path | None = None) -> "EmbeddingExtractor":
        model = build_recognition_model(cfg)
        if weights is not None and Path(weights).exists():
            state = torch.load(weights, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
        return cls(
            model,
            image_size=int(cfg["data"]["image_size"]),
            mean=tuple(cfg["data"].get("mean", (0.5, 0.5, 0.5))),
            std=tuple(cfg["data"].get("std", (0.5, 0.5, 0.5))),
            device=cfg.get("device", "cuda"),
        )

    def _preprocess(self, crops: Iterable[np.ndarray]) -> torch.Tensor:
        out = []
        for img in crops:
            if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            arr = img.astype(np.float32) / 255.0
            out.append(arr.transpose(2, 0, 1))
        tensor = torch.from_numpy(np.stack(out)).to(self.device)
        return (tensor - self.mean) / self.std

    @torch.no_grad()
    def encode(self, crops: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """Embed a list of aligned face crops; returns ``(N, D)`` L2-normalised."""
        if isinstance(crops, np.ndarray):
            if crops.size == 0:
                return np.zeros((0, self.model.embedding_dim), dtype=np.float32)
            if crops.ndim == 4:
                crops = [crops[i] for i in range(crops.shape[0])]
            elif crops.ndim == 3:
                crops = [crops]
            else:
                raise ValueError(f"encode expects (N,H,W,C) or (H,W,C) array, got shape {crops.shape}")
        if len(crops) == 0:
            return np.zeros((0, self.model.embedding_dim), dtype=np.float32)
        feats = []
        for i in range(0, len(crops), self.batch_size):
            batch = self._preprocess(crops[i : i + self.batch_size])
            f = self.model(batch, normalize=True)
            feats.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(feats, axis=0)

    @torch.no_grad()
    def encode_one(self, crop: np.ndarray) -> np.ndarray:
        return self.encode([crop])[0]
