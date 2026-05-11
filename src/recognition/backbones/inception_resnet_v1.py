"""Thin wrapper around facenet-pytorch's InceptionResnetV1.

Using ``facenet-pytorch`` lets us load high-quality pretrained weights
(``vggface2`` / ``casia-webface``) which we can then fine-tune in stage-2.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["InceptionResnetV1"]


class InceptionResnetV1(nn.Module):
    """Wrapper exposing the same ``forward`` contract as our other backbones."""

    def __init__(self, embedding_dim: int = 512, pretrained: str | None = "vggface2"):
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1 as _Inception  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "facenet-pytorch is required for FaceNet. Install with: "
                "pip install facenet-pytorch"
            ) from e

        self._inner = _Inception(pretrained=pretrained, classify=False)
        if embedding_dim != 512:
            self._project = nn.Linear(512, embedding_dim, bias=False)
            nn.init.normal_(self._project.weight, std=0.02)
        else:
            self._project = nn.Identity()
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # facenet-pytorch expects 160x160 input (already enforced by our config),
        # outputs an L2-normalized 512-D embedding.
        feat = self._inner(x)
        return self._project(feat)
