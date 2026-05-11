"""Angular margin softmax heads: ArcFace and CosFace.

Both heads sit on top of an embedding extractor and turn it into a classifier
during training. At inference we drop the head and use the L2-normalized
embedding directly with cosine similarity.

References:
    - Deng et al., "ArcFace: Additive Angular Margin Loss" (CVPR 2019)
    - Wang et al., "CosFace: Large Margin Cosine Loss" (CVPR 2018)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """ArcFace logits: :math:`s \cdot \cos(\theta + m)` for the target class."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb_n = F.normalize(embeddings, p=2, dim=1)
        w_n = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(emb_n, w_n).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale


class CosFaceHead(nn.Module):
    """CosFace logits: :math:`s \cdot (\cos\theta - m)` for the target class."""

    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 64.0, margin: float = 0.35):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb_n = F.normalize(embeddings, p=2, dim=1)
        w_n = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(emb_n, w_n)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        logits = cosine - one_hot * self.margin
        return logits * self.scale
