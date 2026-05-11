"""Factory + thin wrapper that wires backbone + (optional) margin head."""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .backbones import InceptionResnetV1, MobileFaceNet, iresnet18, iresnet50, iresnet100
from .losses import ArcFaceHead, CosFaceHead, OnlineTripletLoss


_BACKBONES = {
    "iresnet18": iresnet18,
    "iresnet50": iresnet50,
    "iresnet100": iresnet100,
    "mobilefacenet": MobileFaceNet,
    "inception_resnet_v1": InceptionResnetV1,
}


class FaceEmbedder(nn.Module):
    """Backbone + L2-normalize wrapper.

    During training, the *raw* (un-normalised) embedding is passed to ArcFace /
    CosFace heads which perform their own L2 normalisation internally.

    During inference (``eval()``), :meth:`forward` returns the L2-normalized
    embedding directly so cosine similarity is just a dot product.
    """

    def __init__(self, backbone: nn.Module, embedding_dim: int):
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, normalize: bool | None = None) -> torch.Tensor:
        feat = self.backbone(x)
        if normalize is None:
            normalize = not self.training
        return F.normalize(feat, p=2, dim=1) if normalize else feat


def _try_load_insightface_pretrained(model: nn.Module, name: str) -> bool:
    """Attempt to populate IResNet weights from a downloaded insightface model.

    Falls back silently if the user hasn't downloaded any.
    """
    try:
        from insightface.utils import face_align  # noqa: F401  ensure pkg present
    except ImportError:
        return False

    candidates = [
        Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx",
    ]
    if not any(c.exists() for c in candidates):
        return False
    # We don't try to translate ONNX weights here; instead we simply mark that
    # users with insightface installed can run inference via the
    # FaceAnalysis app (used by the embed step in pipeline). For training-from-
    # pretrained, recommend providing a .pth via cfg.model.pretrained.
    return False


def build_recognition_model(cfg: DictConfig | dict) -> FaceEmbedder:
    """Build a recognition model from the ``model`` section of a config."""
    model_cfg = cfg.get("model", cfg)
    backbone_name = str(model_cfg.get("backbone")).lower()
    embedding_dim = int(model_cfg.get("embedding_dim", 512))
    pretrained = model_cfg.get("pretrained", "scratch")

    if backbone_name not in _BACKBONES:
        raise ValueError(
            f"Unknown backbone: {backbone_name}. Available: {list(_BACKBONES.keys())}"
        )

    if backbone_name == "inception_resnet_v1":
        weight_tag = pretrained if pretrained in {"vggface2", "casia-webface"} else None
        backbone = InceptionResnetV1(embedding_dim=embedding_dim, pretrained=weight_tag)
    elif backbone_name == "mobilefacenet":
        backbone = MobileFaceNet(embedding_dim=embedding_dim)
    else:
        backbone = _BACKBONES[backbone_name](embedding_dim=embedding_dim)

    if isinstance(pretrained, str) and pretrained not in {"scratch", "vggface2", "casia-webface", "insightface"}:
        if os.path.exists(pretrained):
            state = torch.load(pretrained, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            print(f"Loaded weights from {pretrained}. Missing: {len(missing)} Unexpected: {len(unexpected)}")
        else:
            print(f"WARN: pretrained path not found ({pretrained}), training from scratch.")

    return FaceEmbedder(backbone, embedding_dim=embedding_dim)


def build_loss(cfg: DictConfig | dict, embedding_dim: int, num_classes: int):
    """Build the head/loss object based on the ``loss`` section of a config.

    Returns one of:
        - (ArcFaceHead | CosFaceHead, nn.CrossEntropyLoss)
        - OnlineTripletLoss  (no separate classifier head)
    """
    loss_cfg = cfg.get("loss", cfg)
    loss_type = str(loss_cfg.get("type", "arcface")).lower()
    if loss_type == "arcface":
        head = ArcFaceHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=float(loss_cfg.get("scale", 64.0)),
            margin=float(loss_cfg.get("margin", 0.5)),
            easy_margin=bool(loss_cfg.get("easy_margin", False)),
        )
        return head, nn.CrossEntropyLoss()
    if loss_type == "cosface":
        head = CosFaceHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=float(loss_cfg.get("scale", 64.0)),
            margin=float(loss_cfg.get("margin", 0.35)),
        )
        return head, nn.CrossEntropyLoss()
    if loss_type == "triplet":
        return None, OnlineTripletLoss(
            margin=float(loss_cfg.get("margin", 0.2)),
            mining=str(loss_cfg.get("mining", "semi_hard")),
        )
    raise ValueError(f"Unknown loss type: {loss_type}")
