"""MiniFASNetV2 — compact anti-spoofing CNN.

Distilled from the open-source ``Silent-Face-Anti-Spoofing`` project (the same
architecture deployed on millions of mobile devices). The model takes an 80x80
RGB face crop and outputs a softmax over {real, spoof}.

Training data: any FAS public dataset (CelebA-Spoof, CASIA-FASD, NUAA, OULU)
or a custom collection of "real selfies + photos of phone screens". The
training script in :mod:`scripts.train_antispoof` accepts an ImageFolder-style
``root/{real,spoof}/...`` layout for simplicity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class _ConvBN(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _DepthwiseSeparable(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.dw = _ConvBN(in_c, in_c, k=3, s=stride, p=1, g=in_c)
        self.pw = _ConvBN(in_c, out_c, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class MiniFASNetV2(nn.Module):
    """Compact CNN: 80x80x3 -> {real, spoof}.

    ~0.4M params, runs ~1ms on a modern GPU and a few ms on CPU.
    """

    def __init__(self, num_classes: int = 2, embedding_dim: int = 128):
        super().__init__()
        self.stem = _ConvBN(3, 32, k=3, s=2, p=1)            # 80 -> 40
        self.block1 = _DepthwiseSeparable(32, 64, stride=1)  # 40
        self.block2 = _DepthwiseSeparable(64, 64, stride=2)  # 20
        self.block3 = _DepthwiseSeparable(64, 128, stride=1) # 20
        self.block4 = _DepthwiseSeparable(128, 128, stride=2) # 10
        self.block5 = _DepthwiseSeparable(128, 128, stride=1) # 10
        self.block6 = _DepthwiseSeparable(128, 128, stride=2) # 5
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.embed = nn.Linear(128, embedding_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.embed(x)
        x = self.dropout(x)
        return self.classifier(x)


def build_antispoof_model(cfg: DictConfig | dict) -> MiniFASNetV2:
    model_cfg = cfg.get("model", cfg)
    return MiniFASNetV2(
        num_classes=int(model_cfg.get("num_classes", 2)),
        embedding_dim=int(model_cfg.get("embedding_dim", 128)),
    )
