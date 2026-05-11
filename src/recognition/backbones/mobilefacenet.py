"""MobileFaceNet — efficient backbone for on-device face recognition.

Reference:
    Chen et al., "MobileFaceNets: Efficient CNNs for Accurate Real-Time Face
    Verification on Mobile Devices" (2018).

Architecture summary (for 112x112 input, 128-D embedding):
    Conv 3x3 s2  -> 64
    DW Conv 3x3  -> 64
    [Bottleneck block x 5  blocks of (in, out, stride, expansion)]
    Conv 1x1 -> 512
    DW Conv 7x7 (linear) -> 512
    Conv 1x1 (linear) -> embedding_dim
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["MobileFaceNet"]


class _ConvBlock(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        linear: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.Identity() if linear else nn.PReLU(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _Bottleneck(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int, expansion: int):
        super().__init__()
        hidden = in_c * expansion
        self.use_residual = stride == 1 and in_c == out_c
        self.expand = _ConvBlock(in_c, hidden, kernel=1)
        self.dw = _ConvBlock(hidden, hidden, kernel=3, stride=stride, padding=1, groups=hidden)
        self.project = _ConvBlock(hidden, out_c, kernel=1, linear=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.project(self.dw(self.expand(x)))
        return out + x if self.use_residual else out


class MobileFaceNet(nn.Module):
    """112x112 -> embedding_dim."""

    # (out_c, n_repeat, stride, expansion)
    _CONFIG: list[tuple[int, int, int, int]] = [
        (64, 5, 2, 2),
        (128, 1, 2, 4),
        (128, 6, 1, 2),
        (128, 1, 2, 4),
        (128, 2, 1, 2),
    ]

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.stem = _ConvBlock(3, 64, kernel=3, stride=2, padding=1)
        self.dw_stem = _ConvBlock(64, 64, kernel=3, stride=1, padding=1, groups=64)

        layers: list[nn.Module] = []
        in_c = 64
        for out_c, n, s, t in self._CONFIG:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(_Bottleneck(in_c, out_c, stride=stride, expansion=t))
                in_c = out_c
        self.bottlenecks = nn.Sequential(*layers)

        self.conv_last = _ConvBlock(in_c, 512, kernel=1)
        # Global depthwise: 7x7 covers the spatial map at this depth.
        self.gdc = _ConvBlock(512, 512, kernel=7, stride=1, padding=0, groups=512, linear=True)
        self.linear = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(embedding_dim)

        self._init_weights()
        self.embedding_dim = embedding_dim

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.dw_stem(x)
        x = self.bottlenecks(x)
        x = self.conv_last(x)
        x = self.gdc(x)
        x = self.linear(x)
        x = x.flatten(1)
        return self.bn(x)
