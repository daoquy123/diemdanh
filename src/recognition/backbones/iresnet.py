"""Improved ResNet (IResNet) — the backbone used by ArcFace.

This is a faithful re-implementation of the architecture from
``insightface/recognition/arcface_torch`` (Apache 2.0). The key differences vs
torchvision ResNet are:

* BatchNorm before the first conv (per the IR-SE paper)
* PReLU instead of ReLU
* "BN -> Conv -> BN -> PReLU -> Conv -> BN" basic block
* The final spatial features are flattened and projected to ``embedding_dim``
  by ``BN -> Dropout -> FC -> BN``.

The shapes match insightface checkpoints, so we can load their pretrained
weights directly — useful for the "stage 1: freeze backbone, train head"
transfer-learning recipe.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["iresnet18", "iresnet50", "iresnet100", "IResNet"]


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return out


class IResNet(nn.Module):
    """Improved ResNet for face recognition (ArcFace backbone)."""

    def __init__(
        self,
        block: type[IBasicBlock],
        layers: list[int],
        embedding_dim: int = 512,
        dropout: float = 0.0,
        input_size: int = 112,
    ):
        super().__init__()
        if input_size != 112:
            raise ValueError("IResNet only supports 112x112 input.")
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True) if dropout > 0 else nn.Identity()
        # 112 -> conv stride 1 -> 112 -> 4 stride-2 blocks -> 7x7
        self.fc = nn.Linear(512 * block.expansion * 7 * 7, embedding_dim)
        self.features = nn.BatchNorm1d(embedding_dim, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False  # match insightface ckpt

        self._initialize_weights()
        self.embedding_dim = embedding_dim

    def _make_layer(self, block: type[IBasicBlock], planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        return x


def iresnet18(embedding_dim: int = 512, **kw) -> IResNet:
    return IResNet(IBasicBlock, [2, 2, 2, 2], embedding_dim=embedding_dim, **kw)


def iresnet50(embedding_dim: int = 512, **kw) -> IResNet:
    return IResNet(IBasicBlock, [3, 4, 14, 3], embedding_dim=embedding_dim, **kw)


def iresnet100(embedding_dim: int = 512, **kw) -> IResNet:
    return IResNet(IBasicBlock, [3, 13, 30, 3], embedding_dim=embedding_dim, **kw)
