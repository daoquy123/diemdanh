"""Smoke tests that don't require any pretrained weights or datasets.

Run with::

    pytest tests/

These tests instantiate every pure-PyTorch component on a single random tensor
to make sure shapes, configs and class hierarchies all line up.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.alignment import align_face
from src.recognition.backbones import iresnet18, MobileFaceNet
from src.recognition.losses import (
    ArcFaceHead,
    CosFaceHead,
    OnlineTripletLoss,
    batch_hard_triplet_loss,
)
from src.metrics import verification_metrics, identification_report


def test_iresnet18_forward_shape():
    model = iresnet18(embedding_dim=128).eval()
    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        feat = model(x)
    assert feat.shape == (2, 128)


def test_mobilefacenet_forward_shape():
    model = MobileFaceNet(embedding_dim=128).eval()
    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        feat = model(x)
    assert feat.shape == (2, 128)


def test_arcface_and_cosface_heads():
    head_a = ArcFaceHead(embedding_dim=64, num_classes=10)
    head_c = CosFaceHead(embedding_dim=64, num_classes=10)
    emb = torch.randn(8, 64)
    labels = torch.randint(0, 10, (8,))
    out_a = head_a(emb, labels)
    out_c = head_c(emb, labels)
    assert out_a.shape == (8, 10)
    assert out_c.shape == (8, 10)


def test_online_triplet_loss_runs():
    emb = torch.randn(12, 32, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    loss = OnlineTripletLoss(margin=0.2, mining="semi_hard")(emb, labels)
    loss2 = batch_hard_triplet_loss(emb, labels, margin=0.2)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(loss2)


def test_align_face_fallback():
    img = np.random.randint(0, 255, size=(200, 200, 3), dtype=np.uint8)
    crop = align_face(img, np.zeros((0, 2)))  # no landmarks → center crop
    assert crop.shape == (112, 112, 3)


def test_verification_metrics_synthetic():
    rng = np.random.default_rng(0)
    # genuine pairs cluster, impostor pairs random
    same = rng.normal(loc=0.0, scale=0.1, size=(40, 64))
    a = same / np.linalg.norm(same, axis=1, keepdims=True)
    b = (same + rng.normal(scale=0.05, size=same.shape))
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    diff_a = rng.normal(size=(40, 64));  diff_a /= np.linalg.norm(diff_a, axis=1, keepdims=True)
    diff_b = rng.normal(size=(40, 64));  diff_b /= np.linalg.norm(diff_b, axis=1, keepdims=True)
    e_a = np.concatenate([a, diff_a]).astype(np.float32)
    e_b = np.concatenate([b, diff_b]).astype(np.float32)
    labels = np.array([1] * 40 + [0] * 40)
    res = verification_metrics(e_a, e_b, labels, n_folds=4)
    assert 0.6 < res.accuracy <= 1.0
    assert 0.0 <= res.eer < 0.5


def test_identification_synthetic():
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(5, 16))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    g_emb = centers.copy()
    g_lbl = np.arange(5)
    q_emb = (centers + rng.normal(scale=0.05, size=centers.shape))
    q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_lbl = np.arange(5)
    res = identification_report(q_emb.astype(np.float32), q_lbl, g_emb.astype(np.float32), g_lbl)
    assert res.top1_accuracy >= 0.8
