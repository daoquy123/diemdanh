"""Online triplet losses for FaceNet-style training.

Implements:

* :func:`batch_hard_triplet_loss`     — Hermans et al., "In Defense of the
  Triplet Loss for Person Re-ID" (2017). For each anchor, picks the hardest
  positive and hardest negative in the batch.
* :func:`batch_semi_hard_triplet_loss` — Schroff et al., "FaceNet" (2015).
  Picks the hardest negative that is still further than the chosen positive.

Inputs are expected to be **L2-normalized** embeddings of shape ``(B, D)`` and
labels of shape ``(B,)`` produced by a :class:`PKBatchSampler` (P identities x
K samples each).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_distance_sq(x: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance matrix for L2-normalised embeddings."""
    return torch.cdist(x, x, p=2.0).pow(2).clamp(min=0.0)


def _get_anchor_positive_mask(labels: torch.Tensor) -> torch.Tensor:
    indices_eq = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device).logical_not()
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    return indices_eq & labels_eq


def _get_anchor_negative_mask(labels: torch.Tensor) -> torch.Tensor:
    return labels.unsqueeze(0) != labels.unsqueeze(1)


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    embeddings = F.normalize(embeddings, p=2, dim=1)
    dist = _pairwise_distance_sq(embeddings)

    pos_mask = _get_anchor_positive_mask(labels).float()
    hardest_pos = (dist * pos_mask).max(dim=1).values

    neg_mask = _get_anchor_negative_mask(labels)
    max_dist = dist.max(dim=1, keepdim=True).values
    dist_neg = dist + max_dist * (~neg_mask).float()
    hardest_neg = dist_neg.min(dim=1).values

    return F.relu(hardest_pos - hardest_neg + margin).mean()


def batch_semi_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    embeddings = F.normalize(embeddings, p=2, dim=1)
    dist = _pairwise_distance_sq(embeddings)

    pos_mask = _get_anchor_positive_mask(labels)
    neg_mask = _get_anchor_negative_mask(labels)

    losses = []
    for i in range(embeddings.size(0)):
        if not pos_mask[i].any() or not neg_mask[i].any():
            continue
        d_ap = dist[i][pos_mask[i]].mean()  # average positive distance for stability
        # Semi-hard negatives: d_an > d_ap but d_an < d_ap + margin
        d_an_all = dist[i][neg_mask[i]]
        semi = d_an_all[(d_an_all > d_ap) & (d_an_all < d_ap + margin)]
        if semi.numel() == 0:
            # fall back to hardest negative
            d_an = d_an_all.min()
        else:
            d_an = semi.min()
        losses.append(F.relu(d_ap - d_an + margin))

    if not losses:
        return torch.zeros((), device=embeddings.device, requires_grad=True)
    return torch.stack(losses).mean()


class OnlineTripletLoss(nn.Module):
    """Module wrapper choosing between batch-hard / semi-hard mining."""

    def __init__(self, margin: float = 0.2, mining: str = "semi_hard"):
        super().__init__()
        if mining not in {"semi_hard", "hard"}:
            raise ValueError(f"Unknown triplet mining: {mining}")
        self.margin = margin
        self.mining = mining

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.mining == "hard":
            return batch_hard_triplet_loss(embeddings, labels, self.margin)
        return batch_semi_hard_triplet_loss(embeddings, labels, self.margin)
