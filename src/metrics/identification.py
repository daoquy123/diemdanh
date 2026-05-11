"""Identification (1:N) metrics.

Closed-set classification: compare each query embedding against a gallery of
enrolled identities and assign the nearest one. Compute Top-K accuracy, plus
classification report (precision/recall/F1 per identity).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class IdentificationResult:
    top1_accuracy: float
    top5_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion_matrix: np.ndarray
    classification_report: str


def _topk_accuracy(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """``scores``: (N_query, N_classes); ``labels``: (N_query,)"""
    topk = np.argsort(-scores, axis=1)[:, :k]
    return float(np.any(topk == labels[:, None], axis=1).mean())


def identification_report(
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    threshold: float | None = None,
) -> IdentificationResult:
    """Run closed-set identification.

    Args:
        query_embeddings:  (Nq, D) L2-normalised.
        query_labels:      (Nq,) integer label per query.
        gallery_embeddings:(Ng, D) — usually 1 prototype per identity, but can
                           be multiple (we max-pool similarity per identity).
        gallery_labels:    (Ng,) identity for each gallery embedding.
        threshold:         If given, predictions below this cosine similarity
                           are flagged "unknown" (label = -1).
    """
    sims = query_embeddings @ gallery_embeddings.T  # cosine similarity

    # Aggregate per-identity score (max pooling across gallery shots per id)
    unique_ids = np.unique(gallery_labels)
    per_id_scores = np.zeros((sims.shape[0], len(unique_ids)), dtype=np.float32)
    for j, gid in enumerate(unique_ids):
        per_id_scores[:, j] = sims[:, gallery_labels == gid].max(axis=1)

    label_to_col = {gid: j for j, gid in enumerate(unique_ids)}
    cols = np.array([label_to_col.get(int(l), -1) for l in query_labels])
    valid = cols >= 0  # queries whose label is in the gallery
    s_valid = per_id_scores[valid]
    cols_valid = cols[valid]

    top1 = _topk_accuracy(s_valid, cols_valid, k=1) if len(s_valid) else 0.0
    top5 = _topk_accuracy(s_valid, cols_valid, k=min(5, s_valid.shape[1])) if len(s_valid) else 0.0

    preds_col = np.argmax(per_id_scores, axis=1)
    preds_id = unique_ids[preds_col]
    if threshold is not None:
        max_score = per_id_scores.max(axis=1)
        preds_id = np.where(max_score >= threshold, preds_id, -1)

    # Restrict precision/recall/F1 to the supported labels
    label_set = list(unique_ids)
    if -1 in preds_id and -1 not in label_set:
        label_set = label_set + [-1]
    macro_p = precision_score(query_labels, preds_id, labels=label_set, average="macro", zero_division=0)
    macro_r = recall_score(query_labels, preds_id, labels=label_set, average="macro", zero_division=0)
    macro_f1 = f1_score(query_labels, preds_id, labels=label_set, average="macro", zero_division=0)
    cm = confusion_matrix(query_labels, preds_id, labels=label_set)
    rep = classification_report(query_labels, preds_id, labels=label_set, zero_division=0)

    return IdentificationResult(
        top1_accuracy=top1,
        top5_accuracy=top5,
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        confusion_matrix=cm,
        classification_report=rep,
    )
