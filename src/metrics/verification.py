"""Face verification metrics (1:1 matching).

Implements the LFW-style protocol metrics:

* Best accuracy across thresholds (LFW headline number)
* TAR @ FAR = 1e-3 / 1e-4
* ROC-AUC
* Equal Error Rate (EER)
* 10-fold cross-validated accuracy & threshold (canonical LFW protocol).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class VerificationResult:
    accuracy: float
    best_threshold: float
    tar_at_far_1e3: float
    tar_at_far_1e4: float
    roc_auc: float
    eer: float
    fold_accuracy_mean: float
    fold_accuracy_std: float


def compute_roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper around :func:`sklearn.metrics.roc_curve` returning ``(fpr, tpr, thr)``."""
    fpr, tpr, thr = roc_curve(labels, scores)
    return fpr, tpr, thr


def tar_at_far(scores: np.ndarray, labels: np.ndarray, far_target: float) -> float:
    """True-Accept Rate at the operating point with FAR <= ``far_target``."""
    fpr, tpr, _ = roc_curve(labels, scores)
    idxs = np.where(fpr <= far_target)[0]
    if len(idxs) == 0:
        return 0.0
    return float(tpr[idxs[-1]])


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def _best_threshold(scores: np.ndarray, labels: np.ndarray, candidates: np.ndarray | None = None) -> tuple[float, float]:
    """Return (best_accuracy, best_threshold) maximising accuracy."""
    if candidates is None:
        candidates = np.unique(np.concatenate([scores, [scores.min() - 1e-6, scores.max() + 1e-6]]))
    best_acc, best_thr = 0.0, 0.0
    for t in candidates:
        pred = (scores >= t).astype(np.int64)
        acc = float((pred == labels).mean())
        if acc > best_acc:
            best_acc, best_thr = acc, float(t)
    return best_acc, best_thr


def _kfold_cv_accuracy(scores: np.ndarray, labels: np.ndarray, n_folds: int = 10) -> tuple[float, float, float]:
    """LFW protocol: 10-fold CV — pick threshold on n-1 folds, score on the held-out fold."""
    n = len(scores)
    folds = np.array_split(np.arange(n), n_folds)
    accs: list[float] = []
    best_thr_overall = 0.0
    candidates = np.linspace(scores.min(), scores.max(), 200)
    for i in range(n_folds):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        _, thr = _best_threshold(scores[train], labels[train], candidates=candidates)
        pred = (scores[test] >= thr).astype(np.int64)
        accs.append(float((pred == labels[test]).mean()))
        best_thr_overall = thr
    return float(np.mean(accs)), float(np.std(accs)), best_thr_overall


def verification_metrics(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 10,
) -> VerificationResult:
    """Compute all verification metrics at once.

    Args:
        embeddings_a, embeddings_b: ``(N, D)`` L2-normalised pair embeddings.
        labels:                     ``(N,)`` 1 if same identity else 0.
    """
    if embeddings_a.shape != embeddings_b.shape:
        raise ValueError("embeddings_a and embeddings_b must have the same shape")
    # cosine similarity (already L2-normalised → just dot product)
    scores = (embeddings_a * embeddings_b).sum(axis=1)
    acc, thr = _best_threshold(scores, labels)
    fold_mean, fold_std, _ = _kfold_cv_accuracy(scores, labels, n_folds=n_folds)
    return VerificationResult(
        accuracy=acc,
        best_threshold=thr,
        tar_at_far_1e3=tar_at_far(scores, labels, 1e-3),
        tar_at_far_1e4=tar_at_far(scores, labels, 1e-4),
        roc_auc=float(roc_auc_score(labels, scores)),
        eer=compute_eer(scores, labels),
        fold_accuracy_mean=fold_mean,
        fold_accuracy_std=fold_std,
    )
