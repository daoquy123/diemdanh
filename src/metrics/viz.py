"""Plotting helpers used by the report-generation scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    out_path: str | Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> Path:
    out_path = _ensure_dir(out_path)
    if normalize:
        cm = cm.astype(np.float32)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sum, where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.45), max(5, len(class_names) * 0.4)))
    sns.heatmap(
        cm,
        annot=cm.shape[0] <= 25,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_value: float,
    out_path: str | Path,
    label: str = "ROC",
) -> Path:
    out_path = _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc_value:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_score_histogram(
    scores: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    threshold: float | None = None,
) -> Path:
    out_path = _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores[labels == 1], bins=50, alpha=0.6, label="Genuine (same)", color="tab:green")
    ax.hist(scores[labels == 0], bins=50, alpha=0.6, label="Impostor (diff)", color="tab:red")
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", label=f"thr={threshold:.3f}")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("# pairs")
    ax.set_title("Pair-wise similarity distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_split_metrics_bars(
    metrics_by_split: dict[str, dict[str, float]],
    out_path: str | Path,
    *,
    title: str = "Accuracy & F1 (prototype nearest-neighbour)",
) -> Path:
    """Bar chart for ``accuracy``, ``macro_f1``, ``weighted_f1`` per split (e.g. val / test).

    Each inner dict should have keys ``accuracy``, ``macro_f1``, ``weighted_f1`` in [0, 1].
    """
    out_path = _ensure_dir(out_path)
    keys_order = ("accuracy", "macro_f1", "weighted_f1")
    labels = ("Accuracy", "Macro F1", "Weighted F1")
    splits = [k for k in ("val", "test") if k in metrics_by_split and metrics_by_split[k]]
    if not splits:
        raise ValueError("metrics_by_split must contain at least 'val'")

    n_m = len(keys_order)
    x = np.arange(n_m, dtype=np.float32)
    width = min(0.35, 0.8 / max(1, len(splits)))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, split in enumerate(splits):
        m = metrics_by_split[split]
        vals = [float(m[k]) for k in keys_order]
        offset = width * (i - (len(splits) - 1) / 2)
        ax.bar(x + offset, vals, width, label=split)

    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    label_names: Sequence[str] | None = None,
    perplexity: float = 30.0,
    max_classes: int = 20,
) -> Path:
    out_path = _ensure_dir(out_path)
    # Down-sample classes for legibility
    unique = np.unique(labels)
    if len(unique) > max_classes:
        unique = unique[:max_classes]
        mask = np.isin(labels, unique)
        embeddings = embeddings[mask]
        labels = labels[mask]
    perplexity = float(min(perplexity, max(2.0, (len(embeddings) - 1) / 3)))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        metric="cosine",
    )
    points = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 7))
    palette = sns.color_palette("tab20", n_colors=len(unique))
    for color, lbl in zip(palette, unique):
        mask = labels == lbl
        name = label_names[int(lbl)] if label_names is not None and int(lbl) < len(label_names) else str(lbl)
        ax.scatter(points[mask, 0], points[mask, 1], s=18, alpha=0.75, color=color, label=name)
    ax.set_title(f"t-SNE of face embeddings ({len(unique)} identities)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
