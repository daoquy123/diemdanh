from .verification import (
    verification_metrics,
    compute_eer,
    tar_at_far,
    compute_roc_curve,
)
from .identification import identification_report
from .viz import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_split_metrics_bars,
    plot_tsne_embeddings,
    plot_score_histogram,
)

__all__ = [
    "verification_metrics",
    "compute_eer",
    "tar_at_far",
    "compute_roc_curve",
    "identification_report",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_split_metrics_bars",
    "plot_tsne_embeddings",
    "plot_score_histogram",
]
