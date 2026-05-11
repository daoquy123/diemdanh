"""Redraw report charts from an existing fine-tune run directory.

Usage:
    python -m scripts.redraw_report --run-dir weights/facenet/finetuned_custom
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.metrics import plot_split_metrics_bars


def _plot_history(history: dict, out_dir: Path, model_name: str = "model") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    epochs = history["epoch"]

    axes[0].plot(epochs, history["train_loss"], label="train_loss", linewidth=1.8)
    if any(v is not None for v in history.get("val_loss", [])):
        v_epochs = [e for e, v in zip(epochs, history["val_loss"]) if v is not None]
        v_loss = [v for v in history["val_loss"] if v is not None]
        axes[0].plot(v_epochs, v_loss, label="val_loss", linewidth=1.8)
    if "stage2" in history.get("stage", []):
        s2_start = history["stage"].index("stage2") + 1
        axes[0].axvline(s2_start - 0.5, ls="--", c="gray", lw=1, alpha=0.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"Loss ({model_name})")

    v_epochs = [e for e, v in zip(epochs, history.get("val_acc", [])) if v is not None]
    v_acc = [v for v in history.get("val_acc", []) if v is not None]
    axes[1].plot(v_epochs, v_acc, label="val_acc", linewidth=1.8)
    if any(a is not None for a in history.get("train_acc", [])):
        t_epochs = [e for e, v in zip(epochs, history["train_acc"]) if v is not None]
        t_acc = [v for v in history["train_acc"] if v is not None]
        axes[1].plot(t_epochs, t_acc, label="train_acc", linewidth=1.8)
    if "stage2" in history.get("stage", []):
        s2_start = history["stage"].index("stage2") + 1
        axes[1].axvline(s2_start - 0.5, ls="--", c="gray", lw=1, alpha=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper left")
    axes[1].set_title(f"Accuracy ({model_name})")

    plt.tight_layout()
    out_path = out_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Fine-tune output directory (contains history.json)")
    parser.add_argument("--model-name", default="facenet", help="Label shown in chart titles")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    history_path = run_dir / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")

    history = json.loads(history_path.read_text(encoding="utf-8"))
    out_curve = _plot_history(history, run_dir, model_name=args.model_name)
    print(f"Updated: {out_curve}")

    metrics_path = run_dir / "metrics_classification.json"
    if metrics_path.exists():
        metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
        plot_split_metrics_bars(metrics_data, run_dir / "metrics_f1_accuracy.png")
        print(f"Updated: {run_dir / 'metrics_f1_accuracy.png'}")
    else:
        print("Skip metrics_f1_accuracy.png (metrics_classification.json not found)")


if __name__ == "__main__":
    main()
