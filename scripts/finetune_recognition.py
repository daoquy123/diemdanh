"""Two-stage transfer-learning fine-tune on a small custom dataset.

This is the "research-grade" trainer: stratified train/val split, per-epoch
loss + validation accuracy tracking, before/after t-SNE comparison, confusion
matrix on the held-out val set and a markdown report you can paste into your
thesis verbatim.

Stages:
* Stage 1 — freeze backbone, train head only (warm-up the new classifier).
* Stage 2 — unfreeze the last N residual blocks, fine-tune with low LR.

Example:
    python -m scripts.finetune_recognition \\
        --config configs/recognition/facenet.yaml \\
        --custom-data data/processed/custom

Uses ``data/splits/custom_splits.json`` when present (60/20/20 from ``prepare_data``),
otherwise falls back to a random stratified val split via ``--val-ratio``.

Large ImageFolders: t-SNE uses a subsample (``--tsne-max-samples``, default 5000) so startup
is not millions of forwards; use ``--skip-tsne`` to skip plots entirely.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data import (
    build_eval_transform,
    build_train_dataset,
    build_train_transform,
)
from src.metrics import plot_confusion_matrix, plot_split_metrics_bars, plot_tsne_embeddings
from src.recognition import build_loss, build_recognition_model
from src.utils import (
    dump_config,
    get_logger,
    load_config,
    merge_overrides,
    select_device,
    set_seed,
)

logger = get_logger("logs/finetune_recognition.log")

# Minimum identities required for triplet loss to be meaningful. Below this we
# auto-switch to ArcFace because triplet mining needs intra-class diversity
# AND inter-class separation that 3 identities can't provide.
MIN_IDS_FOR_TRIPLET = 4


# ---------------------------------------------------------------- helpers
def _freeze(module: nn.Module, freeze: bool) -> None:
    for p in module.parameters():
        p.requires_grad = not freeze


def _unfreeze_last_n_blocks(model: nn.Module, n: int) -> None:
    """Unfreeze the last ``n`` "macro-blocks" of the backbone."""
    candidates: list[nn.Module] = []
    backbone = getattr(model, "backbone", model)
    inner = getattr(backbone, "_inner", None) or backbone

    if hasattr(inner, "layer4"):  # IResNet
        candidates = [inner.layer1, inner.layer2, inner.layer3, inner.layer4, inner.bn2, inner.fc]
    elif hasattr(inner, "bottlenecks"):  # MobileFaceNet
        bns = list(inner.bottlenecks)
        candidates = bns + [inner.conv_last, inner.gdc, inner.linear, inner.bn]
    else:  # facenet-pytorch InceptionResnetV1
        for name in ["last_bn", "last_linear", "block8", "mixed_7a", "repeat_3", "mixed_6a", "repeat_2"]:
            mod = getattr(inner, name, None)
            if mod is not None:
                candidates.append(mod)
        candidates = list(reversed(candidates))

    if not candidates:
        logger.warning("Could not locate backbone blocks; unfreezing everything.")
        for p in model.parameters():
            p.requires_grad = True
        return

    for blk in candidates[-n:]:
        for p in blk.parameters():
            p.requires_grad = True


def _build_optimizer(name: str, params, lr: float, wd: float):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def _stratified_split(labels: list[int], val_ratio: float, seed: int):
    """Per-identity stratified split: ``val_ratio`` of each class goes to val."""
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        by_class[lbl].append(idx)
    train_idx, val_idx = [], []
    for lbl, idxs in by_class.items():
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio))) if len(idxs) >= 2 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx


def _indices_from_split_json(dataset, root: Path, split_paths: list[str]) -> list[int]:
    """Map relative paths in ``custom_splits.json`` to indices in ``dataset``."""
    root = Path(root).resolve()
    rel_set = {Path(p).as_posix() for p in split_paths}
    idxs: list[int] = []
    for i, abs_p in enumerate(dataset.paths):
        rel = Path(abs_p).resolve().relative_to(root).as_posix()
        if rel in rel_set:
            idxs.append(i)
    if len(idxs) < len(rel_set):
        logger.warning(
            f"Split JSON lists {len(rel_set)} paths but only {len(idxs)} exist under {root}. "
            "Re-run `python -m scripts.prepare_data --dataset custom` after aligning."
        )
    return sorted(idxs)


def _indices_for_tsne_plot(dataset, *, max_samples: int, max_classes: int, seed: int) -> list[int]:
    """Pick indices for t-SNE plots: first ``max_classes`` label ids (matches ``plot_tsne_embeddings``),
    at most ``max_samples`` images total — avoids embedding the full dataset (100k+ forwards).
    """
    if max_samples <= 0:
        return []
    labels = np.asarray(dataset.labels)
    if labels.size == 0:
        return []
    n_cls = int(labels.max()) + 1
    use_c = min(max_classes, n_cls)
    rng = np.random.default_rng(seed)
    per = max(1, max_samples // use_c)
    picked: list[int] = []
    for c in range(use_c):
        idxs = np.flatnonzero(labels == c)
        if idxs.size == 0:
            continue
        if idxs.size <= per:
            chosen = idxs
        else:
            chosen = rng.choice(idxs, size=per, replace=False)
        picked.extend(int(x) for x in chosen.tolist())
        if len(picked) >= max_samples:
            return picked[:max_samples]
    return picked[:max_samples]


@torch.no_grad()
def _embed_subset(model: nn.Module, dataset, indices: list[int], device: torch.device, batch_size: int = 64):
    """Encode a Subset and return ``(embeddings, labels)``."""
    model.eval()
    sub = Subset(dataset, indices)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        f = model(imgs, normalize=True).cpu().numpy()
        feats.append(f)
        labels.append(lbls.numpy())
    return np.concatenate(feats), np.concatenate(labels)


@torch.no_grad()
def _validate(
    model: nn.Module,
    head: nn.Module | None,
    eval_loader: DataLoader,
    train_emb: np.ndarray,
    train_lbl: np.ndarray,
    device: torch.device,
) -> dict:
    """Compute val loss + cosine-NN classification accuracy on the val split."""
    model.eval()
    if head is not None:
        head.eval()
    feats, lbls = [], []
    losses = []
    ce = nn.CrossEntropyLoss()
    for imgs, ys in eval_loader:
        imgs = imgs.to(device); ys = ys.to(device)
        emb_raw = model(imgs, normalize=False)
        emb_norm = nn.functional.normalize(emb_raw, p=2, dim=1)
        feats.append(emb_norm.cpu().numpy()); lbls.append(ys.cpu().numpy())
        if head is not None:
            logits = head(emb_raw, ys)
            losses.append(ce(logits, ys).item())

    feats = np.concatenate(feats)
    lbls = np.concatenate(lbls)

    # Build per-identity prototypes (mean of training embeddings) once
    unique = np.unique(train_lbl)
    proto = np.stack([train_emb[train_lbl == u].mean(axis=0) for u in unique])
    proto /= (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-9)
    sims = feats @ proto.T  # (Nv, Ncls)
    preds = unique[np.argmax(sims, axis=1)]
    acc = float((preds == lbls).mean())

    return {
        "val_loss": float(np.mean(losses)) if losses else None,
        "val_acc": acc,
        "val_features": feats,
        "val_labels": lbls,
        "val_preds": preds,
    }


def _prototype_nn_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    """Prototype-NN accuracy on provided embeddings/labels."""
    unique = np.unique(labels)
    proto = np.stack([features[labels == u].mean(axis=0) for u in unique])
    proto /= (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-9)
    sims = features @ proto.T
    preds = unique[np.argmax(sims, axis=1)]
    return float((preds == labels).mean())


def _train_one_stage(
    stage_name: str,
    epochs: int,
    lr: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_indices: list[int],
    train_indices: list[int],
    model: nn.Module,
    head: nn.Module | None,
    criterion,
    device: torch.device,
    weight_decay: float,
    optimizer_name: str,
    history: dict,
    full_dataset,
):
    if epochs <= 0:
        logger.info(f"[{stage_name}] epochs={epochs} -> skip stage.")
        return

    train_params = [p for p in model.parameters() if p.requires_grad]
    if head is not None:
        train_params += [p for p in head.parameters() if p.requires_grad]
    if not train_params:
        logger.warning(f"[{stage_name}] no trainable parameters — skipping stage.")
        return

    optimizer = _build_optimizer(optimizer_name, train_params, lr=lr, wd=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    n_train_params = sum(p.numel() for p in train_params)
    logger.info(f"[{stage_name}] epochs={epochs} lr={lr} trainable_params={n_train_params:,}")

    for epoch in range(epochs):
        model.train()
        if head is not None:
            head.train()
        pbar = tqdm(train_loader, desc=f"[{stage_name}] epoch {epoch+1}/{epochs}")
        running, n_seen, train_correct = 0.0, 0, 0
        for batch in pbar:
            imgs, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            emb = model(imgs, normalize=False)
            if head is not None:
                logits = head(emb, labels)
                loss = criterion(logits, labels)
                train_correct += (logits.argmax(1) == labels).sum().item()
            else:
                loss = criterion(emb, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, max_norm=5.0)
            optimizer.step()
            running += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        scheduler.step()
        train_loss = running / max(1, n_seen)
        train_acc = train_correct / max(1, n_seen) if head is not None else None

        # Validation against prototypes (the "real" recognition metric)
        train_emb_now, train_lbl_now = _embed_subset(model, full_dataset, train_indices, device)
        if train_acc is None:
            # Triplet-loss path (no classifier head): use the same prototype-NN rule as val_acc.
            train_acc = _prototype_nn_accuracy(train_emb_now, train_lbl_now)
        val = _validate(model, head, val_loader, train_emb_now, train_lbl_now, device)
        history["epoch"].append(history["epoch"][-1] + 1 if history["epoch"] else 1)
        history["stage"].append(stage_name)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["val_loss"])
        history["val_acc"].append(val["val_acc"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        logger.info(
            f"[{stage_name}] epoch {epoch+1}: train_loss={train_loss:.4f} "
            + (f"train_acc={train_acc:.4f} " if train_acc is not None else "")
            + f"val_acc={val['val_acc']:.4f}"
        )

    # Stash final val for the report (preds + features for confusion / t-SNE)
    history["last_val_features"] = val["val_features"]
    history["last_val_labels"] = val["val_labels"]
    history["last_val_preds"] = val["val_preds"]


# ----------------------------------------------------------------- report
def _nn_split_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    """Accuracy + macro / weighted F1 for multi-class NN predictions (same as val_acc definition)."""
    from sklearn.metrics import accuracy_score, f1_score

    labels = list(range(n_classes))
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, average="weighted", labels=labels, zero_division=0)),
    }


def _plot_history(history: dict, out_dir: Path, model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    epochs = history["epoch"]

    axes[0].plot(epochs, history["train_loss"], label="train_loss", linewidth=1.8)
    if any(v is not None for v in history["val_loss"]):
        v_epochs = [e for e, v in zip(epochs, history["val_loss"]) if v is not None]
        v_loss = [v for v in history["val_loss"] if v is not None]
        axes[0].plot(v_epochs, v_loss, label="val_loss", linewidth=1.8)
    # Stage boundary marker
    if "stage2" in history["stage"]:
        s2_start = history["stage"].index("stage2") + 1
        axes[0].axvline(s2_start - 0.5, ls="--", c="gray", lw=1, alpha=0.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"Loss ({model_name})")

    v_epochs = [e for e, v in zip(epochs, history["val_acc"]) if v is not None]
    v_acc = [v for v in history["val_acc"] if v is not None]
    axes[1].plot(v_epochs, v_acc, label="val_acc", linewidth=1.8)
    if any(a is not None for a in history["train_acc"]):
        t_epochs = [e for e, v in zip(epochs, history["train_acc"]) if v is not None]
        t_acc = [v for v in history["train_acc"] if v is not None]
        axes[1].plot(t_epochs, t_acc, label="train_acc", linewidth=1.8)
    if "stage2" in history["stage"]:
        s2_start = history["stage"].index("stage2") + 1
        axes[1].axvline(s2_start - 0.5, ls="--", c="gray", lw=1, alpha=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper left")
    axes[1].set_title(f"Accuracy ({model_name})")
    plt.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def _write_report(
    cfg,
    history: dict,
    out_dir: Path,
    classes: list[str],
    final_train_acc: float | None,
    n_train: int,
    n_val: int,
    *,
    n_test: int = 0,
    test_acc: float | None = None,
    splits_file: str | None = None,
    val_nn_metrics: dict[str, float] | None = None,
    test_nn_metrics: dict[str, float] | None = None,
) -> None:
    md = [f"# Fine-tune report — {cfg['name']}", ""]
    md.append(f"- Model: `{cfg['name']}` ({cfg['model']['backbone']})")
    md.append(f"- Loss: `{cfg['loss']['type']}`")
    md.append(f"- Identities: {len(classes)} ({', '.join(classes)})")
    md.append(f"- Train images: {n_train} · Val images: {n_val}" + (f" · Test images: {n_test}" if n_test else ""))
    if splits_file:
        md.append(f"- Split file: `{splits_file}` (train/val/test fixed, not re-randomised each run)")
    md.append("")
    md.append("## Final metrics")
    md.append(f"- Final val accuracy: **{history['val_acc'][-1]:.4f}**")
    if test_acc is not None:
        md.append(f"- **Held-out test accuracy** (cosine NN vs train prototypes): **{test_acc:.4f}**")
    if final_train_acc is not None:
        md.append(f"- Final train accuracy: **{final_train_acc:.4f}**")
    md.append(f"- Best val accuracy: **{max(history['val_acc']):.4f}** (epoch {history['val_acc'].index(max(history['val_acc'])) + 1})")
    md.append("")
    md.append("## Training curves")
    md.append("![training curves](training_curves.png)")
    md.append("")
    md.append("## Embedding space (t-SNE)")
    md.append("Before fine-tune (pretrained backbone):")
    md.append("![tsne before](tsne_before.png)")
    md.append("")
    md.append("After fine-tune:")
    md.append("![tsne after](tsne_after.png)")
    md.append("")
    md.append("## Confusion matrix on validation set (after fine-tune)")
    md.append("![confusion](confusion_val.png)")
    md.append("")
    if val_nn_metrics:
        md.append("## Accuracy & F1 (validation, same NN rule as val_acc)")
        md.append("")
        md.append("| metric | value |")
        md.append("|--------|------:|")
        md.append(f"| accuracy | **{val_nn_metrics['accuracy']:.4f}** |")
        md.append(f"| macro F1 | **{val_nn_metrics['macro_f1']:.4f}** |")
        md.append(f"| weighted F1 | **{val_nn_metrics['weighted_f1']:.4f}** |")
        md.append("")
        md.append("![metrics bar chart](metrics_f1_accuracy.png)")
        md.append("")
    if test_acc is not None:
        md.append("## Confusion matrix on held-out test set")
        md.append("![confusion test](confusion_test.png)")
        md.append("")
    if test_nn_metrics:
        md.append("## Accuracy & F1 (held-out test)")
        md.append("")
        md.append("| metric | value |")
        md.append("|--------|------:|")
        md.append(f"| accuracy | **{test_nn_metrics['accuracy']:.4f}** |")
        md.append(f"| macro F1 | **{test_nn_metrics['macro_f1']:.4f}** |")
        md.append(f"| weighted F1 | **{test_nn_metrics['weighted_f1']:.4f}** |")
        md.append("")
    md.append("## Per-epoch metrics")
    md.append("| epoch | stage | train_loss | train_acc | val_loss | val_acc | lr |")
    md.append("|------:|:-----|----------:|---------:|--------:|--------:|--:|")
    for i in range(len(history["epoch"])):
        ta = f"{history['train_acc'][i]:.4f}" if history["train_acc"][i] is not None else "—"
        vl = f"{history['val_loss'][i]:.4f}" if history["val_loss"][i] is not None else "—"
        md.append(
            f"| {history['epoch'][i]} | {history['stage'][i]} | "
            f"{history['train_loss'][i]:.4f} | {ta} | {vl} | "
            f"{history['val_acc'][i]:.4f} | {history['lr'][i]:.2e} |"
        )
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")


# -------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--custom-data", required=True, help="ImageFolder root (e.g. data/processed/custom)")
    parser.add_argument("--pretrained", default=None, help="Path to .pth from stage-0 training (optional)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Only used when no --splits-file (random stratified val split).",
    )
    parser.add_argument(
        "--splits-file",
        type=str,
        default=None,
        help="JSON from prepare_data (train/val/test lists). Default: data/splits/custom_splits.json if it exists.",
    )
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Skip t-SNE plots (fastest). Default embeds a subsample only.",
    )
    parser.add_argument(
        "--tsne-max-samples",
        type=int,
        default=5000,
        help="Max images to forward for t-SNE before/after (default 5000). Set 0 with --skip-tsne or alone to skip.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers. Default: config train.num_workers (fallback 2).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = merge_overrides(load_config(args.config), args.override)
    device = select_device(args.device)
    logger.info(f"Device: {device}")
    custom_data = Path(args.custom_data)

    # Auto-switch triplet → ArcFace for tiny datasets ----------------------
    aug = cfg["data"].get("augment", {})
    eval_tf = build_eval_transform(
        image_size=int(cfg["data"]["image_size"]),
        mean=tuple(cfg["data"]["mean"]),
        std=tuple(cfg["data"]["std"]),
    )
    train_tf = build_train_transform(
        image_size=int(cfg["data"]["image_size"]),
        mean=tuple(cfg["data"]["mean"]),
        std=tuple(cfg["data"]["std"]),
        horizontal_flip=float(aug.get("horizontal_flip", 0.5)),
        color_jitter=dict(aug.get("color_jitter", {})),
        blur=float(aug.get("blur", 0.1)),
        rotation=int(aug.get("rotation", 5)),
    )
    train_set = build_train_dataset(custom_data, transform=train_tf, min_images_per_id=2)
    eval_set = build_train_dataset(custom_data, transform=eval_tf, min_images_per_id=2)
    n_classes = train_set.num_classes
    logger.info(f"Custom set: {len(train_set)} images / {n_classes} identities ({train_set.classes})")

    if cfg["loss"]["type"].lower() == "triplet" and n_classes < MIN_IDS_FOR_TRIPLET:
        logger.warning(
            f"Triplet loss requires >= {MIN_IDS_FOR_TRIPLET} identities; got {n_classes}. "
            "Auto-switching to ArcFace (margin=0.5, scale=32) for fine-tune."
        )
        cfg["loss"] = {"type": "arcface", "margin": 0.5, "scale": 32.0, "easy_margin": False}

    # Train / val / (optional test) indices ---------------------------------
    splits_path = Path(args.splits_file) if args.splits_file else Path("data/splits/custom_splits.json")
    test_idx: list[int] = []
    used_splits_file: str | None = None
    if splits_path.exists():
        split_data = json.loads(splits_path.read_text(encoding="utf-8"))
        train_idx = _indices_from_split_json(train_set, custom_data, split_data.get("train", []))
        val_idx = _indices_from_split_json(eval_set, custom_data, split_data.get("val", []))
        test_idx = _indices_from_split_json(eval_set, custom_data, split_data.get("test", []))
        used_splits_file = str(splits_path).replace("\\", "/")
        logger.info(
            f"Loaded splits from {used_splits_file}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )
    else:
        train_idx, val_idx = _stratified_split(train_set.labels, args.val_ratio, args.seed)
        logger.info(
            f"No split file at {splits_path}; random stratified split val_ratio={args.val_ratio}: "
            f"train={len(train_idx)} val={len(val_idx)} (no held-out test)"
        )

    if not train_idx:
        raise RuntimeError("Train split is empty — check splits JSON or dataset path.")
    if not val_idx:
        raise RuntimeError("Val split is empty — need at least one val image per identity.")

    finetune_cfg = cfg["finetune"]
    bs = int(finetune_cfg.get("batch_size", 32))
    bs = min(bs, max(8, len(train_idx) // 2))  # avoid bs > #train
    num_workers = int(args.num_workers) if args.num_workers is not None else int(cfg["train"].get("num_workers", 2))
    num_workers = max(0, num_workers)
    train_loader = DataLoader(
        Subset(train_set, train_idx), batch_size=bs, shuffle=True, num_workers=num_workers,
        drop_last=len(train_idx) > bs, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(eval_set, val_idx), batch_size=min(bs, len(val_idx)),
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = None
    if test_idx:
        test_loader = DataLoader(
            Subset(eval_set, test_idx), batch_size=min(bs, len(test_idx)),
            shuffle=False, num_workers=num_workers, pin_memory=True,
        )

    # Build model + (optional) load stage-0 weights -----------------------
    model = build_recognition_model(cfg).to(device)
    if args.pretrained and Path(args.pretrained).exists():
        state = torch.load(args.pretrained, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        miss, unx = model.load_state_dict(state, strict=False)
        logger.info(f"Loaded pretrained {args.pretrained} (missing={len(miss)}, unexpected={len(unx)})")

    head, criterion = build_loss(cfg, embedding_dim=model.embedding_dim, num_classes=n_classes)
    if head is not None:
        head = head.to(device)

    # Output dir ----------------------------------------------------------
    out_dir = Path(cfg["train"]["ckpt_dir"]) / "finetuned_custom"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "config.yaml")

    # t-SNE BEFORE fine-tune (subsample — full set is too slow on large corpora)
    _emb_bs = min(256, max(64, len(eval_set) // 2000 + 64)) if device.type == "cuda" else 64
    if not args.skip_tsne and args.tsne_max_samples > 0:
        try:
            idx_tsne = _indices_for_tsne_plot(
                eval_set, max_samples=args.tsne_max_samples, max_classes=20, seed=args.seed
            )
            logger.info(f"t-SNE (before): embedding {len(idx_tsne)} / {len(eval_set)} images (batch_size={_emb_bs})")
            emb_before, lbl_before = _embed_subset(model, eval_set, idx_tsne, device, batch_size=_emb_bs)
            plot_tsne_embeddings(
                emb_before, lbl_before, out_dir / "tsne_before.png",
                label_names=train_set.classes, max_classes=20,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"t-SNE before skipped: {e}")
    else:
        logger.info("t-SNE before: skipped (--skip-tsne or --tsne-max-samples 0)")

    history = {
        "epoch": [], "stage": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "lr": [],
    }

    # Stage 1 -------------------------------------------------------------
    s1 = finetune_cfg["stage1"]
    _freeze(model, freeze=bool(s1.get("freeze_backbone", True)))
    _train_one_stage(
        stage_name="stage1",
        epochs=int(s1.get("epochs", 5)),
        lr=float(s1.get("lr", 1e-3)),
        train_loader=train_loader, val_loader=val_loader,
        train_indices=train_idx, val_indices=val_idx,
        model=model, head=head, criterion=criterion,
        device=device,
        weight_decay=float(finetune_cfg.get("weight_decay", 5e-4)),
        optimizer_name=str(cfg["train"].get("optimizer", "adam")),
        history=history, full_dataset=eval_set,
    )

    # Stage 2 -------------------------------------------------------------
    s2 = finetune_cfg["stage2"]
    if s2.get("freeze_backbone", False):
        _freeze(model, freeze=True)
    else:
        _freeze(model, freeze=False)
        _unfreeze_last_n_blocks(model, int(s2.get("unfreeze_last_n_blocks", 2)))
    _train_one_stage(
        stage_name="stage2",
        epochs=int(s2.get("epochs", 20)),
        lr=float(s2.get("lr", 1e-4)),
        train_loader=train_loader, val_loader=val_loader,
        train_indices=train_idx, val_indices=val_idx,
        model=model, head=head, criterion=criterion,
        device=device,
        weight_decay=float(finetune_cfg.get("weight_decay", 5e-4)),
        optimizer_name=str(cfg["train"].get("optimizer", "adam")),
        history=history, full_dataset=eval_set,
    )

    # Save fine-tuned weights --------------------------------------------
    torch.save(
        {
            "state_dict": model.state_dict(),
            "head": head.state_dict() if head is not None else None,
            "classes": train_set.classes,
            "history": {k: v for k, v in history.items() if k not in {"last_val_features", "last_val_labels", "last_val_preds"}},
        },
        out_dir / "best.pth",
    )

    # t-SNE AFTER + confusion + curves + report --------------------------
    if not args.skip_tsne and args.tsne_max_samples > 0:
        try:
            idx_tsne = _indices_for_tsne_plot(
                eval_set, max_samples=args.tsne_max_samples, max_classes=20, seed=args.seed
            )
            logger.info(f"t-SNE (after): embedding {len(idx_tsne)} / {len(eval_set)} images")
            emb_after, lbl_after = _embed_subset(model, eval_set, idx_tsne, device, batch_size=_emb_bs)
            plot_tsne_embeddings(
                emb_after, lbl_after, out_dir / "tsne_after.png",
                label_names=train_set.classes, max_classes=20,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"t-SNE after skipped: {e}")
    else:
        logger.info("t-SNE after: skipped")

    val_nn_metrics: dict[str, float] | None = None
    test_nn_metrics: dict[str, float] | None = None
    if "last_val_features" in history:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(history["last_val_labels"], history["last_val_preds"], labels=list(range(n_classes)))
        plot_confusion_matrix(cm, train_set.classes, out_dir / "confusion_val.png", title=f"Val confusion — {cfg['name']}")
        val_nn_metrics = _nn_split_metrics(history["last_val_labels"], history["last_val_preds"], n_classes)
        logger.info(
            f"Val (final epoch): acc={val_nn_metrics['accuracy']:.4f} "
            f"macro_f1={val_nn_metrics['macro_f1']:.4f} weighted_f1={val_nn_metrics['weighted_f1']:.4f}"
        )

    test_acc: float | None = None
    if test_loader is not None and test_idx:
        train_emb_final, train_lbl_final = _embed_subset(model, eval_set, train_idx, device)
        te = _validate(model, head, test_loader, train_emb_final, train_lbl_final, device)
        test_acc = float(te["val_acc"])
        logger.success(f"Held-out TEST accuracy (never seen during training): {test_acc:.4f}")
        from sklearn.metrics import confusion_matrix

        cm_t = confusion_matrix(te["val_labels"], te["val_preds"], labels=list(range(n_classes)))
        plot_confusion_matrix(cm_t, train_set.classes, out_dir / "confusion_test.png", title=f"Test confusion — {cfg['name']}")
        test_nn_metrics = _nn_split_metrics(te["val_labels"], te["val_preds"], n_classes)
        logger.info(
            f"Test: acc={test_nn_metrics['accuracy']:.4f} "
            f"macro_f1={test_nn_metrics['macro_f1']:.4f} weighted_f1={test_nn_metrics['weighted_f1']:.4f}"
        )

    if val_nn_metrics:
        try:
            bar_splits: dict[str, dict[str, float]] = {"val": val_nn_metrics}
            if test_nn_metrics:
                bar_splits["test"] = test_nn_metrics
            plot_split_metrics_bars(bar_splits, out_dir / "metrics_f1_accuracy.png")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"metrics_f1_accuracy.png skipped: {e}")
        (out_dir / "metrics_classification.json").write_text(
            json.dumps({"val": val_nn_metrics, "test": test_nn_metrics}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    _plot_history(history, out_dir, cfg["name"])
    final_train_acc = history["train_acc"][-1] if history["train_acc"] else None
    _write_report(
        cfg,
        history,
        out_dir,
        train_set.classes,
        final_train_acc,
        len(train_idx),
        len(val_idx),
        n_test=len(test_idx),
        test_acc=test_acc,
        splits_file=used_splits_file,
        val_nn_metrics=val_nn_metrics,
        test_nn_metrics=test_nn_metrics,
    )
    (out_dir / "history.json").write_text(json.dumps(
        {k: v for k, v in history.items() if k not in {"last_val_features", "last_val_labels", "last_val_preds"}},
        indent=2,
    ))

    logger.success(f"Fine-tune done. Outputs in {out_dir}")
    logger.success(
        "  best.pth, history.json, training_curves.png (loss + acc), "
        "confusion_val.png (+ test if split), metrics_f1_accuracy.png, metrics_classification.json, "
        "tsne_*.png, report.md"
    )


if __name__ == "__main__":
    main()
