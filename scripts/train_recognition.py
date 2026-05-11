"""Train a face recognition embedding model.

Supports both classification-style heads (ArcFace / CosFace + cross-entropy)
and FaceNet-style triplet loss with PK sampling. The choice is driven by the
``loss.type`` field in the config.

Example:
    python -m scripts.train_recognition --config configs/recognition/arcface_r50.yaml
    python -m scripts.train_recognition --config configs/recognition/facenet.yaml --override train.epochs=10
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import (
    PKBatchSampler,
    build_eval_transform,
    build_train_dataset,
    build_train_transform,
)
from src.recognition import build_loss, build_recognition_model
from src.utils import dump_config, get_logger, load_config, merge_overrides, select_device, set_seed

logger = get_logger("logs/train_recognition.log")


def _build_optimizer(name: str, params, lr: float, momentum: float, wd: float):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def _build_scheduler(name: str, optimizer, epochs: int, warmup_epochs: int):
    if name in {None, "none", ""}:
        return None
    if name == "cosine":
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs)
        )
        return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    raise ValueError(f"Unknown scheduler: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None, help="Override data root (else uses configs/data/<train_dataset>.yaml)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--override", nargs="*", default=[], help="OmegaConf-style overrides, e.g. train.batch_size=64")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = merge_overrides(load_config(args.config), args.override)
    device = select_device(args.device)
    logger.info(f"Device: {device}")

    # Dataset ---------------------------------------------------------------
    data_cfg = load_config(f"configs/data/{cfg['data']['train_dataset']}.yaml")
    data_root = Path(args.data_root) if args.data_root else Path(data_cfg["processed_root"])
    if not data_root.exists():
        raise FileNotFoundError(
            f"Aligned dataset not found at {data_root}. Run scripts/prepare_data.py first."
        )

    image_size = int(cfg["data"]["image_size"])
    aug = cfg["data"].get("augment", {})
    train_tf = build_train_transform(
        image_size=image_size,
        mean=tuple(cfg["data"]["mean"]),
        std=tuple(cfg["data"]["std"]),
        horizontal_flip=float(aug.get("horizontal_flip", 0.5)),
        color_jitter=dict(aug.get("color_jitter", {})),
        blur=float(aug.get("blur", 0.1)),
        rotation=int(aug.get("rotation", 5)),
    )
    train_set = build_train_dataset(
        data_root,
        transform=train_tf,
        min_images_per_id=int(data_cfg.get("min_images_per_id", 1)),
    )
    logger.info(f"Train set: {len(train_set)} images / {train_set.num_classes} classes")

    loss_type = str(cfg["loss"]["type"]).lower()
    if loss_type in {"arcface", "cosface"}:
        loader = DataLoader(
            train_set,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=True,
            num_workers=int(cfg["train"]["num_workers"]),
            pin_memory=True,
            drop_last=True,
        )
    else:  # triplet — needs PK sampling
        bs = int(cfg["train"]["batch_size"])
        p = max(2, bs // 3)
        k = max(2, bs // p)
        sampler = PKBatchSampler(
            labels=train_set.labels,
            p=p,
            k=k,
            num_batches=max(50, len(train_set) // (p * k)),
            seed=args.seed,
        )
        loader = DataLoader(
            train_set,
            batch_sampler=sampler,
            num_workers=int(cfg["train"]["num_workers"]),
            pin_memory=True,
        )

    # Model -----------------------------------------------------------------
    model = build_recognition_model(cfg).to(device)
    head, criterion = build_loss(cfg, embedding_dim=model.embedding_dim, num_classes=train_set.num_classes)
    if head is not None:
        head = head.to(device)

    params = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    optimizer = _build_optimizer(
        cfg["train"].get("optimizer", "sgd"),
        params,
        lr=float(cfg["train"]["lr"]),
        momentum=float(cfg["train"].get("momentum", 0.9)),
        wd=float(cfg["train"].get("weight_decay", 5e-4)),
    )
    scheduler = _build_scheduler(
        cfg["train"].get("scheduler", "cosine"),
        optimizer,
        epochs=int(cfg["train"]["epochs"]),
        warmup_epochs=int(cfg["train"].get("warmup_epochs", 0)),
    )

    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, ckpt_dir / "config.yaml")
    log_interval = int(cfg["train"].get("log_interval", 50))

    epochs = int(cfg["train"]["epochs"])
    best_loss = math.inf

    for epoch in range(epochs):
        model.train()
        if head is not None:
            head.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            imgs, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            embeddings = model(imgs, normalize=False)
            if head is not None:  # ArcFace/CosFace + CE
                logits = head(embeddings, labels)
                loss = criterion(logits, labels)
            else:  # triplet
                loss = criterion(embeddings, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            running += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
            if step % log_interval == 0:
                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        if scheduler is not None:
            scheduler.step()
        epoch_loss = running / max(1, n_seen)
        logger.info(f"epoch {epoch+1}: loss={epoch_loss:.4f}")

        ckpt = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "head": head.state_dict() if head is not None else None,
            "optimizer": optimizer.state_dict(),
            "loss": epoch_loss,
        }
        torch.save(ckpt, ckpt_dir / "last.pth")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ckpt, ckpt_dir / "best.pth")
            logger.success(f"saved best: loss={epoch_loss:.4f}")

    logger.success(f"Training done. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
