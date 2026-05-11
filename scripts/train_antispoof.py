"""Train MiniFASNetV2 for face anti-spoofing.

Expected dataset layout (binary classification: real vs spoof)::

    data/raw/antispoof/
      ├── real/        # genuine selfies
      └── spoof/       # photos of phone screens, printed papers, masks, ...

Public datasets you can use:
- CelebA-Spoof  https://github.com/Davidzhangyuanhan/CelebA-Spoof
- CASIA-FASD
- NUAA-IPMI
- Or collect your own: ~500 real + ~500 spoof selfies is enough for a sanity-
  level baseline.

Example:
    python -m scripts.train_antispoof \\
        --config configs/antispoof/minifasnet.yaml \\
        --data-root data/raw/antispoof
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.antispoof import build_antispoof_model
from src.data.datasets import FolderFaceDataset
from src.data.transforms import build_eval_transform, build_train_transform
from src.utils import dump_config, get_logger, load_config, merge_overrides, select_device, set_seed

logger = get_logger("logs/train_antispoof.log")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/antispoof/minifasnet.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = merge_overrides(load_config(args.config), args.override)
    device = select_device(args.device)

    image_size = int(cfg["data"]["image_size"])
    train_tf = build_train_transform(
        image_size=image_size,
        mean=tuple(cfg["data"]["mean"]),
        std=tuple(cfg["data"]["std"]),
        horizontal_flip=float(cfg["data"]["augment"].get("horizontal_flip", 0.5)),
        color_jitter=dict(cfg["data"]["augment"].get("color_jitter", {})),
        blur=float(cfg["data"]["augment"].get("blur", 0.2)),
        rotation=5,
    )
    eval_tf = build_eval_transform(image_size=image_size, mean=tuple(cfg["data"]["mean"]), std=tuple(cfg["data"]["std"]))

    full = FolderFaceDataset(args.data_root, transform=train_tf, min_images_per_id=1)
    if set(full.classes) != {"real", "spoof"}:
        logger.warning(f"Expected classes {{real, spoof}}, got {full.classes}. Continuing — labels will follow that order.")
    val_size = max(1, int(0.1 * len(full)))
    train_size = len(full) - val_size
    train_set, val_set = random_split(full, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    val_set.dataset.transform = eval_tf  # type: ignore[attr-defined]
    logger.info(f"train={train_size} val={val_size}")

    train_loader = DataLoader(train_set, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=int(cfg["train"]["num_workers"]), drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    model = build_antispoof_model(cfg).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        momentum=float(cfg["train"].get("momentum", 0.9)),
        weight_decay=float(cfg["train"].get("weight_decay", 5e-4)),
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["train"]["epochs"]))
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, ckpt_dir / "config.yaml")
    best_acc = 0.0

    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        running, n_seen, correct = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, lbls)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
            correct += (logits.argmax(1) == lbls).sum().item()
            pbar.set_postfix(loss=loss.item(), acc=correct / max(1, n_seen))
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_correct, val_total = 0, 0
            for imgs, lbls in val_loader:
                imgs = imgs.to(device); lbls = lbls.to(device)
                logits = model(imgs)
                val_correct += (logits.argmax(1) == lbls).sum().item()
                val_total += imgs.size(0)
            val_acc = val_correct / max(1, val_total)
        logger.info(f"epoch {epoch+1}: train_loss={running/max(1,n_seen):.4f} train_acc={correct/max(1,n_seen):.4f} val_acc={val_acc:.4f}")

        torch.save({"state_dict": model.state_dict(), "val_acc": val_acc}, ckpt_dir / "last.pth")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "val_acc": val_acc}, ckpt_dir / "best.pth")
            logger.success(f"saved best val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
