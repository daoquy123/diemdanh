"""Evaluate recognition models on fixed train/val/test splits.

Features:
1) Threshold sweep on VAL to pick best threshold.
2) Evaluate on TEST with that threshold.
3) Export hardest identities (lowest recall / most confusions).

Example:
    python -m scripts.evaluate_models ^
      --data-root data/processed/merged_faces ^
      --splits-file data/splits/merged_faces_splits.json ^
      --run facenet|configs/recognition/facenet.yaml|weights/facenet/finetuned_custom/best.pth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset

from src.data import build_eval_transform, build_train_dataset
from src.recognition import build_recognition_model
from src.utils import load_config, select_device


def _indices_from_split_json(dataset, root: Path, split_paths: list[str]) -> list[int]:
    root = Path(root).resolve()
    rel_set = {Path(p).as_posix() for p in split_paths}
    idxs: list[int] = []
    for i, abs_p in enumerate(dataset.paths):
        rel = Path(abs_p).resolve().relative_to(root).as_posix()
        if rel in rel_set:
            idxs.append(i)
    return sorted(idxs)


@torch.no_grad()
def _embed_indices(
    model,
    dataset,
    indices: list[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    sub = Subset(dataset, indices)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feats, labels = [], []
    model.eval()
    for imgs, ys in loader:
        imgs = imgs.to(device, non_blocking=True)
        emb = model(imgs, normalize=True).cpu().numpy().astype(np.float32)
        feats.append(emb)
        labels.append(ys.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def _build_prototypes(train_emb: np.ndarray, train_lbl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(train_lbl)
    proto = np.stack([train_emb[train_lbl == u].mean(axis=0) for u in unique]).astype(np.float32)
    proto /= (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-9)
    return unique, proto


def _predict_with_prototypes(
    query_emb: np.ndarray,
    unique_labels: np.ndarray,
    prototypes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sims = query_emb @ prototypes.T
    pred_idx = np.argmax(sims, axis=1)
    pred_lbl = unique_labels[pred_idx]
    pred_sim = sims[np.arange(len(sims)), pred_idx]
    return pred_lbl, pred_sim


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    acc = float((y_true == y_pred).mean())
    labels = sorted(set(int(x) for x in np.unique(y_true)))
    macro = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    weighted = float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
    return {"accuracy": acc, "macro_f1": macro, "weighted_f1": weighted}


def _sweep_threshold(
    y_true: np.ndarray,
    pred_lbl: np.ndarray,
    pred_sim: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    t_step: float,
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    best_t = t_min
    best_m = {"accuracy": -1.0, "macro_f1": -1.0, "weighted_f1": -1.0, "unknown_rate": 0.0}
    thr = t_min
    while thr <= t_max + 1e-9:
        y_hat = pred_lbl.copy()
        y_hat[pred_sim < thr] = -1  # unknown
        m = _metrics(y_true, y_hat)
        m["unknown_rate"] = float((pred_sim < thr).mean())
        rows.append({"threshold": float(thr), **m})
        # prioritize macro F1, then weighted F1
        if (m["macro_f1"], m["weighted_f1"]) > (best_m["macro_f1"], best_m["weighted_f1"]):
            best_t = float(thr)
            best_m = m
        thr += t_step
    return best_t, best_m, rows


def _hardest_identities(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    top_k: int = 20,
) -> list[dict]:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    hard: list[dict] = []
    for i, name in enumerate(class_names):
        support = int(cm[i].sum())
        if support == 0:
            continue
        correct = int(cm[i, i])
        recall = correct / support
        row = cm[i].copy()
        row[i] = 0
        j = int(np.argmax(row))
        hard.append(
            {
                "identity": name,
                "support": support,
                "recall": round(float(recall), 4),
                "top_confused_with": class_names[j] if row[j] > 0 else None,
                "top_confused_count": int(row[j]),
            }
        )
    hard.sort(key=lambda x: (x["recall"], -x["support"]))
    return hard[:top_k]


def _parse_run(text: str) -> tuple[str, str, str]:
    parts = text.split("|")
    if len(parts) != 3:
        raise ValueError(f"--run must be 'name|config|weights', got: {text}")
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits-file", required=True)
    parser.add_argument("--run", action="append", required=True, help="name|config|weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--thr-min", type=float, default=0.20)
    parser.add_argument("--thr-max", type=float, default=0.80)
    parser.add_argument("--thr-step", type=float, default=0.01)
    parser.add_argument("--top-hard", type=int, default=20)
    parser.add_argument("--max-per-split", type=int, default=0, help="0 means full split")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_data = json.loads(Path(args.splits_file).read_text(encoding="utf-8"))

    # Build split paths once, map to indices per-run (because different models may use different transforms).
    split_train = split_data.get("train", [])
    split_val = split_data.get("val", [])
    split_test = split_data.get("test", [])

    out_dir = Path("reports/eval_models")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for run_text in args.run:
        name, cfg_path, w_path = _parse_run(run_text)
        cfg = load_config(cfg_path)
        eval_tf = build_eval_transform(
            image_size=int(cfg["data"]["image_size"]),
            mean=tuple(cfg["data"]["mean"]),
            std=tuple(cfg["data"]["std"]),
        )
        dataset = build_train_dataset(data_root, transform=eval_tf, min_images_per_id=2)
        train_idx = _indices_from_split_json(dataset, data_root, split_train)
        val_idx = _indices_from_split_json(dataset, data_root, split_val)
        test_idx = _indices_from_split_json(dataset, data_root, split_test)
        if args.max_per_split > 0:
            train_idx = train_idx[: args.max_per_split]
            val_idx = val_idx[: args.max_per_split]
            test_idx = test_idx[: args.max_per_split]

        device = select_device(args.device)
        model = build_recognition_model(cfg).to(device).eval()
        state = torch.load(w_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

        train_emb, train_lbl = _embed_indices(model, dataset, train_idx, device, args.batch_size, args.num_workers)
        val_emb, val_lbl = _embed_indices(model, dataset, val_idx, device, args.batch_size, args.num_workers)
        test_emb, test_lbl = _embed_indices(model, dataset, test_idx, device, args.batch_size, args.num_workers)

        uniq, proto = _build_prototypes(train_emb, train_lbl)
        val_pred, val_sim = _predict_with_prototypes(val_emb, uniq, proto)
        test_pred, test_sim = _predict_with_prototypes(test_emb, uniq, proto)

        val_closed = _metrics(val_lbl, val_pred)
        test_closed = _metrics(test_lbl, test_pred)

        best_t, best_val_thr, sweep_rows = _sweep_threshold(
            val_lbl, val_pred, val_sim, t_min=args.thr_min, t_max=args.thr_max, t_step=args.thr_step
        )
        test_pred_thr = test_pred.copy()
        test_pred_thr[test_sim < best_t] = -1
        test_thr = _metrics(test_lbl, test_pred_thr)
        test_thr["unknown_rate"] = float((test_sim < best_t).mean())

        hard = _hardest_identities(test_lbl, test_pred, dataset.classes, top_k=args.top_hard)
        run_out = out_dir / name
        run_out.mkdir(parents=True, exist_ok=True)
        (run_out / "threshold_sweep_val.json").write_text(json.dumps(sweep_rows, indent=2), encoding="utf-8")
        (run_out / "hardest_identities.json").write_text(json.dumps(hard, indent=2, ensure_ascii=False), encoding="utf-8")

        results[name] = {
            "checkpoint": w_path,
            "closed_set_val": val_closed,
            "closed_set_test": test_closed,
            "best_threshold_on_val": best_t,
            "val_at_best_threshold": best_val_thr,
            "test_at_best_threshold": test_thr,
            "files": {
                "threshold_sweep_val": str(run_out / "threshold_sweep_val.json"),
                "hardest_identities": str(run_out / "hardest_identities.json"),
            },
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {summary_path}")
    for name, r in results.items():
        print(
            f"[{name}] closed-test acc={r['closed_set_test']['accuracy']:.4f} "
            f"macro_f1={r['closed_set_test']['macro_f1']:.4f} | "
            f"thr*={r['best_threshold_on_val']:.2f} "
            f"test@thr acc={r['test_at_best_threshold']['accuracy']:.4f} "
            f"unk={r['test_at_best_threshold']['unknown_rate']:.3f}"
        )


if __name__ == "__main__":
    main()

