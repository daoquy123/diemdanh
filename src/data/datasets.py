"""Dataset classes for face recognition.

We intentionally support three data sources:

* :class:`FolderFaceDataset` – ImageFolder-style ``root/<identity>/<img>``,
  used for both CASIA-WebFace style train and a custom enrolled student set.
* :class:`LFWPairs`           – LFW verification pairs (``pairs.txt``).
* :class:`PKBatchSampler`     – Sampler producing batches of ``P`` identities x
  ``K`` images each, required for triplet-loss FaceNet-style training.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterator, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

ImageTransform = Callable[[np.ndarray], dict] | None

_VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _scan_imagefolder(root: Path, min_per_id: int = 1) -> tuple[list[str], list[int], list[str]]:
    """Walk ``root`` collecting per-identity image paths.

    Returns ``(paths, labels, classes)`` where labels are 0..N-1 indices into
    ``classes``.
    """
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    classes: list[str] = []
    paths: list[str] = []
    labels: list[int] = []

    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = [p for p in class_dir.iterdir() if p.suffix.lower() in _VALID_EXT]
        if len(imgs) < min_per_id:
            continue
        idx = len(classes)
        classes.append(class_dir.name)
        for img in imgs:
            paths.append(str(img))
            labels.append(idx)

    if not paths:
        raise RuntimeError(
            f"No images found under {root}. Expected ImageFolder layout: <root>/<id>/<img>"
        )
    return paths, labels, classes


def _imread_rgb(path: str) -> np.ndarray:
    """Read an image as RGB uint8 ``HxWx3`` numpy array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class FolderFaceDataset(Dataset):
    """ImageFolder dataset: ``root/<identity_name>/<image>``."""

    def __init__(
        self,
        root: str | Path,
        transform: ImageTransform = None,
        min_images_per_id: int = 1,
        return_path: bool = False,
    ):
        self.root = Path(root)
        self.transform = transform
        self.return_path = return_path
        self.paths, self.labels, self.classes = _scan_imagefolder(self.root, min_images_per_id)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = self.labels[idx]
        img = _imread_rgb(path)
        if self.transform is not None:
            out = self.transform(image=img)
            img_tensor = out["image"] if isinstance(out, dict) else out
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.return_path:
            return img_tensor, label, path
        return img_tensor, label


class LFWPairs(Dataset):
    """LFW verification dataset built from the standard ``pairs.txt`` file.

    Each item yields ``(img1, img2, is_same)`` where ``is_same`` is 0/1.

    The directory must follow the canonical LFW layout::

        root/Aaron_Eckhart/Aaron_Eckhart_0001.jpg
    """

    def __init__(
        self,
        root: str | Path,
        pairs_file: str | Path,
        transform: ImageTransform = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.pairs = self._parse_pairs(Path(pairs_file))

    @staticmethod
    def _format_path(root: Path, name: str, idx: int) -> Path:
        return root / name / f"{name}_{int(idx):04d}.jpg"

    def _parse_pairs(self, pairs_file: Path) -> list[tuple[str, str, int]]:
        if not pairs_file.exists():
            raise FileNotFoundError(f"LFW pairs file not found: {pairs_file}")
        pairs: list[tuple[str, str, int]] = []
        with pairs_file.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        # First line is the header (n_folds n_pairs); skip if it has 2 ints.
        if lines and len(lines[0].split()) == 2:
            lines = lines[1:]
        for ln in lines:
            parts = ln.split()
            if len(parts) == 3:
                name, i, j = parts
                p1 = self._format_path(self.root, name, int(i))
                p2 = self._format_path(self.root, name, int(j))
                same = 1
            elif len(parts) == 4:
                name1, i, name2, j = parts
                p1 = self._format_path(self.root, name1, int(i))
                p2 = self._format_path(self.root, name2, int(j))
                same = 0
            else:
                continue
            pairs.append((str(p1), str(p2), same))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, path: str) -> torch.Tensor:
        img = _imread_rgb(path)
        if self.transform is not None:
            out = self.transform(image=img)
            return out["image"] if isinstance(out, dict) else out
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx: int):
        p1, p2, same = self.pairs[idx]
        return self._load(p1), self._load(p2), torch.tensor(same, dtype=torch.long)


class PKBatchSampler(Sampler[list[int]]):
    """Yield batches of ``P`` identities x ``K`` images, the canonical sampler
    for triplet-style mining (FaceNet semi-hard / batch-hard).
    """

    def __init__(
        self,
        labels: Sequence[int],
        p: int = 30,
        k: int = 3,
        num_batches: int | None = None,
        seed: int = 42,
    ):
        self.labels = list(labels)
        self.p = p
        self.k = k
        self.num_batches = num_batches or max(1, len(self.labels) // (p * k))
        self.rng = random.Random(seed)

        self.label_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, lbl in enumerate(self.labels):
            self.label_to_indices[lbl].append(idx)
        self.label_to_indices = {
            lbl: idxs for lbl, idxs in self.label_to_indices.items() if len(idxs) >= 2
        }
        if len(self.label_to_indices) < self.p:
            raise ValueError(
                f"PKBatchSampler needs at least P={self.p} identities with >=2 images, "
                f"got {len(self.label_to_indices)}."
            )

    def __iter__(self) -> Iterator[list[int]]:
        labels_pool = list(self.label_to_indices.keys())
        for _ in range(self.num_batches):
            chosen_labels = self.rng.sample(labels_pool, self.p)
            batch: list[int] = []
            for lbl in chosen_labels:
                idxs = self.label_to_indices[lbl]
                if len(idxs) >= self.k:
                    batch.extend(self.rng.sample(idxs, self.k))
                else:
                    batch.extend(self.rng.choices(idxs, k=self.k))
            yield batch

    def __len__(self) -> int:
        return self.num_batches


def build_train_dataset(
    root: str | Path,
    transform: ImageTransform,
    min_images_per_id: int = 1,
) -> FolderFaceDataset:
    """Convenience builder used by training scripts."""
    return FolderFaceDataset(root, transform=transform, min_images_per_id=min_images_per_id)
