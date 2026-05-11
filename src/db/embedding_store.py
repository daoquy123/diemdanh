"""FAISS-backed gallery of enrolled face embeddings.

Two on-disk artifacts are written next to each other:

* ``<root>/index.faiss`` — the FAISS inner-product index (with L2-normalised
  vectors this is equivalent to cosine similarity)
* ``<root>/meta.npz``    — parallel arrays of ``(name, image_path)`` so that
  search results can be turned back into human-readable identities

A simple JSON-style API (``add``, ``search``, ``remove``) so the Streamlit app
and CLI scripts can both speak to the gallery.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "faiss-cpu is required for the embedding store. "
        "Install with: pip install faiss-cpu"
    ) from e


class FaissEmbeddingStore:
    """Inner-product FAISS index of L2-normalised embeddings.

    Vectors must be L2-normalised before insertion (the recognition model
    already does this in eval mode). The search API returns
    ``(name, similarity)`` tuples sorted descending.
    """

    def __init__(self, embedding_dim: int = 512, root: str | Path = "embeddings_db"):
        self.embedding_dim = embedding_dim
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.faiss"
        self._meta_path = self.root / "meta.npz"

        if self._index_path.exists() and self._meta_path.exists():
            self.index = faiss.read_index(str(self._index_path))
            meta = np.load(self._meta_path, allow_pickle=True)
            self.names: list[str] = list(meta["names"].tolist())
            self.image_paths: list[str] = list(meta["image_paths"].tolist())
            if self.index.d != embedding_dim:
                raise ValueError(
                    f"Embedding dim mismatch: index.d={self.index.d} but config says {embedding_dim}. "
                    "Delete the embedding store or change the config."
                )
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.names = []
            self.image_paths = []

    def __len__(self) -> int:
        return len(self.names)

    @property
    def unique_identities(self) -> list[str]:
        return sorted(set(self.names))

    def add(self, embeddings: np.ndarray, names: Iterable[str], image_paths: Iterable[str]) -> None:
        names = list(names)
        image_paths = list(image_paths)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected (N,{self.embedding_dim}) embeddings, got {embeddings.shape}"
            )
        if len(names) != len(image_paths) or len(names) != embeddings.shape[0]:
            raise ValueError("embeddings, names and image_paths must have matching length.")
        self.index.add(embeddings.astype(np.float32))
        self.names.extend(names)
        self.image_paths.extend(image_paths)

    def remove_identity(self, name: str) -> int:
        """Drop all entries belonging to ``name``. Returns # removed."""
        mask = np.array([n != name for n in self.names], dtype=bool)
        if mask.all():
            return 0
        kept_vectors = self._all_vectors()[mask]
        kept_names = [n for n, keep in zip(self.names, mask) if keep]
        kept_paths = [p for p, keep in zip(self.image_paths, mask) if keep]
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        if len(kept_vectors):
            self.index.add(kept_vectors.astype(np.float32))
        removed = len(self.names) - len(kept_names)
        self.names = kept_names
        self.image_paths = kept_paths
        return removed

    def _all_vectors(self) -> np.ndarray:
        n = self.index.ntotal
        if n == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return self.index.reconstruct_n(0, n)

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[str, float, str]]:
        """Return the ``top_k`` matches above ``threshold``.

        Each result is ``(name, similarity, image_path)``; an empty list means
        "unknown". Aggregates per-identity by **max** similarity so multiple
        enrolled images don't dominate the top-K.
        """
        if self.index.ntotal == 0:
            return []
        emb = embedding.reshape(1, -1).astype(np.float32)
        sims, idxs = self.index.search(emb, k=min(self.index.ntotal, max(top_k, 50)))
        sims, idxs = sims[0], idxs[0]

        best_per_name: dict[str, tuple[float, str]] = {}
        for sim, idx in zip(sims, idxs):
            if idx < 0:
                continue
            sim = float(sim)
            name = self.names[idx]
            path = self.image_paths[idx]
            cur = best_per_name.get(name)
            if cur is None or sim > cur[0]:
                best_per_name[name] = (sim, path)

        ranked = sorted(best_per_name.items(), key=lambda kv: kv[1][0], reverse=True)
        return [
            (name, sim, path) for name, (sim, path) in ranked if sim >= threshold
        ][:top_k]

    def save(self) -> None:
        faiss.write_index(self.index, str(self._index_path))
        np.savez(
            self._meta_path,
            names=np.array(self.names, dtype=object),
            image_paths=np.array(self.image_paths, dtype=object),
        )
