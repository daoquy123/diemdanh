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

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        emb = embedding.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(emb)) + 1e-9
        return emb / norm

    def _score_identities(
        self, embedding: np.ndarray
    ) -> list[tuple[str, float, float, float]]:
        """Per identity: (name, final, proto_sim, best_frame_sim)."""
        if self.index.ntotal == 0:
            return []
        emb = self._normalize(embedding)
        vectors = self._all_vectors()
        buckets: dict[str, list[np.ndarray]] = {}
        for vec, name in zip(vectors, self.names):
            buckets.setdefault(name, []).append(vec)

        scored: list[tuple[str, float, float, float]] = []
        for name, arrs in buckets.items():
            stack = np.stack(arrs).astype(np.float32)
            norms = stack / (np.linalg.norm(stack, axis=1, keepdims=True) + 1e-9)
            frame_sims = norms @ emb
            max_frame = float(frame_sims.max())

            proto = self._normalize(stack.mean(axis=0))
            proto_sim = float(np.dot(emb, proto))

            good = frame_sims >= (max_frame - 0.12)
            if int(good.sum()) >= 5:
                proto_trim = self._normalize(stack[good].mean(axis=0))
                proto_sim = max(proto_sim, float(np.dot(emb, proto_trim)))

            final = max(proto_sim, max_frame)
            scored.append((name, final, proto_sim, max_frame))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[str, float, str]]:
        """Hybrid: max(trimmed-prototype, best enrolled frame) per person."""
        ranked = self._score_identities(embedding)
        return [
            (name, final, "")
            for name, final, _p, _m in ranked
            if final >= threshold
        ][:top_k]

    def search_best_frame(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[str, float, str]]:
        """Legacy: max similarity to any single enrolled frame (can flip between IDs)."""
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

    def rank_all_identities(
        self, embedding: np.ndarray
    ) -> list[tuple[str, float, str, float, float]]:
        """All identities: (name, final, path, proto_sim, max_frame_sim)."""
        return [
            (name, final, "", proto, mx)
            for name, final, proto, mx in self._score_identities(embedding)
        ]

    def save(self) -> None:
        faiss.write_index(self.index, str(self._index_path))
        np.savez(
            self._meta_path,
            names=np.array(self.names, dtype=object),
            image_paths=np.array(self.image_paths, dtype=object),
        )
