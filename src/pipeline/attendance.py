"""End-to-end attendance pipeline orchestrator.

Glues together every stage so the Streamlit app and the CLI ``attendance_run``
script can both consume a single high-level interface:

    pipeline = AttendancePipeline.from_configs(...)
    results  = pipeline.recognize_image(rgb_array)

A ``RecognitionResult`` carries everything we need to render an annotated frame
(bbox, name, similarity, spoof flag) and to log attendance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..alignment import align_face
from ..antispoof import AntiSpoofPredictor
from ..db import AttendanceDB, FaissEmbeddingStore
from ..detection import DetectedFace, build_detector
from ..recognition import EmbeddingExtractor
from ..utils import load_config


@dataclass
class RecognitionResult:
    bbox: np.ndarray
    detection_score: float
    landmarks: np.ndarray
    name: str
    similarity: float
    is_real: bool | None
    real_prob: float | None
    crop: np.ndarray              # aligned 112x112
    embedding: np.ndarray         # L2-normalised 512-D (kept for debug/top-k re-query)
    top_matches: list[tuple[str, float]]  # (name, similarity) ranked DESC, top-5


class AttendancePipeline:
    """Orchestrates Detection → Alignment → (Anti-Spoof) → Recognition → Gallery."""

    def __init__(
        self,
        detector,
        embedder: EmbeddingExtractor,
        gallery: FaissEmbeddingStore,
        anti_spoof: AntiSpoofPredictor | None = None,
        db: AttendanceDB | None = None,
        threshold: float = 0.4,
        align_size: int = 112,
        antispoof_size: int = 80,
    ):
        self.detector = detector
        self.embedder = embedder
        self.gallery = gallery
        self.anti_spoof = anti_spoof
        self.db = db
        self.threshold = threshold
        self.align_size = align_size
        self.antispoof_size = antispoof_size

    # ---------------------------------------------------------------- factory
    @classmethod
    def from_configs(
        cls,
        detection_cfg: str | Path,
        recognition_cfg: str | Path,
        recognition_weights: str | Path | None = None,
        antispoof_cfg: str | Path | None = None,
        antispoof_weights: str | Path | None = None,
        gallery_root: str | Path = "embeddings_db",
        db_path: str | Path = "attendance.db",
        threshold: float = 0.4,
    ) -> "AttendancePipeline":
        det_cfg = load_config(detection_cfg)
        rec_cfg = load_config(recognition_cfg)
        detector = build_detector(det_cfg)
        embedder = EmbeddingExtractor.from_config(rec_cfg, weights=recognition_weights)

        gallery = FaissEmbeddingStore(
            embedding_dim=int(rec_cfg["model"]["embedding_dim"]),
            root=gallery_root,
        )
        anti_spoof = None
        antispoof_size = 80
        if antispoof_cfg is not None:
            as_cfg = load_config(antispoof_cfg)
            anti_spoof = AntiSpoofPredictor.from_config(as_cfg, weights=antispoof_weights)
            antispoof_size = int(as_cfg["data"]["image_size"])

        db = AttendanceDB(db_path) if db_path else None
        align_size = int(det_cfg.get("align_size", 112))
        return cls(
            detector=detector,
            embedder=embedder,
            gallery=gallery,
            anti_spoof=anti_spoof,
            db=db,
            threshold=threshold,
            align_size=align_size,
            antispoof_size=antispoof_size,
        )

    # ---------------------------------------------------------- inference API
    def detect_and_align(self, image_rgb: np.ndarray) -> list[tuple[DetectedFace, np.ndarray]]:
        """Detect faces and return ``(face, aligned_crop)`` pairs."""
        faces = self.detector.detect(image_rgb)
        out = []
        for face in faces:
            crop = align_face(image_rgb, face.landmarks, output_size=self.align_size)
            face.crop = crop
            out.append((face, crop))
        return out

    def recognize_image(self, image_rgb: np.ndarray) -> list[RecognitionResult]:
        """Full pipeline on a single RGB image (HxWx3 uint8)."""
        det_pairs = self.detect_and_align(image_rgb)
        if not det_pairs:
            return []
        crops = [c for _, c in det_pairs]
        embeddings = self.embedder.encode(crops)
        results: list[RecognitionResult] = []
        for (face, crop), emb in zip(det_pairs, embeddings):
            is_real, real_prob = (None, None)
            if self.anti_spoof is not None:
                # Anti-spoof prefers a slightly larger crop than 112; cv2.resize
                # is enough since we already have the aligned face.
                is_real, real_prob = self.anti_spoof.predict(crop)
            # Always pull top-5 with NO threshold first (debug telemetry),
            # then apply threshold on the top-1 to decide known vs Unknown.
            top5 = self.gallery.search(emb, top_k=5, threshold=-1.0)
            top_matches = [(n, s) for n, s, _ in top5]
            if top_matches and top_matches[0][1] >= self.threshold:
                name, sim = top_matches[0]
            else:
                name = "Unknown"
                sim = top_matches[0][1] if top_matches else 0.0
            results.append(
                RecognitionResult(
                    bbox=face.bbox,
                    detection_score=face.score,
                    landmarks=face.landmarks,
                    name=name,
                    similarity=sim,
                    is_real=is_real,
                    real_prob=real_prob,
                    crop=crop,
                    embedding=emb.astype(np.float32),
                    top_matches=top_matches,
                )
            )
        return results

    def log_results(
        self,
        results: Sequence[RecognitionResult],
        source: str = "live",
        require_real: bool = True,
        cooldown_minutes: int = 5,
        session_id: int | None = None,
    ) -> list[str]:
        """Persist attendance for every confidently recognised result.

        Returns the list of student names that were just logged (after cooldown
        filtering) — useful for UI toast notifications.
        """
        if self.db is None:
            return []
        logged: list[str] = []
        for r in results:
            if r.name == "Unknown":
                continue
            if require_real and r.is_real is False:
                continue
            self.db.upsert_student(r.name)
            if self.db.log(
                r.name,
                r.similarity,
                source=source,
                cooldown_minutes=cooldown_minutes,
                session_id=session_id,
            ):
                logged.append(r.name)
        return logged
