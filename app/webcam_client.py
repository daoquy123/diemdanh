"""Browser webcam → server via streamlit-webrtc (getUserMedia on client, frames on server)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

from src.alignment import align_face
from src.pipeline.attendance import AttendancePipeline

# Public STUN for NAT (ngrok / home networks). TURN may be needed on strict firewalls.
RTC_ICE: dict[str, Any] = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
WEBCAM_MEDIA: dict[str, Any] = {"video": True, "audio": False}

_registry: dict[str, dict[str, Any]] = {}
_registry_lock = threading.Lock()


def _get(bid: str) -> dict[str, Any] | None:
    with _registry_lock:
        return _registry.get(bid)


def get_buffer(bid: str) -> dict[str, Any] | None:
    """Read-only handle for UI polling (do not mutate without ``lock``)."""
    return _get(bid)


def register_drop(bid: str) -> None:
    with _registry_lock:
        _registry.pop(bid, None)


def init_enroll_buffer(
    bid: str,
    *,
    pipeline: AttendancePipeline,
    norm_identity: str,
    stash_dir: Path,
    target_frames: int,
    max_seconds: float,
    det_score_min: float,
) -> None:
    stash_dir.mkdir(parents=True, exist_ok=True)
    buf: dict[str, Any] = {
        "kind": "enroll",
        "lock": threading.Lock(),
        "pipeline": pipeline,
        "norm_identity": norm_identity,
        "stash_dir": stash_dir,
        "crops": [],
        "paths": [],
        "t0": time.time(),
        "done": False,
        "target_frames": int(target_frames),
        "max_seconds": float(max_seconds),
        "det_score_min": float(det_score_min),
    }
    with _registry_lock:
        _registry[bid] = buf


def enroll_frame_callback(bid: str) -> Callable[[av.VideoFrame], av.VideoFrame]:
    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        buf = _get(bid)
        if not buf or buf.get("done"):
            return frame
        try:
            rgb = frame.to_ndarray(format="rgb24")
        except Exception:
            return frame
        now = time.time()
        with buf["lock"]:
            if buf["done"]:
                return frame
            if now - buf["t0"] >= buf["max_seconds"] or len(buf["crops"]) >= buf["target_frames"]:
                buf["done"] = True
                return frame

        pipe: AttendancePipeline = buf["pipeline"]
        try:
            faces = pipe.detector.detect(rgb)
        except Exception:
            return frame

        if not faces:
            return frame
        best = max(faces, key=lambda d: d.score)
        if best.score < buf["det_score_min"]:
            return frame
        try:
            crop = align_face(rgb, best.landmarks, output_size=pipe.align_size)
        except Exception:
            return frame
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        with buf["lock"]:
            if buf["done"]:
                return frame
            n = len(buf["crops"])
            if n >= buf["target_frames"]:
                buf["done"] = True
                return frame
            p = buf["stash_dir"] / f"{buf['norm_identity']}_{ts}_{n + 1:04d}.jpg"
            buf["crops"].append(crop)
            cv2.imwrite(str(p), bgr)
            buf["paths"].append(str(p))
            if len(buf["crops"]) >= buf["target_frames"]:
                buf["done"] = True
        return frame

    return callback


def init_attend_buffer(
    bid: str,
    *,
    pipeline: AttendancePipeline,
    user: str,
    session_id: int,
    norm_mssv: Callable[[str], str],
    live_seconds: float,
    warmup_seconds: float,
    streak_needed: int,
    det_score_min: float,
) -> None:
    buf: dict[str, Any] = {
        "kind": "attend",
        "lock": threading.Lock(),
        "pipeline": pipeline,
        "user_norm": norm_mssv(user),
        "norm_mssv": norm_mssv,
        "session_id": int(session_id),
        "t0": time.time(),
        "streak": 0,
        "last_good": None,
        "outcome": "none",
        "done": False,
        "pending_result": None,
        "live_seconds": float(live_seconds),
        "warmup_seconds": float(warmup_seconds),
        "streak_needed": int(streak_needed),
        "det_score_min": float(det_score_min),
    }
    with _registry_lock:
        _registry[bid] = buf


def attend_frame_callback(bid: str) -> Callable[[av.VideoFrame], av.VideoFrame]:
    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        buf = _get(bid)
        if not buf or buf.get("done"):
            return frame
        with buf["lock"]:
            if buf["done"] or buf.get("pending_result") is not None:
                return frame

        try:
            rgb = frame.to_ndarray(format="rgb24")
        except Exception:
            return frame

        elapsed = time.time() - buf["t0"]
        live_s = buf["live_seconds"]
        warm_s = buf["warmup_seconds"]
        streak_n = buf["streak_needed"]
        dmin = buf["det_score_min"]
        u = buf["user_norm"]
        norm_mssv: Callable[[str], str] = buf["norm_mssv"]

        if elapsed >= live_s:
            with buf["lock"]:
                if buf["pending_result"] is None:
                    buf["done"] = True
            return frame

        if elapsed < warm_s:
            return frame

        pipe: AttendancePipeline = buf["pipeline"]
        try:
            results = pipe.recognize_image(rgb)
        except Exception:
            return frame

        ok_frame = (
            len(results) == 1
            and results[0].name != "Unknown"
            and norm_mssv(str(results[0].name)) == u
            and results[0].detection_score >= dmin
        )

        with buf["lock"]:
            if buf["done"] or buf.get("pending_result") is not None:
                return frame
            if ok_frame:
                buf["streak"] += 1
                buf["last_good"] = results[0]
                if buf["streak"] >= streak_n:
                    buf["pending_result"] = results[0]
            else:
                buf["streak"] = 0
        return frame

    return callback
