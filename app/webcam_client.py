"""Browser webcam → server via streamlit-webrtc (getUserMedia on client, frames on server)."""

from __future__ import annotations

import json
import os
import queue
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


def build_rtc_configuration() -> dict[str, Any]:
    """ICE servers for WebRTC. STUN alone fails on some NATs — use TURN env vars then.

    Optional env (Linux/systemd or Windows):

    * ``WEBRTC_ICE_SERVERS_JSON`` — JSON list of objects, e.g.
      ``[{"urls":"turn:relay.example.com:3478","username":"u","credential":"p"}]``
    * ``TURN_URI`` — ``turn:host:3478`` or ``turns:host:5349`` (comma-separated allowed)
    * ``TURN_USERNAME`` / ``TURN_PASSWORD`` — credentials for that TURN server (coturn, Twilio, Metered, …)
    """
    ice_servers: list[dict[str, Any]] = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]},
    ]
    raw = os.environ.get("WEBRTC_ICE_SERVERS_JSON", "").strip()
    if raw:
        try:
            extra = json.loads(raw)
            if isinstance(extra, list):
                for item in extra:
                    if isinstance(item, dict) and "urls" in item:
                        ice_servers.append(item)
        except json.JSONDecodeError:
            pass
    turn_uri = os.environ.get("TURN_URI", "").strip()
    if turn_uri:
        urls = [u.strip() for u in turn_uri.split(",") if u.strip()]
        turn_entry: dict[str, Any] = {"urls": urls[0] if len(urls) == 1 else urls}
        u = os.environ.get("TURN_USERNAME", "").strip()
        p = os.environ.get("TURN_PASSWORD", "").strip()
        if u:
            turn_entry["username"] = u
        if p:
            turn_entry["credential"] = p
        ice_servers.append(turn_entry)
    return {"iceServers": ice_servers}


# Built once at import; restart Streamlit after changing ICE-related env vars.
RTC_ICE: dict[str, Any] = build_rtc_configuration()


def _load_webcam_quality_preset() -> tuple[dict[str, Any], int, float, float]:
    """``WEBCAM_QUALITY_PRESET`` = ``fast`` | ``balanced`` | ``sharp`` (env).

    Không thể vừa “real-time HD” vừa nhẹ CPU — preset chỉ đổi điểm cân bằng.
    """
    p = os.environ.get("WEBCAM_QUALITY_PRESET", "balanced").strip().lower()
    if p == "sharp":
        media: dict[str, Any] = {
            "video": {
                "width": {"ideal": 960, "max": 1280},
                "height": {"ideal": 720, "max": 720},
                "frameRate": {"ideal": 24, "max": 30},
            },
            "audio": False,
        }
        return media, 800, 0.14, 0.34
    if p == "fast":
        media = {
            "video": {
                "width": {"ideal": 480, "max": 640},
                "height": {"ideal": 360, "max": 480},
                "frameRate": {"ideal": 12, "max": 18},
            },
            "audio": False,
        }
        return media, 448, 0.10, 0.20
    # balanced — default: sharper than old 512 without going full sharp
    media = {
        "video": {
            "width": {"ideal": 854, "max": 1280},
            "height": {"ideal": 480, "max": 720},
            "frameRate": {"ideal": 20, "max": 30},
        },
        "audio": False,
    }
    return media, 640, 0.12, 0.24


WEBCAM_MEDIA, WEBCAM_INFER_MAX_SIDE, ENROLL_DETECT_MIN_INTERVAL, ATTEND_RECOGNIZE_MIN_INTERVAL = (
    _load_webcam_quality_preset()
)
# Ảnh lưu đăng ký (1–100); cao hơn = file lớn hơn, nét hơn.
ENROLL_JPEG_QUALITY = max(70, min(100, int(os.environ.get("WEBCAM_JPEG_QUALITY", "92"))))

_registry: dict[str, dict[str, Any]] = {}
_registry_lock = threading.Lock()


def _resize_max_side(rgb: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return rgb
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb
    scale = max_side / m
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def _get(bid: str) -> dict[str, Any] | None:
    with _registry_lock:
        return _registry.get(bid)


def get_buffer(bid: str) -> dict[str, Any] | None:
    """Read-only handle for UI polling (do not mutate without ``lock``)."""
    return _get(bid)


def register_drop(bid: str) -> None:
    with _registry_lock:
        buf = _registry.get(bid)
        if buf is not None:
            buf["_stop_worker"] = True
            fq = buf.get("_frame_q")
            if fq is not None:
                try:
                    fq.put_nowait(None)
                except queue.Full:
                    try:
                        fq.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        fq.put_nowait(None)
                    except queue.Full:
                        pass
        _registry.pop(bid, None)


def _enqueue_latest(q: queue.Queue, rgb: np.ndarray) -> None:
    """Keep only the freshest frame so the worker never falls minutes behind."""
    try:
        q.put_nowait(rgb)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(rgb)
        except queue.Full:
            pass


def _run_enroll_worker(bid: str) -> None:
    while True:
        buf = _get(bid)
        if buf is None or buf.get("_stop_worker"):
            return
        fq: queue.Queue | None = buf.get("_frame_q")
        if fq is None:
            return
        try:
            item = fq.get(timeout=0.35)
        except queue.Empty:
            continue
        if item is None or buf.get("_stop_worker"):
            return

        rgb = item
        now = time.time()
        with buf["lock"]:
            if buf["done"]:
                continue
            if now - buf["t0"] >= buf["max_seconds"] or len(buf["crops"]) >= buf["target_frames"]:
                buf["done"] = True
                continue
            if now - buf["_last_enroll_det_ts"] < ENROLL_DETECT_MIN_INTERVAL:
                continue
            buf["_last_enroll_det_ts"] = now

        rgb_small = _resize_max_side(rgb, WEBCAM_INFER_MAX_SIDE)
        pipe: AttendancePipeline = buf["pipeline"]
        try:
            faces = pipe.detector.detect(rgb_small)
        except Exception:
            continue
        if not faces:
            continue
        best = max(faces, key=lambda d: d.score)
        if best.score < buf["det_score_min"]:
            continue
        try:
            crop = align_face(rgb_small, best.landmarks, output_size=pipe.align_size)
        except Exception:
            continue
        bgr = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        with buf["lock"]:
            if buf["done"]:
                continue
            n = len(buf["crops"])
            if n >= buf["target_frames"]:
                buf["done"] = True
                continue
            p = buf["stash_dir"] / f"{buf['norm_identity']}_{ts}_{n + 1:04d}.jpg"
            buf["crops"].append(crop)
            cv2.imwrite(str(p), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), ENROLL_JPEG_QUALITY])
            buf["paths"].append(str(p))
            if len(buf["crops"]) >= buf["target_frames"]:
                buf["done"] = True


def _run_attend_worker(bid: str) -> None:
    while True:
        buf = _get(bid)
        if buf is None or buf.get("_stop_worker"):
            return
        fq = buf.get("_frame_q")
        if fq is None:
            return
        try:
            item = fq.get(timeout=0.35)
        except queue.Empty:
            continue
        if item is None or buf.get("_stop_worker"):
            return

        rgb = item
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
            continue
        if elapsed < warm_s:
            continue

        now = time.time()
        with buf["lock"]:
            if buf["done"] or buf.get("pending_result") is not None:
                continue
            if now - buf["_last_rec_ts"] < ATTEND_RECOGNIZE_MIN_INTERVAL:
                continue
            buf["_last_rec_ts"] = now

        rgb_small = _resize_max_side(rgb, WEBCAM_INFER_MAX_SIDE)
        pipe: AttendancePipeline = buf["pipeline"]
        try:
            results = pipe.recognize_image(rgb_small)
        except Exception:
            continue

        ok_frame = (
            len(results) == 1
            and results[0].name != "Unknown"
            and norm_mssv(str(results[0].name)) == u
            and results[0].detection_score >= dmin
        )

        with buf["lock"]:
            if buf["done"] or buf.get("pending_result") is not None:
                continue
            if ok_frame:
                buf["streak"] += 1
                buf["last_good"] = results[0]
                if buf["streak"] >= streak_n:
                    buf["pending_result"] = results[0]
            else:
                buf["streak"] = 0


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
        "_last_enroll_det_ts": 0.0,
        "_frame_q": queue.Queue(maxsize=1),
        "_stop_worker": False,
    }
    with _registry_lock:
        _registry[bid] = buf
    threading.Thread(
        target=_run_enroll_worker,
        args=(bid,),
        daemon=True,
        name=f"enroll-{bid[:8]}",
    ).start()


def enroll_frame_callback(bid: str) -> Callable[[av.VideoFrame], av.VideoFrame]:
    """Return immediately after enqueue — heavy work runs in ``_run_enroll_worker`` (smooth preview)."""

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        buf = _get(bid)
        if not buf or buf.get("done"):
            return frame
        try:
            rgb = frame.to_ndarray(format="rgb24")
        except Exception:
            return frame
        with buf["lock"]:
            if buf["done"]:
                return frame
            now = time.time()
            if now - buf["t0"] >= buf["max_seconds"] or len(buf["crops"]) >= buf["target_frames"]:
                buf["done"] = True
                return frame
        _enqueue_latest(buf["_frame_q"], rgb.copy())
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
        "_last_rec_ts": 0.0,
        "_frame_q": queue.Queue(maxsize=1),
        "_stop_worker": False,
    }
    with _registry_lock:
        _registry[bid] = buf
    threading.Thread(
        target=_run_attend_worker,
        args=(bid,),
        daemon=True,
        name=f"attend-{bid[:8]}",
    ).start()


def attend_frame_callback(bid: str) -> Callable[[av.VideoFrame], av.VideoFrame]:
    """Return immediately after enqueue — ``recognize_image`` runs in background thread."""

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
        _enqueue_latest(buf["_frame_q"], rgb.copy())
        return frame

    return callback
