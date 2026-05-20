from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.db import AttendanceDB, FaissEmbeddingStore
from src.detection import build_detector
from src.pipeline import AttendancePipeline
from src.recognition import EmbeddingExtractor
from src.utils import load_config

ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = ROOT / "webapp"

DEFAULT_DETECTION_CFG = ROOT / "configs/detection/retinaface.yaml"
DEFAULT_RECOGNITION_CFG = ROOT / "configs/recognition/facenet_highacc.yaml"
DEFAULT_RECOGNITION_WEIGHTS = ROOT / "weights/facenet_highacc/finetuned_custom/best.pth"
DEFAULT_GALLERY = ROOT / "embeddings_db"
DEFAULT_DB = ROOT / "attendance.db"
# Ngưỡng cosine: báo cáo evaluate ~0.6 trên val; webcam enroll thường cần ~0.65–0.75
DEFAULT_THRESHOLD = 0.68
MIN_MARGIN_TOP1_TOP2 = 0.06  # chênh top1-top2 tối thiểu để tránh nhầm người

app = FastAPI(title="Diem Danh Sinh Vien", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(WEB_ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(WEB_ROOT / "templates"))

_pipeline_lock = threading.Lock()
_gallery_write_lock = threading.Lock()
_pipeline_cached: AttendancePipeline | None = None


def _norm_mssv(mssv: str) -> str:
    return mssv.strip().lower()


def _load_pipeline() -> AttendancePipeline:
    global _pipeline_cached
    with _pipeline_lock:
        if _pipeline_cached is not None:
            return _pipeline_cached
        det_cfg = load_config(str(DEFAULT_DETECTION_CFG))
        # Webcam tối / mặt nhỏ: hạ ngưỡng detect một chút so với batch offline.
        det_cfg["det_thresh"] = min(float(det_cfg.get("det_thresh", 0.5)), 0.4)
        rec_cfg = load_config(str(DEFAULT_RECOGNITION_CFG))
        detector = build_detector(det_cfg)
        weights = str(DEFAULT_RECOGNITION_WEIGHTS) if DEFAULT_RECOGNITION_WEIGHTS.exists() else None
        embedder = EmbeddingExtractor.from_config(rec_cfg, weights=weights)
        gallery = FaissEmbeddingStore(
            embedding_dim=int(rec_cfg["model"]["embedding_dim"]),
            root=str(DEFAULT_GALLERY),
        )
        _pipeline_cached = AttendancePipeline(
            detector=detector,
            embedder=embedder,
            gallery=gallery,
            anti_spoof=None,
            db=None,
            threshold=DEFAULT_THRESHOLD,
            align_size=int(det_cfg.get("align_size", 112)),
            antispoof_size=80,
        )
        return _pipeline_cached


def _reload_gallery(pipe: AttendancePipeline) -> None:
    """Đọc lại gallery từ đĩa sau khi enroll — tránh cache cũ."""
    pipe.gallery = FaissEmbeddingStore(
        embedding_dim=pipe.embedder.model.embedding_dim,
        root=str(DEFAULT_GALLERY),
    )


def _decode_image(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _enhance_webcam(rgb: np.ndarray) -> np.ndarray:
    """Tăng tương phản nhẹ cho phòng tối (webcam hay bị under-expose)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    return np.clip(enhanced.astype(np.float32) * 1.12, 0, 255).astype(np.uint8)


def _save_frame(mssv: str, idx: int, image_rgb: np.ndarray) -> str:
    out_dir = ROOT / "data" / "raw" / "custom" / mssv
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = out_dir / f"{mssv}_{idx:04d}_{ts}.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return str(out_path)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "admin.html", {})


@app.get("/attend", response_class=HTMLResponse)
async def attend_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "attend.html", {})


@app.post("/api/enroll")
async def enroll_student(
    full_name: str = Form(...),
    mssv: str = Form(...),
    frames: list[UploadFile] = File(...),
) -> JSONResponse:
    norm_mssv = _norm_mssv(mssv)
    if len(full_name.strip()) < 3:
        return JSONResponse({"ok": False, "message": "Ho ten khong hop le."}, status_code=400)
    if len(norm_mssv) < 4:
        return JSONResponse({"ok": False, "message": "MSSV khong hop le."}, status_code=400)
    if len(frames) < 25:
        return JSONResponse({"ok": False, "message": "Can it nhat 25 frame."}, status_code=400)

    pipe = _load_pipeline()
    db = AttendanceDB(str(DEFAULT_DB))
    if norm_mssv in {_norm_mssv(x) for x in pipe.gallery.unique_identities}:
        return JSONResponse({"ok": False, "message": "MSSV da ton tai."}, status_code=409)
    if db.student_row(norm_mssv) is not None:
        return JSONResponse({"ok": False, "message": "MSSV da ton tai."}, status_code=409)

    crops: list[np.ndarray] = []
    paths: list[str] = []
    accepted = 0
    for i, upload in enumerate(frames):
        raw = await upload.read()
        rgb = _decode_image(raw)
        if rgb is None:
            continue
        det_pairs = pipe.detect_and_align(rgb)
        if len(det_pairs) != 1:
            continue
        _, crop = det_pairs[0]
        crops.append(crop)
        paths.append(_save_frame(norm_mssv, accepted, rgb))
        accepted += 1
        if accepted >= 100:
            break

    if accepted < 25:
        return JSONResponse(
            {"ok": False, "message": f"Chi lay duoc {accepted} frame hop le. Thu lai."},
            status_code=400,
        )

    emb = pipe.embedder.encode(crops)
    with _gallery_write_lock:
        pipe.gallery = FaissEmbeddingStore(
            embedding_dim=pipe.embedder.model.embedding_dim,
            root=str(DEFAULT_GALLERY),
        )
        pipe.gallery.add(emb, [norm_mssv] * emb.shape[0], paths)
        pipe.gallery.save()
        db.upsert_student(norm_mssv)
    _reload_gallery(pipe)
    return JSONResponse(
        {
            "ok": True,
            "message": f"Da them sinh vien moi {full_name.strip()} mssv: {norm_mssv}.",
            "accepted_frames": accepted,
        }
    )


@app.post("/api/recognize")
async def recognize(
    frame: UploadFile = File(...),
    compare_all: bool = False,
) -> JSONResponse:
    raw = await frame.read()
    rgb = _decode_image(raw)
    if rgb is None:
        return JSONResponse({"ok": False, "message": "Khong doc duoc anh."}, status_code=400)
    pipe = _load_pipeline()
    _reload_gallery(pipe)
    results = pipe.recognize_image(rgb)
    if not results:
        results = pipe.recognize_image(_enhance_webcam(rgb))

    name_labels: dict[str, str] = {}
    if compare_all:
        db = AttendanceDB(str(DEFAULT_DB))
        for row in db.list_students_enriched():
            mssv = str(row.get("mssv", "")).strip().lower()
            ho_ten = str(row.get("ho_ten", "")).strip()
            if mssv:
                name_labels[mssv] = ho_ten

    payload = []
    for r in results:
        top = [{"name": n, "similarity": round(float(s), 4)} for n, s in r.top_matches[:3]]
        margin = 0.0
        if len(r.top_matches) >= 2:
            margin = float(r.top_matches[0][1] - r.top_matches[1][1])
        item = {
            "name": r.name,
            "similarity": round(float(r.similarity), 4),
            "margin": round(margin, 4),
            "detection_score": round(float(r.detection_score), 4),
            "bbox": [float(x) for x in r.bbox.tolist()],
            "top_matches": top,
        }
        if compare_all:
            ranking = []
            for n, sim, _, proto, mx in pipe.gallery.rank_all_identities(r.embedding):
                key = n.strip().lower()
                entry: dict = {
                    "name": n,
                    "similarity": round(float(sim), 4),
                    "similarity_proto": round(float(proto), 4),
                    "similarity_max_frame": round(float(mx), 4),
                }
                label = name_labels.get(key, "")
                if label:
                    entry["full_name"] = label
                ranking.append(entry)
            item["gallery_ranking"] = ranking
        payload.append(item)

    gallery_count = len(pipe.gallery.unique_identities)
    return JSONResponse(
        {
            "ok": True,
            "results": payload,
            "face_count": len(payload),
            "gallery_count": gallery_count,
            "threshold": DEFAULT_THRESHOLD,
            "min_margin": MIN_MARGIN_TOP1_TOP2,
            "match_method": "hybrid_trim_proto_max_frame_tta",
        }
    )
