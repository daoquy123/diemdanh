"""Ứng dụng điểm danh sinh viên — giao diện rút gọn.

* **Admin** (mật khẩu cấu hình qua biến môi trường hoặc Streamlit Secrets, không lưu trong repo):
  * Thêm sinh viên: nhập tên → **camera trình duyệt** (WebRTC) ghi tối đa 100 khung rõ trong 30 giây.
  * Mở phiên điểm danh: chọn thời điểm bắt đầu + thời lượng (ví dụ 5 phút).
* **Sinh viên**: đăng ký / điểm danh qua **camera trình duyệt** (WebRTC; cần HTTPS như ngrok).

Chạy: ``streamlit run app/streamlit_app.py``

Mật khẩu admin: tạo file ``.streamlit/secrets.toml`` (xem ``.streamlit/secrets.toml.example``)
hoặc đặt biến môi trường ``ADMIN_PASSWORD``.
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from app.webcam_client import (
    RTC_ICE,
    WEBCAM_MEDIA,
    attend_frame_callback,
    enroll_frame_callback,
    get_buffer,
    init_attend_buffer,
    init_enroll_buffer,
    register_drop,
)

from src.db import AttendanceDB, FaissEmbeddingStore
from src.db.attendance_db import (
    register_student_account_connection,
    student_full_name_connection,
    verify_student_account_connection,
)
from src.detection import build_detector
from src.pipeline import AttendancePipeline
from src.recognition import EmbeddingExtractor
from src.utils import load_config

# --- Cấu hình mặc định (ẩn khỏi sinh viên; chỉnh trong code nếu cần) ------------
DEFAULT_DETECTION_CFG = ROOT / "configs/detection/retinaface.yaml"
DEFAULT_RECOGNITION_CFG = ROOT / "configs/recognition/facenet_highacc.yaml"
DEFAULT_RECOGNITION_WEIGHTS = ROOT / "weights/facenet_highacc/finetuned_custom/best.pth"
DEFAULT_GALLERY = ROOT / "embeddings_db"
DEFAULT_DB = ROOT / "attendance.db"
DEFAULT_THRESHOLD = 0.85

ENROLL_TARGET_FRAMES = 100
ENROLL_MAX_SECONDS = 30
ENROLL_MIN_FRAMES = 25
DET_SCORE_MIN = 0.45

# Điểm danh trực tiếp: quét camera trong khoảng thời gian này, cần khung liên tiếp khớp MSSV
ATTEND_LIVE_SECONDS = 15.0
ATTEND_MATCH_STREAK = 2
# Sau khi mở camera: chờ tối thiểu bấy nhiêu giây mới bắt đầu tính khớp / cho phép hoàn tất điểm danh
ATTEND_WARMUP_SECONDS = 2.0

st.set_page_config(page_title="Điểm danh sinh viên", layout="centered")


def _admin_password() -> str:
    """Không hardcode mật khẩu trong repo — chỉ env hoặc ``st.secrets``."""
    env = os.environ.get("ADMIN_PASSWORD", "").strip()
    if env:
        return env
    try:
        return str(st.secrets["ADMIN_PASSWORD"]).strip()
    except (FileNotFoundError, KeyError):
        return ""


def _norm_name(name: str) -> str:
    return name.strip().lower()


def _norm_mssv(mssv: str) -> str:
    """MSSV dùng làm khóa đăng nhập / gallery (chữ thường, không khoảng trắng)."""
    return mssv.strip().lower()


def _clear_reg_session() -> None:
    for k in ("reg_step", "reg_full_name", "reg_mssv", "reg_pending_emb", "reg_pending_paths"):
        st.session_state.pop(k, None)


def _mssv_taken(pipe: AttendancePipeline, db: AttendanceDB, mssv: str) -> bool:
    u = _norm_mssv(mssv)
    if not u:
        return False
    # Truy vấn trực tiếp (tránh lỗi khi Streamlit chưa reload class DB cũ thiếu method).
    try:
        hit = db.conn.execute(
            "SELECT 1 FROM student_accounts WHERE username = ? LIMIT 1",
            (u,),
        ).fetchone()
    except sqlite3.OperationalError:
        hit = None
    if hit is not None:
        return True
    if u in {_norm_name(x) for x in pipe.gallery.unique_identities}:
        return True
    if db.student_row(u) is not None:
        return True
    return False


@st.cache_resource(show_spinner="Đang tải mô hình…")
def _load_pipeline_cached(
    det_path: str,
    rec_path: str,
    rec_weights: str | None,
    gallery_root: str,
) -> AttendancePipeline:
    det_cfg = load_config(det_path)
    rec_cfg = load_config(rec_path)
    detector = build_detector(det_cfg)
    embedder = EmbeddingExtractor.from_config(rec_cfg, weights=rec_weights)
    gallery = FaissEmbeddingStore(
        embedding_dim=int(rec_cfg["model"]["embedding_dim"]),
        root=gallery_root,
    )
    align_size = int(det_cfg.get("align_size", 112))
    return AttendancePipeline(
        detector=detector,
        embedder=embedder,
        gallery=gallery,
        anti_spoof=None,
        db=None,
        threshold=DEFAULT_THRESHOLD,
        align_size=align_size,
        antispoof_size=80,
    )


def _get_pipeline() -> AttendancePipeline:
    w = str(DEFAULT_RECOGNITION_WEIGHTS) if DEFAULT_RECOGNITION_WEIGHTS.exists() else None
    pipe = _load_pipeline_cached(
        str(DEFAULT_DETECTION_CFG),
        str(DEFAULT_RECOGNITION_CFG),
        w,
        str(DEFAULT_GALLERY),
    )
    pipe.threshold = DEFAULT_THRESHOLD
    pipe.gallery = FaissEmbeddingStore(
        embedding_dim=pipe.embedder.model.embedding_dim,
        root=str(DEFAULT_GALLERY),
    )
    return pipe


def _get_db() -> AttendanceDB:
    return AttendanceDB(str(DEFAULT_DB))


def _db_list_students_enriched(conn: sqlite3.Connection) -> list[dict]:
    """Truy vấn qua ``conn`` — tránh lỗi khi class ``AttendanceDB`` trong bộ nhớ cũ thiếu method."""
    rows = conn.execute(
        """
        SELECT
            s.id AS id,
            s.name AS mssv,
            s.enrolled_at AS enrolled_at,
            COALESCE(NULLIF(TRIM(a.full_name), ''), '') AS ho_ten,
            CASE WHEN a.username IS NOT NULL THEN 1 ELSE 0 END AS da_dang_ky_tk
        FROM students s
        LEFT JOIN student_accounts a ON lower(trim(s.name)) = a.username
        ORDER BY s.name ASC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def _db_list_attendance_enriched(
    conn: sqlite3.Connection, limit: int = 500, session_id: int | None = None
) -> list[dict]:
    if session_id is None:
        rows = conn.execute(
            """
            SELECT
                a.id AS id,
                a.student_name AS mssv,
                a.timestamp AS thoi_diem,
                a.similarity AS do_tuong_dong,
                a.source AS nguon,
                a.status AS trang_thai,
                a.session_id AS ma_phien,
                sess.title AS ten_phien,
                sess.opens_at AS phien_bat_dau,
                sess.closes_at AS phien_ket_thuc
            FROM attendance a
            LEFT JOIN attendance_sessions sess ON a.session_id = sess.id
            ORDER BY a.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT
                a.id AS id,
                a.student_name AS mssv,
                a.timestamp AS thoi_diem,
                a.similarity AS do_tuong_dong,
                a.source AS nguon,
                a.status AS trang_thai,
                a.session_id AS ma_phien,
                sess.title AS ten_phien,
                sess.opens_at AS phien_bat_dau,
                sess.closes_at AS phien_ket_thuc
            FROM attendance a
            LEFT JOIN attendance_sessions sess ON a.session_id = sess.id
            WHERE a.session_id = ?
            ORDER BY a.id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def _db_already_attended_session(conn: sqlite3.Connection, mssv: str, session_id: int) -> bool:
    u = _norm_mssv(mssv)
    row = conn.execute(
        "SELECT 1 FROM attendance WHERE lower(trim(student_name)) = ? AND session_id = ? LIMIT 1",
        (u, int(session_id)),
    ).fetchone()
    return row is not None


def _identity_exists(pipe: AttendancePipeline, db: AttendanceDB, name: str) -> bool:
    n = _norm_name(name)
    if not n:
        return False
    if n in {_norm_name(x) for x in pipe.gallery.unique_identities}:
        return True
    row = db.conn.execute(
        "SELECT 1 FROM students WHERE lower(trim(name)) = ?", (n,)
    ).fetchone()
    return row is not None


@st.fragment(run_every=timedelta(milliseconds=300))
def _fragment_poll_admin_enroll() -> None:
    bid = st.session_state.get("admin_enroll_bid")
    if not bid:
        return
    buf = get_buffer(bid)
    if not buf:
        return
    pipe = _get_pipeline()
    name = str(st.session_state.get("admin_enroll_name", ""))
    with buf["lock"]:
        n = len(buf["crops"])
        done = buf["done"]
        elapsed = time.time() - buf["t0"]
    st.progress(min(n / ENROLL_TARGET_FRAMES, 1.0))
    st.caption(
        f"Đã thu **{n}** / {ENROLL_TARGET_FRAMES} khung · {elapsed:.1f}s / {ENROLL_MAX_SECONDS}s — "
        "nhìn thẳng camera trình duyệt, đủ sáng, hơi nghiêng hai bên."
    )
    if not done and elapsed < ENROLL_MAX_SECONDS and n < ENROLL_TARGET_FRAMES:
        return
    with buf["lock"]:
        crops = list(buf["crops"])
        paths = list(buf["paths"])
    register_drop(bid)
    st.session_state.pop("admin_enroll_bid", None)
    st.session_state.pop("admin_enroll_name", None)
    err: str | None = None
    if len(crops) < ENROLL_MIN_FRAMES:
        err = (
            f"Hình ảnh không đủ rõ hoặc không thấy bạn trong khung quá lâu. "
            f"Chỉ thu được {len(crops)} khung (cần ít nhất {ENROLL_MIN_FRAMES}). Hãy thử lại."
        )
    elif len(crops) < ENROLL_TARGET_FRAMES and elapsed >= ENROLL_MAX_SECONDS:
        err = (
            f"Chưa đủ {ENROLL_TARGET_FRAMES} khung trong {ENROLL_MAX_SECONDS}s "
            f"(hiện có {len(crops)}). Quay chậm, giữ trong khung hình, đủ sáng."
        )
    if err:
        st.session_state["admin_enroll_result"] = {"err": err}
    else:
        emb = pipe.embedder.encode(crops)
        pipe.gallery.add(emb, [name] * len(crops), paths)
        pipe.gallery.save()
        _get_db().upsert_student(name)
        st.session_state["admin_enroll_result"] = {"err": None, "name": name, "n": len(crops)}
    st.rerun()


@st.fragment(run_every=timedelta(milliseconds=300))
def _fragment_poll_student_reg_enroll() -> None:
    bid = st.session_state.get("stu_reg_enroll_bid")
    if not bid:
        return
    buf = get_buffer(bid)
    if not buf:
        return
    pipe = _get_pipeline()
    u = str(st.session_state.get("stu_reg_enroll_mssv", ""))
    with buf["lock"]:
        n = len(buf["crops"])
        done = buf["done"]
        elapsed = time.time() - buf["t0"]
    st.progress(min(n / ENROLL_TARGET_FRAMES, 1.0))
    st.caption(
        f"Đã thu **{n}** / {ENROLL_TARGET_FRAMES} khung · {elapsed:.1f}s / {ENROLL_MAX_SECONDS}s — "
        "nhìn thẳng camera trình duyệt."
    )
    if not done and elapsed < ENROLL_MAX_SECONDS and n < ENROLL_TARGET_FRAMES:
        return
    with buf["lock"]:
        crops = list(buf["crops"])
        paths = list(buf["paths"])
    register_drop(bid)
    st.session_state.pop("stu_reg_enroll_bid", None)
    st.session_state.pop("stu_reg_enroll_mssv", None)
    err: str | None = None
    if len(crops) < ENROLL_MIN_FRAMES:
        err = (
            f"Hình ảnh không đủ rõ. Chỉ thu được {len(crops)} khung (cần ít nhất {ENROLL_MIN_FRAMES}). "
            "Hãy thử lại."
        )
    elif len(crops) < ENROLL_TARGET_FRAMES and elapsed >= ENROLL_MAX_SECONDS:
        err = f"Chưa đủ {ENROLL_TARGET_FRAMES} khung trong {ENROLL_MAX_SECONDS}s."
    if err:
        st.session_state["stu_reg_enroll_result"] = {"err": err}
    else:
        emb = pipe.embedder.encode(crops)
        st.session_state["reg_pending_emb"] = emb
        st.session_state["reg_pending_paths"] = paths
        st.session_state["reg_step"] = 3
    st.rerun()


@st.fragment(run_every=timedelta(milliseconds=250))
def _fragment_poll_student_attend() -> None:
    bid = st.session_state.get("attend_bid")
    if not bid:
        return
    buf = get_buffer(bid)
    if not buf:
        return
    pipe = _get_pipeline()
    pipe.db = _get_db()
    sid = int(st.session_state.get("attend_sid", 0))
    live_s = float(buf["live_seconds"])
    with buf["lock"]:
        pending = buf.get("pending_result")
        done = buf["done"]
        outcome = buf["outcome"]
        elapsed = time.time() - buf["t0"]
        streak = buf["streak"]
        warm = buf["warmup_seconds"]
        need = buf["streak_needed"]

    warm_left = max(0.0, warm - elapsed)
    left = max(0.0, live_s - elapsed)
    if warm_left > 0:
        st.caption(f"Khởi động camera… còn **{warm_left:.1f}s** · tổng **{left:.1f}s**")
    else:
        st.caption(f"Đang nhận diện… còn **{left:.1f}s** · khớp liên tiếp **{streak}** / {need}")
    st.progress(min(elapsed / live_s, 0.99))

    if pending is not None:
        logged = pipe.log_results(
            [pending],
            source="streamlit-webrtc",
            require_real=False,
            cooldown_minutes=0,
            session_id=sid,
        )
        with buf["lock"]:
            buf["pending_result"] = None
            buf["done"] = True
            buf["outcome"] = "logged" if logged else "duplicate"
        register_drop(bid)
        st.session_state.pop("attend_bid", None)
        st.session_state.pop("attend_sid", None)
        st.session_state["stu_attend_result"] = "logged" if logged else "duplicate"
        st.rerun()
        return

    if done and outcome == "none" and elapsed >= live_s - 1e-6:
        register_drop(bid)
        st.session_state.pop("attend_bid", None)
        st.session_state.pop("attend_sid", None)
        st.session_state["stu_attend_result"] = "none"
        st.rerun()


def _ui_home() -> None:
    st.title("Điểm danh sinh viên")
    open_sess = _get_db().open_session_for_now()
    if open_sess:
        closes = open_sess["closes_at"]
        st.success(f"Có phiên điểm danh đang mở (đến **{closes}**).")
    else:
        st.info("Hiện không có phiên điểm danh nào đang mở.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Đăng nhập Admin", type="primary", use_container_width=True):
            st.session_state["view"] = "admin_login"
            st.rerun()
    with c2:
        if st.button("Sinh viên điểm danh", use_container_width=True):
            st.session_state["view"] = "student_portal"
            st.rerun()


def _ui_admin_login() -> None:
    st.title("Admin")
    pwd_cfg = _admin_password()
    if not pwd_cfg:
        st.error(
            "Chưa cấu hình mật khẩu admin. "
            "Tạo file `.streamlit/secrets.toml` với khóa `ADMIN_PASSWORD`, "
            "hoặc đặt biến môi trường `ADMIN_PASSWORD` (xem `.streamlit/secrets.toml.example`)."
        )
        if st.button("← Quay lại"):
            st.session_state["view"] = "home"
            st.rerun()
        return

    pw = st.text_input("Mật khẩu", type="password")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Đăng nhập", type="primary"):
            if pw == pwd_cfg:
                st.session_state["admin_authed"] = True
                st.session_state["view"] = "admin_menu"
                st.rerun()
            else:
                st.error("Sai mật khẩu.")
    with c2:
        if st.button("← Quay lại"):
            st.session_state["view"] = "home"
            st.rerun()


def _ui_admin_menu() -> None:
    if not st.session_state.get("admin_authed"):
        st.session_state["view"] = "admin_login"
        st.rerun()
        return

    st.title("Trang quản trị")
    db = _get_db()

    if not DEFAULT_RECOGNITION_WEIGHTS.exists():
        st.warning(
            f"Không thấy file trọng số: `{DEFAULT_RECOGNITION_WEIGHTS}`. "
            "Nhận diện có thể kém — hãy huấn luyện / đặt checkpoint đúng đường dẫn."
        )

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        if st.button("Thêm sinh viên", use_container_width=True):
            st.session_state["view"] = "admin_enroll"
            st.rerun()
    with a2:
        if st.button("Mở phiên điểm danh", use_container_width=True):
            st.session_state["view"] = "admin_session"
            st.rerun()
    with a3:
        if st.button("DS sinh viên & điểm danh", use_container_width=True):
            st.session_state["view"] = "admin_roster"
            st.rerun()
    with a4:
        if st.button("Đăng xuất", use_container_width=True):
            st.session_state["admin_authed"] = False
            st.session_state["view"] = "home"
            st.rerun()

    st.divider()
    st.subheader("Phiên gần đây")
    rows = db.list_recent_sessions(8)
    if not rows:
        st.caption("Chưa có phiên nào.")
    else:
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _ui_admin_roster() -> None:
    if not st.session_state.get("admin_authed"):
        st.session_state["view"] = "admin_login"
        st.rerun()
        return

    st.title("Danh sách sinh viên & điểm danh")
    db = _get_db()

    if st.button("← Quay lại menu"):
        st.session_state["view"] = "admin_menu"
        st.rerun()

    st.subheader("Danh sách sinh viên (đã có trên hệ thống)")
    studs = _db_list_students_enriched(db.conn)
    if not studs:
        st.info("Chưa có sinh viên nào trong cơ sở dữ liệu.")
    else:
        st.caption(f"Tổng **{len(studs)}** sinh viên. Cột *đã đăng ký tài khoản* = 1 nếu sinh viên đã tự đăng ký web.")
        st.dataframe(studs, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Đã điểm danh (lịch sử)")
    sess_opts = db.list_recent_sessions(40)
    sid_filter: int | None = None
    if sess_opts:
        labels = ["Tất cả phiên gần đây"] + [
            f"#{r['id']} — {r.get('title') or '(không tên)'} — {r['opens_at']} → {r['closes_at']}"
            for r in sess_opts
        ]
        pick = st.selectbox("Lọc theo phiên", labels, index=0)
        if pick != labels[0]:
            idx = labels.index(pick) - 1
            sid_filter = int(sess_opts[idx]["id"])
    att = _db_list_attendance_enriched(db.conn, limit=500, session_id=sid_filter)
    if not att:
        st.info("Chưa có bản ghi điểm danh nào.")
    else:
        st.caption(f"Hiển thị tối đa **{len(att)}** bản ghi mới nhất (theo bộ lọc phiên).")
        st.dataframe(att, use_container_width=True, hide_index=True)


def _ui_admin_enroll() -> None:
    if not st.session_state.get("admin_authed"):
        st.session_state["view"] = "admin_login"
        st.rerun()
        return

    st.title("Thêm sinh viên")
    db = _get_db()
    pipe = _get_pipeline()

    res_en = st.session_state.pop("admin_enroll_result", None)
    if res_en:
        if res_en.get("err"):
            st.error(res_en["err"])
        else:
            st.success(f"Đã thêm sinh viên **{res_en['name']}** ({res_en['n']} mẫu nhận diện).")
            st.session_state["view"] = "admin_menu"
            st.rerun()

    st.markdown(
        """
**Hướng dẫn quay**
- Ngồi thẳng, nhìn vào camera, đủ ánh sáng (tránh ngược sáng).
- Quay chậm trái / phải một chút để thu nhiều góc.
- Tối đa **30 giây**, hệ thống thu **100** khung hình rõ để đăng ký sinh viên.
"""
    )

    name_raw = st.text_input(
        "Mã định danh trên hệ thống (khuyến nghị: MSSV)",
        placeholder="vd: 22120001",
        help="Nên trùng MSSV sinh viên để đồng bộ với đăng ký tự phục vụ.",
    )
    name = _norm_name(name_raw)

    if name and _identity_exists(pipe, db, name):
        st.error("Sinh viên này đã tồn tại trong hệ thống (gallery hoặc danh sách).")

    if st.button("← Quay lại menu"):
        if st.session_state.get("admin_enroll_bid"):
            register_drop(st.session_state["admin_enroll_bid"])
            st.session_state.pop("admin_enroll_bid", None)
            st.session_state.pop("admin_enroll_name", None)
        st.session_state["view"] = "admin_menu"
        st.rerun()

    if not name:
        st.caption("Nhập tên trước khi bắt đầu quay.")
        return

    if _identity_exists(pipe, db, name):
        return

    if st.button("Bắt đầu ghi hình đăng ký", type="primary"):
        bid = str(uuid.uuid4())
        stash = ROOT / "data" / "raw" / "custom" / _norm_name(name)
        init_enroll_buffer(
            bid,
            pipeline=pipe,
            norm_identity=name,
            stash_dir=stash,
            target_frames=ENROLL_TARGET_FRAMES,
            max_seconds=float(ENROLL_MAX_SECONDS),
            det_score_min=DET_SCORE_MIN,
        )
        st.session_state["admin_enroll_bid"] = bid
        st.session_state["admin_enroll_name"] = name
        st.rerun()

    if st.session_state.get("admin_enroll_bid"):
        bid_ad = st.session_state["admin_enroll_bid"]
        st.info(
            "Dùng **camera trình duyệt** (máy đang mở trang). Bấm **Bật camera**, cho phép quyền; "
            "trang phải là **HTTPS** (ví dụ link ngrok)."
        )
        webrtc_streamer(
            key="webrtc_admin_enroll",
            rtc_configuration=RTC_ICE,
            media_stream_constraints=WEBCAM_MEDIA,
            video_frame_callback=enroll_frame_callback(bid_ad),
            translations={"start": "Bật camera", "stop": "Tắt camera"},
        )
        if st.button("Hủy quay", key="admin_enroll_cancel"):
            register_drop(bid_ad)
            st.session_state.pop("admin_enroll_bid", None)
            st.session_state.pop("admin_enroll_name", None)
            st.rerun()
        _fragment_poll_admin_enroll()


def _ui_admin_session() -> None:
    if not st.session_state.get("admin_authed"):
        st.session_state["view"] = "admin_login"
        st.rerun()
        return

    st.title("Mở phiên điểm danh")
    db = _get_db()

    st.caption("Chọn thời điểm bắt đầu và thời lượng mở cửa điểm danh. Sinh viên chỉ điểm danh được trong khoảng thời gian này.")

    c_d, c_t = st.columns(2)
    with c_d:
        day0 = st.date_input(
            "Ngày bắt đầu",
            value=datetime.now().date(),
            format="DD/MM/YYYY",
        )
    with c_t:
        time0 = st.time_input("Giờ bắt đầu", value=datetime.now().time())
    duration_min = st.number_input("Thời lượng (phút)", min_value=1, max_value=180, value=5, step=1)
    note = st.text_input("Ghi chú (tuỳ chọn)", placeholder="vd: Buổi sáng lớp X")

    if st.button("Tạo phiên", type="primary"):
        opens = datetime.combine(day0, time0)
        closes = opens + timedelta(minutes=float(duration_min))
        sid = db.create_attendance_session(opens, closes, title=note or None)
        st.success(
            f"Đã tạo phiên **#{sid}**: mở từ `{opens.isoformat(timespec='minutes')}` "
            f"đến `{closes.isoformat(timespec='minutes')}`."
        )

    if st.button("← Quay lại menu"):
        st.session_state["view"] = "admin_menu"
        st.rerun()


def _ui_student_portal() -> None:
    st.title("Sinh viên")
    db = _get_db()

    if st.session_state.get("student_user"):
        st.session_state["view"] = "student_attend"
        st.rerun()
        return

    tab_login, tab_reg = st.tabs(["Đăng nhập", "Đăng ký"])

    with tab_login:
        u_in = st.text_input("MSSV", key="stu_login_u", placeholder="vd: 22120001")
        p_in = st.text_input("Mật khẩu", type="password", key="stu_login_p")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Đăng nhập", type="primary", key="stu_login_btn"):
                u = _norm_mssv(u_in)
                if not u or not p_in:
                    st.error("Nhập đủ MSSV và mật khẩu.")
                elif not verify_student_account_connection(db.conn, u, p_in):
                    st.error("Sai MSSV hoặc mật khẩu.")
                else:
                    st.session_state["student_user"] = u
                    st.session_state["student_full_name"] = student_full_name_connection(db.conn, u) or ""
                    st.session_state["view"] = "student_attend"
                    st.rerun()
        with c2:
            if st.button("← Trang chủ", key="stu_login_back"):
                st.session_state["view"] = "home"
                st.rerun()

    with tab_reg:
        if "reg_step" not in st.session_state:
            st.session_state["reg_step"] = 1

        step = int(st.session_state.get("reg_step", 1))
        pipe = _get_pipeline()

        if st.button("← Trang chủ", key="stu_reg_back"):
            _clear_reg_session()
            st.session_state["reg_step"] = 1
            st.session_state["view"] = "home"
            st.rerun()

        # ----- Bước 1: họ tên + MSSV → Tiếp ---------------------------------
        if step == 1:
            st.caption("Nhập họ tên và MSSV, sau đó bấm **Tiếp** để mở camera quét mặt đăng ký.")
            fn_in = st.text_input("Họ và tên", key="reg_fn", placeholder="vd: Nguyễn Văn A")
            mssv_in = st.text_input("MSSV", key="reg_mssv_in", placeholder="vd: 22120001")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Tiếp", type="primary", key="reg_next1"):
                    fn = fn_in.strip()
                    u = _norm_mssv(mssv_in)
                    if len(fn) < 3:
                        st.error("Nhập họ tên đầy đủ (ít nhất 3 ký tự).")
                    elif not re.fullmatch(r"[a-z0-9]{4,32}", u):
                        st.error("MSSV chỉ gồm chữ và số, độ dài 4–32 ký tự (không dấu cách).")
                    elif _mssv_taken(pipe, db, u):
                        st.error("MSSV này đã tồn tại trên hệ thống. Dùng Đăng nhập hoặc MSSV khác.")
                    else:
                        st.session_state["reg_full_name"] = fn
                        st.session_state["reg_mssv"] = u
                        st.session_state["reg_step"] = 2
                        st.rerun()
            with c2:
                if st.button("Làm lại", key="reg_reset1"):
                    _clear_reg_session()
                    st.session_state["reg_step"] = 1
                    st.rerun()

        # ----- Bước 2: camera quét mặt (WebRTC — camera trình duyệt) --------
        elif step == 2:
            fn = st.session_state.get("reg_full_name", "")
            u = st.session_state.get("reg_mssv", "")
            if not fn or not u:
                st.session_state["reg_step"] = 1
                st.rerun()
            reg_flash = st.session_state.pop("stu_reg_enroll_result", None)
            if reg_flash and reg_flash.get("err"):
                st.error(reg_flash["err"])

            st.subheader("Quét mặt đăng ký")
            st.markdown(f"**Họ tên:** {fn}  ·  **MSSV:** `{u}`")
            st.caption(
                f"Camera **trình duyệt** ghi tối đa {ENROLL_TARGET_FRAMES} khung trong {ENROLL_MAX_SECONDS} giây. "
                "Bấm **Bật camera**, cho phép quyền; trang cần **HTTPS** (ngrok)."
            )
            if st.button("← Quay lại sửa thông tin", key="reg_back2"):
                if st.session_state.get("stu_reg_enroll_bid"):
                    register_drop(st.session_state["stu_reg_enroll_bid"])
                    st.session_state.pop("stu_reg_enroll_bid", None)
                    st.session_state.pop("stu_reg_enroll_mssv", None)
                st.session_state["reg_step"] = 1
                st.session_state.pop("reg_pending_emb", None)
                st.session_state.pop("reg_pending_paths", None)
                st.rerun()
            if st.button("Bắt đầu quét mặt", type="primary", key="reg_scan"):
                bid_r = str(uuid.uuid4())
                stash = ROOT / "data" / "raw" / "custom" / _norm_name(u)
                init_enroll_buffer(
                    bid_r,
                    pipeline=pipe,
                    norm_identity=u,
                    stash_dir=stash,
                    target_frames=ENROLL_TARGET_FRAMES,
                    max_seconds=float(ENROLL_MAX_SECONDS),
                    det_score_min=DET_SCORE_MIN,
                )
                st.session_state["stu_reg_enroll_bid"] = bid_r
                st.session_state["stu_reg_enroll_mssv"] = u
                st.rerun()

            if st.session_state.get("stu_reg_enroll_bid"):
                bid_r2 = st.session_state["stu_reg_enroll_bid"]
                webrtc_streamer(
                    key=f"webrtc_stu_reg_{u}",
                    rtc_configuration=RTC_ICE,
                    media_stream_constraints=WEBCAM_MEDIA,
                    video_frame_callback=enroll_frame_callback(bid_r2),
                    translations={"start": "Bật camera", "stop": "Tắt camera"},
                )
                if st.button("Hủy quét", key="stu_reg_enroll_cancel"):
                    register_drop(bid_r2)
                    st.session_state.pop("stu_reg_enroll_bid", None)
                    st.session_state.pop("stu_reg_enroll_mssv", None)
                    st.rerun()
                _fragment_poll_student_reg_enroll()

        # ----- Bước 3: đặt mật khẩu, hoàn tất --------------------------------
        else:
            fn = st.session_state.get("reg_full_name", "")
            u = st.session_state.get("reg_mssv", "")
            emb = st.session_state.get("reg_pending_emb")
            paths = st.session_state.get("reg_pending_paths")
            if not fn or not u or emb is None or not paths:
                st.warning("Thiếu dữ liệu đăng ký. Làm lại từ đầu.")
                _clear_reg_session()
                st.session_state["reg_step"] = 1
                st.rerun()

            st.subheader("Đặt mật khẩu đăng nhập")
            st.caption(f"**{fn}** — MSSV `{u}` · Đã thu {len(paths)} mẫu nhận diện.")
            p_r = st.text_input("Mật khẩu (≥6 ký tự)", type="password", key="stu_reg_p")
            p2 = st.text_input("Nhập lại mật khẩu", type="password", key="stu_reg_p2")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Hoàn tất đăng ký", type="primary", key="stu_reg_finish"):
                    if not p_r or len(p_r) < 6:
                        st.error("Mật khẩu cần ít nhất 6 ký tự.")
                    elif p_r != p2:
                        st.error("Hai lần nhập mật khẩu không khớp.")
                    else:
                        pipe.gallery.add(emb, [u] * emb.shape[0], paths)
                        pipe.gallery.save()
                        ok, msg = register_student_account_connection(db.conn, u, p_r, fn)
                        if not ok:
                            pipe.gallery.remove_identity(u)
                            pipe.gallery.save()
                            st.error(msg)
                        else:
                            db.upsert_student(u)
                            _clear_reg_session()
                            st.session_state["reg_step"] = 1
                            st.session_state["student_user"] = u
                            st.session_state["student_full_name"] = fn
                            st.success("Đăng ký thành công. Đã đăng nhập.")
                            st.session_state["view"] = "student_attend"
                            st.rerun()
            with c2:
                if st.button("Quay lại quét mặt", key="reg_back3"):
                    if st.session_state.get("stu_reg_enroll_bid"):
                        register_drop(st.session_state["stu_reg_enroll_bid"])
                        st.session_state.pop("stu_reg_enroll_bid", None)
                        st.session_state.pop("stu_reg_enroll_mssv", None)
                    st.session_state["reg_step"] = 2
                    st.session_state.pop("reg_pending_emb", None)
                    st.session_state.pop("reg_pending_paths", None)
                    st.rerun()


def _ui_student_attend() -> None:
    st.title("Điểm danh sinh viên")
    db = _get_db()
    user = st.session_state.get("student_user")
    if not user:
        st.session_state["view"] = "student_portal"
        st.rerun()
        return

    fn_disp = (st.session_state.get("student_full_name") or "").strip()
    if fn_disp:
        st.caption(f"Đã đăng nhập: **{fn_disp}** · MSSV: **{user}**")
    else:
        st.caption(f"Đã đăng nhập: MSSV **{user}**")
    st.info(
        "Điểm danh dùng **camera trình duyệt** (điện thoại / laptop của bạn). Bấm **Bật camera** và cho phép quyền; "
        "trang phải là **HTTPS** (link ngrok). Một số mạng chặn WebRTC — khi đó cần mạng khác hoặc cấu hình TURN."
    )
    c_out, _ = st.columns([1, 3])
    with c_out:
        if st.button("Đăng xuất"):
            if st.session_state.get("attend_bid"):
                register_drop(st.session_state["attend_bid"])
                st.session_state.pop("attend_bid", None)
                st.session_state.pop("attend_sid", None)
            st.session_state.pop("student_user", None)
            st.session_state.pop("student_full_name", None)
            st.session_state["view"] = "student_portal"
            st.rerun()

    sess = db.open_session_for_now()
    if not sess:
        st.warning("Hiện không có phiên điểm danh nào đang mở. Quay lại khi giảng viên mở phiên.")
        if st.button("← Trang chủ", key="stu_att_home"):
            if st.session_state.get("attend_bid"):
                register_drop(st.session_state["attend_bid"])
                st.session_state.pop("attend_bid", None)
                st.session_state.pop("attend_sid", None)
            st.session_state.pop("student_user", None)
            st.session_state.pop("student_full_name", None)
            st.session_state["view"] = "home"
            st.rerun()
        return

    sid = int(sess["id"])
    already = _db_already_attended_session(db.conn, user, sid)

    st.info(
        f"Phiên đang mở (hết hạn lúc **{sess['closes_at']}**)."
        f"{' — ' + fn_disp if fn_disp else ''}"
    )

    pipe = _get_pipeline()
    pipe.db = db

    att_flash = st.session_state.pop("stu_attend_result", None)
    if att_flash == "logged":
        label = fn_disp if fn_disp else user
        st.success(f"Điểm danh thành công — **{label}** (MSSV: {user}).")
    elif att_flash == "duplicate":
        st.warning("Bạn đã điểm danh trong phiên này rồi (không ghi thêm bản ghi mới).")
    elif att_flash == "none":
        st.error(
            "Trong thời gian quét không nhận diện đủ ổn định. Nhìn thẳng camera trình duyệt, đủ sáng, một người trong khung, rồi thử lại."
        )

    if already:
        st.success(
            "✅ **Bạn đã điểm danh trong phiên này rồi.** Không cần điểm danh lại. "
            "Đợi phiên mới hoặc liên hệ giảng viên nếu cần chỉnh sửa."
        )
    else:
        st.caption(
            f"Sau khi bấm **Bắt đầu điểm danh**: **{ATTEND_WARMUP_SECONDS:.0f}s** đầu chỉ khởi động camera, "
            f"sau đó hệ thống cần **{ATTEND_MATCH_STREAK}** khung liên tiếp nhận diện đúng MSSV trong tối đa "
            f"**{ATTEND_LIVE_SECONDS:.0f}s**."
        )
        if st.button("Bắt đầu điểm danh", type="primary"):
            if _db_already_attended_session(db.conn, user, sid):
                st.warning("Bạn đã điểm danh trong phiên này rồi.")
                st.rerun()
            bid_a = str(uuid.uuid4())
            init_attend_buffer(
                bid_a,
                pipeline=pipe,
                user=user,
                session_id=sid,
                norm_mssv=_norm_mssv,
                live_seconds=float(ATTEND_LIVE_SECONDS),
                warmup_seconds=float(ATTEND_WARMUP_SECONDS),
                streak_needed=int(ATTEND_MATCH_STREAK),
                det_score_min=DET_SCORE_MIN,
            )
            st.session_state["attend_bid"] = bid_a
            st.session_state["attend_sid"] = sid
            st.rerun()

        if st.session_state.get("attend_bid"):
            bid_at = st.session_state["attend_bid"]
            webrtc_streamer(
                key=f"webrtc_stu_attend_{_norm_mssv(user)}",
                rtc_configuration=RTC_ICE,
                media_stream_constraints=WEBCAM_MEDIA,
                video_frame_callback=attend_frame_callback(bid_at),
                translations={"start": "Bật camera", "stop": "Tắt camera"},
            )
            if st.button("Hủy điểm danh", key="stu_attend_cancel"):
                register_drop(bid_at)
                st.session_state.pop("attend_bid", None)
                st.session_state.pop("attend_sid", None)
                st.rerun()
            _fragment_poll_student_attend()

    if st.button("← Trang chủ", key="stu_attend_home"):
        if st.session_state.get("attend_bid"):
            register_drop(st.session_state["attend_bid"])
            st.session_state.pop("attend_bid", None)
            st.session_state.pop("attend_sid", None)
        st.session_state["view"] = "home"
        st.rerun()


# --------------------------------- main -------------------------------------
if "view" not in st.session_state:
    st.session_state["view"] = "home"

view = st.session_state["view"]
if view == "home":
    _ui_home()
elif view == "admin_login":
    _ui_admin_login()
elif view == "admin_menu":
    _ui_admin_menu()
elif view == "admin_enroll":
    _ui_admin_enroll()
elif view == "admin_session":
    _ui_admin_session()
elif view == "admin_roster":
    _ui_admin_roster()
elif view == "student_portal":
    _ui_student_portal()
elif view == "student_attend":
    _ui_student_attend()
else:
    st.session_state["view"] = "home"
    st.rerun()
