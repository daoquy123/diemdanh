"""SQLite-backed attendance log.

Schema:

* ``students``             — (id, name, enrolled_at)
* ``student_accounts``     — (id, username=MSSV, full_name, salt_hex, password_hash_hex, created_at)
* ``attendance_sessions``  — (id, title, opens_at, closes_at, created_at)
* ``attendance``           — (id, student_name, timestamp, similarity, source, status, session_id)

Kept intentionally tiny — for a real deployment swap with PostgreSQL/MySQL via
SQLAlchemy (already in requirements.txt).
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

_PBKDF2_ITERS = 310_000


def _hash_password_pbkdf2(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS
    ).hex()


def register_student_account_connection(
    conn: sqlite3.Connection,
    username: str,
    password: str,
    full_name: str = "",
) -> tuple[bool, str]:
    """Đăng ký tài khoản sinh viên (MSSV + mật khẩu + họ tên).

    Gọi qua ``conn`` để tránh lỗi ``AttributeError`` khi tiến trình Streamlit
    giữ instance/class ``AttendanceDB`` cũ thiếu method trên class.
    """
    u = username.strip().lower()
    fn = (full_name or "").strip()
    if len(u) < 4:
        return False, "MSSV quá ngắn (tối thiểu 4 ký tự)."
    if len(fn) < 3:
        return False, "Họ tên quá ngắn."
    if len(password) < 6:
        return False, "Mật khẩu cần ít nhất 6 ký tự."
    hit = conn.execute(
        "SELECT 1 FROM student_accounts WHERE username = ? LIMIT 1",
        (u,),
    ).fetchone()
    if hit is not None:
        return False, "MSSV đã được đăng ký."
    salt = secrets.token_bytes(16)
    ph = _hash_password_pbkdf2(password, salt)
    ts = datetime.now().isoformat(timespec="seconds")
    try:
        conn.execute(
            "INSERT INTO student_accounts(username, full_name, salt_hex, password_hash_hex, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (u, fn, salt.hex(), ph, ts),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return False, "MSSV đã được đăng ký."
    return True, "Đăng ký thành công."


def verify_student_account_connection(
    conn: sqlite3.Connection, username: str, password: str
) -> bool:
    u = username.strip().lower()
    row = conn.execute(
        "SELECT salt_hex, password_hash_hex FROM student_accounts WHERE username = ?",
        (u,),
    ).fetchone()
    if row is None:
        return False
    try:
        salt = bytes.fromhex(row["salt_hex"])
    except ValueError:
        return False
    expect = _hash_password_pbkdf2(password, salt)
    return hmac.compare_digest(expect, row["password_hash_hex"])


def student_full_name_connection(conn: sqlite3.Connection, username: str) -> str | None:
    u = username.strip().lower()
    row = conn.execute(
        "SELECT full_name FROM student_accounts WHERE username = ?", (u,)
    ).fetchone()
    if row is None:
        return None
    fn = (row["full_name"] or "").strip()
    return fn or None


class AttendanceDB:
    """Lightweight SQLite store for attendance events."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        enrolled_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_name TEXT NOT NULL,
        timestamp   TEXT NOT NULL,
        similarity  REAL NOT NULL,
        source      TEXT,
        status      TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_name);
    CREATE INDEX IF NOT EXISTS idx_attendance_ts ON attendance(timestamp);
    """

    def __init__(self, path: str | Path = "attendance.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()
        self._migrate_sessions()
        self._migrate_student_accounts()

    # ----- students ---------------------------------------------------------
    def _migrate_student_accounts(self) -> None:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='student_accounts'"
        )
        if cur.fetchone() is None:
            self.conn.execute(
                """
                CREATE TABLE student_accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL DEFAULT '',
                    salt_hex TEXT NOT NULL,
                    password_hash_hex TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_student_accounts_user "
                "ON student_accounts(username)"
            )
            self.conn.commit()
        cols_acc = [r[1] for r in self.conn.execute("PRAGMA table_info(student_accounts)").fetchall()]
        if "full_name" not in cols_acc:
            self.conn.execute("ALTER TABLE student_accounts ADD COLUMN full_name TEXT NOT NULL DEFAULT ''")
            self.conn.commit()

    def _migrate_sessions(self) -> None:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='attendance_sessions'"
        )
        if cur.fetchone() is None:
            self.conn.execute(
                """
                CREATE TABLE attendance_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    opens_at TEXT NOT NULL,
                    closes_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_opens ON attendance_sessions(opens_at)"
            )
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(attendance)").fetchall()]
        if "session_id" not in cols:
            self.conn.execute("ALTER TABLE attendance ADD COLUMN session_id INTEGER")
        self.conn.commit()

    def upsert_student(self, name: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO students(name, enrolled_at) VALUES (?, ?)",
            (name, datetime.utcnow().isoformat()),
        )
        self.conn.commit()

    def student_row(self, name: str) -> dict | None:
        n = name.strip().lower()
        row = self.conn.execute(
            "SELECT * FROM students WHERE lower(trim(name)) = ?", (n,)
        ).fetchone()
        return dict(row) if row else None

    def list_students(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM students ORDER BY name ASC").fetchall()
        return [dict(r) for r in rows]

    def list_students_enriched(self) -> list[dict]:
        """Danh sách sinh viên: MSSV/mã trong ``students`` + họ tên từ ``student_accounts`` nếu có."""
        rows = self.conn.execute(
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

    def list_attendance_enriched(self, limit: int = 500, session_id: int | None = None) -> list[dict]:
        """Lịch sử điểm danh kèm thông tin phiên (nếu có)."""
        if session_id is None:
            rows = self.conn.execute(
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
            rows = self.conn.execute(
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

    # ----- student login (self-service) -------------------------------------
    def student_account_exists(self, username: str) -> bool:
        u = username.strip().lower()
        row = self.conn.execute(
            "SELECT 1 FROM student_accounts WHERE username = ?", (u,)
        ).fetchone()
        return row is not None

    def register_student_account(
        self, username: str, password: str, full_name: str = ""
    ) -> tuple[bool, str]:
        """Tạo tài khoản sinh viên. ``username`` là MSSV (đã chuẩn hoá). ``full_name`` là họ đệm tên."""
        return register_student_account_connection(self.conn, username, password, full_name)

    def student_full_name(self, username: str) -> str | None:
        return student_full_name_connection(self.conn, username)

    def verify_student_account(self, username: str, password: str) -> bool:
        return verify_student_account_connection(self.conn, username, password)

    # ----- attendance sessions ----------------------------------------------
    def create_attendance_session(
        self,
        opens_at: datetime,
        closes_at: datetime,
        title: str | None = None,
    ) -> int:
        now = datetime.now().isoformat(timespec="seconds")
        cur = self.conn.execute(
            "INSERT INTO attendance_sessions(title, opens_at, closes_at, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                title or "",
                opens_at.isoformat(timespec="seconds"),
                closes_at.isoformat(timespec="seconds"),
                now,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_recent_sessions(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM attendance_sessions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def open_session_for_now(self, when: datetime | None = None) -> dict | None:
        """Return the session that ``when`` falls into (inclusive bounds), if any."""
        when = when or datetime.now()
        ts = when.isoformat(timespec="seconds")
        row = self.conn.execute(
            "SELECT * FROM attendance_sessions WHERE opens_at <= ? AND closes_at >= ? "
            "ORDER BY id DESC LIMIT 1",
            (ts, ts),
        ).fetchone()
        return dict(row) if row else None

    # ----- attendance -------------------------------------------------------
    def log(
        self,
        student_name: str,
        similarity: float,
        source: str = "live",
        status: str = "present",
        cooldown_minutes: int = 5,
        session_id: int | None = None,
    ) -> bool:
        """Insert an attendance event. Returns False if a record for the same
        student exists within the cooldown window (avoids spam).

        When ``session_id`` is set, cooldown is scoped to that session only
        (re-attend same session won't duplicate; a new session allows a new log).
        """
        now = datetime.utcnow()
        if cooldown_minutes > 0:
            if session_id is not None:
                last = self.conn.execute(
                    "SELECT timestamp FROM attendance WHERE student_name=? AND session_id=? "
                    "ORDER BY id DESC LIMIT 1",
                    (student_name, session_id),
                ).fetchone()
            else:
                last = self.conn.execute(
                    "SELECT timestamp FROM attendance WHERE student_name=? "
                    "ORDER BY id DESC LIMIT 1",
                    (student_name,),
                ).fetchone()
            if last is not None:
                try:
                    last_ts = datetime.fromisoformat(last["timestamp"])
                    if (now - last_ts).total_seconds() < cooldown_minutes * 60:
                        return False
                except ValueError:
                    pass
        self.conn.execute(
            "INSERT INTO attendance(student_name, timestamp, similarity, source, status, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (student_name, now.isoformat(), float(similarity), source, status, session_id),
        )
        self.conn.commit()
        return True

    def recent(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM attendance ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def report_for_date(self, day: str) -> list[dict]:
        """``day`` in ``YYYY-MM-DD`` (UTC)."""
        rows = self.conn.execute(
            "SELECT * FROM attendance WHERE substr(timestamp, 1, 10) = ? ORDER BY timestamp ASC",
            (day,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()
