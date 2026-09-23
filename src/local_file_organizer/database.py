"""SQLite catalog storage. All writes target the project's own data/ dir,
never the scanned tree."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    name TEXT,
    extension TEXT,
    category TEXT,
    size_bytes INTEGER,
    modified_at TEXT,
    mime_type TEXT,
    sha256 TEXT,
    duplicate_group TEXT,

    duration_seconds REAL,
    width INTEGER,
    height INTEGER,
    fps REAL,
    video_codec TEXT,
    audio_codec TEXT,
    bitrate INTEGER,

    scan_status TEXT,
    scan_error TEXT,

    metadata_status TEXT,
    metadata_error TEXT
);
"""

# Columns added after the initial V1 schema. On an existing database that
# predates them, init_db() adds them via ALTER TABLE instead of requiring a
# full migration — reasonable for a project still in V1.
_ADDED_COLUMNS: dict[str, str] = {
    "metadata_status": "TEXT",
    "metadata_error": "TEXT",
}

_FILE_COLUMNS = [
    "path", "name", "extension", "category", "size_bytes", "modified_at",
    "mime_type", "sha256", "duplicate_group",
    "duration_seconds", "width", "height", "fps", "video_codec",
    "audio_codec", "bitrate",
    "scan_status", "scan_error",
    "metadata_status", "metadata_error",
]


@dataclass
class FileRecord:
    """A single cataloged file.

    Two independent status/error pairs, kept deliberately separate:

    - ``scan_status``/``scan_error`` cover the *basic* inventory step (can
      the file be stat'd, and later hashed for duplicate detection?). This
      is the only pair that means "this row's general metadata may be
      incomplete or wrong".
    - ``metadata_status``/``metadata_error`` cover *specialized* per-category
      extraction (currently: ffprobe for video). A failure here — e.g. a
      corrupt video ffprobe can't parse — never flips ``scan_status``, since
      the file's basic inventory (path, size, mtime, category) is still
      valid. Its value is ``"ok"`` when extraction succeeded, ``"error"``
      when it was attempted and failed, or ``"skipped"`` when the category
      has no specialized extraction (or the basic scan already failed, so
      extraction was never attempted).
    """

    path: str
    name: str = ""
    extension: str = ""
    category: str = ""
    size_bytes: int = 0
    modified_at: str = ""
    mime_type: Optional[str] = None
    sha256: Optional[str] = None
    duplicate_group: Optional[str] = None

    duration_seconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    bitrate: Optional[int] = None

    scan_status: str = "ok"
    scan_error: Optional[str] = None

    metadata_status: str = "skipped"
    metadata_error: Optional[str] = None


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(SCHEMA)
    _migrate_added_columns(conn)
    conn.commit()


def _migrate_added_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the initial schema to a pre-existing
    database, without touching any data already in it."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(files)")}
    for column, sql_type in _ADDED_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE files ADD COLUMN {column} {sql_type}")


def upsert_file(conn: sqlite3.Connection, record: FileRecord) -> None:
    """Insert or update a file row, keyed by its unique path.

    Re-scanning the same tree updates existing rows in place instead of
    creating duplicate entries.
    """
    columns = _FILE_COLUMNS
    placeholders = ", ".join(f":{col}" for col in columns)
    update_clause = ", ".join(
        f"{col} = excluded.{col}" for col in columns if col != "path"
    )
    sql = (
        f"INSERT INTO files ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT(path) DO UPDATE SET {update_clause}"
    )
    values = {col: getattr(record, col) for col in columns}
    conn.execute(sql, values)


def set_duplicate_group(conn: sqlite3.Connection, path: str, group_id: Optional[str]) -> None:
    conn.execute(
        "UPDATE files SET duplicate_group = ? WHERE path = ?", (group_id, path)
    )


def set_scan_error(conn: sqlite3.Connection, path: str, message: str) -> None:
    conn.execute(
        "UPDATE files SET scan_error = ? WHERE path = ?", (message, path)
    )


def clear_duplicate_groups(conn: sqlite3.Connection, paths: list[str]) -> None:
    if not paths:
        return
    placeholders = ", ".join("?" for _ in paths)
    conn.execute(
        f"UPDATE files SET duplicate_group = NULL WHERE path IN ({placeholders})",
        paths,
    )


def fetch_category_totals(conn: sqlite3.Connection, paths: Optional[list[str]] = None):
    if paths is None:
        cur = conn.execute(
            "SELECT category, COUNT(*) AS count, SUM(size_bytes) AS total_bytes "
            "FROM files GROUP BY category ORDER BY category"
        )
        return cur.fetchall()
    placeholders = ", ".join("?" for _ in paths)
    cur = conn.execute(
        f"SELECT category, COUNT(*) AS count, SUM(size_bytes) AS total_bytes "
        f"FROM files WHERE path IN ({placeholders}) GROUP BY category ORDER BY category",
        paths,
    )
    return cur.fetchall()


def fetch_duplicate_groups(conn: sqlite3.Connection):
    cur = conn.execute(
        "SELECT duplicate_group, COUNT(*) AS count, SUM(size_bytes) AS total_bytes, "
        "MIN(size_bytes) AS size_bytes "
        "FROM files WHERE duplicate_group IS NOT NULL "
        "GROUP BY duplicate_group ORDER BY duplicate_group"
    )
    return cur.fetchall()


def fetch_duplicate_files(conn: sqlite3.Connection, group_id: Optional[str] = None):
    if group_id is None:
        cur = conn.execute(
            "SELECT * FROM files WHERE duplicate_group IS NOT NULL "
            "ORDER BY duplicate_group, path"
        )
    else:
        cur = conn.execute(
            "SELECT * FROM files WHERE duplicate_group = ? ORDER BY path",
            (group_id,),
        )
    return cur.fetchall()


_ERROR_CONDITION = (
    "(scan_status = 'error' OR metadata_status = 'error' OR scan_error IS NOT NULL)"
)


def fetch_error_count(conn: sqlite3.Connection, paths: Optional[list[str]] = None) -> int:
    """Count files that could not be *fully* analyzed: the basic scan
    failed (scan_status='error'), specialized metadata extraction failed
    (metadata_status='error', e.g. a corrupt video ffprobe can't parse), or
    an auxiliary step such as hashing left a note in scan_error."""
    if paths is None:
        cur = conn.execute(f"SELECT COUNT(*) FROM files WHERE {_ERROR_CONDITION}")
        return cur.fetchone()[0]
    placeholders = ", ".join("?" for _ in paths)
    cur = conn.execute(
        f"SELECT COUNT(*) FROM files WHERE {_ERROR_CONDITION} AND path IN ({placeholders})",
        paths,
    )
    return cur.fetchone()[0]


def fetch_all_files(conn: sqlite3.Connection):
    cur = conn.execute("SELECT * FROM files ORDER BY path")
    return cur.fetchall()
