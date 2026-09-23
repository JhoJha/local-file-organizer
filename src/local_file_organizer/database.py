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
    similar_group TEXT,

    duration_seconds REAL,
    width INTEGER,
    height INTEGER,
    fps REAL,
    video_codec TEXT,
    audio_codec TEXT,
    bitrate INTEGER,

    perceptual_hash TEXT,
    similarity_method TEXT,

    scan_status TEXT,
    scan_error TEXT,

    metadata_status TEXT,
    metadata_error TEXT
);
CREATE TABLE IF NOT EXISTS video_frame_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    sample_index INTEGER NOT NULL,
    timestamp_seconds REAL NOT NULL,
    status TEXT NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_video_frame_samples_path ON video_frame_samples(file_path);
CREATE TABLE IF NOT EXISTS face_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    source_type TEXT NOT NULL,
    timestamp_ms REAL,
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_width INTEGER NOT NULL,
    bbox_height INTEGER NOT NULL,
    detection_confidence REAL NOT NULL,
    source_width INTEGER NOT NULL,
    source_height INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_face_observations_path ON face_observations(file_path);
CREATE TABLE IF NOT EXISTS person_cluster_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_label TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_category TEXT NOT NULL,
    observation_count INTEGER NOT NULL,
    UNIQUE(person_label, file_path)
);
CREATE INDEX IF NOT EXISTS idx_person_cluster_members_label ON person_cluster_members(person_label);
CREATE INDEX IF NOT EXISTS idx_person_cluster_members_path ON person_cluster_members(file_path);
"""

# Columns added after the initial V1 schema. On an existing database that
# predates them, init_db() adds them via ALTER TABLE instead of requiring a
# full migration — reasonable for a project still in V1.
_ADDED_COLUMNS: dict[str, str] = {
    "metadata_status": "TEXT",
    "metadata_error": "TEXT",
    # V2.1: perceptual hashing for images.
    "similar_group": "TEXT",
    "perceptual_hash": "TEXT",
    "similarity_method": "TEXT",
}

_FILE_COLUMNS = [
    "path", "name", "extension", "category", "size_bytes", "modified_at",
    "mime_type", "sha256", "duplicate_group", "similar_group",
    "duration_seconds", "width", "height", "fps", "video_codec",
    "audio_codec", "bitrate",
    "perceptual_hash", "similarity_method",
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
      extraction was never attempted). Images use this same pair for
      perceptual-hash extraction (V2.1): ``metadata_error`` holds the
      Pillow failure when a file can't be decoded as an image, which never
      flips ``scan_status`` either.
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
    similar_group: Optional[str] = None

    duration_seconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    bitrate: Optional[int] = None

    perceptual_hash: Optional[str] = None
    similarity_method: Optional[str] = None

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
    conn.executescript(SCHEMA)
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


def set_similar_group(conn: sqlite3.Connection, path: str, group_id: Optional[str]) -> None:
    conn.execute(
        "UPDATE files SET similar_group = ? WHERE path = ?", (group_id, path)
    )


def clear_similar_groups(conn: sqlite3.Connection, paths: list[str]) -> None:
    if not paths:
        return
    placeholders = ", ".join("?" for _ in paths)
    conn.execute(
        f"UPDATE files SET similar_group = NULL WHERE path IN ({placeholders})",
        paths,
    )


def fetch_similar_groups(conn: sqlite3.Connection):
    cur = conn.execute(
        "SELECT similar_group, COUNT(*) AS count FROM files "
        "WHERE similar_group IS NOT NULL "
        "GROUP BY similar_group ORDER BY similar_group"
    )
    return cur.fetchall()


def fetch_similar_files(conn: sqlite3.Connection, group_id: Optional[str] = None):
    if group_id is None:
        cur = conn.execute(
            "SELECT * FROM files WHERE similar_group IS NOT NULL "
            "ORDER BY similar_group, path"
        )
    else:
        cur = conn.execute(
            "SELECT * FROM files WHERE similar_group = ? ORDER BY path",
            (group_id,),
        )
    return cur.fetchall()


def fetch_perceptual_hash_count(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM files WHERE perceptual_hash IS NOT NULL")
    return cur.fetchone()[0]


def clear_video_frame_samples(conn: sqlite3.Connection, file_path: str) -> None:
    conn.execute("DELETE FROM video_frame_samples WHERE file_path = ?", (file_path,))


def insert_video_frame_samples(conn: sqlite3.Connection, file_path: str, samples: list) -> None:
    """Bulk-insert frame sampling outcomes for one video file."""
    if not samples:
        return
    conn.executemany(
        "INSERT INTO video_frame_samples "
        "(file_path, sample_index, timestamp_seconds, status, error) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (file_path, s.sample_index, s.timestamp_seconds, s.status, s.error)
            for s in samples
        ],
    )


def fetch_video_frame_sample_counts(
    conn: sqlite3.Connection, file_path: Optional[str] = None
) -> dict[str, int]:
    """Return aggregate generated/error sample counts, globally or for one file."""
    if file_path is None:
        cur = conn.execute("SELECT status, COUNT(*) AS n FROM video_frame_samples GROUP BY status")
    else:
        cur = conn.execute(
            "SELECT status, COUNT(*) AS n FROM video_frame_samples "
            "WHERE file_path = ? GROUP BY status",
            (file_path,),
        )
    counts = {"generated": 0, "error": 0}
    for row in cur.fetchall():
        if row["status"] in counts:
            counts[row["status"]] = row["n"]
    return counts


def clear_face_observations(conn: sqlite3.Connection, file_path: str) -> None:
    conn.execute("DELETE FROM face_observations WHERE file_path = ?", (file_path,))


def insert_face_observations(
    conn: sqlite3.Connection,
    file_path: str,
    source_type: str,
    observations: list,
    *,
    timestamp_ms: Optional[float] = None,
) -> None:
    """Bulk-insert face observations from one image or one video frame."""
    if not observations:
        return
    conn.executemany(
        "INSERT INTO face_observations "
        "(file_path, source_type, timestamp_ms, bbox_x, bbox_y, "
        "bbox_width, bbox_height, detection_confidence, "
        "source_width, source_height) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                file_path,
                source_type,
                timestamp_ms,
                o.bbox_x,
                o.bbox_y,
                o.bbox_width,
                o.bbox_height,
                o.detection_confidence,
                o.source_width,
                o.source_height,
            )
            for o in observations
        ],
    )


def fetch_face_observation_counts(
    conn: sqlite3.Connection, file_path: Optional[str] = None
) -> dict:
    """Return aggregate face-detection counts, globally or for one file."""
    if file_path is None:
        total = conn.execute("SELECT COUNT(*) FROM face_observations").fetchone()[0]
        files_with = conn.execute(
            "SELECT COUNT(DISTINCT file_path) FROM face_observations"
        ).fetchone()[0]
    else:
        total = conn.execute(
            "SELECT COUNT(*) FROM face_observations WHERE file_path = ?",
            (file_path,),
        ).fetchone()[0]
        files_with = (
            1
            if conn.execute(
                "SELECT COUNT(*) FROM face_observations WHERE file_path = ?",
                (file_path,),
            ).fetchone()[0]
            > 0
            else 0
        )
    return {"total": total, "files_with_faces": files_with}


def clear_all_person_cluster_members(conn: sqlite3.Connection) -> None:
    """Delete ALL person_cluster_members rows (full rebuild semantics).

    Unlike every other clear function in this codebase (all per-file),
    clustering is a global/relative operation over the whole batch, not
    something that can be updated incrementally per file — so each
    `cluster-people` run clears the whole table and rebuilds it.
    """
    conn.execute("DELETE FROM person_cluster_members")


def insert_person_cluster_members(conn: sqlite3.Connection, rows: list) -> None:
    """Bulk-insert (person_label, file_path, file_category, observation_count) tuples."""
    if not rows:
        return
    conn.executemany(
        "INSERT INTO person_cluster_members "
        "(person_label, file_path, file_category, observation_count) "
        "VALUES (?, ?, ?, ?)",
        list(rows),
    )


def fetch_person_cluster_summary(conn: sqlite3.Connection) -> list[dict]:
    """One row per distinct person_label with files/images/videos counts.

    `images`/`videos` derive from `file_category`; ordered by person_label.
    """
    cur = conn.execute(
        "SELECT person_label, "
        "COUNT(DISTINCT file_path) AS files, "
        "SUM(CASE WHEN file_category = 'image' THEN 1 ELSE 0 END) AS images, "
        "SUM(CASE WHEN file_category = 'video' THEN 1 ELSE 0 END) AS videos "
        "FROM person_cluster_members "
        "GROUP BY person_label ORDER BY person_label"
    )
    result = []
    for row in cur.fetchall():
        result.append(
            {
                "person_label": row["person_label"],
                "files": row["files"],
                "images": row["images"] or 0,
                "videos": row["videos"] or 0,
            }
        )
    return result
