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

    capture_date TEXT,
    device_make TEXT,
    device_model TEXT,
    device_software TEXT,
    gps_latitude REAL,
    gps_longitude REAL,
    gps_altitude REAL,
    orientation INTEGER,

    scan_status TEXT,
    scan_error TEXT,

    metadata_status TEXT,
    metadata_error TEXT,

    last_seen_at TEXT,
    file_state TEXT DEFAULT 'active',
    extractor_versions TEXT
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
CREATE TABLE IF NOT EXISTS person_cluster_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_label TEXT NOT NULL,
    file_path TEXT NOT NULL,
    source_type TEXT NOT NULL,
    timestamp_ms REAL,
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_width INTEGER NOT NULL,
    bbox_height INTEGER NOT NULL,
    detection_confidence REAL
);
CREATE INDEX IF NOT EXISTS idx_person_cluster_observations_label ON person_cluster_observations(person_label);
CREATE INDEX IF NOT EXISTS idx_person_cluster_observations_path ON person_cluster_observations(file_path);
CREATE TABLE IF NOT EXISTS persons (
    person_id TEXT PRIMARY KEY,
    display_name TEXT,
    created_at TEXT NOT NULL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS person_aliases (
    machine_label TEXT PRIMARY KEY,
    person_id TEXT NOT NULL,
    origin TEXT NOT NULL DEFAULT 'manual',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_person_aliases_person ON person_aliases(person_id);
CREATE TABLE IF NOT EXISTS file_person_links (
    file_path TEXT NOT NULL,
    person_id TEXT NOT NULL,
    origin TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (file_path, person_id)
);
CREATE INDEX IF NOT EXISTS idx_file_person_links_person ON file_person_links(person_id);
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
    # V3.0: EXIF/XMP enrichment via ExifTool (NULL = unknown, never a guess).
    "capture_date": "TEXT",
    "device_make": "TEXT",
    "device_model": "TEXT",
    "device_software": "TEXT",
    "gps_latitude": "REAL",
    "gps_longitude": "REAL",
    "gps_altitude": "REAL",
    "orientation": "INTEGER",
    # F1 incremental scan: fingerprint bookkeeping (NULL = pre-F1 row,
    # treated as "reprocess once").
    "last_seen_at": "TEXT",
    "file_state": "TEXT",
    "extractor_versions": "TEXT",
}

_FILE_COLUMNS = [
    "path", "name", "extension", "category", "size_bytes", "modified_at",
    "mime_type", "sha256", "duplicate_group", "similar_group",
    "duration_seconds", "width", "height", "fps", "video_codec",
    "audio_codec", "bitrate",
    "perceptual_hash", "similarity_method",
    "capture_date", "device_make", "device_model", "device_software",
    "gps_latitude", "gps_longitude", "gps_altitude", "orientation",
    "scan_status", "scan_error",
    "metadata_status", "metadata_error",
    "last_seen_at", "file_state", "extractor_versions",
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

    # V3.0 EXIF/XMP enrichment (ExifTool). NULL/None always means
    # "unknown" (stripped or never present) — the program never guesses a
    # date, device or location.
    capture_date: Optional[str] = None
    device_make: Optional[str] = None
    device_model: Optional[str] = None
    device_software: Optional[str] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude: Optional[float] = None
    orientation: Optional[int] = None

    scan_status: str = "ok"
    scan_error: Optional[str] = None

    metadata_status: str = "skipped"
    metadata_error: Optional[str] = None

    # F1 incremental bookkeeping: last_seen_at is refreshed on every
    # scan that observes the file; file_state is 'active' or 'stale'
    # (seen before, missing now — kept for audit/V5 path_history);
    # extractor_versions records which extractors produced this row.
    last_seen_at: Optional[str] = None
    file_state: str = "active"
    extractor_versions: Optional[str] = None


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        init_db(conn)
    except sqlite3.DatabaseError as exc:
        # A garbage-filled path (not a database at all) used to surface
        # as a raw traceback from deep inside init_db.
        conn.close()
        raise sqlite3.DatabaseError(f"catálogo ilegible o corrupto: {db_path}") from exc
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _migrate_added_columns(conn)
    _create_search_indexes(conn)
    conn.commit()


# V3.2 structured search: indexes over the filter fields so 50k+
# catalogs answer without full table scans. Created AFTER the column
# migration (a legacy/hand-made DB may miss the column entirely) and
# each in its own try/except: indexes are performance-only and must
# never break connect() the way a failed executescript would.
_SEARCH_INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_files_category ON files(category)",
    "CREATE INDEX IF NOT EXISTS idx_files_extension ON files(extension)",
    "CREATE INDEX IF NOT EXISTS idx_files_size_bytes ON files(size_bytes)",
    "CREATE INDEX IF NOT EXISTS idx_files_duration ON files(duration_seconds)",
    "CREATE INDEX IF NOT EXISTS idx_files_capture_date ON files(capture_date)",
    "CREATE INDEX IF NOT EXISTS idx_files_device_model ON files(device_model)",
]


def _create_search_indexes(conn: sqlite3.Connection) -> None:
    for statement in _SEARCH_INDEXES:
        try:
            conn.execute(statement)
        except sqlite3.OperationalError:
            continue


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


def set_digests(conn: sqlite3.Connection, digests: dict[str, str]) -> None:
    """Bulk-persist SHA-256 digests computed during duplicate detection
    (F1: without this the catalog could never reuse them and every scan
    would re-read every byte of every size-collision group).

    A persisted digest also retires a stale ``hash: ...`` scan_error
    note: digest present means the last known hash outcome was success.
    Any other scan_error text is left untouched.
    """
    if not digests:
        return
    conn.executemany(
        "UPDATE files SET sha256 = ?, "
        "scan_error = CASE WHEN scan_error LIKE 'hash:%' THEN NULL "
        "ELSE scan_error END "
        "WHERE path = ?",
        [(digest, path) for path, digest in digests.items()],
    )


def set_scan_error(conn: sqlite3.Connection, path: str, message: str) -> None:
    conn.execute(
        "UPDATE files SET scan_error = ? WHERE path = ?", (message, path)
    )


_SQL_CHUNK_SIZE = 500


class CategoryTotalRow:
    """Row-like object for category totals supporting both key and index access."""

    def __init__(self, category: str, count: int, total_bytes: int):
        self.category = category
        self.count = count
        self.total_bytes = total_bytes

    def __getitem__(self, key):
        if key == 0 or key == "category":
            return self.category
        if key == 1 or key == "count":
            return self.count
        if key == 2 or key == "total_bytes":
            return self.total_bytes
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def clear_duplicate_groups(conn: sqlite3.Connection, paths: list[str]) -> None:
    if not paths:
        return
    for i in range(0, len(paths), _SQL_CHUNK_SIZE):
        chunk = paths[i : i + _SQL_CHUNK_SIZE]
        placeholders = ", ".join("?" for _ in chunk)
        conn.execute(
            f"UPDATE files SET duplicate_group = NULL WHERE path IN ({placeholders})",
            chunk,
        )


def fetch_category_totals(
    conn: sqlite3.Connection,
    paths: Optional[list[str]] = None,
    *,
    active_only: bool = False,
):
    if paths is None:
        where = " WHERE COALESCE(file_state, 'active') = 'active'" if active_only else ""
        cur = conn.execute(
            f"SELECT category, COUNT(*) AS count, SUM(size_bytes) AS total_bytes "
            f"FROM files{where} GROUP BY category ORDER BY category"
        )
        return cur.fetchall()

    if not paths:
        return []

    if len(paths) <= _SQL_CHUNK_SIZE:
        placeholders = ", ".join("?" for _ in paths)
        cur = conn.execute(
            f"SELECT category, COUNT(*) AS count, SUM(size_bytes) AS total_bytes "
            f"FROM files WHERE path IN ({placeholders}) GROUP BY category ORDER BY category",
            paths,
        )
        return cur.fetchall()

    aggregated: dict[str, dict[str, int]] = {}
    for i in range(0, len(paths), _SQL_CHUNK_SIZE):
        chunk = paths[i : i + _SQL_CHUNK_SIZE]
        placeholders = ", ".join("?" for _ in chunk)
        cur = conn.execute(
            f"SELECT category, COUNT(*) AS count, SUM(size_bytes) AS total_bytes "
            f"FROM files WHERE path IN ({placeholders}) GROUP BY category",
            chunk,
        )
        for row in cur.fetchall():
            cat = row["category"]
            if cat not in aggregated:
                aggregated[cat] = {"count": 0, "total_bytes": 0}
            aggregated[cat]["count"] += row["count"]
            aggregated[cat]["total_bytes"] += row["total_bytes"] or 0

    return [
        CategoryTotalRow(cat, aggregated[cat]["count"], aggregated[cat]["total_bytes"])
        for cat in sorted(aggregated)
    ]


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


def fetch_error_count(
    conn: sqlite3.Connection,
    paths: Optional[list[str]] = None,
    *,
    active_only: bool = False,
) -> int:
    """Count files that could not be *fully* analyzed: the basic scan
    failed (scan_status='error'), specialized metadata extraction failed
    (metadata_status='error', e.g. a corrupt video ffprobe can't parse), or
    an auxiliary step such as hashing left a note in scan_error."""
    if paths is None:
        where = f"WHERE {_ERROR_CONDITION}"
        if active_only:
            where += " AND (COALESCE(file_state, 'active') = 'active')"
        cur = conn.execute(f"SELECT COUNT(*) FROM files {where}")
        return cur.fetchone()[0]

    if not paths:
        return 0

    if len(paths) <= _SQL_CHUNK_SIZE:
        placeholders = ", ".join("?" for _ in paths)
        cur = conn.execute(
            f"SELECT COUNT(*) FROM files WHERE {_ERROR_CONDITION} AND path IN ({placeholders})",
            paths,
        )
        return cur.fetchone()[0]

    total = 0
    for i in range(0, len(paths), _SQL_CHUNK_SIZE):
        chunk = paths[i : i + _SQL_CHUNK_SIZE]
        placeholders = ", ".join("?" for _ in chunk)
        cur = conn.execute(
            f"SELECT COUNT(*) FROM files WHERE {_ERROR_CONDITION} AND path IN ({placeholders})",
            chunk,
        )
        total += cur.fetchone()[0]
    return total


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
    for i in range(0, len(paths), _SQL_CHUNK_SIZE):
        chunk = paths[i : i + _SQL_CHUNK_SIZE]
        placeholders = ", ".join("?" for _ in chunk)
        conn.execute(
            f"UPDATE files SET similar_group = NULL WHERE path IN ({placeholders})",
            chunk,
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


def fetch_perceptual_hash_count(
    conn: sqlite3.Connection,
    paths: Optional[list[str]] = None,
    *,
    active_only: bool = False,
) -> int:
    if paths is None:
        where = " WHERE perceptual_hash IS NOT NULL"
        if active_only:
            where += " AND (COALESCE(file_state, 'active') = 'active')"
        cur = conn.execute(f"SELECT COUNT(*) FROM files{where}")
        return cur.fetchone()[0]
    if not paths:
        return 0
    if len(paths) <= _SQL_CHUNK_SIZE:
        placeholders = ", ".join("?" for _ in paths)
        cur = conn.execute(
            f"SELECT COUNT(*) FROM files WHERE perceptual_hash IS NOT NULL AND path IN ({placeholders})",
            paths,
        )
        return cur.fetchone()[0]

    total = 0
    for i in range(0, len(paths), _SQL_CHUNK_SIZE):
        chunk = paths[i : i + _SQL_CHUNK_SIZE]
        placeholders = ", ".join("?" for _ in chunk)
        cur = conn.execute(
            f"SELECT COUNT(*) FROM files WHERE perceptual_hash IS NOT NULL AND path IN ({placeholders})",
            chunk,
        )
        total += cur.fetchone()[0]
    return total


def fetch_exif_counts(
    conn: sqlite3.Connection,
    paths: Optional[list[str]] = None,
    *,
    active_only: bool = False,
) -> dict[str, int]:
    """Count files carrying EXIF/XMP enrichment: capture date, device
    (make or model) and GPS coordinates. Always scoped to the current
    scan when ``paths`` is given, like the other fetch helpers."""
    if paths is None:
        scope = " WHERE (COALESCE(file_state, 'active') = 'active')" if active_only else ""
        pfx = " AND" if scope else " WHERE"
        counts = {}
        counts["with_capture_date"] = conn.execute(
            f"SELECT COUNT(*) FROM files{scope}{pfx} capture_date IS NOT NULL"
        ).fetchone()[0]
        counts["with_device"] = conn.execute(
            f"SELECT COUNT(*) FROM files{scope}{pfx} (device_make IS NOT NULL OR device_model IS NOT NULL)"
        ).fetchone()[0]
        counts["with_gps"] = conn.execute(
            f"SELECT COUNT(*) FROM files{scope}{pfx} (gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL)"
        ).fetchone()[0]
        return counts

    if not paths:
        return {"with_capture_date": 0, "with_device": 0, "with_gps": 0}

    if len(paths) <= _SQL_CHUNK_SIZE:
        placeholders = ", ".join("?" for _ in paths)
        scope = f" AND path IN ({placeholders})"
        counts = {}
        counts["with_capture_date"] = conn.execute(
            f"SELECT COUNT(*) FROM files WHERE capture_date IS NOT NULL{scope}",
            paths,
        ).fetchone()[0]
        counts["with_device"] = conn.execute(
            f"SELECT COUNT(*) FROM files "
            f"WHERE (device_make IS NOT NULL OR device_model IS NOT NULL){scope}",
            paths,
        ).fetchone()[0]
        counts["with_gps"] = conn.execute(
            f"SELECT COUNT(*) FROM files "
            f"WHERE gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL{scope}",
            paths,
        ).fetchone()[0]
        return counts

    counts = {"with_capture_date": 0, "with_device": 0, "with_gps": 0}
    for i in range(0, len(paths), _SQL_CHUNK_SIZE):
        chunk = paths[i : i + _SQL_CHUNK_SIZE]
        placeholders = ", ".join("?" for _ in chunk)
        scope = f" AND path IN ({placeholders})"
        counts["with_capture_date"] += conn.execute(
            f"SELECT COUNT(*) FROM files WHERE capture_date IS NOT NULL{scope}",
            chunk,
        ).fetchone()[0]
        counts["with_device"] += conn.execute(
            f"SELECT COUNT(*) FROM files "
            f"WHERE (device_make IS NOT NULL OR device_model IS NOT NULL){scope}",
            chunk,
        ).fetchone()[0]
        counts["with_gps"] += conn.execute(
            f"SELECT COUNT(*) FROM files "
            f"WHERE gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL{scope}",
            chunk,
        ).fetchone()[0]
    return counts


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


def clear_all_person_cluster_observations(conn: sqlite3.Connection) -> None:
    """Delete ALL person_cluster_observations rows (full rebuild semantics).

    Same global/relative-operation rationale as
    `clear_all_person_cluster_members`: each `cluster-people` run rebuilds
    the whole observation map from scratch, since cluster labels are only
    meaningful relative to the whole batch.
    """
    conn.execute("DELETE FROM person_cluster_observations")


def insert_person_cluster_observations(conn: sqlite3.Connection, rows: list) -> None:
    """Bulk-insert (person_label, file_path, source_type, timestamp_ms,
    bbox_x, bbox_y, bbox_width, bbox_height, detection_confidence) tuples."""
    if not rows:
        return
    conn.executemany(
        "INSERT INTO person_cluster_observations "
        "(person_label, file_path, source_type, timestamp_ms, "
        "bbox_x, bbox_y, bbox_width, bbox_height, detection_confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        list(rows),
    )


def fetch_person_cluster_observations(
    conn: sqlite3.Connection, person_label: Optional[str] = None
):
    """Fetch observation rows globally or for one person_label."""
    if person_label is None:
        cur = conn.execute(
            "SELECT * FROM person_cluster_observations "
            "ORDER BY person_label, file_path, timestamp_ms, bbox_x, id"
        )
    else:
        cur = conn.execute(
            "SELECT * FROM person_cluster_observations WHERE person_label = ? "
            "ORDER BY person_label, file_path, timestamp_ms, bbox_x, id",
            (person_label,),
        )
    return cur.fetchall()


# --- V3.1 canonical identities -------------------------------------------
# Machine cluster labels (P0001, ...) are model evidence; canonical persons
# (PERSON_0007, ...) are human-curated identities. These helpers are plain
# CRUD — the merge/split/resolve rules live in person_identity.py.

def upsert_person(
    conn: sqlite3.Connection,
    person_id: str,
    display_name: Optional[str] = None,
    created_at: Optional[str] = None,
    note: Optional[str] = None,
) -> None:
    """Create a canonical person, or update name/note if it already exists."""
    conn.execute(
        "INSERT INTO persons (person_id, display_name, created_at, note) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(person_id) DO UPDATE SET "
        "display_name = excluded.display_name, note = excluded.note",
        (person_id, display_name, created_at, note),
    )


def fetch_persons(conn: sqlite3.Connection) -> list:
    cur = conn.execute("SELECT * FROM persons ORDER BY person_id")
    return cur.fetchall()


def fetch_person(conn: sqlite3.Connection, person_id: str):
    cur = conn.execute("SELECT * FROM persons WHERE person_id = ?", (person_id,))
    return cur.fetchone()


def next_person_id(conn: sqlite3.Connection) -> str:
    """Next free PERSON_NNNN id, scanning existing canonical ids only
    (machine P000N labels live in a separate namespace and never clash)."""
    taken = set()
    for row in conn.execute("SELECT person_id FROM persons"):
        label = row["person_id"] or ""
        if label.startswith("PERSON_"):
            try:
                taken.add(int(label.split("_", 1)[1]))
            except ValueError:
                continue
    number = 1
    while number in taken:
        number += 1
    return f"PERSON_{number:04d}"


def set_alias(
    conn: sqlite3.Connection, machine_label: str, person_id: str, created_at: str
) -> None:
    """Point one machine cluster at a canonical person (merge = several
    machine labels pointing at the same person_id)."""
    conn.execute(
        "INSERT INTO person_aliases (machine_label, person_id, origin, created_at) "
        "VALUES (?, ?, 'manual', ?) "
        "ON CONFLICT(machine_label) DO UPDATE SET person_id = excluded.person_id",
        (machine_label, person_id, created_at),
    )


def fetch_aliases(conn: sqlite3.Connection) -> dict[str, str]:
    """Map machine_label -> canonical person_id."""
    return {
        row["machine_label"]: row["person_id"]
        for row in conn.execute("SELECT machine_label, person_id FROM person_aliases")
    }


def fetch_machine_labels(conn: sqlite3.Connection) -> set[str]:
    """Distinct P000N labels currently in the cluster table ({} if the
    table does not exist yet — same guard style as the other fetchers)."""
    try:
        rows = conn.execute(
            "SELECT DISTINCT person_label FROM person_cluster_members"
        ).fetchall()
    except sqlite3.OperationalError:
        return set()
    return {row["person_label"] for row in rows}


def set_file_link(
    conn: sqlite3.Connection, file_path: str, person_id: str, origin: str, created_at: str
) -> None:
    """Manual assertion about one file: origin 'manual' (user says this
    file shows the person, even with no detected face) or 'excluded'
    (user says an AUTO-derived membership is wrong for this file)."""
    conn.execute(
        "INSERT INTO file_person_links (file_path, person_id, origin, created_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(file_path, person_id) DO UPDATE SET origin = excluded.origin",
        (file_path, person_id, origin, created_at),
    )


def delete_file_link(conn: sqlite3.Connection, file_path: str, person_id: str) -> None:
    conn.execute(
        "DELETE FROM file_person_links WHERE file_path = ? AND person_id = ?",
        (file_path, person_id),
    )


def fetch_file_links(conn: sqlite3.Connection) -> list:
    cur = conn.execute(
        "SELECT * FROM file_person_links ORDER BY file_path, person_id"
    )
    return cur.fetchall()


# --- F1 incremental scan -------------------------------------------------
# A file is skippable when its cheap fingerprint (size + mtime) matches
# the catalog AND the extractors that produced the row cover everything
# the current scan would run. Fresh/legacy rows (NULL versions) always
# reprocess once; 'other'-category files need no extractor at all.

def fetch_fingerprints(conn: sqlite3.Connection) -> dict[str, dict]:
    """Map path -> stored fingerprint columns for the whole catalog."""
    try:
        rows = conn.execute(
            "SELECT path, size_bytes, modified_at, extractor_versions, "
            "sha256, perceptual_hash, file_state FROM files"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {
        row["path"]: {
            "size_bytes": row["size_bytes"],
            "modified_at": row["modified_at"],
            "extractor_versions": row["extractor_versions"],
            "sha256": row["sha256"],
            "perceptual_hash": row["perceptual_hash"],
            "file_state": row["file_state"],
        }
        for row in rows
    }


def touch_files(conn: sqlite3.Connection, touches: list[tuple[str, str]]) -> None:
    """Bulk-refresh (last_seen_at, active) for fingerprint-matched files."""
    if not touches:
        return
    conn.executemany(
        "UPDATE files SET last_seen_at = ?, file_state = 'active' WHERE path = ?",
        touches,
    )


def mark_stale(conn: sqlite3.Connection, cutoff_iso: str) -> int:
    """Mark rows not observed since cutoff as stale and detach them from
    duplicate/similar groups. Returns the stale count."""
    cur = conn.execute(
        "UPDATE files SET file_state = 'stale', "
        "duplicate_group = NULL, similar_group = NULL "
        "WHERE COALESCE(file_state, 'active') = 'active' "
        "AND (last_seen_at IS NULL OR last_seen_at < ?)",
        (cutoff_iso,),
    )
    return cur.rowcount


def fetch_active_paths(conn: sqlite3.Connection) -> Optional[set[str]]:
    """Paths usable by global views (summary/persons). None when the
    column predates the caller DB (treat as 'everything active')."""
    try:
        rows = conn.execute(
            "SELECT path FROM files WHERE COALESCE(file_state, 'active') = 'active'"
        ).fetchall()
    except sqlite3.OperationalError:
        return None
    return {row["path"] for row in rows}
