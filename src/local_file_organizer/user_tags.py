"""User-defined file tags (T2).

Tags are plain text labels the user attaches to catalog files
("beach", "to-print", ...). They live only in the local catalog
tables ``tags`` / ``file_tag_links`` and are never read by the
organization plan or its static viewer, so they can't leak into
``reports/organization_plan.json``.

A file is referenced either by its catalog path (as stored) or by
its stable file id (``F000123``, numbered by ascending path over the
whole catalog — the same numbering ``query`` and the plan use).
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone

from . import database

FILE_ID_RE = re.compile(r"^F(\d{6})$")

MAX_TAG_LENGTH = 64


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_tag_name(raw: str) -> str:
    """Validate and normalize a tag name. Returns the stripped name."""
    name = (raw or "").strip()
    if not name:
        raise ValueError("tag name must not be empty")
    if len(name) > MAX_TAG_LENGTH:
        raise ValueError(f"tag name too long (max {MAX_TAG_LENGTH} chars): {raw!r}")
    if "," in name:
        raise ValueError(f"tag name must not contain a comma: {raw!r}")
    if any(ord(ch) < 32 for ch in name):
        raise ValueError(f"tag name must not contain control characters: {raw!r}")
    return name


def resolve_file_path(conn: sqlite3.Connection, ref: str) -> str:
    """Map a catalog path or stable file id (F000NNN) to a catalog path."""
    label = (ref or "").strip()
    if not label:
        raise ValueError("file reference must not be empty")
    match = FILE_ID_RE.match(label)
    if match:
        number = int(match.group(1))
        row = conn.execute(
            "SELECT path FROM "
            "(SELECT path, ROW_NUMBER() OVER (ORDER BY path) AS file_num "
            "FROM files) WHERE file_num = ?",
            (number,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown file id: {label}")
        return row["path"]
    row = conn.execute("SELECT path FROM files WHERE path = ?", (label,)).fetchone()
    if row is None:
        raise ValueError(f"file not in catalog: {label}")
    return row["path"]


def attach_tag(conn: sqlite3.Connection, file_ref: str, tag_name: str) -> tuple[str, str]:
    """Attach a tag to a file, creating the tag on first use.

    Idempotent: attaching the same tag twice keeps a single link.
    Returns (file_path, tag_name).
    """
    path = resolve_file_path(conn, file_ref)
    name = normalize_tag_name(tag_name)
    database.upsert_tag(conn, name, _now())
    database.add_file_tag(conn, path, name, _now())
    return path, name


def detach_tag(conn: sqlite3.Connection, file_ref: str, tag_name: str) -> tuple[str, str]:
    """Detach a tag from a file. The tag row itself is kept, so `tags`
    still lists it with zero files. Returns (file_path, tag_name)."""
    path = resolve_file_path(conn, file_ref)
    name = normalize_tag_name(tag_name)
    removed = database.remove_file_tag(conn, path, name)
    if not removed:
        raise ValueError(f"{path} is not tagged with {name!r}")
    return path, name


def resolve_file_tags(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """Map file_path -> set of tag names (all states, like persons)."""
    resolved: dict[str, set[str]] = {}
    for link in database.fetch_file_tag_links(conn):
        resolved.setdefault(link["file_path"], set()).add(link["tag_name"])
    return resolved


def tag_summary(conn: sqlite3.Connection) -> list[dict]:
    """One row per tag: name plus active-file count, ordered by name.

    Stale files (F1) are excluded when the catalog tracks them, so the
    counts stay consistent with search and the other global views.
    Tags without active files are still listed (files=0).
    """
    try:
        tag_rows = database.fetch_tags(conn)
    except sqlite3.OperationalError:
        return []
    active_paths = database.fetch_active_paths(conn)
    counts: dict[str, int] = {row["tag_name"]: 0 for row in tag_rows}
    for link in database.fetch_file_tag_links(conn):
        if active_paths is not None and link["file_path"] not in active_paths:
            continue
        if link["tag_name"] in counts:
            counts[link["tag_name"]] += 1
    return [{"tag_name": name, "files": counts[name]} for name in sorted(counts)]
