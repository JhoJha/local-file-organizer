"""Read-only organization preview (V2.5).

This module only reads the existing SQLite catalog (tables ``files`` and
``person_cluster_members``) and builds an in-memory organization plan plus
a JSON report under ``reports/organization_plan.json``. It never walks the
filesystem, never opens a file from the scanned tree, and never moves,
copies, renames or deletes anything.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def _ext_parts(extension: str | None) -> tuple[str, str]:
    """Return (EXT_UPPER, ext_lower) stripped of the leading dot."""
    raw = (extension or "").strip()
    if raw in ("", "."):
        return "", ""
    lowered = raw.lower()
    if lowered.startswith("."):
        lowered = lowered[1:]
    if not lowered:
        return "", ""
    return lowered.upper(), lowered


def _fetch_person_sets(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """Map file_path -> set of distinct person_label values."""
    person_sets: dict[str, set[str]] = {}
    try:
        rows = conn.execute(
            "SELECT file_path, person_label FROM person_cluster_members"
        ).fetchall()
    except sqlite3.OperationalError:
        return person_sets
    for row in rows:
        person_sets.setdefault(row["file_path"], set()).add(row["person_label"])
    return person_sets


def build_organization_plan(conn: sqlite3.Connection) -> list[dict]:
    """Build the full organization plan, one entry per cataloged file.

    File IDs (``F000001``, ...) are assigned by ascending ``path`` order and
    recomputed fresh on every call (never persisted). ``PAIR_000N`` /
    ``GROUP_000N`` labels are derived fresh each run from the sorted
    distinct person-sets of size exactly 2 / 3+.
    """
    rows = conn.execute(
        "SELECT path, extension, category, scan_status, metadata_status,"
        " duplicate_group, similar_group FROM files ORDER BY path ASC"
    ).fetchall()

    paths = [row["path"] for row in rows]
    file_ids = {path: f"F{i + 1:06d}" for i, path in enumerate(paths)}

    person_sets = _fetch_person_sets(conn)

    # First pass: buckets + person-sets for planned image/video files.
    buckets: dict[str, str] = {}
    person_tuple_by_path: dict[str, tuple[str, ...]] = {}
    for row in rows:
        path = row["path"]
        scan_status = row["scan_status"]
        category = row["category"]
        metadata_status = row["metadata_status"]
        if scan_status == "error":
            buckets[path] = "unsupported"
        elif (
            scan_status == "ok"
            and category in ("image", "video")
            and metadata_status == "error"
        ):
            buckets[path] = "review"
        else:
            buckets[path] = "planned"
        if buckets[path] == "planned" and category in ("image", "video"):
            labels = sorted(person_sets.get(path, set()))
            person_tuple_by_path[path] = tuple(labels)

    # Derive deterministic PAIR_ / GROUP_ labels.
    pair_tuples = sorted(
        {t for t in person_tuple_by_path.values() if len(t) == 2}
    )
    group_tuples = sorted(
        {t for t in person_tuple_by_path.values() if len(t) >= 3}
    )
    pair_labels = {t: f"PAIR_{i + 1:04d}" for i, t in enumerate(pair_tuples)}
    group_labels = {t: f"GROUP_{i + 1:04d}" for i, t in enumerate(group_tuples)}

    entries: list[dict] = []
    for row in rows:
        path = row["path"]
        category = row["category"]
        extension = row["extension"]
        bucket = buckets[path]
        flags: list[str] = []
        if row["duplicate_group"] is not None:
            flags.append("exact_duplicate")
        if row["similar_group"] is not None:
            flags.append("visual_similar_group")

        if bucket == "unsupported":
            destination = "Revisión/Errores_de_escaneo/"
            reason_codes = ["scan_error"]
            confidence = 0.0
        elif bucket == "review":
            destination = "Revisión/Metadata_incompleta/"
            reason_codes = ["metadata_error"]
            confidence = 0.3
        elif category in ("image", "video"):
            person_tuple = person_tuple_by_path.get(path, ())
            n = len(person_tuple)
            if n == 0:
                destination = "Media/Sin_personas/"
                reason_codes = ["no_person_detected"]
                confidence = 0.9
            elif n == 1:
                label = person_tuple[0]
                destination = f"Media/Personas/{label}/"
                reason_codes = [f"single_person:{label}"]
                confidence = 0.9
            elif n == 2:
                pair_label = pair_labels[person_tuple]
                destination = f"Media/Parejas/{pair_label}/"
                reason_codes = [f"pair:{pair_label}"]
                confidence = 0.85
            else:
                group_label = group_labels[person_tuple]
                destination = f"Media/Grupos/{group_label}/"
                reason_codes = [f"group:{group_label}"]
                confidence = 0.8
        elif category == "document":
            ext_upper, ext_lower = _ext_parts(extension)
            if not ext_upper:
                destination = "Documentos/Sin_extension/"
                reason_codes = ["no_extension"]
            else:
                destination = f"Documentos/{ext_upper}/"
                reason_codes = [f"extension:{ext_lower}"]
            confidence = 1.0
        elif category == "audio":
            ext_upper, ext_lower = _ext_parts(extension)
            if not ext_upper:
                destination = "Audio/Sin_extension/"
                reason_codes = ["no_extension"]
            else:
                destination = f"Audio/{ext_upper}/"
                reason_codes = [f"extension:{ext_lower}"]
            confidence = 1.0
        elif category == "archive":
            _ext_upper, ext_lower = _ext_parts(extension)
            destination = "Archivos_comprimidos/"
            reason_codes = [f"extension:{ext_lower}"]
            confidence = 1.0
        elif category == "software":
            _ext_upper, ext_lower = _ext_parts(extension)
            destination = "Software/"
            reason_codes = [f"extension:{ext_lower}"]
            confidence = 1.0
        else:  # 'other' and any unknown category
            ext_upper, ext_lower = _ext_parts(extension)
            if not ext_upper:
                destination = "Otros/Sin_extension/"
                reason_codes = ["no_extension"]
            else:
                destination = f"Otros/{ext_upper}/"
                reason_codes = [f"extension:unknown:{ext_lower}"]
            confidence = 0.7

        entries.append(
            {
                "file_id": file_ids[path],
                "category": category,
                "bucket": bucket,
                "proposed_destination": destination,
                "reason_codes": reason_codes,
                "flags": flags,
                "confidence": confidence,
                "action": "preview_only",
            }
        )
    return entries


def write_plan_json(entries: list[dict], plan_path: Path) -> None:
    """Write the plan entries to ``plan_path`` as a JSON array."""
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, ensure_ascii=False, indent=2)


def plan_counts(entries: list[dict]) -> dict[str, int]:
    """Aggregate bucket/flag counts plus per-destination file counts."""
    by_destination: dict[str, int] = {}
    for entry in entries:
        dest = entry["proposed_destination"]
        by_destination[dest] = by_destination.get(dest, 0) + 1
    planned = sum(1 for e in entries if e["bucket"] == "planned")
    review = sum(1 for e in entries if e["bucket"] == "review")
    unsupported = sum(1 for e in entries if e["bucket"] == "unsupported")
    exact_duplicates = sum(1 for e in entries if "exact_duplicate" in e["flags"])
    visual_similar = sum(
        1 for e in entries if "visual_similar_group" in e["flags"]
    )
    return {
        "catalog_files": len(entries),
        "planned": planned,
        "review": review,
        "unsupported": unsupported,
        "exact_duplicates": exact_duplicates,
        "visual_similar_groups": visual_similar,
        "by_destination": by_destination,
    }
