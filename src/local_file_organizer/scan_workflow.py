"""Application workflow for scanning and cataloguing a local file tree.

This module owns scan orchestration but has no CLI/stdout/stderr dependencies.
The caller may provide callbacks for progress and warnings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import database
from .hashing import DuplicateRecord
from .perceptual_hash import SimilarityRecord, group_similar_images
from .reporting import build_summary_text, export_csv
from .scanner import ScanError
from .snapshot import capture_snapshot, diff_snapshots, save_snapshot


class ScanWorkflowError(Exception):
    """Base exception for scan workflow errors."""


class TargetNotFoundError(ScanWorkflowError):
    """Raised when the requested scan target does not exist."""


class TargetNotDirectoryError(ScanWorkflowError):
    """Raised when the requested scan target is not a directory."""


@dataclass(frozen=True)
class ScanResult:
    """Structured result returned by the scan application workflow."""

    db_path: Path
    csv_path: Path
    summary: str
    walk_errors: tuple[ScanError, ...]
    readonly_diff: Any | None


def generate_scan(
    *,
    target: Path,
    db_path: Path,
    csv_path: Path,
    ignored_dirs: set[str],
    verify_readonly: bool,
    snapshot_dir: Path,
    similarity_threshold: int,
    full: bool,
    progress_interval: int,
    iter_files_fn: Callable[..., Any],
    get_basic_metadata_fn: Callable[..., Any],
    scan_one_file_fn: Callable[..., Any],
    fingerprint_match_fn: Callable[..., bool],
    exiftool_available_fn: Callable[[], bool],
    find_duplicates_fn: Callable[..., Any],
    now_iso_fn: Callable[[], str],
    on_progress: Callable[[int], None] | None = None,
    on_walk_error: Callable[[ScanError], None] | None = None,
    on_hash_error: Callable[[str, Exception], None] | None = None,
) -> ScanResult:
    """Run the complete scan workflow without writing to stdout/stderr.

    The scanned tree is treated as read-only. Project-owned outputs may be written
    to the catalog, CSV report, and optional snapshot directory.
    """
    if not target.exists():
        raise TargetNotFoundError(str(target))
    if not target.is_dir():
        raise TargetNotDirectoryError(str(target))

    walk_errors: list[ScanError] = []

    def _walk_error(err: ScanError) -> None:
        walk_errors.append(err)
        if on_walk_error is not None:
            on_walk_error(err)

    before_snapshot = None
    if verify_readonly:
        before_snapshot = capture_snapshot(
            target, ignored_dirs=ignored_dirs, on_error=_walk_error
        )
        save_snapshot(before_snapshot, snapshot_dir / "before.json")

    conn = database.connect(db_path)
    try:
        scanned_paths: list[str] = []
        duplicate_candidates: list[DuplicateRecord] = []
        image_hash_candidates: list[SimilarityRecord] = []
        known_digests: dict[str, str] = {}
        touches: list[tuple[str, str]] = []
        reused_count = 0
        added_count = 0
        reprocessed_count = 0

        exif_enabled = exiftool_available_fn()
        scan_start_iso = now_iso_fn()
        fingerprints = database.fetch_fingerprints(conn)

        count = 0
        for file_path in iter_files_fn(
            target, ignored_dirs=ignored_dirs, on_error=_walk_error
        ):
            try:
                basic = get_basic_metadata_fn(file_path)
            except OSError:
                record = scan_one_file_fn(
                    file_path, enable_exif=exif_enabled
                )
                record.last_seen_at = scan_start_iso
                record.file_state = "active"
                record.duplicate_group = None
                record.similar_group = None
                database.upsert_file(conn, record)
                scanned_paths.append(record.path)
                if str(file_path) in fingerprints:
                    reprocessed_count += 1
                else:
                    added_count += 1
                count += 1
                if on_progress is not None and count % progress_interval == 0:
                    on_progress(count)
                continue

            stored = fingerprints.get(basic.path)
            if not full and fingerprint_match_fn(stored, basic, exif_enabled):
                touches.append((scan_start_iso, basic.path))
                scanned_paths.append(basic.path)
                duplicate_candidates.append(
                    DuplicateRecord(path=basic.path, size_bytes=basic.size_bytes)
                )
                if stored.get("sha256"):
                    known_digests[basic.path] = stored["sha256"]
                if basic.category == "image" and stored.get("perceptual_hash"):
                    image_hash_candidates.append(
                        SimilarityRecord(
                            path=basic.path,
                            perceptual_hash=stored["perceptual_hash"],
                            size_bytes=basic.size_bytes,
                            extension=basic.extension,
                        )
                    )
                reused_count += 1
                count += 1
                if on_progress is not None and count % progress_interval == 0:
                    on_progress(count)
                continue

            record = scan_one_file_fn(
                file_path, enable_exif=exif_enabled, known_basic=basic
            )
            record.last_seen_at = scan_start_iso
            record.file_state = "active"
            record.duplicate_group = None
            record.similar_group = None
            database.upsert_file(conn, record)
            scanned_paths.append(record.path)
            if stored is None:
                added_count += 1
            else:
                reprocessed_count += 1
            if record.scan_status == "ok":
                duplicate_candidates.append(
                    DuplicateRecord(path=record.path, size_bytes=record.size_bytes)
                )
            if record.category == "image" and record.perceptual_hash:
                image_hash_candidates.append(
                    SimilarityRecord(
                        path=record.path,
                        perceptual_hash=record.perceptual_hash,
                        size_bytes=record.size_bytes,
                        extension=record.extension,
                    )
                )
            count += 1
            if on_progress is not None and count % progress_interval == 0:
                on_progress(count)

        database.touch_files(conn, touches)
        conn.commit()

        def _hash_error(path: str, exc: Exception) -> None:
            if on_hash_error is not None:
                on_hash_error(path, exc)
            database.set_scan_error(conn, path, f"hash: {exc}")

        dup_stats = find_duplicates_fn(
            duplicate_candidates,
            on_hash_error=_hash_error,
            known_digests=known_digests,
        )
        database.clear_duplicate_groups(conn, scanned_paths)
        database.set_digests(conn, dup_stats.digests)
        for path, group_id in dup_stats.group_by_path.items():
            database.set_duplicate_group(conn, path, group_id)
        conn.commit()

        sim_stats = group_similar_images(
            image_hash_candidates, threshold=similarity_threshold
        )
        database.clear_similar_groups(conn, scanned_paths)
        for path, group_id in sim_stats.group_by_path.items():
            database.set_similar_group(conn, path, group_id)
        conn.commit()

        stale_count = database.mark_stale(conn, scan_start_iso)
        conn.commit()

        readonly_diff = None
        if verify_readonly:
            after_snapshot = capture_snapshot(
                target, ignored_dirs=ignored_dirs, on_error=_walk_error
            )
            save_snapshot(after_snapshot, snapshot_dir / "after.json")
            readonly_diff = diff_snapshots(before_snapshot, after_snapshot)

        export_csv(conn, csv_path)

        category_totals = database.fetch_category_totals(
            conn, paths=scanned_paths
        )
        total_files = len(scanned_paths)
        total_bytes = sum(row["total_bytes"] or 0 for row in category_totals)
        error_count = database.fetch_error_count(conn, paths=scanned_paths)
        exif_counts = database.fetch_exif_counts(conn, paths=scanned_paths)

        summary = build_summary_text(
            category_totals=category_totals,
            total_files=total_files,
            total_bytes=total_bytes,
            duplicate_file_count=dup_stats.total_duplicate_files,
            duplicate_group_count=dup_stats.total_duplicate_groups,
            recoverable_bytes=dup_stats.recoverable_space_bytes,
            error_count=error_count,
            image_hash_count=len(image_hash_candidates),
            similar_group_count=sim_stats.total_groups,
            similar_grouped_file_count=sim_stats.total_grouped_files,
            exif_date_count=exif_counts["with_capture_date"],
            exif_device_count=exif_counts["with_device"],
            exif_gps_count=exif_counts["with_gps"],
            reused_count=reused_count,
            added_count=added_count,
            reprocessed_count=reprocessed_count,
            stale_count=stale_count,
        )

        return ScanResult(
            db_path=db_path,
            csv_path=csv_path,
            summary=summary,
            walk_errors=tuple(walk_errors),
            readonly_diff=readonly_diff,
        )
    finally:
        conn.close()
