"""Application workflow for scanning and cataloguing a local file tree.

This module owns scan orchestration and scanning domain logic without
CLI/stdout/stderr dependencies. The caller may provide callbacks for
progress and warnings, or invoke generate_scan directly with defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import database
from .classifier import classify
from .exiftool import EXIFTOOL_VERSION, exiftool_available, get_exif_metadata
from .hashing import DuplicateRecord, find_duplicates
from .metadata import (
    FFPROBE_VERSION,
    get_basic_metadata,
    get_video_metadata,
)
from .perceptual_hash import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DHASH_VERSION,
    SimilarityRecord,
    get_image_metadata,
    group_similar_images,
)
from .reporting import build_summary_text, export_csv
from .scanner import DEFAULT_IGNORED_DIRS, ScanError, iter_files
from .snapshot import SnapshotDiff, capture_snapshot, diff_snapshots, save_snapshot

DEFAULT_PROGRESS_INTERVAL = 500


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
    readonly_diff: SnapshotDiff | None


def required_versions(category: str, exif_enabled: bool) -> set[str]:
    """Extractor versions a scan would run for this category (F1).

    'other' needs nothing beyond stat+classify, so a fingerprinted
    'other' file always skips. Versions are only recorded on success
    (see scan_one_file), so failed extractions are retried next scan.
    """
    required: set[str] = set()
    if category == "image":
        required.add(DHASH_VERSION)
    elif category == "video":
        required.add(FFPROBE_VERSION)
    if exif_enabled and category in ("image", "video"):
        required.add(EXIFTOOL_VERSION)
    return required


def fingerprint_match(
    stored: dict[str, Any] | None, basic: Any, exif_enabled: bool
) -> bool:
    """True when the catalog row is fresh: same bytes (size+mtime) and
    the stored extractor versions cover everything this scan would run.
    """
    if not stored:
        return False
    if stored.get("size_bytes") != basic.size_bytes:
        return False
    if stored.get("modified_at") != basic.modified_at:
        return False
    stored_versions = set((stored.get("extractor_versions") or "").split("|")) - {""}
    return required_versions(basic.category, exif_enabled) <= stored_versions


def scan_one_file(
    path: Path,
    enable_exif: bool = True,
    known_basic: Any = None,
    *,
    get_basic_metadata_fn: Callable[..., Any] = get_basic_metadata,
    get_image_metadata_fn: Callable[..., Any] = get_image_metadata,
    get_video_metadata_fn: Callable[..., Any] = get_video_metadata,
    get_exif_metadata_fn: Callable[..., Any] = get_exif_metadata,
    classify_fn: Callable[[str], str] = classify,
) -> database.FileRecord:
    """Extract metadata for a single file into a FileRecord."""
    if known_basic is None:
        try:
            basic = get_basic_metadata_fn(path)
        except OSError as exc:
            return database.FileRecord(
                path=str(path),
                name=path.name,
                extension=path.suffix.lower(),
                category=classify_fn(path.suffix),
                scan_status="error",
                scan_error=str(exc),
            )
    else:
        basic = known_basic

    record = database.FileRecord(
        path=basic.path,
        name=basic.name,
        extension=basic.extension,
        category=basic.category,
        size_bytes=basic.size_bytes,
        modified_at=basic.modified_at,
        mime_type=basic.mime_type,
        scan_status="ok",
        scan_error=None,
    )

    if basic.category == "video":
        video = get_video_metadata_fn(path)
        record.duration_seconds = video.duration_seconds
        record.width = video.width
        record.height = video.height
        record.fps = video.fps
        record.video_codec = video.video_codec
        record.audio_codec = video.audio_codec
        record.bitrate = video.bitrate
        if video.error:
            record.metadata_status = "error"
            record.metadata_error = video.error
        else:
            record.metadata_status = "ok"
    elif basic.category == "image":
        image = get_image_metadata_fn(path)
        if image.error:
            record.metadata_status = "error"
            record.metadata_error = image.error
        else:
            record.metadata_status = "ok"
            record.perceptual_hash = image.perceptual_hash
            record.similarity_method = image.similarity_method

    exif_ok = False
    if enable_exif and basic.category in ("image", "video"):
        exif = get_exif_metadata_fn(path)
        if exif.error is None:
            exif_ok = True
            record.capture_date = exif.capture_date
            record.device_make = exif.device_make
            record.device_model = exif.device_model
            record.device_software = exif.device_software
            record.gps_latitude = exif.gps_latitude
            record.gps_longitude = exif.gps_longitude
            record.gps_altitude = exif.gps_altitude
            record.orientation = exif.orientation

    versions: set[str] = set()
    if basic.category == "image" and record.metadata_status == "ok":
        versions.add(DHASH_VERSION)
    if basic.category == "video" and record.metadata_status == "ok":
        versions.add(FFPROBE_VERSION)
    if exif_ok:
        versions.add(EXIFTOOL_VERSION)
    record.extractor_versions = "|".join(sorted(versions)) or None

    return record


def _default_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_scan(
    *,
    target: Path,
    db_path: Path,
    csv_path: Path,
    ignored_dirs: set[str] | None = None,
    verify_readonly: bool = False,
    snapshot_dir: Path | None = None,
    similarity_threshold: int = DEFAULT_SIMILARITY_THRESHOLD,
    full: bool = False,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
    iter_files_fn: Callable[..., Any] = iter_files,
    get_basic_metadata_fn: Callable[..., Any] = get_basic_metadata,
    scan_one_file_fn: Callable[..., Any] = scan_one_file,
    fingerprint_match_fn: Callable[..., bool] = fingerprint_match,
    exiftool_available_fn: Callable[[], bool] = exiftool_available,
    find_duplicates_fn: Callable[..., Any] = find_duplicates,
    now_iso_fn: Callable[[], str] = _default_now_iso,
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

    effective_ignored_dirs = (
        set(DEFAULT_IGNORED_DIRS) if ignored_dirs is None else set(ignored_dirs)
    )
    target_snapshot_dir = (
        snapshot_dir
        if snapshot_dir is not None
        else db_path.parent / "snapshots"
    )

    walk_errors: list[ScanError] = []

    def _walk_error(err: ScanError) -> None:
        walk_errors.append(err)
        if on_walk_error is not None:
            on_walk_error(err)

    before_snapshot = None
    if verify_readonly:
        before_snapshot = capture_snapshot(
            target, ignored_dirs=effective_ignored_dirs, on_error=_walk_error
        )
        save_snapshot(before_snapshot, target_snapshot_dir / "before.json")

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
            target, ignored_dirs=effective_ignored_dirs, on_error=_walk_error
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
                target, ignored_dirs=effective_ignored_dirs, on_error=_walk_error
            )
            save_snapshot(after_snapshot, target_snapshot_dir / "after.json")
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
