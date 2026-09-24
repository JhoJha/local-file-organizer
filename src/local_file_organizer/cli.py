"""Command-line interface: scan, summary, duplicates, similar-images, sample-frames, detect-faces, cluster-people.

Every code path in this module is read-only with respect to the scanned
tree: it lists files, reads their metadata/bytes (including decoding
images in memory to compute a perceptual hash, and decoding images or
ephemeral video frames in memory for face detection), and writes results
only under this project's own data/ and reports/ directories.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2

from . import database
from . import person_identity
from .classifier import classify
from .face_clustering import assign_person_labels, cluster_people
from .face_clustering import (
    ClusterableObservation,
    DEFAULT_SIMILARITY_THRESHOLD as DEFAULT_PERSON_SIMILARITY_THRESHOLD,
)
from .face_detection import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MIN_FACE_SIZE_PX,
    detect_faces_in_image_file,
    detect_faces_raw,
)
from .exiftool import EXIFTOOL_VERSION, exiftool_available, get_exif_metadata
from .face_embedding import compute_embedding
from .face_review import (
    DEFAULT_MAX_PREVIEWS_PER_CLUSTER,
    DEFAULT_THUMBNAIL_MAX_PX,
    build_manifest,
    compute_metrics,
    generate_assets,
    load_decisions,
    parse_decisions_payload,
    save_decisions,
    validate_decisions,
    write_review_site,
)
from .hashing import DuplicateRecord, find_duplicates
from .metadata import (
    FFPROBE_VERSION,
    get_basic_metadata,
    get_video_metadata,
    human_readable_size,
)
from .video_sampling import compute_sample_timestamps, sample_video_frames
from .perceptual_hash import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DHASH_VERSION,
    SimilarityRecord,
    get_image_metadata,
    group_similar_images,
    hamming_distance,
)
from .reporting import build_summary_text, export_csv
from .scanner import DEFAULT_IGNORED_DIRS, ScanError, iter_files
from .snapshot import capture_snapshot, diff_snapshots, save_snapshot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "catalog.sqlite"
DEFAULT_CSV_PATH = PROJECT_ROOT / "reports" / "inventory.csv"
DEFAULT_PLAN_PATH = PROJECT_ROOT / "reports" / "organization_plan.json"
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"
DEFAULT_DEBUG_FRAMES_DIR = PROJECT_ROOT / "data" / "debug_frames"
DEFAULT_FACE_REVIEW_DIR = PROJECT_ROOT / "reports" / "face_review"
DEFAULT_FACE_REVIEW_DECISIONS = PROJECT_ROOT / "data" / "face_review_decisions.json"

_PROGRESS_INTERVAL = 500


def _display_path(path: Path) -> str:
    """Render a project-owned output path relative to the repo when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _required_versions(category: str, exif_enabled: bool) -> set[str]:
    """Extractor versions a scan would run for this category (F1).

    'other' needs nothing beyond stat+classify, so a fingerprinted
    'other' file always skips. Versions are only recorded on success
    (see _scan_one_file), so failed extractions are retried next scan.
    """
    required: set[str] = set()
    if category == "image":
        required.add(DHASH_VERSION)
    elif category == "video":
        required.add(FFPROBE_VERSION)
    if exif_enabled and category in ("image", "video"):
        required.add(EXIFTOOL_VERSION)
    return required


def _fingerprint_match(
    stored: Optional[dict], basic, exif_enabled: bool
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
    return _required_versions(basic.category, exif_enabled) <= stored_versions


def _scan_one_file(
    path: Path, enable_exif: bool = True, known_basic=None
) -> database.FileRecord:
    if known_basic is None:
        try:
            basic = get_basic_metadata(path)
        except OSError as exc:
            return database.FileRecord(
                path=str(path),
                name=path.name,
                extension=path.suffix.lower(),
                category=classify(path.suffix),
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
        video = get_video_metadata(path)
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
        image = get_image_metadata(path)
        if image.error:
            record.metadata_status = "error"
            record.metadata_error = image.error
        else:
            record.metadata_status = "ok"
            record.perceptual_hash = image.perceptual_hash
            record.similarity_method = image.similarity_method
    # Other categories have no specialized extraction yet in V1, so
    # metadata_status keeps its "skipped" default.

    # V3.0 EXIF/XMP enrichment for stills and video (QuickTime tags).
    # Pure enrichment: it only fills columns, it never flips
    # metadata_status — a file without EXIF is "unknown", not an error.
    exif_ok = False
    if enable_exif and basic.category in ("image", "video"):
        exif = get_exif_metadata(path)
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

    # F1 incrementality: record which extractors succeeded. Failed ones
    # are omitted so the next scan retries them; a newly appeared binary
    # (e.g. exiftool installed later) triggers reprocessing via the
    # required-vs-stored subset check in _fingerprint_match.
    versions: set[str] = set()
    if basic.category == "image" and record.metadata_status == "ok":
        versions.add(DHASH_VERSION)
    if basic.category == "video" and record.metadata_status == "ok":
        versions.add(FFPROBE_VERSION)
    if exif_ok:
        versions.add(EXIFTOOL_VERSION)
    record.extractor_versions = "|".join(sorted(versions)) or None

    return record


def run_scan(
    target: Path,
    db_path: Path = DEFAULT_DB_PATH,
    csv_path: Path = DEFAULT_CSV_PATH,
    extra_ignored_dirs: Optional[list[str]] = None,
    quiet: bool = False,
    verify_readonly: bool = False,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    similarity_threshold: int = DEFAULT_SIMILARITY_THRESHOLD,
    full: bool = False,
) -> None:
    if not target.exists():
        print(f"ERROR: la ruta no existe: {target}", file=sys.stderr)
        print(
            "No se inventará ni escaneará ninguna otra ruta. Verifica el "
            "punto de montaje o crea la carpeta esperada.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not target.is_dir():
        print(f"ERROR: la ruta no es un directorio: {target}", file=sys.stderr)
        raise SystemExit(1)

    ignored_dirs = set(DEFAULT_IGNORED_DIRS)
    if extra_ignored_dirs:
        ignored_dirs.update(extra_ignored_dirs)

    walk_errors: list[ScanError] = []

    def _on_error(err: ScanError) -> None:
        walk_errors.append(err)
        if not quiet:
            print(f"AVISO: no se pudo listar {err.path}: {err.message}", file=sys.stderr)

    before_snapshot = None
    if verify_readonly:
        before_snapshot = capture_snapshot(target, ignored_dirs=ignored_dirs, on_error=_on_error)
        save_snapshot(before_snapshot, snapshot_dir / "before.json")

    conn = database.connect(db_path)
    scanned_paths: list[str] = []
    duplicate_candidates: list[DuplicateRecord] = []
    image_hash_candidates: list[SimilarityRecord] = []
    known_digests: dict[str, str] = {}
    touches: list[tuple[str, str]] = []
    reused_count = 0
    added_count = 0
    reprocessed_count = 0

    # One PATH lookup for the whole run instead of one per file; when
    # ExifTool is absent the scan behaves exactly like V2.5.1.
    exif_enabled = exiftool_available()

    # F1 incremental: preload fingerprints once (one query, ~50k small
    # rows fit in memory comfortably) instead of one SELECT per file.
    # --full still preloads (for exact new/reprocessed counters) but
    # never takes the reuse shortcut (calibration escape hatch).
    scan_start_iso = datetime.now(timezone.utc).isoformat()
    fingerprints = database.fetch_fingerprints(conn)

    count = 0
    for file_path in iter_files(target, ignored_dirs=ignored_dirs, on_error=_on_error):
        try:
            basic = get_basic_metadata(file_path)
        except OSError:
            record = _scan_one_file(file_path, enable_exif=exif_enabled)
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
            if not quiet and count % _PROGRESS_INTERVAL == 0:
                print(f"Escaneados {count} archivos...")
            continue

        stored = fingerprints.get(basic.path)
        if not full and _fingerprint_match(stored, basic, exif_enabled):
            # Unchanged bytes + fresh extractors: touch only. Hashes are
            # reused from the catalog (no byte reads, no subprocesses).
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
            if not quiet and count % _PROGRESS_INTERVAL == 0:
                print(f"Escaneados {count} archivos...")
            continue

        record = _scan_one_file(file_path, enable_exif=exif_enabled, known_basic=basic)
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
        if not quiet and count % _PROGRESS_INTERVAL == 0:
            print(f"Escaneados {count} archivos...")

    database.touch_files(conn, touches)
    conn.commit()

    def _on_hash_error(path: str, exc: Exception) -> None:
        if not quiet:
            print(f"AVISO: no se pudo calcular hash de {path}: {exc}", file=sys.stderr)
        database.set_scan_error(conn, path, f"hash: {exc}")

    dup_stats = find_duplicates(
        duplicate_candidates,
        on_hash_error=_on_hash_error,
        known_digests=known_digests,
    )
    database.clear_duplicate_groups(conn, scanned_paths)
    database.set_digests(conn, dup_stats.digests)
    for path, group_id in dup_stats.group_by_path.items():
        database.set_duplicate_group(conn, path, group_id)
    conn.commit()

    sim_stats = group_similar_images(image_hash_candidates, threshold=similarity_threshold)
    database.clear_similar_groups(conn, scanned_paths)
    for path, group_id in sim_stats.group_by_path.items():
        database.set_similar_group(conn, path, group_id)
    conn.commit()

    # F1: rows observed before but missing now become stale (kept for
    # audit, detached from groups, excluded from views).
    stale_count = database.mark_stale(conn, scan_start_iso)
    conn.commit()

    if verify_readonly:
        after_snapshot = capture_snapshot(target, ignored_dirs=ignored_dirs, on_error=_on_error)
        save_snapshot(after_snapshot, snapshot_dir / "after.json")
        readonly_diff = diff_snapshots(before_snapshot, after_snapshot)
        print()
        print("=== VERIFICACIÓN READ-ONLY ===")
        print(
            f"added={len(readonly_diff.added)} removed={len(readonly_diff.removed)} "
            f"size_changed={len(readonly_diff.size_changed)} "
            f"mtime_changed={len(readonly_diff.mtime_changed)}"
        )
        if readonly_diff.is_clean:
            print("OK: ningún archivo original cambió durante el escaneo.")
        else:
            print(
                "ALERTA: se detectaron cambios en los archivos originales durante el escaneo.",
                file=sys.stderr,
            )
            for label, paths in (
                ("added", readonly_diff.added),
                ("removed", readonly_diff.removed),
                ("size_changed", readonly_diff.size_changed),
                ("mtime_changed", readonly_diff.mtime_changed),
            ):
                for changed_path in paths[:20]:
                    print(f"  [{label}] {changed_path}", file=sys.stderr)

    export_csv(conn, csv_path)

    category_totals = database.fetch_category_totals(conn, paths=scanned_paths)
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
    print()
    print(summary)
    print()
    print(f"Catálogo SQLite: {db_path}")
    print(f"Reporte CSV: {csv_path}")
    if walk_errors:
        print(f"Directorios no listados por error de permisos: {len(walk_errors)}")

    conn.close()


def run_sample_frames(
    target: Path,
    db_path: Path = DEFAULT_DB_PATH,
    max_samples: int = 30,
    keep_debug_frames: bool = False,
    debug_dir: Path = DEFAULT_DEBUG_FRAMES_DIR,
    redact_paths: bool = False,
    quiet: bool = False,
    verify_readonly: bool = False,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
) -> None:
    """Sample ephemeral video frames for every video under target."""
    if not target.exists():
        print(f"ERROR: la ruta no existe: {target}", file=sys.stderr)
        print(
            "No se inventará ni escaneará ninguna otra ruta. Verifica el "
            "punto de montaje o crea la carpeta esperada.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not target.is_dir():
        print(f"ERROR: la ruta no es un directorio: {target}", file=sys.stderr)
        raise SystemExit(1)

    start = time.perf_counter()
    ignored_dirs = set(DEFAULT_IGNORED_DIRS)
    walk_errors: list[ScanError] = []

    def _on_error(err: ScanError) -> None:
        walk_errors.append(err)
        if not quiet and not redact_paths:
            print(f"AVISO: no se pudo listar {err.path}: {err.message}", file=sys.stderr)

    before_snapshot = None
    if verify_readonly:
        before_snapshot = capture_snapshot(target, ignored_dirs=ignored_dirs, on_error=_on_error)
        save_snapshot(before_snapshot, snapshot_dir / "before.json")

    conn = database.connect(db_path)
    ephemeral_dirs: list[Path] = []
    videos = 0
    videos_ok = 0
    videos_error = 0
    frames_generated = 0
    frames_failed = 0

    for file_path in iter_files(target, ignored_dirs=ignored_dirs, on_error=_on_error):
        if classify(file_path.suffix) != "video":
            continue
        video_index = videos
        videos += 1

        def _on_frame(frame_path: Path, _ts: float, _idx: int) -> None:
            parent = frame_path.parent
            if parent not in ephemeral_dirs:
                ephemeral_dirs.append(parent)

        try:
            video_meta = get_video_metadata(file_path)
            if video_meta.error or video_meta.duration_seconds is None:
                result_error = video_meta.error or "no duration available"
                database.clear_video_frame_samples(conn, str(file_path))
                conn.commit()
                videos_error += 1
                if not quiet and not redact_paths:
                    print(f"Muestreo {file_path}: error: {result_error}")
                continue
            timestamps = compute_sample_timestamps(video_meta.duration_seconds, max_samples)
            per_video_debug: Optional[Path] = None
            if keep_debug_frames:
                per_video_debug = debug_dir / f"{video_index:04d}"
                per_video_debug.mkdir(parents=True, exist_ok=True)
            result = sample_video_frames(
                file_path,
                timestamps,
                on_frame=_on_frame,
                keep_debug_frames=keep_debug_frames,
                debug_dir=per_video_debug,
            )
        except Exception as exc:  # noqa: BLE001 - one bad video never aborts the run
            database.clear_video_frame_samples(conn, str(file_path))
            conn.commit()
            videos_error += 1
            if not quiet and not redact_paths:
                print(f"Muestreo {file_path}: error: {exc}")
            continue

        database.clear_video_frame_samples(conn, str(file_path))
        database.insert_video_frame_samples(conn, str(file_path), result.samples)
        conn.commit()
        frames_generated += result.frames_generated
        frames_failed += result.frames_failed
        if result.video_error is None:
            videos_ok += 1
        else:
            videos_error += 1
        if not quiet:
            if redact_paths:
                if videos % 10 == 0 or videos == 1:
                    print(f"Procesados {videos} videos...")
            else:
                print(
                    f"Muestreo {file_path}: "
                    f"{result.frames_generated} generados, {result.frames_failed} fallidos"
                    + (f" ({result.video_error})" if result.video_error else "")
                )

    if verify_readonly:
        after_snapshot = capture_snapshot(target, ignored_dirs=ignored_dirs, on_error=_on_error)
        save_snapshot(after_snapshot, snapshot_dir / "after.json")
        readonly_diff = diff_snapshots(before_snapshot, after_snapshot)
        print()
        print("=== VERIFICACIÓN READ-ONLY ===")
        print(
            f"added={len(readonly_diff.added)} removed={len(readonly_diff.removed)} "
            f"size_changed={len(readonly_diff.size_changed)} "
            f"mtime_changed={len(readonly_diff.mtime_changed)}"
        )
        if readonly_diff.is_clean:
            print("OK: ningún archivo original cambió durante el escaneo.")
        else:
            print(
                "ALERTA: se detectaron cambios en los archivos originales durante el escaneo.",
                file=sys.stderr,
            )
            for label, paths in (
                ("added", readonly_diff.added),
                ("removed", readonly_diff.removed),
                ("size_changed", readonly_diff.size_changed),
                ("mtime_changed", readonly_diff.mtime_changed),
            ):
                for changed_path in paths[:20]:
                    print(f"  [{label}] {changed_path}", file=sys.stderr)

    temporary_frames_remaining = 0
    for ephemeral in ephemeral_dirs:
        if ephemeral.exists():
            remaining_files = [p for p in ephemeral.rglob("*") if p.is_file()]
            temporary_frames_remaining += len(remaining_files)
    elapsed_seconds = time.perf_counter() - start

    print()
    print("=== VIDEO FRAME SAMPLING ===")
    print(f"videos={videos}")
    print(f"videos_ok={videos_ok}")
    print(f"videos_error={videos_error}")
    print(f"frames_generated={frames_generated}")
    print(f"frames_failed={frames_failed}")
    print(f"temporary_frames_remaining={temporary_frames_remaining}")
    print(f"elapsed_seconds={elapsed_seconds:.2f}")
    if walk_errors:
        print(f"Directorios no listados por error de permisos: {len(walk_errors)}")

    conn.close()


def run_detect_faces(
    target: Path,
    db_path: Path = DEFAULT_DB_PATH,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    min_face_size: int = DEFAULT_MIN_FACE_SIZE_PX,
    max_video_samples: int = 30,
    redact_paths: bool = False,
    quiet: bool = False,
    verify_readonly: bool = False,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
) -> None:
    """Detect faces in images and sampled video frames under target."""
    if not target.exists():
        print(f"ERROR: la ruta no existe: {target}", file=sys.stderr)
        print(
            "No se inventará ni escaneará ninguna otra ruta. Verifica el "
            "punto de montaje o crea la carpeta esperada.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not target.is_dir():
        print(f"ERROR: la ruta no es un directorio: {target}", file=sys.stderr)
        raise SystemExit(1)

    start = time.perf_counter()
    ignored_dirs = set(DEFAULT_IGNORED_DIRS)
    walk_errors: list[ScanError] = []

    def _on_error(err: ScanError) -> None:
        walk_errors.append(err)
        if not quiet and not redact_paths:
            print(f"AVISO: no se pudo listar {err.path}: {err.message}", file=sys.stderr)

    before_snapshot = None
    if verify_readonly:
        before_snapshot = capture_snapshot(target, ignored_dirs=ignored_dirs, on_error=_on_error)
        save_snapshot(before_snapshot, snapshot_dir / "before.json")

    conn = database.connect(db_path)
    images_processed = 0
    video_frames_processed = 0
    face_observations = 0
    detector_errors = 0
    attempted_files: set[str] = set()
    files_with_faces: set[str] = set()
    processed_files = 0

    for file_path in iter_files(target, ignored_dirs=ignored_dirs, on_error=_on_error):
        category = classify(file_path.suffix)
        if category == "image":
            images_processed += 1
            processed_files += 1
            attempted_files.add(str(file_path))
            try:
                result = detect_faces_in_image_file(
                    file_path,
                    confidence_threshold=confidence_threshold,
                    min_face_size_px=min_face_size,
                )
            except Exception as exc:  # noqa: BLE001 - one bad image never aborts the run
                detector_errors += 1
                if not quiet and not redact_paths:
                    print(f"Detección {file_path}: error: {exc}")
                continue
            if result.error is not None:
                detector_errors += 1
                if not quiet and not redact_paths:
                    print(f"Detección {file_path}: error: {result.error}")
                continue
            database.clear_face_observations(conn, str(file_path))
            database.insert_face_observations(
                conn, str(file_path), "image", result.faces, timestamp_ms=None
            )
            conn.commit()
            face_observations += len(result.faces)
            if result.faces:
                files_with_faces.add(str(file_path))
            if not quiet:
                if redact_paths:
                    if processed_files % 10 == 0 or processed_files == 1:
                        print(f"Procesados {processed_files} archivos...")
                else:
                    print(f"Detección {file_path}: {len(result.faces)} caras")
        elif category == "video":
            processed_files += 1
            attempted_files.add(str(file_path))
            try:
                video_meta = get_video_metadata(file_path)
                if video_meta.error or video_meta.duration_seconds is None:
                    result_error = video_meta.error or "no duration available"
                    database.clear_face_observations(conn, str(file_path))
                    database.clear_video_frame_samples(conn, str(file_path))
                    conn.commit()
                    detector_errors += 1
                    if not quiet and not redact_paths:
                        print(f"Detección {file_path}: error: {result_error}")
                    continue
                timestamps = compute_sample_timestamps(
                    video_meta.duration_seconds, max_video_samples
                )
                detected: list = []

                def _on_frame(frame_path: Path, timestamp: float, _idx: int) -> None:
                    face_result = detect_faces_in_image_file(
                        frame_path,
                        confidence_threshold=confidence_threshold,
                        min_face_size_px=min_face_size,
                    )
                    detected.append((timestamp, face_result))

                sampling = sample_video_frames(
                    file_path,
                    timestamps,
                    on_frame=_on_frame,
                    keep_debug_frames=False,
                )
            except Exception as exc:  # noqa: BLE001 - one bad video never aborts the run
                database.clear_face_observations(conn, str(file_path))
                database.clear_video_frame_samples(conn, str(file_path))
                conn.commit()
                detector_errors += 1
                if not quiet and not redact_paths:
                    print(f"Detección {file_path}: error: {exc}")
                continue

            if sampling.video_error is not None:
                database.clear_face_observations(conn, str(file_path))
                database.clear_video_frame_samples(conn, str(file_path))
                conn.commit()
                detector_errors += 1
                if not quiet and not redact_paths:
                    print(f"Detección {file_path}: error: {sampling.video_error}")
                continue

            database.clear_face_observations(conn, str(file_path))
            video_faces = 0
            for timestamp, face_result in detected:
                video_frames_processed += 1
                if face_result.error is not None:
                    continue
                database.insert_face_observations(
                    conn,
                    str(file_path),
                    "video_frame",
                    face_result.faces,
                    timestamp_ms=timestamp * 1000,
                )
                face_observations += len(face_result.faces)
                video_faces += len(face_result.faces)
            if video_faces:
                files_with_faces.add(str(file_path))
            database.clear_video_frame_samples(conn, str(file_path))
            database.insert_video_frame_samples(conn, str(file_path), sampling.samples)
            conn.commit()
            if not quiet:
                if redact_paths:
                    if processed_files % 10 == 0 or processed_files == 1:
                        print(f"Procesados {processed_files} archivos...")
                else:
                    print(
                        f"Detección {file_path}: "
                        f"{video_faces} caras en {len(detected)} frames"
                    )

    if verify_readonly:
        after_snapshot = capture_snapshot(target, ignored_dirs=ignored_dirs, on_error=_on_error)
        save_snapshot(after_snapshot, snapshot_dir / "after.json")
        readonly_diff = diff_snapshots(before_snapshot, after_snapshot)
        print()
        print("=== VERIFICACIÓN READ-ONLY ===")
        print(
            f"added={len(readonly_diff.added)} removed={len(readonly_diff.removed)} "
            f"size_changed={len(readonly_diff.size_changed)} "
            f"mtime_changed={len(readonly_diff.mtime_changed)}"
        )
        if readonly_diff.is_clean:
            print("OK: ningún archivo original cambió durante el escaneo.")
        else:
            print(
                "ALERTA: se detectaron cambios en los archivos originales durante el escaneo.",
                file=sys.stderr,
            )
            for label, paths in (
                ("added", readonly_diff.added),
                ("removed", readonly_diff.removed),
                ("size_changed", readonly_diff.size_changed),
                ("mtime_changed", readonly_diff.mtime_changed),
            ):
                for changed_path in paths[:20]:
                    print(f"  [{label}] {changed_path}", file=sys.stderr)

    files_with = len(files_with_faces)
    files_without = len(attempted_files - files_with_faces)
    elapsed_seconds = time.perf_counter() - start

    print()
    print("=== FACE DETECTION ===")
    print(f"images_processed={images_processed}")
    print(f"video_frames_processed={video_frames_processed}")
    print(f"face_observations={face_observations}")
    print(f"files_with_faces={files_with}")
    print(f"files_without_faces={files_without}")
    print(f"detector_errors={detector_errors}")
    print(f"elapsed_seconds={elapsed_seconds:.2f}")
    if walk_errors:
        print(f"Directorios no listados por error de permisos: {len(walk_errors)}")

    conn.close()


def run_cluster_people(
    target: Path,
    db_path: Path = DEFAULT_DB_PATH,
    similarity_threshold: float = DEFAULT_PERSON_SIMILARITY_THRESHOLD,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    min_face_size: int = DEFAULT_MIN_FACE_SIZE_PX,
    max_video_samples: int = 30,
    redact_paths: bool = False,
    quiet: bool = False,
    verify_readonly: bool = False,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
) -> None:
    """Cluster face embeddings into anonymous person labels (P0001, ...).

    Embeddings live only as transient in-memory arrays during this run and
    are discarded when it ends; only the derived anonymous (person label,
    file) membership rows are persisted (full rebuild of
    `person_cluster_members` on every run).
    """
    if not target.exists():
        print(f"ERROR: la ruta no existe: {target}", file=sys.stderr)
        print(
            "No se inventará ni escaneará ninguna otra ruta. Verifica el "
            "punto de montaje o crea la carpeta esperada.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not target.is_dir():
        print(f"ERROR: la ruta no es un directorio: {target}", file=sys.stderr)
        raise SystemExit(1)

    start = time.perf_counter()
    ignored_dirs = set(DEFAULT_IGNORED_DIRS)
    walk_errors: list[ScanError] = []

    def _on_error(err: ScanError) -> None:
        walk_errors.append(err)
        if not quiet and not redact_paths:
            print(f"AVISO: no se pudo listar {err.path}: {err.message}", file=sys.stderr)

    before_snapshot = None
    if verify_readonly:
        before_snapshot = capture_snapshot(target, ignored_dirs=ignored_dirs, on_error=_on_error)
        save_snapshot(before_snapshot, snapshot_dir / "before.json")

    conn = database.connect(db_path)
    observations: list[ClusterableObservation] = []
    file_of_obs: list[tuple[str, str]] = []  # (file_path, file_category) per observation
    obs_details: list[tuple] = []  # per-observation detail row, parallel to file_of_obs
    face_observations_considered = 0
    processed_files = 0

    for file_path in iter_files(target, ignored_dirs=ignored_dirs, on_error=_on_error):
        category = classify(file_path.suffix)
        if category == "image":
            processed_files += 1
            try:
                _w, _h, rows, error = detect_faces_raw(
                    file_path, confidence_threshold=confidence_threshold
                )
            except Exception as exc:  # noqa: BLE001 - one bad image never aborts the run
                if not quiet and not redact_paths:
                    print(f"Clustering {file_path}: error: {exc}")
                continue
            if error is not None or rows is None:
                if error is not None and not quiet and not redact_paths:
                    print(f"Clustering {file_path}: error: {error}")
                if not quiet and redact_paths:
                    if processed_files % 10 == 0 or processed_files == 1:
                        print(f"Procesados {processed_files} archivos...")
                continue
            img = cv2.imread(str(file_path))
            for row in rows:
                face_observations_considered += 1
                w = round(float(row[2]))
                h = round(float(row[3]))
                if w < min_face_size or h < min_face_size:
                    continue
                if img is None:
                    continue
                emb = compute_embedding(img, row)
                if emb is None:
                    continue
                x = max(0, round(float(row[0])))
                y = max(0, round(float(row[1])))
                observations.append(
                    ClusterableObservation(
                        key=(str(file_path), "image", -1, x),
                        embedding=emb,
                    )
                )
                file_of_obs.append((str(file_path), "image"))
                obs_details.append(
                    (str(file_path), "image", None, x, y, w, h, float(row[14]))
                )
            if not quiet:
                if redact_paths:
                    if processed_files % 10 == 0 or processed_files == 1:
                        print(f"Procesados {processed_files} archivos...")
                else:
                    print(f"Clustering {file_path}: {len(rows)} caras detectadas")
        elif category == "video":
            processed_files += 1
            try:
                video_meta = get_video_metadata(file_path)
                if video_meta.error or video_meta.duration_seconds is None:
                    result_error = video_meta.error or "no duration available"
                    if not quiet and not redact_paths:
                        print(f"Clustering {file_path}: error: {result_error}")
                    continue
                timestamps = compute_sample_timestamps(
                    video_meta.duration_seconds, max_video_samples
                )
                face_counts = {"considered": 0}

                def _on_frame(
                    frame_path: Path, timestamp: float, _idx: int
                ) -> None:
                    _fw2, _fh2, frows2, ferror2 = detect_faces_raw(
                        frame_path, confidence_threshold=confidence_threshold
                    )
                    if ferror2 is not None or frows2 is None:
                        return
                    frame_img2 = cv2.imread(str(frame_path))
                    if frame_img2 is None:
                        return
                    timestamp_ms2 = timestamp * 1000
                    for frow2 in frows2:
                        face_counts["considered"] += 1
                        w2 = round(float(frow2[2]))
                        h2 = round(float(frow2[3]))
                        if w2 < min_face_size or h2 < min_face_size:
                            continue
                        emb2 = compute_embedding(frame_img2, frow2)
                        if emb2 is None:
                            continue
                        x2 = max(0, round(float(frow2[0])))
                        y2 = max(0, round(float(frow2[1])))
                        observations.append(
                            ClusterableObservation(
                                key=(str(file_path), "video_frame", timestamp_ms2, x2),
                                embedding=emb2,
                            )
                        )
                        file_of_obs.append((str(file_path), "video"))
                        obs_details.append(
                            (
                                str(file_path),
                                "video_frame",
                                timestamp_ms2,
                                x2,
                                y2,
                                w2,
                                h2,
                                float(frow2[14]),
                            )
                        )

                sampling = sample_video_frames(
                    file_path,
                    timestamps,
                    on_frame=_on_frame,
                    keep_debug_frames=False,
                )
                face_observations_considered += face_counts["considered"]
                if sampling.video_error is not None:
                    if not quiet and not redact_paths:
                        print(f"Clustering {file_path}: error: {sampling.video_error}")
                    continue
            except Exception as exc:  # noqa: BLE001 - one bad video never aborts the run
                if not quiet and not redact_paths:
                    print(f"Clustering {file_path}: error: {exc}")
                continue

            if not quiet:
                if redact_paths:
                    if processed_files % 10 == 0 or processed_files == 1:
                        print(f"Procesados {processed_files} archivos...")
                # else: per-video line omitted to avoid noisy frame counts;
                # the sanitized summary below is the contract.

    usable_embeddings = len(observations)
    rejected_faces = face_observations_considered - usable_embeddings

    distance_threshold = 1.0 - similarity_threshold
    clusters = cluster_people(observations, threshold=distance_threshold)
    labels = assign_person_labels(clusters)

    # For each (file_path, file_category) group, count observations per person.
    counts_by_pair: dict[tuple[str, str, str], int] = {}
    for obs_index, label in labels.items():
        fpath, fcat = file_of_obs[obs_index]
        key = (label, fpath, fcat)
        counts_by_pair[key] = counts_by_pair.get(key, 0) + 1
    rows_to_insert = [
        (label, fpath, fcat, count) for (label, fpath, fcat), count in counts_by_pair.items()
    ]

    observation_rows_to_insert = [
        (labels[obs_index],) + obs_details[obs_index]
        for obs_index in range(len(obs_details))
    ]

    database.clear_all_person_cluster_members(conn)
    database.insert_person_cluster_members(conn, rows_to_insert)
    database.clear_all_person_cluster_observations(conn)
    database.insert_person_cluster_observations(conn, observation_rows_to_insert)
    conn.commit()

    summary = database.fetch_person_cluster_summary(conn)
    person_clusters = len(summary)
    # Distinct files where exactly 2 distinct person labels are present.
    per_file_labels: dict[str, set[str]] = {}
    for label, fpath, _fcat, _count in rows_to_insert:
        per_file_labels.setdefault(fpath, set()).add(label)
    cooccurrence_files = sum(1 for labels_set in per_file_labels.values() if len(labels_set) == 2)

    if verify_readonly:
        after_snapshot = capture_snapshot(target, ignored_dirs=ignored_dirs, on_error=_on_error)
        save_snapshot(after_snapshot, snapshot_dir / "after.json")
        readonly_diff = diff_snapshots(before_snapshot, after_snapshot)
        print()
        print("=== VERIFICACIÓN READ-ONLY ===")
        print(
            f"added={len(readonly_diff.added)} removed={len(readonly_diff.removed)} "
            f"size_changed={len(readonly_diff.size_changed)} "
            f"mtime_changed={len(readonly_diff.mtime_changed)}"
        )
        if readonly_diff.is_clean:
            print("OK: ningún archivo original cambió durante el escaneo.")
        else:
            print(
                "ALERTA: se detectaron cambios en los archivos originales durante el escaneo.",
                file=sys.stderr,
            )
            for label_name, paths in (
                ("added", readonly_diff.added),
                ("removed", readonly_diff.removed),
                ("size_changed", readonly_diff.size_changed),
                ("mtime_changed", readonly_diff.mtime_changed),
            ):
                for changed_path in paths[:20]:
                    print(f"  [{label_name}] {changed_path}", file=sys.stderr)

    elapsed_seconds = time.perf_counter() - start

    print()
    print("=== PERSON CLUSTERING ===")
    print(f"face_observations_considered={face_observations_considered}")
    print(f"usable_embeddings={usable_embeddings}")
    print(f"mapped_observations={len(obs_details)}")
    print(f"rejected_faces={rejected_faces}")
    print(f"person_clusters={person_clusters}")
    print()
    for entry in summary:
        print(
            f"{entry['person_label']} files={entry['files']} "
            f"images={entry['images']} videos={entry['videos']}"
        )
    print()
    print(f"cooccurrence_files={cooccurrence_files}")
    print(f"elapsed_seconds={elapsed_seconds:.2f}")
    if walk_errors:
        print(f"Directorios no listados por error de permisos: {len(walk_errors)}")

    conn.close()


def _require_catalog(conn, db_path: Path) -> None:
    """Exit(1) unless the catalog exists and holds at least one file row."""
    try:
        count_row = conn.execute("SELECT COUNT(*) FROM files").fetchone()
    except Exception:
        count_row = None
    if not count_row or not count_row[0]:
        conn.close()
        print(
            f"ERROR: no existe el catálogo {db_path}. Ejecuta 'scan' primero.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def run_persons(db_path: Path = DEFAULT_DB_PATH) -> None:
    """List effective persons (canonical + unmapped machine clusters)."""
    if not db_path.exists():
        print(
            f"ERROR: no existe el catálogo {db_path}. Ejecuta 'scan' primero.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    conn = database.connect(db_path)
    _require_catalog(conn, db_path)
    summary = person_identity.effective_summary(conn)
    conn.close()
    conn.close()
    print()
    print("=== PERSONS ===")
    for entry in summary:
        name = f" name={entry['display_name']!r}" if entry["display_name"] else ""
        members = ",".join(entry["machine_labels"])
        print(
            f"{entry['person_id']} files={entry['files']} "
            f"images={entry['images']} videos={entry['videos']}"
            f"{name} clusters=[{members}]"
        )
    print(f"persons={len(summary)}")


def _run_person_mutation(db_path: Path, action) -> str:
    """Open the catalog, apply an identity ``action(conn) -> person_id``
    atomically, and return the resulting canonical id."""
    if not db_path.exists():
        print(
            f"ERROR: no existe el catálogo {db_path}. Ejecuta 'scan' primero.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    conn = database.connect(db_path)
    _require_catalog(conn, db_path)
    try:
        person_id = action(conn)
    except ValueError as exc:
        conn.close()
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    conn.commit()
    conn.close()
    return person_id


def run_person_merge(
    refs: list[str],
    db_path: Path = DEFAULT_DB_PATH,
    name: Optional[str] = None,
) -> None:
    person_id = _run_person_mutation(
        db_path, lambda conn: person_identity.merge_labels(conn, refs, name)
    )
    print(f"merged -> {person_id}")


def run_person_rename(
    ref: str,
    db_path: Path = DEFAULT_DB_PATH,
    name: str = "",
) -> None:
    person_id = _run_person_mutation(
        db_path, lambda conn: person_identity.rename_person(conn, ref, name)
    )
    print(f"renamed -> {person_id}")


def run_person_assign(
    file_path: str,
    ref: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    person_id = _run_person_mutation(
        db_path, lambda conn: person_identity.assign_file(conn, file_path, ref)
    )
    print(f"linked {file_path} -> {person_id}")


def run_person_unassign(
    file_path: str,
    ref: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    person_id = _run_person_mutation(
        db_path, lambda conn: person_identity.unassign_file(conn, file_path, ref)
    )
    print(f"unlinked {file_path} -/-> {person_id}")


def run_person_split(
    machine_label: str,
    file_paths: list[str],
    db_path: Path = DEFAULT_DB_PATH,
    name: Optional[str] = None,
) -> None:
    person_id = _run_person_mutation(
        db_path,
        lambda conn: person_identity.split_files(conn, machine_label, file_paths, name),
    )
    print(f"split -> {person_id} files={len(file_paths)}")


def run_search(
    db_path: Path = DEFAULT_DB_PATH,
    show_files: bool = False,
    order_by: str = "path",
    descending: bool = False,
    limit: int = 100,
    **filter_kwargs,
) -> None:
    """Structured catalog search (V3.2). Prints opaque file IDs plus a
    few columns by default; real paths only under --show-files."""
    from .metadata import human_readable_size
    from .person_identity import resolve_file_persons
    from .query import cli_filters, run_query

    if not db_path.exists():
        print(
            f"ERROR: no existe el catálogo {db_path}. Ejecuta 'scan' primero.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    conn = database.connect(db_path)
    _require_catalog(conn, db_path)
    try:
        filters = cli_filters(**filter_kwargs)
    except ValueError as exc:
        conn.close()
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    try:
        rows, total = run_query(
            conn, filters, order_by=order_by, descending=descending, limit=limit
        )
    except ValueError as exc:
        conn.close()
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    persons_by_path = resolve_file_persons(conn)
    conn.close()

    print()
    print("=== SEARCH ===")
    for row in rows:
        persons = ",".join(sorted(persons_by_path.get(row["path"], ()))) or "-"
        line = (
            f"{row['file_id']} category={row['category']} "
            f"size={human_readable_size(row['size_bytes'] or 0)} "
            f"capture={row['capture_date'] or '-'} "
            f"device={row['device_model'] or row['device_make'] or '-'} "
            f"persons={persons}"
        )
        if show_files:
            line += f" path={row['path']}"
        print(line)
    print(f"matches={total} shown={len(rows)}")


def run_organize_preview(
    db_path: Path = DEFAULT_DB_PATH,
    plan_path: Path = DEFAULT_PLAN_PATH,
    decisions_path: Path = DEFAULT_FACE_REVIEW_DECISIONS,
    show_files: bool = False,
    quiet: bool = False,
) -> None:
    """Build the read-only organization preview from the SQLite catalog.

    Only reads the catalog database and the local human decisions file, and
    writes ``reports/organization_plan.json``; never walks the filesystem or
    touches the scanned tree. A missing decisions file means no decisions.
    """
    from .organization_plan import (
        DESTINATION_REVIEW_LOW_EVIDENCE,
        DESTINATION_REVIEW_METADATA,
        DESTINATION_REVIEW_REJECTED,
        DESTINATION_REVIEW_SCAN,
        DESTINATION_REVIEW_UNCERTAIN,
        DESTINATION_SIN_PERSONAS,
        build_organization_plan,
        plan_counts,
        write_plan_json,
    )

    if not db_path.exists():
        print(
            f"ERROR: no existe el catálogo {db_path}. Ejecuta 'scan' primero.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    conn = database.connect(db_path)
    try:
        count_row = conn.execute("SELECT COUNT(*) FROM files").fetchone()
    except Exception:
        print(
            f"ERROR: no existe el catálogo {db_path}. Ejecuta 'scan' primero.",
            file=sys.stderr,
        )
        conn.close()
        raise SystemExit(1)
    if (count_row[0] if count_row else 0) == 0:
        print(
            f"ERROR: no existe el catálogo {db_path}. Ejecuta 'scan' primero.",
            file=sys.stderr,
        )
        conn.close()
        raise SystemExit(1)

    try:
        decisions = load_decisions(decisions_path)
    except ValueError as exc:
        conn.close()
        print(f"ERROR: decisiones inválidas: {exc}", file=sys.stderr)
        raise SystemExit(1)

    entries = build_organization_plan(conn, decisions=decisions)
    counts = plan_counts(entries)
    by_destination: dict[str, int] = counts["by_destination"]
    write_plan_json(entries, plan_path)

    paths_by_destination: dict[str, list[str]] = {}
    if show_files:
        rows = conn.execute("SELECT path FROM files ORDER BY path ASC").fetchall()
        dest_by_index = [e["proposed_destination"] for e in entries]
        for row, dest in zip(rows, dest_by_index):
            paths_by_destination.setdefault(dest, []).append(row["path"])
    conn.close()

    def _files_word(n: int) -> str:
        return "1 archivo" if n == 1 else f"{n} archivos"

    def _leaf_line(name: str, n: int) -> str:
        return f"{name}  {_files_word(n)}"

    lines = ["=== ORGANIZATION PREVIEW ===", ""]

    media_dests = {d: c for d, c in by_destination.items() if d.startswith("Media/")}
    if media_dests:
        lines.append("Media/")
        personas = sorted(
            (d, c) for d, c in media_dests.items() if d.startswith("Media/Personas/")
        )
        parejas = sorted(
            (d, c) for d, c in media_dests.items() if d.startswith("Media/Parejas/")
        )
        grupos = sorted(
            (d, c) for d, c in media_dests.items() if d.startswith("Media/Grupos/")
        )
        sin = media_dests.get(DESTINATION_SIN_PERSONAS)
        if personas:
            lines.append("  Personas/")
            for dest, c in personas:
                label = dest.split("/")[2]
                lines.append(f"    {label}/  {_files_word(c)}")
                if show_files:
                    for p in sorted(paths_by_destination.get(dest, [])):
                        lines.append(f"      - {p}")
        if parejas:
            lines.append("  Parejas/")
            for dest, c in parejas:
                label = dest.split("/")[2]
                lines.append(f"    {label}/  {_files_word(c)}")
                if show_files:
                    for p in sorted(paths_by_destination.get(dest, [])):
                        lines.append(f"      - {p}")
        if grupos:
            lines.append("  Grupos/")
            for dest, c in grupos:
                label = dest.split("/")[2]
                lines.append(f"    {label}/  {_files_word(c)}")
                if show_files:
                    for p in sorted(paths_by_destination.get(dest, [])):
                        lines.append(f"      - {p}")
        if sin:
            lines.append(f"  {_leaf_line('Sin_personas/', sin)}")
            if show_files:
                for p in sorted(
                    paths_by_destination.get(DESTINATION_SIN_PERSONAS, [])
                ):
                    lines.append(f"      - {p}")
        lines.append("")

    revision = {d: c for d, c in by_destination.items() if d.startswith("Revisión/")}
    if revision:
        lines.append("Revisión/")
        for label, dest in (
            ("Persona_incierta/", DESTINATION_REVIEW_UNCERTAIN),
            ("Cluster_rechazado/", DESTINATION_REVIEW_REJECTED),
            ("Baja_evidencia/", DESTINATION_REVIEW_LOW_EVIDENCE),
            ("Metadata_incompleta/", DESTINATION_REVIEW_METADATA),
            ("Errores_de_escaneo/", DESTINATION_REVIEW_SCAN),
        ):
            n = revision.get(dest)
            if not n:
                continue
            lines.append(f"  {_leaf_line(label, n)}")
            if show_files:
                for p in sorted(paths_by_destination.get(dest, [])):
                    lines.append(f"    - {p}")
        lines.append("")

    documentos = sorted(
        (d, c) for d, c in by_destination.items() if d.startswith("Documentos/")
    )
    if documentos:
        lines.append("Documentos/")
        for dest, c in documentos:
            label = dest.split("/")[1]
            lines.append(f"  {_leaf_line(label + '/', c)}")
            if show_files:
                for p in sorted(paths_by_destination.get(dest, [])):
                    lines.append(f"    - {p}")
        lines.append("")

    audios = sorted(
        (d, c) for d, c in by_destination.items() if d.startswith("Audio/")
    )
    if audios:
        lines.append("Audio/")
        for dest, c in audios:
            label = dest.split("/")[1]
            lines.append(f"  {_leaf_line(label + '/', c)}")
            if show_files:
                for p in sorted(paths_by_destination.get(dest, [])):
                    lines.append(f"    - {p}")
        lines.append("")

    if by_destination.get("Archivos_comprimidos/"):
        n = by_destination["Archivos_comprimidos/"]
        lines.append(f"Archivos_comprimidos/  {_files_word(n)}")
        if show_files:
            for p in sorted(paths_by_destination.get("Archivos_comprimidos/", [])):
                lines.append(f"  - {p}")
        lines.append("")

    if by_destination.get("Software/"):
        n = by_destination["Software/"]
        lines.append(f"Software/  {_files_word(n)}")
        if show_files:
            for p in sorted(paths_by_destination.get("Software/", [])):
                lines.append(f"  - {p}")
        lines.append("")

    otros = sorted(
        (d, c) for d, c in by_destination.items() if d.startswith("Otros/")
    )
    if otros:
        lines.append("Otros/")
        for dest, c in otros:
            label = dest.split("/")[1]
            lines.append(f"  {_leaf_line(label + '/', c)}")
            if show_files:
                for p in sorted(paths_by_destination.get(dest, [])):
                    lines.append(f"    - {p}")
        lines.append("")

    person_destinations = sum(
        c for d, c in by_destination.items() if d.startswith("Media/Personas/")
    )
    pair_destinations = sum(
        c for d, c in by_destination.items() if d.startswith("Media/Parejas/")
    )
    group_destinations = sum(
        c for d, c in by_destination.items() if d.startswith("Media/Grupos/")
    )
    review_destinations = sum(
        c for d, c in by_destination.items() if d.startswith("Revisión/")
    )
    sin_personas = by_destination.get(DESTINATION_SIN_PERSONAS, 0)

    lines.append("FLAGS:")
    lines.append(f"exact_duplicates={counts['exact_duplicates']}")
    lines.append(f"visual_similar_groups={counts['visual_similar_groups']}")
    lines.append("")
    lines.append("=== DESTINATIONS ===")
    lines.append(f"person_destinations={person_destinations}")
    lines.append(f"pair_destinations={pair_destinations}")
    lines.append(f"group_destinations={group_destinations}")
    lines.append(f"review_destinations={review_destinations}")
    lines.append(f"sin_personas={sin_personas}")
    lines.append("")
    lines.append("=== CONFIDENCE ===")
    confidence = counts["confidence_levels"]
    lines.append(f"high={confidence['high']}")
    lines.append(f"medium={confidence['medium']}")
    lines.append(f"low={confidence['low']}")
    lines.append("")
    lines.append("=== ACCOUNTING ===")
    lines.append(f"catalog_files={counts['catalog_files']}")
    lines.append(f"planned={counts['planned']}")
    lines.append(f"review={counts['review']}")
    lines.append(f"unsupported={counts['unsupported']}")
    lines.append("")
    lines.append("PREVIEW ONLY")
    lines.append("moved=0")
    lines.append("copied=0")
    lines.append("renamed=0")
    lines.append("deleted=0")
    print("\n".join(lines))


def run_face_review(
    target: Path,
    db_path: Path = DEFAULT_DB_PATH,
    output_dir: Path = DEFAULT_FACE_REVIEW_DIR,
    decisions_path: Path = DEFAULT_FACE_REVIEW_DECISIONS,
    max_previews_per_cluster: int = DEFAULT_MAX_PREVIEWS_PER_CLUSTER,
    thumb_size: int = DEFAULT_THUMBNAIL_MAX_PX,
    quiet: bool = False,
    verify_readonly: bool = False,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
) -> None:
    """Build a fully local visual review of the person clusters.

    Reads the catalog and the scanned tree (images/frames are decoded in
    memory only) and writes a static, offline review site plus local
    thumbnails under ``output_dir``. Never writes inside the scanned tree.
    """
    if not target.exists():
        print(f"ERROR: la ruta no existe: {target}", file=sys.stderr)
        print(
            "No se inventará ni escaneará ninguna otra ruta. Verifica el "
            "punto de montaje o crea la carpeta esperada.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not target.is_dir():
        print(f"ERROR: la ruta no es un directorio: {target}", file=sys.stderr)
        raise SystemExit(1)

    if not db_path.exists():
        print(
            f"ERROR: no existe el catálogo {db_path}. Ejecuta 'scan' primero.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    conn = database.connect(db_path)
    try:
        count_row = conn.execute("SELECT COUNT(*) FROM files").fetchone()
    except Exception:
        conn.close()
        print(
            f"ERROR: no existe el catálogo {db_path}. Ejecuta 'scan' primero.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if (count_row[0] if count_row else 0) == 0:
        conn.close()
        print(
            f"ERROR: no existe el catálogo {db_path}. Ejecuta 'scan' primero.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Safety: the review output must never live inside the scanned tree.
    target_resolved = target.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == target_resolved or target_resolved in output_resolved.parents:
        conn.close()
        print(
            "ERROR: el directorio de salida está dentro del árbol escaneado; "
            "elige una ruta fuera de él.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if verify_readonly:
        snapshot_resolved = snapshot_dir.resolve()
        if (
            snapshot_resolved == target_resolved
            or target_resolved in snapshot_resolved.parents
        ):
            conn.close()
            print(
                "ERROR: el directorio de snapshots está dentro del árbol "
                "escaneado; elige una ruta fuera de él.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    try:
        decisions = load_decisions(decisions_path)
    except ValueError as exc:
        conn.close()
        print(f"ERROR: decisiones inválidas: {exc}", file=sys.stderr)
        raise SystemExit(1)

    start = time.perf_counter()
    ignored_dirs = set(DEFAULT_IGNORED_DIRS)
    walk_errors: list[ScanError] = []

    def _on_error(err: ScanError) -> None:
        # Paths are deliberately not printed: the summary never leaks the
        # scanned tree.
        walk_errors.append(err)

    before_snapshot = None
    if verify_readonly:
        before_snapshot = capture_snapshot(
            target, ignored_dirs=ignored_dirs, on_error=_on_error
        )
        save_snapshot(before_snapshot, snapshot_dir / "before.json")

    manifest, details = build_manifest(
        conn, max_previews_per_cluster=max_previews_per_cluster
    )
    metrics = compute_metrics(manifest, decisions)
    conn.close()

    asset_stats = generate_assets(
        manifest, details, output_dir / "assets", thumb_size=thumb_size
    )
    write_review_site(manifest, decisions, metrics, output_dir)

    if verify_readonly:
        after_snapshot = capture_snapshot(
            target, ignored_dirs=ignored_dirs, on_error=_on_error
        )
        save_snapshot(after_snapshot, snapshot_dir / "after.json")
        readonly_diff = diff_snapshots(before_snapshot, after_snapshot)
        print()
        print("=== VERIFICACIÓN READ-ONLY ===")
        print(
            f"added={len(readonly_diff.added)} removed={len(readonly_diff.removed)} "
            f"size_changed={len(readonly_diff.size_changed)} "
            f"mtime_changed={len(readonly_diff.mtime_changed)}"
        )
        if readonly_diff.is_clean:
            print("OK: ningún archivo original cambió durante el escaneo.")
        else:
            print(
                "ALERTA: se detectaron cambios en los archivos originales durante el escaneo.",
                file=sys.stderr,
            )
            for label_name, paths in (
                ("added", readonly_diff.added),
                ("removed", readonly_diff.removed),
                ("size_changed", readonly_diff.size_changed),
                ("mtime_changed", readonly_diff.mtime_changed),
            ):
                for changed_path in paths[:20]:
                    print(f"  [{label_name}] {changed_path}", file=sys.stderr)

    elapsed_seconds = time.perf_counter() - start

    print()
    print("=== FACE REVIEW ===")
    print(f"clusters={len(manifest['clusters'])}")
    print("review_pages=1")
    print(f"thumbnails={asset_stats['thumbnails']}")
    print(f"assets_bytes={asset_stats['assets_bytes']}")
    print(f"errors={asset_stats['errors']}")
    print(f"singleton_clusters={metrics['singleton_clusters']}")
    print(f"multi_file_clusters={metrics['multi_file_clusters']}")
    print(f"cross_media_clusters={metrics['cross_media_clusters']}")
    print(f"reviewed={metrics['reviewed']}")
    print(f"confirmed={metrics['confirmed']}")
    print(f"rejected={metrics['rejected']}")
    print(f"uncertain={metrics['uncertain']}")
    print(f"unreviewed={metrics['unreviewed']}")
    print(f"output={_display_path(output_dir / 'index.html')}")
    print(f"manifest={_display_path(output_dir / 'manifest.json')}")
    print(f"elapsed_seconds={elapsed_seconds:.2f}")


def run_face_review_import(
    decisions_file: Path,
    db_path: Path = DEFAULT_DB_PATH,
    decisions_path: Path = DEFAULT_FACE_REVIEW_DECISIONS,
    quiet: bool = False,
) -> None:
    """Import human review decisions, validating labels against the catalog.

    On any validation error nothing is written: the existing decisions file
    stays untouched.
    """
    if not decisions_file.exists():
        print(
            f"ERROR: no existe el archivo de decisiones: {decisions_file}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    try:
        payload = json.loads(decisions_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: JSON de decisiones inválido: {exc}", file=sys.stderr)
        raise SystemExit(1)
    try:
        imported = parse_decisions_payload(payload)
    except ValueError as exc:
        print(f"ERROR: JSON de decisiones inválido: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if not db_path.exists():
        print(
            f"ERROR: no existe el catálogo {db_path}. Ejecuta 'scan' primero.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    conn = database.connect(db_path)
    known_labels = {
        row["person_label"]
        for row in conn.execute(
            "SELECT DISTINCT person_label FROM person_cluster_members"
        ).fetchall()
    }
    conn.close()

    errors = validate_decisions(imported, known_labels)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    try:
        existing = load_decisions(decisions_path)
    except ValueError as exc:
        print(f"ERROR: decisiones locales inválidas: {exc}", file=sys.stderr)
        raise SystemExit(1)
    merged = dict(existing)
    merged.update(imported)
    save_decisions(decisions_path, merged)

    counts = {"confirmed": 0, "rejected": 0, "uncertain": 0}
    for value in merged.values():
        if value in counts:
            counts[value] += 1

    print()
    print("=== FACE REVIEW IMPORT ===")
    print(f"imported={len(imported)}")
    print(f"confirmed={counts['confirmed']}")
    print(f"rejected={counts['rejected']}")
    print(f"uncertain={counts['uncertain']}")
    print(f"total_reviewed={sum(counts.values())}")
    print(f"decisions_file={_display_path(decisions_path)}")


def run_summary(db_path: Path = DEFAULT_DB_PATH) -> None:
    if not db_path.exists():
        print(
            f"ERROR: no existe el catálogo {db_path}. Ejecuta 'scan' primero.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    conn = database.connect(db_path)
    # F1: global views cover active files only (None = pre-F1 caller DB
    # without the column: fall back to the legacy unscoped behavior).
    active_paths = database.fetch_active_paths(conn)
    scoped = None if active_paths is None else sorted(active_paths)
    if scoped is not None and not scoped:
        category_totals = []
        dup_groups = []
        error_count = 0
        image_hash_count = 0
        sim_groups = []
        exif_counts = {"with_capture_date": 0, "with_device": 0, "with_gps": 0}
    else:
        category_totals = database.fetch_category_totals(conn, paths=scoped)
        dup_groups = database.fetch_duplicate_groups(conn)
        error_count = database.fetch_error_count(conn, paths=scoped)
        image_hash_count = database.fetch_perceptual_hash_count(conn, paths=scoped)
        sim_groups = database.fetch_similar_groups(conn)
        exif_counts = database.fetch_exif_counts(conn, paths=scoped)
    total_files = sum(row["count"] for row in category_totals)
    total_bytes = sum(row["total_bytes"] or 0 for row in category_totals)
    duplicate_file_count = sum(row["count"] for row in dup_groups)
    recoverable_bytes = sum(
        (row["count"] - 1) * row["size_bytes"] for row in dup_groups
    )
    similar_grouped_file_count = sum(row["count"] for row in sim_groups)

    summary = build_summary_text(
        category_totals=category_totals,
        total_files=total_files,
        total_bytes=total_bytes,
        duplicate_file_count=duplicate_file_count,
        duplicate_group_count=len(dup_groups),
        recoverable_bytes=recoverable_bytes,
        error_count=error_count,
        image_hash_count=image_hash_count,
        similar_group_count=len(sim_groups),
        similar_grouped_file_count=similar_grouped_file_count,
        exif_date_count=exif_counts["with_capture_date"],
        exif_device_count=exif_counts["with_device"],
        exif_gps_count=exif_counts["with_gps"],
    )
    print(summary)
    conn.close()


def run_duplicates(db_path: Path = DEFAULT_DB_PATH) -> None:
    if not db_path.exists():
        print(
            f"ERROR: no existe el catálogo {db_path}. Ejecuta 'scan' primero.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    conn = database.connect(db_path)
    groups = database.fetch_duplicate_groups(conn)
    if not groups:
        print("No se encontraron grupos de duplicados.")
        conn.close()
        return
    for group in groups:
        group_id = group["duplicate_group"]
        size_human = human_readable_size(group["size_bytes"] or 0)
        print(f"{group_id}  ({group['count']} archivos, {size_human} c/u)")
        for file_row in database.fetch_duplicate_files(conn, group_id):
            print(f"  - {file_row['path']}")
    conn.close()


def run_similar_images(db_path: Path = DEFAULT_DB_PATH) -> None:
    if not db_path.exists():
        print(
            f"ERROR: no existe el catálogo {db_path}. Ejecuta 'scan' primero.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    conn = database.connect(db_path)
    groups = database.fetch_similar_groups(conn)
    if not groups:
        print("No se encontraron grupos de imágenes visualmente similares.")
        conn.close()
        return
    for group in groups:
        group_id = group["similar_group"]
        files = database.fetch_similar_files(conn, group_id)
        hashes = [f["perceptual_hash"] for f in files]
        max_distance = 0
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                max_distance = max(max_distance, hamming_distance(hashes[i], hashes[j]))
        print(group_id)
        print(f"  {len(files)} imágenes")
        print(f"  distancia máx: {max_distance}")
        for file_row in files:
            size_human = human_readable_size(file_row["size_bytes"] or 0)
            print(f"  - {file_row['path']} ({size_human}, {file_row['extension']})")
        print()
    conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local_file_organizer",
        description="Inventario local y read-only de archivos personales.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Escanear una carpeta (solo lectura).")
    scan_parser.add_argument("target", type=Path, help="Ruta a escanear.")
    scan_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Ruta al catálogo SQLite.")
    scan_parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="Ruta del reporte CSV.")
    scan_parser.add_argument(
        "--ignore-dir",
        action="append",
        default=None,
        help="Nombre de directorio adicional a ignorar (repetible).",
    )
    scan_parser.add_argument("--quiet", action="store_true", help="Reducir mensajes de progreso.")
    scan_parser.add_argument(
        "--verify-readonly",
        action="store_true",
        help=(
            "Captura un snapshot (path, size, mtime_ns) del árbol antes y "
            "después del escaneo y verifica que ningún archivo original "
            "haya cambiado."
        ),
    )
    scan_parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
        help="Directorio (bajo el repo) donde guardar before.json/after.json.",
    )
    scan_parser.add_argument(
        "--similarity-threshold",
        type=int,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=(
            "Distancia Hamming máxima (0-64) para agrupar imágenes como "
            f"visualmente similares (por defecto {DEFAULT_SIMILARITY_THRESHOLD})."
        ),
    )
    scan_parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Re-analizar todos los archivos aunque su fingerprint "
            "(tamaño+mtime) no haya cambiado. Sin este flag, el scan es "
            "incremental: reutiliza lo ya catalogado."
        ),
    )

    summary_parser = subparsers.add_parser("summary", help="Mostrar resumen del catálogo existente.")
    summary_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    duplicates_parser = subparsers.add_parser("duplicates", help="Listar grupos de duplicados.")
    duplicates_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    similar_parser = subparsers.add_parser(
        "similar-images", help="Listar grupos de imágenes visualmente similares (hash perceptual)."
    )
    similar_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    sample_parser = subparsers.add_parser(
        "sample-frames", help="Muestrear frames de video de forma efímera (solo lectura)."
    )
    sample_parser.add_argument("target", type=Path, help="Ruta a escanear.")
    sample_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    sample_parser.add_argument("--max-samples", type=int, default=30)
    sample_parser.add_argument("--keep-debug-frames", action="store_true")
    sample_parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_FRAMES_DIR)
    sample_parser.add_argument("--redact-paths", action="store_true")
    sample_parser.add_argument("--verify-readonly", action="store_true")
    sample_parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    sample_parser.add_argument("--quiet", action="store_true")

    faces_parser = subparsers.add_parser(
        "detect-faces", help="Detectar caras en imágenes y frames de video (todo local)."
    )
    faces_parser.add_argument("target", type=Path, help="Ruta a escanear.")
    faces_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    faces_parser.add_argument(
        "--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD
    )
    faces_parser.add_argument("--min-face-size", type=int, default=DEFAULT_MIN_FACE_SIZE_PX)
    faces_parser.add_argument("--max-video-samples", type=int, default=30)
    faces_parser.add_argument("--redact-paths", action="store_true")
    faces_parser.add_argument("--verify-readonly", action="store_true")
    faces_parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    faces_parser.add_argument("--quiet", action="store_true")

    cluster_parser = subparsers.add_parser(
        "cluster-people",
        help="Agrupar caras en personas anónimas (P0001, ...) con embeddings locales.",
    )
    cluster_parser.add_argument("target", type=Path, help="Ruta a escanear.")
    cluster_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    cluster_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_PERSON_SIMILARITY_THRESHOLD,
        help=(
            "Similitud coseno mínima (0-1) para considerar dos caras la misma "
            "persona (por defecto "
            f"{DEFAULT_PERSON_SIMILARITY_THRESHOLD}): mayor = más estricto / "
            "menos fusiones. OJO: dirección opuesta a la de `scan "
            "--similarity-threshold`, que es una distancia Hamming donde "
            "menor = más estricto."
        ),
    )
    cluster_parser.add_argument(
        "--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD
    )
    cluster_parser.add_argument("--min-face-size", type=int, default=DEFAULT_MIN_FACE_SIZE_PX)
    cluster_parser.add_argument("--max-video-samples", type=int, default=30)
    cluster_parser.add_argument("--redact-paths", action="store_true")
    cluster_parser.add_argument("--verify-readonly", action="store_true")
    cluster_parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    cluster_parser.add_argument("--quiet", action="store_true")

    preview_parser = subparsers.add_parser(
        "organize-preview", help="Vista previa de organización (solo lectura)."
    )
    preview_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    preview_parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    preview_parser.add_argument(
        "--decisions",
        type=Path,
        default=DEFAULT_FACE_REVIEW_DECISIONS,
        help=(
            "Decisiones humanas de la revisión de clusters (V2.4.1) que "
            "anulan la confianza heurística; un archivo ausente equivale a "
            "no tener decisiones."
        ),
    )
    preview_parser.add_argument(
        "--show-files",
        action="store_true",
        help=(
            "Muestra las rutas reales de cada archivo bajo cada destino. "
            "Pensado para que lo ejecutes tú manualmente — no debe usarse "
            "en sesiones automatizadas."
        ),
    )
    preview_parser.add_argument("--quiet", action="store_true")

    review_parser = subparsers.add_parser(
        "face-review",
        help="Revisión visual local de clusters de personas (todo offline).",
    )
    review_parser.add_argument("target", type=Path, help="Ruta a escanear.")
    review_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    review_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_FACE_REVIEW_DIR
    )
    review_parser.add_argument(
        "--decisions", type=Path, default=DEFAULT_FACE_REVIEW_DECISIONS
    )
    review_parser.add_argument(
        "--max-previews-per-cluster",
        type=int,
        default=DEFAULT_MAX_PREVIEWS_PER_CLUSTER,
    )
    review_parser.add_argument("--thumb-size", type=int, default=DEFAULT_THUMBNAIL_MAX_PX)
    review_parser.add_argument("--verify-readonly", action="store_true")
    review_parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    review_parser.add_argument("--quiet", action="store_true")

    review_import_parser = subparsers.add_parser(
        "face-review-import",
        help="Importar decisiones humanas de la revisión de clusters.",
    )
    review_import_parser.add_argument(
        "decisions_file", type=Path, help="JSON exportado desde la revisión."
    )
    review_import_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    review_import_parser.add_argument(
        "--decisions", type=Path, default=DEFAULT_FACE_REVIEW_DECISIONS
    )
    review_import_parser.add_argument("--quiet", action="store_true")

    persons_parser = subparsers.add_parser(
        "persons",
        help="Listar personas efectivas (canónicas + clusters sin fusionar).",
    )
    persons_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    merge_parser = subparsers.add_parser(
        "person-merge",
        help="Fusionar 2+ clusters/personas en una identidad canónica.",
    )
    merge_parser.add_argument(
        "refs", nargs="+", help="Etiquetas P000N o PERSON_0007 a fusionar."
    )
    merge_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    merge_parser.add_argument("--name", default=None, help="Nombre para mostrar.")

    rename_parser = subparsers.add_parser(
        "person-rename",
        help="Nombrar una persona (un P000N estrena identidad canónica).",
    )
    rename_parser.add_argument("ref", help="Etiqueta P000N o PERSON_0007.")
    rename_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    rename_parser.add_argument("--name", required=True, help="Nombre para mostrar.")

    assign_parser = subparsers.add_parser(
        "person-assign",
        help="Afirmar manualmente que un archivo muestra a una persona.",
    )
    assign_parser.add_argument("file_path", help="Ruta tal como está en el catálogo.")
    assign_parser.add_argument("ref", help="Etiqueta P000N o PERSON_0007.")
    assign_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    unassign_parser = subparsers.add_parser(
        "person-unassign",
        help="Quitar el vínculo persona-archivo (exclusión si era AUTO).",
    )
    unassign_parser.add_argument("file_path", help="Ruta tal como está en el catálogo.")
    unassign_parser.add_argument("ref", help="Etiqueta P000N o PERSON_0007.")
    unassign_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    split_parser = subparsers.add_parser(
        "person-split",
        help="Separar archivos de un cluster a una persona nueva.",
    )
    split_parser.add_argument("machine_label", help="Cluster P000N origen.")
    split_parser.add_argument(
        "--files", nargs="+", required=True, help="Rutas del catálogo a separar."
    )
    split_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    split_parser.add_argument("--name", default=None, help="Nombre para mostrar.")

    search_parser = subparsers.add_parser(
        "search",
        help="Buscar en el catálogo por filtros estructurados (V3.2).",
    )
    search_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    search_parser.add_argument(
        "--category", default=None, help="video,image,... (coma = OR)."
    )
    search_parser.add_argument(
        "--extension", default=None, help=".mp4,.jpg,... (coma = OR)."
    )
    search_parser.add_argument(
        "--person",
        action="append",
        default=None,
        help="P000N, PERSON_0007 o nombre; repetible (todas deben aparecer).",
    )
    search_parser.add_argument(
        "--device", default=None, help="Fabricante o modelo exacto (ej. POCO X6)."
    )
    search_parser.add_argument("--year", default=None, help="Año de captura (2025).")
    search_parser.add_argument(
        "--min-duration", type=float, default=None, help="Segundos."
    )
    search_parser.add_argument(
        "--max-duration", type=float, default=None, help="Segundos."
    )
    search_parser.add_argument(
        "--min-size", default=None, help="Bytes con sufijo (500MB, 1.5G)."
    )
    search_parser.add_argument(
        "--max-size", default=None, help="Bytes con sufijo (500MB, 1.5G)."
    )
    search_parser.add_argument(
        "--has-gps", action="store_true", help="Solo archivos con GPS."
    )
    search_parser.add_argument(
        "--has-device", action="store_true", help="Solo con dispositivo conocido."
    )
    search_parser.add_argument(
        "--has-date", action="store_true", help="Solo con fecha de captura."
    )
    search_parser.add_argument(
        "--bbox",
        default=None,
        help="Caja GPS: lat_min,lat_max,lon_min,lon_max.",
    )
    search_parser.add_argument(
        "--duplicates", action="store_true", help="Solo duplicados exactos."
    )
    search_parser.add_argument(
        "--similar", action="store_true", help="Solo con similar visual."
    )
    search_parser.add_argument(
        "--sort",
        default="path",
        choices=["path", "size_bytes", "capture_date", "duration_seconds", "category"],
    )
    search_parser.add_argument("--desc", action="store_true")
    search_parser.add_argument("--limit", type=int, default=100)
    search_parser.add_argument(
        "--show-files",
        action="store_true",
        help="Incluye rutas reales (solo ejecución manual).",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        run_scan(
            target=args.target,
            db_path=args.db,
            csv_path=args.csv,
            extra_ignored_dirs=args.ignore_dir,
            quiet=args.quiet,
            verify_readonly=args.verify_readonly,
            snapshot_dir=args.snapshot_dir,
            similarity_threshold=args.similarity_threshold,
            full=args.full,
        )
    elif args.command == "summary":
        run_summary(db_path=args.db)
    elif args.command == "duplicates":
        run_duplicates(db_path=args.db)
    elif args.command == "similar-images":
        run_similar_images(db_path=args.db)
    elif args.command == "sample-frames":
        run_sample_frames(
            target=args.target,
            db_path=args.db,
            max_samples=args.max_samples,
            keep_debug_frames=args.keep_debug_frames,
            debug_dir=args.debug_dir,
            redact_paths=args.redact_paths,
            quiet=args.quiet,
            verify_readonly=args.verify_readonly,
            snapshot_dir=args.snapshot_dir,
        )
    elif args.command == "detect-faces":
        run_detect_faces(
            target=args.target,
            db_path=args.db,
            confidence_threshold=args.confidence_threshold,
            min_face_size=args.min_face_size,
            max_video_samples=args.max_video_samples,
            redact_paths=args.redact_paths,
            quiet=args.quiet,
            verify_readonly=args.verify_readonly,
            snapshot_dir=args.snapshot_dir,
        )
    elif args.command == "cluster-people":
        run_cluster_people(
            target=args.target,
            db_path=args.db,
            similarity_threshold=args.similarity_threshold,
            confidence_threshold=args.confidence_threshold,
            min_face_size=args.min_face_size,
            max_video_samples=args.max_video_samples,
            redact_paths=args.redact_paths,
            quiet=args.quiet,
            verify_readonly=args.verify_readonly,
            snapshot_dir=args.snapshot_dir,
        )
    elif args.command == "organize-preview":
        run_organize_preview(
            db_path=args.db,
            plan_path=args.plan_path,
            decisions_path=args.decisions,
            show_files=args.show_files,
            quiet=args.quiet,
        )
    elif args.command == "face-review":
        run_face_review(
            target=args.target,
            db_path=args.db,
            output_dir=args.output_dir,
            decisions_path=args.decisions,
            max_previews_per_cluster=args.max_previews_per_cluster,
            thumb_size=args.thumb_size,
            quiet=args.quiet,
            verify_readonly=args.verify_readonly,
            snapshot_dir=args.snapshot_dir,
        )
    elif args.command == "face-review-import":
        run_face_review_import(
            decisions_file=args.decisions_file,
            db_path=args.db,
            decisions_path=args.decisions,
            quiet=args.quiet,
        )
    elif args.command == "persons":
        run_persons(db_path=args.db)
    elif args.command == "person-merge":
        run_person_merge(refs=args.refs, db_path=args.db, name=args.name)
    elif args.command == "person-rename":
        run_person_rename(ref=args.ref, db_path=args.db, name=args.name)
    elif args.command == "person-assign":
        run_person_assign(file_path=args.file_path, ref=args.ref, db_path=args.db)
    elif args.command == "person-unassign":
        run_person_unassign(file_path=args.file_path, ref=args.ref, db_path=args.db)
    elif args.command == "person-split":
        run_person_split(
            machine_label=args.machine_label,
            file_paths=args.files,
            db_path=args.db,
            name=args.name,
        )
    elif args.command == "search":
        run_search(
            db_path=args.db,
            show_files=args.show_files,
            order_by=args.sort,
            descending=args.desc,
            limit=args.limit,
            category=args.category,
            extension=args.extension,
            persons=args.person,
            device=args.device,
            year=args.year,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            min_size=args.min_size,
            max_size=args.max_size,
            has_gps=args.has_gps,
            has_device=args.has_device,
            has_date=args.has_date,
            bbox=args.bbox,
            duplicates=args.duplicates,
            similar=args.similar,
        )


if __name__ == "__main__":
    main()
