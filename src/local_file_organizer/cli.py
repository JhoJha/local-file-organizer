"""Command-line interface: scan, summary, duplicates, similar-images, sample-frames, detect-faces, cluster-people.

Every code path in this module is read-only with respect to the scanned
tree: it lists files, reads their metadata/bytes (including decoding
images in memory to compute a perceptual hash, and decoding images or
ephemeral video frames in memory for face detection), and writes results
only under this project's own data/ and reports/ directories.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

from . import database
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
from .face_embedding import compute_embedding
from .hashing import DuplicateRecord, find_duplicates
from .metadata import get_basic_metadata, get_video_metadata, human_readable_size
from .video_sampling import compute_sample_timestamps, sample_video_frames
from .perceptual_hash import (
    DEFAULT_SIMILARITY_THRESHOLD,
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
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"
DEFAULT_DEBUG_FRAMES_DIR = PROJECT_ROOT / "data" / "debug_frames"

_PROGRESS_INTERVAL = 500


def _scan_one_file(path: Path) -> database.FileRecord:
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

    count = 0
    for file_path in iter_files(target, ignored_dirs=ignored_dirs, on_error=_on_error):
        record = _scan_one_file(file_path)
        record.duplicate_group = None
        record.similar_group = None
        database.upsert_file(conn, record)
        scanned_paths.append(record.path)
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

    conn.commit()

    def _on_hash_error(path: str, exc: Exception) -> None:
        if not quiet:
            print(f"AVISO: no se pudo calcular hash de {path}: {exc}", file=sys.stderr)
        database.set_scan_error(conn, path, f"hash: {exc}")

    dup_stats = find_duplicates(duplicate_candidates, on_hash_error=_on_hash_error)
    database.clear_duplicate_groups(conn, scanned_paths)
    for path, group_id in dup_stats.group_by_path.items():
        database.set_duplicate_group(conn, path, group_id)
    conn.commit()

    sim_stats = group_similar_images(image_hash_candidates, threshold=similarity_threshold)
    database.clear_similar_groups(conn, scanned_paths)
    for path, group_id in sim_stats.group_by_path.items():
        database.set_similar_group(conn, path, group_id)
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
                observations.append(
                    ClusterableObservation(
                        key=(str(file_path), "image", -1, x),
                        embedding=emb,
                    )
                )
                file_of_obs.append((str(file_path), "image"))
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
                        observations.append(
                            ClusterableObservation(
                                key=(str(file_path), "video_frame", timestamp_ms2, x2),
                                embedding=emb2,
                            )
                        )
                        file_of_obs.append((str(file_path), "video"))

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

    database.clear_all_person_cluster_members(conn)
    database.insert_person_cluster_members(conn, rows_to_insert)
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


def run_summary(db_path: Path = DEFAULT_DB_PATH) -> None:
    if not db_path.exists():
        print(
            f"ERROR: no existe el catálogo {db_path}. Ejecuta 'scan' primero.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    conn = database.connect(db_path)
    category_totals = database.fetch_category_totals(conn)
    total_files = sum(row["count"] for row in category_totals)
    total_bytes = sum(row["total_bytes"] or 0 for row in category_totals)
    dup_groups = database.fetch_duplicate_groups(conn)
    duplicate_file_count = sum(row["count"] for row in dup_groups)
    recoverable_bytes = sum(
        (row["count"] - 1) * row["size_bytes"] for row in dup_groups
    )
    error_count = database.fetch_error_count(conn)
    image_hash_count = database.fetch_perceptual_hash_count(conn)
    sim_groups = database.fetch_similar_groups(conn)
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


if __name__ == "__main__":
    main()
