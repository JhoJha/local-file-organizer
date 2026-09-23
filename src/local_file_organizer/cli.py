"""Command-line interface: scan, summary, duplicates.

Every code path in this module is read-only with respect to the scanned
tree: it lists files, reads their metadata/bytes, and writes results only
under this project's own data/ and reports/ directories.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from . import database
from .classifier import classify
from .hashing import DuplicateRecord, find_duplicates
from .metadata import get_basic_metadata, get_video_metadata, human_readable_size
from .reporting import build_summary_text, export_csv
from .scanner import DEFAULT_IGNORED_DIRS, ScanError, iter_files
from .snapshot import capture_snapshot, diff_snapshots, save_snapshot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "catalog.sqlite"
DEFAULT_CSV_PATH = PROJECT_ROOT / "reports" / "inventory.csv"
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"

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

    count = 0
    for file_path in iter_files(target, ignored_dirs=ignored_dirs, on_error=_on_error):
        record = _scan_one_file(file_path)
        record.duplicate_group = None
        database.upsert_file(conn, record)
        scanned_paths.append(record.path)
        if record.scan_status == "ok":
            duplicate_candidates.append(
                DuplicateRecord(path=record.path, size_bytes=record.size_bytes)
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
    )
    print()
    print(summary)
    print()
    print(f"Catálogo SQLite: {db_path}")
    print(f"Reporte CSV: {csv_path}")
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

    summary = build_summary_text(
        category_totals=category_totals,
        total_files=total_files,
        total_bytes=total_bytes,
        duplicate_file_count=duplicate_file_count,
        duplicate_group_count=len(dup_groups),
        recoverable_bytes=recoverable_bytes,
        error_count=error_count,
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

    summary_parser = subparsers.add_parser("summary", help="Mostrar resumen del catálogo existente.")
    summary_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    duplicates_parser = subparsers.add_parser("duplicates", help="Listar grupos de duplicados.")
    duplicates_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

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
        )
    elif args.command == "summary":
        run_summary(db_path=args.db)
    elif args.command == "duplicates":
        run_duplicates(db_path=args.db)


if __name__ == "__main__":
    main()
