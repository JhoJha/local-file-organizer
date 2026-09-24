"""CSV export and human-readable summaries. Only ever writes under the
project's own reports/ directory, never into the scanned tree."""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Optional

from .metadata import human_readable_size

CSV_COLUMNS = [
    "path", "name", "extension", "category", "size_bytes", "size_human",
    "modified_at", "mime_type", "sha256", "duplicate_group", "similar_group",
    "duration_seconds", "width", "height", "fps", "video_codec",
    "audio_codec", "bitrate", "perceptual_hash", "similarity_method",
    "capture_date", "device_make", "device_model", "device_software",
    "gps_latitude", "gps_longitude", "gps_altitude", "orientation",
    "last_seen_at", "file_state", "extractor_versions",
    "scan_status", "scan_error",
    "metadata_status", "metadata_error",
]


def export_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    """Write the full catalog to a CSV file. Returns the row count."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cur = conn.execute("SELECT * FROM files ORDER BY category, path")
    rows = cur.fetchall()
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            row_dict = dict(row)
            row_dict["size_human"] = human_readable_size(row_dict["size_bytes"] or 0)
            writer.writerow({key: row_dict.get(key) for key in CSV_COLUMNS})
    return len(rows)


def build_summary_text(
    category_totals,
    total_files: int,
    total_bytes: int,
    duplicate_file_count: int,
    duplicate_group_count: int,
    recoverable_bytes: int,
    error_count: int,
    image_hash_count: int = 0,
    similar_group_count: int = 0,
    similar_grouped_file_count: int = 0,
    exif_date_count: int = 0,
    exif_device_count: int = 0,
    exif_gps_count: int = 0,
    reused_count: Optional[int] = None,
    added_count: Optional[int] = None,
    reprocessed_count: Optional[int] = None,
    stale_count: Optional[int] = None,
) -> str:
    lines = ["=== RESUMEN ===", ""]
    for row in category_totals:
        category = row["category"]
        count = row["count"]
        total = row["total_bytes"] or 0
        lines.append(f"{category:<12} {count:>6} archivos   {human_readable_size(total):>10}")

    lines.append("")
    lines.append("Total:")
    lines.append(f"{total_files} archivos")
    lines.append(human_readable_size(total_bytes))

    lines.append("")
    lines.append("Duplicados:")
    lines.append(f"{duplicate_file_count} archivos en grupos duplicados")
    lines.append(f"{duplicate_group_count} grupos")
    lines.append(f"Espacio potencialmente recuperable: {human_readable_size(recoverable_bytes)}")

    lines.append("")
    lines.append("Imágenes similares (hash perceptual):")
    lines.append(f"{image_hash_count} imágenes con hash perceptual calculado")
    lines.append(f"{similar_group_count} grupos de similares detectados")
    lines.append(f"{similar_grouped_file_count} imágenes agrupadas")
    lines.append(f"{max(image_hash_count - similar_grouped_file_count, 0)} imágenes sin grupo (únicas)")

    lines.append("")
    lines.append("Metadata EXIF (ExifTool):")
    lines.append(f"{exif_date_count} archivos con fecha de captura")
    lines.append(f"{exif_device_count} archivos con dispositivo conocido")
    lines.append(f"{exif_gps_count} archivos con GPS")

    if reused_count is not None:
        lines.append("")
        lines.append("Incremental:")
        lines.append(f"{reused_count} reutilizados sin re-analizar")
        lines.append(f"{added_count} nuevos")
        lines.append(f"{reprocessed_count} reprocesados")
        lines.append(f"{stale_count} marcados stale (ya no existen)")

    lines.append("")
    lines.append("Errores:")
    lines.append(f"{error_count} archivos no pudieron analizarse")

    return "\n".join(lines)
