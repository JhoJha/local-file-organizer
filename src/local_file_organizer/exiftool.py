"""Read-only EXIF/XMP metadata extraction via ExifTool (V3.0).

Nothing in this module writes, moves, renames or deletes the files it
reads: ExifTool is only ever invoked in read mode (``-j`` JSON output to
stdout) against a single existing file.

Design rules, mirroring :mod:`metadata`:

- The ExifTool binary is an *optional system dependency* (``apt install
  libimage-exiftool-perl``). When it is missing, every call returns an
  ``ExifMetadata`` with ``error`` set instead of raising, so a scan
  behaves exactly like V2.5.1.
- EXIF is *enrichment, never verdict*: a file without EXIF tags (e.g.
  stripped by WhatsApp/Telegram/recompression) yields ``None`` fields
  and no error — the program records "unknown" by leaving the columns
  NULL, it never invents a device, date or location.
- Any failure (missing binary, timeout, malformed output, non-zero
  exit, corrupt file) is captured in ``ExifMetadata.error`` rather than
  raised, so a single bad file never aborts the overall scan.

Known limitation: one ExifTool subprocess per file (~0.2-0.5 s, the
perl interpreter startup dominates). Fine for pilot scale; at tens of
thousands of files this wants ``exiftool -stay_open`` batch mode, which
is explicitly left as future work.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .scanner import is_regular_file

EXIFTOOL_BINARY = "exiftool"
EXIFTOOL_TIMEOUT_SECONDS = 30

# Incrementality contract (V3.2/F1): stored per file in
# files.extractor_versions; bump to force EXIF re-extraction.
EXIFTOOL_VERSION = "exiftool_v1"

# Capture-date candidates, in priority order. Different containers store
# the capture moment under different tags: stills use DateTimeOriginal,
# QuickTime/MP4 uses CreateDate/MediaCreateDate/TrackCreateDate, some
# cameras and phones write CreationDate. The first usable value wins.
_DATE_TAGS = (
    "DateTimeOriginal",
    "CreateDate",
    "MediaCreateDate",
    "TrackCreateDate",
    "CreationDate",
)

_DATETIME_RE = re.compile(
    r"^(\d{4}):(\d{2}):(\d{2})[ ](\d{2}):(\d{2}):(\d{2})"
)


@dataclass
class ExifMetadata:
    """EXIF-derived enrichment for one file. All fields are ``None`` when
    the file carries no such metadata (stripped/never present) — NULL in
    the catalog means "unknown", never a guess."""

    capture_date: Optional[str] = None  # normalized "YYYY-MM-DDTHH:MM:SS", naive
    device_make: Optional[str] = None
    device_model: Optional[str] = None
    device_software: Optional[str] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude: Optional[float] = None
    orientation: Optional[int] = None
    error: Optional[str] = None


def exiftool_available() -> bool:
    return shutil.which(EXIFTOOL_BINARY) is not None


def normalize_exif_datetime(raw: object) -> Optional[str]:
    """Normalize an ExifTool datetime to ``YYYY-MM-DDTHH:MM:SS``.

    Returns ``None`` for anything unusable: empty values, the
    all-zeros placeholder some containers write (``0000:00:00 ...``),
    out-of-range components, or non-strings. Never raises.
    """
    if not isinstance(raw, str):
        return None
    match = _DATETIME_RE.match(raw.strip())
    if not match:
        return None
    year, month, day, hour, minute, second = (int(g) for g in match.groups())
    if year == 0 and month == 0 and day == 0:
        return None
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    if hour > 23 or minute > 59 or second > 60:  # 60 = leap second
        return None
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"


def _parse_coordinate(raw: object, low: float, high: float) -> Optional[float]:
    """Parse a GPS coordinate, accepting the decimal floats ``-n`` mode
    produces. Out-of-range or unparseable values become ``None``."""
    if isinstance(raw, bool):
        return None
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not (low <= value <= high):
        return None
    return value


def _clean_text(raw: object) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    return text or None


def parse_exif_tags(tags: dict) -> ExifMetadata:
    """Map a decoded ExifTool JSON object to :class:`ExifMetadata`.

    ``tags`` is one element of the ``-j`` output array. Pure function of
    its input (no subprocess, no filesystem) so it is unit-testable
    without the binary. Never raises on hostile input.
    """
    if not isinstance(tags, dict):
        return ExifMetadata(error="exiftool returned an unexpected JSON shape")

    capture_date = None
    for tag in _DATE_TAGS:
        capture_date = normalize_exif_datetime(tags.get(tag))
        if capture_date is not None:
            break

    orientation = tags.get("Orientation")
    if isinstance(orientation, bool) or not isinstance(orientation, int):
        try:
            orientation = int(orientation)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            orientation = None
    if orientation not in (1, 2, 3, 4, 5, 6, 7, 8):
        orientation = None

    return ExifMetadata(
        capture_date=capture_date,
        device_make=_clean_text(tags.get("Make")),
        device_model=_clean_text(tags.get("Model")),
        device_software=_clean_text(tags.get("Software")),
        gps_latitude=_parse_coordinate(tags.get("GPSLatitude"), -90.0, 90.0),
        gps_longitude=_parse_coordinate(tags.get("GPSLongitude"), -180.0, 180.0),
        gps_altitude=_parse_coordinate(tags.get("GPSAltitude"), -11000.0, 9000.0),
        orientation=orientation,
    )


def get_exif_metadata(path: Path) -> ExifMetadata:
    """Extract EXIF/XMP metadata for one file, read-only. Never raises.

    Mirrors ``metadata.get_video_metadata``: any failure is captured in
    ``ExifMetadata.error``. A file with no EXIF tags is *not* a failure —
    it yields empty fields (unknown), which the catalog stores as NULL.
    """
    if not exiftool_available():
        return ExifMetadata(error="exiftool not found on PATH")

    if not is_regular_file(path):
        return ExifMetadata(error="not a regular file: skipped (FIFO/socket/device?)")

    command = [EXIFTOOL_BINARY, "-j", "-n", "-s", str(path)]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=EXIFTOOL_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ExifMetadata(error="exiftool timed out")
    except OSError as exc:
        return ExifMetadata(error=f"exiftool failed to launch: {exc}")

    if result.returncode != 0:
        stderr = (result.stderr or "").strip().splitlines()
        detail = stderr[0] if stderr else f"exit code {result.returncode}"
        return ExifMetadata(error=f"exiftool failed: {detail}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return ExifMetadata(error=f"exiftool returned invalid JSON: {exc}")

    if not isinstance(payload, list) or not payload:
        return ExifMetadata(error="exiftool returned empty output")
    return parse_exif_tags(payload[0])
