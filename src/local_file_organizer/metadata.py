"""Read-only metadata extraction: general file stats and ffprobe video probing.

Nothing in this module writes, moves, renames or deletes the files it reads.
"""
from __future__ import annotations

import json
import mimetypes
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .classifier import classify

_SIZE_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")

FFPROBE_BINARY = "ffprobe"
FFPROBE_TIMEOUT_SECONDS = 30


def human_readable_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable string (e.g. '31.82 GB')."""
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")
    size = float(num_bytes)
    for unit in _SIZE_UNITS:
        if size < 1024.0 or unit == _SIZE_UNITS[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} {_SIZE_UNITS[-1]}"


@dataclass
class BasicMetadata:
    path: str
    name: str
    extension: str
    category: str
    size_bytes: int
    size_human: str
    modified_at: str
    mime_type: Optional[str]


def get_basic_metadata(path: Path) -> BasicMetadata:
    """Collect cheap, read-only metadata for a single file.

    Only ``os.stat`` and filename inspection are used; file contents are not
    read here.
    """
    stat_result = path.stat()
    extension = path.suffix.lower()
    mime_type, _ = mimetypes.guess_type(path.name)
    modified_at = datetime.fromtimestamp(
        stat_result.st_mtime, tz=timezone.utc
    ).isoformat()
    return BasicMetadata(
        path=str(path),
        name=path.name,
        extension=extension,
        category=classify(extension),
        size_bytes=stat_result.st_size,
        size_human=human_readable_size(stat_result.st_size),
        modified_at=modified_at,
        mime_type=mime_type,
    )


@dataclass
class VideoMetadata:
    duration_seconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    bitrate: Optional[int] = None
    error: Optional[str] = None


def _parse_fps(rate: Optional[str]) -> Optional[float]:
    if not rate:
        return None
    if "/" in rate:
        num_str, _, den_str = rate.partition("/")
        try:
            num, den = float(num_str), float(den_str)
        except ValueError:
            return None
        if den == 0:
            return None
        return round(num / den, 3)
    try:
        return round(float(rate), 3)
    except ValueError:
        return None


def ffprobe_available() -> bool:
    return shutil.which(FFPROBE_BINARY) is not None


def get_video_metadata(path: Path) -> VideoMetadata:
    """Probe a video file read-only via ffprobe. Never mutates the file.

    Any failure (missing binary, timeout, malformed output, non-zero exit)
    is captured in ``VideoMetadata.error`` rather than raised, so a single
    unreadable/corrupt video never aborts the overall scan.
    """
    if not ffprobe_available():
        return VideoMetadata(error="ffprobe not found on PATH")

    command = [
        FFPROBE_BINARY,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return VideoMetadata(error="ffprobe timed out")
    except OSError as exc:
        return VideoMetadata(error=f"ffprobe failed to launch: {exc}")

    if result.returncode != 0:
        return VideoMetadata(error=f"ffprobe exited with code {result.returncode}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return VideoMetadata(error=f"ffprobe returned invalid JSON: {exc}")

    fmt = payload.get("format", {}) or {}
    streams = payload.get("streams", []) or []
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = fmt.get("duration")
    try:
        duration_seconds = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration_seconds = None

    bitrate = fmt.get("bit_rate")
    try:
        bitrate_value = int(bitrate) if bitrate is not None else None
    except (TypeError, ValueError):
        bitrate_value = None

    fps = None
    width = height = None
    video_codec = None
    if video_stream:
        width = video_stream.get("width")
        height = video_stream.get("height")
        video_codec = video_stream.get("codec_name")
        fps = _parse_fps(video_stream.get("avg_frame_rate")) or _parse_fps(
            video_stream.get("r_frame_rate")
        )

    audio_codec = audio_stream.get("codec_name") if audio_stream else None

    return VideoMetadata(
        duration_seconds=duration_seconds,
        width=width,
        height=height,
        fps=fps,
        video_codec=video_codec,
        audio_codec=audio_codec,
        bitrate=bitrate_value,
        error=None,
    )
