"""Ephemeral video frame sampling (V2.2).

Frames are extracted with the external ``ffmpeg`` binary into a temporary
directory that is always removed afterwards; nothing is ever written into
the scanned tree and frame bytes are never persisted unless the caller
explicitly opts into the ``keep_debug_frames`` dev-only escape hatch.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

FFMPEG_BINARY = "ffmpeg"
FFMPEG_TIMEOUT_SECONDS = 30


def compute_sample_timestamps(duration_seconds: float, max_samples: int = 30) -> list[float]:
    """Return sorted, distinct sample timestamps for a video of the given duration."""
    if max_samples < 1:
        raise ValueError("max_samples must be >= 1")
    if duration_seconds <= 0:
        return []
    margin = max(duration_seconds * 0.05, 0.2)
    if margin > duration_seconds / 2:
        margin = duration_seconds / 2
    usable = duration_seconds - 2 * margin
    if usable <= 0:
        return []
    if duration_seconds <= 60:
        target = 8
    elif duration_seconds <= 300:
        target = 12
    elif duration_seconds <= 900:
        target = 20
    else:
        target = 30
    count = min(target, max_samples)
    if count == 1:
        points = [margin]
    else:
        step = usable / (count - 1)
        points = [margin + i * step for i in range(count)]
    rounded = [round(p, 3) for p in points]
    seen: list[float] = []
    for value in rounded:
        if value not in seen:
            seen.append(value)
    return sorted(seen)


def ffmpeg_available() -> bool:
    return shutil.which(FFMPEG_BINARY) is not None


@dataclass
class FrameSample:
    sample_index: int
    timestamp_seconds: float
    status: str
    error: Optional[str] = None


@dataclass
class SamplingResult:
    timestamps_requested: int
    frames_generated: int
    frames_failed: int
    samples: list[FrameSample]
    video_error: Optional[str] = None


def sample_video_frames(
    path: Path,
    timestamps: list[float],
    on_frame: Optional[Callable[[Path, float, int], None]] = None,
    *,
    keep_debug_frames: bool = False,
    debug_dir: Optional[Path] = None,
) -> SamplingResult:
    """Extract one frame per timestamp with ffmpeg, ephemerally by default."""
    if not ffmpeg_available():
        return SamplingResult(
            timestamps_requested=len(timestamps),
            frames_generated=0,
            frames_failed=0,
            samples=[],
            video_error="ffmpeg not found on PATH",
        )
    if not timestamps:
        return SamplingResult(
            timestamps_requested=0,
            frames_generated=0,
            frames_failed=0,
            samples=[],
            video_error=None,
        )
    if keep_debug_frames:
        if debug_dir is None:
            raise ValueError("debug_dir is required when keep_debug_frames=True")
        debug_dir.mkdir(parents=True, exist_ok=True)
        return _extract_frames(path, timestamps, on_frame, debug_dir, cleanup=False)
    with tempfile.TemporaryDirectory(prefix="lfo-frames-") as tmp:
        return _extract_frames(path, timestamps, on_frame, Path(tmp), cleanup=True)


def _extract_frames(
    path: Path,
    timestamps: list[float],
    on_frame: Optional[Callable[[Path, float, int], None]],
    work_dir: Path,
    *,
    cleanup: bool,
) -> SamplingResult:
    # When cleanup is True, work_dir is a TemporaryDirectory managed by the
    # caller: leaving this function (even via an on_frame exception)
    # removes it. When False (debug escape hatch), files are left in place.
    del cleanup  # lifetime is managed by the caller via context / contract
    samples: list[FrameSample] = []
    generated = 0
    failed = 0
    for sample_index, timestamp in enumerate(timestamps):
        out_path = work_dir / f"frame_{sample_index:04d}.jpg"
        command = [
            FFMPEG_BINARY,
            "-loglevel", "error",
            "-ss", str(timestamp),
            "-i", str(path),
            "-frames:v", "1",
            "-q:v", "2",
            "-y", str(out_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=FFMPEG_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            samples.append(
                FrameSample(
                    sample_index=sample_index,
                    timestamp_seconds=timestamp,
                    status="error",
                    error="ffmpeg timed out",
                )
            )
            failed += 1
            continue
        except OSError as exc:
            samples.append(
                FrameSample(
                    sample_index=sample_index,
                    timestamp_seconds=timestamp,
                    status="error",
                    error=f"ffmpeg failed to launch: {exc}",
                )
            )
            failed += 1
            continue
        if result.returncode != 0:
            samples.append(
                FrameSample(
                    sample_index=sample_index,
                    timestamp_seconds=timestamp,
                    status="error",
                    error=f"ffmpeg exited with code {result.returncode}",
                )
            )
            failed += 1
            continue
        if not out_path.is_file():
            samples.append(
                FrameSample(
                    sample_index=sample_index,
                    timestamp_seconds=timestamp,
                    status="error",
                    error="ffmpeg produced no output file",
                )
            )
            failed += 1
            continue
        if on_frame is not None:
            on_frame(out_path, timestamp, sample_index)
        samples.append(
            FrameSample(
                sample_index=sample_index,
                timestamp_seconds=timestamp,
                status="generated",
            )
        )
        generated += 1
    return SamplingResult(
        timestamps_requested=len(timestamps),
        frames_generated=generated,
        frames_failed=failed,
        samples=samples,
        video_error=None,
    )
