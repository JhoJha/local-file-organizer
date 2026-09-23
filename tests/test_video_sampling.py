from __future__ import annotations

import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_file_organizer import video_sampling
from local_file_organizer.video_sampling import (
    compute_sample_timestamps,
    sample_video_frames,
)

FFMPEG_PRESENT = shutil.which("ffmpeg") is not None


def _make_video(out_path: Path, duration: float = 2.0) -> None:
    subprocess.run(
        [
            "ffmpeg", "-f", "lavfi", "-i",
            f"testsrc=duration={duration}:size=160x120:rate=10",
            "-y", str(out_path),
        ],
        capture_output=True,
        timeout=30,
        check=True,
    )


class ComputeSampleTimestampsTests(unittest.TestCase):
    def test_short_duration_tier(self):
        stamps = compute_sample_timestamps(30.0)
        self.assertEqual(len(stamps), 8)

    def test_long_duration_tier(self):
        stamps = compute_sample_timestamps(1200.0)
        self.assertEqual(len(stamps), 30)

    def test_mid_tiers(self):
        self.assertEqual(len(compute_sample_timestamps(200.0)), 12)
        self.assertEqual(len(compute_sample_timestamps(600.0)), 20)

    def test_max_samples_clamp_reduces_count(self):
        full = compute_sample_timestamps(1200.0, max_samples=30)
        clamped = compute_sample_timestamps(1200.0, max_samples=5)
        self.assertEqual(len(full), 30)
        self.assertEqual(len(clamped), 5)

    def test_max_samples_invalid_raises(self):
        with self.assertRaises(ValueError):
            compute_sample_timestamps(30.0, max_samples=0)

    def test_determinism(self):
        self.assertEqual(
            compute_sample_timestamps(123.456),
            compute_sample_timestamps(123.456),
        )

    def test_sorted_no_duplicates(self):
        stamps = compute_sample_timestamps(45.0)
        self.assertEqual(stamps, sorted(stamps))
        self.assertEqual(len(stamps), len(set(stamps)))

    def test_never_includes_endpoints(self):
        duration = 100.0
        stamps = compute_sample_timestamps(duration)
        self.assertTrue(stamps)
        self.assertNotIn(0, stamps)
        self.assertNotIn(duration, stamps)
        self.assertGreater(min(stamps), 0)
        self.assertLess(max(stamps), duration)

    def test_non_positive_duration_returns_empty(self):
        self.assertEqual(compute_sample_timestamps(0), [])
        self.assertEqual(compute_sample_timestamps(-5), [])

    def test_very_short_duration_does_not_crash(self):
        stamps = compute_sample_timestamps(1.0)
        self.assertEqual(stamps, sorted(set(stamps)))


@unittest.skipUnless(FFMPEG_PRESENT, "ffmpeg not installed")
class SampleVideoFramesIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.video = Path(self.tmpdir.name) / "clip.mp4"
        _make_video(self.video, duration=2.0)

    def test_frames_generated_and_temp_cleaned_and_readonly(self):
        before_stat = self.video.stat()
        seen: list[Path] = []

        def on_frame(frame_path: Path, timestamp: float, index: int) -> None:
            seen.append(frame_path)
            self.assertTrue(frame_path.is_file())

        result = sample_video_frames(
            self.video, [0.3, 0.6, 0.9], on_frame=on_frame
        )
        self.assertIsNone(result.video_error)
        self.assertEqual(result.timestamps_requested, 3)
        self.assertEqual(result.frames_generated, 3)
        self.assertEqual(result.frames_failed, 0)
        self.assertTrue(all(s.status == "generated" for s in result.samples))
        self.assertTrue(seen)
        for parent in {p.parent for p in seen}:
            self.assertFalse(parent.exists())
        after_stat = self.video.stat()
        self.assertEqual(after_stat.st_size, before_stat.st_size)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)

    def test_corrupt_video_gives_per_sample_errors(self):
        corrupt = Path(self.tmpdir.name) / "corrupt.mp4"
        corrupt.write_bytes(b"\x00\x01\x02garbage-not-a-video" * 64)
        seen_dirs: list[Path] = []

        def on_frame(frame_path: Path, timestamp: float, index: int) -> None:
            seen_dirs.append(frame_path.parent)

        result = sample_video_frames(corrupt, [0.3, 0.6], on_frame=on_frame)
        self.assertIsNone(result.video_error)
        self.assertTrue(result.samples)
        self.assertTrue(all(s.status == "error" for s in result.samples))
        self.assertGreater(result.frames_failed, 0)
        for parent in set(seen_dirs) | {p.parent for p in seen_dirs}:
            self.assertFalse(parent.exists())

    def test_cleanup_on_exception(self):
        parents: list[Path] = []

        def boom(frame_path: Path, timestamp: float, index: int) -> None:
            parents.append(frame_path.parent)
            raise RuntimeError("consumer exploded")

        with self.assertRaises(RuntimeError):
            sample_video_frames(self.video, [0.3, 0.6], on_frame=boom)
        self.assertTrue(parents)
        for parent in set(parents):
            self.assertFalse(parent.exists())


class FfmpegMissingTests(unittest.TestCase):
    def test_missing_binary_returns_video_error(self):
        with mock.patch.object(video_sampling, "ffmpeg_available", return_value=False):
            result = sample_video_frames(Path("whatever.mp4"), [0.5])
        self.assertIsNotNone(result.video_error)
        self.assertEqual(result.samples, [])


class DatabaseHelpersTests(unittest.TestCase):
    def test_clear_and_insert_replace_rows(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        from local_file_organizer import database

        database.init_db(conn)
        samples = [
            video_sampling.FrameSample(sample_index=0, timestamp_seconds=0.5, status="generated"),
            video_sampling.FrameSample(
                sample_index=1, timestamp_seconds=1.0, status="error", error="boom"
            ),
        ]
        database.insert_video_frame_samples(conn, "/v/clip.mp4", samples)
        counts = database.fetch_video_frame_sample_counts(conn, "/v/clip.mp4")
        self.assertEqual(counts["generated"], 1)
        self.assertEqual(counts["error"], 1)
        database.clear_video_frame_samples(conn, "/v/clip.mp4")
        database.insert_video_frame_samples(conn, "/v/clip.mp4", samples[:1])
        counts = database.fetch_video_frame_sample_counts(conn, "/v/clip.mp4")
        self.assertEqual(counts["generated"], 1)
        self.assertEqual(counts["error"], 0)
        conn.close()


if __name__ == "__main__":
    unittest.main()
