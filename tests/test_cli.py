import io
import os
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from PIL import Image

from local_file_organizer.cli import run_scan, run_similar_images
from local_file_organizer.metadata import ffprobe_available
from local_file_organizer.perceptual_hash import compute_dhash, hamming_distance
from local_file_organizer.snapshot import diff_snapshots, load_snapshot


def _save_gradient(path: Path, size=(64, 64), fmt="PNG"):
    img = Image.new("RGB", size)
    pixels = img.load()
    for x in range(size[0]):
        for y in range(size[1]):
            value = int(255 * x / size[0])
            pixels[x, y] = (value, value, value)
    img.save(path, format=fmt)


class RunScanTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.target = Path(self.tmpdir.name) / "target"
        self.target.mkdir()
        self.output_dir = Path(self.tmpdir.name) / "output"
        self.output_dir.mkdir()
        self.db_path = self.output_dir / "catalog.sqlite"
        self.csv_path = self.output_dir / "inventory.csv"

    def test_nonexistent_path_stops_without_scanning_anything(self):
        missing = Path(self.tmpdir.name) / "does-not-exist"
        with self.assertRaises(SystemExit):
            run_scan(missing, db_path=self.db_path, csv_path=self.csv_path, quiet=True)
        self.assertFalse(self.db_path.exists())

    def test_scan_creates_catalog_and_csv_without_touching_originals(self):
        (self.target / "doc.txt").write_text("hello")
        (self.target / "dup1.bin").write_bytes(b"same-content")
        (self.target / "dup2.bin").write_bytes(b"same-content")

        original_listing_before = sorted(os.listdir(self.target))
        original_mtimes = {
            p.name: p.stat().st_mtime for p in self.target.iterdir()
        }

        run_scan(self.target, db_path=self.db_path, csv_path=self.csv_path, quiet=True)

        # Originals must be untouched: same files, same mtimes, nothing new.
        self.assertEqual(sorted(os.listdir(self.target)), original_listing_before)
        for p in self.target.iterdir():
            self.assertEqual(p.stat().st_mtime, original_mtimes[p.name])

        self.assertTrue(self.db_path.exists())
        self.assertTrue(self.csv_path.exists())

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM files ORDER BY path").fetchall()
        self.assertEqual(len(rows), 3)

        dup_rows = [r for r in rows if r["duplicate_group"] is not None]
        self.assertEqual(len(dup_rows), 2)
        self.assertEqual(dup_rows[0]["duplicate_group"], dup_rows[1]["duplicate_group"])
        conn.close()

    def test_rescanning_is_idempotent(self):
        (self.target / "a.txt").write_text("hello")
        run_scan(self.target, db_path=self.db_path, csv_path=self.csv_path, quiet=True)
        run_scan(self.target, db_path=self.db_path, csv_path=self.csv_path, quiet=True)

        conn = sqlite3.connect(str(self.db_path))
        count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)

    def test_unreadable_file_is_recorded_as_error_and_scan_continues(self):
        if os.geteuid() == 0:
            self.skipTest("permission bits are ignored when running as root")

        readable = self.target / "readable.txt"
        readable.write_text("hello")

        # Two files of the same size (so they enter the hashing pipeline),
        # one of them unreadable, to exercise the "file can't be read"
        # path without touching the file's permissions after the fact.
        blocked = self.target / "blocked.bin"
        twin = self.target / "twin.bin"
        blocked.write_bytes(b"x" * 32)
        twin.write_bytes(b"y" * 32)
        blocked.chmod(0o000)
        self.addCleanup(lambda: blocked.chmod(0o644))

        run_scan(self.target, db_path=self.db_path, csv_path=self.csv_path, quiet=True)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = {r["name"]: r for r in conn.execute("SELECT * FROM files").fetchall()}
        conn.close()

        # The scan must complete and every file (including the unreadable
        # one) must still be cataloged with its basic metadata.
        self.assertEqual(set(rows.keys()), {"readable.txt", "blocked.bin", "twin.bin"})
        self.assertEqual(rows["readable.txt"]["scan_status"], "ok")
        self.assertIsNone(rows["readable.txt"]["scan_error"])

        # blocked.bin's content couldn't be hashed (permission denied), and
        # that failure must be recorded rather than silently dropped or
        # crashing the scan.
        self.assertIsNotNone(rows["blocked.bin"]["scan_error"])
        self.assertIsNone(rows["blocked.bin"]["duplicate_group"])

    def test_valid_video_gets_ffprobe_metadata(self):
        if not ffprobe_available():
            self.skipTest("ffprobe not installed")

        video_path = self.target / "clip.mp4"
        subprocess.run(
            [
                "ffmpeg", "-f", "lavfi", "-i", "testsrc=size=32x32:rate=1",
                "-t", "1", "-pix_fmt", "yuv420p", "-y", str(video_path),
            ],
            capture_output=True,
            check=True,
        )

        run_scan(self.target, db_path=self.db_path, csv_path=self.csv_path, quiet=True)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM files WHERE name = 'clip.mp4'").fetchone()
        conn.close()

        self.assertEqual(row["scan_status"], "ok")
        self.assertIsNone(row["scan_error"])
        self.assertEqual(row["metadata_status"], "ok")
        self.assertIsNone(row["metadata_error"])
        self.assertEqual(row["width"], 32)
        self.assertEqual(row["height"], 32)
        self.assertIsNotNone(row["duration_seconds"])

    def test_corrupt_video_metadata_failure_does_not_mark_scan_failed(self):
        if not ffprobe_available():
            self.skipTest("ffprobe not installed")

        fake_video = self.target / "not-really-a-video.mp4"
        fake_video.write_bytes(b"this is not a valid video container")

        run_scan(self.target, db_path=self.db_path, csv_path=self.csv_path, quiet=True)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM files WHERE name = 'not-really-a-video.mp4'"
        ).fetchone()
        conn.close()

        # The file could be stat'd and cataloged fine: that must stay "ok".
        self.assertEqual(row["category"], "video")
        self.assertEqual(row["scan_status"], "ok")
        self.assertIsNone(row["scan_error"])
        # Only the specialized video-metadata extraction failed.
        self.assertEqual(row["metadata_status"], "error")
        self.assertIsNotNone(row["metadata_error"])

    def test_non_video_file_has_skipped_metadata_status(self):
        (self.target / "doc.txt").write_text("hello")
        run_scan(self.target, db_path=self.db_path, csv_path=self.csv_path, quiet=True)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM files WHERE name = 'doc.txt'").fetchone()
        conn.close()

        self.assertEqual(row["scan_status"], "ok")
        self.assertEqual(row["metadata_status"], "skipped")
        self.assertIsNone(row["metadata_error"])

    def test_valid_image_gets_perceptual_hash(self):
        image_path = self.target / "photo.png"
        _save_gradient(image_path)

        run_scan(self.target, db_path=self.db_path, csv_path=self.csv_path, quiet=True)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM files WHERE name = 'photo.png'").fetchone()
        conn.close()

        self.assertEqual(row["scan_status"], "ok")
        self.assertEqual(row["category"], "image")
        self.assertEqual(row["metadata_status"], "ok")
        self.assertIsNone(row["metadata_error"])
        self.assertIsNotNone(row["perceptual_hash"])
        self.assertEqual(row["similarity_method"], "dhash8x8")

    def test_corrupt_image_metadata_failure_does_not_mark_scan_failed(self):
        fake_image = self.target / "not-really-an-image.png"
        fake_image.write_bytes(b"this is not a valid PNG file")

        run_scan(self.target, db_path=self.db_path, csv_path=self.csv_path, quiet=True)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM files WHERE name = 'not-really-an-image.png'"
        ).fetchone()
        conn.close()

        # The basic inventory step succeeded (it's a real, readable file).
        self.assertEqual(row["category"], "image")
        self.assertEqual(row["scan_status"], "ok")
        self.assertIsNone(row["scan_error"])
        # Only the specialized perceptual-hash extraction failed.
        self.assertEqual(row["metadata_status"], "error")
        self.assertIsNotNone(row["metadata_error"])
        self.assertIsNone(row["perceptual_hash"])

    def test_similar_images_are_grouped_and_duplicate_group_stays_separate(self):
        # Two images with different exact bytes (PNG vs JPEG of the same
        # picture): no SHA-256 duplicate, but they should land in the same
        # perceptual similarity group. A third, unrelated image stays
        # ungrouped.
        _save_gradient(self.target / "a.png", fmt="PNG")
        _save_gradient(self.target / "b.jpg", fmt="JPEG")
        checker = Image.new("RGB", (64, 64))
        pixels = checker.load()
        for x in range(64):
            for y in range(64):
                pixels[x, y] = (255, 255, 255) if (x // 8 + y // 8) % 2 == 0 else (0, 0, 0)
        checker.save(self.target / "c-checker.png", format="PNG")

        run_scan(self.target, db_path=self.db_path, csv_path=self.csv_path, quiet=True)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = {r["name"]: r for r in conn.execute("SELECT * FROM files").fetchall()}
        conn.close()

        self.assertIsNone(rows["a.png"]["duplicate_group"])
        self.assertIsNone(rows["b.jpg"]["duplicate_group"])
        self.assertIsNotNone(rows["a.png"]["similar_group"])
        self.assertEqual(rows["a.png"]["similar_group"], rows["b.jpg"]["similar_group"])
        self.assertIsNone(rows["c-checker.png"]["similar_group"])

        output = io.StringIO()
        with redirect_stdout(output):
            run_similar_images(db_path=self.db_path)
        printed = output.getvalue()
        self.assertIn(rows["a.png"]["similar_group"], printed)
        self.assertIn("2 imágenes", printed)
        self.assertIn("distancia máx:", printed)
        self.assertIn(str(self.target / "a.png"), printed)
        self.assertNotIn("c-checker.png", printed)

    def test_rescanning_images_does_not_duplicate_similar_groups(self):
        _save_gradient(self.target / "a.png")
        _save_gradient(self.target / "b.png")

        run_scan(self.target, db_path=self.db_path, csv_path=self.csv_path, quiet=True)
        run_scan(self.target, db_path=self.db_path, csv_path=self.csv_path, quiet=True)

        conn = sqlite3.connect(str(self.db_path))
        count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        group_count = conn.execute(
            "SELECT COUNT(DISTINCT similar_group) FROM files WHERE similar_group IS NOT NULL"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(count, 2)
        self.assertEqual(group_count, 1)

    def test_similar_groups_reset_when_threshold_tightens(self):
        # V2.1.1 reset: scan with threshold=8 groups a pair whose distance
        # is in (2, 8]; rescanning the same tree with threshold=2 must leave
        # both ungrouped with no stale group IDs behind.
        base = self.target / "base.png"
        _save_gradient(base)
        base_hash = compute_dhash(base)

        variant_path = None
        variant_distance = None
        for square_size, pos in (
            (16, (16, 16)),
            (16, (32, 32)),
            (16, (48, 48)),
            (32, (0, 0)),
            (32, (48, 48)),
            (8, (48, 48)),
            (32, (32, 32)),
        ):
            candidate = self.target / f"variant-{square_size}-{pos[0]}.png"
            img = Image.open(base).copy()
            pixels = img.load()
            for x in range(pos[0], min(64, pos[0] + square_size)):
                for y in range(pos[1], min(64, pos[1] + square_size)):
                    pixels[x, y] = (0, 0, 0)
            img.save(candidate, format="PNG")
            distance = hamming_distance(base_hash, compute_dhash(candidate))
            if 2 < distance <= 8:
                variant_path = candidate
                variant_distance = distance
                break
            candidate.unlink()
        self.assertIsNotNone(
            variant_path, "expected a variant with distance in (2, 8]"
        )
        assert variant_path is not None  # for type checkers
        self.assertGreater(variant_distance, 2)
        self.assertLessEqual(variant_distance, 8)

        run_scan(
            self.target,
            db_path=self.db_path,
            csv_path=self.csv_path,
            quiet=True,
            similarity_threshold=8,
        )
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = {r["name"]: r for r in conn.execute("SELECT * FROM files").fetchall()}
        conn.close()
        self.assertIsNotNone(rows[base.name]["similar_group"])
        self.assertEqual(
            rows[base.name]["similar_group"],
            rows[variant_path.name]["similar_group"],
        )
        # Different bytes: perceptual grouping must not create SHA-256 groups.
        self.assertIsNone(rows[base.name]["duplicate_group"])
        self.assertIsNone(rows[variant_path.name]["duplicate_group"])
        old_group = rows[base.name]["similar_group"]

        run_scan(
            self.target,
            db_path=self.db_path,
            csv_path=self.csv_path,
            quiet=True,
            similarity_threshold=2,
        )
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows_after = {
            r["name"]: r for r in conn.execute("SELECT * FROM files").fetchall()
        }
        remaining = conn.execute(
            "SELECT DISTINCT similar_group FROM files WHERE similar_group IS NOT NULL"
        ).fetchall()
        conn.close()
        self.assertIsNone(rows_after[base.name]["similar_group"])
        self.assertIsNone(rows_after[variant_path.name]["similar_group"])
        self.assertEqual(remaining, [])
        # The old group ID must not linger on any row.
        self.assertNotIn(old_group, [r["similar_group"] for r in rows_after.values()])

    def test_verify_readonly_reports_clean_diff(self):
        (self.target / "a.txt").write_text("hello")
        (self.target / "b.bin").write_bytes(b"world")
        snapshot_dir = self.output_dir / "snapshots"

        run_scan(
            self.target,
            db_path=self.db_path,
            csv_path=self.csv_path,
            quiet=True,
            verify_readonly=True,
            snapshot_dir=snapshot_dir,
        )

        self.assertTrue((snapshot_dir / "before.json").exists())
        self.assertTrue((snapshot_dir / "after.json").exists())

        before = load_snapshot(snapshot_dir / "before.json")
        after = load_snapshot(snapshot_dir / "after.json")
        diff = diff_snapshots(before, after)
        self.assertTrue(diff.is_clean)


class SampleFramesCliTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.target = Path(self.tmpdir.name) / "target"
        self.target.mkdir()
        self.output_dir = Path(self.tmpdir.name) / "output"
        self.output_dir.mkdir()
        self.db_path = self.output_dir / "catalog.sqlite"

    def _make_video(self, name: str, duration: float = 2.0) -> Path:
        out = self.target / name
        subprocess.run(
            [
                "ffmpeg", "-f", "lavfi", "-i",
                f"testsrc=duration={duration}:size=160x120:rate=10",
                "-y", str(out),
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
        return out

    @unittest.skipUnless(
        ffprobe_available() and shutil.which("ffmpeg") is not None,
        "ffmpeg/ffprobe not installed",
    )
    def test_sample_frames_cli_end_to_end(self):
        from local_file_organizer.cli import run_sample_frames

        self._make_video("clip1.mp4")
        self._make_video("clip2.mp4")
        (self.target / "notes.txt").write_text("not a video")

        snapshot_dir = self.output_dir / "snapshots"
        output = io.StringIO()
        with redirect_stdout(output):
            run_sample_frames(
                self.target,
                db_path=self.db_path,
                quiet=True,
                verify_readonly=True,
                snapshot_dir=snapshot_dir,
            )
        printed = output.getvalue()
        self.assertIn("=== VIDEO FRAME SAMPLING ===", printed)
        self.assertIn("temporary_frames_remaining=0", printed)
        self.assertIn("=== VERIFICACIÓN READ-ONLY ===", printed)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM video_frame_samples").fetchall()
        video_paths = {r["file_path"] for r in rows}
        self.assertTrue(rows)
        self.assertIn(str(self.target / "clip1.mp4"), video_paths)
        self.assertIn(str(self.target / "clip2.mp4"), video_paths)
        self.assertNotIn(str(self.target / "notes.txt"), video_paths)
        count_before = conn.execute(
            "SELECT COUNT(*) FROM video_frame_samples WHERE file_path = ?",
            (str(self.target / "clip1.mp4"),),
        ).fetchone()[0]
        conn.close()

        before = load_snapshot(snapshot_dir / "before.json")
        after = load_snapshot(snapshot_dir / "after.json")
        self.assertTrue(diff_snapshots(before, after).is_clean)

        # Re-running must replace rows, not accumulate them.
        with redirect_stdout(io.StringIO()):
            run_sample_frames(self.target, db_path=self.db_path, quiet=True)
        conn = sqlite3.connect(str(self.db_path))
        count_after = conn.execute(
            "SELECT COUNT(*) FROM video_frame_samples WHERE file_path = ?",
            (str(self.target / "clip1.mp4"),),
        ).fetchone()[0]
        conn.close()
        self.assertEqual(count_before, count_after)

    @unittest.skipUnless(
        ffprobe_available() and shutil.which("ffmpeg") is not None,
        "ffmpeg/ffprobe not installed",
    )
    def test_sample_frames_redact_paths_hides_temp_dir(self):
        from local_file_organizer.cli import main

        self._make_video("clip.mp4")
        output = io.StringIO()
        with redirect_stdout(output):
            main(
                [
                    "sample-frames", str(self.target),
                    "--db", str(self.db_path),
                    "--redact-paths",
                    "--quiet",
                ]
            )
        printed = output.getvalue()
        self.assertIn("=== VIDEO FRAME SAMPLING ===", printed)
        self.assertNotIn(str(self.target), printed)
        self.assertNotIn(self.tmpdir.name, printed)

    @unittest.skipUnless(
        ffprobe_available() and shutil.which("ffmpeg") is not None,
        "ffmpeg/ffprobe not installed",
    )
    def test_sample_frames_redact_paths_hides_walk_errors(self):
        from unittest import mock

        from local_file_organizer.cli import run_sample_frames
        from local_file_organizer.scanner import ScanError
        from local_file_organizer.scanner import iter_files as real_iter_files

        self._make_video("clip.mp4")
        secret = str(self.target / "secret-dir")

        def _fake_iter_files(root, ignored_dirs=frozenset(), on_error=None):
            if on_error is not None:
                on_error(ScanError(path=secret, message="permission denied"))
            yield from real_iter_files(
                root, ignored_dirs=ignored_dirs, on_error=on_error
            )

        with mock.patch(
            "local_file_organizer.cli.iter_files", side_effect=_fake_iter_files
        ):
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                run_sample_frames(
                    self.target,
                    db_path=self.db_path,
                    redact_paths=True,
                    quiet=False,
                )
            redacted = out.getvalue() + err.getvalue()
            self.assertIn("=== VIDEO FRAME SAMPLING ===", out.getvalue())
            self.assertNotIn(secret, redacted)

            # Sanity check: without --redact-paths the same walk error IS
            # reported with its path, proving the redacted run above
            # actually exercised the error path.
            out2, err2 = io.StringIO(), io.StringIO()
            with redirect_stdout(out2), redirect_stderr(err2):
                run_sample_frames(
                    self.target,
                    db_path=self.db_path,
                    redact_paths=False,
                    quiet=False,
                )
            self.assertIn(secret, err2.getvalue())


class DetectFacesCliTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.target = Path(self.tmpdir.name) / "target"
        self.target.mkdir()
        self.output_dir = Path(self.tmpdir.name) / "output"
        self.output_dir.mkdir()
        self.db_path = self.output_dir / "catalog.sqlite"

    def _make_video(self, name: str, duration: float = 2.0) -> Path:
        out = self.target / name
        subprocess.run(
            [
                "ffmpeg", "-f", "lavfi", "-i",
                f"testsrc=duration={duration}:size=160x120:rate=10",
                "-y", str(out),
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
        return out

    def test_detect_faces_images_zero_vs_mocked(self):
        from unittest import mock

        from local_file_organizer import cli as cli_module
        from local_file_organizer.cli import run_detect_faces
        from local_file_organizer.face_detection import (
            FaceDetectionResult,
            FaceObservation,
        )

        _save_gradient(self.target / "zero.png")
        _save_gradient(self.target / "face.png")

        real_detect = cli_module.detect_faces_in_image_file

        def _fake_detect(path, **kwargs):
            if Path(path).name == "face.png":
                return FaceDetectionResult(
                    width=64,
                    height=64,
                    faces=[
                        FaceObservation(
                            bbox_x=5,
                            bbox_y=5,
                            bbox_width=30,
                            bbox_height=30,
                            detection_confidence=0.9,
                            source_width=64,
                            source_height=64,
                        )
                    ],
                )
            return real_detect(path, **kwargs)

        with mock.patch.object(cli_module, "detect_faces_in_image_file", side_effect=_fake_detect):
            output = io.StringIO()
            with redirect_stdout(output):
                run_detect_faces(self.target, db_path=self.db_path, quiet=True)
        printed = output.getvalue()
        self.assertIn("=== FACE DETECTION ===", printed)
        self.assertIn("images_processed=2", printed)
        self.assertIn("face_observations=1", printed)
        self.assertIn("files_with_faces=1", printed)
        self.assertIn("files_without_faces=1", printed)
        self.assertIn("detector_errors=0", printed)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM face_observations").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_type"], "image")
        self.assertIsNone(rows[0]["timestamp_ms"])
        count_before = len(rows)
        conn.close()

        # Re-running replaces rows rather than accumulating.
        with mock.patch.object(cli_module, "detect_faces_in_image_file", side_effect=_fake_detect):
            with redirect_stdout(io.StringIO()):
                run_detect_faces(self.target, db_path=self.db_path, quiet=True)
        conn = sqlite3.connect(str(self.db_path))
        count_after = conn.execute("SELECT COUNT(*) FROM face_observations").fetchone()[0]
        conn.close()
        self.assertEqual(count_before, count_after)

    @unittest.skipUnless(
        ffprobe_available() and shutil.which("ffmpeg") is not None,
        "ffmpeg/ffprobe not installed",
    )
    def test_detect_faces_video_mocked_persists_frames_and_samples(self):
        from unittest import mock

        from local_file_organizer import cli as cli_module
        from local_file_organizer.cli import run_detect_faces
        from local_file_organizer.face_detection import (
            FaceDetectionResult,
            FaceObservation,
        )

        self._make_video("clip.mp4")
        _save_gradient(self.target / "still.png")

        def _fake_detect(path, **kwargs):
            return FaceDetectionResult(
                width=160,
                height=120,
                faces=[
                    FaceObservation(
                        bbox_x=1,
                        bbox_y=1,
                        bbox_width=40,
                        bbox_height=40,
                        detection_confidence=0.8,
                        source_width=160,
                        source_height=120,
                    )
                ],
            )

        with mock.patch.object(cli_module, "detect_faces_in_image_file", side_effect=_fake_detect):
            output = io.StringIO()
            with redirect_stdout(output):
                snapshot_dir = self.output_dir / "snapshots"
                run_detect_faces(
                    self.target,
                    db_path=self.db_path,
                    quiet=True,
                    verify_readonly=True,
                    snapshot_dir=snapshot_dir,
                )
        printed = output.getvalue()
        self.assertIn("=== FACE DETECTION ===", printed)
        self.assertIn("=== VERIFICACIÓN READ-ONLY ===", printed)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        video_rows = conn.execute(
            "SELECT * FROM face_observations WHERE file_path = ?",
            (str(self.target / "clip.mp4"),),
        ).fetchall()
        self.assertTrue(video_rows)
        self.assertTrue(all(r["source_type"] == "video_frame" for r in video_rows))
        self.assertTrue(all(r["timestamp_ms"] is not None for r in video_rows))
        # detect-faces also leaves the sampling diagnostics behind.
        sample_rows = conn.execute(
            "SELECT * FROM video_frame_samples WHERE file_path = ?",
            (str(self.target / "clip.mp4"),),
        ).fetchall()
        self.assertTrue(sample_rows)
        conn.close()

        before = load_snapshot(snapshot_dir / "before.json")
        after = load_snapshot(snapshot_dir / "after.json")
        self.assertTrue(diff_snapshots(before, after).is_clean)

    @unittest.skipUnless(
        ffprobe_available() and shutil.which("ffmpeg") is not None,
        "ffmpeg/ffprobe not installed",
    )
    def test_detect_faces_redact_paths_hides_walk_errors(self):
        from unittest import mock

        from local_file_organizer.cli import run_detect_faces
        from local_file_organizer.scanner import ScanError
        from local_file_organizer.scanner import iter_files as real_iter_files

        _save_gradient(self.target / "photo.png")
        secret = str(self.target / "secret-dir")

        def _fake_iter_files(root, ignored_dirs=frozenset(), on_error=None):
            if on_error is not None:
                on_error(ScanError(path=secret, message="permission denied"))
            yield from real_iter_files(
                root, ignored_dirs=ignored_dirs, on_error=on_error
            )

        with mock.patch(
            "local_file_organizer.cli.iter_files", side_effect=_fake_iter_files
        ):
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                run_detect_faces(
                    self.target,
                    db_path=self.db_path,
                    redact_paths=True,
                    quiet=False,
                )
            redacted = out.getvalue() + err.getvalue()
            self.assertIn("=== FACE DETECTION ===", out.getvalue())
            self.assertNotIn(secret, redacted)

            out2, err2 = io.StringIO(), io.StringIO()
            with redirect_stdout(out2), redirect_stderr(err2):
                run_detect_faces(
                    self.target,
                    db_path=self.db_path,
                    redact_paths=False,
                    quiet=False,
                )
            self.assertIn(secret, err2.getvalue())


class ClusterPeopleCliTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.target = Path(self.tmpdir.name) / "target"
        self.target.mkdir()
        self.output_dir = Path(self.tmpdir.name) / "output"
        self.output_dir.mkdir()
        self.db_path = self.output_dir / "catalog.sqlite"
        self.frames_dir = Path(self.tmpdir.name) / "frames"
        self.frames_dir.mkdir()

    def _emb(self, index: int):
        import numpy as np

        v = np.zeros(128, dtype=np.float32)
        v[index] = 1.0
        return v

    def _row(self, x, y=5, w=30, h=30, confidence=0.9):
        full = [0.0] * 15
        full[0], full[1], full[2], full[3] = x, y, w, h
        # Plausible landmark points inside the box (content is irrelevant:
        # compute_embedding is mocked in these tests).
        full[4], full[5] = x + 8, y + 10
        full[6], full[7] = x + 22, y + 10
        full[8], full[9] = x + 15, y + 18
        full[10], full[11] = x + 10, y + 24
        full[12], full[13] = x + 20, y + 24
        full[14] = confidence
        return full

    def _rows_array(self, rows):
        import numpy as np

        return np.array(rows, dtype=np.float32)

    def _embedding_by_x(self, emb_a, emb_b):
        def _fake_compute(img, row):
            if round(float(row[0])) == 10:
                return emb_a
            return emb_b

        return _fake_compute

    def _run(self, **kwargs):
        from local_file_organizer.cli import run_cluster_people

        output = io.StringIO()
        with redirect_stdout(output):
            run_cluster_people(self.target, db_path=self.db_path, quiet=True, **kwargs)
        return output.getvalue()

    def _db_rows(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT person_label, file_path, file_category, observation_count "
            "FROM person_cluster_members ORDER BY person_label, file_path"
        ).fetchall()
        conn.close()
        return [tuple(r) for r in rows]

    def test_two_persons_and_cooccurrence(self):
        from unittest import mock

        from local_file_organizer import cli as cli_module

        emb_a, emb_b = self._emb(0), self._emb(1)
        _save_gradient(self.target / "a.png")
        _save_gradient(self.target / "b.png")
        _save_gradient(self.target / "c.png")

        rows_by_name = {
            "a.png": self._rows_array([self._row(10)]),
            "b.png": self._rows_array([self._row(50)]),
            "c.png": self._rows_array([self._row(10), self._row(50)]),
        }

        def _fake_detect(path, **kwargs):
            return (64, 64, rows_by_name[Path(path).name], None)

        with (
            mock.patch.object(cli_module, "detect_faces_raw", side_effect=_fake_detect),
            mock.patch.object(
                cli_module, "compute_embedding", side_effect=self._embedding_by_x(emb_a, emb_b)
            ),
        ):
            printed = self._run()

        self.assertIn("=== PERSON CLUSTERING ===", printed)
        self.assertIn("face_observations_considered=4", printed)
        self.assertIn("usable_embeddings=4", printed)
        self.assertIn("rejected_faces=0", printed)
        self.assertIn("person_clusters=2", printed)
        # Deterministic key order: a.png's embedding sorts first -> P0001.
        self.assertIn("P0001 files=2 images=2 videos=0", printed)
        self.assertIn("P0002 files=2 images=2 videos=0", printed)
        # c.png holds two distinct persons -> PAIR_-like co-occurrence.
        self.assertIn("cooccurrence_files=1", printed)

        rows = self._db_rows()
        self.assertEqual(len(rows), 4)
        by_pair = {(r[0], Path(r[1]).name): (r[2], r[3]) for r in rows}
        self.assertEqual(by_pair[("P0001", "a.png")], ("image", 1))
        self.assertEqual(by_pair[("P0002", "b.png")], ("image", 1))
        self.assertEqual(by_pair[("P0001", "c.png")], ("image", 1))
        self.assertEqual(by_pair[("P0002", "c.png")], ("image", 1))

        # Re-running fully replaces rather than accumulating.
        with (
            mock.patch.object(cli_module, "detect_faces_raw", side_effect=_fake_detect),
            mock.patch.object(
                cli_module, "compute_embedding", side_effect=self._embedding_by_x(emb_a, emb_b)
            ),
        ):
            self._run()
        self.assertEqual(self._db_rows(), rows)

    def test_video_frames_of_same_person_collapse_to_one_row(self):
        from types import SimpleNamespace
        from unittest import mock

        from local_file_organizer import cli as cli_module
        from local_file_organizer.video_sampling import SamplingResult

        emb_a = self._emb(0)
        clip = self.target / "clip.mp4"
        clip.write_bytes(b"fake-video-bytes")

        def _fake_detect(path, **kwargs):
            return (160, 120, self._rows_array([self._row(10)]), None)

        def _fake_sample(path, timestamps, on_frame=None, **kwargs):
            for idx, ts in enumerate(timestamps):
                frame = self.frames_dir / f"frame_{idx:04d}.png"
                _save_gradient(frame)
                if on_frame is not None:
                    on_frame(frame, ts, idx)
            return SamplingResult(
                timestamps_requested=len(timestamps),
                frames_generated=len(timestamps),
                frames_failed=0,
                samples=[],
                video_error=None,
            )

        with (
            mock.patch.object(
                cli_module,
                "get_video_metadata",
                return_value=SimpleNamespace(error=None, duration_seconds=2.0),
            ),
            mock.patch.object(cli_module, "sample_video_frames", side_effect=_fake_sample),
            mock.patch.object(cli_module, "detect_faces_raw", side_effect=_fake_detect),
            mock.patch.object(cli_module, "compute_embedding", return_value=emb_a),
        ):
            printed = self._run()

        # duration=2.0s -> 8 sample timestamps; every frame yields the same
        # fake embedding, so the whole video is ONE person, not 8 people.
        from local_file_organizer.video_sampling import compute_sample_timestamps

        n_frames = len(compute_sample_timestamps(2.0, 30))
        self.assertIn(f"face_observations_considered={n_frames}", printed)
        self.assertIn(f"usable_embeddings={n_frames}", printed)
        self.assertIn("rejected_faces=0", printed)
        self.assertIn("person_clusters=1", printed)
        self.assertIn("P0001 files=1 images=0 videos=1", printed)
        self.assertIn("cooccurrence_files=0", printed)

        rows = self._db_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "P0001")
        self.assertEqual(rows[0][1], str(clip))
        self.assertEqual(rows[0][2], "video")
        self.assertEqual(rows[0][3], n_frames)

    def test_rejected_counts_cover_filter_and_embedding_failure(self):
        from unittest import mock

        from local_file_organizer import cli as cli_module

        emb_a = self._emb(0)
        _save_gradient(self.target / "ok.png")
        _save_gradient(self.target / "tiny.png")
        _save_gradient(self.target / "bad-emb.png")

        rows_by_name = {
            "ok.png": self._rows_array([self._row(10)]),
            "tiny.png": self._rows_array([self._row(10, w=5, h=5)]),
            "bad-emb.png": self._rows_array([self._row(50)]),
        }

        def _fake_detect(path, **kwargs):
            return (64, 64, rows_by_name[Path(path).name], None)

        def _fake_compute(img, row):
            if round(float(row[0])) == 10:
                return emb_a
            return None

        with (
            mock.patch.object(cli_module, "detect_faces_raw", side_effect=_fake_detect),
            mock.patch.object(cli_module, "compute_embedding", side_effect=_fake_compute),
        ):
            printed = self._run()

        # 3 raw rows examined; 1 survives (1 too small, 1 embedding None).
        self.assertIn("face_observations_considered=3", printed)
        self.assertIn("usable_embeddings=1", printed)
        self.assertIn("rejected_faces=2", printed)
        self.assertIn("person_clusters=1", printed)

    def test_redact_paths_hides_paths_but_not_person_labels(self):
        from unittest import mock

        from local_file_organizer import cli as cli_module
        from local_file_organizer.cli import run_cluster_people
        from local_file_organizer.scanner import ScanError
        from local_file_organizer.scanner import iter_files as real_iter_files

        emb_a = self._emb(0)
        _save_gradient(self.target / "photo.png")
        secret = str(self.target / "secret-dir")

        def _fake_iter_files(root, ignored_dirs=frozenset(), on_error=None):
            if on_error is not None:
                on_error(ScanError(path=secret, message="permission denied"))
            yield from real_iter_files(root, ignored_dirs=ignored_dirs, on_error=on_error)

        def _fake_detect(path, **kwargs):
            return (64, 64, self._rows_array([self._row(10)]), None)

        with (
            mock.patch("local_file_organizer.cli.iter_files", side_effect=_fake_iter_files),
            mock.patch.object(cli_module, "detect_faces_raw", side_effect=_fake_detect),
            mock.patch.object(cli_module, "compute_embedding", return_value=emb_a),
        ):
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                run_cluster_people(
                    self.target,
                    db_path=self.db_path,
                    redact_paths=True,
                    quiet=False,
                )
            redacted = out.getvalue() + err.getvalue()
            self.assertIn("=== PERSON CLUSTERING ===", out.getvalue())
            self.assertNotIn(secret, redacted)
            self.assertNotIn(str(self.target / "photo.png"), redacted)
            # Person labels are NOT paths: unaffected by --redact-paths.
            self.assertIn("P0001", out.getvalue())

            # Sanity check: without --redact-paths the same walk error IS
            # reported with its path, proving the redacted run above
            # actually exercised the error path.
            out2, err2 = io.StringIO(), io.StringIO()
            with redirect_stdout(out2), redirect_stderr(err2):
                run_cluster_people(
                    self.target,
                    db_path=self.db_path,
                    redact_paths=False,
                    quiet=False,
                )
            self.assertIn(secret, err2.getvalue())

    def test_verify_readonly_reports_clean_diff(self):
        from unittest import mock

        from local_file_organizer import cli as cli_module
        from local_file_organizer.snapshot import diff_snapshots, load_snapshot

        emb_a = self._emb(0)
        _save_gradient(self.target / "photo.png")
        snapshot_dir = self.output_dir / "snapshots"

        def _fake_detect(path, **kwargs):
            return (64, 64, self._rows_array([self._row(10)]), None)

        with (
            mock.patch.object(cli_module, "detect_faces_raw", side_effect=_fake_detect),
            mock.patch.object(cli_module, "compute_embedding", return_value=emb_a),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                from local_file_organizer.cli import run_cluster_people

                run_cluster_people(
                    self.target,
                    db_path=self.db_path,
                    quiet=True,
                    verify_readonly=True,
                    snapshot_dir=snapshot_dir,
                )
        printed = output.getvalue()
        self.assertIn("=== PERSON CLUSTERING ===", printed)
        self.assertIn("=== VERIFICACIÓN READ-ONLY ===", printed)

        self.assertTrue((snapshot_dir / "before.json").exists())
        self.assertTrue((snapshot_dir / "after.json").exists())
        before = load_snapshot(snapshot_dir / "before.json")
        after = load_snapshot(snapshot_dir / "after.json")
        self.assertTrue(diff_snapshots(before, after).is_clean)


class OrganizePreviewCliTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.output_dir = Path(self.tmpdir.name) / "output"
        self.output_dir.mkdir()
        self.db_path = self.output_dir / "catalog.sqlite"
        self.plan_path = self.output_dir / "organization_plan.json"

    def _seed(self):
        from local_file_organizer import database

        conn = database.connect(self.db_path)
        base = str(Path(self.tmpdir.name) / "media")
        self.p_single = base + "/single.jpg"
        self.p_zero = base + "/zero.jpg"
        self.p_doc = base + "/informe.pdf"
        self.p_review = base + "/rota.jpg"
        self.p_bad = base + "/ilegible.jpg"
        conn.execute("DELETE FROM person_cluster_members")
        database.upsert_file(
            conn,
            database.FileRecord(
                path=self.p_single, name="single.jpg", extension=".jpg",
                category="image", scan_status="ok", metadata_status="ok"),
        )
        database.upsert_file(
            conn,
            database.FileRecord(
                path=self.p_zero, name="zero.jpg", extension=".jpg",
                category="image", scan_status="ok", metadata_status="ok"),
        )
        database.upsert_file(
            conn,
            database.FileRecord(
                path=self.p_doc, name="informe.pdf", extension=".pdf",
                category="document", scan_status="ok", metadata_status="skipped",
                duplicate_group="DUP_00001"),
        )
        database.upsert_file(
            conn,
            database.FileRecord(
                path=self.p_review, name="rota.jpg", extension=".jpg",
                category="image", scan_status="ok", metadata_status="error"),
        )
        database.upsert_file(
            conn,
            database.FileRecord(
                path=self.p_bad, name="ilegible.jpg", extension=".jpg",
                category="image", scan_status="error", metadata_status="skipped",
                similar_group="SIM_00001"),
        )
        database.insert_person_cluster_members(
            conn, [("P0004", self.p_single, "image", 1)])
        conn.commit()
        conn.close()
        return [self.p_single, self.p_zero, self.p_doc, self.p_review, self.p_bad]

    def test_end_to_end_tree_accounting_and_privacy(self):
        import json

        from local_file_organizer.cli import run_organize_preview

        secrets = self._seed()
        output = io.StringIO()
        with redirect_stdout(output):
            run_organize_preview(
                db_path=self.db_path, plan_path=self.plan_path, quiet=True)
        printed = output.getvalue()
        for token in (
            "=== ORGANIZATION PREVIEW ===",
            "Media/", "Personas/", "P0004/", "Sin_personas/",
            "Documentos/", "PDF/",
            "Revisión/", "Metadata_incompleta/", "Errores_de_escaneo/",
            "FLAGS:", "exact_duplicates=", "visual_similar_groups=",
            "=== ACCOUNTING ===",
            "catalog_files=5", "planned=3", "review=1", "unsupported=1",
            "PREVIEW ONLY", "moved=0", "copied=0", "renamed=0", "deleted=0",
        ):
            self.assertIn(token, printed)
        for secret in secrets:
            self.assertNotIn(secret, printed)

        self.assertTrue(self.plan_path.exists())
        data = json.loads(self.plan_path.read_text(encoding="utf-8"))
        self.assertEqual(len(data), 5)
        for entry in data:
            self.assertEqual(
                set(entry.keys()),
                {"file_id", "category", "bucket", "proposed_destination",
                 "reason_codes", "flags", "confidence", "action"})
        serialized = self.plan_path.read_text(encoding="utf-8")
        for secret in secrets:
            self.assertNotIn(secret, serialized)

    def test_show_files_reveals_paths(self):
        from local_file_organizer.cli import run_organize_preview

        secrets = self._seed()
        default_out = io.StringIO()
        with redirect_stdout(default_out):
            run_organize_preview(
                db_path=self.db_path, plan_path=self.plan_path, quiet=True)
        self.assertNotIn(secrets[0], default_out.getvalue())

        shown_out = io.StringIO()
        with redirect_stdout(shown_out):
            run_organize_preview(
                db_path=self.db_path, plan_path=self.plan_path,
                show_files=True, quiet=True)
        self.assertIn(secrets[0], shown_out.getvalue())

    def test_missing_or_empty_catalog_errors(self):
        from local_file_organizer.cli import run_organize_preview

        missing = self.output_dir / "no-existe.sqlite"
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                with redirect_stdout(io.StringIO()):
                    run_organize_preview(
                        db_path=missing, plan_path=self.plan_path, quiet=True)

        from local_file_organizer import database

        conn = database.connect(self.db_path)
        conn.close()
        err = io.StringIO()
        with self.assertRaises(SystemExit):
            with redirect_stderr(err):
                with redirect_stdout(io.StringIO()):
                    run_organize_preview(
                        db_path=self.db_path, plan_path=self.plan_path, quiet=True)
        self.assertIn("ERROR: no existe el catálogo", err.getvalue())


if __name__ == "__main__":
    unittest.main()
