import os
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

from local_file_organizer.cli import run_scan
from local_file_organizer.metadata import ffprobe_available
from local_file_organizer.snapshot import diff_snapshots, load_snapshot


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


if __name__ == "__main__":
    unittest.main()
