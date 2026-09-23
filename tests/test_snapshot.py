import os
import tempfile
import unittest
from pathlib import Path

from local_file_organizer.snapshot import (
    capture_snapshot,
    diff_snapshots,
    load_snapshot,
    save_snapshot,
)


class CaptureSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)

    def test_captures_size_and_mtime_for_nested_files(self):
        (self.root / "sub").mkdir()
        (self.root / "sub" / "a.txt").write_text("hello")
        (self.root / "b.txt").write_text("hi")

        snapshot = capture_snapshot(self.root)

        self.assertEqual(set(snapshot.keys()), {"sub/a.txt", "b.txt"})
        self.assertEqual(snapshot["sub/a.txt"].size, 5)
        self.assertIsInstance(snapshot["sub/a.txt"].mtime_ns, int)
        self.assertIsNone(snapshot["sub/a.txt"].sha256)

    def test_include_hash_computes_sha256(self):
        (self.root / "a.txt").write_text("hello")
        snapshot = capture_snapshot(self.root, include_hash=True)
        self.assertIsNotNone(snapshot["a.txt"].sha256)

    def test_snapshot_never_writes_inside_root(self):
        (self.root / "a.txt").write_text("hello")
        before_listing = sorted(p.name for p in self.root.iterdir())
        capture_snapshot(self.root)
        after_listing = sorted(p.name for p in self.root.iterdir())
        self.assertEqual(before_listing, after_listing)


class DiffSnapshotsTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)

    def test_clean_diff_when_nothing_changes(self):
        (self.root / "a.txt").write_text("hello")
        before = capture_snapshot(self.root)
        after = capture_snapshot(self.root)
        diff = diff_snapshots(before, after)
        self.assertTrue(diff.is_clean)
        self.assertEqual(diff.added, [])
        self.assertEqual(diff.removed, [])
        self.assertEqual(diff.size_changed, [])
        self.assertEqual(diff.mtime_changed, [])

    def test_detects_added_file(self):
        (self.root / "a.txt").write_text("hello")
        before = capture_snapshot(self.root)
        (self.root / "b.txt").write_text("new")
        after = capture_snapshot(self.root)

        diff = diff_snapshots(before, after)
        self.assertEqual(diff.added, ["b.txt"])
        self.assertFalse(diff.is_clean)

    def test_detects_removed_file(self):
        (self.root / "a.txt").write_text("hello")
        (self.root / "b.txt").write_text("bye")
        before = capture_snapshot(self.root)
        (self.root / "b.txt").unlink()
        after = capture_snapshot(self.root)

        diff = diff_snapshots(before, after)
        self.assertEqual(diff.removed, ["b.txt"])

    def test_detects_size_change(self):
        target = self.root / "a.txt"
        target.write_text("hello")
        before = capture_snapshot(self.root)
        target.write_text("hello world, longer now")
        after = capture_snapshot(self.root)

        diff = diff_snapshots(before, after)
        self.assertEqual(diff.size_changed, ["a.txt"])

    def test_detects_mtime_change_without_size_change(self):
        target = self.root / "a.txt"
        target.write_text("hello")
        before = capture_snapshot(self.root)

        # Same size, but touch the mtime forward.
        stat_before = target.stat()
        new_mtime = stat_before.st_mtime + 5
        os.utime(target, times=(new_mtime, new_mtime))
        after = capture_snapshot(self.root)

        diff = diff_snapshots(before, after)
        self.assertEqual(diff.mtime_changed, ["a.txt"])
        self.assertEqual(diff.size_changed, [])


class SaveLoadSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name) / "scanned"
        self.root.mkdir()
        self.output_dir = Path(self.tmpdir.name) / "snapshots"

    def test_round_trips_through_json(self):
        (self.root / "a.txt").write_text("hello")
        snapshot = capture_snapshot(self.root)

        out_path = self.output_dir / "before.json"
        save_snapshot(snapshot, out_path)
        self.assertTrue(out_path.exists())

        loaded = load_snapshot(out_path)
        self.assertEqual(loaded, snapshot)

    def test_save_does_not_write_inside_scanned_root(self):
        (self.root / "a.txt").write_text("hello")
        snapshot = capture_snapshot(self.root)
        save_snapshot(snapshot, self.output_dir / "before.json")

        self.assertEqual([p.name for p in self.root.iterdir()], ["a.txt"])


if __name__ == "__main__":
    unittest.main()
