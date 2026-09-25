"""Unit tests for the reusable scan application workflow."""

import io
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from local_file_organizer import database
from local_file_organizer.cli import _fingerprint_match, _scan_one_file
from local_file_organizer.hashing import find_duplicates
from local_file_organizer.metadata import get_basic_metadata
from local_file_organizer.scan_workflow import (
    ScanResult,
    TargetNotDirectoryError,
    TargetNotFoundError,
    generate_scan,
)
from local_file_organizer.scanner import DEFAULT_IGNORED_DIRS, iter_files


class ScanWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "tree"
        self.root.mkdir()
        self.db = Path(self.tmp.name) / "catalog.sqlite"
        self.csv = Path(self.tmp.name) / "inventory.csv"
        self.snapshots = Path(self.tmp.name) / "snapshots"

    def _run(self, target=None, **overrides):
        kwargs = {
            "target": target or self.root,
            "db_path": self.db,
            "csv_path": self.csv,
            "ignored_dirs": set(DEFAULT_IGNORED_DIRS),
            "verify_readonly": False,
            "snapshot_dir": self.snapshots,
            "similarity_threshold": 8,
            "full": False,
            "progress_interval": 500,
            "iter_files_fn": iter_files,
            "get_basic_metadata_fn": get_basic_metadata,
            "scan_one_file_fn": _scan_one_file,
            "fingerprint_match_fn": _fingerprint_match,
            "exiftool_available_fn": lambda: False,
            "find_duplicates_fn": find_duplicates,
            "now_iso_fn": lambda: "2026-09-25T21:00:00+00:00",
        }
        kwargs.update(overrides)
        return generate_scan(**kwargs)

    def test_returns_structured_result_without_console_output(self):
        (self.root / "a.txt").write_text("hello", encoding="utf-8")
        (self.root / "b.bin").write_bytes(b"world!!")

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            result = self._run()

        self.assertIsInstance(result, ScanResult)
        self.assertEqual(stdout_buf.getvalue(), "")
        self.assertEqual(stderr_buf.getvalue(), "")
        self.assertTrue(self.db.exists())
        self.assertTrue(self.csv.exists())
        self.assertIn("2 archivos", result.summary)
        self.assertEqual(result.walk_errors, ())
        self.assertIsNone(result.readonly_diff)

    def test_missing_and_non_directory_targets_raise_typed_errors(self):
        missing = Path(self.tmp.name) / "missing"
        with self.assertRaises(TargetNotFoundError):
            self._run(target=missing)

        not_dir = Path(self.tmp.name) / "file.txt"
        not_dir.write_text("x", encoding="utf-8")
        with self.assertRaises(TargetNotDirectoryError):
            self._run(target=not_dir)

    def test_progress_is_callback_not_print(self):
        for index in range(3):
            (self.root / f"{index}.txt").write_text(str(index), encoding="utf-8")
        seen = []

        result = self._run(progress_interval=1, on_progress=seen.append)

        self.assertEqual(seen, [1, 2, 3])
        self.assertIn("3 archivos", result.summary)

    def test_connection_closes_on_unexpected_error(self):
        real_connect = database.connect
        opened = []

        def spy_connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        def broken_iter(*args, **kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover

        with patch(
            "local_file_organizer.scan_workflow.database.connect",
            side_effect=spy_connect,
        ), self.assertRaises(RuntimeError):
            self._run(iter_files_fn=broken_iter)

        self.assertEqual(len(opened), 1)
        with self.assertRaises(sqlite3.ProgrammingError):
            opened[0].execute("SELECT 1")

    def test_readonly_verification_returns_clean_diff(self):
        (self.root / "a.txt").write_text("hello", encoding="utf-8")

        result = self._run(verify_readonly=True)

        self.assertIsNotNone(result.readonly_diff)
        self.assertTrue(result.readonly_diff.is_clean)
        self.assertTrue((self.snapshots / "before.json").exists())
        self.assertTrue((self.snapshots / "after.json").exists())


if __name__ == "__main__":
    unittest.main()
