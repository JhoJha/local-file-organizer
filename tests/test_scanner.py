import os
import tempfile
import unittest
from pathlib import Path

from local_file_organizer.scanner import DEFAULT_IGNORED_DIRS, ScanError, iter_files


class IterFilesTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)

    def test_missing_path_raises(self):
        missing = self.root / "does-not-exist"
        with self.assertRaises(FileNotFoundError):
            list(iter_files(missing))

    def test_finds_nested_files(self):
        (self.root / "sub").mkdir()
        (self.root / "sub" / "a.txt").write_text("a")
        (self.root / "b.txt").write_text("b")

        found = {p.name for p in iter_files(self.root)}
        self.assertEqual(found, {"a.txt", "b.txt"})

    def test_skips_default_protected_dirs(self):
        for ignored_name in (".git", "node_modules", "__pycache__"):
            protected = self.root / ignored_name
            protected.mkdir()
            (protected / "secret.txt").write_text("secret")
        (self.root / "visible.txt").write_text("visible")

        found = {p.name for p in iter_files(self.root, ignored_dirs=DEFAULT_IGNORED_DIRS)}
        self.assertEqual(found, {"visible.txt"})

    def test_custom_ignored_dirs(self):
        (self.root / "build").mkdir()
        (self.root / "build" / "output.txt").write_text("x")
        (self.root / "keep.txt").write_text("y")

        found = {p.name for p in iter_files(self.root, ignored_dirs={"build"})}
        self.assertEqual(found, {"keep.txt"})

    def test_unreadable_subdir_reports_error_and_continues(self):
        if os.geteuid() == 0:
            self.skipTest("permission bits are ignored when running as root")

        blocked = self.root / "blocked"
        blocked.mkdir()
        (blocked / "hidden.txt").write_text("secret")
        (self.root / "ok.txt").write_text("visible")

        original_mode = blocked.stat().st_mode
        blocked.chmod(0o000)
        self.addCleanup(lambda: blocked.chmod(original_mode))

        errors: list[ScanError] = []
        found = {p.name for p in iter_files(self.root, on_error=errors.append)}

        self.assertEqual(found, {"ok.txt"})
        self.assertTrue(errors, "expected a ScanError to be reported")


if __name__ == "__main__":
    unittest.main()
