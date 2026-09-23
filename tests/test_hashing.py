import tempfile
import unittest
from pathlib import Path

from local_file_organizer.hashing import DuplicateRecord, find_duplicates, group_by_size


class GroupBySizeTests(unittest.TestCase):
    def test_unique_sizes_are_dropped(self):
        records = [
            DuplicateRecord(path="a", size_bytes=10),
            DuplicateRecord(path="b", size_bytes=20),
        ]
        self.assertEqual(group_by_size(records), {})

    def test_repeated_sizes_are_grouped(self):
        records = [
            DuplicateRecord(path="a", size_bytes=10),
            DuplicateRecord(path="b", size_bytes=10),
            DuplicateRecord(path="c", size_bytes=20),
        ]
        groups = group_by_size(records)
        self.assertEqual(set(groups.keys()), {10})
        self.assertEqual(len(groups[10]), 2)


class FindDuplicatesTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)

    def _write(self, name: str, content: bytes) -> Path:
        path = self.root / name
        path.write_bytes(content)
        return path

    def test_identical_content_same_size_is_duplicate(self):
        a = self._write("a.bin", b"hello world")
        b = self._write("b.bin", b"hello world")
        c = self._write("c.bin", b"different!!")  # same size, different content
        unique = self._write("d.bin", b"unique-size-content-longer")

        records = [
            DuplicateRecord(path=str(a), size_bytes=a.stat().st_size),
            DuplicateRecord(path=str(b), size_bytes=b.stat().st_size),
            DuplicateRecord(path=str(c), size_bytes=c.stat().st_size),
            DuplicateRecord(path=str(unique), size_bytes=unique.stat().st_size),
        ]
        stats = find_duplicates(records)

        self.assertEqual(stats.total_duplicate_groups, 1)
        self.assertEqual(stats.total_duplicate_files, 2)
        self.assertEqual(set(stats.group_by_path.keys()), {str(a), str(b)})
        self.assertEqual(stats.group_by_path[str(a)], stats.group_by_path[str(b)])
        self.assertTrue(stats.group_by_path[str(a)].startswith("DUP_"))
        self.assertNotIn(str(c), stats.group_by_path)
        self.assertNotIn(str(unique), stats.group_by_path)

    def test_recoverable_space_math(self):
        a = self._write("a.bin", b"x" * 100)
        b = self._write("b.bin", b"x" * 100)
        c = self._write("c.bin", b"x" * 100)
        records = [
            DuplicateRecord(path=str(a), size_bytes=100),
            DuplicateRecord(path=str(b), size_bytes=100),
            DuplicateRecord(path=str(c), size_bytes=100),
        ]
        stats = find_duplicates(records)
        self.assertEqual(stats.duplicated_space_bytes, 300)
        self.assertEqual(stats.recoverable_space_bytes, 200)

    def test_no_duplicates_when_all_unique(self):
        a = self._write("a.bin", b"aaa")
        b = self._write("b.bin", b"bbbbb")
        records = [
            DuplicateRecord(path=str(a), size_bytes=a.stat().st_size),
            DuplicateRecord(path=str(b), size_bytes=b.stat().st_size),
        ]
        stats = find_duplicates(records)
        self.assertEqual(stats.total_duplicate_groups, 0)
        self.assertEqual(stats.group_by_path, {})


if __name__ == "__main__":
    unittest.main()
