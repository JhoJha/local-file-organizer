"""Tests for F1 incremental scan: fingerprint reuse (zero heavy calls on
clean rescan), new/changed/deleted handling, --full equivalence,
version-driven reprocessing and stale exclusion from views.
Only touches temporary directories; ExifTool is mocked out for
determinism (no perl spawns)."""
import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from local_file_organizer import database
from local_file_organizer.cli import (
    _fingerprint_match,
    _required_versions,
    run_scan,
    run_summary,
)
from local_file_organizer.organization_plan import build_organization_plan

NO_EXIF = mock.patch(
    "local_file_organizer.cli.exiftool_available", return_value=False
)


def _write(root: Path, name: str, payload: bytes) -> Path:
    target = root / name
    target.write_bytes(payload)
    return target


def _scan(root: Path, db: Path, **kwargs) -> str:
    kwargs.setdefault("quiet", True)
    buffer = io.StringIO()
    with NO_EXIF, redirect_stdout(buffer):
        run_scan(root, db_path=db, csv_path=db.parent / "i.csv", **kwargs)
    return buffer.getvalue()


def _content_dump(conn) -> list[tuple]:
    rows = conn.execute(
        "SELECT path, category, size_bytes, sha256, perceptual_hash,"
        " capture_date, device_model, duplicate_group, similar_group,"
        " scan_status, metadata_status, file_state FROM files ORDER BY path"
    ).fetchall()
    return [tuple(r) for r in rows]


class FingerprintTests(unittest.TestCase):
    def test_match_requires_size_mtime_and_versions(self):
        basic = mock.Mock(size_bytes=10, modified_at="2024-01-01T00:00:00", category="image")
        good = {"size_bytes": 10, "modified_at": "2024-01-01T00:00:00",
                "extractor_versions": "dhash_v1"}
        self.assertTrue(_fingerprint_match(good, basic, False))
        self.assertFalse(_fingerprint_match(None, basic, False))
        self.assertFalse(_fingerprint_match({**good, "size_bytes": 11}, basic, False))
        self.assertFalse(_fingerprint_match({**good, "modified_at": "X"}, basic, False))
        # newly appeared binary => required superset => reprocess
        self.assertFalse(_fingerprint_match(good, basic, True))
        # vanished binary => required subset => keep data, skip
        richer = {**good, "extractor_versions": "dhash_v1|exiftool_v1"}
        self.assertTrue(_fingerprint_match(richer, basic, False))
        # legacy NULL versions => reprocess once
        self.assertFalse(_fingerprint_match(
            {"size_bytes": 10, "modified_at": "2024-01-01T00:00:00",
             "extractor_versions": None}, basic, False))

    def test_other_category_needs_no_extractor(self):
        self.assertEqual(_required_versions("other", True), set())
        self.assertEqual(_required_versions("archive", False), set())
        self.assertEqual(_required_versions("image", False), {"dhash_v1"})
        self.assertEqual(
            _required_versions("video", True), {"ffprobe_v1", "exiftool_v1"}
        )


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name) / "tree"
        self.root.mkdir()
        # unique sizes => duplicate detection never reads bytes here
        _write(self.root, "a.txt", b"a" * 100)
        _write(self.root, "b.txt", b"b" * 200)
        _write(self.root, "c.bin", b"c" * 300)
        self.db = Path(self.tmp.name) / "c.sqlite"

    def tearDown(self):
        self.tmp.cleanup()

    def _heavy_mocks(self):
        return (
            mock.patch(
                "local_file_organizer.cli.get_image_metadata",
                side_effect=AssertionError("must not decode"),
            ),
            mock.patch(
                "local_file_organizer.cli.get_video_metadata",
                side_effect=AssertionError("must not probe"),
            ),
            mock.patch(
                "local_file_organizer.cli.get_exif_metadata",
                side_effect=AssertionError("must not spawn"),
            ),
            mock.patch("PIL.Image.open", side_effect=AssertionError("must not open")),
            mock.patch("subprocess.run", side_effect=AssertionError("must not spawn")),
        )

    def test_clean_rescan_touches_only(self):
        _scan(self.root, self.db)
        mocks = self._heavy_mocks()
        for m in mocks:
            m.start()
        try:
            out = _scan(self.root, self.db)
        finally:
            for m in mocks:
                m.stop()
        self.assertIn("3 reutilizados sin re-analizar", out)
        self.assertIn("0 nuevos", out)
        self.assertIn("0 reprocesados", out)

    def test_incremental_equals_full(self):
        _scan(self.root, self.db)
        _write(self.root, "d.txt", b"d" * 400)
        (self.root / "a.txt").write_bytes(b"A" * 150)
        _scan(self.root, self.db)
        conn = database.connect(self.db)
        incremental = _content_dump(conn)
        conn.close()
        out = _scan(self.root, self.db, full=True)
        conn = database.connect(self.db)
        full = _content_dump(conn)
        # last_seen_at differs (new timestamp); everything else identical
        conn.close()
        self.assertEqual(incremental, full)
        self.assertIn("0 reutilizados sin re-analizar", out)

    def test_new_changed_deleted_counters(self):
        _scan(self.root, self.db)
        _write(self.root, "new.txt", b"n" * 500)
        (self.root / "a.txt").write_bytes(b"A" * 101)
        os.remove(self.root / "b.txt")
        out = _scan(self.root, self.db)
        self.assertIn("1 reutilizados sin re-analizar", out)
        self.assertIn("1 nuevos", out)
        self.assertIn("1 reprocesados", out)
        self.assertIn("1 marcados stale", out)
        conn = database.connect(self.db)
        state = {
            r["path"]: r["file_state"]
            for r in conn.execute("SELECT path, file_state FROM files")
        }
        conn.close()
        self.assertEqual(state[str(self.root / "b.txt")], "stale")
        self.assertEqual(state[str(self.root / "a.txt")], "active")

    def test_mtime_only_touch_reprocesses(self):
        _scan(self.root, self.db)
        target = self.root / "a.txt"
        stat = target.stat()
        os.utime(target, (stat.st_atime, stat.st_mtime + 5))
        out = _scan(self.root, self.db)
        self.assertIn("1 reprocesados", out)

    def test_stale_excluded_from_views(self):
        _scan(self.root, self.db)
        os.remove(self.root / "b.txt")
        _scan(self.root, self.db)
        conn = database.connect(self.db)
        try:
            plan = build_organization_plan(conn)
            self.assertEqual(len(plan), 2)
        finally:
            conn.close()
        # search excludes stale: query everything via a broad filter
        from local_file_organizer.query import run_query

        conn = database.connect(self.db)
        try:
            rows, total = run_query(
                conn,
                {"version": 1, "and": [
                    {"field": "size_bytes", "op": ">", "value": 0}]},
            )
        finally:
            conn.close()
        self.assertEqual(total, 2)

    def test_summary_counts_active_only(self):
        _scan(self.root, self.db)
        os.remove(self.root / "b.txt")
        buffer = io.StringIO()
        with NO_EXIF, redirect_stdout(buffer):
            run_summary(db_path=self.db)
        self.assertIn("2 archivos", buffer.getvalue())

    def test_versions_recorded_and_legacy_reprocesses(self):
        # extractor-less categories skip on size+mtime alone; an image
        # exercises the version gate (dhash_v1 recorded on success).
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            root.mkdir()
            from PIL import Image

            Image.new("RGB", (32, 32), (1, 2, 3)).save(str(root / "img.png"))
            db = Path(tmp) / "c.sqlite"
            _scan(root, db)
            conn = database.connect(db)
            row = conn.execute(
                "SELECT extractor_versions FROM files"
            ).fetchone()
            self.assertEqual(row["extractor_versions"], "dhash_v1")
            conn.execute("UPDATE files SET extractor_versions = NULL")
            conn.commit()
            conn.close()
            out = _scan(root, db)
        self.assertIn("1 reprocesados", out)
        self.assertIn("0 reutilizados sin re-analizar", out)


class DuplicateReuseTests(unittest.TestCase):
    def test_duplicate_group_survives_without_rehash(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            root.mkdir()
            _write(root, "x1.bin", b"Z" * 1000)
            _write(root, "x2.bin", b"Z" * 1000)
            db = Path(tmp) / "c.sqlite"
            _scan(root, db)
            conn = database.connect(db)
            groups = {
                r["path"]: r["duplicate_group"]
                for r in conn.execute("SELECT path, duplicate_group FROM files")
            }
            conn.close()
            self.assertIsNotNone(groups[str(root / "x1.bin")])
            self.assertEqual(groups[str(root / "x1.bin")], groups[str(root / "x2.bin")])

            hashed = []
            import local_file_organizer.hashing as hashing_mod

            real_find = hashing_mod.find_duplicates

            def spy(records, hash_fn=None, on_hash_error=None, known_digests=None):
                def counting(path):
                    hashed.append(str(path))
                    return hashing_mod.sha256_file(path)

                return real_find(
                    records,
                    hash_fn=counting,
                    on_hash_error=on_hash_error,
                    known_digests=known_digests,
                )

            with mock.patch("local_file_organizer.cli.find_duplicates", side_effect=spy):
                _scan(root, db)
            # both digests came from the catalog: zero byte reads
            self.assertEqual(hashed, [])
            conn = database.connect(db)
            groups2 = {
                r["path"]: r["duplicate_group"]
                for r in conn.execute("SELECT path, duplicate_group FROM files")
            }
            conn.close()
            self.assertEqual(groups, groups2)


class AdversarialTests(unittest.TestCase):
    """Regression tests for breaks found by trying to break the scan."""

    def test_fifo_and_socket_never_hang_the_scan(self):
        import socket
        import threading

        from local_file_organizer.scanner import is_regular_file

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            root.mkdir()
            os.mkfifo(str(root / "tube"))
            (root / "empty.txt").write_bytes(b"")
            sock = socket.socket(socket.AF_UNIX)
            try:
                sock.bind(str(root / "sock"))
                self.assertFalse(is_regular_file(root / "tube"))
                self.assertFalse(is_regular_file(root / "sock"))
                self.assertFalse(is_regular_file(root / "missing"))
                self.assertTrue(is_regular_file(root / "empty.txt"))

                db = Path(tmp) / "c.sqlite"
                errors: list = []

                def target():
                    try:
                        _scan(root, db)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(exc)

                thread = threading.Thread(target=target, daemon=True)
                thread.start()
                thread.join(timeout=30)
                self.assertFalse(thread.is_alive(), "scan hung on FIFO/socket")
                self.assertEqual(errors, [])
            finally:
                sock.close()

            conn = database.connect(db)
            try:
                notes = {
                    r["path"]: (r["scan_status"], r["scan_error"])
                    for r in conn.execute("SELECT path, scan_status, scan_error FROM files")
                }
            finally:
                conn.close()
            self.assertIsNotNone(notes[str(root / "tube")])
            self.assertIn("not a regular file", notes[str(root / "tube")][1])

    def test_broken_symlink_counts_as_new_not_reprocessed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            root.mkdir()
            os.symlink("/nonexistent-target-xyz", str(root / "broken"))
            db = Path(tmp) / "c.sqlite"
            out = _scan(root, db)
        self.assertIn("1 nuevos", out)
        self.assertIn("0 reprocesados", out)

    def test_corrupt_catalog_raises_clear_error(self):
        import sqlite3

        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "c.sqlite"
            db.write_bytes(b"GARBAGE-NOT-SQLITE")
            with self.assertRaises(sqlite3.DatabaseError) as ctx:
                database.connect(db)
        self.assertIn("ilegible o corrupto", str(ctx.exception))

    def test_transient_hash_error_note_retires_on_success(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            root.mkdir()
            _write(root, "h1.bin", b"Q" * 1000)
            _write(root, "h2.bin", b"Q" * 1000)
            db = Path(tmp) / "c.sqlite"
            _scan(root, db)
            conn = database.connect(db)
            # simulate a past transient failure note on an otherwise
            # healthy, digested row
            conn.execute(
                "UPDATE files SET scan_error = 'hash: boom' WHERE path = ?",
                (str(root / "h1.bin"),),
            )
            conn.commit()
            conn.close()
            _scan(root, db)
            conn = database.connect(db)
            try:
                note = conn.execute(
                    "SELECT scan_error FROM files WHERE path = ?",
                    (str(root / "h1.bin"),),
                ).fetchone()[0]
            finally:
                conn.close()
        self.assertIsNone(note)


if __name__ == "__main__":
    unittest.main()
