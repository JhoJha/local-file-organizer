"""Tests for T2 user tags: attach/detach/list, search --tag filtering,
migration on old catalogs, stale-file exclusion and plan privacy.
Only touches temporary directories."""
import io
import json
import sqlite3
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from local_file_organizer import database, user_tags
from local_file_organizer.cli import run_search, run_tag, run_tags, run_untag
from local_file_organizer.query import cli_filters, run_query


def _seed(conn) -> None:
    for path, category in (
        ("/a.jpg", "image"),
        ("/b.jpg", "image"),
        ("/v.mp4", "video"),
    ):
        database.upsert_file(
            conn, database.FileRecord(path=path, category=category)
        )
    conn.commit()


def _ids(rows) -> set[str]:
    return {r["path"] for r in rows}


class AttachDetachTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.conn = database.connect(Path(self.tmp.name) / "t.sqlite")
        _seed(self.conn)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_attach_by_path_and_list(self):
        path, tag = user_tags.attach_tag(self.conn, "/a.jpg", "beach")
        self.assertEqual((path, tag), ("/a.jpg", "beach"))
        self.conn.commit()
        summary = user_tags.tag_summary(self.conn)
        self.assertEqual(summary, [{"tag_name": "beach", "files": 1}])
        self.assertEqual(database.fetch_tags_for_file(self.conn, "/a.jpg"), ["beach"])

    def test_attach_by_file_id(self):
        # Paths sort as /a.jpg, /b.jpg, /v.mp4 -> /b.jpg is F000002.
        path, _ = user_tags.attach_tag(self.conn, "F000002", "beach")
        self.assertEqual(path, "/b.jpg")

    def test_attach_is_idempotent(self):
        user_tags.attach_tag(self.conn, "/a.jpg", "beach")
        user_tags.attach_tag(self.conn, "/a.jpg", "beach")
        self.conn.commit()
        links = database.fetch_file_tag_links(self.conn)
        self.assertEqual(len(links), 1)

    def test_detach_keeps_tag_row_with_zero_files(self):
        user_tags.attach_tag(self.conn, "/a.jpg", "beach")
        self.conn.commit()
        user_tags.detach_tag(self.conn, "/a.jpg", "beach")
        self.conn.commit()
        self.assertIsNotNone(database.fetch_tag(self.conn, "beach"))
        self.assertEqual(user_tags.tag_summary(self.conn), [{"tag_name": "beach", "files": 0}])

    def test_detach_missing_link_fails(self):
        user_tags.attach_tag(self.conn, "/a.jpg", "beach")
        self.conn.commit()
        with self.assertRaises(ValueError):
            user_tags.detach_tag(self.conn, "/a.jpg", "nope")
        with self.assertRaises(ValueError):
            user_tags.detach_tag(self.conn, "/b.jpg", "beach")

    def test_invalid_names_rejected(self):
        with self.assertRaises(ValueError):
            user_tags.attach_tag(self.conn, "/a.jpg", "   ")
        with self.assertRaises(ValueError):
            user_tags.attach_tag(self.conn, "/a.jpg", "x" * 65)
        with self.assertRaises(ValueError):
            user_tags.attach_tag(self.conn, "/a.jpg", "a,b")

    def test_unknown_file_refs_rejected(self):
        with self.assertRaises(ValueError):
            user_tags.attach_tag(self.conn, "/missing.jpg", "beach")
        with self.assertRaises(ValueError):
            user_tags.attach_tag(self.conn, "F000099", "beach")
        with self.assertRaises(ValueError):
            user_tags.attach_tag(self.conn, "", "beach")


class SearchTagTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.conn = database.connect(Path(self.tmp.name) / "t.sqlite")
        _seed(self.conn)
        user_tags.attach_tag(self.conn, "/a.jpg", "beach")
        user_tags.attach_tag(self.conn, "/b.jpg", "beach")
        user_tags.attach_tag(self.conn, "/b.jpg", "family")
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_single_tag_filters(self):
        rows, total = run_query(self.conn, cli_filters(tags=["beach"]))
        self.assertEqual(_ids(rows), {"/a.jpg", "/b.jpg"})
        self.assertEqual(total, 2)

    def test_repeated_tag_is_and(self):
        rows, _ = run_query(self.conn, cli_filters(tags=["beach", "family"]))
        self.assertEqual(_ids(rows), {"/b.jpg"})

    def test_tag_combines_with_other_filters(self):
        rows, _ = run_query(
            self.conn, cli_filters(category="image", tags=["family"])
        )
        self.assertEqual(_ids(rows), {"/b.jpg"})
        rows, _ = run_query(
            self.conn, cli_filters(category="video", tags=["beach"])
        )
        self.assertEqual(_ids(rows), set())

    def test_unknown_tag_matches_zero_rows(self):
        rows, total = run_query(self.conn, cli_filters(tags=["never-created"]))
        self.assertEqual((rows, total), ([], 0))

    def test_tag_injection_is_just_no_match(self):
        rows, total = run_query(
            self.conn, cli_filters(tags=["x' OR '1'='1"])
        )
        self.assertEqual((rows, total), ([], 0))
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0], 3
        )

    def test_stale_files_excluded_from_search(self):
        self.conn.execute(
            "UPDATE files SET file_state = 'stale' WHERE path = '/a.jpg'"
        )
        self.conn.commit()
        rows, total = run_query(self.conn, cli_filters(tags=["beach"]))
        self.assertEqual(_ids(rows), {"/b.jpg"})
        self.assertEqual(total, 1)

    def test_stale_files_excluded_from_summary(self):
        self.conn.execute(
            "UPDATE files SET file_state = 'stale' WHERE path = '/a.jpg'"
        )
        self.conn.commit()
        summary = {
            e["tag_name"]: e["files"] for e in user_tags.tag_summary(self.conn)
        }
        self.assertEqual(summary, {"beach": 1, "family": 1})


class MigrationTests(unittest.TestCase):
    def test_old_db_without_tags_tables_migrates(self):
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "old.sqlite"
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "CREATE TABLE files ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "path TEXT NOT NULL UNIQUE, name TEXT)"
            )
            conn.execute(
                "INSERT INTO files (path, name) VALUES (?, ?)",
                ("/old/photo.png", "photo.png"),
            )
            conn.commit()
            conn.close()

            conn = database.connect(db_path)
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            self.assertIn("tags", tables)
            self.assertIn("file_tag_links", tables)
            # Old row preserved and taggable.
            path, _ = user_tags.attach_tag(conn, "/old/photo.png", "archive")
            conn.commit()
            self.assertEqual(path, "/old/photo.png")
            self.assertEqual(
                database.fetch_tags_for_file(conn, "/old/photo.png"), ["archive"]
            )
            conn.close()


class PrivacyTests(unittest.TestCase):
    def test_tags_never_leak_into_plan(self):
        from local_file_organizer.organization_plan import build_organization_plan

        with TemporaryDirectory() as tmp:
            conn = database.connect(Path(tmp) / "t.sqlite")
            _seed(conn)
            user_tags.attach_tag(conn, "/a.jpg", "secret-tag")
            conn.commit()
            plan = build_organization_plan(conn)
            conn.close()
        payload = json.dumps(plan)
        self.assertNotIn("secret-tag", payload)
        self.assertTrue(all("tag" not in entry for entry in plan))


class CliTests(unittest.TestCase):
    def test_cli_tag_untag_tags_roundtrip(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "c.sqlite"
            conn = database.connect(db)
            _seed(conn)
            conn.close()
            run_tag("/a.jpg", "beach", db_path=db)
            run_tag("F000002", "beach", db_path=db)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                run_tags(db_path=db)
            self.assertIn("beach files=2", buffer.getvalue())
            run_untag("/a.jpg", "beach", db_path=db)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                run_tags(db_path=db)
            self.assertIn("beach files=1", buffer.getvalue())
            conn = database.connect(db)
            tags = user_tags.resolve_file_tags(conn)
            conn.close()
        self.assertEqual(tags["/b.jpg"], {"beach"})
        self.assertNotIn("/a.jpg", tags)

    def test_cli_search_tag_flag(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "c.sqlite"
            conn = database.connect(db)
            _seed(conn)
            conn.close()
            run_tag("/a.jpg", "beach", db_path=db)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                run_search(db_path=db, tags=["beach"])
            out = buffer.getvalue()
        self.assertIn("matches=1", out)
        self.assertIn("tags=beach", out)
        self.assertNotIn("/a.jpg", out)

    def test_cli_errors_exit_1(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "c.sqlite"
            conn = database.connect(db)
            _seed(conn)
            conn.close()
            with self.assertRaises(SystemExit):
                run_tag("/missing.jpg", "beach", db_path=db)
            with self.assertRaises(SystemExit):
                run_untag("/a.jpg", "beach", db_path=db)
            with self.assertRaises(SystemExit):
                run_tag("/a.jpg", "   ", db_path=db)


if __name__ == "__main__":
    unittest.main()
