"""Tests for V3.2 structured search: DSL validation, filters, person
resolution in SQL, injection safety, sorting/limits and CLI redaction.
Only touches temporary directories."""
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from local_file_organizer import database, person_identity
from local_file_organizer.cli import run_search
from local_file_organizer.query import (
    cli_filters,
    parse_bbox,
    parse_size,
    run_query,
)


def _seed(conn) -> None:
    files = [
        # path, ext, category, size, duration, capture, make, model, lat, lon
        ("/v1.mp4", ".mp4", "video", 842_000_000, 1122.0, "2025-08-17T10:00:00",
         "Xiaomi", "POCO X6", None, None),
        ("/v2.mp4", ".mp4", "video", 100_000_000, 300.0, "2024-05-01T10:00:00",
         "Samsung", "Galaxy S9+", 40.4, -3.6),
        ("/a.jpg", ".jpg", "image", 2_000_000, None, "2025-01-10T10:00:00",
         "Xiaomi", "POCO X6", 40.5, -3.7),
        ("/b.jpg", ".jpg", "image", 3_000_000, None, None, None, None, None, None),
        ("/d.pdf", ".pdf", "document", 500_000, None, None, None, None, None, None),
    ]
    for path, ext, cat, size, dur, cap, make, model, lat, lon in files:
        database.upsert_file(
            conn,
            database.FileRecord(
                path=path, extension=ext, category=cat, size_bytes=size,
                duration_seconds=dur, capture_date=cap,
                device_make=make, device_model=model,
                gps_latitude=lat, gps_longitude=lon,
            ),
        )
    database.insert_person_cluster_members(
        conn,
        [
            ("P0001", "/v1.mp4", "video", 2),
            ("P0001", "/a.jpg", "image", 1),
            ("P0002", "/v2.mp4", "video", 1),
            ("P0002", "/a.jpg", "image", 1),
        ],
    )
    conn.commit()


def _ids(rows) -> set[str]:
    return {r["path"] for r in rows}


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.conn = database.connect(Path(self.tmp.name) / "t.sqlite")
        _seed(self.conn)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_unknown_field_rejected(self):
        with self.assertRaises(ValueError):
            run_query(self.conn, {"version": 1, "and": [
                {"field": "hacker", "op": "=", "value": 1}]})

    def test_unknown_op_rejected(self):
        with self.assertRaises(ValueError):
            run_query(self.conn, {"version": 1, "and": [
                {"field": "category", "op": "regexp", "value": "x"}]})

    def test_bad_version_and_empty_rejected(self):
        with self.assertRaises(ValueError):
            run_query(self.conn, {"version": 99, "and": []})
        with self.assertRaises(ValueError):
            run_query(self.conn, {"version": 1, "and": []})
        with self.assertRaises(ValueError):
            run_query(self.conn, {"version": 1, "and": [
                {"field": "size_bytes", "op": "=", "value": "huge"}]})

    def test_bad_sort_and_limit_rejected(self):
        good = {"version": 1, "and": [{"field": "category", "op": "=", "value": "video"}]}
        with self.assertRaises(ValueError):
            run_query(self.conn, good, order_by="path; DROP TABLE files")
        with self.assertRaises(ValueError):
            run_query(self.conn, good, limit=0)

    def test_injection_is_just_no_match(self):
        rows, total = run_query(self.conn, {"version": 1, "and": [
            {"field": "category", "op": "=", "value": "video' OR '1'='1"}]})
        self.assertEqual(total, 0)
        self.assertEqual(rows, [])
        # catalog intact
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0], 5
        )

    def test_like_injection_is_escaped_by_binding(self):
        rows, total = run_query(self.conn, {"version": 1, "and": [
            {"field": "category", "op": "like", "value": "%"}]})
        self.assertEqual(total, 5)  # % matches all: binding, not escaping bug


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.conn = database.connect(Path(self.tmp.name) / "t.sqlite")
        _seed(self.conn)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def q(self, **kwargs):
        return run_query(self.conn, cli_filters(**kwargs))

    def test_category_and_extension(self):
        rows, total = self.q(category="video")
        self.assertEqual(_ids(rows), {"/v1.mp4", "/v2.mp4"})
        self.assertEqual(total, 2)
        rows, _ = self.q(extension=".mp4,.pdf")
        self.assertEqual(_ids(rows), {"/v1.mp4", "/v2.mp4", "/d.pdf"})

    def test_device_matches_make_or_model(self):
        rows, _ = self.q(device="POCO X6")
        self.assertEqual(_ids(rows), {"/v1.mp4", "/a.jpg"})
        rows, _ = self.q(device="xiaomi")  # NOCASE
        self.assertEqual(_ids(rows), {"/v1.mp4", "/a.jpg"})

    def test_year_and_date_presence(self):
        rows, _ = self.q(year="2025")
        self.assertEqual(_ids(rows), {"/v1.mp4", "/a.jpg"})
        rows, _ = self.q(has_date=True)
        self.assertEqual(_ids(rows), {"/v1.mp4", "/v2.mp4", "/a.jpg"})
        with self.assertRaises(ValueError):
            self.q(year="25")

    def test_duration_and_size_ranges(self):
        rows, _ = self.q(min_duration=600)
        self.assertEqual(_ids(rows), {"/v1.mp4"})
        rows, _ = self.q(min_size="500MB")
        self.assertEqual(_ids(rows), {"/v1.mp4"})
        rows, _ = self.q(max_size="1MB")
        self.assertEqual(_ids(rows), {"/d.pdf"})

    def test_gps_and_bbox(self):
        rows, _ = self.q(has_gps=True)
        self.assertEqual(_ids(rows), {"/v2.mp4", "/a.jpg"})
        rows, _ = self.q(bbox="40.0,41.0,-4.0,-3.0")
        self.assertEqual(_ids(rows), {"/v2.mp4", "/a.jpg"})
        rows, _ = self.q(bbox="0.0,1.0,0.0,1.0")
        self.assertEqual(_ids(rows), set())

    def test_person_auto(self):
        rows, _ = self.q(persons=["P0001"])
        self.assertEqual(_ids(rows), {"/v1.mp4", "/a.jpg"})

    def test_person_cooccurrence_is_and(self):
        rows, _ = self.q(persons=["P0001", "P0002"])
        self.assertEqual(_ids(rows), {"/a.jpg"})

    def test_person_follows_alias_and_manual_minus_excluded(self):
        person_identity.merge_labels(self.conn, ["P0001", "P0002"], "María")
        self.conn.commit()
        rows, _ = self.q(persons=["PERSON_0001"])
        self.assertEqual(_ids(rows), {"/v1.mp4", "/v2.mp4", "/a.jpg"})
        rows, _ = self.q(persons=["María"])  # display name
        self.assertEqual(_ids(rows), {"/v1.mp4", "/v2.mp4", "/a.jpg"})
        person_identity.assign_file(self.conn, "/d.pdf", "PERSON_0001")
        person_identity.unassign_file(self.conn, "/v2.mp4", "PERSON_0001")
        self.conn.commit()
        rows, _ = self.q(persons=["PERSON_0001"])
        self.assertEqual(_ids(rows), {"/v1.mp4", "/a.jpg", "/d.pdf"})

    def test_person_unknown_and_raw_vs_canonical(self):
        with self.assertRaises(ValueError):
            self.q(persons=["Nadie"])
        person_identity.merge_labels(self.conn, ["P0001", "P0002"])
        self.conn.commit()
        # P0001 = raw cluster (calibration unit); PERSON_0001 = identity.
        rows, _ = self.q(persons=["P0001"])
        self.assertEqual(_ids(rows), {"/v1.mp4", "/a.jpg"})
        rows, _ = self.q(persons=["PERSON_0001"])
        self.assertEqual(_ids(rows), {"/v1.mp4", "/v2.mp4", "/a.jpg"})

    def test_combined_story_query(self):
        # "videos POCO X6 de 2025 de más de 10 min": the §15 use case.
        rows, total = self.q(
            category="video", device="POCO X6", year="2025", min_duration=600
        )
        self.assertEqual(_ids(rows), {"/v1.mp4"})
        self.assertEqual(total, 1)

    def test_sort_limit_and_total(self):
        rows, total = run_query(
            self.conn,
            cli_filters(category="video,image"),
            order_by="size_bytes",
            descending=True,
            limit=2,
        )
        self.assertEqual(total, 4)
        self.assertEqual([r["path"] for r in rows], ["/v1.mp4", "/v2.mp4"])
        self.assertEqual(rows[0]["file_id"], "F000004")  # path-ordered catalog id

    def test_file_ids_match_preview_numbering(self):
        rows, _ = self.q(category="document")
        self.assertEqual(rows[0]["file_id"], "F000003")  # /d.pdf sorts third


class ParseHelperTests(unittest.TestCase):
    def test_parse_size(self):
        self.assertEqual(parse_size("500MB"), 500 * 1024**2)
        self.assertEqual(parse_size("1.5G"), int(1.5 * 1024**3))
        self.assertEqual(parse_size("1024"), 1024)
        self.assertEqual(parse_size(2048), 2048)
        with self.assertRaises(ValueError):
            parse_size("mucho")
        with self.assertRaises(ValueError):
            parse_size("-5MB")

    def test_parse_bbox(self):
        self.assertEqual(parse_bbox("1,2,3,4"), (1.0, 2.0, 3.0, 4.0))
        with self.assertRaises(ValueError):
            parse_bbox("1,2,3")
        with self.assertRaises(ValueError):
            parse_bbox("5,1,0,10")
        with self.assertRaises(ValueError):
            parse_bbox("0,100,0,10")


class IndexTests(unittest.TestCase):
    def test_search_indexes_exist_after_connect(self):
        with TemporaryDirectory() as tmp:
            conn = database.connect(Path(tmp) / "n.sqlite")
            names = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            }
            conn.close()
        for expected in (
            "idx_files_category", "idx_files_capture_date", "idx_files_device_model",
            "idx_files_size_bytes", "idx_files_duration", "idx_files_extension",
        ):
            self.assertIn(expected, names)


class CliRedactionTests(unittest.TestCase):
    def test_default_output_has_no_paths(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "c.sqlite"
            conn = database.connect(db)
            _seed(conn)
            conn.close()
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                run_search(db_path=db, category="video")
            out = buffer.getvalue()
        self.assertIn("F000004", out)
        self.assertIn("matches=2", out)
        self.assertNotIn("/v1.mp4", out)
        self.assertNotIn(tmp, out)

    def test_show_files_prints_paths(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "c.sqlite"
            conn = database.connect(db)
            _seed(conn)
            conn.close()
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                run_search(db_path=db, show_files=True, category="video")
            out = buffer.getvalue()
        self.assertIn("/v1.mp4", out)

    def test_no_filters_is_error(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "c.sqlite"
            conn = database.connect(db)
            _seed(conn)
            conn.close()
            with self.assertRaises(SystemExit):
                run_search(db_path=db)


if __name__ == "__main__":
    unittest.main()
