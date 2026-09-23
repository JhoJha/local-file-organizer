import sqlite3
import tempfile
import unittest
from pathlib import Path

from local_file_organizer.database import FileRecord, connect, init_db, upsert_file
from local_file_organizer.face_detection import FaceObservation


class UpsertTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = Path(self.tmpdir.name) / "catalog.sqlite"

    def test_rescanning_same_path_updates_instead_of_duplicating(self):
        conn = connect(self.db_path)
        record = FileRecord(
            path="/some/file.txt",
            name="file.txt",
            extension=".txt",
            category="document",
            size_bytes=100,
            modified_at="2024-01-01T00:00:00+00:00",
            mime_type="text/plain",
            scan_status="ok",
        )
        upsert_file(conn, record)
        conn.commit()

        record.size_bytes = 200
        record.modified_at = "2024-02-01T00:00:00+00:00"
        upsert_file(conn, record)
        conn.commit()

        rows = conn.execute("SELECT * FROM files WHERE path = ?", (record.path,)).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["size_bytes"], 200)

        total = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        self.assertEqual(total, 1)
        conn.close()

    def test_path_unique_constraint(self):
        conn = connect(self.db_path)
        record = FileRecord(path="/x.txt")
        upsert_file(conn, record)
        conn.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO files (path) VALUES (?)", (record.path,)
            )
        conn.close()

    def test_metadata_columns_exist_on_fresh_db(self):
        conn = connect(self.db_path)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(files)")}
        self.assertIn("metadata_status", columns)
        self.assertIn("metadata_error", columns)
        conn.close()

    def test_similarity_columns_exist_on_fresh_db(self):
        conn = connect(self.db_path)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(files)")}
        self.assertIn("perceptual_hash", columns)
        self.assertIn("similar_group", columns)
        self.assertIn("similarity_method", columns)
        conn.close()

    def test_migration_adds_similarity_columns_to_pre_v2_1_db(self):
        # Simulate a V1/V2.0 catalog: has metadata_status/metadata_error
        # but predates perceptual_hash/similar_group/similarity_method.
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """
            CREATE TABLE files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                name TEXT,
                category TEXT,
                scan_status TEXT,
                scan_error TEXT,
                metadata_status TEXT,
                metadata_error TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO files (path, name, category, scan_status, metadata_status) "
            "VALUES (?, ?, ?, ?, ?)",
            ("/old/photo.png", "photo.png", "image", "ok", "skipped"),
        )
        conn.commit()
        conn.close()

        conn = connect(self.db_path)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(files)")}
        self.assertIn("perceptual_hash", columns)
        self.assertIn("similar_group", columns)
        self.assertIn("similarity_method", columns)

        row = conn.execute("SELECT * FROM files WHERE path = '/old/photo.png'").fetchone()
        self.assertEqual(row["name"], "photo.png")
        self.assertEqual(row["scan_status"], "ok")
        self.assertIsNone(row["perceptual_hash"])
        self.assertIsNone(row["similar_group"])

        total = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        self.assertEqual(total, 1)
        conn.close()

    def test_migration_adds_metadata_columns_to_pre_existing_db(self):
        # Simulate a catalog created before metadata_status/metadata_error
        # existed: a "files" table without them, holding one row.
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """
            CREATE TABLE files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                name TEXT,
                scan_status TEXT,
                scan_error TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO files (path, name, scan_status) VALUES (?, ?, ?)",
            ("/old/file.txt", "file.txt", "ok"),
        )
        conn.commit()
        conn.close()

        # Reconnecting via the app's normal path must add the missing
        # columns without touching the existing row.
        conn = connect(self.db_path)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(files)")}
        self.assertIn("metadata_status", columns)
        self.assertIn("metadata_error", columns)

        row = conn.execute("SELECT * FROM files WHERE path = '/old/file.txt'").fetchone()
        self.assertEqual(row["name"], "file.txt")
        self.assertEqual(row["scan_status"], "ok")
        self.assertIsNone(row["metadata_status"])
        conn.close()

    def test_init_db_is_idempotent(self):
        conn = connect(self.db_path)
        init_db(conn)  # calling again must not raise or duplicate columns
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(files)")]
        self.assertEqual(len(columns), len(set(columns)))
        conn.close()


class FaceObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = Path(self.tmpdir.name) / "catalog.sqlite"

    def _obs(self, x=10, y=20, w=30, h=40, conf=0.9):
        return FaceObservation(
            bbox_x=x,
            bbox_y=y,
            bbox_width=w,
            bbox_height=h,
            detection_confidence=conf,
            source_width=100,
            source_height=80,
        )

    def test_insert_clear_fetch_round_trip(self):
        from local_file_organizer import database

        conn = connect(self.db_path)
        database.insert_face_observations(
            conn, "/img/photo.png", "image", [self._obs()], timestamp_ms=None
        )
        database.insert_face_observations(
            conn,
            "/v/clip.mp4",
            "video_frame",
            [self._obs(), self._obs(x=1)],
            timestamp_ms=500.0,
        )
        conn.commit()

        counts = database.fetch_face_observation_counts(conn)
        self.assertEqual(counts["total"], 3)
        self.assertEqual(counts["files_with_faces"], 2)

        rows = conn.execute(
            "SELECT * FROM face_observations WHERE file_path = ?",
            ("/v/clip.mp4",),
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["source_type"] == "video_frame" for r in rows))
        self.assertTrue(all(r["timestamp_ms"] == 500.0 for r in rows))
        img_rows = conn.execute(
            "SELECT * FROM face_observations WHERE file_path = ?",
            ("/img/photo.png",),
        ).fetchall()
        self.assertEqual(len(img_rows), 1)
        self.assertIsNone(img_rows[0]["timestamp_ms"])

        # Clearing one file leaves the other's rows behind.
        database.clear_face_observations(conn, "/img/photo.png")
        conn.commit()
        counts = database.fetch_face_observation_counts(conn)
        self.assertEqual(counts["total"], 2)
        self.assertEqual(counts["files_with_faces"], 1)

        # Re-inserting after clear replaces rather than accumulates.
        database.clear_face_observations(conn, "/v/clip.mp4")
        database.insert_face_observations(
            conn, "/v/clip.mp4", "video_frame", [self._obs()], timestamp_ms=1000.0
        )
        conn.commit()
        counts = database.fetch_face_observation_counts(conn, "/v/clip.mp4")
        self.assertEqual(counts["total"], 1)
        conn.close()

    def test_insert_empty_is_noop(self):
        from local_file_organizer import database

        conn = connect(self.db_path)
        database.insert_face_observations(conn, "/img/empty.png", "image", [])
        conn.commit()
        counts = database.fetch_face_observation_counts(conn)
        self.assertEqual(counts, {"total": 0, "files_with_faces": 0})
        conn.close()


class PersonClusterMemberTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = Path(self.tmpdir.name) / "catalog.sqlite"

    def test_insert_clear_all_fetch_round_trip(self):
        from local_file_organizer import database

        conn = connect(self.db_path)
        database.insert_person_cluster_members(
            conn,
            [
                ("P0001", "/img/a.png", "image", 1),
                ("P0001", "/v/clip.mp4", "video", 5),
                ("P0002", "/img/b.png", "image", 2),
            ],
        )
        conn.commit()

        summary = database.fetch_person_cluster_summary(conn)
        self.assertEqual(len(summary), 2)
        by_label = {row["person_label"]: row for row in summary}
        self.assertEqual(
            by_label["P0001"], {"person_label": "P0001", "files": 2, "images": 1, "videos": 1}
        )
        self.assertEqual(
            by_label["P0002"], {"person_label": "P0002", "files": 1, "images": 1, "videos": 0}
        )

        # Full-table clear (NOT scoped to one file_path — different from
        # every other clear function in this codebase, which are per-file).
        database.clear_all_person_cluster_members(conn)
        conn.commit()
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM person_cluster_members").fetchone()[0], 0
        )
        self.assertEqual(database.fetch_person_cluster_summary(conn), [])
        conn.close()

    def test_clear_all_removes_other_files_rows_too(self):
        from local_file_organizer import database

        conn = connect(self.db_path)
        database.insert_person_cluster_members(
            conn,
            [
                ("P0001", "/img/a.png", "image", 1),
                ("P0002", "/img/b.png", "image", 1),
            ],
        )
        conn.commit()
        database.clear_all_person_cluster_members(conn)
        conn.commit()
        remaining = conn.execute("SELECT COUNT(*) FROM person_cluster_members").fetchone()[0]
        self.assertEqual(remaining, 0)
        conn.close()

    def test_insert_empty_is_noop(self):
        from local_file_organizer import database

        conn = connect(self.db_path)
        database.insert_person_cluster_members(conn, [])
        conn.commit()
        self.assertEqual(database.fetch_person_cluster_summary(conn), [])
        conn.close()


if __name__ == "__main__":
    unittest.main()
