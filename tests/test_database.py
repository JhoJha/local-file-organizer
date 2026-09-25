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


class PersonClusterObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = Path(self.tmpdir.name) / "catalog.sqlite"

    def test_table_columns_exist_on_fresh_db(self):
        conn = connect(self.db_path)
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(person_cluster_observations)")
        }
        self.assertEqual(
            columns,
            {
                "id",
                "person_label",
                "file_path",
                "source_type",
                "timestamp_ms",
                "bbox_x",
                "bbox_y",
                "bbox_width",
                "bbox_height",
                "detection_confidence",
            },
        )
        indexes = {
            row["name"]
            for row in conn.execute("PRAGMA index_list(person_cluster_observations)")
        }
        self.assertIn("idx_person_cluster_observations_label", indexes)
        self.assertIn("idx_person_cluster_observations_path", indexes)
        conn.close()

    def test_insert_fetch_clear_all_round_trip(self):
        from local_file_organizer import database

        conn = connect(self.db_path)
        database.insert_person_cluster_observations(
            conn,
            [
                ("P0002", "/img/b.png", "image", None, 1, 2, 3, 4, 0.5),
                ("P0001", "/v/clip.mp4", "video_frame", 1500.0, 5, 6, 7, 8, 0.9),
                ("P0001", "/img/a.png", "image", None, 9, 10, 11, 12, 0.7),
            ],
        )
        conn.commit()

        rows = database.fetch_person_cluster_observations(conn)
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [(r["person_label"], r["file_path"]) for r in rows],
            [
                ("P0001", "/img/a.png"),
                ("P0001", "/v/clip.mp4"),
                ("P0002", "/img/b.png"),
            ],
        )
        video_row = rows[1]
        self.assertEqual(video_row["source_type"], "video_frame")
        self.assertEqual(video_row["timestamp_ms"], 1500.0)
        self.assertEqual(
            (
                video_row["bbox_x"],
                video_row["bbox_y"],
                video_row["bbox_width"],
                video_row["bbox_height"],
            ),
            (5, 6, 7, 8),
        )
        self.assertEqual(video_row["detection_confidence"], 0.9)

        filtered = database.fetch_person_cluster_observations(conn, "P0001")
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(r["person_label"] == "P0001" for r in filtered))

        # Full-table clear (same global rebuild semantics as members).
        database.clear_all_person_cluster_observations(conn)
        conn.commit()
        self.assertEqual(database.fetch_person_cluster_observations(conn), [])
        conn.close()

    def test_insert_empty_is_noop(self):
        from local_file_organizer import database

        conn = connect(self.db_path)
        database.insert_person_cluster_observations(conn, [])
        conn.commit()
        self.assertEqual(database.fetch_person_cluster_observations(conn), [])
        conn.close()

    def test_large_path_list_chunking(self):
        from local_file_organizer import database

        conn = connect(self.db_path)
        n = 1200
        paths = [f"/path/file_{i:04d}.jpg" for i in range(n)]

        # Insert 1200 files in bulk
        conn.executemany(
            "INSERT INTO files (path, name, extension, category, size_bytes, scan_status, "
            "perceptual_hash, capture_date, device_make, gps_latitude, gps_longitude, "
            "duplicate_group, similar_group, file_state) "
            "VALUES (?, 'f.jpg', '.jpg', 'image', 100, 'ok', 'aabb1122', '2025-01-01', "
            "'Camera', 10.0, 20.0, 'DUP_0001', 'SIM_0001', 'active')",
            [(p,) for p in paths],
        )
        conn.commit()

        # Test chunking across 1200 paths (> _SQL_CHUNK_SIZE of 500)
        totals = database.fetch_category_totals(conn, paths=paths)
        self.assertEqual(len(totals), 1)
        self.assertEqual(totals[0]["category"], "image")
        self.assertEqual(totals[0]["count"], n)
        self.assertEqual(totals[0]["total_bytes"], n * 100)

        err_count = database.fetch_error_count(conn, paths=paths)
        self.assertEqual(err_count, 0)

        hash_count = database.fetch_perceptual_hash_count(conn, paths=paths)
        self.assertEqual(hash_count, n)

        exif = database.fetch_exif_counts(conn, paths=paths)
        self.assertEqual(exif["with_capture_date"], n)
        self.assertEqual(exif["with_device"], n)
        self.assertEqual(exif["with_gps"], n)

        # Clear duplicate and similar groups across the 1200 paths
        database.clear_duplicate_groups(conn, paths)
        database.clear_similar_groups(conn, paths)
        conn.commit()

        cur = conn.execute("SELECT COUNT(*) FROM files WHERE duplicate_group IS NOT NULL")
        self.assertEqual(cur.fetchone()[0], 0)
        cur = conn.execute("SELECT COUNT(*) FROM files WHERE similar_group IS NOT NULL")
        self.assertEqual(cur.fetchone()[0], 0)
        conn.close()

    def test_fetch_helpers_active_only(self):
        from local_file_organizer import database

        conn = connect(self.db_path)
        database.upsert_file(
            conn,
            database.FileRecord(
                path="/active.jpg", name="active.jpg", extension=".jpg",
                category="image", size_bytes=200, scan_status="ok",
                metadata_status="ok", perceptual_hash="11223344",
                capture_date="2025-01-01", device_make="Phone",
                gps_latitude=1.0, gps_longitude=2.0, file_state="active",
            ),
        )
        database.upsert_file(
            conn,
            database.FileRecord(
                path="/stale.jpg", name="stale.jpg", extension=".jpg",
                category="image", size_bytes=300, scan_status="error",
                scan_error="stale read error", metadata_status="error",
                perceptual_hash="55667788", capture_date="2025-01-02",
                device_make="OldPhone", gps_latitude=3.0, gps_longitude=4.0,
                file_state="stale",
            ),
        )
        conn.commit()

        # active_only=True excludes stale.jpg
        totals = database.fetch_category_totals(conn, active_only=True)
        self.assertEqual(len(totals), 1)
        self.assertEqual(totals[0]["count"], 1)
        self.assertEqual(totals[0]["total_bytes"], 200)

        # active_only=False includes stale.jpg
        all_totals = database.fetch_category_totals(conn, active_only=False)
        self.assertEqual(len(all_totals), 1)
        self.assertEqual(all_totals[0]["count"], 2)
        self.assertEqual(all_totals[0]["total_bytes"], 500)

        self.assertEqual(database.fetch_error_count(conn, active_only=True), 0)
        self.assertEqual(database.fetch_error_count(conn, active_only=False), 1)

        self.assertEqual(database.fetch_perceptual_hash_count(conn, active_only=True), 1)
        self.assertEqual(database.fetch_perceptual_hash_count(conn, active_only=False), 2)

        exif_active = database.fetch_exif_counts(conn, active_only=True)
        self.assertEqual(exif_active["with_capture_date"], 1)
        self.assertEqual(exif_active["with_device"], 1)
        self.assertEqual(exif_active["with_gps"], 1)

        exif_all = database.fetch_exif_counts(conn, active_only=False)
        self.assertEqual(exif_all["with_capture_date"], 2)
        self.assertEqual(exif_all["with_device"], 2)
        self.assertEqual(exif_all["with_gps"], 2)
        conn.close()


if __name__ == "__main__":
    unittest.main()
