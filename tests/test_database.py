import sqlite3
import tempfile
import unittest
from pathlib import Path

from local_file_organizer.database import FileRecord, connect, init_db, upsert_file


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


if __name__ == "__main__":
    unittest.main()
