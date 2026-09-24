"""Tests for V3.0 ExifTool enrichment: pure parsing, graceful degradation,
database migration and scan wiring. Only touches temporary directories."""
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from local_file_organizer import database
from local_file_organizer.cli import _scan_one_file
from local_file_organizer.exiftool import (
    exiftool_available,
    get_exif_metadata,
    normalize_exif_datetime,
    parse_exif_tags,
)


def _write_exif_jpeg(path: Path) -> None:
    """Write a small JPEG carrying Make/Model/Software/DateTimeOriginal/GPS."""
    piexif = __import__("piexif")
    from PIL import Image

    img = Image.new("RGB", (64, 64), (10, 120, 200))
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: "Xiaomi",
            piexif.ImageIFD.Model: "POCO X6",
            piexif.ImageIFD.Software: "test-sw",
            piexif.ImageIFD.Orientation: 6,
        },
        "Exif": {piexif.ExifIFD.DateTimeOriginal: "2024:07:03 14:22:11"},
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: "N",
            piexif.GPSIFD.GPSLatitude: ((40, 1), (24, 1), (1000, 100)),
            piexif.GPSIFD.GPSLongitudeRef: "W",
            piexif.GPSIFD.GPSLongitude: ((3, 1), (40, 1), (500, 100)),
        },
        "0": {},
        "thumbnail": None,
    }
    img.save(str(path), exif=piexif.dump(exif_dict))


def _write_stripped_jpeg(path: Path) -> None:
    """WhatsApp-style: valid image, zero EXIF tags."""
    from PIL import Image

    Image.new("RGB", (32, 32), (200, 200, 200)).save(str(path))


class NormalizeDatetimeTests(unittest.TestCase):
    def test_valid_normalizes_to_iso(self):
        self.assertEqual(
            normalize_exif_datetime("2024:07:03 14:22:11"),
            "2024-07-03T14:22:11",
        )

    def test_zero_placeholder_rejected(self):
        self.assertIsNone(normalize_exif_datetime("0000:00:00 00:00:00"))

    def test_garbage_rejected(self):
        for raw in ("", "yesterday", "2024-07-03", "2024:13:40 25:61:61", None, 42):
            self.assertIsNone(normalize_exif_datetime(raw), msg=repr(raw))

    def test_trailing_timezone_ignored(self):
        # Some files append an offset; the leading wall-clock still parses.
        self.assertEqual(
            normalize_exif_datetime("2024:07:03 14:22:11-05:00"),
            "2024-07-03T14:22:11",
        )


class ParseTagsTests(unittest.TestCase):
    def test_full_still_tags(self):
        meta = parse_exif_tags(
            {
                "Make": "Xiaomi",
                "Model": "POCO X6",
                "Software": "HyperOS",
                "DateTimeOriginal": "2024:07:03 14:22:11",
                "GPSLatitude": 40.4027,
                "GPSLongitude": -3.668,
                "Orientation": 6,
            }
        )
        self.assertIsNone(meta.error)
        self.assertEqual(meta.capture_date, "2024-07-03T14:22:11")
        self.assertEqual(meta.device_make, "Xiaomi")
        self.assertEqual(meta.device_model, "POCO X6")
        self.assertAlmostEqual(meta.gps_latitude, 40.4027)
        self.assertAlmostEqual(meta.gps_longitude, -3.668)
        self.assertEqual(meta.orientation, 6)

    def test_video_create_date_fallback(self):
        meta = parse_exif_tags({"CreateDate": "2025:08:17 10:00:00"})
        self.assertEqual(meta.capture_date, "2025-08-17T10:00:00")

    def test_datetime_original_wins_over_create_date(self):
        meta = parse_exif_tags(
            {
                "DateTimeOriginal": "2024:07:03 14:22:11",
                "CreateDate": "2025:08:17 10:00:00",
            }
        )
        self.assertEqual(meta.capture_date, "2024-07-03T14:22:11")

    def test_zero_create_date_is_unknown_not_error(self):
        meta = parse_exif_tags({"CreateDate": "0000:00:00 00:00:00"})
        self.assertIsNone(meta.error)
        self.assertIsNone(meta.capture_date)

    def test_empty_tags_is_unknown_not_error(self):
        meta = parse_exif_tags({})
        self.assertIsNone(meta.error)
        self.assertIsNone(meta.capture_date)
        self.assertIsNone(meta.device_make)
        self.assertIsNone(meta.gps_latitude)

    def test_out_of_range_gps_rejected(self):
        meta = parse_exif_tags({"GPSLatitude": 999.0, "GPSLongitude": -200.0})
        self.assertIsNone(meta.gps_latitude)
        self.assertIsNone(meta.gps_longitude)

    def test_bad_orientation_rejected(self):
        for raw in (0, 9, "up", None):
            meta = parse_exif_tags({"Orientation": raw})
            self.assertIsNone(meta.orientation, msg=repr(raw))

    def test_non_dict_is_error(self):
        meta = parse_exif_tags(["not", "a", "dict"])
        self.assertIsNotNone(meta.error)

    def test_blank_strings_become_none(self):
        meta = parse_exif_tags({"Make": "   ", "Model": ""})
        self.assertIsNone(meta.device_make)
        self.assertIsNone(meta.device_model)


class MissingBinaryTests(unittest.TestCase):
    def test_missing_binary_returns_error_without_raising(self):
        with mock.patch(
            "local_file_organizer.exiftool.exiftool_available", return_value=False
        ):
            meta = get_exif_metadata(Path("/tmp/whatever.jpg"))
        self.assertIsNotNone(meta.error)
        self.assertIn("not found", meta.error)
        self.assertIsNone(meta.capture_date)

    def test_nonzero_exit_returns_error_without_raising(self):
        fake = mock.Mock(returncode=1, stdout="", stderr="Error: bad file\n")
        with (
            mock.patch(
                "local_file_organizer.exiftool.exiftool_available",
                return_value=True,
            ),
            mock.patch("subprocess.run", return_value=fake),
        ):
            meta = get_exif_metadata(Path("/tmp/bad.jpg"))
        self.assertIsNotNone(meta.error)

    def test_invalid_json_returns_error_without_raising(self):
        fake = mock.Mock(returncode=0, stdout="not json", stderr="")
        with (
            mock.patch(
                "local_file_organizer.exiftool.exiftool_available",
                return_value=True,
            ),
            mock.patch("subprocess.run", return_value=fake),
        ):
            meta = get_exif_metadata(Path("/tmp/bad.jpg"))
        self.assertIsNotNone(meta.error)


@unittest.skipUnless(exiftool_available(), "exiftool binary not on PATH")
class RealBinaryTests(unittest.TestCase):
    def test_exif_fixture_roundtrip(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "foto.jpg"
            _write_exif_jpeg(target)
            meta = get_exif_metadata(target)
        self.assertIsNone(meta.error)
        self.assertEqual(meta.capture_date, "2024-07-03T14:22:11")
        self.assertEqual(meta.device_make, "Xiaomi")
        self.assertEqual(meta.device_model, "POCO X6")
        self.assertEqual(meta.device_software, "test-sw")
        self.assertAlmostEqual(meta.gps_latitude, 40.4027, places=3)
        self.assertAlmostEqual(meta.gps_longitude, -3.668, places=3)
        self.assertEqual(meta.orientation, 6)

    def test_stripped_image_is_unknown_not_error(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "wa.jpg"
            _write_stripped_jpeg(target)
            meta = get_exif_metadata(target)
        self.assertIsNone(meta.error)
        self.assertIsNone(meta.capture_date)
        self.assertIsNone(meta.device_make)
        self.assertIsNone(meta.gps_latitude)

    def test_corrupt_file_is_error_not_raise(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "roto.jpg"
            target.write_bytes(b"\x00\x01\x02not an image")
            meta = get_exif_metadata(target)
        # Either exiftool reports an error or finds no tags; both are fine,
        # what matters is no exception escapes.
        self.assertIsNone(meta.capture_date)


class MigrationTests(unittest.TestCase):
    def _open_legacy_db(self, db_path: Path) -> sqlite3.Connection:
        """Simulate a pre-V3.0 catalog missing the EXIF columns."""
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE files ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "path TEXT NOT NULL UNIQUE, name TEXT, category TEXT, "
            "scan_status TEXT, metadata_status TEXT)"
        )
        conn.execute(
            "INSERT INTO files (path, name, category, scan_status, metadata_status) "
            "VALUES ('/a.jpg', 'a.jpg', 'image', 'ok', 'ok')"
        )
        conn.commit()
        return conn

    def test_legacy_db_gains_exif_columns_with_data_preserved(self):
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "legacy.sqlite"
            legacy = self._open_legacy_db(db_path)
            legacy.close()
            conn = database.connect(db_path)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(files)")}
            for column in (
                "capture_date",
                "device_make",
                "device_model",
                "device_software",
                "gps_latitude",
                "gps_longitude",
                "gps_altitude",
                "orientation",
            ):
                self.assertIn(column, columns)
            row = conn.execute("SELECT * FROM files WHERE path = '/a.jpg'").fetchone()
            self.assertEqual(row["category"], "image")
            self.assertIsNone(row["capture_date"])
            conn.close()

    def test_exif_counts(self):
        with TemporaryDirectory() as tmp:
            conn = database.connect(Path(tmp) / "c.sqlite")
            database.upsert_file(
                conn,
                database.FileRecord(
                    path="/a.jpg",
                    category="image",
                    capture_date="2024-07-03T14:22:11",
                    device_make="Xiaomi",
                    gps_latitude=40.4,
                    gps_longitude=-3.6,
                ),
            )
            database.upsert_file(
                conn, database.FileRecord(path="/b.jpg", category="image")
            )
            conn.commit()
            counts = database.fetch_exif_counts(conn)
            self.assertEqual(counts["with_capture_date"], 1)
            self.assertEqual(counts["with_device"], 1)
            self.assertEqual(counts["with_gps"], 1)
            scoped = database.fetch_exif_counts(conn, paths=["/b.jpg"])
            self.assertEqual(scoped["with_capture_date"], 0)
            conn.close()


@unittest.skipUnless(exiftool_available(), "exiftool binary not on PATH")
class ScanWiringTests(unittest.TestCase):
    def test_scan_populates_exif_columns(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exif_jpeg(root / "foto.jpg")
            record = _scan_one_file(root / "foto.jpg", enable_exif=True)
        self.assertEqual(record.metadata_status, "ok")
        self.assertEqual(record.capture_date, "2024-07-03T14:22:11")
        self.assertEqual(record.device_model, "POCO X6")
        self.assertIsNotNone(record.gps_latitude)

    def test_scan_without_exif_keeps_v251_behavior(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_stripped_jpeg(root / "wa.jpg")
            record = _scan_one_file(root / "wa.jpg", enable_exif=True)
        self.assertEqual(record.metadata_status, "ok")
        self.assertIsNone(record.capture_date)
        self.assertIsNone(record.device_make)

    def test_scan_with_exif_disabled_never_calls_binary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exif_jpeg(root / "foto.jpg")
            with mock.patch(
                "local_file_organizer.cli.get_exif_metadata",
                side_effect=AssertionError("must not be called"),
            ):
                record = _scan_one_file(root / "foto.jpg", enable_exif=False)
        self.assertIsNone(record.capture_date)


if __name__ == "__main__":
    unittest.main()
