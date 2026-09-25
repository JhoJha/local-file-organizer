"""Unit tests for organization_preview module."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from local_file_organizer import database
from local_file_organizer.organization_preview import (
    CatalogEmptyError,
    CatalogNotFoundError,
    InvalidDecisionsError,
    PlanDesyncError,
    PreviewResult,
    format_preview_report,
    generate_organization_preview,
)


class OrganizationPreviewTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.dir_path = Path(self.tmpdir.name)
        self.db_path = self.dir_path / "catalog.sqlite"
        self.plan_path = self.dir_path / "plan.json"
        self.decisions_path = self.dir_path / "decisions.json"

    def _seed_catalog(self) -> list[str]:
        conn = database.connect(self.db_path)
        base = str(self.dir_path / "media")
        p_photo = base + "/photo.jpg"
        p_doc = base + "/doc.pdf"
        p_audio = base + "/song.mp3"

        database.upsert_file(
            conn,
            database.FileRecord(
                path=p_photo,
                name="photo.jpg",
                extension=".jpg",
                category="image",
                scan_status="ok",
                metadata_status="ok",
                file_state="active",
            ),
        )
        database.upsert_file(
            conn,
            database.FileRecord(
                path=p_doc,
                name="doc.pdf",
                extension=".pdf",
                category="document",
                scan_status="ok",
                metadata_status="skipped",
                file_state="active",
            ),
        )
        database.upsert_file(
            conn,
            database.FileRecord(
                path=p_audio,
                name="song.mp3",
                extension=".mp3",
                category="audio",
                scan_status="ok",
                metadata_status="skipped",
                file_state="active",
            ),
        )
        conn.commit()
        conn.close()
        return [p_photo, p_doc, p_audio]

    def test_generate_preview_result_structure_and_silence(self):
        """Service returns PreviewResult and emits zero bytes to stdout/stderr."""
        self._seed_catalog()

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            result = generate_organization_preview(
                db_path=self.db_path,
                plan_path=self.plan_path,
                show_files=True,
            )

        self.assertEqual(stdout_buf.getvalue(), "")
        self.assertEqual(stderr_buf.getvalue(), "")

        self.assertIsInstance(result, PreviewResult)
        self.assertEqual(result.plan_path, self.plan_path)
        self.assertEqual(len(result.entries), 3)
        self.assertIn("by_destination", result.counts)
        self.assertIn("catalog_files", result.counts)
        self.assertEqual(result.counts["catalog_files"], 3)
        self.assertTrue(self.plan_path.exists())

        # Verify paths_by_destination mapping when show_files=True
        self.assertTrue(len(result.paths_by_destination) > 0)

    def test_catalog_not_found_error_on_missing_db(self):
        """Raises CatalogNotFoundError when database file does not exist."""
        missing = self.dir_path / "nonexistent.sqlite"
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            with self.assertRaises(CatalogNotFoundError):
                generate_organization_preview(
                    db_path=missing,
                    plan_path=self.plan_path,
                )
        self.assertEqual(stdout_buf.getvalue(), "")
        self.assertEqual(stderr_buf.getvalue(), "")

    def test_catalog_not_found_error_on_invalid_db(self):
        """Raises CatalogNotFoundError when database file cannot execute files query."""
        corrupt_db = self.dir_path / "corrupt.sqlite"
        corrupt_db.write_bytes(b"not a valid sqlite file")
        with self.assertRaises(CatalogNotFoundError):
            generate_organization_preview(
                db_path=corrupt_db,
                plan_path=self.plan_path,
            )

    def test_catalog_empty_error_when_zero_files(self):
        """Raises CatalogEmptyError when files table has 0 rows."""
        conn = database.connect(self.db_path)
        conn.close()

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            with self.assertRaises(CatalogEmptyError):
                generate_organization_preview(
                    db_path=self.db_path,
                    plan_path=self.plan_path,
                )
        self.assertEqual(stdout_buf.getvalue(), "")
        self.assertEqual(stderr_buf.getvalue(), "")

    def test_invalid_decisions_error(self):
        """Raises InvalidDecisionsError when decisions file contains invalid JSON."""
        self._seed_catalog()
        self.decisions_path.write_text("{malformed json", encoding="utf-8")

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            with self.assertRaises(InvalidDecisionsError):
                generate_organization_preview(
                    db_path=self.db_path,
                    plan_path=self.plan_path,
                    decisions_path=self.decisions_path,
                )
        self.assertEqual(stdout_buf.getvalue(), "")
        self.assertEqual(stderr_buf.getvalue(), "")

    def test_connection_closed_always(self):
        """Database connection is closed even when an error occurs during preview."""
        self._seed_catalog()
        real_connect = database.connect
        recorded_connections = []

        def spy_connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            recorded_connections.append(conn)
            return conn

        with patch("local_file_organizer.organization_preview.database.connect", side_effect=spy_connect):
            with patch("local_file_organizer.organization_preview.build_organization_plan", side_effect=ValueError("boom")):
                with self.assertRaises(ValueError):
                    generate_organization_preview(
                        db_path=self.db_path,
                        plan_path=self.plan_path,
                    )

        self.assertEqual(len(recorded_connections), 1)
        # Connection should be closed: querying a closed sqlite3 connection raises ProgrammingError
        import sqlite3
        with self.assertRaises(sqlite3.ProgrammingError):
            recorded_connections[0].execute("SELECT 1")

    def test_plan_desync_error(self):
        """Raises PlanDesyncError when active rows count differs from plan entries."""
        self._seed_catalog()
        # Mock build_organization_plan to return fewer entries than active files
        with patch("local_file_organizer.organization_preview.build_organization_plan", return_value=[]):
            with self.assertRaises(PlanDesyncError):
                generate_organization_preview(
                    db_path=self.db_path,
                    plan_path=self.plan_path,
                    show_files=True,
                )

    def test_format_preview_report_is_pure_and_deterministic(self):
        """format_preview_report performs pure string formatting without side effects."""
        result = PreviewResult(
            plan_path=self.plan_path,
            entries=[],
            counts={
                "by_destination": {
                    "Media/Personas/Alice/": 2,
                    "Media/Parejas/Alice_Bob/": 1,
                    "Media/Grupos/Trip/": 3,
                    "Media/Sin_personas/": 4,
                    "Revisión/Persona_incierta/": 1,
                    "Revisión/Cluster_rechazado/": 1,
                    "Revisión/Baja_evidencia/": 1,
                    "Revisión/Metadata_incompleta/": 1,
                    "Revisión/Errores_de_escaneo/": 1,
                    "Documentos/PDF/": 5,
                    "Audio/MP3/": 2,
                    "Archivos_comprimidos/": 3,
                    "Software/": 1,
                    "Otros/DAT/": 1,
                },
                "exact_duplicates": 2,
                "visual_similar_groups": 1,
                "confidence_levels": {"high": 10, "medium": 5, "low": 3},
                "catalog_files": 18,
                "planned": 15,
                "review": 5,
                "unsupported": 0,
            },
            paths_by_destination={
                "Media/Personas/Alice/": ["/media/alice1.jpg", "/media/alice2.jpg"],
                "Documentos/PDF/": ["/docs/report.pdf"],
            },
        )

        report1 = format_preview_report(result, show_files=False)
        report2 = format_preview_report(result, show_files=False)

        # Pure and deterministic
        self.assertEqual(report1, report2)
        self.assertIn("=== ORGANIZATION PREVIEW ===", report1)
        self.assertIn("Personas/", report1)
        self.assertIn("Alice/  2 archivos", report1)
        self.assertIn("Parejas/", report1)
        self.assertIn("Grupos/", report1)
        self.assertIn("Sin_personas/  4 archivos", report1)
        self.assertIn("Revisión/", report1)
        self.assertIn("Documentos/", report1)
        self.assertIn("Audio/", report1)
        self.assertIn("Archivos_comprimidos/  3 archivos", report1)
        self.assertIn("Software/  1 archivo", report1)
        self.assertIn("Otros/", report1)
        self.assertIn("FLAGS:", report1)
        self.assertIn("=== DESTINATIONS ===", report1)
        self.assertIn("=== CONFIDENCE ===", report1)
        self.assertIn("=== ACCOUNTING ===", report1)
        self.assertIn("PREVIEW ONLY", report1)
        self.assertNotIn("/media/alice1.jpg", report1)

        # With show_files=True
        report_shown = format_preview_report(result, show_files=True)
        self.assertIn("/media/alice1.jpg", report_shown)
        self.assertIn("/docs/report.pdf", report_shown)


if __name__ == "__main__":
    unittest.main()
