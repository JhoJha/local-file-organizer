"""Tests for the static offline organization plan viewer (T1).

Every fixture is synthetic (temp directories only); no test touches real
personal files. The viewer must render opaque file ids with working
destination/person/confidence filters, stay fully offline, and never leak
real paths unless the explicit manual ``show_files`` flag is used.
"""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from local_file_organizer import database
from local_file_organizer.organization_plan import (
    VIEW_DIR_NAME,
    render_plan_view_html,
)
from local_file_organizer.organization_preview import generate_organization_preview


class PlanViewTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)
        self.db_path = self.root / "catalog.sqlite"
        self.plan_path = self.root / "output" / "organization_plan.json"

    def _seed(self):
        """One clustered image (P0001 singleton), one plain image, one doc."""
        base = str(self.root / "media")
        p_person = base + "/clustered.jpg"
        p_plain = base + "/plain.jpg"
        p_doc = base + "/notes.pdf"
        conn = database.connect(self.db_path)
        for path, name, ext, category in (
            (p_person, "clustered.jpg", ".jpg", "image"),
            (p_plain, "plain.jpg", ".jpg", "image"),
            (p_doc, "notes.pdf", ".pdf", "document"),
        ):
            database.upsert_file(
                conn,
                database.FileRecord(
                    path=path,
                    name=name,
                    extension=ext,
                    category=category,
                    scan_status="ok",
                    metadata_status="ok" if category == "image" else "skipped",
                    file_state="active",
                ),
            )
        database.insert_person_cluster_members(
            conn, [("P0001", p_person, "image", 1)]
        )
        conn.commit()
        conn.close()
        return [p_person, p_plain, p_doc]


class ViewerEmissionTests(PlanViewTestBase):
    def test_viewer_emitted_alongside_preview_by_default(self):
        self._seed()
        result = generate_organization_preview(
            db_path=self.db_path, plan_path=self.plan_path
        )
        expected = self.plan_path.parent / VIEW_DIR_NAME / "index.html"
        self.assertEqual(result.view_path, expected)
        self.assertTrue(expected.exists())
        html = expected.read_text(encoding="utf-8")
        self.assertIn("Organization Plan Viewer", html)
        self.assertIn("F000001", html)
        self.assertIn('id="filter-destination"', html)

    def test_write_view_false_skips_viewer(self):
        self._seed()
        result = generate_organization_preview(
            db_path=self.db_path, plan_path=self.plan_path, write_view=False
        )
        self.assertIsNone(result.view_path)
        self.assertFalse((self.plan_path.parent / VIEW_DIR_NAME).exists())
        self.assertTrue(self.plan_path.exists())

    def test_custom_view_dir_is_honoured(self):
        self._seed()
        custom = self.root / "custom-view"
        result = generate_organization_preview(
            db_path=self.db_path, plan_path=self.plan_path, view_dir=custom
        )
        self.assertEqual(result.view_path, custom / "index.html")
        self.assertTrue((custom / "index.html").exists())

    def test_plan_json_schema_unchanged(self):
        """The viewer must not leak persons/paths into organization_plan.json."""
        import json

        self._seed()
        generate_organization_preview(
            db_path=self.db_path, plan_path=self.plan_path, show_files=True
        )
        data = json.loads(self.plan_path.read_text(encoding="utf-8"))
        self.assertEqual(len(data), 3)
        for entry in data:
            self.assertEqual(
                set(entry.keys()),
                {
                    "file_id",
                    "category",
                    "bucket",
                    "proposed_destination",
                    "reason_codes",
                    "flags",
                    "confidence_level",
                    "action",
                },
            )


class ViewerFilterTests(PlanViewTestBase):
    def _html(self, **kwargs):
        self._seed()
        result = generate_organization_preview(
            db_path=self.db_path, plan_path=self.plan_path, **kwargs
        )
        return (result.view_path).read_text(encoding="utf-8")

    def test_destination_filter_options(self):
        html = self._html()
        self.assertIn('id="filter-destination"', html)
        for dest in (
            "Media/Sin_personas/",
            "Documentos/PDF/",
            "Revisión/Baja_evidencia/",
        ):
            self.assertIn(f'value="{dest}"', html)
            self.assertIn(f'data-destination="{dest}"', html)

    def test_person_filter_options_and_row_data(self):
        html = self._html()
        self.assertIn('id="filter-person"', html)
        self.assertIn('value="P0001"', html)
        self.assertIn("No person assigned", html)
        # Only the clustered file carries the person label.
        self.assertEqual(html.count('data-persons="P0001"'), 1)
        self.assertEqual(html.count('data-persons=""'), 2)
        self.assertIn("<td>P0001</td>", html)

    def test_confidence_filter_options_and_row_data(self):
        html = self._html()
        self.assertIn('id="filter-confidence"', html)
        for level in ("high", "medium", "low"):
            self.assertIn(f'value="{level}"', html)
        self.assertIn('data-confidence="high"', html)
        self.assertIn('data-confidence="low"', html)

    def test_filter_controls_behaviour_hooks(self):
        html = self._html()
        self.assertIn('id="reset-filters"', html)
        self.assertIn('id="match-count"', html)
        self.assertIn("applyFilters", html)

    def test_summary_counts_rendered(self):
        html = self._html()
        for token in ("planned", "review", "high", "low"):
            self.assertIn(token, html)


class ViewerPrivacyTests(PlanViewTestBase):
    def test_no_real_paths_by_default(self):
        secrets = self._seed()
        result = generate_organization_preview(
            db_path=self.db_path, plan_path=self.plan_path
        )
        html = result.view_path.read_text(encoding="utf-8")
        for secret in secrets:
            self.assertNotIn(secret, html)
        self.assertNotIn("Path (manual view only)", html)

    def test_show_files_embeds_paths_behind_manual_flag(self):
        secrets = self._seed()
        result = generate_organization_preview(
            db_path=self.db_path, plan_path=self.plan_path, show_files=True
        )
        html = result.view_path.read_text(encoding="utf-8")
        for secret in secrets:
            self.assertIn(secret, html)
        self.assertIn("Path (manual view only)", html)


class ViewerOfflineTests(PlanViewTestBase):
    def test_html_has_zero_network_references(self):
        secrets = self._seed()
        assert secrets
        result = generate_organization_preview(
            db_path=self.db_path, plan_path=self.plan_path
        )
        html = result.view_path.read_text(encoding="utf-8")
        for token in (
            "http://",
            "https://",
            "//",
            "<link",
            "@import",
            "fetch(",
            "XMLHttpRequest",
            "WebSocket",
            "sendBeacon",
            "<script src",
        ):
            self.assertNotIn(token, html)
        self.assertIn('lang="en"', html)
        self.assertIn("LOCAL ONLY", html)

    def test_render_is_deterministic(self):
        entries = [
            {
                "file_id": "F000001",
                "category": "image",
                "bucket": "review",
                "proposed_destination": "Revisión/Baja_evidencia/",
                "reason_codes": ["singleton_person_cluster"],
                "flags": [],
                "confidence_level": "low",
                "action": "preview_only",
            }
        ]
        counts = {
            "catalog_files": 1,
            "planned": 0,
            "review": 1,
            "unsupported": 0,
            "by_destination": {"Revisión/Baja_evidencia/": 1},
            "confidence_levels": {"high": 0, "medium": 0, "low": 1},
        }
        persons = {"F000001": ["P0001"]}
        first = render_plan_view_html(entries, counts, persons)
        second = render_plan_view_html(entries, counts, persons)
        self.assertEqual(first, second)
        self.assertIn("F000001", first)
        self.assertIn("P0001", first)


class ViewerCliTests(PlanViewTestBase):
    def test_cli_emits_view_and_prints_location(self):
        from local_file_organizer.cli import run_organize_preview

        self._seed()
        output = io.StringIO()
        with redirect_stdout(output):
            run_organize_preview(db_path=self.db_path, plan_path=self.plan_path)
        self.assertIn("View:", output.getvalue())
        self.assertTrue(
            (self.plan_path.parent / VIEW_DIR_NAME / "index.html").exists()
        )

    def test_cli_no_view_skips_viewer(self):
        from local_file_organizer.cli import run_organize_preview

        self._seed()
        output = io.StringIO()
        with redirect_stdout(output):
            run_organize_preview(
                db_path=self.db_path, plan_path=self.plan_path, write_view=False
            )
        self.assertNotIn("View:", output.getvalue())
        self.assertFalse((self.plan_path.parent / VIEW_DIR_NAME).exists())


if __name__ == "__main__":
    unittest.main()
