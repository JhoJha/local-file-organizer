"""Tests for the static offline organization plan viewer (T1).

Every fixture is synthetic (temp directories only); no test touches real
personal files. The viewer must render opaque file ids with working
destination/person/confidence filters, stay fully offline, and never leak
real paths unless the explicit manual ``show_files`` flag is used.
"""

import io
import re
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

from local_file_organizer import database
from local_file_organizer.organization_plan import (
    VIEW_DIR_NAME,
    generate_plan_view_thumbnails,
    render_plan_view_html,
    write_plan_view_site,
)
from local_file_organizer.organization_preview import generate_organization_preview
from local_file_organizer.video_sampling import SamplingResult

FFMPEG_PRESENT = shutil.which("ffmpeg") is not None


def _make_png(path: Path, size=(400, 300), color=(30, 120, 200)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def _make_jpeg(path: Path, size=(100, 500), color=(200, 40, 40)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG")
    return path


def _make_video(path: Path, duration: float = 2.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-f", "lavfi", "-i",
            f"testsrc=duration={duration}:size=160x120:rate=10",
            "-y", str(path),
        ],
        capture_output=True,
        timeout=60,
        check=True,
    )
    return path


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
                    "src_hash",
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


class ViewerThumbnailBase(PlanViewTestBase):
    def _add_record(self, conn, path: Path, category: str,
                    duration_seconds=None) -> None:
        database.upsert_file(
            conn,
            database.FileRecord(
                path=str(path),
                name=path.name,
                extension=path.suffix.lower(),
                category=category,
                scan_status="ok",
                metadata_status="ok"
                if category in ("image", "video")
                else "skipped",
                duration_seconds=duration_seconds,
                file_state="active",
            ),
        )

    def _seed_real_images(self):
        """Two real images (wide + tall) plus one document; temp dirs only."""
        media = self.root / "media"
        wide = _make_png(media / "a.png", size=(400, 300))
        tall = _make_jpeg(media / "b.jpg", size=(100, 500))
        doc = self.root / "media" / "notes.pdf"
        conn = database.connect(self.db_path)
        self._add_record(conn, wide, "image")
        self._add_record(conn, tall, "image")
        self._add_record(conn, doc, "document")
        conn.commit()
        conn.close()
        return [wide, tall, doc]

    def _referenced_assets(self, html: str) -> list[str]:
        return re.findall(r'(?:src|href)="(assets/[^"]+)"', html)


class ViewerImageThumbnailTests(ViewerThumbnailBase):
    def test_image_thumbs_emitted_downscaled_and_referenced(self):
        self._seed_real_images()
        result = generate_organization_preview(
            db_path=self.db_path, plan_path=self.plan_path
        )
        view_dir = result.view_path.parent
        for file_id in ("F000001", "F000002"):
            thumb = view_dir / "assets" / f"{file_id}.jpg"
            self.assertTrue(thumb.exists(), thumb)
            with Image.open(thumb) as handle:
                self.assertEqual(handle.format, "JPEG")
                self.assertLessEqual(max(handle.size), 256)
        # Documents never get a thumbnail file.
        self.assertFalse((view_dir / "assets" / "F000003.jpg").exists())

        html = result.view_path.read_text(encoding="utf-8")
        self.assertIn('<img src="assets/F000001.jpg"', html)
        self.assertIn('<img src="assets/F000002.jpg"', html)
        self.assertNotIn("assets/F000003.jpg", html)
        self.assertIn('<span class="no-thumb">', html)

    def test_referenced_thumbs_all_exist_never_broken(self):
        self._seed_real_images()
        result = generate_organization_preview(
            db_path=self.db_path, plan_path=self.plan_path
        )
        html = result.view_path.read_text(encoding="utf-8")
        referenced = self._referenced_assets(html)
        self.assertTrue(referenced)
        for relative in referenced:
            self.assertTrue(
                (result.view_path.parent / relative).exists(), relative
            )

    def test_offline_with_thumbs_and_no_paths_by_default(self):
        secrets = [str(p) for p in self._seed_real_images()]
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
        for secret in secrets:
            self.assertNotIn(secret, html)

    def test_thumbs_only_written_under_view_dir(self):
        self._seed_real_images()
        result = generate_organization_preview(
            db_path=self.db_path, plan_path=self.plan_path
        )
        view_assets = result.view_path.parent / "assets"
        stray = [
            path
            for path in self.root.rglob("F*.jpg")
            if view_assets not in path.parents
        ]
        self.assertEqual(stray, [])

    def test_missing_source_yields_placeholder(self):
        missing = self.root / "media" / "ghost.png"
        conn = database.connect(self.db_path)
        self._add_record(conn, missing, "image")
        conn.commit()
        conn.close()
        result = generate_organization_preview(
            db_path=self.db_path, plan_path=self.plan_path
        )
        html = result.view_path.read_text(encoding="utf-8")
        self.assertNotIn('<img src="assets/F000001.jpg"', html)
        self.assertIn('<span class="no-thumb">', html)
        for relative in self._referenced_assets(html):
            self.assertTrue(
                (result.view_path.parent / relative).exists(), relative
            )


class ViewerVideoThumbnailTests(ViewerThumbnailBase):
    @unittest.skipUnless(FFMPEG_PRESENT, "ffmpeg not installed")
    def test_video_middle_frame_thumb_emitted(self):
        clip = _make_video(self.root / "media" / "clip.mp4", duration=2.0)
        conn = database.connect(self.db_path)
        self._add_record(conn, clip, "video", duration_seconds=2.0)
        conn.commit()
        conn.close()
        before = set((self.root / "media").iterdir())
        result = generate_organization_preview(
            db_path=self.db_path, plan_path=self.plan_path
        )
        thumb = result.view_path.parent / "assets" / "F000001.jpg"
        self.assertTrue(thumb.exists())
        with Image.open(thumb) as handle:
            self.assertEqual(handle.format, "JPEG")
            self.assertLessEqual(max(handle.size), 256)
        html = result.view_path.read_text(encoding="utf-8")
        self.assertIn('<img src="assets/F000001.jpg"', html)
        # Only the thumbnail persists; the scanned tree is untouched.
        self.assertEqual(set((self.root / "media").iterdir()), before)

    def test_corrupt_video_yields_placeholder_never_broken(self):
        fake = self.root / "media" / "clip.mp4"
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_bytes(b"fake-video-bytes")
        conn = database.connect(self.db_path)
        self._add_record(conn, fake, "video", duration_seconds=2.0)
        conn.commit()
        conn.close()
        result = generate_organization_preview(
            db_path=self.db_path, plan_path=self.plan_path
        )
        html = result.view_path.read_text(encoding="utf-8")
        self.assertNotIn('<img src="assets/F000001.jpg"', html)
        self.assertIn('<span class="no-thumb">', html)
        self.assertFalse(
            (result.view_path.parent / "assets" / "F000001.jpg").exists()
        )

    def test_middle_timestamp_and_ephemeral_sampling(self):
        frame = _make_png(self.root / "frame.png", size=(64, 64))
        calls: list[dict] = []

        def _fake_sampler(path, timestamps, on_frame=None, **kwargs):
            calls.append(
                {
                    "path": path,
                    "timestamps": list(timestamps),
                    "keep_debug": kwargs.get("keep_debug_frames"),
                }
            )
            if on_frame is not None:
                for index, stamp in enumerate(timestamps):
                    on_frame(frame, stamp, index)
            return SamplingResult(
                timestamps_requested=len(timestamps),
                frames_generated=len(timestamps),
                frames_failed=0,
                samples=[],
                video_error=None,
            )

        entries = [
            {
                "file_id": "F000001",
                "category": "video",
                "bucket": "planned",
                "proposed_destination": "Media/Sin_personas/",
                "reason_codes": ["no_person_detected"],
                "flags": [],
                "confidence_level": "high",
                "action": "preview_only",
            }
        ]
        counts = {
            "catalog_files": 1,
            "planned": 1,
            "review": 0,
            "unsupported": 0,
            "by_destination": {"Media/Sin_personas/": 1},
            "confidence_levels": {"high": 1, "medium": 0, "low": 0},
        }
        view_dir = self.root / "view"
        write_plan_view_site(
            entries,
            counts,
            view_dir,
            sources_by_file_id={
                "F000001": {
                    "path": str(self.root / "media" / "clip.mp4"),
                    "category": "video",
                    "duration_seconds": 4.0,
                }
            },
            video_sampler=_fake_sampler,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["timestamps"], [2.0])
        self.assertFalse(calls[0]["keep_debug"])
        self.assertTrue((view_dir / "assets" / "F000001.jpg").exists())

    def test_sampler_error_yields_placeholder(self):
        def _failing_sampler(path, timestamps, on_frame=None, **kwargs):
            return SamplingResult(
                timestamps_requested=len(timestamps),
                frames_generated=0,
                frames_failed=len(timestamps),
                samples=[],
                video_error="ffmpeg exploded",
            )

        thumbs = generate_plan_view_thumbnails(
            {
                "F000001": {
                    "path": str(self.root / "media" / "clip.mp4"),
                    "category": "video",
                    "duration_seconds": 4.0,
                }
            },
            self.root / "view" / "assets",
            video_sampler=_failing_sampler,
        )
        self.assertEqual(thumbs, {})
        self.assertFalse((self.root / "view" / "assets" / "F000001.jpg").exists())


if __name__ == "__main__":
    unittest.main()
