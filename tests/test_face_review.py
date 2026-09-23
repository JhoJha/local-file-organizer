"""Tests for the V2.4.1 face-cluster calibration review.

Every fixture is synthetic (Pillow-generated PNGs, mocked video sampling);
no test touches real personal files.
"""
import io
import json
import re
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image

from local_file_organizer import database
from local_file_organizer.cli import run_face_review, run_face_review_import
from local_file_organizer.face_review import (
    DECISION_CONFIRMED,
    DECISION_REJECTED,
    DECISION_UNCERTAIN,
    DECISIONS_VERSION,
    MANIFEST_VERSION,
    build_manifest,
    compute_metrics,
    generate_assets,
    load_decisions,
    parse_decisions_payload,
    render_html,
    save_decisions,
    select_samples,
    validate_decisions,
    write_review_site,
)
from local_file_organizer.video_sampling import SamplingResult


def _make_png(path: Path, size=(80, 80), color=(120, 60, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def _make_video_bytes(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-video-bytes")
    return path


class FaceReviewTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)
        self.target = self.root / "target"
        self.target.mkdir()
        self.output_dir = self.root / "review"
        self.output_dir.mkdir()
        self.db_path = self.root / "catalog.sqlite"
        self.decisions_path = self.root / "decisions.json"
        self.conn = database.connect(self.db_path)
        self.addCleanup(self.conn.close)

    def _add_file(self, path, category="image"):
        database.upsert_file(
            self.conn,
            database.FileRecord(
                path=str(path),
                name=path.name,
                extension=path.suffix.lower(),
                category=category,
                scan_status="ok",
                metadata_status="ok",
            ),
        )

    def _add_member(self, label, path, category):
        database.insert_person_cluster_members(
            self.conn, [(label, str(path), category, 1)]
        )

    def _add_observation(
        self, label, path, source_type, timestamp_ms, bbox=(5, 5, 30, 30), confidence=0.9
    ):
        database.insert_person_cluster_observations(
            self.conn,
            [(label, str(path), source_type, timestamp_ms, *bbox, confidence)],
        )


class ManifestTests(FaceReviewTestBase):
    def _seed(self):
        a1 = _make_png(self.target / "a1.png")
        a2 = _make_video_bytes(self.target / "a2.mp4")
        b1 = _make_png(self.target / "b1.png")
        b2 = _make_png(self.target / "b2.png")
        b3 = _make_png(self.target / "b3.png")
        c1 = _make_png(self.target / "c1.png")
        c2 = _make_png(self.target / "c2.png")
        d1 = _make_png(self.target / "d1.png")
        e1 = _make_png(self.target / "e1.png")
        e2 = _make_png(self.target / "e2.png")
        for path in (a1, a2, b1, b2, b3, c1, c2, d1, e1, e2):
            self._add_file(path, "video" if path.suffix == ".mp4" else "image")
        # P0001: 2 files, cross-media (image + video), 2 observations.
        self._add_member("P0001", a1, "image")
        self._add_member("P0001", a2, "video")
        self._add_observation("P0001", a1, "image", None)
        self._add_observation("P0001", a2, "video_frame", 1000.0)
        # P0002: 3 files, 3 image observations.
        for path in (b1, b2, b3):
            self._add_member("P0002", path, "image")
            self._add_observation("P0002", path, "image", None)
        # P0003: 2 files, 4 image observations.
        self._add_member("P0003", c1, "image")
        self._add_member("P0003", c2, "image")
        self._add_observation("P0003", c1, "image", None)
        self._add_observation("P0003", c1, "image", None, bbox=(40, 40, 20, 20))
        self._add_observation("P0003", c2, "image", None)
        self._add_observation("P0003", c2, "image", None, bbox=(45, 45, 20, 20))
        # P0004: singleton.
        self._add_member("P0004", d1, "image")
        self._add_observation("P0004", d1, "image", None)
        # P0005: 2 files, 2 observations.
        self._add_member("P0005", e1, "image")
        self._add_member("P0005", e2, "image")
        self._add_observation("P0005", e1, "image", None)
        self._add_observation("P0005", e2, "image", None)
        self.conn.commit()
        return {
            "a1": a1, "a2": a2, "b1": b1, "b2": b2, "b3": b3,
            "c1": c1, "c2": c2, "d1": d1, "e1": e1, "e2": e2,
        }

    def test_manifest_schema_priority_and_determinism(self):
        paths = self._seed()
        manifest, details = build_manifest(self.conn)

        self.assertEqual(set(manifest.keys()), {"version", "generated_by", "clusters"})
        self.assertEqual(manifest["version"], MANIFEST_VERSION)
        self.assertEqual(manifest["generated_by"], "face-review")

        # Multi-file/cross-media first, then bigger clusters, singleton last.
        self.assertEqual(
            [cluster["cluster_id"] for cluster in manifest["clusters"]],
            ["P0001", "P0002", "P0003", "P0005", "P0004"],
        )
        for cluster in manifest["clusters"]:
            self.assertEqual(
                set(cluster.keys()),
                {
                    "cluster_id", "observations", "file_count", "images",
                    "videos", "cross_media", "samples",
                },
            )
            for sample in cluster["samples"]:
                self.assertEqual(
                    set(sample.keys()),
                    {"observation_id", "file_id", "source_type", "preview"},
                )

        by_id = {cluster["cluster_id"]: cluster for cluster in manifest["clusters"]}
        self.assertTrue(by_id["P0001"]["cross_media"])
        self.assertEqual(by_id["P0001"]["file_count"], 2)
        self.assertEqual(by_id["P0001"]["images"], 1)
        self.assertEqual(by_id["P0001"]["videos"], 1)
        self.assertFalse(by_id["P0002"]["cross_media"])
        self.assertEqual(by_id["P0002"]["file_count"], 3)
        self.assertEqual(by_id["P0003"]["observations"], 4)
        self.assertEqual(by_id["P0004"]["file_count"], 1)
        self.assertFalse(by_id["P0004"]["cross_media"])
        self.assertEqual(by_id["P0005"]["file_count"], 2)

        # File ids follow ascending path order (same criterion as the plan).
        expected_file_ids = {
            str(path): f"F{index + 1:06d}"
            for index, path in enumerate(sorted(paths.values()))
        }
        for cluster in manifest["clusters"]:
            for sample in cluster["samples"]:
                self.assertEqual(
                    sample["file_id"],
                    expected_file_ids[details[sample["observation_id"]]["path"]],
                )

        # The serialized manifest never contains real paths.
        serialized = json.dumps(manifest, ensure_ascii=False)
        self.assertNotIn(str(self.target), serialized)

        second, second_details = build_manifest(self.conn)
        self.assertEqual(
            json.dumps(manifest, ensure_ascii=False),
            json.dumps(second, ensure_ascii=False),
        )
        self.assertEqual(
            json.dumps(details, ensure_ascii=False, default=str),
            json.dumps(second_details, ensure_ascii=False, default=str),
        )


class SelectionTests(FaceReviewTestBase):
    def test_round_robin_mixed_media_and_video_spacing(self):
        a = _make_png(self.target / "a.png")
        b = _make_png(self.target / "b.png")
        clip = _make_video_bytes(self.target / "clip.mp4")
        self._add_file(a)
        self._add_file(b)
        self._add_file(clip, "video")
        self._add_member("P0001", a, "image")
        self._add_member("P0001", b, "image")
        self._add_member("P0001", clip, "video")
        self._add_observation("P0001", a, "image", None, bbox=(1, 1, 10, 10))
        self._add_observation("P0001", b, "image", None, bbox=(2, 2, 10, 10))
        for timestamp_ms in (500.0, 1000.0, 1500.0, 2000.0):
            self._add_observation(
                "P0001", clip, "video_frame", timestamp_ms, bbox=(3, 3, 10, 10)
            )
        self.conn.commit()

        manifest, _details = build_manifest(self.conn, max_previews_per_cluster=4)
        cluster = manifest["clusters"][0]
        self.assertEqual(len(cluster["samples"]), 4)
        self.assertEqual(
            [sample["preview"] for sample in cluster["samples"]],
            [
                "assets/P0001_01.jpg",
                "assets/P0001_02.jpg",
                "assets/P0001_03.jpg",
                "assets/P0001_04.jpg",
            ],
        )
        rows = database.fetch_person_cluster_observations(self.conn)
        timestamp_by_observation = {
            f"O{row['id']:06d}": row["timestamp_ms"] for row in rows
        }
        selected_timestamps = [
            timestamp_by_observation[sample["observation_id"]]
            for sample in cluster["samples"]
        ]
        # Round-robin: one sample per file per round, and the two video
        # frames come from the farthest-point spread (500ms, 2000ms), never
        # consecutive.
        self.assertEqual(selected_timestamps, [None, None, 500.0, 2000.0])
        self.assertEqual(
            [sample["source_type"] for sample in cluster["samples"]],
            ["image", "image", "video", "video"],
        )

        # The cap is respected and the names stay deterministic.
        capped, _ = build_manifest(self.conn, max_previews_per_cluster=2)
        self.assertEqual(
            [sample["preview"] for sample in capped["clusters"][0]["samples"]],
            ["assets/P0001_01.jpg", "assets/P0001_02.jpg"],
        )

    def test_select_samples_spreads_video_frames(self):
        observations = [
            {
                "observation_id": f"O{index + 1:06d}",
                "file_id": "F000001",
                "file_path": "/v/clip.mp4",
                "source_type": "video",
                "timestamp_ms": timestamp_ms,
                "bbox_x": 0,
                "bbox_y": 0,
                "bbox_width": 10,
                "bbox_height": 10,
            }
            for index, timestamp_ms in enumerate((500.0, 1000.0, 1500.0, 2000.0))
        ]
        selected = select_samples(observations, 3)
        self.assertEqual(
            [observation["timestamp_ms"] for observation in selected],
            [500.0, 2000.0, 1000.0],
        )
        self.assertEqual(select_samples(observations, 0), [])


class AssetTests(FaceReviewTestBase):
    def test_image_thumbnail_written(self):
        photo = _make_png(self.target / "photo.png", size=(120, 90))
        self._add_file(photo)
        self._add_member("P0001", photo, "image")
        self._add_observation("P0001", photo, "image", None, bbox=(20, 20, 40, 40))
        self.conn.commit()

        manifest, details = build_manifest(self.conn)
        assets_dir = self.output_dir / "assets"
        stats = generate_assets(manifest, details, assets_dir)

        self.assertEqual(stats["thumbnails"], 1)
        self.assertEqual(stats["errors"], 0)
        self.assertGreater(stats["assets_bytes"], 0)
        preview = assets_dir / "P0001_01.jpg"
        self.assertTrue(preview.exists())
        with Image.open(preview) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertLessEqual(image.width, 192)
            self.assertLessEqual(image.height, 192)
        self.assertEqual(preview.stat().st_size, stats["assets_bytes"])

    def test_video_thumbnail_with_mocked_sampler(self):
        clip = _make_video_bytes(self.target / "clip.mp4")
        self._add_file(clip, "video")
        self._add_member("P0001", clip, "video")
        for timestamp_ms in (1000.0, 2000.0):
            self._add_observation(
                "P0001", clip, "video_frame", timestamp_ms, bbox=(5, 5, 40, 40)
            )
        self.conn.commit()

        manifest, details = build_manifest(self.conn)
        assets_dir = self.output_dir / "assets"
        calls = []
        temporary_dirs = []

        def _fake_sampler(path, timestamps, on_frame=None, **kwargs):
            calls.append({"path": path, "timestamps": list(timestamps), "kwargs": kwargs})
            with tempfile.TemporaryDirectory() as tmp:
                temporary_dirs.append(Path(tmp))
                for index, timestamp in enumerate(timestamps):
                    frame = Path(tmp) / f"frame_{index:04d}.png"
                    _make_png(frame, size=(160, 120))
                    if on_frame is not None:
                        on_frame(frame, timestamp, index)
            return SamplingResult(
                timestamps_requested=len(timestamps),
                frames_generated=len(timestamps),
                frames_failed=0,
                samples=[],
                video_error=None,
            )

        stats = generate_assets(
            manifest, details, assets_dir, video_sampler=_fake_sampler
        )

        self.assertEqual(stats["thumbnails"], 2)
        self.assertEqual(stats["errors"], 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["timestamps"], [1.0, 2.0])
        self.assertFalse(calls[0]["kwargs"].get("keep_debug_frames"))
        self.assertFalse(temporary_dirs[0].exists())
        written_bytes = 0
        for sample in manifest["clusters"][0]["samples"]:
            self.assertIsNotNone(sample["preview"])
            preview = self.output_dir / sample["preview"]
            self.assertTrue(preview.exists())
            written_bytes += preview.stat().st_size
        self.assertEqual(stats["assets_bytes"], written_bytes)

    def test_missing_file_and_video_error_mark_preview_none(self):
        missing = self.target / "missing.png"
        self._add_file(missing)
        self._add_member("P0001", missing, "image")
        self._add_observation("P0001", missing, "image", None, bbox=(0, 0, 10, 10))
        clip = _make_video_bytes(self.target / "clip.mp4")
        self._add_file(clip, "video")
        self._add_member("P0002", clip, "video")
        self._add_observation("P0002", clip, "video_frame", 1000.0, bbox=(0, 0, 10, 10))
        self.conn.commit()

        manifest, details = build_manifest(self.conn)

        def _video_error_sampler(path, timestamps, on_frame=None, **kwargs):
            return SamplingResult(
                timestamps_requested=len(timestamps),
                frames_generated=0,
                frames_failed=len(timestamps),
                samples=[],
                video_error="ffmpeg exploded",
            )

        assets_dir = self.output_dir / "assets"
        stats = generate_assets(
            manifest, details, assets_dir, video_sampler=_video_error_sampler
        )

        self.assertGreater(stats["errors"], 0)
        self.assertEqual(stats["thumbnails"], 0)
        for cluster in manifest["clusters"]:
            for sample in cluster["samples"]:
                self.assertIsNone(sample["preview"])
        self.assertFalse((assets_dir / "P0001_01.jpg").exists())
        self.assertFalse((assets_dir / "P0002_01.jpg").exists())

    def test_frame_never_delivered_marks_preview_none(self):
        clip = _make_video_bytes(self.target / "clip.mp4")
        self._add_file(clip, "video")
        self._add_member("P0001", clip, "video")
        self._add_observation("P0001", clip, "video_frame", 1000.0, bbox=(0, 0, 10, 10))
        self.conn.commit()

        manifest, details = build_manifest(self.conn)

        def _silent_sampler(path, timestamps, on_frame=None, **kwargs):
            return SamplingResult(
                timestamps_requested=len(timestamps),
                frames_generated=0,
                frames_failed=len(timestamps),
                samples=[],
                video_error=None,
            )

        stats = generate_assets(
            manifest, details, self.output_dir / "assets", video_sampler=_silent_sampler
        )
        self.assertEqual(stats["errors"], 1)
        self.assertIsNone(manifest["clusters"][0]["samples"][0]["preview"])


class HtmlTests(FaceReviewTestBase):
    def _build_site(self):
        photo = _make_png(self.target / "photo.png", size=(100, 100))
        self._add_file(photo)
        self._add_member("P0001", photo, "image")
        self._add_observation("P0001", photo, "image", None, bbox=(10, 10, 50, 50))
        self.conn.commit()
        manifest, details = build_manifest(self.conn)
        decisions = {"P0001": DECISION_CONFIRMED}
        metrics = compute_metrics(manifest, decisions)
        generate_assets(manifest, details, self.output_dir / "assets")
        write_review_site(manifest, decisions, metrics, self.output_dir)
        return manifest, metrics

    def test_html_is_offline_and_assets_exist(self):
        manifest, metrics = self._build_site()
        html = (self.output_dir / "index.html").read_text(encoding="utf-8")

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

        referenced = re.findall(r'(?:src|href)="(assets/[^"]+)"', html)
        self.assertTrue(referenced)
        for relative in referenced:
            self.assertTrue((self.output_dir / relative).exists(), relative)

        self.assertIn('lang="es"', html)
        self.assertIn("LOCAL ONLY", html)
        self.assertIn('data-cluster="P0001"', html)
        self.assertIn("observations=1 files=1 images=1 videos=0", html)
        self.assertIn('id="metric-clusters_total"', html)
        self.assertIn('id="metric-reviewed"', html)
        self.assertIn('id="metric-unreviewed"', html)
        self.assertIn('data-decision="confirmed"', html)
        self.assertIn('data-decision="rejected"', html)
        self.assertIn('data-decision="uncertain"', html)
        self.assertIn('data-decision="unreviewed"', html)
        self.assertIn("Misma persona", html)
        self.assertIn("Personas distintas", html)
        self.assertIn("No seguro", html)
        self.assertIn("Limpiar", html)
        self.assertIn('id="status"', html)
        self.assertIn("lfo.face_review.decisions.v1", html)
        self.assertIn("INITIAL_DECISIONS", html)
        self.assertIn('"P0001": "confirmed"', html)
        self.assertIn("review_decisions.json", html)

        self.assertTrue((self.output_dir / "manifest.json").exists())
        self.assertEqual(
            render_html(manifest, {"P0001": DECISION_CONFIRMED}, metrics), html
        )

    def test_html_without_previews_shows_placeholder(self):
        self._add_file(self.target / "a.png")
        self._add_member("P0001", self.target / "a.png", "image")
        self.conn.commit()
        manifest, details = build_manifest(self.conn)
        metrics = compute_metrics(manifest, {})
        generate_assets(manifest, details, self.output_dir / "assets")
        write_review_site(manifest, {}, metrics, self.output_dir)
        html = (self.output_dir / "index.html").read_text(encoding="utf-8")
        self.assertIn("sin preview", html)
        self.assertNotIn("<img", html)


class PrivacyTests(FaceReviewTestBase):
    def test_real_paths_never_appear_in_outputs(self):
        photo = _make_png(self.target / "private-photo.png")
        self._add_file(photo)
        self._add_member("P0001", photo, "image")
        self._add_observation("P0001", photo, "image", None)
        self.conn.commit()

        out = io.StringIO()
        with redirect_stdout(out):
            run_face_review(
                self.target,
                db_path=self.db_path,
                output_dir=self.output_dir,
                decisions_path=self.decisions_path,
                quiet=True,
            )

        for name in ("index.html", "manifest.json"):
            text = (self.output_dir / name).read_text(encoding="utf-8")
            self.assertNotIn(str(self.target), text)
            self.assertNotIn(str(photo), text)
        self.assertNotIn(str(self.target), out.getvalue())

        save_decisions(self.decisions_path, {"P0001": DECISION_CONFIRMED})
        self.assertNotIn(
            str(self.target), self.decisions_path.read_text(encoding="utf-8")
        )


class DecisionsTests(FaceReviewTestBase):
    def test_save_load_roundtrip_and_missing_file(self):
        self.assertEqual(load_decisions(self.decisions_path), {})
        decisions = {"P0002": DECISION_REJECTED, "P0001": DECISION_CONFIRMED}
        save_decisions(self.decisions_path, decisions)
        self.assertEqual(load_decisions(self.decisions_path), decisions)
        payload = json.loads(self.decisions_path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload, {"version": DECISIONS_VERSION, "decisions": decisions}
        )

    def test_parse_decisions_payload_reports_all_errors(self):
        with self.assertRaises(ValueError):
            parse_decisions_payload(["not", "an", "object"])
        with self.assertRaises(ValueError):
            parse_decisions_payload({"version": 2, "decisions": {}})
        with self.assertRaises(ValueError):
            parse_decisions_payload({"version": 1, "decisions": []})
        with self.assertRaises(ValueError) as context:
            parse_decisions_payload({"version": 2, "decisions": {"P0001": "maybe"}})
        message = str(context.exception)
        self.assertIn("version", message)
        self.assertIn("P0001", message)

    def test_load_invalid_file_raises_value_error(self):
        self.decisions_path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_decisions(self.decisions_path)

    def test_validate_unknown_labels(self):
        errors = validate_decisions(
            {"P0001": DECISION_CONFIRMED, "P9999": DECISION_REJECTED}, {"P0001"}
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("P9999", errors[0])


class ImportCliTests(FaceReviewTestBase):
    def _seed_labels(self):
        self._add_file(self.target / "a.png")
        self._add_file(self.target / "b.png")
        self._add_member("P0001", self.target / "a.png", "image")
        self._add_member("P0002", self.target / "b.png", "image")
        self.conn.commit()

    def _write_import(self, payload):
        path = self.root / "import.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_import_merges_over_existing(self):
        self._seed_labels()
        save_decisions(self.decisions_path, {"P0002": DECISION_UNCERTAIN})
        import_file = self._write_import(
            {"version": DECISIONS_VERSION, "decisions": {"P0001": DECISION_CONFIRMED}}
        )

        out = io.StringIO()
        with redirect_stdout(out):
            run_face_review_import(
                import_file,
                db_path=self.db_path,
                decisions_path=self.decisions_path,
            )
        printed = out.getvalue()
        self.assertIn("=== FACE REVIEW IMPORT ===", printed)
        self.assertIn("imported=1", printed)
        self.assertIn("confirmed=1", printed)
        self.assertIn("uncertain=1", printed)
        self.assertIn("total_reviewed=2", printed)
        self.assertIn("decisions_file=", printed)
        self.assertEqual(
            load_decisions(self.decisions_path),
            {"P0001": DECISION_CONFIRMED, "P0002": DECISION_UNCERTAIN},
        )

    def test_import_rejects_and_leaves_file_intact(self):
        self._seed_labels()
        save_decisions(self.decisions_path, {"P0001": DECISION_CONFIRMED})
        original = self.decisions_path.read_text(encoding="utf-8")

        cases = [
            ("{not json", "JSON"),
            (
                json.dumps({"version": 2, "decisions": {"P0001": "confirmed"}}),
                "version",
            ),
            (
                json.dumps({"version": 1, "decisions": {"P9999": "confirmed"}}),
                "P9999",
            ),
            (
                json.dumps({"version": 1, "decisions": {"P0001": "maybe"}}),
                "P0001",
            ),
        ]
        for index, (payload_text, token) in enumerate(cases):
            import_file = self.root / f"bad_import_{index}.json"
            import_file.write_text(payload_text, encoding="utf-8")
            stderr = io.StringIO()
            with self.assertRaises(SystemExit) as context:
                with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                    run_face_review_import(
                        import_file,
                        db_path=self.db_path,
                        decisions_path=self.decisions_path,
                    )
            self.assertEqual(context.exception.code, 1)
            self.assertIn(token, stderr.getvalue())
            self.assertEqual(
                self.decisions_path.read_text(encoding="utf-8"), original
            )


class MetricsTests(FaceReviewTestBase):
    def test_metrics_exact_keys_and_counts(self):
        manifest = {
            "clusters": [
                {"cluster_id": "P0001", "file_count": 1, "cross_media": False},
                {"cluster_id": "P0002", "file_count": 2, "cross_media": False},
                {"cluster_id": "P0003", "file_count": 2, "cross_media": True},
            ]
        }
        decisions = {"P0001": DECISION_CONFIRMED, "P0002": DECISION_REJECTED}
        self.assertEqual(
            compute_metrics(manifest, decisions),
            {
                "clusters_total": 3,
                "reviewed": 2,
                "confirmed": 1,
                "rejected": 1,
                "uncertain": 0,
                "unreviewed": 1,
                "singleton_clusters": 1,
                "multi_file_clusters": 2,
                "cross_media_clusters": 1,
            },
        )
        self.assertEqual(
            compute_metrics({"clusters": []}, {}),
            {
                "clusters_total": 0,
                "reviewed": 0,
                "confirmed": 0,
                "rejected": 0,
                "uncertain": 0,
                "unreviewed": 0,
                "singleton_clusters": 0,
                "multi_file_clusters": 0,
                "cross_media_clusters": 0,
            },
        )


class ReadOnlyTests(FaceReviewTestBase):
    def test_verify_readonly_and_output_inside_target_rejected(self):
        photo = _make_png(self.target / "photo.png")
        self._add_file(photo)
        self._add_member("P0001", photo, "image")
        self._add_observation("P0001", photo, "image", None, bbox=(5, 5, 30, 30))
        self.conn.commit()

        def _tree_state():
            return {
                path.relative_to(self.target).as_posix(): (
                    path.stat().st_size,
                    path.stat().st_mtime_ns,
                )
                for path in sorted(self.target.rglob("*"))
                if path.is_file()
            }

        before = _tree_state()
        out = io.StringIO()
        with redirect_stdout(out):
            run_face_review(
                self.target,
                db_path=self.db_path,
                output_dir=self.output_dir,
                decisions_path=self.decisions_path,
                quiet=True,
                verify_readonly=True,
                snapshot_dir=self.root / "snapshots",
            )
        self.assertEqual(_tree_state(), before)
        printed = out.getvalue()
        self.assertIn("=== VERIFICACIÓN READ-ONLY ===", printed)
        self.assertIn("=== FACE REVIEW ===", printed)

        with self.assertRaises(SystemExit) as context:
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                run_face_review(
                    self.target,
                    db_path=self.db_path,
                    output_dir=self.target / "review-inside",
                    decisions_path=self.decisions_path,
                    quiet=True,
                )
        self.assertEqual(context.exception.code, 1)
        self.assertFalse((self.target / "review-inside").exists())
        self.assertEqual(_tree_state(), before)


class FallbackTests(FaceReviewTestBase):
    def test_members_without_observations(self):
        self._add_file(self.target / "a.png")
        self._add_member("P0001", self.target / "a.png", "image")
        self.conn.commit()
        manifest, details = build_manifest(self.conn)
        self.assertEqual(len(manifest["clusters"]), 1)
        cluster = manifest["clusters"][0]
        self.assertEqual(cluster["samples"], [])
        self.assertEqual(cluster["observations"], 0)
        self.assertEqual(details, {})
        self.assertEqual(manifest["version"], MANIFEST_VERSION)

    def test_empty_catalog_builds_empty_manifest(self):
        manifest, details = build_manifest(self.conn)
        self.assertEqual(manifest["clusters"], [])
        self.assertEqual(details, {})

    def test_missing_observations_table(self):
        legacy_path = self.root / "legacy.sqlite"
        legacy = sqlite3.connect(str(legacy_path))
        legacy.row_factory = sqlite3.Row
        legacy.executescript(
            "CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT);"
            "CREATE TABLE person_cluster_members (id INTEGER PRIMARY KEY, "
            "person_label TEXT, file_path TEXT, file_category TEXT, "
            "observation_count INTEGER);"
        )
        legacy.execute(
            "INSERT INTO person_cluster_members "
            "(person_label, file_path, file_category, observation_count) "
            "VALUES ('P0001', '/x/a.png', 'image', 1)"
        )
        legacy.commit()
        manifest, details = build_manifest(legacy)
        self.assertEqual(len(manifest["clusters"]), 1)
        self.assertEqual(manifest["clusters"][0]["samples"], [])
        self.assertEqual(details, {})
        legacy.close()


class ClusterPeopleObservationTests(FaceReviewTestBase):
    @staticmethod
    def _row(x, y=5, w=30, h=30, confidence=0.9):
        full = [0.0] * 15
        full[0], full[1], full[2], full[3] = x, y, w, h
        full[14] = confidence
        return full

    @staticmethod
    def _embedding(index):
        import numpy as np

        vector = np.zeros(128, dtype=np.float32)
        vector[index] = 1.0
        return vector

    def test_cluster_people_persists_observation_map_and_rebuilds(self):
        import numpy as np

        from local_file_organizer import cli as cli_module
        from local_file_organizer.cli import run_cluster_people

        _make_png(self.target / "a.png", size=(64, 64))
        _make_png(self.target / "b.png", size=(64, 64))
        rows_by_name = {
            "a.png": np.array([self._row(10)], dtype=np.float32),
            "b.png": np.array([self._row(50)], dtype=np.float32),
        }

        def _fake_detect(path, **kwargs):
            return (64, 64, rows_by_name[Path(path).name], None)

        def _fake_compute(img, row):
            return self._embedding(0) if round(float(row[0])) == 10 else self._embedding(1)

        with (
            mock.patch.object(cli_module, "detect_faces_raw", side_effect=_fake_detect),
            mock.patch.object(
                cli_module, "compute_embedding", side_effect=_fake_compute
            ),
        ):
            out = io.StringIO()
            with redirect_stdout(out):
                run_cluster_people(self.target, db_path=self.db_path, quiet=True)
        self.assertIn("mapped_observations=2", out.getvalue())

        rows = database.fetch_person_cluster_observations(self.conn)
        self.assertEqual(len(rows), 2)
        by_label = {row["person_label"]: row for row in rows}
        self.assertEqual(set(by_label), {"P0001", "P0002"})
        self.assertEqual(by_label["P0001"]["file_path"], str(self.target / "a.png"))
        self.assertEqual(by_label["P0002"]["file_path"], str(self.target / "b.png"))
        for row in rows:
            self.assertEqual(row["source_type"], "image")
            self.assertIsNone(row["timestamp_ms"])
            self.assertAlmostEqual(row["detection_confidence"], 0.9)
            self.assertEqual(row["bbox_y"], 5)
            self.assertEqual(row["bbox_width"], 30)
            self.assertEqual(row["bbox_height"], 30)
        self.assertEqual(by_label["P0001"]["bbox_x"], 10)
        self.assertEqual(by_label["P0002"]["bbox_x"], 50)

        # A second run with no detected faces rebuilds (clears) the table.
        def _fake_detect_none(path, **kwargs):
            return (64, 64, None, None)

        with mock.patch.object(
            cli_module, "detect_faces_raw", side_effect=_fake_detect_none
        ):
            with redirect_stdout(io.StringIO()):
                run_cluster_people(self.target, db_path=self.db_path, quiet=True)
        self.assertEqual(database.fetch_person_cluster_observations(self.conn), [])

    def test_video_observations_persist_timestamp_and_bbox(self):
        import numpy as np
        from types import SimpleNamespace

        from local_file_organizer import cli as cli_module
        from local_file_organizer.cli import run_cluster_people

        _make_video_bytes(self.target / "clip.mp4")

        def _fake_detect(path, **kwargs):
            rows = np.array(
                [self._row(10, y=7, w=40, h=40, confidence=0.8)], dtype=np.float32
            )
            return (160, 120, rows, None)

        def _fake_sample(path, timestamps, on_frame=None, **kwargs):
            with tempfile.TemporaryDirectory() as tmp:
                for index, timestamp in enumerate(timestamps):
                    frame = Path(tmp) / f"frame_{index:04d}.png"
                    _make_png(frame, size=(160, 120))
                    if on_frame is not None:
                        on_frame(frame, timestamp, index)
            return SamplingResult(
                timestamps_requested=len(timestamps),
                frames_generated=len(timestamps),
                frames_failed=0,
                samples=[],
                video_error=None,
            )

        with (
            mock.patch.object(
                cli_module,
                "get_video_metadata",
                return_value=SimpleNamespace(error=None, duration_seconds=2.0),
            ),
            mock.patch.object(cli_module, "sample_video_frames", side_effect=_fake_sample),
            mock.patch.object(cli_module, "detect_faces_raw", side_effect=_fake_detect),
            mock.patch.object(
                cli_module, "compute_embedding", return_value=self._embedding(0)
            ),
        ):
            with redirect_stdout(io.StringIO()):
                run_cluster_people(self.target, db_path=self.db_path, quiet=True)

        rows = database.fetch_person_cluster_observations(self.conn)
        self.assertTrue(rows)
        self.assertTrue(all(row["source_type"] == "video_frame" for row in rows))
        self.assertTrue(all(row["timestamp_ms"] is not None for row in rows))
        self.assertTrue(all(row["bbox_y"] == 7 for row in rows))
        self.assertTrue(all(row["bbox_width"] == 40 for row in rows))
        self.assertAlmostEqual(rows[0]["detection_confidence"], 0.8)


class FaceReviewCliTests(FaceReviewTestBase):
    def test_face_review_and_import_end_to_end(self):
        from local_file_organizer.cli import main

        photo = _make_png(self.target / "photo.png")
        self._add_file(photo)
        self._add_member("P0001", photo, "image")
        self._add_observation("P0001", photo, "image", None, bbox=(5, 5, 30, 30))
        self.conn.commit()

        out = io.StringIO()
        with redirect_stdout(out):
            main(
                [
                    "face-review",
                    str(self.target),
                    "--db",
                    str(self.db_path),
                    "--output-dir",
                    str(self.output_dir),
                    "--decisions",
                    str(self.decisions_path),
                    "--quiet",
                ]
            )
        printed = out.getvalue()
        for token in (
            "=== FACE REVIEW ===",
            "clusters=1",
            "review_pages=1",
            "thumbnails=1",
            "assets_bytes=",
            "errors=0",
            "singleton_clusters=1",
            "multi_file_clusters=0",
            "cross_media_clusters=0",
            "reviewed=0",
            "confirmed=0",
            "rejected=0",
            "uncertain=0",
            "unreviewed=1",
            "output=",
            "manifest=",
            "elapsed_seconds=",
        ):
            self.assertIn(token, printed)
        self.assertTrue((self.output_dir / "index.html").exists())
        self.assertTrue((self.output_dir / "manifest.json").exists())
        self.assertTrue((self.output_dir / "assets" / "P0001_01.jpg").exists())

        import_file = self.root / "review_decisions.json"
        import_file.write_text(
            json.dumps({"version": 1, "decisions": {"P0001": "confirmed"}}),
            encoding="utf-8",
        )
        out_import = io.StringIO()
        with redirect_stdout(out_import):
            main(
                [
                    "face-review-import",
                    str(import_file),
                    "--db",
                    str(self.db_path),
                    "--decisions",
                    str(self.decisions_path),
                ]
            )
        printed_import = out_import.getvalue()
        self.assertIn("=== FACE REVIEW IMPORT ===", printed_import)
        self.assertIn("imported=1", printed_import)
        self.assertIn("total_reviewed=1", printed_import)
        self.assertEqual(
            load_decisions(self.decisions_path), {"P0001": DECISION_CONFIRMED}
        )

        out_second = io.StringIO()
        with redirect_stdout(out_second):
            main(
                [
                    "face-review",
                    str(self.target),
                    "--db",
                    str(self.db_path),
                    "--output-dir",
                    str(self.output_dir),
                    "--decisions",
                    str(self.decisions_path),
                    "--quiet",
                ]
            )
        printed_second = out_second.getvalue()
        self.assertIn("reviewed=1", printed_second)
        self.assertIn("confirmed=1", printed_second)
        self.assertIn("unreviewed=0", printed_second)


if __name__ == "__main__":
    unittest.main()
