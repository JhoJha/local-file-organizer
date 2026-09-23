"""Unit tests for the V2.5 read-only organization preview.

All fixtures are hand-built SQLite rows via database helpers — no real
scanning, images, videos or models needed.
"""
import json
import tempfile
import unittest
from pathlib import Path

from local_file_organizer import database
from local_file_organizer.organization_plan import (
    build_organization_plan,
    plan_counts,
    write_plan_json,
)


def _record(path, category, extension, scan_status="ok", metadata_status="skipped",
            duplicate_group=None, similar_group=None):
    return database.FileRecord(
        path=path,
        name=Path(path).name,
        extension=extension,
        category=category,
        scan_status=scan_status,
        metadata_status=metadata_status,
        duplicate_group=duplicate_group,
        similar_group=similar_group,
    )


def _make_conn():
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    conn = database.connect(db_path)
    return conn, db_path


class OrganizationPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = Path(self.tmpdir.name) / "catalog.sqlite"
        self.conn = database.connect(self.db_path)
        self.addCleanup(self.conn.close)

    def _by_path(self, entries, paths):
        """Map fixture path -> entry via sorted-path file_id ordering."""
        sorted_paths = sorted(paths)
        id_to_entry = {e["file_id"]: e for e in entries}
        result = {}
        for i, p in enumerate(sorted_paths):
            result[p] = id_to_entry[f"F{i + 1:06d}"]
        return result

    def test_bucket_rules_and_destinations(self):
        base = "/data/user/fotos"
        paths = {
            "single": f"{base}/single.jpg",
            "pair": f"{base}/pair.jpg",
            "group": f"{base}/group.jpg",
            "zero": f"{base}/zero.jpg",
            "doc": "/data/user/docs/informe.pdf",
            "audio": "/data/user/musica/tema.wav",
            "archive": "/data/user/descargas/pack.zip",
            "software": "/data/user/descargas/app.exe",
            "other_ext": "/data/user/misc/notas.tmp",
            "other_noext": "/data/user/misc/README",
            "review": f"{base}/rota.jpg",
            "unsupported": f"{base}/ilegible.jpg",
            "video_zero": "/data/user/videos/clip.mp4",
        }
        fixtures = [
            _record(paths["single"], "image", ".jpg", metadata_status="ok"),
            _record(paths["pair"], "image", ".jpg", metadata_status="ok"),
            _record(paths["group"], "image", ".jpg", metadata_status="ok"),
            _record(paths["zero"], "image", ".jpg", metadata_status="ok"),
            _record(paths["doc"], "document", ".pdf"),
            _record(paths["audio"], "audio", ".wav"),
            _record(paths["archive"], "archive", ".zip"),
            _record(paths["software"], "software", ".exe"),
            _record(paths["other_ext"], "other", ".tmp"),
            _record(paths["other_noext"], "other", ""),
            _record(paths["review"], "image", ".jpg", metadata_status="error"),
            _record(paths["unsupported"], "image", ".jpg", scan_status="error"),
            _record(paths["video_zero"], "video", ".mp4", metadata_status="ok"),
        ]
        for rec in fixtures:
            database.upsert_file(self.conn, rec)
        database.insert_person_cluster_members(self.conn, [
            ("P0004", paths["single"], "image", 1),
            ("P0001", paths["pair"], "image", 1),
            ("P0002", paths["pair"], "image", 1),
            ("P0001", paths["group"], "image", 1),
            ("P0002", paths["group"], "image", 1),
            ("P0003", paths["group"], "image", 1),
        ])
        self.conn.commit()

        entries = build_organization_plan(self.conn)
        by_path = self._by_path(entries, list(paths.values()))

        self.assertEqual(by_path[paths["single"]]["bucket"], "planned")
        self.assertEqual(
            by_path[paths["single"]]["proposed_destination"], "Media/Personas/P0004/")
        self.assertEqual(
            by_path[paths["single"]]["reason_codes"], ["single_person:P0004"])
        self.assertEqual(by_path[paths["single"]]["confidence"], 0.9)

        self.assertEqual(by_path[paths["pair"]]["bucket"], "planned")
        self.assertTrue(
            by_path[paths["pair"]]["proposed_destination"].startswith("Media/Parejas/PAIR_"))
        self.assertEqual(by_path[paths["pair"]]["confidence"], 0.85)

        self.assertEqual(by_path[paths["group"]]["bucket"], "planned")
        self.assertTrue(
            by_path[paths["group"]]["proposed_destination"].startswith("Media/Grupos/GROUP_"))
        self.assertEqual(by_path[paths["group"]]["confidence"], 0.8)

        self.assertEqual(by_path[paths["zero"]]["proposed_destination"], "Media/Sin_personas/")
        self.assertEqual(by_path[paths["zero"]]["reason_codes"], ["no_person_detected"])
        self.assertEqual(by_path[paths["video_zero"]]["proposed_destination"], "Media/Sin_personas/")

        self.assertEqual(by_path[paths["doc"]]["proposed_destination"], "Documentos/PDF/")
        self.assertEqual(by_path[paths["doc"]]["reason_codes"], ["extension:pdf"])
        self.assertEqual(by_path[paths["doc"]]["confidence"], 1.0)

        self.assertEqual(by_path[paths["audio"]]["proposed_destination"], "Audio/WAV/")
        self.assertEqual(by_path[paths["audio"]]["reason_codes"], ["extension:wav"])

        self.assertEqual(
            by_path[paths["archive"]]["proposed_destination"], "Archivos_comprimidos/")
        self.assertEqual(
            by_path[paths["archive"]]["reason_codes"], ["extension:zip"])

        self.assertEqual(by_path[paths["software"]]["proposed_destination"], "Software/")
        self.assertEqual(by_path[paths["software"]]["reason_codes"], ["extension:exe"])

        self.assertEqual(by_path[paths["other_ext"]]["proposed_destination"], "Otros/TMP/")
        self.assertEqual(
            by_path[paths["other_ext"]]["reason_codes"], ["extension:unknown:tmp"])
        self.assertEqual(by_path[paths["other_ext"]]["confidence"], 0.7)

        self.assertEqual(
            by_path[paths["other_noext"]]["proposed_destination"], "Otros/Sin_extension/")
        self.assertEqual(by_path[paths["other_noext"]]["reason_codes"], ["no_extension"])

        self.assertEqual(by_path[paths["review"]]["bucket"], "review")
        self.assertEqual(
            by_path[paths["review"]]["proposed_destination"], "Revisión/Metadata_incompleta/")
        self.assertEqual(by_path[paths["review"]]["reason_codes"], ["metadata_error"])
        self.assertEqual(by_path[paths["review"]]["confidence"], 0.3)

        self.assertEqual(by_path[paths["unsupported"]]["bucket"], "unsupported")
        self.assertEqual(
            by_path[paths["unsupported"]]["proposed_destination"],
            "Revisión/Errores_de_escaneo/")
        self.assertEqual(by_path[paths["unsupported"]]["reason_codes"], ["scan_error"])
        self.assertEqual(by_path[paths["unsupported"]]["confidence"], 0.0)

        for entry in entries:
            self.assertEqual(entry["action"], "preview_only")

    def test_flags_attach_independently(self):
        p1 = "/data/user/a.jpg"
        p2 = "/data/user/b.pdf"
        database.upsert_file(
            self.conn, _record(p1, "image", ".jpg", metadata_status="ok",
                               duplicate_group="DUP_00001", similar_group="SIM_00001"))
        database.upsert_file(
            self.conn, _record(p2, "document", ".pdf", duplicate_group="DUP_00001"))
        self.conn.commit()
        entries = build_organization_plan(self.conn)
        by_id = {e["file_id"]: e for e in entries}
        # Sorted: /data/user/a.jpg -> F000001, /data/user/b.pdf -> F000002
        self.assertEqual(by_id["F000001"]["flags"], ["exact_duplicate", "visual_similar_group"])
        self.assertEqual(by_id["F000001"]["proposed_destination"], "Media/Sin_personas/")
        self.assertEqual(by_id["F000002"]["flags"], ["exact_duplicate"])
        self.assertEqual(by_id["F000002"]["proposed_destination"], "Documentos/PDF/")

    def test_accounting_invariant(self):
        files = [
            ("/data/u/a.jpg", "image", ".jpg", "ok", "ok"),
            ("/data/u/b.mp4", "video", ".mp4", "ok", "error"),
            ("/data/u/c.jpg", "image", ".jpg", "error", "skipped"),
            ("/data/u/d.pdf", "document", ".pdf", "ok", "skipped"),
            ("/data/u/e", "other", "", "ok", "skipped"),
        ]
        for path, cat, ext, ss, ms in files:
            database.upsert_file(
                self.conn, _record(path, cat, ext, scan_status=ss, metadata_status=ms))
        self.conn.commit()
        entries = build_organization_plan(self.conn)
        counts = plan_counts(entries)
        self.assertEqual(counts["catalog_files"], 5)
        self.assertEqual(
            counts["catalog_files"],
            counts["planned"] + counts["review"] + counts["unsupported"])
        self.assertEqual(counts["planned"], 3)
        self.assertEqual(counts["review"], 1)
        self.assertEqual(counts["unsupported"], 1)
        buckets = [e["bucket"] for e in entries]
        self.assertEqual(len(buckets), len(set(
            e["file_id"] for e in entries)))  # one entry per file, no dupes

    def test_pair_group_determinism_and_order(self):
        fa = "/data/u/pair_a.jpg"
        fb = "/data/u/pair_b.jpg"
        fc = "/data/u/group_c.jpg"
        for p in (fa, fb, fc):
            database.upsert_file(self.conn, _record(p, "image", ".jpg", metadata_status="ok"))
        database.insert_person_cluster_members(self.conn, [
            ("P0001", fa, "image", 1), ("P0002", fa, "image", 1),
            ("P0001", fb, "image", 1), ("P0003", fb, "image", 1),
            ("P0001", fc, "image", 1), ("P0002", fc, "image", 1),
            ("P0003", fc, "image", 1),
        ])
        self.conn.commit()
        first = build_organization_plan(self.conn)
        second = build_organization_plan(self.conn)
        self.assertEqual(first, second)
        by_path = self._by_path(first, [fa, fb, fc])
        dest_a = by_path[fa]["proposed_destination"]
        dest_b = by_path[fb]["proposed_destination"]
        dest_c = by_path[fc]["proposed_destination"]
        # ("P0001","P0002") sorts before ("P0001","P0003")
        self.assertIn("PAIR_0001", dest_a)
        self.assertIn("PAIR_0002", dest_b)
        self.assertNotEqual(dest_a, dest_b)
        self.assertEqual(by_path[fa]["reason_codes"], ["pair:PAIR_0001"])
        self.assertEqual(by_path[fb]["reason_codes"], ["pair:PAIR_0002"])
        self.assertIn("GROUP_0001", dest_c)

    def test_file_id_determinism_sorted_by_path(self):
        for p in ("/data/u/c.jpg", "/data/u/a.jpg", "/data/u/b.jpg"):
            database.upsert_file(self.conn, _record(p, "image", ".jpg", metadata_status="ok"))
        self.conn.commit()
        first = build_organization_plan(self.conn)
        second = build_organization_plan(self.conn)
        self.assertEqual(
            [e["file_id"] for e in first], ["F000001", "F000002", "F000003"])
        self.assertEqual(first, second)
        # a.jpg < b.jpg < c.jpg so F000001 must be the 'other'/media entry
        # corresponding to a.jpg — verify via destination stability across runs
        # and, more directly, that re-running after no changes is identical.

    def test_json_contains_no_real_paths(self):
        secret_paths = ["/data/personal/secreto_a.jpg", "/data/personal/secreto_b.pdf"]
        database.upsert_file(
            self.conn, _record(secret_paths[0], "image", ".jpg", metadata_status="ok"))
        database.upsert_file(self.conn, _record(secret_paths[1], "document", ".pdf"))
        self.conn.commit()
        entries = build_organization_plan(self.conn)
        plan_path = Path(self.tmpdir.name) / "organization_plan.json"
        write_plan_json(entries, plan_path)
        serialized = plan_path.read_text(encoding="utf-8")
        for secret in secret_paths:
            self.assertNotIn(secret, serialized)
        # Schema check
        data = json.loads(serialized)
        self.assertEqual(len(data), 2)
        for entry in data:
            self.assertEqual(
                set(entry.keys()),
                {"file_id", "category", "bucket", "proposed_destination",
                 "reason_codes", "flags", "confidence", "action"})
            self.assertNotIn("path", entry)


if __name__ == "__main__":
    unittest.main()
