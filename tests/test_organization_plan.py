"""Unit tests for the V2.5.1 read-only organization preview with confidence.

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
    person_confidence,
    person_evidence,
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


class PersonEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = Path(self.tmpdir.name) / "catalog.sqlite"
        self.conn = database.connect(self.db_path)
        self.addCleanup(self.conn.close)

    def test_aggregates_files_media_and_observations(self):
        img1, img2, vid = "/data/u/a.jpg", "/data/u/b.jpg", "/data/u/c.mp4"
        for path, cat in ((img1, "image"), (img2, "image"), (vid, "video")):
            database.upsert_file(
                self.conn, _record(path, cat, Path(path).suffix, metadata_status="ok")
            )
        database.insert_person_cluster_members(self.conn, [
            ("P0001", img1, "image", 2),
            ("P0001", img2, "image", 1),
            ("P0001", vid, "video", 3),
            ("P0002", img1, "image", 5),
        ])
        self.conn.commit()

        evidence = person_evidence(self.conn)
        self.assertEqual(evidence["P0001"], {
            "files": 3, "images": 2, "videos": 1,
            "observations": 6, "cross_media": True,
        })
        self.assertEqual(evidence["P0002"], {
            "files": 1, "images": 1, "videos": 0,
            "observations": 5, "cross_media": False,
        })

    def test_missing_table_returns_empty_mapping(self):
        conn = database.connect(Path(self.tmpdir.name) / "empty.sqlite")
        self.addCleanup(conn.close)
        conn.execute("DROP TABLE IF EXISTS person_cluster_members")
        self.assertEqual(person_evidence(conn), {})


class PersonConfidenceTests(unittest.TestCase):
    def _evidence(self, files, observations):
        return {
            "files": files, "images": files, "videos": 0,
            "observations": observations, "cross_media": False,
        }

    def test_human_decisions_override_heuristics(self):
        weak = self._evidence(1, 1)
        strong = self._evidence(10, 50)
        self.assertEqual(person_confidence(weak, "confirmed"), "high")
        self.assertEqual(person_confidence(strong, "rejected"), "low")
        self.assertEqual(person_confidence(strong, "uncertain"), "low")

    def test_evidence_thresholds_without_decisions(self):
        self.assertEqual(person_confidence(self._evidence(3, 3), None), "high")
        self.assertEqual(person_confidence(self._evidence(3, 2), "medium"), "medium")
        self.assertEqual(person_confidence(self._evidence(2, 10), None), "medium")
        self.assertEqual(person_confidence(self._evidence(1, 100), None), "low")


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

    def _add_images(self, paths, metadata_status="ok"):
        for p in paths:
            database.upsert_file(
                self.conn,
                _record(p, "image", ".jpg", metadata_status=metadata_status),
            )

    def _add_members(self, rows):
        database.insert_person_cluster_members(self.conn, rows)
        self.conn.commit()

    def _build(self, paths, decisions=None):
        entries = build_organization_plan(self.conn, decisions=decisions)
        return entries, self._by_path(entries, paths)

    # -- single-person routing ---------------------------------------------

    def test_singleton_unreviewed_routes_to_low_evidence(self):
        p = "/data/u/sola.jpg"
        self._add_images([p])
        self._add_members([("P0001", p, "image", 1)])

        entries, by_path = self._build([p])
        entry = by_path[p]
        self.assertEqual(entry["bucket"], "review")
        self.assertEqual(entry["proposed_destination"], "Revisión/Baja_evidencia/")
        self.assertEqual(
            entry["reason_codes"],
            ["singleton_person_cluster", "unreviewed_cluster"],
        )
        self.assertEqual(entry["confidence_level"], "low")
        self.assertEqual(entry["flags"], [])
        self.assertEqual(entry["action"], "preview_only")

    def test_singleton_confirmed_routes_to_person_high(self):
        p = "/data/u/confirmada.jpg"
        self._add_images([p])
        self._add_members([("P0001", p, "image", 1)])

        entries, by_path = self._build([p], decisions={"P0001": "confirmed"})
        entry = by_path[p]
        self.assertEqual(entry["bucket"], "planned")
        self.assertEqual(entry["proposed_destination"], "Media/Personas/P0001/")
        self.assertEqual(
            entry["reason_codes"],
            ["single_person:P0001", "confirmed_person_cluster"],
        )
        self.assertEqual(entry["confidence_level"], "high")

    def test_multi_file_confirmed_overrides_heuristic_to_high(self):
        p1, p2 = "/data/u/m1.jpg", "/data/u/m2.jpg"
        self._add_images([p1, p2])
        self._add_members([
            ("P0002", p1, "image", 1),
            ("P0002", p2, "image", 1),
        ])

        _entries, by_path = self._build(
            [p1, p2], decisions={"P0002": "confirmed"}
        )
        for p in (p1, p2):
            self.assertEqual(by_path[p]["bucket"], "planned")
            self.assertEqual(by_path[p]["proposed_destination"], "Media/Personas/P0002/")
            self.assertEqual(by_path[p]["confidence_level"], "high")
            self.assertIn("confirmed_person_cluster", by_path[p]["reason_codes"])

    def test_evidence_thresholds_without_decisions(self):
        high1, high2, high3 = (
            "/data/u/high1.jpg", "/data/u/high2.jpg", "/data/u/high3.jpg"
        )
        med1, med2 = "/data/u/med1.jpg", "/data/u/med2.jpg"
        weak = "/data/u/weak.jpg"
        self._add_images([high1, high2, high3, med1, med2, weak])
        self._add_members([
            ("P_HIGH", high1, "image", 1),
            ("P_HIGH", high2, "image", 1),
            ("P_HIGH", high3, "image", 1),
            ("P_MED", med1, "image", 1),
            ("P_MED", med2, "image", 1),
            ("P_LOW", weak, "image", 9),
        ])

        _entries, by_path = self._build(
            [high1, high2, high3, med1, med2, weak]
        )
        for p in (high1, high2, high3):
            self.assertEqual(by_path[p]["proposed_destination"], "Media/Personas/P_HIGH/")
            self.assertEqual(by_path[p]["confidence_level"], "high")
        for p in (med1, med2):
            self.assertEqual(by_path[p]["proposed_destination"], "Media/Personas/P_MED/")
            self.assertEqual(by_path[p]["confidence_level"], "medium")
        self.assertEqual(by_path[weak]["bucket"], "review")
        self.assertEqual(
            by_path[weak]["proposed_destination"], "Revisión/Baja_evidencia/"
        )
        self.assertEqual(by_path[weak]["confidence_level"], "low")

    def test_uncertain_cluster_routes_to_persona_incierta(self):
        p1, p2, p3 = "/data/u/u1.jpg", "/data/u/u2.jpg", "/data/u/u3.jpg"
        self._add_images([p1, p2, p3])
        self._add_members([
            ("P_UNC", p1, "image", 1),
            ("P_UNC", p2, "image", 1),
            ("P_UNC", p3, "image", 1),
        ])

        _entries, by_path = self._build(
            [p1, p2, p3], decisions={"P_UNC": "uncertain"}
        )
        for p in (p1, p2, p3):
            self.assertEqual(by_path[p]["bucket"], "review")
            self.assertEqual(
                by_path[p]["proposed_destination"], "Revisión/Persona_incierta/"
            )
            self.assertEqual(
                by_path[p]["reason_codes"], ["uncertain_person_cluster"]
            )
            self.assertEqual(by_path[p]["confidence_level"], "low")

    def test_rejected_cluster_routes_to_cluster_rechazado(self):
        p1, p2, p3 = "/data/u/r1.jpg", "/data/u/r2.jpg", "/data/u/r3.jpg"
        self._add_images([p1, p2, p3])
        self._add_members([
            ("P_REJ", p1, "image", 1),
            ("P_REJ", p2, "image", 1),
            ("P_REJ", p3, "image", 1),
        ])

        _entries, by_path = self._build(
            [p1, p2, p3], decisions={"P_REJ": "rejected"}
        )
        for p in (p1, p2, p3):
            self.assertEqual(by_path[p]["bucket"], "review")
            self.assertEqual(
                by_path[p]["proposed_destination"], "Revisión/Cluster_rechazado/"
            )
            self.assertEqual(
                by_path[p]["reason_codes"], ["rejected_person_cluster"]
            )
            self.assertEqual(by_path[p]["confidence_level"], "low")

    # -- pair routing ------------------------------------------------------

    def test_pair_with_uncertain_member_is_review(self):
        p = "/data/u/pair_unc.jpg"
        self._add_images([p])
        self._add_members([("P1", p, "image", 1), ("P2", p, "image", 1)])

        _entries, by_path = self._build([p], decisions={"P2": "uncertain"})
        entry = by_path[p]
        self.assertEqual(entry["bucket"], "review")
        self.assertEqual(entry["proposed_destination"], "Revisión/Persona_incierta/")
        self.assertEqual(
            entry["reason_codes"], ["pair_contains_uncertain_person"]
        )
        self.assertEqual(entry["confidence_level"], "low")

    def test_pair_with_rejected_member_takes_precedence(self):
        p = "/data/u/pair_rej.jpg"
        self._add_images([p])
        self._add_members([("P1", p, "image", 1), ("P2", p, "image", 1)])

        _entries, by_path = self._build(
            [p], decisions={"P1": "rejected", "P2": "uncertain"}
        )
        entry = by_path[p]
        self.assertEqual(entry["bucket"], "review")
        self.assertEqual(
            entry["proposed_destination"], "Revisión/Cluster_rechazado/"
        )
        self.assertEqual(
            entry["reason_codes"], ["pair_contains_rejected_person"]
        )

    def test_pair_with_low_evidence_member_is_review(self):
        p = "/data/u/pair_low.jpg"
        self._add_images([p])
        self._add_members([("P1", p, "image", 1), ("P2", p, "image", 1)])

        _entries, by_path = self._build([p])
        entry = by_path[p]
        self.assertEqual(entry["bucket"], "review")
        self.assertEqual(entry["proposed_destination"], "Revisión/Persona_incierta/")
        self.assertEqual(
            entry["reason_codes"], ["pair_contains_low_evidence_person"]
        )

    def test_pair_all_confirmed_is_planned_high(self):
        p = "/data/u/pair_ok.jpg"
        self._add_images([p])
        self._add_members([("P1", p, "image", 1), ("P2", p, "image", 1)])

        _entries, by_path = self._build(
            [p], decisions={"P1": "confirmed", "P2": "confirmed"}
        )
        entry = by_path[p]
        self.assertEqual(entry["bucket"], "planned")
        self.assertEqual(entry["proposed_destination"], "Media/Parejas/PAIR_0001/")
        self.assertEqual(
            entry["reason_codes"],
            ["pair:PAIR_0001", "confirmed_person_cluster"],
        )
        self.assertEqual(entry["confidence_level"], "high")

    def test_pair_confirmed_plus_medium_is_planned_medium(self):
        p1, p2 = "/data/u/pair_a.jpg", "/data/u/pair_b.jpg"
        self._add_images([p1, p2])
        self._add_members([
            ("P1", p1, "image", 1), ("P2", p1, "image", 1),
            ("P2", p2, "image", 1),
        ])

        _entries, by_path = self._build([p1, p2], decisions={"P1": "confirmed"})
        entry = by_path[p1]
        self.assertEqual(entry["bucket"], "planned")
        self.assertEqual(entry["proposed_destination"], "Media/Parejas/PAIR_0001/")
        self.assertEqual(entry["reason_codes"], ["pair:PAIR_0001"])
        self.assertEqual(entry["confidence_level"], "medium")

    # -- group routing -----------------------------------------------------

    def test_group_with_uncertain_member_is_review(self):
        p = "/data/u/group_unc.jpg"
        self._add_images([p])
        self._add_members([
            ("P1", p, "image", 1), ("P2", p, "image", 1), ("P3", p, "image", 1),
        ])

        _entries, by_path = self._build([p], decisions={"P2": "uncertain"})
        entry = by_path[p]
        self.assertEqual(entry["bucket"], "review")
        self.assertEqual(entry["proposed_destination"], "Revisión/Persona_incierta/")
        self.assertEqual(
            entry["reason_codes"], ["group_contains_uncertain_person"]
        )
        self.assertEqual(entry["confidence_level"], "low")

    def test_group_with_rejected_member_is_review(self):
        p = "/data/u/group_rej.jpg"
        self._add_images([p])
        self._add_members([
            ("P1", p, "image", 1), ("P2", p, "image", 1), ("P3", p, "image", 1),
        ])

        _entries, by_path = self._build([p], decisions={"P3": "rejected"})
        entry = by_path[p]
        self.assertEqual(entry["bucket"], "review")
        self.assertEqual(
            entry["proposed_destination"], "Revisión/Cluster_rechazado/"
        )
        self.assertEqual(
            entry["reason_codes"], ["group_contains_rejected_person"]
        )

    def test_group_with_low_evidence_member_is_review(self):
        p = "/data/u/group_low.jpg"
        self._add_images([p])
        self._add_members([
            ("P1", p, "image", 1), ("P2", p, "image", 1), ("P3", p, "image", 1),
        ])

        _entries, by_path = self._build([p])
        entry = by_path[p]
        self.assertEqual(entry["bucket"], "review")
        self.assertEqual(entry["proposed_destination"], "Revisión/Persona_incierta/")
        self.assertEqual(
            entry["reason_codes"], ["group_contains_low_evidence_person"]
        )

    def test_group_all_confirmed_is_planned_high(self):
        p = "/data/u/group_ok.jpg"
        self._add_images([p])
        self._add_members([
            ("P1", p, "image", 1), ("P2", p, "image", 1), ("P3", p, "image", 1),
        ])

        _entries, by_path = self._build(
            [p],
            decisions={"P1": "confirmed", "P2": "confirmed", "P3": "confirmed"},
        )
        entry = by_path[p]
        self.assertEqual(entry["bucket"], "planned")
        self.assertEqual(entry["proposed_destination"], "Media/Grupos/GROUP_0001/")
        self.assertEqual(
            entry["reason_codes"],
            ["group:GROUP_0001", "confirmed_person_cluster"],
        )
        self.assertEqual(entry["confidence_level"], "high")

    def test_group_high_evidence_unreviewed_is_planned(self):
        paths = ["/data/u/g1.jpg", "/data/u/g2.jpg", "/data/u/g3.jpg"]
        self._add_images(paths)
        members = []
        for p in paths:
            for label in ("P1", "P2", "P3"):
                members.append((label, p, "image", 1))
        self._add_members(members)

        _entries, by_path = self._build(paths)
        for p in paths:
            self.assertEqual(by_path[p]["bucket"], "planned")
            self.assertEqual(
                by_path[p]["proposed_destination"], "Media/Grupos/GROUP_0001/"
            )
            self.assertEqual(by_path[p]["reason_codes"], ["group:GROUP_0001"])
            self.assertEqual(by_path[p]["confidence_level"], "high")

    # -- no-person and non-media routing -----------------------------------

    def test_no_person_media_is_high(self):
        img, vid = "/data/u/libre.jpg", "/data/u/libre.mp4"
        database.upsert_file(
            self.conn, _record(img, "image", ".jpg", metadata_status="ok"))
        database.upsert_file(
            self.conn, _record(vid, "video", ".mp4", metadata_status="ok"))
        self.conn.commit()

        _entries, by_path = self._build([img, vid])
        for p in (img, vid):
            self.assertEqual(by_path[p]["bucket"], "planned")
            self.assertEqual(
                by_path[p]["proposed_destination"], "Media/Sin_personas/"
            )
            self.assertEqual(by_path[p]["reason_codes"], ["no_person_detected"])
            self.assertEqual(by_path[p]["confidence_level"], "high")

    def test_non_media_destinations_and_confidence(self):
        doc = "/data/u/informe.pdf"
        audio = "/data/u/tema.wav"
        archive = "/data/u/pack.zip"
        software = "/data/u/app.exe"
        other_ext = "/data/u/notas.tmp"
        other_noext = "/data/u/README"
        fixtures = [
            _record(doc, "document", ".pdf"),
            _record(audio, "audio", ".wav"),
            _record(archive, "archive", ".zip"),
            _record(software, "software", ".exe"),
            _record(other_ext, "other", ".tmp"),
            _record(other_noext, "other", ""),
        ]
        for rec in fixtures:
            database.upsert_file(self.conn, rec)
        self.conn.commit()

        _entries, by_path = self._build(
            [doc, audio, archive, software, other_ext, other_noext]
        )
        self.assertEqual(by_path[doc]["proposed_destination"], "Documentos/PDF/")
        self.assertEqual(by_path[doc]["confidence_level"], "high")
        self.assertEqual(by_path[audio]["proposed_destination"], "Audio/WAV/")
        self.assertEqual(by_path[audio]["confidence_level"], "high")
        self.assertEqual(
            by_path[archive]["proposed_destination"], "Archivos_comprimidos/")
        self.assertEqual(by_path[archive]["confidence_level"], "high")
        self.assertEqual(by_path[software]["proposed_destination"], "Software/")
        self.assertEqual(by_path[software]["confidence_level"], "high")
        self.assertEqual(by_path[other_ext]["proposed_destination"], "Otros/TMP/")
        self.assertEqual(
            by_path[other_ext]["reason_codes"], ["extension:unknown:tmp"])
        self.assertEqual(by_path[other_ext]["confidence_level"], "medium")
        self.assertEqual(
            by_path[other_noext]["proposed_destination"], "Otros/Sin_extension/")
        self.assertEqual(by_path[other_noext]["reason_codes"], ["no_extension"])
        self.assertEqual(by_path[other_noext]["confidence_level"], "medium")

    def test_scan_and_metadata_errors_are_low(self):
        rota = "/data/u/rota.jpg"
        ilegible = "/data/u/ilegible.jpg"
        database.upsert_file(
            self.conn, _record(rota, "image", ".jpg", metadata_status="error"))
        database.upsert_file(
            self.conn, _record(ilegible, "image", ".jpg", scan_status="error"))
        self.conn.commit()

        _entries, by_path = self._build([rota, ilegible])
        self.assertEqual(by_path[rota]["bucket"], "review")
        self.assertEqual(
            by_path[rota]["proposed_destination"], "Revisión/Metadata_incompleta/")
        self.assertEqual(by_path[rota]["reason_codes"], ["metadata_error"])
        self.assertEqual(by_path[rota]["confidence_level"], "low")
        self.assertEqual(by_path[ilegible]["bucket"], "unsupported")
        self.assertEqual(
            by_path[ilegible]["proposed_destination"],
            "Revisión/Errores_de_escaneo/")
        self.assertEqual(by_path[ilegible]["reason_codes"], ["scan_error"])
        self.assertEqual(by_path[ilegible]["confidence_level"], "low")

    # -- flags, accounting, determinism, JSON ------------------------------

    def test_flags_attach_independently(self):
        p1 = "/data/u/a.jpg"
        p2 = "/data/u/b.pdf"
        database.upsert_file(
            self.conn, _record(p1, "image", ".jpg", metadata_status="ok",
                               duplicate_group="DUP_00001", similar_group="SIM_00001"))
        database.upsert_file(
            self.conn, _record(p2, "document", ".pdf", duplicate_group="DUP_00001"))
        self.conn.commit()
        entries = build_organization_plan(self.conn)
        by_id = {e["file_id"]: e for e in entries}
        # Sorted: /data/u/a.jpg -> F000001, /data/u/b.pdf -> F000002
        self.assertEqual(
            by_id["F000001"]["flags"],
            ["exact_duplicate", "visual_similar_group"],
        )
        self.assertEqual(by_id["F000001"]["proposed_destination"], "Media/Sin_personas/")
        self.assertEqual(by_id["F000001"]["confidence_level"], "high")
        self.assertEqual(by_id["F000002"]["flags"], ["exact_duplicate"])
        self.assertEqual(by_id["F000002"]["proposed_destination"], "Documentos/PDF/")
        self.assertEqual(by_id["F000002"]["confidence_level"], "high")

    def test_accounting_invariant_with_review(self):
        files = [
            ("/data/u/a.jpg", "image", ".jpg", "ok", "ok"),
            ("/data/u/b.mp4", "video", ".mp4", "ok", "error"),
            ("/data/u/c.jpg", "image", ".jpg", "error", "skipped"),
            ("/data/u/d.pdf", "document", ".pdf", "ok", "skipped"),
            ("/data/u/e.jpg", "image", ".jpg", "ok", "ok"),
            ("/data/u/f1.jpg", "image", ".jpg", "ok", "ok"),
            ("/data/u/f2.jpg", "image", ".jpg", "ok", "ok"),
            ("/data/u/f3.jpg", "image", ".jpg", "ok", "ok"),
            ("/data/u/g.jpg", "image", ".jpg", "ok", "ok"),
        ]
        for path, cat, ext, ss, ms in files:
            database.upsert_file(
                self.conn, _record(path, cat, ext, scan_status=ss, metadata_status=ms))
        self._add_members([
            ("P_SINGLE", "/data/u/a.jpg", "image", 1),
            ("P_STRONG", "/data/u/f1.jpg", "image", 1),
            ("P_STRONG", "/data/u/f2.jpg", "image", 1),
            ("P_STRONG", "/data/u/f3.jpg", "image", 1),
            ("P_REJ", "/data/u/g.jpg", "image", 1),
        ])
        entries = build_organization_plan(
            self.conn, decisions={"P_REJ": "rejected"}
        )
        counts = plan_counts(entries)
        self.assertEqual(counts["catalog_files"], 9)
        self.assertEqual(
            counts["catalog_files"],
            counts["planned"] + counts["review"] + counts["unsupported"],
        )
        self.assertEqual(counts["planned"], 5)
        self.assertEqual(counts["review"], 3)
        self.assertEqual(counts["unsupported"], 1)
        levels = counts["confidence_levels"]
        self.assertEqual(sum(levels.values()), counts["catalog_files"])
        self.assertEqual(levels, {"high": 5, "medium": 0, "low": 4})
        self.assertEqual(len({e["file_id"] for e in entries}), len(entries))

    def test_pair_group_determinism_and_order(self):
        fa = "/data/u/pair_a.jpg"
        fb = "/data/u/pair_b.jpg"
        fc = "/data/u/group_c.jpg"
        self._add_images([fa, fb, fc])
        self._add_members([
            ("P0001", fa, "image", 1), ("P0002", fa, "image", 1),
            ("P0001", fb, "image", 1), ("P0003", fb, "image", 1),
            ("P0001", fc, "image", 1), ("P0002", fc, "image", 1),
            ("P0003", fc, "image", 1),
        ])
        decisions = {
            "P0001": "confirmed", "P0002": "confirmed", "P0003": "confirmed",
        }
        first = build_organization_plan(self.conn, decisions=decisions)
        second = build_organization_plan(self.conn, decisions=decisions)
        self.assertEqual(first, second)
        by_path = self._by_path(first, [fa, fb, fc])
        dest_a = by_path[fa]["proposed_destination"]
        dest_b = by_path[fb]["proposed_destination"]
        dest_c = by_path[fc]["proposed_destination"]
        # ("P0001","P0002") sorts before ("P0001","P0003")
        self.assertIn("PAIR_0001", dest_a)
        self.assertIn("PAIR_0002", dest_b)
        self.assertNotEqual(dest_a, dest_b)
        self.assertEqual(by_path[fa]["reason_codes"], ["pair:PAIR_0001", "confirmed_person_cluster"])
        self.assertEqual(by_path[fb]["reason_codes"], ["pair:PAIR_0002", "confirmed_person_cluster"])
        self.assertIn("GROUP_0001", dest_c)
        self.assertEqual(
            by_path[fc]["reason_codes"],
            ["group:GROUP_0001", "confirmed_person_cluster"],
        )
        # Rebuilding without decisions keeps the same order/counts, only
        # the routing changes (all labels are still singletons here).
        third = build_organization_plan(self.conn)
        self.assertEqual(len(third), 3)
        self.assertEqual(
            [e["file_id"] for e in third], [e["file_id"] for e in first]
        )

    def test_file_id_determinism_sorted_by_path(self):
        for p in ("/data/u/c.jpg", "/data/u/a.jpg", "/data/u/b.jpg"):
            database.upsert_file(
                self.conn, _record(p, "image", ".jpg", metadata_status="ok"))
        self.conn.commit()
        first = build_organization_plan(self.conn)
        second = build_organization_plan(self.conn)
        self.assertEqual(
            [e["file_id"] for e in first], ["F000001", "F000002", "F000003"])
        self.assertEqual(first, second)

    def test_json_contains_no_real_paths_and_exact_schema(self):
        secret_paths = [
            "/data/personal/secreto_a.jpg",
            "/data/personal/secreto_b.pdf",
        ]
        database.upsert_file(
            self.conn,
            _record(secret_paths[0], "image", ".jpg", metadata_status="ok"))
        database.upsert_file(
            self.conn, _record(secret_paths[1], "document", ".pdf"))
        self.conn.commit()
        entries = build_organization_plan(self.conn)
        plan_path = Path(self.tmpdir.name) / "organization_plan.json"
        write_plan_json(entries, plan_path)
        serialized = plan_path.read_text(encoding="utf-8")
        for secret in secret_paths:
            self.assertNotIn(secret, serialized)
        data = json.loads(serialized)
        self.assertEqual(len(data), 2)
        for entry in data:
            self.assertEqual(
                set(entry.keys()),
                {"file_id", "category", "bucket", "proposed_destination",
                 "reason_codes", "flags", "confidence_level", "action"})
            self.assertNotIn("path", entry)
            self.assertNotIn("confidence", entry)
            self.assertEqual(entry["action"], "preview_only")


if __name__ == "__main__":
    unittest.main()
