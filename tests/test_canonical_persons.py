"""Tests for V3.1 canonical identities: resolution, merge/rename/assign/
unassign/split, evidence pooling and organize-preview compatibility.
Only touches temporary directories."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from local_file_organizer import database, person_identity
from local_file_organizer.cli import (
    run_person_assign,
    run_person_merge,
    run_person_rename,
    run_person_split,
    run_person_unassign,
    run_persons,
)
from local_file_organizer.organization_plan import (
    build_organization_plan,
    person_evidence,
)


def _seed(conn) -> None:
    """Catalog: 3 images in P0001, 2 images in P0002, 1 imageless video."""
    for path, category in (
        ("/a1.jpg", "image"),
        ("/a2.jpg", "image"),
        ("/a3.jpg", "image"),
        ("/b1.jpg", "image"),
        ("/b2.jpg", "image"),
        ("/v1.mp4", "video"),
    ):
        database.upsert_file(
            conn, database.FileRecord(path=path, category=category)
        )
    database.insert_person_cluster_members(
        conn,
        [
            ("P0001", "/a1.jpg", "image", 2),
            ("P0001", "/a2.jpg", "image", 1),
            ("P0001", "/a3.jpg", "image", 1),
            ("P0002", "/b1.jpg", "image", 3),
            ("P0002", "/b2.jpg", "image", 1),
        ],
    )
    conn.commit()


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.conn = database.connect(Path(self.tmp.name) / "t.sqlite")
        _seed(self.conn)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_unmapped_clusters_resolve_to_themselves(self):
        resolved = person_identity.resolve_file_persons(self.conn)
        self.assertEqual(resolved["/a1.jpg"], {"P0001"})
        self.assertEqual(resolved["/b1.jpg"], {"P0002"})
        self.assertNotIn("/v1.mp4", resolved)

    def test_empty_catalog_resolves_empty(self):
        with TemporaryDirectory() as tmp:
            conn = database.connect(Path(tmp) / "e.sqlite")
            self.assertEqual(person_identity.resolve_file_persons(conn), {})
            conn.close()


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.conn = database.connect(Path(self.tmp.name) / "t.sqlite")
        _seed(self.conn)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_merge_creates_canonical_and_aliases(self):
        target = person_identity.merge_labels(self.conn, ["P0001", "P0002"], "María")
        self.assertEqual(target, "PERSON_0001")
        self.conn.commit()
        resolved = person_identity.resolve_file_persons(self.conn)
        self.assertEqual(resolved["/a1.jpg"], {"PERSON_0001"})
        self.assertEqual(resolved["/b2.jpg"], {"PERSON_0001"})
        row = database.fetch_person(self.conn, "PERSON_0001")
        self.assertEqual(row["display_name"], "María")

    def test_merge_reuses_existing_canonical(self):
        first = person_identity.merge_labels(self.conn, ["P0001", "P0002"])
        second = person_identity.merge_labels(self.conn, ["P0002", first])
        self.assertEqual(first, second)

    def test_merge_needs_two_distinct_refs(self):
        with self.assertRaises(ValueError):
            person_identity.merge_labels(self.conn, ["P0001"])
        with self.assertRaises(ValueError):
            person_identity.merge_labels(self.conn, ["P0001", "P0001"])
        with self.assertRaises(ValueError):
            person_identity.merge_labels(self.conn, ["P0001", "P9999"])

    def test_evidence_pools_after_merge(self):
        person_identity.merge_labels(self.conn, ["P0001", "P0002"])
        evidence = person_evidence(self.conn)
        self.assertEqual(set(evidence), {"PERSON_0001"})
        self.assertEqual(evidence["PERSON_0001"]["files"], 5)
        self.assertEqual(evidence["PERSON_0001"]["observations"], 2 + 1 + 1 + 3 + 1)


class RenameTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.conn = database.connect(Path(self.tmp.name) / "t.sqlite")
        _seed(self.conn)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_rename_cluster_mints_canonical(self):
        target = person_identity.rename_person(self.conn, "P0001", "María")
        self.assertEqual(target, "PERSON_0001")
        self.conn.commit()
        self.assertEqual(
            person_identity.resolve_file_persons(self.conn)["/a1.jpg"],
            {"PERSON_0001"},
        )

    def test_rename_canonical_updates_name(self):
        target = person_identity.merge_labels(self.conn, ["P0001", "P0002"])
        person_identity.rename_person(self.conn, target, "Familia")
        self.assertEqual(
            database.fetch_person(self.conn, target)["display_name"], "Familia"
        )

    def test_rename_rejects_garbage(self):
        with self.assertRaises(ValueError):
            person_identity.rename_person(self.conn, "P0001", "   ")
        with self.assertRaises(ValueError):
            person_identity.rename_person(self.conn, "NADIE", "X")
        with self.assertRaises(ValueError):
            person_identity.rename_person(self.conn, "PERSON_0099", "X")


class AssignUnassignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.conn = database.connect(Path(self.tmp.name) / "t.sqlite")
        _seed(self.conn)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_assign_faceless_video(self):
        canonical = person_identity.assign_file(self.conn, "/v1.mp4", "P0001")
        self.assertEqual(canonical, "P0001")
        self.conn.commit()
        self.assertEqual(
            person_identity.resolve_file_persons(self.conn)["/v1.mp4"], {"P0001"}
        )

    def test_assign_rejects_unknowns(self):
        with self.assertRaises(ValueError):
            person_identity.assign_file(self.conn, "/missing.jpg", "P0001")
        with self.assertRaises(ValueError):
            person_identity.assign_file(self.conn, "/v1.mp4", "P9999")

    def test_unassign_manual_removes_link(self):
        person_identity.assign_file(self.conn, "/v1.mp4", "P0001")
        self.conn.commit()
        person_identity.unassign_file(self.conn, "/v1.mp4", "P0001")
        self.conn.commit()
        self.assertNotIn("/v1.mp4", person_identity.resolve_file_persons(self.conn))

    def test_unassign_auto_creates_exclusion_others_stay(self):
        person_identity.unassign_file(self.conn, "/a1.jpg", "P0001")
        self.conn.commit()
        resolved = person_identity.resolve_file_persons(self.conn)
        self.assertNotIn("/a1.jpg", resolved)
        self.assertEqual(resolved["/a2.jpg"], {"P0001"})

    def test_unassign_non_linked_fails(self):
        with self.assertRaises(ValueError):
            person_identity.unassign_file(self.conn, "/v1.mp4", "P0001")
        with self.assertRaises(ValueError):
            person_identity.unassign_file(self.conn, "/a1.jpg", "P0002")


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.conn = database.connect(Path(self.tmp.name) / "t.sqlite")
        _seed(self.conn)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_split_moves_files_to_new_person(self):
        new_id = person_identity.split_files(
            self.conn, "P0001", ["/a1.jpg"], "Sospechoso"
        )
        self.assertEqual(new_id, "PERSON_0001")
        self.conn.commit()
        resolved = person_identity.resolve_file_persons(self.conn)
        self.assertEqual(resolved["/a1.jpg"], {"PERSON_0001"})
        self.assertEqual(resolved["/a2.jpg"], {"P0001"})
        self.assertEqual(
            database.fetch_person(self.conn, new_id)["display_name"], "Sospechoso"
        )

    def test_split_rejects_foreign_files(self):
        with self.assertRaises(ValueError):
            person_identity.split_files(self.conn, "P0001", ["/b1.jpg"])
        with self.assertRaises(ValueError):
            person_identity.split_files(self.conn, "P0001", [])
        with self.assertRaises(ValueError):
            person_identity.split_files(self.conn, "PERSON_0001", ["/a1.jpg"])


class ReviewStatusTests(unittest.TestCase):
    def test_rejected_wins_over_confirmed(self):
        status = person_identity.combined_review_status(
            {"P0001": "confirmed", "P0002": "rejected"}, "PERSON_0001", {"P0001", "P0002"}
        )
        self.assertEqual(status, "rejected")

    def test_uncertain_wins_over_confirmed(self):
        status = person_identity.combined_review_status(
            {"P0001": "confirmed", "P0002": "uncertain"}, "PERSON_0001", {"P0001", "P0002"}
        )
        self.assertEqual(status, "uncertain")

    def test_canonical_decision_counts(self):
        status = person_identity.combined_review_status(
            {"PERSON_0001": "confirmed"}, "PERSON_0001", {"P0001"}
        )
        self.assertEqual(status, "confirmed")

    def test_no_decisions_is_none(self):
        self.assertIsNone(person_identity.combined_review_status({}, "P0001", set()))


class PreviewCompatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.conn = database.connect(Path(self.tmp.name) / "t.sqlite")
        _seed(self.conn)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _destinations(self):
        return {
            e["file_id"]: e["proposed_destination"]
            for e in build_organization_plan(self.conn)
        }

    def test_no_curation_uses_machine_labels(self):
        dests = self._destinations()
        self.assertTrue(
            any(d == "Media/Personas/P0001/" for d in dests.values()),
            dests,
        )

    def test_merge_reroutes_to_canonical(self):
        person_identity.merge_labels(self.conn, ["P0001", "P0002"], "María")
        dests = self._destinations()
        self.assertTrue(
            any(d == "Media/Personas/PERSON_0001/" for d in dests.values()),
            dests,
        )
        self.assertFalse(any("/P0001/" in d or "/P0002/" in d for d in dests.values()))

    def test_rejected_member_routes_canonical_to_review(self):
        person_identity.merge_labels(self.conn, ["P0001", "P0002"])
        entries = build_organization_plan(self.conn, decisions={"P0002": "rejected"})
        dests = {e["proposed_destination"] for e in entries}
        self.assertIn("Revisión/Cluster_rechazado/", dests)


class CliTests(unittest.TestCase):
    def test_cli_merge_assign_persons_roundtrip(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "c.sqlite"
            conn = database.connect(db)
            _seed(conn)
            conn.close()
            run_person_merge(["P0001", "P0002"], db_path=db, name="María")
            run_person_assign("/v1.mp4", "PERSON_0001", db_path=db)
            run_persons(db_path=db)  # must not raise
            run_person_rename("PERSON_0001", db_path=db, name="Familia")
            run_person_unassign("/v1.mp4", "PERSON_0001", db_path=db)
            run_person_split("P0001", ["/a1.jpg"], db_path=db, name="Otro")
            conn = database.connect(db)
            resolved = person_identity.resolve_file_persons(conn)
            conn.close()
        self.assertEqual(resolved["/a2.jpg"], {"PERSON_0001"})
        self.assertEqual(resolved["/a1.jpg"], {"PERSON_0002"})
        self.assertNotIn("/v1.mp4", resolved)

    def test_cli_errors_exit_1(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "c.sqlite"
            conn = database.connect(db)
            _seed(conn)
            conn.close()
            with self.assertRaises(SystemExit):
                run_person_merge(["P0001", "P9999"], db_path=db)
            with self.assertRaises(SystemExit):
                run_person_assign("/missing.jpg", "P0001", db_path=db)


if __name__ == "__main__":
    unittest.main()
