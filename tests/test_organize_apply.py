"""Tests for organize-apply dry-run/rollback (temp dirs only, synthetic trees)."""
import hashlib
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from local_file_organizer import database
from local_file_organizer import organize_apply as apply_mod
from local_file_organizer.cli import run_organize_apply
from local_file_organizer.organization_preview import generate_organization_preview


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class ApplyTree:
    """A synthetic target tree plus a catalog and plan describing it."""

    FILES = (
        # (relative name, category, scan_status, metadata_status, content)
        ("broken.bin", "other", "error", "skipped", b"broken-bytes"),
        ("corrupt.jpg", "image", "ok", "error", b"corrupt-image-bytes"),
        ("doc.pdf", "document", "ok", "skipped", b"pdf-bytes"),
        ("photo.jpg", "image", "ok", "ok", b"photo-bytes"),
        ("song.mp3", "audio", "ok", "skipped", b"mp3-bytes"),
    )

    def __init__(self, base: Path) -> None:
        self.base = base
        self.target = base / "target"
        self.db_path = base / "catalog.sqlite"
        self.plan_path = base / "reports" / "organization_plan.json"
        self.snapshot_dir = base / "snapshots"
        self.manifest_path = base / "reports" / "manifest.json"
        for name, _cat, _scan, _meta, content in self.FILES:
            _write(self.target / name, content)
        conn = database.connect(self.db_path)
        try:
            for name, cat, scan, meta, _content in self.FILES:
                ext = Path(name).suffix
                database.upsert_file(
                    conn,
                    database.FileRecord(
                        path=str(self.target / name),
                        name=name,
                        extension=ext,
                        category=cat,
                        scan_status=scan,
                        metadata_status=meta,
                        file_state="active",
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        generate_organization_preview(
            db_path=self.db_path,
            plan_path=self.plan_path,
            write_view=False,
        )
        self.entries = json.loads(self.plan_path.read_text(encoding="utf-8"))

    def contents(self) -> dict[str, bytes]:
        return {
            str(p.relative_to(self.target)): p.read_bytes()
            for p in sorted(self.target.rglob("*"))
            if p.is_file()
        }

    def planned_ids(self) -> list[str]:
        return [e["file_id"] for e in self.entries if e["bucket"] == "planned"]

    def gated_ids(self) -> list[str]:
        return [e["file_id"] for e in self.entries if e["bucket"] != "planned"]


class OrganizeApplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tree = ApplyTree(Path(self.tmp.name))

    def _apply(self, **kwargs):
        params = {
            "target": self.tree.target,
            "db_path": self.tree.db_path,
            "plan_path": self.tree.plan_path,
            "snapshot_dir": self.tree.snapshot_dir,
            "manifest_path": self.tree.manifest_path,
            "timestamp": "20200101T000000Z",
        }
        params.update(kwargs)
        return apply_mod.run_apply(**params)

    def test_dry_run_changes_nothing(self):
        before = self.tree.contents()
        preview = apply_mod.plan_apply_dry_run(
            target=self.tree.target,
            db_path=self.tree.db_path,
            plan_path=self.tree.plan_path,
        )
        self.assertEqual(len(preview.moves), 3)
        self.assertEqual(len(preview.skipped_review), 2)
        self.assertEqual(preview.missing_sources, ())
        self.assertEqual(self.tree.contents(), before)
        self.assertFalse(self.tree.snapshot_dir.exists())
        self.assertFalse(self.tree.manifest_path.exists())

    def test_real_run_moves_exactly_planned_set(self):
        result = self._apply()
        self.assertEqual(len(result.moved), 3)
        self.assertEqual(
            sorted(m.file_id for m in result.moved),
            sorted(self.tree.planned_ids()),
        )
        self.assertEqual(len(result.skipped_review), 2)
        self.assertEqual(result.skipped_identical, ())
        # Planned files reached their destinations with identical bytes.
        expected = {
            "doc.pdf": "Documentos/PDF/doc.pdf",
            "photo.jpg": "Media/Sin_personas/photo.jpg",
            "song.mp3": "Audio/MP3/song.mp3",
        }
        for name, dest in expected.items():
            content = next(c for n, _c, _s, _m, c in ApplyTree.FILES if n == name)
            dst = self.tree.target / dest
            self.assertTrue(dst.is_file(), f"missing {dest}")
            self.assertEqual(dst.read_bytes(), content)
            self.assertFalse((self.tree.target / name).exists())
        # Gated entries stayed in place.
        self.assertTrue((self.tree.target / "corrupt.jpg").is_file())
        self.assertTrue((self.tree.target / "broken.bin").is_file())
        # Proof artifacts exist.
        self.assertTrue(result.manifest_path.is_file())
        self.assertTrue(result.before_snapshot.is_file())
        self.assertTrue(result.after_snapshot.is_file())

    def test_manifest_holds_opaque_ids_and_checksums(self):
        result = self._apply()
        payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["manifest_version"], 1)
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(len(payload["entries"]), 3)
        by_id = {e["file_id"]: e for e in payload["entries"]}
        for name, _cat, _scan, _meta, content in ApplyTree.FILES:
            for move in result.moved:
                if move.src.name == name:
                    entry = by_id[move.file_id]
                    self.assertEqual(entry["sha256"], _sha256(content))
                    self.assertEqual(entry["src"], str(move.src))
                    self.assertEqual(entry["dst"], str(move.dst))

    def test_rollback_restores_original_layout(self):
        result = self._apply()
        rollback = apply_mod.run_rollback(
            manifest_path=result.manifest_path,
            snapshot_dir=self.tree.snapshot_dir,
            timestamp="20200102T000000Z",
        )
        self.assertEqual(len(rollback.restored), 3)
        for name, _cat, _scan, _meta, content in ApplyTree.FILES:
            path = self.tree.target / name
            if name in ("corrupt.jpg", "broken.bin"):
                self.assertTrue(path.is_file())
            else:
                self.assertTrue(path.is_file(), f"not restored: {name}")
                self.assertEqual(path.read_bytes(), content)
        self.assertFalse((self.tree.target / "Documentos" / "PDF" / "doc.pdf").exists())
        self.assertFalse((self.tree.target / "Audio" / "MP3" / "song.mp3").exists())
        self.assertFalse(
            (self.tree.target / "Media" / "Sin_personas" / "photo.jpg").exists()
        )
        # Rollback never deletes: empty destination directories may remain,
        # but they must hold no files.
        stray = [
            p for p in self.tree.target.rglob("*") if p.is_file()
            and str(p.relative_to(self.tree.target)) not in
            {f[0] for f in ApplyTree.FILES}
        ]
        self.assertEqual(stray, [])

    def test_rollback_dry_run_changes_nothing(self):
        result = self._apply()
        before = self.tree.contents()
        preview = apply_mod.plan_rollback_dry_run(result.manifest_path)
        self.assertEqual(len(preview.moves), 3)
        self.assertEqual(self.tree.contents(), before)

    def test_refuses_source_outside_target_root(self):
        outside = Path(self.tmp.name) / "elsewhere" / "extra.pdf"
        _write(outside, b"extra-bytes")
        conn = database.connect(self.tree.db_path)
        try:
            database.upsert_file(
                conn,
                database.FileRecord(
                    path=str(outside),
                    name="extra.pdf",
                    extension=".pdf",
                    category="document",
                    scan_status="ok",
                    metadata_status="skipped",
                    file_state="active",
                ),
            )
            conn.commit()
        finally:
            conn.close()
        generate_organization_preview(
            db_path=self.tree.db_path,
            plan_path=self.tree.plan_path,
            write_view=False,
        )
        before = self.tree.contents()
        with self.assertRaises(apply_mod.TargetRootError):
            self._apply()
        with self.assertRaises(apply_mod.TargetRootError):
            apply_mod.plan_apply_dry_run(
                target=self.tree.target,
                db_path=self.tree.db_path,
                plan_path=self.tree.plan_path,
            )
        self.assertEqual(self.tree.contents(), before)
        self.assertFalse(self.tree.manifest_path.exists())

    def test_review_gating_and_include_review(self):
        result = self._apply()
        self.assertEqual(len(result.skipped_review), 2)
        # A fresh tree: with --include-review every entry moves.
        second_base = Path(self.tmp.name) / "second"
        second_base.mkdir()
        second = ApplyTree(second_base)
        full = apply_mod.run_apply(
            target=second.target,
            db_path=second.db_path,
            plan_path=second.plan_path,
            snapshot_dir=second.snapshot_dir,
            manifest_path=second.manifest_path,
            include_review=True,
            timestamp="20200103T000000Z",
        )
        self.assertEqual(len(full.moved), 5)
        self.assertEqual(full.skipped_review, ())
        self.assertTrue(
            (second.target / "Revisión" / "Metadata_incompleta"
             / "corrupt.jpg").is_file()
        )
        self.assertTrue(
            (second.target / "Revisión" / "Errores_de_escaneo"
             / "broken.bin").is_file()
        )

    def test_collision_with_different_bytes_is_error(self):
        dst = self.tree.target / "Documentos" / "PDF" / "doc.pdf"
        _write(dst, b"someone-else-bytes")
        before = self.tree.contents()
        with self.assertRaises(apply_mod.CollisionError):
            self._apply()
        self.assertEqual(self.tree.contents(), before)
        self.assertFalse(self.tree.manifest_path.exists())

    def test_intra_run_duplicate_destination_is_refused(self):
        _write(self.tree.target / "sub_a" / "dup.pdf", b"first-bytes")
        _write(self.tree.target / "sub_b" / "dup.pdf", b"second-bytes")
        conn = database.connect(self.tree.db_path)
        try:
            for sub in ("sub_a", "sub_b"):
                database.upsert_file(
                    conn,
                    database.FileRecord(
                        path=str(self.tree.target / sub / "dup.pdf"),
                        name="dup.pdf",
                        extension=".pdf",
                        category="document",
                        scan_status="ok",
                        metadata_status="skipped",
                        file_state="active",
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        generate_organization_preview(
            db_path=self.tree.db_path,
            plan_path=self.tree.plan_path,
            write_view=False,
        )
        before = self.tree.contents()
        with self.assertRaises(apply_mod.CollisionError):
            self._apply()
        self.assertEqual(self.tree.contents(), before)
        self.assertFalse(self.tree.manifest_path.exists())

    def test_symlink_destination_is_never_overwritten(self):
        dst = self.tree.target / "Documentos" / "PDF" / "doc.pdf"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(self.tree.target / "song.mp3")
        with self.assertRaises(apply_mod.CollisionError):
            self._apply()
        self.assertTrue(dst.is_symlink())

    def test_identical_destination_is_skipped_not_an_error(self):
        dst = self.tree.target / "Documentos" / "PDF" / "doc.pdf"
        _write(dst, b"pdf-bytes")
        result = self._apply()
        self.assertEqual(len(result.moved), 2)
        self.assertEqual(len(result.skipped_identical), 1)
        # The skipped source stays where it is; the twin is untouched.
        self.assertTrue((self.tree.target / "doc.pdf").is_file())
        self.assertEqual(dst.read_bytes(), b"pdf-bytes")

    def test_missing_source_aborts_without_moving(self):
        (self.tree.target / "doc.pdf").unlink()
        before = self.tree.contents()
        with self.assertRaises(apply_mod.MissingSourceError):
            self._apply()
        self.assertEqual(self.tree.contents(), before)

    def test_plan_desync_is_refused(self):
        conn = database.connect(self.tree.db_path)
        try:
            database.upsert_file(
                conn,
                database.FileRecord(
                    path=str(self.tree.target / "late.pdf"),
                    name="late.pdf",
                    extension=".pdf",
                    category="document",
                    scan_status="ok",
                    metadata_status="skipped",
                    file_state="active",
                ),
            )
            conn.commit()
        finally:
            conn.close()
        with self.assertRaises(apply_mod.PlanDesyncError):
            self._apply()

    def test_same_count_path_swap_is_refused(self):
        conn = database.connect(self.tree.db_path)
        try:
            old = conn.execute(
                "SELECT path FROM files ORDER BY path LIMIT 1").fetchone()["path"]
            conn.execute("UPDATE files SET path = ? WHERE path = ?",
                         (old + ".swapped", old))
            conn.commit()
        finally:
            conn.close()
        before = self.tree.contents()
        with self.assertRaises(apply_mod.PlanDesyncError):
            self._apply()
        self.assertEqual(self.tree.contents(), before)

    def test_after_snapshot_failure_keeps_complete_manifest(self):
        real_save = apply_mod.save_snapshot
        def flaky(snapshot, path):
            if path.name.startswith("apply_after_"):
                raise OSError("snapshot disk full")
            return real_save(snapshot, path)
        apply_mod.save_snapshot = flaky
        try:
            result = self._apply()
        finally:
            apply_mod.save_snapshot = real_save
        payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(len(payload["entries"]), len(result.moved))

    def test_outputs_inside_target_are_refused(self):
        with self.assertRaises(apply_mod.UnsafeOutputError):
            self._apply(snapshot_dir=self.tree.target / "snaps")
        with self.assertRaises(apply_mod.UnsafeOutputError):
            self._apply(manifest_path=self.tree.target / "manifest.json")
        self.assertEqual(len(self.tree.contents()), 5)

    def test_rollback_refuses_checksum_mismatch(self):
        result = self._apply()
        moved_doc = self.tree.target / "Documentos" / "PDF" / "doc.pdf"
        with open(moved_doc, "ab") as handle:
            handle.write(b"tampered")
        with self.assertRaises(apply_mod.RollbackError):
            apply_mod.run_rollback(
                manifest_path=result.manifest_path,
                snapshot_dir=self.tree.snapshot_dir,
            )
        # Tampered file is still where apply left it.
        self.assertTrue(moved_doc.is_file())

    def test_rollback_refuses_blocked_original_path(self):
        result = self._apply()
        _write(self.tree.target / "doc.pdf", b"new-unrelated-bytes")
        with self.assertRaises(apply_mod.RollbackError):
            apply_mod.run_rollback(
                manifest_path=result.manifest_path,
                snapshot_dir=self.tree.snapshot_dir,
            )

    def test_default_manifest_path_sits_next_to_plan(self):
        manifest = apply_mod.default_manifest_path(
            self.tree.plan_path, "20200101T000000Z"
        )
        self.assertEqual(manifest.parent, self.tree.plan_path.parent)
        self.assertTrue(manifest.name.startswith("apply_manifest_"))

    def test_missing_plan_and_bad_plan(self):
        with self.assertRaises(apply_mod.PlanNotFoundError):
            apply_mod.plan_apply_dry_run(
                target=self.tree.target,
                db_path=self.tree.db_path,
                plan_path=self.tree.base / "nope.json",
            )
        bad = self.tree.base / "bad.json"
        bad.write_text('{"not": "a list"}', encoding="utf-8")
        with self.assertRaises(apply_mod.PlanInvalidError):
            apply_mod.plan_apply_dry_run(
                target=self.tree.target,
                db_path=self.tree.db_path,
                plan_path=bad,
            )

    def test_cli_dry_run_prints_and_changes_nothing(self):
        before = self.tree.contents()
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_organize_apply(
                target=self.tree.target,
                db_path=self.tree.db_path,
                plan_path=self.tree.plan_path,
                apply=False,
            )
        output = buf.getvalue()
        self.assertIn("DRY RUN", output)
        self.assertIn("would_move=3", output)
        self.assertEqual(self.tree.contents(), before)


if __name__ == "__main__":
    unittest.main()
