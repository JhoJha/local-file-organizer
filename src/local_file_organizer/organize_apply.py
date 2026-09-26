"""Real organization apply with dry-run default and rollback (V2.6).

SAFETY-CRITICAL module. This is the ONLY module allowed to rename files
inside the scanned tree, and only under all of these conditions:

- The caller passes an explicit target root; every source path resolved
  from the catalog and every computed destination must stay inside it.
  Anything outside aborts the whole run before moving anything.
- Dry-run is the default: without an explicit ``apply=True`` nothing on
  disk changes and no manifest or snapshot is written.
- Real runs write before/after snapshots (``--verify-readonly``-style
  proof) plus a rollback manifest, so every move is undoable.
- Entries with bucket ``review``/``unsupported`` are skipped unless the
  caller explicitly opts in with ``include_review=True``.
- Destinations are never overwritten: an existing destination with
  different bytes aborts the run; symlinks are never overwritten.
- Nothing is ever deleted: the only mutating primitive used here is
  ``os.rename`` (move), plus creating destination directories.

The service layer below performs no stdout/stderr output; it raises typed
exceptions and returns structured results. CLI printing lives in
``cli.py`` (``run_organize_apply`` / ``run_organize_rollback``).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import database
from .hashing import sha256_file
from .snapshot import capture_snapshot, save_snapshot

MANIFEST_VERSION = 1
MANIFEST_PREFIX = "apply_manifest_"
BEFORE_PREFIX = "apply_before_"
AFTER_PREFIX = "apply_after_"

#: Buckets that are never moved unless the caller passes ``include_review``.
GATED_BUCKETS = frozenset({"review", "unsupported"})


class OrganizeApplyError(Exception):
    """Base exception for organize-apply / organize-rollback errors."""


class PlanNotFoundError(OrganizeApplyError):
    """Raised when the plan JSON file does not exist."""


class PlanInvalidError(OrganizeApplyError):
    """Raised when the plan JSON is malformed or has an unexpected shape."""


class CatalogNotFoundError(OrganizeApplyError):
    """Raised when the SQLite catalog does not exist or cannot be read."""


class PlanDesyncError(OrganizeApplyError):
    """Raised when active catalog files do not match the plan entries."""


class TargetRootError(OrganizeApplyError):
    """Raised when the target root is missing, not a directory, or when a
    plan source path or computed destination would fall outside it."""


class UnsafeOutputError(OrganizeApplyError):
    """Raised when the snapshot directory or manifest path lives inside the
    target root (apply artifacts must never pollute the organized tree)."""


class ApplyRefusedError(OrganizeApplyError):
    """Base class for pre-flight refusals: nothing was moved."""


class MissingSourceError(ApplyRefusedError):
    """Raised when a planned source file no longer exists on disk."""


class CollisionError(ApplyRefusedError):
    """Raised when a destination already exists with different bytes (or is
    a symlink / directory): apply never overwrites, it refuses."""


class SourceTypeError(ApplyRefusedError):
    """Raised when a planned source is a directory instead of a file."""


class ApplyInterruptedError(OrganizeApplyError):
    """Raised when a real run fails midway. Some moves may have completed;
    ``manifest_path`` points at the partial manifest so the completed moves
    stay undoable via ``organize-rollback``."""

    def __init__(self, message: str, manifest_path: Path) -> None:
        super().__init__(message)
        self.manifest_path = manifest_path


class RollbackError(OrganizeApplyError):
    """Base class for rollback errors: nothing was moved."""


@dataclass(frozen=True)
class PlannedMove:
    """One validated file relocation, fully inside the target root."""

    file_id: str
    src: Path
    dst: Path
    bucket: str
    proposed_destination: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ApplyPreview:
    """Dry-run outcome: what a real run would do. Nothing was changed and
    no manifest or snapshot was written."""

    target_root: Path
    moves: tuple[PlannedMove, ...]
    skipped_review: tuple[str, ...]
    existing_destinations: tuple[str, ...]
    missing_sources: tuple[str, ...]
    dry_run: bool = True


@dataclass(frozen=True)
class ApplyResult:
    """Outcome of a real (``apply=True``) run."""

    target_root: Path
    moved: tuple[PlannedMove, ...]
    skipped_review: tuple[str, ...]
    skipped_identical: tuple[str, ...]
    manifest_path: Path
    before_snapshot: Path
    after_snapshot: Path


@dataclass(frozen=True)
class RollbackPreview:
    """Dry-run outcome of a rollback. Nothing was changed."""

    manifest_path: Path
    moves: tuple[PlannedMove, ...]
    dry_run: bool = True


@dataclass(frozen=True)
class RollbackResult:
    """Outcome of a real rollback run."""

    manifest_path: Path
    restored: tuple[PlannedMove, ...]
    before_snapshot: Path
    after_snapshot: Path


@dataclass
class _ResolvedPlan:
    entries: list[dict]
    paths_by_file_id: dict[str, str]
    target_root: Path


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_plan_entries(plan_path: Path) -> list[dict]:
    """Read and validate the plan JSON (a list of plan entry dicts)."""
    if not plan_path.exists():
        raise PlanNotFoundError(f"plan file not found: {plan_path}")
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanInvalidError(f"invalid plan JSON in {plan_path}: {exc}") from exc
    if not isinstance(payload, list):
        raise PlanInvalidError(f"invalid plan JSON in {plan_path}: expected a list")
    required = {"file_id", "bucket", "proposed_destination"}
    for entry in payload:
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise PlanInvalidError(
                f"invalid plan JSON in {plan_path}: every entry needs "
                f"{sorted(required)}"
            )
    return payload


def resolve_source_paths(
    db_path: Path, entries: list[dict]
) -> dict[str, str]:
    """Map each plan ``file_id`` to its catalog source path.

    File ids (``F000001``, ...) are assigned by ascending ``path`` order and
    recomputed on every preview, so the mapping zips the active catalog rows
    (same ordering) with the plan entries positionally and verifies the ids
    match. Any count or id mismatch means the plan no longer describes this
    catalog and the run is refused instead of moving the wrong files.
    """
    if not db_path.exists():
        raise CatalogNotFoundError(f"catalog not found: {db_path}")
    try:
        conn = database.connect(db_path)
    except Exception as exc:
        raise CatalogNotFoundError(f"catalog not found: {db_path}") from exc
    try:
        try:
            rows = conn.execute(
                "SELECT path FROM files "
                "WHERE COALESCE(file_state, 'active') = 'active' "
                "ORDER BY path ASC"
            ).fetchall()
        except Exception as exc:
            raise CatalogNotFoundError(
                f"catalog not found or unreadable: {db_path}"
            ) from exc
    finally:
        conn.close()
    if len(rows) != len(entries):
        raise PlanDesyncError(
            f"plan desync: {len(rows)} active catalog files but "
            f"{len(entries)} plan entries; regenerate the plan with "
            f"organize-preview before applying"
        )
    mapping: dict[str, str] = {}
    for index, (row, entry) in enumerate(zip(rows, entries)):
        expected = f"F{index + 1:06d}"
        if entry["file_id"] != expected:
            raise PlanDesyncError(
                f"plan desync: entry {index} has id {entry['file_id']!r}, "
                f"expected {expected!r}; regenerate the plan with "
                f"organize-preview before applying"
            )
        want = hashlib.sha256(row["path"].encode("utf-8")).hexdigest()
        if entry.get("src_hash") != want:
            raise PlanDesyncError(
                f"plan desync: source mismatch for {entry['file_id']!r}; "
                f"regenerate the plan with organize-preview before applying"
            )
        mapping[entry["file_id"]] = row["path"]
    return mapping


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _ensure_inside(path: Path, root: Path, role: str) -> None:
    if path != root and root not in path.parents:
        raise TargetRootError(
            f"refused: {role} escapes the target root "
            f"({path} is outside {root})"
        )


def _resolve_plan(
    target: Path,
    db_path: Path,
    plan_path: Path,
    include_review: bool,
) -> tuple[_ResolvedPlan, list[dict], list[str]]:
    """Load the plan, resolve sources, and split movable vs gated entries.

    Returns (resolved, movable_entries, skipped_review_ids). Raises before
    touching the tree on any structural problem (missing target, plan or
    catalog issues, desync, paths escaping the target root).
    """
    target_root = _absolute(target)
    if not target_root.is_dir():
        raise TargetRootError(
            f"refused: target root is not a directory: {target_root}"
        )
    entries = load_plan_entries(plan_path)
    paths_by_file_id = resolve_source_paths(db_path, entries)
    movable: list[dict] = []
    skipped_review: list[str] = []
    for entry in entries:
        bucket = entry["bucket"]
        if bucket in GATED_BUCKETS and not include_review:
            skipped_review.append(entry["file_id"])
            continue
        movable.append(entry)
    resolved = _ResolvedPlan(
        entries=entries,
        paths_by_file_id=paths_by_file_id,
        target_root=target_root,
    )
    return resolved, movable, skipped_review


def _destination_for(
    src: Path, proposed_destination: str, target_root: Path
) -> Path:
    """Compute the in-target destination for one source file.

    Layout: ``<target>/<proposed_destination>/<basename>``. The proposed
    destination must be a relative path without ``..`` segments; anything
    else is refused so a crafted plan can never redirect a move outside
    the target root.
    """
    dest_rel = Path(proposed_destination)
    if dest_rel.is_absolute() or ".." in dest_rel.parts:
        raise TargetRootError(
            f"refused: unsafe proposed destination "
            f"{proposed_destination!r}: must be a relative path without '..'"
        )
    dst = _absolute(target_root / dest_rel / src.name)
    _ensure_inside(dst, target_root, "destination")
    return dst


def _check_output_locations(
    target_root: Path, snapshot_dir: Path, manifest_path: Path
) -> None:
    """Refuse apply artifacts that would land inside the organized tree."""
    snapshot_abs = _absolute(snapshot_dir)
    if snapshot_abs == target_root or target_root in snapshot_abs.parents:
        raise UnsafeOutputError(
            f"refused: snapshot dir {snapshot_abs} is inside the target "
            f"root {target_root}; choose a path outside it"
        )
    manifest_abs = _absolute(manifest_path)
    if manifest_abs == target_root or target_root in manifest_abs.parents:
        raise UnsafeOutputError(
            f"refused: manifest path {manifest_abs} is inside the target "
            f"root {target_root}; choose a path outside it"
        )


def _stat_moves(
    resolved: _ResolvedPlan, movable: list[dict]
) -> tuple[list[dict], list[str], list[str]]:
    """Cheap dry-run stats (no hashing): ready, existing-dst, missing-src.

    Used only for the dry-run preview so it stays fast and obviously
    side-effect free. The real run performs the full hashing pre-flight.
    """
    ready: list[dict] = []
    existing: list[str] = []
    missing: list[str] = []
    for entry in movable:
        src = _absolute(Path(resolved.paths_by_file_id[entry["file_id"]]))
        _ensure_inside(src, resolved.target_root, "source")
        dst = _destination_for(
            src, entry["proposed_destination"], resolved.target_root
        )
        if src == dst:
            continue  # Already in place: nothing to do.
        if not os.path.lexists(src):
            missing.append(entry["file_id"])
            continue
        if dst.is_symlink() or dst.exists():
            existing.append(entry["file_id"])
            continue
        ready.append(entry)
    return ready, existing, missing


def plan_apply_dry_run(
    target: Path,
    db_path: Path,
    plan_path: Path,
    include_review: bool = False,
) -> ApplyPreview:
    """Preview the apply without changing anything (no hashing, no writes)."""
    resolved, movable, skipped_review = _resolve_plan(
        target, db_path, plan_path, include_review
    )
    ready, existing, missing = _stat_moves(resolved, movable)
    moves = tuple(
        PlannedMove(
            file_id=entry["file_id"],
            src=_absolute(Path(resolved.paths_by_file_id[entry["file_id"]])),
            dst=_destination_for(
                _absolute(Path(resolved.paths_by_file_id[entry["file_id"]])),
                entry["proposed_destination"],
                resolved.target_root,
            ),
            bucket=entry["bucket"],
            proposed_destination=entry["proposed_destination"],
            sha256="",
            size_bytes=-1,
        )
        for entry in ready
    )
    return ApplyPreview(
        target_root=resolved.target_root,
        moves=moves,
        skipped_review=tuple(skipped_review),
        existing_destinations=tuple(existing),
        missing_sources=tuple(missing),
    )


def _preflight_moves(
    resolved: _ResolvedPlan, movable: list[dict]
) -> tuple[list[PlannedMove], list[str]]:
    """Full pre-flight for a real run: hash sources, detect collisions.

    Returns (moves, skipped_identical). Raises (moving nothing) when a
    source is missing, is a directory, escapes the target root, or when a
    destination already exists with different bytes (or is a symlink or a
    directory). A destination with byte-identical content is skipped, not
    treated as an error.
    """
    moves: list[PlannedMove] = []
    skipped_identical: list[str] = []
    missing: list[str] = []
    wrong_type: list[str] = []
    outside: list[str] = []
    collisions: list[str] = []
    for entry in movable:
        file_id = entry["file_id"]
        src = _absolute(Path(resolved.paths_by_file_id[file_id]))
        if src != resolved.target_root and resolved.target_root not in src.parents:
            outside.append(file_id)
            continue
        dst = _destination_for(
            src, entry["proposed_destination"], resolved.target_root
        )
        if src == dst:
            continue  # Already in place: nothing to do.
        if not os.path.lexists(src):
            missing.append(file_id)
            continue
        if src.is_dir():
            wrong_type.append(file_id)
            continue
        try:
            digest = sha256_file(src)
            size_bytes = src.stat().st_size
        except OSError:
            missing.append(file_id)
            continue
        if dst.is_symlink() or dst.exists():
            if (
                not dst.is_symlink()
                and dst.is_file()
                and _bytes_match(dst, digest)
            ):
                skipped_identical.append(file_id)
                continue
            collisions.append(file_id)
            continue
        moves.append(
            PlannedMove(
                file_id=file_id,
                src=src,
                dst=dst,
                bucket=entry["bucket"],
                proposed_destination=entry["proposed_destination"],
                sha256=digest,
                size_bytes=size_bytes,
            )
        )
    problems: list[str] = []
    if outside:
        problems.append(
            f"sources outside target root ({len(outside)}): "
            + ", ".join(outside[:10])
        )
    if missing:
        problems.append(
            f"missing sources ({len(missing)}): " + ", ".join(missing[:10])
        )
    if wrong_type:
        problems.append(
            f"sources that are directories ({len(wrong_type)}): "
            + ", ".join(wrong_type[:10])
        )
    if collisions:
        problems.append(
            f"destination collisions with different bytes "
            f"({len(collisions)}): " + ", ".join(collisions[:10])
        )
    if outside:
        raise TargetRootError(
            "refused: " + "; ".join(problems) + "; nothing was moved"
        )
    if missing:
        raise MissingSourceError(
            "refused: " + "; ".join(problems) + "; nothing was moved"
        )
    if wrong_type:
        raise SourceTypeError(
            "refused: " + "; ".join(problems) + "; nothing was moved"
        )
    if collisions:
        raise CollisionError(
            "refused: " + "; ".join(problems) + "; nothing was moved"
        )
    return moves, skipped_identical


def _bytes_match(path: Path, digest: str) -> bool:
    try:
        return sha256_file(path) == digest
    except OSError:
        return False


def default_manifest_path(plan_path: Path, timestamp: str) -> Path:
    """Default rollback manifest location: next to the plan (``reports/``)."""
    return plan_path.parent / f"{MANIFEST_PREFIX}{timestamp}.json"


def write_manifest(
    manifest_path: Path,
    target_root: Path,
    include_review: bool,
    moves: list[PlannedMove],
    status: str,
) -> Path:
    """Write the rollback manifest (opaque ids + src/dst + checksums).

    Only ``open(path, "w")`` in this module besides the safety-gate
    allowlist justification: manifests always land outside the target root
    (enforced by ``_check_output_locations`` before any move).
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_root": str(target_root),
        "include_review": include_review,
        "status": status,
        "entries": [
            {
                "file_id": move.file_id,
                "src": str(move.src),
                "dst": str(move.dst),
                "sha256": move.sha256,
                "size_bytes": move.size_bytes,
            }
            for move in moves
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return manifest_path


def load_manifest(manifest_path: Path) -> dict:
    """Read and validate a rollback manifest."""
    if not manifest_path.exists():
        raise RollbackError(f"manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RollbackError(
            f"invalid manifest JSON in {manifest_path}: {exc}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("manifest_version") != MANIFEST_VERSION
        or not isinstance(payload.get("target_root"), str)
        or not isinstance(payload.get("entries"), list)
    ):
        raise RollbackError(
            f"invalid manifest in {manifest_path}: unsupported or corrupt format"
        )
    for entry in payload["entries"]:
        if (
            not isinstance(entry, dict)
            or not {"file_id", "src", "dst", "sha256"}.issubset(entry)
        ):
            raise RollbackError(
                f"invalid manifest in {manifest_path}: corrupt entry"
            )
    return payload


def run_apply(
    target: Path,
    db_path: Path,
    plan_path: Path,
    snapshot_dir: Path,
    manifest_path: Path | None = None,
    include_review: bool = False,
    timestamp: str | None = None,
) -> ApplyResult:
    """Execute the planned moves (real run: the caller confirmed ``--apply``).

    Order: structural checks -> full hashing pre-flight (refuses without
    moving anything) -> before snapshot -> moves -> manifest -> after
    snapshot (best-effort proof). A mid-run failure still writes a partial
    manifest so the completed moves stay undoable, then raises ``ApplyInterruptedError``.
    """
    stamp = timestamp or _utc_timestamp()
    resolved, movable, skipped_review = _resolve_plan(
        target, db_path, plan_path, include_review
    )
    manifest = (
        _absolute(manifest_path)
        if manifest_path is not None
        else _absolute(default_manifest_path(plan_path, stamp))
    )
    _check_output_locations(resolved.target_root, snapshot_dir, manifest)
    moves, skipped_identical = _preflight_moves(resolved, movable)

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    before_path = _absolute(snapshot_dir / f"{BEFORE_PREFIX}{stamp}.json")
    after_path = _absolute(snapshot_dir / f"{AFTER_PREFIX}{stamp}.json")
    save_snapshot(capture_snapshot(resolved.target_root), before_path)

    completed: list[PlannedMove] = []
    try:
        for move in moves:
            move.dst.parent.mkdir(parents=True, exist_ok=True)
            # Pre-flight guaranteed the destination does not exist, so
            # os.rename can never overwrite here; it only relocates.
            os.rename(move.src, move.dst)
            completed.append(move)
    except OSError as exc:
        partial = write_manifest(
            manifest, resolved.target_root, include_review, completed, "partial"
        )
        save_snapshot(capture_snapshot(resolved.target_root), after_path)
        raise ApplyInterruptedError(
            f"interrupted after {len(completed)}/{len(moves)} moves "
            f"({exc}); partial manifest at {partial}; "
            f"restore with organize-rollback",
            partial,
        ) from exc

    write_manifest(manifest, resolved.target_root, include_review, completed, "complete")
    try:
        save_snapshot(capture_snapshot(resolved.target_root), after_path)
    except OSError:
        pass  # Snapshot is proof; the manifest above already keeps undo.
    return ApplyResult(
        target_root=resolved.target_root,
        moved=tuple(completed),
        skipped_review=tuple(skipped_review),
        skipped_identical=tuple(skipped_identical),
        manifest_path=manifest,
        before_snapshot=before_path,
        after_snapshot=after_path,
    )


def _rollback_moves(payload: dict) -> list[PlannedMove]:
    target_root = _absolute(Path(payload["target_root"]))
    moves: list[PlannedMove] = []
    for entry in payload["entries"]:
        src = _absolute(Path(entry["src"]))
        dst = _absolute(Path(entry["dst"]))
        _ensure_inside(src, target_root, "rollback source")
        _ensure_inside(dst, target_root, "rollback destination")
        moves.append(
            PlannedMove(
                file_id=entry["file_id"],
                src=dst,  # Rollback direction: current location -> original.
                dst=src,
                bucket="",
                proposed_destination="",
                sha256=entry["sha256"],
                size_bytes=entry.get("size_bytes", -1),
            )
        )
    return moves


def plan_rollback_dry_run(manifest_path: Path) -> RollbackPreview:
    """Preview a rollback without changing anything."""
    payload = load_manifest(_absolute(manifest_path))
    return RollbackPreview(
        manifest_path=_absolute(manifest_path),
        moves=tuple(_rollback_moves(payload)),
    )


def run_rollback(
    manifest_path: Path,
    snapshot_dir: Path,
    timestamp: str | None = None,
) -> RollbackResult:
    """Undo every move recorded in a rollback manifest (reverse order).

    Pre-flight verifies each current location still holds the exact bytes
    recorded in the manifest (checksum match) and that no original path is
    now occupied by different bytes; any mismatch refuses the whole
    rollback without moving anything.
    """
    stamp = timestamp or _utc_timestamp()
    manifest_abs = _absolute(manifest_path)
    payload = load_manifest(manifest_abs)
    target_root = _absolute(Path(payload["target_root"]))
    if not target_root.is_dir():
        raise RollbackError(f"target root is not a directory: {target_root}")
    moves = list(reversed(_rollback_moves(payload)))

    snapshot_abs = _absolute(snapshot_dir)
    if snapshot_abs == target_root or target_root in snapshot_abs.parents:
        raise UnsafeOutputError(
            f"refused: snapshot dir {snapshot_abs} is inside the target "
            f"root {target_root}; choose a path outside it"
        )

    problems: list[str] = []
    for move in moves:
        if not os.path.lexists(move.src):
            problems.append(f"missing current location {move.file_id}")
        elif not _bytes_match(move.src, move.sha256):
            problems.append(f"checksum mismatch at {move.file_id}")
        if move.dst.is_symlink() or move.dst.exists():
            if not (
                not move.dst.is_symlink()
                and move.dst.is_file()
                and _bytes_match(move.dst, move.sha256)
            ):
                problems.append(f"blocked original path {move.file_id}")
    if problems:
        raise RollbackError(
            "refused rollback (" + "; ".join(problems[:10]) + "); "
            "nothing was moved"
        )

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    before_path = _absolute(snapshot_dir / f"{BEFORE_PREFIX}rollback_{stamp}.json")
    after_path = _absolute(snapshot_dir / f"{AFTER_PREFIX}rollback_{stamp}.json")
    save_snapshot(capture_snapshot(target_root), before_path)

    restored: list[PlannedMove] = []
    for move in moves:
        if move.src == move.dst:
            continue
        if move.dst.exists() and _bytes_match(move.dst, move.sha256):
            continue  # Already restored (identical bytes in place).
        move.dst.parent.mkdir(parents=True, exist_ok=True)
        # Pre-flight guaranteed the original path is free (or identical),
        # so os.rename can never overwrite here; it only moves back.
        os.rename(move.src, move.dst)
        restored.append(move)
    save_snapshot(capture_snapshot(target_root), after_path)
    return RollbackResult(
        manifest_path=manifest_abs,
        restored=tuple(restored),
        before_snapshot=before_path,
        after_snapshot=after_path,
    )
