"""Canonical person identities over machine face clusters (V3.1).

A machine cluster label (``P0001``, ...) is only model evidence — the
model may split one real person into several clusters or (worse) merge
two people. A canonical person (``PERSON_0007`` + optional display name)
is the human-curated identity the user actually filters and organizes
by.

Resolution rule (``resolve_file_persons``), per file:

1. AUTO: machine labels from ``person_cluster_members`` mapped through
   ``person_aliases``; a machine label with no alias resolves to itself,
   so a catalog with zero curation behaves exactly like V2.5.
2. MANUAL: ``file_person_links`` with origin ``'manual'`` are added —
   this is how a user asserts "this video shows María" even when no
   face was ever detected in it.
3. EXCLUDED: links with origin ``'excluded'`` remove a person derived
   in step 1/2 — how a user corrects a wrong AUTO membership for one
   file without touching the rest of the cluster.

Exclusions always win over additions. Nothing here touches the scanned
tree; only the catalog database is read/written, and every operation
commits atomically from the CLI layer.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone

from . import database

MACHINE_RE = re.compile(r"^P\d{4}$")
CANONICAL_RE = re.compile(r"^PERSON_\d{4}$")

ORIGIN_MANUAL = "manual"
ORIGIN_EXCLUDED = "excluded"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_machine_label(ref: str) -> bool:
    return MACHINE_RE.match(ref or "") is not None


def is_canonical_id(ref: str) -> bool:
    return CANONICAL_RE.match(ref or "") is not None


def _machine_sets(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """Map file_path -> set of machine labels ({} when the cluster table
    is missing — same conservative guard as organization_plan)."""
    sets: dict[str, set[str]] = {}
    try:
        rows = conn.execute(
            "SELECT file_path, person_label FROM person_cluster_members"
        ).fetchall()
    except sqlite3.OperationalError:
        return sets
    for row in rows:
        sets.setdefault(row["file_path"], set()).add(row["person_label"])
    return sets


def resolve_file_persons(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """Effective canonical persons per file, applying aliases, manual
    links and exclusions (see module docstring)."""
    aliases = database.fetch_aliases(conn)
    resolved: dict[str, set[str]] = {}
    for file_path, labels in _machine_sets(conn).items():
        resolved[file_path] = {aliases.get(label, label) for label in labels}
    for link in database.fetch_file_links(conn):
        persons = resolved.setdefault(link["file_path"], set())
        if link["origin"] == ORIGIN_MANUAL:
            persons.add(link["person_id"])
        elif link["origin"] == ORIGIN_EXCLUDED:
            persons.discard(link["person_id"])
    return {path: persons for path, persons in resolved.items() if persons}


def member_machine_labels(conn: sqlite3.Connection, person_id: str) -> set[str]:
    """Machine clusters currently resolving to ``person_id``: aliased
    members plus the id itself when it is an unmapped machine label."""
    members = {
        machine
        for machine, canonical in database.fetch_aliases(conn).items()
        if canonical == person_id
    }
    if is_machine_label(person_id) and person_id in database.fetch_machine_labels(conn):
        members.add(person_id)
    return members


def _resolve_ref_to_canonical(
    conn: sqlite3.Connection, ref: str, machine_labels: set[str]
) -> str:
    """Map a user-supplied P000N/PERSON_0007 ref to its canonical id,
    validating that it names something real."""
    if is_canonical_id(ref):
        if database.fetch_person(conn, ref) is None:
            raise ValueError(f"persona canónica desconocida: {ref}")
        return ref
    if is_machine_label(ref):
        if ref not in machine_labels and ref not in database.fetch_aliases(conn):
            raise ValueError(f"cluster facial desconocido: {ref}")
        return database.fetch_aliases(conn).get(ref, ref)
    raise ValueError(f"referencia de persona inválida: {ref!r} (usa P000N o PERSON_0007)")


def merge_labels(
    conn: sqlite3.Connection, refs: list[str], name: str | None = None
) -> str:
    """Fuse 2+ clusters/persons into one canonical person.

    Reuses the first canonical id found among the refs, otherwise mints
    a fresh PERSON_0007. Every machine label involved is aliased to it.
    Returns the surviving canonical id. Never merges automatically —
    only this explicit user call does.
    """
    if len(set(refs)) < 2:
        raise ValueError("fusionar requiere al menos 2 referencias distintas")
    machine_labels = database.fetch_machine_labels(conn)
    aliases = database.fetch_aliases(conn)
    canonicals = [_resolve_ref_to_canonical(conn, ref, machine_labels) for ref in refs]

    target = next((c for c in canonicals if is_canonical_id(c)), None)
    involved_machine: set[str] = set()
    for ref, canonical in zip(refs, canonicals):
        if is_machine_label(ref):
            involved_machine.add(ref)
        else:  # an existing canonical: bring all its machine members along
            involved_machine.update(member_machine_labels(conn, canonical))
    # Machine labels that already resolve to one of the merged canonicals
    # move together (transitive intent of the explicit merge).
    for machine, canonical in aliases.items():
        if canonical in canonicals:
            involved_machine.add(machine)

    if target is None:
        target = database.next_person_id(conn)
        database.upsert_person(conn, target, name, _now())
    elif name is not None:
        database.upsert_person(conn, target, name, _now())
    for machine in sorted(involved_machine):
        database.set_alias(conn, machine, target, _now())
    return target


def rename_person(conn: sqlite3.Connection, ref: str, name: str) -> str:
    """Name a person. Naming a raw P000N cluster mints its canonical
    identity implicitly (alias P000N -> new PERSON_0007); naming a
    PERSON_0007 just updates it. Returns the canonical id."""
    if not (name or "").strip():
        raise ValueError("el nombre no puede estar vacío")
    machine_labels = database.fetch_machine_labels(conn)
    if is_machine_label(ref):
        if ref not in machine_labels and ref not in database.fetch_aliases(conn):
            raise ValueError(f"cluster facial desconocido: {ref}")
        target = database.fetch_aliases(conn).get(ref)
        if target is None:
            target = database.next_person_id(conn)
            database.upsert_person(conn, target, name.strip(), _now())
            database.set_alias(conn, ref, target, _now())
        else:
            database.upsert_person(conn, target, name.strip(), _now())
        return target
    if is_canonical_id(ref):
        if database.fetch_person(conn, ref) is None:
            raise ValueError(f"persona canónica desconocida: {ref}")
        database.upsert_person(conn, ref, name.strip(), _now())
        return ref
    raise ValueError(f"referencia de persona inválida: {ref!r}")


def _require_catalog_file(conn: sqlite3.Connection, file_path: str) -> str:
    row = conn.execute(
        "SELECT category FROM files WHERE path = ?", (file_path,)
    ).fetchone()
    if row is None:
        raise ValueError(f"archivo fuera del catálogo: {file_path}")
    return row["category"] or "other"


def assign_file(conn: sqlite3.Connection, file_path: str, ref: str) -> str:
    """Manually assert that a file shows a person (works even with zero
    detected faces). Returns the canonical id linked."""
    _require_catalog_file(conn, file_path)
    canonical = _resolve_ref_to_canonical(
        conn, ref, database.fetch_machine_labels(conn)
    )
    database.delete_file_link(conn, file_path, canonical)
    database.set_file_link(conn, file_path, canonical, ORIGIN_MANUAL, _now())
    return canonical


def unassign_file(conn: sqlite3.Connection, file_path: str, ref: str) -> str:
    """Remove a person's link to a file. A MANUAL link is deleted; an
    AUTO-derived membership becomes an explicit exclusion so the rest of
    the cluster is untouched. Returns the canonical id affected."""
    _require_catalog_file(conn, file_path)
    canonical = _resolve_ref_to_canonical(
        conn, ref, database.fetch_machine_labels(conn)
    )
    effective = resolve_file_persons(conn).get(file_path, set())
    if canonical not in effective:
        raise ValueError(f"{file_path} no está vinculado a {canonical}")
    database.delete_file_link(conn, file_path, canonical)
    if canonical in _machine_sets(conn).get(file_path, set()) or any(
        database.fetch_aliases(conn).get(m) == canonical
        for m in _machine_sets(conn).get(file_path, set())
    ):
        database.set_file_link(conn, file_path, canonical, ORIGIN_EXCLUDED, _now())
    return canonical


def split_files(
    conn: sqlite3.Connection,
    machine_label: str,
    file_paths: list[str],
    name: str | None = None,
) -> str:
    """File-level split: move files out of a machine cluster into a fresh
    canonical person (exclusion + manual link per file). The rest of the
    cluster keeps resolving where it did. Returns the new canonical id."""
    if not is_machine_label(machine_label):
        raise ValueError(f"se esperaba un cluster P000N, no {machine_label!r}")
    machine_labels = database.fetch_machine_labels(conn)
    if machine_label not in machine_labels:
        raise ValueError(f"cluster facial desconocido: {machine_label}")
    if not file_paths:
        raise ValueError("split requiere al menos 1 archivo")
    source = database.fetch_aliases(conn).get(machine_label, machine_label)
    auto_files = {
        path
        for path, labels in _machine_sets(conn).items()
        if machine_label in labels
    }
    for file_path in file_paths:
        _require_catalog_file(conn, file_path)
        if file_path not in auto_files:
            raise ValueError(f"{file_path} no pertenece al cluster {machine_label}")
    target = database.next_person_id(conn)
    database.upsert_person(conn, target, name, _now())
    for file_path in file_paths:
        database.delete_file_link(conn, file_path, source)
        database.set_file_link(conn, file_path, source, ORIGIN_EXCLUDED, _now())
        database.delete_file_link(conn, file_path, target)
        database.set_file_link(conn, file_path, target, ORIGIN_MANUAL, _now())
    return target


def effective_summary(conn: sqlite3.Connection) -> list[dict]:
    """One row per effective person: id, display name, file/image/video
    counts and member machine labels, ordered by person_id."""
    resolved = resolve_file_persons(conn)
    categories: dict[str, str] = {}
    try:
        for row in conn.execute("SELECT path, category FROM files"):
            categories[row["path"]] = row["category"] or "other"
    except sqlite3.OperationalError:
        pass
    names = {row["person_id"]: row["display_name"] for row in database.fetch_persons(conn)}
    aliases = database.fetch_aliases(conn)
    members: dict[str, set[str]] = {}
    for machine, canonical in aliases.items():
        members.setdefault(canonical, set()).add(machine)

    files_by_person: dict[str, set[str]] = {}
    for file_path, persons in resolved.items():
        for person in persons:
            files_by_person.setdefault(person, set()).add(file_path)

    summary = []
    for person_id in sorted(files_by_person):
        files = files_by_person[person_id]
        images = sum(1 for f in files if categories.get(f) == "image")
        videos = sum(1 for f in files if categories.get(f) == "video")
        own = {person_id} if is_machine_label(person_id) else set()
        summary.append(
            {
                "person_id": person_id,
                "display_name": names.get(person_id),
                "files": len(files),
                "images": images,
                "videos": videos,
                "machine_labels": sorted(members.get(person_id, set()) | own),
                "canonical": person_id in names,
            }
        )
    return summary


def combined_review_status(
    decisions: dict[str, str], person_id: str, member_labels: set[str]
) -> str | None:
    """Conservative merge of V2.4.1 human decisions over a canonical
    person and its member machine clusters: any rejection/uncertainty
    wins over confirmations (prefer Revisión over a wrong assignment)."""
    statuses = {decisions.get(person_id)}
    statuses.update(decisions.get(label) for label in member_labels)
    statuses.discard(None)
    if not statuses:
        return None
    if "rejected" in statuses:
        return "rejected"
    if "uncertain" in statuses:
        return "uncertain"
    if "confirmed" in statuses:
        return "confirmed"
    return None
