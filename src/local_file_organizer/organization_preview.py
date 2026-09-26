"""Organization preview use case and report formatting.

Extracts preview generation and reporting out of cli.py so that organization
planning can be invoked programmatically without CLI dependencies or stdout side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import database
from .face_review import load_decisions
from .organization_plan import (
    DESTINATION_REVIEW_LOW_EVIDENCE,
    DESTINATION_REVIEW_METADATA,
    DESTINATION_REVIEW_REJECTED,
    DESTINATION_REVIEW_SCAN,
    DESTINATION_REVIEW_UNCERTAIN,
    DESTINATION_SIN_PERSONAS,
    VIEW_DIR_NAME,
    build_organization_plan,
    plan_counts,
    write_plan_json,
    write_plan_view_site,
)
from .person_identity import resolve_file_persons


class OrganizationPreviewError(Exception):
    """Base exception for organization preview errors."""


class CatalogNotFoundError(OrganizationPreviewError):
    """Raised when the SQLite catalog does not exist or cannot be read."""


class CatalogEmptyError(OrganizationPreviewError):
    """Raised when the SQLite catalog contains zero files."""


class InvalidDecisionsError(OrganizationPreviewError):
    """Raised when the human decisions file is invalid or malformed."""


class PlanDesyncError(OrganizationPreviewError, RuntimeError):
    """Raised when active files in catalog do not match generated plan entries."""


@dataclass(frozen=True)
class PreviewResult:
    """Structured result of an organization preview generation."""

    plan_path: Path
    entries: list[dict[str, Any]]
    counts: dict[str, Any]
    paths_by_destination: dict[str, list[str]]
    view_path: Path | None = None


def generate_organization_preview(
    db_path: Path,
    plan_path: Path,
    decisions_path: Path | None = None,
    show_files: bool = False,
    view_dir: Path | None = None,
    write_view: bool = True,
) -> PreviewResult:
    """Generate the organization plan and return structured preview data.

    This function is strictly read-only on the catalog database and decisions file,
    and only writes the output plan JSON to `plan_path` plus the static offline
    viewer (`index.html`) under `view_dir` (default: `<plan_dir>/organization_view`).
    Pass `write_view=False` to skip the viewer. Real paths are embedded in the
    viewer only when `show_files` is true (explicit manual runs); otherwise the
    viewer holds opaque file ids only. It produces no stdout/stderr
    and raises typed exceptions instead of terminating the process.
    """
    if not db_path.exists():
        raise CatalogNotFoundError(f"no existe el catálogo {db_path}")

    try:
        conn = database.connect(db_path)
    except Exception as exc:
        raise CatalogNotFoundError(f"no existe el catálogo {db_path}") from exc

    try:
        try:
            count_row = conn.execute("SELECT COUNT(*) FROM files").fetchone()
        except Exception as exc:
            raise CatalogNotFoundError(f"no existe el catálogo {db_path}") from exc

        if (count_row[0] if count_row else 0) == 0:
            raise CatalogEmptyError(f"el catálogo {db_path} está vacío")

        if decisions_path is not None:
            try:
                decisions = load_decisions(decisions_path)
            except ValueError as exc:
                raise InvalidDecisionsError(str(exc)) from exc
        else:
            decisions = {}

        entries = build_organization_plan(conn, decisions=decisions)
        counts = plan_counts(entries)
        write_plan_json(entries, plan_path)

        paths_by_destination: dict[str, list[str]] = {}
        persons_by_file_id: dict[str, list[str]] = {}
        paths_by_file_id: dict[str, str] | None = None
        if show_files or write_view:
            rows = conn.execute(
                "SELECT path FROM files WHERE COALESCE(file_state, 'active') = 'active' ORDER BY path ASC"
            ).fetchall()
            dest_by_index = [e["proposed_destination"] for e in entries]
            aligned = len(rows) == len(dest_by_index)
            if not aligned and show_files:
                raise PlanDesyncError(
                    f"desincronización en organize-preview: {len(rows)} archivos activos pero {len(dest_by_index)} entradas de plan"
                )
            if aligned:
                person_sets = resolve_file_persons(conn)
                for row, entry in zip(rows, entries):
                    persons = sorted(person_sets.get(row["path"], ()))
                    if persons:
                        persons_by_file_id[entry["file_id"]] = persons
                if show_files:
                    paths_by_file_id = {
                        entry["file_id"]: row["path"]
                        for row, entry in zip(rows, entries)
                    }
                    for row, dest in zip(rows, dest_by_index):
                        paths_by_destination.setdefault(dest, []).append(row["path"])

        view_path: Path | None = None
        if write_view:
            resolved_view_dir = (
                view_dir if view_dir is not None else plan_path.parent / VIEW_DIR_NAME
            )
            view_path = write_plan_view_site(
                entries,
                counts,
                resolved_view_dir,
                persons_by_file_id=persons_by_file_id,
                paths_by_file_id=paths_by_file_id,
            )

        return PreviewResult(
            plan_path=plan_path,
            entries=entries,
            counts=counts,
            paths_by_destination=paths_by_destination,
            view_path=view_path,
        )
    finally:
        conn.close()


def _files_word(n: int) -> str:
    return "1 archivo" if n == 1 else f"{n} archivos"


def _leaf_line(name: str, n: int) -> str:
    return f"{name}  {_files_word(n)}"


def format_preview_report(result: PreviewResult, show_files: bool = False) -> str:
    """Format preview results into a human-readable text report.

    Pure function: performs no I/O, does not read disk or database, and produces no side effects.
    """
    by_destination: dict[str, int] = result.counts["by_destination"]
    counts = result.counts
    paths_by_destination = result.paths_by_destination

    lines = ["=== ORGANIZATION PREVIEW ===", ""]

    media_dests = {d: c for d, c in by_destination.items() if d.startswith("Media/")}
    if media_dests:
        lines.append("Media/")
        personas = sorted(
            (d, c) for d, c in media_dests.items() if d.startswith("Media/Personas/")
        )
        parejas = sorted(
            (d, c) for d, c in media_dests.items() if d.startswith("Media/Parejas/")
        )
        grupos = sorted(
            (d, c) for d, c in media_dests.items() if d.startswith("Media/Grupos/")
        )
        sin = media_dests.get(DESTINATION_SIN_PERSONAS)
        if personas:
            lines.append("  Personas/")
            for dest, c in personas:
                label = dest.split("/")[2]
                lines.append(f"    {label}/  {_files_word(c)}")
                if show_files:
                    for p in sorted(paths_by_destination.get(dest, [])):
                        lines.append(f"      - {p}")
        if parejas:
            lines.append("  Parejas/")
            for dest, c in parejas:
                label = dest.split("/")[2]
                lines.append(f"    {label}/  {_files_word(c)}")
                if show_files:
                    for p in sorted(paths_by_destination.get(dest, [])):
                        lines.append(f"      - {p}")
        if grupos:
            lines.append("  Grupos/")
            for dest, c in grupos:
                label = dest.split("/")[2]
                lines.append(f"    {label}/  {_files_word(c)}")
                if show_files:
                    for p in sorted(paths_by_destination.get(dest, [])):
                        lines.append(f"      - {p}")
        if sin:
            lines.append(f"  {_leaf_line('Sin_personas/', sin)}")
            if show_files:
                for p in sorted(paths_by_destination.get(DESTINATION_SIN_PERSONAS, [])):
                    lines.append(f"      - {p}")
        lines.append("")

    revision = {d: c for d, c in by_destination.items() if d.startswith("Revisión/")}
    if revision:
        lines.append("Revisión/")
        for label, dest in (
            ("Persona_incierta/", DESTINATION_REVIEW_UNCERTAIN),
            ("Cluster_rechazado/", DESTINATION_REVIEW_REJECTED),
            ("Baja_evidencia/", DESTINATION_REVIEW_LOW_EVIDENCE),
            ("Metadata_incompleta/", DESTINATION_REVIEW_METADATA),
            ("Errores_de_escaneo/", DESTINATION_REVIEW_SCAN),
        ):
            n = revision.get(dest)
            if not n:
                continue
            lines.append(f"  {_leaf_line(label, n)}")
            if show_files:
                for p in sorted(paths_by_destination.get(dest, [])):
                    lines.append(f"    - {p}")
        lines.append("")

    documentos = sorted(
        (d, c) for d, c in by_destination.items() if d.startswith("Documentos/")
    )
    if documentos:
        lines.append("Documentos/")
        for dest, c in documentos:
            label = dest.split("/")[1]
            lines.append(f"  {_leaf_line(label + '/', c)}")
            if show_files:
                for p in sorted(paths_by_destination.get(dest, [])):
                    lines.append(f"    - {p}")
        lines.append("")

    audios = sorted((d, c) for d, c in by_destination.items() if d.startswith("Audio/"))
    if audios:
        lines.append("Audio/")
        for dest, c in audios:
            label = dest.split("/")[1]
            lines.append(f"  {_leaf_line(label + '/', c)}")
            if show_files:
                for p in sorted(paths_by_destination.get(dest, [])):
                    lines.append(f"    - {p}")
        lines.append("")

    if by_destination.get("Archivos_comprimidos/"):
        n = by_destination["Archivos_comprimidos/"]
        lines.append(f"Archivos_comprimidos/  {_files_word(n)}")
        if show_files:
            for p in sorted(paths_by_destination.get("Archivos_comprimidos/", [])):
                lines.append(f"  - {p}")
        lines.append("")

    if by_destination.get("Software/"):
        n = by_destination["Software/"]
        lines.append(f"Software/  {_files_word(n)}")
        if show_files:
            for p in sorted(paths_by_destination.get("Software/", [])):
                lines.append(f"  - {p}")
        lines.append("")

    otros = sorted((d, c) for d, c in by_destination.items() if d.startswith("Otros/"))
    if otros:
        lines.append("Otros/")
        for dest, c in otros:
            label = dest.split("/")[1]
            lines.append(f"  {_leaf_line(label + '/', c)}")
            if show_files:
                for p in sorted(paths_by_destination.get(dest, [])):
                    lines.append(f"    - {p}")
        lines.append("")

    person_destinations = sum(
        c for d, c in by_destination.items() if d.startswith("Media/Personas/")
    )
    pair_destinations = sum(
        c for d, c in by_destination.items() if d.startswith("Media/Parejas/")
    )
    group_destinations = sum(
        c for d, c in by_destination.items() if d.startswith("Media/Grupos/")
    )
    review_destinations = sum(
        c for d, c in by_destination.items() if d.startswith("Revisión/")
    )
    sin_personas = by_destination.get(DESTINATION_SIN_PERSONAS, 0)

    lines.append("FLAGS:")
    lines.append(f"exact_duplicates={counts['exact_duplicates']}")
    lines.append(f"visual_similar_groups={counts['visual_similar_groups']}")
    lines.append("")
    lines.append("=== DESTINATIONS ===")
    lines.append(f"person_destinations={person_destinations}")
    lines.append(f"pair_destinations={pair_destinations}")
    lines.append(f"group_destinations={group_destinations}")
    lines.append(f"review_destinations={review_destinations}")
    lines.append(f"sin_personas={sin_personas}")
    lines.append("")
    lines.append("=== CONFIDENCE ===")
    confidence = counts["confidence_levels"]
    lines.append(f"high={confidence['high']}")
    lines.append(f"medium={confidence['medium']}")
    lines.append(f"low={confidence['low']}")
    lines.append("")
    lines.append("=== ACCOUNTING ===")
    lines.append(f"catalog_files={counts['catalog_files']}")
    lines.append(f"planned={counts['planned']}")
    lines.append(f"review={counts['review']}")
    lines.append(f"unsupported={counts['unsupported']}")
    lines.append("")
    lines.append("PREVIEW ONLY")
    lines.append("moved=0")
    lines.append("copied=0")
    lines.append("renamed=0")
    lines.append("deleted=0")

    return "\n".join(lines)
