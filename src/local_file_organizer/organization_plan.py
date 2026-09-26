"""Read-only organization preview with confidence (V2.5.1).

This module only reads the existing SQLite catalog (tables ``files`` and
``person_cluster_members``) and builds an in-memory organization plan plus
a JSON report under ``reports/organization_plan.json``. It never walks the
filesystem, never opens a file from the scanned tree, and never moves,
copies, renames or deletes anything.

Confidence is expressed as transparent levels (``high``/``medium``/``low``)
derived from countable evidence plus the human decisions of V2.4.1. No
probability is ever invented, stored or displayed: the guiding principle is
to prefer a false negative (route to ``Revisión/``) over a false positive
(assign a file to the wrong person).
"""
from __future__ import annotations

import hashlib
import html
import json
import sqlite3
from pathlib import Path

from .face_review import (
    DECISION_CONFIRMED,
    DECISION_REJECTED,
    DECISION_UNCERTAIN,
)
from .person_identity import (
    combined_review_status,
    member_machine_labels,
    resolve_file_persons,
)

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

DESTINATION_SIN_PERSONAS = "Media/Sin_personas/"
DESTINATION_REVIEW_REJECTED = "Revisión/Cluster_rechazado/"
DESTINATION_REVIEW_UNCERTAIN = "Revisión/Persona_incierta/"
DESTINATION_REVIEW_LOW_EVIDENCE = "Revisión/Baja_evidencia/"
DESTINATION_REVIEW_METADATA = "Revisión/Metadata_incompleta/"
DESTINATION_REVIEW_SCAN = "Revisión/Errores_de_escaneo/"

_EMPTY_EVIDENCE = {
    "files": 0,
    "images": 0,
    "videos": 0,
    "observations": 0,
    "cross_media": False,
}


def _ext_parts(extension: str | None) -> tuple[str, str]:
    """Return (EXT_UPPER, ext_lower) stripped of the leading dot."""
    raw = (extension or "").strip()
    if raw in ("", "."):
        return "", ""
    lowered = raw.lower()
    if lowered.startswith("."):
        lowered = lowered[1:]
    if not lowered:
        return "", ""
    return lowered.upper(), lowered


def _fetch_person_sets(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """Map file_path -> set of distinct person_label values."""
    person_sets: dict[str, set[str]] = {}
    try:
        rows = conn.execute(
            "SELECT file_path, person_label FROM person_cluster_members"
        ).fetchall()
    except sqlite3.OperationalError:
        return person_sets
    for row in rows:
        person_sets.setdefault(row["file_path"], set()).add(row["person_label"])
    return person_sets


def person_evidence(conn: sqlite3.Connection) -> dict[str, dict]:
    """Aggregate countable evidence per effective person.

    Machine cluster rows are grouped by canonical identity (aliases), so
    merged clusters pool their evidence; manually assigned files count
    toward ``files``/``images``/``videos`` (human-verified) but never
    toward ``observations`` (no embedding behind them). With zero
    curation the output is identical to the pre-V3.1 machine-level
    version. Nothing is persisted.
    """
    evidence: dict[str, dict] = {}
    try:
        rows = conn.execute(
            "SELECT person_label, file_path, file_category, observation_count "
            "FROM person_cluster_members"
        ).fetchall()
    except sqlite3.OperationalError:
        return evidence
    try:
        aliases = {
            row["machine_label"]: row["person_id"]
            for row in conn.execute(
                "SELECT machine_label, person_id FROM person_aliases"
            )
        }
    except sqlite3.OperationalError:
        aliases = {}
    try:
        catalog_category = {
            row["path"]: row["category"]
            for row in conn.execute("SELECT path, category FROM files")
        }
    except sqlite3.OperationalError:
        catalog_category = {}
    manual_files: dict[str, set[str]] = {}
    try:
        for row in conn.execute(
            "SELECT file_path, person_id FROM file_person_links "
            "WHERE origin = 'manual'"
        ):
            manual_files.setdefault(row["person_id"], set()).add(row["file_path"])
    except sqlite3.OperationalError:
        pass

    grouped: dict[str, dict] = {}
    for row in rows:
        canonical = aliases.get(row["person_label"], row["person_label"])
        entry = grouped.setdefault(
            canonical,
            {"files": set(), "images": 0, "videos": 0, "observations": 0},
        )
        path = row["file_path"]
        if path not in entry["files"]:
            entry["files"].add(path)
            # file_category mirrors the catalog category for clustered
            # files; fall back to the catalog when they disagree.
            category = catalog_category.get(path, row["file_category"])
            if category == "image":
                entry["images"] += 1
            elif category == "video":
                entry["videos"] += 1
        entry["observations"] += row["observation_count"] or 0
    for person_id, paths in manual_files.items():
        entry = grouped.setdefault(
            person_id, {"files": set(), "images": 0, "videos": 0, "observations": 0}
        )
        for path in paths:
            if path in entry["files"]:
                continue
            entry["files"].add(path)
            category = catalog_category.get(path)
            if category == "image":
                entry["images"] += 1
            elif category == "video":
                entry["videos"] += 1
    for person_id, entry in grouped.items():
        images = entry["images"]
        videos = entry["videos"]
        evidence[person_id] = {
            "files": len(entry["files"]),
            "images": images,
            "videos": videos,
            "observations": entry["observations"],
            "cross_media": images > 0 and videos > 0,
        }
    return evidence


def person_confidence(evidence: dict, review_status: str | None) -> str:
    """Transparent confidence level for one person cluster.

    Human review always overrides the heuristic: ``confirmed`` -> ``high``,
    ``rejected``/``uncertain`` -> ``low``. Without a decision, countable
    evidence decides: 3+ files and 3+ observations -> ``high``, 2+ files ->
    ``medium``, anything else (singleton / weak evidence) -> ``low``.

    Embedding cohesion is deliberately not persisted (privacy, V2.4), so
    ``observations`` acts as a transparent proxy for evidence strength; no
    probability is ever shown or stored.
    """
    if review_status == DECISION_CONFIRMED:
        return CONFIDENCE_HIGH
    if review_status in (DECISION_REJECTED, DECISION_UNCERTAIN):
        return CONFIDENCE_LOW
    if evidence["files"] >= 3 and evidence["observations"] >= 3:
        return CONFIDENCE_HIGH
    if evidence["files"] >= 2:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


def build_organization_plan(
    conn: sqlite3.Connection, decisions: dict[str, str] | None = None
) -> list[dict]:
    """Build the full organization plan, one entry per cataloged file.

    ``decisions`` maps ``person_label -> review status`` (V2.4.1 human
    review); it overrides the heuristic confidence and can route a cluster
    to ``Revisión/``. File IDs (``F000001``, ...) are assigned by ascending
    ``path`` order and recomputed fresh on every call (never persisted).
    ``PAIR_000N`` / ``GROUP_000N`` labels are derived fresh each run from
    the sorted distinct person-sets of size exactly 2 / 3+ of all
    person-routable image/video files, independently of any decision, so
    the numbering stays deterministic and stable.
    """
    decisions = decisions or {}
    rows = conn.execute(
        "SELECT path, extension, category, scan_status, metadata_status,"
        " duplicate_group, similar_group FROM files "
        "WHERE COALESCE(file_state, 'active') = 'active' ORDER BY path ASC"
    ).fetchall()

    paths = [row["path"] for row in rows]
    file_ids = {path: f"F{i + 1:06d}" for i, path in enumerate(paths)}

    person_sets = resolve_file_persons(conn)
    evidence = person_evidence(conn)

    # First pass: base buckets + person-sets for person-routable media.
    buckets: dict[str, str] = {}
    person_tuple_by_path: dict[str, tuple[str, ...]] = {}
    for row in rows:
        path = row["path"]
        scan_status = row["scan_status"]
        category = row["category"]
        metadata_status = row["metadata_status"]
        if scan_status == "error":
            buckets[path] = "unsupported"
        elif (
            scan_status == "ok"
            and category in ("image", "video")
            and metadata_status == "error"
        ):
            buckets[path] = "review"
        else:
            buckets[path] = "planned"
        if buckets[path] == "planned" and category in ("image", "video"):
            labels = sorted(person_sets.get(path, set()))
            person_tuple_by_path[path] = tuple(labels)

    # Derive deterministic PAIR_ / GROUP_ labels from every ordered tuple.
    pair_tuples = sorted(
        {t for t in person_tuple_by_path.values() if len(t) == 2}
    )
    group_tuples = sorted(
        {t for t in person_tuple_by_path.values() if len(t) >= 3}
    )
    pair_labels = {t: f"PAIR_{i + 1:04d}" for i, t in enumerate(pair_tuples)}
    group_labels = {t: f"GROUP_{i + 1:04d}" for i, t in enumerate(group_tuples)}

    def _status_for(label: str) -> str | None:
        # V2.4.1 decisions are keyed by machine cluster; a canonical
        # person inherits its members' decisions conservatively
        # (rejected/uncertain win over confirmed).
        return combined_review_status(
            decisions, label, member_machine_labels(conn, label)
        )

    def _level_for(label: str) -> str:
        return person_confidence(
            evidence.get(label, _EMPTY_EVIDENCE), _status_for(label)
        )

    entries: list[dict] = []
    for row in rows:
        path = row["path"]
        category = row["category"]
        extension = row["extension"]
        bucket = buckets[path]
        flags: list[str] = []
        if row["duplicate_group"] is not None:
            flags.append("exact_duplicate")
        if row["similar_group"] is not None:
            flags.append("visual_similar_group")

        if bucket == "unsupported":
            destination = DESTINATION_REVIEW_SCAN
            reason_codes = ["scan_error"]
            confidence_level = CONFIDENCE_LOW
        elif bucket == "review":
            destination = DESTINATION_REVIEW_METADATA
            reason_codes = ["metadata_error"]
            confidence_level = CONFIDENCE_LOW
        elif category in ("image", "video"):
            person_tuple = person_tuple_by_path.get(path, ())
            n = len(person_tuple)
            statuses = {label: _status_for(label) for label in person_tuple}
            levels = {label: _level_for(label) for label in person_tuple}
            if n == 0:
                destination = DESTINATION_SIN_PERSONAS
                reason_codes = ["no_person_detected"]
                confidence_level = CONFIDENCE_HIGH
            elif n == 1:
                label = person_tuple[0]
                status = statuses[label]
                confidence_level = levels[label]
                if confidence_level in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM):
                    destination = f"Media/Personas/{label}/"
                    reason_codes = [f"single_person:{label}"]
                    if status == DECISION_CONFIRMED:
                        reason_codes.append("confirmed_person_cluster")
                elif status == DECISION_REJECTED:
                    destination = DESTINATION_REVIEW_REJECTED
                    reason_codes = ["rejected_person_cluster"]
                    bucket = "review"
                    confidence_level = CONFIDENCE_LOW
                elif status == DECISION_UNCERTAIN:
                    destination = DESTINATION_REVIEW_UNCERTAIN
                    reason_codes = ["uncertain_person_cluster"]
                    bucket = "review"
                    confidence_level = CONFIDENCE_LOW
                else:
                    destination = DESTINATION_REVIEW_LOW_EVIDENCE
                    reason_codes = [
                        "singleton_person_cluster",
                        "unreviewed_cluster",
                    ]
                    bucket = "review"
                    confidence_level = CONFIDENCE_LOW
            elif n == 2:
                pair_label = pair_labels[person_tuple]
                if any(s == DECISION_REJECTED for s in statuses.values()):
                    destination = DESTINATION_REVIEW_REJECTED
                    reason_codes = ["pair_contains_rejected_person"]
                    bucket = "review"
                    confidence_level = CONFIDENCE_LOW
                elif any(s == DECISION_UNCERTAIN for s in statuses.values()):
                    destination = DESTINATION_REVIEW_UNCERTAIN
                    reason_codes = ["pair_contains_uncertain_person"]
                    bucket = "review"
                    confidence_level = CONFIDENCE_LOW
                elif any(level == CONFIDENCE_LOW for level in levels.values()):
                    destination = DESTINATION_REVIEW_UNCERTAIN
                    reason_codes = ["pair_contains_low_evidence_person"]
                    bucket = "review"
                    confidence_level = CONFIDENCE_LOW
                else:
                    destination = f"Media/Parejas/{pair_label}/"
                    reason_codes = [f"pair:{pair_label}"]
                    if all(s == DECISION_CONFIRMED for s in statuses.values()):
                        reason_codes.append("confirmed_person_cluster")
                    confidence_level = (
                        CONFIDENCE_HIGH
                        if all(level == CONFIDENCE_HIGH for level in levels.values())
                        else CONFIDENCE_MEDIUM
                    )
            else:
                group_label = group_labels[person_tuple]
                if any(s == DECISION_REJECTED for s in statuses.values()):
                    destination = DESTINATION_REVIEW_REJECTED
                    reason_codes = ["group_contains_rejected_person"]
                    bucket = "review"
                    confidence_level = CONFIDENCE_LOW
                elif any(s == DECISION_UNCERTAIN for s in statuses.values()):
                    destination = DESTINATION_REVIEW_UNCERTAIN
                    reason_codes = ["group_contains_uncertain_person"]
                    bucket = "review"
                    confidence_level = CONFIDENCE_LOW
                elif any(level == CONFIDENCE_LOW for level in levels.values()):
                    destination = DESTINATION_REVIEW_UNCERTAIN
                    reason_codes = ["group_contains_low_evidence_person"]
                    bucket = "review"
                    confidence_level = CONFIDENCE_LOW
                else:
                    destination = f"Media/Grupos/{group_label}/"
                    reason_codes = [f"group:{group_label}"]
                    if all(s == DECISION_CONFIRMED for s in statuses.values()):
                        reason_codes.append("confirmed_person_cluster")
                    confidence_level = (
                        CONFIDENCE_HIGH
                        if all(level == CONFIDENCE_HIGH for level in levels.values())
                        else CONFIDENCE_MEDIUM
                    )
        elif category == "document":
            ext_upper, ext_lower = _ext_parts(extension)
            if not ext_upper:
                destination = "Documentos/Sin_extension/"
                reason_codes = ["no_extension"]
            else:
                destination = f"Documentos/{ext_upper}/"
                reason_codes = [f"extension:{ext_lower}"]
            confidence_level = CONFIDENCE_HIGH
        elif category == "audio":
            ext_upper, ext_lower = _ext_parts(extension)
            if not ext_upper:
                destination = "Audio/Sin_extension/"
                reason_codes = ["no_extension"]
            else:
                destination = f"Audio/{ext_upper}/"
                reason_codes = [f"extension:{ext_lower}"]
            confidence_level = CONFIDENCE_HIGH
        elif category == "archive":
            _ext_upper, ext_lower = _ext_parts(extension)
            destination = "Archivos_comprimidos/"
            reason_codes = [f"extension:{ext_lower}"]
            confidence_level = CONFIDENCE_HIGH
        elif category == "software":
            _ext_upper, ext_lower = _ext_parts(extension)
            destination = "Software/"
            reason_codes = [f"extension:{ext_lower}"]
            confidence_level = CONFIDENCE_HIGH
        else:  # 'other' and any unknown category
            ext_upper, ext_lower = _ext_parts(extension)
            if not ext_upper:
                destination = "Otros/Sin_extension/"
                reason_codes = ["no_extension"]
            else:
                destination = f"Otros/{ext_upper}/"
                reason_codes = [f"extension:unknown:{ext_lower}"]
            confidence_level = CONFIDENCE_MEDIUM

        entries.append(
            {
                "file_id": file_ids[path],
                "category": category,
                "bucket": bucket,
                "proposed_destination": destination,
                "reason_codes": reason_codes,
                "flags": flags,
                "confidence_level": confidence_level,
                "action": "preview_only",
                "src_hash": hashlib.sha256(path.encode("utf-8")).hexdigest(),
            }
        )
    return entries


def write_plan_json(entries: list[dict], plan_path: Path) -> None:
    """Write the plan entries to ``plan_path`` as a JSON array."""
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Static offline plan viewer (T1)
# ---------------------------------------------------------------------------

VIEW_DIR_NAME = "organization_view"
VIEW_INDEX_FILENAME = "index.html"

CONFIDENCE_LEVELS = (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
)


def build_plan_view_rows(
    entries: list[dict],
    persons_by_file_id: dict[str, list[str]] | None = None,
    paths_by_file_id: dict[str, str] | None = None,
) -> list[dict]:
    """Derive viewer rows from plan entries plus optional overlays.

    The plan entries themselves never carry persons or real paths (privacy:
    ``organization_plan.json`` only holds opaque ``F000NNN`` ids). The
    ``persons`` facet comes from the caller-supplied overlay (resolved from
    the catalog at preview time), and ``path`` is only attached when the
    caller explicitly passes ``paths_by_file_id`` (manual ``--show-files``
    runs). With both overlays omitted the rows contain no personal data
    beyond opaque ids and proposed destinations.
    """
    persons_by_file_id = persons_by_file_id or {}
    rows = []
    for entry in entries:
        file_id = entry["file_id"]
        persons = sorted(set(persons_by_file_id.get(file_id, ())))
        path = (paths_by_file_id or {}).get(file_id)
        rows.append(
            {
                "file_id": file_id,
                "category": entry["category"],
                "bucket": entry["bucket"],
                "destination": entry["proposed_destination"],
                "confidence": entry["confidence_level"],
                "persons": persons,
                "flags": list(entry["flags"]),
                "reasons": list(entry["reason_codes"]),
                "path": path,
            }
        )
    return rows


_VIEW_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 1rem 3rem;
  font-family: system-ui, sans-serif;
  background: #f5f5f5; color: #1c1c1c;
}
header.site { padding: 1rem 0; }
h1 { font-size: 1.4rem; margin: 0 0 0.25rem; }
.local-only {
  display: inline-block; margin: 0 0 0.75rem; padding: 0.2rem 0.5rem;
  background: #1c1c1c; color: #fff; font-size: 0.8rem;
  border-radius: 4px; letter-spacing: 0.02em;
}
dl.summary { display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; margin: 0 0 0.75rem; }
dl.summary div { margin: 0; }
dl.summary dt { font-size: 0.7rem; text-transform: uppercase; color: #555; }
dl.summary dd { margin: 0; font-size: 1.1rem; font-weight: 600; }
form.filters {
  display: flex; gap: 1rem; flex-wrap: wrap; align-items: end;
  background: #fff; border: 1px solid #ddd; border-radius: 6px;
  padding: 0.75rem; margin: 0 0 1rem;
}
form.filters label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.8rem; color: #444; }
form.filters select { padding: 0.3rem 0.5rem; font-size: 0.85rem; max-width: 22rem; }
button { padding: 0.35rem 0.7rem; border: 1px solid #999; border-radius: 4px; background: #fff; cursor: pointer; font-size: 0.85rem; }
#match-line { font-size: 0.85rem; color: #444; }
table { width: 100%; border-collapse: collapse; background: #fff; font-size: 0.82rem; }
th, td { text-align: left; padding: 0.35rem 0.5rem; border-bottom: 1px solid #e5e5e5; vertical-align: top; }
thead th { background: #ececec; position: sticky; top: 0; }
td.id { font-family: monospace; white-space: nowrap; }
td.dest { font-family: monospace; }
td.reasons { font-family: monospace; color: #444; }
"""

_VIEW_JS = """
(function () {
  "use strict";
  var destSelect = document.getElementById("filter-destination");
  var personSelect = document.getElementById("filter-person");
  var confidenceSelect = document.getElementById("filter-confidence");
  var matchCount = document.getElementById("match-count");
  var rows = Array.prototype.slice.call(document.querySelectorAll("tbody tr"));
  function personsOf(row) {
    var raw = row.getAttribute("data-persons") || "";
    return raw.split(" ").filter(function (p) { return p.length > 0; });
  }
  function applyFilters() {
    var dest = destSelect.value;
    var person = personSelect.value;
    var confidence = confidenceSelect.value;
    var visible = 0;
    rows.forEach(function (row) {
      var show = true;
      if (dest !== "all" && row.getAttribute("data-destination") !== dest) { show = false; }
      if (show && confidence !== "all" && row.getAttribute("data-confidence") !== confidence) { show = false; }
      if (show && person !== "all") {
        var persons = personsOf(row);
        if (person === "none") {
          if (persons.length !== 0) { show = false; }
        } else if (persons.indexOf(person) === -1) { show = false; }
      }
      row.style.display = show ? "" : "none";
      if (show) { visible += 1; }
    });
    matchCount.textContent = String(visible);
  }
  [destSelect, personSelect, confidenceSelect].forEach(function (el) {
    el.addEventListener("change", applyFilters);
  });
  document.getElementById("reset-filters").addEventListener("click", function () {
    destSelect.value = "all";
    personSelect.value = "all";
    confidenceSelect.value = "all";
    applyFilters();
  });
  applyFilters();
})();
"""


def _view_option(value: str, label: str, selected: bool = False) -> str:
    selected_attr = " selected" if selected else ""
    return f'<option value="{html.escape(value)}"{selected_attr}>{html.escape(label)}</option>'


def _render_filter_bar(rows: list[dict]) -> str:
    """Render the destination/person/confidence filter controls."""
    destinations = sorted({row["destination"] for row in rows})
    persons = sorted({label for row in rows for label in row["persons"]})
    dest_options = [_view_option("all", "All destinations", True)]
    dest_options += [_view_option(d, d) for d in destinations]
    person_options = [_view_option("all", "All persons", True)]
    person_options.append(_view_option("none", "No person assigned"))
    person_options += [_view_option(label, label) for label in persons]
    confidence_options = [_view_option("all", "All confidence levels", True)]
    confidence_options += [_view_option(level, level) for level in CONFIDENCE_LEVELS]
    return (
        '<form class="filters" id="filters">\n'
        '  <label>Destination\n'
        f'    <select id="filter-destination">\n      {" ".join(dest_options)}\n    </select>\n'
        "  </label>\n"
        "  <label>Person\n"
        f'    <select id="filter-person">\n      {" ".join(person_options)}\n    </select>\n'
        "  </label>\n"
        "  <label>Confidence\n"
        f'    <select id="filter-confidence">\n      {" ".join(confidence_options)}\n    </select>\n'
        "  </label>\n"
        '  <button type="button" id="reset-filters">Reset filters</button>\n'
        f'  <span id="match-line"><span id="match-count">{len(rows)}</span>'
        f" of {len(rows)} shown</span>\n"
        "</form>"
    )


def _render_view_rows(rows: list[dict], include_paths: bool) -> str:
    """Render one table row per plan entry with filterable data attributes."""
    rendered = []
    for row in rows:
        persons = row["persons"]
        persons_attr = " ".join(persons)
        persons_cell = ", ".join(persons) if persons else "-"
        flags_cell = ", ".join(row["flags"]) if row["flags"] else "-"
        reasons_cell = ", ".join(row["reasons"]) if row["reasons"] else "-"
        cells = (
            f'<td class="id">{html.escape(row["file_id"])}</td>'
            f"<td>{html.escape(row['category'])}</td>"
            f'<td class="dest">{html.escape(row["destination"])}</td>'
            f"<td>{html.escape(row['confidence'])}</td>"
            f"<td>{html.escape(persons_cell)}</td>"
            f"<td>{html.escape(flags_cell)}</td>"
            f'<td class="reasons">{html.escape(reasons_cell)}</td>'
        )
        if include_paths:
            cells += f"<td>{html.escape(row['path'] or '-')}</td>"
        rendered.append(
            f'<tr data-file-id="{html.escape(row["file_id"])}" '
            f'data-destination="{html.escape(row["destination"])}" '
            f'data-confidence="{html.escape(row["confidence"])}" '
            f'data-persons="{html.escape(persons_attr)}">'
            f"{cells}</tr>"
        )
    return "\n".join(rendered)


def _render_view_summary(counts: dict) -> str:
    """Render aggregate counts (destinations, confidence, accounting)."""
    confidence = counts.get("confidence_levels", {})
    fields = [
        ("files", counts.get("catalog_files", 0)),
        ("planned", counts.get("planned", 0)),
        ("review", counts.get("review", 0)),
        ("unsupported", counts.get("unsupported", 0)),
        ("high", confidence.get(CONFIDENCE_HIGH, 0)),
        ("medium", confidence.get(CONFIDENCE_MEDIUM, 0)),
        ("low", confidence.get(CONFIDENCE_LOW, 0)),
    ]
    items = "\n      ".join(
        f"<div><dt>{html.escape(name)}</dt><dd>{value}</dd></div>"
        for name, value in fields
    )
    return f'<dl class="summary">\n      {items}\n  </dl>'


def render_plan_view_html(
    entries: list[dict],
    counts: dict,
    persons_by_file_id: dict[str, list[str]] | None = None,
    paths_by_file_id: dict[str, str] | None = None,
) -> str:
    """Render the fully static, offline plan viewer page (deterministic).

    The page inlines its CSS/JS, references no remote resources, and makes
    no network calls: filtering runs locally in the browser over the rows
    rendered into the table. Real paths only appear when the caller passes
    ``paths_by_file_id`` (manual ``--show-files`` runs); otherwise the page
    holds opaque ``F000NNN`` ids only.
    """
    rows = build_plan_view_rows(entries, persons_by_file_id, paths_by_file_id)
    include_paths = paths_by_file_id is not None
    headers = (
        "<th>File</th><th>Category</th><th>Destination</th>"
        "<th>Confidence</th><th>Persons</th><th>Flags</th><th>Reasons</th>"
    )
    if include_paths:
        headers += "<th>Path (manual view only)</th>"
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Organization Plan Viewer</title>\n"
        f"<style>{_VIEW_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        '<header class="site">\n'
        "  <h1>Organization Plan Viewer</h1>\n"
        "  <p class=\"local-only\">LOCAL ONLY — offline static view, no network</p>\n"
        f"  {_render_view_summary(counts)}\n"
        f"  {_render_filter_bar(rows)}\n"
        "</header>\n"
        "<main>\n"
        "<table>\n"
        f"<thead><tr>{headers}</tr></thead>\n"
        "<tbody>\n"
        f"{_render_view_rows(rows, include_paths)}\n"
        "</tbody>\n"
        "</table>\n"
        "</main>\n"
        "<script>\n"
        f"{_VIEW_JS}"
        "</script>\n"
        "</body>\n"
        "</html>\n"
    )


def write_plan_view_site(
    entries: list[dict],
    counts: dict,
    output_dir: Path,
    persons_by_file_id: dict[str, list[str]] | None = None,
    paths_by_file_id: dict[str, str] | None = None,
) -> Path:
    """Write ``index.html`` (the offline plan viewer) under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / VIEW_INDEX_FILENAME
    with open(index_path, "w", encoding="utf-8") as handle:
        handle.write(
            render_plan_view_html(
                entries, counts, persons_by_file_id, paths_by_file_id
            )
        )
    return index_path


def plan_counts(entries: list[dict]) -> dict[str, int]:
    """Aggregate bucket/flag/confidence counts plus per-destination counts."""
    by_destination: dict[str, int] = {}
    confidence_levels = {
        CONFIDENCE_HIGH: 0,
        CONFIDENCE_MEDIUM: 0,
        CONFIDENCE_LOW: 0,
    }
    for entry in entries:
        dest = entry["proposed_destination"]
        by_destination[dest] = by_destination.get(dest, 0) + 1
        level = entry["confidence_level"]
        confidence_levels[level] = confidence_levels.get(level, 0) + 1
    planned = sum(1 for e in entries if e["bucket"] == "planned")
    review = sum(1 for e in entries if e["bucket"] == "review")
    unsupported = sum(1 for e in entries if e["bucket"] == "unsupported")
    exact_duplicates = sum(1 for e in entries if "exact_duplicate" in e["flags"])
    visual_similar = sum(
        1 for e in entries if "visual_similar_group" in e["flags"]
    )
    return {
        "catalog_files": len(entries),
        "planned": planned,
        "review": review,
        "unsupported": unsupported,
        "exact_duplicates": exact_duplicates,
        "visual_similar_groups": visual_similar,
        "by_destination": by_destination,
        "confidence_levels": confidence_levels,
    }
