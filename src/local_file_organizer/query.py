"""Structured catalog search (V3.2).

A filter is a plain versioned dict so the *same* language can later be
persisted by smart collections (V3.3) and evaluated by the rule engine
(V3.4) without rewriting anything::

    {"version": 1, "and": [
        {"field": "category", "op": "=", "value": "video"},
        {"field": "person", "op": "=", "value": "PERSON_0001"},
        {"field": "duration_seconds", "op": ">", "value": 600},
    ]}

Every flag narrows (AND). Repeating a filter (e.g. two ``person``
conditions) means the file must satisfy ALL of them — that is also how
co-occurrence queries ("files showing A and B") are expressed.

Safety rules:

- Only whitelisted fields/ops ever reach SQL; everything else raises
  ``ValueError``. All values travel as bound parameters — hostile input
  such as ``x' OR '1'='1`` can only ever match zero rows, never escape.
- Person matching: ``P000N`` matches the raw machine cluster (exact
  calibration/review unit); ``PERSON_0007`` and display names match the
  V3.1 effective identity (aliases + manual links minus exclusions),
  computed in SQL via EXISTS — so one file with two persons never
  duplicates result rows.
- File IDs (``F000123``) are assigned by ascending catalog ``path`` over
  the *whole* catalog (``ROW_NUMBER``), identical to the IDs in
  ``reports/organization_plan.json`` — the two outputs stay comparable.
- Results never include real paths unless the caller formats them in
  (the CLI only does so under ``--show-files``).
"""
from __future__ import annotations

import re
import sqlite3

from . import database

QUERY_VERSION = 1

# field -> (column expression, value kind). "special" fields are handled
# by dedicated builders below instead of a direct column comparison.
_FIELD_COLUMNS = {
    "category": ("f.category", "text"),
    "extension": ("f.extension", "text"),
    "size_bytes": ("f.size_bytes", "number"),
    "duration_seconds": ("f.duration_seconds", "number"),
    "width": ("f.width", "number"),
    "height": ("f.height", "number"),
    "capture_date": ("f.capture_date", "text"),
    "capture_year": ("substr(f.capture_date, 1, 4)", "text"),
    "device_make": ("f.device_make", "text"),
    "device_model": ("f.device_model", "text"),
    "scan_status": ("f.scan_status", "text"),
    "metadata_status": ("f.metadata_status", "text"),
}

_SIMPLE_OPS = {"=", "!=", ">", ">=", "<", "<=", "like"}
_SPECIAL_FIELDS = {
    "person",
    "device",  # make OR model, case-insensitive exact
    "has_gps",
    "has_device",
    "has_capture_date",
    "exact_duplicate",
    "visual_similar",
    "gps_bbox",  # [lat_min, lat_max, lon_min, lon_max]
}

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([bkmgtp])?b?\s*$", re.IGNORECASE)
_SIZE_MULT = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}


def parse_size(raw: str) -> int:
    """Parse '500MB'/'1.5g'/1024 into bytes. Raises ValueError on garbage."""
    if isinstance(raw, (int, float)):
        value = int(raw)
        if value < 0:
            raise ValueError(f"tamaño inválido: {raw!r}")
        return value
    match = _SIZE_RE.match(str(raw or ""))
    if not match:
        raise ValueError(f"tamaño inválido: {raw!r} (ejemplos: 500MB, 1.5G, 1024)")
    number, suffix = match.groups()
    return int(float(number) * _SIZE_MULT[(suffix or "b").lower()])


def parse_bbox(raw: str) -> tuple[float, float, float, float]:
    """Parse 'lat_min,lat_max,lon_min,lon_max'. Raises ValueError."""
    try:
        lat_min, lat_max, lon_min, lon_max = (float(p) for p in str(raw).split(","))
    except ValueError:
        raise ValueError(
            f"bbox inválida: {raw!r} (formato: lat_min,lat_max,lon_min,lon_max)"
        )
    if not (lat_min <= lat_max and lon_min <= lon_max):
        raise ValueError(f"bbox inválida (mínimos > máximos): {raw!r}")
    if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        raise ValueError(f"latitud fuera de rango: {raw!r}")
    if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
        raise ValueError(f"longitud fuera de rango: {raw!r}")
    return (lat_min, lat_max, lon_min, lon_max)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _effective_persons_cte(conn: sqlite3.Connection) -> str:
    """CTE fragment resolving V3.1 effective persons in SQL. Missing
    tables (hand-made legacy DBs) degrade to empty inputs, never to an
    error."""
    if _table_exists(conn, "person_cluster_members"):
        auto = (
            "SELECT m.file_path AS file_path, "
            "COALESCE(a.person_id, m.person_label) AS person_id "
            "FROM person_cluster_members m "
            "LEFT JOIN person_aliases a ON a.machine_label = m.person_label"
        )
        if not _table_exists(conn, "person_aliases"):
            auto = (
                "SELECT file_path, person_label AS person_id "
                "FROM person_cluster_members"
            )
    else:
        auto = "SELECT '' AS file_path, '' AS person_id WHERE 0"
    if _table_exists(conn, "file_person_links"):
        manual = (
            "SELECT file_path, person_id FROM file_person_links "
            "WHERE origin = 'manual'"
        )
        excluded = (
            "SELECT file_path, person_id FROM file_person_links "
            "WHERE origin = 'excluded'"
        )
    else:
        manual = "SELECT '' AS file_path, '' AS person_id WHERE 0"
        excluded = "SELECT '' AS file_path, '' AS person_id WHERE 0"
    return (
        "effective_persons(file_path, person_id) AS ("
        f"SELECT file_path, person_id FROM ({auto}) "
        f"UNION SELECT file_path, person_id FROM ({manual}) "
        f"EXCEPT SELECT file_path, person_id FROM ({excluded}))"
    )


def _resolve_person_ref(conn: sqlite3.Connection, ref: str) -> tuple[str, str]:
    """Accept P000N / PERSON_0007 / display name; return (kind, id) where
    kind is 'machine' (match raw cluster rows) or 'canonical' (match the
    effective V3.1 identity). Display names must match exactly one
    person (NOCASE)."""
    from .person_identity import is_canonical_id, is_machine_label

    if is_machine_label(ref):
        if ref in database.fetch_machine_labels(conn):
            return ("machine", ref)
        raise ValueError(f"cluster facial desconocido: {ref}")
    if is_canonical_id(ref):
        if _table_exists(conn, "persons") and database.fetch_person(conn, ref) is not None:
            return ("canonical", ref)
        raise ValueError(f"persona canónica desconocida: {ref}")
    if not _table_exists(conn, "persons"):
        raise ValueError(f"persona desconocida: {ref!r}")
    rows = conn.execute(
        "SELECT person_id FROM persons WHERE display_name = ? COLLATE NOCASE",
        (ref,),
    ).fetchall()
    if len(rows) == 1:
        return ("canonical", rows[0]["person_id"])
    if not rows:
        raise ValueError(
            f"persona desconocida: {ref!r} (usa P000N, PERSON_0007 o un nombre único)"
        )
    raise ValueError(f"nombre ambiguo: {ref!r} ({len(rows)} personas)")


def _condition_sql(conn: sqlite3.Connection, cond: dict) -> tuple[str, list]:
    """Compile one condition to (SQL fragment, params). Raises ValueError
    on anything outside the whitelist."""
    if not isinstance(cond, dict):
        raise ValueError(f"condición inválida: {cond!r}")
    field = cond.get("field")
    op = cond.get("op", "=")
    value = cond.get("value")

    if field in _FIELD_COLUMNS:
        column, kind = _FIELD_COLUMNS[field]
        if op not in _SIMPLE_OPS and op != "in":
            raise ValueError(f"operador inválido para {field}: {op!r}")
        if op == "in":
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f"{field} in requiere una lista no vacía")
            if kind == "number":
                try:
                    value = [float(v) for v in value]
                except (TypeError, ValueError):
                    raise ValueError(f"{field} in requiere números: {value!r}")
            placeholders = ", ".join("?" for _ in value)
            return f"{column} IN ({placeholders})", list(value)
        if value is None:
            raise ValueError(f"{field} requiere un valor (usa is_null si buscas vacíos)")
        if kind == "number":
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{field} requiere un número, no {value!r}")
        elif op == "like" and not isinstance(value, str):
            raise ValueError(f"{field} like requiere texto")
        return f"{column} {op} ?", [value]

    if field not in _SPECIAL_FIELDS:
        known = sorted(_FIELD_COLUMNS) + sorted(_SPECIAL_FIELDS)
        raise ValueError(f"campo desconocido: {field!r} (conocidos: {', '.join(known)})")

    if field == "person":
        if op != "=" or not isinstance(value, str) or not value:
            raise ValueError("person requiere op '=' y una referencia no vacía")
        kind, person_id = _resolve_person_ref(conn, value)
        if kind == "machine":
            if not _table_exists(conn, "person_cluster_members"):
                raise ValueError(f"cluster facial desconocido: {value}")
            return (
                "EXISTS (SELECT 1 FROM person_cluster_members m "
                "WHERE m.file_path = f.path AND m.person_label = ?)",
                [person_id],
            )
        return (
            "EXISTS (SELECT 1 FROM effective_persons e "
            "WHERE e.file_path = f.path AND e.person_id = ?)",
            [person_id],
        )
    if field == "device":
        if op != "=" or not isinstance(value, str) or not value:
            raise ValueError("device requiere op '=' y texto no vacío")
        return (
            "(f.device_make = ? COLLATE NOCASE OR f.device_model = ? COLLATE NOCASE)",
            [value, value],
        )
    if field in ("has_gps", "has_device", "has_capture_date", "exact_duplicate", "visual_similar"):
        if op not in ("=", "!=") or not isinstance(value, bool):
            raise ValueError(f"{field} requiere op '='/'!=' y true/false")
        fragments = {
            "has_gps": "f.gps_latitude IS NOT NULL AND f.gps_longitude IS NOT NULL",
            "has_device": "f.device_make IS NOT NULL OR f.device_model IS NOT NULL",
            "has_capture_date": "f.capture_date IS NOT NULL",
            "exact_duplicate": "f.duplicate_group IS NOT NULL",
            "visual_similar": "f.similar_group IS NOT NULL",
        }
        expr = fragments[field]
        false_expr = {
            "has_gps": "NOT (f.gps_latitude IS NOT NULL AND f.gps_longitude IS NOT NULL)",
            "has_device": "f.device_make IS NULL AND f.device_model IS NULL",
            "has_capture_date": "f.capture_date IS NULL",
            "exact_duplicate": "f.duplicate_group IS NULL",
            "visual_similar": "f.similar_group IS NULL",
        }[field]
        if (op == "=") == value:
            return expr, []
        return false_expr, []
    if field == "gps_bbox":
        if op != "=":
            raise ValueError("gps_bbox solo admite op '='")
        lat_min, lat_max, lon_min, lon_max = parse_bbox(value) if isinstance(value, str) else value
        try:
            lat_min, lat_max, lon_min, lon_max = (
                float(lat_min),
                float(lat_max),
                float(lon_min),
                float(lon_max),
            )
        except (TypeError, ValueError):
            raise ValueError(f"gps_bbox inválida: {value!r}")
        return (
            "f.gps_latitude BETWEEN ? AND ? AND f.gps_longitude BETWEEN ? AND ?",
            [lat_min, lat_max, lon_min, lon_max],
        )
    raise ValueError(f"campo no implementado: {field!r}")  # pragma: no cover


_SORTABLE = {"path", "size_bytes", "capture_date", "duration_seconds", "category"}


def build_sql(
    conn: sqlite3.Connection,
    filters: dict,
    order_by: str = "path",
    descending: bool = False,
    limit: int = 100,
) -> tuple[str, list]:
    """Compile a v1 filter dict to (SQL, params). Raises ValueError."""
    if not isinstance(filters, dict) or filters.get("version") != QUERY_VERSION:
        raise ValueError(f"filtros con versión desconocida: {filters!r}")
    conditions = filters.get("and", [])
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("filtros vacíos: 'and' requiere al menos 1 condición")
    if order_by not in _SORTABLE:
        raise ValueError(f"orden inválido: {order_by!r} (usa: {', '.join(sorted(_SORTABLE))})")
    if not isinstance(limit, int) or limit < 1:
        raise ValueError(f"límite inválido: {limit!r}")

    fragments: list[str] = []
    params: list = []
    for cond in conditions:
        fragment, cond_params = _condition_sql(conn, cond)
        fragments.append(f"({fragment})")
        params.extend(cond_params)
    # F1: stale rows (seen before, missing now) never match searches.
    fragments.append("(COALESCE(f.file_state, 'active') = 'active')")
    where = " AND ".join(fragments)
    direction = "DESC" if descending else "ASC"
    # Two levels: the inner query numbers EVERY catalog row by path (so
    # file_ids stay stable vs organize-preview), the outer one filters.
    # The outer alias is deliberately `f` so all condition fragments
    # (including the person EXISTS correlation on f.path) just work.
    sql = (
        f"WITH {_effective_persons_cte(conn)} "
        "SELECT f.* FROM (SELECT c.*, "
        "ROW_NUMBER() OVER (ORDER BY c.path) AS file_num FROM files c) AS f "
        f"WHERE {where} ORDER BY f.{order_by} {direction} LIMIT {limit}"
    )
    return sql, params


def count_sql(conn: sqlite3.Connection, filters: dict) -> tuple[str, list]:
    """Same WHERE as build_sql but COUNT(*). Raises ValueError."""
    sql, params = build_sql(conn, filters)
    inner = sql.rsplit(" ORDER BY f.", 1)[0]
    return f"SELECT COUNT(*) FROM ({inner}) AS counted", params


def run_query(
    conn: sqlite3.Connection,
    filters: dict,
    order_by: str = "path",
    descending: bool = False,
    limit: int = 100,
) -> tuple[list[dict], int]:
    """Execute filters; return (rows, total). Each row carries file_id
    (F000123, estable entre search y organize-preview) plus the catalog
    columns. Raises ValueError on bad filters."""
    sql, params = build_sql(conn, filters, order_by, descending, limit)
    rows = [
        dict(row, file_id=f"F{row['file_num']:06d}")
        for row in conn.execute(sql, params).fetchall()
    ]
    total_sql, total_params = count_sql(conn, filters)
    total = conn.execute(total_sql, total_params).fetchone()[0]
    return rows, total


def cli_filters(
    category: str | None = None,
    extension: str | None = None,
    persons: list[str] | None = None,
    device: str | None = None,
    year: str | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    min_size: str | None = None,
    max_size: str | None = None,
    has_gps: bool = False,
    has_device: bool = False,
    has_date: bool = False,
    bbox: str | None = None,
    duplicates: bool = False,
    similar: bool = False,
) -> dict:
    """Map CLI flags to a v1 filter dict (all AND). Raises ValueError."""
    conditions: list[dict] = []

    def _multi(raw: str | None, field: str, lower: bool = False) -> None:
        if not raw:
            return
        values = [v.strip() for v in str(raw).split(",") if v.strip()]
        if not values:
            return
        if lower:
            values = [v if v.startswith(".") else f".{v}" for v in values]
            values = [v.lower() for v in values]
        if len(values) == 1:
            conditions.append({"field": field, "op": "=", "value": values[0]})
        else:
            # OR within one field ("video OR audio"), AND with the rest.
            conditions.append({"field": field, "op": "in", "value": values})

    _multi(category, "category")
    _multi(extension, "extension", lower=True)
    for ref in persons or []:
        conditions.append({"field": "person", "op": "=", "value": ref})
    if device:
        conditions.append({"field": "device", "op": "=", "value": device})
    if year:
        if not (isinstance(year, str) and year.isdigit() and len(year) == 4):
            raise ValueError(f"año inválido: {year!r} (formato: 2025)")
        conditions.append({"field": "capture_year", "op": "=", "value": year})
    if min_duration is not None:
        conditions.append({"field": "duration_seconds", "op": ">=", "value": min_duration})
    if max_duration is not None:
        conditions.append({"field": "duration_seconds", "op": "<=", "value": max_duration})
    if min_size is not None:
        conditions.append({"field": "size_bytes", "op": ">=", "value": parse_size(min_size)})
    if max_size is not None:
        conditions.append({"field": "size_bytes", "op": "<=", "value": parse_size(max_size)})
    if has_gps:
        conditions.append({"field": "has_gps", "op": "=", "value": True})
    if has_device:
        conditions.append({"field": "has_device", "op": "=", "value": True})
    if has_date:
        conditions.append({"field": "has_capture_date", "op": "=", "value": True})
    if bbox:
        conditions.append({"field": "gps_bbox", "op": "=", "value": parse_bbox(bbox)})
    if duplicates:
        conditions.append({"field": "exact_duplicate", "op": "=", "value": True})
    if similar:
        conditions.append({"field": "visual_similar", "op": "=", "value": True})
    if not conditions:
        raise ValueError("search requiere al menos 1 filtro")
    return {"version": QUERY_VERSION, "and": conditions}
