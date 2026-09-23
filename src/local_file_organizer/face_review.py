"""Local face-cluster calibration review (V2.4.1).

Turns the anonymous person clusters produced by V2.4 into a fully local,
offline visual review: a deterministic manifest of preview samples (no
real paths, only opaque file/observation ids), local JPEG thumbnails under
the review output directory, a static HTML page with no remote resources
of any kind, and a human-decisions JSON file that the CLI can export and
re-import.

Everything here is stdlib + Pillow. The scanned tree is only ever read
(images decoded into memory, video frames sampled ephemerally); nothing is
written inside it, and no network call is ever made.
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
from pathlib import Path
from typing import Callable

from PIL import Image

from .video_sampling import sample_video_frames

DECISION_CONFIRMED = "confirmed"
DECISION_REJECTED = "rejected"
DECISION_UNCERTAIN = "uncertain"
DECISION_UNREVIEWED = "unreviewed"
ALLOWED_DECISIONS = frozenset(
    {DECISION_CONFIRMED, DECISION_REJECTED, DECISION_UNCERTAIN}
)
DECISIONS_VERSION = 1
DEFAULT_MAX_PREVIEWS_PER_CLUSTER = 6
DEFAULT_THUMBNAIL_MAX_PX = 192
MANIFEST_VERSION = 1

CROP_MARGIN_RATIO = 0.30
JPEG_QUALITY = 85
_TIMESTAMP_TOLERANCE_SECONDS = 1e-6


# ---------------------------------------------------------------------------
# Human decisions (local JSON)
# ---------------------------------------------------------------------------


def parse_decisions_payload(payload) -> dict[str, str]:
    """Validate a decisions payload and return its decisions mapping.

    Accepts ``{"version": 1, "decisions": {...}}``. A ``version`` key inside
    the decisions mapping itself is ignored (some exporters include it
    twice). Raises ``ValueError`` with every problem found, not just the
    first one.
    """
    if not isinstance(payload, dict):
        raise ValueError("el JSON de decisiones debe ser un objeto")
    errors: list[str] = []
    version = payload.get("version")
    if version != DECISIONS_VERSION:
        errors.append(
            f"version debe ser {DECISIONS_VERSION} (encontrado: {version!r})"
        )
    decisions = payload.get("decisions")
    if not isinstance(decisions, dict):
        errors.append("decisions debe ser un objeto")
        decisions = {}
    else:
        for label, value in decisions.items():
            if label == "version":
                continue
            if not isinstance(label, str):
                errors.append(f"label inválido: {label!r}")
            if value not in ALLOWED_DECISIONS:
                errors.append(f"estado inválido para {label!r}: {value!r}")
    if errors:
        raise ValueError("; ".join(errors))
    return {label: value for label, value in decisions.items() if label != "version"}


def load_decisions(path: Path) -> dict[str, str]:
    """Load decisions from a local JSON file; missing file means no decisions.

    The file's labels are NOT checked against the database here (the
    database may change between runs); that strict check belongs to the
    import flow, which validates against current cluster labels.
    """
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"no se pudo leer {path}: {exc}") from exc
    return parse_decisions_payload(payload)


def validate_decisions(
    decisions: dict[str, str], known_labels: set[str]
) -> list[str]:
    """Return one error per decision whose label is not a current cluster."""
    errors = [
        f"label desconocido en el catálogo: {label}"
        for label in sorted(decisions)
        if label not in known_labels
    ]
    return errors


def save_decisions(path: Path, decisions: dict[str, str]) -> None:
    """Atomically write decisions as local JSON (tmp file + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": DECISIONS_VERSION, "decisions": decisions}
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Manifest and preview selection
# ---------------------------------------------------------------------------


def _normalize_source_type(source_type: str) -> str:
    return "video" if source_type == "video_frame" else source_type


def _farthest_point_order(count: int) -> list[int]:
    """Deterministic farthest-point ordering over list indices.

    Starts at index 0, then repeatedly picks the index whose distance to
    the already-chosen set is largest (ties go to the lowest index). This
    spreads video frames across time instead of picking consecutive ones.
    """
    if count <= 1:
        return list(range(count))
    order = [0]
    remaining = list(range(1, count))
    while remaining:
        best_index = remaining[0]
        best_distance = -1
        for index in remaining:
            distance = min(abs(index - chosen) for chosen in order)
            if distance > best_distance:
                best_distance = distance
                best_index = index
        order.append(best_index)
        remaining.remove(best_index)
    return order


def select_samples(
    observations: list[dict],
    max_previews_per_cluster: int = DEFAULT_MAX_PREVIEWS_PER_CLUSTER,
) -> list[dict]:
    """Select up to ``max_previews_per_cluster`` observations for preview.

    Within each file observations are sorted by (images before videos,
    timestamp, bbox_x, observation_id) and reordered farthest-point to
    avoid consecutive frames; then files (ordered by file_id) are visited
    round-robin, one sample per file per round.
    """
    if max_previews_per_cluster <= 0:
        return []
    grouped: dict[str, list[dict]] = {}
    for observation in observations:
        grouped.setdefault(observation["file_id"], []).append(observation)

    ordered_by_file: dict[str, list[dict]] = {}
    for file_id, items in grouped.items():
        items = sorted(
            items,
            key=lambda o: (
                o["source_type"] == "video",
                o["timestamp_ms"] or -1,
                o["bbox_x"],
                o["observation_id"],
            ),
        )
        ordered_by_file[file_id] = [
            items[index] for index in _farthest_point_order(len(items))
        ]

    selected: list[dict] = []
    file_ids = sorted(ordered_by_file)
    round_index = 0
    while len(selected) < max_previews_per_cluster:
        progressed = False
        for file_id in file_ids:
            items = ordered_by_file[file_id]
            if round_index < len(items):
                selected.append(items[round_index])
                progressed = True
                if len(selected) >= max_previews_per_cluster:
                    break
        if not progressed:
            break
        round_index += 1
    return selected


def _cluster_priority(cluster: dict) -> tuple:
    """Review priority: multi-file, then cross-media, then bigger clusters."""
    return (
        1 if cluster["file_count"] == 1 else 0,
        0 if cluster["cross_media"] else 1,
        -cluster["file_count"],
        -cluster["observations"],
        cluster["cluster_id"],
    )


def _fetch_canonical_labels(conn: sqlite3.Connection) -> list[str]:
    try:
        rows = conn.execute(
            "SELECT DISTINCT person_label FROM person_cluster_members "
            "ORDER BY person_label"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [row["person_label"] for row in rows]


def _fetch_cluster_observation_rows(conn: sqlite3.Connection) -> list:
    try:
        return conn.execute(
            "SELECT id, person_label, file_path, source_type, timestamp_ms, "
            "bbox_x, bbox_y, bbox_width, bbox_height "
            "FROM person_cluster_observations "
            "ORDER BY person_label, file_path, timestamp_ms, bbox_x, id"
        ).fetchall()
    except sqlite3.OperationalError:
        return []


def build_manifest(
    conn: sqlite3.Connection,
    *,
    max_previews_per_cluster: int = DEFAULT_MAX_PREVIEWS_PER_CLUSTER,
) -> tuple[dict, dict[str, dict]]:
    """Build the deterministic review manifest plus a private details map.

    The manifest never contains real paths or timestamps, only opaque
    ``F000NNN``/``O000NNN`` ids. The second return value maps every
    observation id to its real path, bbox, timestamp and source type; it is
    used only to render thumbnails and is never serialized.
    """
    file_rows = conn.execute("SELECT path FROM files ORDER BY path ASC").fetchall()
    file_ids = {
        row["path"]: f"F{index + 1:06d}" for index, row in enumerate(file_rows)
    }

    labels = _fetch_canonical_labels(conn)
    observation_rows = _fetch_cluster_observation_rows(conn)

    observations_by_label: dict[str, list[dict]] = {}
    details: dict[str, dict] = {}
    for row in observation_rows:
        source_type = _normalize_source_type(row["source_type"])
        observation = {
            "observation_id": f"O{row['id']:06d}",
            "file_id": file_ids.get(row["file_path"], "F000000"),
            "file_path": row["file_path"],
            "source_type": source_type,
            "timestamp_ms": row["timestamp_ms"],
            "bbox_x": row["bbox_x"],
            "bbox_y": row["bbox_y"],
            "bbox_width": row["bbox_width"],
            "bbox_height": row["bbox_height"],
        }
        observations_by_label.setdefault(row["person_label"], []).append(observation)
        details[observation["observation_id"]] = {
            "path": observation["file_path"],
            "bbox": (
                observation["bbox_x"],
                observation["bbox_y"],
                observation["bbox_width"],
                observation["bbox_height"],
            ),
            "timestamp_ms": observation["timestamp_ms"],
            "source_type": source_type,
        }

    clusters: list[dict] = []
    for label in labels:
        observations = observations_by_label.get(label, [])
        selected = select_samples(observations, max_previews_per_cluster)
        samples = [
            {
                "observation_id": observation["observation_id"],
                "file_id": observation["file_id"],
                "source_type": observation["source_type"],
                "preview": f"assets/{label}_{index:02d}.jpg",
            }
            for index, observation in enumerate(selected, start=1)
        ]
        images = sum(1 for o in observations if o["source_type"] == "image")
        videos = sum(1 for o in observations if o["source_type"] == "video")
        clusters.append(
            {
                "cluster_id": label,
                "observations": len(observations),
                "file_count": len({o["file_path"] for o in observations}),
                "images": images,
                "videos": videos,
                "cross_media": images > 0 and videos > 0,
                "samples": samples,
            }
        )

    clusters.sort(key=_cluster_priority)
    manifest = {
        "version": MANIFEST_VERSION,
        "generated_by": "face-review",
        "clusters": clusters,
    }
    return manifest, details


def write_manifest(manifest: dict, path: Path) -> None:
    """Write the manifest as JSON, preserving the intentional key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Thumbnails (local assets)
# ---------------------------------------------------------------------------


def _crop_box(image: Image.Image, bbox: tuple) -> tuple:
    """Crop box with a 30% margin on the bbox's longest side, clamped."""
    bbox_x, bbox_y, bbox_width, bbox_height = bbox
    margin = round(CROP_MARGIN_RATIO * max(bbox_width, bbox_height))
    left = max(0, bbox_x - margin)
    top = max(0, bbox_y - margin)
    right = min(image.width, bbox_x + bbox_width + margin)
    bottom = min(image.height, bbox_y + bbox_height + margin)
    if right <= left or bottom <= top:
        raise ValueError("bbox fuera de los límites de la imagen")
    return (left, top, right, bottom)


def _write_thumbnail(
    image: Image.Image, bbox: tuple, destination: Path, thumb_size: int
) -> int:
    """Crop, downscale and write a JPEG; returns the bytes written.

    The JPEG is encoded in memory first, so a failed encode never leaves a
    partial file behind.
    """
    crop = image.crop(_crop_box(image, bbox))
    crop.thumbnail((thumb_size, thumb_size))
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    data = buffer.getvalue()
    destination.write_bytes(data)
    return len(data)


def generate_assets(
    manifest: dict,
    details: dict[str, dict],
    assets_dir: Path,
    *,
    thumb_size: int = DEFAULT_THUMBNAIL_MAX_PX,
    video_sampler: Callable = sample_video_frames,
) -> dict:
    """Render local JPEG thumbnails for every selected sample.

    Video frames are sampled once per file (all selected timestamps at
    once, ephemerally); the crop/save happens inside the sampler callback
    because the frame only lives in a temporary directory. Any failure
    marks the sample's ``preview`` as ``None`` and counts an error instead
    of aborting the run. Returns ``{"thumbnails", "assets_bytes", "errors"}``.
    """
    assets_dir.mkdir(parents=True, exist_ok=True)
    thumbnails = 0
    assets_bytes = 0
    errors = 0

    for cluster in manifest.get("clusters", []):
        image_samples: list[tuple[dict, dict]] = []
        video_samples_by_path: dict[str, list[tuple[dict, dict]]] = {}
        for sample in cluster.get("samples", []):
            detail = details.get(sample["observation_id"])
            if detail is None:
                errors += 1
                sample["preview"] = None
                continue
            if detail["source_type"] == "video":
                video_samples_by_path.setdefault(detail["path"], []).append(
                    (sample, detail)
                )
            else:
                image_samples.append((sample, detail))

        for sample, detail in image_samples:
            try:
                with Image.open(detail["path"]) as handle:
                    image = handle.convert("RGB")
                size = _write_thumbnail(
                    image,
                    detail["bbox"],
                    assets_dir / Path(sample["preview"]).name,
                    thumb_size,
                )
                thumbnails += 1
                assets_bytes += size
            except Exception:  # noqa: BLE001 - one bad sample never aborts the run
                errors += 1
                sample["preview"] = None

        for path, items in video_samples_by_path.items():
            video_thumbnails, video_bytes, video_errors = _generate_video_assets(
                path, items, assets_dir, thumb_size, video_sampler
            )
            thumbnails += video_thumbnails
            assets_bytes += video_bytes
            errors += video_errors

    return {
        "thumbnails": thumbnails,
        "assets_bytes": assets_bytes,
        "errors": errors,
    }


def _generate_video_assets(
    path: str,
    items: list[tuple[dict, dict]],
    assets_dir: Path,
    thumb_size: int,
    video_sampler: Callable,
) -> tuple[int, int, int]:
    """Sample one video once and write every selected frame's thumbnail.

    Returns ``(thumbnails_written, bytes_written, errors)``. Samples that
    never get a written preview are marked ``preview=None``.
    """
    saved_ids: set[str] = set()
    thumbnail_count = 0
    bytes_written = 0

    timestamps: list[float] = []
    for _sample, detail in items:
        timestamp_ms = detail["timestamp_ms"]
        if timestamp_ms is None:
            continue
        seconds = round(timestamp_ms / 1000.0, 6)
        if seconds not in timestamps:
            timestamps.append(seconds)
    timestamps.sort()

    def _on_frame(frame_path: Path, timestamp: float, _index: int) -> None:
        nonlocal thumbnail_count, bytes_written
        matching = [
            (sample, detail)
            for sample, detail in items
            if detail["timestamp_ms"] is not None
            and abs(detail["timestamp_ms"] / 1000.0 - timestamp)
            <= _TIMESTAMP_TOLERANCE_SECONDS
        ]
        if not matching:
            return
        with Image.open(frame_path) as handle:
            image = handle.convert("RGB")
        for sample, detail in matching:
            if sample["observation_id"] in saved_ids:
                continue
            bytes_written += _write_thumbnail(
                image,
                detail["bbox"],
                assets_dir / Path(sample["preview"]).name,
                thumb_size,
            )
            saved_ids.add(sample["observation_id"])
            thumbnail_count += 1

    def _unsaved() -> list[dict]:
        return [
            sample
            for sample, _detail in items
            if sample["observation_id"] not in saved_ids
        ]

    try:
        if timestamps:
            result = video_sampler(
                Path(path),
                timestamps,
                on_frame=_on_frame,
                keep_debug_frames=False,
            )
        else:
            result = None
    except Exception:  # noqa: BLE001 - one bad video never aborts the run
        unsaved = _unsaved()
        for sample in unsaved:
            sample["preview"] = None
        return thumbnail_count, bytes_written, 1 if unsaved else 0

    if result is None or result.video_error is not None:
        unsaved = _unsaved()
        for sample in unsaved:
            sample["preview"] = None
        return thumbnail_count, bytes_written, len(unsaved)

    unsaved = _unsaved()
    for sample in unsaved:
        sample["preview"] = None
    return thumbnail_count, bytes_written, len(unsaved)


# ---------------------------------------------------------------------------
# Static offline HTML
# ---------------------------------------------------------------------------


def _escape_json_for_script(value) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def compute_metrics(manifest: dict, decisions: dict[str, str]) -> dict:
    """Exact review metrics for a manifest plus human decisions."""
    clusters = manifest.get("clusters", [])
    confirmed = 0
    rejected = 0
    uncertain = 0
    for cluster in clusters:
        decision = decisions.get(cluster["cluster_id"], DECISION_UNREVIEWED)
        if decision == DECISION_CONFIRMED:
            confirmed += 1
        elif decision == DECISION_REJECTED:
            rejected += 1
        elif decision == DECISION_UNCERTAIN:
            uncertain += 1
    reviewed = confirmed + rejected + uncertain
    return {
        "clusters_total": len(clusters),
        "reviewed": reviewed,
        "confirmed": confirmed,
        "rejected": rejected,
        "uncertain": uncertain,
        "unreviewed": len(clusters) - reviewed,
        "singleton_clusters": sum(
            1 for cluster in clusters if cluster["file_count"] == 1
        ),
        "multi_file_clusters": sum(
            1 for cluster in clusters if cluster["file_count"] > 1
        ),
        "cross_media_clusters": sum(
            1 for cluster in clusters if cluster["cross_media"]
        ),
    }


_METRIC_FIELDS = (
    "clusters_total",
    "reviewed",
    "confirmed",
    "rejected",
    "uncertain",
    "unreviewed",
    "singleton_clusters",
    "multi_file_clusters",
    "cross_media_clusters",
)

_CSS = """
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
dl.metrics { display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; margin: 0 0 0.75rem; }
dl.metrics div { margin: 0; }
dl.metrics dt { font-size: 0.7rem; text-transform: uppercase; color: #555; }
dl.metrics dd { margin: 0; font-size: 1.1rem; font-weight: 600; }
.actions { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }
button {
  padding: 0.35rem 0.7rem; border: 1px solid #999; border-radius: 4px;
  background: #fff; cursor: pointer; font-size: 0.85rem;
}
button.active { background: #1c1c1c; color: #fff; border-color: #1c1c1c; }
#status { font-size: 0.85rem; color: #444; }
main { display: grid; gap: 1rem; }
section.cluster {
  background: #fff; border: 1px solid #ddd; border-radius: 6px;
  padding: 0.75rem;
}
section.cluster[data-state="confirmed"] { border-color: #2e7d32; }
section.cluster[data-state="rejected"] { border-color: #c62828; }
section.cluster[data-state="uncertain"] { border-color: #ef6c00; }
section.cluster header { display: flex; justify-content: space-between; align-items: baseline; }
section.cluster h2 { font-size: 1rem; margin: 0; }
.state { font-size: 0.8rem; color: #555; }
.stats { margin: 0.25rem 0 0.5rem; font-size: 0.8rem; color: #444; font-family: monospace; }
.previews { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.5rem; }
figure { margin: 0; }
.previews img {
  display: block; max-width: 192px; max-height: 192px;
  border: 1px solid #ccc; border-radius: 4px; background: #eee;
}
.no-preview { margin: 0; color: #777; font-size: 0.85rem; font-style: italic; }
.decisions { display: flex; gap: 0.5rem; flex-wrap: wrap; }
"""

_JS = """
(function () {
  "use strict";
  var STORAGE_KEY = "lfo.face_review.decisions.v1";
  var ALLOWED = ["confirmed", "rejected", "uncertain"];
  var DECISION_UNREVIEWED = "unreviewed";
  var clustersById = {};
  CLUSTERS.forEach(function (cluster) { clustersById[cluster.cluster_id] = cluster; });
  var state = {};

  function decidedOnly() {
    var out = {};
    Object.keys(state).forEach(function (id) {
      if (ALLOWED.indexOf(state[id]) !== -1) { out[id] = state[id]; }
    });
    return out;
  }

  function readLocal() {
    try {
      var raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) { return {}; }
      var parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object" && parsed.decisions && typeof parsed.decisions === "object") {
        return parsed.decisions;
      }
      return {};
    } catch (err) {
      return {};
    }
  }

  function writeLocal() {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, decisions: state }));
    } catch (err) {
      setStatus("Aviso: no se pudo guardar en localStorage.");
    }
  }

  function applyInitial() {
    var local = readLocal();
    Object.keys(local).forEach(function (id) {
      if (clustersById[id] && (ALLOWED.indexOf(local[id]) !== -1 || local[id] === DECISION_UNREVIEWED)) {
        state[id] = local[id];
      }
    });
    Object.keys(INITIAL_DECISIONS).forEach(function (id) {
      if (!Object.prototype.hasOwnProperty.call(state, id) && clustersById[id]) {
        state[id] = INITIAL_DECISIONS[id];
      }
    });
  }

  function setText(id, value) {
    var node = document.getElementById(id);
    if (node) { node.textContent = String(value); }
  }

  function setStatus(message) {
    var node = document.getElementById("status");
    if (node) { node.textContent = message; }
  }

  function updateMetrics() {
    var counts = { confirmed: 0, rejected: 0, uncertain: 0 };
    Object.keys(state).forEach(function (id) {
      if (Object.prototype.hasOwnProperty.call(counts, state[id])) { counts[state[id]] += 1; }
    });
    var reviewed = counts.confirmed + counts.rejected + counts.uncertain;
    setText("metric-reviewed", reviewed);
    setText("metric-confirmed", counts.confirmed);
    setText("metric-rejected", counts.rejected);
    setText("metric-uncertain", counts.uncertain);
    setText("metric-unreviewed", CLUSTERS.length - reviewed);
  }

  function refreshCluster(clusterId) {
    var decision = state[clusterId] || DECISION_UNREVIEWED;
    var section = document.querySelector('[data-cluster="' + clusterId + '"]');
    if (!section) { return; }
    section.setAttribute("data-state", decision);
    var stateNode = section.querySelector(".state");
    if (stateNode) { stateNode.textContent = decision; }
    var buttons = section.querySelectorAll("button[data-decision]");
    for (var i = 0; i < buttons.length; i += 1) {
      if (buttons[i].getAttribute("data-decision") === decision) {
        buttons[i].classList.add("active");
      } else {
        buttons[i].classList.remove("active");
      }
    }
  }

  function refreshAll() {
    Object.keys(clustersById).forEach(refreshCluster);
    updateMetrics();
  }

  function exportDecisions() {
    var payload = { version: 1, decisions: decidedOnly() };
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    var url = URL.createObjectURL(blob);
    var anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "review_decisions.json";
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
    setStatus("Exportadas " + Object.keys(payload.decisions).length + " decisiones.");
  }

  function importDecisions(file) {
    var reader = new FileReader();
    reader.onload = function () {
      try {
        var payload = JSON.parse(String(reader.result));
        if (!payload || payload.version !== 1 || !payload.decisions || typeof payload.decisions !== "object") {
          throw new Error("formato o version invalidos");
        }
        var applied = 0;
        Object.keys(payload.decisions).forEach(function (id) {
          var value = payload.decisions[id];
          if (ALLOWED.indexOf(value) === -1) {
            throw new Error("estado invalido para " + id);
          }
          if (clustersById[id]) {
            state[id] = value;
            applied += 1;
          }
        });
        refreshAll();
        writeLocal();
        setStatus("Importadas " + applied + " decisiones.");
      } catch (err) {
        setStatus("Error al importar: " + err.message);
      }
    };
    reader.onerror = function () { setStatus("Error al leer el archivo."); };
    reader.readAsText(file);
  }

  function init() {
    applyInitial();
    refreshAll();
    var buttons = document.querySelectorAll("button[data-decision]");
    for (var i = 0; i < buttons.length; i += 1) {
      buttons[i].addEventListener("click", function (event) {
        var button = event.currentTarget;
        var section = button.closest(".cluster");
        if (!section) { return; }
        var clusterId = section.getAttribute("data-cluster");
        state[clusterId] = button.getAttribute("data-decision");
        refreshCluster(clusterId);
        updateMetrics();
        writeLocal();
        setStatus(clusterId + ": " + state[clusterId]);
      });
    }
    var exportButton = document.getElementById("export");
    if (exportButton) { exportButton.addEventListener("click", exportDecisions); }
    var importInput = document.getElementById("import");
    if (importInput) {
      importInput.addEventListener("change", function () {
        if (importInput.files && importInput.files[0]) { importDecisions(importInput.files[0]); }
      });
    }
  }

  init();
})();
"""


def _render_metric_rows(metrics: dict) -> str:
    rows = []
    for field in _METRIC_FIELDS:
        rows.append(
            f'<div><dt>{field}</dt><dd id="metric-{field}">'
            f"{metrics.get(field, 0)}</dd></div>"
        )
    return "\n      ".join(rows)


def _render_cluster_sections(clusters: list[dict]) -> str:
    sections = []
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        previews = []
        for index, sample in enumerate(cluster["samples"], start=1):
            preview = sample.get("preview")
            if preview:
                previews.append(
                    f'<figure><img src="{preview}" loading="lazy" '
                    f'alt="{cluster_id} muestra {index}"></figure>'
                )
        if not previews:
            previews.append('<p class="no-preview">sin preview</p>')
        sections.append(
            f'<section class="cluster" data-cluster="{cluster_id}" '
            f'data-state="{DECISION_UNREVIEWED}">\n'
            f"  <header><h2>{cluster_id}</h2>"
            f'<span class="state" id="state-{cluster_id}">'
            f"{DECISION_UNREVIEWED}</span></header>\n"
            f'  <p class="stats">observations={cluster["observations"]} '
            f'files={cluster["file_count"]} images={cluster["images"]} '
            f'videos={cluster["videos"]}</p>\n'
            f'  <div class="previews">\n    '
            + "\n    ".join(previews)
            + "\n  </div>\n"
            "  <div class=\"decisions\">\n"
            '    <button type="button" data-decision="confirmed">Misma persona</button>\n'
            '    <button type="button" data-decision="rejected">Personas distintas</button>\n'
            '    <button type="button" data-decision="uncertain">No seguro</button>\n'
            '    <button type="button" data-decision="unreviewed">Limpiar</button>\n'
            "  </div>\n"
            "</section>"
        )
    return "\n".join(sections)


def render_html(manifest: dict, decisions: dict[str, str], metrics: dict) -> str:
    """Render the fully static, offline review page (deterministic bytes)."""
    clusters = manifest.get("clusters", [])
    initial_decisions = {
        cluster["cluster_id"]: decisions[cluster["cluster_id"]]
        for cluster in clusters
        if cluster["cluster_id"] in decisions
    }
    clusters_json = _escape_json_for_script(clusters)
    initial_json = _escape_json_for_script(initial_decisions)
    sections = _render_cluster_sections(clusters)
    metric_rows = _render_metric_rows(metrics)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="es">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Revisión de clusters de personas</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        '<header class="site">\n'
        "  <h1>Revisión de clusters de personas</h1>\n"
        '  <p class="local-only">LOCAL ONLY — sin red, sin servicios externos</p>\n'
        '  <dl class="metrics">\n      '
        f"{metric_rows}\n"
        "  </dl>\n"
        '  <div class="actions">\n'
        '    <button type="button" id="export">Exportar decisiones JSON</button>\n'
        '    <label>Importar decisiones JSON <input type="file" id="import" '
        'accept=".json,application/json"></label>\n'
        '    <span id="status"></span>\n'
        "  </div>\n"
        "</header>\n"
        "<main>\n"
        f"{sections}\n"
        "</main>\n"
        "<script>\n"
        f"const CLUSTERS = {clusters_json};\n"
        f"const INITIAL_DECISIONS = {initial_json};\n"
        f"{_JS}"
        "</script>\n"
        "</body>\n"
        "</html>\n"
    )


def write_review_site(
    manifest: dict, decisions: dict[str, str], metrics: dict, output_dir: Path
) -> None:
    """Write ``index.html`` and ``manifest.json`` under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(
        render_html(manifest, decisions, metrics), encoding="utf-8"
    )
    write_manifest(manifest, output_dir / "manifest.json")


__all__ = [
    "ALLOWED_DECISIONS",
    "DECISION_CONFIRMED",
    "DECISION_REJECTED",
    "DECISION_UNCERTAIN",
    "DECISION_UNREVIEWED",
    "DECISIONS_VERSION",
    "DEFAULT_MAX_PREVIEWS_PER_CLUSTER",
    "DEFAULT_THUMBNAIL_MAX_PX",
    "MANIFEST_VERSION",
    "build_manifest",
    "compute_metrics",
    "generate_assets",
    "load_decisions",
    "parse_decisions_payload",
    "render_html",
    "save_decisions",
    "select_samples",
    "validate_decisions",
    "write_manifest",
    "write_review_site",
]
