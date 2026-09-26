"""Threshold calibration against curated PERSON_XXXX ground truth (T2).

Read-only calibration harness: memory-only recomputed embeddings are
clustered over a grid of (similarity-threshold, merge-floor) pairs and
each cell is scored against the human-curated canonical identities
(`PERSON_XXXX` from `person_aliases`/`file_person_links`, resolved via
`person_identity.resolve_file_persons`). Each cell reports pairwise
precision/recall/F0.5 (beta=0.5 weighs precision higher — false merges
are worse than missed merges, the same rationale as the 0.50 default),
plus singleton-rate, cluster-count and the suggestion count for that
merge floor. Cluster metrics are merge-floor-invariant by construction
(suggestions are advisory only, never applied); the floor column moves
only the suggestion count.

Privacy: this module never persists anything — no embeddings, no
labels, no database writes — and it never prints embeddings, paths, or
vectors, only aggregate metrics. It never changes the 0.50/0.40
defaults; it only reports what other thresholds would have scored.
CPU only, zero new dependencies, fully deterministic.
"""
from __future__ import annotations

from .face_clustering import assign_person_labels, cluster_people
from .person_identity import is_canonical_id
from .person_suggest import suggest_merges

# F-beta weight for the calibration score: beta=0.5 counts precision
# twice as much as recall (1+beta^2 weighting), matching the product
# rule that a wrongful merge is worse than a missed merge.
F_BETA = 0.5


def build_observation_truth(
    file_of_obs: list[tuple[str, str]],
    resolved_persons: dict[str, set[str]],
) -> dict[int, frozenset[str]]:
    """Map observation index -> canonical ground-truth person set.

    `resolved_persons` is `person_identity.resolve_file_persons(conn)`:
    effective persons per file (aliases + manual links minus exclusions).
    Only canonical `PERSON_XXXX` identities count as ground truth —
    files resolving solely to raw machine labels (`P000N`, i.e. never
    curated) are unlabeled and excluded from scoring, otherwise the
    harness would measure self-consistency against stale machine labels
    instead of human curation. Files with several canonical persons
    (group photos) keep the whole set; a predicted pair sharing at
    least one of them counts as a true same-person pair.
    """
    truth: dict[int, frozenset[str]] = {}
    for idx, (file_path, _category) in enumerate(file_of_obs):
        canonical = frozenset(
            p for p in resolved_persons.get(file_path, set()) if is_canonical_id(p)
        )
        if canonical:
            truth[idx] = canonical
    return truth


def pairwise_scores(
    clusters: list[list[int]], truth: dict[int, frozenset[str]]
) -> dict[str, float]:
    """Pairwise precision/recall/F0.5 of predicted clusters vs truth.

    Only observations present in `truth` take part; every other
    observation is ignored (it still occupies its cluster, it just
    contributes no pairs). A labeled pair is predicted-same when both
    members land in one predicted cluster, and truly-same when their
    canonical sets intersect. Empty pair denominators score 0.0 (an
    over-split-to-singletons run has recall 0, not undefined).
    Deterministic: pair iteration order never affects the counts.
    """
    cluster_of: dict[int, int] = {}
    for number, idxs in enumerate(clusters):
        for idx in idxs:
            cluster_of[idx] = number
    labeled = sorted(truth)
    tp = fp = fn = 0
    for pos, idx_a in enumerate(labeled):
        for idx_b in labeled[pos + 1 :]:
            predicted_same = cluster_of.get(idx_a) == cluster_of.get(idx_b)
            truly_same = bool(truth[idx_a] & truth[idx_b])
            if predicted_same and truly_same:
                tp += 1
            elif predicted_same:
                fp += 1
            elif truly_same:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta2 = F_BETA * F_BETA
    denom = beta2 * precision + recall
    f_score = (1.0 + beta2) * precision * recall / denom if denom > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f0.5": float(f_score),
        "pairs": float(tp + fp + fn),
        "true_pairs": float(tp + fn),
    }


def calibration_metrics(
    clusters: list[list[int]], truth: dict[int, frozenset[str]]
) -> dict[str, float]:
    """Pairwise scores plus singleton-rate and cluster-count for one cell."""
    scores = pairwise_scores(clusters, truth)
    singleton_rate = (
        sum(1 for idxs in clusters if len(idxs) == 1) / len(clusters)
        if clusters
        else 0.0
    )
    scores["singleton_rate"] = float(singleton_rate)
    scores["cluster_count"] = float(len(clusters))
    scores["labeled_observations"] = float(len(truth))
    return scores


def sweep_grid(
    observations,
    file_of_obs: list[tuple[str, str]],
    weights: list[float] | None,
    truth: dict[int, frozenset[str]],
    *,
    similarity_thresholds: list[float],
    merge_floors: list[float],
    linkage: str = "complete",
) -> list[dict[str, float]]:
    """Score every (similarity-threshold, merge-floor) cell, deterministically.

    Each similarity threshold rebuilds the clusters (`cluster_people` at
    distance `1.0 - similarity`, quality-weighted tracklet/suggest
    centroids via `weights`); each merge floor only recounts
    `suggest_merges` inside `[floor, split)` for that row's clusters.
    Rows come out sorted by (similarity, floor) regardless of input
    order. Raises ValueError on empty grids.
    """
    if not similarity_thresholds:
        raise ValueError("similarity_thresholds no puede estar vacío")
    if not merge_floors:
        raise ValueError("merge_floors no puede estar vacío")
    rows: list[dict[str, float]] = []
    for similarity in sorted(similarity_thresholds):
        clusters = cluster_people(
            observations, threshold=1.0 - similarity, linkage=linkage
        )
        labels = assign_person_labels(clusters)
        metrics = calibration_metrics(clusters, truth)
        for floor in sorted(merge_floors):
            candidates = suggest_merges(
                observations,
                clusters,
                labels,
                file_of_obs,
                split_threshold=similarity,
                merge_floor=floor,
                weights=weights,
            )
            rows.append(
                {
                    "similarity_threshold": float(similarity),
                    "merge_floor": float(floor),
                    **metrics,
                    "suggestions": float(len(candidates)),
                }
            )
    return rows


def format_results_table(
    rows: list[dict[str, float]],
    *,
    observations: int,
    labeled: int,
    ground_truth_persons: int,
    linkage: str,
    default_similarity: float,
    default_floor: float,
) -> str:
    """Render sweep rows as the `person-calibrate` stdout table (no paths)."""
    lines = [
        "=== PERSON CALIBRATION ===",
        f"observations={observations} labeled={labeled} "
        f"unlabeled={observations - labeled} "
        f"ground_truth_persons={ground_truth_persons} linkage={linkage}",
        "similarity_threshold merge_floor precision recall f0.5 "
        "singleton_rate clusters suggestions",
    ]
    for row in rows:
        lines.append(
            f"{row['similarity_threshold']:.2f} "
            f"{row['merge_floor']:.2f} "
            f"{row['precision']:.3f} "
            f"{row['recall']:.3f} "
            f"{row['f0.5']:.3f} "
            f"{row['singleton_rate']:.3f} "
            f"{int(row['cluster_count'])} "
            f"{int(row['suggestions'])}"
        )
    lines.append(
        f"defaults: similarity_threshold={default_similarity:.2f} "
        f"merge_floor={default_floor:.2f} (unchanged; this command only reports)"
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "F_BETA",
    "build_observation_truth",
    "calibration_metrics",
    "format_results_table",
    "pairwise_scores",
    "sweep_grid",
]
