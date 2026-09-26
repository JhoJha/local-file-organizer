"""Merge suggestions over person clusters (S2).

After `cluster-people` splits observations into anonymous clusters
(P0001, ...), some real people remain split across two clusters whose
embeddings are close but not close enough to have merged. This module
lists those doubtful pairs as suggestions with evidence; the user
confirms with the existing `person-merge` command. There is no
auto-merge under any flag: suggestions only.

Similarity metric (SUGGEST_SIMILARITY_METRIC): cosine similarity
between cluster centroids (mean embedding per cluster). A candidate
pair is listed only inside the merge band: similarity at or above
DEFAULT_MERGE_FLOOR_SIMILARITY and strictly below the split threshold
(the same threshold `cluster-people` used, default 0.50). Pairs at or
above the split threshold would normally already be one cluster;
pairs below the floor are too dissimilar to suggest.

Privacy: embeddings are transient in-memory arrays, exactly as in
`face_clustering` — this module never persists anything.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .face_clustering import DEFAULT_SIMILARITY_THRESHOLD
from .face_embedding import cosine_distance, weighted_mean

# Pairwise cluster similarity metric used for suggestions.
SUGGEST_SIMILARITY_METRIC = "centroid_cosine"

# Conservative default merge floor (cosine similarity). The suggest band
# is [floor, split): 0.40 is deliberately close to the 0.50 split
# threshold so only near-miss pairs are suggested (false merges are
# worse than missed merges, same rationale as the split threshold).
DEFAULT_MERGE_FLOOR_SIMILARITY = 0.40


@dataclass
class MergeCandidate:
    """One suggested merge with evidence for the user to confirm."""

    label_a: str  # P000N, alphabetically first
    label_b: str  # P000N, alphabetically second
    mean_similarity: float  # centroid cosine similarity
    obs_a: int  # observations in cluster A
    obs_b: int  # observations in cluster B
    files_a: int  # distinct files in cluster A
    files_b: int  # distinct files in cluster B

    def merge_command(self) -> str:
        """Exact invocation the user runs to confirm this merge."""
        return f"python -m local_file_organizer person-merge {self.label_a} {self.label_b}"


def _centroid(embeddings: list[np.ndarray], weights=None) -> np.ndarray | None:
    """Mean embedding, or None when it has zero norm (unmeasurable).

    Quality weighting (T2): `weights` is one non-negative float per
    embedding (confidence*size*sharpness from the ingestion gate; the
    CLI passes one per observation). `None` or uniform weights take the
    exact pre-T2 plain-mean path; invalid weights fall back to it (see
    `face_embedding.weighted_mean`). Similarity math is untouched: the
    centroid is still the raw (non-normalized) mean — `cosine_distance`
    normalizes internally, so rankings only shift because low-quality
    observations now steer the centroid less, which is the intended fix.
    """
    stacked = np.stack([np.asarray(e, dtype=np.float64).reshape(-1) for e in embeddings])
    mean = weighted_mean(stacked, weights)
    if float(np.linalg.norm(mean)) == 0.0:
        return None
    return mean


def suggest_merges(
    observations,
    clusters: list[list[int]],
    labels: dict[int, str],
    file_of_obs: list[tuple[str, str]],
    split_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    merge_floor: float = DEFAULT_MERGE_FLOOR_SIMILARITY,
    weights: list[float] | None = None,
) -> list[MergeCandidate]:
    """List candidate merges inside [merge_floor, split_threshold).

    Deterministic: candidates are sorted by similarity descending, then
    by (label_a, label_b). Pairs that already merged (at/above split)
    or are too dissimilar (below floor) are absent. `weights` (one
    quality weight per observation, `None` = plain centroids) is sliced
    per cluster and forwarded to `_centroid`; a length mismatch falls
    back to plain centroids. The band math and the ordering are
    identical either way.
    """
    if len(clusters) < 2:
        return []
    if weights is not None and len(weights) != len(observations):
        weights = None
    centroids: dict[int, np.ndarray] = {}
    for number, idxs in enumerate(clusters):
        cluster_weights = None if weights is None else [weights[i] for i in idxs]
        centroid = _centroid(
            [observations[i].embedding for i in idxs], weights=cluster_weights
        )
        if centroid is not None:
            centroids[number] = centroid
    cluster_labels = [labels[clusters[n][0]] for n in range(len(clusters))]
    info = [
        (cluster_labels[n], len(idxs), len({file_of_obs[i][0] for i in idxs}))
        for n, idxs in enumerate(clusters)
    ]
    candidates: list[MergeCandidate] = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if i not in centroids or j not in centroids:
                continue
            sim = 1.0 - cosine_distance(centroids[i], centroids[j])
            if sim < merge_floor or sim >= split_threshold:
                continue
            first, second = sorted((info[i], info[j]))  # by label (unique)
            candidates.append(
                MergeCandidate(
                    label_a=first[0],
                    label_b=second[0],
                    mean_similarity=float(sim),
                    obs_a=first[1],
                    obs_b=second[1],
                    files_a=first[2],
                    files_b=second[2],
                )
            )
    candidates.sort(key=lambda c: (-c.mean_similarity, c.label_a, c.label_b))
    return candidates


__all__ = [
    "DEFAULT_MERGE_FLOOR_SIMILARITY",
    "SUGGEST_SIMILARITY_METRIC",
    "MergeCandidate",
    "suggest_merges",
]
