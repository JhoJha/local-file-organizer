"""Anonymous person clustering over face embeddings (V2.4).

Greedy complete-linkage clustering over cosine distance. This mirrors the
algorithmic shape of `perceptual_hash.group_similar_images` (same
anti-transitive-merge rationale: A~B and B~C must NOT force A~C together
unless A and C are themselves within threshold — i.e. a candidate only
joins the first existing cluster whose *every* current member is within
threshold, otherwise it starts a new cluster).

The one deliberate difference from V2.1's grouping: in V2.1, a singleton
image (nothing else similar enough) stays ungrouped (`similar_group =
NULL`) because "similarity grouping" is only meaningful for 2+. Here,
every face observation must end up with *some* person label, even if it
is the only observation of that person in the whole dataset (a person
seen once is still `P0001`, not left unlabeled) — this milestone is about
identity, not just similarity grouping. So singleton clusters are kept.

Privacy: embeddings live only as transient in-memory numpy arrays during
a single `cluster-people` run and are discarded when it ends. This module
never persists anything; the CLI persists only the derived anonymous
(person label, file) membership rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .face_embedding import cosine_distance

# Cosine SIMILARITY (not distance). OpenCV's official SFace sample uses a
# cosine-similarity match threshold of 0.363 (calibrated on LFW to balance
# precision/recall). This project deliberately sets a *stricter* (higher)
# similarity requirement — 0.50 — because the product requirement is
# "false positives [wrongly merging two different people] are worse than
# false negatives [splitting one person into two clusters]" — raising the
# bar for what counts as "the same person" trades some extra splitting
# for fewer wrongful merges, which is the explicitly desired direction.
DEFAULT_SIMILARITY_THRESHOLD = 0.50


@dataclass
class ClusterableObservation:
    key: tuple  # fully deterministic sort key — used ONLY for ordering, never for similarity
    embedding: np.ndarray  # (128,) float32


def cluster_people(
    observations: list[ClusterableObservation],
    threshold: float = 1.0 - DEFAULT_SIMILARITY_THRESHOLD,
) -> list[list[int]]:
    """Greedy complete-linkage clustering over cosine distance.

    `threshold` is a DISTANCE threshold (lower = more similar), already
    converted from the cosine-similarity CLI flag by the caller
    (`distance = 1.0 - similarity`).

    Returns a list of clusters, each a list of indices into
    `observations`, sorted deterministically (see below). Every
    observation belongs to exactly one cluster, including size-1 clusters
    (unlike perceptual_hash's group_similar_images, which drops
    singletons).
    """
    n = len(observations)
    if n == 0:
        return []
    # Sort by .key first (determinism — same input always produces the
    # same construction order, hence the same cluster contents and the
    # same eventual P-label assignment).
    order = sorted(range(n), key=lambda i: observations[i].key)
    clusters: list[list[int]] = []
    for idx in order:
        placed = False
        for cluster in clusters:
            if all(
                cosine_distance(
                    observations[idx].embedding,
                    observations[member].embedding,
                )
                <= threshold
                for member in cluster
            ):
                cluster.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])
    # Sort the resulting clusters by the .key of their earliest
    # (smallest-key) member, so P-label assignment order is deterministic
    # regardless of internal construction order.
    clusters.sort(key=lambda idxs: min(observations[i].key for i in idxs))
    return clusters


def assign_person_labels(clusters: list[list[int]]) -> dict[int, str]:
    """Assign anonymous P-labels: P0001 for the first cluster, etc."""
    labels: dict[int, str] = {}
    for number, idxs in enumerate(clusters, start=1):
        label = f"P{number:04d}"
        for i in idxs:
            labels[i] = label
    return labels


__all__ = [
    "ClusterableObservation",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "assign_person_labels",
    "cluster_people",
]
