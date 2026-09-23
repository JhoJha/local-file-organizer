"""Tests for V2.4 anonymous person clustering (pure numpy, no model).

Clustering LOGIC is tested here in complete isolation: no images, no
detector, no SFace model — hand-built 128-d vectors only.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from local_file_organizer.face_clustering import (
    ClusterableObservation,
    assign_person_labels,
    cluster_people,
)
from local_file_organizer.face_embedding import cosine_distance


def _unit_on_circle(angle_rad: float) -> np.ndarray:
    """Unit vector in the (e0, e1) plane at `angle_rad`, embedded in 128-d."""
    v = np.zeros(128, dtype=np.float64)
    v[0] = math.cos(angle_rad)
    v[1] = math.sin(angle_rad)
    return v


def _obs(key, embedding) -> ClusterableObservation:
    return ClusterableObservation(key=key, embedding=np.asarray(embedding))


def _keys_of(clusters, observations) -> set[frozenset]:
    return frozenset(
        frozenset(observations[i].key for i in cluster) for cluster in clusters
    )


class TransitiveSafetyTests(unittest.TestCase):
    def test_chain_does_not_transitively_merge(self):
        # THE single most important test in this milestone: A~B and B~C
        # must NOT force A,B,C together when A and C are themselves beyond
        # threshold (greedy complete-linkage, mirroring V2.1's
        # group_similar_images rationale — no union-find/single-linkage).
        theta = math.radians(40)
        a = _unit_on_circle(0.0)
        b = _unit_on_circle(theta)
        c = _unit_on_circle(2 * theta)
        threshold = 0.5
        self.assertLessEqual(cosine_distance(a, b), threshold)
        self.assertLessEqual(cosine_distance(b, c), threshold)
        self.assertGreater(cosine_distance(a, c), threshold)

        observations = [_obs(("a",), a), _obs(("b",), b), _obs(("c",), c)]
        clusters = cluster_people(observations, threshold=threshold)
        # Never one cluster of all three.
        self.assertEqual(len(clusters), 2)
        self.assertTrue(all(len(cluster) < 3 for cluster in clusters))
        # Every observation belongs to exactly one cluster.
        flat = sorted(i for cluster in clusters for i in cluster)
        self.assertEqual(flat, [0, 1, 2])


class BasicGroupingTests(unittest.TestCase):
    def test_identical_embeddings_cluster_together(self):
        v = _unit_on_circle(0.3)
        observations = [
            _obs(("a",), v.copy()),
            _obs(("b",), v.copy()),
            _obs(("c",), v.copy()),
        ]
        clusters = cluster_people(observations, threshold=0.5)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(sorted(clusters[0]), [0, 1, 2])

    def test_far_apart_embeddings_stay_singletons_and_are_kept(self):
        # Orthogonal basis vectors: pairwise cosine distance 1.0.
        observations = [
            _obs(("a",), np.eye(128, dtype=np.float64)[0]),
            _obs(("b",), np.eye(128, dtype=np.float64)[1]),
            _obs(("c",), np.eye(128, dtype=np.float64)[2]),
        ]
        clusters = cluster_people(observations, threshold=0.5)
        # Unlike V2.1's group_similar_images (which drops singletons),
        # every observation keeps its own cluster here.
        self.assertEqual(len(clusters), 3)
        self.assertTrue(all(len(cluster) == 1 for cluster in clusters))
        flat = sorted(i for cluster in clusters for i in cluster)
        self.assertEqual(flat, [0, 1, 2])

    def test_empty_input_gives_no_clusters(self):
        self.assertEqual(cluster_people([], threshold=0.5), [])


class DeterminismTests(unittest.TestCase):
    def test_shuffled_input_order_gives_same_outcome(self):
        theta = math.radians(40)
        base = [
            _obs(("a",), _unit_on_circle(0.0)),
            _obs(("b",), _unit_on_circle(theta)),
            _obs(("c",), _unit_on_circle(2 * theta)),
            _obs(("d",), _unit_on_circle(math.pi)),
        ]
        shuffled = [base[2], base[0], base[3], base[1]]
        clusters_base = cluster_people(base, threshold=0.5)
        clusters_shuffled = cluster_people(shuffled, threshold=0.5)
        # Same final cluster *contents* (compared as key sets).
        self.assertEqual(
            _keys_of(clusters_base, base), _keys_of(clusters_shuffled, shuffled)
        )
        # Same P-label for the same underlying observation.
        labels_base = assign_person_labels(clusters_base)
        labels_shuffled = assign_person_labels(clusters_shuffled)
        key_to_label_base = {base[i].key: labels_base[i] for i in labels_base}
        key_to_label_shuffled = {
            shuffled[i].key: labels_shuffled[i] for i in labels_shuffled
        }
        self.assertEqual(key_to_label_base, key_to_label_shuffled)


class LabelFormattingTests(unittest.TestCase):
    def test_p_labels_are_zero_padded_and_ordered(self):
        observations = [
            _obs(("a",), np.eye(128, dtype=np.float64)[0]),
            _obs(("b",), np.eye(128, dtype=np.float64)[1]),
        ]
        clusters = cluster_people(observations, threshold=0.5)
        self.assertEqual(len(clusters), 2)
        labels = assign_person_labels(clusters)
        ordered = [labels[i] for i in clusters[0]] + [
            labels[i] for i in clusters[1]
        ]
        self.assertEqual(ordered, ["P0001", "P0002"])
        # P0001 goes to the cluster whose smallest key is smallest.
        first_keys = [observations[i].key for i in clusters[0]]
        self.assertIn(("a",), first_keys)


class ThresholdSensitivityTests(unittest.TestCase):
    def test_looser_threshold_merges_more_stricter_merges_less(self):
        theta = math.radians(40)
        observations = [
            _obs(("a",), _unit_on_circle(0.0)),
            _obs(("b",), _unit_on_circle(theta)),
            _obs(("c",), _unit_on_circle(2 * theta)),
        ]
        loose = cluster_people(observations, threshold=0.9)
        self.assertEqual(len(loose), 1)  # everything merges
        strict = cluster_people(observations, threshold=0.1)
        self.assertEqual(len(strict), 3)  # nothing merges
        middle = cluster_people(observations, threshold=0.5)
        self.assertEqual(len(middle), 2)  # partial merge


if __name__ == "__main__":
    unittest.main()
