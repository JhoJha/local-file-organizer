"""Tests for T2 threshold calibration (pure synthetic vectors, no models).

No real faces, no detector, no database: hand-built 128-d vectors and
invented ground-truth person sets. The CLI read-only wiring (catalog +
target) is covered in tests/test_cli.py.
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
from local_file_organizer.threshold_calibration import (
    build_observation_truth,
    calibration_metrics,
    format_results_table,
    pairwise_scores,
    sweep_grid,
)


def _unit(angle_rad: float) -> np.ndarray:
    v = np.zeros(128, dtype=np.float64)
    v[0] = math.cos(angle_rad)
    v[1] = math.sin(angle_rad)
    return v


def _obs(key, embedding) -> ClusterableObservation:
    return ClusterableObservation(key=key, embedding=np.asarray(embedding))


def _clustered(embeddings, threshold):
    observations = [_obs((i,), e) for i, e in enumerate(embeddings)]
    clusters = cluster_people(observations, threshold=threshold)
    labels = assign_person_labels(clusters)
    file_of_obs = [(f"/f{i}.jpg", "image") for i in range(len(embeddings))]
    return observations, clusters, labels, file_of_obs


class PairwiseScoresTests(unittest.TestCase):
    def test_perfect_clustering_scores_one(self):
        v = _unit(0.0)
        _observations, clusters, _labels, _files = _clustered([v.copy(), v.copy()], 0.5)
        truth = {0: frozenset({"PERSON_0001"}), 1: frozenset({"PERSON_0001"})}
        scores = pairwise_scores(clusters, truth)
        self.assertEqual(scores["precision"], 1.0)
        self.assertEqual(scores["recall"], 1.0)
        self.assertEqual(scores["f0.5"], 1.0)

    def test_over_split_kills_recall_not_precision_denominator(self):
        # Two far-apart observations of one person: no predicted-same
        # pair -> tp=fp=0, fn=1 -> precision 0 (empty denominator), recall 0.
        a, b = _unit(0.0), _unit(math.pi)
        _observations, clusters, _labels, _files = _clustered([a, b], 0.5)
        self.assertEqual(len(clusters), 2)
        truth = {0: frozenset({"PERSON_0001"}), 1: frozenset({"PERSON_0001"})}
        scores = pairwise_scores(clusters, truth)
        self.assertEqual(scores["precision"], 0.0)
        self.assertEqual(scores["recall"], 0.0)
        self.assertEqual(scores["f0.5"], 0.0)

    def test_wrongful_merge_kills_precision_keeps_recall(self):
        # Loose threshold merges two different people: fp=1, fn=0.
        a, b = _unit(0.0), _unit(math.radians(80))  # dist ~0.826
        _observations, clusters, _labels, _files = _clustered([a, b], 0.9)
        self.assertEqual(len(clusters), 1)
        truth = {0: frozenset({"PERSON_0001"}), 1: frozenset({"PERSON_0002"})}
        scores = pairwise_scores(clusters, truth)
        self.assertEqual(scores["precision"], 0.0)
        self.assertEqual(scores["recall"], 0.0)  # no true pairs exist either
        # F0.5 still 0 when both are 0.
        self.assertEqual(scores["f0.5"], 0.0)

    def test_partial_merge_gives_mid_precision(self):
        # Three observations, truth = {P1: 0,1 / P2: 2}, predicted all
        # together: pairs (0,1)=TP, (0,2)+(1,2)=FP -> P=1/3, R=1.0.
        clusters = [[0, 1, 2]]
        truth = {
            0: frozenset({"PERSON_0001"}),
            1: frozenset({"PERSON_0001"}),
            2: frozenset({"PERSON_0002"}),
        }
        scores = pairwise_scores(clusters, truth)
        self.assertAlmostEqual(scores["precision"], 1 / 3)
        self.assertEqual(scores["recall"], 1.0)
        # F0.5 weighs precision higher than F1 would (F1 = 0.5 here).
        beta2 = 0.25
        expected = (1 + beta2) * (1 / 3) * 1.0 / (beta2 * (1 / 3) + 1.0)
        self.assertAlmostEqual(scores["f0.5"], expected)

    def test_unlabeled_observations_contribute_no_pairs(self):
        clusters = [[0, 1]]
        scores = pairwise_scores(clusters, {0: frozenset({"PERSON_0001"})})
        self.assertEqual(scores["pairs"], 0.0)
        self.assertEqual(scores["precision"], 0.0)
        self.assertEqual(scores["recall"], 0.0)

    def test_group_photo_overlap_counts_as_same(self):
        clusters = [[0, 1]]
        truth = {
            0: frozenset({"PERSON_0001", "PERSON_0002"}),
            1: frozenset({"PERSON_0002", "PERSON_0003"}),
        }
        scores = pairwise_scores(clusters, truth)
        self.assertEqual(scores["precision"], 1.0)
        self.assertEqual(scores["recall"], 1.0)


class ObservationTruthTests(unittest.TestCase):
    def test_only_canonical_ids_are_ground_truth(self):
        file_of_obs = [("/a.jpg", "image"), ("/b.jpg", "image"), ("/c.jpg", "image")]
        resolved = {
            "/a.jpg": {"PERSON_0001"},
            "/b.jpg": {"P0002"},  # never curated: excluded
            "/c.jpg": {"PERSON_0001", "PERSON_0002"},
        }
        truth = build_observation_truth(file_of_obs, resolved)
        self.assertEqual(
            truth,
            {0: frozenset({"PERSON_0001"}), 2: frozenset({"PERSON_0001", "PERSON_0002"})},
        )

    def test_missing_files_are_unlabeled(self):
        truth = build_observation_truth([("/x.jpg", "image")], {})
        self.assertEqual(truth, {})


class SweepGridTests(unittest.TestCase):
    def _two_person_setup(self):
        # a1/a2 identical (PERSON_0001, always together), b at 63 degrees
        # (PERSON_0002, sim ~0.454): all merged at split 0.40, b apart
        # at split 0.50.
        a, b = _unit(0.0), _unit(math.radians(63))
        observations = [_obs((0,), a), _obs((1,), a.copy()), _obs((2,), b)]
        file_of_obs = [
            ("/a1.jpg", "image"),
            ("/a2.jpg", "image"),
            ("/b.jpg", "image"),
        ]
        truth = {
            0: frozenset({"PERSON_0001"}),
            1: frozenset({"PERSON_0001"}),
            2: frozenset({"PERSON_0002"}),
        }
        return observations, file_of_obs, truth

    def test_grid_shape_and_order(self):
        observations, file_of_obs, truth = self._two_person_setup()
        rows = sweep_grid(
            observations, file_of_obs, None, truth,
            similarity_thresholds=[0.50, 0.40], merge_floors=[0.35, 0.30],
        )
        self.assertEqual(len(rows), 4)
        keys = [(r["similarity_threshold"], r["merge_floor"]) for r in rows]
        self.assertEqual(keys, [(0.40, 0.30), (0.40, 0.35), (0.50, 0.30), (0.50, 0.35)])
        for row in rows:
            for key in (
                "precision", "recall", "f0.5", "singleton_rate",
                "cluster_count", "suggestions", "pairs",
            ):
                self.assertIn(key, row)

    def test_loose_cell_merges_strict_cell_splits(self):
        observations, file_of_obs, truth = self._two_person_setup()
        rows = sweep_grid(
            observations, file_of_obs, None, truth,
            similarity_thresholds=[0.40, 0.50], merge_floors=[0.30],
        )
        loose, strict = rows
        self.assertEqual(loose["cluster_count"], 1.0)
        self.assertLess(loose["precision"], 1.0)
        self.assertEqual(strict["cluster_count"], 2.0)
        self.assertEqual(strict["precision"], 1.0)
        self.assertEqual(strict["recall"], 1.0)
        # Merge band [0.30, split): the in-between pair is suggested at
        # split 0.50 but already merged at split 0.40.
        self.assertEqual(strict["suggestions"], 1.0)
        self.assertEqual(loose["suggestions"], 0.0)

    def test_cluster_metrics_ignore_merge_floor(self):
        observations, file_of_obs, truth = self._two_person_setup()
        rows = sweep_grid(
            observations, file_of_obs, None, truth,
            similarity_thresholds=[0.50], merge_floors=[0.30, 0.40],
        )
        for key in ("precision", "recall", "f0.5", "singleton_rate", "cluster_count"):
            self.assertEqual(rows[0][key], rows[1][key])

    def test_empty_grids_raise(self):
        observations, file_of_obs, truth = self._two_person_setup()
        with self.assertRaises(ValueError):
            sweep_grid(observations, file_of_obs, None, truth,
                        similarity_thresholds=[], merge_floors=[0.30])
        with self.assertRaises(ValueError):
            sweep_grid(observations, file_of_obs, None, truth,
                        similarity_thresholds=[0.50], merge_floors=[])

    def test_sweep_is_deterministic(self):
        observations, file_of_obs, truth = self._two_person_setup()
        kwargs = dict(similarity_thresholds=[0.40, 0.50], merge_floors=[0.30, 0.40])
        self.assertEqual(
            sweep_grid(observations, file_of_obs, [2.0, 7.0, 1.0], truth, **kwargs),
            sweep_grid(observations, file_of_obs, [2.0, 7.0, 1.0], truth, **kwargs),
        )

    def test_singleton_rate_counts_size_one_clusters(self):
        truth = {0: frozenset({"PERSON_0001"}), 1: frozenset({"PERSON_0001"})}
        metrics = calibration_metrics([[0], [1]], truth)
        self.assertEqual(metrics["singleton_rate"], 1.0)
        self.assertEqual(metrics["cluster_count"], 2.0)


class FormatTableTests(unittest.TestCase):
    def test_table_has_header_columns_and_defaults_line(self):
        table = format_results_table(
            [
                {
                    "similarity_threshold": 0.5, "merge_floor": 0.4,
                    "precision": 1.0, "recall": 0.5, "f0.5": 0.8,
                    "singleton_rate": 0.25, "cluster_count": 4.0,
                    "suggestions": 2.0,
                }
            ],
            observations=10, labeled=8, ground_truth_persons=3,
            linkage="complete", default_similarity=0.50, default_floor=0.40,
        )
        self.assertIn("=== PERSON CALIBRATION ===", table)
        self.assertIn("observations=10 labeled=8 unlabeled=2", table)
        self.assertIn("ground_truth_persons=3 linkage=complete", table)
        self.assertIn("similarity_threshold merge_floor precision recall f0.5", table)
        self.assertIn("0.50 0.40 1.000 0.500 0.800 0.250 4 2", table)
        self.assertIn("defaults: similarity_threshold=0.50 merge_floor=0.40", table)


if __name__ == "__main__":
    unittest.main()
