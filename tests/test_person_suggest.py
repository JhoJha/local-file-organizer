"""Tests for S2 person-suggest (pure synthetic vectors + temp dirs only)."""
from __future__ import annotations

import io
import math
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from local_file_organizer.face_clustering import (
    ClusterableObservation,
    assign_person_labels,
    cluster_people,
)
from local_file_organizer.person_suggest import (
    DEFAULT_MERGE_FLOOR_SIMILARITY,
    suggest_merges,
)

SPLIT = 0.50
FLOOR = DEFAULT_MERGE_FLOOR_SIMILARITY


def _unit(angle_rad: float) -> np.ndarray:
    v = np.zeros(128, dtype=np.float64)
    v[0] = math.cos(angle_rad)
    v[1] = math.sin(angle_rad)
    return v


def _obs(key, embedding) -> ClusterableObservation:
    return ClusterableObservation(key=key, embedding=np.asarray(embedding))


def _clustered(embeddings, files):
    observations = [_obs((f, i), e) for i, (e, f) in enumerate(zip(embeddings, files))]
    clusters = cluster_people(observations, threshold=1.0 - SPLIT)
    labels = assign_person_labels(clusters)
    file_of_obs = [(f, "image") for f in files]
    return observations, clusters, labels, file_of_obs


class BandTests(unittest.TestCase):
    def test_in_band_pair_listed_with_evidence(self):
        a, b = _unit(0.0), _unit(math.radians(63))  # sim ~0.454
        observations, clusters, labels, file_of_obs = _clustered(
            [a, b], ["/a.jpg", "/b.jpg"]
        )
        self.assertEqual(len(clusters), 2)
        candidates = suggest_merges(
            observations, clusters, labels, file_of_obs,
            split_threshold=SPLIT, merge_floor=FLOOR,
        )
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual((candidate.label_a, candidate.label_b), ("P0001", "P0002"))
        self.assertGreaterEqual(candidate.mean_similarity, FLOOR)
        self.assertLess(candidate.mean_similarity, SPLIT)
        self.assertEqual((candidate.obs_a, candidate.obs_b), (1, 1))
        self.assertEqual((candidate.files_a, candidate.files_b), (1, 1))
        self.assertEqual(
            candidate.merge_command(),
            "python -m local_file_organizer person-merge P0001 P0002",
        )

    def test_above_split_pair_absent_already_one_cluster(self):
        v = _unit(0.3)
        observations, clusters, labels, file_of_obs = _clustered(
            [v.copy(), v.copy()], ["/a.jpg", "/b.jpg"]
        )
        self.assertEqual(len(clusters), 1)
        self.assertEqual(
            suggest_merges(observations, clusters, labels, file_of_obs,
                           split_threshold=SPLIT, merge_floor=FLOOR), [],
        )

    def test_below_floor_pair_absent(self):
        observations, clusters, labels, file_of_obs = _clustered(
            [np.eye(128)[0], np.eye(128)[1]], ["/a.jpg", "/b.jpg"]
        )
        self.assertEqual(len(clusters), 2)
        self.assertEqual(
            suggest_merges(observations, clusters, labels, file_of_obs,
                           split_threshold=SPLIT, merge_floor=FLOOR), [],
        )

    def test_deterministic_ordering_similarity_then_labels(self):
        v0 = _unit(0.0)
        v1 = _unit(math.radians(61))  # sim(v0,v1) ~0.485
        v2 = _unit(math.radians(-63))  # sim(v0,v2) ~0.454, sim(v1,v2) below floor
        observations, clusters, labels, file_of_obs = _clustered(
            [v0, v1, v2], ["/a.jpg", "/b.jpg", "/c.jpg"]
        )
        self.assertEqual(len(clusters), 3)
        first = suggest_merges(
            observations, clusters, labels, file_of_obs,
            split_threshold=SPLIT, merge_floor=FLOOR,
        )
        second = suggest_merges(
            observations, clusters, labels, file_of_obs,
            split_threshold=SPLIT, merge_floor=FLOOR,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        sims = [c.mean_similarity for c in first]
        self.assertEqual(sims, sorted(sims, reverse=True))
        self.assertEqual((first[0].label_a, first[0].label_b), ("P0001", "P0002"))


def _save_gradient(path: Path) -> None:
    img = Image.new("RGB", (64, 64))
    pixels = img.load()
    for x in range(64):
        for y in range(64):
            value = int(255 * x / 64)
            pixels[x, y] = (value, value, value)
    img.save(path, format="PNG")


class CliTests(unittest.TestCase):
    def test_suggest_output_contains_merge_command(self):
        from unittest import mock

        from local_file_organizer import cli as cli_module
        from local_file_organizer.cli import run_person_suggest

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            target.mkdir()
            _save_gradient(target / "a.png")
            _save_gradient(target / "b.png")
            emb_a, emb_b = _unit(0.0), _unit(math.radians(63))

            def _row(x):
                row = [0.0] * 15
                row[0], row[1], row[2], row[3] = x, 5, 30, 30
                row[4], row[5] = x + 8, 15
                row[6], row[7] = x + 22, 15
                row[14] = 0.9
                return row

            rows_by_name = {
                "a.png": np.array([_row(10)], dtype=np.float32),
                "b.png": np.array([_row(50)], dtype=np.float32),
            }

            def _fake_detect(path, **kwargs):
                return (64, 64, rows_by_name[Path(path).name], None)

            def _fake_compute(img, row):
                return emb_a if round(float(row[0])) == 10 else emb_b

            with (
                mock.patch.object(cli_module, "detect_faces_raw", side_effect=_fake_detect),
                mock.patch.object(cli_module, "compute_embedding", side_effect=_fake_compute),
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    run_person_suggest(target, quiet=True)
            printed = output.getvalue()
        self.assertIn("=== PERSON SUGGESTIONS ===", printed)
        self.assertIn("P0001 P0002", printed)
        self.assertIn("similarity=", printed)
        self.assertIn("obs=1/1", printed)
        self.assertIn("files=1/1", printed)
        self.assertIn(
            "python -m local_file_organizer person-merge P0001 P0002", printed
        )
        self.assertIn("suggestions=1", printed)


if __name__ == "__main__":
    unittest.main()
