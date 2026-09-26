"""Tests for S1 video temporal tracklets (pure numpy, no models).

All embeddings/frames are synthetic: hand-built 128-d vectors and
invented bboxes/timestamps. No real video files, no detector, no SFace.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from local_file_organizer.face_clustering import (
    ClusterableObservation,
    cluster_people,
)
from local_file_organizer.face_tracking import (
    FrameFace,
    bbox_iou,
    build_tracklets,
)


def _unit_on_circle(angle_rad: float) -> np.ndarray:
    v = np.zeros(128, dtype=np.float32)
    v[0] = math.cos(angle_rad)
    v[1] = math.sin(angle_rad)
    return v


def _face(
    ts_ms: float,
    bbox=(10, 5, 30, 30),
    embedding=None,
    confidence: float = 0.9,
) -> FrameFace:
    if embedding is None:
        embedding = _unit_on_circle(0.0)
    return FrameFace(
        timestamp_ms=ts_ms,
        bbox=bbox,
        embedding=np.asarray(embedding, dtype=np.float32),
        confidence=confidence,
    )


def _tracklet_signature(tracklets) -> list:
    return sorted(
        (
            tuple((m.timestamp_ms, m.bbox) for m in t.members),
            tuple(np.asarray(t.representative_embedding()).tolist()),
        )
        for t in tracklets
    )


class BboxIouTests(unittest.TestCase):
    def test_identical_boxes_give_one(self):
        self.assertAlmostEqual(bbox_iou((10, 5, 30, 30), (10, 5, 30, 30)), 1.0)

    def test_disjoint_boxes_give_zero(self):
        self.assertEqual(bbox_iou((0, 0, 10, 10), (50, 50, 10, 10)), 0.0)

    def test_zero_area_boxes_give_zero(self):
        self.assertEqual(bbox_iou((10, 5, 0, 30), (10, 5, 30, 30)), 0.0)


class TrackletLinkingTests(unittest.TestCase):
    def test_same_face_across_frames_collapses_to_one_tracklet(self):
        faces = [_face(ts) for ts in (1000.0, 2000.0, 3000.0, 4000.0)]
        tracklets = build_tracklets(faces)
        self.assertEqual(len(tracklets), 1)
        self.assertEqual(len(tracklets[0].members), 4)

    def test_different_embedding_stays_separate(self):
        faces = [
            _face(1000.0, embedding=_unit_on_circle(0.0)),
            _face(2000.0, embedding=_unit_on_circle(math.pi)),
        ]
        tracklets = build_tracklets(faces)
        self.assertEqual(len(tracklets), 2)

    def test_disjoint_bbox_stays_separate_despite_same_embedding(self):
        faces = [_face(1000.0, bbox=(10, 5, 30, 30)), _face(2000.0, bbox=(200, 150, 30, 30))]
        tracklets = build_tracklets(faces)
        self.assertEqual(len(tracklets), 2)

    def test_no_gap_bridging_across_faceless_frame(self):
        faces = [_face(1000.0), _face(3000.0)]
        tracklets = build_tracklets(faces, frame_ms=[1000.0, 2000.0, 3000.0])
        self.assertEqual(len(tracklets), 2)

    def test_two_persons_tracked_independently(self):
        box_a, box_b = (10, 5, 30, 30), (100, 5, 30, 30)
        faces = [
            _face(1000.0, bbox=box_a, embedding=_unit_on_circle(0.0)),
            _face(1000.0, bbox=box_b, embedding=_unit_on_circle(math.pi)),
            _face(2000.0, bbox=box_a, embedding=_unit_on_circle(0.0)),
            _face(2000.0, bbox=box_b, embedding=_unit_on_circle(math.pi)),
        ]
        tracklets = build_tracklets(faces)
        self.assertEqual(len(tracklets), 2)
        self.assertTrue(all(len(t.members) == 2 for t in tracklets))

    def test_representative_embedding_is_mean(self):
        emb_a = _unit_on_circle(0.0)
        emb_b = _unit_on_circle(math.radians(10))
        tracklets = build_tracklets([_face(1000.0, embedding=emb_a), _face(2000.0, embedding=emb_b)])
        self.assertEqual(len(tracklets), 1)
        expected = ((emb_a.astype(np.float64) + emb_b.astype(np.float64)) / 2).astype(np.float32)
        np.testing.assert_allclose(tracklets[0].representative_embedding(), expected, rtol=1e-5)

    def test_representative_member_is_highest_confidence(self):
        faces = [_face(1000.0, confidence=0.7), _face(2000.0, confidence=0.95)]
        rep = build_tracklets(faces)[0].representative_member()
        self.assertEqual(rep.timestamp_ms, 2000.0)
        self.assertAlmostEqual(rep.confidence, 0.95)


class TrackletDeterminismTests(unittest.TestCase):
    def test_shuffled_input_and_repeat_runs_give_identical_tracklets(self):
        faces = [_face(ts, confidence=0.5 + ts / 100000.0) for ts in (1000.0, 2000.0, 3000.0)]
        faces += [_face(2000.0, bbox=(100, 5, 30, 30), embedding=_unit_on_circle(math.pi))]
        first = _tracklet_signature(build_tracklets(faces))
        second = _tracklet_signature(build_tracklets(list(reversed(faces))))
        third = _tracklet_signature(build_tracklets(list(faces)))
        self.assertEqual(first, second)
        self.assertEqual(first, third)


class TrackletClusteringEndToEndTests(unittest.TestCase):
    def test_tracklet_representatives_collapse_drifting_singletons(self):
        # 8 adjacent frames drift 10 degrees each: consecutive frames link
        # (similarity ~0.985) but first-to-last drift (70 degrees) exceeds
        # the clustering threshold, so raw observations split.
        angles = [math.radians(10 * i) for i in range(8)]
        faces = [
            _face(1000.0 * (i + 1), embedding=_unit_on_circle(a)) for i, a in enumerate(angles)
        ]
        tracklets = build_tracklets(faces)
        self.assertEqual(len(tracklets), 1)
        raw = [ClusterableObservation(key=(i,), embedding=f.embedding) for i, f in enumerate(faces)]
        self.assertEqual(len(cluster_people(raw, threshold=0.5)), 2)
        reps = [
            ClusterableObservation(key=(0,), embedding=tracklets[0].representative_embedding())
        ]
        self.assertEqual(len(cluster_people(reps, threshold=0.5)), 1)


if __name__ == "__main__":
    unittest.main()
