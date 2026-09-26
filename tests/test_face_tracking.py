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
    face_size_px: float = 0.0,
    sharpness: float = 0.0,
) -> FrameFace:
    if embedding is None:
        embedding = _unit_on_circle(0.0)
    return FrameFace(
        timestamp_ms=ts_ms,
        bbox=bbox,
        embedding=np.asarray(embedding, dtype=np.float32),
        confidence=confidence,
        face_size_px=face_size_px,
        sharpness=sharpness,
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
        # T1 update justification: the tracklet mean is now L2-normalized
        # (face-quality T1 hygiene — representatives live on the same unit
        # hypersphere as the normalized per-frame embeddings). Cosine
        # distance is scale-invariant, so no clustering ranking changes;
        # only the magnitude convention changed, hence this expectation.
        emb_a = _unit_on_circle(0.0)
        emb_b = _unit_on_circle(math.radians(10))
        tracklets = build_tracklets([_face(1000.0, embedding=emb_a), _face(2000.0, embedding=emb_b)])
        self.assertEqual(len(tracklets), 1)
        mean = (emb_a.astype(np.float64) + emb_b.astype(np.float64)) / 2
        expected = (mean / np.linalg.norm(mean)).astype(np.float32)
        np.testing.assert_allclose(tracklets[0].representative_embedding(), expected, rtol=1e-5)

    def test_representative_embedding_is_unit_norm(self):
        faces = [_face(1000.0 * (i + 1)) for i in range(3)]
        rep = build_tracklets(faces)[0].representative_embedding()
        self.assertAlmostEqual(float(np.linalg.norm(rep)), 1.0, places=5)

    def test_representative_member_is_highest_confidence(self):
        faces = [_face(1000.0, confidence=0.7), _face(2000.0, confidence=0.95)]
        rep = build_tracklets(faces)[0].representative_member()
        self.assertEqual(rep.timestamp_ms, 2000.0)
        self.assertAlmostEqual(rep.confidence, 0.95)

    def test_quality_defaults_fall_back_to_confidence(self):
        # Fixtures without gate info keep the old behavior: quality == confidence.
        member = _face(1000.0, confidence=0.7)
        self.assertAlmostEqual(member.quality, 0.7)

    def test_representative_member_prefers_quality_over_raw_confidence(self):
        # quality = conf*size*sharpness: a larger+sharper frame wins even
        # with lower raw detector confidence (0.80*900*50 vs 0.95*900*5).
        low_conf = _face(
            1000.0, confidence=0.80, face_size_px=900.0, sharpness=50.0
        )
        high_conf = _face(
            2000.0, confidence=0.95, face_size_px=900.0, sharpness=5.0
        )
        rep = build_tracklets([high_conf, low_conf])[0].representative_member()
        self.assertEqual(rep.timestamp_ms, 1000.0)

    def test_representative_member_tie_break_is_deterministic(self):
        # Identical quality scores -> earliest timestamp, then bbox order,
        # regardless of input order.
        late = _face(
            2000.0,
            bbox=(20, 5, 30, 30),
            confidence=0.9,
            face_size_px=100.0,
            sharpness=10.0,
        )
        early = _face(
            1000.0,
            bbox=(10, 5, 30, 30),
            confidence=0.9,
            face_size_px=100.0,
            sharpness=10.0,
        )
        for ordering in ([late, early], [early, late]):
            rep = build_tracklets(ordering)[0].representative_member()
            self.assertEqual(rep.timestamp_ms, 1000.0)
            self.assertEqual(rep.bbox, (10, 5, 30, 30))


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


class WeightedRepresentativeTests(unittest.TestCase):
    def test_none_derives_quality_weights(self):
        # weights=None weighs by member quality: the representative leans
        # toward the sharp/large/confident frame, away from the plain mean.
        low = _face(
            1000.0,
            embedding=_unit_on_circle(0.0),
            confidence=0.9,
            face_size_px=900.0,
            sharpness=5.0,
        )
        high = _face(
            2000.0,
            embedding=_unit_on_circle(math.radians(60)),
            confidence=0.9,
            face_size_px=900.0,
            sharpness=50.0,
        )
        tracklets = build_tracklets([low, high])
        self.assertEqual(len(tracklets), 1)
        rep = tracklets[0].representative_embedding()
        plain = (
            low.embedding.astype(np.float64) + high.embedding.astype(np.float64)
        ) / 2
        plain = plain / np.linalg.norm(plain)
        # Representative is closer (higher cosine similarity) to the
        # high-quality member than the plain mean is.
        self.assertGreater(float(rep @ high.embedding), float(plain @ high.embedding))
        self.assertLess(float(rep @ low.embedding), float(plain @ low.embedding))

    def test_uniform_explicit_weights_reproduce_plain_mean_exactly(self):
        # T2 acceptance: uniform weights -> the old mean, bit for bit,
        # even when member qualities differ.
        low = _face(
            1000.0,
            embedding=_unit_on_circle(0.0),
            confidence=0.9,
            face_size_px=900.0,
            sharpness=5.0,
        )
        high = _face(
            2000.0,
            embedding=_unit_on_circle(math.radians(60)),
            confidence=0.9,
            face_size_px=900.0,
            sharpness=50.0,
        )
        tracklets = build_tracklets([low, high])
        rep = tracklets[0].representative_embedding(weights=[2.0, 2.0])
        mean = (
            low.embedding.astype(np.float64) + high.embedding.astype(np.float64)
        ) / 2
        expected = (mean / np.linalg.norm(mean)).astype(np.float32)
        np.testing.assert_array_equal(rep, expected)

    def test_invalid_weights_fall_back_to_plain_mean(self):
        faces = [
            _face(1000.0, embedding=_unit_on_circle(0.0), confidence=0.9),
            _face(2000.0, embedding=_unit_on_circle(math.radians(20)), confidence=0.9),
        ]
        tracklet = build_tracklets(faces)[0]
        expected = tracklet.representative_embedding(weights=[1.0, 1.0])
        for bad in ([1.0], [1.0, -1.0], [0.0, 0.0], [float("nan"), 1.0]):
            np.testing.assert_array_equal(
                tracklet.representative_embedding(weights=bad), expected
            )

    def test_zero_weight_member_is_ignored(self):
        # A zero weight is valid (not a fallback): the representative is
        # the L2-normalized surviving member.
        faces = [
            _face(1000.0, embedding=_unit_on_circle(0.0), confidence=0.9),
            _face(2000.0, embedding=_unit_on_circle(math.radians(20)), confidence=0.9),
        ]
        tracklet = build_tracklets(faces)[0]
        rep = tracklet.representative_embedding(weights=[0.0, 3.0])
        np.testing.assert_allclose(
            rep, faces[1].embedding.astype(np.float32), rtol=1e-5
        )

    def test_weighted_representative_stays_unit_norm(self):
        faces = [
            _face(
                1000.0 * (i + 1),
                embedding=_unit_on_circle(math.radians(10 * i)),
                confidence=0.5 + 0.1 * i,
                face_size_px=900.0,
                sharpness=5.0 * (i + 1),
            )
            for i in range(3)
        ]
        rep = build_tracklets(faces)[0].representative_embedding()
        self.assertAlmostEqual(float(np.linalg.norm(rep)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
