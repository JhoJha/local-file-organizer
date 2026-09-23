"""Tests for V2.4 face embeddings (SFace via OpenCV FaceRecognizerSF).

Embeddings are NEVER persisted: these tests only hold them as transient
in-memory arrays and never write them anywhere.
"""
from __future__ import annotations

import hashlib
import unittest

import numpy as np

from local_file_organizer.face_embedding import (
    EMBEDDING_DIM,
    MODEL_PATH,
    MODEL_SHA256,
    compute_embedding,
    cosine_distance,
)


def _fake_row():
    return np.array(
        [80, 80, 120, 120, 110, 120, 150, 120, 130, 145, 115, 165, 145, 165, 0.9],
        dtype=np.float32,
    )


def _synthetic_image():
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    img[:] = (100, 150, 200)
    return img


class ModelIntegrityTests(unittest.TestCase):
    def test_vendored_model_sha256(self):
        digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, MODEL_SHA256)


class RealPipelineSmokeTests(unittest.TestCase):
    def test_compute_embedding_returns_128d_float32(self):
        # Real SFace model, synthetic content: proves alignCrop/feature run
        # end-to-end even though the "face" content is meaningless.
        emb = compute_embedding(_synthetic_image(), _fake_row())
        self.assertIsNotNone(emb)
        assert emb is not None
        self.assertEqual(emb.shape, (128,))
        self.assertEqual(emb.dtype, np.float32)
        self.assertEqual(EMBEDDING_DIM, 128)

    def test_determinism_self_similarity(self):
        img = _synthetic_image()
        row = _fake_row()
        e1 = compute_embedding(img, row)
        e2 = compute_embedding(img, row)
        self.assertIsNotNone(e1)
        self.assertIsNotNone(e2)
        assert e1 is not None and e2 is not None
        self.assertLess(cosine_distance(e1, e2), 1e-4)


class CosineDistanceTests(unittest.TestCase):
    def test_identical_vectors_give_zero(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.assertAlmostEqual(cosine_distance(a, a.copy()), 0.0, places=6)

    def test_orthogonal_vectors_give_one(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(cosine_distance(a, b), 1.0, places=6)

    def test_zero_norm_does_not_raise(self):
        a = np.zeros(4, dtype=np.float32)
        b = np.ones(4, dtype=np.float32)
        self.assertEqual(cosine_distance(a, b), 1.0)


class EmbeddingFailureTests(unittest.TestCase):
    def test_malformed_row_returns_none_not_raise(self):
        img = _synthetic_image()
        # Wrong length: not a 15-float YuNet row.
        self.assertIsNone(compute_embedding(img, np.array([1, 2, 3], dtype=np.float32)))
        self.assertIsNone(compute_embedding(img, np.zeros((7,), dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
