"""Tests for V2.3 local face detection (YuNet via OpenCV FaceDetectorYN).

Real-face detection cannot be meaningfully tested with synthetic images:
YuNet is a trained CNN and correctly finds zero faces in flat or abstract
pictures. So the real pipeline is exercised only for the zero-face case,
while parsing/filtering logic uses a stub detector patched in at the
``get_face_detector`` boundary.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from local_file_organizer import face_detection
from local_file_organizer.face_detection import (
    MODEL_PATH,
    MODEL_SHA256,
    detect_faces_in_image_file,
)


def _save_solid(path: Path, color=(200, 50, 50), size=(64, 64)):
    Image.new("RGB", size, color=color).save(path, format="PNG")


class _StubDetector:
    """Minimal stand-in for cv2's FaceDetectorYN: rows are Nx15 floats
    (columns 0-3: x, y, w, h bbox; column 14: confidence)."""

    def __init__(self, rows):
        self._rows = rows
        self.input_size = None

    def setInputSize(self, size):
        self.input_size = size

    def detect(self, img):
        if not self._rows:
            return 1, None
        return 1, np.array(self._rows, dtype=np.float32)


def _row(x, y, w, h, confidence=0.9):
    full = [0.0] * 15
    full[0], full[1], full[2], full[3] = x, y, w, h
    full[14] = confidence
    return full


class RealDetectorZeroFaceTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.dir = Path(self.tmpdir.name)

    def test_solid_image_has_no_faces_but_pipeline_runs(self):
        # Exercises the real cv2/model pipeline end-to-end (model loads,
        # setInputSize/detect run without crashing); YuNet correctly finds
        # zero faces in a flat image with no photographic texture.
        path = self.dir / "solid.png"
        _save_solid(path)
        result = detect_faces_in_image_file(path)
        self.assertIsNone(result.error)
        self.assertEqual(result.faces, [])
        self.assertEqual((result.width, result.height), (64, 64))


class MockedDetectorTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.dir = Path(self.tmpdir.name)
        self.image = self.dir / "any.png"
        _save_solid(self.image, size=(100, 80))

    def _detect_with_rows(self, rows, **kwargs):
        stub = _StubDetector(rows)
        with mock.patch.object(face_detection, "get_face_detector", return_value=stub):
            return detect_faces_in_image_file(self.image, **kwargs), stub

    def test_single_face_parsed(self):
        result, stub = self._detect_with_rows([_row(10, 20, 30, 40, 0.95)])
        self.assertIsNone(result.error)
        self.assertEqual(len(result.faces), 1)
        face = result.faces[0]
        self.assertEqual(
            (face.bbox_x, face.bbox_y, face.bbox_width, face.bbox_height),
            (10, 20, 30, 40),
        )
        self.assertAlmostEqual(face.detection_confidence, 0.95)
        # Source dims describe the whole image, not the face box.
        self.assertEqual((face.source_width, face.source_height), (100, 80))
        self.assertEqual((result.width, result.height), (100, 80))
        # Detector got (width, height) — cv2 shape is (h, w, c), order matters.
        self.assertEqual(stub.input_size, (100, 80))

    def test_multiple_faces(self):
        rows = [_row(1, 2, 30, 30), _row(40, 50, 25, 25), _row(70, 10, 20, 20)]
        result, _ = self._detect_with_rows(rows)
        self.assertEqual(len(result.faces), 3)

    def test_fractional_bbox_is_rounded_not_truncated(self):
        result, _ = self._detect_with_rows([_row(10.6, 20.4, 30, 30)])
        self.assertEqual((result.faces[0].bbox_x, result.faces[0].bbox_y), (11, 20))

    def test_negative_xy_clamped_to_zero(self):
        # cv2 can return small negative x/y for boxes near the edge.
        result, _ = self._detect_with_rows([_row(-3.2, -1.0, 30, 30)])
        self.assertEqual((result.faces[0].bbox_x, result.faces[0].bbox_y), (0, 0))

    def test_small_faces_filtered_by_min_face_size(self):
        rows = [_row(5, 5, 10, 10), _row(40, 40, 50, 50)]
        result, _ = self._detect_with_rows(rows, min_face_size_px=20)
        self.assertEqual(len(result.faces), 1)
        self.assertEqual((result.faces[0].bbox_width, result.faces[0].bbox_height), (50, 50))

    def test_min_face_size_boundary_is_inclusive(self):
        # A face exactly at the threshold survives; one px under does not.
        rows = [_row(0, 0, 20, 20), _row(30, 30, 19, 20)]
        result, _ = self._detect_with_rows(rows, min_face_size_px=20)
        self.assertEqual(len(result.faces), 1)
        self.assertEqual(result.faces[0].bbox_width, 20)

    def test_confidence_threshold_forwarded_to_detector(self):
        seen = {}

        def _fake_get(confidence_threshold=0.6, **_kwargs):
            seen["confidence"] = confidence_threshold
            return _StubDetector([])

        with mock.patch.object(face_detection, "get_face_detector", side_effect=_fake_get):
            detect_faces_in_image_file(self.image, confidence_threshold=0.85)
        self.assertEqual(seen["confidence"], 0.85)


class CorruptImageTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.dir = Path(self.tmpdir.name)

    def test_corrupt_image_returns_error_not_raise(self):
        path = self.dir / "bad.png"
        path.write_bytes(b"this is not a valid PNG file")
        result = detect_faces_in_image_file(path)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.faces, [])
        self.assertEqual((result.width, result.height), (0, 0))


class ModelIntegrityTests(unittest.TestCase):
    def test_vendored_model_sha256(self):
        digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, MODEL_SHA256)


if __name__ == "__main__":
    unittest.main()
