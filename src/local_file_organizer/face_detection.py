"""Local face detection with YuNet via OpenCV's FaceDetectorYN (V2.3).

Read-only: image files and ephemeral video frame files are only decoded
into memory, never modified. A single bad file never raises — failures
are captured in ``FaceDetectionResult.error``. This milestone only
detects that *a* face-shaped region exists; no embeddings, clustering,
or person IDs (that is V2.4).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_NMS_THRESHOLD = 0.3
DEFAULT_TOP_K = 5000
DEFAULT_MIN_FACE_SIZE_PX = 20

# Lazily-created detectors, keyed by (confidence, nms, top_k) so tests can
# request fresh instances with different thresholds.
_detector_cache: dict = {}


@dataclass
class FaceObservation:
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    detection_confidence: float
    # Width/height of the *whole* source image/frame, not the face box:
    # lets future milestones compute relative face size without re-reading.
    source_width: int
    source_height: int


@dataclass
class FaceDetectionResult:
    width: int  # source image/frame width (0 if error)
    height: int  # source image/frame height (0 if error)
    faces: list[FaceObservation]
    error: Optional[str] = None


def get_face_detector(
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
):
    """Lazily create (and cache) a YuNet FaceDetectorYN instance."""
    key = (confidence_threshold, nms_threshold, top_k)
    if key not in _detector_cache:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"face detection model not found: {MODEL_PATH}")
        _detector_cache[key] = cv2.FaceDetectorYN_create(
            str(MODEL_PATH),
            "",
            (0, 0),
            confidence_threshold,
            nms_threshold,
            top_k,
        )
    return _detector_cache[key]


def detect_faces_raw(
    path: Path, *, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> tuple[int, int, Optional[np.ndarray], Optional[str]]:
    """Detect faces and return the raw YuNet detection rows, never raising.

    Returns (width, height, rows, error). rows is cv2's raw Nx15 float
    array (or None if no faces / on error) — NOT filtered by min_face_size,
    NOT converted to FaceObservation. Same read-only, never-raises contract
    as detect_faces_in_image_file: any failure comes back as `error` with
    rows=None, width=height=0.
    """
    try:
        img = cv2.imread(str(path))
        if img is None:
            return (0, 0, None, "could not decode image")
        height, width = img.shape[0], img.shape[1]
        detector = get_face_detector(confidence_threshold=confidence_threshold)
        detector.setInputSize((width, height))
        _retval, faces = detector.detect(img)
        if faces is None:
            return (width, height, None, None)
        rows = np.asarray(faces, dtype=np.float32)
        if rows.size == 0:
            return (width, height, None, None)
        return (width, height, rows, None)
    except Exception as exc:  # one bad file must never crash the batch
        return (0, 0, None, str(exc))


def detect_faces_in_image_file(
    path: Path,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    min_face_size_px: int = DEFAULT_MIN_FACE_SIZE_PX,
) -> FaceDetectionResult:
    """Detect faces in an image file (or ephemeral video frame), never raising.

    Any failure — unreadable/corrupt file, detector error — is captured in
    ``FaceDetectionResult.error`` with ``faces == []``.
    """
    try:
        width, height, rows, error = detect_faces_raw(
            path, confidence_threshold=confidence_threshold
        )
        if error is not None:
            return FaceDetectionResult(width=0, height=0, faces=[], error=error)
        observations: list[FaceObservation] = []
        if rows is not None:
            for row in rows:
                # cv2 returns Nx15 floats: columns 0-3 are x, y, w, h bbox,
                # column 14 is the confidence score.
                x = max(0, round(float(row[0])))
                y = max(0, round(float(row[1])))
                w = max(0, round(float(row[2])))
                h = max(0, round(float(row[3])))
                confidence = float(row[14])
                # Second quality gate (size) on top of the detector's own
                # confidence threshold: faces smaller than this are dropped.
                # Boundary: exactly min_face_size_px survives (>= comparison).
                if w < min_face_size_px or h < min_face_size_px:
                    continue
                observations.append(
                    FaceObservation(
                        bbox_x=x,
                        bbox_y=y,
                        bbox_width=w,
                        bbox_height=h,
                        detection_confidence=confidence,
                        source_width=width,
                        source_height=height,
                    )
                )
        return FaceDetectionResult(
            width=width, height=height, faces=observations, error=None
        )
    except Exception as exc:  # one bad file must never crash the batch
        return FaceDetectionResult(width=0, height=0, faces=[], error=str(exc))
