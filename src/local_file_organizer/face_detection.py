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

# Minimum face-ROI sharpness (variance of the Laplacian over the
# grayscale ROI) for a detection to survive the quality gate below.
# Deliberately a conservative floor, not a deblurring bar: sharp real
# faces score in the tens-to-hundreds, while solid-color or
# decode-artifact boxes score ~0. The floor only rejects those
# near-flat, uninformative ROIs; small or low-contrast but usable faces
# still pass. Boundary is inclusive (>= survives).
DEFAULT_BLUR_THRESHOLD = 2.0

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


def face_roi_sharpness(
    image_bgr: np.ndarray, x: int, y: int, w: int, h: int
) -> float:
    """Variance of the Laplacian over the grayscale face ROI, never raising.

    Higher = sharper. Returns 0.0 when the ROI is empty, out of bounds,
    or unmeasurable for any reason (which the quality gate rejects).
    Pure in-memory measure over the already-loaded image: no file I/O.
    """
    try:
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.ndim != 3
            or w <= 0
            or h <= 0
        ):
            return 0.0
        img_h, img_w = image_bgr.shape[0], image_bgr.shape[1]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(img_w, x + w), min(img_h, y + h)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        roi = image_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if not np.isfinite(sharpness) or sharpness < 0.0:
            return 0.0
        return sharpness
    except Exception:
        return 0.0


def passes_quality_gate(
    image_bgr: np.ndarray,
    raw_face_row: np.ndarray,
    *,
    min_face_size_px: int = DEFAULT_MIN_FACE_SIZE_PX,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
) -> tuple[bool, float, float]:
    """Shared ingestion quality gate over one raw YuNet row, never raising.

    Re-checks confidence (`row[14]`) and size (`row[2:4]`) explicitly —
    `detect_faces_raw` rows are unfiltered, so every ingestion site
    (cluster/suggest × image/video) must apply the same bar — then
    applies the blur gate (`face_roi_sharpness` over the already-loaded
    image). Returns `(ok, face_size_px, sharpness)` where `face_size_px`
    is the `w*h` box area and `sharpness` the Laplacian variance; both
    are 0.0 when the gate rejects. All comparisons are inclusive
    (>= survives). A `None`/undecodable image never passes.
    """
    try:
        row = np.asarray(raw_face_row, dtype=np.float32).reshape(-1)
        if row.shape != (15,):
            return (False, 0.0, 0.0)
        if float(row[14]) < confidence_threshold:
            return (False, 0.0, 0.0)
        w = round(float(row[2]))
        h = round(float(row[3]))
        if w < min_face_size_px or h < min_face_size_px:
            return (False, 0.0, 0.0)
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3:
            return (False, 0.0, 0.0)
        x = max(0, round(float(row[0])))
        y = max(0, round(float(row[1])))
        face_size_px = float(w * h)
        sharpness = face_roi_sharpness(image_bgr, x, y, w, h)
        if sharpness < blur_threshold:
            return (False, 0.0, 0.0)
        return (True, face_size_px, sharpness)
    except Exception:
        return (False, 0.0, 0.0)
