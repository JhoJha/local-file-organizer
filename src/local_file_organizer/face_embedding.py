"""Local face embeddings with SFace via OpenCV's FaceRecognizerSF (V2.4).

Privacy: embeddings are NEVER persisted. They live only as transient
in-memory numpy arrays during a single `cluster-people` run, used to build
the clusters, and are discarded the moment that run ends. Never write an
embedding vector to SQLite, to a file, or to stdout/logs, under any flag.
Only the derived, anonymous clustering result (a person label and which
files it appears in) is persisted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "face_recognition_sface_2021dec.onnx"
MODEL_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
EMBEDDING_DIM = 128

_recognizer = None


def get_face_recognizer():
    """Lazily create (and cache) an SFace FaceRecognizerSF instance."""
    global _recognizer
    if _recognizer is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"face recognition model not found: {MODEL_PATH}")
        _recognizer = cv2.FaceRecognizerSF_create(str(MODEL_PATH), "")
    return _recognizer


def compute_embedding(
    image_bgr: np.ndarray, raw_face_row: np.ndarray
) -> Optional[np.ndarray]:
    """Compute a 128-d face embedding for one raw detection row, never raising.

    `raw_face_row` is one row (15 floats) from
    `face_detection.detect_faces_raw`'s output — has the landmarks
    `alignCrop` needs. Returns a plain 1-D `(128,)` float32 numpy array,
    or None on any failure. Pure in-memory transform: does not read or
    write any file. Never log or persist its return value beyond the
    caller's immediate use in the same run.
    """
    try:
        row = np.asarray(raw_face_row, dtype=np.float32).reshape(-1)
        if row.shape != (15,):
            return None
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3:
            return None
        recognizer = get_face_recognizer()
        aligned = recognizer.alignCrop(image_bgr, row)
        feature = recognizer.feature(aligned)
        arr = np.asarray(feature, dtype=np.float32).reshape(-1)
        if arr.shape != (EMBEDDING_DIM,):
            return None
        return arr
    except Exception:
        return None


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance `1 - dot(a,b)/(norm(a)*norm(b))`.

    Guards against zero-norm vectors (returns 1.0, i.e. "maximally
    different", rather than dividing by zero).
    """
    try:
        fa = np.asarray(a, dtype=np.float64).reshape(-1)
        fb = np.asarray(b, dtype=np.float64).reshape(-1)
        na = float(np.linalg.norm(fa))
        nb = float(np.linalg.norm(fb))
        if na == 0.0 or nb == 0.0:
            return 1.0
        cos_sim = float(np.dot(fa, fb) / (na * nb))
        return float(1.0 - cos_sim)
    except Exception:
        return 1.0
