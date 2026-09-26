"""Temporal face tracklets within a single video (S1).

Same-face detections across adjacent sampled frames are linked into a
tracklet, so that one person lingering in a video contributes ONE
representative embedding to the global clustering instead of one raw
observation per sampled frame (the pilot symptom: every frame clustered
independently produced hundreds of singleton clusters).

Linking rule (deterministic): detections are sorted by
(timestamp_ms, bbox) and consumed frame by frame in timestamp order.
Within a frame, each detection (in bbox order) joins the open tracklet
— a tracklet whose last member is in the immediately previous sampled
frame — with the highest bbox IoU, provided IoU >= TRACKLET_MIN_IOU
*and* cosine similarity to that last member >= TRACKLET_SIMILARITY.
Each open tracklet matches at most one detection per frame; unmatched
detections start new tracklets. No gap bridging: a sampled frame with
no linkable detection ends the chain, so frames t-1 and t+1 never link
directly across t.

Privacy: embeddings stay transient in-memory numpy arrays, exactly like
the raw observations they replace — this module never persists
anything. CPU only, no new dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .face_embedding import cosine_distance, weighted_mean

# Minimum cosine SIMILARITY between a detection and a tracklet's last
# member to allow linking. Deliberately the same strictness (0.50) as
# face_clustering.DEFAULT_SIMILARITY_THRESHOLD: linking two different
# people into one tracklet is worse than splitting one person in two,
# because a wrongful link can never be undone downstream.
TRACKLET_SIMILARITY = 0.50

# Minimum bbox IoU between a detection and a tracklet's last member.
# Lenient on purpose: sampled frames are seconds apart (not 30fps
# video), so the same face may have moved noticeably; the strict
# embedding gate above is what prevents wrong merges.
TRACKLET_MIN_IOU = 0.2


@dataclass
class FrameFace:
    """One face detection in one sampled video frame (memory-only)."""

    timestamp_ms: float
    bbox: tuple[int, int, int, int]  # (x, y, width, height)
    embedding: np.ndarray  # (128,) float32, never persisted
    confidence: float
    # Quality fields, populated by the ingestion quality gate
    # (face_detection.passes_quality_gate): box AREA (w*h) in px and
    # Laplacian-variance sharpness of the ROI. They default to 0.0 so
    # callers without quality info (older tests, synthetic fixtures)
    # keep working: `quality` then falls back to plain confidence.
    # Naming note (R2-001): `face_size_px` is a box AREA (w*h, in px^2),
    # not a side length — it compares against nothing, it only feeds the
    # quality product below. The side-length gate is `min_face_size_px`
    # (each of w/h must clear it) in face_detection/cli. The historic
    # field name is kept for backward compatibility.
    face_size_px: float = 0.0
    sharpness: float = 0.0
    # Best-frame score: confidence*size*sharpness when quality info is
    # present, else confidence. None means "derive in __post_init__".
    quality: Optional[float] = None

    def __post_init__(self) -> None:
        if self.quality is None:
            if self.face_size_px > 0.0 and self.sharpness > 0.0:
                self.quality = (
                    float(self.confidence)
                    * float(self.face_size_px)
                    * float(self.sharpness)
                )
            else:
                self.quality = float(self.confidence)


@dataclass
class Tracklet:
    """One person followed across adjacent sampled frames."""

    members: list[FrameFace] = field(default_factory=list)

    def representative_embedding(self, weights=None) -> np.ndarray:
        """L2-normalized mean embedding of the members (float32).

        Averaging same-face observations denoises per-frame jitter; the
        mean is then L2-normalized to unit norm so the tracklet
        representative lives on the same hypersphere as the (also
        normalized) per-frame embeddings — cosine distance is
        scale-invariant, so this changes no ranking, it only keeps
        magnitudes comparable for the downstream clustering. A
        zero-norm mean (degenerate all-zero members) cannot be
        normalized and is returned as-is.

        Quality weighting (T2): `weights=None` (the default) derives one
        weight per member from `quality` (confidence*size*sharpness when
        the ingestion gate populated it, else plain confidence), so
        sharp, large, confident frames steer the representative and
        blurry marginal frames count less. An explicit `weights` list
        (one non-negative float per member) overrides that; uniform
        explicit weights reproduce the old plain mean exactly (same code
        path, bit for bit). Invalid weights (wrong length, non-finite,
        negative, all-zero) fall back to the plain mean.
        """
        stacked = np.stack(
            [np.asarray(m.embedding, dtype=np.float64) for m in self.members]
        )
        if weights is None:
            weights = [
                m.quality if m.quality is not None else m.confidence
                for m in self.members
            ]
        mean = weighted_mean(stacked, weights)
        norm = float(np.linalg.norm(mean))
        if norm == 0.0 or not np.isfinite(norm):
            return np.asarray(mean, dtype=np.float32)
        return np.asarray(mean / norm, dtype=np.float32)

    def representative_member(self) -> FrameFace:
        """Best-quality member (ties: earliest frame, then bbox order).

        Ranks by `quality` (confidence*size*sharpness when the
        ingestion gate populated it, else plain confidence), so the
        persisted observation detail row (bbox, timestamp, confidence
        shown in review) carries the sharpest detection as evidence.
        The (timestamp_ms, bbox) tie-break keeps the choice
        deterministic for identical scores.
        """
        return min(
            self.members,
            key=lambda m: (
                -(m.quality if m.quality is not None else m.confidence),
                m.timestamp_ms,
                m.bbox,
            ),
        )


def bbox_iou(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes. Zero-area boxes give 0.0."""
    ax, ay, aw, ah = (float(v) for v in box_a)
    bx, by, bw, bh = (float(v) for v in box_b)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    inter_w = min(ax + aw, bx + bw) - max(ax, bx)
    inter_h = min(ay + ah, by + bh) - max(ay, by)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def build_tracklets(
    detections: list[FrameFace],
    *,
    frame_ms: list[float] | None = None,
    min_iou: float = TRACKLET_MIN_IOU,
    similarity: float = TRACKLET_SIMILARITY,
) -> list[Tracklet]:
    """Link per-frame detections into tracklets (see module docstring).

    `frame_ms` is the full sampled timestamp sequence (in ms) of the
    video, including frames where nothing was detected: a face-less
    frame still breaks every open chain (no gap bridging). When omitted,
    adjacency is judged over the frames that produced detections only.
    Input order does not matter: detections are grouped per timestamp
    and frames are consumed in timestamp order with in-frame bbox
    order, so the same input always yields the same tracklets.
    """
    by_ts: dict[float, list[FrameFace]] = {}
    for detection in detections:
        by_ts.setdefault(detection.timestamp_ms, []).append(detection)
    if frame_ms is None:
        stamps = sorted(by_ts)
    else:
        stamps = sorted(set(frame_ms) | set(by_ts))
    tracklets: list[Tracklet] = []
    # Tracklets ending in the previous sampled frame, by tracklet index
    # (insertion order == creation order, used as the IoU tie-break).
    previous_open: dict[int, Tracklet] = {}
    for stamp in stamps:
        frame = sorted(by_ts.get(stamp, []), key=lambda d: d.bbox)
        available = dict(previous_open)
        current_open: dict[int, Tracklet] = {}
        for detection in frame:
            best_key: int | None = None
            best_iou = 0.0
            for key, tracklet in available.items():
                iou = bbox_iou(detection.bbox, tracklet.members[-1].bbox)
                if iou < min_iou:
                    continue
                last_emb = tracklet.members[-1].embedding
                if 1.0 - cosine_distance(detection.embedding, last_emb) < similarity:
                    continue
                if best_key is None or iou > best_iou:
                    best_key = key
                    best_iou = iou
            if best_key is None:
                tracklets.append(Tracklet(members=[detection]))
                current_open[len(tracklets) - 1] = tracklets[-1]
            else:
                tracklet = available.pop(best_key)
                tracklet.members.append(detection)
                current_open[best_key] = tracklet
        previous_open = current_open
    return tracklets


__all__ = [
    "TRACKLET_MIN_IOU",
    "TRACKLET_SIMILARITY",
    "FrameFace",
    "Tracklet",
    "bbox_iou",
    "build_tracklets",
]
