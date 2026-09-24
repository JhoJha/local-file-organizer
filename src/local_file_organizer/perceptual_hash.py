"""Read-only perceptual image hashing (dHash) and similarity grouping.

Nothing in this module writes, moves, renames or deletes the files it
reads: images are only opened for decoding, resized in memory, and
discarded. This is deliberately a *different* concept from the exact
SHA-256 duplicate detection in :mod:`hashing`: two visually similar but
byte-different images (recompressed, resized, re-saved) get a low Hamming
distance here while never matching on SHA-256, and vice versa.

Algorithm choice (V2.1): dHash (difference hash), not the classical
pHash (DCT-based). dHash needs only a grayscale downscale and adjacent-
pixel comparisons — no frequency-domain transform — which keeps the
implementation small and easy to reason about while still being a real
perceptual hash (robust to minor recompression/resizing, unlike a byte
hash). The trade-off is weaker robustness to rotation and heavy cropping
than pHash; acceptable for a conservative first iteration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image

from .scanner import is_regular_file

SIMILARITY_METHOD = "dhash8x8"
_HASH_SIZE = 8  # 8x8 -> 64-bit hash, from a resized (hash_size+1) x hash_size image

# Incrementality contract (V3.2/F1): this version string is stored per
# file in files.extractor_versions. Bump it if the algorithm changes so
# the next scan reprocesses images instead of trusting stale hashes.
DHASH_VERSION = "dhash_v1"

# Hamming distance threshold (out of 64 bits) below which two images are
# considered "similar enough" to group. Chosen conservatively: dHash
# distances of 0-5 are near-identical (recompression/minor resize), 6-10
# still tend to be genuinely related images, and above that false
# positives (unrelated images that happen to share coarse light/dark
# structure) become common. 8 favors precision over recall, per the
# "conservative grouping" requirement for this iteration.
DEFAULT_SIMILARITY_THRESHOLD = 8


@dataclass
class ImageHashResult:
    perceptual_hash: Optional[str] = None
    similarity_method: Optional[str] = None
    error: Optional[str] = None


def compute_dhash(path: Path, hash_size: int = _HASH_SIZE) -> str:
    """Compute a difference hash (dHash) for an image, read-only.

    Opens the image, converts to grayscale, and resizes to
    ``(hash_size + 1) x hash_size``. Each bit compares a pixel to its
    right neighbor (1 if strictly brighter). The result is packed into a
    hex string. Raises on any failure; callers should catch broadly since
    Pillow surfaces many distinct exception types for malformed or
    unidentifiable image data.
    """
    with Image.open(path) as img:
        small = img.convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        )
        pixels = list(small.getdata())

    bits = []
    row_width = hash_size + 1
    for row in range(hash_size):
        offset = row * row_width
        for col in range(hash_size):
            bits.append(1 if pixels[offset + col] > pixels[offset + col + 1] else 0)

    value = 0
    for bit in bits:
        value = (value << 1) | bit
    hex_digits = (hash_size * hash_size) // 4
    return format(value, f"0{hex_digits}x")


def get_image_metadata(path: Path) -> ImageHashResult:
    """Compute perceptual-hash metadata for one image, never raising.

    Mirrors ``metadata.get_video_metadata``: any failure (corrupt file,
    truncated data, an extension that doesn't actually contain an image)
    is captured in ``ImageHashResult.error`` instead of propagating, so a
    single bad image never aborts the overall scan.
    """
    if not is_regular_file(path):
        return ImageHashResult(error="not a regular file: skipped (FIFO/socket/device?)")
    try:
        digest = compute_dhash(path)
    except Exception as exc:  # Pillow raises many distinct types for bad image data.
        return ImageHashResult(error=f"{type(exc).__name__}: {exc}")
    return ImageHashResult(perceptual_hash=digest, similarity_method=SIMILARITY_METHOD)


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Bit-difference count between two equal-length hex-encoded hashes."""
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


@dataclass
class SimilarityRecord:
    path: str
    perceptual_hash: str
    size_bytes: int = 0
    extension: str = ""


@dataclass
class SimilarityStats:
    # path -> assigned group id, e.g. "SIM_00001"
    group_by_path: dict[str, str]
    # group id -> maximum pairwise Hamming distance within that group
    group_max_distance: dict[str, int]
    total_grouped_files: int
    total_groups: int


def max_pairwise_distance(hashes: list[str]) -> int:
    """Maximum Hamming distance over all pairs in ``hashes`` (0 if <2)."""
    best = 0
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            best = max(best, hamming_distance(hashes[i], hashes[j]))
    return best


def group_similar_images(
    records: Iterable[SimilarityRecord],
    threshold: int = DEFAULT_SIMILARITY_THRESHOLD,
) -> SimilarityStats:
    """Group images so every group satisfies ``max_pairwise <= threshold``.

    Greedy complete-linkage threshold clustering (O(n^2), fine at
    personal-collection scale): the input is first sorted by path for
    determinism, then each image joins the first existing cluster whose
    *every* member is within ``threshold`` bits; otherwise it starts a new
    cluster. A singleton cluster is left ungrouped, same as exact-duplicate
    detection leaves unique files without a ``duplicate_group``.

    This deliberately prefers false negatives over false positives: unlike
    union-find/connected-components (where A~B and B~C forces A,B,C
    together even if A-C exceeds the threshold), here C only joins a group
    containing A when ``distance(C, A) <= threshold`` as well.
    """
    items = sorted(list(records), key=lambda r: r.path)
    n = len(items)

    clusters: list[list[int]] = []
    for idx in range(n):
        placed = False
        for cluster in clusters:
            if all(
                hamming_distance(
                    items[idx].perceptual_hash, items[member].perceptual_hash
                )
                <= threshold
                for member in cluster
            ):
                cluster.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])

    clusters = [indices for indices in clusters if len(indices) > 1]
    # Explicit sort by smallest path: with sorted input the creation order
    # already satisfies this, but sorting again keeps group IDs stable even
    # if the construction loop ever changes.
    clusters.sort(key=lambda idxs: min(items[i].path for i in idxs))

    group_by_path: dict[str, str] = {}
    group_max_distance: dict[str, int] = {}
    total_grouped_files = 0

    for index, idxs in enumerate(clusters, start=1):
        group_id = f"SIM_{index:05d}"
        max_distance = max_pairwise_distance(
            [items[i].perceptual_hash for i in idxs]
        )
        for i in idxs:
            group_by_path[items[i].path] = group_id
        group_max_distance[group_id] = max_distance
        total_grouped_files += len(idxs)

    return SimilarityStats(
        group_by_path=group_by_path,
        group_max_distance=group_max_distance,
        total_grouped_files=total_grouped_files,
        total_groups=len(clusters),
    )
