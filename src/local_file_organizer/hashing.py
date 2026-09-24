"""Read-only exact-duplicate detection.

Strategy (never hashes indiscriminately):
1. Group candidate files by size.
2. Discard groups where the size is unique (can't have a duplicate).
3. Only for repeated sizes, compute a SHA-256 digest.
4. Group equal hashes together and assign a stable DUP_##### id per group.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .scanner import is_regular_file

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def sha256_file(path: Path, chunk_size: int = _HASH_CHUNK_SIZE) -> str:
    """Compute the SHA-256 digest of a file by streaming it in chunks.

    Opens the file for reading only; never writes, truncates or moves it.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class DuplicateRecord:
    path: str
    size_bytes: int


def group_by_size(records: Iterable[DuplicateRecord]) -> dict[int, list[DuplicateRecord]]:
    """Group records by size_bytes, keeping only groups with 2+ members."""
    by_size: dict[int, list[DuplicateRecord]] = defaultdict(list)
    for record in records:
        by_size[record.size_bytes].append(record)
    return {size: items for size, items in by_size.items() if len(items) > 1}


@dataclass
class DuplicateStats:
    # path -> assigned group id, e.g. "DUP_00001"
    group_by_path: dict[str, str]
    total_duplicate_files: int
    total_duplicate_groups: int
    duplicated_space_bytes: int
    recoverable_space_bytes: int
    # path -> SHA-256 computed during this run (F1: persisted to the
    # catalog so future scans reuse digests without re-reading bytes).
    digests: dict[str, str]


def find_duplicates(
    records: Iterable[DuplicateRecord],
    hash_fn=sha256_file,
    on_hash_error=None,
    known_digests: Optional[dict[str, str]] = None,
) -> DuplicateStats:
    """Run the full size -> hash duplicate-detection pipeline.

    ``hash_fn`` is injectable for testing. ``on_hash_error(path, exc)`` is
    called (if provided) when a candidate file can't be hashed; that file is
    simply excluded from duplicate grouping rather than aborting the run.

    ``known_digests`` (F1 incremental) maps path -> stored SHA-256: files
    whose bytes were not re-read reuse their catalog digest, so only
    new/changed files in size-collision groups pay for hashing. A stored
    None (never hashed) falls back to hashing now.
    """
    size_groups = group_by_size(records)
    known = known_digests or {}

    hash_groups: dict[str, list[DuplicateRecord]] = defaultdict(list)
    digests: dict[str, str] = dict(known)
    for candidates in size_groups.values():
        for record in candidates:
            digest = known.get(record.path)
            if digest is None:
                if not is_regular_file(Path(record.path)):
                    if on_hash_error is not None:
                        on_hash_error(record.path, OSError("not a regular file"))
                    digests.pop(record.path, None)
                    continue
                try:
                    digest = hash_fn(Path(record.path))
                except OSError as exc:
                    if on_hash_error is not None:
                        on_hash_error(record.path, exc)
                    # A fresh read failure invalidates any previously
                    # stored digest for this path.
                    digests.pop(record.path, None)
                    continue
                digests[record.path] = digest
            hash_groups[digest].append(record)

    confirmed_groups = {
        digest: items for digest, items in hash_groups.items() if len(items) > 1
    }

    # Deterministic ordering: largest recoverable space first, then hash.
    ordered = sorted(
        confirmed_groups.items(),
        key=lambda kv: (-kv[1][0].size_bytes, kv[0]),
    )

    group_by_path: dict[str, str] = {}
    total_duplicate_files = 0
    duplicated_space_bytes = 0
    recoverable_space_bytes = 0

    for index, (_digest, items) in enumerate(ordered, start=1):
        group_id = f"DUP_{index:05d}"
        for record in items:
            group_by_path[record.path] = group_id
        total_duplicate_files += len(items)
        group_size = items[0].size_bytes
        duplicated_space_bytes += group_size * len(items)
        recoverable_space_bytes += group_size * (len(items) - 1)

    return DuplicateStats(
        group_by_path=group_by_path,
        total_duplicate_files=total_duplicate_files,
        total_duplicate_groups=len(ordered),
        duplicated_space_bytes=duplicated_space_bytes,
        recoverable_space_bytes=recoverable_space_bytes,
        digests=digests,
    )
