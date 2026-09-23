"""Read-only integrity snapshots.

Captures (relative_path, size, mtime_ns) — and optionally a SHA-256 for
small sets — for every file under a directory, so two snapshots taken
before and after a scan can be diffed to *prove* the scan didn't add,
remove, or modify anything in the tree it read.

Snapshot files themselves are written only to a path the caller supplies,
which must live under this project (e.g. under data/snapshots/) — never
inside the scanned tree.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .hashing import sha256_file
from .scanner import DEFAULT_IGNORED_DIRS, iter_files


@dataclass(frozen=True)
class SnapshotEntry:
    size: int
    mtime_ns: int
    sha256: Optional[str] = None


def capture_snapshot(
    root: Path,
    ignored_dirs: Iterable[str] = DEFAULT_IGNORED_DIRS,
    include_hash: bool = False,
    on_error=None,
) -> dict[str, SnapshotEntry]:
    """Build a map of posix-style relative path -> SnapshotEntry for every
    file under ``root``.

    Only ``os.stat`` (via ``Path.stat``) is used per file; content is only
    read when ``include_hash=True``, which is meant for small sets (e.g. a
    pilot folder) — never enable it for a large, unfiltered collection.
    """
    root = root.resolve()
    snapshot: dict[str, SnapshotEntry] = {}
    for file_path in iter_files(root, ignored_dirs=ignored_dirs, on_error=on_error):
        try:
            stat_result = file_path.stat()
            digest = sha256_file(file_path) if include_hash else None
        except OSError:
            continue
        relative = file_path.relative_to(root).as_posix()
        snapshot[relative] = SnapshotEntry(
            size=stat_result.st_size, mtime_ns=stat_result.st_mtime_ns, sha256=digest
        )
    return snapshot


def save_snapshot(snapshot: dict[str, SnapshotEntry], output_path: Path) -> None:
    """Write a snapshot as JSON to ``output_path``.

    Callers must pass a path under this project (e.g. data/snapshots/); this
    function does not validate that on its own.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        path: {"size": e.size, "mtime_ns": e.mtime_ns, "sha256": e.sha256}
        for path, e in snapshot.items()
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_snapshot(input_path: Path) -> dict[str, SnapshotEntry]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return {
        path: SnapshotEntry(
            size=entry["size"], mtime_ns=entry["mtime_ns"], sha256=entry.get("sha256")
        )
        for path, entry in payload.items()
    }


@dataclass
class SnapshotDiff:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    size_changed: list[str] = field(default_factory=list)
    mtime_changed: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (self.added or self.removed or self.size_changed or self.mtime_changed)


def diff_snapshots(
    before: dict[str, SnapshotEntry], after: dict[str, SnapshotEntry]
) -> SnapshotDiff:
    """Compare two snapshots and report exactly what changed, if anything."""
    diff = SnapshotDiff()
    before_keys = set(before)
    after_keys = set(after)

    diff.added = sorted(after_keys - before_keys)
    diff.removed = sorted(before_keys - after_keys)

    for path in sorted(before_keys & after_keys):
        b, a = before[path], after[path]
        if b.size != a.size:
            diff.size_changed.append(path)
        if b.mtime_ns != a.mtime_ns:
            diff.mtime_changed.append(path)

    return diff
