"""Read-only recursive directory walk.

This module only lists and stats files. It never creates, moves, renames or
deletes anything inside the scanned tree.
"""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

# Configurable: names are matched case-sensitively against each directory's
# basename, anywhere in the tree.
DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".cache",
        ".npm",
        ".docker",
    }
)


@dataclass
class ScanError:
    path: str
    message: str


def is_regular_file(path: Path) -> bool:
    """True only for regular files (symlinks are followed, like stat).

    FIFOs, sockets, devices and anything un-stat-able return False.
    Content readers (hashing, image decoding, ffprobe/exiftool spawns)
    must check this first: opening a FIFO for reading blocks forever,
    and no timeout can interrupt a blocked read — a single FIFO in the
    tree used to hang the whole scan. The inventory itself (stat-only)
    still records such entries.
    """
    try:
        return stat.S_ISREG(os.stat(path).st_mode)
    except OSError:
        return False


def iter_files(
    root: Path,
    ignored_dirs: Iterable[str] = DEFAULT_IGNORED_DIRS,
    on_error: Optional[callable] = None,
) -> Iterator[Path]:
    """Yield every file path under ``root``, skipping ignored directories.

    Permission errors while listing a directory are reported via
    ``on_error(ScanError)`` and that subtree is simply skipped; the walk
    continues elsewhere. ``sudo`` is never invoked and permissions are never
    changed to work around an unreadable path.
    """
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    ignored = set(ignored_dirs)

    def _on_walk_error(exc: OSError) -> None:
        if on_error is not None:
            on_error(ScanError(path=exc.filename or str(root), message=str(exc)))

    for dirpath, dirnames, filenames in os.walk(
        root, onerror=_on_walk_error, followlinks=False
    ):
        dirnames[:] = [d for d in dirnames if d not in ignored]
        current_dir = Path(dirpath)
        for filename in filenames:
            yield current_dir / filename
