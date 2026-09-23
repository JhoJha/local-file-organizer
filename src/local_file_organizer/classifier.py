"""Extension-based file classification into general categories."""
from __future__ import annotations

# Mapping is intentionally simple and easy to extend: add extensions to an
# existing set, or add a new "category: {extensions}" entry.
CATEGORY_EXTENSIONS: dict[str, set[str]] = {
    "video": {
        ".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".m4v", ".flv",
        ".mpeg", ".mpg",
    },
    "image": {
        ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".heic",
    },
    "document": {
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt",
        ".csv", ".odt",
    },
    "audio": {
        ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".opus",
    },
    "archive": {
        ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz",
    },
    "software": {
        ".exe", ".msi", ".iso", ".apk", ".deb", ".appimage",
    },
}

OTHER_CATEGORY = "other"

_EXTENSION_TO_CATEGORY: dict[str, str] = {
    ext: category
    for category, extensions in CATEGORY_EXTENSIONS.items()
    for ext in extensions
}


def classify(extension: str) -> str:
    """Return the general category for a given file extension.

    ``extension`` may be passed with or without the leading dot and in any
    case; unknown extensions map to :data:`OTHER_CATEGORY`.
    """
    if not extension:
        return OTHER_CATEGORY
    normalized = extension.lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return _EXTENSION_TO_CATEGORY.get(normalized, OTHER_CATEGORY)
