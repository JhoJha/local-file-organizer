"""Local read-only preview server for the organization plan viewer.

Serves the static viewer directory (``index.html`` plus ``assets/``) and the
original target files addressed only by opaque id (``/file/F000NNN``,
resolved through the catalog plus the plan with the same positional/id
discipline as ``organize-apply``). Bound strictly to 127.0.0.1 for manual
local runs only: any other bind host is refused, directory listing is
disabled, only GET and HEAD are served (anything else is refused with
405), and every served byte must resolve inside the explicit target root
(outside paths, traversal attempts and symlink escapes all get a plain
404 with no path disclosure). Nothing is ever written or mutated.
"""

from __future__ import annotations

import mimetypes
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from . import organize_apply as apply_mod

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PREVIEW_SERVE_PORT = 8123

_FILE_ID_RE = re.compile(r"^F\d{6}$")
_SAFE_ASSET_RE = re.compile(r"^[A-Za-z0-9._-]+\.[A-Za-z0-9]+$")


class PreviewServeError(Exception):
    """Raised when the preview server cannot start or serve safely."""


def resolve_preview_files(db_path: Path, plan_path: Path) -> dict[str, str]:
    """Map opaque ``F000NNN`` ids to catalog source paths.

    Delegates to ``organize_apply.resolve_source_paths`` so ids stay bound
    to the catalog positionally (count plus id plus source hash must match,
    otherwise the plan is stale and the server refuses to start).
    """
    try:
        entries = apply_mod.load_plan_entries(plan_path)
        return apply_mod.resolve_source_paths(db_path, entries)
    except apply_mod.OrganizeApplyError as exc:
        raise PreviewServeError(str(exc)) from exc


def _real(path: Path) -> Path:
    return Path(os.path.realpath(path))


def _inside(candidate: Path, root: Path) -> bool:
    return candidate != root and root in candidate.parents


def _make_handler(
    view_root: Path, target_root: Path, paths_by_file_id: dict[str, str]
) -> type[BaseHTTPRequestHandler]:
    real_view = _real(view_root)
    real_target = _real(target_root)

    class PreviewHandler(BaseHTTPRequestHandler):
        server_version = "PreviewServe/1.0"

        def log_message(self, *args: object) -> None:  # keep stderr quiet
            pass

        def _send(self, status: int, ctype: str, data: bytes, head_only: bool) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if not head_only and data:
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def _deny(self, head_only: bool) -> None:
            self._send(404, "text/plain; charset=utf-8", b"not found", head_only)

        def _refuse_method(self) -> None:
            self._send(405, "text/plain; charset=utf-8", b"method not allowed", False)

        do_POST = _refuse_method
        do_PUT = _refuse_method
        do_DELETE = _refuse_method
        do_PATCH = _refuse_method
        do_OPTIONS = _refuse_method

        def do_GET(self) -> None:
            self._route(head_only=False)

        def do_HEAD(self) -> None:
            self._route(head_only=True)

        def _route(self, head_only: bool) -> None:
            try:
                path = unquote(urlsplit(self.path).path)
            except Exception:  # noqa: BLE001 - malformed request targets 404
                self._deny(head_only)
                return
            if "\x00" in path:
                self._deny(head_only)
                return
            if path in ("/", "/index.html"):
                self._serve_static(real_view / "index.html", head_only)
            elif path == "/file" or path.startswith("/file/"):
                rest = path[len("/file/"):] if path.startswith("/file/") else ""
                self._serve_file_id(rest.strip("/"), head_only)
            elif path.startswith("/assets/"):
                self._serve_static(real_view / "assets" / path[len("/assets/"):], head_only)
            else:
                self._deny(head_only)

        def _serve_static(self, candidate: Path, head_only: bool) -> None:
            name = candidate.name
            if not _SAFE_ASSET_RE.match(name) and candidate.name != "index.html":
                self._deny(head_only)
                return
            real = _real(candidate)
            if not _inside(real, real_view) or not real.is_file():
                self._deny(head_only)
                return
            try:
                data = real.read_bytes()
            except OSError:
                self._deny(head_only)
                return
            ctype = mimetypes.guess_type(str(real))[0] or "application/octet-stream"
            if real.suffix.lower() == ".html":
                ctype = "text/html; charset=utf-8"
            self._send(200, ctype, data, head_only)

        def _serve_file_id(self, file_id: str, head_only: bool) -> None:
            if not _FILE_ID_RE.match(file_id) or file_id not in paths_by_file_id:
                self._deny(head_only)
                return
            real = _real(Path(paths_by_file_id[file_id]))
            if not _inside(real, real_target) or not real.is_file():
                self._deny(head_only)
                return
            try:
                data = real.read_bytes()
            except OSError:
                self._deny(head_only)
                return
            ctype = mimetypes.guess_type(str(real))[0] or "application/octet-stream"
            self._send(200, ctype, data, head_only)

    return PreviewHandler


def create_preview_server(
    *,
    view_dir: Path,
    target: Path,
    paths_by_file_id: dict[str, str],
    host: str = LOOPBACK_HOST,
    port: int = DEFAULT_PREVIEW_SERVE_PORT,
) -> ThreadingHTTPServer:
    """Build (bound, not yet serving) the read-only preview server.

    Refuses any bind host other than 127.0.0.1, a missing viewer page, a
    missing target directory, or an out-of-range port. Callers start it
    with ``serve_forever()``; tests introspect ``server_address`` instead
    of opening LAN sockets.
    """
    if host != LOOPBACK_HOST:
        raise PreviewServeError(
            f"refused: preview-serve binds {LOOPBACK_HOST} only, got {host!r}"
        )
    if not 0 <= port <= 65535:
        raise PreviewServeError(f"refused: invalid port {port!r}")
    if not target.is_dir():
        raise PreviewServeError(f"refused: target root is not a directory: {target}")
    if not (view_dir / "index.html").is_file():
        raise PreviewServeError(f"refused: no viewer page at {view_dir / 'index.html'}")
    handler = _make_handler(view_dir, target, paths_by_file_id)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def run_preview_server(
    *,
    target: Path,
    db_path: Path,
    plan_path: Path,
    view_dir: Path | None = None,
    host: str = LOOPBACK_HOST,
    port: int = DEFAULT_PREVIEW_SERVE_PORT,
) -> None:
    """Resolve ids and serve the viewer plus originals until interrupted."""
    resolved_view = view_dir if view_dir is not None else plan_path.parent / "organization_view"
    mapping = resolve_preview_files(db_path, plan_path)
    server = create_preview_server(
        view_dir=resolved_view, target=target, paths_by_file_id=mapping, host=host, port=port
    )
    bound_port = server.server_address[1]
    print(f"Serving viewer {resolved_view} and {len(mapping)} files by id")
    print(f"Open http://{LOOPBACK_HOST}:{bound_port}/ (local only, read-only)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
