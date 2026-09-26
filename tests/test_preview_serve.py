"""Tests for preview-serve (V2 read-only localhost server) and Abrir buttons.

Every fixture is synthetic (temp directories only); no test touches real
personal files. The server binds 127.0.0.1 only (ephemeral ports in tests,
asserted via introspection, never a LAN socket), serves the viewer plus
originals by opaque id, and refuses outside-root bytes, traversal, unknown
ids, and non-GET methods.
"""

import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from local_file_organizer import database, preview_serve
from local_file_organizer.organization_preview import generate_organization_preview


def _add_record(conn, path: Path, category: str) -> None:
    database.upsert_file(
        conn,
        database.FileRecord(
            path=str(path),
            name=path.name,
            extension=path.suffix.lower(),
            category=category,
            scan_status="ok",
            metadata_status="skipped",
            file_state="active",
        ),
    )


class PreviewServeTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)
        self.target = self.root / "target"
        self.inside_a = self.target / "a.txt"
        self.inside_b = self.target / "b.bin"
        self.outside = self.root / "outside" / "secret.txt"
        for path, data in (
            (self.inside_a, b"hello preview"),
            (self.inside_b, b"\x00\x01\x02binary"),
            (self.outside, b"top secret bytes"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.db_path = self.root / "catalog.sqlite"
        self.plan_path = self.root / "output" / "organization_plan.json"
        conn = database.connect(self.db_path)
        for path in (self.inside_a, self.inside_b, self.outside):
            _add_record(conn, path, "document")
        conn.commit()
        conn.close()
        result = generate_organization_preview(
            db_path=self.db_path, plan_path=self.plan_path
        )
        self.view_dir = result.view_path.parent
        self.mapping = preview_serve.resolve_preview_files(self.db_path, self.plan_path)
        self.by_path = {path: fid for fid, path in self.mapping.items()}
        self.server = preview_serve.create_preview_server(
            view_dir=self.view_dir,
            target=self.target,
            paths_by_file_id=self.mapping,
            host="127.0.0.1",
            port=0,
        )
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def _get(self, path: str, method: str = "GET"):
        request = urllib.request.Request(self._url(path), method=method)
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def test_serves_viewer_index(self):
        for path in ("/", "/index.html"):
            status, body = self._get(path)
            self.assertEqual(status, 200, path)
            self.assertIn(b"Organization Plan Viewer", body)

    def test_serves_file_bytes_by_id(self):
        file_id = self.by_path[str(self.inside_a)]
        status, body = self._get(f"/file/{file_id}")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"hello preview")
        status, body = self._get(f"/file/{file_id}", method="HEAD")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"")

    def test_outside_root_refused_without_disclosure(self):
        file_id = self.by_path[str(self.outside)]
        status, body = self._get(f"/file/{file_id}")
        self.assertEqual(status, 404)
        self.assertNotIn(b"top secret", body)
        self.assertNotIn(str(self.outside).encode(), body)

    def test_traversal_refused(self):
        file_id = self.by_path[str(self.inside_a)]
        for nasty in (
            "/file/../secret",
            "/file/%2e%2e/secret",
            f"/file/{file_id}/../../outside/secret.txt",
            "/assets/../../outside/secret.txt",
            "/assets/%2e%2e/%2e%2e/x",
            "/etc/passwd",
        ):
            status, _ = self._get(nasty)
            self.assertEqual(status, 404, nasty)

    def test_non_get_refused(self):
        file_id = self.by_path[str(self.inside_a)]
        for method in ("POST", "PUT", "DELETE"):
            status, _ = self._get(f"/file/{file_id}", method=method)
            self.assertEqual(status, 405, method)
        status, _ = self._get("/", method="POST")
        self.assertEqual(status, 405)

    def test_unknown_id_404(self):
        for path in ("/file/F999999", "/file/not-an-id", "/file/"):
            status, _ = self._get(path)
            self.assertEqual(status, 404, path)

    def test_binds_localhost_only(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        for host in ("0.0.0.0", "localhost", "::"):
            with self.assertRaises(preview_serve.PreviewServeError, msg=host):
                preview_serve.create_preview_server(
                    view_dir=self.view_dir,
                    target=self.target,
                    paths_by_file_id=self.mapping,
                    host=host,
                    port=0,
                )


if __name__ == "__main__":
    unittest.main()
