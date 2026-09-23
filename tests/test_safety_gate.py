"""Safety gate: the source code must never call filesystem-mutating
operations. This is a static, defense-in-depth check on top of the
behavioral tests in test_cli.py/test_scanner.py, so a future change that
introduces a destructive call fails CI immediately instead of only being
caught by chance in a runtime test.
"""
import re
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "local_file_organizer"

FORBIDDEN_PATTERNS = [
    r"\bos\.unlink\s*\(",
    r"\bos\.remove\s*\(",
    r"\bos\.rmdir\s*\(",
    r"\bos\.removedirs\s*\(",
    r"\bos\.rename\s*\(",
    r"\bos\.replace\s*\(",
    r"\bPath(\.[a-zA-Z_]+)?\.unlink\s*\(",
    r"\bPath(\.[a-zA-Z_]+)?\.rmdir\s*\(",
    r"\bPath(\.[a-zA-Z_]+)?\.rename\s*\(",
    r"\bPath(\.[a-zA-Z_]+)?\.replace\s*\(",
    r"\bshutil\.move\s*\(",
    r"\bshutil\.copy2?\s*\(",
    r"\bshutil\.copytree\s*\(",
    r"\bshutil\.rmtree\s*\(",
    r"\bos\.chmod\s*\(",
    r"\bos\.chown\s*\(",
    r"\bos\.utime\s*\(",
]

# face_review.save_decisions() uses tmp-file + os.replace to atomically
# write the human decisions JSON, always to a caller-supplied path under
# the project (never the scanned tree). It is the only module allowed to
# use os.replace, and only for that atomic local JSON write.
ALLOWED_REPLACE_FILES = {"face_review.py"}


class SafetyGateTests(unittest.TestCase):
    def test_no_destructive_or_write_operations_in_source(self):
        python_files = sorted(SRC_DIR.rglob("*.py"))
        self.assertTrue(python_files, "expected to find source files to scan")

        violations = []
        for path in python_files:
            text = path.read_text(encoding="utf-8")
            for pattern in FORBIDDEN_PATTERNS:
                if (
                    pattern == r"\bos\.replace\s*\("
                    and path.name in ALLOWED_REPLACE_FILES
                ):
                    continue
                for match in re.finditer(pattern, text):
                    violations.append(f"{path}: {match.group(0)!r}")

        self.assertEqual(
            violations,
            [],
            "Found forbidden filesystem-mutating calls in source:\n"
            + "\n".join(violations),
        )

    def test_writes_are_confined_to_known_output_modules(self):
        # database.connect() (SQLite), reporting.export_csv() (CSV),
        # snapshot.save_snapshot() (JSON), organization_plan.write_plan_json()
        # (JSON) and face_review (manifest/HTML/thumbnails/decisions JSON) are
        # the only places allowed to write to disk, and each only writes to a
        # path supplied by the CLI (defaulting to this project's own data/
        # and reports/ directories), never to the scanned tree. Every other
        # module may only *read* (stat, open for reading, hash).
        allowed_write_files = {
            "database.py",
            "reporting.py",
            "snapshot.py",
            "organization_plan.py",
            "face_review.py",
        }
        write_call_patterns = [
            r"open\(([^)]*)\)",
            r"\.write_text\s*\(",
            r"\.write_bytes\s*\(",
        ]
        for path in sorted(SRC_DIR.rglob("*.py")):
            if path.name in allowed_write_files or path.name == "__init__.py":
                continue
            text = path.read_text(encoding="utf-8")

            for match in re.finditer(r"open\(([^)]*)\)", text):
                args = match.group(1)
                self.assertNotRegex(
                    args,
                    r"[\"']([wa])[\"']|mode\s*=\s*[\"'][wa]",
                    f"unexpected write-mode open() in {path}: {match.group(0)}",
                )
            for pattern in write_call_patterns[1:]:
                self.assertNotRegex(
                    text,
                    pattern,
                    f"unexpected write call in {path} (only {sorted(allowed_write_files)} may write)",
                )


if __name__ == "__main__":
    unittest.main()
