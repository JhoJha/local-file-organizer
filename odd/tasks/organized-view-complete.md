# Feature: organized-view-complete

## Objective
Make the final plan real: view organized files as browsable output and select/filter by tag, person, and other facets, plus a safe real apply with dry-run/rollback.

## Problem
Today the organized plan exists only as `reports/organization_plan.json` (preview_only) with CLI text output. There is no HTML viewer for the plan, no user tag model, and no `organize-apply`. `search` + `persons` filtering exists only as CLI text.

## Why
User request (2026-09-26): "poder ver mis archivos organizados y seleccionar por etiqueta, persona, etc" with scope `Completo con apply` (viewer + tags + real apply).

## Scope
- T1: read-only HTML viewer for `organization_plan.json` with filters (destination/person/confidence), offline static like `face_review`, opaque IDs only (no real paths by default).
- T2: user tag model (create/tag/untag/filter) integrated into `query.py` + `search` CLI.
- T3: `organize-apply` with dry-run default, rollback, extended safety-gate. Real moves only via explicit apply.

Out of scope: cloud sync, generative AI, OCR/full-text (still pending per README), V3.3 collections / V3.4 rules engine beyond tags.

## Constraints
- LOCAL ONLY: never touch scanned tree except via explicit `organize-apply`; all writes under `data/` and `reports/` otherwise.
- Privacy: viewer/plan JSON use opaque IDs (`F000NNN`, `P000N`/`PERSON_XXXX`); real paths only behind explicit `--show-files`-style manual flag, never by default.
- No `sudo`, no permission changes; read errors recorded, scan continues.
- Existing safety-gate `tests/test_safety_gate.py` must keep passing and extend to apply.

## Authorized scope
User explicitly authorized implementation of viewer + tags + apply (answer: `Completo con apply`). Work-unit commits on feature branch are authorized. Push/PR/merge remain user decisions under ordinary repo policy.

## Tasks
- [x] T1 — plan-viewer-html: static offline viewer for organization plan with destination/person/confidence filters.
  - Route: delegated direct (writer trigger fired: 5 files touched).
  - Acceptance: MET. `organize-preview` emits `reports/organization_view/index.html` offline with destination/person/confidence filters, opaque IDs, no paths by default (`--show-files` manual only). Flags `--view-dir`/`--no-view` added. Commit 819c85c.
  - Checks: observed — unittest 310 OK (295+15), ruff clean.
- [x] T2 — user-tags model + search integration.
  - Route: delegated direct (writer trigger fired: 6 files touched).
  - Acceptance: MET. `tag`/`untag`/`tags` CLI (path or F000NNN refs), persist with auto-migration, `search --tag` repeatable AND, stale excluded, tags never reach plan JSON/viewer. Commit f4cad6a (slice-2 = 819c85c..f4cad6a).
  - Checks: observed — unittest 329 OK (310+19), ruff clean.
- [x] T3 — organize-apply dry-run/rollback.
  - Route: delegated direct (writer trigger fired: safety-critical, 5 files touched).
  - Acceptance: MET. `organize-apply` dry-run default, real run only with `--apply`/`--yes` + snapshots, rollback manifest + `organize-rollback`, target-root containment, review/unsupported skipped unless `--include-review`, no overwrites/deletes, safety-gate extended deliberately. Commit e42b472 (slice-3 = f4cad6a..e42b472).
  - Checks: observed — unittest 347 OK (329+18), ruff clean.
  - Correction (native review, lineage review-6b113ac458f7e511): reviewer found R3-001 (deterministic BLOCKER: no source-path binding) + R3-002 (inferential CRITICAL, refuter-corroborated: manifest after after-snapshot). One bounded correction committed as 4cc978c (55 lines ≤ 60 cap: src_hash per plan entry + fail-closed verify, manifest-before-snapshot; 349 tests OK, ruff clean).
  - Recovery (user-authorized) → successor lineage review-7a1c94be02f5d683 → 4R round found R4-duplicate-destination-overwrite (deterministic BLOCKER: intra-run same-destination overwrite). One bounded correction committed as fcd5f4e (37 lines ≤ 40 cap: seen-set in preflight + regression test; 350 tests OK, ruff clean) → targeted validation PASSED → APPROVED with 3 advisory WARNINGs (non-blocking follow-ups, no correction offered). Authority burned. Reviewed boundary now fcd5f4e.

## TDD
- Resolved mode: unknown (no project/session TDD config found; test files present does not enable TDD).
- Source: none. Runner candidates: README `PYTHONPATH=src python -m unittest discover -s tests -v` and dev `pytest>=8`. Writer must report which runner was used as evidence; ordinary functional checks required (not no checks).

## Delivery
- Strategy: `ask-on-risk` (default, user did not specify).
- Chain strategy: `feature-branch-chain` (user chose `Cadena con tracker`, 2026-09-26). Slices: T1=819c85c; T2 and T3 each land as separate work-unit commits on feature/organized-view-complete (the tracker); PRs (when user asks) chain tracker-first. Recorded boundaries: slice-1 (T1 viewer) = 37b1726..819c85c, reviewed APPROVED.
- Forecast (authored +/- excl. generated): T1 ~250, T2 ~300, T3 ~450+; total >400, so `ask-on-risk` will trigger chain-strategy question before the commit that pushes running count over ~400. Slice boundaries recorded here when known.
- Running count: T1 ~660 + T2 ~636 + T3 ~1000 native lines (slices reviewed separately under feature-branch-chain).

## RDD
- Mode: on (global on, clone-local unset, read 2026-09-26).
- Candidate per task = work-unit commit (never a checkbox, never accumulated branch). After each commit: `gentle-ai review assess --cwd <repo> --agent opencode --base-ref <last reviewed boundary> --committed-only --json`, follow `review_due`/`next_transition`. First boundary = branch point of feature branch.
- T1: tier medium, 660 lines → granted → review-reliability → approved, 0 findings → acknowledged, burned (lineage review-0e66ac246a828059). Boundary was 819c85c.
- T2: tier medium (executable_change cli.py), 636 changed lines, review_due slice_budget_reached → granted → 1 lens (review-reliability) → approved, 0 findings → acknowledged, authority burned (lineage review-4716150d86982b05). Reviewed boundary now f4cad6a; T3 assesses with --base-ref f4cad6a.

## Progress
- 2026-09-26: feature doc created. Base refactor committed as 37b1726 on refactor/hito-2b-scan; branched feature/organized-view-complete.
- 2026-09-26: T1 closed (commit 819c85c `feat(preview): emit static offline HTML viewer for organization plan`). Files: organization_plan.py, organization_preview.py, cli.py, tests/test_plan_view.py (15 new), README.md.
- 2026-09-26: T2 closed (commit f4cad6a `feat(tags): add user tag model with search integration`). Files: database.py, user_tags.py (new), query.py, cli.py, tests/test_user_tags.py (19 new), README.md.
- 2026-09-26: T3 closed (commit e42b472 `feat(apply): add organize-apply with dry-run default and rollback`). Files: organize_apply.py (new), cli.py, tests/test_organize_apply.py (18 new), tests/test_safety_gate.py (extended), README.md.
- 2026-09-26: T3 reviewed APPROVED (successor lineage review-7a1c94be02f5d683, authority burned). Advisory follow-ups (WARNING, separate later work): crafted-plan containment test, directory-source refusal test, rollback interrupt coverage.
- Next: DONE — all 3 tasks closed and reviewed. Push/PR/merge are user decisions under ordinary repo policy.

## Follow-ups (advisory, non-blocking)
- Crafted-plan containment refusal deserves a direct test (organize_apply.py:340).
- Directory-as-source refusal path deserves a direct test (organize_apply.py:490).
- Rollback interrupt/partial behavior deserves tests (organize_apply.py:710).

## Verification evidence
- T1: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests` → 310 tests OK. `ruff check src tests` → clean. Manual synthetic /tmp catalog run: viewer emitted, 0 network tokens, 0 path leaks by default.
- T2: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests` → 329 tests OK (8.8s). `ruff check src tests` → clean. CLI smoke on temp catalog (tag→tags→search --tag→untag→tags): expected output, counts 1→0.
- T3: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests` → 347 tests OK (.venv python 3.14.4). `ruff check src tests` → clean. CLI smoke: apply/rollback --help render; dry-run asserts DRY RUN + would_move=3, tree unchanged.

## Verification evidence
- T1: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests` → 310 tests OK. `ruff check src tests` → clean. Manual synthetic /tmp catalog run: viewer emitted, 0 network tokens, 0 path leaks by default.
- T2: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests` → 329 tests OK (8.8s). `ruff check src tests` → clean. CLI smoke on temp catalog (tag→tags→search --tag→untag→tags): expected output, counts 1→0.

## Relevant files
- src/local_file_organizer/organization_plan.py — plan JSON builder + static viewer renderer (T1).
- src/local_file_organizer/organization_preview.py — preview generation + viewer emission (T1).
- src/local_file_organizer/query.py — structured filters + tag field (T2).
- src/local_file_organizer/user_tags.py — tag model (T2, new).
- src/local_file_organizer/cli.py — all CLI commands incl. tag/search/preview/apply/rollback (T1-T3).
- src/local_file_organizer/database.py — catalog schema/migrations.
- reports/organization_plan.json — current preview_only artifact.
- tests/test_safety_gate.py — read-only gate to extend.
