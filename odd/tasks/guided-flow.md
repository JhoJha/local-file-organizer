# Feature: guided-flow

## Objective
Make the tool usable end to end: installable `lfo` command plus a guided `organize` command that runs the pipeline in order and resumes.

## Problem
Organizing a folder requires ~20 standalone subcommands in the right order, and there is no console entry point.

## Why
User request (2026-09-28): "quiero que sea funcional" -> usable. Approved items 1 and 2.

## Scope
- T1 — console entry point `lfo` in `[project.scripts]`; README install section (pipx).
- T2 — `organize <folder>` guided command: scan -> sample-frames -> detect-faces -> cluster-people -> face-review (pause: user reviews HTML) -> organize-preview (pause: confirm) -> organize-apply. Resumable, progress output, reuses existing functions. Existing subcommands stay unchanged.

Out of scope: web UI, cli.py split, schema migrations.

## Constraints
- Originals never modified outside the existing organize-apply safety path; apply only after explicit confirmation (`--yes` for non-interactive).
- No new dependencies. Existing tests must stay green.

## Authorized scope
User authorized items 1 and 2. Work-unit commits on feature/guided-flow authorized; push/PR/merge are user decisions.

## TDD
Mode: off (no project config). Runner: `pytest`. Ordinary checks: `pytest -q`, `ruff check`.

## Tasks
- [x] T1 — lfo entry point + README install. Route: delegated direct (bundled with T2 writer). Commit: 6f7b2e4.
- [x] T2 — guided `organize` command + tests + README. Route: delegated direct (writer trigger: cli.py + new module + tests). Commit: 5dd1f12.

## Verification
- `python -m pytest -q`: 436 passed.
- `ruff check src tests`: all checks passed.

## Next step
Done — both tasks implemented and verified.
