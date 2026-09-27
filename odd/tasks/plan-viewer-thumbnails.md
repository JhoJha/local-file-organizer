# Feature: plan-viewer-thumbnails

## Objective
The organization plan viewer shows opaque IDs but no visual content. Add thumbnails to every viewer row plus an "Open" action to view the full file — all local-only.

## Problem
User pilot feedback (2026-09-26, screenshot): "no puedo ver los archivos, sería bueno tener previsualización y la opción de verlo". Today `reports/organization_view/index.html` lists F000NNN/category/destination/persons with no images and no way to open the file.

## Why
Pilot continuation on user's folder copy. Viewing what each entry IS is required before any real apply.

## Scope
- V1 — thumbnails in viewer: images → downscaled JPEG (long edge ~256px) into `organization_view/assets/` with opaque names (`F000NNN.jpg`); videos → middle sampled frame via the existing ephemeral V2.2 sampling, saved as thumbnail (no persistent frames beyond the thumb). Viewer table gains a thumbnail column; page stays fully offline/static. Thumbnails are a derived local cache like `face_review/assets/` (documented, never inside the scanned tree).
- V2 — open action: new `preview-serve` command, a read-only HTTP server bound strictly to 127.0.0.1 serving the view dir plus target files addressed by opaque id (`/file/F000NNN`); viewer "Abrir" buttons link there. Refuses any path outside the target root; directory listing off; no uploads/mutations. Only for local manual runs.
- Privacy: no real paths in HTML by default (unchanged rule); thumbnails contain image content by design (same posture as face_review thumbnails — local only, documented).

Out of scope: editing files from viewer, cloud/sync, video streaming/transcoding (single frame thumb only), face-review changes.

## Constraints
- LOCAL ONLY: thumbs under `reports/organization_view/assets/` (or `--view-dir`); server binds 127.0.0.1, GET-only, target-root containment enforced.
- Existing privacy tests keep passing; viewer still path-free by default.
- Safety gate untouched (no new mutating primitives: Pillow decode + JPEG write to reports only; server is read-only).

## Authorized scope
User explicitly requested thumbnails + open action during pilot. Work-unit commits on feature/organized-view-complete authorized. Push/PR/merge remain user decisions.

## Tasks
- [x] V1 — viewer thumbnails (images + video frame thumbs, assets, offline page).
  - Route: delegated direct (writer trigger fired).
  - Acceptance: MET. Preview column with F000NNN.jpg thumbs (images 256px, videos middle frame), placeholders on failure, offline, no paths default, plan schema untouched. Split per user call (size:exception declined): slice-A efa2cab (code, 207 lines, existing 350 tests OK in isolated worktree) + slice-B 9f551df (tests/docs, 304 lines, full 359 OK, ruff clean).
  - Checks: slice-A self-verified (350 OK in isolated worktree + ruff; native assess medium/under_budget → no review due, stays pending). Slice-B: full suite 359 OK + ruff clean, native 4R APPROVED with 0 findings (lineage review-0f23a88cef0cf37b, burned). Pilot viewer regenerated with all 30 thumbs, served at http://localhost:8091.
- [x] V2 — preview-serve localhost read-only server + Abrir buttons.
  - Route: delegated direct.
  - Acceptance: MET in two slices per user call: slice-1 6b00333 (server+CLI+server tests, 412 lines, 366 OK + ruff) + slice-2 5aea5c3 (Abrir buttons+README, 74 lines, 367 OK + ruff).
  - Checks: slice-1 server (6b00333, 412 lines): 366 OK + ruff, native reliability review APPROVED 0 findings (lineage review-354a7d76e221d1c7, burned). Slice-2 buttons (5aea5c3, 74 lines): 367 OK + ruff, native assess medium/under_budget → no review due, self-verified pending. V2 proven live on pilot data (file-by-id 200, traversal 404, POST 405, unknown 404).

## TDD
- Resolved mode: unknown (unchanged). Ordinary functional checks required; writer reports runner used.

## Delivery
- Strategy: `ask-on-risk` (inherited). Forecast: V1 ~200, V2 ~250; if running total pushes the branch over ~400 new lines vs last reviewed boundary, ask chain question before committing (tracker already feature-branch-chain).

## RDD
- Mode: on. Candidate per task = work-unit commit. Assess with --base-ref <last reviewed boundary> (currently fcd5f4e), follow review_due/consent flow.

## Progress
- 2026-09-26: feature doc created from pilot feedback. V1 done + reviewed (see above).
- Next: V2 preview-serve (Abrir buttons) — needs only a go-ahead.

## Verification evidence
- V1 slice-A: 350 OK isolated + ruff clean (self-verification, native review not due).
- V1 slice-B: 359 OK + ruff clean + native 4R approved, 0 findings.

## Relevant files
- src/local_file_organizer/organization_plan.py — viewer renderer (T1).
- src/local_file_organizer/organization_preview.py — preview generation.
- src/local_file_organizer/face_review.py — thumbnail precedent (assets pattern).
- src/local_file_organizer/video_sampling.py — ephemeral frame sampling to reuse.
- src/local_file_organizer/cli.py — new preview-serve command (V2).
