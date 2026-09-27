# Feature: smart-persons

## Objective
Make person identity genuinely smart on the same machine (OpenCV/CPU only, no server, no new deps): collapse the hundreds of singleton video clusters via temporal tracking, and turn remaining merge decisions into one-click suggestions with evidence.

## Problem
Pilot evidence: 604 face observations → 172 clusters (104 singletons), because every sampled video frame is clustered independently and the 0.50 threshold splits aggressively. Manual curation of 171 clusters is too much; the system must do the easy joins itself and only ask about the doubtful ones.

## Why
User request (2026-09-26): "quiero un sistema muy inteligente pero específicamente para eso, no tengo infraestructura para el resto". Full package authorized (tracking + suggestions).

## Scope
- S1 — video temporal tracking: within each video, link face detections across adjacent sampled frames (bbox IoU + embedding cosine) into tracklets; feed tracklets (representative embedding each) into the existing global clustering instead of raw observations. Embeddings stay memory-only (existing privacy rule). Deterministic.
- S2 — merge suggestions: after clustering, compute pairwise cluster similarity and list candidates in the merge band (below split threshold, above a merge floor) with evidence (observations, files, mean similarity) via new `person-suggest` CLI; user confirms with existing `person-merge`. No auto-merge (conservative principle preserved).
- Privacy/locality: no new dependencies, no network, no persisted embeddings, CPU only.

Out of scope: model swaps, GPU, server components, auto-merge without confirm, face-review UI changes.

## Constraints
- Reuse existing sampling (video_sampling) and embedding (face_embedding, SFace) paths.
- Deterministic output (same input → same tracklets/suggestions).
- Tests temp-dirs/synthetic only; no real personal files.
- Safety gate untouched (no new mutating primitives).

## Authorized scope
User authorized the full package. Work-unit commits on feature/organized-view-complete authorized. Push/PR/merge remain user decisions.

## Tasks
- [x] S1 — video tracklets into clustering.
  - Route: delegated direct.
  - Acceptance: MET. Adjacent-frame linking (IoU≥0.2 + cosine≥0.50, greedy timestamp order, no gap bridging), one mean embedding per tracklet into global clustering, memory-only, deterministic. Commit be9de47 (386 lines, within budget).
  - Checks: observed — 379 OK (367+12), ruff clean. Native 4R APPROVED with 0 findings (lineage review-42ef6a93fb7c7de3, burned). Reviewed boundary now be9de47.
- [x] S2 — person-suggest with evidence + confirm via person-merge. Committed as 589178f with user-granted size:exception (494 lines, atomic unit).
  - Acceptance: MET. person_suggest.py (centroid-cosine, merge floor 0.40, band [floor, split)), deterministic order, exact person-merge invocation, no auto-merge. 5 new tests.
  - Checks: observed — 384 OK (379+5), ruff clean. Native reliability review APPROVED with 0 findings (lineage review-33dc2e23939a559f, burned). Reviewed boundary now 589178f.

## TDD
- Resolved mode: unknown. Ordinary functional checks required; writer reports runner used.

## Delivery
- Strategy: `ask-on-risk` (tracker feature-branch-chain). Forecast S1 ~300, S2 ~250; each slice assessed separately vs ~400; chain question before any commit that would exceed it.

## RDD
- Mode: on. Candidate per task = work-unit commit. Assess with --base-ref <last reviewed boundary> (currently 5aea5c3), follow review_due/consent flow.

## Progress
- 2026-09-26: feature doc created, full package authorized. Next: S1.
- 2026-09-26: S1 closed + reviewed (be9de47). S2 closed (589178f) + reviewed APPROVED — feature DONE.

## Verification evidence
- S2: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests` → 384 tests OK. `ruff check src tests` → clean. Native reliability APPROVED 0 findings.

## Relevant files
- src/local_file_organizer/face_detection.py — YuNet detection output (bboxes).
- src/local_file_organizer/face_embedding.py — SFace embeddings (memory-only).
- src/local_file_organizer/face_clustering.py — global clustering to feed.
- src/local_file_organizer/video_sampling.py — frame sampling to reuse.
- src/local_file_organizer/person_identity.py — canonical persons + merge.
- src/local_file_organizer/cli.py — new person-suggest command (S2).
