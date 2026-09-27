# Feature: face-quality

## Objective
Mejorar distinción facial sin nuevas dependencias, sin GPU, sin reentrenar: higiene de embeddings + quality gating + centroide ponderado + calibración de umbral con ground truth local.

## Problem
SFace congelado (128-d) con umbral 0.50 estricto + media simple en tracklets + sin filtro de blur causa over-split: piloto 604 obs → 172 clusters (104 singletons). Curar une, pero el modelo se vuelve a equivocar igual.

## Why
User request (2026-09-26): "implementa lo que encontraste en estos papers que mejoren nuestra clasificacion facial". Solo cero-deps, CPU-only, LOCAL ONLY.

## Scope
- T1 — higiene cero-deps: L2-normalize en compute_embedding + check aligned no-vacío, L2 en tracklet mean, blur gate (Laplacian var con cv2 ya instalado) + re-check conf/size en los 4 sitios de ingesta, best-frame por conf*size*sharpness.
- T2 — centroide ponderado + calibración: representative_embedding(weights) + _centroid(weights), nuevo `person-calibrate` read-only que barre split/merge-floor y reporta precision/recall/F0.5 contra PERSON_XXXX, sin persistir embeddings. Linkage average opcional solo como flag experimental (default complete).

Out of scope: swap a ArcFace/GhostFaceNet/SCRFD (requiere onnxruntime + revalidación + licencias), reentrenar SFace, persistir embeddings, auto-merge, cambios en organize-apply/safety-gate.

## Constraints
- LOCAL ONLY, CPU-only, sin nuevas dependencias (solo cv2/numpy ya usados).
- Privacidad: embeddings solo memoria, nunca a SQLite/archivo/stdout. Calibración recomputa en memoria.
- Determinismo: mismo input → mismos clusters/suggests. Nuevos campos con defaults + tie-breaks estables.
- No bajar defaults 0.50/0.40 en T1; T2 solo reporta, no cambia defaults sin evidencia.

## Authorized scope
User autorizó explícitamente implementar mejoras de papers. Work-unit commits en feature branch autorizados. Push/PR/merge son decisiones del usuario bajo repo policy ordinaria.

## Tasks
- [x] T1 — hygiene-quality-gates. Commit ce9c10b (10 archivos, +448/-30, slice-1 = 589178f..ce9c10b).
  - Route: delegated direct (writer trigger: 3+ archivos no-triviales).
  - Acceptance: MET. compute_embedding L2 + aligned vacío/zero-norm→None; tracklet mean L2; passes_quality_gate compartido (conf+size+blur floor 2.0) en 4 sitios; FrameFace quality + best-frame conf*size*sharpness determinista. Defaults 0.50/0.40 intactos.
  - Checks: observed — 398 OK (384+14), ruff clean (padre spot-check coincide). Native 4R APPROVED (lineage review-811f5864a1073ff1, burned) con 2 advisory non-blocking (R2-001 WARNING naming area vs side, R2-002 SUGGESTION image sites descartan size/sharpness). Reviewed boundary ahora ce9c10b.
  - Checks: unittest 384 OK base + nuevos, ruff clean.
- [x] T2 — weighted-centroid-calibrate. Commit 8416557 (13 archivos, +1355/-200, slice-2 = ce9c10b..8416557, overage honesto por helper compartido anti-drift).
  - Route: delegated direct.
  - Acceptance: MET. representative_embedding(weights=None) deriva quality + uniformes reproducen media vieja bit a bit; _centroid(weights) igual; `person-calibrate` barre grid e imprime precision/recall/F0.5/singleton-rate/count vs PERSON_XXXX, read-only memoria-only; linkage complete default + average experimental; R2-001/R2-002 de T1 cerrados.
  - Checks: observed — 430 OK (398+32), ruff clean (padre spot-check coincide) + demo sintética calibrate read-only. Native 4R APPROVED (lineage review-9fa30d77313ce1bd, burned) con 3 advisory non-blocking (R2-DUP-GRIDS, R2-LINKAGE-LITERALS WARNING + R2-PARALLEL-ARRAYS SUGGESTION, trabajo futuro separado). Reviewed boundary ahora 8416557.
  - Checks: unittest + ruff clean, demo sintética.

## TDD
- Resolved mode: unknown/disabled (no project/session TDD config; test files present does not enable TDD).
- Source: none. Runner: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests` + `ruff check src tests`. Writer reporta runner usado; checks funcionales ordinarios requeridos.

## Delivery
- Strategy: `ask-on-risk` (default).
- Chain strategy: `feature-branch-chain` (heredado).
- Forecast (authored +/- excl. generated): T1 ~180, T2 ~260; total ~440 >400, pregunta chain antes del commit que supere ~400. Heurística advisory-only, no cap duro.
- Running count: T1 478 (10 paths) vs forecast 180 — superó advisory; T2 1555 (13 paths) vs forecast 260 — overage honesto por helper compartido + calibración completa; ambos slices separados bajo feature-branch-chain ya cacheado.
- Slice boundaries: slice-1 (T1) = 589178f..ce9c10b, reviewed APPROVED; slice-2 (T2) = ce9c10b..8416557, reviewed APPROVED.

## RDD
- Mode: on (global on, clone-local unset).
- Candidate per task = work-unit commit (nunca checkbox ni rama acumulada). Tras cada commit: `gentle-ai review assess --cwd <repo> --agent opencode --base-ref <last reviewed boundary> --committed-only --json` con --untracked-scope=exclude. First boundary = 589178f (S2 reviewed). Seguir review_due/consent flow. odd/ y .atl/ excluidos.

## Progress
- 2026-09-26: feature doc creado desde mapping + research papers. Next: T1.
- 2026-09-26: T1 cerrado + revisado (ce9c10b, APPROVED). Next: T2.
- 2026-09-26: T2 cerrado + revisado (8416557, APPROVED). Feature DONE — push/PR/merge son decisiones del usuario.

## Verification evidence
- T1: 398 OK + ruff clean + 4R APPROVED (review-811f5864a1073ff1).
- T2: 430 OK + ruff clean + demo calibrate sintética + 4R APPROVED (review-9fa30d77313ce1bd).

## Relevant files
- src/local_file_organizer/face_embedding.py — compute_embedding L2 + aligned check, cosine_distance.
- src/local_file_organizer/face_detection.py — detect_faces_raw, thresholds, min-size gate.
- src/local_file_organizer/face_tracking.py — FrameFace, build_tracklets, representative_embedding/member.
- src/local_file_organizer/face_clustering.py — cluster_people, thresholds, linkage.
- src/local_file_organizer/person_suggest.py — _centroid, suggest_merges band.
- src/local_file_organizer/video_sampling.py — timestamps/frames para quality.
- src/local_file_organizer/cli.py — run_cluster_people, run_person_suggest, nuevos flags/calibrate.
- tests/test_face_*.py, test_person_suggest.py — no romper salvo update justificado.
