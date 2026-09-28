"""Guided orchestration for the ``organize`` command.

This module chains the existing read-only pipeline steps (scan,
sample-frames, detect-faces, cluster-people, face-review) followed by the
write-guarded pair organize-preview / organize-apply, by calling the same
functions the individual subcommands in :mod:`local_file_organizer.cli`
already call — no subprocesses are spawned and no new behavior is added to
any of them.

It only adds:

- numbered progress output ("[3/7] Detectando rostros...");
- a supervised pause after face-review so a human can open the generated
  HTML report and curate cluster decisions before continuing (skippable
  with ``review=False``);
- an explicit confirmation gate before organize-apply, satisfied either
  interactively or with ``yes=True``.

Every step is a direct call into the already-idempotent pipeline
functions: re-running ``organize`` on the same target resumes safely
because ``scan`` is incremental by default and the later steps upsert
their own tables/files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import cli

InputFunc = Callable[[str], str]
PrintFunc = Callable[[str], None]

STEP_LABELS = (
    "Escaneando la carpeta",
    "Muestreando frames de video",
    "Detectando rostros",
    "Agrupando personas",
    "Generando revisión de clusters",
    "Generando vista previa de organización",
    "Aplicando organización",
)

_TOTAL_STEPS = len(STEP_LABELS)

_CONFIRM_YES = {"y", "yes", "s", "si", "sí"}


@dataclass
class GuidedOrganizeResult:
    """Outcome of a guided ``organize`` run."""

    stopped_at: Optional[str]
    applied: bool
    confirmed: bool


def _default_input(prompt: str) -> str:
    return input(prompt)


def _step(index: int, print_func: PrintFunc) -> None:
    print_func(f"[{index}/{_TOTAL_STEPS}] {STEP_LABELS[index - 1]}...")


def run_guided_organize(
    target: Path,
    *,
    db_path: Path = cli.DEFAULT_DB_PATH,
    csv_path: Path = cli.DEFAULT_CSV_PATH,
    plan_path: Path = cli.DEFAULT_PLAN_PATH,
    snapshot_dir: Path = cli.DEFAULT_SNAPSHOT_DIR,
    face_review_dir: Path = cli.DEFAULT_FACE_REVIEW_DIR,
    face_review_decisions: Path = cli.DEFAULT_FACE_REVIEW_DECISIONS,
    review: bool = True,
    dry_run: bool = False,
    yes: bool = False,
    quiet: bool = False,
    input_func: InputFunc = _default_input,
    print_func: PrintFunc = print,
) -> GuidedOrganizeResult:
    """Run the full guided pipeline against ``target``.

    Mirrors the defaults and options each existing subcommand uses; the
    original subcommands themselves are left untouched by this module.

    ``review=False`` skips the interactive pause after face-review (but the
    HTML report is still generated). ``dry_run=True`` stops after printing
    the organize-preview summary. ``yes=True`` bypasses the interactive
    apply confirmation, matching ``organize-apply --yes``.
    """
    _step(1, print_func)
    cli.run_scan(
        target=target,
        db_path=db_path,
        csv_path=csv_path,
        snapshot_dir=snapshot_dir,
        quiet=quiet,
    )

    _step(2, print_func)
    cli.run_sample_frames(
        target=target,
        db_path=db_path,
        snapshot_dir=snapshot_dir,
        quiet=quiet,
    )

    _step(3, print_func)
    cli.run_detect_faces(
        target=target,
        db_path=db_path,
        snapshot_dir=snapshot_dir,
        quiet=quiet,
    )

    _step(4, print_func)
    cli.run_cluster_people(
        target=target,
        db_path=db_path,
        snapshot_dir=snapshot_dir,
        quiet=quiet,
    )

    _step(5, print_func)
    cli.run_face_review(
        target=target,
        db_path=db_path,
        output_dir=face_review_dir,
        decisions_path=face_review_decisions,
        snapshot_dir=snapshot_dir,
        quiet=quiet,
    )
    review_index = face_review_dir / "index.html"
    print_func(f"Revisión de clusters generada en: {review_index}")

    if review:
        input_func(
            "Revisa/edita los clusters en el navegador y pulsa Enter para "
            "continuar (o Ctrl+C para salir)... "
        )
    else:
        print_func("--no-review: se omite la pausa de revisión de clusters.")

    _step(6, print_func)
    cli.run_organize_preview(
        db_path=db_path,
        plan_path=plan_path,
        decisions_path=face_review_decisions,
        quiet=quiet,
    )

    if dry_run:
        print_func("--dry-run: deteniendo antes de organize-apply.")
        return GuidedOrganizeResult(
            stopped_at="organize-preview", applied=False, confirmed=False
        )

    confirmed = yes
    if not confirmed:
        answer = input_func(
            "¿Aplicar esta organización moviendo los archivos? [y/N] "
        )
        confirmed = answer.strip().lower() in _CONFIRM_YES

    if not confirmed:
        print_func("Organización cancelada: no se movió ningún archivo.")
        return GuidedOrganizeResult(
            stopped_at="organize-preview", applied=False, confirmed=False
        )

    _step(7, print_func)
    cli.run_organize_apply(
        target=target,
        db_path=db_path,
        plan_path=plan_path,
        apply=True,
        snapshot_dir=snapshot_dir,
    )
    return GuidedOrganizeResult(stopped_at=None, applied=True, confirmed=True)
