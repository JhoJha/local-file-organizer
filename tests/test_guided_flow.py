"""Tests for the guided ``organize`` orchestration (no heavy pipeline work).

All underlying pipeline functions (scan, sample-frames, detect-faces,
cluster-people, face-review, organize-preview, organize-apply) are
monkeypatched to record calls instead of touching ffmpeg/OpenCV/the
filesystem, since this module only tests orchestration order, pausing,
and the confirmation gate.
"""
from pathlib import Path

import pytest

from local_file_organizer import cli
from local_file_organizer.guided_flow import run_guided_organize

PIPELINE_FUNCS = (
    "run_scan",
    "run_sample_frames",
    "run_detect_faces",
    "run_cluster_people",
    "run_face_review",
    "run_organize_preview",
    "run_organize_apply",
)


@pytest.fixture
def calls(monkeypatch):
    """Monkeypatch every pipeline function on the ``cli`` module."""
    order = []

    def _make(name):
        def _fake(**kwargs):
            order.append(name)

        return _fake

    for name in PIPELINE_FUNCS:
        monkeypatch.setattr(cli, name, _make(name))
    return order


def _run(calls, *, target=Path("/tmp/target"), input_answers=None, **kwargs):
    answers = list(input_answers or [])
    prints = []

    def fake_input(prompt):
        prints.append(("INPUT", prompt))
        return answers.pop(0) if answers else ""

    def fake_print(msg):
        prints.append(("PRINT", msg))

    result = run_guided_organize(
        target,
        input_func=fake_input,
        print_func=fake_print,
        **kwargs,
    )
    return result, prints


def test_step_ordering_with_yes_runs_full_pipeline(calls):
    result, prints = _run(calls, yes=True)
    assert calls == list(PIPELINE_FUNCS)
    assert result.applied is True
    assert result.confirmed is True
    assert result.stopped_at is None


def test_confirmation_declined_skips_apply(calls):
    result, prints = _run(calls, input_answers=["", "n"])
    # everything up to organize-preview ran, but organize-apply did not
    assert calls == list(PIPELINE_FUNCS[:-1])
    assert result.applied is False
    assert result.confirmed is False
    assert result.stopped_at == "organize-preview"


def test_confirmation_accepted_runs_apply(calls):
    result, prints = _run(calls, input_answers=["", "y"])
    assert calls == list(PIPELINE_FUNCS)
    assert result.applied is True
    assert result.confirmed is True


def test_dry_run_stops_after_preview_without_asking_to_apply(calls):
    result, prints = _run(calls, dry_run=True, input_answers=[""])
    assert calls == list(PIPELINE_FUNCS[:-1])
    assert result.applied is False
    assert result.stopped_at == "organize-preview"
    # only the review pause prompt was asked, never the apply confirmation
    input_prompts = [msg for kind, msg in prints if kind == "INPUT"]
    assert len(input_prompts) == 1


def test_no_review_skips_the_pause_prompt(calls):
    # review=False means input_func is never called for the review pause;
    # yes=True means it is not called for confirmation either.
    _run(calls, review=False, yes=True)
    assert calls == list(PIPELINE_FUNCS)


def test_review_true_pauses_before_organize_preview(calls):
    seen_before_preview = []

    def fake_input(prompt):
        seen_before_preview.append(list(calls))
        return ""

    run_guided_organize(
        Path("/tmp/target"),
        review=True,
        yes=True,
        input_func=fake_input,
        print_func=lambda msg: None,
    )
    # the review pause must happen after cluster-people/face-review but
    # before organize-preview runs
    assert seen_before_preview == [
        ["run_scan", "run_sample_frames", "run_detect_faces",
         "run_cluster_people", "run_face_review"]
    ]
