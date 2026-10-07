"""Tests for the perf-analyze checkpoint state."""

from __future__ import annotations

import json

import pytest

from agent_flow.workflows.perf_analyze import state as state_module


def test_save_load_round_trip(tmp_path):
    path = tmp_path / state_module.STATE_FILENAME
    original = state_module.WorkflowState(
        task_path=str(tmp_path / "task.yaml"),
        benchmarker_done=True,
        projector_done=True,
        analyzer_done=False,
        reporter_done=False,
        done=False,
        stage=state_module.STAGE_ANALYZER,
    )
    state_module.save_state(path, original)
    loaded = state_module.load_state(path)
    assert loaded == original


def test_projector_stage_round_trip(tmp_path):
    path = tmp_path / state_module.STATE_FILENAME
    original = state_module.WorkflowState(
        task_path="t",
        benchmarker_done=True,
        stage=state_module.STAGE_PROJECTOR,
    )
    state_module.save_state(path, original)
    assert state_module.load_state(path) == original


def test_defaults(tmp_path):
    s = state_module.WorkflowState(task_path="t")
    assert s.stage == state_module.STAGE_BENCHMARKER
    assert s.benchmarker_done is False
    assert s.projector_done is False
    assert s.analyzer_done is False
    assert s.reporter_done is False
    assert s.done is False


def test_saved_payload_carries_version(tmp_path):
    path = tmp_path / state_module.STATE_FILENAME
    state_module.save_state(path, state_module.WorkflowState(task_path="t"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == state_module.SCHEMA_VERSION
    assert payload["stage"] == state_module.STAGE_BENCHMARKER


def test_load_rejects_unknown_version(tmp_path):
    path = tmp_path / state_module.STATE_FILENAME
    path.write_text(
        json.dumps({"version": 999, "task_path": "t", "stage": state_module.STAGE_BENCHMARKER}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="version"):
        state_module.load_state(path)


def test_load_rejects_unknown_stage(tmp_path):
    path = tmp_path / state_module.STATE_FILENAME
    path.write_text(
        json.dumps({"version": state_module.SCHEMA_VERSION, "task_path": "t", "stage": "nonsense"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stage"):
        state_module.load_state(path)


def test_valid_stages_are_the_four_roles():
    assert state_module._VALID_STAGES == (
        state_module.STAGE_BENCHMARKER,
        state_module.STAGE_PROJECTOR,
        state_module.STAGE_ANALYZER,
        state_module.STAGE_REPORTER,
    )


def test_load_legacy_profiler_checkpoint_maps_to_analyzer(tmp_path):
    """A v1 checkpoint from before the profiler→analyzer rename still loads.

    The rename kept SCHEMA_VERSION at 1: the legacy ``"profiler"`` stage
    string and ``profiler_done`` key map onto their analyzer equivalents
    instead of hard-failing an in-flight workspace.
    """
    path = tmp_path / state_module.STATE_FILENAME
    path.write_text(
        json.dumps(
            {
                "version": state_module.SCHEMA_VERSION,
                "task_path": "t",
                "benchmarker_done": True,
                "projector_done": True,
                "profiler_done": True,
                "reporter_done": False,
                "done": False,
                "stage": "profiler",
            }
        ),
        encoding="utf-8",
    )
    loaded = state_module.load_state(path)
    assert loaded.stage == state_module.STAGE_ANALYZER
    assert loaded.analyzer_done is True
    # The new key wins when both are present.
    path.write_text(
        json.dumps(
            {
                "version": state_module.SCHEMA_VERSION,
                "task_path": "t",
                "profiler_done": True,
                "analyzer_done": False,
                "stage": state_module.STAGE_ANALYZER,
            }
        ),
        encoding="utf-8",
    )
    assert state_module.load_state(path).analyzer_done is False


def test_load_pre_projector_checkpoint_defaults_projector_done_false(tmp_path):
    """A v1 checkpoint written before the projector existed still loads.

    Adding ``projector_done`` was purely additive (SCHEMA_VERSION stayed
    at 1), so payloads without the field must load with it False rather
    than failing an in-flight workspace.
    """
    path = tmp_path / state_module.STATE_FILENAME
    path.write_text(
        json.dumps(
            {
                "version": state_module.SCHEMA_VERSION,
                "task_path": "t",
                "benchmarker_done": True,
                "analyzer_done": True,
                "reporter_done": False,
                "done": False,
                "stage": state_module.STAGE_REPORTER,
            }
        ),
        encoding="utf-8",
    )
    loaded = state_module.load_state(path)
    assert loaded.projector_done is False
    assert loaded.benchmarker_done is True
    assert loaded.stage == state_module.STAGE_REPORTER
