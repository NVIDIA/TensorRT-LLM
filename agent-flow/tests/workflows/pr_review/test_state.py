from __future__ import annotations

import json

import pytest

from agent_flow.workflows.pr_review import state as state_module


def test_stage_roundtrips_through_save_and_load(tmp_path):
    path = tmp_path / state_module.STATE_FILENAME
    for stage in (
        state_module.STAGE_S1_REVIEWER,
        state_module.STAGE_S1_CODER,
        state_module.STAGE_S2_REVIEWER,
        state_module.STAGE_S2_CODER,
    ):
        st = state_module.WorkflowState(
            pr_context_path=str(tmp_path / "pr_context.md"),
            num_rounds=7,
            next_round_index=3,
            stage=stage,
            done=False,
        )
        state_module.save_state(path, st)
        loaded = state_module.load_state(path)
        assert loaded.stage == stage
        assert loaded.num_rounds == 7
        assert loaded.next_round_index == 3
        assert loaded.done is False


def test_save_state_round_trips_pr_context_path(tmp_path):
    path = tmp_path / state_module.STATE_FILENAME
    state_module.save_state(
        path,
        state_module.WorkflowState(
            pr_context_path=str(tmp_path / "pr_context.md"),
            num_rounds=5,
            next_round_index=2,
            stage=state_module.STAGE_S2_CODER,
            done=True,
        ),
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == state_module.SCHEMA_VERSION
    assert data["pr_context_path"] == str(tmp_path / "pr_context.md")
    loaded = state_module.load_state(path)
    assert loaded.done is True
    assert loaded.stage == state_module.STAGE_S2_CODER


def test_load_state_rejects_unknown_version(tmp_path):
    path = tmp_path / state_module.STATE_FILENAME
    path.write_text(
        '{"version": 999, "pr_context_path": "p", "num_rounds": 1, "stage": "s1_reviewer"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        state_module.load_state(path)


def test_load_state_rejects_unknown_stage(tmp_path):
    path = tmp_path / state_module.STATE_FILENAME
    path.write_text(
        f'{{"version": {state_module.SCHEMA_VERSION}, "pr_context_path": "p", '
        f'"num_rounds": 1, "stage": "bogus"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported stage"):
        state_module.load_state(path)
