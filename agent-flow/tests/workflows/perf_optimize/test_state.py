"""Tests for the perf-optimize checkpoint state."""

from __future__ import annotations

import json

import pytest

from agent_flow.workflows.perf_optimize import state as state_module


def test_save_load_round_trip(tmp_path):
    path = tmp_path / state_module.STATE_FILENAME
    original = state_module.WorkflowState(
        task_path=str(tmp_path / "task.yaml"),
        max_rounds=5,
        max_attempts_per_item=2,
        max_items_per_round=4,
        item_execution="serial",
        round_index=1,
        item_index=2,
        attempt_index=1,
        current_item_id="opt-002",
        profile_required=False,
        last_profiled_analysis_dir="/ws/rounds/round_1/analysis",
        approach_violation="the attempt changed tuning/extra_llm_api_options.yaml",
        last_nsys_dir="/ws/rounds/round_1/item_1_opt-001/attempt_1/profile",
        reuse_analysis_dir="/ws/previous-perf-analyze",
        reuse_pending=True,
        campaign_git_branch="perf-optimize/ws-20260701-120000",
        campaign_git_base_commit="abc123def456",
        item_batch=[
            {
                "current_item_id": "opt-002",
                "item_index": 2,
                "attempt_index": 1,
                "phase": "evaluator",
            }
        ],
        benchmarker_done=True,
        reporter_done=False,
        done=False,
        stage=state_module.STAGE_OPTIMIZER_EVALUATOR,
    )
    state_module.save_state(path, original)
    loaded = state_module.load_state(path)
    assert loaded == original


def test_defaults():
    s = state_module.WorkflowState(task_path="t")
    assert s.stage == state_module.STAGE_BENCHMARKER
    assert s.max_rounds == 3
    assert s.max_attempts_per_item == 3
    # 1 (not the task-schema default of 3) keeps a checkpoint with no
    # recorded item budget on the narrowest round size.
    assert s.max_items_per_round == 1
    assert s.item_execution == "parallel"
    assert s.round_index == 0
    assert s.item_index == 0
    assert s.attempt_index == 0
    # A fresh campaign has not established a current runtime profile.
    assert s.profile_required is True
    assert s.last_profiled_analysis_dir == ""
    assert s.current_item_id == ""
    assert s.approach_violation == ""
    assert s.last_nsys_dir == ""
    assert s.reuse_analysis_dir == ""
    assert s.reuse_pending is False
    assert s.campaign_git_branch == ""
    assert s.campaign_git_base_commit == ""
    assert s.item_batch == []
    assert s.item_worktree_path == ""
    assert s.benchmarker_done is False
    assert s.projector_done is False
    assert s.reporter_done is False
    assert s.done is False


def test_saved_payload_carries_version(tmp_path):
    path = tmp_path / state_module.STATE_FILENAME
    state_module.save_state(path, state_module.WorkflowState(task_path="t"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == state_module.SCHEMA_VERSION
    assert payload["stage"] == state_module.STAGE_BENCHMARKER


def test_load_defaults_item_fields_missing_from_old_checkpoints(tmp_path):
    """Checkpoints written before the item loop existed must resume as-was."""
    path = tmp_path / state_module.STATE_FILENAME
    path.write_text(
        json.dumps(
            {
                "version": state_module.SCHEMA_VERSION,
                "task_path": "t",
                "stage": state_module.STAGE_QA,
                "round_index": 1,
            }
        ),
        encoding="utf-8",
    )
    loaded = state_module.load_state(path)
    assert loaded.item_index == 0
    # No budget on disk — use the narrowest round, since nothing records
    # how wide the campaign was started.
    assert loaded.max_items_per_round == 1
    # Likewise for the fields added with the analysis reuse.
    assert loaded.reuse_analysis_dir == ""
    assert loaded.reuse_pending is False
    # A checkpoint without profile-currency evidence buys one conservative
    # profile rather than asserting that an unknown runtime is unchanged.
    assert loaded.profile_required is True
    assert loaded.last_profiled_analysis_dir == ""
    assert loaded.item_execution == "parallel"


def test_load_preserves_an_explicit_current_profile_marker(tmp_path):
    """New checkpoints can prove a zero-accept runtime is safe to re-plan."""
    path = tmp_path / state_module.STATE_FILENAME
    path.write_text(
        json.dumps(
            {
                "version": state_module.SCHEMA_VERSION,
                "task_path": "t",
                "stage": state_module.STAGE_ANALYZER,
                "profile_required": False,
                "last_profiled_analysis_dir": "/ws/rounds/round_1/analysis",
            }
        ),
        encoding="utf-8",
    )
    loaded = state_module.load_state(path)

    assert loaded.profile_required is False


def test_load_rejects_unknown_item_execution(tmp_path):
    path = tmp_path / state_module.STATE_FILENAME
    path.write_text(
        json.dumps(
            {
                "version": state_module.SCHEMA_VERSION,
                "task_path": "t",
                "stage": state_module.STAGE_QA,
                "item_execution": "threads",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="item_execution"):
        state_module.load_state(path)


@pytest.mark.parametrize("version", [1, 2, 999])
def test_load_rejects_unknown_version(tmp_path, version):
    """Version-1 checkpoints (per-round QA gate semantics) must not resume."""
    path = tmp_path / state_module.STATE_FILENAME
    path.write_text(
        json.dumps({"version": version, "task_path": "t", "stage": state_module.STAGE_BENCHMARKER}),
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


@pytest.mark.parametrize(
    "stage", [state_module.STAGE_OPTIMIZER_EVALUATOR, state_module.STAGE_INTEGRATOR]
)
def test_load_rejects_batch_stage_without_item_batch(tmp_path, stage):
    """Parallel stages must carry their durable batch."""
    path = tmp_path / state_module.STATE_FILENAME
    path.write_text(
        json.dumps(
            {
                "version": state_module.SCHEMA_VERSION,
                "task_path": "t",
                "stage": stage,
                "item_batch": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="item_batch"):
        state_module.load_state(path)


def test_load_accepts_parallel_stage_with_batch(tmp_path):
    path = tmp_path / state_module.STATE_FILENAME
    state_module.save_state(
        path,
        state_module.WorkflowState(
            task_path="t",
            item_batch=[{"current_item_id": "opt-001", "phase": "optimizer"}],
            stage=state_module.STAGE_OPTIMIZER_EVALUATOR,
        ),
    )
    loaded = state_module.load_state(path)
    assert loaded.item_batch[0]["current_item_id"] == "opt-001"


def test_valid_stages_expose_one_parallel_pair_stage_and_integrator():
    assert state_module._VALID_STAGES == (
        state_module.STAGE_BENCHMARKER,
        state_module.STAGE_PROJECTOR,
        state_module.STAGE_ANALYZER,
        state_module.STAGE_OPTIMIZER_EVALUATOR,
        state_module.STAGE_INTEGRATOR,
        state_module.STAGE_QA,
        state_module.STAGE_REPORTER,
    )


def test_round_stages_are_the_loop_roles():
    # qa is deliberately absent: it runs once after the round loop as the
    # campaign's final verification, not as a per-round gate. The
    # projector is likewise absent — it runs once, before round 1.
    assert state_module.ROUND_STAGES == (
        state_module.STAGE_ANALYZER,
        state_module.STAGE_OPTIMIZER_EVALUATOR,
        state_module.STAGE_INTEGRATOR,
    )
    assert state_module.STAGE_PROJECTOR not in state_module.ROUND_STAGES


def test_projector_done_round_trip_and_old_checkpoint_default(tmp_path):
    """``projector_done`` round-trips; pre-projector checkpoints load False."""
    path = tmp_path / state_module.STATE_FILENAME
    state_module.save_state(
        path,
        state_module.WorkflowState(
            task_path="t",
            projector_done=True,
            stage=state_module.STAGE_PROJECTOR,
        ),
    )
    assert state_module.load_state(path).projector_done is True

    # A v2 checkpoint written before the projector existed carries no
    # ``projector_done`` key — it must load (same SCHEMA_VERSION) with
    # the field defaulted off.
    path.write_text(
        json.dumps(
            {
                "version": state_module.SCHEMA_VERSION,
                "task_path": "t",
                "stage": state_module.STAGE_QA,
            }
        ),
        encoding="utf-8",
    )
    loaded = state_module.load_state(path)
    assert loaded.projector_done is False
