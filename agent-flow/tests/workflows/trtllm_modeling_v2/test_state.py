# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trtllm-modeling-v2 checkpoint state."""

from __future__ import annotations

import json

import pytest

from agent_flow.workflows.trtllm_modeling_v2 import state as state_mod
from agent_flow.workflows.trtllm_modeling_v2.state import (
    ITEM_DONE,
    ITEM_PENDING,
    SCHEMA_VERSION,
    STAGE_ASSEMBLE,
    STAGE_PREFLIGHT,
    WorkflowState,
    load_state,
    save_state,
)


def test_round_trip_preserves_every_field(tmp_path):
    path = tmp_path / "state.json"
    original = WorkflowState(
        task_path="/x/task.yaml",
        stage=STAGE_ASSEMBLE,
        done=False,
        slug="qwen3-30b-a3b-sm_103-tp1",
        branch="trtllm-modeling-v2/qwen3-30b-a3b-sm_103-tp1",
        devices=[2, 3],
        plan_round=1,
        plan_approved=True,
        queue=[{"n": 1, "mode": "onboard", "target": "moe_gelu", "status": ITEM_DONE}],
        unit_index=1,
        unit_turns=3,
        units_done=["core"],
        audit_clean=False,
        human_stops=2,
    )
    save_state(path, original)
    assert load_state(path) == original


def test_default_state_starts_at_preflight():
    assert WorkflowState(task_path="t").stage == STAGE_PREFLIGHT


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "state.json"
    save_state(path, WorkflowState(task_path="t"))
    save_state(path, WorkflowState(task_path="t", stage=STAGE_ASSEMBLE))
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_unknown_version_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 99, "task_path": "t", "stage": "plan"}))
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        load_state(path)


def test_unknown_stage_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": SCHEMA_VERSION, "task_path": "t", "stage": "tuning"}))
    with pytest.raises(ValueError, match="Unsupported stage"):
        load_state(path)


def test_queue_with_unknown_status_is_rejected(tmp_path):
    """A silently-dropped queue row is a catalog entry that never gets built."""
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "version": SCHEMA_VERSION,
                "task_path": "t",
                "stage": "smith",
                "queue": [{"n": 1, "status": "probably-fine"}],
            }
        )
    )
    with pytest.raises(ValueError, match="probably-fine"):
        load_state(path)


def test_queue_must_be_a_list_of_mappings(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "version": SCHEMA_VERSION,
                "task_path": "t",
                "stage": "smith",
                "queue": ["not-a-mapping"],
            }
        )
    )
    with pytest.raises(ValueError, match="must be a mapping"):
        load_state(path)


def test_missing_optional_fields_default(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": SCHEMA_VERSION, "task_path": "t", "stage": "plan"}))
    loaded = load_state(path)
    assert loaded.queue == []
    assert loaded.human_stops == 0
    assert loaded.unit_index == 0
    assert loaded.plan_approved is False


def test_item_statuses_are_distinct():
    statuses = {
        ITEM_PENDING,
        ITEM_DONE,
        state_mod.ITEM_RUNNING,
        state_mod.ITEM_FAILED,
        state_mod.ITEM_DEFERRED,
    }
    assert len(statuses) == 5


# ------------------------------------------------------- remediable verdicts


def test_the_verdict_survives_a_round_trip():
    """A closed run that cannot say why it closed has to be started over."""
    path_state = WorkflowState(task_path="t", done=True, verdict="delivery-failed")
    assert path_state.verdict == "delivery-failed"


def test_a_state_written_before_verdicts_existed_still_loads(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps({"version": SCHEMA_VERSION, "task_path": "t", "stage": STAGE_PREFLIGHT}),
        encoding="utf-8",
    )
    assert load_state(path).verdict == ""


@pytest.mark.parametrize(
    "verdict",
    [
        "delivery-failed",
        "audit-uncovered",
        "witness-unconfigured:host_offload",
        "assembler-blocked:knowledge_gap",
        "assembler-blocked:environment",
    ],
)
def test_verdicts_naming_a_repair_outside_the_run_are_remediable(verdict):
    """These close a run whose agent work is done.

    The repair is an edit in the repo or in task.yaml -- a missing CI test
    file, an entry needing certification, a witness block never configured --
    and the run cannot observe it except by running its own check again.
    """
    assert state_mod.is_remediable(verdict)


@pytest.mark.parametrize(
    "verdict",
    [
        "ceiling",
        "plan-failed",
        "smith-blocked",
        "assembly-exhausted:mtp",
        "assembler-blocked:vocabulary_gap",
        "delivered",
        "",
    ],
)
def test_budgets_and_agent_judgments_are_not_remediable(verdict):
    """Re-running these re-dispatches paid turns with nothing changed.

    That is how a workflow burns a night discovering the same thing four
    times. They need `--clean` and a human deciding what to do differently.
    """
    assert not state_mod.is_remediable(verdict)


def test_a_prefix_is_not_enough_to_count_as_remediable():
    """`startswith` alone would match a verdict that merely begins the same."""
    assert not state_mod.is_remediable("delivery-failed-somehow-else")


def test_a_deferred_row_survives_a_round_trip(tmp_path):
    """A resume that forgot a deferral would assemble that unit without the entry."""
    path = tmp_path / "state.json"
    row = {
        "n": 2,
        "mode": "onboard",
        "target": "moe/drafter_moe",
        "unit": "dspark",
        "status": state_mod.ITEM_DEFERRED,
        "deferred_to": "dspark",
        "deferred_why": "smith decided BLOCKED",
    }
    save_state(path, WorkflowState(task_path="t", queue=[row]))
    assert load_state(path).queue == [row]


def test_re_entering_an_assembler_block_restores_the_unit_turn_budget():
    """The budget bounds the unit's own self-heal loop.

    The operator's repair happened outside it, so the resumed unit gets its
    turns back.
    """
    import inspect

    from agent_flow.workflows.trtllm_modeling_v2.workflow import TrtllmModelingV2Workflow

    src = inspect.getsource(TrtllmModelingV2Workflow._init_state)
    branch = src[src.index("re-entered a closed run") :]
    branch = branch[: branch.index("state.done = False")]
    assert 'state.verdict.startswith("assembler-blocked")' in branch
    assert "state.unit_turns = 0" in branch
