# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trtllm-modeling-v2 checkpoint state."""

from __future__ import annotations

import json

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run import state as state_mod
from agent_flow.workflows.trtllm_modeling_v2.run.state import (
    SCHEMA_VERSION,
    STAGE_PLANNED,
    STAGE_PREFLIGHT,
    WorkflowState,
    load_state,
    save_state,
)


def test_round_trip_preserves_every_field(tmp_path):
    path = tmp_path / "state.json"
    original = WorkflowState(
        task_path="/x/task.yaml",
        stage=STAGE_PLANNED,
        done=False,
        slug="qwen3-30b-a3b-sm_103-tp1",
        branch="trtllm-modeling-v2/qwen3-30b-a3b-sm_103-tp1",
        devices=[2, 3],
        human_stops=2,
        tune={"phase": "campaign"},
    )
    save_state(path, original)
    assert load_state(path) == original


def test_default_state_starts_at_preflight():
    assert WorkflowState(task_path="t").stage == STAGE_PREFLIGHT


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "state.json"
    save_state(path, WorkflowState(task_path="t"))
    save_state(path, WorkflowState(task_path="t", stage=STAGE_PLANNED))
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_unknown_version_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 1, "task_path": "t", "stage": "plan"}))
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        load_state(path)


def test_unknown_stage_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": SCHEMA_VERSION, "task_path": "t", "stage": "smith"}))
    with pytest.raises(ValueError, match="Unsupported stage"):
        load_state(path)


def test_missing_optional_fields_default(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps({"version": SCHEMA_VERSION, "task_path": "t", "stage": STAGE_PREFLIGHT})
    )
    loaded = load_state(path)
    assert loaded.verdict == "" and loaded.human_stops == 0 and loaded.tune == {}


@pytest.mark.parametrize(
    "verdict",
    [
        "delivery-failed",
        "delivery-failed:contract",
        "witness-unconfigured:host_offload",
        "session-lost:planned",
        "escalated",
        "planner-stalled",
    ],
)
def test_verdicts_naming_a_repair_outside_the_run_are_remediable(verdict):
    assert state_mod.is_remediable(verdict)


@pytest.mark.parametrize("verdict", ["ceiling", "planner-refused", "delivered", ""])
def test_agent_judgments_and_success_are_not_remediable(verdict):
    assert not state_mod.is_remediable(verdict)


def test_a_prefix_is_not_enough_to_count_as_remediable():
    assert not state_mod.is_remediable("delivery-failed-somehow-else")
