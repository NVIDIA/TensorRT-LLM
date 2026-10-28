# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint a resumed run continues from, and which closed runs re-enter."""

from __future__ import annotations

import json

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run.records import state as state_mod
from agent_flow.workflows.trtllm_modeling_v2.run.records.state import (
    STAGE_EXECUTION,
    WorkflowState,
    load_state,
    save_state,
)


def test_round_trip_preserves_every_field(tmp_path):
    path = tmp_path / "state.json"
    original = WorkflowState(
        task_path="/x/task.yaml",
        stage=STAGE_EXECUTION,
        done=False,
        slug="qwen3-30b-a3b-sm_103-tp1",
        branch="trtllm-modeling-v2/qwen3-30b-a3b-sm_103-tp1",
        devices=[2, 3],
        tune={"phase": "campaign"},
    )
    save_state(path, original)
    assert load_state(path) == original


def test_unknown_version_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 1, "task_path": "t", "stage": "plan"}))
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        load_state(path)


@pytest.mark.parametrize("verdict", ["delivery-failed:contract", "escalated"])
def test_verdicts_naming_a_repair_outside_the_run_are_remediable(verdict):
    assert state_mod.is_remediable(verdict)


@pytest.mark.parametrize("verdict", ["ceiling", "delivered"])
def test_agent_judgments_and_success_are_not_remediable(verdict):
    assert not state_mod.is_remediable(verdict)
