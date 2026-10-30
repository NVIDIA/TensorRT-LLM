# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint a resumed run continues from, and which closed runs re-enter."""

from __future__ import annotations

import json

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run.records.state import (
    STAGE_EXECUTION,
    WorkflowState,
    load_state,
    save_state,
)


def test_round_trip_preserves_every_field_and_an_unknown_version_is_rejected(tmp_path):
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
    path.write_text(json.dumps({"version": 1, "task_path": "t", "stage": "plan"}))
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        load_state(path)
