# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One decision table: what each role's prompt lists is what the run validates and handles."""

from __future__ import annotations


def test_the_planner_hears_why_an_assembler_blocked_and_a_macros_rows(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.execution.handoff import TaskResult
    from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage

    from .test_workflow import _workflow

    stage = ExecutionStage(_workflow(tmp_path))
    said = stage._blocked_report(
        {
            "decision": "BLOCKED",
            "blocked_reason": "vocabulary_gap",
            "missing_entry": "attention/x",
            "summary": "the call has no entry",
        }
    )
    assert said == "blocked_reason vocabulary_gap; missing_entry attention/x; the call has no entry"
    rows = stage._macro_rows(
        TaskResult(
            "MACRO",
            data={
                "constituents": [{"target": "gemm/a", "tier": 1}, {"target": "norm/b", "tier": 2}]
            },
        )
    )
    assert "`gemm/a` (tier 1); `norm/b` (tier 2)" in rows
