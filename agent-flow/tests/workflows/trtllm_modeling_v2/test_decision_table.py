# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One decision table: what each role's prompt lists is what the run validates and handles."""

from __future__ import annotations

import re

from agent_flow.workflows.trtllm_modeling_v2.agents import decisions as D
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import DEFAULT_PROMPTS
from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator

#: The roles whose turn is a task's: their decisions end it done or failed.
_TASK_ROLES = ("kernel_writer", "assembler", "surveyor")


def test_each_role_validates_exactly_the_decisions_its_table_row_names():
    for role, table in D.DECISIONS.items():
        assert D.DECISIONS_BY_AGENT[role] == tuple(d.name for d in table), role


def test_every_decision_a_task_role_writes_ends_its_task_one_way():
    for role in _TASK_ROLES:
        for decision in D.DECISIONS[role]:
            assert decision.finishes is not None, (role, decision.name)
            assert (decision.name in orchestrator.FINISHED) == decision.finishes, decision.name
    # Every decision that ends a task done is a role's, or one the run writes itself.
    assert orchestrator.FINISHED == D.finishing() | {orchestrator.TUNED, orchestrator.CHECKED}


def test_every_prompt_lists_its_decisions_and_no_other():
    for role in D.DECISIONS:
        prompt = getattr(DEFAULT_PROMPTS, role)
        assert D.decision_list(role) in prompt or " | ".join(D.names(role)) in prompt, role
        listed = set(re.findall(r"^- `([A-Z_]+)`: ", prompt, flags=re.M))
        assert listed <= set(D.names(role)), (role, listed - set(D.names(role)))


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
