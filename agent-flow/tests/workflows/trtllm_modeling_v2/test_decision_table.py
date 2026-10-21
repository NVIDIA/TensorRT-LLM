# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One decision table: what each role's prompt lists is what the run validates and handles."""

from __future__ import annotations

import re

from agent_flow.workflows.trtllm_modeling_v2.agents import decisions as D
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import DEFAULT_PROMPTS
from agent_flow.workflows.trtllm_modeling_v2.run import (
    contract,  # noqa: F401 -- imported first
    orchestrator,
    progress,
)

#: The roles whose turn is a task's: their decisions end it done or failed.
_TASK_ROLES = ("smith", "assembler", "reviewer", "plan_reviewer", "plan_drafter")


def test_each_role_validates_exactly_the_decisions_its_table_row_names():
    for role, table in D.DECISIONS.items():
        assert progress.DECISIONS_BY_AGENT[role] == tuple(d.name for d in table), role


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


def test_a_blocked_assembler_tells_the_planner_why_and_what_is_missing():
    assembler = DEFAULT_PROMPTS.assembler
    assert "fails to the planner at once with `blocked_reason`, `missing_entry`" in assembler
    assert "ends the run with the audit trail" not in assembler


def test_each_review_kind_has_its_own_dispatch(tmp_path):
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.run.stages.assemble import AssembleStage

    class Stage(AssembleStage):
        workspace = tmp_path
        task_data = {"trtllm_repo_path": "/repo"}
        _preflight = SimpleNamespace(target_rel="models/fam/target")

        def _cli(self):
            return "cli"

        def _report_instruction(self, role):
            return ""

    prompt = Stage()._regression_review_prompt(["k3-x@1"], ("models/fam/target",))
    assert "This is a **regression review**" in prompt and "k3-x@1" in prompt
    assert "cli tests affected --workspace" in prompt and "cli tests compare" in prompt
    assert "boot one such model at tp1" in prompt
    assert "Run nothing else of the suite, and not the contract gates" in prompt


def test_the_planner_hears_why_an_assembler_blocked_and_a_macros_rows(tmp_path):
    import yaml

    from agent_flow.workflows.trtllm_modeling_v2.onboard.vocabulary import QueueItem
    from agent_flow.workflows.trtllm_modeling_v2.run.stages.planned import PlannedStage

    decision = tmp_path / "core-assembler-1.yaml"
    decision.write_text(
        yaml.safe_dump(
            {
                "decision": "BLOCKED",
                "blocked_reason": "vocabulary_gap",
                "missing_entry": "attention/x",
                "summary": "the call has no entry",
            }
        )
    )

    class Stage(PlannedStage):
        def _smith_entry_for(self, item):
            return {
                "constituents": [{"target": "gemm/a", "tier": 1}, {"target": "norm/b", "tier": 2}]
            }

    stage = Stage()
    stage.__dict__["_decision_files"] = {"core": decision}
    said = stage._blocked_report("core")
    assert said == "blocked_reason vocabulary_gap; missing_entry attention/x; the call has no entry"
    rows = stage._macro_rows(QueueItem(n=1, mode="onboard", target="x/y"), "MACRO")
    assert "`gemm/a` (tier 1); `norm/b` (tier 2)" in rows
