# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A completed worker turn survives a restart: the next process adopts its answer."""

from __future__ import annotations

from agent_flow.workflows.trtllm_modeling_v2.agents.turns import InProcessTurnExecutor, TurnRequest
from agent_flow.workflows.trtllm_modeling_v2.run.execution.handoff import mark_decision_dispatch
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage
from agent_flow.workflows.trtllm_modeling_v2.run.graph import Graph, Task

from .test_workflow import _workflow


def _completed(workspace, *, task="core-tp", attempt=2):
    decision = mark_decision_dispatch(
        workspace,
        "assembler",
        5,
        unit="core",
        identity={"task_id": task, "task_attempt": attempt},
    )

    class Answer:
        async def aforward(self, prompt):
            decision.write_text("decision: UNIT_READY\nsummary: measured on the worker tree\n")
            return "UNIT_READY"

    records = workspace / "turns"
    original = InProcessTurnExecutor(lambda request: Answer(), records=records)
    original.run(
        TurnRequest(
            role="assembler", what=task, work=task, prompt="assemble", decision_file=str(decision)
        )
    )
    return original, decision


def _resumed(workspace):
    def unexpected(request):
        raise AssertionError("a completed worker turn must not start an agent")

    return InProcessTurnExecutor(unexpected, records=workspace / "turns")


def _relaunched(tmp_path, *, attempt):
    """A workflow relaunched on a running task's second attempt, after a turn of ``attempt``."""
    workspace = tmp_path / "ws"
    _, answer = _completed(workspace, attempt=attempt)
    Graph(tasks=[Task(id="core-tp", kind="feature", goal="g", status="running", attempts=2)]).save(
        workspace / "graph.yaml"
    )
    workflow = _workflow(tmp_path)
    workflow._turns = _resumed(workspace)
    return workflow, answer


def test_a_relaunch_adopts_without_another_agent_call_or_decision_copy(tmp_path):
    workflow, answer = _relaunched(tmp_path, attempt=2)
    workspace = workflow.workspace
    result = ExecutionStage(workflow)._dispatch_task("assembler", "core", "core-tp", "assemble")
    assert result.decision == "UNIT_READY"
    assert result.result_file.is_file()
    assert len(list((workspace / "turns").glob("*.request.json"))) == 1
    # Read where the earlier dispatch reserved it: no path is reserved again.
    assert [p.name for p in (workspace / "decisions").glob("*.yaml")] == [answer.name]
    assert result.data["decision"] == "UNIT_READY"
    assert "decision adopted on resume" in workflow.journal_path.read_text()
