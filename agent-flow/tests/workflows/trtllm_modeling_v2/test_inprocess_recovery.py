# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A completed worker turn survives a restart: the next process adopts its answer."""

from __future__ import annotations

import json

import pytest

from agent_flow.workflows.trtllm_modeling_v2.agents.turns import InProcessTurnExecutor, TurnRequest
from agent_flow.workflows.trtllm_modeling_v2.run.execution.handoff import (
    decision_value,
    mark_decision_dispatch,
)
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


def test_an_unconsumed_answer_can_survive_more_than_one_restart(tmp_path):
    original, answer = _completed(tmp_path)
    assert original.adopt_answered("assembler", "core-tp", lambda path: True) is None
    for _ in range(2):
        resumed = _resumed(tmp_path)
        adopted = resumed.adopt_answered("assembler", "core-tp", lambda path: True)
        assert adopted.decision_file == answer
        assert resumed.adopt_answered("assembler", "core-tp", lambda path: True) is None


def test_a_finished_task_consumes_answers_before_a_real_retry(tmp_path, monkeypatch):
    _completed(tmp_path)
    resumed = _resumed(tmp_path)

    def fails(*args, **kwargs):
        raise OSError("quota")

    with monkeypatch.context() as patch:
        # Nothing to consume: the history is not even read.
        patch.setattr(resumed, "_requests", fails)
        assert resumed.consume([]) == []
    with monkeypatch.context() as patch:
        # A marker that cannot be written leaves the answer as it was, not the task failed.
        patch.setattr(
            "agent_flow.workflows.trtllm_modeling_v2.agents.turns.write_json_atomic", fails
        )
        assert resumed.consume(["core-tp"]) == []
    assert resumed.consume(["core-tp"]) == ["001-assembler"]
    assert resumed.consume(["core-tp"]) == []
    assert _resumed(tmp_path).adopt_answered("assembler", "core-tp", lambda path: True) is None


#: What the latest request of a turn carries that keeps it from being adopted.
_UNUSABLE_REQUESTS = {
    "another-work": {"work": "another-task"},
    "relative-decision": {"decision_file": "relative.yaml"},
    "untyped-decision": {"decision_file": None},
}


@pytest.mark.parametrize(
    "problem",
    [
        "raised",
        "incomplete",
        "missing-result",
        "invalid-decision",
        "superseded",
        *_UNUSABLE_REQUESTS,
    ],
)
def test_an_unusable_latest_turn_is_not_adopted(tmp_path, problem):
    _, answer = _completed(tmp_path)
    records = tmp_path / "turns"
    request = json.loads((records / "001-assembler.request.json").read_text())
    if problem == "raised":
        (records / "001-assembler.done").write_text('{"outcome":"raised","error":"lost"}')
    elif problem == "incomplete":
        (records / "001-assembler.done").unlink()
    elif problem == "missing-result":
        (records / "001-assembler.result.md").unlink()
    elif problem == "invalid-decision":
        answer.write_text("decision: UNKNOWN\n")
    elif problem == "superseded":
        # A newer failed request supersedes the older answer.
        (records / "002-assembler.request.json").write_text(
            json.dumps({**request, "id": "002-assembler"})
        )
        (records / "002-assembler.done").write_text('{"outcome":"raised"}')
    else:
        (records / "001-assembler.request.json").write_text(
            json.dumps({**request, **_UNUSABLE_REQUESTS[problem]})
        )
    assert (
        _resumed(tmp_path).adopt_answered(
            "assembler", "core-tp", lambda path: decision_value(path, "assembler") is not None
        )
        is None
    )


def test_a_new_task_attempt_cannot_adopt_an_old_answer(tmp_path):
    workflow, answer = _relaunched(tmp_path, attempt=1)
    with pytest.raises(AssertionError, match="must not start an agent"):
        ExecutionStage(workflow)._dispatch_task("assembler", "core", "core-tp", "genuine retry")
    assert answer.read_text().startswith("decision: UNIT_READY")
