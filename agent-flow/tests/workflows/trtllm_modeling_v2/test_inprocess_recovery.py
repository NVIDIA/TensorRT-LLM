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


def test_a_relaunch_adopts_without_another_agent_call_or_decision_copy(tmp_path):
    workspace = tmp_path / "ws"
    _, answer = _completed(workspace)
    Graph(tasks=[Task(id="core-tp", kind="feature", goal="g", status="running", attempts=2)]).save(
        workspace / "graph.yaml"
    )
    workflow = _workflow(tmp_path)
    workflow._turns = _resumed(workspace)

    execution = ExecutionStage(workflow)
    result = execution._dispatch_task("assembler", "core", "core-tp", "assemble")
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


def test_a_finished_task_consumes_answers_before_a_real_retry(tmp_path):
    _completed(tmp_path)
    resumed = _resumed(tmp_path)
    assert resumed.consume(["core-tp"]) == ["001-assembler"]
    assert resumed.consume(["core-tp"]) == []
    assert _resumed(tmp_path).adopt_answered("assembler", "core-tp", lambda path: True) is None


@pytest.mark.parametrize("problem", ["raised", "incomplete", "missing-result", "invalid-decision"])
def test_an_unusable_latest_turn_is_not_adopted(tmp_path, problem):
    _, answer = _completed(tmp_path)
    records = tmp_path / "turns"
    if problem == "raised":
        (records / "001-assembler.done").write_text('{"outcome":"raised","error":"lost"}')
    elif problem == "incomplete":
        (records / "001-assembler.done").unlink()
    elif problem == "missing-result":
        (records / "001-assembler.result.md").unlink()
    else:
        answer.write_text("decision: UNKNOWN\n")
    assert (
        _resumed(tmp_path).adopt_answered(
            "assembler", "core-tp", lambda path: decision_value(path, "assembler") is not None
        )
        is None
    )


def test_a_newer_failed_request_supersedes_an_older_answer(tmp_path):
    _completed(tmp_path)
    request = json.loads((tmp_path / "turns/001-assembler.request.json").read_text())
    request["id"] = "002-assembler"
    (tmp_path / "turns/002-assembler.request.json").write_text(json.dumps(request))
    (tmp_path / "turns/002-assembler.done").write_text('{"outcome":"raised"}')
    assert _resumed(tmp_path).adopt_answered("assembler", "core-tp", lambda path: True) is None


def test_work_identity_prevents_adopting_another_tasks_same_named_turn(tmp_path):
    _completed(tmp_path)
    request_path = tmp_path / "turns/001-assembler.request.json"
    request = json.loads(request_path.read_text())
    request["work"] = "another-task"
    request_path.write_text(json.dumps(request))
    assert _resumed(tmp_path).adopt_answered("assembler", "core-tp", lambda path: True) is None


@pytest.mark.parametrize("decision", ["relative.yaml", None, [], 42])
def test_a_request_without_an_absolute_decision_path_is_not_adopted(tmp_path, decision):
    _completed(tmp_path)
    path = tmp_path / "turns/001-assembler.request.json"
    request = json.loads(path.read_text())
    request["decision_file"] = decision
    path.write_text(json.dumps(request))
    assert _resumed(tmp_path).adopt_answered("assembler", "core-tp", lambda path: True) is None


def test_empty_consumption_does_not_scan_history(tmp_path, monkeypatch):
    resumed = _resumed(tmp_path)

    def unexpected(*args):
        raise AssertionError("no completed work to consume")

    monkeypatch.setattr(resumed, "_requests", unexpected)
    assert resumed.consume([]) == []


def test_a_new_task_attempt_cannot_adopt_an_old_answer(tmp_path):
    workspace = tmp_path / "ws"
    _, answer = _completed(workspace, attempt=1)
    Graph(tasks=[Task(id="core-tp", kind="feature", goal="g", status="running", attempts=2)]).save(
        workspace / "graph.yaml"
    )
    workflow = _workflow(tmp_path)
    workflow._turns = _resumed(workspace)
    with pytest.raises(AssertionError, match="must not start an agent"):
        ExecutionStage(workflow)._dispatch_task("assembler", "core", "core-tp", "genuine retry")
    assert answer.read_text().startswith("decision: UNIT_READY")


def test_consumption_io_failure_does_not_turn_completed_work_into_a_task_failure(
    tmp_path, monkeypatch
):
    _completed(tmp_path)

    def no_space(*args, **kwargs):
        raise OSError("quota")

    monkeypatch.setattr(
        "agent_flow.workflows.trtllm_modeling_v2.agents.turns.write_json_atomic", no_space
    )
    assert _resumed(tmp_path).consume(["core-tp"]) == []
