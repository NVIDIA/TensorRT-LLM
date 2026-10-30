# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fault containment: whatever raises becomes an event, the loop goes on, and no result is lost."""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run import graph as g

from .test_orchestrator import PlannerAnswer, _orchestrator, _services, _task

_KEYS = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]


@pytest.fixture(autouse=True)
def _quick_retries(monkeypatch):
    monkeypatch.setattr(
        "agent_flow.workflows.trtllm_modeling_v2.run.orchestrator.PLANNER_RETRY_S", 0.01
    )


def test_a_planner_turn_that_raised_is_asked_again_on_the_same_events(tmp_path):
    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            core = _task("core", "assemble", _KEYS, deps=["x"])
            return PlannerAnswer(
                "GRAPH_UPDATED", "plan", g.Graph(tasks=[_task("x", "investigate"), core])
            )
        if len(turns) == 2:
            raise TimeoutError("the turn ran past its bound")
        if len(turns) == 3:
            # The core goes on without what x failed to find.
            tasks = [t for t in orchestrator.graph.tasks if t.id != "core"]
            tasks.append(_task("core", "assemble", _KEYS))
            return PlannerAnswer("GRAPH_UPDATED", "the core next", g.Graph(tasks=tasks))
        return PlannerAnswer("DONE", "green")

    services = _services(plan, decisions={"x": "BLOCKED"})
    orchestrator = _orchestrator(tmp_path, services)
    assert orchestrator.run() == "delivered"
    assert any("ended BLOCKED" in e for e in turns[1].events)
    # The third turn reads what the second could not answer, and why it is asked again.
    assert any("ended BLOCKED" in e for e in turns[2].events)
    assert any("your previous turn raised TimeoutError" in e for e in turns[2].events)


def test_a_planner_that_keeps_raising_ends_the_run_with_its_own_verdict(tmp_path):
    def plan(turn):
        raise RuntimeError("no credentials")

    verdict = _orchestrator(tmp_path, _services(plan)).run()
    assert verdict == "planner-failed: RuntimeError: no credentials"


def test_an_unreadable_proposal_is_refused_and_sent_back(tmp_path):
    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            return PlannerAnswer(
                "GRAPH_UPDATED", "plan", None, problem="graph.proposed.yaml could not be read: x"
            )
        graph = g.Graph(tasks=[_task("core", "assemble", _KEYS)])
        return (
            PlannerAnswer("GRAPH_UPDATED", "plan", graph)
            if len(turns) == 2
            else (PlannerAnswer("DONE", "green"))
        )

    assert _orchestrator(tmp_path, _services(plan)).run() == "delivered"
    assert turns[1].refusals == ["graph.proposed.yaml could not be read: x"]


def test_a_refused_revision_leaves_its_events_for_the_next_turn(tmp_path):
    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(tasks=[_task("core", "assemble", _KEYS), _task("x", "investigate")])
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        if len(turns) == 2:
            bad = g.Graph(tasks=[*orchestrator.graph.tasks, _task("y", "nonsense-kind")])
            return PlannerAnswer("GRAPH_UPDATED", "bad", bad)
        return PlannerAnswer("DONE", "green")

    services = _services(plan, decisions={"x": "BLOCKED"})
    orchestrator = _orchestrator(tmp_path, services)
    assert orchestrator.run() == "delivered"
    assert any("ended BLOCKED" in e for e in turns[1].events)
    assert turns[2].refusals and any("ended BLOCKED" in e for e in turns[2].events)


def test_checks_that_raised_leave_the_work_landed_and_run_again(tmp_path):
    """A lost session is not the task's failure: it is done, and its checks run as re-checks."""
    raised = []

    def plan(turn):
        if not turn.events:
            return PlannerAnswer(
                "GRAPH_UPDATED", "plan", g.Graph(tasks=[_task("core", "assemble", _KEYS)])
            )
        return PlannerAnswer("DONE", "green")

    services = _services(plan)
    check = services.check

    def flaky(chk):
        if chk.key == "core:boot@tp1" and not raised:
            raised.append(chk.key)
            raise RuntimeError("session lost")
        return check(chk)

    services.check = flaky
    orchestrator = _orchestrator(tmp_path, services)
    assert orchestrator.run() == "delivered"
    core = orchestrator.graph.by_id()["core"]
    assert core.status == "done" and core.attempts <= 1
    assert services.ran_checks.count("core:boot@tp1") == 1  # recorded once: the re-check


def test_a_landed_change_given_up_later_is_still_reviewed():
    shared = _task("k", "kernel", touches=["catalog/shared.py"], priority="nice_to_have")
    shared.status, shared.attempts = "abandoned", 1
    graph = g.Graph(tasks=[shared])
    assert g.regression_batch(graph, ["models/x/"], {"k": 1}) == ["k"]
    assert g.regression_batch(graph, ["models/x/"], {}) == []


def test_tasks_that_ended_are_recorded_when_the_loop_itself_fails(tmp_path):
    orchestrator = _orchestrator(tmp_path, _services(lambda turn: PlannerAnswer("DONE")))
    task = _task("core", "assemble", [])
    task.status = "running"
    orchestrator.graph = g.Graph(tasks=[task])
    done: Future = Future()
    done.set_result(("UNIT_READY", []))
    orchestrator._running = {"core": done, "other": Future()}
    orchestrator._record_finished()
    assert task.status == "done"


def test_an_infrastructure_retry_is_counted_per_attempt(tmp_path):
    orchestrator = _orchestrator(tmp_path, _services(lambda turn: PlannerAnswer("DONE")))
    task = _task("core", "assemble", [])
    orchestrator.graph = g.Graph(tasks=[task])
    task.status, task.attempts = "running", 1
    orchestrator._finish(task, "raised OSError: lost", [])
    assert task.status == "pending"
    task.status = "running"
    orchestrator._finish(task, "raised OSError: lost", [])
    assert task.status == "failed"  # the attempt's one retry is spent
    task.status, task.attempts = "running", 2  # the planner retries it
    orchestrator._finish(task, "raised OSError: lost", [])
    assert task.status == "pending"  # a new attempt gets a retry of its own
    assert SimpleNamespace(**orchestrator.retries) == SimpleNamespace(**{"core@1": 1, "core@2": 1})
