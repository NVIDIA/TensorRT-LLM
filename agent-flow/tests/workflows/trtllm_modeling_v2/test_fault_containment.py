# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fault containment: whatever raises becomes an event, the loop goes on, and no result is lost."""

from __future__ import annotations

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run import graph as g

from .test_orchestrator import _KEYS, PlannerAnswer, _orchestrator, _services, _task


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
