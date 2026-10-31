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
                "GRAPH_UPDATED", "plan", g.Graph(tasks=[_task("x", "kernel"), core])
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


def test_a_kernel_that_ends_macro_is_split_not_failed(tmp_path):
    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = [
                _task("k", "kernel", touches=["catalog/k.py"]),
                _task("core", "assemble", _KEYS),
            ]
            return PlannerAnswer("GRAPH_UPDATED", "plan", g.Graph(tasks=graph))
        return PlannerAnswer("DONE", "green")

    orchestrator = _orchestrator(tmp_path, _services(plan, decisions={"k": "MACRO"}))
    assert orchestrator.run() == "delivered"
    assert orchestrator.graph.by_id()["k"].status == "split"
    # A split task lands nothing and holds no dependent: the run is delivered past it.
    assert g.Graph.load(tmp_path / g.GRAPH_FILENAME).by_id()["k"].status == "split"
