# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The loop keeps moving while the planner thinks.

Every fake that waits for another thread waits on an event with a deadline,
so a loop that serializes what should overlap fails the test instead of
hanging it.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator as orchestrator_mod
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)

#: How long a fake waits for another thread before the test counts it as never.
_WAIT = 10.0
_CONTRACT = c.Contract(
    checks=(
        c.Check("core", "boot", "tp1", c.TIER_INNER),
        c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER),
    )
)
_KEYS = [check.key for check in _CONTRACT.checks]


def _task(tid, satisfies=(), deps=()):
    return g.Task(
        id=tid,
        kind="feature",
        goal=f"do {tid}",
        depends_on=list(deps),
        touches=[f"models/{tid}.py"],
        gpus=1,
        est_minutes=10,
        satisfies=list(satisfies),
    )


def _passed(check):
    return SimpleNamespace(name=check.gate, member=check.member, passed=True, detail="")


def _orchestrator(tmp_path, plan, work, **kw):
    services = Services(plan=plan, work=work, check=_passed, free_gpus=lambda: 8)
    status = c.ContractStatus.load(tmp_path)
    return GraphOrchestrator(services, _CONTRACT, status, tmp_path, poll_s=0.01, **kw)


def test_a_ready_task_starts_while_a_slow_planner_turn_is_in_flight(tmp_path):
    turns = []
    in_turn_two, in_turn_three, b_started = threading.Event(), threading.Event(), threading.Event()
    seen = {}

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            # a runs a check of its own, so its success needs a turn though b depends on it.
            graph = g.Graph(
                tasks=[
                    _task("a", satisfies=_KEYS[:1]),
                    _task("b", satisfies=_KEYS[1:], deps=["a"]),
                    _task("c"),
                ]
            )
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        if len(turns) == 2:
            in_turn_two.set()
            seen["b started during the turn"] = b_started.wait(_WAIT)
        if len(turns) == 3:
            in_turn_three.set()
        return PlannerAnswer("DONE", "all green")

    def work(task):
        if task.id == "c":
            return "UNIT_READY\nthe checkpoint has a second attention variant"
        if task.id == "a":
            in_turn_two.wait(_WAIT)  # a ends while the planner reads c's report
        if task.id == "b":
            b_started.set()
            in_turn_three.wait(_WAIT)
        return "UNIT_READY"

    assert _orchestrator(tmp_path, plan, work).run() == "delivered"
    assert seen["b started during the turn"]
    # What arrived during the second turn went to the third, not to the void.
    assert any("task c (feature) reports:" in e for e in turns[1].events)
    assert not any(e.startswith("task a ") for e in turns[1].events)
    assert any(e.startswith("task a (feature) UNIT_READY") for e in turns[2].events)


def test_a_failed_task_wakes_the_planner_inside_the_window(tmp_path):
    turns, seen = [], {}
    woken = threading.Event()

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(tasks=[_task("a"), _task("slow", satisfies=_KEYS)])
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        woken.set()
        return PlannerAnswer("DONE", "the rest holds")

    def work(task):
        if task.id == "a":
            return "BLOCKED"
        seen["planner woke while slow ran"] = woken.wait(_WAIT)
        return "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, plan, work, event_window_s=60)
    assert orchestrator.run() == "delivered"
    assert seen["planner woke while slow ran"]
    assert "task a (feature) ended BLOCKED" in turns[1].events


def test_a_stop_ends_the_run_at_once_without_waiting_for_the_turns_in_flight(tmp_path):
    release, started, turns = threading.Event(), threading.Event(), []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            return PlannerAnswer(
                "GRAPH_UPDATED", "plan", g.Graph(tasks=[_task("a", satisfies=_KEYS)])
            )
        return PlannerAnswer("DONE", "all green")

    def work(task):
        started.set()
        release.wait(_WAIT)
        return "UNIT_READY"

    with orchestrator_mod.stoppable(tmp_path) as stop:
        orchestrator = _orchestrator(tmp_path, plan, work)
        threading.Thread(target=lambda: started.wait(_WAIT) and stop("the host failed")).start()
        begun = time.monotonic()
        assert orchestrator.run() == "stopped: the host failed"
        assert time.monotonic() - begun < _WAIT / 2
    release.set()
    assert len(turns) == 1
    assert json.loads((tmp_path / "inflight.json").read_text())["stopped"]["how"] == (
        "closed:stopped: the host failed"
    )
