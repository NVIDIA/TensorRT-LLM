# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The loop keeps moving while the planner thinks.

Every fake that waits for another thread waits on an event with a deadline,
so a loop that serializes what should overlap fails the test instead of
hanging it.
"""

from __future__ import annotations

import _thread
import threading
import time
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run import inbox
from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator as orchestrator_mod
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c

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


def test_a_failure_blocking_pending_work_wakes_the_planner_inside_the_window(tmp_path):
    turns, seen = [], {}
    woken = threading.Event()

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(
                tasks=[_task("a"), _task("dependent", deps=["a"]), _task("slow", satisfies=_KEYS)]
            )
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        woken.set()
        return PlannerAnswer("DONE", "the rest holds")

    def work(task):
        if task.id == "a":
            return "BLOCKED"
        seen["planner woke while slow ran"] = woken.wait(_WAIT)
        return "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, plan, work, event_window_s=60)
    assert orchestrator.run() == "planner-stalled"
    assert seen["planner woke while slow ran"]
    assert "task a (feature) ended BLOCKED" in turns[1].events


@pytest.mark.parametrize("decision", ["UNIT_READY\nordinary report", "BLOCKED"])
def test_nonblocking_task_events_are_batched(tmp_path, decision):
    orchestrator = _orchestrator(tmp_path, lambda turn: None, lambda task: None, event_window_s=60)
    task = _task("a")
    orchestrator.graph = g.Graph(tasks=[task, _task("independent")])
    orchestrator._finish(task, decision, [])
    assert orchestrator.events
    assert not orchestrator._events_due()
    orchestrator._first_event_at = time.monotonic() - 61
    assert orchestrator._events_due()


def test_default_slots_reserve_one_planner_and_allow_eight_workers():
    from agent_flow.workflows.trtllm_modeling_v2.agents.slots import AgentGate, Priority

    gate = AgentGate(orchestrator_mod.DEFAULT_MAX_PARALLEL_TASKS)
    assert gate.task_slots == 8
    gate._active[Priority.REQUIRED] = 8
    assert gate._admissible(Priority.PLANNER)
    assert not gate._admissible(Priority.REQUIRED)


def test_ctrl_c_ends_the_run_at_once_without_waiting_for_the_turns_in_flight(tmp_path):
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

    def interrupt():
        if started.wait(_WAIT):
            _thread.interrupt_main()

    orchestrator = _orchestrator(tmp_path, plan, work)
    threading.Thread(target=interrupt).start()
    begun = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        orchestrator.run()
    assert time.monotonic() - begun < _WAIT / 2
    release.set()
    assert len(turns) == 1


def test_a_burst_of_operator_messages_reaches_the_planner_as_one(tmp_path):
    first = inbox.post(tmp_path, "focus on dep4")
    inbox.post(tmp_path, "and drop the vision work")
    stamp = first.stat().st_mtime
    assert inbox.take(tmp_path, now=stamp + inbox.SETTLE_S / 2) == []
    taken = inbox.take(tmp_path, now=stamp + inbox.SETTLE_S + 1)
    assert [text.rsplit(": ", 1)[-1] for text in taken] == [
        "focus on dep4",
        "and drop the vision work",
    ]


def test_an_operator_message_reaches_the_planner_while_a_long_task_runs(tmp_path, monkeypatch):
    started, waiting, note, read = (threading.Event() for _ in range(4))
    turns, seen = [], {}
    original_wait = orchestrator_mod.wait

    def wait(*args, **kwargs):
        if started.is_set():
            waiting.set()
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(orchestrator_mod, "wait", wait)

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            return PlannerAnswer(
                "GRAPH_UPDATED", "plan", g.Graph(tasks=[_task("slow", satisfies=_KEYS)])
            )
        if "late operator evidence" in turn.events:
            read.set()
        return PlannerAnswer("DONE", "all green")

    def work(task):
        started.set()
        seen["read before the task ended"] = read.wait(_WAIT)
        return "UNIT_READY"

    def stage_notes():
        if note.is_set():
            note.clear()
            return ["late operator evidence"]
        return []

    def operator():
        assert waiting.wait(_WAIT)
        note.set()

    orchestrator = _orchestrator(tmp_path, plan, work)
    orchestrator.s.stage_notes = stage_notes
    writer = threading.Thread(target=operator)
    writer.start()
    try:
        assert orchestrator.run() == "delivered"
    finally:
        writer.join(_WAIT)
    assert seen["read before the task ended"]


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


def test_a_stop_before_the_loop_starts_nothing(tmp_path):
    def plan(turn):
        raise AssertionError("no planner turn starts after a stop")

    with orchestrator_mod.stoppable(tmp_path) as stop:
        stop("the host failed")
        orchestrator = _orchestrator(tmp_path, plan, lambda task: "UNIT_READY")
        assert orchestrator.run() == "stopped: the host failed"
    assert not orchestrator_mod._STOPS
