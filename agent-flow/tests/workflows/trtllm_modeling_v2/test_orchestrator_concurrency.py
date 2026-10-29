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


def _slow_run(tmp_path):
    """One planner turn, then one task that runs until released."""
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

    return _orchestrator(tmp_path, plan, work), started, release, turns


def test_ctrl_c_ends_the_run_at_once_without_waiting_for_the_turns_in_flight(tmp_path):
    orchestrator, started, release, turns = _slow_run(tmp_path)
    threading.Thread(target=lambda: started.wait(_WAIT) and _thread.interrupt_main()).start()
    begun = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        orchestrator.run()
    assert time.monotonic() - begun < _WAIT / 2
    release.set()
    assert len(turns) == 1


def test_a_features_own_work_waits_for_the_core_to_boot(tmp_path):
    started, seen = [], {}

    def task(tid, kind, ref, satisfies=()):
        return g.Task(
            id=tid,
            kind=kind,
            ref=ref,
            goal=f"do {tid}",
            touches=[f"models/{tid}.py"],
            gpus=1,
            est_minutes=10,
            satisfies=list(satisfies),
        )

    def plan(turn):
        if not started:
            graph = g.Graph(
                tasks=[
                    task("core", "assemble", "core", satisfies=_KEYS),
                    task("mtp", "feature", "mtp"),
                    task("k1-draft-head", "kernel", "gemm/draft_head"),
                ]
            )
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        return PlannerAnswer("DONE", "all green")

    def work(t):
        started.append(t.id)
        if t.id == "core":
            time.sleep(0.3)  # long enough for the loop to start whatever it would
            seen["beside the core"] = set(started) - {"core"}
        return "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, plan, work, after_core=(["mtp"], [_KEYS[0]]))
    assert orchestrator.run() == "delivered"
    # The kernel the feature needs starts beside the core; the feature waits for its boot.
    assert seen["beside the core"] == {"k1-draft-head"}
    assert started.index("mtp") > started.index("core")


def test_a_burst_of_short_operator_steers_reaches_the_planner_as_one(tmp_path):
    with pytest.raises(ValueError, match="more than 1200"):
        inbox.post(tmp_path, "x" * (inbox.MESSAGE_MAX_CHARS + 1))
    inbox.post(tmp_path, "drop the vision work")
    last = inbox.post(tmp_path, "stop the dep4 boot loop", now=True)
    assert inbox.take(tmp_path, now=last.stat().st_mtime + inbox.SETTLE_S / 2) == []
    taken = inbox.take(tmp_path, now=last.stat().st_mtime + inbox.SETTLE_S + 1)
    assert [text.rsplit(": ", 1)[-1] for text in taken] == [
        "drop the vision work",
        "stop the dep4 boot loop",
    ]
    assert [inbox.is_urgent(e) for e in taken] == [False, True]
    assert all(inbox.is_message(e) for e in taken) and ".now" not in taken[1]


@pytest.mark.parametrize("now", [False, True])
def test_an_operator_message_wakes_the_planner_only_when_sent_now(tmp_path, monkeypatch, now):
    """The loop looks for messages while a long task runs, not only once it ends."""
    started, waiting, read = (threading.Event() for _ in range(3))
    prefix = inbox.URGENT_PREFIX if now else inbox.OPERATOR_PREFIX
    message = f"{prefix} (m): drop the vision work"
    turns, seen, sent = [], {}, []
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
        if message in turn.events:
            read.set()
        return PlannerAnswer("DONE", "all green")

    def work(task):
        started.set()
        seen["read before the task ended"] = read.wait(_WAIT if now else 0.5)
        return "UNIT_READY"

    def stage_notes():
        # Only once the loop waits on the running task: the message ends no future.
        if waiting.is_set() and not sent:
            sent.append(message)
            return [message]
        return []

    # The run's event window: a message that is not urgent waits for it while work runs.
    orchestrator = _orchestrator(tmp_path, plan, work, event_window_s=900.0)
    orchestrator.s.stage_notes = stage_notes
    assert orchestrator.run() == "delivered"
    # Sent `--now`, it wakes a turn while the task runs; otherwise the next turn reads it.
    assert seen["read before the task ended"] is now
    assert any(message in turn.events for turn in turns[1:])


def test_a_stop_ends_the_run_at_once_without_waiting_for_the_turns_in_flight(tmp_path):
    with orchestrator_mod.stoppable(tmp_path) as stop:
        orchestrator, started, release, turns = _slow_run(tmp_path)
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
