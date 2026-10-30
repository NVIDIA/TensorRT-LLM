# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks are batched: one check takes every change to shared code landed."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c

_WAIT = 10.0
_PKG = "tensorrt_llm/_torch/_experimental/modeling_v2"
_TARGET = f"{_PKG}/models/fam/newmodel__sm_100__unified"
_OWN = g.own_paths(_TARGET)
_BOOT = "core:boot@tp1"
_CONTRACT = c.Contract(checks=(c.Check("core", "boot", "tp1", c.TIER_INNER),))


def _shared(tid, status="pending", **kw):
    return g.Task(
        id=tid,
        kind="kernel",
        goal=f"do {tid}",
        touches=[f"{_PKG}/catalog/{tid}.py"],
        status=status,
        **kw,
    )


def _core(**kw):
    return g.Task(
        id="core",
        kind="assemble",
        goal="core",
        touches=[f"{_TARGET}/modeling.py"],
        satisfies=[_BOOT],
        **kw,
    )


def _review(tid, deps=(), status="pending", **kw):
    return g.Task(
        id=tid,
        kind=g.CHECK_KIND,
        goal="regression check",
        depends_on=list(deps),
        regression=True,
        status=status,
        **kw,
    )


# ------------------------------------------------------------------ the rule


@pytest.mark.parametrize("status,left", [("done", ["k2"]), ("failed", ["k1", "k2"])])
def test_a_review_covers_its_batch_and_nothing_landed_after_it_unless_it_failed(status, left):
    graph = g.Graph(
        tasks=[
            _shared("k1", "done"),
            _shared("k2", "done"),
            _review("r0", status=status, covers=["k1@1"]),
        ]
    )
    assert g.regression_batch(graph, _OWN) == left
    assert list(g.unreviewed_shared_changes(graph, _OWN)) == left


# ------------------------------------------------------------------- the loop


def _orchestrator(tmp_path, plan, work, regression=None, **kw):
    services = Services(
        plan=plan,
        work=work,
        check=lambda chk: SimpleNamespace(name=chk.gate, member=chk.member, passed=True, detail=""),
        free_gpus=lambda: 8,
        regression=regression or (lambda task: []),
    )
    kw.setdefault("poll_s", 0.01)
    status = c.ContractStatus.load(tmp_path)
    return GraphOrchestrator(services, _CONTRACT, status, tmp_path, own_paths=_OWN, **kw)


def _plan(*answers):
    """A planner whose turns answer in order; DONE once they run out."""
    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) <= len(answers):
            answer = answers[len(turns) - 1]
            return answer(turn) if callable(answer) else answer
        return PlannerAnswer("DONE", "all green")

    plan.turns = turns  # type: ignore[attr-defined]
    return plan


def _graph(*tasks):
    return PlannerAnswer("GRAPH_UPDATED", "plan", g.Graph(tasks=list(tasks)))


def _wait_for(predicate):
    deadline = time.monotonic() + _WAIT
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_one_check_runs_at_a_time_and_the_next_takes_what_landed_meanwhile(tmp_path):
    orchestrator, reviewed = None, []

    def work(task):
        if task.id == "k2":
            # k2 lands while r1 runs: it is r1's to leave for the next check.
            _wait_for(lambda: orchestrator.graph.by_id()["r1"].status == "running")
        return "LANDED" if task.kind == "kernel" else "UNIT_READY"

    def regression(task):
        if task.id == "r1":
            _wait_for(lambda: orchestrator.graph.by_id()["k2"].status == "done")
        reviewed.append(task.id)
        return []

    plan = _plan(_graph(_shared("k1"), _shared("k2"), _core(), _review("r1", ["k1"]),
                        _review("r2", ["k2"]), _review("r3", ["k2"])))  # fmt: skip
    orchestrator = _orchestrator(tmp_path, plan, work, regression=regression)
    assert orchestrator.run() == "delivered"
    by_id = orchestrator.graph.by_id()
    assert reviewed == ["r1", "r2"]
    assert by_id["r1"].covers == ["k1@1"] and by_id["r2"].covers == ["k2@1"]
    assert by_id["r3"].status == "done" and by_id["r3"].covers == []
    events = [
        *(e for t in plan.turns for e in t.events),
        *(event.text for event in orchestrator.events),
    ]
    said = [e for e in events if e.startswith("task r3")]
    assert said == [
        "task r3 (regression check) closed: no change to shared code was left for it to "
        "check: r2 covered what it follows"
    ]


def test_the_run_is_not_delivered_while_a_landed_change_is_in_no_checks_batch(tmp_path):
    """A check that failed covers nothing: the planner saying DONE does not deliver the run."""
    reviewed = []

    def work(task):
        return "LANDED" if task.kind == "kernel" else "UNIT_READY"

    def regression(task):
        reviewed.append(task.id)
        return ["tests/x.py::test_y (passed at the base, fails now)"] if len(reviewed) == 1 else []

    def retry(turn):
        if not any("no regression check covers k1" in e for e in turn.events):
            return PlannerAnswer("DONE", "the check's findings are not ours")
        assert "the batch the next one takes: k1" in turn.regression
        return _graph(_shared("k1", "done"), _core(status="done"), _review("r1"))

    plan = _plan(_graph(_shared("k1"), _core(), _review("r1", ["k1"])), retry, retry, retry)
    orchestrator = _orchestrator(tmp_path, plan, work, regression=regression)
    assert orchestrator.run() == "delivered"
    assert reviewed == ["r1", "r1"]
    assert orchestrator.graph.by_id()["r1"].covers == ["k1@1"]


def test_a_regression_check_waits_while_the_gates_are_busy(tmp_path):
    orchestrator, order = None, []

    def check(chk):
        _wait_for(lambda: orchestrator.graph.by_id()["k1"].status == "done")
        time.sleep(0.2)  # long enough for a regression check to have started beside it
        assert orchestrator.graph.by_id()["r1"].status == "pending"
        order.append(chk.key if hasattr(chk, "key") else "gate")
        return SimpleNamespace(name=chk.gate, member=chk.member, passed=True, detail="")

    def regression(task):
        order.append(task.id)
        return []

    plan = _plan(_graph(_shared("k1"), _core(), _review("r1", ["k1"])))
    orchestrator = _orchestrator(tmp_path, plan, lambda task: "LANDED" if task.kind == "kernel"
                                 else "UNIT_READY", regression=regression)  # fmt: skip
    orchestrator.s.check = check
    assert orchestrator.run() == "delivered"
    assert order[-1] == "r1" and len(order) >= 2
