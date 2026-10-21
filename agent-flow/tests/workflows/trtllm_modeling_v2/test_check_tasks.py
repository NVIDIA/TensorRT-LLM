# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A `check` task's work is only contract checks: the orchestrator runs them, with no agent turn.

Re-certifying a member, verifying one nobody changed, running a stale check
again: an agent turn adds nothing to any of them but its wait. A check task
takes no task slot and no devices of its own; only its failure wakes the
planner, which then sends an agent.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import yaml

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run import receipts as receipts_mod
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    CHECKED,
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)
from agent_flow.workflows.trtllm_modeling_v2.run.stages.planned import TASKS_DIRNAME, PlannedStage

_WAIT = 10.0
_BOOT, _ACC, _TP2 = "core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:boot@tp2"
_ENVELOPE = "core:envelope@tp1"
_CONTRACT = c.Contract(
    checks=(
        c.Check("core", "boot", "tp1", c.TIER_INNER),
        c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER),
        c.Check("core", "boot", "tp2", c.TIER_INNER),
        c.Check("core", "envelope", "tp1", c.TIER_DELIVERY),
    )
)


def _task(tid, kind="assemble", satisfies=(), deps=(), gpus=1, touches=None):
    return g.Task(
        id=tid,
        kind=kind,
        goal=f"do {tid}",
        depends_on=list(deps),
        touches=list(touches if touches is not None else [f"models/{tid}.py"]),
        gpus=gpus,
        satisfies=list(satisfies),
    )


def _check_task(tid, satisfies, deps=()):
    return _task(tid, g.CHECK_KIND, satisfies, deps, gpus=0, touches=[])


def _run(tmp_path, tasks, *, work=None, failing=(), gpus=4, **kw):
    turns, worked, ran = [], [], []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            return PlannerAnswer("GRAPH_UPDATED", "plan", g.Graph(tasks=list(tasks)))
        failed = any("checks failed" in e for e in turn.events)
        return PlannerAnswer("ESCALATE" if failed else "DONE", "seen")

    def do(task):
        worked.append(task.id)
        return work(task) if work is not None else "UNIT_READY"

    def check(chk):
        ran.append(chk.key)
        return SimpleNamespace(
            name=chk.gate, member=chk.member, passed=chk.key not in failing, detail=""
        )

    services = Services(plan=plan, work=do, check=check, free_gpus=lambda: gpus)
    status = c.ContractStatus.load(tmp_path)
    kw.setdefault("poll_s", 0.01)
    orchestrator = GraphOrchestrator(services, _CONTRACT, status, tmp_path, **kw)
    verdict = orchestrator.run()
    return SimpleNamespace(
        verdict=verdict, turns=turns, worked=worked, ran=ran, orchestrator=orchestrator
    )


def test_a_check_task_runs_its_checks_with_no_agent_turn(tmp_path):
    run = _run(
        tmp_path,
        [
            _task("core", satisfies=[_BOOT, _ACC, _ENVELOPE]),
            _check_task("core-tp2", [_TP2], deps=["core"]),
        ],
    )
    assert run.verdict == "delivered"
    assert run.worked == ["core"]  # no worker for the check task
    assert run.ran.count(_TP2) == 1
    assert run.orchestrator.graph.by_id()["core-tp2"].status == "done"
    # A pass waits for the next turn like any finished task's (here the run is delivered
    # first, the delivery tier having been rehearsed); a failure wakes the planner.
    heard = [*(e for t in run.turns for e in t.events), *run.orchestrator.events]
    said = next(e for e in heard if e.startswith("task core-tp2"))
    assert said == "task core-tp2 (check) ran its checks with no agent turn; they passed"


def test_a_failing_check_task_fails_and_wakes_the_planner_and_no_agent(tmp_path):
    run = _run(
        tmp_path,
        [
            _task("core", satisfies=[_BOOT, _ACC, _ENVELOPE]),
            _check_task("core-tp2", [_TP2], deps=["core"]),
        ],
        failing={_TP2},
    )
    assert run.verdict == "escalated"
    assert run.worked == ["core"]
    assert run.orchestrator.graph.by_id()["core-tp2"].status == "failed"
    said = next(e for t in run.turns for e in t.events if e.startswith("task core-tp2"))
    assert "with no agent turn" in said and f"checks failed: {_TP2}" in said
    assert "`investigate`" in said


def test_a_check_task_takes_no_slot_and_no_devices(tmp_path):
    """Its checks run while a worker holds the only slot and all four devices."""
    started = threading.Event()
    services_ran = []

    def plan(turn):
        if not services_ran:
            services_ran.append(turn)
            graph = g.Graph(
                tasks=[
                    _task("core", satisfies=[_BOOT, _ACC, _ENVELOPE], gpus=4),
                    _check_task("tp2", [_TP2]),
                ]
            )
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        return PlannerAnswer("DONE", "all green")

    def work(task):
        assert started.wait(_WAIT), "the check task waited for the worker's slot or devices"
        return "UNIT_READY"

    def check(chk):
        if chk.key == _TP2:
            started.set()
        return SimpleNamespace(name=chk.gate, member=chk.member, passed=True, detail="")

    services = Services(plan=plan, work=work, check=check, free_gpus=lambda: 4)
    orchestrator = GraphOrchestrator(
        services, _CONTRACT, c.ContractStatus.load(tmp_path), tmp_path, poll_s=0.01,
        max_parallel=1,
    )  # fmt: skip
    assert orchestrator.run() == "delivered"
    assert orchestrator._held(orchestrator.graph.by_id()["tp2"]) == 0


def test_a_check_tasks_delivery_tier_checks_wait_for_the_delivery_tier(tmp_path):
    run = _run(
        tmp_path,
        [
            _task("core", satisfies=[_BOOT, _ACC]),
            _check_task("final", [_TP2, _ENVELOPE], deps=["core"]),
        ],
    )
    assert run.verdict == "delivered"
    assert run.ran[-1] == _ENVELOPE and run.ran.count(_ENVELOPE) == 1


def test_a_check_task_running_when_the_process_stopped_starts_again(tmp_path):
    g.Graph(
        tasks=[
            _task("core", satisfies=[_BOOT, _ACC]),
            _check_task("tp2", [_TP2], deps=["core"]),
        ]
    ).save(tmp_path / g.GRAPH_FILENAME)
    saved = g.Graph.load(tmp_path / g.GRAPH_FILENAME)
    saved.by_id()["core"].status = "done"
    saved.by_id()["tp2"].status = "running"
    saved.save(tmp_path / g.GRAPH_FILENAME)

    def plan(turn):
        return PlannerAnswer("DONE", "all green")

    ran = []

    def check(chk):
        ran.append(chk.key)
        return SimpleNamespace(name=chk.gate, member=chk.member, passed=True, detail="")

    services = Services(plan=plan, work=lambda task: "UNIT_READY", check=check, free_gpus=lambda: 4)
    orchestrator = GraphOrchestrator(
        services, _CONTRACT, c.ContractStatus.load(tmp_path), tmp_path, poll_s=0.01
    )
    orchestrator.run()
    assert _TP2 in ran and orchestrator.graph.by_id()["tp2"].status == "done"


# ------------------------------------------------------------------ admission


def _errors(*tasks):
    return g.validate(g.Graph(tasks=list(tasks)), known_checks=[k.key for k in _CONTRACT.checks])


def test_a_check_task_names_contract_checks_and_changes_nothing():
    assert _errors(_check_task("tp2", [_TP2])) == []
    errors = _errors(
        _task("a", g.CHECK_KIND, [], gpus=0, touches=[]),
        _task("b", g.CHECK_KIND, ["core:boot@tp8"], gpus=0, touches=[]),
        _task("c", g.CHECK_KIND, [_TP2], gpus=2, touches=["models/c.py"]),
    )
    assert any("'a'" in e and "names none" in e for e in errors)
    assert any("'b'" in e and "core:boot@tp8 is no check of the contract" in e for e in errors)
    assert any("'c'" in e and "`touches` stay empty" in e for e in errors)
    assert any("'c'" in e and "`resources.gpus: 0`" in e for e in errors)


def test_the_loop_refuses_a_check_task_naming_no_contract_check(tmp_path):
    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(tasks=[_task("core", satisfies=[_BOOT, _ACC, _TP2]),
                                   _check_task("x", ["core:boot@tp9"])])  # fmt: skip
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        return PlannerAnswer("ESCALATE", "seen")

    services = Services(
        plan=plan,
        work=lambda task: "UNIT_READY",
        check=lambda chk: SimpleNamespace(name=chk.gate, member=chk.member, passed=True),
        free_gpus=lambda: 4,
    )
    orchestrator = GraphOrchestrator(
        services, _CONTRACT, c.ContractStatus.load(tmp_path), tmp_path, poll_s=0.01
    )
    assert orchestrator.run() == "escalated"
    assert any("is no check of the contract" in r for r in turns[1].refusals)


# ------------------------------------------------------------------ receipt


def test_a_check_task_that_passed_leaves_a_receipt(tmp_path):
    stage = PlannedStage()
    stage.workspace = tmp_path
    stage._turns = None
    task = _check_task("tp2", [_TP2])
    task.status = "done"
    stage._task_finished(task)
    receipt = yaml.safe_load(
        (tmp_path / TASKS_DIRNAME / f"tp2{receipts_mod.RECEIPT_SUFFIX}").read_text()
    )
    assert receipt["decision"] == CHECKED and _TP2 in receipt["note"]
    task.status = "failed"
    (tmp_path / TASKS_DIRNAME / f"tp2{receipts_mod.RECEIPT_SUFFIX}").unlink()
    stage._task_finished(task)
    assert not (tmp_path / TASKS_DIRNAME / f"tp2{receipts_mod.RECEIPT_SUFFIX}").exists()
