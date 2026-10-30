# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The graph-driven loop under a fake planner, fake workers and fake checks."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c

_CONTRACT = c.Contract(
    checks=(
        c.Check("core", "boot", "tp1", c.TIER_INNER),
        c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER),
        c.Check("core", "serve", "tp1", c.TIER_DELIVERY),
    )
)
_KEYS = [check.key for check in _CONTRACT.checks]


def _task(tid, kind, satisfies=(), deps=(), priority="required", touches=None):
    return g.Task(
        id=tid,
        kind=kind,
        goal=f"do {tid}",
        depends_on=list(deps),
        touches=list(touches if touches is not None else [f"models/{tid}/"]),
        gpus=1,
        est_minutes=10,
        satisfies=list(satisfies),
        priority=priority,
    )


def _plan_once(*tasks):
    """A planner whose first turn writes ``tasks`` and later turns say DONE."""
    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            return PlannerAnswer("GRAPH_UPDATED", "first plan", g.Graph(tasks=list(tasks)))
        return PlannerAnswer("DONE", "all green")

    plan.turns = turns  # type: ignore[attr-defined]
    return plan


def _services(plan, *, decisions=None, failing=(), gpus=4):
    ran_checks = []

    def check(chk):
        ran_checks.append(chk.key)
        return SimpleNamespace(name=chk.gate, member=chk.member, passed=chk.key not in failing,
                               detail="", tree_state="")  # fmt: skip

    services = Services(
        plan=plan,
        work=lambda task: (decisions or {}).get(task.id, "UNIT_READY"),
        check=check,
        free_gpus=lambda: gpus,
    )
    services.ran_checks = ran_checks  # type: ignore[attr-defined]
    return services


def _orchestrator(tmp_path, services, **kw):
    status = c.ContractStatus.load(tmp_path)
    return GraphOrchestrator(services, _CONTRACT, status, tmp_path, poll_s=0, **kw)


def test_a_plan_whose_checks_pass_is_delivered_with_the_delivery_tier_last(tmp_path):
    plan = _plan_once(
        _task("k", "kernel", touches=["catalog/k.py"]),
        _task("core", "assemble", _KEYS, deps=["k"]),
    )
    services = _services(plan)
    assert _orchestrator(tmp_path, services).run() == "delivered"
    # The serve is claimed by the task but run as the delivery tier, once everything else holds.
    assert services.ran_checks[-1] == "core:serve@tp1"
    saved = g.Graph.load(tmp_path / g.GRAPH_FILENAME)
    assert {t.id: t.status for t in saved.tasks} == {"k": "done", "core": "done"}


def test_a_failed_check_wakes_the_planner_with_the_evidence(tmp_path):
    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            return PlannerAnswer(
                "GRAPH_UPDATED", "plan", g.Graph(tasks=[_task("core", "assemble", _KEYS)])
            )
        return PlannerAnswer("CEILING", "no attention kernel")

    services = _services(plan, failing={"core:accuracy[gsm8k]@tp1"})
    assert _orchestrator(tmp_path, services).run() == "ceiling"
    assert any("checks failed: core:accuracy[gsm8k]@tp1" in e for e in turns[1].events)
    assert "FAILED core:accuracy[gsm8k]@tp1" in turns[1].evaluation.render()


# ----------------------------------------------------------------- retries


@pytest.mark.parametrize(
    "decision, retried",
    [("raised TimeoutError: no answer from the driver", True), ("BLOCKED", False)],
)
def test_only_an_infrastructure_failure_is_retried_and_only_once(tmp_path, decision, retried):
    def failing(orchestrator):
        task = _task("tune", "tune", touches=[])
        task.status = "running"
        orchestrator.graph = g.Graph(tasks=[task])
        orchestrator._finish(task, decision, [])
        return task.status

    assert failing(_orchestrator(tmp_path, _services(_plan_once()))) == (
        "pending" if retried else "failed"
    )
    # The count is kept with the planner's events: a restarted process does not retry again.
    assert failing(_orchestrator(tmp_path, _services(_plan_once()))) == "failed"


# ------------------------------------------------------------ a tune campaign


def test_a_campaign_starts_once_its_checks_hold_and_the_run_does_not_wait_for_it(tmp_path):
    """The campaign follows the onboarding; the onboarding's verdict never waits for it."""
    logs: list[str] = []
    started: list[tuple[str, tuple[str, ...]]] = []
    release = threading.Event()

    def work(task):
        started.append((task.id, tuple(services.ran_checks)))
        if task.kind == "tune":
            assert release.wait(10)
            return "TUNED\ntuned: best branch `tune/x/best`"
        return "UNIT_READY"

    # Listed first, the campaign still waits for the checks it is gated on.
    services = _services(
        _plan_once(_task("tune", "tune", touches=[]), _task("core", "assemble", _KEYS))
    )
    services.work, services.log = work, logs.append
    services.marker = lambda: "v1"

    def correct(marker, evaluation):
        milestone = c.Milestone(sha="a" * 40, marker=marker, checks=len(evaluation.passed))
        c.record_milestone(tmp_path, milestone)
        return milestone

    services.correct = correct
    orchestrator = _orchestrator(
        tmp_path, services, required_kinds=("tune",), start_after={"tune": _KEYS}
    )
    try:
        assert orchestrator.run() == "delivered"
        # Delivered with the campaign still running: handed over, not waited for.
        assert orchestrator.campaign is not None and not orchestrator.campaign.done()
    finally:
        release.set()
    assert started[0] == ("core", ()) and started[1][0] == "tune"
    assert set(_KEYS) <= set(started[1][1])
    assert orchestrator.campaign.result(10)[0].startswith("TUNED")
    assert [m.marker for m in c.milestones(tmp_path)] == ["v1"]
    # While the core works, the journal says what the campaign waits for, once per change.
    waits = [line for line in logs if line.startswith("[gated] tune (tune) waits for")]
    assert waits and waits[0].endswith(f"{', '.join(_KEYS)} to pass")
    assert len(waits) == len(set(waits))


def test_an_escalation_does_not_wait_for_a_campaign(tmp_path):
    turns, escalated, release = [], threading.Event(), threading.Event()

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(
                tasks=[
                    _task("vision", "feature", ["core:boot@tp1"]),
                    _task("core", "assemble", _KEYS[1:]),
                    _task("tune", "tune", touches=[]),
                ]
            )
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        escalated.set()
        return PlannerAnswer("ESCALATE", "the checkpoint has no image processor")

    def work(task):
        if task.id == "vision":
            return "BLOCKED"
        if task.kind == "tune":
            assert release.wait(10)
            return "TUNED"
        assert escalated.wait(10)
        return "UNIT_READY"

    services = _services(plan)
    services.work = work
    orchestrator = _orchestrator(
        tmp_path, services, required_kinds=("tune",), start_after={"tune": [_KEYS[1]]}
    )
    try:
        assert orchestrator.run() == "escalated"
        assert orchestrator.campaign is not None and not orchestrator.campaign.done()
    finally:
        release.set()
    assert orchestrator.campaign.result(10)[0] == "TUNED"


def test_a_campaign_takes_at_most_two_agent_slots_while_the_onboarding_works(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator as orchestrator_mod

    loop = _orchestrator(tmp_path, _services(_plan_once()), max_parallel=9)
    idle = loop.idle_agent_slots()
    assert idle > orchestrator_mod.CAMPAIGN_SLOTS_BESIDE_ONBOARDING == 2
    campaign = _task("tune", "tune", touches=[])
    campaign.status = "running"
    loop.graph = g.Graph(tasks=[campaign, _task("vision", "feature")])
    assert loop.campaign_agent_slots() == 2
    loop._foreground_demand = idle - 1
    assert loop.campaign_agent_slots() == 1
    # The onboarding's work all ended: the campaign may take every idle slot.
    loop.graph.by_id()["vision"].status = "done"
    loop._foreground_demand = 0
    assert loop.campaign_agent_slots() == idle
