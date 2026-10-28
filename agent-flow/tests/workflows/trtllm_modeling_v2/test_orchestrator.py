# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The graph-driven loop under a fake planner, fake workers and fake checks."""

from __future__ import annotations

import errno
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
        _task(
            "core",
            "assemble",
            ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:serve@tp1"],
            deps=["k"],
        ),  # fmt: skip
    )
    services = _services(plan)
    assert _orchestrator(tmp_path, services).run() == "delivered"
    # The serve is claimed by the task but run as the delivery tier, once everything else holds.
    assert services.ran_checks[-1] == "core:serve@tp1"
    saved = g.Graph.load(tmp_path / g.GRAPH_FILENAME)
    assert {t.id: t.status for t in saved.tasks} == {"k": "done", "core": "done"}


def test_a_graph_that_leaves_a_required_check_unclaimed_is_sent_back(tmp_path):
    answers = [
        PlannerAnswer(
            "GRAPH_UPDATED", "too little", g.Graph(tasks=[_task("core", "assemble", [])])
        ),
        PlannerAnswer("ESCALATE", "cannot"),
    ]
    seen = []

    def plan(turn):
        seen.append(turn.refusals)
        return answers[len(seen) - 1]

    assert _orchestrator(tmp_path, _services(plan)).run() == "escalated"
    assert any("core:boot@tp1" in r for r in seen[1])


def test_a_failed_check_wakes_the_planner_with_the_evidence(tmp_path):
    turns = []
    keys = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:serve@tp1"]

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            return PlannerAnswer(
                "GRAPH_UPDATED", "plan", g.Graph(tasks=[_task("core", "assemble", keys)])
            )
        return PlannerAnswer("CEILING", "no attention kernel")

    services = _services(plan, failing={"core:accuracy[gsm8k]@tp1"})
    assert _orchestrator(tmp_path, services).run() == "ceiling"
    assert any("checks failed: core:accuracy[gsm8k]@tp1" in e for e in turns[1].events)
    assert "FAILED core:accuracy[gsm8k]@tp1" in turns[1].evaluation.render()


def test_a_failed_nice_to_have_is_abandoned_not_retried(tmp_path):
    keys = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:serve@tp1"]
    plan = _plan_once(
        _task("core", "assemble", keys),
        _task("eplb-ish", "feature", priority="nice_to_have"),
    )
    services = _services(plan, decisions={"eplb-ish": "BLOCKED"})
    orchestrator = _orchestrator(tmp_path, services)
    assert orchestrator.run() == "delivered"
    assert orchestrator.graph.by_id()["eplb-ish"].status == "abandoned"


def test_a_planner_that_never_makes_progress_is_stopped(tmp_path):
    def plan(turn):
        return PlannerAnswer("DONE", "trust me")

    orchestrator = _orchestrator(tmp_path, _services(plan), revision_cap=2)
    orchestrator.graph = g.Graph(tasks=[_task("idle", "investigate")])
    orchestrator.graph.tasks[0].status = "done"
    assert orchestrator.run() == "planner-stalled"


def test_checks_a_later_task_made_stale_run_again_on_the_final_target(tmp_path):
    keys = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:serve@tp1"]
    plan = _plan_once(
        _task("core", "assemble", keys[:2]),
        _task("tweak", "feature", ["core:serve@tp1"], deps=["core"]),
    )
    services = _services(plan)
    tree = {"marker": "v1"}
    services.marker = lambda: tree["marker"]
    original = services.work

    def work(task):
        if task.id == "tweak":
            tree["marker"] = "v2"  # the feature changed the target after core's checks
        return original(task)

    services.work = work
    assert _orchestrator(tmp_path, services).run() == "delivered"
    ran = services.ran_checks  # type: ignore[attr-defined]
    assert ran.count("core:boot@tp1") == 2 and ran.count("core:accuracy[gsm8k]@tp1") == 2


def test_a_stale_benchmark_waits_for_a_milestone_while_required_work_runs(tmp_path):
    # One check at a time, in the contract's order: mtp's benchmark would run before its boot.
    contract = c.Contract(
        checks=tuple(
            c.Check(unit, gate, "tp1", c.TIER_INNER)
            for unit, gates in (
                ("core", ("boot", "accuracy[gsm8k]")),
                ("mtp", ("accuracy[gsm8k]", "boot")),
            )
            for gate in gates
        )
    )
    plan = _plan_once(
        _task("core", "assemble", ["core:boot@tp1", "core:accuracy[gsm8k]@tp1"]),
        _task("mtp", "feature", ["mtp:boot@tp1", "mtp:accuracy[gsm8k]@tp1"], deps=["core"]),
        _task("tweak", "feature", deps=["mtp"]),
        _task("slow", "feature", deps=["core"]),
    )
    services = _services(plan)
    tree = {"marker": "v1"}
    services.marker = lambda: tree["marker"]
    release, seen = threading.Event(), {}
    check = services.check

    def watch(chk):
        result = check(chk)
        if chk.key == "mtp:boot@tp1" and services.ran_checks.count(chk.key) == 2:
            # The landing's re-checks, while `slow` still runs.
            seen["benchmarks"] = services.ran_checks.count("mtp:accuracy[gsm8k]@tp1")
            release.set()
        return result

    def work(task):
        if task.id == "tweak":
            tree["marker"] = "v2"  # mtp's checks go stale
        if task.id == "slow":
            release.wait(10)
        return "UNIT_READY"

    services.check, services.work = watch, work
    status = c.ContractStatus.load(tmp_path)
    loop = GraphOrchestrator(services, contract, status, tmp_path, poll_s=0, max_parallel_checks=1)
    assert loop.run() == "delivered"
    assert seen == {"benchmarks": 1}
    assert services.ran_checks.count("mtp:accuracy[gsm8k]@tp1") == 2


def test_a_tune_task_waits_for_the_core_checks_then_runs_beside_other_work(tmp_path):
    started: list[tuple[str, tuple[str, ...]]] = []
    plan = _plan_once(
        _task("tune", "tune", touches=[]),
        _task(
            "core",
            "assemble",
            ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:serve@tp1"],
        ),  # fmt: skip
    )
    services = _services(plan)

    def work(task):
        started.append((task.id, tuple(services.ran_checks)))
        return "TUNED" if task.kind == "tune" else "UNIT_READY"

    services.work = work
    orchestrator = _orchestrator(
        tmp_path,
        services,
        required_kinds=("tune",),
        start_after={"tune": ["core:boot@tp1", "core:accuracy[gsm8k]@tp1"]},
    )
    assert orchestrator.run() == "delivered"
    assert started[0] == ("core", ())
    assert started[1][0] == "tune"
    assert {"core:boot@tp1", "core:accuracy[gsm8k]@tp1"} <= set(started[1][1])


def test_a_task_left_running_by_a_dead_process_starts_again(tmp_path):
    task = _task(
        "core", "assemble", ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:serve@tp1"]
    )
    task.status = "running"
    g.Graph(tasks=[task]).save(tmp_path / g.GRAPH_FILENAME)
    ran = []
    services = _services(_plan_once())
    services.work = lambda t: ran.append(t.id) or "UNIT_READY"
    assert _orchestrator(tmp_path, services).run() == "delivered"
    assert ran == ["core"]


# ----------------------------------------------------------------- retries


def test_an_integration_failure_starts_the_task_again_once_then_fails_it(tmp_path):
    keys = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:serve@tp1"]
    turns, attempts = [], []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(tasks=[_task("core", "assemble", keys)])
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        return PlannerAnswer("ESCALATE", "the branch keeps conflicting")

    def work(task):
        attempts.append(task.id)
        return "INTEGRATION_FAILED\nbranch mv2/x/core conflicts with the integration head in: a.py"

    services = _services(plan)
    services.work = work
    # The retry's event waits for the window; the second failure wakes the planner.
    orchestrator = _orchestrator(tmp_path, services, event_window_s=60)
    assert orchestrator.run() == "escalated"
    assert attempts == ["core", "core"]
    assert orchestrator.graph.by_id()["core"].status == "failed"
    assert orchestrator.retries == {"core@1": 1}
    events = turns[1].events
    assert any(
        e.startswith("task core (assemble) ended INTEGRATION_FAILED, a failure of the run's")
        and "conflicts with the integration head" in e
        for e in events
    )
    assert "task core (assemble) ended INTEGRATION_FAILED" in events


def test_a_worker_turn_that_raised_starts_again_and_the_run_goes_on(tmp_path):
    keys = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:serve@tp1"]
    attempts = []

    def work(task):
        attempts.append(task.id)
        if len(attempts) == 1:
            raise OSError("the dispatch session went away")
        return "UNIT_READY"

    services = _services(_plan_once(_task("core", "assemble", keys)))
    services.work = work
    assert _orchestrator(tmp_path, services).run() == "delivered"
    assert attempts == ["core", "core"]


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


def test_full_support_tells_the_campaign_to_wrap_up_and_the_run_is_delivered(tmp_path):
    """The campaign gives way to full support: told to stop, it wraps up and nothing waits for it."""
    stopped = threading.Event()
    reasons: list[str] = []
    logs: list[str] = []

    def work(task):
        if task.kind == "tune":
            # Runs until told to stop, as a campaign with hours of budget left would.
            assert stopped.wait(timeout=30)
            return "TUNED\ntuned: best branch `tune/x/best`"
        return "UNIT_READY"

    def stop(reason):
        reasons.append(reason)
        stopped.set()

    services = _services(
        _plan_once(_task("core", "assemble", _KEYS), _task("tune", "tune", touches=[]))
    )
    services.work, services.stop_campaign, services.log = work, stop, logs.append
    services.marker = lambda: "v1"

    def correct(marker, evaluation):
        milestone = c.Milestone(sha="a" * 40, marker=marker, checks=len(evaluation.passed))
        c.record_milestone(tmp_path, milestone)
        return milestone

    services.correct = correct
    orchestrator = _orchestrator(
        tmp_path, services, required_kinds=("tune",), start_after={"tune": _KEYS[:2]}
    )
    assert orchestrator.run() == "delivered"
    assert reasons == ["full support holds on the target as it is"]
    assert orchestrator.graph.by_id()["tune"].status == "done"
    # The campaign's report is journaled for the record; no planner turn had to read it.
    assert any("best branch `tune/x/best`" in line for line in logs)
    assert [m.marker for m in c.milestones(tmp_path)] == ["v1"]


def test_an_escalation_lets_the_work_that_does_not_wait_on_it_finish(tmp_path):
    turns, escalated = [], threading.Event()

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(
                tasks=[
                    _task("vision", "feature", ["core:boot@tp1"]),
                    _task("core", "assemble", ["core:accuracy[gsm8k]@tp1", "core:serve@tp1"]),
                ]
            )
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        assert not turn.refusals
        escalated.set()
        return PlannerAnswer("ESCALATE", "the checkpoint has no image processor")

    def work(task):
        if task.id == "vision":
            return "BLOCKED"
        assert escalated.wait(10)
        return "UNIT_READY"

    services = _services(plan)
    services.work = work
    orchestrator = _orchestrator(tmp_path, services)
    assert orchestrator.run() == "escalated"
    assert orchestrator.graph.by_id()["core"].status == "done"
    # What finished after the escalation reached the planner before the run ended.
    assert any(event.startswith("task core (assemble) UNIT_READY") for event in turns[-1].events)
    assert any(event.startswith("you escalated") for event in turns[-1].events)


def test_a_full_filesystem_holds_new_work_until_the_state_is_written(tmp_path, monkeypatch):
    refused, started_after = [], []
    services = _services(_plan_once(_task("core", "assemble", _KEYS)))
    services.work = lambda task: started_after.append(len(refused)) or "UNIT_READY"
    orchestrator = _orchestrator(tmp_path, services)
    write_graph = orchestrator._write_graph

    def quota_then_room():
        if len(refused) < 3:
            refused.append(True)
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        write_graph()

    monkeypatch.setattr(orchestrator, "_write_graph", quota_then_room)
    assert orchestrator.run() == "delivered"
    assert started_after == [3]
    assert g.Graph.load(tmp_path / g.GRAPH_FILENAME).by_id()["core"].status == "done"
