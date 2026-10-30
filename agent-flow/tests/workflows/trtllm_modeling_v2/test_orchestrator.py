# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The graph-driven loop under a fake planner, fake workers and fake checks."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)

_CONTRACT = c.Contract(
    checks=(
        c.Check("core", "boot", "tp1", c.TIER_INNER),
        c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER),
        c.Check("core", "envelope", "tp1", c.TIER_DELIVERY),
    )
)


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
            ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"],
            deps=["k"],
        ),  # fmt: skip
    )
    services = _services(plan)
    assert _orchestrator(tmp_path, services).run() == "delivered"
    # The envelope is claimed by the task but run as the delivery tier, once everything else holds.
    assert services.ran_checks[-1] == "core:envelope@tp1"
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
    keys = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]

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
    keys = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]
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


def test_work_that_needs_more_devices_than_exist_goes_back_to_the_planner(tmp_path):
    keys = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]
    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            big = _task("core", "assemble", keys)
            big.gpus = 8
            return PlannerAnswer("GRAPH_UPDATED", "plan", g.Graph(tasks=[big]))
        return PlannerAnswer("ESCALATE", "needs a bigger node")

    assert _orchestrator(tmp_path, _services(plan, gpus=4)).run() == "escalated"
    assert any("cannot start on 4 free GPUs" in e for e in turns[1].events)


def test_checks_a_later_task_made_stale_run_again_on_the_final_target(tmp_path):
    keys = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]
    plan = _plan_once(
        _task("core", "assemble", keys[:2]),
        _task("tweak", "feature", ["core:envelope@tp1"], deps=["core"]),
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


def test_a_tune_task_waits_for_the_core_checks_then_runs_beside_other_work(tmp_path):
    started: list[tuple[str, tuple[str, ...]]] = []
    plan = _plan_once(
        _task("tune", "tune", touches=[]),
        _task(
            "core",
            "assemble",
            ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"],
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


def test_a_gated_task_that_can_never_start_goes_back_to_the_planner(tmp_path):
    seen = []

    def plan(turn):
        seen.append(turn)
        if len(seen) == 1:
            return PlannerAnswer("GRAPH_UPDATED", "tune only", g.Graph(tasks=[
                _task("tune", "tune", ["core:boot@tp1", "core:accuracy[gsm8k]@tp1",
                                       "core:envelope@tp1"], touches=[])]))  # fmt: skip
        return PlannerAnswer("ESCALATE", "no core")

    orchestrator = _orchestrator(tmp_path, _services(plan), start_after={"tune": ["core:boot@tp1"]})
    assert orchestrator.run() == "escalated"
    assert any("tune (waits for core:boot@tp1)" in e for e in seen[1].events)


def test_a_report_reaches_the_planner_before_the_run_is_delivered(tmp_path):
    plan = _plan_once(
        _task(
            "core",
            "assemble",
            ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"],
        ),  # fmt: skip
    )
    services = _services(plan, decisions={"core": "UNIT_READY\nbest branch tune/x/best"})
    assert _orchestrator(tmp_path, services).run() == "delivered"
    assert len(plan.turns) == 2
    assert "task core (assemble) reports: best branch tune/x/best" in plan.turns[1].events


def test_a_task_left_running_by_a_dead_process_starts_again(tmp_path):
    task = _task(
        "core", "assemble", ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]
    )
    task.status = "running"
    g.Graph(tasks=[task]).save(tmp_path / g.GRAPH_FILENAME)
    ran = []
    services = _services(_plan_once())
    services.work = lambda t: ran.append(t.id) or "UNIT_READY"
    assert _orchestrator(tmp_path, services).run() == "delivered"
    assert ran == ["core"]


def test_events_a_dead_process_left_unread_reach_the_next_planner_turn(tmp_path):
    first = _orchestrator(tmp_path, _services(_plan_once()))
    first.events.append("critic (advisory) on this graph: REJECT, the boot keywords")
    first.unread_reports = True
    first._save_events()
    task = _task(
        "core", "assemble", ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]
    )
    g.Graph(tasks=[task]).save(tmp_path / g.GRAPH_FILENAME)
    plan = _plan_once()
    second = _orchestrator(tmp_path, _services(plan))
    assert second.events == ["critic (advisory) on this graph: REJECT, the boot keywords"]
    assert second.run() == "delivered"
    assert "REJECT, the boot keywords" in plan.turns[0].events[0]
    after = _orchestrator(tmp_path, _services(_plan_once())).events
    assert not [e for e in after if "REJECT" in e]  # read once, not again


def test_work_that_lands_counts_as_progress_even_before_a_check_turns_green(tmp_path):
    """Kernel tasks claim no check; their landing is still the run moving forward."""
    kernels = [_task(f"k{i}", "kernel", []) for i in range(4)]
    core = _task(
        "core", "assemble", ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]
    )
    core.depends_on = [k.id for k in kernels]

    def plan(turn):
        # Every event gets a revision, as a planner reading kernel landings would write.
        graph = orchestrator.graph if orchestrator.graph.tasks else g.Graph([*kernels, core])
        return PlannerAnswer("GRAPH_UPDATED", "keep going", g.Graph(tasks=list(graph.tasks)))

    services = _services(plan)
    services.work = lambda t: "LANDED" if t.kind == "kernel" else "UNIT_READY"
    orchestrator = _orchestrator(tmp_path, services, revision_cap=2, max_parallel=1)
    assert orchestrator.run() == "delivered"


def test_checks_that_pass_in_the_task_thread_reset_the_stall_count(tmp_path):
    services = _services(_plan_once())
    orchestrator = _orchestrator(tmp_path, services)
    task = _task("core", "assemble", ["core:boot@tp1"])
    orchestrator.graph = g.Graph(tasks=[task])
    orchestrator.revisions_since_progress = 5
    failed = orchestrator._run_checks(task.satisfies)  # as the task's thread does
    task.status = "running"
    orchestrator._finish(task, "BLOCKED", failed)
    assert task.status == "failed" and orchestrator.revisions_since_progress == 0


def test_a_landing_runs_the_riskiest_stale_checks_again_beside_the_work_after_it(tmp_path):
    """Always green: checks made stale by a landing run again while later work is still going.

    core's checks go stale when ``tweak`` lands, and run again while ``tweak2`` is still at
    work, not only once the graph has run out of work.
    """
    keys = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]
    tasks = [
        _task("core", "assemble", keys[:2]),
        _task("tweak", "feature", [], deps=["core"]),
        _task("tweak2", "feature", ["core:envelope@tp1"], deps=["tweak"]),
    ]
    rechecked = threading.Event()
    seen = {}
    services = _services(_plan_once(*tasks))
    tree = {"marker": "v1"}
    services.marker = lambda: tree["marker"]
    check = services.check
    ran: list[str] = []

    def counted(chk):
        ran.append(chk.key)
        if ran.count("core:boot@tp1") == 2:
            rechecked.set()
        return check(chk)

    def work(task):
        if task.id == "tweak":
            tree["marker"] = "v2"
        if task.id == "tweak2":
            seen["re-checked while tweak2 ran"] = rechecked.wait(timeout=10)
        return "UNIT_READY"

    services.work, services.check = work, counted
    orchestrator = _orchestrator(tmp_path, services, max_parallel=1)
    assert orchestrator.run() == "delivered"
    assert seen["re-checked while tweak2 ran"] is True
    assert ran.count("core:boot@tp1") == 2  # once on v1, once again on v2


# ----------------------------------------------------------------- retries


def test_an_integration_failure_starts_the_task_again_once_then_fails_it(tmp_path):
    keys = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]
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
    keys = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]
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
    [
        ("TUNE_FAILED (tune-failed:baseline)", True),
        ("raised TimeoutError: no answer from the driver", True),
        ("INTEGRATION_FAILED\nuncommitted changes in worktrees/tune", True),
        ("TUNE_FAILED (tune-failed:spec)", False),
        ("BLOCKED", False),
        ("checks raised RuntimeError: the spool is gone", False),
    ],
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


# --------------------------------------------------------------- tune gate


def _gate_run(tmp_path, *verdicts):
    """A tune task gated on the core's inner checks, whose verdicts are all stale."""
    status = c.ContractStatus.load(tmp_path)
    for gate, passed in verdicts:
        result = SimpleNamespace(name=gate, member="tp1", passed=passed, detail="")
        status.record("core", result, marker="v1")
    g.Graph(tasks=[_task("tune", "tune", touches=[])]).save(tmp_path / g.GRAPH_FILENAME)
    started, turns = [], []

    def plan(turn):
        turns.append(turn)
        return PlannerAnswer("ESCALATE", "stop here")

    services = _services(plan)
    services.work = lambda task: started.append(task.id) or "TUNED"
    services.marker = lambda: "v2"  # a task integrated since every verdict above
    start_after = {"tune": ["core:boot@tp1", "core:accuracy[gsm8k]@tp1"]}
    orchestrator = GraphOrchestrator(
        services, _CONTRACT, status, tmp_path, poll_s=0, start_after=start_after
    )
    assert orchestrator.run() == "escalated"
    return started, turns


def test_a_stale_core_pass_releases_the_tune_gate(tmp_path):
    started, _ = _gate_run(tmp_path, ("boot", True), ("accuracy[gsm8k]", True))
    assert started == ["tune"]


def test_every_recorded_outcome_is_handed_to_the_finished_hook(tmp_path):
    keys = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]
    plan = _plan_once(
        _task("core", "assemble", keys),
        _task("extra", "feature", priority="nice_to_have"),
        _task("flaky", "feature"),
    )
    attempts, outcomes = [], []

    def work(task):
        attempts.append(task.id)
        if task.id == "extra":
            return "BLOCKED"
        if task.id == "flaky" and attempts.count("flaky") == 1:
            raise OSError("the dispatch session went away")
        return "UNIT_READY"

    def finished(task):
        # Called once the outcome is on disk, where a resumed process reads it.
        on_disk = g.Graph.load(tmp_path / g.GRAPH_FILENAME).by_id()[task.id].status
        outcomes.append((task.id, task.status, on_disk))

    services = _services(plan)
    services.work, services.finished = work, finished
    assert _orchestrator(tmp_path, services).run() == "delivered"
    assert sorted(outcomes) == [
        ("core", "done", "done"),
        ("extra", "abandoned", "abandoned"),
        ("flaky", "done", "done"),
        ("flaky", "pending", "pending"),
    ]


def test_a_later_core_failure_holds_the_tune_gate(tmp_path):
    started, turns = _gate_run(tmp_path, ("boot", True), ("accuracy[gsm8k]", True), ("boot", False))
    assert started == []
    assert any("tune (waits for core:boot@tp1)" in e for e in turns[0].events)
