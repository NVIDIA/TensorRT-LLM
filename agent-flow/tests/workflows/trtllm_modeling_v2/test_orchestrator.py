# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 orchestrator behavior and regression checks."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import test_results
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.sessions import SessionSpec
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run import inbox
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run.validation import regression
from agent_flow.workflows.trtllm_modeling_v2.run.workspace.integration import TaskNotStarted

_orchestrator_CONTRACT = c.Contract(
    checks=(
        c.Check("core", "boot", "tp1", c.TIER_INNER),
        c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER),
        c.Check("core", "serve", "tp1", c.TIER_DELIVERY),
    )
)


_KEYS = [check.key for check in _orchestrator_CONTRACT.checks]


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


def _orchestrator_orchestrator(tmp_path, services, **kw):
    status = c.ContractStatus.load(tmp_path)
    return GraphOrchestrator(services, _orchestrator_CONTRACT, status, tmp_path, poll_s=0, **kw)


def test_a_plan_whose_checks_pass_is_delivered_with_the_delivery_tier_last(tmp_path):
    plan = _plan_once(
        _task("k", "kernel", touches=["catalog/k.py"]),
        _task("core", "assemble", _KEYS, deps=["k"]),
    )
    services = _services(plan)
    assert _orchestrator_orchestrator(tmp_path, services).run() == "delivered"
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
    assert _orchestrator_orchestrator(tmp_path, services).run() == "ceiling"
    assert any("checks failed: core:accuracy[gsm8k]@tp1" in e for e in turns[1].events)
    assert "FAILED core:accuracy[gsm8k]@tp1" in turns[1].evaluation.render()


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

    assert failing(_orchestrator_orchestrator(tmp_path, _services(_plan_once()))) == (
        "pending" if retried else "failed"
    )
    # The count is kept with the planner's events: a restarted process does not retry again.
    assert failing(_orchestrator_orchestrator(tmp_path, _services(_plan_once()))) == "failed"


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
    orchestrator = _orchestrator_orchestrator(
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
    orchestrator = _orchestrator_orchestrator(
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

    loop = _orchestrator_orchestrator(tmp_path, _services(_plan_once()), max_parallel=9)
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


def test_unavailable_tune_measurement_leaves_a_resumable_task_without_failure(tmp_path):
    orch = _orchestrator_orchestrator(tmp_path, _services(_plan_once()))
    task = _task("tune", "tune", touches=[])
    task.status = "running"
    orch.graph = g.Graph(tasks=[task])
    orch._finish(task, "TUNE_PENDING\nallocation expired", [])
    assert task.status == "pending" and task.id in orch._deferred_campaigns
    assert not any("failed" in str(event) for event in orch.events)


@pytest.fixture
def _quick_retries(monkeypatch):
    monkeypatch.setattr(
        "agent_flow.workflows.trtllm_modeling_v2.run.orchestrator.PLANNER_RETRY_S", 0.01
    )


@pytest.mark.usefixtures("_quick_retries")
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
    orchestrator = _orchestrator_orchestrator(tmp_path, services)
    assert orchestrator.run() == "delivered"
    assert any("ended BLOCKED" in e for e in turns[1].events)
    # The third turn reads what the second could not answer, and why it is asked again.
    assert any("ended BLOCKED" in e for e in turns[2].events)
    assert any("your previous turn raised TimeoutError" in e for e in turns[2].events)


@pytest.mark.usefixtures("_quick_retries")
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

    orchestrator = _orchestrator_orchestrator(tmp_path, _services(plan, decisions={"k": "MACRO"}))
    assert orchestrator.run() == "delivered"
    assert orchestrator.graph.by_id()["k"].status == "split"
    # A split task lands nothing and holds no dependent: the run is delivered past it.
    assert g.Graph.load(tmp_path / g.GRAPH_FILENAME).by_id()["k"].status == "split"


def _replanning(*reasons):
    """A planner that writes the same one-task graph on every turn: the retry is the status."""
    turns = []

    def plan(turn):
        turns.append(turn)
        reason = reasons[min(len(turns), len(reasons)) - 1]
        return PlannerAnswer(
            "GRAPH_UPDATED", reason, g.Graph(tasks=[_task("core", "assemble", _KEYS)])
        )

    plan.turns = turns  # type: ignore[attr-defined]
    return plan


def test_an_attempt_whose_worktree_was_never_prepared_does_not_spend_a_worker_run(tmp_path):
    runs = []

    def work(task):
        runs.append(task.attempts)
        if task.attempts <= 3:
            raise TaskNotStarted("worktree checkout: Disk quota exceeded")
        return "BLOCKED"

    services = _services(_replanning("storage repaired; retry the same task"))
    services.work = work
    runner = _orchestrator_orchestrator(tmp_path, services, revision_cap=10)
    assert runner.run() == "planner-refused"
    # Each launch that never started a worker gets its one infrastructure retry, keeps
    # its number, and leaves the task its MAX_ATTEMPTS worker runs.
    assert runs == [1, 1, 2, 2, 3, 3, 4, 5, 6]
    assert runner.not_started_attempts == {"core": [1, 2, 3]}
    assert _orchestrator_orchestrator(tmp_path, services).not_started_attempts == {
        "core": [1, 2, 3]
    }
    # The run sets how many worker runs a task gets; a long goal is adopted, and said.
    failed = g.Graph(tasks=[_task("core", "assemble", _KEYS)])
    failed.tasks[0].status, failed.tasks[0].attempts = "failed", 3
    retry = g.Graph(tasks=[_task("core", "assemble", _KEYS)])
    retry.tasks[0].goal = "x" * (g.MAX_GOAL_CHARS + 1)
    assert "failed on attempt 3 of 3" in g.apply_revision(failed, retry, "retry")[1][0]
    adopted, refused = g.apply_revision(failed, retry, "retry", max_attempts=4)
    assert not refused and adopted.tasks[0].attempts == 4
    assert "it was adopted" in g.long_goals(failed, retry)[0]


_CONCURRENT_CONTRACT = c.Contract(
    checks=(
        c.Check("core", "boot", "tp1", c.TIER_INNER),
        c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER),
    )
)


_CONCURRENT_KEYS = [check.key for check in _CONCURRENT_CONTRACT.checks]


def _passed(check):
    return SimpleNamespace(name=check.gate, member=check.member, passed=True, detail="")


def _concurrent_orchestrator(tmp_path, plan, work, **kw):
    services = Services(plan=plan, work=work, check=_passed, free_gpus=lambda: 8)
    status = c.ContractStatus.load(tmp_path)
    return GraphOrchestrator(services, _CONCURRENT_CONTRACT, status, tmp_path, poll_s=0.01, **kw)


def test_feature_work_on_the_core_waits_for_its_boot(tmp_path):
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
            mtp = task("mtp", "feature", "mtp")
            mtp.depends_on_soft = ["core"]
            graph = g.Graph(
                tasks=[
                    task("core", "assemble", "core", satisfies=_CONCURRENT_KEYS),
                    mtp,
                    task("mtp-test", "feature", "mtp"),
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

    orchestrator = _concurrent_orchestrator(
        tmp_path, plan, work, after_core=(["mtp"], [_CONCURRENT_KEYS[0]])
    )
    assert orchestrator.run() == "delivered"
    # The kernel the feature needs, and the feature's work that does not build on the core,
    # start beside the core; the feature's work on the core waits for its boot.
    assert seen["beside the core"] == {"k1-draft-head", "mtp-test"}
    assert started.index("mtp") > started.index("core")


def test_a_burst_of_operator_steers_reaches_the_planner_as_one(tmp_path):
    long = inbox.post(tmp_path, "x" * (inbox.MESSAGE_MAX_CHARS + 1))
    inbox.post(tmp_path, "drop the vision work")
    last = inbox.post(tmp_path, "stop the dep4 boot loop", now=True)
    assert inbox.take(tmp_path, now=last.stat().st_mtime + inbox.SETTLE_S / 2) == []
    taken = inbox.take(tmp_path, now=last.stat().st_mtime + inbox.SETTLE_S + 1)
    # A long one reaches the planner as its head and the path of the whole.
    assert f"1201 characters; the whole message: {long.parent / inbox.TAKEN_DIRNAME}" in taken[0]
    assert [text.rsplit(": ", 1)[-1] for text in taken[1:]] == [
        "drop the vision work",
        "stop the dep4 boot loop",
    ]
    assert [inbox.is_urgent(e) for e in taken] == [False, False, True]
    assert all(inbox.is_message(e) for e in taken) and ".now" not in taken[2]


_WAIT = 10.0


_PKG = "tensorrt_llm/_torch/_experimental/modeling_v2"


_TARGET = f"{_PKG}/models/fam/newmodel__sm_100__unified"


_OWN = g.own_paths(_TARGET)


_BOOT = "core:boot@tp1"


_regression_batches_CONTRACT = c.Contract(checks=(c.Check("core", "boot", "tp1", c.TIER_INNER),))


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


def _regression_batches_orchestrator(tmp_path, plan, work, regression=None, **kw):
    services = Services(
        plan=plan,
        work=work,
        check=lambda chk: SimpleNamespace(name=chk.gate, member=chk.member, passed=True, detail=""),
        free_gpus=lambda: 8,
        regression=regression or (lambda task: []),
    )
    kw.setdefault("poll_s", 0.01)
    status = c.ContractStatus.load(tmp_path)
    return GraphOrchestrator(
        services, _regression_batches_CONTRACT, status, tmp_path, own_paths=_OWN, **kw
    )


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
    orchestrator = _regression_batches_orchestrator(tmp_path, plan, work, regression=regression)
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
    orchestrator = _regression_batches_orchestrator(tmp_path, plan, lambda task: "LANDED" if task.kind == "kernel"
                                 else "UNIT_READY", regression=regression)  # fmt: skip
    orchestrator.s.check = check
    assert orchestrator.run() == "delivered"
    assert order[-1] == "r1" and len(order) >= 2


def test_regression_jobs_fit_the_gate_sessions_and_keep_the_test_budget(tmp_path, monkeypatch):
    # Two-hour allocations handing over 30 min before their end: a two-hour job never starts.
    sessions = (
        SessionSpec("gates", "/g", time_limit_s=7200, handover_lead_s=1800),
        SessionSpec("work", "/w", role="work", scheduler={"default_wall": "5m"}),
    )
    env = RunEnvironment("session_spool", "python", spool="/g", sessions=sessions)
    observed = []

    class Executor:
        def execute(self, request):
            observed.append(request)
            return SimpleNamespace(
                returncode=0, stdout="MV2FLOW_TEST_ENV 0123456789abcdef\n", stderr=""
            )

    monkeypatch.setattr(regression, "select_command_executor", lambda *args, **kwargs: Executor())
    wf = SimpleNamespace(task_data={"trtllm_repo_path": str(tmp_path)}, _gate_env=lambda: env)
    tree_at, run, python = regression.job_runner(wf, "regression")
    files = [f"tests/test_{i}.py" for i in range(6)]

    def recorded(cache, commit, env_digest=""):
        # Each test job finishes one file before its wall limit.
        done = files[: max(0, len(observed) - 1)]
        return test_results.Results(commit, env_digest, {f: {"outcome": "passed"} for f in done})

    monkeypatch.setattr(test_results, "recorded", recorded)
    results, jobs, _, _ = regression.record_at(
        "c0ffee",
        files,
        cache_dir=tmp_path,
        out=tmp_path,
        tree_at=tree_at,
        run_job=run,
        python=python,
    )
    # The probe is short, and the three two-hour jobs' budget is four 90-minute ones.
    assert [r.timeout_s for r in observed] == [regression.PROBE_TIMEOUT_S, *[5400] * 4]
    assert jobs == 4 and results.unrun(files) == files[4:]
    assert "--selection" in observed[1].argv
    # A shorter declared default wall bounds the window; off a spool a job runs two hours.
    walled = SessionSpec(
        "g", "/g", time_limit_s=7200, handover_lead_s=1800, scheduler={"default_wall": "75m"}
    )
    walled_env = RunEnvironment("session_spool", "p", spool="/g", sessions=(walled,))
    assert regression.job_timeout(walled_env) == 4500
    assert regression.job_timeout(RunEnvironment("local", "p")) == regression.JOB_TIMEOUT_S
    # Only the sessions gate jobs reach size them: a longer `work` session on another spool
    # never receives one, while an `any` session on its own spool does.
    longer = SessionSpec("other", "/o", role="work", time_limit_s=4 * 3600, handover_lead_s=600)
    both = RunEnvironment("session_spool", "p", spool="/g", sessions=(walled, longer))
    assert regression.job_timeout(both) == 4500
    shared = replace(longer, role="any")
    pooled = RunEnvironment("session_spool", "p", spool="/g", sessions=(walled, shared))
    assert regression.job_timeout(pooled) == regression.JOB_TIMEOUT_S
    assert regression.max_jobs(regression.JOB_TIMEOUT_S) == 3
