# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 orchestrator behavior and regression checks."""

from __future__ import annotations

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
from agent_flow.workflows.trtllm_modeling_v2.run.validation import test_jobs
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


_STANDING = c.Contract(
    checks=(
        c.Check("core", "boot", "tp1", c.TIER_INNER),
        c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER),
        c.Check("mtp", "accuracy[gsm8k]", "tp1", c.TIER_INNER),
        c.Check("mtp", "acceptance", "tp1", c.TIER_PROOF),
    ),
    confirm="mtp:accuracy[gsm8k]@tp1",
)
_CORE = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1"]
_MTP = ["mtp:accuracy[gsm8k]@tp1", "mtp:acceptance@tp1"]


def _moving_target(plan, *, failing=lambda key, marker: False):
    """Services whose target changes each time a task's work ends; ``ran`` is (check, state) pairs."""
    landed = []
    ran = []

    def work(task):
        landed.append(task.id)
        return "UNIT_READY"

    def check(chk):
        marker = f"m{len(landed)}"
        ran.append((chk.key, marker))
        return SimpleNamespace(name=chk.gate, member=chk.member, detail="", tree_state="",
                               passed=not failing(chk.key, marker))  # fmt: skip

    services = Services(
        plan=plan, work=work, check=check, free_gpus=lambda: 4, marker=lambda: f"m{len(landed)}"
    )
    services.ran = ran  # type: ignore[attr-defined]
    return services


def _standing_run(tmp_path, services) -> str:
    status = c.ContractStatus.load(tmp_path)
    return GraphOrchestrator(services, _STANDING, status, tmp_path, poll_s=0).run()


def test_a_check_that_passed_stands_when_a_later_task_lands(tmp_path):
    plan = _plan_once(
        _task("core", "assemble", _CORE), _task("mtp", "feature", _MTP, deps=["core"])
    )
    services = _moving_target(plan)
    assert _standing_run(tmp_path, services) == "delivered"
    # Each check ran once, where its own task landed. The core's were not run again on
    # the state mtp left, and mtp's own checks, which ran on the final tree, confirm it.
    assert sorted(services.ran) == sorted([(k, "m1") for k in _CORE] + [(k, "m2") for k in _MTP])


def test_the_final_tree_is_confirmed_by_one_check_when_none_ran_on_it(tmp_path):
    plan = _plan_once(
        _task("core", "assemble", _CORE),
        _task("mtp", "feature", _MTP, deps=["core"]),
        _task("tidy", "kernel", deps=["mtp"], touches=["catalog/k.py"]),
    )
    services = _moving_target(plan)
    assert _standing_run(tmp_path, services) == "delivered"
    # `tidy` claims no check and changed the target last: one check runs on what it left.
    assert services.ran[-1] == ("mtp:accuracy[gsm8k]@tp1", "m3")
    assert sorted(services.ran[:-1]) == sorted(
        [(k, "m1") for k in _CORE] + [(k, "m2") for k in _MTP]
    )


def test_a_failure_on_the_final_tree_reaches_the_planner(tmp_path):
    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            tasks = [
                _task("core", "assemble", _CORE),
                _task("mtp", "feature", _MTP, deps=["core"]),
                _task("tidy", "kernel", deps=["mtp"], touches=["catalog/k.py"]),
            ]
            return PlannerAnswer("GRAPH_UPDATED", "plan", g.Graph(tasks=tasks))
        if any(
            f"the check of the final tree failed: {_STANDING.confirm}" in e for e in turn.events
        ):
            return PlannerAnswer("CEILING", "the last change broke the drafter")
        return PlannerAnswer("DONE", "all green")

    services = _moving_target(plan, failing=lambda key, marker: marker == "m3")
    assert _standing_run(tmp_path, services) == "ceiling"
    assert services.ran[-1] == (_STANDING.confirm, "m3")


def test_a_pass_stands_and_a_failure_counts_where_it_was_measured(tmp_path):
    boot, benchmark = _STANDING.checks[:2]
    contract = c.Contract(checks=(boot, benchmark), confirm=benchmark.key)
    status = c.ContractStatus.load(tmp_path)

    def result(check, passed):
        return SimpleNamespace(name=check.gate, member=check.member, passed=passed, detail="")

    status.record("core", result(boot, True), marker="m1")
    status.record("core", result(benchmark, False), marker="m1")
    here = c.evaluate(contract, status, marker="m1")
    assert [x.key for x in here.passed] == [boot.key]
    assert [x.key for x, _ in here.failed] == [benchmark.key]
    # On the state a later change left, the pass stands and the failure is due again.
    later = c.evaluate(contract, status, marker="m2")
    assert [x.key for x in later.passed] == [boot.key]
    assert not later.failed and [x.key for x in later.missing] == [benchmark.key]
    # Standing is not having run there: only m1 was measured.
    assert c.confirmed(contract, status, "m1") and not c.confirmed(contract, status, "m2")
    assert c.passed_here(contract, status, boot, "m1")
    assert not c.passed_here(contract, status, boot, "m2")
    # The delivery report says where the standing pass was measured.
    table = c.measured_where(later, status, "m2", since=lambda tree: "3")
    assert "| core:boot@tp1 | unknown | 3 |" in table
    assert "0, the final tree" in c.measured_where(here, status, "m1")


def test_a_failed_check_runs_again_with_its_repair_and_nothing_else_does(tmp_path):
    contract = c.Contract(checks=_STANDING.checks[:2], confirm=_CORE[1])
    repaired = []

    def plan(turn):
        if any("checks failed: core:boot@tp1" in e for e in turn.events):
            repaired.append(turn)
            graph = g.Graph.load(tmp_path / g.GRAPH_FILENAME)
            graph.tasks.append(_task("fix", "assemble", _CORE[:1], touches=["models/core/"]))
            return PlannerAnswer("GRAPH_UPDATED", "repair", graph)
        if not repaired:
            core = _task("core", "assemble", _CORE)
            return PlannerAnswer("GRAPH_UPDATED", "plan", g.Graph(tasks=[core]))
        return PlannerAnswer("DONE", "all green")

    services = _moving_target(plan, failing=lambda key, marker: (key, marker) == (_CORE[0], "m1"))
    status = c.ContractStatus.load(tmp_path)
    assert GraphOrchestrator(services, contract, status, tmp_path, poll_s=0).run() == "delivered"
    # The repair claims the boot, which runs on what it left. The benchmark that passed
    # beside the failure stands, and the boot's pass confirms the final tree.
    assert sorted(services.ran) == [(_CORE[1], "m1"), (_CORE[0], "m1"), (_CORE[0], "m2")]


@pytest.mark.parametrize(
    "decision, retried",
    [("raised TimeoutError: no answer from the driver", True), ("BLOCKED", False)],
)
def test_only_an_infrastructure_failure_is_retried_and_only_once(tmp_path, decision, retried):
    def failing(orchestrator):
        task = _task("core", "assemble", touches=[])
        task.status = "running"
        orchestrator.graph = g.Graph(tasks=[task])
        orchestrator._finish(task, decision, [])
        return task.status

    assert failing(_orchestrator_orchestrator(tmp_path, _services(_plan_once()))) == (
        "pending" if retried else "failed"
    )
    # The count is kept with the planner's events: a restarted process does not retry again.
    assert failing(_orchestrator_orchestrator(tmp_path, _services(_plan_once()))) == "failed"


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


def test_test_jobs_fit_the_gate_sessions_and_keep_the_test_budget(tmp_path, monkeypatch):
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

    monkeypatch.setattr(test_jobs, "select_command_executor", lambda *args, **kwargs: Executor())
    wf = SimpleNamespace(task_data={"trtllm_repo_path": str(tmp_path)}, _gate_env=lambda: env)
    tree_at, run, python = test_jobs.job_runner(wf, "delivery-tests")
    files = [f"tests/test_{i}.py" for i in range(6)]

    def recorded(cache, commit, env_digest=""):
        # Each test job finishes one file before its wall limit.
        done = files[: max(0, len(observed) - 1)]
        return test_results.Results(commit, env_digest, {f: {"outcome": "passed"} for f in done})

    monkeypatch.setattr(test_results, "recorded", recorded)
    results, jobs, _, _ = test_jobs.record_at(
        "c0ffee",
        files,
        cache_dir=tmp_path,
        out=tmp_path,
        tree_at=tree_at,
        run_job=run,
        python=python,
    )
    # The probe is short, and the three two-hour jobs' budget is four 90-minute ones.
    assert [r.timeout_s for r in observed] == [test_jobs.PROBE_TIMEOUT_S, *[5400] * 4]
    assert jobs == 4 and results.unrun(files) == files[4:]
    assert "--selection" in observed[1].argv
    # A shorter declared default wall bounds the window; off a spool a job runs two hours.
    walled = SessionSpec(
        "g", "/g", time_limit_s=7200, handover_lead_s=1800, scheduler={"default_wall": "75m"}
    )
    walled_env = RunEnvironment("session_spool", "p", spool="/g", sessions=(walled,))
    assert test_jobs.job_timeout(walled_env) == 4500
    assert test_jobs.job_timeout(RunEnvironment("local", "p")) == test_jobs.JOB_TIMEOUT_S
    # Only the sessions gate jobs reach size them: a longer `work` session on another spool
    # never receives one, while an `any` session on its own spool does.
    longer = SessionSpec("other", "/o", role="work", time_limit_s=4 * 3600, handover_lead_s=600)
    both = RunEnvironment("session_spool", "p", spool="/g", sessions=(walled, longer))
    assert test_jobs.job_timeout(both) == 4500
    shared = replace(longer, role="any")
    pooled = RunEnvironment("session_spool", "p", spool="/g", sessions=(walled, shared))
    assert test_jobs.job_timeout(pooled) == test_jobs.JOB_TIMEOUT_S
    assert test_jobs.max_jobs(test_jobs.JOB_TIMEOUT_S) == 3
