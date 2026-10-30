# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness is delivered beside a running tune campaign, and recorded where it was delivered."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.infra.vcs import git_output
from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.journal import Journal
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    PlannerTurn,
    Services,
)
from agent_flow.workflows.trtllm_modeling_v2.run.stages.planned import PlannedStage

_BOOT = c.Check("core", "boot", "tp1", c.TIER_INNER)
_ACCURACY = c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER)
_ENVELOPE = c.Check("core", "envelope", "tp1", c.TIER_DELIVERY)
_CONTRACT = c.Contract(checks=(_BOOT, _ACCURACY, _ENVELOPE))
_KEYS = [_BOOT.key, _ACCURACY.key, _ENVELOPE.key]
_INNER = {"tune": [_BOOT.key, _ACCURACY.key]}


def _task(tid, kind, satisfies=(), deps=(), touches=None):
    return g.Task(
        id=tid,
        kind=kind,
        goal=f"do {tid}",
        depends_on=list(deps),
        touches=list(touches if touches is not None else [f"models/{tid}.py"]),
        gpus=1,
        est_minutes=10,
        satisfies=list(satisfies),
    )


def _plan_once(*tasks):
    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            return PlannerAnswer("GRAPH_UPDATED", "first plan", g.Graph(tasks=list(tasks)))
        return PlannerAnswer("DONE", "all green")

    plan.turns = turns  # type: ignore[attr-defined]
    return plan


class _Run:
    """A fake workflow: checks pass, the target is ``tree['marker']``, each state has a commit."""

    def __init__(self, plan, work, shas=None):
        self.tree = {"marker": "v1"}
        self.shas = shas or {"v1": "a" * 40, "v2": "b" * 40}
        self.ran: list[tuple[str, str]] = []
        self.named: list[str] = []
        self.logs: list[str] = []
        self.services = Services(
            plan=plan,
            work=work,
            check=self.check,
            free_gpus=lambda: 8,
            marker=lambda: self.tree["marker"],
            correct=self.correct,
            log=self.logs.append,
        )

    def check(self, chk):
        self.ran.append((chk.key, self.tree["marker"]))
        return SimpleNamespace(name=chk.gate, member=chk.member, passed=True, detail="")

    def correct(self, marker, evaluation):
        self.named.append(marker)
        return {"sha": self.shas[marker], "ref": "user/x/model-correct", "elapsed_s": 7200.0}


def _orchestrator(tmp_path, services, **kw):
    status = c.ContractStatus.load(tmp_path)
    return GraphOrchestrator(services, _CONTRACT, status, tmp_path, poll_s=0, **kw)


def test_the_delivery_tier_runs_beside_a_campaign_and_correctness_is_recorded_before_it_ends(
    tmp_path,
):
    recorded = threading.Event()
    seen: dict[str, bool] = {}

    def work(task):
        if task.kind == "tune":
            # The campaign is still running when correctness is delivered.
            seen["recorded during the campaign"] = recorded.wait(timeout=30)
            return "TUNED\ntuned: best branch `tune/x/best`"
        return "UNIT_READY"

    plan = _plan_once(_task("core", "assemble", _KEYS), _task("tune", "tune", touches=[]))
    run = _Run(plan, work)
    named = run.correct

    def correct(marker, evaluation):
        out = named(marker, evaluation)
        recorded.set()
        return out

    run.services.correct = correct
    orchestrator = _orchestrator(
        tmp_path, run.services, required_kinds=("tune",), start_after=_INNER
    )
    assert orchestrator.run() == "delivered"
    assert seen == {"recorded during the campaign": True}
    assert (_ENVELOPE.key, "v1") in run.ran
    [milestone] = c.milestones(tmp_path)
    assert (milestone.sha, milestone.marker, milestone.ref) == (
        "a" * 40,
        "v1",
        "user/x/model-correct",
    )
    assert milestone.checks == 3 and milestone.elapsed_s == 7200.0
    assert run.named == ["v1"]  # once per state of the target, however often it is evaluated
    assert any(turn.marker == "v1" for turn in plan.turns)


def test_full_support_tells_the_campaign_to_wrap_up_and_the_run_is_delivered(tmp_path):
    """The campaign gives way to full support: told to stop, it wraps up and nothing waits for it."""
    stopped = threading.Event()
    reasons: list[str] = []

    def work(task):
        if task.kind == "tune":
            # Runs until told to stop, as a campaign with hours of budget left would.
            assert stopped.wait(timeout=30)
            return "TUNED\ntuned: best branch `tune/x/best`"
        return "UNIT_READY"

    plan = _plan_once(_task("core", "assemble", _KEYS), _task("tune", "tune", touches=[]))
    run = _Run(plan, work)

    def stop(reason):
        reasons.append(reason)
        stopped.set()

    run.services.stop_campaign = stop
    orchestrator = _orchestrator(
        tmp_path, run.services, required_kinds=("tune",), start_after=_INNER
    )
    assert orchestrator.run() == "delivered"
    assert reasons == ["full support holds on the target as it is"]
    assert orchestrator.graph.by_id()["tune"].status == "done"
    # The campaign's report is journaled for the record; no planner turn had to read it.
    assert any("best branch `tune/x/best`" in line for line in run.logs)
    assert [m.marker for m in c.milestones(tmp_path)] == ["v1"]


def test_a_task_that_builds_on_the_campaign_is_refused():
    """The campaign's best branch is an artifact of the run: nothing depends on it or integrates it."""
    tune = _task("tune", "tune", touches=[])
    merge = _task("integrate-tune", "integrate", deps=["tune"])
    errors = g.validate(g.Graph(tasks=[tune, merge]))
    assert any("depends on 'tune', a `tune` task" in e for e in errors)
    soft = _task("feature", "feature")
    soft.depends_on_soft = ["tune"]
    assert any("depends on 'tune'" in e for e in g.validate(g.Graph(tasks=[tune, soft])))


def test_delivery_files_that_fail_hold_the_milestone_and_reach_the_planner(tmp_path):
    """Full support includes the files on disk: until they check out, nothing is recorded."""
    turns: list[PlannerTurn] = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            return PlannerAnswer(
                "GRAPH_UPDATED", "first plan", g.Graph(tasks=[_task("core", "assemble", _KEYS)])
            )
        if any("delivery files on disk do not check out" in e for e in turn.events):
            tasks = [*orchestrator.graph.tasks, _task("ci", "feature")]
            return PlannerAnswer("GRAPH_UPDATED", "write the CI test", g.Graph(tasks=tasks))
        return PlannerAnswer("DONE", "all green")

    def work(task):
        if task.id == "ci":
            run.tree["marker"] = "v2"  # the CI accuracy test lands on the target
        return "UNIT_READY"

    run = _Run(plan, work)
    read: list[str] = []

    def delivery_files():
        read.append(run.tree["marker"])
        return [] if run.tree["marker"] == "v2" else ["tests:accuracy-file: missing"]

    run.services.delivery_files = delivery_files
    orchestrator = _orchestrator(tmp_path, run.services)
    assert orchestrator.run() == "delivered"
    # Correctness was never recorded on the tree the files were missing from.
    assert [m.marker for m in c.milestones(tmp_path)] == ["v2"]
    assert run.named == ["v2"]
    said = [e for t in turns for e in t.events if "do not check out" in e]
    assert len(said) == 1 and "tests:accuracy-file: missing" in said[0]
    # Read once per state of the target, however often the run looked.
    assert sorted(set(read)) == ["v1", "v2"] and read.count("v1") == 1


def test_no_stall_is_judged_while_work_is_still_running(tmp_path):
    """Revisions while a long task runs are no stall: its outcome may yet be the progress."""
    release = threading.Event()

    def work(task):
        if task.id == "long":
            assert release.wait(timeout=30)
            return "UNIT_READY"
        return "BLOCKED"  # a short one fails at once and wakes the planner

    def short(i):
        t = _task(f"s{i}", "investigate")
        t.priority = "nice_to_have"
        return t

    turns: list[PlannerTurn] = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            tasks = [_task("long", "assemble", _KEYS), short(0)]
            return PlannerAnswer("GRAPH_UPDATED", "plan", g.Graph(tasks=tasks))
        if len(turns) >= 5:
            release.set()
            return PlannerAnswer("DONE", "waiting for long")
        # Each failure gets a revision that tries something else, which fails too.
        tasks = [*orchestrator.graph.tasks, short(len(turns))]
        return PlannerAnswer("GRAPH_UPDATED", "try another", g.Graph(tasks=tasks))

    run = _Run(plan, work)
    orchestrator = _orchestrator(tmp_path, run.services, revision_cap=2)
    assert orchestrator.run() == "delivered"
    assert len(turns) >= 5  # four revisions, past the cap of 2, while the long task ran


def _holding(*checks):
    return c.Evaluation(passed=tuple(checks), failed=(), missing=())


def test_a_later_failure_leaves_the_milestone_and_is_journaled_once(tmp_path):
    run = _Run(_plan_once(), lambda task: "UNIT_READY")
    orchestrator = _orchestrator(tmp_path, run.services)
    orchestrator._correctness("v1", _holding(_BOOT, _ACCURACY, _ENVELOPE))
    verdict = c.Verdict("core", "boot", "tp1", passed=False, marker="v2")
    failing = c.Evaluation(passed=(_ACCURACY,), failed=((_BOOT, verdict),), missing=(_ENVELOPE,))
    orchestrator._correctness("v2", failing)
    orchestrator._correctness("v2", failing)
    assert run.named == ["v1"]
    assert [m.marker for m in c.milestones(tmp_path)] == ["v1"]
    assert run.logs == [
        f"correctness stays delivered at {'a' * 10}: on the target as it is now "
        "core:boot@tp1 failed"
    ]
    assert c.latest_milestone(tmp_path).describe("v2", failing) == (
        f"Correctness delivered at {'a' * 10} (`user/x/model-correct`), 2.0 h after the run "
        "started; the target has moved since, and on it 1 required checks failed and 1 are "
        "stale or missing. It moves forward once the contract holds on a later commit."
    )


def test_a_state_no_commit_holds_is_asked_again_and_an_unknown_one_never(tmp_path):
    answers = [None, {"sha": "c" * 40}]
    asked: list[str] = []

    def correct(marker, evaluation):
        asked.append(marker)
        return answers[len(asked) - 1]

    run = _Run(_plan_once(), lambda task: "UNIT_READY")
    run.services.correct = correct
    orchestrator = _orchestrator(tmp_path, run.services)
    holding = _holding(_BOOT, _ACCURACY, _ENVELOPE)
    orchestrator._correctness("", holding)  # not established: names no tree
    orchestrator._correctness("v1", c.Evaluation(passed=(), failed=(), missing=()))
    assert asked == [] and c.milestones(tmp_path) == []
    orchestrator._correctness("v1", holding)  # uncommitted, say: nothing recorded
    assert c.milestones(tmp_path) == []
    orchestrator._correctness("v1", holding)  # committed as it was measured
    [milestone] = c.milestones(tmp_path)
    assert asked == ["v1", "v1"] and milestone.sha == "c" * 40 and milestone.ref == ""
    assert milestone.describe() == f"Correctness delivered at {'c' * 10}."


def test_a_resumed_run_keeps_the_milestone_it_recorded(tmp_path):
    c.record_milestone(tmp_path, c.Milestone(sha="a" * 40, marker="v1"))
    run = _Run(_plan_once(), lambda task: "UNIT_READY")
    orchestrator = _orchestrator(tmp_path, run.services)
    orchestrator._correctness("v1", _holding(_BOOT, _ACCURACY, _ENVELOPE))
    assert run.named == [] and len(c.milestones(tmp_path)) == 1
    standing = c.standing(_holding(_BOOT), tmp_path, "v1")
    assert standing.endswith(f"Correctness delivered at {'a' * 10}; the target is still there.")
    assert c.standing(_holding(_BOOT), tmp_path, "").endswith(f"at {'a' * 10}.")  # not established
    assert c.standing(_holding(_BOOT), tmp_path / "elsewhere") == "1 passed, 0 failed, 0 missing"


# ------------------------------------------------------- naming the commit


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=T", "-c", "user.email=t@example.invalid", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class _Marking(PlannedStage):
    """What naming the correct commit uses of the workflow: the checkout, the journal, the marker."""

    def __init__(self, workspace: Path, repo: Path):
        self.workspace = workspace
        self.task_data = {"trtllm_repo_path": str(repo)}
        self.journal = Journal(path=workspace / "journal.md")
        self.journal.open(slug="t")
        self._git = git_output
        self._planned_state = SimpleNamespace(branch="trtllm-modeling-v2/t")
        self.marker = "v1"

    def _target_marker(self):
        return self.marker


def _checkout(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "tensorrt_llm" / "models").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "user/x/model-mv2", str(repo))
    (repo / "tensorrt_llm" / "models" / "x.py").write_text("X = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "x")
    return repo


def _journal_lines(stage: _Marking, what: str) -> list[str]:
    return [line for line in stage.journal.path.read_text().splitlines() if f"**{what}" in line]


def test_the_commit_correctness_holds_on_is_named_by_a_branch_and_a_journal_line(tmp_path):
    repo = _checkout(tmp_path)
    stage = _Marking(tmp_path / "ws", repo)
    head = _git(repo, "rev-parse", "HEAD")
    # A file no gate reads does not keep the commit from holding what the checks measured.
    (repo / "tests").mkdir()
    (repo / "tests" / "notes.txt").write_text("scratch\n")
    named = stage._mark_correct("v1", _holding(_BOOT, _ACCURACY, _ENVELOPE))
    assert named["sha"] == head and named["ref"] == "user/x/model-mv2-correct"
    assert 0 <= named["elapsed_s"] < 600
    assert _git(repo, "rev-parse", "user/x/model-mv2-correct") == head
    [line] = _journal_lines(stage, "CORRECTNESS DELIVERED")
    assert f"**CORRECTNESS DELIVERED {head}**" in line
    assert "h after the run started" in line and "all 3 required checks" in line
    assert "branch `user/x/model-mv2-correct`" in line

    # A later commit the contract holds on moves the branch forward.
    c.record_milestone(stage.workspace, c.Milestone(sha=head, marker="v1"))
    (repo / "tensorrt_llm" / "models" / "x.py").write_text("X = 2\n")
    _git(repo, "commit", "-q", "-am", "x2")
    stage.marker = "v2"
    later = stage._mark_correct("v2", _holding(_BOOT))
    assert _git(repo, "rev-parse", "user/x/model-mv2-correct") == later["sha"] != head
    assert f"moved forward from {head[:10]}" in _journal_lines(stage, "CORRECTNESS DELIVERED")[-1]


def test_no_commit_is_named_for_a_moved_target(tmp_path):
    repo = _checkout(tmp_path)
    stage = _Marking(tmp_path / "ws", repo)
    holding = _holding(_BOOT)
    stage.marker = "v2"  # an integration landed after the checks were read
    assert stage._mark_correct("v1", holding) is None
    assert _journal_lines(stage, "CORRECTNESS DELIVERED") == []
    assert git_output(repo, "rev-parse", "--verify", "--quiet", "user/x/model-mv2-correct") is None
    # The checks measured the head's snapshot: an uncommitted edit is not in what they held on.
    stage.marker = "v1"
    (repo / "tensorrt_llm" / "models" / "x.py").write_text("X = 3\n")
    assert stage._mark_correct("v1", holding)["sha"] == _git(repo, "rev-parse", "HEAD")


def test_a_detached_checkout_is_named_by_the_runs_own_branch(tmp_path):
    repo = _checkout(tmp_path)
    _git(repo, "checkout", "-q", "--detach")
    stage = _Marking(tmp_path / "ws", repo)
    named = stage._mark_correct("v1", _holding(_BOOT))
    assert named["ref"] == "trtllm-modeling-v2/t-correct"
    assert _git(repo, "rev-parse", "trtllm-modeling-v2/t-correct") == named["sha"]


def test_the_planners_request_says_where_correctness_stands(tmp_path):
    class Planning(_Marking):
        task_path = Path("task.yaml")
        progress_path = Path("progress.yaml")

        def _dispatch_task(self, role, unit, what, prompt):
            self.prompts.append(prompt)
            return "DONE"

    repo = _checkout(tmp_path)
    stage = Planning(tmp_path / "ws", repo)
    stage.prompts = []
    c.record_milestone(stage.workspace, c.Milestone(sha="a" * 40, marker="v1", ref="b-correct"))
    turn = PlannerTurn(c.evaluate(_CONTRACT, []), [], marker="v2")
    stage._planner_turn(turn, _CONTRACT)
    standing = stage.prompts[-1].split("## Contract standing", 1)[1]
    assert f"Correctness delivered at {'a' * 10} (`b-correct`); the target has moved since" in (
        standing
    )


def test_the_journal_knows_when_the_run_opened(tmp_path):
    journal = Journal(path=tmp_path / "journal.md")
    assert journal.opened_at() is None
    journal.open(slug="t")
    opened = journal.opened_at()
    journal.open(slug="t")  # a resume appends; the run's start stays
    journal.event("run opened", "not the first")
    assert journal.opened_at() == opened and opened.tzinfo is not None


def test_a_milestone_round_trips_through_its_file(tmp_path):
    first = c.Milestone(sha="a" * 40, marker="v1", ref="r", at=1.0, elapsed_s=None, checks=3)
    c.record_milestone(tmp_path, first)
    line = json.loads((tmp_path / c.MILESTONE_FILENAME).read_text())
    line["added_later"] = True  # a field a later flow writes is not one this one reads
    (tmp_path / c.MILESTONE_FILENAME).write_text(json.dumps(line) + "\n")
    assert c.milestones(tmp_path) == [first]


# ------------------------------------------------------------- the whole run


def test_a_planned_run_records_correctness_before_its_campaign_ends(
    tmp_path, task_file, monkeypatch
):
    import agent_flow.workflows.trtllm_modeling_v2.tune.stage as stage

    from .test_workflow import Script, _install, _plan_with_tune, _state, _with_perf, _workflow

    seen: list[list[c.Milestone]] = []

    def fake_tune(wf, state, base, attempt=1, should_stop=lambda: False):
        seen.append(c.milestones(wf.workspace))
        state.tune.update(phase="done", base_commit=base)
        return "TUNED", "tuned from c0ffee1: best branch `tune/x/best`"

    monkeypatch.setattr(stage, "run_tune_task", fake_tune)
    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script(None, planner=_plan_with_tune))
    with workflow:
        workflow.run(_with_perf(task_file, enabled=True))
    state = _state(tmp_path)
    assert state.verdict == "tuned"
    [[milestone]] = seen  # recorded once, before the campaign had ended
    # The fake checkout has no branch checked out: the run's own branch name stands in.
    assert (milestone.sha, milestone.ref) == ("c0ffee1", f"{state.branch}-correct")
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "**CORRECTNESS DELIVERED c0ffee1**" in journal
    assert "Correctness delivered at c0ffee1" in journal.split("planned run: delivered", 1)[1]


def test_the_runs_clock_starts_when_it_goes_past_preflight_not_at_a_dry_run(tmp_path):
    """A dry run opened the journal 26 minutes before the run did, and every milestone was late."""
    journal = Journal(path=tmp_path / "journal.md")
    journal.open(slug="t")  # a --dry-run
    opened = journal.opened_at()
    time.sleep(1.1)  # the journal stamps seconds
    journal.open(slug="t")  # the real run
    journal.mark_started()
    started = journal.opened_at()
    assert started is not None and started > opened
    journal.mark_started()  # a resume: the clock stays
    assert journal.opened_at() == started
