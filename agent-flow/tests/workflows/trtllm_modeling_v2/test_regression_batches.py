# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-model regression reviews are batched: one review takes every change to shared code landed.

A review per change to shared code re-ran the other models' tests once per
change. A regression review now covers, as it starts, every landed change no
earlier review covered (its batch). A ready one waits, while a task that
changes shared code runs, up to the window after the batch's first landing;
one runs at a time; one whose batch is empty closes with no turn. The rule
keeps its strength: every landed change is in some review's batch before the
run is delivered, a failed review covers nothing, and a change that lands
again on a retry is a new one.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)

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


def _core(satisfies=(_BOOT,), **kw):
    return g.Task(
        id="core",
        kind="assemble",
        goal="core",
        touches=[f"{_TARGET}/modeling.py"],
        satisfies=list(satisfies),
        **kw,
    )


def _review(tid, deps=(), status="pending", **kw):
    return g.Task(
        id=tid,
        kind="review",
        goal="cross-model regression",
        depends_on=list(deps),
        regression=True,
        status=status,
        **kw,
    )


def _errors(*tasks):
    return g.validate(g.Graph(tasks=list(tasks)), own_paths=_OWN)


# ------------------------------------------------------------------ the rule


def test_a_landed_change_joins_the_open_reviews_batch_with_no_edge():
    assert _errors(_shared("k1", "done"), _review("xmodel")) == []
    [error] = _errors(_shared("k1", "done"))
    assert "no cross-model regression review follows 1 task(s)" in error
    assert "Add one required `review` task with `regression: true`" in error
    assert "its batch (k1)" in error and "before the delivery tier" in error


def test_the_refusal_names_the_open_review_and_its_batch():
    [error] = _errors(_shared("k1", "done"), _shared("k2"), _review("xmodel"))
    assert ": k2." in error
    assert "Add k2 to the `depends_on` of the regression review `xmodel`" in error
    assert "Its batch so far, the changes landed that no review covered yet: k1" in error
    assert _errors(_shared("k1", "done"), _shared("k2"), _review("xmodel", ["k2"])) == []


def test_a_started_review_covers_its_batch_and_nothing_landed_after_it():
    graph = g.Graph(
        tasks=[
            _shared("k1", "done"),
            _shared("k2", "done"),
            _review("r0", status="done", covers=["k1@1"]),
        ]
    )
    assert g.regression_batch(graph, _OWN) == ["k2"]
    assert list(g.unreviewed_shared_changes(graph, _OWN)) == ["k2"]


def test_a_failed_review_covers_nothing():
    graph = g.Graph(tasks=[_shared("k1", "done"), _review("r0", status="failed", covers=["k1@1"])])
    assert g.regression_batch(graph, _OWN) == ["k1"]
    assert list(g.unreviewed_shared_changes(graph, _OWN)) == ["k1"]


def test_a_retry_that_lands_again_is_a_new_change():
    graph = g.Graph(
        tasks=[_shared("k1", "done", attempts=2), _review("r0", status="done", covers=["k1@1"])]
    )
    assert g.regression_batch(graph, _OWN) == ["k1"]
    assert list(g.unreviewed_shared_changes(graph, _OWN)) == ["k1"]


def test_a_failed_task_whose_commits_landed_joins_the_batch():
    graph = g.Graph(tasks=[_shared("k1", "failed")])
    assert g.regression_batch(graph, _OWN) == []
    assert g.regression_batch(graph, _OWN, {"k1": 1}) == ["k1"]
    retried = g.Graph(tasks=[_shared("k1", "failed", attempts=2)])
    assert g.regression_batch(retried, _OWN, {"k1": 1}) == []  # attempt 2 did not land


def test_a_review_recorded_before_batches_covers_what_it_depended_on_that_was_done():
    graph = g.Graph(
        tasks=[
            _shared("k1", "done"),
            _shared("k2", "done"),
            _review("r0", ["k1"], status="done"),
        ]
    )
    assert g.regression_batch(graph, _OWN) == ["k2"]


def test_a_revision_keeps_a_started_reviews_batch_and_a_retry_records_anew():
    current = g.Graph(
        tasks=[
            _shared("k1", "done"),
            _review("r0", status="done", covers=["k1@1"]),
            _review("r1", status="failed", covers=["k1@1"]),
        ]
    )
    proposed = g.Graph(
        tasks=[
            _shared("k1", "done"),
            _review("r0", status="done", covers=[]),
            _review("r1", covers=["k1@1"]),
            _review("r2", covers=["k1@1"]),
        ]
    )
    adopted, errors = g.apply_revision(current, proposed, "retry r1")
    assert errors == []
    by_id = adopted.by_id()
    assert by_id["r0"].covers == ["k1@1"]
    assert by_id["r1"].covers is None and by_id["r1"].status == "pending"
    assert by_id["r2"].covers is None


# ------------------------------------------------------------------- the loop


def _orchestrator(tmp_path, plan, work, failing=(), **kw):
    services = Services(
        plan=plan,
        work=work,
        check=lambda chk: SimpleNamespace(
            name=chk.gate, member=chk.member, passed=chk.key not in failing, detail=""
        ),
        free_gpus=lambda: 8,
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


def test_one_review_takes_every_change_that_lands_and_a_second_closes_with_no_turn(tmp_path):
    orchestrator, reviewed = None, []

    def work(task):
        if task.id == "k2":
            # k1 lands first; r1 is ready then but waits while k2 runs.
            _wait_for(lambda: orchestrator.graph.by_id()["k1"].status == "done")
            time.sleep(0.1)
            assert orchestrator.graph.by_id()["r1"].status == "pending"
            return "LANDED"
        if task.kind == "review":
            reviewed.append(task.id)
            return "APPROVE"
        return "LANDED" if task.kind == "kernel" else "UNIT_READY"

    plan = _plan(_graph(_shared("k1"), _shared("k2"), _core(), _review("r1", ["k1"]),
                        _review("r2", ["k2"])))  # fmt: skip
    orchestrator = _orchestrator(tmp_path, plan, work)
    assert orchestrator.run() == "delivered"
    by_id = orchestrator.graph.by_id()
    assert reviewed == ["r1"]
    assert by_id["r1"].covers == ["k1@1", "k2@1"]
    assert by_id["r2"].status == "done" and by_id["r2"].covers == []
    events = [*(e for t in plan.turns for e in t.events), *orchestrator.events]
    said = [e for e in events if e.startswith("task r2")]
    assert said == [
        "task r2 (review) closed with no turn: no change to shared code was left for it to "
        "review: r1 covered what it follows"
    ]


def test_past_the_window_a_review_starts_and_only_one_runs_at_a_time(tmp_path):
    orchestrator, order = None, []
    r1_running = threading.Event()

    def work(task):
        if task.id == "k2":
            assert r1_running.wait(_WAIT), "r1 waited for k2 past its window"
            return "LANDED"
        if task.id == "r1":
            r1_running.set()
            # k2 lands while r1 runs; r2 is ready then but waits for r1.
            _wait_for(lambda: orchestrator.graph.by_id()["k2"].status == "done")
            time.sleep(0.1)
            assert orchestrator.graph.by_id()["r2"].status == "pending"
            order.append("r1")
            return "APPROVE"
        if task.id == "r2":
            order.append("r2")
            return "APPROVE"
        return "LANDED" if task.kind == "kernel" else "UNIT_READY"

    plan = _plan(_graph(_shared("k1"), _shared("k2"), _core(), _review("r1", ["k1"]),
                        _review("r2", ["k2"])))  # fmt: skip
    orchestrator = _orchestrator(tmp_path, plan, work, regression_window_s=0)
    assert orchestrator.run() == "delivered"
    by_id = orchestrator.graph.by_id()
    assert order == ["r1", "r2"]
    assert by_id["r1"].covers == ["k1@1"] and by_id["r2"].covers == ["k2@1"]


def test_the_run_is_not_delivered_while_a_landed_change_is_in_no_reviews_batch(tmp_path):
    """A review that rejects covers nothing: the planner saying DONE does not deliver the run."""
    reviewed = []

    def work(task):
        if task.kind == "review":
            reviewed.append(task.id)
            return "REJECT" if len(reviewed) == 1 else "APPROVE"
        return "LANDED" if task.kind == "kernel" else "UNIT_READY"

    def retry(turn):
        if not any("no cross-model regression review covers k1" in e for e in turn.events):
            return PlannerAnswer("DONE", "the review's findings are not ours")
        assert "the batch the next one takes: k1" in turn.regression
        return _graph(_shared("k1", "done"), _core(status="done"), _review("r1"))

    plan = _plan(_graph(_shared("k1"), _core(), _review("r1", ["k1"])), retry, retry, retry)
    orchestrator = _orchestrator(tmp_path, plan, work)
    assert orchestrator.run() == "delivered"
    assert reviewed == ["r1", "r1"]
    assert orchestrator.graph.by_id()["r1"].covers == ["k1@1"]


def test_a_failed_task_whose_commits_landed_keeps_its_review_from_closing(tmp_path):
    """Its checks failed after its branch went in: the review has that change to read."""
    reviewed = []

    def work(task):
        if task.kind == "review":
            reviewed.append(task.id)
            return "APPROVE"
        return "LANDED" if task.kind == "kernel" else "UNIT_READY"

    def drop_the_edge(turn):
        if not any("k1 (kernel) LANDED; checks failed" in e for e in turn.events):
            return PlannerAnswer("DONE", "waiting")
        return _graph(
            _shared("k1", "failed", satisfies=[_BOOT]), _core(satisfies=[]), _review("r1")
        )

    plan = _plan(
        _graph(_shared("k1", satisfies=[_BOOT]), _core(satisfies=[]), _review("r1", ["k1"])),
        drop_the_edge,
        drop_the_edge,
        PlannerAnswer("ESCALATE", "k1's check fails"),
    )
    orchestrator = _orchestrator(tmp_path, plan, work, failing={_BOOT})
    orchestrator.run()
    assert reviewed == ["r1"]
    assert orchestrator.graph.by_id()["r1"].covers == ["k1@1"]


def test_the_planner_hears_where_the_reviews_stand(tmp_path):
    orchestrator = _orchestrator(tmp_path, _plan(), lambda task: "UNIT_READY")
    orchestrator.graph = g.Graph(
        tasks=[_shared("k1", "done"), _review("r0", status="running", covers=[]), _review("r1")]
    )
    orchestrator._landings["k1"] = (time.time() - 600, 1)
    said = orchestrator._regression_standing()
    assert "the batch the next one takes: k1, the first landed 10 min ago." in said
    assert "Running: r0; the others wait for it." in said
    assert "Not started: r1." in said and "30 min" in said
    without = GraphOrchestrator(
        Services(plan=_plan(), work=lambda t: "", check=lambda chk: None, free_gpus=lambda: 1),
        _CONTRACT,
        c.ContractStatus.load(tmp_path),
        tmp_path,
    )
    assert without._regression_standing() == ""


def test_a_failed_receipt_whose_guards_landed_joins_the_batch(tmp_path):
    """A turn that did not finish may still land commits (a failed receipt's guards)."""
    reviewed = []

    def work(task):
        if task.kind == "review":
            reviewed.append(task.id)
            return "APPROVE"
        return (
            "FAILED_RECEIPT\nits guard commits are integrated" if task.id == "k1" else "UNIT_READY"
        )

    def drop_the_edge(turn):
        if not any("task k1 (kernel) ended FAILED_RECEIPT" in e for e in turn.events):
            return PlannerAnswer("DONE", "waiting")
        return _graph(_shared("k1", "failed"), _core(), _review("r1"))

    plan = _plan(
        _graph(_shared("k1"), _core(), _review("r1", ["k1"])),
        drop_the_edge,
        drop_the_edge,
    )
    orchestrator = _orchestrator(tmp_path, plan, work)
    orchestrator.s.landed = lambda task: task.id == "k1"
    assert orchestrator.run() == "delivered"
    assert reviewed == ["r1"] and orchestrator.graph.by_id()["r1"].covers == ["k1@1"]


def test_the_stage_tells_a_failed_turn_that_landed_from_one_that_did_not(tmp_path):
    import json

    from agent_flow.workflows.trtllm_modeling_v2.run.integration import INTEGRATED_FILENAME
    from agent_flow.workflows.trtllm_modeling_v2.run.stages.planned import PlannedStage

    root = tmp_path / "worktrees"
    root.mkdir()

    class Stage(PlannedStage):
        def __init__(self, lands):
            self.workspace = tmp_path
            self._integration = SimpleNamespace(root=root)
            self.lands = lands

        def _work_task_turn(self, task):
            if self.lands:
                (root / INTEGRATED_FILENAME).write_text(json.dumps({task.id: "abc123"}))
            return "FAILED_RECEIPT\nthe entry refuses g=0"

        def _keep_receipt(self, task, decision):
            pass

        def _improver_task_ended(self, task, decision):
            pass

    task = _shared("k1")
    quiet = Stage(lands=False)
    quiet._work_task(task)
    assert not quiet._landed_unfinished(task)
    landed = Stage(lands=True)
    landed._work_task(task)
    assert landed._landed_unfinished(task)
    task.attempts = 2  # the next attempt has landed nothing yet
    assert not landed._landed_unfinished(task)
