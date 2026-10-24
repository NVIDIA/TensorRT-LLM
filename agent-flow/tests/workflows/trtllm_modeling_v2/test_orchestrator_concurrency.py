# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The loop keeps moving while the planner thinks and while checks run.

Every fake that waits for another thread to reach a point waits on an event
with a deadline, so a loop that serializes what should overlap fails the test
instead of hanging it.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator as orchestrator_mod
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    EVENTS_FILENAME,
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)

#: How long a fake waits for another thread before the test counts it as never.
_WAIT = 10.0
_CONTRACT = c.Contract(
    checks=(
        c.Check("core", "boot", "tp1", c.TIER_INNER),
        c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER),
    )
)
_KEYS = [check.key for check in _CONTRACT.checks]


def _task(tid, kind="feature", satisfies=(), deps=(), gpus=1):
    return g.Task(
        id=tid,
        kind=kind,
        goal=f"do {tid}",
        depends_on=list(deps),
        touches=[f"models/{tid}.py"],
        gpus=gpus,
        est_minutes=10,
        satisfies=list(satisfies),
    )


def _result(check, passed=True):
    return SimpleNamespace(
        name=check.gate, member=check.member, passed=passed, detail="", tree_state=""
    )


def _services(plan, work=None, check=None, gpus=8):
    return Services(
        plan=plan,
        work=work or (lambda task: "UNIT_READY"),
        check=check or _result,
        free_gpus=lambda: gpus,
    )


def _orchestrator(tmp_path, services, contract=_CONTRACT, **kw):
    kw.setdefault("poll_s", 0.01)
    status = c.ContractStatus.load(tmp_path)
    return GraphOrchestrator(services, contract, status, tmp_path, **kw)


def _left_unread(tmp_path, *events):
    """Events a dead process left for the planner: the next one reads them first."""
    state = {"events": list(events), "unread_reports": False}
    (tmp_path / EVENTS_FILENAME).write_text(json.dumps(state), encoding="utf-8")


# ---------------------------------------------------------------- planner


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

    assert _orchestrator(tmp_path, _services(plan, work)).run() == "delivered"
    assert seen["b started during the turn"]
    # What arrived during the second turn went to the third, not to the void.
    assert any("task c (feature) reports:" in e for e in turns[1].events)
    assert not any(e.startswith("task a ") for e in turns[1].events)
    assert any(e.startswith("task a (feature) UNIT_READY") for e in turns[2].events)


def test_the_run_is_not_delivered_while_a_planner_turn_may_still_add_work(tmp_path):
    turns, ran = [], []
    in_turn_two = threading.Event()
    orchestrator = None

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(tasks=[_task("core", "assemble", _KEYS), _task("side")])
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        if len(turns) == 2:
            in_turn_two.set()
            deadline = time.monotonic() + _WAIT
            while orchestrator.graph.by_id()["core"].status != "done":
                assert time.monotonic() < deadline
                time.sleep(0.01)
            time.sleep(0.2)  # the contract holds all this while
            graph = g.Graph(tasks=[_task("core", "assemble", _KEYS), _task("side"), _task("extra")])
            return PlannerAnswer("GRAPH_UPDATED", "one more thing", graph)
        return PlannerAnswer("DONE", "all green")

    def work(task):
        ran.append(task.id)
        if task.id == "side":
            return "UNIT_READY\nthe tokenizer needs a chat template"  # wakes the planner
        if task.id == "core":
            in_turn_two.wait(_WAIT)
        return "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, _services(plan, work))
    assert orchestrator.run() == "delivered"
    assert sorted(ran) == ["core", "extra", "side"]


def test_with_nothing_to_run_the_loop_waits_for_the_planner_instead_of_spinning(tmp_path):
    looks = []

    def plan(turn):
        time.sleep(0.3)
        return PlannerAnswer("ESCALATE", "cannot")

    services = _services(plan)
    services.free_gpus = lambda: looks.append(1) or 8
    assert _orchestrator(tmp_path, services, poll_s=0.001).run() == "escalated"
    assert len(looks) <= 3


def test_ready_work_stuck_during_a_planner_turn_is_not_counted_against_the_planner(tmp_path):
    g.Graph(tasks=[_task("big", gpus=8)]).save(tmp_path / g.GRAPH_FILENAME)
    _left_unread(tmp_path, "operator: big needs a node of its own")
    turns = []

    def plan(turn):
        turns.append(turn)
        time.sleep(0.2)  # the loop looks many times while big cannot start
        return PlannerAnswer("ESCALATE", "no such node")

    orchestrator = _orchestrator(tmp_path, _services(plan, gpus=4))
    assert orchestrator.run() == "escalated"
    # Once, before the turn: the loop does not tell (or count) it again while the turn runs.
    assert sum("cannot start" in e for e in turns[0].events) == 1
    assert orchestrator.revisions_since_progress == 1
    assert not [e for e in orchestrator.events if "cannot start" in e]


def test_the_critic_holds_nothing_and_its_words_reach_the_next_turn(tmp_path):
    """The tasks a reshaping revision adds start while the critic reads it."""
    g.Graph(tasks=[_task("w"), _task("x", satisfies=_KEYS, deps=["w"])]).save(
        tmp_path / g.GRAPH_FILENAME
    )
    _left_unread(tmp_path, "operator: there are three more ops to onboard")
    new_started, critique_read, turns, seen = threading.Event(), threading.Event(), [], {}

    def plan(turn):
        turns.append(turn)
        if any("critic (advisory) on this graph: REJECT: n2" in e for e in turn.events):
            critique_read.set()
        if len(turns) == 1:
            graph = g.Graph(
                tasks=[_task("w"), _task("x", satisfies=_KEYS, deps=["w"]), _task("n1"),
                       _task("n2"), _task("n3")]
            )  # fmt: skip
            return PlannerAnswer("GRAPH_UPDATED", "three more ops", graph)
        return PlannerAnswer("DONE", "all green")

    def critique(graph, reason):
        seen["new tasks started while the critic read"] = new_started.wait(_WAIT)
        return "REJECT: n2 and n3 share a forward"

    def work(task):
        if task.id.startswith("n"):
            new_started.set()
        if task.id == "x":
            seen["a later turn read the critique"] = critique_read.wait(_WAIT)
        return "UNIT_READY"

    services = _services(plan, work)
    services.critique = critique
    assert _orchestrator(tmp_path, services).run() == "delivered"
    assert seen["new tasks started while the critic read"]
    assert seen["a later turn read the critique"]


def test_a_planner_turn_runs_while_the_critic_reads(tmp_path):
    """A failure during the critic's read wakes the planner at once, not after the read."""
    critic_reading, planner_woken, turns, seen = threading.Event(), threading.Event(), [], {}

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(tasks=[_task("a"), _task("b", satisfies=_KEYS)])
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        planner_woken.set()
        return PlannerAnswer("DONE", "b will hold")

    def critique(graph, reason):
        critic_reading.set()
        seen["planner woke during the read"] = planner_woken.wait(_WAIT)
        return ""

    def work(task):
        if task.id == "a":
            critic_reading.wait(_WAIT)
            return "BLOCKED"
        planner_woken.wait(_WAIT)
        return "UNIT_READY"

    services = _services(plan, work)
    services.critique = critique
    assert _orchestrator(tmp_path, services).run() == "delivered"
    assert seen["planner woke during the read"]
    assert "task a (feature) ended BLOCKED" in turns[1].events


# ---------------------------------------------------------------- wake-ups


def test_a_success_during_a_planner_turn_waits_for_the_window(tmp_path):
    turns, seen = [], {}
    in_turn_two, second_turn_over = threading.Event(), threading.Event()
    orchestrator = None

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(tasks=[_task("a"), _task("b"), _task("slow", satisfies=_KEYS)])
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        in_turn_two.set()
        deadline = time.monotonic() + _WAIT
        while not any(e.startswith("task b ") for e in orchestrator.events):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        second_turn_over.set()
        return PlannerAnswer("DONE", "all green")

    def work(task):
        if task.id == "a":
            return "UNIT_READY\nthe vision tower is replicated per rank"  # wakes the planner
        if task.id == "b":
            in_turn_two.wait(_WAIT)  # b's success arrives while the planner reads a's report
            return "UNIT_READY"
        second_turn_over.wait(_WAIT)
        time.sleep(0.3)  # long enough for the loop to wake the planner, were it going to
        seen["turns while slow ran"] = len(turns)
        return "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, _services(plan, work), event_window_s=60)
    assert orchestrator.run() == "delivered"
    assert not any(e.startswith("task b ") for e in turns[1].events)
    assert seen["turns while slow ran"] == 2 and len(turns) == 2


def test_a_failed_task_wakes_the_planner_inside_the_window(tmp_path):
    turns, seen = [], {}
    woken = threading.Event()

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(tasks=[_task("a"), _task("slow", satisfies=_KEYS)])
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        woken.set()
        return PlannerAnswer("DONE", "the rest holds")

    def work(task):
        if task.id == "a":
            return "BLOCKED"
        seen["planner woke while slow ran"] = woken.wait(_WAIT)
        return "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, _services(plan, work), event_window_s=60)
    assert orchestrator.run() == "delivered"
    assert seen["planner woke while slow ran"]
    assert "task a (feature) ended BLOCKED" in turns[1].events


def test_with_only_a_campaign_running_the_planner_reads_events_at_once(tmp_path):
    """A campaign runs for hours: a success beside it is not held for the event window."""
    turns, seen = [], {}
    woken = threading.Event()

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(tasks=[_task("a"), _task("campaign", kind="tune", satisfies=_KEYS)])
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        woken.set()
        return PlannerAnswer("DONE", "the campaign will finish")

    def work(task):
        if task.id == "a":
            return "UNIT_READY"
        seen["planner woke while the campaign ran"] = woken.wait(_WAIT)
        return "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, _services(plan, work), event_window_s=600)
    assert orchestrator.run() == "delivered"
    assert seen["planner woke while the campaign ran"]
    assert any(e.startswith("task a ") for e in turns[1].events)


# --------------------------------------------------- events a revision answered


def _wait_until(predicate):
    deadline = time.monotonic() + _WAIT
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def test_a_failure_the_revision_retried_does_not_wake_the_planner_again(tmp_path):
    """A failure the previous revision already retried is not news to the planner."""
    turns, logs, orchestrator = [], [], None
    in_turn_two = threading.Event()

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(tasks=[_task("r"), _task("a"), _task("slow", satisfies=_KEYS)])
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        if len(turns) == 2:
            in_turn_two.set()
            # The turn reads the graph on disk after a failed, and retries it.
            _wait_until(lambda: orchestrator.graph.by_id()["a"].status == "failed")
            graph = copy.deepcopy(orchestrator.graph)
            graph.by_id()["a"].status = "pending"
            return PlannerAnswer("GRAPH_UPDATED", "retry a", graph)
        return PlannerAnswer("DONE", "all green")

    def work(task):
        if task.id == "r":
            return "UNIT_READY\nthe tokenizer needs a chat template"  # wakes turn two
        if task.id == "a" and task.attempts == 1:
            in_turn_two.wait(_WAIT)
            return "BLOCKED"
        if task.id == "slow":
            _wait_until(lambda: orchestrator.graph.by_id()["a"].status == "done")
        return "UNIT_READY"

    services = _services(plan, work)
    services.log = logs.append
    orchestrator = _orchestrator(tmp_path, services, event_window_s=60)
    assert orchestrator.run() == "delivered"
    assert not any("task a (feature) ended BLOCKED" in e for t in turns[2:] for e in t.events)
    assert any("already answered 1 event(s) of its turn" in line for line in logs)
    assert orchestrator.graph.by_id()["a"].attempts == 2


def test_a_success_the_revision_wrote_as_done_does_not_wake_the_planner(tmp_path):
    """Events the previous revision already handled do not wake the planner."""
    turns, orchestrator, seen = [], None, {}
    in_turn_two, turn_two_over = threading.Event(), threading.Event()

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(tasks=[_task("r"), _task("c"), _task("slow", satisfies=_KEYS)])
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        in_turn_two.set()
        _wait_until(lambda: orchestrator.graph.by_id()["c"].status == "done")
        graph = copy.deepcopy(orchestrator.graph)
        turn_two_over.set()
        return PlannerAnswer("GRAPH_UPDATED", "c landed; nothing changes", graph)

    def work(task):
        if task.id == "r":
            return "UNIT_READY\nthe vision tower is replicated per rank"  # wakes turn two
        if task.id == "c":
            in_turn_two.wait(_WAIT)
            return "UNIT_READY"
        turn_two_over.wait(_WAIT)
        time.sleep(0.5)  # well past the window, were c's success still waiting
        seen["turns while slow ran"] = len(turns)
        return "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, _services(plan, work), event_window_s=0.1)
    assert orchestrator.run() == "delivered"
    assert seen["turns while slow ran"] == 2
    assert not any(e.startswith("task c ") for t in turns[2:] for e in t.events)


def test_a_failure_the_revision_kept_as_failed_still_wakes_the_planner(tmp_path):
    """A turn that copies the graph copies a failure it may never have read."""
    turns, orchestrator = [], None
    in_turn_two = threading.Event()

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(tasks=[_task("r"), _task("a"), _task("slow", satisfies=_KEYS)])
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        if len(turns) == 2:
            in_turn_two.set()
            _wait_until(lambda: orchestrator.graph.by_id()["a"].status == "failed")
            return PlannerAnswer("GRAPH_UPDATED", "unchanged", copy.deepcopy(orchestrator.graph))
        return PlannerAnswer("DONE", "a is not needed")

    def work(task):
        if task.id == "r":
            return "UNIT_READY\na report"
        if task.id == "a":
            in_turn_two.wait(_WAIT)
            return "BLOCKED"
        _wait_until(lambda: len(turns) >= 3)
        return "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, _services(plan, work), event_window_s=60)
    assert orchestrator.run() == "delivered"
    assert "task a (feature) ended BLOCKED" in turns[2].events


_RECEIPT_NOTE = "its 1 commit(s), the entry's own files only, are integrated"


def test_a_failures_report_goes_with_the_failure_the_revision_answered(tmp_path):
    """What a failed run reports beside its end is answered with it: no turn rewrites the graph."""
    turns, orchestrator, seen = [], None, {}
    in_turn_two, turn_two_over = threading.Event(), threading.Event()

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(tasks=[_task("r"), _task("a"), _task("slow", satisfies=_KEYS)])
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        if len(turns) == 2:
            in_turn_two.set()
            # The turn reads the graph on disk after a failed, and drops it.
            _wait_until(lambda: orchestrator.graph.by_id()["a"].status == "failed")
            graph = copy.deepcopy(orchestrator.graph)
            graph.tasks = [t for t in graph.tasks if t.id != "a"]
            turn_two_over.set()
            return PlannerAnswer("GRAPH_UPDATED", "drop a", graph)
        return PlannerAnswer("DONE", "nothing changed")

    def work(task):
        if task.id == "r":
            return "UNIT_READY\nthe tokenizer needs a chat template"  # wakes turn two
        if task.id == "a":
            in_turn_two.wait(_WAIT)
            return f"FAILED_RECEIPT\n{_RECEIPT_NOTE}"
        turn_two_over.wait(_WAIT)
        time.sleep(0.5)  # well past the window, were a's report still waiting
        seen["turns while slow ran"] = len(turns)
        return "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, _services(plan, work), event_window_s=0.1)
    assert orchestrator.run() == "delivered"
    assert seen["turns while slow ran"] == 2
    assert not any(e.startswith("task a ") for t in turns[2:] for e in t.events)
    # Nothing is left unread, so the idle run is delivered without another turn.
    assert len(turns) == 2


def test_a_failures_report_the_revision_kept_as_failed_still_reaches_the_planner(tmp_path):
    """A turn that copies the graph copies a failure, and its report, it may never have read."""
    turns, orchestrator = [], None
    in_turn_two = threading.Event()

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(tasks=[_task("r"), _task("a"), _task("slow", satisfies=_KEYS)])
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        if len(turns) == 2:
            in_turn_two.set()
            _wait_until(lambda: orchestrator.graph.by_id()["a"].status == "failed")
            return PlannerAnswer("GRAPH_UPDATED", "unchanged", copy.deepcopy(orchestrator.graph))
        return PlannerAnswer("DONE", "a is not needed")

    def work(task):
        if task.id == "r":
            return "UNIT_READY\na report"
        if task.id == "a":
            in_turn_two.wait(_WAIT)
            return f"FAILED_RECEIPT\n{_RECEIPT_NOTE}"
        _wait_until(lambda: len(turns) >= 3)
        return "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, _services(plan, work), event_window_s=60)
    assert orchestrator.run() == "delivered"
    assert "task a (feature) ended FAILED_RECEIPT" in turns[2].events
    assert f"task a (feature) reports: {_RECEIPT_NOTE}" in turns[2].events


def test_a_finished_runs_report_reaches_the_planner_though_the_revision_wrote_it_done(tmp_path):
    """A finished run's report may still need acting on (a campaign's best branch to merge)."""
    turns, orchestrator = [], None
    in_turn_two = threading.Event()

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(tasks=[_task("r"), _task("c"), _task("slow", satisfies=_KEYS)])
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        if len(turns) == 2:
            in_turn_two.set()
            _wait_until(lambda: orchestrator.graph.by_id()["c"].status == "done")
            return PlannerAnswer("GRAPH_UPDATED", "c landed", copy.deepcopy(orchestrator.graph))
        return PlannerAnswer("DONE", "merged")

    def work(task):
        if task.id == "r":
            return "UNIT_READY\na report"
        if task.id == "c":
            in_turn_two.wait(_WAIT)
            return "UNIT_READY\nbest branch mv2/run/c-best still needs merging"
        _wait_until(lambda: len(turns) >= 3)
        return "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, _services(plan, work), event_window_s=60)
    assert orchestrator.run() == "delivered"
    assert not any(e.startswith("task c (feature) UNIT_READY") for e in turns[2].events)
    assert "task c (feature) reports: best branch mv2/run/c-best still needs merging" in (
        turns[2].events
    )


@pytest.mark.parametrize(
    ("outcome", "written", "adopted", "answered"),
    [
        (("a", 1, "failed"), {}, {}, True),  # dropped
        (("a", 1, "failed"), {"a": ("pending", 1)}, {"a": ("pending", 2)}, True),  # retried
        (("a", 1, "failed"), {"a": ("failed", 1)}, {"a": ("failed", 1)}, False),
        (("a", 1, "abandoned"), {"a": ("pending", 1)}, {"a": ("pending", 1)}, True),
        (("a", 1, "abandoned"), {"a": ("abandoned", 1)}, {"a": ("abandoned", 1)}, False),
        (("a", 1, "done"), {"a": ("done", 1)}, {"a": ("done", 1)}, True),
        (("a", 2, "done"), {"a": ("done", 1)}, {"a": ("done", 2)}, False),
        (("a", 1, "done"), {"a": ("running", 1)}, {"a": ("done", 1)}, False),
    ],
)
def test_which_events_a_revision_answered(outcome, written, adopted, answered):
    tasks = {}
    for tid, (status, attempts) in adopted.items():
        tasks[tid] = _task(tid)
        tasks[tid].status, tasks[tid].attempts = status, attempts
    assert orchestrator_mod._answered(orchestrator_mod._Outcome(*outcome), written, tasks) is (
        answered
    )


# --------------------------------------------------------------------- stop


def test_a_stop_ends_the_run_at_once_without_waiting_for_the_turns_in_flight(tmp_path):
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

    with orchestrator_mod.stoppable(tmp_path) as stop:
        orchestrator = _orchestrator(tmp_path, _services(plan, work))
        threading.Thread(target=lambda: started.wait(_WAIT) and stop("the host failed")).start()
        begun = time.monotonic()
        assert orchestrator.run() == "stopped: the host failed"
        assert time.monotonic() - begun < _WAIT / 2
    release.set()
    assert len(turns) == 1
    assert json.loads((tmp_path / "inflight.json").read_text())["stopped"]["how"] == (
        "closed:stopped: the host failed"
    )


def test_a_stop_asked_before_the_loop_starts_ends_it_as_it_starts(tmp_path):
    turns = []
    with orchestrator_mod.stoppable(tmp_path) as stop:
        stop("the substrate gate failed")
        orchestrator = _orchestrator(tmp_path, _services(lambda turn: turns.append(turn)))
        assert orchestrator.run() == "stopped: the substrate gate failed"
    assert turns == []
    # Outside it, a run on the same workspace is not stopped.
    assert not _orchestrator(tmp_path, _services(None))._stop.done()


# ------------------------------------------------------------------ checks


def test_a_tasks_checks_run_at_the_same_time(tmp_path):
    contract = c.Contract(
        checks=tuple(c.Check("core", "boot", m, c.TIER_INNER) for m in ("tp1", "tp2", "tp4", "tp8"))
    )
    keys = [check.key for check in contract.checks]
    together = threading.Barrier(len(keys), timeout=_WAIT)

    def check(chk):
        together.wait()  # passes only once every check of the task has started
        return _result(chk)

    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            return PlannerAnswer(
                "GRAPH_UPDATED", "plan", g.Graph(tasks=[_task("core", "assemble", keys)])
            )
        return PlannerAnswer("DONE", "all green")

    orchestrator = _orchestrator(tmp_path, _services(plan, check=check), contract=contract)
    assert orchestrator.run() == "delivered"
    assert {v.member for v in orchestrator.status.verdicts if v.passed} == {
        "tp1",
        "tp2",
        "tp4",
        "tp8",
    }


def test_the_check_pool_bounds_how_many_checks_run_at_once(tmp_path):
    contract = c.Contract(
        checks=tuple(c.Check("core", "boot", m, c.TIER_INNER) for m in ("tp1", "tp2", "tp4", "tp8"))
    )
    pairs = threading.Barrier(2, timeout=_WAIT)
    lock, running, peak = threading.Lock(), [0], [0]

    def check(chk):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        pairs.wait()
        with lock:
            running[0] -= 1
        return _result(chk, passed=chk.member != "tp4")

    orchestrator = _orchestrator(
        tmp_path, _services(None, check=check), contract=contract, max_parallel_checks=2
    )
    assert orchestrator._run_checks(chk.key for chk in contract.checks) == ["core:boot@tp4"]
    assert peak[0] == 2


def test_admission_bounds_the_checks_by_what_the_sessions_can_start(tmp_path):
    """A bigger pool, but the sessions' free devices decide how many run at once."""
    from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CheckAdmission

    contract = c.Contract(
        checks=tuple(c.Check("core", "boot", m, c.TIER_INNER) for m in ("tp1", "tp2", "tp4", "tp8"))
    )
    # No session has a free device: only the floor of two runs.
    admission = CheckAdmission(lambda env, gpus, wall: 0, floor=2, poll_s=0.01)
    pairs = threading.Barrier(2, timeout=_WAIT)
    lock, running, peak = threading.Lock(), [0], [0]

    def check(chk):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        pairs.wait()
        with lock:
            running[0] -= 1
        return _result(chk)

    services = _services(None, check=check)
    services.admit = lambda chk: admission.slot(SimpleNamespace(world_size=1))
    orchestrator = _orchestrator(
        tmp_path, services, contract=contract, max_parallel_checks=admission.cap
    )
    assert orchestrator._run_checks(chk.key for chk in contract.checks) == []
    assert peak[0] == 2
    assert admission.in_flight == 0


def test_a_check_is_recorded_under_the_target_as_it_was_when_it_started(tmp_path):
    tree = {"marker": "v1"}

    def check(chk):
        tree["marker"] = "v2"  # another task integrates while this check runs
        return _result(chk)

    services = _services(None, check=check)
    services.marker = lambda: tree["marker"]
    orchestrator = _orchestrator(tmp_path, services)
    assert orchestrator._run_checks(_KEYS[:1]) == []
    assert orchestrator.status.verdicts[-1].marker == "v1"


def test_stale_rechecks_run_beside_a_task_the_planner_adds(tmp_path):
    status = c.ContractStatus.load(tmp_path)
    for check in _CONTRACT.checks:
        status.record(check.unit, _result(check), marker="v1")
    core = _task("core", "assemble", _KEYS)
    core.status = "done"
    g.Graph(tasks=[core]).save(tmp_path / g.GRAPH_FILENAME)
    _left_unread(tmp_path, "operator: a fix for the v2 regression is next")
    fix_started, seen, turns = threading.Event(), {}, []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(tasks=[_task("core", "assemble", _KEYS), _task("fix")])
            return PlannerAnswer("GRAPH_UPDATED", "the fix", graph)
        return PlannerAnswer("DONE", "all green")

    def check(chk):
        seen.setdefault("fix started while the re-checks ran", fix_started.wait(_WAIT))
        return _result(chk)

    def work(task):
        fix_started.set()
        return "UNIT_READY"

    services = _services(plan, work, check)
    services.marker = lambda: "v2"
    orchestrator = GraphOrchestrator(services, _CONTRACT, status, tmp_path, poll_s=0.01)
    assert orchestrator.run() == "delivered"
    assert seen["fix started while the re-checks ran"]
    assert [v.marker for v in orchestrator.status.verdicts] == ["v1", "v1", "v2", "v2"]


def test_a_check_already_passed_on_this_target_does_not_run_again(tmp_path):
    ran = []

    def check(chk):
        ran.append(chk.key)
        return _result(chk)

    status = c.ContractStatus.load(tmp_path)
    status.record("core", _result(_CONTRACT.checks[0]), marker="v1")
    services = _services(None, check=check)
    services.marker = lambda: "v1"
    orchestrator = GraphOrchestrator(
        services, _CONTRACT, status, tmp_path, poll_s=0.01, max_parallel_checks=1
    )
    # The boot verdict is held on v1; of two accuracy checks the second reads the first's.
    assert orchestrator._run_checks([_KEYS[0], _KEYS[1], _KEYS[1]]) == []
    assert ran == [_KEYS[1]]


def test_without_an_established_marker_every_check_runs(tmp_path):
    ran = []

    def check(chk):
        ran.append(chk.key)
        return _result(chk)

    status = c.ContractStatus.load(tmp_path)
    status.record("core", _result(_CONTRACT.checks[0]), marker="")
    orchestrator = GraphOrchestrator(_services(None, check=check), _CONTRACT, status, tmp_path)
    assert orchestrator._run_checks(_KEYS) == []
    assert ran == _KEYS


# ------------------------------------------- successes the graph already answered


def test_a_success_its_dependents_wait_for_takes_no_turn_of_its_own(tmp_path):
    """The graph already says what follows: the dependent starts on it, its receipt in hand."""
    turns, orchestrator, seen = [], None, {}

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(
                tasks=[
                    _task("k", kind="kernel"),
                    _task("dep", deps=["k"]),
                    _task("last", satisfies=_KEYS, deps=["dep"]),
                ]
            )
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        return PlannerAnswer("GRAPH_UPDATED", "nothing changes", copy.deepcopy(orchestrator.graph))

    def work(task):
        if task.id == "dep":
            time.sleep(0.5)  # well past the window, were k's success to wake the planner
            seen["turns while dep ran"] = len(turns)
            return "UNIT_READY\nthe receipt names a limit"  # a report wakes the next turn
        return "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, _services(plan, work), event_window_s=0.1)
    assert orchestrator.run() == "delivered"
    assert seen["turns while dep ran"] == 1
    # It waited for the next turn something else woke, and reached it.
    assert any(e.startswith("task k (kernel) UNIT_READY") for e in turns[1].events)


@pytest.mark.parametrize(
    ("satisfies", "dependent", "quiet"),
    [
        ((), "hard", True),
        ((), "soft", True),
        ((), None, False),  # nothing in the graph says what follows it
        (tuple(_KEYS[:1]), "hard", False),  # a verdict moved, and with it the runtime facts
    ],
)
def test_which_successes_the_graph_already_answered(tmp_path, satisfies, dependent, quiet):
    task = _task("k", kind="kernel", satisfies=satisfies)
    other = _task("dep", deps=["k"] if dependent == "hard" else [])
    if dependent == "soft":
        other.depends_on_soft = ["k"]
    orchestrator = _orchestrator(tmp_path, _services(lambda turn: None))
    orchestrator.graph = g.Graph(tasks=[task, other])
    assert orchestrator._answered_ahead(task) is quiet


# ------------------------------------------------- the orchestrator's own notes


def test_a_note_about_tasks_the_orchestrator_added_is_answered_by_a_revision_that_holds_them(
    tmp_path,
):
    """Promote, defect fix, delivery files, kernel rows: a revision with the tasks already has them."""
    orchestrator = _orchestrator(tmp_path, _services(lambda turn: None))
    orchestrator.graph = g.Graph(tasks=[_task("a"), _task("promote-2")])
    orchestrator._note("the orchestrator added promote-2", added=("promote-2",))
    orchestrator._note("a worker said something")
    orchestrator._drop_answered({"a": ("done", 1)})
    assert len(orchestrator.events) == 2, "a revision without the task does not answer it"
    orchestrator._drop_answered({"a": ("done", 1), "promote-2": ("pending", 1)})
    assert orchestrator.events == ["a worker said something"]


def test_soft_dependents_of_a_failed_run_are_answered_with_the_run(tmp_path):
    orchestrator = _orchestrator(tmp_path, _services(lambda turn: None))
    failed = _task("k1")
    failed.status = "failed"
    dependent = _task("core")
    dependent.depends_on_soft = ["k1"]
    orchestrator.graph = g.Graph(tasks=[failed, dependent])
    orchestrator._note_soft_dependents(failed)
    assert orchestrator.events and "soft-depend on k1" in orchestrator.events[0]
    retried = copy.deepcopy(orchestrator.graph)
    retried.by_id()["k1"].attempts = 2
    orchestrator.graph = retried
    orchestrator._drop_answered({"k1": ("pending", 2), "core": ("pending", 1)})
    assert orchestrator.events == []


def test_a_check_task_that_passes_wakes_no_turn_of_its_own(tmp_path):
    """Only a failed check needs the planner; a green one is read by the next turn."""
    orchestrator = _orchestrator(tmp_path, _services(lambda turn: None))
    check = _task("mm-dp", kind=g.CHECK_KIND, satisfies=_KEYS[:1])
    check.status = "running"
    orchestrator.graph = g.Graph(tasks=[check])
    orchestrator._finish(check, "CHECKED", [])
    assert orchestrator.events and not orchestrator._loud()
    failing = _task("mm-tp", kind=g.CHECK_KIND, satisfies=_KEYS[:1])
    failing.status = "running"
    orchestrator.graph.tasks.append(failing)
    orchestrator._finish(failing, "CHECKED", list(_KEYS[:1]))
    assert orchestrator._loud()


def test_a_task_the_orchestrator_adds_goes_before_the_regression_review_its_change_needs(
    tmp_path,
):
    """The delivery files touch shared test lists: the graph it leaves passes its own validation."""
    spec = {
        "id": "delivery-files",
        "kind": "feature",
        "goal": "write the delivery files",
        "touches": ["tests/integration/test_lists/test-db/"],
        "depends_on": ["core"],
    }
    services = _services(lambda turn: None)
    services.delivery_task = lambda graph: spec
    orchestrator = _orchestrator(tmp_path, services, own_paths=("models/newmodel",))
    review = g.Task(
        id="regress", kind="review", goal="g", regression=True, depends_on=["core"], touches=[]
    )
    orchestrator.graph = g.Graph(tasks=[_task("core"), review])
    orchestrator._add_delivery_task()
    assert "delivery-files" in review.depends_on
    assert not g.unreviewed_shared_changes(orchestrator.graph, ("models/newmodel",))
