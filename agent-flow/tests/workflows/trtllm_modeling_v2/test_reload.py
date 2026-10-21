# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A reload drains the orchestrator, then re-executes it; a stop by hand is recorded as a cost.

A run that loads a flow fix by killing and relaunching its orchestrator cuts
short whatever was in flight: a rebase, gate jobs, a planner answer about to
be read. These pin the reload instead: while it drains, no task, integration,
check or planner turn starts and what is in flight runs to its end; then the
loop checkpoints and hands over to the re-exec. Every start records what the
stop before it cut short.

Every fake that waits for another thread waits on an event with a deadline,
so a loop that never drains fails the test instead of hanging it.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import cli as cli_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.driver import core as driver
from agent_flow.workflows.trtllm_modeling_v2.infra.driver import reload as reload_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import write_json_atomic
from agent_flow.workflows.trtllm_modeling_v2.infra.transport import (
    ORCHESTRATOR_HEARTBEAT,
    ExternalAgentTransport,
)
from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import drain as drain_mod
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator as orchestrator_mod
from agent_flow.workflows.trtllm_modeling_v2.run.integration import Outcome
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)
from agent_flow.workflows.trtllm_modeling_v2.run.stages import planned as planned_mod

from .test_workflow import Script, _install, _workflow

#: How long a fake waits for another thread before the test counts it as never.
_WAIT = 10.0
_CONTRACT = c.Contract(
    checks=(
        c.Check("core", "boot", "tp1", c.TIER_INNER),
        c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER),
    )
)
_BOOT = "core:boot@tp1"


class Replaced(Exception):
    """What the fake re-exec raises: a real one never returns, as the process is gone."""


def _task(tid, satisfies=(), deps=()):
    return g.Task(
        id=tid,
        kind="feature",
        goal=f"do {tid}",
        depends_on=list(deps),
        touches=[f"models/{tid}.py"],
        gpus=1,
        est_minutes=10,
        satisfies=list(satisfies),
    )


def _passed(check):
    return SimpleNamespace(name=check.gate, member=check.member, passed=True, detail="")


def _until(predicate, what):
    deadline = time.monotonic() + _WAIT
    while not predicate():
        assert time.monotonic() < deadline, f"never: {what}"
        time.sleep(0.005)


def _phase(workspace: Path):
    return (reload_mod.read_record(workspace) or {}).get("phase")


class Reexec:
    """The reload service: notes what the run looked like when it was handed over.

    ``note`` adds what the test itself kept; ``then`` runs before it raises.
    """

    def __init__(self, workspace: Path, *, error=None, note=None, then=None):
        self.workspace = workspace
        self.error = error or Replaced()
        self.note = note
        self.then = then
        self.calls: list[dict] = []

    def __call__(self, record):
        self.calls.append(
            {
                "record": record,
                "graph": {
                    t.id: t.status for t in g.Graph.load(self.workspace / g.GRAPH_FILENAME).tasks
                },
                "inflight": reload_mod.read_inflight(self.workspace),
                "request": reload_mod.read_request(self.workspace),
                "note": self.note() if self.note is not None else None,
            }
        )
        if self.then is not None:
            self.then()
        raise self.error


def _start(orchestrator):
    """Run ``orchestrator`` in a thread; the box gets its verdict or what it raised."""
    box: dict = {}

    def run():
        try:
            box["verdict"] = orchestrator.run()
        except BaseException as exc:  # noqa: BLE001 -- handed to the test to assert on
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


def _replaced(thread, box):
    thread.join(_WAIT)
    assert not thread.is_alive(), "the run neither re-executed nor ended"
    assert isinstance(box.get("error"), Replaced), box


def _orchestrator(workspace, *, plan=None, work=None, check=None, reload=None, log=None):
    services = Services(
        plan=plan or (lambda turn: PlannerAnswer("DONE", "all green")),
        work=work or (lambda task: "UNIT_READY"),
        check=check or _passed,
        free_gpus=lambda: 8,
        reload=reload,
        log=log or (lambda message: None),
    )
    status = c.ContractStatus.load(workspace)
    return GraphOrchestrator(services, _CONTRACT, status, workspace, poll_s=0.01)


def _blocking_check(started: threading.Event, release: threading.Event, key=_BOOT):
    def check(chk):
        if chk.key == key:
            started.set()
            assert release.wait(_WAIT), "the test never released the check"
        return _passed(chk)

    return check


# ------------------------------------------------------------------ draining


def test_a_reload_waits_for_the_check_in_flight_and_starts_nothing_meanwhile(tmp_path):
    g.Graph(tasks=[_task("a", [_BOOT]), _task("b", deps=["a"])]).save(tmp_path / g.GRAPH_FILENAME)
    started, release = threading.Event(), threading.Event()
    worked, logged = [], []
    reexec = Reexec(tmp_path)
    orchestrator = _orchestrator(
        tmp_path,
        work=lambda task: worked.append(task.id) or "UNIT_READY",
        check=_blocking_check(started, release),
        reload=reexec,
        log=logged.append,
    )
    runner = _start(orchestrator)
    assert started.wait(_WAIT)
    reload_mod.request(tmp_path, by="operator")
    _until(lambda: _phase(tmp_path) == reload_mod.PHASE_DRAINING, "the drain began")
    record = reload_mod.read_record(tmp_path)
    assert record["waiting_for"] == [f"check {_BOOT}"]
    time.sleep(0.1)
    assert not reexec.calls, "re-executed with a check in flight"
    release.set()
    _replaced(*runner)

    (call,) = reexec.calls
    assert worked == ["a"], "b became ready during the drain and must not start"
    assert call["graph"] == {"a": "done", "b": "pending"}
    assert call["request"] is None, "the request is answered before the re-exec"
    assert call["inflight"]["stopped"]["how"] == "reload"
    assert call["inflight"]["held"] == []
    assert call["record"]["phase"] == reload_mod.PHASE_REEXEC
    assert [w["what"] for w in call["record"]["waited"]] == [f"check {_BOOT}"]
    assert call["record"]["left"] == [] and call["record"]["forced"] is False
    assert any(line.startswith("[reload] requested by operator") for line in logged)


def test_a_check_that_would_start_during_the_drain_waits_for_the_new_process(tmp_path):
    """The task's turn ends while the drain waits for another task's check."""
    g.Graph(tasks=[_task("a", [_BOOT]), _task("b", ["core:accuracy[gsm8k]@tp1"])]).save(
        tmp_path / g.GRAPH_FILENAME
    )
    started, release, turn_ends = threading.Event(), threading.Event(), threading.Event()
    checked = []

    def work(task):
        if task.id == "b":
            assert turn_ends.wait(_WAIT)
        return "UNIT_READY"

    blocking = _blocking_check(started, release)

    def check(chk):
        checked.append(chk.key)
        return blocking(chk)

    reexec = Reexec(tmp_path, note=lambda: list(checked))
    orchestrator = _orchestrator(tmp_path, work=work, check=check, reload=reexec)
    runner = _start(orchestrator)
    assert started.wait(_WAIT)
    reload_mod.request(tmp_path)
    _until(lambda: _phase(tmp_path) == reload_mod.PHASE_DRAINING, "the drain began")
    turn_ends.set()
    time.sleep(0.1)
    release.set()
    _replaced(*runner)

    (call,) = reexec.calls
    assert call["graph"] == {"a": "done", "b": "running"}, "b resumes in the next process"
    assert call["note"] == [_BOOT], "b's check started during the drain"


def test_the_drain_waits_for_the_planner_turn_in_flight_and_adopts_its_graph(tmp_path):
    in_turn, answer = threading.Event(), threading.Event()
    worked = []

    def plan(turn):
        in_turn.set()
        assert answer.wait(_WAIT)
        graph = g.Graph(tasks=[_task("a", [_BOOT, "core:accuracy[gsm8k]@tp1"])])
        return PlannerAnswer("GRAPH_UPDATED", "plan", graph)

    reexec = Reexec(tmp_path)
    orchestrator = _orchestrator(
        tmp_path, plan=plan, work=lambda t: worked.append(t.id) or "UNIT_READY", reload=reexec
    )
    runner = _start(orchestrator)
    assert in_turn.wait(_WAIT)
    reload_mod.request(tmp_path)
    _until(lambda: _phase(tmp_path) == reload_mod.PHASE_DRAINING, "the drain began")
    assert reload_mod.read_record(tmp_path)["waiting_for"] == ["planner turn"]
    answer.set()
    _replaced(*runner)

    (call,) = reexec.calls
    assert call["graph"] == {"a": "pending"}, "the answer is applied, and nothing starts"
    assert worked == []
    assert [w["what"] for w in call["record"]["waited"]] == ["planner turn"]


def test_a_drain_past_its_deadline_says_what_it_waits_for_and_keeps_waiting(tmp_path):
    g.Graph(tasks=[_task("a", [_BOOT])]).save(tmp_path / g.GRAPH_FILENAME)
    started, release = threading.Event(), threading.Event()
    logged = []
    reexec = Reexec(tmp_path)
    orchestrator = _orchestrator(
        tmp_path, check=_blocking_check(started, release), reload=reexec, log=logged.append
    )
    runner = _start(orchestrator)
    assert started.wait(_WAIT)
    reload_mod.request(tmp_path, timeout_s=0.05)
    _until(lambda: (reload_mod.read_record(tmp_path) or {}).get("overdue"), "overdue")
    assert any("past its deadline" in line and f"check {_BOOT}" in line for line in logged)
    time.sleep(0.1)
    assert not reexec.calls, "an overdue drain without --force keeps waiting"
    release.set()
    _replaced(*runner)
    assert len(reexec.calls) == 1


def test_a_forced_drain_goes_on_at_its_deadline_without_what_is_left(tmp_path):
    g.Graph(tasks=[_task("a", [_BOOT])]).save(tmp_path / g.GRAPH_FILENAME)
    started, release = threading.Event(), threading.Event()
    reexec = Reexec(tmp_path, then=release.set)
    orchestrator = _orchestrator(tmp_path, check=_blocking_check(started, release), reload=reexec)
    runner = _start(orchestrator)
    assert started.wait(_WAIT)
    reload_mod.request(tmp_path, timeout_s=0.05, force=True)
    _replaced(*runner)

    (call,) = reexec.calls
    assert call["record"]["left"] == [f"check {_BOOT}"] and call["record"]["forced"]
    held = call["inflight"]["held"]
    assert [(h["kind"], h["what"]) for h in held] == [("check", _BOOT)], (
        "what a forced reload cuts short stays in the ledger for the next start"
    )


def test_a_withdrawn_request_ends_the_drain_and_the_run_goes_on(tmp_path):
    g.Graph(tasks=[_task("a", [_BOOT, "core:accuracy[gsm8k]@tp1"])]).save(
        tmp_path / g.GRAPH_FILENAME
    )
    started, release = threading.Event(), threading.Event()
    reexec = Reexec(tmp_path)
    orchestrator = _orchestrator(tmp_path, check=_blocking_check(started, release), reload=reexec)
    thread, box = _start(orchestrator)
    assert started.wait(_WAIT)
    reload_mod.request(tmp_path)
    _until(lambda: _phase(tmp_path) == reload_mod.PHASE_DRAINING, "the drain began")
    assert reload_mod.cancel(tmp_path) is not None
    _until(lambda: _phase(tmp_path) == reload_mod.PHASE_CANCELLED, "the drain ended")
    release.set()
    thread.join(_WAIT)

    assert box.get("verdict") == "delivered", box
    assert reexec.calls == []
    assert not orchestrator.inflight.closed


def test_a_reexec_that_fails_leaves_the_run_going_on_in_this_process(tmp_path):
    g.Graph(tasks=[_task("a", [_BOOT, "core:accuracy[gsm8k]@tp1"])]).save(
        tmp_path / g.GRAPH_FILENAME
    )
    reload_mod.request(tmp_path)
    logged = []
    reexec = Reexec(tmp_path, error=OSError(8, "Exec format error"))
    orchestrator = _orchestrator(tmp_path, reload=reexec, log=logged.append)

    assert orchestrator.run() == "delivered"
    assert len(reexec.calls) == 1
    record = reload_mod.read_record(tmp_path)
    assert record["phase"] == reload_mod.PHASE_FAILED and "Exec format error" in record["error"]
    assert reload_mod.read_request(tmp_path) is None, "a failed reload is not tried again"
    assert any("the run goes on in this process" in line for line in logged)
    ledger = reload_mod.read_inflight(tmp_path)
    assert ledger["stopped"]["how"] == "closed:delivered"


def test_a_request_whose_reexec_failed_is_not_tried_again_until_it_is_asked_again(
    tmp_path, monkeypatch
):
    """A request the failed re-exec could not remove would otherwise drain the run in a loop."""
    g.Graph(tasks=[_task("a", [_BOOT, "core:accuracy[gsm8k]@tp1"])]).save(
        tmp_path / g.GRAPH_FILENAME
    )
    monkeypatch.setattr(orchestrator_mod.reload_mod, "remove_request", lambda workspace: None)
    reload_mod.request(tmp_path)
    reexec = Reexec(tmp_path, error=OSError(13, "Permission denied"))
    assert _orchestrator(tmp_path, reload=reexec).run() == "delivered"
    assert len(reexec.calls) == 1
    assert reload_mod.read_request(tmp_path) is not None


def test_an_orchestrator_that_cannot_reexec_leaves_a_request_alone(tmp_path):
    g.Graph(tasks=[_task("a", [_BOOT, "core:accuracy[gsm8k]@tp1"])]).save(
        tmp_path / g.GRAPH_FILENAME
    )
    reload_mod.request(tmp_path)
    assert _orchestrator(tmp_path).run() == "delivered"
    assert reload_mod.read_record(tmp_path) is None


# --------------------------------------------------------------- the next start


def test_a_request_older_than_the_process_is_answered_by_its_start(tmp_path):
    g.Graph(tasks=[_task("a", [_BOOT, "core:accuracy[gsm8k]@tp1"])]).save(
        tmp_path / g.GRAPH_FILENAME
    )
    reload_mod.request(tmp_path, now=drain_mod.PROCESS_STARTED_AT - 60)
    reexec = Reexec(tmp_path)
    orchestrator = _orchestrator(tmp_path, reload=reexec)
    assert reload_mod.read_request(tmp_path) is None
    assert orchestrator.restart["answered_request"]["requested_at"] < drain_mod.PROCESS_STARTED_AT
    assert "is answered by this start" in drain_mod.describe_restart(orchestrator.restart)
    assert orchestrator.run() == "delivered"
    assert reexec.calls == []


def test_a_hard_stop_is_recorded_with_what_it_cut_short(tmp_path):
    g.Graph(tasks=[_task("a", [_BOOT]), _task("b")]).save(tmp_path / g.GRAPH_FILENAME)
    graph = g.Graph.load(tmp_path / g.GRAPH_FILENAME)
    graph.tasks[0].status = "running"
    graph.save(tmp_path / g.GRAPH_FILENAME)
    write_json_atomic(
        tmp_path / reload_mod.INFLIGHT_FILENAME,
        {
            "pid": 4242,
            "host": "gone",
            "started_at": 1.0,
            "stopped": None,
            "held": [
                {"kind": "check", "what": _BOOT, "since": 2.0},
                {"kind": "integration", "what": "k46", "since": 3.0},
            ],
        },
    )
    orchestrator = _orchestrator(tmp_path)

    record = orchestrator.restart
    assert record["after"] == drain_mod.AFTER_HARD_STOP
    assert record["tasks_reset"] == ["a"]
    assert [(c["kind"], c["what"]) for c in record["cut"]] == [
        ("check", _BOOT),
        ("integration", "k46"),
    ]
    line = drain_mod.describe_restart(record)
    assert "pid 4242 on gone" in line and "stopped without a reload" in line
    assert f"cut short: check {_BOOT}, integration k46" in line
    assert "tasks set back to pending: a" in line
    ledger = reload_mod.read_inflight(tmp_path)
    assert ledger["pid"] == os.getpid() and ledger["held"] == [], "this process keeps its own"


def test_after_a_reload_the_start_costs_nothing_and_completes_the_reload(tmp_path):
    g.Graph(tasks=[_task("a", [_BOOT, "core:accuracy[gsm8k]@tp1"])]).save(
        tmp_path / g.GRAPH_FILENAME
    )
    write_json_atomic(
        tmp_path / reload_mod.INFLIGHT_FILENAME,
        {"pid": os.getpid(), "host": "h", "started_at": 1.0, "held": [],
         "stopped": {"how": "reload", "at": 5.0}},
    )  # fmt: skip
    reload_mod.write_record(
        tmp_path,
        {"id": "r1", "phase": reload_mod.PHASE_REEXEC, "pid": os.getpid(), "reexec_at": 5.0,
         "drained_s": 12.0, "waited": [{"what": f"check {_BOOT}", "s": 12.0}], "left": []},
    )  # fmt: skip
    orchestrator = _orchestrator(tmp_path, reload=Reexec(tmp_path))
    assert orchestrator.restart["after"] == drain_mod.AFTER_RELOAD
    assert "nothing was cut short" in drain_mod.describe_restart(orchestrator.restart)

    assert orchestrator.run() == "delivered"
    record = reload_mod.read_record(tmp_path)
    assert record["phase"] == reload_mod.PHASE_RELOADED and record["reloaded_at"] >= 5.0


def test_a_drain_its_process_never_finished_is_marked_abandoned(tmp_path):
    reload_mod.write_record(tmp_path, {"id": "r1", "phase": reload_mod.PHASE_DRAINING})
    _orchestrator(tmp_path)
    assert _phase(tmp_path) == reload_mod.PHASE_ABANDONED


# --------------------------------------------------------------------- the door


def test_the_door_holds_new_work_back_but_not_work_already_through_it(tmp_path):
    inflight = drain_mod.InFlight(tmp_path)
    inflight.close()
    with inflight.hold("turn", "planner", door=False):
        # Nested in this thread: it never waits for itself.
        with inflight.hold("check", _BOOT):
            on_disk = reload_mod.read_inflight(tmp_path)
            assert [h["what"] for h in on_disk["held"]] == ["planner", _BOOT]
    entered = threading.Event()

    def later():
        with inflight.hold("integration", "k1"):
            entered.set()

    thread = threading.Thread(target=later, daemon=True)
    thread.start()
    assert not entered.wait(0.1), "new work went through a closed door"
    changed = inflight.changed()
    inflight.open()
    assert entered.wait(_WAIT) and changed.result(_WAIT) is None
    thread.join(_WAIT)
    assert inflight.held() == []


# ------------------------------------------------------------------ the workflow


def test_a_planned_run_reloads_between_turns_and_resumes_where_it_was(
    tmp_path, task_file, monkeypatch
):
    """The planner's turn is answered while a reload waits; the next process runs the task."""
    asked = []

    def planner(turn, prompt):
        from .test_workflow import plan_every_unit

        if turn == 1:
            reload_mod.request(tmp_path / "ws", by="driver")
        return plan_every_unit(turn, prompt)

    def replaced():
        asked.append(json.loads((tmp_path / "ws" / reload_mod.RECORD_FILENAME).read_text()))
        raise Replaced()

    first = _workflow(tmp_path)
    first.reexec = replaced
    calls = _install(monkeypatch, first, Script(None, planner=planner))
    with first, pytest.raises(Replaced):
        first.run(task_file)
    assert [name for name, _ in calls] == ["planner"], "no task starts while it drains"
    assert asked and asked[0]["phase"] == reload_mod.PHASE_REEXEC
    journal = (tmp_path / "ws" / "journal.md").read_text(encoding="utf-8")
    assert "**reload**" in journal

    second = _workflow(tmp_path)
    second.reexec = replaced
    calls = _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)
    assert [name for name, _ in calls] == ["assembler"], "the plan is not asked again"
    assert _phase(tmp_path / "ws") == reload_mod.PHASE_RELOADED
    journal = (tmp_path / "ws" / "journal.md").read_text(encoding="utf-8")
    assert "**restart (reload)**" in journal and "nothing was cut short" in journal
    starts = [
        json.loads(line)
        for line in (tmp_path / "ws" / reload_mod.RESTARTS_FILENAME).read_text().splitlines()
    ]
    assert [s["after"] for s in starts] == [drain_mod.AFTER_FIRST, drain_mod.AFTER_RELOAD]


class _Integration:
    """Integrates by noting what the run held in flight at the time."""

    def __init__(self, workflow, root: Path):
        self.workflow, self.repo, self.root = workflow, root / "repo", root
        self.seen = []

    def open(self, task_id, touches, full=False, attempt=1):
        self.seen.append(("open", [h["what"] for h in self.workflow._inflight.held()]))
        return self.root / "worktrees" / task_id

    def integrate(self, task_id):
        self.seen.append(("integrate", [h["kind"] for h in self.workflow._inflight.held()]))
        return Outcome(True, f"integrated {task_id}", 1)

    def close(self, task_id):
        pass


def test_a_tasks_worktree_and_integration_are_in_flight_and_wait_at_a_closed_door(
    tmp_path, monkeypatch
):
    workflow = _workflow(tmp_path)
    workflow._inflight = drain_mod.InFlight(workflow.workspace)
    integration = _Integration(workflow, tmp_path)
    workflow._integration = integration
    monkeypatch.setattr(planned_mod, "worktree_block", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        type(workflow), "_work_in_tree", lambda self, task, block, note="": "UNIT_READY"
    )
    task = g.Task(id="k1", kind="kernel", goal="g", touches=["ops/x.py"])

    assert workflow._work_task_turn(task) == "UNIT_READY"
    assert integration.seen == [("open", ["k1"]), ("integrate", ["integration"])]

    integration.seen.clear()
    workflow._inflight.close()
    box = {}
    thread = threading.Thread(
        target=lambda: box.setdefault("decision", workflow._work_task_turn(task)), daemon=True
    )
    thread.start()
    time.sleep(0.1)
    assert integration.seen == [], "a worktree was made while a reload drained"
    workflow._inflight.open()
    thread.join(_WAIT)
    assert box["decision"] == "UNIT_READY" and len(integration.seen) == 2


def test_only_a_turn_in_this_process_is_waited_for(tmp_path):
    workflow = _workflow(tmp_path)
    inflight = drain_mod.InFlight(workflow.workspace)
    workflow._inflight = inflight
    with workflow._turn_in_flight("assembler", "core"):
        assert [h["what"] for h in inflight.held()] == ["assembler core"]
    external = _workflow(
        tmp_path, agent_transport=ExternalAgentTransport(tmp_path / "ws", poll_s=0.01)
    )
    external._inflight = inflight
    with external._turn_in_flight("assembler", "core"):
        assert inflight.held() == [], "the next process adopts an external turn"


def test_the_start_names_the_turns_the_previous_process_left(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    transport = ExternalAgentTransport(workspace, poll_s=0.01)
    dispatch = transport.root
    asked_at = drain_mod.PROCESS_STARTED_AT - 30
    for request_id, what, answered, read in (
        ("181-planner", "planner", True, True),
        ("182-planner", "planner", True, False),
        ("183-assembler", "core-tp", True, True),
        ("184-smith", "onboard x", False, False),
        ("185-assembler", "core-dp", True, True),
    ):
        (dispatch / f"{request_id}.request.json").write_text(
            json.dumps({"id": request_id, "role": request_id[4:], "what": what,
                        "requested_at": asked_at}),
            encoding="utf-8",
        )  # fmt: skip
        if answered:
            (dispatch / f"{request_id}.done").write_text("1\n", encoding="utf-8")
        if read:
            (dispatch / f"{request_id}.read").write_text("{}", encoding="utf-8")
    (dispatch / "185-assembler.consumed").write_text("{}", encoding="utf-8")
    workflow = _workflow(tmp_path, agent_transport=transport)
    record = drain_mod.restart_record(
        {"pid": 9, "host": "h", "started_at": asked_at - 1, "held": [], "stopped": None},
        reset=["core-tp"],
    )

    workflow._record_restart(record)

    assert [t["id"] for t in record["turns"]] == ["182-planner", "183-assembler", "184-smith"]
    journal = (workspace / "journal.md").read_text(encoding="utf-8")
    assert "**restart (hard-stop)**" in journal
    assert "turns still with the driver (waited on): 184-smith onboard x" in journal
    assert "answers no process read: 182-planner planner" in journal
    assert "answered turns of tasks with no outcome (adopted): 183-assembler core-tp" in journal
    (line,) = (workspace / reload_mod.RESTARTS_FILENAME).read_text().splitlines()
    assert json.loads(line)["after"] == drain_mod.AFTER_HARD_STOP


def test_the_start_names_the_check_jobs_a_hard_stop_left_in_the_spools(tmp_path):
    started = drain_mod.PROCESS_STARTED_AT - 600
    stamp = int((started + 60) * 1000)
    spool = tmp_path / "spool"
    for directory, name in (
        ("running", f"gate-{stamp}-4242-a1b2c3-1"),  # the previous process's check
        ("pending", f"gate-{stamp}-4242-a1b2c3-2"),
        ("running", f"gate-{stamp}-777-a1b2c3-1"),  # another process's
        ("running", f"gate-{int((started - 60) * 1000)}-4242-a1b2c3-1"),  # before it started
        ("running", f"gate-boot-{stamp}-4242-a1b2c3-1"),  # a standalone gate an agent ran
        ("done", f"gate-{stamp}-4242-a1b2c3-3"),  # in already
    ):
        (spool / directory).mkdir(parents=True, exist_ok=True)
        (spool / directory / f"{name}.sh").write_text("#!/bin/bash\n")

    left = drain_mod.jobs_left(
        [spool, tmp_path / "no-such-spool"], pid=4242, since=started,
        before=drain_mod.PROCESS_STARTED_AT,
    )  # fmt: skip

    assert left == [
        f"gate-{stamp}-4242-a1b2c3-2 (pending in {spool})",
        f"gate-{stamp}-4242-a1b2c3-1 (running in {spool})",
    ]
    record = {"after": drain_mod.AFTER_HARD_STOP, "jobs": left, "previous": {"pid": 4242}}
    assert "check jobs it left queued or running, whose checks run again" in (
        drain_mod.describe_restart(record)
    )


# ----------------------------------------------------------------------- the CLI


def test_the_relaunch_is_the_same_command_line_and_environment_without_clean(monkeypatch):
    seen = {}
    monkeypatch.setenv("MV2FLOW_TEST_VAR", "before")
    reexec = cli_mod._relauncher(
        ["/venv/bin/python", "-m", "agent_flow.workflows.trtllm_modeling_v2.cli",
         "--task", "t.yaml", "--clean", "--workspace", "ws"],
        os.environ,
    )  # fmt: skip
    monkeypatch.setenv("MV2FLOW_TEST_VAR", "after")
    monkeypatch.setattr(
        cli_mod.os, "execve", lambda path, argv, env: seen.update(path=path, argv=argv, env=env)
    )
    reexec()
    assert seen["path"] == sys.executable
    assert seen["argv"] == [
        sys.executable, "-m", "agent_flow.workflows.trtllm_modeling_v2.cli",
        "--task", "t.yaml", "--workspace", "ws",
    ]  # fmt: skip
    assert seen["env"]["MV2FLOW_TEST_VAR"] == "before"


# -------------------------------------------------------------------- the driver


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    (ws / "dispatch").mkdir(parents=True)
    (ws / "task.yaml").write_text(yaml.safe_dump({"trtllm_repo_path": str(tmp_path)}))
    write_json_atomic(
        ws / "dispatch" / ORCHESTRATOR_HEARTBEAT, {"pid": 7, "host": "h", "at": time.time()}
    )
    return ws


def test_driver_reload_asks_once_and_a_later_call_only_changes_how(workspace, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli_mod.main(["driver", "--workspace", str(workspace), "reload", "--by", "me"])
    assert exit_info.value.code == 0 and "requested" in capsys.readouterr().out
    first = reload_mod.read_request(workspace)
    assert first["by"] == "me" and not first["force"]
    status = driver.status(driver.load_config(workspace))
    assert status["reload"]["phase"] == "requested"

    cfg = driver.load_config(workspace)
    again = driver.request_reload(cfg, force=True, timeout_s=60)
    assert again["id"] == first["id"] and again["requested_at"] == first["requested_at"]
    assert again["force"] and again["timeout_s"] == 60

    with pytest.raises(SystemExit) as exit_info:
        cli_mod.main(["driver", "--workspace", str(workspace), "reload", "--cancel"])
    assert exit_info.value.code == 0 and f"withdrew reload {first['id']}" in capsys.readouterr().out
    assert reload_mod.read_request(workspace) is None


def test_driver_reload_is_refused_when_no_orchestrator_runs(workspace, capsys):
    write_json_atomic(
        workspace / "dispatch" / ORCHESTRATOR_HEARTBEAT,
        {"pid": 7, "host": "h", "at": time.time() - 3600},
    )
    with pytest.raises(SystemExit) as exit_info:
        cli_mod.main(["driver", "--workspace", str(workspace), "reload"])
    assert exit_info.value.code == 1
    assert "launching it again loads the current code" in capsys.readouterr().err
    assert reload_mod.read_request(workspace) is None


def test_status_shows_a_drain_and_what_the_orchestrator_holds(workspace):
    now = time.time()
    write_json_atomic(
        workspace / reload_mod.INFLIGHT_FILENAME,
        {"pid": 7, "host": "h", "started_at": now - 600, "stopped": None,
         "held": [{"kind": "check", "what": _BOOT, "since": now - 240}]},
    )  # fmt: skip
    reload_mod.request(workspace)
    request = reload_mod.read_request(workspace)
    reload_mod.write_record(
        workspace,
        {"id": request["id"], "phase": reload_mod.PHASE_DRAINING, "since": now - 300,
         "deadline": now + 1500, "waiting_for": [f"check {_BOOT}"]},
    )  # fmt: skip
    text = driver.format_status(driver.status(driver.load_config(workspace)))
    assert f"orchestrator holds: check {_BOOT} 4m" in text
    assert f"reload: draining (draining for 5m (deadline in 25m); waiting for check {_BOOT})" in (
        text
    )


@pytest.mark.parametrize(
    "record, kind",
    [
        ({"phase": reload_mod.PHASE_RELOADED, "reexec_at": 1.0, "drained_s": 3.0}, "RELOADED"),
        ({"phase": reload_mod.PHASE_FAILED, "error": "OSError: no"}, "RELOAD_FAILED"),
        (
            {"phase": reload_mod.PHASE_DRAINING, "overdue": True, "since": 1.0, "deadline": 2.0,
             "waiting_for": ["integration k46"]},
            "RELOAD_OVERDUE",
        ),
    ],
)  # fmt: skip
def test_wait_returns_each_reload_outcome_once(workspace, record, kind):
    reload_mod.write_record(workspace, {"id": "r1", **record})
    cfg = driver.load_config(workspace)
    event = driver.wait(cfg, timeout_s=5, poll_s=0.01, watch_sessions=False)
    assert event.kind == kind
    if kind == "RELOAD_OVERDUE":
        assert "integration k46" in event.detail and "reload --force" in event.detail
    again = driver.wait(cfg, timeout_s=0.05, poll_s=0.01, watch_sessions=False)
    assert again.kind == "TIMEOUT"


def test_the_wait_loop_documents_the_reload_events():
    from agent_flow.workflows.trtllm_modeling_v2.infra.driver import cli as driver_cli

    for event in ("RELOADED", "RELOAD_OVERDUE", "RELOAD_FAILED", "reload"):
        assert event in driver_cli.__doc__
