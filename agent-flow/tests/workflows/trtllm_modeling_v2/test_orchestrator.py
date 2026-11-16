# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run loop with no planner: tasks by code, one attempt counter, blocked receipts."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli, test_results
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import (
    RunEnvironment,
    resolve_environment,
)
from agent_flow.workflows.trtllm_modeling_v2.operators import rows as rows_mod
from agent_flow.workflows.trtllm_modeling_v2.run import escalation as esc
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator as orch
from agent_flow.workflows.trtllm_modeling_v2.run.execution import tasks as tasks_mod
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    Deployment,
    GraphOrchestrator,
    Services,
    compile_tasks,
    seed,
    stoppable,
)
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run.validation import regression
from agent_flow.workflows.trtllm_modeling_v2.run.workspace.integration import TaskNotStarted

_PKG = "tensorrt_llm/_torch/_experimental/modeling_v2"
_TARGET = f"{_PKG}/models/fam/newmodel__sm_100__unified"
_OWN = g.own_paths(_TARGET)

_CONTRACT = c.Contract(
    checks=(
        c.Check("core", "boot", "tp1"),
        c.Check("core", "accuracy[gsm8k]", "tp1"),
        c.Check("core", "serve", "tp1"),
    )
)
_KEYS = tuple(check.key for check in _CONTRACT.checks)
_CORE = Deployment("core", checks=_KEYS, gpus=1, wall=600)

_VOCABULARY = {
    "steps": [],
    "queue": [
        {"n": 1, "mode": "onboard", "target": "ops/norm/rms", "depends_on": [],
         "outcome": "an RMS norm entry at this checkpoint's epsilon",
         "search_questions": ["Which wrapper applies the norm in one call?"],
         "definitions": ["OpWrapper", "CELLS"], "leads": ["another target's norm"],
         "cells": [{"dtype": "bf16"}], "optional_cells": [{"dtype": "fp8"}]},
        {"n": 2, "mode": "verify", "target": "ops/attention/gqa", "depends_on": [],
         "surface": "head_dim 128", "cells": [{"head_dim": 128}]},
    ],
    "rows": [
        {"id": "weights", "kind": "weights", "outcome": "every checkpoint key is loaded or named",
         "required": ["no key is left unmapped"], "writes": [f"{_TARGET}/weights.py"],
         "check": {"id": "cpu:weights", "wall": 600}},
        {"id": "routing", "kind": "routing", "outcome": "the engine chooses the target",
         "required": ["the target is registered for this config"], "depends_on": ["weights"]},
    ],
}  # fmt: skip


def _survey():
    return g.Task(id="survey", kind="survey", goal="map the checkpoint", ref=_TARGET)


def _workspace(tmp_path, vocabulary=_VOCABULARY, graph=True):
    (tmp_path / "vocabulary.yaml").write_text(yaml.safe_dump(vocabulary), encoding="utf-8")
    if graph:
        seed(tmp_path, [_survey()])
    return tmp_path


def _services(*, decisions=None, failing=(), deployments=(_CORE,), work=None, **more):
    """Fake services: every worker says ``ready`` unless ``decisions`` says otherwise."""
    ran, started, log = [], [], []

    def check(chk):
        ran.append(chk.key)
        failed = failing(chk.key) if callable(failing) else chk.key in failing
        return SimpleNamespace(name=chk.gate, member=chk.member, passed=not failed,
                               detail="", tree_state="")  # fmt: skip

    def default_work(task):
        started.append((task.id, task.attempts))
        said = (decisions or {}).get(task.id, "ready")
        return said(task) if callable(said) else said

    # One state of the target for the whole run, unless a case says how it moves.
    more.setdefault("key", lambda check, commit="": "the target")
    services = Services(
        work=work or default_work,
        check=check,
        deployments=lambda: list(deployments),
        log=log.append,
        **more,
    )
    services.ran, services.started, services.logged = ran, started, log  # type: ignore[attr-defined]
    return services


def _orchestrator(tmp_path, services, contract=_CONTRACT, **kw):
    kw.setdefault("poll_s", 0.01)
    kw.setdefault("own_paths", _OWN)
    return GraphOrchestrator(services, contract, c.ContractStatus.load(tmp_path), tmp_path, **kw)


def _waiting(tmp_path):
    """What waits for an answer in the saved graph: each blocked task, then the run's own question."""
    graph = g.Graph.load(tmp_path / g.GRAPH_FILENAME)
    found = [
        {"id": t.id, "task": t.id, "source": t.block.get("by"),
         "blocker_kind": t.block.get("kind"), "observed": t.last,
         "next_question": t.block.get("question", ""), "blocks": g.blocked_by(graph, t.id)}
        for t in graph.tasks
        if t.status == "blocked"
    ]  # fmt: skip
    asked = graph.run_blocked
    if asked:
        found.append({"id": "run", "task": "", "blocker_kind": asked.get("kind"),
                      "observed": asked.get("observed"), "next_question": asked.get("question")})  # fmt: skip
    return found


def _answer(tmp_path, name, verb, args):
    """Leave an answer as `resolve` does beside a running controller."""
    task = g.Graph.load(tmp_path / g.GRAPH_FILENAME).by_id().get(name)
    return esc.leave_answer(tmp_path, name, verb, args, task.attempts if task else 0)


def _statuses(tmp_path):
    return {t.id: t.status for t in g.Graph.load(tmp_path / g.GRAPH_FILENAME).tasks}


def test_rows_and_deployments_make_the_task_list_with_no_planner_and_it_is_delivered(tmp_path):
    seen = {}

    def skeleton(files):
        # Before any row's task starts: nothing but the survey has run yet.
        # Asked as the task list is made, and again as each task starts: the first is kept.
        seen.setdefault("before", list(services.started))
        seen.setdefault("files", list(files))
        return "committed"

    services = _services(skeleton=skeleton)
    orchestrator = _orchestrator(_workspace(tmp_path), services)
    assert orchestrator.run() == "delivered"
    by_id = orchestrator.graph.by_id()
    # The survey's rows, the deployment's own two, and the delivery's task behind the session.
    assert list(by_id) == ["survey", "k1-rms", "k2-gqa", "weights", "routing",
                           "assemble-core", "session-core", "delivery-files"]  # fmt: skip
    # An operator row is a kernel task on its entry and test; any other row a task of its kind.
    assert by_id["k1-rms"].kind == "kernel" and by_id["k1-rms"].check == "kernel:1"
    assert by_id["k1-rms"].touches == [
        f"{_TARGET}/ops/norm/rms.py",
        "tests/unittest/_torch/modeling_v2/models/fam/newmodel__sm_100__unified/norm/"
        "test_modeling_v2_rms.py",
    ]
    assert by_id["weights"].kind == "weights" and by_id["weights"].check == "cpu:weights"
    assert by_id["routing"].depends_on == ["survey", "weights"]
    # The assembly waits for every row of the target; its session is a check with no agent.
    assert by_id["assemble-core"].depends_on == ["survey", "k1-rms", "k2-gqa", "weights", "routing"]
    session = by_id["session-core"]
    assert session.kind == g.CHECK_KIND and session.satisfies == list(_KEYS)
    assert session.depends_on == ["survey", "assemble-core"] and session.check == "deployment:core"
    # The skeleton is committed before the fan-out, for the files the rows own.
    assert seen["before"] == [("survey", 1)]
    assert (
        f"{_TARGET}/ops/norm/rms.py" in seen["files"] and f"{_TARGET}/weights.py" in seen["files"]
    )
    # Every check the session claims ran once, as its task; the final pass found each
    # with a row under the candidate's key and measured none again.
    assert sorted(services.ran) == sorted(_KEYS)
    assert any(
        line.startswith("[final] candidate") and "measured now: none" in line
        for line in services.logged
    )
    # Every row a task works on is kept for its card, the deployment's own among them.
    kept = {row.id: row for row in rows_mod.load_work(tmp_path)}
    assert set(kept) == {"1", "2", "weights", "routing", "assemble-core", "delivery-files"}
    assert kept["1"].optional == ("dtype: fp8",)
    # What to look for is one list, whichever keys the survey sorted it under.
    assert kept["1"].leads == (
        "another target's norm", "`OpWrapper`", "`CELLS`", "Which wrapper applies the norm in one call?",
    )  # fmt: skip
    assert kept["2"].required[0] == "surface: head_dim 128"


def test_the_rows_tasks_run_side_by_side_on_one_counter_of_agent_turns(tmp_path):
    gate, peak, lock = threading.Event(), [0, 0], threading.Lock()

    def work(task):
        if task.kind != "survey":
            with lock:
                peak[0] += 1
                peak[1] = max(peak[1], peak[0])
            gate.wait(2)
            with lock:
                peak[0] -= 1
        return "ready"

    services = _services(work=work)
    orchestrator = _orchestrator(_workspace(tmp_path), services, max_parallel=3)
    threading.Timer(0.4, gate.set).start()
    assert orchestrator.run() == "delivered"
    # Both kernel rows and the weights row ran together, and never more than the counter.
    assert peak[1] == 3


def test_a_blocked_receipt_opens_an_escalation_while_its_sibling_goes_on(tmp_path):
    receipt = {"blocker_kind": "engine_defect", "command": "pytest tests/x.py::t", "result": "1 failed",
               "tried": ["a guard on the input"], "ruled_out": ["the test's own setup"],
               "next_question": "is the cached-prefix path meant to take int64?"}  # fmt: skip
    services = _services(
        decisions={"k1-rms": "blocked\nthe wrapped op fails at the largest row"},
        receipt=lambda task: receipt if task.id == "k1-rms" else {},
    )
    orchestrator = _orchestrator(_workspace(tmp_path), services)
    assert orchestrator.run() == "escalated"
    statuses = _statuses(tmp_path)
    # The sibling rows finished; only what depends on the blocked one waits.
    assert statuses["k2-gqa"] == statuses["weights"] == statuses["routing"] == "done"
    assert statuses["k1-rms"] == "blocked" and statuses["assemble-core"] == "pending"
    (record,) = _waiting(tmp_path)
    assert record["task"] == "k1-rms" and record["source"] == "worker"
    assert record["blocks"] == ["assemble-core", "session-core", "delivery-files"]
    # The kind and the question are the worker's; what it tried stays in its receipt.
    assert record["blocker_kind"] == "engine_defect"
    assert record["next_question"].startswith("is the cached-prefix path")
    assert "the wrapped op fails at the largest row" in record["observed"]
    # The outer session answers; the same command then finishes the run.
    _answer(tmp_path, "k1-rms", "retry", {"note": "int64 is right: cast"})
    services = _services()
    resumed = _orchestrator(tmp_path, services)
    assert resumed.run() == "delivered"
    assert services.started[0] == ("k1-rms", 2)
    assert esc.read_answer(tmp_path, "k1-rms")["applied"] is True and _waiting(tmp_path) == []
    assert "int64 is right: cast" in (tmp_path / esc.NOTES_FILENAME).read_text()


def test_a_resolution_is_applied_by_the_running_controller_while_other_tasks_go_on(tmp_path):
    release = threading.Event()

    def work(task):
        if task.id == "k1-rms" and task.attempts == 1:
            return "blocked\nno such op"
        if task.id == "k2-gqa":
            # Still running while the escalation is open and while it is resolved.
            assert release.wait(5)
        return "ready"

    services = _services(work=work)
    orchestrator = _orchestrator(_workspace(tmp_path), services)

    def outer_session():
        deadline = time.monotonic() + 5
        while not _waiting(tmp_path):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        left = _answer(tmp_path, "k1-rms", "retry", {"note": "use the other op"})
        assert esc.wait_settled(tmp_path, "k1-rms", left["at"], 5, 0.01)["applied"] is True
        release.set()

    watcher = threading.Thread(target=outer_session)
    watcher.start()
    assert orchestrator.run() == "delivered"
    watcher.join()
    assert orchestrator.graph.by_id()["k1-rms"].attempts == 2
    assert esc.pending_answers(tmp_path) == []


def test_a_controller_with_everything_left_blocked_stays_for_an_answer_and_takes_it(tmp_path):
    """A whole run's controller left the moment its last row blocked; the answer came 28 s later."""

    def work(task):
        return "blocked\nno such op" if task.id == "k1-rms" and task.attempts == 1 else "ready"

    def stays(services):
        return [line for line in services.logged if line.startswith("[loop] everything left waits")]

    services = _services(work=work)
    orchestrator = _orchestrator(_workspace(tmp_path), services, answer_wait_s=60)

    def outer_session():
        # It answers once nothing else goes on: before this change no controller was left.
        deadline = time.monotonic() + 5
        while not stays(services):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        left = _answer(tmp_path, "k1-rms", "retry", {"note": "use the other op"})
        assert esc.wait_settled(tmp_path, "k1-rms", left["at"], 5, 0.01)["applied"] is True

    watcher = threading.Thread(target=outer_session)
    watcher.start()
    # The same controller takes the answer and ends the run: none is started again.
    assert orchestrator.run() == "delivered"
    watcher.join()
    (said,) = stays(services)
    assert "for an answer (k1-rms): this controller is alive and takes one until " in said
    assert "(1 min), then leaves with exit 76" in said
    assert orchestrator.graph.by_id()["k1-rms"].attempts == 2
    # With no answer it leaves as it did, that much later, having said so once.
    (tmp_path / "alone").mkdir()
    unanswered = _services(decisions={"k1-rms": "blocked\nno such op"})
    alone = _orchestrator(_workspace(tmp_path / "alone"), unanswered, answer_wait_s=0.3)
    began = time.monotonic()
    assert alone.run() == "escalated" and time.monotonic() - began >= 0.3
    assert len(stays(unanswered)) == 1


@pytest.mark.parametrize("verb", ["retry", "skip"])
def test_a_task_in_its_turn_is_answered_and_its_turn_is_ended_first(tmp_path, verb):
    """A worker left the purpose for two hours, and `resolve skip` said "not blocked (it is running)"."""
    from agent_flow.workflows.trtllm_modeling_v2.agents.turns import TurnEnded

    in_turn, ended, seen, settled = threading.Event(), threading.Event(), [], []

    def work(task):
        seen.append((task.id, task.attempts))
        if task.id == "weights" and task.attempts == 1:
            # A turn that never ends by itself: only the controller ends it.
            in_turn.set()
            assert ended.wait(5), "nothing ended the turn"
            raise TurnEnded("003-worker: the turn was ended before it answered.")
        return "ready"

    def end_turn(task):
        was = task.id == "weights" and in_turn.is_set() and not ended.is_set()
        ended.set()
        return was

    services = _services(work=work, end_turn=end_turn)
    orchestrator = _orchestrator(_workspace(tmp_path), services)

    def outer_session():
        assert in_turn.wait(5)
        left = _answer(tmp_path, "weights", verb, {"note": "load the keys as they are named"})
        settled.append(esc.wait_settled(tmp_path, "weights", left["at"], 5, 0.01))

    watcher = threading.Thread(target=outer_session)
    watcher.start()
    verdict = orchestrator.run()
    watcher.join()
    (answer,) = settled
    weights = orchestrator.graph.by_id()["weights"]
    # The turn was ended once, and what became of the attempt is the answer: no failure
    # of the attempt was recorded, and none of the task's attempts was spent on it.
    assert answer["applied"] is True
    assert sum("the turn is ended" in line for line in services.logged) == 1
    assert not any(line.startswith("[retry] weights") for line in services.logged)
    if verb == "retry":
        assert verdict == "delivered" and "weights run again with the note" in answer["said"]
        assert [a for tid, a in seen if tid == "weights"] == [1, 2]
        assert (weights.status, weights.attempts, weights.budget) == ("done", 2, g.MAX_ATTEMPTS + 1)
    else:
        # Skipped: reported not done, never run again, and what waited for it does not start.
        assert "weights is skipped" in answer["said"] and verdict != "delivered"
        assert [a for tid, a in seen if tid == "weights"] == [1]
        assert (weights.status, weights.attempts) == ("skipped", 1)
        assert _statuses(tmp_path)["routing"] == "pending"
    assert esc.pending_answers(tmp_path) == []


def test_one_attempt_counter_a_turn_that_raised_runs_again_and_then_is_blocked(tmp_path):
    def raises(task):
        raise TimeoutError("no answer from the driver")

    services = _services(decisions={"weights": raises})
    orchestrator = _orchestrator(_workspace(tmp_path), services, max_attempts=2)
    assert orchestrator.run() == "escalated"
    weights = orchestrator.graph.by_id()["weights"]
    assert [a for tid, a in services.started if tid == "weights"] == [1, 2]
    assert (weights.status, weights.attempts, weights.budget) == ("blocked", 2, 2)
    (record,) = _waiting(tmp_path)
    assert record["source"] == "gate" and record["blocker_kind"] == "attempts_exhausted"
    assert "TimeoutError: no answer from the driver" in record["observed"]
    # The loop opened it, so the loop asks: every verb that answers it is named.
    asked = record["next_question"]
    assert asked.startswith("`weights` used its 2 attempt(s)")
    assert all(
        verb in asked
        for verb in ("resolve retry weights --note", "change-requirement weights", "skip weights")
    )
    # So does every block the loop makes itself: each kind names a verb that exists.
    for kind, text in orch.ASKS.items():
        named = [verb for verb in esc.VERBS if f"`resolve {verb}" in text]
        assert named and "{task}" in text, kind


def test_a_tree_that_could_not_be_prepared_spends_no_attempt_and_goes_to_the_outer_session(
    tmp_path,
):
    def no_tree(task):
        raise TaskNotStarted("worktree checkout: Disk quota exceeded")

    services = _services(decisions={"weights": no_tree})
    orchestrator = _orchestrator(_workspace(tmp_path), services)
    assert orchestrator.run() == "escalated"
    weights = orchestrator.graph.by_id()["weights"]
    assert (weights.status, weights.attempts) == ("blocked", 1)
    assert _waiting(tmp_path)[0]["blocker_kind"] == "tree_not_prepared"


def test_a_session_that_fails_goes_back_to_the_assembly_and_runs_again_after_it(tmp_path):
    attempts = []

    def failing(key):
        # The first session fails its accuracy; the one after the repair passes.
        return key == "core:accuracy[gsm8k]@tp1" and attempts.count("assemble-core") < 2

    def work(task):
        attempts.append(task.id)
        return "ready"

    services = _services(work=work, failing=failing)
    orchestrator = _orchestrator(_workspace(tmp_path), services)
    assert orchestrator.run() == "delivered"
    by_id = orchestrator.graph.by_id()
    assert by_id["assemble-core"].attempts == 2 and by_id["session-core"].attempts == 2
    assert attempts.count("assemble-core") == 2 and attempts.count("k1-rms") == 1
    assert any("back to assemble-core" in line for line in services.logged)


_DECODE = {"id": "decode-graphs", "kind": "feature", "needs": "model",
           "outcome": "a decode step replays a captured graph",
           "required": ["the second decode step runs no eager kernel"],
           "writes": [f"{_TARGET}/decode.py"]}  # fmt: skip
_WARMUP = {"id": "graph-warmup", "kind": "feature", "depends_on": ["decode-graphs"],
           "outcome": "the graphs are captured at start", "required": ["no capture at request time"]}  # fmt: skip


def test_the_files_the_delivery_requires_for_ci_get_a_task_by_code(tmp_path):
    """Known from the spec before the survey: no survey has to think of them."""
    test = "tests/integration/defs/accuracy/modeling_v2/fam/target/test_accuracy.py"
    lists = "tests/integration/test_lists/test-db"
    first, wide = Deployment("core", checks=("core:deployment@tp1",)), Deployment("wide")
    compiled = compile_tasks([], [first, wide], own_paths=_OWN, ci=(test, lists))
    by_id = {t.id: t for t in compiled.tasks}
    task = by_id[rows_mod.DELIVERY_ROW]
    # One task, after the first deployment's session and beside the further deployments:
    # nothing of the delivery's form stands before `onboarded`. Its write scope is the two
    # places and the target's own files, where it repairs what a landing recorded. It
    # names no check of the gate's (its card then gives no command to run): the landing
    # asks the delivery's own rules of its files.
    assert task.depends_on == ["session-core"] and task.kind == "feature"
    assert by_id["assemble-wide"].depends_on == ["session-core"]
    assert task.touches == [test, lists, *_OWN] and task.check == ""
    row = next(r for r in compiled.rows if r.id == rows_mod.DELIVERY_ROW)
    said = " ".join(row.required)
    assert f"`{test}` exists" in said and f"a list under `{lists}/` names it" in said
    assert "it is not run here, and you do not load the model to try it" in said
    assert "each file your dispatch lists as breaking a rule of the delivery" in said
    assert compiled.unplaced == {}
    # A survey row that names the test's file takes nothing from the task and does not
    # get the file: a run's survey wrote such a row, its own check ran the accuracy test
    # on no device, and the delivery found the list and the anchor's row missing.
    owned = rows_mod.Row(
        "ci-accuracy-test", "feature", "the accuracy test", writes=(test, "package/contracts.py")
    )
    again = {
        t.id: t for t in compile_tasks([owned], [first], own_paths=_OWN, ci=(test, lists)).tasks
    }
    assert again[rows_mod.DELIVERY_ROW].touches == [test, lists, *_OWN]
    assert again["ci-accuracy-test"].touches == ["package/contracts.py"]
    assert f"`{test}` is not in your write scope: the task `delivery-files`" in (
        again["ci-accuracy-test"].goal
    )
    # The third file the delivery requires for CI is where it reads the anchor's row: the
    # task holds it and its card names the rule, whoever owns the test.
    reference = "tests/integration/defs/accuracy/references/gsm8k.yaml"
    three = compile_tasks([], [first], own_paths=_OWN, ci=(test, lists, reference))
    told = next(r for r in three.rows if r.id == rows_mod.DELIVERY_ROW)
    assert list(told.writes) == [test, lists, reference, *_OWN]
    assert {t.id: t for t in three.tasks}[rows_mod.DELIVERY_ROW].touches == list(told.writes)
    assert f"`{reference}` has an entry under the model's key" in " ".join(told.required)
    beside = compile_tasks([owned], [first], own_paths=_OWN, ci=(test, lists, reference))
    assert {t.id: t for t in beside.tasks}[rows_mod.DELIVERY_ROW].touches == list(told.writes)
    # So does a run that owes no CI file (no deployment scores a benchmark). A run with
    # no deployment has no target to deliver, and no such task.
    alone = {t.id: t for t in compile_tasks([], [first], own_paths=_OWN).tasks}
    assert alone[rows_mod.DELIVERY_ROW].touches == list(_OWN)
    assert "accuracy test" not in alone[rows_mod.DELIVERY_ROW].goal
    assert compile_tasks([], [], own_paths=_OWN, ci=(test, lists)).tasks == []
    # Where the run may not commit them, the task is blocked on its write scope at once.
    kept_out = compile_tasks(
        [], [first], own_paths=_OWN, allowed_paths=["package/"], ci=(test, lists)
    )
    assert "outside the paths the run may commit" in kept_out.unplaced[rows_mod.DELIVERY_ROW]
    # Its card: what the delivery checks of it.
    rows_mod.save_work(tmp_path, compiled.rows)
    kept = rows_mod.row_of(tmp_path, rows_mod.DELIVERY_ROW)
    assert kept.check == "" and kept.required == row.required
    from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import card as card_mod

    text = card_mod.render_card(task, kept, command="", notes=tmp_path / "n.md", spec="task.yaml")
    assert "a list under" in text and "it is not run here" in text
    # It runs no job, so it asks the pool for no holder while it is due.
    orchestrator = _orchestrator(_workspace(tmp_path, graph=False), _services())
    task.status, task.depends_on = "pending", []
    orchestrator.graph.tasks[:] = [task]
    assert orchestrator.demand() == {}
    # And the report does not call it landed on its worker's receipt.
    from agent_flow.workflows.trtllm_modeling_v2.run.records import report

    task.status = "done"
    g.Graph([task]).save(tmp_path / g.GRAPH_FILENAME)
    assert not [line for line in report.not_verified(tmp_path) if rows_mod.DELIVERY_ROW in line]


def test_a_row_whose_check_needs_the_assembled_model_comes_after_the_assembly(tmp_path):
    """A whole run placed a decode row before the assembly its test builds: six tasks waited on it."""
    vocabulary = {**_VOCABULARY, "rows": [*_VOCABULARY["rows"], _DECODE, _WARMUP]}
    order, held = [], {}

    def work(task):
        order.append(task.id)
        held[task.id] = orchestrator.holds_onboarded(task.id)
        return "ready"

    services = _services(work=work)
    checked = services.check
    services.check = lambda chk: order.append(chk.key) or checked(chk)
    orchestrator = _orchestrator(_workspace(tmp_path, vocabulary), services)
    assert orchestrator.run() == "delivered"
    by_id = orchestrator.graph.by_id()
    decode, warmup, assembly = by_id["decode-graphs"], by_id["graph-warmup"], by_id["assemble-core"]
    # The assembly does not wait for it, nor for the row that waits for it: they wait for
    # the assembly. And the first deployment's session waits for neither: it is the
    # milestone, and a whole run's sat eighteen minutes behind such a row. They start
    # once it passed (`onboarded`), as a further deployment's tasks do.
    assert assembly.depends_on == ["survey", "k1-rms", "k2-gqa", "weights", "routing"]
    assert decode.depends_on == ["survey", "assemble-core", "session-core"]
    assert warmup.depends_on == ["survey", "decode-graphs", "assemble-core", "session-core"]
    assert by_id["session-core"].depends_on == ["survey", "assemble-core"]
    # While the milestone was to come, the assembly's landing was all it waited for, and
    # no other task's; once the session passed, nothing holds it any more.
    assert [task for task, waited in held.items() if waited] == ["assemble-core"]
    assert set(held) >= {"k1-rms", "weights", "decode-graphs", "graph-warmup"}
    assert not orchestrator.holds_onboarded("assemble-core")
    first_check = min(order.index(key) for key in _KEYS)
    assert order.index("assemble-core") < first_check < order.index("decode-graphs")
    assert order.index("decode-graphs") < order.index("graph-warmup")
    # What such a row changed of the model is the final pass's to measure: the key of the
    # session's row is asked again on the candidate.
    assert orchestrator.final is not None and orchestrator.final.evaluation().holds
    # The model's files are in its write scope, and its card's outcome says why; a row
    # that only waits for it keeps to its own files.
    assert f"{_TARGET}/decode.py" in decode.touches and set(_OWN) <= set(decode.touches)
    assert decode.goal.startswith("a decode step replays a captured graph. Your check runs the")
    assert "the target's own files are in your write scope" in decode.goal
    assert not set(_OWN) & set(warmup.touches) and warmup.goal == _WARMUP["outcome"]
    assert rows_mod.row_of(tmp_path, "decode-graphs").needs == "model"


def test_a_task_that_is_done_may_still_ask_and_its_question_holds_nothing(tmp_path):
    asks = "change-requirement: the anchor of `core` on gsm8k reads 61.77, the reference row 41.77"
    services = _services(
        receipt=lambda task: {"next_question": asks} if task.id == "weights" else {}
    )
    orchestrator = _orchestrator(_workspace(tmp_path), services)
    # Nothing waited for it: every task is done, and the run closes.
    assert orchestrator.run() == "delivered"
    assert set(_statuses(tmp_path).values()) == {"done"} and _waiting(tmp_path) == []
    saved = g.Graph.load(tmp_path / g.GRAPH_FILENAME)
    assert saved.by_id()["weights"].block == {"by": "worker", "kind": "question", "question": asks}
    assert esc.waiting(saved) == [f"weights (weights), done, asks and holds nothing: {asks}"]
    assert any(line.startswith("[asks] weights (weights), done") for line in services.logged)


def test_a_state_row_is_judged_by_its_own_test_whatever_its_deployment():
    """A row is judged by its own test.

    A whole run: a `state` row with no check of its own got a gate that named a deployment
    the spec did not have, and its worker stopped before any job.
    """
    state = rows_mod.Row("kv-state", "state", "the cache keeps its pages", required=("r",),
                         check=rows_mod.check_id("state", "kv-state"), gpus=1)  # fmt: skip
    assert state.check == "cpu:kv-state"
    pair = Deployment("pd", kind=orch.DISAGGREGATED, checks=("pd:deployment@pd2",))
    for deployments in ([_CORE], [pair]):
        compiled = compile_tasks([state], deployments, target_rel=_TARGET, own_paths=_OWN)
        task = next(t for t in compiled.tasks if t.id == "kv-state")
        assert task.check == "cpu:kv-state" and rows_mod.own_test("kv-state", _OWN) in task.touches
        (kept,) = [r for r in compiled.rows if r.id == "kv-state"]
        assert (kept.check, kept.gpus) == ("cpu:kv-state", 1)


def test_a_routing_rows_scope_is_the_familys_routing_and_a_new_familys_registration(tmp_path):
    """Assessed on a large model of a new family: its routing row was given a file in the target.

    The delivery reads the family's `routing.py`, and the engine finds a family through
    the one line that imports it in the models' package. A row that named no file got
    neither, and a worker that wrote them was outside its scope.
    """
    from agent_flow.workflows.trtllm_modeling_v2.checks import delivery

    models = tmp_path / _PKG / "models"
    (models / "known").mkdir(parents=True)
    registry = f"{_PKG}/models/__init__.py"
    (tmp_path / registry).write_text(
        '"""Families."""\n\nfrom .known import routing as known_routing  # noqa: F401\n'
    )
    row = rows_mod.Row("routing", "routing", "the engine chooses the target.", required=("r",),
                       check="cpu:routing")  # fmt: skip

    def task(family, **stated):
        target = f"{_PKG}/models/{family}/small__sm_100__unified"
        own = g.own_paths(target)
        compiled = compile_tasks([replace(row, **stated)], [_CORE], target_rel=target,
                                 own_paths=own, repo_root=str(tmp_path))  # fmt: skip
        return next(t for t in compiled.tasks if t.id == "routing"), own

    # A family the package does not import yet: its routing file and the line that registers it.
    fresh, own = task("fresh")
    family = f"{_PKG}/models/fresh/routing.py"
    assert fresh.touches == [family, registry, rows_mod.own_test("routing", own)]
    assert f"through its family's routing, `{family}`" in fresh.goal
    assert f"registered by its one line in `{registry}`" in fresh.goal
    assert fresh.goal.endswith("Your write scope holds both")
    # A family that is registered: its routing file, and the package is left alone.
    known, own = task("known")
    assert known.touches == [f"{_PKG}/models/known/routing.py", rows_mod.own_test("routing", own)]
    assert registry not in known.goal and known.goal.endswith("Your write scope holds it")
    # What a row names stays first, and the delivery's places follow.
    named = (f"{_PKG}/models/fresh/select.py", "tests/unittest/test_select.py")
    stated, own = task("fresh", writes=named)
    assert stated.touches == [*named, family, registry]
    # No tree to read (a plan made before the checkout is known): the family's file alone.
    assert delivery.routing_places("", f"{_PKG}/models/fresh/small") == (family,)
    assert delivery.routing_places(tmp_path, "elsewhere/small") == ()


def test_where_the_delivery_looks_for_an_entrys_test_is_in_the_tasks_write_scope():
    """A whole run: the entry's cases moved where the delivery wants them, and the scope refused.

    The row's `writes` named the entry and a test file of its own choosing; the delivery
    matches an entry's test by its place and name. The task could obey one of the flow's
    rules or the other. The place is known when the task is made.
    """
    from agent_flow.workflows.trtllm_modeling_v2.checks import delivery

    tests = _OWN[1].rstrip("/")
    entry = f"{_TARGET}/ops/attention/paged_kv_binding.py"
    wanted = f"{tests}/attention/test_modeling_v2_paged_kv_binding.py"
    rows = [
        rows_mod.Row("state", "feature", "the cache binds pages.", required=("r",),
                     writes=(entry, f"{tests}/test_cache.py"), check="cpu:state"),
        rows_mod.Row("glue", "feature", "glue", required=("r",),
                     writes=(f"{_TARGET}/glue.py",), check="cpu:glue"),
    ]  # fmt: skip
    compiled = compile_tasks(rows, [_CORE], target_rel=_TARGET, own_paths=_OWN)
    by_id = {t.id: t for t in compiled.tasks}
    state, glue = by_id["state"], by_id["glue"]
    # What the row named stays first; the delivery's places for its entry follow.
    assert state.touches[:2] == [entry, f"{tests}/test_cache.py"]
    assert state.touches[2:] == [
        wanted,
        f"{tests}/attention/test_modeling_v2_paged_kv_binding_op_matrix.py",
        f"{tests}/attention/_paged_kv_binding_op_matrix.py",
    ]
    assert state.goal.startswith("the cache binds pages. The delivery looks for an entry's test")
    assert f"`{entry}` is tested in `{wanted}`" in state.goal
    # The row itself is kept as the survey wrote it: its key does not move with the card.
    assert [r.outcome for r in compiled.rows if r.id == "state"] == ["the cache binds pages."]
    # A row that writes no entry is told nothing and given nothing.
    assert glue.touches == [f"{_TARGET}/glue.py", rows_mod.own_test("glue", _OWN)]
    assert glue.goal == "glue"
    # A category's directory gets the directory its tests mirror it in; the catalog's
    # entry its catalog test; a helper, an initializer and a file elsewhere nothing.
    places = lambda path: delivery.entry_test_places("", _TARGET, path)  # noqa: E731
    assert places(f"{_TARGET}/ops/attention/") == (f"{tests}/attention/",)
    assert places(f"{_PKG}/catalog/norm/rms.py")[0] == (
        f"{delivery.CATALOG_TESTS_DIR}/norm/test_modeling_v2_rms.py"
    )
    for path in (f"{_TARGET}/ops/attention/_shared.py", f"{_TARGET}/ops/attention/__init__.py",
                 f"{_TARGET}/modeling.py", f"{_TARGET}/ops/", "tensorrt_llm/other.py"):  # fmt: skip
        assert places(path) == (), path


def test_a_row_names_the_deployment_whose_model_it_needs_and_a_wrong_word_is_refused():
    rows = [
        rows_mod.Row("decode", "feature", "x", deployment="pd", required=("r",), needs="model",
                     check="cpu:decode"),
        rows_mod.Row("plain", "feature", "y", deployment="pd", required=("r",), check="cpu:plain"),
    ]  # fmt: skip
    deployments = [
        Deployment("core", checks=("core:deployment@tp1",)),
        Deployment("pd", kind=orch.DISAGGREGATED, checks=("pd:deployment@pd2",)),
    ]
    by_id = {t.id: t for t in compile_tasks(rows, deployments, own_paths=_OWN).tasks}
    # After both sides of the deployment it names, before its one session; a row of the
    # same deployment that needs nothing is still what the two sides wait for.
    assert by_id["decode"].depends_on == ["session-core", "pd-ctx-pd", "pd-gen-pd"]
    assert by_id["pd-hop-pd"].depends_on == ["pd-ctx-pd", "pd-gen-pd", "decode"]
    assert (
        "plain" in by_id["pd-ctx-pd"].depends_on and "decode" not in by_id["pd-ctx-pd"].depends_on
    )
    assert g.validate(g.Graph(list(by_id.values()))) == []
    # With no deployment to assemble there is nothing to come after: the row is as any other.
    alone = compile_tasks(rows[:1], [], own_paths=_OWN).tasks
    assert [t.id for t in alone] == ["decode"] and alone[0].goal == "x"
    # The key takes one word.
    entry = {"id": "decode", "kind": "feature", "outcome": "x", "required": ["r"]}
    assert rows_mod.typed_rows({"rows": [{**entry, "needs": "model"}]})[0].needs == "model"
    assert rows_mod.typed_rows({"rows": [entry]})[0].needs == ""
    with pytest.raises(rows_mod.RowError, match=r"rows\[0\].needs is 'the assembly'; say `model`"):
        rows_mod.typed_rows({"rows": [{**entry, "needs": "the assembly"}]})


def _not_observed(services, key, why):
    """Make ``key`` end as the ledger's third outcome: nothing failed, and ``why`` it certifies nothing."""
    checked = services.check

    def check(chk):
        result = checked(chk)
        result.unverified = why if chk.key == key else ""
        return result

    services.check = check
    return services


def test_a_check_not_verified_ends_its_task_done_and_the_run_partial_with_no_escalation(tmp_path):
    key, why = "core:accuracy[gsm8k]@tp1", "the endpoint returned no scores to judge"
    pinned = []
    services = _not_observed(_services(onboarded=lambda task: pinned.append(task)), key, why)
    orchestrator = _orchestrator(_workspace(tmp_path), services)
    assert orchestrator.run() == "partial"
    # The task that owns the check is done, and says which of its checks was not verified.
    session = orchestrator.graph.by_id()["session-core"]
    assert session.status == "done" and session.attempts == 1
    assert session.unverified == [f"{key}: {why}"]
    saved = g.Graph.load(tmp_path / g.GRAPH_FILENAME).by_id()
    assert saved["session-core"].unverified == [f"{key}: {why}"]
    assert saved["k1-rms"].unverified == [] and "unverified" not in saved["k1-rms"].as_dict()
    # Nothing is escalated, nothing is repaired, and the check ran once on that target.
    assert _waiting(tmp_path) == [] and _statuses(tmp_path)["assemble-core"] == "done"
    assert orchestrator.graph.by_id()["assemble-core"].attempts == 1
    assert services.ran.count(key) == 1
    assert any(f"NOT VERIFIED: {key}: {why}" in line for line in services.logged)
    assert any(line.startswith("[partial] every task is done") for line in services.logged)
    # Onboarded is pinned on the first deployment's session all the same, with what it says.
    assert [task.id for task in pinned] == ["session-core"]
    assert pinned[0].unverified == [f"{key}: {why}"]
    # A later start finds the same verdict on the same target: it does not measure again,
    # and what was published is not published a second time (the task keeps `onboarded`).
    again = _not_observed(_services(onboarded=lambda task: pinned.append(task)), key, why)
    assert _orchestrator(tmp_path, again).run() == "partial" and again.ran == []
    assert len(pinned) == 1 and saved["session-core"].onboarded


def test_a_landing_whose_check_could_not_verify_is_done_as_such_and_the_run_is_partial(tmp_path):
    why = "kernel:1: the row names no optional cell to observe"
    holder = {}

    def work(task):
        if task.id == "k1-rms":
            # The landing says what its card's check could not verify (``tasks._work_and_land``).
            holder["orchestrator"].mark_integrated(task.id, task.attempts, why)
        return "ready"

    services = _services(work=work)
    orchestrator = holder["orchestrator"] = _orchestrator(_workspace(tmp_path), services)
    # Every contract check passed, and the run is not delivered: one landing is not verified.
    assert orchestrator.run() == "partial"
    row = g.Graph.load(tmp_path / g.GRAPH_FILENAME).by_id()["k1-rms"]
    assert row.status == "done" and row.attempts == 1 and row.unverified == [why]
    assert _waiting(tmp_path) == [] and set(_statuses(tmp_path).values()) == {"done"}
    assert any(f"NOT VERIFIED: {why}" in line for line in services.logged)
    assert any(line.startswith("[partial]") and why in line for line in services.logged)


def test_a_session_that_keeps_failing_is_blocked_once_the_assembly_has_no_attempt_left(tmp_path):
    # Another check left not verified changes nothing: a check that fails is never partial.
    services = _not_observed(
        _services(failing={"core:boot@tp1"}), "core:accuracy[gsm8k]@tp1", "no scores"
    )
    orchestrator = _orchestrator(_workspace(tmp_path), services, max_attempts=2)
    assert orchestrator.run() == "escalated"
    by_id = orchestrator.graph.by_id()
    assert by_id["assemble-core"].attempts == 2 and by_id["assemble-core"].status == "done"
    assert by_id["session-core"].status == "blocked"
    (record,) = _waiting(tmp_path)
    assert record["task"] == "session-core" and record["blocker_kind"] == "check_failed"
    assert "core:boot@tp1" in record["observed"]


def test_work_that_cannot_start_before_its_hosts_job_ends_is_continued_at_once(
    tmp_path, monkeypatch
):
    """On one host inside a job, a task waited until the job's limit on a job that could not fit."""
    workspace = _workspace(tmp_path)
    waiting, release, told = threading.Event(), threading.Event(), []
    late = "job session-core states a wall of 2h and this host's watcher ends in 1h"

    def held(key):
        # The session's job waits in the spool: nothing serves it before the job ends.
        waiting.set()
        release.wait(5)
        return False

    with stoppable(workspace) as stop:

        def continue_now(where, why):
            told.append((where, why))
            stop(f"{why}: stopped for the limit")
            release.set()
            return True

        monkeypatch.setattr(orch.controller_mod, "continue_now", continue_now)
        # The pool says so from the moment that job is ready, and not while a row's turn runs.
        services = _services(failing=held, late=lambda: late if waiting.is_set() else "")
        assert _orchestrator(workspace, services).run() == f"stopped: {late}: stopped for the limit"
    # Said once, as the limit's own stop: every row's turn was over, only the session waited.
    assert told == [(workspace, late)]
    assert services.logged.count(f"[stop] to be continued: {late}") == 1
    assert _statuses(tmp_path)["assemble-core"] == "done"
    # A run that is not in such a place is told nothing: the service's answer is "".
    assert Services(work=lambda task: "ready", check=lambda chk: None).late() == ""


def test_a_turn_cut_by_a_stop_runs_again_as_the_attempt_it_was(tmp_path):
    workspace = _workspace(tmp_path)
    with stoppable(workspace) as stop:
        held = threading.Event()

        def work(task):
            if task.id == "weights":
                stop("the controller's job ends: stopped for the limit")
                held.wait(1)
            return "ready"

        first = _orchestrator(workspace, _services(work=work))
        assert first.run().startswith("stopped: ")
        held.set()
    assert _statuses(tmp_path)["weights"] == "running"
    services = _services()
    resumed = _orchestrator(workspace, services)
    # It starts first, and as attempt 1: the cut turn spent nothing.
    assert resumed.run() == "delivered"
    assert ("weights", 1) in services.started
    assert resumed.graph.by_id()["weights"].attempts == 1


def test_a_disaggregated_deployment_is_state_then_both_sides_then_one_hop(tmp_path):
    contract = c.Contract(
        checks=(
            c.Check("core", "boot", "tp1"),
            c.Check("kv_fp8", "accuracy[gsm8k]", "tp1"),
            c.Check("pd", "disagg", "tp2"),
        )
    )
    deployments = [
        Deployment("core", checks=("core:boot@tp1",), gpus=1),
        Deployment("kv_fp8", checks=("kv_fp8:accuracy[gsm8k]@tp1",), relative_to="core", gpus=1),
        Deployment("pd", kind=orch.DISAGGREGATED, checks=("pd:disagg@tp2",), gpus=2, row_gpus=4),
    ]
    rows = rows_mod.load_rows(_workspace(tmp_path, graph=False) / "vocabulary.yaml")
    compiled = compile_tasks(rows[2:], deployments, own_paths=_OWN, target_rel=_TARGET)
    by_id = {t.id: t for t in compiled.tasks}
    assert contract.checks  # the three deployments' own
    assert list(by_id) == ["weights", "routing", "assemble-core", "session-core",
                           "assemble-kv_fp8", "session-kv_fp8", "pd-state-pd", "pd-ctx-pd",
                           "pd-gen-pd", "pd-hop-pd", "delivery-files"]  # fmt: skip
    # Every further deployment starts once the first one's session passed, side by side.
    assert by_id["assemble-kv_fp8"].depends_on == ["session-core"]
    assert by_id["pd-state-pd"].depends_on == ["session-core"]
    # What shows the transfer before the session: the row's own test. The native tests the
    # deployment names are the contract's to judge, once. Never a check with nothing to run.
    state = by_id["pd-state-pd"]
    assert state.kind == "state" and state.check == "cpu:pd-state-pd"
    assert rows_mod.test_files(state.touches) == [f"{_OWN[1]}/test_modeling_v2_pd_state_pd.py"]
    kept = {row.id: row for row in compiled.rows}
    for side in ("pd-ctx-pd", "pd-gen-pd"):
        assert by_id[side].depends_on == ["pd-state-pd", "session-core"]
        # Each side is judged by its own test, on one node's devices at most, before the
        # other side is written and before the session's nodes are asked for.
        assert by_id[side].kind == "assemble" and by_id[side].check == f"cpu:{side}"
        assert rows_mod.test_files(by_id[side].touches) == [rows_mod.own_test(side, _OWN)]
        assert kept[side].gpus == 4 and not by_id[side].satisfies  # a row check, no checkpoint
    # Every check of its rows is one the front door has, with what the card will say.
    for task in (by_id[name] for name in ("pd-state-pd", "pd-ctx-pd", "pd-gen-pd")):
        argv = tasks_mod.check_argv(
            task.check,
            task_path="/ws/task.yaml",
            tree="/tree",
            gpus=kept[task.id].gpus,
            tests=tuple(rows_mod.test_files(task.touches)),
        )
        said = gate_cli._parse(argv)
        assert said.name == "cpu" and said.row == task.id and said.test
        assert str(said.gpus) == str(kept[task.id].gpus)
    hop = by_id["pd-hop-pd"]
    assert hop.kind == g.CHECK_KIND and hop.satisfies == ["pd:disagg@tp2"]
    # The one session waits for both sides; a baseline's session only when judged against it.
    assert hop.depends_on == ["pd-ctx-pd", "pd-gen-pd"]
    assert by_id["session-kv_fp8"].depends_on == ["assemble-kv_fp8", "session-core"]
    relative = [Deployment("core", checks=("core:boot@tp1",)),
                Deployment("pd", kind=orch.DISAGGREGATED, checks=("pd:disagg@tp2",), relative_to="core")]  # fmt: skip
    hop = {t.id: t for t in compile_tasks([], relative, own_paths=_OWN).tasks}["pd-hop-pd"]
    assert hop.depends_on == ["pd-ctx-pd", "pd-gen-pd", "session-core"]
    # A state row the survey wrote for the deployment is its state task: none is added.
    state = rows_mod.Row(
        "pd-kv", rows_mod.STATE, "kv pages move", deployment="pd", check="cpu:pd-kv"
    )
    stated = {t.id: t for t in compile_tasks([state], deployments, own_paths=_OWN).tasks}
    assert "pd-state-pd" not in stated and stated["pd-ctx-pd"].depends_on == [
        "pd-kv",
        "session-core",
    ]
    # A spec that names an onboarded ref: its rows are the delta, and nothing waits for a first.
    delta = {
        t.id: t
        for t in compile_tasks([state], deployments[2:], own_paths=_OWN, baseline=True).tasks
    }
    assert list(delta) == ["pd-kv", "pd-ctx-pd", "pd-gen-pd", "pd-hop-pd", "delivery-files"]
    assert delta["pd-kv"].depends_on == [] and delta["pd-ctx-pd"].depends_on == ["pd-kv"]
    # Its workers hear that the target stands and what they may change of it; in a run
    # that onboards from the start no card says so.
    heard = "The target is onboarded already (the spec's `onboarded` ref)"
    assert all(heard in delta[t].goal for t in ("pd-ctx-pd", "pd-gen-pd"))
    assert heard not in stated["pd-ctx-pd"].goal
    alone = {t.id: t for t in compile_tasks([], deployments, own_paths=_OWN, baseline=True).tasks}
    assert heard in alone["pd-state-pd"].goal
    # An aggregated deployment of such a run is no first one: the model is not written anew.
    assert "which already serves the first deployment" in alone["assemble-core"].goal
    # On a baseline no deployment waits for another's session, and neither does the delivery's task.
    waits = {d for t in alone.values() if t.kind != g.CHECK_KIND for d in t.depends_on}
    assert not any(d.startswith(("session-", "pd-hop-")) for d in waits)


def test_onboarded_is_pinned_once_when_the_first_deployments_session_passes(tmp_path):
    contract = c.Contract(
        checks=(
            c.Check("core", "boot", "tp1"),
            c.Check("kv_fp8", "accuracy[gsm8k]", "tp1"),
        )
    )
    deployments = (
        Deployment("core", checks=("core:boot@tp1",)),
        Deployment("kv_fp8", checks=("kv_fp8:accuracy[gsm8k]@tp1",)),
    )
    pinned, order = [], []

    def work(task):
        order.append(task.id)
        return "ready"

    services = _services(
        work=work,
        deployments=deployments,
        onboarded=lambda task: pinned.append((task.id, list(order))),
    )
    orchestrator = _orchestrator(_workspace(tmp_path), services, contract=contract)
    assert orchestrator.run() == "delivered"
    ((session, before),) = pinned
    # Pinned once, by the first session, before any further deployment's task started.
    assert session == "session-core" and "assemble-kv_fp8" not in before
    assert order.index("assemble-kv_fp8") > order.index("assemble-core")


def test_a_row_let_in_later_keeps_its_own_files_and_gets_its_directories_as_it_starts(tmp_path):
    """Found by rehearsing a run: the row came back with one path, and no skeleton."""
    asked: list[list[str]] = []
    spec = {"commit_policy": {"allowed_paths": [
        f"{_TARGET}/ops/norm/", f"{_OWN[1]}/", f"{_TARGET}/weights.py", f"{_PKG}/models/fam/routing.py",
    ]}}  # fmt: skip

    def services():
        return _services(
            skeleton=lambda files: asked.append(sorted(files)) or "committed",
            hooks=lambda: esc.Hooks(task_data=spec),
        )

    allowed = tuple(spec["commit_policy"]["allowed_paths"])
    blocked_from_the_start = services()
    first = _orchestrator(_workspace(tmp_path), blocked_from_the_start, allowed_paths=allowed)
    assert first.run() == "escalated"
    assert ("k2-gqa", 1) not in blocked_from_the_start.started
    held = g.Graph.load(tmp_path / g.GRAPH_FILENAME).by_id()["k2-gqa"]
    entry = f"{_TARGET}/ops/attention/gqa.py"
    # Blocked for the policy alone: it says which file, and the row keeps the files it owns.
    assert held.status == "blocked" and entry in held.touches and len(held.touches) == 2
    (record,) = _waiting(tmp_path)
    said = f"outside the paths the run may commit (`commit_policy.allowed_paths`): {entry}"
    assert said in record["observed"] and record["blocker_kind"] == "write_scope"
    # The skeleton was asked for the rows that could start, never for the one that could not.
    assert f"{_TARGET}/ops/norm/rms.py" in asked[0]
    assert not any("attention" in path for files in asked for path in files)
    _answer(tmp_path, record["id"], "allow-shared",
               {"path": f"{_TARGET}/ops/attention/", "note": "the row owns it"})  # fmt: skip
    again = services()
    assert _orchestrator(tmp_path, again, allowed_paths=allowed).run() == "delivered"
    # As the row starts, its directories are made: both of its files, and no other row's.
    assert asked[-1] == sorted(held.touches)
    assert [task for task, _ in again.started][0] == "k2-gqa"
    landed = g.Graph.load(tmp_path / g.GRAPH_FILENAME).by_id()["k2-gqa"]
    assert landed.status == "done" and set(held.touches) < set(landed.touches)
    assert spec["commit_policy"]["allowed_paths"][-1] == f"{_TARGET}/ops/attention/"


def test_a_rows_own_file_the_run_may_not_commit_blocks_it_before_its_work(tmp_path):
    """A whole run's row did its work and was refused at its landing, with its attempts spent."""
    allowed = (f"{_TARGET}/ops/", f"{_OWN[1]}/", f"{_PKG}/models/fam/routing.py")
    services = _services()
    assert _orchestrator(_workspace(tmp_path), services, allowed_paths=allowed).run() == "escalated"
    (record,) = _waiting(tmp_path)
    assert (record["task"], record["blocker_kind"]) == ("weights", "write_scope")
    assert record["observed"].endswith(f"(`commit_policy.allowed_paths`): {_TARGET}/weights.py")
    # As a kernel row's entry is: before any work, with the verb that lets it in.
    assert ("weights", 1) not in services.started
    assert "`resolve allow-shared weights --path" in record["next_question"]
    _answer(tmp_path, "weights", "allow-shared",
            {"path": f"{_TARGET}/weights.py", "note": "the row owns it"})  # fmt: skip
    again = _services()
    assert _orchestrator(tmp_path, again, allowed_paths=allowed).run() == "delivered"
    assert again.started[0] == ("weights", 2)


def test_a_wrong_anchor_is_changed_by_resolve_and_its_session_is_measured_again(
    tmp_path, monkeypatch
):
    held = {"anchor": 61.77}

    def failing(key):
        # The model scores 41.5: under an anchor stated 20 points high the benchmark fails.
        return key == "core:accuracy[gsm8k]@tp1" and held["anchor"] > 50

    def change(workspace, deployment, benchmark, value, unit, reason):
        was, held["anchor"] = held["anchor"], float(value)
        return f"anchor {deployment}:{benchmark}: declared {was} -> declared {value}"

    monkeypatch.setattr(esc, "set_anchor", change)

    def services():
        return _services(failing=failing)

    first = services()
    assert _orchestrator(_workspace(tmp_path), first, max_attempts=2).run() == "escalated"
    (record,) = _waiting(tmp_path)
    assert record["task"] == "session-core" and "core:accuracy[gsm8k]@tp1" in record["observed"]
    _answer(
        tmp_path,
        record["id"],
        "change-requirement",
        {"anchor": 41.77, "deployment": "core", "benchmark": "gsm8k",
         "note": "61.77 was a typing error"},
    )  # fmt: skip
    again = services()
    resumed = _orchestrator(tmp_path, again, max_attempts=2)
    assert resumed.run() == "delivered"
    said = esc.read_answer(tmp_path, record["id"])["said"]
    assert "anchor core:gsm8k: declared 61.77 -> declared 41.77" in said
    assert "`session-core` is queued for a new load" in said
    # No check waits for a later tier: the session's other two had passed beside the one
    # that failed, and both passes were judged by the old requirement.
    assert "2 pass(es) by the old requirement no longer count" in said
    # The session was measured again, once, every check of it; no worker ran again for it,
    # and the final pass measured nothing a second time.
    assert sorted(again.ran) == sorted(_KEYS)
    # The one task that starts is the delivery's, which waited behind that session.
    assert again.started == [("delivery-files", 1)]
    voided = [v for v in c.ContractStatus.load(tmp_path).verdicts if v.void]
    assert sorted(f"{v.unit}:{v.gate}@{v.member}" for v in voided) == [
        "core:boot@tp1",
        "core:serve@tp1",
    ]
    assert "typing error" in voided[0].void


def test_a_row_a_later_survey_adds_is_waited_for_by_a_task_the_graph_already_holds(tmp_path):
    """A whole run's survey added a target's row on its third attempt, and the assembly started without it."""
    services = _services()
    orchestrator = _orchestrator(_workspace(tmp_path), services)
    orchestrator.graph.by_id()["survey"].status = "done"
    orchestrator._expand()
    tasks = orchestrator.graph.by_id()
    before = list(tasks["assemble-core"].depends_on)
    tasks["weights"].status, tasks["routing"].status = "done", "running"
    # The survey again: one more row of the target's, which two rows it had now wait for.
    weights, routing = _VOCABULARY["rows"]
    replay = {"id": "replay", "kind": "feature", "outcome": "the decoder's replay is bounded",
              "required": ["a replay longer than the bound is refused"]}  # fmt: skip
    later = {
        **_VOCABULARY,
        "rows": [
            {**weights, "depends_on": ["replay"]},
            {**routing, "depends_on": ["weights", "replay"]},
            replay,
        ],
    }
    (tmp_path / "vocabulary.yaml").write_text(yaml.safe_dump(later), encoding="utf-8")
    orchestrator._expand()
    tasks = orchestrator.graph.by_id()
    assert tasks["replay"].status == "pending"
    # The assembly has not started: it waits for the row, and for all it waited for before.
    assert tasks["assemble-core"].depends_on == [*before, "replay"]
    # A task that is done is as it was. One that runs is not stopped, and that is said.
    assert "replay" not in tasks["weights"].depends_on
    assert "replay" in tasks["routing"].depends_on
    (said,) = [line for line in services.logged if "routing now waits for replay too" in line]
    assert "It is running and is not stopped: its next attempt waits" in said
    # Again, as every start does: nothing more.
    waits = {t.id: list(t.depends_on) for t in orchestrator.graph.tasks}
    orchestrator._expand()
    assert {t.id: list(t.depends_on) for t in orchestrator.graph.tasks} == waits
    # Rows that turn an edge round would close a circle: the edge is left out, and said.
    turned = {**_VOCABULARY, "rows": [weights, routing, {**replay, "depends_on": ["routing"]}]}
    (tmp_path / "vocabulary.yaml").write_text(yaml.safe_dump(turned), encoding="utf-8")
    orchestrator._expand()
    assert "routing" not in orchestrator.graph.by_id()["replay"].depends_on
    assert any("replay is not made to wait for routing" in line for line in services.logged)


def test_the_pool_is_asked_for_the_shapes_the_ready_and_running_tasks_need(tmp_path):
    contract = c.Contract(
        checks=(
            c.Check("core", "deployment", "tp1"),
            c.Check("wide", "deployment", "tp16"),
        )
    )
    deployments = (
        Deployment("core", checks=("core:deployment@tp1",), gpus=1, wall=4200),
        Deployment("wide", checks=("wide:deployment@tp16",), nodes=2, wall=9000),
    )
    services = _services(deployments=deployments)
    orchestrator = _orchestrator(_workspace(tmp_path), services, contract=contract)
    status = {t.id: t for t in orchestrator.graph.tasks}
    # The survey alone: one node, for the fan-out that follows it.
    assert orchestrator.demand() == {1: float(rows_mod.DEFAULT_WALL_S)}
    status["survey"].status = "done"
    orchestrator._expand()
    by_id = orchestrator.graph.by_id()
    # The rows' tasks are ready: one node, the longest wall a row's check states.
    assert orchestrator.demand() == {1: float(rows_mod.DEFAULT_WALL_S)}
    for tid in ("k1-rms", "k2-gqa", "weights", "routing"):
        by_id[tid].status = "done"
    # The first deployment's assembly is due: its session's own shape and wall.
    assert orchestrator.demand() == {1: 4200.0}
    by_id["assemble-core"].status = "done"
    by_id["session-core"].status = "running"
    assert orchestrator.demand() == {1: 4200.0}
    # Onboarded: the wide deployment's tasks are due. They ask one node for their own
    # work; its whole nodes are asked for by its session's job, while that is in the spool.
    by_id["session-core"].status = "done"
    assert orchestrator.demand() == {1: float(rows_mod.DEFAULT_WALL_S)}
    # Blocked work asks nothing.
    by_id["assemble-wide"].status = "blocked"
    assert orchestrator.demand() == {}


_TWO = c.Contract((c.Check("core", "deployment", "tp1"), c.Check("wide", "deployment", "tp16")))
_CORE_KEY, _WIDE_KEY = (check.key for check in _TWO.checks)
_TWO_DEPLOYMENTS = (
    Deployment("core", checks=(_CORE_KEY,)),
    Deployment("wide", checks=(_WIDE_KEY,)),
)


def _moving(state):
    """A key that follows the build ``state`` names, and work whose wide assembly rebuilds it."""

    def key(check, commit=""):
        return f"{check.unit} on {state['library']}"

    def work(task):
        if task.id == "assemble-wide":
            # Its landing changes a library every deployment loads.
            state["library"] = state["next"]
        return "ready"

    return key, work


def test_the_final_pass_measures_again_only_the_checks_whose_key_changed(tmp_path):
    state = {"library": "build-1", "next": "build-2"}
    key, work = _moving(state)
    services = _services(deployments=_TWO_DEPLOYMENTS, work=work, key=key)
    orchestrator = _orchestrator(_workspace(tmp_path), services, contract=_TWO)
    assert orchestrator.run() == "delivered"
    # In the middle of the run each session ran once, and the landing that changed the
    # library started nothing. The final pass then measured the first deployment again,
    # whose row was for the old library, and not the second, whose row has the candidate's key.
    assert services.ran == [_CORE_KEY, _WIDE_KEY, _CORE_KEY]
    final = orchestrator.final
    assert final.rows[_CORE_KEY].measured and final.rows[_CORE_KEY].key == "core on build-2"
    assert not final.rows[_WIDE_KEY].measured and final.rows[_WIDE_KEY].row.passed
    assert final.waived == {} and final.evaluation().holds
    said = next(line for line in services.logged if line.startswith("[final] candidate"))
    assert f"not measured again, a row has the candidate's key: {_WIDE_KEY}" in said
    assert f"measured now: {_CORE_KEY}" in said
    # The same candidate again: every requested check has its row, and nothing is measured.
    again = _services(deployments=_TWO_DEPLOYMENTS, key=key)
    assert _orchestrator(tmp_path, again, contract=_TWO).run() == "delivered" and again.ran == []


def test_a_check_with_no_content_identity_is_never_reused(tmp_path):
    services = _services(deployments=_TWO_DEPLOYMENTS, key=lambda check, commit="": "")
    assert _orchestrator(_workspace(tmp_path), services, contract=_TWO).run() == "delivered"
    # Each session ran as its task and once more in the final pass: nothing vouches that
    # what was measured is what the candidate holds.
    assert services.ran.count(_CORE_KEY) == 2 and services.ran.count(_WIDE_KEY) == 2


def test_a_check_that_fails_in_the_final_pass_blocks_its_task_and_a_repair_is_a_new_candidate(
    tmp_path,
):
    state = {"library": "build-1", "next": "build-2", "broken": True}
    key, work = _moving(state)

    def failing(name):
        # The second deployment's landing broke the first, which had passed before it.
        return name == _CORE_KEY and state["library"] == "build-2" and state["broken"]

    services = _services(deployments=_TWO_DEPLOYMENTS, work=work, key=key, failing=failing)
    orchestrator = _orchestrator(_workspace(tmp_path), services, contract=_TWO)
    assert orchestrator.run() == "escalated"
    (record,) = _waiting(tmp_path)
    assert record["task"] == "session-core" and record["blocker_kind"] == "final_check_failed"
    # It asks what whoever answers it has to decide, and names the verbs.
    assert (
        "resolve retry session-core --task <the task whose work broke it>"
        in record["next_question"]
    )
    assert (
        "the final pass on the candidate" in record["observed"] and _CORE_KEY in record["observed"]
    )
    # Nothing was sent back by itself: no task ran a second time for it.
    assert orchestrator.graph.by_id()["assemble-core"].attempts == 1
    assert _statuses(tmp_path)["session-core"] == "blocked"
    # A controller that stays for the answer looks again: the pass that failed is still
    # what the closing report shows.
    assert orchestrator._spent(None) == "escalated" and orchestrator.final_record["checks"]
    # The session that watches the run has it repaired: the repair is a new candidate, and
    # every requested check is asked for its key there again.
    state.update(library="build-3", broken=False)
    _answer(tmp_path, record["id"], "retry", {"note": "the library was fixed"})
    again = _services(deployments=_TWO_DEPLOYMENTS, key=key, failing=failing)
    resumed = _orchestrator(tmp_path, again, contract=_TWO)
    assert resumed.run() == "delivered"
    assert again.ran == [_CORE_KEY, _WIDE_KEY] and again.started == []
    assert resumed.final.rows[_WIDE_KEY].measured and not resumed.final.rows[_CORE_KEY].measured


def test_a_requested_check_whose_task_was_skipped_is_not_run_and_the_run_is_partial(tmp_path):
    services = _services(deployments=_TWO_DEPLOYMENTS, decisions={"assemble-wide": "blocked"})
    assert _orchestrator(_workspace(tmp_path), services, contract=_TWO).run() == "escalated"
    (record,) = _waiting(tmp_path)
    # No final pass ran on a target a resolution is about to change.
    assert not any(line.startswith("[final]") for line in services.logged)
    _answer(tmp_path, record["id"], "skip", {"note": "the wide shape is not needed yet"})
    again = _services(deployments=_TWO_DEPLOYMENTS)
    resumed = _orchestrator(tmp_path, again, contract=_TWO)
    assert resumed.run() == "partial"
    assert resumed.final.waived == {_WIDE_KEY: "its task `session-wide` is pending"}
    assert list(resumed.final.rows) == [_CORE_KEY] and again.ran == []
    said = next(line for line in again.logged if line.startswith("[partial]"))
    assert (
        "skipped: assemble-wide" in said and "behind a skipped required task: session-wide" in said
    )
    assert f"requested checks not run: {_WIDE_KEY}" in said


def test_a_final_pass_overtaken_by_a_resolution_is_not_judged(tmp_path):
    from concurrent.futures import Future

    services = _services(deployments=_TWO_DEPLOYMENTS)
    orchestrator = _orchestrator(_workspace(tmp_path), services, contract=_TWO)
    assert orchestrator.run() == "delivered"
    passed = orchestrator.final
    # A pass that comes in after a resolution reopened a task: its candidate is stale.
    orchestrator.graph.by_id()["session-wide"].status = "pending"
    orchestrator._final = Future()
    orchestrator._final.set_result(passed)
    assert orchestrator._final_done() == "" and orchestrator._final is None
    assert any("that candidate is not judged" in line for line in services.logged)
    # So is one that enumerated a contract the run no longer has.
    orchestrator.graph.by_id()["session-wide"].status = "done"
    orchestrator.recontract(c.Contract(_TWO.checks))
    orchestrator._final = Future()
    orchestrator._final.set_result(passed)
    assert orchestrator._final_done() == ""


def _tests(verdict="CLEAN", failures=(), at_base=()):
    """What the final tests found, as ``regression.run_final_tests`` returns it."""
    return regression.TestsResult(
        "c0ffee", "ba5e", verdict, list(failures), list(at_base), 5, 1, "3 test file(s)", 1
    )


def test_the_final_tests_run_once_on_the_frozen_candidate_and_the_report_says_what_it_found(
    tmp_path,
):
    from agent_flow.workflows.trtllm_modeling_v2.run.records import report

    old = "tests/test_old.py::test_x (fails at the candidate and at the base): broken before"
    calls = []

    def final_tests(candidate):
        # Every task has ended and every requested check has its row by now.
        calls.append((candidate, sorted(services.ran), set(_statuses(tmp_path).values())))
        return _tests(at_base=[old])

    services = _services(freeze=lambda: "c0ffee", final_tests=final_tests)
    orchestrator = _orchestrator(_workspace(tmp_path), services)
    assert orchestrator.run() == "delivered"
    # Once, on the frozen candidate, and not before the final pass.
    assert calls == [("c0ffee", sorted(_KEYS), {"done"})]
    assert any(line.startswith("[final] final tests: CLEAN") for line in services.logged)
    # The pass is the loop's own record; the closing report is where it is written down.
    record = orchestrator.final_record
    assert (record["candidate"], record["verdict"]) == ("c0ffee", "delivered")
    assert [(c["check"], c["outcome"], c["key"]) for c in record["checks"]] == [
        (key, "passed", "the target") for key in _KEYS
    ]
    assert all(c["how"].startswith("not measured again") for c in record["checks"])
    assert record["tests"]["passed"] and record["tests"]["at_base"] == [old]
    assert record["tests"]["gpus"] == 1 and record["tests"]["base"] == "ba5e"
    # The closing report: each requested check with its outcome and key, then the final
    # tests, and a case that fails at the base too reported as such.
    text = report.render(tmp_path, verdict="delivered", final=record)
    assert "## The final pass" in text and "On the frozen candidate `c0ffee`" in text
    assert "| `core:boot@tp1` | passed | `the target` | not measured again" in text
    assert "**Final tests**: CLEAN at c0ffee against ba5e" in text
    assert "each stating 1 device(s)" in text
    assert "Fail at the base too, and fail nothing:" in text and f"- {old}" in text
    # The same candidate again: the pass is asked again, and finds every row.
    again = _services(freeze=lambda: "c0ffee", final_tests=lambda candidate: _tests())
    assert _orchestrator(tmp_path, again).run() == "delivered" and again.ran == []


def test_a_report_with_no_judged_pass_says_what_the_ledger_holds_of_the_final_tests(tmp_path):
    """A pass that was interrupted may have run them: never "the final tests did not run"."""
    from agent_flow.workflows.trtllm_modeling_v2.run.records import report

    _workspace(tmp_path)
    text = report.render(tmp_path, verdict="stopped")
    assert "No final pass was judged" in text and "did not run" not in text
    assert "the ledger holds no result of the final tests" in text
    # Their row is in the ledger, written before the process that ran them ended.
    c.ContractStatus.load(tmp_path).record(
        "row",
        SimpleNamespace(
            name=regression.FINAL_TESTS, member="", passed=True, detail="CLEAN at c0ffee"
        ),
        marker="key:abc",
    )
    text = report.render(tmp_path, verdict="stopped")
    assert (
        "the final tests left a result in the ledger that no pass judged (CLEAN at c0ffee)" in text
    )


def test_accuracy_tests_left_to_ci_are_listed_and_keep_nothing_from_delivered(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.records import report

    left = "ci/test_accuracy.py (2 case(s)): left to CI: the session measured `gsm8k` on `core`"
    tests = regression.TestsResult(
        "c0ffee", "ba5e", "CLEAN", [], [], 3, 1, "3 test file(s)", 1, 0, [], [left]
    )
    services = _services(freeze=lambda: "c0ffee", final_tests=lambda candidate: tests)
    orchestrator = _orchestrator(_workspace(tmp_path), services)
    assert orchestrator.run() == "delivered"
    record = orchestrator.final_record
    assert record["tests"]["left"] == [left] and record["tests"]["passed"]
    text = report.render(tmp_path, verdict="delivered", final=record)
    assert "collected and not run by the final tests:" in text and f"- {left}" in text
    assert "1 accuracy test file(s) collected and left to CI" in text


def test_final_tests_that_failed_nothing_and_do_not_certify_close_the_run_partial(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.records import report

    why = "every test that ran was skipped (3 skipped, none passed)"
    tests = regression.TestsResult(
        "c0ffee", "ba5e", regression.NOT_VERIFIED, [], [], 3, 1, "3 test file(s)", 1, 0, [why]
    )
    asked = []
    services = _services(
        freeze=lambda: "c0ffee",
        final_tests=lambda candidate: tests,
        delivery_files=lambda: asked.append(1) or [],
    )
    orchestrator = _orchestrator(_workspace(tmp_path), services)
    # Not delivered and not escalated: the third outcome, listed.
    assert orchestrator.run() == "partial" and _waiting(tmp_path) == []
    assert asked == [], "a run that closes partial is not asked for its delivery files"
    assert any(
        line.startswith("[partial]") and f"final tests: {why}" in line for line in services.logged
    )
    record = orchestrator.final_record
    assert record["verdict"] == "partial" and record["tests"]["unverified"] == [why]
    assert not record["tests"]["passed"]
    text = report.render(tmp_path, verdict="partial", final=record)
    assert "Not verified by the final tests (nothing failed; they do not certify):" in text
    assert f"- {why}" in text


def test_the_final_pass_measures_a_chain_of_relative_checks_one_after_another(tmp_path):
    """Each judged against the one before: three waves, never two checks of the chain at once."""
    first = c.Check("first", "deployment", "tp1")
    second = c.Check("second", "deployment", "tp1", after=first.key)
    third = c.Check("third", "deployment", "tp1", after=second.key)
    # Named out of order: the order measured is the chain's, not the list's.
    contract = c.Contract((third, first, second))
    ended = []

    def failing(key):
        # A check judged against another's result fails when that result is not in yet.
        check = next(k for k in contract.checks if k.key == key)
        if check is second:
            time.sleep(0.05)  # were the third measured at once with it, it would be first
        ended.append(key)
        return bool(check.after) and check.after not in ended[:-1]

    seed(tmp_path, [])
    services = _services(deployments=(), failing=failing)
    final = _orchestrator(tmp_path, services, contract=contract)._final_pass()
    assert ended == [first.key, second.key, third.key]
    assert final.evaluation().holds and all(row.measured for row in final.rows.values())


@pytest.mark.parametrize("what", ["a commit", "a build"])
def test_a_target_that_moved_while_the_final_pass_ran_is_not_judged_on_the_old_candidate(
    tmp_path, what
):
    """The session that watches the run commits by hand, or builds a library again, mid-pass."""
    state = {"head": "c0ffee", "library": "build-1"}
    calls = []

    def final_tests(candidate):
        calls.append(candidate)
        if len(calls) == 1:  # while the first pass's tests run, the target moves
            state.update({"head": "f00d"} if what == "a commit" else {"library": "build-2"})
        return _tests()

    services = _services(
        freeze=lambda: state["head"],
        key=lambda check, commit="": f"{check.unit} on {state['library']}",
        final_tests=final_tests,
    )
    orchestrator = _orchestrator(_workspace(tmp_path), services)
    assert orchestrator.run() == "delivered"
    said = [line for line in services.logged if "that candidate is not judged" in line]
    assert len(said) == 1 and "c0ffee" in said[0]
    record = orchestrator.final_record
    if what == "a commit":
        # The commit changed no key: the second pass reads every row and measures nothing.
        assert "the target's head is f00d now" in said[0]
        assert calls == ["c0ffee", "f00d"] and record["candidate"] == "f00d"
        assert sorted(services.ran) == sorted(_KEYS)
    else:
        # The build changed every key: each check is measured once more, on what stands.
        assert "the key of" in said[0] and "is another now" in said[0]
        assert calls == ["c0ffee", "c0ffee"] and len(services.ran) == 2 * len(_KEYS)
        assert {c["key"] for c in record["checks"]} == {"core on build-2"}
        assert all(c["how"] == "measured in the final pass" for c in record["checks"])


def test_final_tests_that_fail_escalate_the_run_and_a_repair_is_a_new_candidate(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.records import report

    broke = "tests/test_shared.py::test_y (fails at the candidate; passed at the base): wrong"
    found = [_tests("REGRESSED", failures=[broke]), _tests()]
    calls = []
    head = ["candidate-1"]

    def final_tests(candidate):
        calls.append(candidate)
        return found[len(calls) - 1]

    def services():
        return _services(freeze=lambda: head[0], final_tests=final_tests)

    first = services()
    failed = _orchestrator(_workspace(tmp_path), first)
    assert failed.run() == "escalated"
    (record,) = _waiting(tmp_path)
    assert record["id"] == "run" and record["blocker_kind"] == "final_tests_failed"
    assert record["next_question"].startswith("Which task's work broke each failing case?")
    assert broke in record["observed"]
    # Every task stays done: nothing is sent back by itself, and nothing runs again
    # while the escalation stands.
    assert set(_statuses(tmp_path).values()) == {"done"}
    assert _orchestrator(tmp_path, services()).run() == "escalated" and len(calls) == 1
    text = report.render(tmp_path, verdict="escalated", final=failed.final_record)
    assert "Fail the final tests (the candidate's result; the base's):" in text
    assert f"- {broke}" in text
    # The session that watches the run sends the assembly back: its repair is a new
    # candidate, and the final tests run on that one.
    _answer(tmp_path, "run", "retry", {"task": "assemble-core", "note": "fix test_y"})
    head[0] = "candidate-2"
    again = services()
    repaired = _orchestrator(tmp_path, again)
    assert repaired.run() == "delivered"
    assert calls == ["candidate-1", "candidate-2"]
    assert repaired.final_record["candidate"] == "candidate-2"


def test_a_final_pass_that_failed_for_no_fault_of_the_target_runs_again_with_no_task_sent_back(
    tmp_path,
):
    """A lost job, or a defect of the flow a newer flow put right: nothing of the target changes."""
    lost = _tests("INCOMPLETE", failures=["the test job recorded nothing (exit 1)"])
    lost.held = 0
    lost.job_said = "FileNotFoundError: [Errno 2] No such file or directory: 'ws/regression/x.txt'"
    found = [lost, _tests()]
    calls = []

    def final_tests(candidate):
        calls.append(candidate)
        return found[len(calls) - 1]

    def services():
        return _services(freeze=lambda: "c0ffee", final_tests=final_tests)

    first = services()
    assert _orchestrator(_workspace(tmp_path), first).run() == "escalated"
    (record,) = _waiting(tmp_path)
    assert record["id"] == "run" and "`resolve retry run --note <why>`" in record["next_question"]
    # No case failed, so nobody is asked whose work broke one: the tests did not run, the
    # job's own last line is quoted, and the environment is looked at first.
    assert record["blocker_kind"] == "final_tests_not_run"
    assert record["next_question"].startswith(
        "The final tests did not run to a result: no case failed"
    )
    assert f"The test job's last line: `{lost.job_said}`" in record["next_question"]
    assert "--task" not in record["next_question"]
    assert "the final tests did not run to a result: INCOMPLETE" in record["observed"]
    # The outer session names no task: none runs again, and the pass does, on the same target.
    _answer(tmp_path, "run", "retry", {"note": "the job was lost with its allocation"})
    again = services()
    assert _orchestrator(tmp_path, again).run() == "delivered"
    assert again.started == [] and calls == ["c0ffee", "c0ffee"] and _waiting(tmp_path) == []
    # On a blocked task `retry` is that task again; on the run, a task that is named
    # with `--task` is sent back.


def test_the_final_tests_do_not_run_on_a_candidate_a_requested_check_failed_on(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.records import report

    state = {"library": "build-1", "next": "build-2"}
    key, work = _moving(state)
    calls = []
    services = _services(
        deployments=_TWO_DEPLOYMENTS,
        work=work,
        key=key,
        failing=lambda name: name == _CORE_KEY and state["library"] == "build-2",
        final_tests=lambda candidate: calls.append(candidate) or _tests(),
    )
    orchestrator = _orchestrator(_workspace(tmp_path), services, contract=_TWO)
    assert orchestrator.run() == "escalated"
    assert calls == []
    record = orchestrator.final_record
    assert record["tests"] is None and record["tests_said"].startswith("not run: a requested")
    assert [(c["check"], c["outcome"], c["how"]) for c in record["checks"]] == [
        (_CORE_KEY, "failed", "measured in the final pass"),
        (_WIDE_KEY, "passed", "not measured again: a row had the candidate's key"),
    ]
    text = report.render(tmp_path, verdict="escalated", final=record)
    assert f"| `{_CORE_KEY}` | failed | `core on build-2` | measured in the final pass" in text
    # A controller started again with the task blocked judges no pass, and its report
    # says so; one that had judged a pass before the block says that none stands.
    resumed = _services(deployments=_TWO_DEPLOYMENTS, key=key, final_tests=lambda c: _tests())
    again = _orchestrator(tmp_path, resumed, contract=_TWO)
    assert again.run() == "escalated" and again.final_record is None
    assert "No final pass was judged" in report.render(tmp_path, verdict="escalated")
    orchestrator.final_record, orchestrator._leaves_at = record, None
    assert orchestrator._spent(None) == "escalated"
    assert "No final pass stands: not run: session-core blocked" in report.render(
        tmp_path, verdict="escalated", final=orchestrator.final_record
    )


def test_a_check_that_was_running_when_the_controller_stopped_is_read_from_the_ledger(tmp_path):
    """A resume starts again what was running; a check whose result is on disk is not run again."""
    first = _services(deployments=_TWO_DEPLOYMENTS)
    assert _orchestrator(_workspace(tmp_path), first, contract=_TWO).run() == "delivered"
    assert first.ran == [_CORE_KEY, _WIDE_KEY]
    # As a process that died left it: both sessions recorded as running. The first had
    # written its row before the process died; the second's row never reached the ledger.
    path = tmp_path / g.GRAPH_FILENAME
    data = yaml.safe_load(path.read_text())
    for task in data["tasks"]:
        if task["id"] in ("session-core", "session-wide"):
            task["status"] = "running"
    path.write_text(yaml.safe_dump(data))
    ledger = tmp_path / c.STATUS_FILENAME
    kept = [line for line in ledger.read_text().splitlines() if '"unit": "core"' in line]
    ledger.write_text("\n".join(kept) + "\n")
    again = _services(deployments=_TWO_DEPLOYMENTS)
    resumed = _orchestrator(tmp_path, again, contract=_TWO)
    assert resumed.run() == "delivered"
    # Reconciled from what is on disk: the row that is there is read, and only the check
    # with no row is asked for, once. No worker's turn runs again.
    assert again.ran == [_WIDE_KEY] and again.started == []
    assert any(
        line.startswith("[resume] the tasks running when the last process stopped start again")
        for line in again.logged
    )


def test_test_jobs_fit_one_holder_keep_the_test_budget_and_state_their_devices(
    tmp_path, monkeypatch
):
    # Holders of 100 minutes: a job starts only where it ends before the holder does, so a
    # two-hour job never starts.
    env = RunEnvironment("slurm", "python", spool="/g", holder_time_s=6000.0, gpus_per_node=4)
    observed, asked = [], []

    class Executor:
        def execute(self, request):
            observed.append(request)
            return SimpleNamespace(
                returncode=0, stdout="MV2FLOW_TEST_ENV 0123456789abcdef\n", stderr=""
            )

    def select(*args, **kwargs):
        asked.append(kwargs)
        return Executor()

    monkeypatch.setattr(regression, "select_command_executor", select)
    wf = SimpleNamespace(task_data={"trtllm_repo_path": str(tmp_path)}, _gate_env=lambda: env)
    tree_at, run, python = regression.job_runner(wf, regression.FINAL_TESTS, gpus=2)
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
    # Each job states the devices its tests state, not the node's four, and its name.
    assert {kwargs["gpus"] for kwargs in asked} == {"2"}
    assert {kwargs["name_prefix"] for kwargs in asked} == {"final-tests"}
    # With nothing said, one device: never a whole node by default.
    assert regression.SessionJobs(env, "final-tests").gpus == 1
    assert {kwargs["nodes"] for kwargs in asked} == {0} and not any(k["once"] for k in asked)
    # A row that states whole nodes: the job states them, and its lead runs it once.
    # Each job carries the identity its results are kept under.
    asked.clear()
    wide = regression.SessionJobs(env, "final-tests", gpus=4, nodes=2, identity="abc123")
    wide("true", tmp_path, timeout_s=60)
    assert (asked[0]["nodes"], asked[0]["once"]) == (2, True)
    # And what a session's one-rank engine is given, from the same place: a test that
    # constructs an engine must not spawn its worker (a container forbids it).
    single = RunEnvironment("local", "p").launch_env
    assert single == {"TLLM_WORKER_USE_SINGLE_PROCESS": "1"}
    assert observed[-1].env == {**single, regression.IDENTITY_VAR: "abc123"}
    assert all(job.env and job.env.items() >= single.items() for job in observed)
    # Whatever the run's own width, and nothing where the spec sets the variable itself
    # (the run's own variables reach the job as they reach every command).
    four = RunEnvironment("local", "p", world_size=4)
    assert four.launch_env == {} and regression.one_rank_env(four) == single
    stated = RunEnvironment("local", "p", env={"TLLM_WORKER_USE_SINGLE_PROCESS": "0"})
    assert regression.one_rank_env(stated) == {}
    # The window is one holder's time less the slack; a holder with no end bounds nothing.
    shorter = RunEnvironment("slurm", "p", spool="/g", holder_time_s=5100.0)
    assert regression.job_timeout(shorter) == 4500
    assert regression.job_timeout(RunEnvironment("local", "p")) == regression.JOB_TIMEOUT_S
    # The spec states how long a holder lasts: Slurm's time.
    slurm = {"environment": {"slurm": {"time": "01:40:00"}, "spool": "/g"}}
    assert resolve_environment(slurm).holder_time_s == 6000.0
    assert regression.max_jobs(regression.JOB_TIMEOUT_S) == 3
