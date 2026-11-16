# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Escalations and `resolve`: each verb, one writer of the graph, the record of every change."""

from __future__ import annotations

import fcntl
import json
import re
import shlex
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import cli
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import card as card_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor import status as monitor
from agent_flow.workflows.trtllm_modeling_v2.operators import rows as rows_mod
from agent_flow.workflows.trtllm_modeling_v2.operators.coverage import VocabularyError
from agent_flow.workflows.trtllm_modeling_v2.run import escalation as esc
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run import resolve
from agent_flow.workflows.trtllm_modeling_v2.run.execution import tasks as tasks_mod

_TARGET = "tensorrt_llm/_torch/_experimental/modeling_v2/models/fam/new__sm_100__unified"


def _graph(tmp_path, *tasks):
    graph = g.Graph(tasks=list(tasks))
    graph.save(tmp_path / g.GRAPH_FILENAME)
    return graph


def _blocked(tmp_path, task_id="k1", question=""):
    """A graph with one blocked kernel task, and what depends on it."""
    task = g.Task(id=task_id, kind="kernel", goal="the entry", touches=["a.py"], status="blocked",
                  attempts=3, budget=3, ref="1", check="kernel:1", last="2 of 20 cells fail",
                  block={"by": esc.WORKER, "kind": "engine_defect", "question": question})  # fmt: skip
    core = g.Task(id="assemble-core", kind="assemble", goal="core", depends_on=[task_id])
    session = g.Task(id="session-core", kind=g.CHECK_KIND, goal="s", depends_on=["assemble-core"],
                     satisfies=["core:boot@tp1"])  # fmt: skip
    return _graph(tmp_path, g.Task(id="survey", kind="survey", goal="s", status="done", ref=_TARGET),
                  task, core, session)  # fmt: skip


def _apply(tmp_path, graph, verb, args, hooks=None, name="k1"):
    """Leave an answer as `resolve` does beside a running controller, and apply it as one does."""
    task = graph.by_id().get(name)
    answer = esc.leave_answer(tmp_path, name, verb, args, task.attempts if task else 0)
    said = esc.apply_pending(
        tmp_path, graph, hooks or esc.Hooks(), lambda: graph.save(tmp_path / g.GRAPH_FILENAME)
    )
    return answer, said


def test_retry_adds_an_attempt_and_its_note_reaches_every_worker(tmp_path):
    graph = _blocked(tmp_path, question="is the FP8 cell owed?")
    # What waits is one line: who blocked it, what was seen, what it asks, what it holds.
    (line,) = esc.waiting(graph)
    assert line.startswith("k1 (kernel), attempt 3, worker/engine_defect: 2 of 20 cells fail")
    assert "it asks: is the FP8 cell owed?" in line
    assert line.endswith("it holds assemble-core, session-core")
    # A verb with no reason is not applied, and says so; the task stays blocked.
    _, said = _apply(tmp_path, graph, "retry", {})
    assert "needs its reason" in said[0] and esc.read_answer(tmp_path, "k1")["applied"] is False
    assert graph.by_id()["k1"].status == "blocked"
    _, said = _apply(tmp_path, graph, "retry", {"note": "leave the long FP8 cells out"})
    task = graph.by_id()["k1"]
    assert (task.status, task.attempts, task.budget, task.block) == ("pending", 4, 4, {})
    assert "leave the long FP8 cells out" in task.last
    assert "leave the long FP8 cells out" in (tmp_path / esc.NOTES_FILENAME).read_text()
    # The answer's own file says what became of it; the graph on disk holds the change.
    settled = esc.read_answer(tmp_path, "k1")
    assert settled["applied"] is True and "run again with the note" in settled["said"]
    assert g.Graph.load(tmp_path / g.GRAPH_FILENAME).by_id()["k1"].attempts == 4
    assert esc.pending_answers(tmp_path) == [] and esc.waiting(graph) == []
    # A worker's own question, once answered, costs its task none of its attempts: with
    # two it has two left to try. A block the loop made for a failure gives nothing back.
    for by, budget in ((esc.WORKER, 3), (esc.GATE, 2)):
        asked = g.Task(id="row", kind="feature", goal="x", status="blocked", attempts=1,
                       budget=2, block={"by": by, "kind": "blocked"})  # fmt: skip
        graph = _graph(tmp_path, asked)
        _apply(tmp_path, graph, "retry", {"note": "go on"}, name="row")
        assert (asked.status, asked.attempts, asked.budget) == ("pending", 2, budget)


def test_an_answer_is_for_one_attempt_and_is_applied_once(tmp_path):
    graph = _blocked(tmp_path)
    save = lambda: graph.save(tmp_path / g.GRAPH_FILENAME)  # noqa: E731
    _apply(tmp_path, graph, "retry", {"note": "again"})
    for _ in range(3):  # settled: later passes find nothing to apply
        assert esc.apply_pending(tmp_path, graph, esc.Hooks(), save) == []
    assert graph.by_id()["k1"].attempts == 4
    assert (tmp_path / esc.NOTES_FILENAME).read_text().count("again") == 1
    # A process that applied it and stopped before it could say so finds the answer again:
    # its task has moved on, and the answer is settled as that. Nothing is applied twice.
    left = {
        k: v for k, v in esc.read_answer(tmp_path, "k1").items() if k not in ("applied", "said")
    }
    esc.answer_path(tmp_path, "k1").write_text(yaml.safe_dump(left))
    (said,) = esc.apply_pending(tmp_path, graph, esc.Hooks(), save)
    assert "was not applied" in said and "`k1` is not blocked (it is pending, attempt 4)" in said
    assert graph.by_id()["k1"].attempts == 4
    # A later answer to the same block replaces one that was not applied yet.
    graph.by_id()["k1"].status = "blocked"
    esc.leave_answer(tmp_path, "k1", "skip", {"note": "first"}, 4)
    esc.leave_answer(tmp_path, "k1", "retry", {"note": "second"}, 4)
    (said,) = esc.apply_pending(tmp_path, graph, esc.Hooks(), save)
    assert "run again with the note" in said and graph.by_id()["k1"].attempts == 5
    # An answer to an attempt the task has left changes nothing.
    graph.by_id()["k1"].status = "blocked"
    esc.leave_answer(tmp_path, "k1", "retry", {"note": "late"}, 4)
    (said,) = esc.apply_pending(tmp_path, graph, esc.Hooks(), save)
    assert "`k1` is blocked on attempt 5, and this answers attempt 4" in said
    # Neither does a verb that is none, a name the graph does not have, or a pool verb on a task.
    for name, verb, why in (
        ("k1", "undo", "is not a verb"),
        ("nope", "retry", "the graph has no task 'nope'"),
        ("k1", "pool-retry", "name `pool`"),
        ("run", "retry", "the run asks nothing"),
    ):
        esc.leave_answer(tmp_path, name, verb, {"note": "x"}, 5)
        (said,) = esc.apply_pending(tmp_path, graph, esc.Hooks(), save)
        assert why in said, said
    # The run's own question: `retry run` runs the final pass again, and a task may be
    # named to be sent back, blocked or not.
    graph.run_blocked = {"kind": "final_tests_failed", "observed": "2 cases fail"}
    _, said = _apply(tmp_path, graph, "retry", {"note": "a job was lost"}, name="run")
    assert "the final pass runs again on the target as it is" in said[0] and not graph.run_blocked
    graph.run_blocked = {"kind": "not_delivered", "observed": "a file"}
    graph.by_id()["survey"].status = "done"
    _, said = _apply(tmp_path, graph, "retry", {"note": "repair it", "task": "survey"}, name="run")
    assert "survey run again with the note" in said[0] and not graph.run_blocked


def test_skip_reports_the_task_not_done_and_only_a_spare_ones_dependents_go_on(tmp_path):
    # Required: nothing produced what its dependents need, so they do not start.
    graph = _blocked(tmp_path)
    _, said = _apply(tmp_path, graph, "skip", {"note": "the first deployment needs no FP8 cell"})
    task = graph.by_id()["k1"]
    assert task.status == "skipped" and "needs no FP8 cell" in task.last
    assert "assemble-core, session-core depend on it and do not start" in said[0]
    assert g.ready(graph) == []
    # Only nice to have: what depended on it goes on and is told so.
    other = tmp_path / "spare"
    other.mkdir()
    graph = _blocked(other)
    graph.by_id()["k1"].priority = "nice_to_have"
    _, said = _apply(other, graph, "skip", {"note": "an optional cell"})
    assert "assemble-core, session-core go on without it" in said[0]
    assert [t.id for t in g.ready(graph)] == ["assemble-core"]


def test_allow_shared_widens_the_write_scope_and_records_a_policy_it_had_to_widen(tmp_path):
    graph = _blocked(tmp_path)
    spec = {"commit_policy": {"allowed_paths": ["tensorrt_llm/_torch/_experimental/"]}}
    changed = []
    hooks = esc.Hooks(task_data=spec, respec=changed.append)
    _apply(tmp_path, graph, "allow-shared",
           {"path": "tensorrt_llm/_torch/modules/", "note": "the norm lives in a shared module"}, hooks)  # fmt: skip
    task = graph.by_id()["k1"]
    assert task.touches == ["a.py", "tensorrt_llm/_torch/modules/"] and task.status == "pending"
    assert spec["commit_policy"]["allowed_paths"][-1] == "tensorrt_llm/_torch/modules/"
    assert changed == [["commit_policy.allowed_paths"]]
    (change,) = esc.spec_changes(tmp_path)
    assert change["class"] == "requirement" and change["why"] == "the norm lives in a shared module"


def test_a_split_leaves_the_task_as_the_join_of_typed_rows(tmp_path):
    graph = _blocked(tmp_path)
    (tmp_path / "task.yaml").write_text(yaml.safe_dump({"trtllm_repo_path": str(tmp_path)}))
    rows = [
        {"id": "norm-pre", "kind": "operator", "mode": "onboard", "target": "ops/norm/pre",
         "outcome": "the pre-norm half", "cells": [{"dtype": "bf16"}]},
        {"id": "norm-post", "kind": "feature", "outcome": "the post-norm half",
         "required": ["matches the reference"], "depends_on": ["norm-pre"],
         "writes": [f"{_TARGET}/post.py"]},
        # A row that states an empty check id has no check of its own: no job is made for it.
        {"id": "norm-note", "kind": "feature", "outcome": "what the halves share is written down",
         "writes": [f"{_TARGET}/shared.py"], "check": {"id": ""}},
    ]  # fmt: skip
    hooks = resolve.offline_hooks(tmp_path, graph)
    _, said = _apply(
        tmp_path, graph, "split", {"rows": rows, "note": "two kernels, not one"}, hooks
    )
    by_id = graph.by_id()
    assert "k1 is now the join of norm-pre, norm-post, norm-note" in said[0]
    parent = by_id["k1"]
    # The original is an ordinary join: no agent, no files, done once its children are.
    assert parent.kind == g.CHECK_KIND and parent.touches == [] and parent.status == "pending"
    assert parent.depends_on == ["norm-pre", "norm-post", "norm-note"] and not parent.satisfies
    assert by_id["norm-note"].check == "" and by_id["norm-post"].check == "cpu:norm-post"
    # A survey's row always has a check: an empty id there is the kind's own.
    (surveyed,) = rows_mod.typed_rows({"rows": [{**rows[2], "required": ["written"]}]})
    assert surveyed.check == "cpu:norm-note"
    assert by_id["norm-pre"].kind == "kernel" and by_id["norm-pre"].check == "kernel:ops/norm/pre"
    assert by_id["norm-pre"].touches[0] == f"{_TARGET}/ops/norm/pre.py"
    assert by_id["norm-post"].kind == "feature" and by_id["norm-post"].depends_on == ["norm-pre"]
    # What waited for the original still waits for it, and so for its children.
    assert by_id["assemble-core"].depends_on == ["k1"] and g.validate(graph) == []
    assert {row.id for row in rows_mod.load_work(tmp_path)} == {
        "norm-pre",
        "norm-post",
        "norm-note",
    }
    # A child sent back with a task that waits for the join holds that task: the join was
    # done and stays done, so without an edge of its own the task started at once.
    for child in parent.depends_on:
        by_id[child].status = "done"
    parent.status = "done"
    esc._unblock(graph, {"task": "assemble-core"}, "its session hung", by_id["norm-post"])
    assert by_id["norm-post"].status == "pending" and parent.status == "done"
    assert by_id["assemble-core"].depends_on == ["k1", "norm-post"]
    assert [t.id for t in g.ready(graph)] == ["norm-post"]
    # Rows that are not usable change nothing.
    (tmp_path / "second").mkdir()
    again = _blocked(tmp_path / "second", task_id="k9")
    hooks = resolve.offline_hooks(tmp_path / "second", again)
    _, said = _apply(tmp_path / "second", again, "split",
                     {"rows": [{"id": "x", "kind": "magic"}], "note": "n"}, hooks, "k9")  # fmt: skip
    assert "was not applied" in said[0] and "kind is 'magic'" in said[0]
    assert again.by_id()["k9"].kind == "kernel" and len(again.tasks) == 4


def test_one_key_of_an_options_file_is_changed_in_the_file_and_recorded(tmp_path):
    """`--set deployments.pd.ctx.options.<key>` was refused, and whole option maps were inlined."""
    options = tmp_path / "engines" / "pd.ctx.yaml"
    options.parent.mkdir()
    options.write_text("tensor_parallel_size: 4\nkv_cache_config: {dtype: fp8}\n")
    spec = {"deployments": [{"name": "pd", "ctx": {"options": str(options)}}]}
    key = "deployments.pd.ctx.options.bounded_replay_on_generation"
    # The trial before a change is applied writes nothing and leaves the spec as it is.
    esc._check_keys(spec, {key: True})
    assert "bounded" not in options.read_text()
    record = esc.record_change(tmp_path, spec, key, True, why="the control ran so", command="c1")
    # The file says what runs, the spec still names the file, and the change is recorded.
    assert yaml.safe_load(options.read_text()) == {
        "tensor_parallel_size": 4,
        "kv_cache_config": {"dtype": "fp8"},
        "bounded_replay_on_generation": True,
    }
    assert spec["deployments"][0]["ctx"]["options"] == str(options)
    assert (record["from"], record["to"], record["key"]) == (None, True, key)
    # A key further down, and a file that is not there.
    assert esc.set_spec(spec, "deployments.pd.ctx.options.kv_cache_config.dtype", "bf16") == "fp8"
    assert yaml.safe_load(options.read_text())["kv_cache_config"] == {"dtype": "bf16"}
    gone = {"deployments": [{"name": "tp1", "options": str(tmp_path / "none.yaml")}]}
    with pytest.raises(esc.ResolveError, match="names an options file that does not exist"):
        esc.set_spec(gone, "deployments.tp1.options.max_batch_size", 8)


def test_a_requirement_or_the_environment_is_changed_and_the_change_is_recorded(
    tmp_path, monkeypatch
):
    graph = _blocked(tmp_path)
    spec = {
        "environment": {"env": {"TMPDIR": "/tmp"}},
        "accuracy": {"anchor": 41.77, "units": [{"name": "kv_fp8", "tolerance": 2.0}]},
    }
    seen = []
    hooks = esc.Hooks(task_data=spec, respec=seen.append)
    _apply(tmp_path, graph, "edit-env",
           {"set": {"environment.env.TMPDIR": "/var/tmp"}, "note": "the node's /tmp is 2 GiB"}, hooks)  # fmt: skip
    assert (
        spec["environment"]["env"]["TMPDIR"] == "/var/tmp"
        and graph.by_id()["k1"].status == "pending"
    )
    # No verb is forbidden: an anchor that is mistaken is changed, and the flow records it.
    graph = _blocked(tmp_path, task_id="k2")
    _apply(tmp_path, graph, "change-requirement",
           {"set": {"accuracy.anchor": 40.5, "accuracy.units.kv_fp8.tolerance": 3.0},
            "note": "the anchor was measured with another chat template"}, hooks, "k2")  # fmt: skip
    assert spec["accuracy"]["anchor"] == 40.5 and spec["accuracy"]["units"][0]["tolerance"] == 3.0
    changes = esc.spec_changes(tmp_path)
    assert [(c["key"], c["class"], c["from"], c["to"]) for c in changes] == [
        ("environment.env.TMPDIR", "environment", "/tmp", "/var/tmp"),
        ("accuracy.anchor", "requirement", 41.77, 40.5),
        ("accuracy.units.kv_fp8.tolerance", "requirement", 2.0, 3.0),
    ]
    assert changes[1]["why"] == "the anchor was measured with another chat template"
    assert changes[1]["escalation"] == "k2"
    assert "A requirement changed (k2)" in (tmp_path / esc.NOTES_FILENAME).read_text()
    # A later start reads the user's file and puts every recorded change back over it.
    fresh = {"environment": {"env": {"TMPDIR": "/tmp"}},
             "accuracy": {"anchor": 41.77, "units": [{"name": "kv_fp8", "tolerance": 2.0}]}}  # fmt: skip
    assert len(esc.reapply_changes(tmp_path, fresh)) == 3 and fresh == spec
    # An anchor goes through the session's own command.
    graph = _blocked(tmp_path, task_id="k3")
    asked = []
    monkeypatch.setattr(esc, "set_anchor", lambda ws, *a: asked.append(a) or "anchor set")
    hooks = esc.Hooks(task_data=spec)
    _, said = _apply(tmp_path, graph, "change-requirement",
                     {"anchor": 39.0, "deployment": "tp1", "benchmark": "gsm8k", "note": "re-measured"},
                     hooks, "k3")  # fmt: skip
    assert asked == [("tp1", "gsm8k", 39.0, None, "re-measured")] and said[-1].endswith(
        "anchor set; `k3` runs again"
    )
    # A spec key that looks like an anchor is set, and the result says how a frozen one changes.
    graph = _blocked(tmp_path, task_id="k4")
    _, said = _apply(tmp_path, graph, "change-requirement",
                     {"set": {"accuracy.anchor": 40.0}, "note": "again"}, hooks, "k4")  # fmt: skip
    assert "a frozen one is changed with --deployment, --benchmark and --anchor" in said[-1]


def test_a_changed_anchor_is_the_session_steps_own_change_and_its_session_is_judged_again(
    tmp_path, capsys
):
    from agent_flow.workflows.trtllm_modeling_v2.checks import anchors
    from agent_flow.workflows.trtllm_modeling_v2.run import evidence
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as contract_mod

    # A run whose first session failed its score against an anchor stated 20 points high.
    assemble = g.Task(id="assemble-tp1", kind="assemble", goal="a", status="done", attempts=2)
    session = g.Task(id="session-tp1", kind=g.CHECK_KIND, goal="s", status="blocked", attempts=2,
                     depends_on=["assemble-tp1"], check="deployment:tp1",
                     last="accuracy[gsm8k]: 41.5 vs floor 59.9",
                     block={"by": esc.GATE, "kind": "check_failed"},
                     satisfies=["tp1:deployment@tp1", "tp1:tests@tp1"])  # fmt: skip
    _graph(tmp_path, assemble, session)
    (tmp_path / "task.yaml").write_text("{}\n")
    anchors.change(tmp_path, "tp1", "gsm8k", reason="as the spec stated it", value=61.77)
    status = contract_mod.ContractStatus.load(tmp_path)
    for gate, passed in (("deployment", False), ("tests", True)):
        status.record(
            "tp1", SimpleNamespace(name=gate, member="tp1", passed=passed, detail=""), marker="m"
        )
    argv = ["--workspace", str(tmp_path), "change-requirement", "session-tp1",
            "--deployment", "tp1", "--benchmark", "gsm8k", "--anchor", "41.77",
            "--reason", "61.77 was a typing error: the measured anchor is 41.77"]  # fmt: skip
    assert resolve.main(argv) == 0
    out = capsys.readouterr().out
    assert "anchor tp1:gsm8k: declared 61.77 -> declared 41.77" in out
    assert "`session-tp1` is queued for a new load, and 1 pass(es) by the old" in out
    saved = g.Graph.load(tmp_path / g.GRAPH_FILENAME).by_id()
    assert (saved["session-tp1"].status, saved["session-tp1"].attempts) == ("pending", 3)
    assert saved["assemble-tp1"].status == "done"  # its code is not what changed
    # The session step's own record holds from, to and why.
    (change,) = [c for c in anchors.changes(tmp_path) if "typing error" in c["reason"]]
    assert change["from"]["value"] == 61.77 and change["to"]["value"] == 41.77
    assert anchors.frozen(tmp_path, "tp1", "gsm8k").value == 41.77
    # The pass judged by the old anchor is voided in the ledger; nothing is removed.
    latest = contract_mod.ContractStatus.load(tmp_path).latest()
    assert not latest["tp1:tests@tp1"].passed and "typing error" in latest["tp1:tests@tp1"].void
    assert len(evidence.verdicts(tmp_path)) == 3


def test_the_pools_verbs_reach_the_pool_and_its_values(tmp_path):
    graph = _graph(tmp_path)
    spec = {"environment": {"slurm": {"qos": "QOS", "time": "04:00:00"}}}
    calls = []
    hooks = esc.Hooks(
        task_data=spec,
        pool_retry=lambda close, note: calls.append(("retry", close, note)) or "asked again",
        pool_adjust=lambda: calls.append(("adjust",)) or "the pool took the new values",
    )
    _, said = _apply(tmp_path, graph, "pool-adjust",
                     {"set": {"qos": "OTHER", "time": "02:00:00"}, "note": "the QoS is full"},
                     hooks, "pool")  # fmt: skip
    assert spec["environment"]["slurm"] == {"qos": "OTHER", "time": "02:00:00"}
    assert calls == [("adjust",)] and "the pool took the new values" in said[0]
    assert {c["class"] for c in esc.spec_changes(tmp_path)} == {"environment"}
    _apply(tmp_path, graph, "pool-retry", {"note": "squeue shows nothing", "close_unknown": 2},
           hooks, "pool")  # fmt: skip
    assert calls[-1] == ("retry", 2, "squeue shows nothing")


def test_resolve_takes_the_workspace_lock_when_no_controller_runs_and_hands_over_when_one_does(
    tmp_path, capsys
):
    _blocked(tmp_path)
    # A controller holds the lock: the answer is left beside the task's receipt for its next pass.
    held = (tmp_path / resolve.LOCK_FILENAME).open("w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    argv = ["--workspace", str(tmp_path), "retry", "k1", "--note", "cast to int64"]
    assert resolve.main([*argv, "--wait", "0"]) == 3
    assert g.Graph.load(tmp_path / g.GRAPH_FILENAME).by_id()["k1"].status == "blocked"
    (answer,) = esc.pending_answers(tmp_path)
    assert (answer["verb"], answer["args"], answer["attempt"]) == (
        "retry", {"note": "cast to int64"}, 3,
    )  # fmt: skip
    # The controller leaves while the answer waits for it (a run that stops with everything
    # left blocked is answered at that moment): the command applies it itself.
    threading.Timer(0.5, held.close).start()
    assert resolve.main([*argv, "--wait", "30"]) == 0
    assert "left for the running controller" in capsys.readouterr().err
    saved = g.Graph.load(tmp_path / g.GRAPH_FILENAME).by_id()["k1"]
    assert (saved.status, saved.attempts) == ("pending", 4) and esc.pending_answers(tmp_path) == []
    # No controller: the verb is applied here, with no file. One on a task that is not
    # blocked says why and exits 1, as does one with no reason or on a name that is none.
    assert resolve.main(argv) == 1
    assert "retry k1: NOT applied: `k1` is not blocked (it is pending" in capsys.readouterr().out
    assert resolve.main(["--workspace", str(tmp_path), "retry", "nope", "--note", "x"]) == 1
    graph = g.Graph.load(tmp_path / g.GRAPH_FILENAME)
    graph.by_id()["k1"].status = "blocked"
    graph.save(tmp_path / g.GRAPH_FILENAME)
    assert resolve.main(["--workspace", str(tmp_path), "skip", "k1"]) == 1
    assert resolve.main(["--workspace", str(tmp_path), "skip", "k1", "--reason", "not owed"]) == 0
    assert "skip k1: applied: k1 is skipped" in capsys.readouterr().out
    # The run's journal has each answer, applied here or by a controller.
    journal = (tmp_path / "journal.md").read_text()
    assert "**resolve** — [answer] k1: k1 run again with the note" in journal
    assert "**resolve** — [answer] k1: k1 is skipped" in journal


def _left_running(tmp_path):
    """A workspace as a stopped controller left it: one row's task `running`, and what waits for it."""
    (tmp_path / "task.yaml").write_text("{}\n")
    lost = g.Task(id="parity", kind="feature", goal="match another framework", status="running",
                  touches=["a.py"], ref="parity")  # fmt: skip
    after = g.Task(id="assemble-core", kind="assemble", goal="core", depends_on=["parity"])
    return _graph(tmp_path, g.Task(id="survey", kind="survey", goal="s", status="done", ref=_TARGET),
                  lost, after)  # fmt: skip


def test_a_task_left_running_is_skipped_or_retried_by_resolve_with_no_controller(tmp_path, capsys):
    """A controller was stopped under a worker two hours into one turn; `skip` said "not blocked"."""
    _left_running(tmp_path)
    argv = ["--workspace", str(tmp_path)]
    # A verb that does not end a turn is refused on a running task, and says which do.
    assert resolve.main([*argv, "allow-shared", "parity", "--path", "b.py", "--note", "x"]) == 1
    assert "A running task takes `retry` and `skip`, which end its turn" in capsys.readouterr().out
    # No controller runs (the one that left it running is gone): `retry` is applied here.
    # The attempt that was cut costs none: the next is attempt 2 of a budget one larger.
    assert resolve.main([*argv, "retry", "parity", "--note", "serve it; do not compare"]) == 0
    saved = g.Graph.load(tmp_path / g.GRAPH_FILENAME).by_id()["parity"]
    assert (saved.status, saved.attempts, saved.budget) == ("pending", 2, g.MAX_ATTEMPTS + 1)
    assert "serve it; do not compare" in (tmp_path / esc.NOTES_FILENAME).read_text()
    # And `skip`, on the workspace as that controller left it.
    _left_running(tmp_path)
    assert resolve.main([*argv, "skip", "parity", "--reason", "nobody asked for this row"]) == 0
    assert "skip parity: applied: parity is skipped" in capsys.readouterr().out
    saved = g.Graph.load(tmp_path / g.GRAPH_FILENAME)
    assert saved.by_id()["parity"].status == "skipped"
    # It was required, so what depends on it does not start: nothing produced what it needs.
    assert g.ready(saved) == []
    # A row nobody asked for is said to be so with the same answer, and the rest goes on.
    _left_running(tmp_path)
    said = [*argv, "skip", "parity", "--nice-to-have", "--reason", "nobody asked for it"]
    assert resolve.main(said) == 0
    assert "parity is skipped and will be reported not done; assemble-core go on without it" in (
        capsys.readouterr().out
    )
    saved = g.Graph.load(tmp_path / g.GRAPH_FILENAME)
    parity = saved.by_id()["parity"]
    assert (parity.status, parity.priority) == ("skipped", "nice_to_have")
    assert "it was required; this answer made it nice to have" in parity.last
    assert [t.id for t in g.ready(saved)] == ["assemble-core"]


def test_a_turn_that_has_run_an_hour_is_said_by_the_monitor_and_holds_nothing(tmp_path):
    """A row's worker was 110 minutes into its first turn when somebody looked."""
    graph = _left_running(tmp_path)
    cfg = monitor.load_config(tmp_path)
    # Its turn's record, as the turn executor leaves it: started 70 minutes ago, not done.
    turns = tmp_path.resolve() / "turns"
    turns.mkdir()
    now = 1_000_000.0
    record = {"id": "007-worker", "work": "parity", "started_at": now - 70 * 60,
              "activity_file": str(turns / "007-worker.activity.log")}  # fmt: skip
    (turns / "007-worker.request.json").write_text(json.dumps(record))
    assert monitor.poll_once(cfg, now=now - 20 * 60) is None, "fifty minutes is not long"
    event = monitor.poll_once(cfg, now=now)
    assert event.kind == "TURN_LONG" and event.detail.startswith(
        f"parity {turns / '007-worker.activity.log'}  parity (feature), attempt 1, has been in "
        "one turn (007-worker) for 70 minutes and holds nothing"
    )
    assert "`resolve skip parity`" in event.detail and "--ignore 007-worker" in event.detail
    # In hand by the turn's own name, so a block of the same task is still said.
    assert monitor.poll_once(cfg, now=now, ignore=["007-worker"]) is None
    # A turn that is over (its checks run) is not long, nor one of a task that is not running.
    (turns / "007-worker.done").write_text("{}")
    assert monitor.poll_once(cfg, now=now) is None
    (turns / "007-worker.done").unlink()
    graph.by_id()["parity"].status = "pending"
    graph.save(tmp_path / g.GRAPH_FILENAME)
    assert monitor.poll_once(cfg, now=now) is None


def test_monitor_wait_returns_a_blocked_task_at_once_and_the_runs_own_question(tmp_path):
    (tmp_path / "task.yaml").write_text("{}\n")
    cfg = monitor.load_config(tmp_path)
    assert monitor.poll_once(cfg) is None
    graph = _blocked(tmp_path, question="is the FP8 cell owed?")
    event = monitor.wait(cfg, timeout_s=0)
    assert event.kind == "TASK_BLOCKED"
    receipt = tmp_path.resolve() / "tasks" / "k1.3.receipt.yaml"
    assert event.detail.startswith(f"k1 {receipt}  k1 (kernel), attempt 3, worker/engine_defect")
    assert "it asks: is the FP8 cell owed?" in event.detail
    assert "holds assemble-core, session-core" in event.detail
    assert monitor.status(cfg)["blocked"][0].startswith("k1 (kernel), attempt 3")
    # One the caller has in hand (asked of a human, say) is passed over for the next thing.
    assert monitor.poll_once(cfg, ignore=["k1"]) is None
    # The run's own question: a final pass that did not hold, on no task.
    graph.run_blocked = {"kind": "final_tests_failed", "observed": "2 cases fail",
                         "question": "which task's work broke them?"}  # fmt: skip
    graph.save(tmp_path / g.GRAPH_FILENAME)
    event = monitor.poll_once(cfg, ignore=["k1"])
    assert event.kind == "RUN_BLOCKED"
    assert event.detail.startswith(
        f"{tmp_path.resolve() / 'journal.md'}  the run, final_tests_failed: 2 cases fail"
    )
    assert "it asks: which task's work broke them?" in event.detail
    assert monitor.status(cfg)["blocked"][1].startswith("the run, final_tests_failed")
    assert monitor.poll_once(cfg, ignore=["k1", "run"]) is None


def test_a_done_task_that_still_asks_is_answered_without_being_sent_back(tmp_path, monkeypatch):
    """A whole run: a survey blocked on a requirement it doubted, and started over after the answer."""
    asks = "change-requirement: the anchor of `core` on gsm8k reads 61.77, the reference row 41.77"
    survey = g.Task(id="survey", kind="survey", goal="s", status="done", ref=_TARGET,
                    block={"by": esc.WORKER, "kind": "question", "question": asks})  # fmt: skip
    session = g.Task(id="session-core", kind=g.CHECK_KIND, goal="s", status="done",
                     check="deployment:core", satisfies=["core:deployment@tp1"])  # fmt: skip
    graph = _graph(tmp_path, survey, session)
    assert esc.waiting(graph) == [f"survey (survey), done, asks and holds nothing: {asks}"]
    (tmp_path / "task.yaml").write_text("{}\n")
    cfg = monitor.load_config(tmp_path)
    event = monitor.poll_once(cfg)
    receipt = tmp_path.resolve() / "tasks" / "survey.1.receipt.yaml"
    assert event.kind == "TASK_ASKS" and event.detail.startswith(
        f"survey {receipt}  survey (survey)"
    )
    assert monitor.poll_once(cfg, ignore=["survey"]) is None
    # The requirement is changed, the session it names is measured again, and the task
    # that asked is not sent back: its work did not wait for the answer.
    monkeypatch.setattr(esc, "set_anchor", lambda *a: "anchor set")
    _, said = _apply(tmp_path, graph, "change-requirement",
                     {"anchor": 41.77, "deployment": "core", "benchmark": "gsm8k", "note": "typo"},
                     None, "survey")  # fmt: skip
    assert "anchor set" in said[0] and "`survey`" not in said[0]
    assert (survey.status, survey.attempts, survey.block) == ("done", 1, {})
    assert graph.by_id()["session-core"].status == "pending"
    assert esc.waiting(graph) == [] and monitor.poll_once(cfg) is None
    # `retry` on one that asks is its work again, as it says.
    survey.block = {"by": esc.WORKER, "kind": "question", "question": asks}
    _apply(tmp_path, graph, "retry", {"note": "survey again with the new anchor"}, name="survey")
    assert (survey.status, survey.attempts, survey.block) == ("pending", 2, {})


def test_a_clean_start_drops_an_answer_left_for_an_earlier_run(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    workspace = tmp_path / "ws"
    esc.leave_answer(workspace, "k1", "retry", {"note": "for the run before"}, 1)
    receipt = workspace / "tasks" / "k1.1.receipt.yaml"
    receipt.write_text("task: k1\n")
    # What a worker noted for the task (its card names the file) is that run's too.
    notes = workspace / "tasks" / "k1.notes.md"
    notes.write_text("the head dim is 128\n")
    # A start that continues keeps both: its controller applies the answer on its first pass.
    TrtllmModelingV2Workflow(workspace=workspace)
    assert len(esc.pending_answers(workspace)) == 1 and notes.is_file()
    TrtllmModelingV2Workflow(workspace=workspace, clean=True)
    assert esc.pending_answers(workspace) == [] and receipt.is_file() and not notes.exists()


def test_feedback_appends_to_the_notes_every_card_names(tmp_path, capsys):
    assert cli._feedback(["--workspace", str(tmp_path), "the FP8 cells may be left out"]) == 0
    assert cli._feedback(["--workspace", str(tmp_path), "TMPDIR is /var/tmp"]) == 0
    notes = (tmp_path / esc.NOTES_FILENAME).read_text()
    assert notes.index("the FP8 cells may be left out") < notes.index("TMPDIR is /var/tmp")
    assert notes.count("## 20") == 2
    with pytest.raises(SystemExit):
        cli._feedback(["--workspace", str(tmp_path), "  "])


def test_a_card_is_rendered_from_its_row_with_its_check_and_job_line_filled(tmp_path):
    row = rows_mod.Row(
        id="3", kind=rows_mod.OPERATOR, outcome="certify the QK-norm attention entry",
        config_axes={"head_dim": 128, "dtype": "bf16"},
        leads=("another target's attention wrapper", "`OpWrapper`",
               "Which wrapper applies normalization and rotation in one call?"),
        required=("prefill and decode with a cached prefix",), optional=("one FP8 KV cell",),
        check="kernel:3", gpus=1, wall=1800,
    )  # fmt: skip
    task = g.Task(id="k3-qk-norm", kind="kernel", goal=row.outcome, depends_on=["survey"],
                  touches=[f"{_TARGET}/ops/attention/qk.py", "tests/x/test_qk.py"],
                  ref="3", check="kernel:3", attempts=2, last="2 of 20 cells failed")  # fmt: skip
    command = tasks_mod.check_command("kernel:3", task_path="/ws/task.yaml", tree="/trees/k3")
    assert command == "trtllm-modeling-v2 gate kernel --task /ws/task.yaml --row 3 --tree /trees/k3"
    text = card_mod.render_card(task, row, command=command, notes="/ws/operator_notes.md",
                                spec="/ws/task.yaml", skipped=["`k1` (skipped: not needed)"])  # fmt: skip
    # What it is for, what to hand back, how it is judged: in that order.
    order = ["## Your task: `k3-qk-norm` (kernel, attempt 2 of 3)", "**What it is for.**",
             "**Inputs.**", "head_dim=128, dtype=bf16", "**Required**", "**Optional**",
             "**Reference code** (leads to search for with",
             "- lead: another target's attention wrapper", "- lead: `OpWrapper`",
             "- lead: Which wrapper applies", "**Hand back.** Commits that change only",
             "A receipt: `ready` carries",
             "`tried`, `command`, `result`, `ruled_out`, `next_question`",
             "**How it is judged.** By the check `kernel:3`", command, "`gpus=1 wall=1800`",
             "**Notes.**",
             "/ws/operator_notes.md", "**Your previous attempt.** 2 of 20 cells failed",
             "**Skipped before you.**"]  # fmt: skip
    places = [text.index(part) for part in order]
    assert places == sorted(places)
    assert "No shared initializer is yours to change" in text and len(text.split()) < 600
    assert "planned as `gpus=1 wall=1800`" in " ".join(text.split())
    assert "records its result in the run's ledger for that pin" in text
    # A kernel row's job states what its row states: more devices, and its wall.
    places = {"task_path": "/ws/task.yaml", "tree": "/t", "gpus": 2, "wall": 900}
    assert tasks_mod.check_command("kernel:3", **places) == (
        "trtllm-modeling-v2 gate kernel --task /ws/task.yaml --row 3 --gpus 2 --timeout 900 "
        "--tree /t"
    )
    # A deployment's devices and wall are its session's own.
    said = tasks_mod.check_command("deployment:tp2", **places)
    assert "--gpus" not in said and "--timeout" not in said
    # A row's own tests are the gate's too: a job of the pool, under the engine's Python,
    # with the test files of its write scope, the devices and nodes it states, and its wall.
    own = g.Task(id="weights", kind="weights", goal="w", ref="weights", check="cpu:weights",
                 touches=[f"{_TARGET}/weights.py", "tests/x/test_w.py"])  # fmt: skip
    plain = tasks_mod.check_command("cpu:weights", task_path="/ws/task.yaml", tree="/t",
                                    tests=rows_mod.test_files(own.touches), nodes=2,
                                    wall=600)  # fmt: skip
    assert plain == (
        "trtllm-modeling-v2 gate cpu --task /ws/task.yaml --row weights "
        "--test tests/x/test_w.py --nodes 2 --timeout 600 --tree /t"
    )
    said = card_mod.render_card(own, None, command=plain, notes="/ws/n.md", spec="/ws/task.yaml")
    assert "records its result in the run's ledger for that pin" in said
    assert "your own shell may not have it" in said and "lands nothing as passed" in said
    assert tasks_mod.check_argv("review:x", task_path="/ws/task.yaml", tree="/t") is None


def test_typed_rows_are_validated_together_and_the_skeleton_names_only_missing_initializers(
    tmp_path,
):
    vocabulary = {
        "steps": [],
        "queue": [],
        "rows": [
            {"id": "Weights", "kind": "weights", "outcome": "x", "required": ["y"]},
            {"id": "routing", "kind": "planner", "required": ["y"]},
            {"id": "state", "kind": "state", "outcome": "o", "depends_on": ["nope"],
             "check": {"gpus": 2, "nodes": 1}},
        ],
    }  # fmt: skip
    path = tmp_path / "vocabulary.yaml"
    path.write_text(yaml.safe_dump(vocabulary))
    with pytest.raises(VocabularyError) as raised:
        rows_mod.load_rows(path)
    said = str(raised.value)
    for problem in ("rows[0].id is 'Weights'", "rows[1].kind is 'planner'",
                    "rows[1] states no `outcome`",
                    "states `gpus` and `nodes`", "row 'state' depends on 'nope'"):  # fmt: skip
        assert problem in said
    vocabulary["rows"] = [{"id": "state", "kind": "state", "outcome": "o", "required": ["r"],
                           "deployment": "pd", "check": {"gpus": 2, "wall": 900}}]  # fmt: skip
    path.write_text(yaml.safe_dump(vocabulary))
    (row,) = rows_mod.load_rows(path)
    assert (row.check, row.job_line, row.deployment) == ("cpu:state", "gpus=2 wall=900", "pd")
    # The skeleton: every directory on the way to a row's entry that is not yet a package.
    tree = tmp_path / "tree"
    family = tree / "pkg" / "models" / "fam"
    family.mkdir(parents=True)
    for directory in (tree / "pkg", tree / "pkg" / "models", family):
        (directory / "__init__.py").write_text("")
    files = ["pkg/models/fam/new/ops/norm/rms.py", "pkg/models/fam/new/ops/attention/gqa.py",
             "tests/unit/models/fam/new/norm/test_rms.py"]  # fmt: skip
    owed = tasks_mod.skeleton_files(
        tree, files, ("pkg/models/fam/new", "tests/unit/models/fam/new")
    )
    assert owed == [
        "tests/unit/models/fam/new/__init__.py",
        "pkg/models/fam/new/__init__.py",
        "pkg/models/fam/new/ops/__init__.py",
        "pkg/models/fam/new/ops/norm/__init__.py",
        "pkg/models/fam/new/ops/attention/__init__.py",
        "tests/unit/models/fam/new/norm/__init__.py",
    ]
    for rel in owed:
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text("")
    assert (
        tasks_mod.skeleton_files(tree, files, ("pkg/models/fam/new", "tests/unit/models/fam/new"))
        == []
    )
    assert json.loads(json.dumps(row.as_dict()))["kind"] == "state" and Path(path).is_file()


def test_the_wait_command_the_guide_shows_is_one_the_monitor_takes(tmp_path):
    """Found by running the guide's loop against the command: the workspace goes before `wait`."""
    guide = (Path(esc.__file__).parents[1] / "OPERATING.md").read_text(encoding="utf-8")
    shown = re.findall(r"monitor (--workspace <ws> wait[^`\n]*)", guide)
    assert len(shown) == 3 and "monitor wait --" not in guide
    for form in shown:
        argv = shlex.split(form.replace("<ws>", str(tmp_path)).replace("<task>", "k1"))
        args = monitor.build_parser().parse_args(argv)
        assert (args.action, args.workspace) == ("wait", str(tmp_path))
        assert args.ignore == (["k1"] if "--ignore" in form else [])
    # The other order is a usage error, which a watcher must not read as "nothing new".
    with pytest.raises(SystemExit) as refused:
        monitor.build_parser().parse_args(["wait", "--workspace", str(tmp_path)])
    assert refused.value.code == 2
    # The guide's other commands, as their own parsers take them.
    shown = re.search(
        r"trtllm-modeling-v2 resolve (--workspace <ws> <verb> <task\|run\|pool>)", guide
    )
    taken = resolve._parser().parse_args(shlex.split(shown.group(1).replace("|", "-")))
    assert (taken.verb, taken.name) == ("<verb>", "<task-run-pool>")
    assert 'trtllm-modeling-v2 feedback --workspace <ws> "<note>"' in guide
    assert cli.main is not None and "feedback" in cli.SUBCOMMANDS and "monitor" in cli.SUBCOMMANDS
