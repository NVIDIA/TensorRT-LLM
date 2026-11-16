# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 graph behavior and regression checks."""

from __future__ import annotations

import json

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.records.state import (
    STAGE_EXECUTION,
    WorkflowState,
    load_state,
    save_state,
)


def _t(tid, kind="kernel", deps=(), touches=(), goal=None, **kw):
    return g.Task(
        id=tid,
        kind=kind,
        goal=goal or f"do {tid}",
        depends_on=list(deps),
        touches=list(touches),
        **kw,
    )


def _graph(*tasks):
    return g.Graph(tasks=list(tasks))


def test_tasks_that_name_the_same_file_never_run_together():
    graph = _graph(
        _t("a", touches=["models/x/modeling.py"]),
        _t("b", touches=["models/x/", "./models/x/modeling.py"]),
        _t("c", touches=["models/x/weights.py"]),
    )
    assert [t.id for t in g.next_to_start(graph)] == ["a", "c"]
    graph.tasks[0].status = "running"
    assert [t.id for t in g.next_to_start(graph)] == ["c"]
    graph.tasks[0].status = "done"
    # A directory is what a task may write too: `b` was given `models/x/`, so the file `c`
    # names inside it waits for `b`.
    assert [t.id for t in g.next_to_start(graph)] == ["b"]
    graph.tasks[1].status = "done"
    assert [t.id for t in g.next_to_start(graph)] == ["c"]


def test_a_skipped_required_dependency_and_a_blocked_one_hold_their_dependents():
    graph = _graph(_t("a"), _t("b", deps=["a"]), _t("c", deps=["b"]), _t("d"))
    assert [t.id for t in g.ready(graph)] == ["a", "d"]
    graph.tasks[0].status = "blocked"
    assert [t.id for t in g.ready(graph)] == ["d"]
    assert g.blocked_by(graph, "a") == ["b", "c"]
    # Skipped and required: nothing produced what `b` needs, so it waits too.
    graph.tasks[0].status = "skipped"
    assert [t.id for t in g.ready(graph)] == ["d"]
    # Skipped and only nice to have: what depended on it goes on.
    graph.tasks[0].priority = "nice_to_have"
    assert [t.id for t in g.ready(graph)] == ["b", "d"]
    # Required tasks start before optional ones, each in the graph's order.
    graph = _graph(_t("x", priority="nice_to_have"), _t("y"), _t("z"))
    assert [t.id for t in g.next_to_start(graph)] == ["y", "z", "x"]


def test_a_graph_is_saved_with_what_is_blocked_and_an_older_flows_is_refused(tmp_path):
    path = tmp_path / g.GRAPH_FILENAME
    task = _t("k", touches=["a.py"], check="kernel:3", status="blocked", last="it failed",
              block={"by": "gate", "kind": "attempts_exhausted", "question": "which verb?"},
              attempts=2, budget=4, ref="3")  # fmt: skip
    asked = {"kind": "final_tests_failed", "observed": "2 cases fail"}
    g.Graph(tasks=[task], run_blocked=asked).save(path)
    loaded = g.Graph.load(path)
    assert loaded.tasks == [task] and loaded.run_blocked == asked
    # What an earlier flow kept of its escalations and commands is read past.
    saved = yaml.safe_load(path.read_text())
    saved["tasks"][0]["escalation"], saved["consumed"] = "esc-k-2", ["esc-k-1.retry.0123456789"]
    path.write_text(yaml.safe_dump(saved))
    assert g.Graph.load(path).tasks == [task]
    g.Graph(tasks=[task]).save(path)
    assert g.removed_kinds(path) == []
    path.write_text(json.dumps({
        "tasks": [{"id": "t", "kind": "tune"}, {"id": "p", "kind": "kernel", "status": "failed"},
                  {"id": "r", "kind": "check", "regression": True}],
        "revisions": [{"number": 1, "reason": "plan"}],
    }))  # fmt: skip
    assert g.removed_kinds(path) == [
        "t (tune)", "p (failed)", "r (a regression check)", "the planner's revisions",
    ]  # fmt: skip
    errors = g.validate(_graph(_t("c", kind="check", touches=["x.py"], satisfies=["nope"])),
                        known_checks=["core:boot@tp1"], required_checks=["core:boot@tp1"])  # fmt: skip
    assert any("nope is no check of the contract" in e for e in errors)
    assert any("its `touches` stay empty" in e for e in errors)
    assert any("required check 'core:boot@tp1' is in no task's `satisfies`" in e for e in errors)
    assert any(
        "status 'failed' is not one of" in e for e in g.validate(_graph(_t("x", status="failed")))
    )


def test_round_trip_preserves_every_field_and_an_unknown_version_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    original = WorkflowState(
        task_path="/x/task.yaml",
        stage=STAGE_EXECUTION,
        done=False,
        slug="qwen3-30b-a3b-sm_103-tp1",
        branch="trtllm-modeling-v2/qwen3-30b-a3b-sm_103-tp1",
        devices=[2, 3],
    )
    save_state(path, original)
    assert load_state(path) == original
    path.write_text(json.dumps({"version": 1, "task_path": "t", "stage": "plan"}))
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        load_state(path)
