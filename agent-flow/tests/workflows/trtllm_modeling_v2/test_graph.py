# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 graph behavior and the run's own paths."""

from __future__ import annotations

import json

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.records.state import (
    STAGE_EXECUTION,
    WorkflowState,
    load_state,
    save_state,
)


def _t(tid, kind="kernel", deps=(), touches=(), gpus=1, minutes=60, goal=None, **kw):
    return g.Task(
        id=tid,
        kind=kind,
        goal=goal or f"do {tid}",
        depends_on=list(deps),
        touches=list(touches),
        gpus=gpus,
        est_minutes=minutes,
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
    assert [t.id for t in g.next_to_start(graph, free_gpus=8)] == ["a", "c"]
    graph.tasks[0].status = "running"
    assert [t.id for t in g.next_to_start(graph, free_gpus=8)] == ["c"]
    graph.tasks[0].status = "done"
    assert [t.id for t in g.next_to_start(graph, free_gpus=8)] == ["b", "c"]


def test_an_investigation_never_gates_a_check_and_a_check_never_gates_a_kernel():
    probe = g.Task(id="probe", kind="investigate", goal="find out why")
    extra = g.Task(id="extra", kind="feature", goal="nice", priority="nice_to_have")
    gate = g.Task(id="gate", kind="check", goal="run", satisfies=["core:boot@tp1"])
    core = g.Task(id="core", kind="assemble", goal="core", satisfies=["core:boot@tp1"],
                  depends_on=["probe", "extra"])  # fmt: skip
    kernel = g.Task(id="k1", kind="kernel", goal="entry", depends_on=["gate"])
    errors = g.validate(g.Graph(tasks=[probe, extra, gate, core, kernel]))
    assert any("'core': depends on 'probe', an `investigate` task" in e for e in errors)
    assert any("'core': depends on 'extra', a nice_to_have task" in e for e in errors)
    assert any("'k1': depends on 'gate', a `check` task" in e for e in errors)
    core.depends_on, kernel.depends_on = [], []
    assert not [
        e
        for e in g.validate(g.Graph(tasks=[probe, extra, gate, core, kernel]))
        if "informs" in e or "never waits" in e
    ]


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


def test_a_task_touches_only_the_runs_own_paths_and_its_commits_are_held_to_them():
    """Code other models run is not the run's: a shared kernel lands as a local variant."""
    from agent_flow.workflows.trtllm_modeling_v2.infra.git import CommitPolicy

    package = "tensorrt_llm/_torch/_experimental/modeling_v2"
    target = f"{package}/models/fam/tgt"
    own = g.own_paths(target)
    assert own == (
        f"{package}/models/fam",
        "tests/unittest/_torch/modeling_v2/models/fam",
        f"{package}/models/__init__.py",
        f"{package}/_router_index.py",
        "tests/integration",
    )
    assert g.own_paths("") == ()
    allowed = ("tensorrt_llm/", "tests/")
    inside = _t("sibling", touches=[f"{package}/models/fam/ops/gemm/mm__r1.py"])
    shared = _t("runtime", "feature", touches=["tensorrt_llm/_torch/pyexecutor/py_executor.py"])
    errors = g.validate(_graph(inside, shared), allowed_paths=allowed, own_paths=own)
    assert len(errors) == 1 and "task 'runtime': touches" in errors[0]
    assert "code other models run" in errors[0]
    # The operator's own allowed paths decide alone: no own-path rule then.
    assert g.validate(_graph(inside, shared), allowed_paths=allowed) == []
    policy = CommitPolicy(allowed_paths=allowed).within(own)
    assert policy.path_problem(f"{package}/models/fam/tgt/modeling.py") == ""
    assert policy.path_problem(f"{package}/models/__init__.py") == ""
    assert policy.path_problem(f"{package}/models/family_two/x.py")
    assert policy.path_problem("tensorrt_llm/_torch/pyexecutor/py_executor.py")
    assert CommitPolicy(allowed_paths=allowed).within(()) == CommitPolicy(allowed_paths=allowed)
