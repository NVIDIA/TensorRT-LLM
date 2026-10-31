# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The task graph: what makes a planner's graph admissible, and what starts next."""

from __future__ import annotations

from agent_flow.workflows.trtllm_modeling_v2.run import graph as g


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
    assert not [e for e in g.validate(g.Graph(tasks=[probe, extra, gate, core, kernel]))
                if "informs" in e or "never waits" in e]  # fmt: skip
