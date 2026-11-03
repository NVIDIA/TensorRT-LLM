# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 graph behavior and regression checks."""

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
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c


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


def test_a_regression_check_may_wait_for_a_nice_to_have_task_and_not_for_ever():
    extra = g.Task(id="extra", kind="feature", goal="nice", priority="nice_to_have",
                   touches=["tensorrt_llm/_torch/shared.py"])  # fmt: skip
    review = g.Task(id="regression-2", kind="check", goal="run the package tests",
                    regression=True, depends_on=["extra"])  # fmt: skip
    graph = g.Graph(tasks=[extra, review])
    assert not [e for e in g.validate(graph) if "informs" in e]
    assert [t.id for t in g.ready(graph)] == ["extra"]
    # Ended without landing: nothing of it to review, so the check is not held for ever.
    extra.status, extra.attempts = "failed", 1
    assert [t.id for t in g.ready(graph)] == ["regression-2"]
    # Any other check still may not wait for it.
    review.regression = False
    assert any("'regression-2': depends on 'extra', a nice_to_have task" in e
               for e in g.validate(graph))  # fmt: skip
    assert g.ready(graph) == []


def test_a_campaign_waits_for_the_cores_required_checks_only():
    """A failing variant or feature does not hold the core's tuning back; delivery owes it."""
    contract = c.Contract(
        checks=(
            c.Check("core", "boot", "tp4", c.TIER_INNER),
            c.Check("core", "accuracy[gsm8k]", "dep4", c.TIER_INNER),
            c.Check("core", "serve", "tep4", c.TIER_DELIVERY),
            c.Check("core", "boot", "tp8", c.TIER_INNER, required=False),
            c.Check("disagg", "disagg", "dep4", c.TIER_PROOF),
            c.Check("kv_nvfp4", "accuracy[gsm8k]", "dep4", c.TIER_PROOF),
        )
    )
    core = ["core:boot@tp4", "core:accuracy[gsm8k]@dep4", "core:serve@tep4"]
    assert [k.key for k in c.core_checks(contract)] == core
    assert [k.key for k in c.core_checks(contract, "core")] == core
    assert [k.key for k in c.core_checks(contract, "disagg")] == ["disagg:disagg@dep4"]
    assert c.core_checks(c.Contract(checks=())) == ()


def test_round_trip_preserves_every_field_and_an_unknown_version_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    original = WorkflowState(
        task_path="/x/task.yaml",
        stage=STAGE_EXECUTION,
        done=False,
        slug="qwen3-30b-a3b-sm_103-tp1",
        branch="trtllm-modeling-v2/qwen3-30b-a3b-sm_103-tp1",
        devices=[2, 3],
        tune={"phase": "campaign"},
    )
    save_state(path, original)
    assert load_state(path) == original
    path.write_text(json.dumps({"version": 1, "task_path": "t", "stage": "plan"}))
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        load_state(path)
