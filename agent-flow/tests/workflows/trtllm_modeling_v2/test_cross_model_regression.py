# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task that changes code other models run is followed by a regression check the orchestrator adds."""

from __future__ import annotations

from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)

_PKG = "tensorrt_llm/_torch/_experimental/modeling_v2"
_TARGET = f"{_PKG}/models/fam/ckpt__sm_100__unified"
_TESTS = "tests/unittest/_torch/modeling_v2/models/fam/ckpt__sm_100__unified"
_OWN = g.own_paths(_TARGET)
_CATALOG = f"{_PKG}/catalog/ssm/conv1d.py"
_EXECUTOR = "tensorrt_llm/_torch/pyexecutor/"


def _t(tid, kind="kernel", deps=(), touches=(), **kw):
    return g.Task(
        id=tid, kind=kind, goal=f"do {tid}", depends_on=list(deps), touches=list(touches), **kw
    )


def _check(tid="regression", deps=(), **kw):
    kw.setdefault("regression", True)
    return _t(tid, g.CHECK_KIND, deps, **kw)


def _follow(*tasks):
    """The graph after the orchestrator followed its shared changes, and the check it added."""
    graph = g.Graph(tasks=list(tasks))
    added = g.add_regression_check(graph, _OWN)
    return graph, added


def test_work_inside_the_target_and_its_tests_needs_no_regression_check():
    core = _t("core", "assemble", touches=[f"{_TARGET}/modeling.py", f"{_TESTS}/test_x.py"])
    ops = _t("k1", touches=[f"{_TARGET}/ops/norm/", f"./{_TESTS}/norm/test_rmsnorm.py"])
    graph, added = _follow(core, ops)
    assert added is None and len(graph.tasks) == 2


def test_a_change_to_shared_code_is_followed_by_one_regression_check():
    graph, added = _follow(
        _t("k1", touches=[_CATALOG, f"{_TARGET}/ops/a.py"]),
        _t("core", "assemble", ["k1"], touches=[_EXECUTOR, f"{_TARGET}/modeling.py"]),
        _t("mine", "feature", ["core"], touches=[f"{_TARGET}/mtp.py"]),
    )
    assert added is not None and added.id == g.REGRESSION_CHECK_ID
    assert added.kind == g.CHECK_KIND and added.regression and added.priority == "required"
    assert added.depends_on == ["k1", "core"] and added.satisfies == [] and added.gpus == 0
    # Admissible as it stands, and nothing more is added while it is pending.
    assert g.validate(graph) == []
    assert g.add_regression_check(graph, _OWN) is None and len(graph.tasks) == 4


# ------------------------------------------------------------- the run loop


def test_the_loop_adds_the_check_runs_it_with_no_turn_and_delivers(tmp_path):
    contract = c.Contract(checks=(c.Check("core", "boot", "tp1", c.TIER_INNER),))
    core = _t("core", "assemble", touches=[_EXECUTOR], satisfies=["core:boot@tp1"])
    answers = [
        PlannerAnswer("GRAPH_UPDATED", "first plan", g.Graph(tasks=[core])),
        PlannerAnswer("DONE", "all green"),
    ]
    turns = []

    def plan(turn):
        turns.append(turn)
        return answers[min(len(turns), len(answers)) - 1]

    worked, checked = [], []

    def work(task):
        worked.append(task.id)
        return "UNIT_READY"

    services = Services(
        plan=plan,
        work=work,
        check=lambda chk: SimpleNamespace(
            name=chk.gate, member=chk.member, passed=True, detail="", tree_state=""
        ),
        free_gpus=lambda: 4,
        regression=lambda task: checked.append((task.id, task.covers)) or [],
    )
    status = c.ContractStatus.load(tmp_path)
    orchestrator = GraphOrchestrator(services, contract, status, tmp_path, poll_s=0, own_paths=_OWN)
    assert orchestrator.run() == "delivered"
    assert all(not t.refusals for t in turns)
    # No agent turn for the check: only core worked, and the check covered what it landed.
    assert worked == ["core"]
    assert checked == [("regression", ["core@1"])]


def test_a_failed_regression_check_reaches_the_planner(tmp_path):
    contract = c.Contract(checks=(c.Check("core", "boot", "tp1", c.TIER_INNER),))
    core = _t("core", "assemble", touches=[_CATALOG], satisfies=["core:boot@tp1"])
    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            return PlannerAnswer("GRAPH_UPDATED", "first plan", g.Graph(tasks=[core]))
        if any("tests/x.py::test_y" in event for event in turn.events):
            return PlannerAnswer("ESCALATE", "seen")
        return PlannerAnswer("DONE", "waiting")

    services = Services(
        plan=plan,
        work=lambda task: "UNIT_READY",
        check=lambda chk: SimpleNamespace(
            name=chk.gate, member=chk.member, passed=True, detail="", tree_state=""
        ),
        free_gpus=lambda: 4,
        regression=lambda task: ["tests/x.py::test_y (passed at the base, fails now)"],
    )
    status = c.ContractStatus.load(tmp_path)
    orchestrator = GraphOrchestrator(services, contract, status, tmp_path, poll_s=0, own_paths=_OWN)
    assert orchestrator.run() == "escalated"
    failed = next(e for t in turns for e in t.events if "tests/x.py::test_y" in e)
    assert "regression" in failed


def test_after_a_failed_check_the_next_one_follows_the_next_change():
    """The same tree would fail the same way: nothing is added until a fix is planned."""
    landed = _t("k1", touches=[_CATALOG], status="done")
    failed = _check(deps=["k1"], status="failed", covers=["k1@1"])
    _, added = _follow(landed, failed)
    assert added is None
    fix = _t("fix", touches=[_CATALOG])
    _, added = _follow(landed, failed, fix)
    assert added is not None and added.depends_on == ["fix"]
