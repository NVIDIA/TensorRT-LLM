# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task that changes code other models run is followed by a cross-model regression review."""

from __future__ import annotations

from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.integration import INTEGRATED_KINDS
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


def _review(tid="xmodel", deps=(), **kw):
    kw.setdefault("regression", True)
    return _t(tid, "review", deps, **kw)


def _errors(*tasks):
    return g.validate(g.Graph(tasks=list(tasks)), own_paths=_OWN)


def test_the_runs_own_paths_are_its_target_and_the_directory_its_tests_mirror_it_in():
    assert _OWN == (_TARGET, _TESTS)
    assert g.own_paths(_TARGET + "/") == _OWN
    assert g.own_paths("") == ()


def test_work_inside_the_target_and_its_tests_needs_no_regression_review():
    core = _t("core", "assemble", touches=[f"{_TARGET}/modeling.py", f"{_TESTS}/test_x.py"])
    ops = _t("k1", touches=[f"{_TARGET}/ops/norm/", f"./{_TESTS}/norm/test_rmsnorm.py"])
    assert _errors(core, ops) == []


def test_a_change_to_shared_code_without_a_regression_review_is_refused_by_name():
    errors = _errors(
        _t("k1", touches=[_CATALOG, f"{_TARGET}/ops/a.py"]),
        _t("core", "assemble", ["k1"], touches=[_EXECUTOR, f"{_TARGET}/modeling.py"]),
        _t("mine", "feature", ["core"], touches=[f"{_TARGET}/mtp.py"]),
    )
    assert len(errors) == 1
    error = errors[0]
    assert "no cross-model regression review follows 2 task(s)" in error
    assert ": k1, core." in error  # the ids, ready to be a review's depends_on
    assert _CATALOG in error and "tensorrt_llm/_torch/pyexecutor" in error
    assert f"{_TARGET}/modeling.py" not in error.split("The shared paths")[1]
    assert "`regression: true`" in error and "before the delivery tier" in error


def test_one_required_regression_review_after_them_all_admits_the_graph():
    k1 = _t("k1", touches=[_CATALOG])
    core = _t("core", "assemble", ["k1"], touches=[_EXECUTOR])
    assert _errors(k1, core, _review(deps=["core"])) == []  # k1 through core
    assert _errors(k1, core, _review(deps=["k1", "core"])) == []


def test_a_family_directory_or_another_target_is_shared():
    family = _t("k1", touches=[f"{_PKG}/models/fam/"])
    other = _t("k2", touches=[f"{_PKG}/models/other/x__sm_100__unified/ops/a.py"])
    native = _t("k3", touches=["cpp/tensorrt_llm/kernels/conv.cu"])
    errors = _errors(family, other, native)
    assert ": k1, k2, k3." in errors[0]


def test_only_work_that_lands_on_the_target_is_counted():
    # A tune campaign commits on its own branches, an investigation on a scratch
    # tree, a review nowhere; the integrate task that merges a campaign does land.
    assert (
        _errors(
            _t("tune", "tune", touches=[_EXECUTOR]),
            _t("why", "investigate", touches=[_CATALOG]),
            _t("read", "review", touches=[_EXECUTOR]),
        )
        == []
    )
    assert "integrate-tune" in _errors(_t("integrate-tune", "integrate", touches=[_EXECUTOR]))[0]
    assert INTEGRATED_KINDS == {"kernel", "assemble", "feature", "integrate"}


def test_failed_and_abandoned_tasks_are_not_counted():
    assert (
        _errors(
            _t("k1", touches=[_CATALOG], status="failed"),
            _t("extra", "feature", touches=[_EXECUTOR], status="abandoned"),
        )
        == []
    )


def test_only_a_required_live_regression_review_counts():
    k1 = _t("k1", touches=[_CATALOG])
    assert _errors(k1, _review(deps=["k1"], priority="nice_to_have"))
    assert _errors(k1, _review(deps=["k1"], status="failed"))
    assert _errors(k1, _review(deps=["k1"], status="abandoned"))
    assert _errors(k1, _review(deps=["k1"], regression=False))


def test_regression_marks_only_a_review():
    errors = g.validate(g.Graph(tasks=[_t("k1", regression=True)]))
    assert errors == [
        "task 'k1': `regression: true` marks a cross-model regression review, "
        "and a 'kernel' task is not a review"
    ]


def test_without_the_runs_own_paths_nothing_is_asked():
    assert g.validate(g.Graph(tasks=[_t("k1", touches=[_CATALOG])])) == []


def test_a_review_that_started_follows_only_the_tasks_that_were_done():
    # A dependency written onto a review after it started held nothing back.
    late = _t("late", touches=[_CATALOG])
    early = _t("early", touches=[_EXECUTOR], status="done")
    for status in ("running", "done"):
        errors = _errors(early, late, _review(deps=["early", "late"], status=status))
        assert ": late." in errors[0]
    assert _errors(early, late, _review(deps=["early", "late"])) == []


def test_a_resumed_run_needs_one_review_after_the_shared_work_already_done():
    """A graph from before the rule: done shared work, a done review not marked."""
    done = dict(status="done")
    tasks = [
        _t("k1", touches=[_CATALOG], **done),
        _t("core", "assemble", ["k1"], touches=[_EXECUTOR], **done),
        _t("fix", "feature", ["core"], touches=["tensorrt_llm/_torch/speculative/"], **done),
        _review("xmodel-regression", deps=["core"], regression=False, **done),
    ]
    errors = _errors(*tasks)
    assert ": k1, core, fix." in errors[0]
    # Marked, the done review follows what it depended on, and only that.
    tasks[-1].regression = True
    assert ": fix." in _errors(*tasks)[0]
    # A new review after the rest, held back by nothing already done, admits it.
    assert _errors(*tasks, _review("xmodel-2", deps=["fix"])) == []
    assert [t.id for t in g.ready(g.Graph(tasks=[*tasks, _review("xmodel-2", deps=["fix"])]))] == [
        "xmodel-2"
    ]


def test_regression_is_read_and_written_with_the_graph(tmp_path):
    path = tmp_path / g.GRAPH_FILENAME
    path.write_text(
        "tasks:\n"
        "  - {id: a, kind: review, goal: x}\n"
        "  - {id: b, kind: review, goal: y, regression: true}\n"
        "  - {id: c, kind: review, goal: z, regression: 'false'}\n"
    )
    graph = g.Graph.load(path)
    assert [t.regression for t in graph.tasks] == [False, True, False]
    graph.save(path)
    assert [t.regression for t in g.Graph.load(path).tasks] == [False, True, False]


def test_a_revision_without_the_review_is_refused_and_one_with_it_adopted():
    current = g.Graph(tasks=[_t("k1", touches=[_CATALOG], status="done")])
    proposed = g.Graph(tasks=[_t("k1", touches=[_CATALOG]), _t("k2", touches=[_EXECUTOR])])
    adopted, errors = g.apply_revision(current, proposed, "add k2", own_paths=_OWN)
    assert adopted is current and ": k1, k2." in errors[0]
    proposed.tasks.append(_review(deps=["k1", "k2"]))
    adopted, errors = g.apply_revision(current, proposed, "add k2", own_paths=_OWN)
    assert errors == [] and adopted.by_id()["xmodel"].regression


# ------------------------------------------------------------- the run loop


def test_the_loop_sends_a_graph_without_the_review_back_to_the_planner(tmp_path):
    contract = c.Contract(checks=(c.Check("core", "boot", "tp1", c.TIER_INNER),))
    core = _t("core", "assemble", touches=[_EXECUTOR], satisfies=["core:boot@tp1"])
    answers = [
        PlannerAnswer("GRAPH_UPDATED", "first plan", g.Graph(tasks=[core])),
        PlannerAnswer(
            "GRAPH_UPDATED",
            "add the regression review",
            g.Graph(tasks=[core, _review(deps=["core"])]),
        ),
        PlannerAnswer("DONE", "all green"),
    ]
    turns = []

    def plan(turn):
        turns.append(turn)
        return answers[min(len(turns), len(answers)) - 1]

    worked = []

    def work(task):
        worked.append(task.id)
        return "APPROVE" if task.kind == "review" else "UNIT_READY"

    services = Services(
        plan=plan,
        work=work,
        check=lambda chk: SimpleNamespace(
            name=chk.gate, member=chk.member, passed=True, detail="", tree_state=""
        ),
        free_gpus=lambda: 4,
    )
    status = c.ContractStatus.load(tmp_path)
    orchestrator = GraphOrchestrator(services, contract, status, tmp_path, poll_s=0, own_paths=_OWN)
    assert orchestrator.run() == "delivered"
    assert any("no cross-model regression review follows 1 task(s)" in r for r in turns[1].refusals)
    assert worked == ["core", "xmodel"]
