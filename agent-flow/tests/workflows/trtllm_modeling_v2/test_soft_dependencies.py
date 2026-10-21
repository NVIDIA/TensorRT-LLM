# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A soft dependency starts its dependent beside it and holds it back only at its integration.

A forward that needs a one-line engine fix only where its own checks run
should not wait hours for the fix to land before it starts. With
``depends_on_soft`` it starts once the fix has started; when its worker
finishes, its integration waits for the fix to land or end, so its branch is
rebased onto a head that holds the fix and its checks run there. A fix that
fails is named in the dependent's receipt and the planner's events.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator as orchestrator_mod
from agent_flow.workflows.trtllm_modeling_v2.run.integration import (
    Integration,
    soft_dependency_block,
)
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)

from .test_integration import _ENV, _commit, _git, _repo, _Stage

_WAIT = 10.0
_CONTRACT = c.Contract(checks=(c.Check("core", "boot", "tp1", c.TIER_INNER),))


def _t(tid, kind="feature", deps=(), soft=(), status="pending", **kw):
    return g.Task(
        id=tid,
        kind=kind,
        goal=f"do {tid}",
        depends_on=list(deps),
        depends_on_soft=list(soft),
        touches=[f"models/{tid}.py"],
        status=status,
        **kw,
    )


# ------------------------------------------------------------------ the graph


def test_a_soft_dependency_lets_its_dependent_start_once_it_has_started():
    for status, starts in [
        ("pending", False),
        ("running", True),
        ("done", True),
        ("failed", True),
        ("abandoned", True),
    ]:
        graph = g.Graph(tasks=[_t("fix", "kernel", status=status), _t("core", soft=["fix"])])
        assert ("core" in [t.id for t in g.ready(graph)]) is starts, status
    # A hard dependency beside it still holds it until done.
    graph = g.Graph(
        tasks=[
            _t("fix", "kernel", status="running"),
            _t("k1", "kernel", status="running"),
            _t("core", deps=["k1"], soft=["fix"]),
        ]
    )
    assert [t.id for t in g.ready(graph)] == []


def test_soft_edges_are_read_and_written_with_the_graph(tmp_path):
    path = tmp_path / g.GRAPH_FILENAME
    path.write_text(
        "tasks:\n"
        "  - {id: fix, kind: kernel, goal: x}\n"
        "  - {id: core, kind: assemble, goal: y, depends_on_soft: [fix]}\n"
    )
    graph = g.Graph.load(path)
    assert graph.by_id()["core"].depends_on_soft == ["fix"]
    graph.save(path)
    assert g.Graph.load(path).by_id()["core"].depends_on_soft == ["fix"]


def test_inadmissible_soft_edges_are_named():
    errors = g.validate(
        g.Graph(
            tasks=[
                _t("fix", "kernel"),
                _t("read", "review"),
                _t("a", soft=["nope"]),
                _t("b", soft=["b"]),
                _t("c", deps=["fix"], soft=["fix"]),
                _t("d", soft=["read"]),
                _t("e", "review", soft=["fix"]),
            ]
        )
    )
    assert "task 'a': soft-depends on unknown task 'nope'" in errors
    assert "task 'b': soft-depends on itself" in errors
    assert any("'c'" in e and "both `depends_on` and `depends_on_soft`" in e for e in errors)
    assert any("'d'" in e and "a 'review' task, which lands nothing" in e for e in errors)
    assert any("'e'" in e and "a 'review' task is not integrated" in e for e in errors)


def test_soft_and_hard_edges_together_form_no_cycle():
    errors = g.validate(g.Graph(tasks=[_t("a", deps=["b"]), _t("b", "kernel", soft=["a"])]))
    assert any(e.startswith("dependency cycle:") for e in errors)


def test_a_soft_dependency_is_on_its_dependents_critical_path():
    graph = g.Graph(
        tasks=[
            _t("fix", "kernel", est_minutes=10),
            _t("other", "kernel", est_minutes=30),
            _t("core", soft=["fix"], est_minutes=300),
        ]
    )
    path = g.critical_path_minutes(graph)
    assert path["fix"] == 310 and path["other"] == 30
    assert [t.id for t in g.next_to_start(graph, 8)] == ["fix", "other"]


# ------------------------------------------------------------------- the loop


def _loop(tmp_path, tasks, work):
    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            return PlannerAnswer("GRAPH_UPDATED", "plan", g.Graph(tasks=list(tasks)))
        return PlannerAnswer("DONE", "all green")

    services = Services(
        plan=plan,
        work=work,
        check=lambda chk: SimpleNamespace(name=chk.gate, member=chk.member, passed=True, detail=""),
        free_gpus=lambda: 8,
    )
    orchestrator = GraphOrchestrator(
        services, _CONTRACT, c.ContractStatus.load(tmp_path), tmp_path, poll_s=0.01
    )
    return orchestrator, turns


def test_the_dependent_runs_beside_its_soft_dependency_and_integrates_after_it(tmp_path):
    started, seen, orchestrator = threading.Event(), {}, None
    fix = _t("fix", "kernel")
    core = _t("core", "assemble", soft=["fix"], satisfies=["core:boot@tp1"])

    def work(task):
        if task.id == "fix":
            # The fix finishes only once its dependent has started beside it.
            seen["core started while fix ran"] = started.wait(_WAIT)
            return "LANDED"
        started.set()
        seen["at its integration"] = orchestrator.soft_dependencies("core")
        seen["fix then"] = orchestrator.graph.by_id()["fix"].status
        return "UNIT_READY"

    orchestrator, _ = _loop(tmp_path, [fix, core], work)
    assert orchestrator.run() == "delivered"
    assert seen == {
        "core started while fix ran": True,
        "at its integration": {"fix": "done"},
        "fix then": "done",
    }


def test_a_soft_dependency_that_fails_is_named_and_the_dependent_goes_on(tmp_path):
    started, seen, orchestrator = threading.Event(), {}, None

    def work(task):
        if task.id == "fix":
            started.wait(_WAIT)
            return "BLOCKED"
        started.set()
        seen["ended"] = orchestrator.soft_dependencies("core")
        return "UNIT_READY"

    tasks = [
        _t("fix", "kernel"),
        _t("core", "assemble", soft=["fix"], satisfies=["core:boot@tp1"]),
    ]
    orchestrator, turns = _loop(tmp_path, tasks, work)
    assert orchestrator.run() == "delivered"
    assert seen["ended"] == {"fix": "failed"}
    events = [e for t in turns for e in t.events]
    assert any(
        "core soft-depend on fix, which ended failed: they go on without its change" in e
        for e in events
    )


def test_a_task_waiting_on_its_soft_dependencies_is_released_when_the_loop_ends(tmp_path):
    seen, returned, orchestrator = {}, threading.Event(), None
    with orchestrator_mod.stoppable(tmp_path) as stop:

        def work(task):
            if task.id == "fix":
                returned.wait(_WAIT)
                return "LANDED"
            stop("the host failed")
            seen["core"] = orchestrator.soft_dependencies("core")
            returned.set()
            return "UNIT_READY"

        core = _t("core", soft=["fix"], satisfies=["core:boot@tp1"])
        orchestrator, _ = _loop(tmp_path, [_t("fix", "kernel"), core], work)
        assert orchestrator.run() == "stopped: the host failed"
        assert returned.wait(_WAIT)
    assert seen["core"] is None


# ------------------------------------------------------------ the stage's gate


def test_the_integration_waits_and_lands_onto_a_head_that_holds_the_soft_dependency(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)
    fix_tree = integration.open("fix")
    _commit(fix_tree, "b.py", "B = 2\n", "the engine fix")

    def worker(turn):
        tree = integration.tree("core")
        _commit(tree, "a.py", "A = 2\n", "the forward")
        return "UNIT_READY"

    stage = _Stage(tmp_path / "ws", integration, worker)
    waited = []

    def soft(task_id):
        # The fix lands while the dependent waits at its integration.
        waited.append(task_id)
        assert integration.integrate("fix").merged
        return {"fix": "done"}

    stage._soft_dependencies = soft
    core = _t("core", "assemble", soft=["fix"])
    assert stage._work_task_turn(core) == "UNIT_READY"
    assert waited == ["core"]
    subjects = _git(repo, "log", "--format=%s", "-3").splitlines()
    assert subjects == ["the forward", "the engine fix", "base"]
    assert (repo / "a.py").read_text() == "A = 2\n" and (repo / "b.py").read_text() == "B = 2\n"


def test_a_soft_dependency_that_ended_without_landing_is_in_the_decision(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)

    def worker(turn):
        _commit(integration.tree("core"), "a.py", "A = 2\n", "the forward")
        return "UNIT_READY\nthe forward is in"

    stage = _Stage(tmp_path / "ws", integration, worker)
    stage._soft_dependencies = lambda task_id: {"fix": "failed", "k1": "done"}
    decision = stage._work_task_turn(_t("core", "assemble", soft=["fix", "k1"]))
    head, *notes = decision.splitlines()
    assert head == "UNIT_READY" and notes[0] == "the forward is in"
    assert notes[1] == (
        "soft dependencies fix (failed) ended without landing: this task integrated without "
        "their change"
    )
    assert _git(repo, "log", "-1", "--format=%s") == "the forward"


def test_a_loop_that_ends_first_leaves_the_task_unintegrated(tmp_path):
    repo = _repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _ENV)

    def worker(turn):
        _commit(integration.tree("core"), "a.py", "A = 2\n", "the forward")
        return "UNIT_READY"

    stage = _Stage(tmp_path / "ws", integration, worker)
    stage._soft_dependencies = lambda task_id: None
    decision = stage._work_task_turn(_t("core", "assemble", soft=["fix"]))
    assert decision.startswith("the run ended")
    assert _git(repo, "log", "-1", "--format=%s") == "base"


def test_the_worker_is_told_where_each_soft_dependency_stands(tmp_path):
    block = soft_dependency_block(
        {"fix": "running", "k1": "done", "k2": "failed"}, tmp_path / "tasks", "/repo"
    )
    assert block.startswith("## Soft dependencies")
    assert "`fix` is running: its change lands while you work" in block
    assert f"`{tmp_path / 'tasks' / 'k1.receipt.yaml'}`" in block
    assert "`k2` ended failed: its change will not land" in block
    assert "rebase $(git -C /repo rev-parse HEAD)" in block
    assert soft_dependency_block({}, tmp_path, "/repo") == ""
