# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Continuous verification: the target is checked as it changes, not only at the end."""

from __future__ import annotations

import threading

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g

from .test_orchestrator import PlannerAnswer, _orchestrator, _plan_once, _services, _task

_KEYS = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]


def test_a_members_delivery_tier_is_rehearsed_once_its_other_checks_hold(tmp_path):
    """The envelope runs while later work is still going, as soon as tp1's other checks hold."""
    rehearsed = threading.Event()
    seen = {}
    services = _services(
        _plan_once(_task("core", "assemble", _KEYS), _task("later", "feature", deps=["core"]))
    )
    check = services.check

    def counted(chk):
        if chk.key == "core:envelope@tp1":
            rehearsed.set()
        return check(chk)

    def work(task):
        if task.id == "later":
            seen["envelope ran beside later"] = rehearsed.wait(timeout=10)
        return "UNIT_READY"

    services.check, services.work = counted, work
    orchestrator = _orchestrator(tmp_path, services)
    assert orchestrator.run() == "delivered"
    assert seen["envelope ran beside later"] is True
    assert services.ran_checks.count("core:envelope@tp1") == 1  # its pass stands on one tree


def test_a_failed_rehearsal_reaches_the_planner_while_the_work_goes_on(tmp_path):
    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            tasks = [_task("core", "assemble", _KEYS), _task("later", "feature", deps=["core"])]
            return PlannerAnswer("GRAPH_UPDATED", "plan", g.Graph(tasks=tasks))
        if any("core:envelope@tp1" in e for e in turn.events):
            return PlannerAnswer("ESCALATE", "the envelope does not fit")
        return PlannerAnswer("GRAPH_UPDATED", "keep going", g.Graph(tasks=orchestrator.graph.tasks))

    gate = threading.Event()
    services = _services(plan, failing={"core:envelope@tp1"})
    services.work = (
        lambda task: (gate.wait(timeout=10) and "UNIT_READY")
        if task.id == "later"
        else ("UNIT_READY")
    )
    orchestrator = _orchestrator(tmp_path, services)
    timer = threading.Timer(3, gate.set)
    timer.start()
    try:
        assert orchestrator.run() == "escalated"
    finally:
        gate.set()
        timer.cancel()
    said = [e for t in turns for e in t.events if "core:envelope@tp1" in e]
    assert said and "beside the work in flight" in said[0]


def test_the_orchestrator_adds_the_task_that_writes_the_delivery_files(tmp_path):
    asked = []

    def delivery_task(graph):
        asked.append(len(graph.tasks))
        if any(t.id == "delivery-files" for t in graph.tasks):
            return None
        return {
            "id": "delivery-files",
            "kind": "feature",
            "goal": "write the CI accuracy test",
            "touches": ["tests/integration/defs/accuracy/test_x.py"],
            "depends_on": ["core"],
            "resources": {"gpus": 0, "est_minutes": 45},
        }

    worked = []
    services = _services(_plan_once(_task("core", "assemble", _KEYS)))
    services.delivery_task = delivery_task
    services.work = lambda task: worked.append(task.id) or "UNIT_READY"
    orchestrator = _orchestrator(tmp_path, services)
    assert orchestrator.run() == "delivered"
    assert worked == ["core", "delivery-files"]
    assert orchestrator.graph.revisions[-1].reason == (
        "orchestrator: delivery-files writes the delivery files"
    )


def test_once_no_required_task_is_open_nothing_more_lands_while_the_target_is_checked(tmp_path):
    services = _services(_plan_once())
    orchestrator = _orchestrator(tmp_path, services)
    done = _task("core", "assemble", _KEYS)
    done.status = "done"
    extra = _task("polish", "feature", priority="nice_to_have")
    probe = _task("probe", "investigate", priority="nice_to_have")
    orchestrator.graph = g.Graph(tasks=[done, extra, probe])
    # A nice_to_have that would change the tree waits; one that changes nothing may run.
    assert orchestrator._closing_holds() == {"polish"}
    open_ = _task("mtp", "feature")
    orchestrator.graph.tasks.append(open_)
    assert orchestrator._closing_holds() == set()


def test_rehearsal_runs_each_delivery_check_once(tmp_path):
    services = _services(_plan_once())
    orchestrator = _orchestrator(tmp_path, services)
    boot, acc, env = (orchestrator._check_by_key(k) for k in _KEYS)
    holding = c.Evaluation(passed=(boot, acc), failed=(), missing=(env,))
    assert orchestrator._rehearsal(holding) == ["core:envelope@tp1"]
    assert orchestrator._rehearsal(holding) == []
    partial = c.Evaluation(passed=(boot,), failed=(), missing=(acc,))
    fresh = _orchestrator(tmp_path / "other", _services(_plan_once()))
    assert fresh._rehearsal(partial) == []


def test_the_delivery_files_task_is_built_only_when_the_files_are_missing(tmp_path):
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.onboard.delivery import (
        ACCURACY_TESTS_DIR,
        TEST_DB_DIR,
    )
    from agent_flow.workflows.trtllm_modeling_v2.run.stages.deliver import DeliverStage

    repo = tmp_path / "repo"
    (repo / ACCURACY_TESTS_DIR).mkdir(parents=True)
    (repo / TEST_DB_DIR).mkdir(parents=True)

    class Stage(DeliverStage):
        task_data = {"trtllm_repo_path": str(repo), "accuracy": {}}
        _preflight = SimpleNamespace(family="newmodel")

        def _model_key(self, unit=None):
            return "org/new-model"

    stage = Stage()
    core = g.Task(id="core", kind="assemble", goal="core", ref="core")
    spec = stage._delivery_task(g.Graph(tasks=[core]))
    test = f"{ACCURACY_TESTS_DIR}/test_modeling_v2_newmodel.py"
    assert spec["id"] == "delivery-files" and spec["depends_on"] == ["core"]
    assert spec["touches"] == [test, f"{TEST_DB_DIR}/"]
    # One is enough: a graph that has it, or a task writing the test, needs no other.
    task = g.Task.from_dict(spec)
    assert stage._delivery_task(g.Graph(tasks=[core, task])) is None
    # Files already in place: nothing to write.
    (repo / test).write_text("def test_x(): pass\n")
    (repo / TEST_DB_DIR / "l0_x.yml").write_text("- test_modeling_v2_newmodel.py\n")
    assert stage._delivery_task(g.Graph(tasks=[core])) is None


def test_with_the_anchor_pending_the_delivery_files_task_writes_no_reference_entry(tmp_path):
    """An agent asked for a reference entry with no written anchor could only invent one."""
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.onboard.delivery import ACCURACY_TESTS_DIR
    from agent_flow.workflows.trtllm_modeling_v2.run.stages.deliver import DeliverStage

    repo = tmp_path / "repo"
    (repo / ACCURACY_TESTS_DIR).mkdir(parents=True)

    class Stage(DeliverStage):
        task_data = {
            "trtllm_repo_path": str(repo),
            "accuracy": {"reference_yaml": "refs/gsm8k.yaml", "anchor": "pending"},
        }
        _preflight = SimpleNamespace(family="newmodel")

        def _model_key(self, unit=None):
            return "org/new-model"

    core = g.Task(id="core", kind="assemble", goal="core", ref="core")
    spec = Stage()._delivery_task(g.Graph(tasks=[core]))
    assert "refs/gsm8k.yaml" not in spec["touches"]
    assert "write no reference entry" in spec["goal"] and "accuracy:reference" not in spec["goal"]
