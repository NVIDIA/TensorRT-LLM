# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A failed task retried under its own id: the planner asks, the loop runs it again, up to a bound."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.workspace.integration import TaskNotStarted

from .test_orchestrator import _KEYS, PlannerAnswer, _orchestrator, _services, _task


def _replanning(*reasons):
    """A planner that writes the same one-task graph on every turn: the retry is the status."""
    turns = []

    def plan(turn):
        turns.append(turn)
        reason = reasons[min(len(turns), len(reasons)) - 1]
        return PlannerAnswer(
            "GRAPH_UPDATED", reason, g.Graph(tasks=[_task("core", "assemble", _KEYS)])
        )

    plan.turns = turns  # type: ignore[attr-defined]
    return plan


def test_a_failed_task_runs_again_under_its_own_id(tmp_path):
    runs = []

    def work(task):
        runs.append((task.id, task.attempts))
        return "BLOCKED" if len(runs) == 1 else "UNIT_READY"

    services = _services(_replanning("plan", "core failed on a fixed flow defect: retry"))
    services.work = work
    assert _orchestrator(tmp_path, services).run() == "delivered"
    assert runs == [("core", 1), ("core", 2)]
    saved = g.Graph.load(tmp_path / g.GRAPH_FILENAME).by_id()["core"]
    assert (saved.status, saved.attempts) == ("done", 2)


def test_retries_stop_at_the_bound_with_the_reason(tmp_path):
    runs = []

    def work(task):
        runs.append(task.attempts)
        return "BLOCKED"

    plan = _replanning("again")
    services = _services(plan)
    services.work = work
    assert _orchestrator(tmp_path, services, revision_cap=10).run() == "planner-refused"
    assert runs == list(range(1, g.MAX_ATTEMPTS + 1))
    assert any(f"failed on attempt {g.MAX_ATTEMPTS} of" in r for r in plan.turns[-1].refusals)


def test_an_attempt_whose_worktree_was_never_prepared_does_not_spend_a_worker_run(tmp_path):
    runs = []

    def work(task):
        runs.append(task.attempts)
        if task.attempts <= 3:
            raise TaskNotStarted("worktree checkout: Disk quota exceeded")
        return "BLOCKED"

    services = _services(_replanning("storage repaired; retry the same task"))
    services.work = work
    runner = _orchestrator(tmp_path, services, revision_cap=10)
    assert runner.run() == "planner-refused"
    # Each launch that never started a worker gets its one infrastructure retry, keeps
    # its number, and leaves the task its MAX_ATTEMPTS worker runs.
    assert runs == [1, 1, 2, 2, 3, 3, 4, 5, 6]
    assert runner.not_started_attempts == {"core": [1, 2, 3]}
    assert _orchestrator(tmp_path, services).not_started_attempts == {"core": [1, 2, 3]}


def test_a_worktree_that_cannot_be_prepared_fails_before_the_worker_runs(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage

    from .test_workflow import _workflow

    stage = ExecutionStage(_workflow(tmp_path))
    invoked = []

    def unavailable(*args, **kwargs):
        raise OSError(122, "Disk quota exceeded")

    stage._integration = SimpleNamespace(open=unavailable)
    stage._work_in_tree = lambda *args: invoked.append(True)
    with pytest.raises(TaskNotStarted, match="Disk quota exceeded"):
        stage._work_task_turn(_task("core", "assemble", _KEYS))
    assert invoked == []
