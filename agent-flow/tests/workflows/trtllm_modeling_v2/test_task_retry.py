# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A failed task retried under its own id: the planner asks, the loop runs it again, up to a bound."""

from __future__ import annotations

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
