# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A failed task retried under its own id: the planner asks, the loop runs it again, up to a bound."""

from __future__ import annotations

from agent_flow.workflows.trtllm_modeling_v2.run import graph as g

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
