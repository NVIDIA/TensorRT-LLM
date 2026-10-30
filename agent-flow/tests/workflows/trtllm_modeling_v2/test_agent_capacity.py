# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One agent-capacity model: a task takes a slot only while it needs an agent, by priority."""

from __future__ import annotations

import threading

from agent_flow.workflows.trtllm_modeling_v2.infra.execution.agent_gate import (
    Priority,
    current_priority,
)
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g

from .test_orchestrator import _orchestrator, _plan_once, _services, _task

_KEYS = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]


def test_a_task_waiting_on_its_soft_dependency_frees_its_slot_and_is_met_at_integration(
    tmp_path,
):
    """A task waiting at its integration frees its slot; the integration, not the checks, ends it.

    With two task slots, ``long`` and ``waiter`` hold both. ``waiter`` finishes its turn and
    waits for ``long`` at its integration: its slot goes to ``other`` meanwhile, and ``long``'s
    integration, not the end of its checks, lets ``waiter`` go on.
    """
    other_started = threading.Event()
    long_release = threading.Event()
    seen: dict[str, object] = {}
    long = _task("long", "assemble", _KEYS, touches=["models/long/"])
    long.est_minutes = 300
    waiter = _task("waiter", "feature", touches=["models/waiter/"])
    waiter.depends_on_soft = ["long"]
    other = _task("other", "feature", touches=["models/other/"])
    other.depends_on_soft = ["waiter"]  # it may start once waiter has

    def work(task):
        if task.id == "long":
            # Its commits land only once other has started in the slot waiter freed.
            seen["other started while waiter waited"] = other_started.wait(timeout=10)
            orchestrator.mark_integrated("long", task.attempts)
            # What follows the integration (its checks) runs long.
            assert long_release.wait(timeout=30)
        elif task.id == "waiter":
            seen["soft"] = orchestrator.soft_dependencies("waiter")
            long_release.set()
        elif task.id == "other":
            other_started.set()
        return "UNIT_READY"

    services = _services(_plan_once(long, waiter, other))
    services.work = work
    # Three agents: two slots for task turns, one for the planner.
    orchestrator = _orchestrator(tmp_path, services, max_parallel=3)
    assert orchestrator.run() == "delivered"
    assert seen["other started while waiter waited"] is True
    assert seen["soft"] == {"long": "integrated"}


def test_a_pending_dependency_a_running_task_waits_for_starts_first(tmp_path):
    """Priority inheritance: what a running task waits for is not held back by an exclusive task."""
    services = _services(_plan_once())
    orchestrator = _orchestrator(tmp_path, services)
    waiter = _task("waiter", "feature", touches=["models/a/"])
    waiter.depends_on_soft = ["dep"]
    waiter.status = "running"
    dep = _task("dep", "feature", touches=["models/b/"])
    rebuild = _task("rebuild", "kernel", touches=["cpp/x.cu"])
    rebuild.exclusive, rebuild.est_minutes = True, 600
    orchestrator.graph = g.Graph(tasks=[waiter, dep, rebuild])
    orchestrator._agentless.add("waiter")  # its thread is in soft_dependencies
    assert orchestrator._waited_for() == {"dep"}
    chosen = g.next_to_start(orchestrator.graph, 8, resumed=orchestrator._waited_for())
    assert [t.id for t in chosen] == ["dep"]
    # Without it the exclusive task, ranked first, would hold every task after it back.
    assert [t.id for t in g.next_to_start(orchestrator.graph, 8)] == []


def test_each_turn_takes_its_slot_at_its_tasks_priority(tmp_path):
    seen: dict[str, Priority] = {}
    core = _task("core", "assemble", _KEYS)
    core.est_minutes = 300
    side = _task("side", "investigate", priority="nice_to_have")
    tune = _task("tune", "tune", touches=[])

    def work(task):
        seen[task.id] = current_priority()
        return "TUNED" if task.kind == "tune" else "UNIT_READY"

    services = _services(_plan_once(core, side, tune))
    services.work = work
    planner = services.plan

    def plan(turn):
        seen.setdefault("planner", current_priority())
        return planner(turn)

    services.plan = plan
    assert _orchestrator(tmp_path, services, required_kinds=("tune",)).run() == "delivered"
    assert seen["planner"] == Priority.PLANNER
    assert seen["core"] == Priority.CRITICAL
    assert seen["side"] == Priority.SIDE
    assert seen["tune"] == Priority.BACKGROUND


def test_a_campaigns_turns_get_only_the_slots_the_foreground_leaves_idle(tmp_path):
    services = _services(_plan_once())
    orchestrator = _orchestrator(tmp_path, services, max_parallel=8)
    orchestrator._foreground_demand = 5
    assert orchestrator.idle_agent_slots() == 2  # seven task slots, five wanted
    orchestrator._foreground_demand = 9
    assert orchestrator.idle_agent_slots() == 0
