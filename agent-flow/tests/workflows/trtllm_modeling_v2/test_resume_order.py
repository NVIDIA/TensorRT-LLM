# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task running when the process stopped starts again before anything that conflicts with it.

Its worker may still be at it, and its answer may already be written: a task
ranked ahead of it on the critical path that shares one of its files would
start in its place and hold it back for as long as that task runs.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)

_CHECKS = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]
_CONTRACT = c.Contract(
    checks=(
        c.Check("core", "boot", "tp1", c.TIER_INNER),
        c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER),
        c.Check("core", "envelope", "tp1", c.TIER_DELIVERY),
    )
)


def _t(tid, touches=(), minutes=60, gpus=1, **kw):
    return g.Task(
        id=tid,
        kind=kw.pop("kind", "feature"),
        goal=f"do {tid}",
        touches=list(touches),
        gpus=gpus,
        est_minutes=minutes,
        **kw,
    )


def _ids(tasks):
    return [t.id for t in tasks]


def _conflicting_pair():
    """The guard tests and a longer task that edits the same test file."""
    return g.Graph(
        tasks=[
            _t("thop-guard-tests", ["tests/test_thop.py"], minutes=30),
            _t("k46-unpadded", ["ops/thop.py", "tests/test_thop.py"], minutes=200),
        ]
    )


def test_the_resumed_task_is_chosen_over_a_longer_one_it_conflicts_with():
    graph = _conflicting_pair()
    assert _ids(g.next_to_start(graph, free_gpus=8)) == ["k46-unpadded"]
    assert _ids(g.next_to_start(graph, free_gpus=8, resumed={"thop-guard-tests"})) == [
        "thop-guard-tests"
    ]


def test_a_resumed_task_waiting_for_devices_keeps_its_files():
    graph = _conflicting_pair()
    graph.tasks[0].gpus = 4
    graph.tasks.append(_t("other", ["ops/other.py"], minutes=10))
    started = g.next_to_start(graph, free_gpus=2, resumed={"thop-guard-tests"})
    # The conflicting task does not take its place; unrelated work still starts.
    assert _ids(started) == ["other"]
    assert _ids(g.next_to_start(graph, free_gpus=6, resumed={"thop-guard-tests"})) == [
        "thop-guard-tests",
        "other",
    ]


def test_resuming_moves_only_the_resumed_task():
    graph = g.Graph(
        tasks=[
            _t("a", minutes=10),
            _t("b", minutes=300),
            _t("c", minutes=50, priority="nice_to_have"),
            _t("d", minutes=100),
        ]
    )
    assert _ids(g.next_to_start(graph, free_gpus=8)) == ["b", "d", "a", "c"]
    assert _ids(g.next_to_start(graph, free_gpus=8, resumed={"c"})) == ["c", "b", "d", "a"]
    # An id that is not ready, or not in the graph, changes nothing.
    graph.tasks[0].status = "done"
    assert _ids(g.next_to_start(graph, free_gpus=8, resumed={"a", "ghost"})) == ["b", "d", "c"]


def test_a_resumed_task_still_waits_for_its_gate():
    graph = _conflicting_pair()
    graph.tasks[0].kind = "tune"
    started = g.next_to_start(
        graph, free_gpus=8, gated_kinds={"tune"}, resumed={"thop-guard-tests"}
    )
    assert _ids(started) == ["k46-unpadded"]


def _services(ran):
    def plan(turn):
        return PlannerAnswer("DONE", "all green")

    def check(chk):
        return SimpleNamespace(name=chk.gate, member=chk.member, passed=True, detail="",
                               tree_state="")  # fmt: skip

    return Services(
        plan=plan,
        work=lambda task: ran.append(task.id) or "UNIT_READY",
        check=check,
        free_gpus=lambda: 4,
    )


def test_a_restarted_orchestrator_starts_the_interrupted_task_first(tmp_path):
    graph = _conflicting_pair()
    graph.tasks[0].status = "running"
    graph.tasks[1].satisfies = list(_CHECKS)
    graph.save(tmp_path / g.GRAPH_FILENAME)
    ran: list[str] = []
    orchestrator = GraphOrchestrator(
        _services(ran), _CONTRACT, c.ContractStatus.load(tmp_path), tmp_path, poll_s=0
    )
    assert orchestrator.run() == "delivered"
    assert ran == ["thop-guard-tests", "k46-unpadded"]


def test_a_fresh_orchestrator_keeps_the_critical_path_order(tmp_path):
    graph = _conflicting_pair()
    graph.tasks[1].satisfies = list(_CHECKS)
    graph.save(tmp_path / g.GRAPH_FILENAME)
    ran: list[str] = []
    orchestrator = GraphOrchestrator(
        _services(ran), _CONTRACT, c.ContractStatus.load(tmp_path), tmp_path, poll_s=0
    )
    assert orchestrator.run() == "delivered"
    assert ran == ["k46-unpadded", "thop-guard-tests"]
