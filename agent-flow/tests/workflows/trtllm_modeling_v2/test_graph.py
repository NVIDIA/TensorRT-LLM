# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The task graph: what makes a planner's graph admissible, and what starts next."""

from __future__ import annotations

from agent_flow.workflows.trtllm_modeling_v2.run import graph as g


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


def test_a_goal_the_planner_writes_stays_short():
    long = _t("a", goal="x" * (g.MAX_GOAL_CHARS + 1))
    current = _graph(long)
    # Kept as it was: the planner did not write it this turn.
    assert g.apply_revision(current, _graph(_t("a", goal=long.goal)), "keep")[1] == []
    refusals = g.apply_revision(_graph(), _graph(_t("a", goal=long.goal)), "add")[1]
    assert refusals and "characters" in refusals[0]


def test_every_inadmissible_property_is_named():
    graph = _graph(
        _t("a", "wizard", deps=["b", "ghost"], touches=["/etc/passwd"], gpus=16),
        _t("b", deps=["a"]),
        _t("c"),
        _t("c"),
    )
    errors = "\n".join(
        g.validate(graph, required_checks=["core:boot@tp1"], allowed_paths=["models/"], max_gpus=8)
    )
    assert "'c' is used more than once" in errors
    assert "kind 'wizard'" in errors
    assert "unknown task 'ghost'" in errors
    assert "outside ['models/']" in errors
    assert "asks for 16 GPUs" in errors
    assert "dependency cycle" in errors
    assert "'core:boot@tp1' is in no task" in errors


def test_required_work_on_the_longest_chain_starts_first():
    graph = _graph(
        _t("short", minutes=10),
        _t("long", minutes=10),
        _t("after-long", deps=["long"], minutes=300),
        _t("extra", priority="nice_to_have"),
    )
    started = [t.id for t in g.next_to_start(graph, free_gpus=1)]
    assert started == ["long"]


def test_nice_to_have_waits_while_required_work_waits_for_devices():
    graph = _graph(_t("big", gpus=4), _t("extra", priority="nice_to_have", gpus=1))
    assert g.next_to_start(graph, free_gpus=2) == []
    assert [t.id for t in g.next_to_start(graph, free_gpus=5)] == ["big", "extra"]


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


def test_an_exclusive_task_runs_alone():
    graph = _graph(
        _t("rebuild", touches=["cpp/x.cu"], minutes=90, exclusive=True),
        _t("a", touches=["catalog/a.py"]),
        _t("tune", kind="tune", touches=[]),
    )
    # The campaign works on its own branches from the commit it pinned: it does not wait.
    assert [t.id for t in g.next_to_start(graph, free_gpus=8)] == ["rebuild", "tune"]
    graph.tasks[0].status = "running"
    assert [t.id for t in g.next_to_start(graph, free_gpus=8)] == ["tune"]
    assert not g.touches_conflict(graph.tasks[1], graph.tasks[2])
    assert g.touches_conflict(graph.tasks[1], graph.tasks[0])
    assert not g.touches_conflict(graph.tasks[0], graph.tasks[2])


def test_a_task_waits_for_its_dependencies():
    graph = _graph(_t("a"), _t("b", deps=["a"]))
    assert [t.id for t in g.ready(graph)] == ["a"]
    graph.tasks[0].status = "done"
    assert [t.id for t in g.ready(graph)] == ["b"]


def test_a_revision_keeps_what_the_orchestrator_owns(tmp_path):
    current = _graph(_t("a", status="done"), _t("b", status="running"), _t("c"))
    proposed = _graph(_t("a"), _t("d", deps=["a"]))
    adopted, errors = g.apply_revision(current, proposed, "split c into d")
    assert errors == []
    by_id = adopted.by_id()
    assert by_id["a"].status == "done"
    assert by_id["b"].status == "running"  # dropped by the planner, still running
    assert "c" not in by_id
    assert adopted.revisions[-1].reason == "split c into d"
    path = adopted.save(tmp_path / g.GRAPH_FILENAME)
    assert g.Graph.load(path).by_id()["b"].status == "running"


def test_a_revision_without_a_reason_or_with_errors_is_refused():
    current = _graph(_t("a"))
    assert g.apply_revision(current, _graph(_t("b")), " ")[1]
    adopted, errors = g.apply_revision(current, _graph(_t("b", deps=["nope"])), "why")
    assert adopted is current and errors


def test_a_gated_kind_never_starts_while_others_do():
    graph = _graph(_t("tune", kind="tune"), _t("k1", touches=["catalog/a.py"]))
    assert [t.id for t in g.next_to_start(graph, 4, gated_kinds={"tune"})] == ["k1"]
    assert {t.id for t in g.next_to_start(graph, 4)} == {"tune", "k1"}


def test_a_planner_cannot_mark_a_task_running_or_done():
    """A task marked running that no worker runs never finishes."""
    current = g.Graph(
        tasks=[g.Task(id="a", kind="kernel", goal="x"), g.Task(id="b", kind="kernel", goal="y")]
    )
    current.tasks[1].status = "done"
    proposed = g.Graph(tasks=[g.Task(id="a", kind="kernel", goal="x"), g.Task(id="b", kind="kernel", goal="y"),
                              g.Task(id="c", kind="kernel", goal="z")])  # fmt: skip
    proposed.tasks[0].status = "running"  # copied from a stale graph.yaml
    proposed.tasks[1].status = "pending"
    proposed.tasks[2].status = "done"
    adopted, errors = g.apply_revision(current, proposed, "r")
    assert errors == []
    assert [(t.id, t.status) for t in adopted.tasks] == [
        ("a", "pending"),
        ("b", "done"),
        ("c", "pending"),
    ]


def test_a_planner_retries_a_failed_task_in_place(tmp_path):
    """A failed task written back as pending runs again under its id, as its next attempt."""
    current = _graph(_t("a", status="failed"), _t("b", status="done"), _t("c", attempts=2))
    # The attempt count is the orchestrator's, whatever the planner writes.
    proposed = _graph(_t("a"), _t("b"), _t("c"), _t("d", attempts=7))
    adopted, errors = g.apply_revision(current, proposed, "a failed on a bench defect, now fixed")
    assert errors == []
    assert [(t.id, t.status, t.attempts) for t in adopted.tasks] == [
        ("a", "pending", 2),
        ("b", "done", 1),
        ("c", "pending", 2),
        ("d", "pending", 1),
    ]
    path = adopted.save(tmp_path / g.GRAPH_FILENAME)
    assert g.Graph.load(path).by_id()["a"].attempts == 2


def test_a_task_that_failed_every_attempt_is_not_retried_again():
    current = _graph(_t("a", status="failed", attempts=g.MAX_ATTEMPTS), _t("b"))
    adopted, errors = g.apply_revision(current, _graph(_t("a"), _t("b")), "once more")
    assert adopted is current
    assert errors == [
        f"task 'a' failed on attempt {g.MAX_ATTEMPTS} of {g.MAX_ATTEMPTS} and is not retried "
        "again: find the cause first (an `investigate` task), reshape the work, or drop the task"
    ]
    # Leaving it failed, or dropping it, is admissible.
    assert g.apply_revision(current, _graph(_t("a", status="failed"), _t("b")), "r")[1] == []
    assert g.apply_revision(current, _graph(_t("b")), "drop a")[1] == []
