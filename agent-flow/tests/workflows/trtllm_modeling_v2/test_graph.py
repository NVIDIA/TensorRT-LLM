# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The task graph: what makes a planner's graph admissible, and what starts next."""

from __future__ import annotations

from agent_flow.workflows.trtllm_modeling_v2.run import graph as g


def _t(tid, kind="kernel", deps=(), touches=(), gpus=1, minutes=60, **kw):
    return g.Task(
        id=tid,
        kind=kind,
        goal=f"do {tid}",
        depends_on=list(deps),
        touches=list(touches),
        gpus=gpus,
        est_minutes=minutes,
        **kw,
    )


def _graph(*tasks):
    return g.Graph(tasks=list(tasks))


def test_a_sound_graph_is_admissible():
    graph = _graph(
        _t("k1", touches=["catalog/a.py"], satisfies=["core:boot@tp1"]),
        _t("core", "assemble", deps=["k1"], touches=["models/x/"], satisfies=["core:accuracy@tp1"]),
    )
    assert (
        g.validate(
            graph,
            required_checks=["core:boot@tp1", "core:accuracy@tp1"],
            allowed_paths=["catalog/", "models/"],
            max_gpus=8,
        )
        == []
    )


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


def test_tasks_on_equal_chains_start_in_the_order_their_ids_number_them():
    """The kernel rows the orchestrator adds start before the planner wires them, on equal chains.

    The caller starts only as many as it has task slots, so the order is what
    starts: row 3 before row 10, as the survey numbered its queue, not as the
    ids sort as strings.
    """
    rows = (10, 2, 1, 3, 12, 11)
    graph = _graph(*(_t(f"k{n}-row", minutes=60) for n in rows), _t("eng", minutes=60))
    started = [t.id for t in g.next_to_start(graph, free_gpus=8)]
    assert started == ["eng", "k1-row", "k2-row", "k3-row", "k10-row", "k11-row", "k12-row"]


def test_nice_to_have_waits_while_required_work_waits_for_devices():
    graph = _graph(_t("big", gpus=4), _t("extra", priority="nice_to_have", gpus=1))
    assert g.next_to_start(graph, free_gpus=2) == []
    assert [t.id for t in g.next_to_start(graph, free_gpus=5)] == ["big", "extra"]


def test_a_task_starts_on_what_it_holds_not_what_it_declares():
    """A tune task whose jobs all run on the perf sessions holds none of the run's devices."""
    graph = _graph(_t("tune", kind="tune", gpus=4), _t("k1", gpus=1))
    assert [t.id for t in g.next_to_start(graph, free_gpus=1)] == ["k1"]
    held = lambda task: 0 if task.kind == "tune" else task.gpus  # noqa: E731
    assert sorted(t.id for t in g.next_to_start(graph, free_gpus=1, held=held)) == ["k1", "tune"]


def test_tasks_that_share_a_directory_start_together():
    """Each task has its own worktree and is rebased at integration."""
    graph = _graph(
        _t("a", touches=["models/x/modeling.py", "models/x/ops/"]),
        _t("b", touches=["models/x/"]),
        _t("c", touches=["models/x/ops/norm.py", "catalog/ssm/"]),
        _t("d", touches=["models/x", "catalog/ssm/"]),
    )
    assert [t.id for t in g.next_to_start(graph, free_gpus=8)] == ["a", "b", "c", "d"]
    graph.tasks[0].status = "running"
    assert [t.id for t in g.next_to_start(graph, free_gpus=8)] == ["b", "c", "d"]


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


def test_a_running_campaign_holds_no_exclusive_task_back():
    graph = _graph(
        _t("tune", kind="tune", touches=[], status="running"),
        _t("rebuild", touches=["cpp/x.cu"], minutes=90, exclusive=True),
        _t("after", touches=["catalog/a.py"]),
    )
    assert [t.id for t in g.next_to_start(graph, free_gpus=8)] == ["rebuild"]


def test_a_campaign_ranks_after_the_feature_work_whatever_its_estimate():
    graph = _graph(
        _t("tune", kind="tune", touches=[], minutes=1200),
        _t("feature", touches=["models/x/"], minutes=60),
    )
    assert [t.id for t in g.next_to_start(graph, free_gpus=1)] == ["feature"]


def test_the_run_drains_to_a_waiting_exclusive_task():
    """Nothing ranked after it starts meanwhile, or it would never find the run empty."""
    graph = _graph(
        _t("first", minutes=300),
        _t("running", status="running"),
        _t("rebuild", minutes=200, exclusive=True),
        _t("after", minutes=10),
    )
    assert [t.id for t in g.next_to_start(graph, free_gpus=8)] == ["first"]
    graph.tasks[0].status = "running"
    assert g.next_to_start(graph, free_gpus=8) == []
    graph.tasks[0].status = graph.tasks[1].status = "done"
    assert [t.id for t in g.next_to_start(graph, free_gpus=8)] == ["rebuild"]


def test_exclusive_is_read_and_written_with_the_graph(tmp_path):
    path = tmp_path / g.GRAPH_FILENAME
    path.write_text(
        "tasks:\n"
        "  - {id: a, kind: kernel, goal: x}\n"
        "  - {id: b, kind: kernel, goal: y, exclusive: true}\n"
        "  - {id: c, kind: kernel, goal: z, exclusive: 'false'}\n"
    )
    graph = g.Graph.load(path)
    assert [t.exclusive for t in graph.tasks] == [False, True, False]
    graph.save(path)
    assert [t.exclusive for t in g.Graph.load(path).tasks] == [False, True, False]
    assert graph.tasks[1].as_dict()["exclusive"] is True


def test_a_task_waits_for_its_dependencies():
    graph = _graph(_t("a"), _t("b", deps=["a"]))
    assert [t.id for t in g.ready(graph)] == ["a"]
    graph.tasks[0].status = "done"
    assert [t.id for t in g.ready(graph)] == ["b"]


def test_a_failed_dependency_whose_receipt_landed_its_latest_attempt_is_met():
    graph = _graph(_t("a", status="failed", attempts=2), _t("b", deps=["a"]))
    assert g.ready(graph) == []
    # The landing of an earlier attempt says nothing of this one.
    assert g.ready(graph, {"a": 1}) == []
    assert [t.id for t in g.ready(graph, {"a": 2})] == ["b"]
    assert [t.id for t in g.next_to_start(graph, free_gpus=8, receipts_landed={"a": 2})] == ["b"]


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


def test_the_planner_is_stalled_after_the_cap():
    assert not g.stalled(2, cap=3)
    assert g.stalled(3, cap=3)


def test_a_gated_kind_never_starts_while_others_do():
    graph = _graph(_t("tune", kind="tune"), _t("k1", touches=["catalog/a.py"]))
    assert [t.id for t in g.next_to_start(graph, 4, gated_kinds={"tune"})] == ["k1"]
    assert {t.id for t in g.next_to_start(graph, 4)} == {"tune", "k1"}


def test_a_required_kind_must_have_a_live_task():
    graph = _graph(_t("k1"), _t("tune", kind="tune", status="abandoned"))
    errors = g.validate(graph, required_kinds=("tune",))
    assert errors == ["the task asks for `tune` work and the graph has no `tune` task"]
    assert g.validate(_graph(_t("tune", kind="tune")), required_kinds=("tune",)) == []


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


def test_only_pending_retries_a_failed_task():
    current = _graph(_t("a", status="failed"), _t("b", status="failed"), _t("c", status="failed"))
    proposed = _graph(_t("a", status="running"), _t("b", status="done"), _t("c", status="failed"))
    adopted, errors = g.apply_revision(current, proposed, "r")
    assert errors == []
    assert [(t.status, t.attempts) for t in adopted.tasks] == [("failed", 1)] * 3


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


def test_a_path_is_inside_the_roots_that_hold_it():
    roots = ["tensorrt_llm/_torch/", "tests/unittest"]
    assert g.inside("tensorrt_llm/_torch/a.py", roots)
    assert g.inside("tests/unittest", roots)
    assert not g.inside("tensorrt_llm/_torch_other/a.py", roots)
    assert g.inside("anything/at/all.py", [])


def test_a_task_waiting_ready_long_enough_starts_before_a_longer_path():
    """A leaf with a short path is not passed over for ever by tasks on longer paths."""
    leaf = g.Task(id="k3b", kind="kernel", goal="g", est_minutes=10, gpus=0, touches=["a.py"])
    head = g.Task(id="core", kind="feature", goal="g", est_minutes=120, gpus=0, touches=["b.py"])
    tail = g.Task(id="mtp", kind="feature", goal="g", est_minutes=120, gpus=0, touches=["c.py"])
    tail.depends_on = ["core"]
    graph = g.Graph(tasks=[leaf, head, tail])
    assert [t.id for t in g.next_to_start(graph, 8)][:1] == ["core"]
    assert [t.id for t in g.next_to_start(graph, 8, waited={"k3b": 400.0})][:1] == ["k3b"]
