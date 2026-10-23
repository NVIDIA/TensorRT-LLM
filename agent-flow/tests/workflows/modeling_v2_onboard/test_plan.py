# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``plan.md`` parsing and the workflow's plan rules."""

from __future__ import annotations

import pytest

from agent_flow.orchestration import NodeState
from agent_flow.workflows.modeling_v2_onboard.plan import PlanError, parse_plan
from agent_flow.workflows.modeling_v2_onboard.prompts import PLAN_EXAMPLE


def _plan(nodes_yaml: str, sections: list[str]) -> str:
    body = "\n".join(f"### {s}\nDo it.\n" for s in sections)
    return f"# Plan\n\n## Tasks\n\n{body}\n## Execution Graph\n\n```yaml\nnodes:\n{nodes_yaml}```\n"


_VALID_NODES = """\
  - id: a
    type: new_op
  - id: b
    type: target
    tags: [module_accuracy]
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [a, b]
"""


def _problems(text: str, **kwargs) -> list[str]:
    with pytest.raises(PlanError) as info:
        parse_plan(text, **kwargs)
    return info.value.problems


def test_planner_example_is_a_valid_plan():
    spec = parse_plan(PLAN_EXAMPLE)
    assert spec.sink_id() == "land"
    assert spec.tags["join-core"] == ("merge", "target")


def test_tags_put_the_primary_type_first():
    spec = parse_plan(_plan(_VALID_NODES, ["a", "b", "land"]))
    assert spec.tags == {
        "a": ("new_op",),
        "b": ("target", "module_accuracy"),
        "land": ("merge",),
    }
    assert spec.graph.by_id["land"].isolation == "shared"


def test_numbered_heading_is_accepted():
    text = _plan(_VALID_NODES, ["a", "b", "land"]).replace(
        "## Execution Graph", "## 4. Execution Graph"
    )
    assert parse_plan(text).sink_id() == "land"


def test_missing_graph_is_rejected():
    assert "Execution Graph" in _problems("# Plan\n\nno graph\n")[0]


def test_malformed_graph_is_rejected():
    text = _plan("  - id: a\n    type: new_op\n    depends_on: [ghost]\n", ["a"])
    assert "ghost" in _problems(text)[0]


def test_every_rule_violation_is_reported_together():
    nodes = """\
  - id: a
    type: new_op
    tags: [bogus]
  - id: m
    type: merge
  - id: j
    type: target
    depends_on: [a, m]
  - id: x
    type: e2e_accuracy
    isolation: shared
"""
    problems = "\n".join(_problems(_plan(nodes, ["a", "j", "x"])))
    assert "unknown tag(s) ['bogus']" in problems
    assert "task 'm': tag `merge` and `kind: merge` must go together" in problems
    assert "task 'j' depends on 2 tasks" in problems
    assert "no `### m` section" in problems
    assert "exactly one final task" in problems


def test_final_task_must_be_shared_and_tagged_merge():
    nodes = "  - id: a\n    type: new_op\n  - id: b\n    type: target\n    depends_on: [a]\n"
    assert any("isolation: shared" in p for p in _problems(_plan(nodes, ["a", "b"])))

    nodes = (
        "  - id: a\n    type: new_op\n"
        "  - id: b\n    type: target\n    isolation: shared\n    depends_on: [a]\n"
    )
    assert any("must be tagged `merge`" in p for p in _problems(_plan(nodes, ["a", "b"])))


def test_shared_task_that_is_not_final_is_rejected():
    nodes = (
        "  - id: a\n    type: merge\n    kind: merge\n    isolation: shared\n"
        "  - id: b\n    type: merge\n    kind: merge\n    depends_on: [a]\n"
    )
    assert any("isolation: shared" in p for p in _problems(_plan(nodes, ["a", "b"])))


def test_unsafe_ids_and_nested_children_are_rejected():
    nodes = (
        "  - id: a/b\n    type: new_op\n"
        "    children:\n      - id: c\n        type: new_op\n"
        "  - id: land\n    type: merge\n    kind: merge\n    isolation: shared\n"
        "    depends_on: [a/b]\n"
    )
    problems = "\n".join(_problems(_plan(nodes, ["a/b", "land"])))
    assert "ids may use" in problems
    assert "nested `children`" in problems


def test_a_done_task_cannot_be_changed_or_removed():
    previous = parse_plan(_plan(_VALID_NODES, ["a", "b", "land"]))
    states = {"a": NodeState.DONE, "b": NodeState.FAILED, "land": NodeState.PENDING}

    changed = _VALID_NODES.replace("type: new_op", "type: new_op\n    tags: [target]")
    problems = _problems(_plan(changed, ["a", "b", "land"]), previous=previous, states=states)
    assert problems == [
        "task 'a' is DONE; its graph entry and section cannot change; add a new task "
        "for follow-up work instead"
    ]
    rewritten = _plan(_VALID_NODES, ["a", "b", "land"]).replace("### a\nDo it.", "### a\nRedo.")
    assert _problems(rewritten, previous=previous, states=states) == problems

    without_a = """\
  - id: b
    type: target
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [b]
"""
    problems = _problems(_plan(without_a, ["b", "land"]), previous=previous, states=states)
    assert problems == ["task 'a' is DONE and cannot be removed"]


def test_a_running_task_cannot_be_changed_or_removed():
    previous = parse_plan(_plan(_VALID_NODES, ["a", "b", "land"]))
    states = {"a": NodeState.RUNNING, "b": NodeState.PENDING, "land": NodeState.PENDING}
    rewritten = _plan(_VALID_NODES, ["a", "b", "land"]).replace("### a\nDo it.", "### a\nRedo.")
    assert _problems(rewritten, previous=previous, states=states) == [
        "task 'a' is RUNNING; its graph entry and section cannot change; list it under "
        "`cancel` to change it"
    ]
    # The same edit is fine once a is no longer running (e.g. cancelled).
    states["a"] = NodeState.PENDING
    assert parse_plan(rewritten, previous=previous, states=states).entry("a")[-1] == "Redo."


def test_a_task_id_that_prefixes_another_has_its_own_section():
    nodes = """\
  - id: gsm8k
    type: e2e_accuracy
  - id: gsm8k-final
    type: e2e_accuracy
    depends_on: [gsm8k]
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [gsm8k-final]
"""
    problems = _problems(_plan(nodes, ["gsm8k-final", "land"]))
    assert problems == ["task 'gsm8k' has no `### gsm8k` section describing it"]


def test_a_revision_may_rewrite_unfinished_tasks_and_add_new_ones():
    previous = parse_plan(_plan(_VALID_NODES, ["a", "b", "land"]))
    states = {"a": NodeState.DONE, "b": NodeState.FAILED, "land": NodeState.PENDING}
    revised = """\
  - id: a
    type: new_op
  - id: op2
    type: new_op
  - id: b
    type: target
    depends_on: [op2]
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [a, b]
"""
    spec = parse_plan(_plan(revised, ["a", "op2", "b", "land"]), previous=previous, states=states)
    assert spec.graph.by_id["b"].depends_on == ("op2",)


def test_priorities_default_by_role_and_can_be_set():
    spec = parse_plan(PLAN_EXAMPLE)
    assert spec.priorities["op-gated-delta"] == "accuracy"
    assert spec.priorities["feat-fp8-kv"] == "feature"
    assert spec.priorities["perf-moe-gemm"] == "perf"
    # Defaults: gsm8k is the first gate on a target; gsm8k-final has it upstream.
    bare = PLAN_EXAMPLE.replace("    priority: accuracy\n", "").replace(
        "    priority: feature\n", ""
    )
    defaults = parse_plan(bare).priorities
    assert defaults["op-gated-delta"] == "accuracy" and defaults["gsm8k"] == "accuracy"
    assert defaults["feat-fp8-kv"] == "feature" and defaults["gsm8k-final"] == "feature"
    assert defaults["perf-moe-gemm"] == "perf"


def test_a_gate_on_later_code_is_not_a_first_gate_without_the_first_upstream():
    # The tp4 gate checks code built on top of what `gate` checks, but joins do
    # not wait for checks, so `gate` is not upstream of it.
    nodes = """\
  - id: target
    type: target
  - id: gate
    type: e2e_accuracy
    depends_on: [target]
  - id: target-tp4
    type: target
    depends_on: [target]
  - id: gate-tp4
    type: e2e_accuracy
    depends_on: [target-tp4]
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [gate, gate-tp4]
"""
    ids = ["target", "gate", "target-tp4", "gate-tp4", "land"]
    priorities = parse_plan(_plan(nodes, ids)).priorities
    assert priorities["target"] == priorities["gate"] == "accuracy"
    assert priorities["target-tp4"] == priorities["gate-tp4"] == "feature"


def test_an_unknown_priority_is_rejected():
    text = _plan(
        _VALID_NODES.replace("type: new_op", "type: new_op\n    priority: urgent"),
        ["a", "b", "land"],
    )
    assert _problems(text) == [
        "task 'a': `priority` must be one of accuracy, feature, perf, got 'urgent'"
    ]


def test_changing_a_running_task_priority_is_allowed():
    previous = parse_plan(_plan(_VALID_NODES, ["a", "b", "land"]))
    states = {"a": NodeState.RUNNING, "b": NodeState.DONE, "land": NodeState.PENDING}
    text = _plan(
        _VALID_NODES.replace("type: new_op", "type: new_op\n    priority: perf"), ["a", "b", "land"]
    )
    assert parse_plan(text, previous=previous, states=states).priorities["a"] == "perf"


def test_a_perf_task_on_the_way_to_the_first_gate_is_accuracy():
    nodes = """\
  - id: op
    type: new_op
  - id: fuse
    type: perf
    depends_on: [op]
  - id: protocol
    type: e2e_accuracy
  - id: tgt
    type: merge
    kind: merge
    tags: [target]
    depends_on: [op, fuse]
  - id: gate
    type: merge
    kind: merge
    tags: [e2e_accuracy]
    depends_on: [tgt, protocol]
  - id: tune
    type: perf
    depends_on: [tgt]
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [gate, tune]
"""
    spec = parse_plan(_plan(nodes, ["op", "fuse", "protocol", "tgt", "gate", "tune", "land"]))
    assert spec.priorities == {
        "op": "accuracy",
        "fuse": "accuracy",
        "protocol": "accuracy",
        "tgt": "accuracy",
        "gate": "accuracy",
        "tune": "perf",
        "land": "feature",
    }


_SIDE_NODES = """\
  - id: a
    type: new_op
  - id: s
    type: perf
    depends_on: [a]
    side: true
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [a]
"""


def test_a_side_task_does_not_lead_to_the_final_task():
    spec = parse_plan(_plan(_SIDE_NODES, ["a", "s", "land"]))
    assert spec.side == {"s"}
    assert spec.sink_id() == "land"
    assert spec.priorities["s"] == "perf"
    # Without `side`, s would be a second final task.
    unmarked = _SIDE_NODES.replace("    side: true\n", "")
    assert any(
        "exactly one final task" in p for p in _problems(_plan(unmarked, ["a", "s", "land"]))
    )


def test_side_is_only_for_perf_work_off_the_main_line():
    nodes = """\
  - id: a
    type: new_op
    side: true
  - id: s
    type: perf
    side: yes please
  - id: t
    type: perf
    side: true
    priority: accuracy
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [a, s, t]
"""
    problems = _problems(_plan(nodes, ["a", "s", "t", "land"]))
    assert problems == [
        "task 'a': only a `perf` task that is not a `merge` task can be `side`",
        "task 's': `side` must be true or false, got 'yes please'",
        "task 'land' depends on side task(s) ['t']; a side branch is never merged into "
        "the main line — drop `side` from a task to merge it",
        "task 't' is `side`, so its `priority` is perf",
    ]


def test_side_can_change_on_a_done_or_running_task():
    previous = parse_plan(_plan(_SIDE_NODES, ["a", "s", "land"]))
    unmarked = """\
  - id: a
    type: new_op
  - id: s
    type: perf
    depends_on: [a]
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [a, s]
"""
    for state in (NodeState.DONE, NodeState.RUNNING):
        states = {"a": NodeState.DONE, "s": state, "land": NodeState.PENDING}
        spec = parse_plan(_plan(unmarked, ["a", "s", "land"]), previous=previous, states=states)
        assert spec.side == frozenset()


def test_landing_tasks_may_chain_after_the_final_task():
    chained = """\
  - id: a
    type: new_op
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [a]
  - id: more
    type: merge
    kind: merge
    depends_on: [land, a]
  - id: land-2
    type: merge
    kind: merge
    isolation: shared
    depends_on: [more]
"""
    assert parse_plan(_plan(chained, ["a", "land", "more", "land-2"])).sink_id() == "land-2"

    side_by_side = """\
  - id: a
    type: new_op
  - id: early
    type: merge
    kind: merge
    isolation: shared
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [a]
  - id: join
    type: merge
    kind: merge
    depends_on: [early, land]
  - id: last
    type: merge
    kind: merge
    isolation: shared
    depends_on: [join]
"""
    problems = _problems(_plan(side_by_side, ["a", "early", "land", "join", "last"]))
    assert len(problems) == 1 and "would land side by side" in problems[0]
