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
    states = {"a": NodeState.DONE, "b": NodeState.FAILED, "land": NodeState.BLOCKED}

    changed = _VALID_NODES.replace("type: new_op", "type: new_op\n    tags: [target]")
    problems = _problems(_plan(changed, ["a", "b", "land"]), previous=previous, states=states)
    assert problems == [
        "task 'a' is DONE and its graph entry cannot change; add a new task for "
        "follow-up work instead"
    ]

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


def test_a_revision_may_rewrite_unfinished_tasks_and_add_new_ones():
    previous = parse_plan(_plan(_VALID_NODES, ["a", "b", "land"]))
    states = {"a": NodeState.DONE, "b": NodeState.FAILED, "land": NodeState.BLOCKED}
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
