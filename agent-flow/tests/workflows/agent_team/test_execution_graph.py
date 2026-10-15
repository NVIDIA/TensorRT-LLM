"""Tests for extracting the ``## Execution Graph`` block from plan.md.

Covers the brief's contract: a valid nested graph is extracted and parsed into
an :class:`ExecutionGraph`; a plan with no such section yields ``None`` (the
legacy/linear single-cursor path); and a section whose fenced block is
missing/empty/unterminated or whose YAML is invalid fails loudly with an
:class:`ExecutionGraphError`. All schema validation is delegated to
``parse_execution_graph`` — this module only does the markdown scan.
"""

from __future__ import annotations

import pytest

from agent_flow.orchestration import ExecutionGraph, ExecutionGraphError
from agent_flow.workflows.agent_team.execution_graph import extract_execution_graph

PLAN_WITH_GRAPH = """\
# Plan

Some prose describing the approach.

## Approach

Do the thing, carefully.

## Execution Graph

```yaml
nodes:
  - id: s1
    type: stage
    depends_on: []
    kind: impl
    children:
      - {id: s1.g1, type: goal, depends_on: [], kind: impl}
      - {id: s1.g2, type: goal, depends_on: [s1.g1], kind: impl}
  - id: s2
    type: stage
    depends_on: [s1]
    kind: impl
```

## Risks

Nothing notable.
"""

PLAN_NO_GRAPH = """\
# Plan

## Approach

A linear plan with no declared DAG.

## Risks

None.
"""


def test_extracts_valid_nested_graph():
    """A ``## Execution Graph`` + fenced yaml yields a parsed ExecutionGraph."""
    graph = extract_execution_graph(PLAN_WITH_GRAPH)
    assert isinstance(graph, ExecutionGraph)
    assert set(graph.by_id) == {"s1", "s1.g1", "s1.g2", "s2"}
    assert graph.by_id["s2"].depends_on == ("s1",)
    assert graph.by_id["s1.g2"].depends_on == ("s1.g1",)


def test_bare_fence_without_language_tag_is_accepted():
    """The ```` ```yaml ```` language tag is optional; a bare ```` ``` ```` works."""
    plan = "## Execution Graph\n\n```\nnodes:\n  - {id: a, type: stage}\n```\n"
    graph = extract_execution_graph(plan)
    assert isinstance(graph, ExecutionGraph)
    assert set(graph.by_id) == {"a"}


def test_numbered_section_heading_is_accepted():
    """A numbered heading still parses — planners often number sections (`## 7. ...`).

    Regression: a real concurrent run aborted because the PlanDrafter numbered every
    section, writing `## 7. Execution Graph`, and the exact-match heading regex missed
    it → ``None`` → "no Execution Graph" crash despite a perfectly good graph.
    """
    for heading in ("## 7. Execution Graph", "## 3) Execution Graph", "## 12 Execution Graph"):
        plan = f"{heading}\n\n```yaml\nnodes:\n  - {{id: a, type: stage}}\n```\n"
        graph = extract_execution_graph(plan)
        assert graph is not None, f"heading {heading!r} should parse"
        assert {n.id for n in graph.nodes} == {"a"}


def test_no_section_returns_none():
    """No ``## Execution Graph`` section => legacy/linear plan => ``None``."""
    assert extract_execution_graph(PLAN_NO_GRAPH) is None
    assert extract_execution_graph("") is None


def test_section_but_no_fenced_block_raises():
    """A section with prose but no fenced block must fail loudly, not return None."""
    plan = "## Execution Graph\n\nTODO: fill in the DAG.\n\n## Risks\n\nNone.\n"
    with pytest.raises(ExecutionGraphError):
        extract_execution_graph(plan)


def test_empty_fenced_block_raises():
    """An empty fenced block is malformed, not 'no graph'."""
    plan = "## Execution Graph\n\n```yaml\n```\n"
    with pytest.raises(ExecutionGraphError):
        extract_execution_graph(plan)


def test_whitespace_only_fenced_block_raises():
    """A block that holds only blank lines is treated as empty and rejected."""
    plan = "## Execution Graph\n\n```yaml\n\n   \n```\n"
    with pytest.raises(ExecutionGraphError):
        extract_execution_graph(plan)


def test_unterminated_fenced_block_raises():
    """An opening fence with no closing fence is a malformed block."""
    plan = "## Execution Graph\n\n```yaml\nnodes:\n  - {id: a, type: stage}\n"
    with pytest.raises(ExecutionGraphError):
        extract_execution_graph(plan)


def test_invalid_yaml_in_block_propagates():
    """A parseable block with a dependency cycle propagates the parser's error."""
    plan = (
        "## Execution Graph\n\n"
        "```yaml\n"
        "nodes:\n"
        "  - {id: a, type: stage, depends_on: [b]}\n"
        "  - {id: b, type: stage, depends_on: [a]}\n"
        "```\n"
    )
    with pytest.raises(ExecutionGraphError) as excinfo:
        extract_execution_graph(plan)
    message = str(excinfo.value)
    assert "a" in message and "b" in message


def test_fence_belonging_to_a_later_section_is_ignored():
    """Only the Execution Graph section's own block counts.

    A fence under a following ``## `` heading must not be adopted.
    """
    plan = (
        "## Execution Graph\n\n"
        "No block declared here yet.\n\n"
        "## Appendix\n\n"
        "```yaml\n"
        "nodes: []\n"
        "```\n"
    )
    with pytest.raises(ExecutionGraphError):
        extract_execution_graph(plan)


def test_takes_first_fenced_block_in_section():
    """When the section holds two fenced blocks, the first one is the graph."""
    plan = (
        "## Execution Graph\n\n"
        "```yaml\n"
        "nodes:\n"
        "  - {id: first, type: stage}\n"
        "```\n\n"
        "```yaml\n"
        "nodes:\n"
        "  - {id: second, type: stage}\n"
        "```\n"
    )
    graph = extract_execution_graph(plan)
    assert isinstance(graph, ExecutionGraph)
    assert set(graph.by_id) == {"first"}
