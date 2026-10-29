"""Tests for the execution-graph data model and YAML parser.

Covers the brief's cases: valid nested parse, cycle rejection, unknown
dependency, duplicate id, bad enum value, and a Python<->YAML round-trip that
verifies ids/edges/children survive parsing intact. Also pins the precise,
id-naming error messages the scheduler (Task 5) and human-facing tooling rely
on.
"""

from __future__ import annotations

import pytest

from agent_flow.orchestration.graph import (
    ExecutionGraph,
    ExecutionGraphError,
    Node,
    NodeState,
    parse_execution_graph,
)

VALID = """
nodes:
  - id: s1
    type: stage
    depends_on: []
    kind: impl
    children:
      - {id: s1.g1, type: goal, depends_on: [], kind: impl}
      - {id: s1.g2, type: goal, depends_on: [], kind: impl}
      - {id: s1.gm, type: goal, depends_on: [s1.g1, s1.g2], kind: merge}
  - id: s2
    type: stage
    depends_on: [s1]
    kind: impl
"""


def test_parse_valid_nested_graph():
    """A nested stage->goal graph parses with a global id index and merge kinds."""
    g = parse_execution_graph(VALID)
    assert set(g.by_id) == {"s1", "s1.g1", "s1.g2", "s1.gm", "s2"}
    assert g.by_id["s2"].depends_on == ("s1",)
    assert g.by_id["s1.gm"].kind == "merge"


def test_defaults_applied_for_optional_fields():
    """Omitted ``depends_on``/``kind``/``isolation`` fall back to model defaults."""
    g = parse_execution_graph("nodes:\n  - {id: a, type: t}")
    node = g.by_id["a"]
    assert node.depends_on == ()
    assert node.kind == "impl"
    assert node.isolation == "worktree"
    assert node.children == ()


def test_children_and_edges_are_tuples():
    """Parsed collections are immutable tuples (Node is a frozen value object)."""
    g = parse_execution_graph(VALID)
    s1 = g.by_id["s1"]
    assert isinstance(s1.children, tuple)
    assert isinstance(s1.depends_on, tuple)
    assert tuple(c.id for c in s1.children) == ("s1.g1", "s1.g2", "s1.gm")
    assert g.by_id["s1.gm"].depends_on == ("s1.g1", "s1.g2")


def test_flatten_is_preorder_over_all_levels():
    """``flatten`` yields every node across nesting in stable pre-order."""
    g = parse_execution_graph(VALID)
    assert tuple(n.id for n in g.flatten()) == (
        "s1",
        "s1.g1",
        "s1.g2",
        "s1.gm",
        "s2",
    )


def test_round_trip_structure_equals_hand_built_model():
    """Parsing reconstructs exactly the hand-built frozen Node tree (ids/edges/children)."""
    expected = ExecutionGraph(
        nodes=(
            Node(
                id="s1",
                type="stage",
                depends_on=(),
                kind="impl",
                children=(
                    Node(id="s1.g1", type="goal", kind="impl"),
                    Node(id="s1.g2", type="goal", kind="impl"),
                    Node(
                        id="s1.gm",
                        type="goal",
                        depends_on=("s1.g1", "s1.g2"),
                        kind="merge",
                    ),
                ),
            ),
            Node(id="s2", type="stage", depends_on=("s1",), kind="impl"),
        )
    )
    assert parse_execution_graph(VALID) == expected


def test_reject_cycle():
    """A sibling dependency cycle is rejected with the offending members named."""
    with pytest.raises(ExecutionGraphError) as excinfo:
        parse_execution_graph(
            "nodes:\n  - {id: a, type: t, depends_on: [b]}\n  - {id: b, type: t, depends_on: [a]}"
        )
    message = str(excinfo.value)
    assert "a" in message and "b" in message


def test_reject_cycle_names_nested_level():
    """Cycles are detected per nesting level, not just at the top level."""
    text = (
        "nodes:\n"
        "  - id: s1\n"
        "    type: stage\n"
        "    children:\n"
        "      - {id: c1, type: goal, depends_on: [c2]}\n"
        "      - {id: c2, type: goal, depends_on: [c1]}\n"
    )
    with pytest.raises(ExecutionGraphError) as excinfo:
        parse_execution_graph(text)
    assert "c1" in str(excinfo.value) and "c2" in str(excinfo.value)


def test_reject_unknown_dependency():
    """A ``depends_on`` target with no matching sibling id is rejected."""
    with pytest.raises(ExecutionGraphError) as excinfo:
        parse_execution_graph("nodes:\n  - {id: a, type: t, depends_on: [ghost]}")
    assert "ghost" in str(excinfo.value)


def test_reject_dependency_on_non_sibling():
    """``depends_on`` resolves to siblings only, even when the id exists elsewhere."""
    text = (
        "nodes:\n"
        "  - id: s1\n"
        "    type: stage\n"
        "    children:\n"
        # s2 is a top-level id, not a sibling of s1.g1 -> must be rejected.
        "      - {id: s1.g1, type: goal, depends_on: [s2]}\n"
        "  - {id: s2, type: stage}\n"
    )
    with pytest.raises(ExecutionGraphError) as excinfo:
        parse_execution_graph(text)
    assert "s2" in str(excinfo.value) and "s1.g1" in str(excinfo.value)


def test_reject_duplicate_id():
    """Ids must be globally unique, including across nesting levels."""
    text = "nodes:\n  - id: dup\n    type: stage\n    children:\n      - {id: dup, type: goal}\n"
    with pytest.raises(ExecutionGraphError) as excinfo:
        parse_execution_graph(text)
    assert "dup" in str(excinfo.value)


def test_reject_bad_kind():
    """An out-of-range ``kind`` names the node and the bad value."""
    with pytest.raises(ExecutionGraphError) as excinfo:
        parse_execution_graph("nodes:\n  - {id: a, type: t, kind: bogus}")
    message = str(excinfo.value)
    assert "a" in message and "bogus" in message


def test_reject_bad_isolation():
    """An out-of-range ``isolation`` names the node and the bad value."""
    with pytest.raises(ExecutionGraphError) as excinfo:
        parse_execution_graph("nodes:\n  - {id: a, type: t, isolation: nope}")
    message = str(excinfo.value)
    assert "a" in message and "nope" in message


def test_reject_not_a_mapping():
    """A top-level YAML list (not a mapping) is rejected."""
    with pytest.raises(ExecutionGraphError):
        parse_execution_graph("- just\n- a\n- list")


def test_reject_missing_nodes_key():
    """A mapping without a ``nodes`` key is rejected."""
    with pytest.raises(ExecutionGraphError) as excinfo:
        parse_execution_graph("other: value")
    assert "nodes" in str(excinfo.value)


def test_reject_node_missing_id():
    """Each node must carry an ``id``."""
    with pytest.raises(ExecutionGraphError):
        parse_execution_graph("nodes:\n  - {type: t}")


def test_reject_node_missing_type():
    """Each node must carry a ``type``; the message names the offending id."""
    with pytest.raises(ExecutionGraphError) as excinfo:
        parse_execution_graph("nodes:\n  - {id: a}")
    assert "a" in str(excinfo.value)


def test_empty_nodes_list_is_valid():
    """An empty graph is a valid (degenerate) graph."""
    g = parse_execution_graph("nodes: []")
    assert g.nodes == ()
    assert g.by_id == {}


def test_node_state_enum_values():
    """NodeState carries the runtime lifecycle states the scheduler consumes."""
    assert {s.value for s in NodeState} == {
        "pending",
        "running",
        "done",
        "failed",
        "blocked",
        "interrupted",
    }
    # str-mixin so states compare/serialize as plain strings.
    assert NodeState.PENDING == "pending"
