# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution-graph data model and YAML parser.

This module is the workflow-agnostic core of the concurrent DAG engine. It is
deliberately self-contained: it depends only on the standard library and
``pyyaml`` and must never import agent, backend, or ``layers`` code. Later tasks
layer isolation (worktrees) and a scheduler on top of the pure model defined
here.

The model is a forest of :class:`Node` objects. Each node may carry ``children``
that form their own sub-DAG. ``depends_on`` edges reference *siblings within the
same subgraph* (never a parent, child, or cousin), while ids are unique
*globally across nesting* so a single flat index can address any node. Every
sibling level must be acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, get_args

import yaml

NodeKind = Literal["impl", "merge"]
"""What a node *does* to the workspace. ``impl`` produces work; ``merge`` folds
its dependencies together. Opaque to parsing beyond membership validation."""

Isolation = Literal["worktree", "shared"]
"""How a node's execution is isolated. ``worktree`` runs in its own git worktree
(Task 4); ``shared`` runs in the parent workspace."""

_VALID_KINDS: tuple[str, ...] = get_args(NodeKind)
_VALID_ISOLATIONS: tuple[str, ...] = get_args(Isolation)


class NodeState(str, Enum):
    """Runtime lifecycle state of a node, consumed by the scheduler (Task 5).

    Mixes in :class:`str` so states serialize and compare as plain strings
    (``NodeState.PENDING == "pending"``). This model module never assigns state
    — nodes are parsed as static structure — but the enum lives here so the
    scheduler and the data model share one vocabulary.
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"
    INTERRUPTED = "interrupted"


class ExecutionGraphError(ValueError):
    """Raised when a graph is structurally invalid or its YAML is malformed.

    Every message names the offending id(s): the duplicate id, the unresolved
    ``depends_on`` target, the members of a cycle, or the node with a bad enum
    value.
    """


@dataclass(frozen=True)
class Node:
    """A single unit of work in the execution graph.

    Attributes:
        id: Globally unique identifier (unique across every nesting level).
        type: Opaque workflow label (e.g. ``"stage"``, ``"goal"``). The engine
            never interprets it.
        depends_on: Ids of sibling nodes that must complete first. Resolves
            within this node's own subgraph only.
        kind: Whether the node implements work or merges its dependencies.
        isolation: Where the node executes relative to its parent workspace.
        children: Nested sub-DAG. Children form their own sibling scope for
            ``depends_on`` resolution and acyclicity.
    """

    id: str
    type: str
    depends_on: tuple[str, ...] = ()
    kind: NodeKind = "impl"
    isolation: Isolation = "worktree"
    children: tuple[Node, ...] = ()


@dataclass(frozen=True)
class ExecutionGraph:
    """A validated forest of :class:`Node` objects.

    Construction validates the whole tree eagerly (:meth:`__post_init__`) so an
    ``ExecutionGraph`` instance is always well-formed: globally unique ids,
    every ``depends_on`` resolving to a same-level sibling, and every sibling
    level acyclic. ``by_id`` is a flat index spanning all nesting levels.
    """

    nodes: tuple[Node, ...]
    by_id: dict[str, Node] = field(init=False, default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        index: dict[str, Node] = {}
        for node in self.flatten():
            if node.id in index:
                raise ExecutionGraphError(f"duplicate node id: {node.id!r}")
            index[node.id] = node
        # ``frozen=True`` blocks normal assignment; go through object.__setattr__.
        object.__setattr__(self, "by_id", index)
        self._validate_level(self.nodes)

    def flatten(self) -> tuple[Node, ...]:
        """Return every node across all nesting levels in stable pre-order."""
        result: list[Node] = []

        def walk(nodes: tuple[Node, ...]) -> None:
            for node in nodes:
                result.append(node)
                walk(node.children)

        walk(self.nodes)
        return tuple(result)

    def _validate_level(self, nodes: tuple[Node, ...]) -> None:
        """Validate one sibling level, then recurse into each node's children."""
        sibling_ids = {node.id for node in nodes}
        for node in nodes:
            for dep in node.depends_on:
                if dep not in sibling_ids:
                    raise ExecutionGraphError(
                        f"node {node.id!r} depends on {dep!r}, which is not a "
                        f"sibling in the same subgraph"
                    )
        _detect_cycle(nodes)
        for node in nodes:
            self._validate_level(node.children)


def _detect_cycle(nodes: tuple[Node, ...]) -> None:
    """Raise if the sibling ``depends_on`` edges form a cycle (three-color DFS).

    Assumes every ``depends_on`` target is a known sibling (dependency
    resolution runs first), so edges point only within ``nodes``.
    """
    edges = {node.id: node.depends_on for node in nodes}
    white, gray, black = 0, 1, 2
    color = dict.fromkeys(edges, white)
    path: list[str] = []

    def visit(current: str) -> None:
        color[current] = gray
        path.append(current)
        for target in edges[current]:
            if color[target] == gray:
                start = path.index(target)
                cycle = path[start:] + [target]
                raise ExecutionGraphError(
                    "dependency cycle detected among siblings: " + " -> ".join(cycle)
                )
            if color[target] == white:
                visit(target)
        path.pop()
        color[current] = black

    for node_id in edges:
        if color[node_id] == white:
            visit(node_id)


def _parse_node(raw: object) -> Node:
    """Build a :class:`Node` from a raw mapping, validating fields and enums."""
    if not isinstance(raw, dict):
        raise ExecutionGraphError(f"each node must be a mapping, got {type(raw).__name__}")

    if "id" not in raw:
        raise ExecutionGraphError("node is missing required key 'id'")
    node_id = raw["id"]
    if not isinstance(node_id, str) or not node_id:
        raise ExecutionGraphError(f"node id must be a non-empty string, got {node_id!r}")

    if "type" not in raw:
        raise ExecutionGraphError(f"node {node_id!r} is missing required key 'type'")
    node_type = raw["type"]
    if not isinstance(node_type, str) or not node_type:
        raise ExecutionGraphError(
            f"node {node_id!r} has invalid type {node_type!r}; expected a non-empty string"
        )

    depends_on_raw = raw.get("depends_on") or ()
    if not isinstance(depends_on_raw, (list, tuple)):
        raise ExecutionGraphError(
            f"node {node_id!r} has invalid depends_on {depends_on_raw!r}; expected a list"
        )
    depends_on = tuple(depends_on_raw)

    kind = raw.get("kind", "impl")
    if kind not in _VALID_KINDS:
        raise ExecutionGraphError(
            f"node {node_id!r} has invalid kind {kind!r}; expected one of {', '.join(_VALID_KINDS)}"
        )

    isolation = raw.get("isolation", "worktree")
    if isolation not in _VALID_ISOLATIONS:
        raise ExecutionGraphError(
            f"node {node_id!r} has invalid isolation {isolation!r}; "
            f"expected one of {', '.join(_VALID_ISOLATIONS)}"
        )

    children_raw = raw.get("children") or ()
    if not isinstance(children_raw, (list, tuple)):
        raise ExecutionGraphError(
            f"node {node_id!r} has invalid children {children_raw!r}; expected a list"
        )
    children = tuple(_parse_node(child) for child in children_raw)

    return Node(
        id=node_id,
        type=node_type,
        depends_on=depends_on,
        kind=kind,
        isolation=isolation,
        children=children,
    )


def parse_execution_graph(text: str) -> ExecutionGraph:
    """Parse the body of a ``## Execution Graph`` block into an ExecutionGraph.

    Args:
        text: YAML source for a mapping with a top-level ``nodes:`` list.

    Returns:
        A validated :class:`ExecutionGraph`.

    Raises:
        ExecutionGraphError: On malformed YAML, a non-mapping root, a missing
            ``nodes`` key, a bad enum value, an unresolved ``depends_on``
            target, a duplicate id, or a dependency cycle. Every message names
            the offending id(s).
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ExecutionGraphError(f"invalid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ExecutionGraphError(
            f"execution graph must be a mapping with a 'nodes' key, got {type(data).__name__}"
        )
    if "nodes" not in data:
        raise ExecutionGraphError("execution graph is missing required key 'nodes'")

    nodes_raw = data["nodes"] or []
    if not isinstance(nodes_raw, (list, tuple)):
        raise ExecutionGraphError(f"'nodes' must be a list, got {type(nodes_raw).__name__}")

    nodes = tuple(_parse_node(node) for node in nodes_raw)
    return ExecutionGraph(nodes=nodes)
