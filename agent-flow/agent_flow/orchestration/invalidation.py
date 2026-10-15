# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute which nodes a replan invalidates: changed ∪ transitive dependents.

A pause-based replan swaps the execution graph mid-run. Only the *invalidation
frontier* — nodes whose definition changed, plus every node that has (or will
have) consumed a changed node's output — may be cancelled and re-run; DONE and
unchanged in-flight nodes must be left untouched. This is a pure function of the
two graphs plus an optional per-node content signature for prose/criteria
changes the structural diff cannot see.
"""

from __future__ import annotations

from collections.abc import Callable

from agent_flow.orchestration.graph import ExecutionGraph, Node


def _effective_deps(graph: ExecutionGraph) -> dict[str, tuple[str, ...]]:
    """Map each node id to its own ``depends_on`` plus every ancestor's.

    Mirrors ``NodeScheduler._effective_deps``: a nested goal inherits its
    ancestor stages' cross-stage dependencies, so a change to an upstream stage
    must invalidate the downstream stage's goals too.
    """
    result: dict[str, tuple[str, ...]] = {}

    def walk(nodes: tuple[Node, ...], inherited: tuple[str, ...]) -> None:
        for node in nodes:
            own = tuple(node.depends_on) + inherited
            result[node.id] = own
            walk(node.children, own)

    walk(graph.nodes, ())
    return result


def _changed_ids(
    old_graph: ExecutionGraph,
    new_graph: ExecutionGraph,
    content_changed: Callable[[str], bool] | None,
) -> set[str]:
    """Ids present in ``new_graph`` whose definition differs from ``old_graph``."""
    changed: set[str] = set()
    for node_id, node in new_graph.by_id.items():
        old = old_graph.by_id.get(node_id)
        if old is None:
            changed.add(node_id)  # added node
            continue
        if old.kind != node.kind or tuple(old.depends_on) != tuple(node.depends_on):
            changed.add(node_id)
            continue
        if content_changed is not None and content_changed(node_id):
            changed.add(node_id)
    return changed


def invalidation_set(
    old_graph: ExecutionGraph,
    new_graph: ExecutionGraph,
    *,
    content_changed: Callable[[str], bool] | None = None,
) -> set[str]:
    """Return the node ids to cancel + reset: changed ∪ transitive dependents.

    All returned ids exist in ``new_graph``. "Changed" is the structural diff
    (added / kind / depends_on) optionally widened by ``content_changed``; the
    frontier then expands to every node that transitively depends on a changed
    node via effective (own + inherited-ancestor) ``depends_on`` edges.
    """
    changed = _changed_ids(old_graph, new_graph, content_changed)
    effective = _effective_deps(new_graph)

    dependents: dict[str, set[str]] = {node_id: set() for node_id in new_graph.by_id}
    for node_id, deps in effective.items():
        for dep in deps:
            if dep in dependents:
                dependents[dep].add(node_id)

    frontier: set[str] = set(changed)
    stack = list(changed)
    while stack:
        current = stack.pop()
        for dependent in dependents.get(current, ()):
            if dependent not in frontier:
                frontier.add(dependent)
                stack.append(dependent)

    return {node_id for node_id in frontier if node_id in new_graph.by_id}
