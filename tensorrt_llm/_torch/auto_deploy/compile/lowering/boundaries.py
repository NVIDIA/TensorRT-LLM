from __future__ import annotations

from collections.abc import Iterable, Mapping

from torch.fx import GraphModule, Node
from torch.fx.node import map_arg

from .datamodel import BoundaryValue, ModeContext, RegionSpec, SupportDecision, ValueType


def graph_order(graph_module: GraphModule) -> dict[Node, int]:
    return {node: index for index, node in enumerate(graph_module.graph.nodes)}


def executable_nodes(graph_module: GraphModule) -> tuple[Node, ...]:
    return tuple(
        node
        for node in graph_module.graph.nodes
        if node.op in {"call_function", "call_method", "call_module"}
    )


def _iter_input_nodes(value) -> tuple[Node, ...]:
    nodes: list[Node] = []

    def collect(node: Node) -> Node:
        nodes.append(node)
        return node

    map_arg(value, collect)
    return tuple(nodes)


def _sort_by_graph_order(nodes: Iterable[Node], order: Mapping[Node, int]) -> tuple[Node, ...]:
    return tuple(sorted(nodes, key=lambda node: order.get(node, len(order))))


def analyze_region_boundaries(
    graph_module: GraphModule,
    nodes: Iterable[Node],
    *,
    mode: ModeContext | None = None,
    region_id: str = "region_0",
    support: Mapping[str, SupportDecision] | None = None,
) -> RegionSpec:
    """Compute read-only FX dataflow boundaries for a region."""

    order = graph_order(graph_module)
    region_nodes = _sort_by_graph_order(nodes, order)
    region_set = set(region_nodes)

    external_inputs: set[Node] = set()
    for node in region_nodes:
        for input_node in _iter_input_nodes((node.args, node.kwargs)):
            if input_node not in region_set:
                external_inputs.add(input_node)

    internal_outputs = {
        node for node in region_nodes if any(user not in region_set for user in node.users)
    }

    inputs = tuple(
        BoundaryValue(node=node, type=ValueType.from_node(node))
        for node in _sort_by_graph_order(external_inputs, order)
    )
    outputs = tuple(
        BoundaryValue(node=node, type=ValueType.from_node(node))
        for node in _sort_by_graph_order(internal_outputs, order)
    )

    return RegionSpec(
        region_id=region_id,
        mode=mode or ModeContext("default"),
        source_nodes=region_nodes,
        boundary_inputs=inputs,
        boundary_outputs=outputs,
        constraints=tuple(
            constraint
            for decision in (support or {}).values()
            for constraint in decision.constraints
        ),
        debug_name=region_id,
        support=dict(support or {}),
    )
