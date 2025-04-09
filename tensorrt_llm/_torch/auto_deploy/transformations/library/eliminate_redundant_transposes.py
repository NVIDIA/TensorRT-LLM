"""Graph transformation to eliminate redundant transpose operations in the model graph.

This transformation identifies and removes patterns where transpose operations with the same
dimensions are applied consecutively, which cancel each other out:
x = x.transpose(1, 2)
x = x.transpose(1, 2)
"""

from typing import Dict, List, Set, Tuple

import torch
from torch.fx import GraphModule, Node

from ...utils.logger import ad_logger
from .._graph import canonicalize_graph


def _is_transpose_op(node: Node) -> bool:
    """Check if a node is a transpose operation."""
    return node.op == "call_function" and (
        node.target == torch.transpose or node.target == torch.ops.aten.transpose.int
    )


def _are_transpose_args_same(node1: Node, node2: Node) -> bool:
    """Check if two transpose nodes have the same dimension arguments."""
    # Get the dimension arguments for both nodes
    # Args structure: (input_tensor, dim1, dim2)
    if len(node1.args) < 3 or len(node2.args) < 3:
        return False

    dim1_node1, dim2_node1 = node1.args[1], node1.args[2]
    dim1_node2, dim2_node2 = node2.args[1], node2.args[2]

    # Check if the dimensions are the same
    return dim1_node1 == dim1_node2 and dim2_node1 == dim2_node2


def eliminate_redundant_transposes(gm: GraphModule) -> GraphModule:
    """Eliminate redundant transpose operations in the graph.

    This transformation identifies pairs of consecutive transpose operations with
    the same dimension arguments and removes both operations, as they cancel out.
    """
    ad_logger.info("Eliminating redundant transpose operations")
    ad_logger.debug("Before eliminating redundant transposes: " + str(gm))

    # Map nodes to their users to efficiently find transpose chains
    node_to_users: Dict[Node, List[Node]] = {}
    for node in gm.graph.nodes:
        for user in node.users:
            if user not in node_to_users:
                node_to_users[user] = []
            node_to_users[user].append(node)

    # Find transpose nodes
    transpose_nodes = [n for n in gm.graph.nodes if _is_transpose_op(n)]

    # Find pairs of redundant transpose operations
    nodes_to_eliminate: Set[Tuple[Node, Node]] = set()

    for node in transpose_nodes:
        if node not in node_to_users:
            continue

        for parent in node_to_users[node]:
            # Check if parent is also a transpose with the same dimensions
            if _is_transpose_op(parent) and _are_transpose_args_same(node, parent):
                # If the parent node has only one user (the current node),
                # and the current node is only used for the transpose operation,
                # then both nodes can be eliminated
                if len(list(parent.users)) == 1 and len(list(node.users)) > 0:
                    nodes_to_eliminate.add((parent, node))

    # Eliminate redundant transpose pairs
    for parent, node in nodes_to_eliminate:
        # Replace all uses of the second transpose with the input to the first transpose
        original_input = parent.args[0]
        node.replace_all_uses_with(original_input)

        ad_logger.debug(f"Eliminated redundant transpose pair: {parent} -> {node}")

    # Clean up the graph
    if nodes_to_eliminate:
        gm.graph.eliminate_dead_code()
        gm = canonicalize_graph(gm)

    ad_logger.debug("After eliminating redundant transposes: " + str(gm))
    return gm
