"""Graph transformation to eliminate redundant transpose operations in the model graph.

This transformation identifies and removes patterns where transpose operations with the same
dimensions are applied consecutively, which cancel each other out:
x = x.transpose(1, 2)
x = x.transpose(1, 2)
"""

from typing import Set, Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


def _is_transpose_op(node: Node) -> bool:
    """Check if the node is a transpose operation."""
    return is_op(node, torch.ops.aten.transpose)


def _is_contiguous_op(node: Node) -> bool:
    """Check if the node is a contiguous operation."""
    return is_op(node, torch.ops.aten.contiguous)


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


@TransformRegistry.register("eliminate_redundant_transposes")
class EliminateRedundantTransposes(BaseTransform):
    """Eliminate redundant transpose operations in the graph.

    This transformation identifies pairs of consecutive transpose operations with
    the same dimension arguments and removes both operations, as they cancel out.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph

        # Find pairs of redundant transpose operations
        nodes_to_eliminate: Set[Tuple[Node, Node]] = set()

        for t_node in gm.graph.nodes:
            # check if there is a transpose operation
            if not _is_transpose_op(t_node):
                continue

            # check if it's already part of a pair
            if any(t_node in pair for pair in nodes_to_eliminate):
                continue

            # check if there is only one user
            if len(t_node.users) > 1:
                continue

            # check if the user is a contiguous operation
            t_comp_node = list(t_node.users)[0]

            # check if the user is a contiguous operation
            has_contiguous = False
            while _is_contiguous_op(t_comp_node) and len(t_comp_node.users) == 1:
                has_contiguous = True
                t_comp_node = list(t_comp_node.users)[0]

            # check if the user is a transpose operation
            if not _is_transpose_op(t_comp_node):
                continue

            # check if the transpose operation has the same dimension arguments
            if not _are_transpose_args_same(t_node, t_comp_node):
                continue

            # add the pair to the set
            nodes_to_eliminate.add((t_node, t_comp_node, has_contiguous))

        # Eliminate redundant transpose pairs
        for t_node, t_comp_node, has_contiguous in nodes_to_eliminate:
            # Replace all uses of the second transpose with the input to the first transpose
            original_input = t_node.args[0]
            t_comp_node.replace_all_uses_with(original_input)

            # if there is a contiguous operation that we skipped, let add it after t_comp_node as new
            # graph node that call contiguous on t_comp_node
            if has_contiguous:
                with graph.inserting_after(original_input):
                    new_contiguous_node = graph.call_function(
                        torch.ops.aten.contiguous.default, args=(original_input,)
                    )
                original_input.replace_all_uses_with(new_contiguous_node)
                new_contiguous_node.replace_input_with(new_contiguous_node, original_input)

        # Clean up the graph
        if nodes_to_eliminate:
            gm.graph.eliminate_dead_code()

        info = TransformInfo(
            skipped=False,
            num_matches=len(nodes_to_eliminate),
            is_clean=False,
            has_valid_shapes=False,
        )

        return gm, info
