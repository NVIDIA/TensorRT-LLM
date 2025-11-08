"""Fusion transform for fusing activation functions into causal_conv1d operations."""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


def _match_causal_conv_activation_pattern(
    graph: GraphModule,
    target_op,
) -> List[Tuple[Node, Node, str]]:
    """
    Match the causal_conv + activation pattern in the graph.

    The pattern corresponds to:
        conv_out = cuda_cached_causal_conv1d(...)
        out = activation(conv_out)

    Args:
        graph: The graph module to search
        target_op: The target causal conv op to match

    Returns:
        A list of tuples (conv_node, activation_node, activation_name) for each match
    """
    matches = []

    for node in graph.nodes:
        if not is_op(node, target_op):
            continue

        # Check if this node has exactly one user and it's an activation
        if len(node.users) != 1:
            continue

        activation_node = list(node.users.keys())[0]
        if activation_node.op != "call_function":
            continue

        # Detect activation type
        activation_name: Optional[str] = None
        if activation_node.target in (torch.ops.aten.silu.default, F.silu):
            activation_name = "silu"
        # Can extend to support more activations here:
        # elif activation_node.target in (torch.ops.aten.gelu.default, F.gelu):
        #     activation_name = "gelu"

        if activation_name is not None:
            matches.append((node, activation_node, activation_name))

    return matches


@TransformRegistry.register("fuse_causal_conv_activation")
class FuseCausalConvActivation(BaseTransform):
    """Fuses activation functions into cached CUDA causal_conv1d operations.

    This transform detects patterns like:
        conv_out = cuda_cached_causal_conv1d(...)
        out = silu(conv_out)

    And replaces them with:
        out = cuda_cached_causal_conv1d(..., activation="silu")

    This optimization allows the backend CUDA kernels to fuse the activation,
    reducing memory bandwidth and improving performance.

    Note: This runs AFTER insert_cached_causal_conv, so it operates on the
    cached CUDA operations, not the uncached torch operations.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph

        # Step 1: Identify causal_conv + activation pattern
        matches = _match_causal_conv_activation_pattern(
            graph,
            target_op=torch.ops.auto_deploy.cuda_cached_causal_conv1d,
        )

        # Step 2: Replace matched patterns with fused version
        for conv_node, activation_node, activation_name in matches:
            with graph.inserting_after(conv_node):
                # Create new call with fused activation
                # Replace the last arg (activation=None) with activation_name
                new_args = list(conv_node.args[:-1]) + [activation_name]
                fused_node = graph.call_function(
                    torch.ops.auto_deploy.cuda_cached_causal_conv1d,
                    args=tuple(new_args),
                )

                # Replace all uses of activation_node with fused_node
                activation_node.replace_all_uses_with(fused_node)

                # Remove the old nodes
                graph.erase_node(activation_node)
                graph.erase_node(conv_node)

        gm.recompile()

        info = TransformInfo(
            skipped=False,
            num_matches=len(matches),
            is_clean=False,
            has_valid_shapes=False,
        )
        return gm, info
