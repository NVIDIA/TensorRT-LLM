# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transform to gather hidden states before LM head using logit_gather_ids.

This moves the gather operation into the model graph before the LM head,
enabling CUDA graph capture and reducing computation by only computing
logits for the tokens that are actually needed.
"""

from typing import Optional, Tuple

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...utils._graph import add_graph_input
from ...utils.logger import ad_logger
from ...utils.node_utils import is_linear_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


# Register the custom op for gather_logits_before_lm_head
@torch.library.custom_op("auto_deploy::gather_logits_before_lm_head", mutates_args=())
def gather_logits_before_lm_head(
    hidden_states: torch.Tensor,
    logit_gather_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather hidden states using logit_gather_ids before LM head.

    Args:
        hidden_states: Hidden states tensor [batch, seq_len, hidden] or
            [1, total_tokens, hidden] or [total_tokens, hidden]
        logit_gather_ids: Gather indices [max_batch_size], one per sequence
            (index of token to gather)

    Returns:
        Gathered hidden states [batch, hidden] for generate, [1, max_batch_size, hidden] for packed
    """
    # Generate format: [batch, 1, hidden] -> seq_len == 1
    # Packed format: [1, total_tokens, hidden] -> seq_len > 1
    if hidden_states.shape[1] == 1:
        return hidden_states.clone()

    gathered = hidden_states[:, logit_gather_ids.long(), :]
    return gathered


@gather_logits_before_lm_head.register_fake
def gather_logits_before_lm_head_fake(
    hidden_states: torch.Tensor,
    logit_gather_ids: torch.Tensor,
) -> torch.Tensor:
    """Fake implementation for tracing - matches real implementation logic."""
    max_batch_size = logit_gather_ids.shape[0]
    hidden_size = hidden_states.shape[-1]
    # Check sequence length dimension to distinguish formats (consistent with real implementation)
    if hidden_states.shape[1] == 1:
        batch_size = hidden_states.shape[0]
        return hidden_states.new_empty(batch_size, 1, hidden_size)
    # Packed format: return [1, max_batch_size, hidden_size] (keep 3D)
    return hidden_states.new_empty(1, max_batch_size, hidden_size)


def _get_model_device(gm: GraphModule) -> torch.device:
    """Get the device of the model from its parameters/buffers."""
    for param in gm.parameters():
        return param.device
    for buffer in gm.buffers():
        return buffer.device
    return torch.device("cuda")


def _find_input_node(gm: GraphModule, input_name: str) -> Optional[Node]:
    """Find an input node by name in the graph."""
    for node in gm.graph.nodes:
        if node.op == "placeholder" and node.name == input_name:
            return node
    return None


def _find_lm_head_node(gm: GraphModule) -> Optional[Node]:
    """Find the LM head linear node (last linear layer before output).

    Returns:
        The LM head node if found, None otherwise.
    """
    output_node = None
    for node in gm.graph.nodes:
        if node.op == "output":
            output_node = node
            break

    if output_node is None:
        return None

    # Find the linear node that directly feeds into the output
    # We traverse backwards from the output to find the LM head
    def find_direct_linear(node: Node, visited: set, depth: int = 0) -> Optional[Tuple[Node, int]]:
        """Find linear nodes, returning (node, depth) tuple."""
        if node in visited:
            return None
        visited.add(node)

        # Check if this is a linear op
        if is_linear_op(node):
            return (node, depth)

        # Check all input nodes, preferring closer nodes
        best_result = None
        best_depth = float("inf")
        for input_node in node.all_input_nodes:
            result = find_direct_linear(input_node, visited, depth + 1)
            if result is not None:
                node_result, node_depth = result
                if node_depth < best_depth:
                    best_result = node_result
                    best_depth = node_depth

        return (best_result, best_depth) if best_result is not None else None

    visited = set()
    if isinstance(output_node.args[0], (list, tuple)):
        # Output is a tuple/list - check first element (logits)
        logits_node = output_node.args[0][0] if output_node.args[0] else None
    else:
        logits_node = output_node.args[0]

    if logits_node is None:
        return None

    result = find_direct_linear(logits_node, visited)
    if result is None:
        return None

    lm_head_node, _ = result
    return lm_head_node


class GatherLogitsBeforeLmHeadConfig(TransformConfig):
    """Configuration for GatherLogitsBeforeLmHead transform."""

    enabled: bool = Field(
        default=True,
        description="Whether to enable the gather-logits-before-lm-head optimization.",
    )
    max_batch_size: int = Field(
        default=1024, description="Maximum batch size for the logit_gather_ids buffer."
    )


@TransformRegistry.register("gather_logits_before_lm_head")
class GatherLogitsBeforeLmHeadTransform(BaseTransform):
    """Transform to gather hidden states before LM head using logit_gather_ids.

    This transform inserts a gather operation before the LM head linear layer
    to select only the hidden states that need logits computed. The output is always
    [max_batch_size, hidden_size] for CUDA graph compatibility.

    Benefits:
    - Reduces computation by only computing logits for needed tokens
    - Eliminates Python loop overhead
    - Enables CUDA graph capture of the gather
    - Moves gather into the graph for better optimization

    NOTE: This transform requires logit_gather_ids to be populated in SequenceInfo.
    """

    config: GatherLogitsBeforeLmHeadConfig

    @classmethod
    def get_config_class(cls):
        return GatherLogitsBeforeLmHeadConfig

    def _apply(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # Return early if disabled
        if not self.config.enabled:
            ad_logger.debug("GatherLogitsBeforeLmHead transform is disabled")
            return gm, TransformInfo(skipped=True, num_matches=0)

        ad_logger.info("Applying GatherLogitsBeforeLmHead transform...")

        # Find the LM head node
        lm_head_node = _find_lm_head_node(gm)
        if lm_head_node is None:
            ad_logger.warning(
                "Could not find LM head node, skipping GatherLogitsBeforeLmHead transform"
            )
            return gm, TransformInfo(skipped=True, num_matches=0)

        ad_logger.info(f"Found LM head node: {lm_head_node.name}")

        # Get the input to the LM head (hidden states)
        hidden_states_node = lm_head_node.args[0]

        # Find input nodes for logit_gather_ids and seq_len
        logit_gather_ids_node = _find_input_node(gm, "logit_gather_ids")
        seq_len_node = _find_input_node(gm, "seq_len")

        # Add logit_gather_ids as input if it doesn't exist
        if logit_gather_ids_node is None:
            ad_logger.info("logit_gather_ids input node not found, adding it to the graph")
            # Get example value from SequenceInfo if available
            val = None
            if hasattr(cm, "info") and "logit_gather_ids" in cm.info._args_device:
                val = cm.info._args_device["logit_gather_ids"]
            logit_gather_ids_node = add_graph_input(gm, "logit_gather_ids", val=val)

        if seq_len_node is None:
            ad_logger.warning(
                "Could not find seq_len input node. "
                "This should not happen if cached attention is enabled."
            )
            return gm, TransformInfo(skipped=True, num_matches=0)

        ad_logger.info("Found logit_gather_ids and seq_len input nodes")

        # Insert gather operation before LM head
        with gm.graph.inserting_before(lm_head_node):
            gather_node = gm.graph.call_function(
                torch.ops.auto_deploy.gather_logits_before_lm_head,
                args=(hidden_states_node, logit_gather_ids_node),
            )

        # Update LM head to use gathered hidden states
        new_args = (gather_node,) + tuple(lm_head_node.args[1:])
        lm_head_node.args = new_args

        # Mark the model so executor knows gather is already done
        gm._gather_logits_before_lm_head_applied = True

        # Recompile
        gm.graph.lint()
        gm.recompile()

        ad_logger.info(
            f"Successfully inserted gather_logits_before_lm_head before: {lm_head_node.name}"
        )
        return gm, TransformInfo(skipped=False, num_matches=1)
