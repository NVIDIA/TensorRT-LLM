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

"""Transform to gather hidden states before LM head using logits_gather_mask.

This moves the gather operation into the model graph before the LM head,
enabling CUDA graph capture and reducing computation by only computing
logits for the tokens that are actually needed.
"""

from typing import Tuple

import torch
from torch.fx import GraphModule

from ...utils._graph import add_graph_input
from ...utils.logger import ad_logger
from ...utils.node_utils import is_linear_op, is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


@torch.library.custom_op("auto_deploy::gather_logits_before_lm_head", mutates_args=())
def gather_logits_before_lm_head(
    hidden_states: torch.Tensor,
    logits_gather_mask: torch.Tensor,  # long tensor
) -> torch.Tensor:
    """Gather hidden states using logits_gather_mask before LM head.

    Args:
        hidden_states: Hidden states tensor [b, 1, hidden] or [1, s_total, hidden]
        logits_gather_mask: mask for gathering logits. Only used for mixed batches

    Returns:
        Gathered and flattened hidden states [num_gathered_tokens, hidden]
    """
    # final shape is [total_tokens, hidden] from [b, 1, hidden] or [1, total_tokens, hidden]
    is_decode_only = hidden_states.shape[1] == 1
    hidden_states = hidden_states.squeeze(int(is_decode_only))

    if is_decode_only:
        return hidden_states.clone()
    return hidden_states[logits_gather_mask[: hidden_states.shape[0]], :]


@gather_logits_before_lm_head.register_fake
def gather_logits_before_lm_head_fake(
    hidden_states: torch.Tensor,
    logits_gather_mask: torch.Tensor,
) -> torch.Tensor:
    is_decode_only = hidden_states.shape[1] == 1
    hidden_states = hidden_states.squeeze(int(is_decode_only))
    return torch.empty_like(hidden_states)


@TransformRegistry.register("gather_logits_before_lm_head")
class GatherLogitsBeforeLmHeadTransform(BaseTransform):
    """Transform to gather hidden states before LM head using logits_gather_mask.

    This transform inserts a gather operation before the LM head linear layer
    to select only the hidden states that need logits computed. The output is always
    [b, hidden_size] in decode-only for CUDA graph compatibility.

    Benefits:
    - Reduces computation by only computing logits for needed tokens
    - Eliminates Python loop overhead
    - Enables CUDA graph capture of the gather
    - Moves gather into the graph for better optimization
    """

    def _apply(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        ad_logger.info("Applying GatherLogitsBeforeLmHead transform...")

        # assume lm head node is the input to the output node
        lm_head_node = gm.graph.find_nodes(op="output")[0].all_input_nodes[0]
        if is_op(lm_head_node, torch.ops.aten.to):
            lm_head_node = lm_head_node.all_input_nodes[0]

        if is_linear_op(lm_head_node):
            node_to_gather = lm_head_node.all_input_nodes[0]
            ad_logger.info(f"Found LM head node: {lm_head_node.name}")
        else:
            node_to_gather = lm_head_node
            ad_logger.warning("lm_head node is not linear, using it as the node to gather")

        # Add logits_gather_mask as input in the graph and the sequence info interface
        logits_gather_mask_node = add_graph_input(gm, "logits_gather_mask")
        cm.info.register_arg(
            "logits_gather_mask", torch.ones(cm.info.max_num_tokens, dtype=torch.bool)
        )
        with gm.graph.inserting_after(node_to_gather):
            gathered_node = gm.graph.call_function(
                torch.ops.auto_deploy.gather_logits_before_lm_head.default,
                args=(node_to_gather, logits_gather_mask_node),
            )
        node_to_gather.replace_all_uses_with(gathered_node)
        gathered_node.replace_input_with(gathered_node, node_to_gather)

        return gm, TransformInfo(skipped=False, num_matches=1)
