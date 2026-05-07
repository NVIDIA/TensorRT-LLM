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

from ...utils._graph import get_lm_head_node
from ...utils.node_utils import is_linear_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


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
        self._log_info("Applying GatherLogitsBeforeLmHead transform...")

        # assume lm head node is the input to the output node
        lm_head_node = get_lm_head_node(gm)

        if is_linear_op(lm_head_node):
            node_to_gather = lm_head_node.all_input_nodes[0]
            self._log_info(f"Found LM head node: {lm_head_node.name}")
        else:
            # Walk backward through SINGLE-INPUT elementwise/unary ops
            # (e.g. Gemma4 softcapping: linear → div → tanh → mul) to find the
            # actual lm_head linear node.  Only follow nodes that have exactly
            # one tensor input to avoid branching into the model body (e.g.
            # residual adds, fused allreduce+norm ops).
            current = lm_head_node
            while current is not None and not is_linear_op(current):
                tensor_inputs = [n for n in current.all_input_nodes if n.op != "get_attr"]
                if len(tensor_inputs) != 1:
                    # Multi-input or no-input node — stop walking; the lm_head
                    # is not in this graph (common for VLMs where only the text
                    # backbone is exported and the lm_head is applied externally).
                    current = None
                    break
                current = tensor_inputs[0]

            if current is not None and is_linear_op(current):
                node_to_gather = current.all_input_nodes[0]
                self._log_info(
                    f"Found LM head linear through post-processing chain: {current.name}"
                )
            else:
                node_to_gather = lm_head_node
                self._log_info(
                    f"lm_head linear not in graph; inserting gather before "
                    f"output node ({lm_head_node.name})"
                )

        # Add logits_gather_mask as input in the graph and the sequence info interface
        logits_gather_indices_node = self._add_or_retrieve_input(gm, cm, "token_gather_indices")
        batch_info_host_node = self._add_or_retrieve_input(gm, cm, "batch_info_host")

        with gm.graph.inserting_after(node_to_gather):
            gathered_node = gm.graph.call_function(
                torch.ops.auto_deploy.gather_tokens.default,
                args=(node_to_gather, logits_gather_indices_node, batch_info_host_node),
            )
        node_to_gather.replace_all_uses_with(gathered_node)
        gathered_node.replace_input_with(gathered_node, node_to_gather)

        return gm, TransformInfo(skipped=False, num_matches=1)
