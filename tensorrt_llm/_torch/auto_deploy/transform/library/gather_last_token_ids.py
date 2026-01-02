# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import add_graph_input
from ...utils.logger import ad_logger
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


@TransformRegistry.register("gather_last_token_ids")
class GatherLastTokenIds(BaseTransform):
    def _find_last_linear_simple(self, gm: GraphModule) -> Node:
        """Find the final torch_linear_simple node that produces logits.
        This is the last matmul operation in the graph
        that produces the vocabulary logits.
        """
        graph = gm.graph

        # Look for the output node and trace back to find MatMul
        output_node = graph.find_nodes(op="output")[0]

        # Look for the last torch_linear_simple node
        linear_simple_nodes = graph.find_nodes(
            op="call_function", target=torch.ops.auto_deploy.torch_linear_simple.default, sort=True
        )
        assert len(linear_simple_nodes) > 0, "No linear simple nodes found"
        last_linear_simple_node = linear_simple_nodes[-1]

        # Verify that the last linear simple node is the one producing the logits
        if last_linear_simple_node not in output_node.args[0]:
            ad_logger.warning(
                f"Last linear simple node {last_linear_simple_node.name} is not the one producing the logits"
            )
            return None

        return last_linear_simple_node

    def _add_last_token_ids_input(self, gm: GraphModule) -> Node:
        """Add last_token_ids as a graph input.

        Shape: int64[batch_size, num_selected_tokens]
        """
        graph = gm.graph

        # Find token_ids or input_ids placeholder to get batch_size dimension
        token_ids_node = None
        for node in graph.nodes:
            if node.op == "placeholder" and (
                "token" in node.name.lower() or "input_ids" in node.name
            ):
                token_ids_node = node
                break

        if token_ids_node is None:
            # Fallback: use first placeholder
            token_ids_node = graph.find_nodes(op="placeholder", sort=True)[0]

        ad_logger.info(f"Using {token_ids_node.name} to extract batch_size dimension")

        # Get symbolic batch_size dimension
        token_ids_meta = token_ids_node.meta.get("val")
        assert token_ids_meta is not None, "token_ids_meta is None"
        batch_size_dim = token_ids_meta.size(0)
        ad_logger.info(f"Extracted batch_size={batch_size_dim}")

        # Add last_token_ids placeholder: int64[batch_size]
        num_selected_tokens = 2
        input_shape = (batch_size_dim, num_selected_tokens)
        last_token_ids_example = torch.zeros(input_shape, dtype=torch.int64, device="meta")
        last_token_ids_node = add_graph_input(gm, name="last_token_ids", val=last_token_ids_example)

        ad_logger.info(f"Added last_token_ids placeholder: {last_token_ids_node.name}")

        return last_token_ids_node

    def _insert_gather_nd(
        self,
        gm: GraphModule,
        linear_simple: Node,
        last_token_ids_node: Node,
    ) -> bool:
        """Insert GatherND operation before MatMul.

        The pattern to create:
        1. Create GatherND with the linear input and last_token_ids
        2. Replace MatMul input with GatherND output
        """
        graph = gm.graph

        # Get the input to linear_simple (should be from the previous layer)
        linear_input = linear_simple.args[0]

        ad_logger.info(
            f"Linear input: {linear_input.name if hasattr(linear_input, 'name') else linear_input}"
        )

        # 1. Insert GatherND operation before linear_simple
        # GatherND takes the hidden states (linear_input) and gathers based on last_token_ids
        with graph.inserting_before(linear_simple):
            unsqueeze_node = graph.call_function(
                torch.ops.aten.unsqueeze.default, args=(last_token_ids_node, -1)
            )
            gather_nd_node = graph.call_function(
                torch.ops.auto_deploy.torch_onnx_gather_nd.default,
                args=(linear_input, unsqueeze_node, 1),  # Use linear_input, not linear_simple!
            )
            ad_logger.info(f"Created GatherND node: {gather_nd_node.name}")

        # 2. Replace linear_simple's first input with GatherND output
        linear_simple.replace_input_with(linear_input, gather_nd_node)
        ad_logger.info("Replaced Linear input with GatherND output")

        return True

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        ad_logger.info("Adding last_token_ids gather operation")

        # Step 1: Find last linear simple node
        linear_simple_node = self._find_last_linear_simple(gm)

        if linear_simple_node is None:
            ad_logger.info("Could not find last linear simple node, skipping")
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Step 2: Add last_token_ids input
        last_token_ids_node = self._add_last_token_ids_input(gm)

        # Step 3: Insert GatherND operation
        success = self._insert_gather_nd(gm, linear_simple_node, last_token_ids_node)

        if not success:
            ad_logger.info("Failed to insert GatherND operation")
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        return gm, TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)
