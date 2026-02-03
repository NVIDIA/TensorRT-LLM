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

from typing import List, Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


@TransformRegistry.register("short_reshape_attention_output")
class ShortReshapeAttentionOutput(BaseTransform):
    """Transform that simplifies reshape operations after attention outputs.

    This transform optimizes reshape nodes that follow AttentionPlugin outputs
    by replacing fixed shape dimensions with dynamic symbolic dimensions derived
    from the input tensor. This ensures the reshape operation correctly handles
    dynamic batch sizes and sequence lengths.

    The transform identifies patterns where:
        1. A reshape node follows an AttentionPlugin output
        2. The reshape's input can be traced back to a linear projection

    It then replaces the reshape with a new one that uses symbolic dimensions
    from the linear projection's input, resulting in the shape [batch, seq, -1].
    """

    def _lookup_ascending_node(self, node: Node, target, max_depth: int = 3) -> Node:
        """Recursively search for an ancestor node with a specific target operation.

        Traverses the graph backwards through node arguments to find a node
        that matches the specified target operation type.

        Args:
            node: Starting node for the backward search.
            target: Target operation to search for (e.g., torch.ops.aten.reshape.default).
            max_depth: Maximum number of levels to traverse (default: 3).

        Returns:
            The matching ancestor node if found, None otherwise.
        """
        if max_depth == 0:
            return None
        if node.target == target:
            return node

        # Helper function to check a single node
        def check_node(n):
            if isinstance(n, Node):
                result = self._lookup_ascending_node(n, target, max_depth - 1)
                if result is not None:
                    return result
            return None

        # Check all arguments
        for arg in node.args:
            if isinstance(arg, (tuple, list)):
                for item in arg:
                    result = check_node(item)
                    if result is not None:
                        return result
            else:
                result = check_node(arg)
                if result is not None:
                    return result
        return None

    def _find_reshape_attention_output(self, gm: GraphModule) -> List[Node]:
        """Find reshape nodes that follow AttentionPlugin outputs.

        Searches for reshape operations that:
            1. Have an AttentionPlugin in their input chain
            2. Can be traced back to a torch_linear_simple operation

        The linear operation is needed to extract the original input shape
        for constructing the new reshape with symbolic dimensions.

        Args:
            gm: The GraphModule to search.

        Returns:
            List of (reshape_node, linear_node) tuples representing the
            matched patterns.
        """
        reshape_linear_pairs = []
        reshape_nodes = gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.reshape.default
        )

        for node in reshape_nodes:
            # Looking for AttentionPlugin from the input of the reshape node
            attention_plugin_node = self._lookup_ascending_node(
                node.args[0], torch.ops.auto_deploy.torch_onnx_attention_plugin.default
            )
            if attention_plugin_node is None:
                continue

            # Looking for torch_linear_simple from the input of the AttentionPlugin node
            linear_simple_node = self._lookup_ascending_node(
                attention_plugin_node.args[0], torch.ops.auto_deploy.torch_linear_simple.default
            )
            if linear_simple_node is None:
                continue

            reshape_linear_pairs.append((node, linear_simple_node))

        return reshape_linear_pairs

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        num_matches = 0
        reshape_linear_pairs = self._find_reshape_attention_output(gm)
        self._log_info(f"Found {len(reshape_linear_pairs)} reshape_linear_pairs")

        graph = gm.graph
        for reshape_node, linear_simple_node in reshape_linear_pairs:
            # Get the input tensor of reshape node (keep it unchanged)
            shape_src = linear_simple_node.args[0]
            data_input = reshape_node.args[0]

            # Extract first two dimensions from the input tensor
            with graph.inserting_before(reshape_node):
                dim0 = graph.call_function(torch.ops.aten.sym_size.int, args=(shape_src, 0))
                dim1 = graph.call_function(torch.ops.aten.sym_size.int, args=(shape_src, 1))

                # Create new shape using input tensor's first two dimensions
                new_shape = [dim0, dim1, -1]

                # Create a NEW reshape node instead of modifying the existing one
                # This ensures the graph dependencies are correctly established
                new_reshape_node = graph.call_function(
                    torch.ops.aten.reshape.default, args=(data_input, new_shape)
                )

            # Replace all uses of the old reshape node with the new one
            reshape_node.replace_all_uses_with(new_reshape_node)

            # Remove the old reshape node from the graph
            graph.erase_node(reshape_node)

            num_matches += 1

        info = TransformInfo(
            skipped=False, num_matches=num_matches, is_clean=False, has_valid_shapes=False
        )
        return gm, info
