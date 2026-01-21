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

import operator
from typing import Tuple

import torch
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import run_shape_prop
from ...utils.logger import ad_logger
from ...utils.node_utils import is_op, sync_weight_meta_dtype
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


@TransformRegistry.register("adapt_to_edgellm")
class AdaptToEdgeLLM(BaseTransform):
    """Transform that adapts the model graph for EdgeLLM deployment.

    This transform performs several modifications to make the model compatible
    with EdgeLLM runtime requirements:

        1. Converts all model weights to float16 precision
        2. Adds float32 cast after the final linear layer (for logits output)
        3. Inserts float16 casts after attention output reshapes
        4. Changes any bfloat16 casts to float16 (EdgeLLM may not support bfloat16)

    These modifications ensure proper data type handling throughout the model
    while maintaining numerical precision where needed (e.g., logits output).
    """

    def _add_cast_after_last_linear(self, gm: GraphModule) -> torch.fx.Node:
        """Add a float32 cast operation after the final linear layer.

        The final linear layer produces logits which are typically kept in
        float32 for numerical stability during softmax and sampling operations.

        Args:
            gm: The GraphModule to modify.

        Returns:
            The newly created cast node.
        """
        graph = gm.graph
        linear_nodes = graph.find_nodes(
            op="call_function", target=torch.ops.auto_deploy.torch_linear_simple.default, sort=True
        )
        assert len(linear_nodes) > 0, "No linear nodes found"
        last_linear_node = linear_nodes[-1]
        with graph.inserting_after(last_linear_node):
            cast_node = graph.call_function(
                torch.ops.aten.to.dtype, args=(last_linear_node, torch.float32)
            )
        last_linear_node.replace_all_uses_with(cast_node)
        # Restore cast_node's input to last_linear_node
        cast_node.update_arg(0, last_linear_node)
        return cast_node

    def _insert_cast_after_attn_reshape(self, gm: GraphModule) -> int:
        """Insert float16 cast after reshape nodes following AttentionPlugin output.

        The AttentionPlugin may output tensors that need explicit casting to float16
        before being consumed by subsequent operations (e.g., linear projections).
        This ensures consistent data types throughout the attention block.

        Graph transformation:
            Before: AttentionPlugin[0] -> Reshape -> MatMul
            After:  AttentionPlugin[0] -> Reshape -> Cast(to float16) -> MatMul

        Args:
            gm: The GraphModule to modify.

        Returns:
            Number of cast nodes inserted.
        """
        graph = gm.graph

        # Find all AttentionPlugin nodes
        attention_plugin_nodes = graph.find_nodes(
            op="call_function", target=torch.ops.auto_deploy.torch_onnx_attention_plugin.default
        )

        num_inserted = 0
        for attn_node in attention_plugin_nodes:
            # Find getitem[0] for this AttentionPlugin (first output)
            for user in attn_node.users:
                if is_op(user, operator.getitem) and user.args[1] == 0:
                    getitem_0_node = user
                    # Find reshape nodes that use this getitem[0]
                    for reshape_user in list(getitem_0_node.users):
                        if is_op(reshape_user, torch.ops.aten.reshape.default):
                            reshape_node = reshape_user
                            # Insert cast (to float16) after reshape
                            with graph.inserting_after(reshape_node):
                                cast_node = graph.call_function(
                                    torch.ops.aten.to.dtype,
                                    args=(reshape_node, torch.float16),
                                )
                            reshape_node.replace_all_uses_with(cast_node)
                            # Fix: restore cast_node's input to reshape_node
                            # (replace_all_uses_with also replaced it)
                            cast_node.update_arg(0, reshape_node)
                            num_inserted += 1
                            ad_logger.debug(f"Inserted cast (to float16) after {reshape_node.name}")

        return num_inserted

    def _change_cast_bfloat16_to_float16(self, gm: GraphModule) -> int:
        """Replace all bfloat16 cast operations with float16 casts.

        EdgeLLM or certain hardware backends may not support bfloat16 natively.
        This method converts all bfloat16 casts to float16 for compatibility.

        Args:
            gm: The GraphModule to modify.

        Returns:
            Number of cast operations changed.
        """
        graph = gm.graph
        cast_nodes = graph.find_nodes(op="call_function", target=torch.ops.aten.to.dtype)
        num_changed = 0
        for cast_node in cast_nodes:
            if cast_node.args[1] == torch.bfloat16:
                cast_node.update_arg(1, torch.float16)
                num_changed += 1
        return num_changed

    def _to_float16(self, gm: GraphModule) -> int:
        """Convert floating point type parameters and buffers to float16
        precision.

        This preserves lower precision dtypes (e.g., float8_e4m3fn for
        quantized weights) while converting higher precision dtypes (e.g.,
        bfloat16, float32, float64) to float16 for EdgeLLM compatibility.

        Args:
            gm: The GraphModule to convert.

        Returns:
            Number of tensors converted from higher precision dtypes to
            float16.
        """
        num_converted = 0
        dtypes = [torch.float32, torch.float64, torch.bfloat16]
        for _name, param in gm.named_parameters():
            if param.dtype in dtypes:
                param.data = param.data.to(torch.float16)
                num_converted += 1
        for _name, buffer in gm.named_buffers():
            if buffer.dtype in dtypes:
                buffer.data = buffer.data.to(torch.float16)
                num_converted += 1
        return num_converted

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        """Apply EdgeLLM adaptation: convert to FP16 and insert necessary casts."""
        num_converted = self._to_float16(gm)
        num_synced = sync_weight_meta_dtype(gm)
        ad_logger.info(f"Converted {num_converted} float32 tensors to float16")
        ad_logger.info(f"Synced {num_synced} weight meta dtypes")
        logits_cast = self._add_cast_after_last_linear(gm)
        assert logits_cast is not None, "Failed to add cast after last linear"
        num_attn_casts = self._insert_cast_after_attn_reshape(gm)
        num_bfloat16_casts = self._change_cast_bfloat16_to_float16(gm)

        # NOTE(yoco) Run shape without lift_to_meta()
        # If we run shape inference with has_valid_shapes=True,
        # it will fail because it will lift some placeholder to meta
        # and cause a device mismatch error.
        run_shape_prop(gm)

        ad_logger.info(f"Changed {num_bfloat16_casts} bfloat16 casts to float16")
        ad_logger.info(f"Adapted EdgeLLM model (inserted {num_attn_casts} attention casts)")

        return gm, TransformInfo(
            skipped=False, num_matches=num_attn_casts, is_clean=False, has_valid_shapes=False
        )
