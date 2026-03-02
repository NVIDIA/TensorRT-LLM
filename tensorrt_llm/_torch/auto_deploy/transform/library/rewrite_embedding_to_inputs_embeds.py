# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from pathlib import Path
from typing import Optional, Tuple

import safetensors.torch
import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import add_graph_input, remove_graph_input
from ...utils.logger import ad_logger
from ...utils.node_utils import get_weight_tensor
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class RewriteEmbeddingToInputsEmbedsConfig(TransformConfig):
    """Configuration for the rewrite embedding to inputs_embeds transform."""

    output_dir: Path = Field(
        description="The directory to save the exported embedding weights.",
        default=Path("."),
    )


@TransformRegistry.register("rewrite_embedding_to_inputs_embeds")
class RewriteEmbeddingToInputsEmbeds(BaseTransform):
    """Transform that rewrites the graph to accept inputs_embeds instead of input_ids.

    NOTE(yoco) This is a temporary solution. We will change this after EdgeLLM
    and TensorRT-LLM align on how to handle multimodal models.

    This transform performs the following operations:
    1. Detects the pattern: input_ids (placeholder) → embedding(weight, input_ids)
    2. Extracts the embedding weight tensor
    3. Exports the embedding weight to safetensors format
    4. Removes the embedding node from the graph
    5. Replaces the input_ids placeholder with inputs_embeds placeholder

    This is necessary for EdgeLLM to support multimodal models where the embedding
    lookup is performed at runtime before passing to the TensorRT engine.
    """

    config: RewriteEmbeddingToInputsEmbedsConfig

    @classmethod
    def get_config_class(cls):
        """Return the configuration class for this transform."""
        return RewriteEmbeddingToInputsEmbedsConfig

    def _find_embedding_pattern(self, gm: GraphModule) -> Optional[Tuple[Node, Node, torch.Tensor]]:
        """Find input_ids → embedding(weight, input_ids) pattern.

        Returns:
            (input_ids_node, embedding_node, weight_tensor) or None if pattern not found
        """
        graph = gm.graph

        # Find input_ids placeholder
        placeholder_nodes = graph.find_nodes(op="placeholder")
        input_ids_node = next(
            (node for node in placeholder_nodes if node.target == "input_ids"), None
        )

        if input_ids_node is None:
            ad_logger.info("No input_ids placeholder found, skipping transform")
            return None

        # Find embedding node that uses input_ids
        embedding_node = None
        for user in input_ids_node.users:
            if (
                user.op == "call_function"
                and user.target == torch.ops.aten.embedding.default
                and user.args[1] == input_ids_node
            ):
                embedding_node = user
                break

        if embedding_node is None:
            ad_logger.warning("input_ids placeholder found but no embedding node uses it directly")
            return None

        extra_users = [
            u
            for u in input_ids_node.users
            if u is not embedding_node and u.target != torch.ops.aten.sym_size.int
        ]
        if extra_users:
            ad_logger.warning(
                "input_ids has non-embedding consumers; skipping rewrite to avoid breaking the graph"
            )
            return None

        # Extract weight tensor
        weight_tensor = get_weight_tensor(embedding_node)
        ad_logger.info(
            f"Found embedding pattern: {input_ids_node.name} → {embedding_node.name}, "
            f"weight shape: {weight_tensor.shape}"
        )
        return (input_ids_node, embedding_node, weight_tensor)

    def _export_embedding_weights(self, weight: torch.Tensor, output_path: Path):
        """Export embedding weight to safetensors format.

        The weight tensor should already be float16 after adapt_to_edgellm transform.

        Args:
            weight: The embedding weight tensor (should be float16)
            output_path: Path to save the safetensors file
        """
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure the weight is on CPU and contiguous
        weight_cpu = weight.detach().cpu().contiguous()

        # Verify that the weight is already float16
        if weight_cpu.dtype != torch.float16:
            raise ValueError(
                f"Embedding weight must be float16 (should be converted by adapt_to_edgellm), "
                f"got {weight_cpu.dtype}"
            )

        # Create state dict
        state_dict = {"weight": weight_cpu}

        # Save to safetensors
        safetensors.torch.save_file(state_dict, output_path)
        ad_logger.info(
            f"Exported embedding weights to {output_path} with dtype {weight_cpu.dtype} "
            f"and shape {weight_cpu.shape}"
        )

    def _replace_with_inputs_embeds(
        self,
        gm: GraphModule,
        input_ids_node: Node,
        embedding_node: Node,
    ):
        """Replace input_ids placeholder with inputs_embeds and remove embedding node.

        Args:
            gm: The GraphModule to modify
            input_ids_node: The input_ids placeholder node
            embedding_node: The embedding node to remove
        """
        graph = gm.graph

        # Get the embedding output meta, which has the correct shape and symbolic dimensions
        # embedding_node output is [batch_size, seq_len, hidden_size] with symbolic dimensions
        embedding_output_meta = embedding_node.meta.get("val")
        if embedding_output_meta is None:
            raise ValueError("embedding_node has no meta['val']")

        ad_logger.info(
            f"Creating inputs_embeds placeholder with shape {embedding_output_meta.shape} "
            f"on device {embedding_output_meta.device}"
        )

        # Use the embedding output meta as template for inputs_embeds
        # This preserves symbolic dimensions and device
        # The dtype should already be float16 after adapt_to_edgellm transform
        if embedding_output_meta.dtype != torch.float16:
            raise ValueError(
                f"Embedding output must be float16 (should be converted by adapt_to_edgellm), "
                f"got {embedding_output_meta.dtype}"
            )

        # Add new inputs_embeds placeholder using add_graph_input utility
        # We pass the embedding output meta directly to preserve symbolic dimensions
        inputs_embeds_node = add_graph_input(
            gm, name="inputs_embeds", val=embedding_output_meta, add_kwargs=True
        )

        # Verify the created placeholder has correct dtype
        if inputs_embeds_node.meta.get("val") is None:
            raise RuntimeError("inputs_embeds node should have meta['val']")
        if inputs_embeds_node.meta["val"].dtype != torch.float16:
            raise ValueError(
                f"inputs_embeds dtype should be float16, got {inputs_embeds_node.meta['val'].dtype}"
            )

        ad_logger.info(f"Created inputs_embeds placeholder: {inputs_embeds_node.name}")

        # Replace all uses of embedding_node with inputs_embeds_node
        embedding_node.replace_all_uses_with(inputs_embeds_node)
        ad_logger.info(f"Replaced {embedding_node.name} with {inputs_embeds_node.name}")

        # Remove the embedding node from graph
        graph.erase_node(embedding_node)
        ad_logger.info(f"Removed {embedding_node.name} from graph")

        # Handle input_ids users (e.g., sym_size operations for shape extraction)
        # Replace them with equivalent operations on inputs_embeds
        input_ids_users = list(input_ids_node.users.keys())
        for user in input_ids_users:
            if user.target == torch.ops.aten.sym_size.int:
                # This is extracting a dimension from input_ids
                # We need to replace it with the same dimension from inputs_embeds
                dim_index = user.args[1]  # The dimension being extracted
                with graph.inserting_after(inputs_embeds_node):
                    new_sym_size = graph.call_function(
                        torch.ops.aten.sym_size.int,
                        args=(inputs_embeds_node, dim_index),
                    )
                user.replace_all_uses_with(new_sym_size)
                graph.erase_node(user)
                ad_logger.info(
                    f"Replaced {user.name} (sym_size on input_ids dim {dim_index}) "
                    f"with sym_size on inputs_embeds"
                )

        # Now we can safely remove input_ids node using remove_graph_input
        # This properly maintains pytree_info and other metadata
        removed_name = remove_graph_input(gm, input_ids_node)
        ad_logger.info(f"Removed {removed_name} from graph")

    def _apply(
        self,
        gm: GraphModule,
        _cm: CachedSequenceInterface,
        _factory: ModelFactory,
        _shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        """Apply the transform to rewrite embedding to inputs_embeds.

        Args:
            gm: The GraphModule to transform
            cm: CachedSequenceInterface (unused)
            factory: ModelFactory to get model config
            shared_config: SharedConfig (unused)

        Returns:
            Tuple of (modified GraphModule, TransformInfo)
        """
        ad_logger.info("Rewriting embedding to inputs_embeds for EdgeLLM")

        # Step 1: Find the embedding pattern
        pattern_result = self._find_embedding_pattern(gm)

        if pattern_result is None:
            ad_logger.info("Embedding pattern not found, skipping transform")
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        input_ids_node, embedding_node, weight_tensor = pattern_result

        # Step 2: Export embedding weights to safetensors
        output_path = self.config.output_dir / "embedding.safetensors"
        self._export_embedding_weights(weight_tensor, output_path)

        # Step 3: Replace input_ids with inputs_embeds and remove embedding node
        self._replace_with_inputs_embeds(gm, input_ids_node, embedding_node)

        ad_logger.info("Successfully rewrote graph to use inputs_embeds")

        gm.recompile()
        return gm, TransformInfo(
            skipped=False, num_matches=1, is_clean=False, has_valid_shapes=False
        )
