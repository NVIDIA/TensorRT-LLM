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
from typing import List, Tuple

import torch
import transformers
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import add_graph_input, add_graph_output, remove_graph_input
from ...utils.logger import ad_logger
from ...utils.node_utils import extract_op_args, is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


class MatchResult:
    """Container for matched RoPE + attention pattern nodes.

    This class stores all the relevant nodes and metadata from a successfully
    matched RoPE (Rotary Position Embedding) + attention pattern in the graph.
    It is used to facilitate the pattern replacement during the fusion transform.

    Attributes:
        q: The original query tensor node before view/reshape.
        k: The original key tensor node before view/reshape.
        v: The original value tensor node before view/reshape.
        cos: The cosine embedding node for RoPE.
        sin: The sine embedding node for RoPE.
        attn_node: The attention operation node to be replaced.
        rope_node: The RoPE operation node to be fused.
        head_dim: Dimension of each attention head.
        num_q_heads: Number of query attention heads.
        num_kv_heads: Number of key-value attention heads (for GQA/MQA).
    """

    def __init__(
        self,
        q: Node,
        k: Node,
        v: Node,
        cos: Node,
        sin: Node,
        attn_node: Node,
        rope_node: Node,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
    ):
        """Initialize MatchResult with matched pattern nodes and metadata.

        Args:
            q: Query tensor node.
            k: Key tensor node.
            v: Value tensor node.
            cos: RoPE cosine node.
            sin: RoPE sine node.
            attn_node: Attention operation node.
            rope_node: RoPE operation node.
            head_dim: Attention head dimension.
            num_q_heads: Number of query heads.
            num_kv_heads: Number of key-value heads.
        """
        self.q = q
        self.k = k
        self.v = v
        self.cos = cos
        self.sin = sin
        self.attn_node = attn_node
        self.rope_node = rope_node
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

    def __repr__(self):
        """Return string representation of the match result."""
        return (
            f"MatchResult(q={self.q.name}, k={self.k.name}, v={self.v.name}, "
            f"cos={self.cos.name}, sin={self.sin.name}, head_dim={self.head_dim}, "
            f"num_q_heads={self.num_q_heads}, num_kv_heads={self.num_kv_heads})"
        )


@TransformRegistry.register("fuse_rope_attention")
class FuseRopeAttention(BaseTransform):
    """Transform that fuses RoPE and attention operations into a single AttentionPlugin.

    This transform identifies patterns in the graph where RoPE (Rotary Position
    Embedding) is applied to query and key tensors followed by scaled dot-product
    attention, and replaces them with a fused AttentionPlugin operation.

    The fusion provides several benefits:
        - Reduced memory bandwidth by eliminating intermediate tensors
        - Enables KV-cache integration for efficient autoregressive generation
        - Allows backend-specific optimizations (e.g., TensorRT, EdgeLLM)

    Pattern matched (backwards from attention):
        1. torch_attention(rope_q, rope_k, view_v, attn_mask, ...)
        2. rope_q = rope[1], rope_k = rope[0]
        3. rope = torch_rope_with_explicit_cos_sin(cont_q, cont_k, cos, sin, 2)
        4. cont_q = contiguous(view_q), cont_k = contiguous(view_k)
        5. view_q = view(q, ...), view_k = view(k, ...), view_v = view(v, ...)

    The transform also:
        - Adds new graph inputs: context_lengths, rope_rotary_cos_sin, kvcache_start_index
        - Adds past_key_values inputs for each attention layer
        - Adds present_key_values outputs for each attention layer
        - Removes the position_ids placeholder (no longer needed after fusion)
    """

    def _get_batch_size_and_max_seq_len(self, gm: GraphModule) -> Tuple[int, int]:
        """Get batch size and max sequence length from the graph.

        Args:
            gm: The GraphModule to get batch size and max sequence length from.

        Returns:
            Tuple of (batch_size_dim, max_seq_len_dim, batch_size_sym_node, max_seq_len_sym_node).

        Note:
            For clarity, we return the symbolic nodes as well, which can be
            used in operations like view or reshape.
        """
        graph = gm.graph
        input_ids_node = graph.find_nodes(op="placeholder", target="input_ids")[0]
        position_ids_node = graph.find_nodes(op="placeholder", target="position_ids")[0]
        input_ids_meta = input_ids_node.meta.get("val")
        batch_size_dim = input_ids_meta.size(0)
        max_seq_len_dim = input_ids_meta.size(1)

        # We scan all the sym_size.int nodes to find the symbolic nodes for
        # batch size and max sequence length. The symbolic nodes has two sources.
        # 1. The input_ids placeholder.
        # 2. The position_ids placeholder.
        # Both of their shapes are (batch_size, max_seq_len).
        sym_ints = graph.find_nodes(op="call_function", target=torch.ops.aten.sym_size.int)
        for sym_int in sym_ints:
            if sym_int.args[0] != input_ids_node and sym_int.args[0] != position_ids_node:
                continue
            if sym_int.args[1] == 0:
                batch_size_sym_node = sym_int
            elif sym_int.args[1] == 1:
                max_seq_len_sym_node = sym_int
        assert batch_size_sym_node is not None and max_seq_len_sym_node is not None

        return batch_size_dim, max_seq_len_dim, batch_size_sym_node, max_seq_len_sym_node

    def _get_config_head_dim(self, model_config: transformers.PretrainedConfig) -> int:
        """Get head dimension from model config."""
        if hasattr(model_config, "head_dim"):
            return model_config.head_dim
        else:
            return model_config.hidden_size // model_config.num_attention_heads

    def _match_rope_attention_pattern(
        self, gm: GraphModule, model_config: transformers.PretrainedConfig
    ) -> List[MatchResult]:
        """Match RoPE + attention patterns in the computation graph.

        Traverses the graph backwards from attention nodes to identify the
        complete pattern of RoPE application followed by attention computation.

        Pattern structure (backwards from attention):
            1. torch_attention(rope_q, rope_k, bind_v, attn_mask, ...)
            2. rope_q, rope_k = rope[1], rope[0]
            3. rope = torch_rope_with_explicit_cos_sin(bind_q, bind_k, cos, sin, 2)

        Args:
            gm: The GraphModule to search for patterns.

        Returns:
            List of MatchResult objects, each containing the matched nodes
            and extracted metadata (head_dim, num_q_heads, num_kv_heads).
        """
        matches = []
        graph = gm.graph
        head_dim = self._get_config_head_dim(model_config)

        # Iterate through all nodes to find attention ops
        for attn_node in graph.nodes:
            if not is_op(attn_node, torch.ops.auto_deploy.torch_attention):
                continue

            if attn_node.args[10] != "bsnd":
                ad_logger.error(
                    f"  Skipping: attention layout is not bsnd: {attn_node.kwargs.get('layout', None)}"
                )
                continue

            ad_logger.debug(f"Found attention node: {attn_node.name}")

            # Extract attention inputs: (rope_q, rope_k, v, attn_mask, ...)
            if len(attn_node.args) < 4:
                ad_logger.error(f"  Skipping: insufficient args ({len(attn_node.args)})")
                continue

            rope_q_node, rope_k_node, bind_v = extract_op_args(attn_node, "query", "key", "value")

            # Step 1: Match rope_q and rope_k as getitem[1] and getitem[0] from rope output
            if not (is_op(rope_q_node, operator.getitem) and is_op(rope_k_node, operator.getitem)):
                ad_logger.error("  Skipping: rope_q or rope_k not getitem")
                continue

            # Verify they come from the same rope node
            assert rope_q_node.target == operator.getitem
            assert rope_k_node.target == operator.getitem
            rope_node_from_q, rope_q_idx = rope_q_node.args
            rope_node_from_k, rope_k_idx = rope_k_node.args

            if rope_node_from_q != rope_node_from_k:
                ad_logger.error("  Skipping: rope_q and rope_k come from different rope nodes")
                continue

            rope_node = rope_node_from_q

            # Verify getitem indices: rope[0] = rope_k, rope[1] = rope_q
            if rope_k_idx != 0 or rope_q_idx != 1:
                ad_logger.error(
                    f"  Skipping: incorrect getitem indices (k={rope_k_idx}, q={rope_q_idx})"
                )
                continue

            # Step 2: Match the rope node
            if not is_op(rope_node, torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin):
                ad_logger.error("  Skipping: not a rope node")
                continue

            # Extract rope inputs: (cont_q, cont_k, cos, sin, 2)
            if len(rope_node.args) < 5:
                ad_logger.error(f"  Skipping: rope has insufficient args ({len(rope_node.args)})")
                continue

            bind_k = rope_node.args[0]
            bind_q = rope_node.args[1]
            cos_node = rope_node.args[2]
            sin_node = rope_node.args[3]

            num_q_heads = model_config.num_attention_heads
            num_kv_heads = model_config.num_key_value_heads

            # Successfully matched the pattern!
            match = MatchResult(
                q=bind_q,
                k=bind_k,
                v=bind_v,
                cos=cos_node,
                sin=sin_node,
                attn_node=attn_node,
                rope_node=rope_node,
                head_dim=head_dim,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
            )
            matches.append(match)
            ad_logger.debug(f"  ✓ Matched pattern: {match}")

        return matches

    def _add_global_placeholders(self, gm: GraphModule, factory: ModelFactory) -> tuple:
        """Add global input placeholders required by the fused AttentionPlugin.

        Creates three new graph inputs that are shared across all attention layers:
            - context_lengths: Actual sequence length for each batch element
            - rope_rotary_cos_sin: Precomputed RoPE embeddings
            - kvcache_start_index: Starting position in KV-cache for each batch

        These inputs enable dynamic sequence handling and efficient KV-cache
        management during autoregressive generation.

        Args:
            gm: GraphModule to add placeholders to.
            cm: CachedSequenceInterface for registering dynamic shapes.

        Returns:
            Tuple of (context_lengths_node, rope_rotary_cos_sin_node,
            kvcache_start_index_node).
        """

        graph = gm.graph

        # Find token_ids placeholder to get batch_size symbolic dimension
        token_ids_node = None
        for node in graph.nodes:
            if node.op == "placeholder" and "token" in node.name.lower():
                token_ids_node = node
                break

        if token_ids_node is None:
            # Fallback: use first placeholder
            token_ids_node = graph.find_nodes(op="placeholder", sort=True)[0]

        batch_size_dim, max_seq_len_dim, _, _ = self._get_batch_size_and_max_seq_len(gm)
        ad_logger.debug(f"Extracted batch_size={batch_size_dim}, max_seq_len={max_seq_len_dim}")

        # 1. Add context_lengths placeholder: int32[batch_size]
        context_lengths_example = torch.zeros(batch_size_dim, dtype=torch.int32, device="meta")
        context_lengths_node = add_graph_input(
            gm, name="context_lengths", val=context_lengths_example
        )
        ad_logger.debug(f"Added context_lengths placeholder: {context_lengths_node.name}")

        # 2. Add rope_rotary_cos_sin placeholder: float32[rope_batch_size, rope_max_position_length, 64]
        # Create with concrete example tensor
        model_config, _ = factory._get_model_config()
        head_dim = self._get_config_head_dim(model_config)
        rope_example = torch.zeros(
            batch_size_dim, max_seq_len_dim, head_dim, dtype=torch.float32, device="meta"
        )
        rope_rotary_cos_sin_node = add_graph_input(gm, name="rope_rotary_cos_sin", val=rope_example)
        ad_logger.debug(f"Added rope_rotary_cos_sin placeholder: {rope_rotary_cos_sin_node.name}")

        # 3. Add kvcache_start_index placeholder: int32[batch_size]
        kvcache_start_index_example = torch.zeros(batch_size_dim, dtype=torch.int32, device="meta")
        kvcache_start_index_node = add_graph_input(
            gm, name="kvcache_start_index", val=kvcache_start_index_example
        )
        ad_logger.debug(f"Added kvcache_start_index placeholder: {kvcache_start_index_node.name}")

        return context_lengths_node, rope_rotary_cos_sin_node, kvcache_start_index_node

    def _perform_replacement(
        self,
        gm: GraphModule,
        cm: "CachedSequenceInterface",
        matches: List[MatchResult],
        context_lengths_node: Node,
        rope_rotary_cos_sin_node: Node,
        kvcache_start_index_node: Node,
    ) -> int:
        """Replace matched RoPE + attention patterns with fused AttentionPlugin.

        For each matched pattern, this method:
            1. Creates a past_key_values input placeholder for KV-cache
            2. Reshape Q, K, V to (batch_size, seq_len, -1)
            3. Concatenates reshaped Q, K, V tensors into a single QKV tensor
            4. Inserts the fused AttentionPlugin operation
            5. Creates getitem nodes to extract attention output and present KV-cache
            6. Replaces the original attention node with the fused output
            7. Adds present_key_values to graph outputs

        Args:
            gm: The GraphModule being transformed.
            cm: CachedSequenceInterface for shape management.
            matches: List of matched patterns to replace.
            context_lengths_node: Shared context lengths input node.
            rope_rotary_cos_sin_node: Shared RoPE embeddings input node.
            kvcache_start_index_node: Shared KV-cache index input node.

        Returns:
            Number of patterns successfully replaced.
        """
        graph = gm.graph
        past_len = 4096  # Does not matter, this value will be replaced by symbolic dimension

        # Get batch size & max sequence length
        batch_size_dim, _, batch_size_sym_node, max_seq_len_sym_node = (
            self._get_batch_size_and_max_seq_len(gm)
        )

        # Process each match
        for match_id, match in enumerate(matches):
            ad_logger.debug(f"Processing match {match_id}: {match}")

            # 1. Create past_key_values_<id> placeholder
            # Shape: float16[batch_size, 2, num_kv_heads, past_len, head_dim]
            past_key_values_example = torch.zeros(
                batch_size_dim,
                2,
                match.num_kv_heads,
                past_len,
                match.head_dim,
                dtype=torch.float16,
                device="meta",
            )
            past_key_values_node = add_graph_input(
                gm, name=f"past_key_values_{match_id}", val=past_key_values_example
            )

            ad_logger.debug(f"Added past_key_values_{match_id} placeholder")

            # 2. Reshape Q, K, V to (batch_size, seq_len, -1)
            with graph.inserting_before(match.attn_node):
                q_node = graph.call_function(
                    torch.ops.aten.view.default,
                    args=(match.q, (batch_size_sym_node, max_seq_len_sym_node, -1)),
                )
                k_node = graph.call_function(
                    torch.ops.aten.view.default,
                    args=(match.k, (batch_size_sym_node, max_seq_len_sym_node, -1)),
                )
                v_node = graph.call_function(
                    torch.ops.aten.view.default,
                    args=(match.v, (batch_size_sym_node, max_seq_len_sym_node, -1)),
                )

            ad_logger.debug(
                f"Reshaped Q, K, V to (batch_size, seq_len, -1): {q_node.name}, {k_node.name}, {v_node.name}"
            )

            # 3. Concatenate reshaped Q, K, V to (batch_size, seq_len, -1)
            with graph.inserting_before(match.attn_node):
                qkv_node = graph.call_function(
                    torch.ops.aten.cat.default, args=([q_node, k_node, v_node], -1)
                )

            ad_logger.debug(f"Created qkv concat node: {qkv_node.name}")

            # 4. Create AttentionPlugin node
            enable_tree_attention = 0
            head_dim = match.head_dim
            num_kv_heads = match.num_kv_heads
            num_q_heads = match.num_q_heads

            with graph.inserting_before(match.attn_node):
                cast_node = graph.call_function(
                    torch.ops.aten.to.dtype, args=(qkv_node, torch.float16)
                )
                rope_attn_node = graph.call_function(
                    torch.ops.auto_deploy.torch_onnx_attention_plugin.default,
                    args=(
                        cast_node,
                        past_key_values_node,
                        context_lengths_node,
                        rope_rotary_cos_sin_node,
                        kvcache_start_index_node,
                        enable_tree_attention,
                        head_dim,
                        num_kv_heads,
                        num_q_heads,
                    ),
                )

            ad_logger.debug(f"Created AttentionPlugin node: {rope_attn_node.name}")

            # 5. Create getitem nodes for rope_attn outputs
            with graph.inserting_after(rope_attn_node):
                rope_attn_output = graph.call_function(operator.getitem, args=(rope_attn_node, 0))
                rope_attn_present_kv = graph.call_function(
                    operator.getitem,
                    args=(rope_attn_node, 1),
                    name=f"present_key_values_{match_id}",
                )

            ad_logger.debug(
                f"Created getitem nodes: {rope_attn_output.name}, {rope_attn_present_kv.name}"
            )

            # 6. Replace all uses of attn_node with rope_attn_output
            match.attn_node.replace_all_uses_with(rope_attn_output)
            ad_logger.debug(f"Replaced {match.attn_node.name} with {rope_attn_output.name}")

            # 7. Add present_key_values_<id> to graph outputs using add_graph_output
            add_graph_output(gm, rope_attn_present_kv, f"present_key_values_{match_id}")
            ad_logger.debug(f"Added present_key_values_{match_id} to graph outputs")

        # 8. Eliminate dead code (remove old pattern nodes)
        graph.eliminate_dead_code()
        ad_logger.debug("Eliminated dead code")

        return len(matches)

    def _remove_position_ids_placeholder(self, gm: GraphModule) -> str:
        """Remove the position_ids placeholder from the graph.

        After fusing RoPE into AttentionPlugin, position_ids is no longer needed
        as an input since positional information is now provided through
        rope_rotary_cos_sin and kvcache_start_index.

        This method also updates any sym_size operations that were reading from
        position_ids to use input_ids instead.

        Args:
            gm: The GraphModule to modify.

        Returns:
            Name of the removed position_ids node.
        """
        graph = gm.graph
        position_ids_node = graph.find_nodes(op="placeholder", target="position_ids")[0]
        input_ids_node = graph.find_nodes(op="placeholder", target="input_ids")[0]

        # Get size from token_ids placeholder instead of position_ids placeholder
        sym_size_int_nodes = graph.find_nodes(
            op="call_function", target=torch.ops.aten.sym_size.int
        )
        for node in sym_size_int_nodes:
            if node.args[0] == position_ids_node:
                new_args = (input_ids_node, *node.args[1:])
                node.args = new_args

        return remove_graph_input(gm, position_ids_node)

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        ad_logger.debug("Fusing rope attention")

        # Perform pattern matching
        model_config, _ = factory._get_model_config()
        matches = self._match_rope_attention_pattern(gm, model_config)
        cnt = len(matches)

        ad_logger.info(f"Matched {cnt} rope attention patterns")

        if cnt == 0:
            ad_logger.info("No patterns matched, skipping replacement")
            info = TransformInfo(skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True)
            return gm, info

        # Add global placeholders
        context_lengths_node, rope_rotary_cos_sin_node, kvcache_start_index_node = (
            self._add_global_placeholders(gm, factory)
        )

        # Perform replacement for all matches
        num_replaced = self._perform_replacement(
            gm,
            cm,
            matches,
            context_lengths_node,
            rope_rotary_cos_sin_node,
            kvcache_start_index_node,
        )

        # Remove position_ids placeholder
        removed_node_name = self._remove_position_ids_placeholder(gm)
        ad_logger.debug(f"Removed position_ids placeholder: {removed_node_name}")

        info = TransformInfo(
            skipped=False, num_matches=num_replaced, is_clean=True, has_valid_shapes=True
        )
        ad_logger.info(f"Fused {num_replaced} rope attention patterns")

        return gm, info
