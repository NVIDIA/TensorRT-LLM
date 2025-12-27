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

"""Gemma3 + FlashInfer custom mask generator.

This module provides the mask generator for Gemma3 VLM models with FlashInfer backend.
It handles:
- Bidirectional attention for image tokens
- Causal attention for text tokens
- FlashInfer's flattened mask format

Key design:
- Generator activates required args in SequenceInfo at transform time
- Generator adds graph inputs (same mechanism as cu_seqlen)
- Executor provides args from SequenceInfo.named_args at runtime
- Mask op receives args as function arguments
"""

from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.fx import GraphModule, Node

from ...utils._graph import add_graph_input
from ..custom_mask_registry import CustomMaskGeneratorRegistry

# =============================================================================
# Custom op for Gemma3 + FlashInfer mask computation
# =============================================================================


def _get_context_mask_with_bidir_images(image_token_mask: Tensor) -> Tensor:
    """Generate attention mask for a single context sequence (Gemma3 style).

    Args:
        image_token_mask: Boolean tensor of shape [seq_len] where True = image token.

    Returns:
        Boolean mask of shape [seq_len, seq_len] where True = attention allowed.
        The mask is causal (lower triangular) with bidirectional override for
        image-image token pairs.
    """
    seq_len = image_token_mask.shape[0]
    device = image_token_mask.device

    # Base causal mask: lower triangular (query can attend to key if key_pos <= query_pos)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    # Image-image bidirectional: if both query and key are image tokens, allow attention
    is_image_q = image_token_mask.unsqueeze(1)  # [seq_len, 1]
    is_image_k = image_token_mask.unsqueeze(0)  # [1, seq_len]
    bidir_image = is_image_q & is_image_k  # [seq_len, seq_len]

    # Override causal restriction for image-image pairs
    mask = mask | bidir_image

    return mask


@torch.library.custom_op("auto_deploy::gemma3_flashinfer_mask", mutates_args=())
def gemma3_flashinfer_mask(
    _ad_token_type_ids: Tensor,
    cu_seqlen: Tensor,
    seq_len: Tensor,
    sliding_window: int,
) -> Tensor:
    """Compute Gemma3 attention mask in FlashInfer format.

    This op creates a flattened boolean mask suitable for FlashInfer's
    BatchPrefillWithPagedKVCacheWrapper custom_mask parameter.

    Args:
        _ad_token_type_ids: Token type IDs [total_tokens] where 1 = image token.
            Uses _ad_ prefix to bypass outer VLM model kwargs consumption.
        cu_seqlen: Cumulative sequence lengths [num_seqs + 1].
        seq_len: Sequence lengths [num_seqs].
        sliding_window: Sliding window size (-1 = FlashInfer handles it natively).

    Returns:
        Flattened boolean mask for attention layers.
        Shape: [sum(seq_len[i]^2) for context sequences]
    """
    device = _ad_token_type_ids.device

    # Convert _ad_token_type_ids to boolean image mask
    image_token_mask = _ad_token_type_ids.flatten() == 1

    # Identify context sequences (seq_len > 1)
    num_contexts = (seq_len > 1).sum().item()

    if num_contexts == 0:
        return torch.empty(0, dtype=torch.bool, device=device)

    masks: List[Tensor] = []
    cu_seqlen_ctx = cu_seqlen[: num_contexts + 1]

    for i in range(num_contexts):
        start = cu_seqlen_ctx[i].item()
        end = cu_seqlen_ctx[i + 1].item()

        # Extract image token mask for this sequence
        seq_image_mask = image_token_mask[start:end]

        # Generate Gemma3-style mask (causal + bidirectional for images)
        mask_i = _get_context_mask_with_bidir_images(seq_image_mask)

        masks.append(mask_i.flatten())

    return torch.cat(masks).contiguous()


@gemma3_flashinfer_mask.register_fake
def gemma3_flashinfer_mask_fake(
    _ad_token_type_ids: Tensor,
    cu_seqlen: Tensor,
    seq_len: Tensor,
    sliding_window: int,
) -> Tensor:
    """Fake implementation for tracing."""
    # Return empty tensor - actual size depends on runtime batch
    return torch.empty(0, dtype=torch.bool, device=_ad_token_type_ids.device)


# =============================================================================
# Generator function - registered in CustomMaskGeneratorRegistry
# =============================================================================


def _get_or_add_graph_input(
    gm: GraphModule, cm: Any, name: str, skip_activate: bool = False
) -> Node:
    """Get existing graph input or add it at transform time.

    This is the same mechanism used for cu_seqlen and other metadata:
    1. Activate the arg in SequenceInfo (unless skipped)
    2. Add as graph input placeholder
    3. Executor provides it from SequenceInfo.named_args at runtime

    Args:
        gm: The GraphModule being transformed.
        cm: CachedSequenceInterface for activating args.
        name: Name of the input to get/add.
        skip_activate: If True, skip activate_arg. Use for inputs that come
            via _extra_args (like _ad_token_type_ids) rather than _input_buffer.

    Returns:
        The placeholder node for the input.
    """
    # Check if placeholder already exists
    for node in gm.graph.nodes:
        if node.op == "placeholder" and node.target == name:
            return node

    # Activate the arg in SequenceInfo (unless skipped for extra args)
    if not skip_activate:
        cm.info.activate_arg(name)

    # Add as graph input (same as cu_seqlen, seq_len, etc.)
    return add_graph_input(gm, name)


@CustomMaskGeneratorRegistry.register("gemma3_text", "flashinfer")
def gemma3_flashinfer_generator(
    gm: GraphModule,
    cm: Any,  # CachedSequenceInterface
    layer_idx: int,
    metadata: Dict[str, Any],
    attn_descriptor: Any,
    meta_nodes_std: List[Node],
) -> Node:
    """Generate mask for Gemma3 + FlashInfer.

    This generator uses the same mechanism as cu_seqlen for getting inputs:
    1. Activate args in SequenceInfo
    2. Add graph inputs
    3. Executor provides them from SequenceInfo.named_args at runtime

    Args:
        gm: The GraphModule being transformed.
        cm: CachedSequenceInterface for activating args.
        layer_idx: Layer index (from marker).
        metadata: Layer metadata dict (from marker). Contains sliding_window.
        attn_descriptor: Attention descriptor for backend info.
        meta_nodes_std: Standard metadata nodes (includes cu_seqlen at index 1).

    Returns:
        The mask node inserted into the graph.
    """
    sliding_window = metadata.get("sliding_window", -1)

    # === Cache masks by sliding_window to avoid duplicates ===
    # All layers with same sliding_window can share the same mask
    cache_key = f"_gemma3_mask_sw_{sliding_window}"
    if hasattr(gm, cache_key):
        return getattr(gm, cache_key)

    # === Get/add required inputs (same mechanism as cu_seqlen) ===

    # _ad_token_type_ids - comes via _extra_args, skip activate
    # Uses _ad_ prefix to bypass outer VLM model (e.g., Gemma3Model) kwargs consumption
    _ad_token_type_ids_node = _get_or_add_graph_input(
        gm, cm, "_ad_token_type_ids", skip_activate=True
    )

    # cu_seqlen - from FlashInfer standard metadata (index 1)
    cu_seqlen_node = meta_nodes_std[1]

    # seq_len - activated and added as graph input
    seq_len_node = _get_or_add_graph_input(gm, cm, "seq_len")

    # === Insert mask op ===
    # Find insertion point - after last placeholder
    last_placeholder = None
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            last_placeholder = node

    with gm.graph.inserting_after(last_placeholder):
        mask_node = gm.graph.call_function(
            torch.ops.auto_deploy.gemma3_flashinfer_mask,
            args=(_ad_token_type_ids_node, cu_seqlen_node, seq_len_node, sliding_window),
        )

    # Cache for reuse by other layers with same sliding_window
    setattr(gm, cache_key, mask_node)

    return mask_node
