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

"""Custom operations for ONNX export of attention mechanisms.

This module provides placeholder custom operations for exporting attention-related
operations to ONNX format. These operations serve as intermediate representations
during the graph transformation pipeline and are intended to be replaced by actual
backend implementations during deployment.
"""

from typing import Tuple

import torch


@torch.library.custom_op("auto_deploy::torch_onnx_attention_plugin", mutates_args=())
def attention_plugin(
    # Inputs
    qkv: torch.Tensor,
    past_key_values: torch.Tensor,
    context_lengths: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    kvcache_start_index: torch.Tensor,
    # Attributes
    enable_tree_attention: int,
    head_size: int,
    num_kv_heads: int,
    num_q_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused attention operation with integrated RoPE (Rotary Position Embedding).

    This custom operation combines rotary position embedding, and scaled
    dot-product attention into a single fused operation. It also handles
    KV-cache management for efficient autoregressive generation.

    Note:
        This is a placeholder implementation for ONNX export. The actual computation
        is performed by the backend runtime (e.g., TensorRT, EdgeLLM
    Args:
        qkv: Concatenated query, key, value tensor of shape
            [batch_size, seq_len, (num_q_heads + 2 * num_kv_heads) * head_size].
        past_key_values: KV-cache tensor of shape
            [batch_size, 2, num_kv_heads, past_seq_len, head_size].
        context_lengths: Sequence lengths for each batch element, shape [batch_size].
        rope_rotary_cos_sin: Precomputed RoPE cosine and sine values of shape
            [batch_size, max_seq_len, head_size].
        kvcache_start_index: Starting index in KV-cache for each batch element,
            shape [batch_size].
        enable_tree_attention: Flag to enable tree attention mode (0 or 1).
        head_size: Dimension of each attention head.
        num_kv_heads: Number of key-value heads (for grouped-query attention).
        num_q_heads: Number of query heads.

    Returns:
        A tuple containing:
            - attention_output: Attention output tensor of shape
              [batch_size, seq_len, num_q_heads, head_size].
            - present_key_values: Updated KV-cache tensor of shape
              [batch_size, 2, num_kv_heads, present_seq_len, head_size].
    """
    return qkv.new_empty(0), past_key_values.new_empty(0)


@attention_plugin.register_fake
def attention_plugin_fake(
    qkv: torch.Tensor,
    past_key_values: torch.Tensor,
    context_lengths: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    kvcache_start_index: torch.Tensor,
    enable_tree_attention: int,
    head_size: int,
    num_kv_heads: int,
    num_q_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation of attention_plugin for torch.compile shape inference.

    This function computes the output shapes without performing actual computation,
    enabling torch.compile to trace through the custom operation.

    Args:
        qkv: Concatenated QKV tensor.
        past_key_values: Previous KV-cache tensor.
        context_lengths: Sequence lengths per batch.
        rope_rotary_cos_sin: RoPE embedding values.
        kvcache_start_index: KV-cache start indices.
        enable_tree_attention: Tree attention flag.
        head_size: Attention head dimension.
        num_kv_heads: Number of KV heads.
        num_q_heads: Number of query heads.

    Returns:
        Tuple of empty tensors with correct shapes for attention output and
        present KV-cache.
    """
    batch_size = qkv.size(0)
    seq_len = qkv.size(1)
    past_len = past_key_values.size(3)
    present_kv_len = seq_len + past_len
    attn_shape = (batch_size, seq_len, num_q_heads, head_size)
    present_kv_shape = (batch_size, 2, num_kv_heads, present_kv_len, head_size)
    return torch.empty(attn_shape, device=qkv.device, dtype=qkv.dtype), torch.empty(
        present_kv_shape, device=past_key_values.device, dtype=past_key_values.dtype
    )


def _fake_gather_nd(data: torch.Tensor, indices: torch.Tensor, batch_dims: int) -> torch.Tensor:
    """Compute output shape for GatherND operation without actual gathering.

    This helper function creates an empty tensor with the correct output shape
    for the GatherND operation, used for both the actual op and its fake
    implementation.

    Args:
        data: Source tensor of shape [batch_size, seq_len, embedding_dim].
        indices: Index tensor of shape [batch_size, num_selected] or
            [batch_size, num_selected, index_depth].
        batch_dims: Number of leading batch dimensions (must be 1).

    Returns:
        Empty tensor with shape [batch_size, num_selected, embedding_dim].

    Raises:
        AssertionError: If batch_dims != 1, data is not 3D, or indices is not 2D/3D.
    """
    assert batch_dims == 1, "Current only support batch_dims = 1"
    assert data.ndim == 3, "Current only support 3D data tensor"
    assert indices.ndim == 2 or indices.ndim == 3, "Current only support 2D or 3D indices tensor"

    dim_batch_size = indices.size(0)
    dim_selected_token = indices.size(1)
    dim_emb = data.size(-1)
    result_shape = [dim_batch_size, dim_selected_token, dim_emb]
    return torch.empty(result_shape, device=data.device, dtype=data.dtype)


@torch.library.custom_op("auto_deploy::torch_onnx_gather_nd", mutates_args=())
def gather_nd(
    data: torch.Tensor,
    indices: torch.Tensor,
    batch_dims: int,
) -> torch.Tensor:
    """N-dimensional gather operation following ONNX gather_nd semantics.

    Gathers slices from the data tensor based on indices, supporting batched
    operations. This operation is commonly used for selecting specific tokens
    from a sequence based on their positions.

    Note:
        This is a placeholder implementation for ONNX export. The actual
        computation is performed by the backend runtime.

    Args:
        data: Source tensor to gather from, shape [batch_size, seq_len, embedding_dim].
        indices: Index tensor specifying which elements to gather,
            shape [batch_size, num_selected] or [batch_size, num_selected, index_depth].
        batch_dims: Number of leading dimensions to treat as batch dimensions.
            Currently only batch_dims=1 is supported.

    Returns:
        Gathered tensor of shape [batch_size, num_selected, embedding_dim].
    """
    return _fake_gather_nd(data, indices, batch_dims)


@gather_nd.register_fake
def gather_nd_fake(
    data: torch.Tensor,
    indices: torch.Tensor,
    batch_dims: int,
) -> torch.Tensor:
    """Fake implementation of gather_nd for torch.compile shape inference.

    Args:
        data: Source tensor to gather from.
        indices: Index tensor for gathering.
        batch_dims: Number of batch dimensions.

    Returns:
        Empty tensor with the correct output shape.
    """
    return _fake_gather_nd(data, indices, batch_dims)
