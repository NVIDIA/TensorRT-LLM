# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Torch reference implementation for Multi-head Latent Attention (MLA).

This module provides the source op for MLA that:
- Accepts compressed_kv (before kv_b_proj) for FlashInfer-compatible caching
- Expands compressed_kv using kv_b_proj_weight for attention computation
- Computes standard attention with the expanded K, V
"""

import math
from typing import Optional

import torch


@torch.library.custom_op("auto_deploy::torch_mla", mutates_args=())
def torch_mla(
    q_nope: torch.Tensor,  # [B, S, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [B, S, N, qk_rope_head_dim] (RoPE applied)
    compressed_kv: torch.Tensor,  # [B, S, kv_lora_rank] - BEFORE kv_b_proj
    kpe: torch.Tensor,  # [B, S, 1, qk_rope_head_dim] (RoPE applied)
    kv_b_proj_weight: torch.Tensor,  # [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    is_causal: bool = True,
    scale: Optional[float] = None,
    layout: str = "bsnd",
    enable_sharding: bool = False,
    layer_type: str = "mla",
) -> torch.Tensor:
    """Multi-head Latent Attention (MLA) with FlashInfer-compatible compressed KV.

    This op expands ``compressed_kv`` with ``kv_b_proj_weight`` and computes
    standard dot-product attention. For prefill, this is the direct matmul/softmax
    formulation; a separate cached path may use weight absorption elsewhere.

    Args:
        q_nope: Query non-positional component. Shape ``[B, S, N, qk_nope_head_dim]``
            when ``layout == "bsnd"``, or ``[B, N, S, qk_nope_head_dim]`` when
            ``layout == "bnsd"``.
        q_pe: Query positional (RoPE) component. Shape ``[B, S, N, qk_rope_head_dim]``
            or ``[B, N, S, qk_rope_head_dim]`` matching ``layout``.
        compressed_kv: Compressed KV latent ``[B, S, kv_lora_rank]`` **before**
            ``kv_b_proj`` expansion.
        kpe: Key positional (RoPE) encodings ``[B, S, 1, qk_rope_head_dim]`` (or the
            ``bnsd`` transpose consistent with ``layout``).
        kv_b_proj_weight: Unpacked projection weights of shape
            ``[num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]``. This is
            argument index 4 (the fifth argument).
        is_causal: If ``True`` and ``s_q == s_k``, apply a causal upper-triangular
            mask to attention logits.
        scale: Softmax temperature; default ``1 / sqrt(qk_nope_head_dim + qk_rope_head_dim)``.
        layout: ``"bsnd"`` or ``"bnsd"`` for batch/sequence/head dimension ordering.
        enable_sharding: When ``True``, ``apply_sharding_hints`` shards ``kv_b_proj_weight``
            **column-wise along the head dimension**: the weight is treated as a
            stacked per-head projection, so each TP rank keeps the slice of rows
            corresponding to its local heads (out_features grouped by head). When
            ``False``, the hint pass does not apply that head-parallel rewrite to
            ``kv_b_proj_weight``.
        layer_type: Layer classification for selective sharding via ``shard_layers``
            config. Values: ``"mha"``, ``"mla"``, ``"mlp"``, ``"moe"``, ``"ssm"``,
            ``"delta"``, ``"unknown"``.

    Sharding hint arguments (graph-level metadata for ``apply_sharding_hints``):
        ``enable_sharding``: When ``True``, ``apply_sharding_hints`` shards ``kv_b_proj_weight``
        (arg ``kv_b_proj_weight`` / arg[4]) columnwise along the head dimension.
        ``layer_type``: Selects whether MLA nodes are rewritten for a given
        ``shard_layers`` configuration.

    Returns:
        Attention output: ``[B, S, N, v_head_dim]`` for ``bsnd``, or ``[B, N, S, v_head_dim]``
        for ``bnsd``, consistent with ``layout``.
    """
    if layout not in ("bnsd", "bsnd"):
        raise ValueError(f"layout must be 'bnsd' or 'bsnd', got {layout!r}")

    # Get dimensions
    if layout == "bsnd":
        bs, s_q, num_heads, qk_nope_head_dim = q_nope.shape
        qk_rope_head_dim = q_pe.shape[-1]
    else:
        bs, num_heads, s_q, qk_nope_head_dim = q_nope.shape
        qk_rope_head_dim = q_pe.shape[-1]

    s_k = compressed_kv.shape[1]

    # Infer dimensions from kv_b_proj_weight
    # kv_b_proj_weight: [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads  # qk_nope_head_dim + v_head_dim
    v_head_dim = kv_head_dim - qk_nope_head_dim

    # Set scale
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    if scale is None:
        scale = 1.0 / math.sqrt(qk_head_dim)

    # =========================================================================
    # Expand compressed_kv using kv_b_proj_weight (this is the prefill path)
    # =========================================================================
    # compressed_kv: [B, S, kv_lora_rank]
    # kv_b_proj_weight: [num_heads * kv_head_dim, kv_lora_rank]
    # kv = compressed_kv @ kv_b_proj_weight.T -> [B, S, num_heads * kv_head_dim]
    kv = torch.matmul(compressed_kv, kv_b_proj_weight.t())

    # Reshape to [B, S, N, kv_head_dim]
    kv = kv.view(bs, s_k, num_heads, kv_head_dim)

    # Split into k_nope and value_states
    k_nope, value_states = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)

    # k_nope and value_states are always [B, S, N, D] from the kv reshape above.
    # We need them in [B, N, S, D] for attention computation.
    k_nope = k_nope.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()

    # Convert inputs to computation layout [B, N, S, D] if they come in bsnd format
    if layout == "bsnd":
        # [B, S, N, D] -> [B, N, S, D]
        q_nope = q_nope.transpose(1, 2).contiguous()
        q_pe = q_pe.transpose(1, 2).contiguous()
        kpe = kpe.transpose(1, 2).contiguous()

    # kpe is [B, 1, S, qk_rope_head_dim], expand to num_heads
    kpe_expanded = kpe.expand(bs, num_heads, s_k, qk_rope_head_dim)

    # Construct full query and key states
    # query_states: [B, N, S, qk_head_dim]
    query_states = torch.cat([q_nope, q_pe], dim=-1)
    # key_states: [B, N, S, qk_head_dim]
    key_states = torch.cat([k_nope, kpe_expanded], dim=-1)

    # Compute attention scores: Q @ K^T
    attn_scores = (
        torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
    )  # [B, N, s_q, s_k]

    # Apply causal mask if specified
    if is_causal and s_q == s_k:
        causal_mask = torch.triu(
            torch.ones(s_q, s_k, device=q_nope.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Compute attention weights and output
    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q_nope.dtype)
    attn_out = torch.matmul(attn_weights, value_states)  # [B, N, s_q, v_head_dim]

    # Convert back to requested layout
    if layout == "bsnd":
        return attn_out.transpose(1, 2).contiguous()  # [B, S, N, v_head_dim]
    else:
        return attn_out.contiguous()  # [B, N, S, v_head_dim]


@torch_mla.register_fake
def torch_mla_fake(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    is_causal: bool = True,
    scale: Optional[float] = None,
    layout: str = "bsnd",
    enable_sharding: bool = False,
    layer_type: str = "mla",
) -> torch.Tensor:
    """Fake implementation for torch_mla."""
    # Infer v_head_dim from kv_b_proj_weight
    qk_nope_head_dim = q_nope.shape[-1]
    num_heads = q_nope.shape[2] if layout == "bsnd" else q_nope.shape[1]
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    # Output shape depends on layout
    if layout == "bsnd":
        # Input: [B, S, N, D], Output: [B, S, N, v_head_dim]
        return q_nope.new_empty(
            q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], v_head_dim
        ).contiguous()
    else:
        # Input: [B, N, S, D], Output: [B, N, S, v_head_dim]
        return q_nope.new_empty(
            q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], v_head_dim
        ).contiguous()
