# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Shared MLA RoPE utilities for auto_deploy custom models.

Contains helper functions for RoPE weight de-interleaving,
used by DeepSeek V3 and GLM4 MoE Lite model implementations.
"""

from typing import Dict

import torch


def _rope_deinterleave_load_hook(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    *args,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    num_heads: int,
    kv_lora_rank: int,
    num_layers: int,
):
    """Pre-load hook that permutes RoPE weight columns from interleaved to non-interleaved.

    For q_b_proj: output shape is [num_heads * (qk_nope_head_dim + qk_rope_head_dim), q_lora_rank]
      -> permute the last qk_rope_head_dim columns within each head's block

    For kv_a_proj_with_mqa: output shape is [kv_lora_rank + qk_rope_head_dim, hidden_size]
      -> permute the last qk_rope_head_dim rows (the k_pe portion)
    """
    d = qk_rope_head_dim
    perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
    qk_head_dim = qk_nope_head_dim + d

    for layer_idx in range(num_layers):
        layer_prefix = f"{prefix}model.layers.{layer_idx}.self_attn."

        # --- q_b_proj.weight ---
        q_key = layer_prefix + "q_b_proj.weight"
        if q_key in state_dict:
            w = state_dict[q_key]
            w = w.view(num_heads, qk_head_dim, -1)
            w_nope = w[:, :qk_nope_head_dim, :]
            w_rope = w[:, qk_nope_head_dim:, :]
            w_rope = w_rope[:, perm, :]
            w = torch.cat([w_nope, w_rope], dim=1)
            state_dict[q_key] = w.view(-1, w.shape[-1])

        # --- kv_a_proj_with_mqa.weight ---
        kv_key = layer_prefix + "kv_a_proj_with_mqa.weight"
        if kv_key in state_dict:
            w = state_dict[kv_key]
            w_kv = w[:kv_lora_rank, :]
            w_pe = w[kv_lora_rank:, :]
            w_pe = w_pe[perm, :]
            state_dict[kv_key] = torch.cat([w_kv, w_pe], dim=0)

        # --- kv_a_proj_with_mqa.bias (if present) ---
        kv_bias_key = layer_prefix + "kv_a_proj_with_mqa.bias"
        if kv_bias_key in state_dict:
            b = state_dict[kv_bias_key]
            b_kv = b[:kv_lora_rank]
            b_pe = b[kv_lora_rank:]
            b_pe = b_pe[perm]
            state_dict[kv_bias_key] = torch.cat([b_kv, b_pe])
