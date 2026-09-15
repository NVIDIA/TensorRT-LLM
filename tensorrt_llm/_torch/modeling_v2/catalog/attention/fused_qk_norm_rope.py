# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-place fused per-head QK RMS norm + RoPE on a packed QKV tensor."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    base: float,
    is_neox: bool,
    position_ids: torch.Tensor,
    factor: float = 1.0,
    low: float = 0.0,
    high: float = 0.0,
    attention_factor: float = 1.0,
    is_qk_norm: bool = True,
    use_gemma: bool = False,
    use_mrope: bool = False,
    mrope_section1: int = 0,
    mrope_section2: int = 0,
) -> None:
    """In-place on qkv: per-head RMS norm of q/k heads, then RoPE. Returns None."""
    torch.ops.trtllm.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        is_qk_norm,
        use_gemma,
        use_mrope,
        mrope_section1,
        mrope_section2,
    )
