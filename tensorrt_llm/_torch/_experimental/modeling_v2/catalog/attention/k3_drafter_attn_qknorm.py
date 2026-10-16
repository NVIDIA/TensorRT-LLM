# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DSpark drafter block attention on the raw projection output: per-head q / k RMSNorm and NeoX RoPE inside
the kernel, then k3_drafter_attn."""

import torch

import tensorrt_llm._torch.cute_dsl_kernels.k3_drafter.op  # noqa: F401 — registers torch.ops.trtllm.k3_drafter_attn_qknorm


def k3_drafter_attn_qknorm(
    qkv: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    rope_base: float,
    cache: torch.Tensor,
    page_table: torch.Tensor,
    ctx_len: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    out: torch.Tensor,
) -> None:
    """``k3_drafter_attn`` after RMSNorm (``q_w`` / ``k_w``, ``eps``) and NeoX RoPE (``rope_base``, one position per
    row of ``qkv``) of every q and k head, computed in the kernel; ``qkv`` is not modified. Returns None."""
    torch.ops.trtllm.k3_drafter_attn_qknorm(
        qkv,
        q_w,
        k_w,
        positions,
        eps,
        rope_base,
        cache,
        page_table,
        ctx_len,
        num_heads,
        num_kv_heads,
        out,
    )
