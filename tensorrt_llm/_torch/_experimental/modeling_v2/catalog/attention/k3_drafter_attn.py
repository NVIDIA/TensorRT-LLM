# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DSpark drafter block attention: each request's draft block attends densely to its paged context and to the
block's own K / V."""

import torch

import tensorrt_llm._torch.cute_dsl_kernels.k3_drafter.op  # noqa: F401 — registers torch.ops.trtllm.k3_drafter_attn


def k3_drafter_attn(
    qkv: torch.Tensor,
    cache: torch.Tensor,
    page_table: torch.Tensor,
    ctx_len: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    out: torch.Tensor,
) -> None:
    """Write ``out`` [R T, num_heads * 64]: request r's T rows of ``qkv`` attend to its ``ctx_len[r]`` cached rows
    (pages: row r of ``page_table``) and to its block's own k / v in ``qkv``. Returns None."""
    torch.ops.trtllm.k3_drafter_attn(qkv, cache, page_table, ctx_len, num_heads, num_kv_heads, out)
