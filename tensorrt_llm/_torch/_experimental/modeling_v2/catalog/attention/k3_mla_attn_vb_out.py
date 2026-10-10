# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's MLA decode attention over the paged latent cache, with v_b and the output gate in the same launch, over
a caller-owned :class:`K3MlaAttnWorkspace`; also the plain forms (the attention output itself)."""

from typing import Optional

import torch

from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import (  # noqa: F401 — registers the ops
    op as _k3_mla_op,
)

from .k3_mla_attn_workspace import K3MlaAttnWorkspace


def k3_mla_attn_vb_out(
    q: torch.Tensor,
    pool: torch.Tensor,
    row_stride: int,
    page_table: torch.Tensor,
    page_offset: int,
    seq_len: torch.Tensor,
    softmax_scale: float,
    w_vb: torch.Tensor,
    out: torch.Tensor,
    workspace: K3MlaAttnWorkspace,
    gate: Optional[torch.Tensor] = None,
    gate_col0: int = 0,
) -> None:
    """Write ``out`` [M, heads * 128] = per head ``bf16(bf16(o) @ w_vb[h]^T)`` (times the gate's sigmoid with
    ``gate``) for the decode attention o of R <= 8 requests of T <= 8 tokens over their pages of ``pool``. Uses
    ``workspace``'s partial slots and, past 7 clusters, advances its counters."""
    torch.ops.trtllm.k3_mla_attn_vb_out(
        q,
        pool,
        row_stride,
        page_table,
        page_offset,
        seq_len,
        softmax_scale,
        w_vb,
        out,
        workspace.buffer,
        gate,
        gate_col0,
    )


def k3_mla_attn_out(
    q: torch.Tensor,
    pool: torch.Tensor,
    row_stride: int,
    page_table: torch.Tensor,
    page_offset: int,
    seq_len: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor,
    workspace: K3MlaAttnWorkspace,
) -> None:
    """Write the attention output itself, ``out`` [M, heads * 512] bf16 (no v_b), over ``workspace``."""
    torch.ops.trtllm.k3_mla_attn_out(
        q, pool, row_stride, page_table, page_offset, seq_len, softmax_scale, out, workspace.buffer
    )


def k3_mla_attn(
    q: torch.Tensor,
    pool: torch.Tensor,
    row_stride: int,
    page_table: torch.Tensor,
    seq_len: torch.Tensor,
    softmax_scale: float,
    workspace: K3MlaAttnWorkspace,
) -> torch.Tensor:
    """Return the attention output [M, heads * 512] bf16 in a new tensor (page offset 0), over ``workspace``."""
    return torch.ops.trtllm.k3_mla_attn(
        q, pool, row_stride, page_table, seq_len, softmax_scale, workspace.buffer
    )
