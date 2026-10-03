# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's MLA decode query path (q_a RMSNorm, q_b projection, k_b absorption) in one launch, which also stores the
step's latent KV rows into the paged latent cache; also the query path alone and the form that stores the rows
densely."""

import torch

from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import (  # noqa: F401 — registers the ops
    op as _k3_mla_op,
)


def k3_mla_qkv(
    ag: torch.Tensor,
    w_qa: torch.Tensor,
    eps: float,
    w_qb: torch.Tensor,
    w_kb: torch.Tensor,
    w_kv: torch.Tensor,
    kv_eps: float,
    pool: torch.Tensor,
    row_stride: int,
    page_table: torch.Tensor,
    page_offset: int,
    seq_len: torch.Tensor,
    trigger_early: bool = True,
) -> torch.Tensor:
    """Return ``fused_q`` [M, heads * 576] bf16 of the M <= 64 rows of ``ag``, and store each token's cache row
    ``[bf16(rmsnorm(ag[t, 1536:2048]) * w_kv) | ag[t, 2048:2112]]`` into ``pool`` at its request's position."""
    return torch.ops.trtllm.k3_mla_qkv(
        ag,
        w_qa,
        eps,
        w_qb,
        w_kb,
        w_kv,
        kv_eps,
        pool,
        row_stride,
        page_table,
        page_offset,
        seq_len,
        trigger_early,
    )


def k3_mla_q(
    ag: torch.Tensor,
    w_qa: torch.Tensor,
    eps: float,
    w_qb: torch.Tensor,
    w_kb: torch.Tensor,
    trigger_early: bool = True,
) -> torch.Tensor:
    """Return ``fused_q`` alone (no cache row stored): the same bits as ``k3_mla_qkv``'s."""
    return torch.ops.trtllm.k3_mla_q(ag, w_qa, eps, w_qb, w_kb, trigger_early)


def k3_mla_qkv_out(
    ag: torch.Tensor,
    w_qa: torch.Tensor,
    eps: float,
    w_qb: torch.Tensor,
    w_kb: torch.Tensor,
    w_kv: torch.Tensor,
    kv_eps: float,
    kv_out: torch.Tensor,
    trigger_early: bool = True,
) -> torch.Tensor:
    """Return ``fused_q`` and store the cache rows densely into ``kv_out`` [M, 576] instead of the pool."""
    return torch.ops.trtllm.k3_mla_qkv_out(
        ag, w_qa, eps, w_qb, w_kb, w_kv, kv_eps, kv_out, trigger_early
    )
