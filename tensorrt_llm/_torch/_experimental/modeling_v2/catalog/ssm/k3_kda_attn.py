# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's KDA projection and speculative verify of one request's 8 tokens in one launch, over a caller-owned
:class:`K3KdaBuffers`; and ``k3_kda_qkvg``, the projection stream alone."""

import torch

from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_attn import (  # noqa: F401 — registers the ops
    op as _k3_kda_attn_op,
)
from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_verify import (  # noqa: F401 — k3_kda_attn's helpers
    op as _k3_kda_verify_op,
)

from .k3_kda_buffers import K3KdaBuffers


def k3_kda_attn(
    x: torch.Tensor,
    w: torch.Tensor,
    w_fb: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    onorm_w: torch.Tensor,
    cs_q: torch.Tensor,
    cs_k: torch.Tensor,
    cs_v: torch.Tensor,
    ssm: torch.Tensor,
    state_tok: torch.Tensor,
    slots: torch.Tensor,
    pending: torch.Tensor,
    buffers: K3KdaBuffers,
    num_spec: int,
    lower_bound: float,
    scale: float,
    eps: float,
) -> torch.Tensor:
    """Return the gated-norm core output bf16 [8, 6, 128] of one request's golden token and 7 drafts ``x`` bf16
    [8, 7168]: the projection ``x w^T`` and the verify of ``ssm/k3_kda_verify`` on it, in one launch. Updates the
    slot's conv caches, state and per-draft states in place; advances ``buffers`` by one launch."""
    return torch.ops.trtllm.k3_kda_attn(
        x,
        w,
        w_fb,
        w_q,
        w_k,
        w_v,
        a_log,
        dt_bias,
        onorm_w,
        cs_q,
        cs_k,
        cs_v,
        ssm,
        state_tok,
        slots,
        pending,
        buffers.p1,
        buffers.part,
        buffers.epoch,
        num_spec,
        lower_bound,
        scale,
        eps,
    )


def k3_kda_qkvg(x: torch.Tensor, w: torch.Tensor, buffers: K3KdaBuffers) -> None:
    """The projection ``x w^T`` of T <= 8 tokens (``x`` bf16 [T, 7168], ``w`` bf16 [3208, 7168]) published into
    ``buffers`` (made with ``ctas=CTAS``): buffer ``e = buffers.epoch[0]`` (before the call) holds the rows; advances
    ``buffers`` by one launch. Returns None."""
    torch.ops.trtllm.k3_kda_qkvg(x, w, buffers.p1, buffers.part, buffers.epoch)
