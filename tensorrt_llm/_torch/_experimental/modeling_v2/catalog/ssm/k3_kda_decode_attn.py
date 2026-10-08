# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's KDA projection and plain one-token decode of up to 8 requests in one launch, over a caller-owned
:class:`K3KdaBuffers`."""

import torch

from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_attn import (  # noqa: F401 — registers the op
    op as _k3_kda_attn_op,
)
from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_verify import (  # noqa: F401 — the op's helpers
    op as _k3_kda_verify_op,
)

from .k3_kda_buffers import K3KdaBuffers


def k3_kda_decode_attn(
    x: torch.Tensor,
    w: torch.Tensor,
    w_fb: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    onorm_w: torch.Tensor,
    conv: torch.Tensor,
    ssm: torch.Tensor,
    slots: torch.Tensor,
    buffers: K3KdaBuffers,
    lower_bound: float,
    scale: float,
    eps: float,
) -> torch.Tensor:
    """Return the gated-norm core output bf16 [R, 6, 128] of one token of each of R <= 8 requests ``x`` bf16
    [R, 7168]: the projection ``x w^T`` and the plain decode of ``ssm/kda_decode`` on it, in one launch. Updates the
    slots' conv and state pools in place; advances ``buffers`` by one launch."""
    return torch.ops.trtllm.k3_kda_decode_attn(
        x,
        w,
        w_fb,
        w_q,
        w_k,
        w_v,
        a_log,
        dt_bias,
        onorm_w,
        conv,
        ssm,
        slots,
        buffers.p1,
        buffers.part,
        buffers.epoch,
        lower_bound,
        scale,
        eps,
    )
