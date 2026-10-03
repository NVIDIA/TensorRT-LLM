# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's KDA speculative verify of N requests of 1 + num_spec tokens, from the fused projection rows to the
gated-norm core output, committing the state after every verify token."""

from typing import Optional

import torch

from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_verify import (  # noqa: F401 — registers the op
    op as _k3_kda_verify_op,
)


def k3_kda_verify(
    proj: torch.Tensor,
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
    num_spec: int,
    lower_bound: float,
    scale: float,
    eps: float,
    g_ext: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the gated-norm core output bf16 [N (1 + num_spec), H, 128] of the verify tokens whose fused
    projection rows are ``proj``; updates the slots' conv caches, state and per-draft states in place."""
    return torch.ops.trtllm.k3_kda_verify(
        proj,
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
        num_spec,
        lower_bound,
        scale,
        eps,
        g_ext,
    )
