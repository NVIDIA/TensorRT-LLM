# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-token KDA decode of B requests: the causal conv, the gated delta rule on each request's state slot and the
gated RMS norm, the conv and state pools updated in place."""

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def kda_decode(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    w_q_t: torch.Tensor,
    w_k_t: torch.Tensor,
    w_v_t: torch.Tensor,
    bias_q: torch.Tensor,
    bias_k: torch.Tensor,
    bias_v: torch.Tensor,
    conv_state_q: torch.Tensor,
    conv_state_k: torch.Tensor,
    conv_state_v: torch.Tensor,
    a_log: torch.Tensor,
    g: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: torch.Tensor,
    onorm_g: torch.Tensor,
    onorm_weight: torch.Tensor,
    ssm_state_indices: Optional[torch.Tensor],
    state: torch.Tensor,
    apply_onorm: bool,
    update_conv_cache: bool,
    use_lower_bound: bool,
    apply_beta_sigmoid: bool,
    lower_bound: float,
    scale: float,
    onorm_eps: float,
    output: torch.Tensor,
) -> None:
    """Writes the decode output into ``output`` [B, 1, HV, 128] bf16; updates ``state`` at ``ssm_state_indices``
    and, with ``update_conv_cache``, the conv pools there. Returns None."""
    torch.ops.trtllm.kda_decode(
        x_q,
        x_k,
        x_v,
        w_q_t,
        w_k_t,
        w_v_t,
        bias_q,
        bias_k,
        bias_v,
        conv_state_q,
        conv_state_k,
        conv_state_v,
        a_log,
        g,
        dt_bias,
        beta,
        onorm_g,
        onorm_weight,
        ssm_state_indices,
        state,
        apply_onorm,
        update_conv_cache,
        use_lower_bound,
        apply_beta_sigmoid,
        lower_bound,
        scale,
        onorm_eps,
        output=output,
    )
