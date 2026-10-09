# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 attention-residual selection fused with the RMSNorm that consumes it."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def attn_res_rmsnorm_fwd(
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_eps: float,
) -> torch.Tensor:
    """Return rmsnorm(attn_res_fwd's bf16 mixture) * output_rms_weight as a new bf16 [T, 1, 7168] tensor.

    `res_weight`, `rms_weight` and `rms_eps` score the candidates as in `attn_res_fwd`;
    `output_rms_weight` and `output_rms_eps` belong to the trailing norm, which rounds the normalized
    value to bf16 before the weight multiply. With PDL enabled the kernel releases its dependents before
    it writes the output, so a dependent PDL kernel must wait on the grid dependency before reading it.
    """
    return torch.ops.trtllm.attn_res_rmsnorm_fwd(
        layer_residual,
        block_residual,
        res_weight,
        rms_weight,
        output_rms_weight,
        rms_eps,
        output_rms_eps,
        early_trigger=True,
    )
