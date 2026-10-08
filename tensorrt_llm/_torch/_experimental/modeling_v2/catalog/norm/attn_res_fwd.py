# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 attention-residual selection: a per-token softmax mixture of residual snapshots."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def attn_res_fwd(
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix each token's N = K + 1 candidates: the K snapshots of block_residual, then layer_residual.

    A candidate's logit is its fp32 RMSNorm (weight `rms_weight`, `rms_eps`) dotted with `res_weight`;
    the softmax of the logits weights the raw candidates. `layer_residual` is bf16 [T, 1, H] and
    `block_residual` bf16 [K, T, 1, H], both contiguous.

    Returns `(output, rsigma, probs, logits)`: the bf16 [T, 1, H] mixture and the fp32 [N, T, 1]
    per-candidate rsqrt(mean square + eps), softmax probabilities and logits, all newly allocated.
    """
    return torch.ops.trtllm.attn_res_fwd(
        layer_residual, block_residual, res_weight, rms_weight, rms_eps
    )
