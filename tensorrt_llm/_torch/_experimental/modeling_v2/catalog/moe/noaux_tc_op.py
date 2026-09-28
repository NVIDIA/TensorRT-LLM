# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V3 style MoE routing: sigmoid + bias-corrected group-limited top-k."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def noaux_tc_op(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route each token to `topk` experts by biased sigmoid score.

    Selection uses `sigmoid(router_logits) + bias`; the combine weights are
    gathered from the *unbiased* `sigmoid(router_logits)`, renormalized to sum
    to 1 and scaled by `routed_scaling_factor`.

    Returns `(topk_weights [T, topk] in router_logits.dtype, topk_ids
    [T, topk] int32)`, both newly allocated and contiguous.
    """
    # Pure-metadata guards for three domains the op does not police. All three
    # were observed on this machine to return a plausible-looking wrong answer
    # instead of raising:
    #  - the kernel addresses `router_logits` as a dense [T, num_experts]
    #    row-major buffer and `bias` as a dense [num_experts] buffer, both from
    #    data_ptr(), ignoring strides;
    #  - topk > num_experts makes the kernel read past the end of a row and
    #    emit expert ids >= num_experts at non-zero weight.
    assert router_logits.is_contiguous(), (
        "router_logits must be contiguous; a strided view is read as a dense "
        "[num_tokens, num_experts] buffer and silently gives wrong routing"
    )
    assert bias.is_contiguous(), (
        "bias must be contiguous; a strided view is read as a dense "
        "[num_experts] buffer and silently gives wrong routing"
    )
    assert topk <= router_logits.shape[-1], (
        f"topk ({topk}) exceeds num_experts ({router_logits.shape[-1]}); the "
        "kernel reads out of bounds and emits out-of-range expert ids"
    )
    return torch.ops.trtllm.noaux_tc_op(
        router_logits, bias, n_group, topk_group, topk, routed_scaling_factor
    )
