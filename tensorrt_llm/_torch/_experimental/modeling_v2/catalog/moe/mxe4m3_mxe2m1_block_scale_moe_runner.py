# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP4-weight / MXFP8-activation mixture-of-experts layer: routing (or given
top-k) + grouped FC1 GEMM + clamped gated activation + MXFP8 requantization +
grouped FC2 GEMM + routing-weighted combine, in one trtllm-gen call."""

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

# routing_method_type values for which the kernel never reads routing_bias:
# 0 Default, 1 Renormalize, 4 RenormalizeNaive, 6 SigmoidRenorm. Observed on
# this machine: a +-1e3 bias leaves the result bitwise unchanged.
_ROUTING_BIAS_IGNORED = (0, 1, 4, 6)


def mxe4m3_mxe2m1_block_scale_moe_runner(
    routing_logits: Optional[torch.Tensor],
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    valid_hidden_size: Optional[int],
    valid_intermediate_size: Optional[int],
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int,
    act_type: int,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    use_dp: bool = False,
) -> torch.Tensor:
    """Run one MXFP4-weight MoE layer over MXFP8 (e4m3 + UE8M0) activations.

    Returns a fresh `[num_tokens, valid_hidden_size]` bf16 tensor, or — when
    `output` is given — an empty `[0]` tensor, the result having been written
    into `output`.
    """
    # Pure-metadata guard: the kernel takes raw data pointers and assumes a
    # dense row-major layout for every tensor. A strided view is accepted
    # without complaint and silently reads the wrong elements (observed on
    # this machine for every tensor argument listed here, hidden_states_scale
    # included).
    for name, tensor in (
        ("routing_logits", routing_logits),
        ("routing_bias", routing_bias),
        ("hidden_states", hidden_states),
        ("hidden_states_scale", hidden_states_scale),
        ("gemm1_weights", gemm1_weights),
        ("gemm1_weights_scale", gemm1_weights_scale),
        ("gemm1_bias", gemm1_bias),
        ("gemm1_alpha", gemm1_alpha),
        ("gemm1_beta", gemm1_beta),
        ("gemm1_clamp_limit", gemm1_clamp_limit),
        ("gemm2_weights", gemm2_weights),
        ("gemm2_weights_scale", gemm2_weights_scale),
        ("gemm2_bias", gemm2_bias),
        ("topk_weights", topk_weights),
        ("topk_ids", topk_ids),
        ("output", output),
    ):
        if tensor is not None:
            assert tensor.is_contiguous(), (
                f"{name} must be contiguous; a strided view is read as if dense "
                "and silently produces wrong results"
            )
    # Pure-metadata guard: routing_bias is a no-op for every non-grouped
    # routing method, and for any routing method once topk_ids/topk_weights
    # carry the routing. A caller expecting the bias to shift expert selection
    # gets a silently different model.
    if routing_bias is not None:
        assert topk_ids is None, (
            "routing_bias is ignored when topk_ids/topk_weights are given "
            "(routing has already happened); fold it into the router logits"
        )
        assert routing_method_type not in _ROUTING_BIAS_IGNORED, (
            f"routing_bias is silently ignored for routing_method_type="
            f"{routing_method_type}; add it to routing_logits before the call"
        )
    return torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        valid_hidden_size,
        valid_intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        act_type,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        tune_max_num_tokens=tune_max_num_tokens,
        use_dp=use_dp,
    )
