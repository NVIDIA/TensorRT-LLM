# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4-weight / NVFP4-activation mixture-of-experts layer: caller-supplied
top-k + grouped FC1 GEMM + gated activation + NVFP4 requantization + grouped
FC2 GEMM + routing-weighted combine, in one trtllm-gen call."""

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def fp4_block_scale_moe_runner(
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
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int,
    do_finalize: bool,
    act_type: int = 0,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    use_dp: bool = False,
) -> list[torch.Tensor]:
    """Run one NVFP4-weight MoE layer over NVFP4 (e2m1 + e4m3 block scale) activations.

    With `do_finalize=True` returns a one-element list holding the combined
    `[num_tokens, hidden]` bf16 result — or an empty `[0]` tensor when `output`
    was given, the result having been written there. With `do_finalize=False`
    returns three tensors (per-slot expert rows, an unwritten scale buffer, and
    the expanded-index -> permuted-row map).
    """
    # Pure-metadata guard: the kernel takes raw data pointers and assumes a
    # dense row-major layout for every tensor. A strided view is accepted
    # without complaint and silently reads (or writes) the wrong elements —
    # observed on this machine for hidden_states, topk_weights and the weight
    # operands.
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
        ("output1_scale_scalar", output1_scale_scalar),
        ("output1_scale_gate_scalar", output1_scale_gate_scalar),
        ("output2_scale_scalar", output2_scale_scalar),
        ("topk_weights", topk_weights),
        ("topk_ids", topk_ids),
        ("output", output),
    ):
        if tensor is not None:
            assert tensor.is_contiguous(), (
                f"{name} must be contiguous; a strided view is read as if dense "
                "and silently produces wrong results"
            )
    # Pure-metadata guard: with the block-scale layout this entry contracts
    # (row shuffle + 128x4 scale swizzle), a hidden size that is not a multiple
    # of 256 makes the kernel read the FC1 weight scales for the wrong blocks.
    # Observed on this machine at hidden 384 / 640 / 896: no error, 300-3400
    # bf16 ulp wrong.
    assert gemm1_weights.shape[-1] * 2 % 256 == 0, (
        "hidden size (gemm1_weights.shape[-1] * 2) must be a multiple of 256; "
        "the prepared block-scale layout is silently misread otherwise"
    )
    return torch.ops.trtllm.fp4_block_scale_moe_runner(
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
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        do_finalize,
        act_type,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        tune_max_num_tokens=tune_max_num_tokens,
        use_dp=use_dp,
    )
