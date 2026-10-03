# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 decode MoE input in one launch: noaux_tc routing (896 experts, top-16) + MXFP8 quantization."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def kimi_k3_noaux_tc_mxfp8_quant(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    hidden_states: torch.Tensor,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route each token to 16 of 896 experts and quantize its [3584] hidden row to MXFP8.

    Returns `(topk_ids [M, 16] int32, topk_weights [M, 16] bf16, data [M, 3584] float8_e4m3fn,
    scales [M, 112] uint8)`, all newly allocated and contiguous: ids first (the opposite order
    to noaux_tc_op), linear UE8M0 scales, one per 32 elements.
    """
    return torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant(
        router_logits, bias, hidden_states, routed_scaling_factor
    )
