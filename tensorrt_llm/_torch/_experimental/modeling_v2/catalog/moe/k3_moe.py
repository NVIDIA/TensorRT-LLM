# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's routed experts at decode size: this rank's routed partial from the persistent CuTe DSL kernel ``k3_moe``
(FC1 + SiTU + FC2 with the routing-weighted combine over the TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers, read in place),
after the routing and MXFP8 quantization of its producer, on caller-owned state: a :class:`K3MoeState` and one
:class:`K3MoeLayer` per MoE layer for up to 8 tokens, a :class:`K3MoeWideState` and one :class:`K3MoeWideLayer` per
layer for up to 64."""

from typing import Optional, Tuple

import torch

# The state types; is_supported reads metadata only. Importing k3_route_quant's op registers trtllm::k3_route_quant.
from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe.op import (
    K3MoeHeadWorkspace,
    K3MoeLayer,
    K3MoeState,
    K3MoeWideLayer,
    K3MoeWideState,
    is_supported,
)
from tensorrt_llm._torch.cute_dsl_kernels.k3_route_quant import op as _k3_route_quant_op  # noqa: F401

__all__ = [
    "K3MoeHeadWorkspace",
    "K3MoeLayer",
    "K3MoeState",
    "K3MoeWideLayer",
    "K3MoeWideState",
    "is_supported",
    "k3_moe",
    "k3_moe_fused_front",
    "k3_moe_wide",
]


def k3_moe(
    latent: torch.Tensor,
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    layer: K3MoeLayer,
) -> torch.Tensor:
    """This rank's routed partial ``[M, 3584]`` bf16 for ``M <= 8`` tokens: ``trtllm::k3_route_quant`` of
    ``router_logits`` (fp32 ``[M, 896]``) and ``latent`` (bf16 ``[M, 3584]``), then ``k3_moe`` on ``layer``'s experts
    (global ids ``[local_expert_offset, local_expert_offset + num_local)``). Writes ``layer``'s state's scratch (left
    armed) and ``layer``'s counters (left zero)."""
    return layer(latent, router_logits, e_score_correction_bias, local_expert_offset, routed_scaling_factor)


def k3_moe_fused_front(
    x: torch.Tensor,
    w_front: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    shared_cols: int,
    gate_cap: float,
    linear_cap: float,
    head: K3MoeHeadWorkspace,
    layer: K3MoeLayer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(routed partial [M, 3584] bf16, shared activation [M, shared_cols] bf16)`` for the MoE input ``x`` (bf16
    ``[M <= 8, 7168]``): ``trtllm::k3_moe_front`` over ``head`` (see ``moe/k3_moe_front``), then ``k3_moe`` on
    ``layer``'s experts. Advances ``head`` by one front call; writes ``layer``'s scratch and counters as
    :func:`k3_moe`."""
    return layer.front(
        x, w_front, e_score_correction_bias, local_expert_offset, routed_scaling_factor, shared_cols, gate_cap,
        linear_cap, head,
    )  # fmt: skip


def k3_moe_wide(
    latent: torch.Tensor,
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    layer: K3MoeWideLayer,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """This rank's routed partial ``[M, 3584]`` bf16 for ``1 <= M <= 64`` tokens: ``trtllm::k3_route_quant`` (its
    dependents launched early, as ``k3_moe``'s PDL producer), then the m_max 64 build of ``k3_moe`` on ``layer``'s
    experts. ``out``: bf16, contiguous, at least ``[M, 3584]``; its first M rows are the result (a new tensor without
    it). Writes ``layer``'s state's scratch (left armed) and ``layer``'s counters (left zero)."""
    ids, weights, x_fp8, x_sf = torch.ops.trtllm.k3_route_quant(
        router_logits, e_score_correction_bias, latent, routed_scaling_factor, early_trigger=True
    )
    return layer(x_fp8, x_sf, ids, weights, local_expert_offset, out=out)
