# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5 fused-MoE adapter glue for the CUTEDSL_SMALL_BS backend.

The GLM-5 fused-MoE backend in TRT-LLM uses two fused CuTe DSL kernels:

  * up/gate/silu kernel (`fused_expert_up_gate_silu_fp8`) - takes the
    pre-sigmoid router logits and the activations, runs sigmoid + bias +
    topK + renormalize internally, then runs the per-expert up_proj and
    gate_proj GEMMs and the SiLU activation.
  * down/combine kernel (`fused_expert_down_combine_fp8`) - runs the
    per-expert down_proj GEMM and combines the topK expert outputs back
    into a single tensor (optionally adding a residual).

RMSNorm + router GEMM stay in `DeepseekV3Gate.forward` upstream of the MoE
call; the MoE backend receives router logits and returns the (un-allreduced)
MoE output.

This module is a thin adapter:

  1. Import both kernels from sibling modules in this package.
  2. Provide `glm5_fused_moe_apply()` that runs the two kernels in sequence
     and returns the MoE output.
  3. Provide `is_glm5_fused_shape()` for backend dispatch.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

# GLM-5 per-device constants after TP=8.  We only enable the fused path when
# all of these match; otherwise we fall back.
GLM5_NUM_EXPERTS = 256
GLM5_TOP_K = 8
GLM5_HIDDEN = 6144
GLM5_INTERMEDIATE_PER_DEVICE = 256


_up_gate_silu_fn = None
_down_combine_fn = None
_load_error: Optional[str] = None

try:
    from .fused_moe_cute_dsl_glm5_down_combine import (
        fused_expert_down_combine_fp8 as _down_combine_fn,
    )
    from .fused_moe_cute_dsl_glm5_up_gate_silu import (
        fused_expert_up_gate_silu_fp8 as _up_gate_silu_fn,
    )
except ImportError as _exc:
    _load_error = f"failed to import GLM-5 fused kernels: {_exc!r}"


def is_glm5_fused_shape(
    num_experts: int, hidden_size: int, intermediate_size_per_partition: int, top_k: int
) -> bool:
    """Return True iff the (num_experts, hidden_size, intermediate, top_k)
    tuple matches the GLM-5 per-device shape after TP=8."""
    return (
        num_experts == GLM5_NUM_EXPERTS
        and hidden_size == GLM5_HIDDEN
        and intermediate_size_per_partition == GLM5_INTERMEDIATE_PER_DEVICE
        and top_k == GLM5_TOP_K
    )


def kernels_available() -> Tuple[bool, Optional[str]]:
    """Return (True, None) when both fused kernels are importable, else
    (False, error_message)."""
    if _up_gate_silu_fn is not None and _down_combine_fn is not None:
        return True, None
    return False, _load_error


def glm5_fused_moe_apply(
    x: torch.Tensor,  # [M, H] bf16
    router_logits: torch.Tensor,  # [M, E] fp32 or bf16
    e_score_correction_bias: torch.Tensor,  # [E] fp32
    gate_weight_per_expert: torch.Tensor,  # [E, N, H] fp8e4m3
    up_weight_per_expert: torch.Tensor,  # [E, N, H] fp8e4m3
    gate_scale_per_expert: torch.Tensor,  # [E, N//128, H//128] fp32
    up_scale_per_expert: torch.Tensor,  # [E, N//128, H//128] fp32
    down_weight_per_expert: torch.Tensor,  # [E, H, N] fp8e4m3
    down_scale_per_expert: torch.Tensor,  # [E, H//128, N//128] fp32
    residual: Optional[torch.Tensor] = None,  # [M, H] bf16 or None
    top_k: int = GLM5_TOP_K,
    routed_scaling_factor: float = 2.5,
) -> torch.Tensor:
    """Run the up/gate/silu kernel followed by the down/combine kernel.

    Allreduce is NOT folded - the caller (`TritonFusedMoE`) does it via
    `reducescatter_or_allreduce`.

    The standard TRT-LLM MoE contract returns only the MoE output (the
    surrounding decoder layer adds the residual).  The down/combine kernel's
    `x_in` parameter always adds a residual; pass `residual=None` (default)
    to use a zero tensor and match the TRT-LLM contract.

    Returns: out [M, H] bfloat16 (NOT all-reduced; no residual when
    `residual=None`).
    """
    if _up_gate_silu_fn is None or _down_combine_fn is None:
        raise RuntimeError(f"GLM-5 fused kernels unavailable: {_load_error}")

    # The up/gate/silu kernel expects pre-sigmoid router logits in fp32; it
    # does sigmoid + bias + topK + renormalize internally.
    if router_logits.dtype != torch.float32:
        scores_in = router_logits.to(torch.float32)
    else:
        scores_in = router_logits
    bias_in = e_score_correction_bias.to(torch.float32)

    gated_hidden, probs, indices = _up_gate_silu_fn(
        x,
        scores_in,
        bias_in,
        gate_weight_per_expert,
        up_weight_per_expert,
        gate_scale_per_expert,
        up_scale_per_expert,
        top_k=top_k,
        routed_scaling_factor=routed_scaling_factor,
    )

    # The down/combine kernel unconditionally adds `x_in` to its output;
    # pass a zero tensor when the caller didn't supply an explicit residual.
    if residual is None:
        residual = torch.zeros_like(x)
    out = _down_combine_fn(
        gated_hidden,
        down_weight_per_expert,
        down_scale_per_expert,
        indices,
        probs,
        residual,
    )

    return out


__all__ = [
    "GLM5_NUM_EXPERTS",
    "GLM5_TOP_K",
    "GLM5_HIDDEN",
    "GLM5_INTERMEDIATE_PER_DEVICE",
    "is_glm5_fused_shape",
    "kernels_available",
    "glm5_fused_moe_apply",
]
