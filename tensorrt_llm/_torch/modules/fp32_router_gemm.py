# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP32 MoE router projection.

A router that keeps its weight in FP32 has no good path through cuBLAS at any
shape. For ``[4, 6144] x [6144, 128]`` cuBLAS picks a 32-way split-K TF32 kernel
plus a ``splitKreduce``, and the FP32 activation it wants costs a separate cast
of the BF16 hidden states first. Three kernels, and the TF32 tensor cores round
both operands to a 10-bit mantissa on the way in.

The existing small-router kernels do not apply: ``dsv3_router_gemm_op`` and
``tinygemm2`` both take a BF16 weight, which is the one thing an FP32 router
cannot give up.

Two kernels cover the range, and :func:`router_gemm` picks between them:

* Up to ``MAX_GEMV_TOKENS``, the GEMV in this module. One CTA per expert
  reduces over the hidden dimension with plain FP32 FMA, widening the BF16
  activation in-register, so the cast disappears, there are no partials to
  reduce, and the multiply keeps the full FP32 mantissa. It is deliberately
  shaped for a handful of tokens, which is where decode lives.
* Above it, the CuTe DSL gate GEMM, which keeps the activation BF16 and moves
  the precision onto the weight instead -- the FP32 weight is pre-split at load
  into a sum of BF16 terms the tensor cores can take. It needs Blackwell and
  the ``nvidia-cutlass-dsl`` package, so it stays a soft dependency.

Anything neither can serve falls back to ``F.linear`` on a widened activation,
which is what the model did before either existed.

Numerics move, and in the direction of the reference: both kernels are closer
to a true FP32 router than the TF32 split-K they replace, so this wants an eval
rather than a bitwise comparison.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# Measured on B200 at 128 experts / 6144 hidden, against the cuBLAS sequence:
# 3.4x at 4 tokens, 1.6x at 12, 1.5x at 16, and 0.85x by 32. The turn comes from
# the register tile: the activation block is BLOCK_M x BLOCK_K, so past 16
# tokens BLOCK_K has to shrink faster than the extra rows pay for, and cuBLAS'
# tiling wins. Streaming K per token the way dsv3RouterGemm does would push this
# out, at the cost of a C++ kernel with a separate instantiation per token count.
MAX_GEMV_TOKENS = 16

# Elements of the activation tile held in registers per CTA, which sets the K
# block. Larger was monotonically better across the whole tuning sweep, all the
# way to covering the hidden dimension in a single pass: at one CTA per expert
# the kernel is latency-bound, occupancy is already capped by having fewer CTAs
# than SMs, so the only lever is loads in flight per thread. This lets a 4-token
# batch take the whole 6144 in one iteration, worth 4.26 -> 2.53 us.
_TILE_ELEMS = 32768


@triton.jit
def _fp32_router_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    num_tokens,
    hidden_size,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < num_tokens
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k0 in range(0, hidden_size, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hidden_size
        w = tl.load(w_ptr + expert * stride_wn + offs_k * stride_wk, mask=k_mask, other=0.0)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # tl.sum over an FP32 product, not tl.dot: this has to stay off the
        # tensor cores, which would round both operands to TF32.
        acc += tl.sum(x.to(tl.float32) * w[None, :], axis=1)

    tl.store(out_ptr + offs_m * stride_om + expert * stride_on, acc, mask=m_mask)


def _gemv_applies(hidden_states: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        hidden_states.is_cuda
        and hidden_states.dim() == 2
        and weight.dim() == 2
        and weight.dtype == torch.float32
        and hidden_states.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and hidden_states.shape[1] == weight.shape[1]
        and 0 < hidden_states.shape[0] <= MAX_GEMV_TOKENS
        and hidden_states.stride(1) == 1
        and weight.stride(1) == 1
    )


def fp32_router_gemm(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Router logits in FP32: ``hidden_states.float() @ weight.T``.

    ``weight`` is ``[num_experts, hidden_size]`` FP32, in the ``nn.Linear``
    layout the HF checkpoint already uses. Falls back to ``F.linear`` for shapes
    the GEMV is not built for.
    """
    if not _gemv_applies(hidden_states, weight):
        return torch.nn.functional.linear(hidden_states.to(torch.float32), weight)

    num_tokens, hidden_size = hidden_states.shape
    num_experts = weight.shape[0]
    out = torch.empty((num_tokens, num_experts), dtype=torch.float32, device=hidden_states.device)

    block_m = triton.next_power_of_2(num_tokens)
    # No point blocking past the hidden dimension; the tail just masks off.
    block_k = min(
        triton.next_power_of_2(hidden_size),
        max(128, triton.next_power_of_2(_TILE_ELEMS // block_m)),
    )

    _fp32_router_gemm_kernel[(num_experts,)](
        hidden_states,
        weight,
        out,
        num_tokens,
        hidden_size,
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        # A wide block has enough loads to keep 8 warps busy; a narrow one only
        # spreads thinner, and 4 measured faster at BLOCK_K=2048.
        num_warps=8 if block_k >= 4096 else 4,
    )
    return out


def cute_gate_ops():
    """The CuTe DSL gate-GEMM helpers, or None when unavailable.

    The CuTe DSL ops are registered only when the nvidia-cutlass-dsl package is
    importable, so this stays a soft dependency. Resolved through the module
    rather than imported directly, since importing the runner pulls in cutlass.
    """
    try:
        from ..custom_ops import cute_dsl_custom_ops
    except ImportError:
        return None
    if not hasattr(cute_dsl_custom_ops, "minimax_m3_gate_gemm_is_supported"):
        return None
    return cute_dsl_custom_ops


def split_router_weight(weight: torch.Tensor) -> Optional[torch.Tensor]:
    """Pre-split an FP32 router weight for the CuTe DSL gate GEMM.

    Returns None when that kernel is unavailable -- the package missing or the
    GPU not Blackwell -- which leaves the caller with nothing to pass and
    :func:`router_gemm` on its fallback above the GEMV band. Meant to run once
    at load: the split is a few small elementwise passes, but the result is a
    second copy of the weight.

    Deciding the architecture here rather than per call is what keeps
    :func:`router_gemm` free of it, since the arch query is lru_cached and
    Dynamo traces through an lru_cache instead of honouring it.
    """
    ops = cute_gate_ops()
    if ops is None or not ops.minimax_m3_gate_arch_is_supported():
        return None
    return ops.minimax_m3_gate_split_weight(weight)


def router_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    weight_split: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Router logits in FP32: ``hidden_states.float() @ weight.T``.

    ``weight`` is ``[num_experts, hidden_size]`` FP32, in the ``nn.Linear``
    layout the HF checkpoint already uses. ``weight_split`` is what
    :func:`split_router_weight` made of it, or None to leave the wide band on
    the ``F.linear`` fallback.

    The split at ``MAX_GEMV_TOKENS`` is where the GEMV's register tile stops
    paying; see that constant. Both branches are decided by shape alone, so a
    traced graph keeps whichever kernel it was captured with.
    """
    if hidden_states.dim() == 2 and hidden_states.shape[0] > MAX_GEMV_TOKENS:
        ops = cute_gate_ops()
        if ops is not None and ops.minimax_m3_gate_gemm_is_supported(
            hidden_states, weight_split, ops.MINIMAX_M3_GATE_GEMM_TERMS
        ):
            return torch.ops.trtllm.cute_dsl_minimax_m3_gate_gemm(
                hidden_states, weight_split, ops.MINIMAX_M3_GATE_GEMM_TERMS
            )
    return fp32_router_gemm(hidden_states, weight)
