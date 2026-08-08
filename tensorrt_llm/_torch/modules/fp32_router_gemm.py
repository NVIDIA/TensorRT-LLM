# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP32 MoE router projection for decode-shaped batches.

A router that keeps its weight in FP32 has no good path through cuBLAS at decode
shapes. For ``[4, 6144] x [6144, 128]`` cuBLAS picks a 32-way split-K TF32 kernel
plus a ``splitKreduce``, and the FP32 activation it wants costs a separate cast
of the BF16 hidden states first. Three kernels, and the TF32 tensor cores round
both operands to a 10-bit mantissa on the way in.

The existing small-router kernels do not apply: ``dsv3_router_gemm_op`` and
``tinygemm2`` both take a BF16 weight, which is the one thing an FP32 router
cannot give up.

This is a GEMV band instead. One CTA per expert reduces over the hidden
dimension with plain FP32 FMA, widening the BF16 activation in-register, so the
cast disappears, there are no partials to reduce, and the multiply keeps the
full FP32 mantissa. It is deliberately shaped for a handful of tokens, which is
where decode lives; anything wider falls back to ``F.linear``, both because
cuBLAS wins there and because prefill should keep the numerics it has.

Numerics move, and in the direction of the reference: this is closer to a true
FP32 router than the TF32 split-K it replaces, so it wants an eval rather than a
bitwise comparison.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Measured on B200 at 128 experts / 6144 hidden, against the cuBLAS sequence:
# 3.4x at 1 token, 2.7x at 4, 1.8x at 8, and a loss by 16. One CTA per expert
# stops paying once the batch is wide enough for cuBLAS to tile it.
MAX_GEMV_TOKENS = 8

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
        num_warps=8,
    )
    return out
