# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact MoE finalize for the routed experts' unfinalized output, optionally with the shared expert added.

The TRTLLM-Gen MoE runner with ``do_finalize=False`` returns the FC2 output in permuted order, the per-token top-k
routing weights and ``expanded_idx_to_permuted_idx``. Its own finalize (``finalizeKernelVecLoad``) computes, per token
and hidden element, ``acc = 0; acc += weight_k * fc2[permuted_idx_k]`` for k in order (skipping ``-1``) in float32
and rounds to BF16 once, but issues the k gathers one after another, so at prefill sizes it runs ~2.6x its byte
floor (153 us per layer at 16K rows on SM107). This kernel issues the gathers together and keeps the per-element
order, so the result is bitwise the runner's: a BF16 x BF16 product is exact in float32, so fused and unfused
multiply-add agree as well. ``shared`` adds the shared expert as ``shared_output.add_(routed_output)`` does
(``bf16(float(shared) + float(routed))``), which removes that separate pass over the hidden states.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

#: (BLOCK_H, num_warps): the tile only splits the hidden dimension, so it never changes the bits. Tuned on SM107
#: (2K / 4K / 16K rows: 11.2 / 20.5 / 74.9 us, 1.52 / 1.39 / 1.27x the byte floor; the runner's finalize
#: 23 / 40 / 153 us).
CONFIGS = ((2048, 4),)


@triton.jit
def _moe_finalize_kernel(fc2, idx, weights, shared, out, ld_fc2, H: tl.constexpr, TOPK: tl.constexpr,
                         BLOCK_H: tl.constexpr, SHARED: tl.constexpr):  # fmt: skip
    t = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    cmask = cols < H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        p = tl.load(idx + t * TOPK + k)
        w = tl.load(weights + t * TOPK + k).to(tl.float32)
        valid = p >= 0
        x = tl.load(
            fc2 + tl.maximum(p, 0).to(tl.int64) * ld_fc2 + cols, mask=cmask & valid, other=0.0
        )
        acc = tl.where(valid, tl.fma(w, x.to(tl.float32), acc), acc)
    r = acc.to(tl.bfloat16)
    if SHARED:
        s = tl.load(shared + t * H + cols, mask=cmask, other=0.0)
        r = (s.to(tl.float32) + r.to(tl.float32)).to(tl.bfloat16)
    tl.store(out + t * H + cols, r, mask=cmask)


def m3_moe_finalize(
    fc2: torch.Tensor,
    weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    shared: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """BF16 [tokens, H] = sum_k weights[:, k] * fc2[expanded_idx_to_permuted_idx[:, k]] (+ ``shared``), bitwise to the
    TRTLLM-Gen finalize (and to ``shared.add_(that)``); None when the operands are not the runner's layout."""
    if (
        fc2.dim() != 2
        or fc2.dtype != torch.bfloat16
        or fc2.stride(1) != 1
        or weights.dim() != 2
        or expanded_idx_to_permuted_idx.shape != weights.shape
        or expanded_idx_to_permuted_idx.dtype != torch.int32
        or not weights.is_contiguous()
        or not expanded_idx_to_permuted_idx.is_contiguous()
        or (
            shared is not None
            and (
                shared.dtype != torch.bfloat16
                or not shared.is_contiguous()
                or shared.shape != (weights.shape[0], fc2.shape[1])
            )
        )
    ):
        return None
    tokens, topk = weights.shape
    hidden = fc2.shape[1]
    out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=fc2.device)
    if tokens == 0:
        return out
    block_h, warps = CONFIGS[0]
    _moe_finalize_kernel[(tokens, triton.cdiv(hidden, block_h))](
        fc2,
        expanded_idx_to_permuted_idx,
        weights,
        shared if shared is not None else out,
        out,
        fc2.stride(0),
        hidden,
        topk,
        block_h,
        shared is not None,
        num_warps=warps,
    )
    return out
