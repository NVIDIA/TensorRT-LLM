# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer's logits weights in one kernel (Triton): ``(w * qs) * scale`` in FP32.

``w * qs.squeeze(-1) * scale`` in torch is two elementwise kernels, the product with the query
scales and then the constant. ``_idx_weights`` reads the projection's strided ``w`` and the query
scales once and writes the result, rounding the product and then the scaled product exactly as
the two kernels do.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _idx_weights_kernel(
    w_ptr, qs_ptr, out_ptr, rows, w_row_stride, scale, HEADS: tl.constexpr, BLOCK_M: tl.constexpr
):
    """out[r, h] = (w[r, h] * qs[r, h]) * scale: w rows ``w_row_stride`` apart, qs and out dense."""
    r = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)[:, None]
    h = tl.arange(0, HEADS)[None, :]
    m = r < rows
    w = tl.load(w_ptr + r * w_row_stride + h, mask=m, other=0.0)
    q = tl.load(qs_ptr + r * HEADS + h, mask=m, other=0.0)
    p = w * q
    tl.store(out_ptr + r * HEADS + h, p * scale, mask=m)


def _idx_weights(w: torch.Tensor, qs: torch.Tensor, scale: float) -> torch.Tensor:
    """A new dense FP32 ``[rows, heads]`` tensor; ``w`` FP32, unit inner stride; ``qs`` dense."""
    rows, heads = w.shape
    out = torch.empty(rows, heads, dtype=torch.float32, device=w.device)
    if rows == 0:
        return out
    block_m = 32
    _idx_weights_kernel[(triton.cdiv(rows, block_m),)](
        w, qs, out, rows, w.stride(0), scale, HEADS=heads, BLOCK_M=block_m, num_warps=4
    )
    return out
