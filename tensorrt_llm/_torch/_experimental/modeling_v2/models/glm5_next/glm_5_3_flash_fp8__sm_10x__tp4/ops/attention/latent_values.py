# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The absorbed value projection of a sparse MLA decode step, written row-major for o_proj.

``out[r, h * V : (h + 1) * V] = latent[r, h, :] @ w_v_t[h]`` (BF16 inputs, FP32
accumulation), and zero for rows that selected no key. The module path runs a batched GEMM
that returns head-major ``[H, R, V]``, a transposing copy and a select over the result.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


@triton.jit
def _latent_values_kernel(
    lat_ptr,
    w_ptr,
    keep_ptr,
    out_ptr,
    R,
    H: tl.constexpr,
    L: tl.constexpr,
    V: tl.constexpr,
    BR: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    h = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    r = tl.program_id(2) * BR + tl.arange(0, BR)
    rm = r < R
    acc = tl.zeros((BR, BN), tl.float32)
    for k0 in range(0, L, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(lat_ptr + (r[:, None] * H + h) * L + k[None, :], mask=rm[:, None], other=0.0)
        b = tl.load(w_ptr + (h * L + k[:, None]) * V + n[None, :])
        acc += tl.dot(a, b)
    keep = tl.load(keep_ptr + r, mask=rm, other=0) != 0
    acc = tl.where(keep[:, None], acc, 0.0)
    tl.store(
        out_ptr + r[:, None] * (H * V) + h * V + n[None, :], acc.to(tl.bfloat16), mask=rm[:, None]
    )


class LatentValues(OpWrapper):
    """``latent`` BF16 [R, H, L], ``w_v_t`` BF16 [H, L, V], ``keep`` bool [R] -> BF16 [R, H * V]."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS: tuple[Cell, ...] = (
        Cell(why="MTP-3 verify of one request", spec=dict(r=4, h=16, l=512, v=256)),
        Cell(why="MTP-3 verify of eight requests", spec=dict(r=32, h=16, l=512, v=256)),
        Cell(why="CUDA-graph batch of 64 verify requests", spec=dict(r=256, h=16, l=512, v=256)),
    )

    def __call__(self, latent, w_v_t, keep):
        rows, heads, lat = latent.shape
        v = w_v_t.shape[2]
        out = torch.empty((rows, heads * v), dtype=torch.bfloat16, device=latent.device)
        _latent_values_kernel[(heads, v // 32, triton.cdiv(rows, 16))](
            latent, w_v_t, keep, out, rows, heads, lat, v, 16, 32, 128, num_warps=4, num_stages=4
        )
        return out

    def reference(self, latent, w_v_t, keep):
        with true_fp32_matmul():
            out = torch.einsum("rhl,hlv->rhv", latent.float(), w_v_t.float())
        out = torch.where(keep[:, None, None], out, 0.0)
        return out.reshape(latent.shape[0], -1).to(torch.bfloat16)

    def is_valid(self, latent, w_v_t, keep):
        assert latent.dtype == torch.bfloat16 and latent.is_contiguous() and latent.dim() == 3
        assert w_v_t.dtype == torch.bfloat16 and w_v_t.is_contiguous()
        assert keep.dtype == torch.bool and keep.shape == (latent.shape[0],)
        assert w_v_t.shape[2] % 32 == 0 and latent.shape[2] % 128 == 0


latent_values = LatentValues()
