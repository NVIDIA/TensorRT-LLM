# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 casts of head-major ``[H, R, D]`` batched-GEMM outputs into row-major ``[R, H * D]``.

The sequence-parallel MLA context rows run their per-head products (query into the latent
space, latent output into the value space) as ``torch.bmm`` over the heads, which returns
head-major tensors. Each consumer wants the rows contiguous and in FP8, so each cast reads
the head-major tensor once and writes the row-major FP8 copy, in place of a transposing copy
followed by the cast.

``saturate_fp8``: the attention query, ``clamp(+-448)`` then E4M3 (the saturating conversion
the sparse attention applies to its query).

``quantize_1x128``: the o_proj input, the 1 x 128 block quantization ``fp8_quantize_1x128``
computes (scale ``448 / amax`` with ``amax`` floored at BF16 ``1e-10``, FP32 product, saturating E4M3), with
its K-block-major ``[H * D / 128, R]`` FP32 dequantization scales.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_FP8_MAX = 448.0


@triton.jit
def _saturate_kernel(x_ptr, out_ptr, R, H: tl.constexpr, D: tl.constexpr, BR: tl.constexpr):
    h = tl.program_id(1)
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    d = tl.arange(0, D)
    mask = (r < R)[:, None]
    x = tl.load(x_ptr + (h * R + r[:, None]).to(tl.int64) * D + d[None, :], mask=mask)
    x = tl.minimum(tl.maximum(x.to(tl.float32), -448.0), 448.0)
    tl.store(
        out_ptr + (r[:, None].to(tl.int64) * H + h) * D + d[None, :], x.to(tl.float8e4nv), mask=mask
    )


@triton.jit
def _quantize_kernel(
    x_ptr, out_ptr, scale_ptr, R, H: tl.constexpr, D: tl.constexpr, BR: tl.constexpr
):
    h = tl.program_id(1)
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    mask = (r < R)[:, None, None]
    d = tl.arange(0, D).reshape(D // 128, 128)
    x = tl.load(x_ptr + (h * R + r[:, None, None]).to(tl.int64) * D + d[None, :, :], mask=mask)
    # The floor is 1e-10 rounded to BF16, as the quantizer compares in BF16.
    amax = tl.maximum(tl.max(tl.abs(x), axis=2).to(tl.float32), 1.000444171950221e-10)
    quant = tl.math.div_rn(tl.full(amax.shape, 448.0, tl.float32), amax)
    q = (x.to(tl.float32) * quant[:, :, None]).to(tl.float8e4nv)
    tl.store(out_ptr + (r[:, None, None].to(tl.int64) * H + h) * D + d[None, :, :], q, mask=mask)
    blk = h * (D // 128) + tl.arange(0, D // 128)
    dequant = tl.math.div_rn(tl.full(quant.shape, 1.0, tl.float32), quant)
    tl.store(scale_ptr + blk[None, :] * R + r[:, None], dequant, mask=(r < R)[:, None])


class SaturateFp8(OpWrapper):
    """``clamp(x, -448, 448)`` as E4M3, ``[H, R, D]`` BF16 -> ``[R, H, D]``."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS: tuple[Cell, ...] = (
        Cell(why="MLA context query, 13K prompt / 4 ranks", spec=dict(h=64, r=3328, d=512)),
        Cell(why="MLA context query, ragged rows", spec=dict(h=64, r=517, d=512)),
    )

    def __call__(self, x):
        h, r, d = x.shape
        out = torch.empty((r, h, d), dtype=torch.float8_e4m3fn, device=x.device)
        _saturate_kernel[(triton.cdiv(r, 8), h)](x, out, r, h, d, 8, num_warps=4)
        return out

    def reference(self, x):
        return x.transpose(0, 1).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)

    def is_valid(self, x):
        assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.dim() == 3
        assert x.shape[2] % 128 == 0


class Quantize1x128(OpWrapper):
    """``fp8_quantize_1x128`` of ``x.transpose(0, 1).reshape(R, H * D)`` for ``[H, R, D]`` BF16:
    E4M3 ``[R, H * D]`` and FP32 ``[H * D / 128, R]``."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS: tuple[Cell, ...] = (
        Cell(why="MLA context values, 13K prompt / 4 ranks", spec=dict(h=64, r=3328, d=256)),
        Cell(why="MLA context values, ragged rows", spec=dict(h=64, r=517, d=256)),
    )

    def __call__(self, x):
        h, r, d = x.shape
        out = torch.empty((r, h * d), dtype=torch.float8_e4m3fn, device=x.device)
        scale = torch.empty((h * d // 128, r), dtype=torch.float32, device=x.device)
        _quantize_kernel[(triton.cdiv(r, 8), h)](x, out, scale, r, h, d, 8, num_warps=4)
        return out, scale

    def reference(self, x):
        h, r, d = x.shape
        q, s = torch.ops.trtllm.fp8_quantize_1x128(x.transpose(0, 1).reshape(r, h * d))
        return q, s.view(h * d // 128, -1)[:, :r]

    def is_valid(self, x):
        assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.dim() == 3
        assert x.shape[2] % 128 == 0


saturate_fp8 = SaturateFp8()
quantize_1x128 = Quantize1x128()
