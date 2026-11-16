# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A DFlash 2 drafter hop's tail in one launch: the finish-side block convolution, the residual add and the RMSNorm.

Where a drafter forward does not take the fused drafter hop (``ops/allreduce/drafter_hop.py``: the TP8 block
decodes of 65..128 rows, 9..16 requests), each of its two hops per layer is three launches after the
projection's built-in all-reduce: ``dflash2_block_conv`` (the finish side, ``ops/spec/dflash2_block_conv.py``),
then flashinfer's CuTe-DSL ``FusedAddRMSNormKernel`` (the next norm, in place on the residual). This entry is the
last two in one kernel, statement for statement as the fused drafter hop computes them after its all-reduce
(``drafter_hop``'s module docstring), so the output and the residual are bit for bit those of the two launches:

* convolution: each coefficient, tap product, mask product and running sum rounded to BF16 where
  ``dflash2_block_conv`` rounds, unfused FP32 add / multiply in between;
* add + RMSNorm, flashinfer's statements for an 8192-wide BF16 row: ``h = x + r`` in FP32 (stored as the
  residual, rounded once); per flashinfer thread ``f`` (0..127) one FMA chain ``s = fma(h, h, s)`` from ``+0``
  over its 64 columns ``8 f + 1024 b + e`` (``b`` then ``e`` ascending); a butterfly over each warp of 32
  threads (lane offsets 1, 2, 4, 8, 16, which every lane ends at the same pairwise tree); the same over the 4
  warp sums padded with zeros; ``rstd = rsqrt.approx.ftz(fma(s, 2**-13, eps))``; ``y = (w + 0) * (h * rstd)``.

One program per row (``f`` its 128 lanes), every FP operation an inline PTX instruction (no contraction, no
reordering). The butterfly's pairwise tree is built with size-2 reductions over an order-preserving reshape.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

#: The row width and the convolution geometry this kernel is written for.
HIDDEN, TAPS, GROUP_SIZE = 8192, 2, 16
#: flashinfer threads per row and columns per thread.
THREADS, PER_THREAD = 128, 64


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _f32_fma(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;", "=r,r,r,r", [a, b, c], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _bf16_round(a):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 r; cvt.rn.bf16.f32 r, $1; cvt.f32.bf16 $0, r; }",
        "=r,r",
        [a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rsqrt_ftz(a):
    return tl.inline_asm_elementwise(
        "rsqrt.approx.ftz.f32 $0, $1;", "=r,r", [a], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _pair_sum(x, N: tl.constexpr):
    """``x[2 i] + x[2 i + 1]`` for ``i < N // 2`` (one add each), in index order."""
    return tl.sum(tl.reshape(x, (N // 2, 2)), axis=1)


@triton.jit
def _finish_h(x_ptr, delta_ptr, base_ptr, res_ptr, row64, cols, keep, has_prev, x_row_stride,
              delta_row_stride, delta_tap_stride, delta_group_stride, base_tap_stride, res_row_stride,
              GROUP: tl.constexpr):  # fmt: skip
    """``h = finish_conv(x)[cols] + residual[cols]`` in FP32, statement for statement (both passes)."""
    groups = cols // GROUP
    current = tl.load(x_ptr + row64 * x_row_stride + cols).to(tl.float32)
    shifted = tl.load(x_ptr + (row64 - 1) * x_row_stride + cols, mask=(cols >= 0) & has_prev, other=0.0).to(
        tl.float32
    )
    d0 = tl.load(delta_ptr + row64 * delta_row_stride + groups * delta_group_stride).to(tl.float32)
    d1 = tl.load(delta_ptr + row64 * delta_row_stride + delta_tap_stride + groups * delta_group_stride).to(
        tl.float32
    )
    b0 = tl.load(base_ptr + cols).to(tl.float32)
    b1 = tl.load(base_ptr + base_tap_stride + cols).to(tl.float32)
    r = tl.load(res_ptr + row64 * res_row_stride + cols).to(tl.float32)
    # The finish-side convolution, rounded where dflash2_block_conv rounds.
    acc = _bf16_round(_f32_mul(_bf16_round(_f32_add(b0, d0)), current))
    term = _bf16_round(_f32_mul(_bf16_round(_f32_add(b1, d1)), shifted))
    term = _bf16_round(_f32_mul(term, keep))
    acc = _bf16_round(_f32_add(acc, term))
    # flashinfer's residual add (FP32, kept unrounded for the norm).
    return _f32_add(acc, r)


@triton.jit
def _drafter_finish_norm_kernel(
    x_ptr,
    delta_ptr,
    base_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    eps,
    block_size,
    x_row_stride,
    delta_row_stride,
    delta_tap_stride,
    delta_group_stride,
    base_tap_stride,
    res_row_stride,
    out_row_stride,
    GROUP: tl.constexpr,
    LANES: tl.constexpr,
):
    row = tl.program_id(0)
    row64 = row.to(tl.int64)
    f = tl.arange(0, LANES)
    keep = tl.zeros([LANES], dtype=tl.float32) + (row % block_size >= 1).to(tl.float32)
    has_prev = row >= 1
    # Pass 1: each flashinfer thread's FMA chain over its 64 columns (blocks b, then e).
    chain = tl.zeros([LANES], dtype=tl.float32)
    for b in tl.static_range(8):
        for e in tl.static_range(8):
            h = _finish_h(
                x_ptr, delta_ptr, base_ptr, res_ptr, row64, 8 * f + 1024 * b + e, keep, has_prev,
                x_row_stride, delta_row_stride, delta_tap_stride, delta_group_stride, base_tap_stride,
                res_row_stride, GROUP,
            )  # fmt: skip
            chain = _f32_fma(h, h, chain)
    # Warp butterflies: every lane ends at the pairwise tree of its warp's 32 sums.
    s = _pair_sum(chain, LANES)
    s = _pair_sum(s, LANES // 2)
    s = _pair_sum(s, LANES // 4)
    s = _pair_sum(s, LANES // 8)
    s = _pair_sum(s, LANES // 16)
    # The 4 warp sums (padded with zeros, which add nothing to a non-negative sum).
    s = _pair_sum(s, LANES // 32)
    total = tl.sum(s, axis=0)  # the last pair: one add
    rstd = _rsqrt_ftz(_f32_fma(total, 1.0 / 8192.0, eps))
    # Every lane has read its residual columns before any is overwritten.
    tl.debug_barrier()
    # Pass 2: the same h (the same statements on the same inputs), the residual and the output.
    for b in tl.static_range(8):
        for e in tl.static_range(8):
            cols = 8 * f + 1024 * b + e
            h = _finish_h(
                x_ptr, delta_ptr, base_ptr, res_ptr, row64, cols, keep, has_prev,
                x_row_stride, delta_row_stride, delta_tap_stride, delta_group_stride, base_tap_stride,
                res_row_stride, GROUP,
            )  # fmt: skip
            w = tl.load(w_ptr + cols).to(tl.float32)
            y = _f32_mul(_f32_add(w, tl.zeros([LANES], dtype=tl.float32)), _f32_mul(h, rstd))
            # Lane f owns columns 8 f + 1024 b + e of this row: no other lane reads what it stores.
            tl.store(res_ptr + row64 * res_row_stride + cols, h.to(res_ptr.dtype.element_ty))
            tl.store(out_ptr + row64 * out_row_stride + cols, y.to(out_ptr.dtype.element_ty))


class DrafterFinishNorm(OpWrapper):
    """``out, residual <- RMSNorm(residual + finish_conv(x))``: a drafter hop's tail after its all-reduce."""

    ARCHS = frozenset({Arch.SM_107})

    note = """
    One Triton program per row (128 lanes = flashinfer's 128 threads of an 8192-wide row), 4 warps,
    FMA contraction off and every FP op inline PTX. ``residual`` is updated in place (as the built-in
    norm does); returns a fresh ``[T, 8192]`` normed tensor.

    Two passes over the row: the first computes ``h`` and the FMA chains, the second the same ``h``
    again from the same (unmodified) inputs, then stores the residual and the output; so the output
    multiplies the FP32 ``h``, as flashinfer and the fused drafter hop do.
    """

    def __call__(self, x, delta, base, residual, weight, eps: float, block_size: int) -> torch.Tensor:
        rows = x.shape[0]
        out = torch.empty_like(x)
        if rows == 0:
            return out
        _drafter_finish_norm_kernel[(rows,)](
            x,
            delta,
            base,
            residual,
            weight,
            out,
            float(eps),
            int(block_size),
            x.stride(0),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            base.stride(0),
            residual.stride(0),
            out.stride(0),
            GROUP=GROUP_SIZE,
            LANES=THREADS,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return out

    def reference(self, x, delta, base, residual, weight, eps: float, block_size: int) -> torch.Tensor:
        """FP64 conv + add + RMSNorm (unrounded), residual left unchanged; for the tolerance band only."""
        rows = x.shape[0]
        coefficients = base.double().unsqueeze(0) + delta.double().repeat_interleave(GROUP_SIZE, dim=-1)
        xd = x.double()
        position = torch.arange(rows, device=x.device) % block_size
        shifted = torch.zeros_like(xd)
        shifted[1:] = xd[:-1]
        conv = coefficients[:, 0] * xd + coefficients[:, 1] * shifted * (position >= 1).double().unsqueeze(-1)
        h = conv + residual.double()
        return (h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps) * weight.double()).to(x.dtype)

    def is_valid(self, x, delta, base, residual, weight, eps: float, block_size: int) -> None:
        assert x.dtype == delta.dtype == base.dtype == residual.dtype == weight.dtype == torch.bfloat16, "BF16"
        assert x.dim() == 2 and x.shape[1] == HIDDEN and x.stride(-1) == 1, f"x [T, {HIDDEN}], unit column stride"
        assert residual.shape == x.shape and residual.stride(-1) == 1, "residual like x"
        assert tuple(base.shape) == (TAPS, HIDDEN) and base.stride(-1) == 1, f"base [{TAPS}, {HIDDEN}]"
        assert tuple(delta.shape) == (x.shape[0], TAPS, HIDDEN // GROUP_SIZE), "delta [T, taps, groups]"
        assert tuple(weight.shape) == (HIDDEN,) and weight.is_contiguous(), f"weight [{HIDDEN}]"
        assert block_size >= 1


drafter_finish_norm = DrafterFinishNorm()


def takes(x, delta, base, residual, weight) -> bool:
    """Whether the kernel takes these operands (else the caller runs the built-in launches)."""
    return (
        x.dtype == delta.dtype == base.dtype == residual.dtype == weight.dtype == torch.bfloat16
        and x.dim() == 2
        and x.shape[1] == HIDDEN
        and x.stride(-1) == 1
        and residual.shape == x.shape
        and residual.stride(-1) == 1
        and tuple(base.shape) == (TAPS, HIDDEN)
        and base.stride(-1) == 1
        and tuple(delta.shape) == (x.shape[0], TAPS, HIDDEN // GROUP_SIZE)
        and tuple(weight.shape) == (HIDDEN,)
        and weight.is_contiguous()
    )
