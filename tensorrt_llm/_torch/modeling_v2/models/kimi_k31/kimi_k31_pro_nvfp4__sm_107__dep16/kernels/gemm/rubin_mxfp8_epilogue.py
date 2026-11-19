# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The catalog MXFP8 quantization as a Triton epilogue: E4M3 codes and R128c4 UE8M0 scales.

A producer kernel whose result is rounded to BF16 can hand the next MXFP8 GEMM its operand itself
instead of storing the BF16 rows for ``mxfp8_quantize`` to read back. These helpers issue, on each
1x32 group of BF16 values, the instructions of that quantizer (``cvt_warp_fp16_to_mxfp8``, built
without ``--use_fast_math``: its SASS flushes nothing), so the codes and the scale bytes are the
quantizer's, bit for bit, on every input:

* ``vecMax``: the maximum of ``|x|`` over the group's even elements and over its odd ones (a NaN
  skipped, as ``__hmax2`` skips it), then ``even > odd ? even : odd`` (``cuda_max`` of two BF16):
  a NaN only when every odd element is a NaN;
* ``sf = vecMax * 0x3B124925`` (``mul.rn.f32``: the quantizer's ``rcp.approx.ftz(448)`` as the
  compiler folds it, the FP32 nearest 1 / 448); the byte ``cvt.rp.satfinite.ue8m0x2.f32(sf)``;
* the multiplier: 0 for ``vecMax == 0``, 1 for byte 0, else ``exp2f(127 - byte)`` as the
  quantizer's SASS computes it: ``ex2.approx.ftz.f32`` with the argument halved and the result
  squared below -126;
* each value widened to FP32 by its bits, ``mul.rn.f32`` by the multiplier, then
  ``cvt.rn.satfinite.e4m3x2.f32``.

Every floating-point step is inline PTX, so neither Triton's conversions nor its flush-to-zero
choices take part. The scale of row ``m`` and group ``k`` goes to the R128c4 byte the dense
block-scaled GEMM reads (``r128c4_offset``); ``rubin_mxfp8_gemm`` takes the codes and that buffer
as its input operand.
"""

import triton
import triton.language as tl


@triton.jit
def mx_magnitude(yb):
    """The 15 magnitude bits of BF16 values as int32, a NaN as -1.

    A maximum of these is the maximum of ``|x|``'s bits; it skips NaN and is -1 only when every
    value is a NaN.
    """
    mag = yb.to(tl.int16, bitcast=True).to(tl.int32) & 0x7FFF
    return tl.where(mag > 0x7F80, -1, mag)


@triton.jit
def mx_amax(yb, GROUPS: tl.constexpr):
    """The quantizer's ``vecMax`` of each 1x32 group of BF16 ``yb`` ([GROUPS * 32]) as
    ``mx_magnitude`` bits (-1 for a NaN).

    The maximum over the even elements and over the odd ones (NaN skipped unless all are NaN), then
    ``even > odd ? even : odd``: a NaN only when every odd element is a NaN; a group whose even
    elements are all NaN takes the odd maximum.
    """
    mag = tl.reshape(mx_magnitude(yb), (GROUPS, 32))
    odd = (tl.arange(0, 32) % 2 == 1)[None, :]
    even_max = tl.max(tl.where(odd, -1, mag), axis=1)
    odd_max = tl.max(tl.where(odd, mag, -1), axis=1)
    return tl.where(odd_max < 0, -1, tl.maximum(even_max, odd_max))


@triton.jit
def mx_scale(amax):
    """``(UE8M0 byte, FP32 multiplier)`` of groups whose ``mx_amax`` is ``amax``.

    The quantizer's instructions in one PTX block: ``mul.rn.f32`` by 0x3B124925,
    ``cvt.rp.satfinite.ue8m0x2.f32``, ``setp.neu`` against 0, and ``exp2f(127 - byte)`` as
    ``ex2.approx.ftz.f32`` with the argument halved and the result squared below -126.
    """
    byte, mult = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32 a, e;
        .reg .b16 e2;
        .reg .f32 f, s, t, r, k, q;
        .reg .pred pq, pn, pe, pr;
        setp.lt.s32 pq, $2, 0;
        shl.b32 a, $2, 16;
        @pq mov.b32 a, 0x7fff0000;
        mov.b32 f, a;
        mul.rn.f32 s, f, 0f3B124925;
        cvt.rp.satfinite.ue8m0x2.f32 e2, s, s;
        cvt.u32.u16 e, e2;
        and.b32 e, e, 255;
        setp.neu.f32 pn, f, 0f00000000;
        setp.ne.u32 pe, e, 0;
        cvt.rn.f32.u32 t, e;
        mov.f32 k, 0f42FE0000;
        sub.rn.f32 t, k, t;
        setp.geu.f32 pr, t, 0fC2FC0000;
        @!pr mul.rn.f32 t, t, 0f3F000000;
        ex2.approx.ftz.f32 r, t;
        @!pr mul.rn.f32 r, r, r;
        mov.f32 q, 0f3F800000;
        @pe mov.f32 q, r;
        @!pn mov.f32 q, 0f00000000;
        mov.b32 $0, e;
        mov.b32 $1, q;
        }
        """,
        constraints="=r,=r,r",
        args=[amax],
        dtype=(tl.int32, tl.int32),
        is_pure=True,
        pack=1,
    )
    return byte, mult.to(tl.float32, bitcast=True)


@triton.jit
def mx_codes(yb, mult):
    """E4M3 codes of the BF16 values ``yb`` times their group's multiplier ``mult``.

    Each value enters as its bits, widened to FP32 by a shift (no conversion can flush a subnormal),
    then ``mul.rn.f32`` and ``cvt.rn.satfinite.e4m3x2.f32``, as the quantizer converts it.
    """
    code = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32 b, c;
        .reg .b16 c2;
        .reg .f32 x, q, z;
        shl.b32 b, $1, 16;
        mov.b32 x, b;
        mul.rn.f32 q, x, $2;
        mov.f32 z, 0f00000000;
        cvt.rn.satfinite.e4m3x2.f32 c2, z, q;
        cvt.u32.u16 c, c2;
        and.b32 $0, c, 255;
        }
        """,
        constraints="=r,r,f",
        args=[yb.to(tl.int16, bitcast=True).to(tl.int32), mult],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    return code.to(tl.uint8).to(tl.float8e4nv, bitcast=True)


@triton.jit
def r128c4_offset(row, kb, kb_tiles):
    """The R128c4 byte of the scale of row ``row``, group ``kb`` (``kb_tiles`` = pad4(K / 32) / 4).

    Rows in tiles of 128, scale columns in groups of 4: ``mxfp8_quantize``'s swizzled order.
    """
    return (
        ((row // 128) * kb_tiles + kb // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + kb % 4
    )
