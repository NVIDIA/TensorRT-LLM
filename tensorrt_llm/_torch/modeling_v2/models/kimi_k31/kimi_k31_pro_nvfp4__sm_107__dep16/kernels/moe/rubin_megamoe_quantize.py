# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 quantization of a MegaMoE chunk's rows straight into its bucket (K3.1 Rubin targets).

``fp4_quantize(x, global_scale, 16, False, False)`` (the engine's ``quantize_with_block_size``,
``cpp/tensorrt_llm/kernels/quantization.cuh``, with linear block scales) writing its two outputs
into rows ``[0, n)`` of a bucket's ``activation`` / ``activation_sf`` instead of new tensors. The
same launch empties the bucket's routing past those rows (expert ids -1, weights 0), which is what
the launcher's staging leaves there. Nothing else is written.

The bits are the C++ kernel's (built without fast math). Per 16-value block: ``amax`` is the
largest BF16 magnitude, compared as integers; the scale byte is ``e4m3(S * (amax * rcp(6)))``;
each value's code is ``e2m1(x * rcp(float(scale) * rcp(S)))``, or 0 in an all-zero block (``S``:
the global scale, ``rcp``: ``rcp.approx.ftz.f32``). The products, both reciprocals and both
conversions are the C++ kernel's own instructions as inline PTX (``mul.rn.f32``,
``rcp.approx.ftz.f32``, ``cvt.rn.satfinite.e4m3x2.f32``, and ``fp32_vec_to_e2m1``'s
``cvt.rn.satfinite.e2m1x2.f32`` packing of eight values into one word). A block holding a NaN is
not served: the C++ kernel's BF16 maximum skips it, this one does not.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: Values per block scale, and the blocks one program quantizes (hidden: a multiple of their
#: product).
SF_VEC = 16
BLOCKS_PER_PROGRAM = 32

# A block's scale byte ($0) and its values' output scale ($1) from its maximum ($2) and the global
# scale ($3): cvt_warp_fp16_to_fp4's arithmetic, instruction for instruction. (A jitted kernel
# reads a module string only as a constexpr.)
_SCALE_PTX = tl.constexpr("""
{
.reg .f32 c6, t, s, z, sfv, rs;
.reg .b16 h, lo, hi;
.reg .b32 hh;
.reg .pred nz;
mov.f32 z, 0f00000000;
mov.f32 c6, 0f40C00000;
rcp.approx.ftz.f32 c6, c6;
mul.rn.f32 t, $2, c6;
mul.rn.f32 s, $3, t;
cvt.rn.satfinite.e4m3x2.f32 h, z, s;
cvt.u32.u16 $0, h;
cvt.rn.f16x2.e4m3x2 hh, h;
mov.b32 {lo, hi}, hh;
cvt.f32.f16 sfv, lo;
rcp.approx.ftz.f32 rs, $3;
mul.rn.f32 t, sfv, rs;
rcp.approx.ftz.f32 t, t;
setp.neu.f32 nz, $2, z;
selp.f32 $1, t, z, nz;
}
""")

# One word ($0) of codes: eight consecutive values ($1..$8) times the output scale ($9), packed as
# fp32_vec_to_e2m1 packs them (value 2i in the low nibble of byte i, value 2i + 1 in its high one).
_E2M1_PTX = tl.constexpr("""
{
.reg .f32 f<8>;
.reg .b8 byte<4>;
mul.rn.f32 f0, $1, $9;
mul.rn.f32 f1, $2, $9;
mul.rn.f32 f2, $3, $9;
mul.rn.f32 f3, $4, $9;
mul.rn.f32 f4, $5, $9;
mul.rn.f32 f5, $6, $9;
mul.rn.f32 f6, $7, $9;
mul.rn.f32 f7, $8, $9;
cvt.rn.satfinite.e2m1x2.f32 byte0, f1, f0;
cvt.rn.satfinite.e2m1x2.f32 byte1, f3, f2;
cvt.rn.satfinite.e2m1x2.f32 byte2, f5, f4;
cvt.rn.satfinite.e2m1x2.f32 byte3, f7, f6;
mov.b32 $0, {byte0, byte1, byte2, byte3};
}
""")


@triton.jit
def _halves(t, BLOCKS: tl.constexpr, N: tl.constexpr):
    """``t [BLOCKS, 2, N]`` -> its even and odd entries along the last axis, ``[BLOCKS, 2, N/2]``."""
    return tl.split(tl.reshape(t, [BLOCKS, 2, N // 2, 2]))


@triton.jit(do_not_specialize=["rows"])
def _quantize_into_bucket(
    x_ptr,
    scale_ptr,
    act_ptr,
    sf_ptr,
    ids_ptr,
    weights_ptr,
    rows,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    """Program (r, p): row r's blocks [p * BLOCKS, (p + 1) * BLOCKS) when r < rows; else, for
    p == 0, row r's routing emptied. ``act_ptr`` is the activation as int32 words."""
    row = tl.program_id(0)
    part = tl.program_id(1)
    if row < rows:
        r = row.to(tl.int64)
        blk = part * BLOCKS + tl.arange(0, BLOCKS)
        x = tl.load(x_ptr + r * HIDDEN + blk[:, None] * 16 + tl.arange(0, 16)[None, :])
        mag = x.to(tl.int16, bitcast=True).to(tl.int32) & 0x7FFF
        amax = (tl.max(mag, axis=1) << 16).to(tl.float32, bitcast=True)
        gscale = tl.zeros([BLOCKS], dtype=tl.float32) + tl.load(scale_ptr)
        sf, scale = tl.inline_asm_elementwise(
            _SCALE_PTX,
            "=r,=f,f,f",
            [amax, gscale],
            dtype=(tl.int32, tl.float32),
            is_pure=True,
            pack=1,
        )
        # Value j of word w of each block, as eight [BLOCKS, 2] tensors.
        w8 = tl.reshape(x.to(tl.float32), [BLOCKS, 2, 8])
        e4, o4 = _halves(w8, BLOCKS, 8)  # values 0, 2, 4, 6 / 1, 3, 5, 7
        e2a, e2b = _halves(e4, BLOCKS, 4)  # 0, 4 / 2, 6
        o2a, o2b = _halves(o4, BLOCKS, 4)  # 1, 5 / 3, 7
        x0, x4 = _halves(e2a, BLOCKS, 2)
        x2, x6 = _halves(e2b, BLOCKS, 2)
        x1, x5 = _halves(o2a, BLOCKS, 2)
        x3, x7 = _halves(o2b, BLOCKS, 2)
        bscale = scale[:, None] + tl.zeros([BLOCKS, 2], dtype=tl.float32)
        words = tl.inline_asm_elementwise(
            _E2M1_PTX,
            "=r,f,f,f,f,f,f,f,f,f",
            [
                tl.reshape(x0, [BLOCKS, 2]),
                tl.reshape(x1, [BLOCKS, 2]),
                tl.reshape(x2, [BLOCKS, 2]),
                tl.reshape(x3, [BLOCKS, 2]),
                tl.reshape(x4, [BLOCKS, 2]),
                tl.reshape(x5, [BLOCKS, 2]),
                tl.reshape(x6, [BLOCKS, 2]),
                tl.reshape(x7, [BLOCKS, 2]),
                bscale,
            ],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        out = act_ptr + r * (HIDDEN // 8) + blk[:, None] * 2 + tl.arange(0, 2)[None, :]
        tl.store(out, words)
        tl.store(sf_ptr + r * (HIDDEN // 16) + blk, sf.to(tl.uint8))
    elif part == 0:
        k = tl.arange(0, TOP_K)
        tl.store(ids_ptr + row * TOP_K + k, tl.full([TOP_K], -1, tl.int32))
        tl.store(weights_ptr + row * TOP_K + k, tl.zeros([TOP_K], dtype=tl.float32))


def quantize_into_bucket(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    activation: torch.Tensor,
    activation_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    """Launch on the current stream: ``x``'s ``n`` rows into rows ``[0, n)`` of ``activation`` /
    ``activation_sf``, ids -1 and weights 0 into rows ``[n, T)`` of ``topk_idx`` /
    ``topk_weights`` (``T`` their rows). One program per (row, ``BLOCKS_PER_PROGRAM`` blocks).

    Nothing is re-validated here: the shapes, dtypes and density the entry's ``is_valid`` states.
    """
    rows, hidden = x.shape
    grid = (topk_idx.shape[0], hidden // (SF_VEC * BLOCKS_PER_PROGRAM))
    _quantize_into_bucket[grid](
        x,
        global_scale,
        activation.view(torch.int32),
        activation_sf,
        topk_idx,
        topk_weights,
        rows,
        HIDDEN=hidden,
        TOP_K=topk_idx.shape[1],
        BLOCKS=BLOCKS_PER_PROGRAM,
    )


__all__ = ["BLOCKS_PER_PROGRAM", "SF_VEC", "quantize_into_bucket"]
