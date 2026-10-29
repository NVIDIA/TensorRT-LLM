# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
#
# This kernel reproduces the arithmetic of FlashMLA's sm100 sparse fp8 decode and its split-K
# combine (https://github.com/deepseek-ai/FlashMLA, commit
# 15f13e5030374295491c5ce31b02d7e63a7772c6), which is distributed under the MIT License:
#
# MIT License
#
# Copyright (c) 2025 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""The CuTe DSL kernel behind `cute_dsl_ds_mla_sparse_decode`: sparse MLA decode of 8 or 16
query heads over the fp8_ds_mla latent pool (sm_100f).

One query row per sequence, top-k 2048 selected pool rows. A 656-byte pool row holds 512 e4m3
codes, four fp32 power-of-two scales (one per 128 codes) and 64 bf16 rope values. The
arithmetic is FlashMLA's sm100 ``head64`` sparse fp8 decode (V32 layout) and its split-K
combine, operation for operation and in the same order; the entry's ``note`` maps every stage
to FlashMLA's source. What differs is the work layout: the heads are not padded to 64, a
sequence's 2048 selections split over a cluster of 1, 2, 4 or 8 CTAs (a fixed partition,
where FlashMLA's scheduler partitions by load), and the split-K partials merge through the
cluster's shared memory instead of a second kernel.

Per CTA: a contiguous 2048/splits slice of one row's selections, in tiles of 64 tokens. The
QK MMA is swapped, S^T = K Q^T with M = 64 tokens and N = 16 heads (8 heads run with 8 zero
query rows), as two fp32 accumulators -- FlashMLA's dual-GEMM half-sums. PV is
O^T = V^T P^T, M = 128 latent columns per stripe, N = 16. Warp roles:

* W9: the query TMA, the index ring, the odd raw slots' TMA gathers.
* W10: the even raw slots' TMA gathers (whole 656-byte rows, ``tile::gather4``).
* W12-15: dequantization into the bf16 K/V tile (SW128, K-major).
* W8: the QK MMAs. W11: the PV MMAs.
* W0-3: the online softmax, one token per thread.
* W4-7: the O rescale, the softmax denominator (FlashMLA's summation order), the epilogue.
"""

from __future__ import annotations

import struct
from collections.abc import Callable

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives.descriptors as _d
import torch
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

D_NOPE = 512
D_ROPE = 64
D_QK = D_NOPE + D_ROPE
ROW_BYTES = 656
ROW_WORDS = ROW_BYTES // 4
SCALE_WORD = D_NOPE // 4
TOPK = 2048
BT = 64  # tokens per tile: the QK MMA is M = 64 (rows on TMEM lanes 32*(r//16) + r%16)
N_MMA = 16
NUM_STRIPES = 4
CHUNK_BYTES = BT * 128  # one 64-column bf16 chunk (or one 128-code e4m3 stripe) of a tile
CHUNK_WORDS = CHUNK_BYTES // 4
ROPE_CHUNK = 8
BUF_WORDS = 9 * CHUNK_WORDS  # one tile: 8 NoPE chunks + the rope chunk
NUM_BUFS = 2
KV_WORDS = NUM_BUFS * BUF_WORDS
Q_CHUNK_BYTES = N_MMA * 128
RAW_TOK = 16  # tokens per raw slot (one whole 656-byte row each, TMA box of 82 int64)
RAW_SLOT_BYTES = RAW_TOK * ROW_BYTES  # bytes one slot's TMA transactions deliver
RAW_GROUP_BYTES = 2688  # one gather4 (4 rows of 656 B) padded to a 128-byte boundary
RAW_GROUP_WORDS = RAW_GROUP_BYTES // 4
RAW_SLOT_WORDS = (RAW_TOK // 4) * RAW_GROUP_WORDS
RAW_PER_TILE = BT // RAW_TOK
NUM_RAW = 5
ST_S = 2
ST_P = 2
ST_I = 4
ST_C = 1
S_COLS = 2 * N_MMA  # one S stage: FlashMLA's two QK half-sums, 16 columns each
TMEM_O = ST_S * S_COLS
TMEM_COLS = 128
PF_STRIDE = 66  # fp32 P row stride (words) of the li hand-off: conflict-free both ways
THREADS = 512
W_CORR0 = 4
W_QK = 8
W_IDX = 9
W_TMA = 10
W_PV = 11
W_DQ0 = 12
BAR_SOFTMAX = 1
BAR_CORR = 2
MAX_INIT = -1.0e30
LOG2_E_F32 = 1.44269504  # FlashMLA's `LOG_2_E`, a float constant
SUPPORTED_HEADS = (8, 16)
SUPPORTED_SPLITS = (1, 2, 4, 8)
SW128 = prims.Tcgen05SmemSwizzle.SWIZZLE_128B


def f32(x: float) -> float:
    """Round a Python float to fp32."""
    return struct.unpack("f", struct.pack("f", x))[0]


def scale_log2_of(softmax_scale: float) -> float:
    """FlashMLA's `sm_scale_div_log2`: the fp32 product `float(scale) * 1.44269504f`."""
    return f32(f32(softmax_scale) * f32(LOG2_E_F32))


# --------------------------------------------------------------------------- small helpers
@dsl_user_op
def _smem_desc(start_address, leading_byte_offset=0, stride_byte_offset=0, *, loc=None, ip=None):
    """A tcgen05 SMEM matrix descriptor (128B swizzle), byte offsets."""
    if isinstance(start_address, (_d.Array, cutlass.Pointer)):
        addr = cutlass.Int32(_d.llvm.ptrtoint(cutlass.Int32.mlir_type, start_address.ir_value()))
    else:
        addr = cutlass.Int32(start_address)
    lbo = cutlass.Int32(_d._drop_low_bits(leading_byte_offset, 4, "leading_byte_offset"))
    sbo = cutlass.Int32(_d._drop_low_bits(stride_byte_offset, 4, "stride_byte_offset"))
    return _d.Tcgen05SmemDesc(
        cutlass.Int64(
            _d._nvvm.tcgen05_mma_smem_desc_v2(
                addr >> 4,
                lbo,
                sbo,
                cutlass.Int8(1),
                cutlass.Int8(0),
                cutlass.Boolean(0),
                cutlass.Boolean(0),
                cutlass.Int8(int(SW128)),
                loc=loc,
                ip=ip,
            )
        )
    )


def _ptx1(code, *args):
    return cutlass.Float32(
        prims.inline_ptx_hl(code, write_only_types=[cutlass.Float32], read_only_args=list(args))
    )


def _ex2(x):  # exp2f under --use_fast_math
    return _ptx1("ex2.approx.ftz.f32 {$w0}, {$r0};", x)


def _lg2(x):  # log2f under --use_fast_math
    return _ptx1("lg2.approx.ftz.f32 {$w0}, {$r0};", x)


def _rcp(x):  # __fdividef(1.0f, x)
    return _ptx1("div.approx.ftz.f32 {$w0}, 0f3F800000, {$r0};", x)


def _fma_rn(a, b, c):  # ku::float2_fma: fma.rn.f32x2 (inline asm, no ftz)
    return _ptx1("fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};", a, b, c)


def _fma_ftz(a, b, c):  # fmaf under --use_fast_math
    return _ptx1("fma.rn.ftz.f32 {$w0}, {$r0}, {$r1}, {$r2};", a, b, c)


def _mul_rn(a, b):  # ku::float2_mul: mul.f32x2 (inline asm, no ftz)
    return _ptx1("mul.rn.f32 {$w0}, {$r0}, {$r1};", a, b)


def _mul_ftz(a, b):  # float * under --use_fast_math
    return _ptx1("mul.rn.ftz.f32 {$w0}, {$r0}, {$r1};", a, b)


def _sub_ftz(a, b):
    return _ptx1("sub.rn.ftz.f32 {$w0}, {$r0}, {$r1};", a, b)


def _add_ftz(a, b):  # float + under --use_fast_math
    return _ptx1("add.rn.ftz.f32 {$w0}, {$r0}, {$r1};", a, b)


def _add_rn(a, b):  # ku::float2_add: add.f32x2 (inline asm, no ftz)
    return _ptx1("add.rn.f32 {$w0}, {$r0}, {$r1};", a, b)


def _mul_into_zero(a, b):  # `0.0f += a * b`, contracted: fma.rn.ftz(a, b, +0)
    return _ptx1("fma.rn.ftz.f32 {$w0}, {$r0}, {$r1}, 0f00000000;", a, b)


def _fmax(a, b):
    return _ptx1("max.ftz.f32 {$w0}, {$r0}, {$r1};", a, b)


def _bf16x2(lo, hi) -> cutlass.Int32:
    """cvt.rn.bf16x2.f32 (`__float22bfloat162_rn`): lo in bits 0-15."""
    return cutlass.Int32(
        prims.inline_ptx_hl(
            "cvt.rn.bf16x2.f32 {$w0}, {$r0}, {$r1};",
            write_only_types=[cutlass.Int32],
            read_only_args=[hi, lo],
        )
    )


def _scales_to_bf16x2(lo, hi) -> cutlass.Int32:
    """Two fp32 scales (as their bit patterns) to bf16x2, lo in bits 0-15: `cvt.rz.ue8m0x2.f32`
    (`__nv_cvt_float2_to_e8m0x2(.., __NV_NOSAT, cudaRoundZero)`), then `cvt.rn.bf16x2.ue8m0x2`
    (`__nv_cvt_e8m0x2_to_bf162raw`)."""
    return cutlass.Int32(
        prims.inline_ptx_hl(
            "{\n\t.reg .b16 e;\n\tcvt.rz.ue8m0x2.f32 e, {$r1}, {$r0};\n\t"
            "cvt.rn.bf16x2.ue8m0x2 {$w0}, e;\n\t}",
            write_only_types=[cutlass.Int32],
            read_only_args=[lo, hi],
        )
    )


def _sel(flag, a, b):
    """flag (Int32, nonzero = true) ? a : b, fp32 registers (selp)."""
    return _ptx1(
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, {$r0}, 0;\n\tselp.f32 {$w0}, {$r1}, {$r2}, p;\n\t}",
        flag,
        a,
        b,
    )


def _sel_i32(flag, a, b):
    """flag ? a : b (Int32 registers)."""
    return cutlass.Int32(
        prims.inline_ptx_hl(
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, {$r0}, 0;\n\tselp.b32 {$w0}, {$r1}, {$r2}, p;\n\t}",
            write_only_types=[cutlass.Int32],
            read_only_args=[flag, a, b],
        )
    )


# Constants live in the PTX text: inline-asm float operands must be registers.
F32_NEG_INF = "0fFF800000"
F32_POS_INF = "0f7F800000"
F32_ZERO = "0f00000000"
F32_SIX = "0f40C00000"


def _cmp_i32(op, a, lit) -> cutlass.Int32:
    """1 if (a <op> lit) for an fp32 PTX literal, else 0."""
    return cutlass.Int32(
        prims.inline_ptx_hl(
            "{\n\t.reg .pred p;\n\tsetp."
            + op
            + ".f32 p, {$r0}, "
            + lit
            + ";\n\tselp.s32 {$w0}, 1, 0, p;\n\t}",
            write_only_types=[cutlass.Int32],
            read_only_args=[a],
        )
    )


def _sel_lit(flag, a, lit):
    """flag ? a : lit (fp32 PTX literal)."""
    return _ptx1(
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, {$r0}, 0;\n\t"
        "selp.f32 {$w0}, {$r1}, " + lit + ", p;\n\t}",
        flag,
        a,
    )


def _shfl_bfly(v, offset):
    return cutlass.Float32(
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF, val=v, offset=offset, mask_and_clamp=0x1F, kind=prims.Shfl.BFLY
        )
    )


def _pick_word(raw4, half, w: int):
    """raw4[half * 2 + w] for a runtime half in {0, 1}."""
    return _sel_i32(half, raw4[2 + w], raw4[w])


def _max_scatter16(vals, lane):
    """Warp max of 16 per-lane values; returns, on lane l, the max of value (l >> 1) & 15 over
    all 32 lanes (16 shuffles instead of 80)."""
    v = list(vals)
    n = 16
    for off in (16, 8, 4, 2):
        up = (lane >> cutlass.Int32(off.bit_length() - 1)) & cutlass.Int32(1)
        half = n // 2
        nv = []
        for i in range(half):
            send = _sel(up, v[i], v[half + i])
            keep = _sel(up, v[half + i], v[i])
            other = _shfl_bfly(send, off)
            nv.append(_fmax(keep, other))
        v = nv
        n = half
    return _fmax(v[0], _shfl_bfly(v[0], 1))


def _butterfly_sum(terms):
    """The sum a xor butterfly over lanes 0..n-1 (n a power of two) produces: pairs i, i + n/2
    first, then halving."""
    while len(terms) > 1:
        hn = len(terms) // 2
        terms = [_add_ftz(terms[i], terms[i + hn]) for i in range(hn)]
    return terms[0]


@cute.jit
def _issue_tile_raw(
    kt, st, parity: cutlass.Constexpr[int], lane, sIdx, raw_full, raw_empty, sRaw, tma_raw
) -> None:
    """Gather the raw slots q = parity, parity + 2 of tile kt (16 whole 656-byte rows each)
    into the shared ring; lanes 0..3 issue one gather4 each."""
    rl = lane % cutlass.Int32(RAW_TOK // 4)
    for qq in cutlass.range_constexpr(RAW_PER_TILE // 2):
        q: cutlass.Constexpr[int] = 2 * qq + parity
        rows4 = (
            sIdx.data_ptr()
            + st * cutlass.Int32(BT)
            + cutlass.Int32(q * RAW_TOK)
            + rl * cutlass.Int32(4)
        ).load(count=4, alignment=16)
        g = kt * cutlass.Int32(RAW_PER_TILE) + cutlass.Int32(q)
        slot = g % cutlass.Int32(NUM_RAW)
        gph = (g // cutlass.Int32(NUM_RAW)) % cutlass.Int32(2)
        _wait(raw_empty.data_ptr() + slot, gph ^ cutlass.Int32(1))
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(raw_full.data_ptr() + slot, RAW_SLOT_BYTES)
        prims.bar_warp_sync(0xFFFFFFFF)
        if lane < cutlass.Int32(RAW_TOK // 4):
            _tma_gather4_lanes(
                sRaw.data_ptr()
                + slot * cutlass.Int32(RAW_SLOT_WORDS)
                + rl * cutlass.Int32(RAW_GROUP_WORDS),
                tma_raw.get_ptr(),
                raw_full.data_ptr() + slot,
                cutlass.Int32(0),
                rows4[0],
                rows4[1],
                rows4[2],
                rows4[3],
            )


@cute.jit
def _wait(bar, phase):
    while not prims.mbarrier_try_wait_parity(bar, phase):
        pass


@cute.jit
def _tma_gather4_lanes(smem_dst, tma_desc, mbarrier, col, row0, row1, row2, row3) -> None:
    """Every lane issues its own `tile::gather4` TMA transaction (4 rows of one box): one
    warp instruction moves 128 rows."""
    local_barrier = prims.cvta_to(mbarrier, prims.CvtaSpace.SHARED)
    prims.inline_ptx_hl(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.L2::cache_hint"
        " [{$r0}], [{$r1}, {{$r2}, {$r3}, {$r4}, {$r5}, {$r6}}], [{$r7}], {$r8};",
        read_only_args=[
            smem_dst,
            tma_desc,
            col,
            row0,
            row1,
            row2,
            row3,
            local_barrier,
            cutlass.Int64(0),
        ],
    )


# ----------------------------------------------------------------------------------- kernel
@cute.kernel
def _dsmla_decode_kernel(
    tma_q: cutlass.GridConstant[cuda.TensorMap],
    tma_raw: cutlass.GridConstant[cuda.TensorMap],
    indices: cute.Tensor,
    out: cute.Tensor,
    scale_log2: cutlass.Float32,
    num_heads: cutlass.Constexpr[int],
    num_splits: cutlass.Constexpr[int],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    warp = cute.arch.warp_idx()
    lane = tidx % cutlass.Int32(32)
    split_x, _, row = cute.arch.block_idx()
    split = split_x
    if cutlass.const_expr(num_splits > 1):
        split = cute.arch.block_idx_in_cluster()
    tokens_per_cta: cutlass.Constexpr[int] = TOPK // num_splits
    num_kt: cutlass.Constexpr[int] = tokens_per_cta // BT
    tok_base = row * cutlass.Int32(TOPK) + split * cutlass.Int32(tokens_per_cta)

    # ---- shared memory ------------------------------------------------------------------
    # The KV tile is typed fp32 so that it doubles as the fp32 split-K partial afterwards;
    # dequantized bf16 pairs are stored as bit-cast 32-bit words.
    sKV = cutlass.Array(cutlass.Float32, KV_WORDS, space=cutlass.AddressSpace.smem, alignment=1024)
    sRaw = cutlass.Array(
        cutlass.Int32, NUM_RAW * RAW_SLOT_WORDS, space=cutlass.AddressSpace.smem, alignment=1024
    )
    sQ = cutlass.Array(
        cutlass.Int32, 9 * Q_CHUNK_BYTES // 4, space=cutlass.AddressSpace.smem, alignment=1024
    )
    sP = cutlass.Array(
        cutlass.Int16, ST_P * N_MMA * 64, space=cutlass.AddressSpace.smem, alignment=1024
    )
    sIdx = cutlass.Array(cutlass.Int32, ST_I * BT, space=cutlass.AddressSpace.smem, alignment=16)
    sMax = cutlass.Array(
        cutlass.Float32, 2 * 4 * N_MMA, space=cutlass.AddressSpace.smem, alignment=16
    )
    sCorr = cutlass.Array(
        cutlass.Float32, ST_C * N_MMA, space=cutlass.AddressSpace.smem, alignment=16
    )
    sFlag = cutlass.Array(cutlass.Int32, ST_C, space=cutlass.AddressSpace.smem, alignment=16)
    sFin = cutlass.Array(cutlass.Float32, N_MMA, space=cutlass.AddressSpace.smem, alignment=16)
    sLse = cutlass.Array(cutlass.Float32, N_MMA, space=cutlass.AddressSpace.smem, alignment=16)
    # fp32 P of the current tile [head][token] for the denominator chains, and their partials.
    sPf = cutlass.Array(
        cutlass.Float32, N_MMA * PF_STRIDE, space=cutlass.AddressSpace.smem, alignment=16
    )
    sLi = cutlass.Array(cutlass.Float32, 2 * N_MMA, space=cutlass.AddressSpace.smem, alignment=16)
    q_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    idx_full = cutlass.Array(cutlass.Int64, ST_I, space=cutlass.AddressSpace.smem)
    idx_empty = cutlass.Array(cutlass.Int64, ST_I, space=cutlass.AddressSpace.smem)
    raw_full = cutlass.Array(cutlass.Int64, NUM_RAW, space=cutlass.AddressSpace.smem)
    raw_empty = cutlass.Array(cutlass.Int64, NUM_RAW, space=cutlass.AddressSpace.smem)
    kv_full = cutlass.Array(cutlass.Int64, NUM_BUFS, space=cutlass.AddressSpace.smem)
    kv_empty = cutlass.Array(cutlass.Int64, NUM_BUFS, space=cutlass.AddressSpace.smem)
    s_full = cutlass.Array(cutlass.Int64, ST_S, space=cutlass.AddressSpace.smem)
    s_empty = cutlass.Array(cutlass.Int64, ST_S, space=cutlass.AddressSpace.smem)
    p_full = cutlass.Array(cutlass.Int64, ST_P, space=cutlass.AddressSpace.smem)
    p_empty = cutlass.Array(cutlass.Int64, ST_P, space=cutlass.AddressSpace.smem)
    c_full = cutlass.Array(cutlass.Int64, ST_C, space=cutlass.AddressSpace.smem)
    c_empty = cutlass.Array(cutlass.Int64, ST_C, space=cutlass.AddressSpace.smem)
    o_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    o_empty = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    fin_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    if warp == 0:
        if prims.elect_sync():
            prims.mbarrier_init(q_full, 1)
            for i in cutlass.range_constexpr(ST_I):
                prims.mbarrier_init(idx_full.data_ptr() + i, 1)
                prims.mbarrier_init(idx_empty.data_ptr() + i, 6)
            for i in cutlass.range_constexpr(NUM_RAW):
                prims.mbarrier_init(raw_full.data_ptr() + i, 1)
                prims.mbarrier_init(raw_empty.data_ptr() + i, 4)
            for b in cutlass.range_constexpr(NUM_BUFS):
                prims.mbarrier_init(kv_full.data_ptr() + b, 4 * RAW_PER_TILE)
                prims.mbarrier_init(kv_empty.data_ptr() + b, 1)
            for i in cutlass.range_constexpr(ST_S):
                prims.mbarrier_init(s_full.data_ptr() + i, 1)
                prims.mbarrier_init(s_empty.data_ptr() + i, 4)
            for i in cutlass.range_constexpr(ST_P):
                prims.mbarrier_init(p_full.data_ptr() + i, 4)
                prims.mbarrier_init(p_empty.data_ptr() + i, 1)
            for i in cutlass.range_constexpr(ST_C):
                prims.mbarrier_init(c_full.data_ptr() + i, 4)
                prims.mbarrier_init(c_empty.data_ptr() + i, 4)
            prims.mbarrier_init(o_full, 1)
            prims.mbarrier_init(o_empty, 4)
            prims.mbarrier_init(fin_full, 1)
    prims.fence_mbarrier_init()
    if cutlass.const_expr(num_splits > 1):
        prims.barrier_cluster_arrive_relaxed()
    if warp == W_PV:
        prims.tcgen05_alloc(tmem_i32, TMEM_COLS, group="cta_1")
    if cutlass.const_expr(num_splits > 1):
        prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    tmem_raw = tmem_i32.load()
    if warp == W_PV:
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")
    # Everything below reads what the preceding kernels wrote (query, indices, pool rows).
    prims.griddepcontrol(prims.GridDepAction.WAIT)

    # =====================================================================================
    # W9: query TMA (9 chunks of 64 bf16 x heads, SW128) and the per-tile index ring.
    # =====================================================================================
    if warp == W_IDX:
        if cutlass.const_expr(num_heads < N_MMA):
            # Rows num_heads..15 of every query chunk are zero (8-head mode runs N = 16).
            zero = cutlass.Vector.from_elements(
                tuple(cutlass.Int32(0) for _ in range(4)), dtype=cutlass.Int32
            )
            for c in cutlass.range_constexpr(9):
                for rep_ in cutlass.range_constexpr((N_MMA - num_heads) * 128 // 16 // 32):
                    off = c * (Q_CHUNK_BYTES // 4) + num_heads * 32 + (lane + rep_ * 32) * 4
                    (sQ.data_ptr() + off).store(zero, alignment=16)
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            prims.bar_warp_sync(0xFFFFFFFF)
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(q_full, 9 * num_heads * 128)
            for c in cutlass.range_constexpr(9):
                prims.cp_async_bulk_tensor_shared_cta_global(
                    sQ.data_ptr() + c * (Q_CHUNK_BYTES // 4),
                    tma_q.get_ptr(),
                    [c * 64, 0, row],
                    q_full,
                )
        idx_ptr = indices.iterator.raw_ptr()
        for kt in cutlass.range(0, num_kt, 1, unroll=1):
            st = kt % cutlass.Int32(ST_I)
            ph = (kt // cutlass.Int32(ST_I)) % cutlass.Int32(2)
            _wait(idx_empty.data_ptr() + st, ph ^ cutlass.Int32(1))
            if lane < cutlass.Int32(BT // 4):
                vals = (idx_ptr + tok_base + kt * cutlass.Int32(BT) + lane * cutlass.Int32(4)).load(
                    count=4, alignment=16
                )
                (sIdx.data_ptr() + st * cutlass.Int32(BT) + lane * cutlass.Int32(4)).store(
                    vals, alignment=16
                )
            prims.bar_warp_sync(0xFFFFFFFF)
            if prims.elect_sync():
                prims.mbarrier_arrive(idx_full.data_ptr() + st)
            # W9 also issues the odd raw slots of this tile (W10 the even ones): the TMA unit
            # serves one warp's requests one after another, two warps' in parallel.
            _issue_tile_raw(kt, st, 1, lane, sIdx, raw_full, raw_empty, sRaw, tma_raw)
            prims.bar_warp_sync(0xFFFFFFFF)
            if prims.elect_sync():
                prims.mbarrier_arrive(idx_empty.data_ptr() + st)

    # =====================================================================================
    # W10: TMA gathers of the even raw slots (16 whole 656-byte rows each) into the 5-slot raw
    # ring. A -1 index is an out-of-bounds row: TMA zero-fills it, codes and scales (FlashMLA
    # sets `tma_coords = -1` and the scale bytes 0; the token is masked to -inf either way).
    # =====================================================================================
    if warp == W_TMA:
        for kt in cutlass.range(0, num_kt, 1, unroll=1):
            st = kt % cutlass.Int32(ST_I)
            ph = (kt // cutlass.Int32(ST_I)) % cutlass.Int32(2)
            _wait(idx_full.data_ptr() + st, ph)
            _issue_tile_raw(kt, st, 0, lane, sIdx, raw_full, raw_empty, sRaw, tma_raw)
            prims.bar_warp_sync(0xFFFFFFFF)
            if prims.elect_sync():
                prims.mbarrier_arrive(idx_empty.data_ptr() + st)

    # =====================================================================================
    # W12-15: dequantization, eight threads per token row. bf16(e4m3) * bf16(scale) with the
    # scale rounded toward zero to a power of two (FlashMLA: e8m0 RZ, then e8m0 -> bf16), in
    # the SW128 K-major layout the QK MMA reads and the PV MMA reads MN-major; the rope is
    # copied as it is.
    # =====================================================================================
    if warp >= W_DQ0:
        dq = tidx - cutlass.Int32(W_DQ0 * 32)
        t = dq >> cutlass.Int32(3)  # token within a 16-token raw slot
        part = dq & cutlass.Int32(7)  # 16-byte pieces part, part+8, part+16, part+24 of the codes
        for kt in cutlass.range(0, num_kt, 1, unroll=1):
            b = kt % cutlass.Int32(NUM_BUFS)
            u = kt // cutlass.Int32(NUM_BUFS)
            kvb = b * cutlass.Int32(BUF_WORDS)
            for q in cutlass.range_constexpr(RAW_PER_TILE):
                g = kt * cutlass.Int32(RAW_PER_TILE) + cutlass.Int32(q)
                slot = g % cutlass.Int32(NUM_RAW)
                gph = (g // cutlass.Int32(NUM_RAW)) % cutlass.Int32(2)
                _wait(raw_full.data_ptr() + slot, gph)
                if cutlass.const_expr(q == 0):
                    _wait(kv_empty.data_ptr() + b, (u % cutlass.Int32(2)) ^ cutlass.Int32(1))
                rrow = (
                    slot * cutlass.Int32(RAW_SLOT_WORDS)
                    + (t >> cutlass.Int32(2)) * cutlass.Int32(RAW_GROUP_WORDS)
                    + (t & cutlass.Int32(3)) * cutlass.Int32(ROW_WORDS)
                )
                trow = cutlass.Int32(q * RAW_TOK) + t  # row of the 64-token tile
                sw = trow % cutlass.Int32(8)
                sc = (sRaw.data_ptr() + rrow + cutlass.Int32(SCALE_WORD)).load(
                    count=4, alignment=16
                )
                # FlashMLA's two conversions: the fp32 scales to e8m0 toward zero, then to bf16.
                pairs = [_scales_to_bf16x2(sc[0], sc[1]), _scales_to_bf16x2(sc[2], sc[3])]
                for m in cutlass.range_constexpr(NUM_STRIPES):
                    sbits = (pairs[m // 2] >> cutlass.Int32(16 * (m % 2))) & cutlass.Int32(0xFFFF)
                    scale2 = sbits | (sbits << cutlass.Int32(16))
                    k = part + cutlass.Int32(8 * m)  # 16-code piece of the row
                    raw4 = (sRaw.data_ptr() + rrow + k * cutlass.Int32(4)).load(
                        count=4, alignment=16
                    )
                    chunk = k >> cutlass.Int32(2)
                    # Threads part and part+4 of a token write the same piece slots of two
                    # chunks 8 KB apart (same banks): the upper four take the halves in the
                    # opposite order so that each store instruction hits 8 distinct bank groups.
                    flip = part >> cutlass.Int32(2)
                    for hh in cutlass.range_constexpr(2):
                        half = cutlass.Int32(hh) ^ flip
                        packed = []
                        for w in cutlass.range_constexpr(2):
                            word = _pick_word(raw4, half, w)
                            lo = prims.cvt_e4m3x2_word_to_bf16x2(word, 0)
                            hi = prims.cvt_e4m3x2_word_to_bf16x2(word, 16)
                            packed.append(prims.mul_bf16x2(lo, scale2))
                            packed.append(prims.mul_bf16x2(hi, scale2))
                        vec = cutlass.Vector.from_elements(tuple(packed), dtype=cutlass.Int32)
                        piece = (k & cutlass.Int32(3)) * cutlass.Int32(2) + half
                        dst = (
                            kvb
                            + chunk * cutlass.Int32(CHUNK_WORDS)
                            + trow * cutlass.Int32(32)
                            + ((piece ^ sw) * cutlass.Int32(4))
                        )
                        (sKV.data_ptr() + dst).store(vec.bitcast(cutlass.Float32), alignment=16)
                # Rope: 8 pieces of 8 bf16, copied as they are.
                rope4 = (
                    sRaw.data_ptr() + rrow + cutlass.Int32(SCALE_WORD + 4) + part * cutlass.Int32(4)
                ).load(count=4, alignment=16)
                rdst = (
                    kvb
                    + cutlass.Int32(ROPE_CHUNK * CHUNK_WORDS)
                    + trow * cutlass.Int32(32)
                    + ((part ^ sw) * cutlass.Int32(4))
                )
                (sKV.data_ptr() + rdst).store(rope4.bitcast(cutlass.Float32), alignment=16)
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                prims.bar_warp_sync(0xFFFFFFFF)
                if prims.elect_sync():
                    prims.mbarrier_arrive(raw_empty.data_ptr() + slot)
                    prims.mbarrier_arrive(kv_full.data_ptr() + b)

    # =====================================================================================
    # W8: QK. S^T[tok, h] = K[tok, :] . Q[h, :] as FlashMLA's two half-sums (below), each
    # its own fp32 accumulator in TMEM.
    # =====================================================================================
    if warp == W_QK:
        idesc_qk = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=cutlass.BFloat16,
            b_dtype=cutlass.BFloat16,
            n_dim=N_MMA,
            m_dim=BT,
        )
        _wait(q_full, 0)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        for kt in cutlass.range(0, num_kt, 1, unroll=1):
            sst = kt % cutlass.Int32(ST_S)
            sph = (kt // cutlass.Int32(ST_S)) % cutlass.Int32(2)
            _wait(s_empty.data_ptr() + sst, sph ^ cutlass.Int32(1))
            # FlashMLA's dual GEMM (V32 layout): half 0 = rope dims 0-31, then NoPE chunks
            # 0, 2, 4, 6; half 1 = rope dims 32-63, then chunks 1, 3, 5, 7. Each half is its own
            # fp32 accumulator, K = 16 per MMA in that order; the softmax adds the two.
            s_tm0 = tmem_raw + sst * cutlass.Int32(S_COLS)
            b = kt % cutlass.Int32(NUM_BUFS)
            u = kt // cutlass.Int32(NUM_BUFS)
            kvb = sKV.data_ptr() + b * cutlass.Int32(BUF_WORDS)
            _wait(kv_full.data_ptr() + b, u % cutlass.Int32(2))
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            dkr = _smem_desc(kvb + ROPE_CHUNK * CHUNK_WORDS, 16, 1024)
            dqr = _smem_desc(sQ.data_ptr() + ROPE_CHUNK * (Q_CHUNK_BYTES // 4), 16, 1024)
            for step in cutlass.range_constexpr(2):
                for half in cutlass.range_constexpr(2):
                    s_tmem = cutlass.inttoptr(s_tm0 + half * N_MMA, 6, cutlass.Float32)
                    kb: cutlass.Constexpr[int] = 2 * half + step
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16,
                            prims.CTAGroup.CTA_1,
                            s_tmem,
                            dkr.advance_start_address(kb * 32),
                            dqr.advance_start_address(kb * 32),
                            idesc_qk,
                            step != 0,
                        )
            for j2 in cutlass.range_constexpr(4):
                for kb in cutlass.range_constexpr(4):
                    for half in cutlass.range_constexpr(2):
                        c: cutlass.Constexpr[int] = 2 * j2 + half
                        s_tmem = cutlass.inttoptr(s_tm0 + half * N_MMA, 6, cutlass.Float32)
                        dk = _smem_desc(kvb + c * CHUNK_WORDS, 16, 1024)
                        dq = _smem_desc(sQ.data_ptr() + c * (Q_CHUNK_BYTES // 4), 16, 1024)
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F16,
                                prims.CTAGroup.CTA_1,
                                s_tmem,
                                dk.advance_start_address(kb * 32),
                                dq.advance_start_address(kb * 32),
                                idesc_qk,
                                True,
                            )
            if prims.elect_sync():
                prims.tcgen05_commit(s_full.data_ptr() + sst, group=prims.CTAGroup.CTA_1)

    # =====================================================================================
    # W11: PV. O^T[d, h] += V^T[d, tok] . P^T[tok, h] per 128-column stripe; V^T is the
    # dequantized NoPE tile read MN-major (two 64-column chunks 16 KB apart).
    # =====================================================================================
    if warp == W_PV:
        idesc_pv = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=cutlass.BFloat16,
            b_dtype=cutlass.BFloat16,
            n_dim=N_MMA,
            m_dim=128,
            a_major=1,
            b_major=0,
        )
        for kt in cutlass.range(0, num_kt, 1, unroll=1):
            pst = kt % cutlass.Int32(ST_P)
            pph = (kt // cutlass.Int32(ST_P)) % cutlass.Int32(2)
            _wait(p_full.data_ptr() + pst, pph)
            _wait(o_empty, kt % cutlass.Int32(2))
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            b = kt % cutlass.Int32(NUM_BUFS)
            kvb = sKV.data_ptr() + b * cutlass.Int32(BUF_WORDS)
            dp = _smem_desc(sP.data_ptr() + pst * cutlass.Int32(N_MMA * 64), 16, 1024)
            for s in cutlass.range_constexpr(NUM_STRIPES):
                o_tmem = cutlass.inttoptr(tmem_raw + TMEM_O + s * N_MMA, 6, cutlass.Float32)
                dv = _smem_desc(kvb + 2 * s * CHUNK_WORDS, CHUNK_BYTES, 1024)
                for kk in cutlass.range_constexpr(BT // 16):
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16,
                            prims.CTAGroup.CTA_1,
                            o_tmem,
                            dv.advance_start_address(kk * 2048),
                            dp.advance_start_address(kk * 32),
                            idesc_pv,
                            (kt > cutlass.Int32(0)) or (kk != 0),
                        )
            if prims.elect_sync():
                prims.tcgen05_commit(kv_empty.data_ptr() + b, group=prims.CTAGroup.CTA_1)
                prims.tcgen05_commit(o_full, group=prims.CTAGroup.CTA_1)
                prims.tcgen05_commit(p_empty.data_ptr() + pst, group=prims.CTAGroup.CTA_1)

    # =====================================================================================
    # W0-3: online softmax, one token per thread (TMEM lane), 16 heads per thread.
    # =====================================================================================
    if warp < 4:
        # M = 64 accumulator: token 16*warp + lane on lanes 0..15 of the warp's quadrant.
        t = warp * cutlass.Int32(16) + lane % cutlass.Int32(16)
        live = cutlass.Int32(1)
        if lane >= cutlass.Int32(16):
            live = cutlass.Int32(0)
        base = prims.TmemAddr(tmem_raw)
        my_lane_addr = ((base.row_id + warp * cutlass.Int32(32)) << cutlass.Int32(16)) | base.col_id
        neg_inf = cutlass.Float32(-float("inf"))
        mi = cutlass.Vector.from_elements(
            tuple(cutlass.Float32(MAX_INIT) for _ in range(N_MMA)), dtype=cutlass.Float32
        )
        real_mi = cutlass.Vector.from_elements(
            tuple(neg_inf for _ in range(N_MMA)), dtype=cutlass.Float32
        )
        p_byte = t * cutlass.Int32(2)
        for kt in cutlass.range(0, num_kt, 1, unroll=1):
            sst = kt % cutlass.Int32(ST_S)
            sph = (kt // cutlass.Int32(ST_S)) % cutlass.Int32(2)
            idx_st = kt % cutlass.Int32(ST_I)
            idx_ph = (kt // cutlass.Int32(ST_I)) % cutlass.Int32(2)
            _wait(idx_full.data_ptr() + idx_st, idx_ph)
            valid = live
            if sIdx[idx_st * cutlass.Int32(BT) + t] == cutlass.Int32(-1):
                valid = cutlass.Int32(0)
            prims.bar_warp_sync(0xFFFFFFFF)
            if prims.elect_sync():
                prims.mbarrier_arrive(idx_empty.data_ptr() + idx_st)
            _wait(s_full.data_ptr() + sst, sph)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            raw = prims.tcgen05_ld(
                "32x32b",
                cutlass.inttoptr(my_lane_addr + sst * cutlass.Int32(S_COLS), 6, cutlass.Float32),
                num=S_COLS,
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.bar_warp_sync(0xFFFFFFFF)
            if prims.elect_sync():
                prims.mbarrier_arrive(s_empty.data_ptr() + sst)
            # The two QK half-sums added (ku::float2_add); invalid tokens masked to -inf.
            sv = [
                _sel_lit(valid, _add_rn(raw[h], raw[N_MMA + h]), F32_NEG_INF) for h in range(N_MMA)
            ]
            # Per-head max over this tile's 128 tokens.
            hmax = _max_scatter16(sv, lane)
            mbuf = (kt % cutlass.Int32(2)) * cutlass.Int32(4 * N_MMA) + warp * cutlass.Int32(N_MMA)
            if lane % cutlass.Int32(2) == cutlass.Int32(0):
                sMax[mbuf + (lane >> cutlass.Int32(1))] = hmax
            prims.barrier_cta_sync(BAR_SOFTMAX, thread_count=128)
            mb0 = (kt % cutlass.Int32(2)) * cutlass.Int32(4 * N_MMA)
            cur = []
            for h in cutlass.range_constexpr(N_MMA):
                m = sMax[mb0 + h]
                for w in cutlass.range_constexpr(1, 4):
                    m = _fmax(m, sMax[mb0 + w * N_MMA + h])
                cur.append(_mul_ftz(m, scale_log2))
            # FlashMLA's lazy rescale: move the running max only when some head's tile max
            # exceeds it by more than 6 (log2 units); all live heads vote together.
            should = cutlass.Int32(0)
            for h in cutlass.range_constexpr(num_heads):
                should = should | _cmp_i32("gt", _sub_ftz(cur[h], mi[h]), F32_SIX)
            new_max = [_sel(should, _fmax(cur[h], mi[h]), mi[h]) for h in range(N_MMA)]
            # exp2(0) is exactly 1: heads (or tiles) that keep their max get factor 1.
            sfo = [_ex2(_sub_ftz(mi[h], new_max[h])) for h in range(N_MMA)]
            real_mi = cutlass.Vector.from_elements(
                tuple(_fmax(real_mi[h], cur[h]) for h in range(N_MMA)), dtype=cutlass.Float32
            )
            mi = cutlass.Vector.from_elements(tuple(new_max), dtype=cutlass.Float32)
            pv = [_ex2(_fma_rn(sv[h], scale_log2, -new_max[h])) for h in range(N_MMA)]
            # P^T [h][tok] bf16, K-major SW128 (one 64-token atom per head row).
            pst = kt % cutlass.Int32(ST_P)
            pph = (kt // cutlass.Int32(ST_P)) % cutlass.Int32(2)
            _wait(p_empty.data_ptr() + pst, pph ^ cutlass.Int32(1))
            p_katom = pst * cutlass.Int32(N_MMA * 64)
            for h2 in cutlass.range_constexpr(N_MMA // 2):
                pk = _bf16x2(pv[2 * h2], pv[2 * h2 + 1])
                for e in cutlass.range_constexpr(2):
                    h: cutlass.Constexpr[int] = 2 * h2 + e
                    piece = (p_byte // cutlass.Int32(16)) ^ cutlass.Int32(h % 8)
                    elt = (
                        p_katom
                        + cutlass.Int32(h * 64)
                        + piece * cutlass.Int32(8)
                        + (p_byte % cutlass.Int32(16)) // cutlass.Int32(2)
                    )
                    if live != cutlass.Int32(0):
                        sP[elt] = cutlass.Int16(
                            (pk >> cutlass.Int32(16 * e)) & cutlass.Int32(0xFFFF)
                        )
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            prims.bar_warp_sync(0xFFFFFFFF)
            if prims.elect_sync():
                prims.mbarrier_arrive(p_full.data_ptr() + pst)
            # Hand-off to the correction warps: the fp32 P of this tile (they sum it in
            # FlashMLA's order), the factor on the old O and l, and whether O is rescaled.
            cst = kt % cutlass.Int32(ST_C)
            cph = (kt // cutlass.Int32(ST_C)) % cutlass.Int32(2)
            _wait(c_empty.data_ptr() + cst, cph ^ cutlass.Int32(1))
            if live != cutlass.Int32(0):
                for h in cutlass.range_constexpr(N_MMA):
                    sPf[cutlass.Int32(h * PF_STRIDE) + t] = pv[h]
            if warp == 0:
                if lane == cutlass.Int32(0):
                    for h in cutlass.range_constexpr(N_MMA):
                        sCorr[cst * cutlass.Int32(N_MMA) + h] = sfo[h]
                    sFlag[cst] = should
            prims.bar_warp_sync(0xFFFFFFFF)
            if prims.elect_sync():
                prims.mbarrier_arrive(c_full.data_ptr() + cst)
        # Final max. No valid index at all: max = -inf, and l = 0 (FlashMLA).
        if tidx == cutlass.Int32(0):
            for h in cutlass.range_constexpr(N_MMA):
                has = _cmp_i32("ne", real_mi[h], F32_NEG_INF)
                sFin[h] = _sel_lit(has, mi[h], F32_NEG_INF)
            prims.mbarrier_arrive(fin_full)

    # =====================================================================================
    # W4-7: the O rescale per tile, the softmax denominator, then the epilogue. Thread = TMEM
    # lane = latent column of every 128-column O stripe.
    # =====================================================================================
    if warp >= W_CORR0 and warp < W_QK:
        cw = warp - cutlass.Int32(W_CORR0)
        col = cw * cutlass.Int32(32) + lane
        base = prims.TmemAddr(tmem_raw)
        my_addr = ((base.row_id + cw * cutlass.Int32(32)) << cutlass.Int32(16)) | base.col_id
        # Warps 4 and 5 keep FlashMLA's denominator: warp 4 the tokens 0-31 of every tile, warp
        # 5 the tokens 32-63; lane = 2 * head + parity (even or odd tokens).
        lh = lane >> cutlass.Int32(1)
        lpar = lane & cutlass.Int32(1)
        pf0 = lh * cutlass.Int32(PF_STRIDE) + cw * cutlass.Int32(32) + lpar
        li_part = cutlass.Float32(0.0)
        for kt in cutlass.range(0, num_kt, 1, unroll=1):
            cst = kt % cutlass.Int32(ST_C)
            cph = (kt // cutlass.Int32(ST_C)) % cutlass.Int32(2)
            _wait(c_full.data_ptr() + cst, cph)
            flag = sFlag[cst]
            corr = (sCorr.data_ptr() + cst * cutlass.Int32(N_MMA)).load(count=N_MMA, alignment=16)
            if cw < cutlass.Int32(2):
                # ku::float2_add chain from 0: d0 + d2 + ... (exact first add), in token order.
                tsum = sPf[pf0]
                for i2 in cutlass.range_constexpr(1, 16):
                    tsum = _add_rn(tsum, sPf[pf0 + cutlass.Int32(2 * i2)])
                sfo_h = sCorr[cst * cutlass.Int32(N_MMA) + lh]
                both = _add_ftz(tsum, _shfl_bfly(tsum, 1))
                li_part = _fma_ftz(li_part, sfo_h, both)
            prims.bar_warp_sync(0xFFFFFFFF)
            if prims.elect_sync():
                prims.mbarrier_arrive(c_empty.data_ptr() + cst)
            if kt > cutlass.Int32(0):
                _wait(o_full, (kt - cutlass.Int32(1)) % cutlass.Int32(2))
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                if flag != cutlass.Int32(0):
                    for s in cutlass.range_constexpr(NUM_STRIPES):
                        tp = cutlass.inttoptr(my_addr + TMEM_O + s * N_MMA, 6, cutlass.Float32)
                        ov = prims.tcgen05_ld("32x32b", tp, num=N_MMA)
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        nv = cutlass.Vector.from_elements(
                            tuple(_mul_rn(ov[h], corr[h]) for h in range(N_MMA)),
                            dtype=cutlass.Float32,
                        )
                        prims.tcgen05_st("32x32b", tp, nv)
                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.bar_warp_sync(0xFFFFFFFF)
            if prims.elect_sync():
                prims.mbarrier_arrive(o_empty)
        _wait(o_full, (cutlass.Int32(num_kt) - cutlass.Int32(1)) % cutlass.Int32(2))
        _wait(fin_full, 0)
        if cw < cutlass.Int32(2):
            if lpar == cutlass.Int32(0):
                sLi[cw * cutlass.Int32(N_MMA) + lh] = li_part
        prims.barrier_cta_sync(BAR_CORR, thread_count=128)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        fin_m = sFin.data_ptr().load(count=N_MMA, alignment=16)
        # l = l(tokens 0-31) + l(tokens 32-63); 0 when no index was valid.
        fin_l = [
            _sel_lit(
                _cmp_i32("ne", fin_m[h], F32_NEG_INF), _add_ftz(sLi[h], sLi[N_MMA + h]), F32_ZERO
            )
            for h in range(N_MMA)
        ]
        osc = [
            _sel_lit(_cmp_i32("ne", fin_l[h], F32_ZERO), _rcp(fin_l[h]), F32_ZERO)
            for h in range(N_MMA)
        ]
        if cutlass.const_expr(num_splits == 1):
            optr = out.iterator.raw_ptr()
            obase = row * cutlass.Int32(num_heads * D_NOPE)
            for s in cutlass.range_constexpr(NUM_STRIPES):
                tp = cutlass.inttoptr(my_addr + TMEM_O + s * N_MMA, 6, cutlass.Float32)
                ov = prims.tcgen05_ld("32x32b", tp, num=N_MMA)
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                for h2 in cutlass.range_constexpr(num_heads // 2):
                    pk = _bf16x2(
                        _mul_rn(ov[2 * h2], osc[2 * h2]), _mul_rn(ov[2 * h2 + 1], osc[2 * h2 + 1])
                    )
                    for e in cutlass.range_constexpr(2):
                        h: cutlass.Constexpr[int] = 2 * h2 + e
                        optr[obase + h * D_NOPE + s * 128 + col] = cutlass.Int16(
                            (pk >> cutlass.Int32(16 * e)) & cutlass.Int32(0xFFFF)
                        )
        else:
            # fp32 partial O / l into the (now idle) KV tile, lse = lg2(l) + max (FlashMLA).
            part = sKV
            for s in cutlass.range_constexpr(NUM_STRIPES):
                tp = cutlass.inttoptr(my_addr + TMEM_O + s * N_MMA, 6, cutlass.Float32)
                ov = prims.tcgen05_ld("32x32b", tp, num=N_MMA)
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                for h in cutlass.range_constexpr(num_heads):
                    part[h * D_NOPE + s * 128 + col] = _mul_rn(ov[h], osc[h])
            if cw == cutlass.Int32(0) and lane < cutlass.Int32(1):
                for h in cutlass.range_constexpr(N_MMA):
                    sLse[h] = _add_ftz(_lg2(fin_l[h]), fin_m[h])

    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp == W_PV:
        prims.tcgen05_dealloc(
            cutlass.inttoptr(tmem_raw, 6, cutlass.Float32), TMEM_COLS, group="cta_1"
        )
    if cutlass.const_expr(num_splits > 1):
        # FlashMLA's combine (smxx/decode/combine/combine.cu) over the cluster's partials:
        # each CTA merges 1/num_splits of the [heads x 512] output.
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()
        total4: cutlass.Constexpr[int] = num_heads * D_NOPE // 4
        per_cta4: cutlass.Constexpr[int] = total4 // num_splits
        iters: cutlass.Constexpr[int] = (per_cta4 + THREADS - 1) // THREADS
        part_local = sKV.data_ptr()
        for it in cutlass.range_constexpr(iters):
            v4 = split * cutlass.Int32(per_cta4) + tidx + cutlass.Int32(it * THREADS)
            if tidx + cutlass.Int32(it * THREADS) < cutlass.Int32(per_cta4):
                h = v4 // cutlass.Int32(D_NOPE // 4)
                d0 = (v4 % cutlass.Int32(D_NOPE // 4)) * cutlass.Int32(4)
                lses = []
                for p in cutlass.range_constexpr(num_splits):
                    peer_lse = prims.mapa(sLse.data_ptr(), cutlass.Int32(p))
                    lses.append(cutlass.Float32((peer_lse + h).load()))
                max_lse = lses[0]
                for p in cutlass.range_constexpr(1, num_splits):
                    max_lse = _fmax(max_lse, lses[p])
                max_lse = _sel_lit(_cmp_i32("ne", max_lse, F32_NEG_INF), max_lse, F32_ZERO)
                # combine.cu sums over lanes with a xor butterfly (offsets 16 .. 1; lanes past
                # the splits hold 0): the same adds as halving the list pairwise.
                sum_lse = _butterfly_sum(
                    [_ex2(_sub_ftz(lses[p], max_lse)) for p in range(num_splits)]
                )
                global_lse = _sel_lit(
                    _cmp_i32("ne", sum_lse, F32_ZERO), _add_ftz(_lg2(sum_lse), max_lse), F32_POS_INF
                )
                acc = []
                for p in cutlass.range_constexpr(num_splits):
                    w = _ex2(_sub_ftz(lses[p], global_lse))
                    peer_o = prims.mapa(part_local, cutlass.Int32(p))
                    o4 = (peer_o + h * cutlass.Int32(D_NOPE) + d0).load(count=4, alignment=16)
                    if cutlass.const_expr(p == 0):
                        # FlashMLA: result = 0; result += w * o (an fma onto 0 is the product).
                        acc = [_mul_into_zero(w, o4[e]) for e in range(4)]
                    else:
                        acc = [_fma_ftz(w, o4[e], acc[e]) for e in range(4)]
                lo = _bf16x2(acc[0], acc[1])
                hi = _bf16x2(acc[2], acc[3])
                ob = (row * cutlass.Int32(num_heads) + h) * cutlass.Int32(D_NOPE) + d0
                m16 = cutlass.Int32(0xFFFF)
                halves = (
                    lo & m16,
                    (lo >> cutlass.Int32(16)) & m16,
                    hi & m16,
                    (hi >> cutlass.Int32(16)) & m16,
                )
                (out.iterator.raw_ptr() + ob).store(
                    cutlass.Vector.from_elements(
                        tuple(cutlass.Int16(x) for x in halves), dtype=cutlass.Int16
                    ),
                    alignment=8,
                )
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


@cute.jit
def _launch(
    q: cute.Tensor,
    pool: cute.Tensor,
    indices: cute.Tensor,
    out: cute.Tensor,
    scale_log2: cutlass.Float32,
    stream: cuda_driver.CUstream,
    num_heads: cutlass.Constexpr[int],
    num_splits: cutlass.Constexpr[int],
) -> None:
    rows = indices.shape[0]
    h_q = q.shape[1]
    # The query: the first `num_heads` of `h_q` heads, 64 dims per box (SW128).
    tma_q = cuda.create_tensor_map_tiled(
        global_address=q.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[D_QK, h_q, rows],
        global_strides=[D_QK * 2 // 16, h_q * (D_QK * 2 // 16)],
        box_dims=[64, num_heads, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    # Whole pool rows, 82 x int64 a box; rows past the pool (the -1 slots) read as zeros.
    tma_raw = cuda.create_tensor_map_tiled(
        global_address=pool.iterator.toint(),
        dtype=cutlass.Int64,
        global_dims=[ROW_BYTES // 8, pool.shape[0]],
        global_strides=[ROW_BYTES // 16],
        box_dims=[ROW_BYTES // 8, 1],
        swizzle=cuda.TensorMapSwizzle.none,
    )
    _dsmla_decode_kernel(tma_q, tma_raw, indices, out, scale_log2, num_heads, num_splits).launch(
        grid=(num_splits, 1, rows),
        block=(THREADS, 1, 1),
        cluster=(num_splits, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
        use_pdl=True,
    )


_COMPILED: dict[tuple[int, int, int], Callable] = {}


def compile_variant(num_heads: int, num_splits: int, device_index: int) -> Callable:
    """The launcher for `num_heads` real heads and `num_splits` CTAs per row, compiled once per
    process and device. Rows, the query's head count and the pool's row count are runtime."""
    key = (device_index, num_heads, num_splits)
    if key not in _COMPILED:
        if num_heads not in SUPPORTED_HEADS or num_splits not in SUPPORTED_SPLITS:
            raise ValueError(f"no variant for {num_heads} heads and {num_splits} splits")
        make = cute.runtime.make_fake_compact_tensor
        rows, h_q, n_rows = cute.sym_int64(), cute.sym_int64(), cute.sym_int64()
        cutlass.cuda.initialize_cuda_context(device_index)
        _COMPILED[key] = cute.compile(
            _launch,
            make(cutlass.BFloat16, (rows, h_q, D_QK), stride_order=(2, 1, 0), assumed_align=16),
            make(cutlass.Uint8, (n_rows, ROW_BYTES), stride_order=(1, 0), assumed_align=16),
            make(cutlass.Int32, (rows, TOPK), stride_order=(1, 0), assumed_align=16),
            make(
                cutlass.Int16, (rows, num_heads, D_NOPE), stride_order=(2, 1, 0), assumed_align=16
            ),
            cutlass.Float32(0.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            num_heads,
            num_splits,
            options="--enable-tvm-ffi --opt-level 2",
        )
    return _COMPILED[key]


def pick_splits(batch: int, num_sms: int) -> int:
    """CTAs per row (the cluster size) for a batch of `batch` rows. Measured on GB300
    at 8.7K and 32K contexts: an 8-CTA cluster stops paying past ~15 rows, a 4-CTA one past ~38,
    two past ~76; the bounds scale with the SM count."""
    per = num_sms / 152.0
    if batch <= 15 * per:
        return 8
    if batch <= 38 * per:
        return 4
    if batch <= 76 * per:
        return 2
    return 1


def launch(
    fn: Callable,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run a compiled variant on the current stream; `out` written and returned."""
    pool = kv_cache.view(torch.uint8).view(-1, ROW_BYTES)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    fn(q, pool, indices, out.view(torch.int16), scale_log2_of(softmax_scale), stream)
    return out
