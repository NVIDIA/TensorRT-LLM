# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MXFP8 decode GEMV on tcgen05: ``y[M, N] = x[M, K] @ W[N, K]^T`` for M <= 8, block-scaled E4M3 x E4M3.

The weight is the MMA's A operand (128 output features per CTA), the activation is B (8 token columns),
fp32 accumulation in TMEM. Both operands come in their MXFP8 storage: E4M3 values (K contiguous) and UE8M0
scales in the 128x4-swizzled layout, whose 512-byte atom per (128 rows, 128 K) is exactly the source of a
``tcgen05.cp`` 32x128b.warpx4 scale copy, so both scale tensors are bulk-copied unchanged. For an activation of at
most 32 rows the atom of k-tile ``k`` sits at byte ``512 k``.

A cluster of SPLIT CTAs shares one 128-row weight tile; rank r streams the interleaved k-tiles r, r + SPLIT, ...
through a RING-stage ring. Before the grid dependency each CTA issues its first RING weight tiles (and their scale
atoms) and prefetches the rest into L2; after it, the activation slice of the CTA (values and scale atoms) lands in
one buffer. Rank r owns rows [r * 128 / SPLIT, (r + 1) * 128 / SPLIT) of the tile: the other ranks store their fp32
partials of those rows into its shared memory and arrive on its barrier (release, cluster scope); the owner adds the
SPLIT partials in rank order and stores bf16.

Warps: 0 weight TMA, 1 activation TMA, 2 TMEM allocation + scale copies + MMA, 3 idle, 4-7 epilogue (warp 4 + i
reads TMEM lanes 32 i .. 32 i + 31).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass import dsl_user_op
from cutlass.experimental import primitives as prims

CTA_M = 128  # weight rows per CTA = the MMA's M  # capacity: fixed: a kernel tile constant
MMA_N = 8  # token columns
CTA_K = 128  # one k-tile: 128 E4M3 bytes per row = one 128-byte swizzle row  # capacity: fixed: a kernel tile constant
MMA_INST_K = 32
K_BLOCKS = (
    CTA_K // MMA_INST_K
)  # 4 MMAs per k-tile, one per scale byte of the atom  # capacity: fixed: a kernel tile constant
THREADS = 256
SF_ATOM_BYTES = 512
SF_ATOM_U16 = SF_ATOM_BYTES // 2
SF_COLS = 4  # TMEM columns of one scale atom
W_TILE_BYTES = CTA_M * CTA_K
X_TILE_BYTES = MMA_N * CTA_K
EVICT_FIRST = 0x12F0000000000000
A_COLS_PER_MMA = MMA_INST_K // 4  # TMEM columns of one 128 x K32 E4M3 granule of A

a_dtype = cutlass.Float8E4M3FN
b_dtype = cutlass.Float8E4M3FN


def num_k_tiles(k_in: int) -> int:
    return k_in // CTA_K


def acc_cols(nacc: int, mma_n: int = MMA_N) -> int:
    # nacc independent accumulators of MMA_N columns: MMA j of the CTA accumulates into j % nacc, so the M128 x N8
    # MMAs do not form one serial dependency chain; the epilogue adds them.
    return max(16, nacc * mma_n)


def a_col0(nacc: int, my_tiles: int, mma_n: int = MMA_N) -> int:
    # a_tmem: [acc | SFA per k-tile | one SFB slot (reused in issue order) | A: 32 columns (128 x 128 E4M3) per k-tile]
    return (acc_cols(nacc, mma_n) + (my_tiles + 1) * SF_COLS + 31) // 32 * 32


def tmem_cols(
    ring: int, nacc: int, a_tmem: bool = False, my_tiles: int = 0, mma_n: int = MMA_N
) -> int:
    need = acc_cols(nacc, mma_n) + 2 * ring * SF_COLS
    if a_tmem:
        need = a_col0(nacc, my_tiles, mma_n) + my_tiles * K_BLOCKS * A_COLS_PER_MMA
    cols = 32
    while cols < need:
        cols *= 2
    return cols


@dsl_user_op
def read_globaltimer(*, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int64(
        _llvm.inline_asm(
            _T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _mapa_u32(smem_ptr, peer, *, loc=None, ip=None):
    """The shared::cluster address of this CTA's shared-memory location in cluster CTA ``peer``."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [smem_ptr.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(peer).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;", "=r,r,r", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _st_async_v4(dst, w0, w1, w2, w3, mbar, *, loc=None, ip=None):
    """st.async of four int32 words to a shared::cluster address, completing ``mbar`` there by 16 bytes."""
    from cutlass._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip), cutlass.Int32(w0).ir_value(loc=loc, ip=ip),
         cutlass.Int32(w1).ir_value(loc=loc, ip=ip), cutlass.Int32(w2).ir_value(loc=loc, ip=ip),
         cutlass.Int32(w3).ir_value(loc=loc, ip=ip), cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];", "r,r,r,r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _ld_shared_v4_f32(smem_ptr, *, loc=None, ip=None):
    """ld.shared.v4.f32 of four fp32 at a 16-byte-aligned shared-memory pointer."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    out = _llvm.inline_asm(
        _llvm.StructType.get_literal([_T.f32()] * 4), [smem_ptr.toint(loc=loc, ip=ip).ir_value()],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip
    return [
        cutlass.Float32(_llvm.extractvalue(_T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def _test_wait_cluster(mbar, parity, *, loc=None, ip=None):
    """mbarrier.test_wait.parity.acquire.cluster on a barrier of this CTA completed by other CTAs' st.async."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    done = cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [mbar.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(parity).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2;\n\t"
            "selp.u32 $0, 1, 0, p;\n\t}", "=r,r,r,~{memory}", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip
    return done != cutlass.Int32(0)


@dsl_user_op
def _absmax(acc, v, *, loc=None, ip=None):
    """max(acc, |v|)."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Float32(
        _llvm.inline_asm(
            _T.f32(), [cutlass.Float32(acc).ir_value(loc=loc, ip=ip), cutlass.Float32(v).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .f32 a;\n\tabs.f32 a, $2;\n\tmax.f32 $0, $1, a;\n\t}", "=f,f,f", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _rcp_approx_ftz(x, *, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Float32(
        _llvm.inline_asm(
            _T.f32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)], "rcp.approx.ftz.f32 $0, $1;", "=f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _e4m3x4(v0, v1, v2, v3, *, loc=None, ip=None):
    """Four f32 to E4M3 (RN, satfinite), packed little-endian into one int32 word (v0 in the low byte)."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
            "{\n\t.reg .b16 lo, hi;\n\tcvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\tmov.b32 $0, {lo, hi};\n\t}", "=r,f,f,f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _ld_volatile_s32(addr, *, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)], "ld.volatile.global.s32 $0, [$1];", "=r,l",
            has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _st_global_b32(addr, value, *, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(
        None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(value).ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;", "l,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _bf16x2_word(lo, hi, *, loc=None, ip=None):
    """bf16(lo) | bf16(hi) << 16 (RN), with -0.0 halves turned into +0.0 (0x80000000 marks an empty exchange word)."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Float32(hi).ir_value(loc=loc, ip=ip), cutlass.Float32(lo).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 w, l, h;\n\t.reg .pred pl, ph;\n\tcvt.rn.bf16x2.f32 w, $1, $2;\n\t"
            "and.b32 l, w, 0xFFFF;\n\tsetp.eq.u32 pl, l, 0x8000;\n\t@pl and.b32 w, w, 0xFFFF0000;\n\t"
            "shr.u32 h, w, 16;\n\tsetp.eq.u32 ph, h, 0x8000;\n\t@ph and.b32 w, w, 0x0000FFFF;\n\t"
            "mov.b32 $0, w;\n\t}", "=r,f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _bf16_bits(x, *, loc=None, ip=None):
    """bf16(x) (RN) as the fp32 bit pattern of the rounded value."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 h, z;\n\tcvt.rn.bf16.f32 h, $1;\n\tmov.b16 z, 0;\n\tmov.b32 $0, {z, h};\n\t}", "=r,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


def _bf16_round(x):
    return _bf16_bits(x).bitcast(cutlass.Float32)


@dsl_user_op
def _rope_bits(c, a, s, b, *, loc=None, ip=None):
    """bf16(c * a + s * b) with separately rounded products (no FMA), as fp32 bits: the stock RoPE arithmetic."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (c, a, s, b)],
            "{\n\t.reg .f32 p, q, r;\n\t.reg .b16 h, z;\n\tmul.rn.f32 p, $1, $2;\n\tmul.rn.f32 q, $3, $4;\n\t"
            "add.rn.f32 r, p, q;\n\tcvt.rn.bf16.f32 h, r;\n\tmov.b16 z, 0;\n\tmov.b32 $0, {z, h};\n\t}", "=r,f,f,f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


def _din_load4(table, hidden, hidden_half, word_idx):
    """Four int32 words at word_idx of the hidden rows (hidden_half == 1) or the embedding table: one CTA-uniform
    base-address select instead of a branch."""
    addr = cutlass.select_(
        hidden_half == cutlass.Int32(1), hidden.data_ptr(0).toint(), table.data_ptr(0).toint()
    )
    ptr = cutlass.inttoptr(
        addr + cutlass.Int64(word_idx) * cutlass.Int64(4), mem_space=1, dtype=cutlass.Int32
    )
    return ptr.load(count=4, alignment=16)


def _stamp(timers, timed: bool, bx, slot: int):
    # Diagnostic: %globaltimer of milestone `slot` of CTA bx (timers [grid, 8] int64).
    if timed:
        timers.store(read_globaltimer(), idx=bx * cutlass.Int32(8) + cutlass.Int32(slot))


def _k_tile(rank, i: int, split: int, my_tiles: int, contig: bool):
    # contig: rank r streams k-tiles [r * my_tiles, (r + 1) * my_tiles); else the interleaved r, r + split, ...
    if contig:
        return rank * cutlass.Int32(my_tiles) + cutlass.Int32(i)
    return rank + cutlass.Int32(i * split)


@cute.kernel
def m3_mx_gemv_kernel(
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],  # W [N, K] E4M3 as bytes, box 128 x 128
    tma_desc_wsf: cutlass.GridConstant[
        cuda.TensorMap
    ],  # W scale atoms [N / 128 * K / 128, 256] u16
    tma_desc_x: cutlass.GridConstant[cuda.TensorMap],  # x [M, K] E4M3 as bytes, box 128 x 8
    tma_desc_xsf: cutlass.GridConstant[cuda.TensorMap],  # x scale atoms [K / 128, 256] u16
    y: cutlass.Array,  # [M * N] bf16, token-major
    x_words: cutlass.Array,  # x_bf16: the bf16 activation [M, K] as int32 words
    timers: cutlass.Array,  # [grid, 8] int64 milestones when `timed`
    num_tokens: cutlass.Int32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    split: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    contig: cutlass.Constexpr[bool],
    timed: cutlass.Constexpr[bool],
    nacc: cutlass.Constexpr[int],
    a_tmem: cutlass.Constexpr[bool],
    st_async: cutlass.Constexpr[bool],
    x_bf16: cutlass.Constexpr[bool],
    ar_ws: cutlass.Int64,  # ar_push: the all-reduce workspace's multicast mapping (int32 words)
    ar_flags: cutlass.Int64,  # ar_push: its call flags (parity of the next call = flags[0] & 1)
    ar_rank: cutlass.Int32,
    ar_push: cutlass.Constexpr[
        int
    ],  # 1: push the output rows into the next all-reduce; 2: and trigger after it
    ar_world: cutlass.Constexpr[int],
    ar_maxt: cutlass.Constexpr[int],
    mma_n: cutlass.Constexpr[int],  # token columns of the MMA (8, 16 or 48): M <= mma_n
    din_i: cutlass.Array,  # din: int32 token ids [M]
    din_table: cutlass.Array,  # din: embedding [V, K / 2] bf16 as int32 words
    din_hidden: cutlass.Array,  # din: hidden rows [M, K / 2] bf16 as int32 words
    din_w: cutlass.Array,  # din: [embedding norm | hidden norm] weights [K] bf16 as int32 words
    din_f: cutlass.Array,  # din: fp32 [eps_e, eps_h]
    din_vocab: cutlass.Int32,
    din_g: cutlass.Array,  # din, flag 1: int64 hidden row of each token, as int32 words
    din_res: cutlass.Array,  # din, flag 2: the gathered hidden rows out [M, K / 2] bf16 as int32 words
    din_flags: cutlass.Int32,  # din: 1 = gather the hidden rows, 2 = write them to din_res
    rope_cs: cutlass.Array,  # rope: fp32 (cos, sin) [positions][64][2]
    rope_i: cutlass.Array,  # rope: int32 cache lengths [M / T]
    rope_bt: cutlass.Array,  # rope: int32 K / V page tables [M / T][2][max_blocks]
    rope_q: cutlass.Array,  # rope: E4M3 Q out [M][hq][128] as uint8
    rope_pool: cutlass.Array,  # rope: paged E4M3 cache [pages][hkv][tokens per page][128] as uint8
    rope_scale: cutlass.Array,  # rope: fp32 [1] E4M3 quantization scale
    rope_dims: cutlass.Array,  # rope: int32 [max_blocks, tokens-per-page log2, hq, hkv, T]
    din: cutlass.Constexpr[bool],
    rope: cutlass.Constexpr[bool],
):
    k_tiles = num_k_tiles(k_in)
    x_tile_bytes = mma_n * CTA_K
    my_tiles = k_tiles // split
    rows_per_rank = CTA_M // split
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp_id = cute.arch.warp_idx()
    rank = cutlass.Int32(0)
    if cutlass.const_expr(split > 1):
        rank = cute.arch.block_idx_in_cluster()
    tma_ptr_w = tma_desc_w.get_ptr()
    tma_ptr_wsf = tma_desc_wsf.get_ptr()
    tma_ptr_x = tma_desc_x.get_ptr()
    tma_ptr_xsf = tma_desc_xsf.get_ptr()
    n_tile = bx // cutlass.Int32(split)
    m_offset = n_tile * cutlass.Int32(CTA_M)

    smem_a = cutlass.Array(
        cutlass.Uint8, ring * W_TILE_BYTES, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_b = cutlass.Array(
        cutlass.Uint8, my_tiles * x_tile_bytes, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_sfa = cutlass.Array(
        cutlass.Uint8, ring * SF_ATOM_BYTES, space=cutlass.AddressSpace.smem, alignment=128
    )
    smem_sfb = cutlass.Array(
        cutlass.Uint8, my_tiles * SF_ATOM_BYTES, space=cutlass.AddressSpace.smem, alignment=128
    )
    w_full = cutlass.Array(cutlass.Int64, ring, space=cutlass.AddressSpace.smem, alignment=8)
    mma_done = cutlass.Array(cutlass.Int64, ring, space=cutlass.AddressSpace.smem, alignment=8)
    act_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    acc_done = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    reduce_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    # Partials of the rows this rank owns: [source rank][row][token] fp32 (its own slot stays unused). Rows are padded
    # to mma_n + 4 words: the owner's 16-byte row reads (lane = row) then hit all 32 banks once per 8 lanes; at a
    # stride of mma_n they were 8- to 16-way conflicted (4.3 us of the 48-token QKV tail).
    mb_stride = mma_n + 4
    mailbox = cutlass.Array(
        cutlass.Float32,
        split * rows_per_rank * mb_stride,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    # din: per-token RMS factor of the half of x this CTA's k-tiles lie in.
    din_rs = None
    if cutlass.const_expr(din):
        din_rs = cutlass.Array(cutlass.Float32, 8, space=cutlass.AddressSpace.smem, alignment=16)

    if warp_id == 0:
        prims.prefetch_tensormap(tma_ptr_w)
        prims.prefetch_tensormap(tma_ptr_wsf)
        prims.prefetch_tensormap(tma_ptr_x)
        prims.prefetch_tensormap(tma_ptr_xsf)
        if prims.elect_sync():
            for s in cutlass.range_constexpr(ring):
                prims.mbarrier_init(w_full.subview(s), 1)
                prims.mbarrier_init(mma_done.subview(s), 1)
            prims.mbarrier_init(act_full, 1)
            prims.mbarrier_init(acc_done, 1)
            if cutlass.const_expr(split > 1 and st_async):
                # Completed by the peers' st.async bytes; this CTA's own arrival carries the expected count.
                prims.mbarrier_init(reduce_ready, 1)
                prims.mbarrier_arrive_expect_tx(
                    reduce_ready, (split - 1) * rows_per_rank * mma_n * 4
                )
            elif cutlass.const_expr(split > 1):
                prims.mbarrier_init(reduce_ready, (split - 1) * rows_per_rank)
    if warp_id == 2:
        prims.tcgen05_alloc(tmem_ptr_i32, tmem_cols(ring, nacc, a_tmem, my_tiles, mma_n))
        prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    if cutlass.const_expr(split > 1):
        prims.barrier_cluster_arrive_relaxed()
        prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    if tx == 0:
        _stamp(timers, timed, bx, 0)

    if warp_id == 0:
        if prims.elect_sync():
            for i in cutlass.range_constexpr(ring, my_tiles):
                k_global = _k_tile(rank, i, split, my_tiles, contig)
                prims.cp_async_bulk_tensor_prefetch(
                    tma_ptr_w, [k_global * cutlass.Int32(CTA_K), m_offset], []
                )
            if cutlass.const_expr((n_out // CTA_M) * split >= 80):
                # Wide grids launch in part only after the predecessor (e.g. the fused attention) frees its SMs, which
                # loses their pre-wait weight prefetch: the CTAs of the first half also stage their partner's
                # (bx + grid / 2) weights into L2.
                if bx < cutlass.Int32((n_out // CTA_M) * split // 2):
                    pb = bx + cutlass.Int32((n_out // CTA_M) * split // 2)
                    p_rank = pb % cutlass.Int32(split)
                    p_m_offset = (pb // cutlass.Int32(split)) * cutlass.Int32(CTA_M)
                    for i in cutlass.range_constexpr(my_tiles):
                        p_k = _k_tile(p_rank, i, split, my_tiles, contig)
                        prims.cp_async_bulk_tensor_prefetch(
                            tma_ptr_w, [p_k * cutlass.Int32(CTA_K), p_m_offset], []
                        )
            for i in cutlass.range_constexpr(my_tiles):
                stage = i % ring
                if cutlass.const_expr(i >= ring):
                    while not cute.arch.mbarrier_try_wait(
                        mma_done.subview(stage).data_ptr(), (i // ring - 1) % 2
                    ):
                        pass
                k_global = _k_tile(rank, i, split, my_tiles, contig)
                prims.mbarrier_arrive_expect_tx(w_full.subview(stage), W_TILE_BYTES + SF_ATOM_BYTES)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_a.subview(stage * W_TILE_BYTES),
                    tma_ptr_w,
                    (k_global * cutlass.Int32(CTA_K), m_offset),
                    w_full.subview(stage),
                    l2_cache_hint=EVICT_FIRST,
                )
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_sfa.subview(stage * SF_ATOM_BYTES),
                    tma_ptr_wsf,
                    (cutlass.Int32(0), n_tile * cutlass.Int32(k_tiles) + k_global),
                    w_full.subview(stage),
                    l2_cache_hint=EVICT_FIRST,
                )
    elif warp_id == 1:
        prims.griddepcontrol(prims.GridDepAction.WAIT)
        if cutlass.const_expr(trigger_early):
            # Dependents launch once this grid's own dependency is satisfied (not at entry): consumers that read
            # step metadata or caches before their own grid wait rely on every grid ahead of them having passed its
            # wait, so the chain of releases never runs ahead of a producer still in flight.
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
        if tx == 32:
            _stamp(timers, timed, bx, 1)
        if cutlass.const_expr(not x_bf16):
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(act_full, my_tiles * (x_tile_bytes + SF_ATOM_BYTES))
                for i in cutlass.range_constexpr(my_tiles):
                    k_global = _k_tile(rank, i, split, my_tiles, contig)
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_b.subview(i * x_tile_bytes),
                        tma_ptr_x,
                        (k_global * cutlass.Int32(CTA_K), cutlass.Int32(0)),
                        act_full,
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_sfb.subview(i * SF_ATOM_BYTES),
                        tma_ptr_xsf,
                        (cutlass.Int32(0), k_global),
                        act_full,
                    )
    elif warp_id == 2:
        tmem_raw = tmem_ptr_i32.load()
        base_col = tmem_raw & 0xFFFF
        base_row = tmem_raw >> 16
        sfa_col0 = base_col + acc_cols(nacc, mma_n)
        sfb_col0 = sfa_col0 + ring * SF_COLS
        idesc = prims.Tcgen05MxInstrDesc.build(
            a_dtype=a_dtype, b_dtype=b_dtype, scale_format=1, n_dim=mma_n, m_dim=CTA_M
        )
        s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
        if cutlass.const_expr(a_tmem):
            # A staged in TMEM: before the grid wait every weight k-tile and its scale atom go on into TMEM as they
            # land (the copies of a stage refilled later are committed to its mma_done); once the activation lands,
            # its scale atoms and the MMAs, A read from TMEM.
            base_row = tmem_raw >> 16
            base_col = tmem_raw & 0xFFFF
            sfa_col0 = base_col + acc_cols(nacc, mma_n)
            sfb_col0 = sfa_col0 + my_tiles * SF_COLS
            a_base = tmem_raw + cutlass.Int32(a_col0(nacc, my_tiles, mma_n))
            s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
            for i in cutlass.range_constexpr(my_tiles):
                stage = i % ring
                while not cute.arch.mbarrier_try_wait(
                    w_full.subview(stage).data_ptr(), (i // ring) % 2
                ):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                desc_a = prims.Tcgen05SmemDesc.build(
                    smem_a.subview(stage * W_TILE_BYTES), leading_byte_offset=16, stride_byte_offset=1024,
                    base_offset=0, layout=2,
                )  # fmt: skip
                desc_sfa = prims.Tcgen05SmemDesc.build(
                    smem_sfa.subview(stage * SF_ATOM_BYTES), leading_byte_offset=16, stride_byte_offset=128,
                    base_offset=0, layout=0,
                )  # fmt: skip
                if prims.elect_sync():
                    for kb in cutlass.range_constexpr(K_BLOCKS):
                        prims.tcgen05_cp(
                            prims.Tcgen05CpShape.SHAPE_128X256B,
                            cutlass.inttoptr(
                                a_base + cutlass.Int32((i * K_BLOCKS + kb) * A_COLS_PER_MMA),
                                6,
                                cutlass.Int32,
                            ),
                            desc_a + ((MMA_INST_K >> 4) * kb),
                            group=prims.CTAGroup.CTA_1,
                        )
                    prims.tcgen05_cp(
                        s2t_shape,
                        cutlass.inttoptr((base_row << 16) | (sfa_col0 + i * SF_COLS), 6, cutlass.Int32),
                        desc_sfa, multicast=s2t_multicast,
                    )  # fmt: skip
                    if cutlass.const_expr(i + ring < my_tiles):
                        prims.tcgen05_commit(mma_done.subview(stage))
            if cutlass.const_expr(x_bf16):
                # Completed by a thread's arrive: a suspended try_wait wakes late on those.
                while not cute.arch.mbarrier_test_wait(act_full.data_ptr(), 0):
                    pass
            else:
                while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
                    pass
            if tx == 64:
                _stamp(timers, timed, bx, 2)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            if prims.elect_sync():
                for i in cutlass.range_constexpr(my_tiles):
                    desc_sfb = prims.Tcgen05SmemDesc.build(
                        smem_sfb.subview(i * SF_ATOM_BYTES), leading_byte_offset=16, stride_byte_offset=128,
                        base_offset=0, layout=0,
                    )  # fmt: skip
                    desc_b = prims.Tcgen05SmemDesc.build(
                        smem_b.subview(i * x_tile_bytes), leading_byte_offset=16, stride_byte_offset=1024,
                        base_offset=0, layout=2,
                    )  # fmt: skip
                    sfa_tmem = cutlass.inttoptr(
                        (base_row << 16) | (sfa_col0 + i * SF_COLS), 6, cutlass.Int32
                    )
                    sfb_tmem = cutlass.inttoptr((base_row << 16) | sfb_col0, 6, cutlass.Int32)
                    prims.tcgen05_cp(s2t_shape, sfb_tmem, desc_sfb, multicast=s2t_multicast)
                    for kb in cutlass.range_constexpr(K_BLOCKS):
                        prims.tcgen05_mma_block_scale(
                            prims.MMABlockScaleKind.MXF8F6F4, prims.CTAGroup.CTA_1,
                            cutlass.inttoptr(tmem_raw, 6, cutlass.Float32),
                            cutlass.inttoptr(
                                a_base + cutlass.Int32((i * K_BLOCKS + kb) * A_COLS_PER_MMA), 6, cutlass.Int32
                            ),
                            desc_b + ((MMA_INST_K >> 4) * kb), idesc.set_sf_ids(a_sf_id=kb, b_sf_id=kb),
                            i > 0 or kb > 0,
                            sfa_tmem, sfb_tmem,
                        )  # fmt: skip
            if tx == 64:
                _stamp(timers, timed, bx, 3)
            if prims.elect_sync():
                prims.tcgen05_commit(acc_done)
        else:
            if cutlass.const_expr(x_bf16):
                # Completed by a thread's arrive: a suspended try_wait wakes late on those.
                while not cute.arch.mbarrier_test_wait(act_full.data_ptr(), 0):
                    pass
            else:
                while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
                    pass
            if tx == 64:
                _stamp(timers, timed, bx, 2)
            for i in cutlass.range_constexpr(my_tiles):
                stage = i % ring
                while not cute.arch.mbarrier_try_wait(
                    w_full.subview(stage).data_ptr(), (i // ring) % 2
                ):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                sfa_tmem = (base_row << 16) | (sfa_col0 + stage * SF_COLS)
                sfb_tmem = (base_row << 16) | (sfb_col0 + stage * SF_COLS)
                desc_sfa = prims.Tcgen05SmemDesc.build(
                    smem_sfa.subview(stage * SF_ATOM_BYTES), leading_byte_offset=16, stride_byte_offset=128,
                    base_offset=0, layout=0,
                )  # fmt: skip
                desc_sfb = prims.Tcgen05SmemDesc.build(
                    smem_sfb.subview(i * SF_ATOM_BYTES), leading_byte_offset=16, stride_byte_offset=128,
                    base_offset=0, layout=0,
                )  # fmt: skip
                desc_a = prims.Tcgen05SmemDesc.build(
                    smem_a.subview(stage * W_TILE_BYTES), leading_byte_offset=16, stride_byte_offset=1024,
                    base_offset=0, layout=2,
                )  # fmt: skip
                desc_b = prims.Tcgen05SmemDesc.build(
                    smem_b.subview(i * x_tile_bytes), leading_byte_offset=16, stride_byte_offset=1024,
                    base_offset=0, layout=2,
                )  # fmt: skip
                if prims.elect_sync():
                    prims.tcgen05_cp(s2t_shape, cutlass.inttoptr(sfa_tmem, 6, cutlass.Int32), desc_sfa,
                                     multicast=s2t_multicast)  # fmt: skip
                    prims.tcgen05_cp(s2t_shape, cutlass.inttoptr(sfb_tmem, 6, cutlass.Int32), desc_sfb,
                                     multicast=s2t_multicast)  # fmt: skip
                    for kb in cutlass.range_constexpr(K_BLOCKS):
                        inc = ((MMA_INST_K * 8 // 8) >> 4) * kb
                        j = i * K_BLOCKS + kb
                        prims.tcgen05_mma_block_scale(
                            prims.MMABlockScaleKind.MXF8F6F4, prims.CTAGroup.CTA_1,
                            cutlass.inttoptr(tmem_raw + cutlass.Int32((j % nacc) * mma_n), 6, cutlass.Float32),
                            desc_a + inc, desc_b + inc, idesc.set_sf_ids(a_sf_id=kb, b_sf_id=kb), j >= nacc,
                            cutlass.inttoptr(sfa_tmem, 6, cutlass.Int32), cutlass.inttoptr(sfb_tmem, 6, cutlass.Int32),
                        )  # fmt: skip
                    prims.tcgen05_commit(mma_done.subview(stage))
            if tx == 64:
                _stamp(timers, timed, bx, 3)
            if prims.elect_sync():
                prims.tcgen05_commit(acc_done)
    elif warp_id >= 4:
        lane = tx % 32
        row = (warp_id % 4) * 32 + lane
        if cutlass.const_expr(x_bf16):
            # The bf16 activation of this CTA's k-tiles to MXFP8 in shared memory, as
            # torch.ops.trtllm.mxfp8_quantize(x, True) does (UE8M0 = round-up(qz_amax * rcp(448)),
            # E4M3 RN-satfinite): one 32-element block (token qz_t, k-tile qz_i, block qz_kb) per thread and round,
            # the E4M3 bytes into the 128-byte-swizzled B tile, the qz_scale byte into the tile's qz_scale atom.
            qz_tid = tx - cutlass.Int32(128)
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            din_half_k = k_in // 2
            din_hidden_half = cutlass.Int32(0)
            if cutlass.const_expr(din):
                # Draft input x = [rmsnorm(table[ids]) * w_e, rmsnorm(hidden) * w_h] (an out-of-range id embeds to
                # zero): this CTA's k-tiles lie in one half, so it needs that half's statistic of every token. 16
                # threads per token, 16-byte loads, all of a thread's loads in flight at once.
                din_hidden_half = cutlass.select_(
                    rank * cutlass.Int32(my_tiles * CTA_K) >= cutlass.Int32(din_half_k),
                    cutlass.Int32(1),
                    cutlass.Int32(0),
                )
                din_tok = qz_tid // cutlass.Int32(16)
                din_part = qz_tid % cutlass.Int32(16)
                din_tok_ld = cutlass.select_(din_tok < num_tokens, din_tok, cutlass.Int32(0))
                din_id = din_i.load(idx=din_tok_ld)
                din_valid = (din_id >= cutlass.Int32(0)) & (din_id < din_vocab)
                din_hrow = din_tok_ld
                if (din_flags & cutlass.Int32(1)) != cutlass.Int32(0):
                    din_hrow = din_g.load(idx=din_tok_ld * cutlass.Int32(2))
                din_row = cutlass.select_(
                    din_hidden_half == cutlass.Int32(1),
                    din_hrow,
                    cutlass.select_(din_valid, din_id, cutlass.Int32(0)),
                )
                # The first hidden-half CTA of tile 0 reads whole hidden rows here: it also writes them out.
                din_copy = (
                    ((din_flags & cutlass.Int32(2)) != cutlass.Int32(0))
                    & (bx == cutlass.Int32(split // 2))
                    & (din_hidden_half == cutlass.Int32(1))
                    & (din_tok < num_tokens)
                )
                din_base = din_row * cutlass.Int32(din_half_k // 2)
                din_ss = cutlass.Float32(0.0)
                din_vecs_per_part = din_half_k // 8 // 16
                for din_round in cutlass.range_constexpr(0, din_vecs_per_part, 16):
                    din_n = min(16, din_vecs_per_part - din_round)
                    din_loaded = []
                    for din_j in cutlass.range_constexpr(din_n):
                        din_widx = din_base + (
                            din_part + cutlass.Int32((din_round + din_j) * 16)
                        ) * cutlass.Int32(4)
                        din_loaded.append(
                            _din_load4(din_table, din_hidden, din_hidden_half, din_widx)
                        )
                    if din_copy:
                        for din_j in cutlass.range_constexpr(din_n):
                            din_v = din_loaded[din_j]
                            din_res.store(
                                (din_v[0], din_v[1], din_v[2], din_v[3]),
                                idx=din_tok * cutlass.Int32(din_half_k // 2)
                                + (din_part + cutlass.Int32((din_round + din_j) * 16))
                                * cutlass.Int32(4),
                                alignment=16,
                            )
                    for din_j in cutlass.range_constexpr(din_n):
                        for din_e in cutlass.range_constexpr(4):
                            din_word = cutlass.Int32(din_loaded[din_j][din_e])
                            din_lo = (din_word << cutlass.Int32(16)).bitcast(cutlass.Float32)
                            din_hi = (din_word & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                            din_ss = din_ss + din_lo * din_lo + din_hi * din_hi
                for din_o in cutlass.range_constexpr(4):
                    din_ss = din_ss + cute.arch.shuffle_sync_bfly(din_ss, offset=1 << din_o)
                if din_part == cutlass.Int32(0):
                    din_eps = din_f.load(idx=din_hidden_half)
                    din_r = cute.math.rsqrt(
                        din_ss / cutlass.Float32(din_half_k) + din_eps, fastmath=True
                    )
                    din_keep = din_valid | (din_hidden_half == cutlass.Int32(1))
                    din_rs.store(
                        cutlass.select_(din_keep, din_r, cutlass.Float32(0.0)), idx=din_tok
                    )
                prims.barrier_cta_sync(3, thread_count=128)
            qz_b_words = cutlass.Array(smem_b.data_ptr(0), shape=(my_tiles * x_tile_bytes // 4,), dtype=cutlass.Int32,
                                    alignment=16)  # fmt: skip
            qz_items = my_tiles * K_BLOCKS * mma_n
            qz_rounds = (qz_items + 127) // 128
            # A group's loads first (clamped to a valid row), then its math: one L2 round trip per group of
            # QZ_GROUP rounds instead of one per round, without holding every round's data in registers.
            QZ_GROUP = qz_rounds if qz_rounds <= 8 else 4
            qz_loaded = []
            for qz_r in cutlass.range_constexpr(qz_rounds):
                qz_item = qz_tid + cutlass.Int32(qz_r * 128)
                qz_t = qz_item % cutlass.Int32(mma_n)
                qz_kb = (qz_item // cutlass.Int32(mma_n)) % cutlass.Int32(K_BLOCKS)
                qz_i = qz_item // cutlass.Int32(mma_n * K_BLOCKS)
                if cutlass.const_expr(qz_r * 128 + 127 >= qz_items):
                    qz_i = cutlass.select_(
                        qz_item < cutlass.Int32(qz_items), qz_i, cutlass.Int32(0)
                    )
                qz_k_global = rank + qz_i * cutlass.Int32(split)
                if cutlass.const_expr(contig):
                    qz_k_global = rank * cutlass.Int32(my_tiles) + qz_i
                qz_t_ld = cutlass.select_(qz_t < num_tokens, qz_t, cutlass.Int32(0))
                qz_x_base = (
                    qz_t_ld * cutlass.Int32(k_in // 2)
                    + qz_k_global * cutlass.Int32(CTA_K // 2)
                    + qz_kb * cutlass.Int32(MMA_INST_K // 2)
                )
                qz_wvecs = None
                if cutlass.const_expr(din):
                    # The draft input's raw values (table row of the id / hidden row) and its norm weights.
                    qz_col = qz_k_global * cutlass.Int32(CTA_K) + qz_kb * cutlass.Int32(MMA_INST_K)
                    qz_hcol = qz_col - din_hidden_half * cutlass.Int32(din_half_k)
                    qz_id = din_i.load(idx=qz_t_ld)
                    qz_hrow = qz_t_ld
                    if (din_flags & cutlass.Int32(1)) != cutlass.Int32(0):
                        qz_hrow = din_g.load(idx=qz_t_ld * cutlass.Int32(2))
                    qz_row = cutlass.select_(
                        din_hidden_half == cutlass.Int32(1),
                        qz_hrow,
                        cutlass.select_(
                            (qz_id >= cutlass.Int32(0)) & (qz_id < din_vocab),
                            qz_id,
                            cutlass.Int32(0),
                        ),
                    )
                    qz_src_w = qz_row * cutlass.Int32(din_half_k // 2) + qz_hcol // cutlass.Int32(2)
                    qz_vecs = [
                        _din_load4(
                            din_table,
                            din_hidden,
                            din_hidden_half,
                            qz_src_w + cutlass.Int32(qz_v * 4),
                        )
                        for qz_v in range(4)
                    ]
                    qz_wvecs = [
                        din_w.load(
                            idx=qz_col // cutlass.Int32(2) + cutlass.Int32(qz_v * 4),
                            vector_size=4,
                            alignment=16,
                        )
                        for qz_v in range(4)
                    ]
                else:
                    qz_vecs = [
                        x_words.load(
                            idx=qz_x_base + cutlass.Int32(qz_v * 4), vector_size=4, alignment=16
                        )
                        for qz_v in range(4)
                    ]
                qz_loaded.append((qz_item, qz_t, qz_kb, qz_i, qz_vecs, qz_wvecs))
                if cutlass.const_expr((qz_r + 1) % QZ_GROUP == 0 or qz_r + 1 == qz_rounds):
                    for qz_g in cutlass.range_constexpr(len(qz_loaded)):
                        qz_item, qz_t, qz_kb, qz_i, qz_vecs, qz_wvecs = qz_loaded[qz_g]
                        qz_vals = []
                        qz_amax = cutlass.Float32(0.0)
                        if cutlass.const_expr(din):
                            qz_rs = din_rs.load(
                                idx=cutlass.select_(qz_t < num_tokens, qz_t, cutlass.Int32(0))
                            )
                        for qz_v in cutlass.range_constexpr(4):
                            for qz_e4 in cutlass.range_constexpr(4):
                                qz_word = cutlass.Int32(qz_vecs[qz_v][qz_e4])
                                qz_lo = (qz_word << cutlass.Int32(16)).bitcast(cutlass.Float32)
                                qz_hi = (qz_word & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                                if cutlass.const_expr(din):
                                    # bf16(x * rs * w), the draft-input kernel's rounding.
                                    qz_wword = cutlass.Int32(qz_wvecs[qz_v][qz_e4])
                                    qz_lo = _bf16_round(
                                        qz_lo
                                        * qz_rs
                                        * (qz_wword << cutlass.Int32(16)).bitcast(cutlass.Float32)
                                    )
                                    qz_hi = _bf16_round(
                                        qz_hi
                                        * qz_rs
                                        * (qz_wword & cutlass.Int32(-65536)).bitcast(
                                            cutlass.Float32
                                        )
                                    )
                                qz_vals.append(qz_lo)
                                qz_vals.append(qz_hi)
                                qz_amax = _absmax(_absmax(qz_amax, qz_lo), qz_hi)
                        qz_bits = (qz_amax * _rcp_approx_ftz(cutlass.Float32(448.0))).bitcast(
                            cutlass.Int32
                        )
                        qz_e = (
                            (qz_bits >> cutlass.Int32(23)) & cutlass.Int32(0xFF)
                        ) + cutlass.select_(
                            (qz_bits & cutlass.Int32(0x7FFFFF)) != cutlass.Int32(0),
                            cutlass.Int32(1),
                            cutlass.Int32(0),
                        )
                        qz_e = cutlass.select_(qz_e > cutlass.Int32(254), cutlass.Int32(254), qz_e)
                        qz_scale = ((cutlass.Int32(254) - qz_e) << cutlass.Int32(23)).bitcast(
                            cutlass.Float32
                        )
                        qz_scale = cutlass.select_(
                            qz_amax != cutlass.Float32(0.0), qz_scale, cutlass.Float32(0.0)
                        )
                        qz_q = [_e4m3x4(qz_vals[4 * qz_j] * qz_scale, qz_vals[4 * qz_j + 1] * qz_scale,
                                        qz_vals[4 * qz_j + 2] * qz_scale, qz_vals[4 * qz_j + 3] * qz_scale)
                                for qz_j in range(8)]  # fmt: skip
                        if qz_item < cutlass.Int32(qz_items):
                            if qz_t < num_tokens:
                                qz_row_w = qz_i * cutlass.Int32(
                                    x_tile_bytes // 4
                                ) + qz_t * cutlass.Int32(CTA_K // 4)
                                qz_c0 = qz_kb * cutlass.Int32(2)
                                qz_b_words.store(
                                    (qz_q[0], qz_q[1], qz_q[2], qz_q[3]),
                                    idx=qz_row_w
                                    + (qz_c0 ^ (qz_t & cutlass.Int32(7))) * cutlass.Int32(4),
                                    alignment=16,
                                )
                                qz_b_words.store(
                                    (qz_q[4], qz_q[5], qz_q[6], qz_q[7]),
                                    idx=qz_row_w
                                    + ((qz_c0 + cutlass.Int32(1)) ^ (qz_t & cutlass.Int32(7)))
                                    * cutlass.Int32(4),
                                    alignment=16,
                                )
                                smem_sfb.store(
                                    cutlass.Uint8(qz_e),
                                    idx=qz_i * cutlass.Int32(SF_ATOM_BYTES)
                                    + (qz_t % cutlass.Int32(32)) * cutlass.Int32(16)
                                    + (qz_t // cutlass.Int32(32)) * cutlass.Int32(4)
                                    + qz_kb,
                                )
                    qz_loaded = []
            # Generic-proxy stores made visible to the MMA's async-proxy reads, then one arrival for all.
            prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
            prims.barrier_cta_sync(2, thread_count=128)
            if qz_tid == cutlass.Int32(0):
                prims.mbarrier_arrive(act_full)
        if cutlass.const_expr(st_async):
            while not cute.arch.mbarrier_test_wait(acc_done.data_ptr(), 0):
                pass
        else:
            while not cute.arch.mbarrier_try_wait(acc_done.data_ptr(), 0):
                pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        # tcgen05.ld widths are powers of two: the accumulator columns in descending power-of-two pieces.
        acc_all = []
        ld_col = 0
        for ld_w in (128, 64, 32, 16, 8):
            if cutlass.const_expr((mma_n * nacc - ld_col) >= ld_w):
                piece = prims.tcgen05_ld(
                    "32x32b",
                    cutlass.inttoptr(
                        tmem_ptr_i32.load() + cutlass.Int32(ld_col), 6, cutlass.Float32
                    ),
                    num=ld_w,
                )
                acc_all = acc_all + [piece[c] for c in range(ld_w)]
                ld_col += ld_w
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        acc = []
        for t in cutlass.range_constexpr(mma_n):
            v = cutlass.Float32(acc_all[t])
            for a in cutlass.range_constexpr(1, nacc):
                v = v + cutlass.Float32(acc_all[a * mma_n + t])
            acc.append(v)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        if tx == 128:
            _stamp(timers, timed, bx, 4)
        n = m_offset + row
        if cutlass.const_expr(split == 1):
            for t in cutlass.range_constexpr(mma_n):
                if cutlass.Int32(t) < num_tokens:
                    y.store(
                        cutlass.BFloat16(cutlass.Float32(acc[t])), idx=cutlass.Int32(t * n_out) + n
                    )
            if cutlass.const_expr(ar_push > 0):
                prims.griddepcontrol(prims.GridDepAction.WAIT)
                # The parity of each (token, column slice) slot: the all-reduce's CTA t * 8 + slice flag. All loads are
                # issued before the first store so they overlap (one L2 round trip, not one per token).
                ar_par = [
                    _ld_volatile_s32(
                        ar_flags
                        + cutlass.Int64(4)
                        * (cutlass.Int64(t * 8) + cutlass.Int64(n // cutlass.Int32(n_out // 8)))
                    )
                    & cutlass.Int32(1)
                    for t in range(mma_n)
                ]
                for t in cutlass.range_constexpr(mma_n):
                    ar_v = cutlass.Float32(acc[t])
                    ar_hi = cute.arch.shuffle_sync_bfly(ar_v, offset=1)
                    if (lane % cutlass.Int32(2)) == cutlass.Int32(0):
                        if cutlass.Int32(t) < num_tokens:
                            ar_off = (
                                cutlass.Int64(ar_par[t])
                                * cutlass.Int64(ar_maxt * ar_world * (n_out // 2))
                                + (cutlass.Int64(t * ar_world) + cutlass.Int64(ar_rank))
                                * cutlass.Int64(n_out // 2)
                                + (cutlass.Int64(n) // cutlass.Int64(2))
                            )
                            _st_global_b32(
                                ar_ws + ar_off * cutlass.Int64(4), _bf16x2_word(ar_v, ar_hi)
                            )
        else:
            owner = row // cutlass.Int32(rows_per_rank)
            local = row % cutlass.Int32(rows_per_rank)
            if cutlass.const_expr(rope):
                # Rank r owns head dims [r * P, (r + 1) * P) and their rotation partners 64 + ..., P = rows / 2.
                pairs = rows_per_rank // 2
                owner = (row % cutlass.Int32(64)) // cutlass.Int32(pairs)
                local = (row // cutlass.Int32(64)) * cutlass.Int32(pairs) + (
                    row % cutlass.Int32(64)
                ) % cutlass.Int32(pairs)
            if owner != rank:
                base = (rank * cutlass.Int32(rows_per_rank) + local) * cutlass.Int32(mb_stride)
                if cutlass.const_expr(st_async):
                    dst = _mapa_u32(mailbox.subview(base).data_ptr(), owner)
                    mb = _mapa_u32(reduce_ready.data_ptr(), owner)
                    words = [cutlass.Float32(acc[t]).bitcast(cutlass.Int32) for t in range(mma_n)]
                    for g in cutlass.range_constexpr(mma_n // 4):
                        _st_async_v4(
                            dst + cutlass.Int32(16 * g),
                            words[4 * g],
                            words[4 * g + 1],
                            words[4 * g + 2],
                            words[4 * g + 3],
                            mb,
                        )
                else:
                    for t in cutlass.range_constexpr(mma_n):
                        prims.mapa(mailbox.subview(base + cutlass.Int32(t)), owner).store(
                            cutlass.Float32(acc[t])
                        )
                    prims.mbarrier_arrive(
                        prims.mapa(reduce_ready, owner), scope=prims.MemScope.CLUSTER
                    )
            else:
                if cutlass.const_expr(st_async):
                    while not _test_wait_cluster(reduce_ready.data_ptr(), 0):
                        pass
                else:
                    while not prims.mbarrier_try_wait_parity(
                        reduce_ready, 0, scope=prims.MBarrierScope.CLUSTER
                    ):
                        pass
                if lane == 0:
                    if row // cutlass.Int32(rows_per_rank) == rank:
                        if (row % cutlass.Int32(rows_per_rank)) == 0:
                            _stamp(timers, timed, bx, 5)
                totals = []
                for g in cutlass.range_constexpr(mma_n // 4):
                    sums = [cutlass.Float32(0.0)] * 4
                    for q in cutlass.range_constexpr(split):
                        parts = _ld_shared_v4_f32(
                            mailbox.subview(
                                (cutlass.Int32(q * rows_per_rank) + local)
                                * cutlass.Int32(mb_stride)
                                + cutlass.Int32(4 * g)
                            ).data_ptr()
                        )
                        sums = [sums[j] + cutlass.select_(rank == cutlass.Int32(q), cutlass.Float32(acc[4 * g + j]),
                                                          parts[j]) for j in range(4)]  # fmt: skip
                    for j in cutlass.range_constexpr(4):
                        totals.append(sums[j])
                        if cutlass.const_expr(rope):
                            # The own-rank mailbox slot is unused: it carries the totals to the RoPE stage.
                            mailbox.store(
                                sums[j],
                                idx=(rank * cutlass.Int32(rows_per_rank) + local)
                                * cutlass.Int32(mb_stride)
                                + cutlass.Int32(4 * g + j),
                            )
                        elif cutlass.Int32(4 * g + j) < num_tokens:
                            y.store(
                                cutlass.BFloat16(sums[j]),
                                idx=cutlass.Int32((4 * g + j) * n_out) + n,
                            )
                if cutlass.const_expr(ar_push > 0):
                    # Every lane of an owner warp is an owner (rows_per_rank >= 32), so lane pairs hold row pairs.
                    prims.griddepcontrol(prims.GridDepAction.WAIT)
                    ar_par = [
                        _ld_volatile_s32(
                            ar_flags
                            + cutlass.Int64(4)
                            * (cutlass.Int64(t * 8) + cutlass.Int64(n // cutlass.Int32(n_out // 8)))
                        )
                        & cutlass.Int32(1)
                        for t in range(mma_n)
                    ]
                    for t in cutlass.range_constexpr(mma_n):
                        ar_hi = cute.arch.shuffle_sync_bfly(totals[t], offset=1)
                        if (lane % cutlass.Int32(2)) == cutlass.Int32(0):
                            if cutlass.Int32(t) < num_tokens:
                                ar_off = (
                                    cutlass.Int64(ar_par[t])
                                    * cutlass.Int64(ar_maxt * ar_world * (n_out // 2))
                                    + (cutlass.Int64(t * ar_world) + cutlass.Int64(ar_rank))
                                    * cutlass.Int64(n_out // 2)
                                    + (cutlass.Int64(n) // cutlass.Int64(2))
                                )
                                _st_global_b32(
                                    ar_ws + ar_off * cutlass.Int64(4),
                                    _bf16x2_word(totals[t], ar_hi),
                                )
        prims.barrier_cta_sync(1, thread_count=128)
        if cutlass.const_expr(rope):
            # Token t of request t / T sits at cache position kv_len - T + t % T. Q / K heads: GPT-NeoX RoPE over
            # the 128-wide head, bf16(cos * x -/+ sin * pair) on the bf16 projection; then E4M3 of x * scale, Q to
            # rope_q, K / V into the page of that position.
            pairs = rows_per_rank // 2
            rp_e = tx - cutlass.Int32(128)
            rp_p = rp_e % cutlass.Int32(pairs)
            rp_t = rp_e // cutlass.Int32(pairs)
            if rp_t < num_tokens:
                rp_max_blocks = rope_dims.load(idx=0)
                rp_tpb_log2 = rope_dims.load(idx=1)
                rp_hq = rope_dims.load(idx=2)
                rp_hkv = rope_dims.load(idx=3)
                rp_T = rope_dims.load(idx=4)
                rp_d = rank * cutlass.Int32(pairs) + rp_p
                rp_a = _bf16_round(
                    mailbox.load(
                        idx=(rank * cutlass.Int32(rows_per_rank) + rp_p) * cutlass.Int32(mb_stride)
                        + rp_t
                    )
                )
                rp_b = _bf16_round(
                    mailbox.load(
                        idx=(rank * cutlass.Int32(rows_per_rank) + cutlass.Int32(pairs) + rp_p)
                        * cutlass.Int32(mb_stride)
                        + rp_t
                    )
                )
                rp_req = rp_t // rp_T
                rp_tok = rope_i.load(idx=rp_req) - rp_T + (rp_t - rp_req * rp_T)
                rp_head = n_tile
                if rp_head < rp_hq + rp_hkv:
                    rp_c = rope_cs.load(idx=(rp_tok * cutlass.Int32(64) + rp_d) * cutlass.Int32(2))
                    rp_s = rope_cs.load(
                        idx=(rp_tok * cutlass.Int32(64) + rp_d) * cutlass.Int32(2)
                        + cutlass.Int32(1)
                    )
                    rp_ra = _rope_bits(rp_c, rp_a, -rp_s, rp_b).bitcast(cutlass.Float32)
                    rp_rb = _rope_bits(rp_c, rp_b, rp_s, rp_a).bitcast(cutlass.Float32)
                    rp_a = rp_ra
                    rp_b = rp_rb
                rp_qs = rope_scale.load(idx=0)
                rp_w = _e4m3x4(
                    rp_a * rp_qs, rp_b * rp_qs, cutlass.Float32(0.0), cutlass.Float32(0.0)
                )
                rp_lo = cutlass.Uint8(rp_w & cutlass.Int32(0xFF))
                rp_hi = cutlass.Uint8((rp_w >> cutlass.Int32(8)) & cutlass.Int32(0xFF))
                if rp_head < rp_hq:
                    rp_q = (rp_t * rp_hq + rp_head) * cutlass.Int32(128) + rp_d
                    rope_q.store(rp_lo, idx=rp_q)
                    rope_q.store(rp_hi, idx=rp_q + cutlass.Int32(64))
                else:
                    rp_kv = cutlass.select_(
                        rp_head < rp_hq + rp_hkv, cutlass.Int32(0), cutlass.Int32(1)
                    )
                    rp_kh = rp_head - rp_hq - rp_kv * rp_hkv
                    rp_page = rope_bt.load(
                        idx=(rp_req * cutlass.Int32(2) + rp_kv) * rp_max_blocks
                        + (rp_tok >> rp_tpb_log2)
                    )
                    # 64-bit byte offset: the pool view can exceed 2^31 bytes.
                    rp_dst = (
                        (cutlass.Int64(rp_page) * cutlass.Int64(rp_hkv) + cutlass.Int64(rp_kh))
                        << cutlass.Int64(rp_tpb_log2 + cutlass.Int32(7))
                    ) + cutlass.Int64(
                        (rp_tok & ((cutlass.Int32(1) << rp_tpb_log2) - cutlass.Int32(1)))
                        * cutlass.Int32(128)
                        + rp_d
                    )
                    rp_addr = rope_pool.data_ptr(0).toint() + rp_dst
                    cutlass.inttoptr(rp_addr, mem_space=1, dtype=cutlass.Uint8).store(rp_lo)
                    cutlass.inttoptr(
                        rp_addr + cutlass.Int64(64), mem_space=1, dtype=cutlass.Uint8
                    ).store(rp_hi)
        if tx == 128:
            _stamp(timers, timed, bx, 6)
            if cutlass.const_expr(ar_push == 2):
                # The all-reduce polls the pushed words; it launches after this CTA's rows are out, not at grid end.
                prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
        if warp_id == 4:
            prims.tcgen05_dealloc(
                cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32),
                tmem_cols(ring, nacc, a_tmem, my_tiles, mma_n),
            )


def _bytes_map(t, cols, rows, box_rows, swizzle):
    return cuda.create_tensor_map_tiled(
        global_address=t.iterator.toint(),
        dtype=cutlass.Uint8,
        global_dims=[cols, rows],
        global_strides=[cols // 16],
        box_dims=[CTA_K, box_rows],
        swizzle=swizzle,
    )


def _atom_map(t, atoms):
    return cuda.create_tensor_map_tiled(
        global_address=t.iterator.toint(),
        dtype=cutlass.Uint16,
        global_dims=[SF_ATOM_U16, atoms],
        global_strides=[SF_ATOM_BYTES // 16],
        box_dims=[SF_ATOM_U16, 1],
        swizzle=cuda.TensorMapSwizzle.none,
    )


@cute.jit
def m3_mx_gemv(
    w: cute.Tensor,  # [N, K] E4M3 as uint8, K contiguous
    wsf: cute.Tensor,  # weight scales, 128x4-swizzled, as uint8
    x: cute.Tensor,  # [M, K] E4M3 as uint8, M <= 8
    xsf: cute.Tensor,  # activation scales, 128x4-swizzled (atom k at byte 512 k), as uint8
    y: cute.Tensor,  # [M * N] bf16
    x_words: cute.Tensor,  # x_bf16: bf16 x [M, K] as int32 words (x / xsf are then unused placeholders)
    timers: cute.Tensor,  # [grid * 8] int64, written when `timed`
    num_tokens: cutlass.Int32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    split: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    contig: cutlass.Constexpr[bool],
    timed: cutlass.Constexpr[bool],
    nacc: cutlass.Constexpr[int],
    a_tmem: cutlass.Constexpr[bool],
    st_async: cutlass.Constexpr[bool],
    x_bf16: cutlass.Constexpr[bool],
    ar_ws: cutlass.Int64,
    ar_flags: cutlass.Int64,
    ar_rank: cutlass.Int32,
    ar_push: cutlass.Constexpr[int],
    ar_world: cutlass.Constexpr[int],
    ar_maxt: cutlass.Constexpr[int],
    mma_n: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
    din_i: cute.Tensor,  # din / rope inputs: see m3_mx_gemv_kernel (placeholders when off)
    din_table: cute.Tensor,
    din_hidden: cute.Tensor,
    din_w: cute.Tensor,
    din_f: cute.Tensor,
    din_vocab: cutlass.Int32,
    din_g: cute.Tensor,
    din_res: cute.Tensor,
    din_flags: cutlass.Int32,
    rope_cs: cute.Tensor,
    rope_i: cute.Tensor,
    rope_bt: cute.Tensor,
    rope_q: cute.Tensor,
    rope_pool: cute.Tensor,
    rope_scale: cute.Tensor,
    rope_dims: cute.Tensor,
    din: cutlass.Constexpr[bool],
    rope: cutlass.Constexpr[bool],
) -> None:
    k_tiles = num_k_tiles(k_in)
    tma_desc_w = _bytes_map(w, k_in, n_out, CTA_M, cuda.TensorMapSwizzle.s128b)
    tma_desc_wsf = _atom_map(wsf, (n_out // CTA_M) * k_tiles)
    tma_desc_x = _bytes_map(x, k_in, num_tokens, mma_n, cuda.TensorMapSwizzle.s128b)
    tma_desc_xsf = _atom_map(xsf, k_tiles)
    m3_mx_gemv_kernel(
        tma_desc_w, tma_desc_wsf, tma_desc_x, tma_desc_xsf, y, x_words, timers, num_tokens, n_out, k_in, split, ring,
        trigger_early, contig, timed, nacc, a_tmem, st_async, x_bf16, ar_ws, ar_flags, ar_rank, ar_push, ar_world,
        ar_maxt, mma_n,
        din_i, din_table, din_hidden, din_w, din_f, din_vocab, din_g, din_res, din_flags, rope_cs,
        rope_i, rope_bt, rope_q, rope_pool, rope_scale, rope_dims, din, rope,
    ).launch(
        grid=((n_out // CTA_M) * split, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(split, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )  # fmt: skip
