# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 Helix generation tail in one kernel: split-KV fold, V up-projection, exchange, combine.

Under Helix every CP rank decodes all ``H`` MLA heads over its ``1 / P`` slice of the KV cache, and
each rank then needs, for its ``H / P`` heads, the partial outputs and softmax statistics of all
``P`` ranks. With the V up-projection before the exchange (``MLA._helix_project_v_partials``), one
launch of :class:`K3HelixTailKernel` computes what the packed split-KV reduction
(``fp4_mla_packed_reduce.reduce_packed_splits``), the per-head V up-projection BMM, the Helix
all-to-all and ``helix_post_process_native`` compute one after another. Every CTA (16 warps) owns
one head and ``tiles_per_cta`` tiles of 16 tokens; it first sends, then receives:

1. ``W_UV`` of its head goes into shared memory with bulk copies issued before
   ``griddepcontrol.wait`` (it does not depend on the previous grid).
2. Fold. Warp ``w`` builds 256-column halves ``(row w // 2, half w % 2)`` and ``(row 8 + w // 2,
   half w % 2)`` of the tile's latent rows in shared memory. A row whose request ran several
   decode tasks is folded from the FP32 partials with the statistics and partials of all its tasks
   (up to :data:`FOLD_TASKS`) loaded in one round, then the arithmetic of the reduction kernel's
   ``_reduce_packed_row`` in its order (beyond :data:`FOLD_TASKS` tasks, ``_reduce_packed_row``
   itself): the BF16 row and its ``(max, sum)`` statistics are bitwise the reduction's. A row the
   decode stored directly (single-task request, or no split partials at all) is copied with its
   statistics.
3. V up-projection. ``mma.sync`` BF16 with FP32 accumulation: warp ``w`` computes output columns
   ``8 w .. 8 w + 7`` of the 16 rows over K = 512. K is permuted inside every 32-column block so
   that each thread's A and B fragments of two consecutive K16 steps are one 16-byte shared load;
   the per-column dot products are unchanged, only their FP32 summation order differs from
   cuBLAS'.
4. Push. The BF16 outputs (staged token-major in shared memory) and the statistics go with
   16-byte (8-byte) stores straight into the head owner's Lamport receive buffer, record
   ``(token, this rank)``: the BF16 values of the owner's ``H / P`` heads, then their FP32
   ``(max, sum)`` pairs. Rows of tokens this rank owns no KV for (``zero_kv_mask``) are sent as
   the no-op contribution (zeros, ``(-inf, 0)``), and no word equal to the sentinel
   ``0xffffffff`` is sent (such words go as ``0x7fffffff``, also NaN).
5. Receive. CTA ``b`` combines tokens ``b, b + CTAs, ..`` in passes of 512 columns: poll the
   token's ``P`` records, combine them bitwise like ``helix_post_process_native`` (same ``expf``,
   reciprocal, butterflies and FMA order), store the BF16 ``[T, H / P * 128]`` output (optionally
   also o_proj's MXFP8 operand of ``bf16(out * gate)``) and re-arm the words read. A CTA only
   polls after its own pushes, and the grid is at most one CTA per SM, so every sender is
   scheduled whatever the receivers wait for. Each CTA releases the dependent grid (o_proj)
   between its pushes and its receive loop.

Receive buffers are Lamport buffers: every word starts as the sentinel, so a word that is not the
sentinel has arrived. A buffer has two halves used by alternate launches; a launch re-arms what it
read before it exits, and the next store into that half comes two launches later from a rank that
has finished the launch in between. The half is the parity of a launch counter: every CTA takes a
ticket from a packed 64-bit word (high half: launch number, low half: CTAs of the launch that have
arrived; the last arrival advances the launch number). Launches that share a workspace must be
ordered on one stream on every rank, in the same sequence on every rank. Every CTA takes its
ticket after ``griddepcontrol.wait``, when every earlier launch on the stream has completed, so
tickets of consecutive launches never interleave.

Numerics: the fold (latent rows and statistics) and the combine are bitwise those of the separate
kernels; the V up-projection rounds the same FP32 dot products to BF16 after another summation
order than the cuBLAS BMM, so the result agrees with the four-kernel path to BF16 rounding.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from ...attention.backends.fp4_mla.fp4_mla_cutedsl_mufu16 import LOG2_E
from ...attention.backends.fp4_mla.fp4_mla_packed_reduce import (
    PackedSplitPartials,
    _reduce_packed_row,
)
from . import fp8_block_scale_low_m_gemm as low_m_gemm
from .raw_ptx import add_f32 as _fadd
from .raw_ptx import asm as _asm
from .raw_ptx import atom_add_u64 as _atom_add_u64
from .raw_ptx import bulk_g2s as _bulk_g2s
from .raw_ptx import fv as _fv
from .raw_ptx import iv as _iv
from .raw_ptx import ld_nc_u8 as _ld_nc_u8
from .raw_ptx import ld_nc_v2 as _ld_nc_v2
from .raw_ptx import ld_nc_v4 as _ld_nc_v4
from .raw_ptx import ld_volatile_v4 as _ld_vol_v4
from .raw_ptx import lds_b32 as _lds_b32
from .raw_ptx import lds_b64 as _lds_b64
from .raw_ptx import lds_v4 as _lds_v4
from .raw_ptx import lv as _lv
from .raw_ptx import mbar_expect_tx as _mbar_expect_tx
from .raw_ptx import mbar_init_fenced as _mbar_init
from .raw_ptx import mbar_try_wait as _mbar_try_wait
from .raw_ptx import mma_bf16 as _mma_bf16
from .raw_ptx import mul_f32 as _fmul
from .raw_ptx import pack_bf16x2 as _pack_bf16x2
from .raw_ptx import shfl_bfly_b32 as _shfl_bfly_b32
from .raw_ptx import shfl_bfly_f32 as _shfl_bfly
from .raw_ptx import st_sentinel_v4 as _st_sentinel_v4
from .raw_ptx import st_u8 as _st_u8
from .raw_ptx import st_v2 as _st_v2
from .raw_ptx import st_v4 as _st_v4
from .raw_ptx import sts_b32 as _sts_b32
from .raw_ptx import sts_b64 as _sts_b64
from .raw_ptx import sts_v4 as _sts_v4
from .raw_ptx import unpack as _unpack
from .utils import griddepcontrol_launch_dependents as _griddepcontrol_launch_dependents
from .utils import griddepcontrol_wait as _griddepcontrol_wait

__all__ = [
    "LATENT_DIM",
    "SENTINEL",
    "V_HEAD_DIM",
    "can_run",
    "k3_helix_tail",
    "record_bytes",
]

#: Latent (kv_lora_rank) and value (v_head_dim) widths of a head.
LATENT_DIM = 512
V_HEAD_DIM = 128
#: The empty-word sentinel of the receive buffers (and its replacement in sent data).
SENTINEL = -1
#: FP32 bits of -inf (the max of a no-op contribution).
_NEG_INF_BITS = -8388608  # 0xff800000
#: Largest supported CP size (one 16-lane shuffle group holds the statistics of all sources).
MAX_CP = 16
#: Tokens per tile (the M of one ``mma.sync`` tile).
ROWS = 16
#: Threads per CTA: 16 warps; a warp folds two 256-column row halves and owns one n8 tile.
THREADS = 512
_WARPS = THREADS // 32
#: Decode tasks of a row folded with every load in one round; rows with more use
#: ``_reduce_packed_row``.
FOLD_TASKS = 8
#: Bytes of one head's ``W_UV`` (``[V_HEAD_DIM, LATENT_DIM]`` BF16) and of one latent row.
_W_HEAD_BYTES = V_HEAD_DIM * LATENT_DIM * 2
_LAT_ROW_BYTES = LATENT_DIM * 2
#: Row stride of ``W_UV`` in shared memory (64 bytes of padding against bank conflicts).
_W_ROW_BYTES = _LAT_ROW_BYTES + 64
#: Warps whose lane 0 issues an equal share of ``W_UV``'s row copies (the copies of one thread
#: issue one at a time).
_W_ISSUE_WARPS = 4
#: Row stride of the token-major BF16 output staging (128 values + 16 bytes against bank conflicts).
_C_ROW_BYTES = V_HEAD_DIM * 2 + 16


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Inline PTX helpers
# ---------------------------------------------------------------------------


@dsl_user_op
def _ld_vol_v2(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.volatile.global.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 2, loc=loc, ip=ip)


@dsl_user_op
def _all_valid2(w0, w1, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(w0, loc=loc, ip=ip), _iv(w1, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.b32 p, $1, -1;\nsetp.ne.and.b32 p, $2, -1, p;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _prep4(w0, w1, w2, w3, zero_kv, is_stats, *, loc=None, ip=None):
    """Words of one 16-byte vector as sent: the no-op contribution on zero-KV rows (zeros, or
    ``(-inf, 0, -inf, 0)`` for a statistics vector), and no word equal to the sentinel."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(w, loc=loc, ip=ip) for w in (w0, w1, w2, w3)]
        + [_iv(zero_kv, loc=loc, ip=ip), _iv(is_stats, loc=loc, ip=ip)],
        "{\n.reg .pred pz, ps, p;\n.reg .b32 m, x;\n"
        "setp.ne.u32 pz, $8, 0;\nsetp.ne.u32 ps, $9, 0;\n"
        "selp.b32 m, 0xff800000, 0, ps;\n"
        "selp.b32 x, m, $4, pz;\nsetp.eq.b32 p, x, -1;\nselp.b32 $0, 0x7fffffff, x, p;\n"
        "selp.b32 x, 0, $5, pz;\nsetp.eq.b32 p, x, -1;\nselp.b32 $1, 0x7fffffff, x, p;\n"
        "selp.b32 x, m, $6, pz;\nsetp.eq.b32 p, x, -1;\nselp.b32 $2, 0x7fffffff, x, p;\n"
        "selp.b32 x, 0, $7, pz;\nsetp.eq.b32 p, x, -1;\nselp.b32 $3, 0x7fffffff, x, p;\n}",
        "=r,=r,=r,=r,r,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _all_valid4(w0, w1, w2, w3, *, loc=None, ip=None):
    """1 if none of the four words is the sentinel."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(w, loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, $1, -1;\nsetp.ne.and.b32 p, $2, -1, p;\n"
            "setp.ne.and.b32 p, $3, -1, p;\nsetp.ne.and.b32 p, $4, -1, p;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_sentinel_v2(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nmov.b32 t, -1;\nst.global.v2.b32 [$0], {t, t};\n}",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _as_f32(bits, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_iv(bits, loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _fmax(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rcp(a, *, loc=None, ip=None):
    """IEEE reciprocal (what ``1.F / x`` compiles to without fast math)."""
    return Float32(
        _asm(T.f32(), [_fv(a, loc=loc, ip=ip)], "rcp.rn.f32 $0, $1;", "=f,f", loc=loc, ip=ip)
    )


@dsl_user_op
def _corrected_exp(mx, gmax, total, *, loc=None, ip=None):
    """``total * expf(mx - gmax)`` with CUDA's (non fast-math) ``expf`` instruction sequence."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(mx, loc=loc, ip=ip), _fv(gmax, loc=loc, ip=ip), _fv(total, loc=loc, ip=ip)],
            "{\n.reg .f32 d, t, j, f, e, s;\n.reg .b32 b;\n"
            "sub.rn.f32 d, $1, $2;\n"
            "fma.rn.sat.f32 t, d, 0f3BBB989D, 0f3F000000;\n"
            "fma.rm.f32 t, t, 0f437C0000, 0f4B400001;\n"
            "add.rn.f32 j, t, 0fCB40007F;\n"
            "neg.f32 j, j;\n"
            "fma.rn.f32 f, d, 0f3FB8AA3B, j;\n"
            "fma.rn.f32 f, d, 0f32A57060, f;\n"
            "ex2.approx.ftz.f32 e, f;\n"
            "mov.b32 b, t;\nshl.b32 b, b, 23;\nmov.b32 s, b;\n"
            "mul.rn.f32 e, s, e;\n"
            "mul.rn.f32 $0, e, $3;\n}",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_idx(value, lane, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(value, loc=loc, ip=ip), _iv(lane, loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 31, -1;",
            "=f,f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2_fma(word, scale, acc_lo, acc_hi, *, loc=None, ip=None):
    """``(fma(lo, scale, acc_lo), fma(hi, scale, acc_hi))`` of the BF16 pair in ``word``."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [
            _iv(word, loc=loc, ip=ip),
            _fv(scale, loc=loc, ip=ip),
            _fv(acc_lo, loc=loc, ip=ip),
            _fv(acc_hi, loc=loc, ip=ip),
        ],
        "{\n.reg .b32 l, h;\n.reg .f32 fl, fh;\n"
        "shl.b32 l, $2, 16;\nand.b32 h, $2, 0xffff0000;\n"
        "mov.b32 fl, l;\nmov.b32 fh, h;\n"
        "fma.rn.f32 $0, fl, $3, $4;\nfma.rn.f32 $1, fh, $3, $5;\n}",
        "=f,=f,r,f,f,f",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 2, loc=loc, ip=ip)


@dsl_user_op
def _opaque_f32(value, *, loc=None, ip=None):
    """``value`` through an opaque move: the compiler cannot merge expressions on it with
    expressions on ``value``."""
    return Float32(
        _asm(T.f32(), [_fv(value, loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,f", loc=loc, ip=ip)
    )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@cute.jit
def _fold_half(
    mPartial: cute.Tensor,
    mRowMax: cute.Tensor,
    mRowSum: cute.Tensor,
    sLat: cute.Tensor,
    sMax: cute.Tensor,
    sSum: cute.Tensor,
    lane: Int32,
    row: Int32,
    local_row: Int32,
    half: Int32,
    task_begin: Int32,
    request_tasks: Int32,
    physical_m: Int32,
    scale: Float32,
    output_scale: Float32,
) -> None:
    """``_reduce_packed_row`` of one 256-column half (``band = half`` of 256 columns) of a row
    with at most :data:`FOLD_TASKS` tasks, with the statistics and partials of every task loaded
    before any is consumed (the reduction issues them in three rounds). The weights, the
    denominator and the numerators are accumulated in the reduction's task order with the same
    operations, so the BF16 row and its statistics are bitwise the reduction's."""
    band_columns: cutlass.Constexpr = 256
    columns_per_lane: cutlass.Constexpr = 4
    band_base = half * Int32(band_columns) + lane * Int32(columns_per_lane)
    gPartial_arr = cutlass.make_array_view(mPartial)
    gOut_arr = cutlass.make_array_view(sLat)
    sums = []
    maxima = []
    vecs = []
    for k in cutlass.range_constexpr(FOLD_TASKS):
        # Out-of-range slots re-read the last valid task with zero weight (as the reduction).
        task = task_begin + cutlass.min(Int32(k), request_tasks - Int32(1))
        sums.append(mRowSum[task, local_row])
        maxima.append(mRowMax[task, local_row])
        partial_row = (task * physical_m + local_row) * Int32(LATENT_DIM) + band_base
        for group in cutlass.range_constexpr(2):
            vecs.append(
                gPartial_arr.subview(partial_row + Int32(group * 128))
                .data_ptr()
                .load(count=columns_per_lane, alignment=4 * columns_per_lane)
            )
    global_row_max = cutlass.Float32(-cutlass.Float32.inf)
    for k in cutlass.range_constexpr(FOLD_TASKS):
        if Int32(k) < request_tasks:
            if sums[k] != cutlass.Float32(0.0):
                global_row_max = cute.arch.fmax(global_row_max, maxima[k] * scale)
    numerator0 = cutlass.Vector.from_elements(
        tuple(cutlass.Float32(0.0) for _ in range(columns_per_lane)), dtype=cutlass.Float32
    )
    numerator1 = cutlass.Vector.from_elements(
        tuple(cutlass.Float32(0.0) for _ in range(columns_per_lane)), dtype=cutlass.Float32
    )
    denominator = cutlass.Float32(0.0)
    for k in cutlass.range_constexpr(FOLD_TASKS):
        task_weight = cutlass.Float32(0.0)
        if Int32(k) < request_tasks:
            task_row_sum = sums[k]
            if task_row_sum != cutlass.Float32(0.0):
                # A product of its own, as the reduction's: its contraction with the subtraction
                # must not depend on the global-max product of the same task.
                task_max = _opaque_f32(maxima[k])
                task_weight = cute.exp2(task_max * scale - global_row_max, fastmath=True)
                denominator += task_row_sum * task_weight
        # Empty tasks keep zero weight; their payload is never trusted.
        if task_weight != cutlass.Float32(0.0):
            numerator0 = numerator0 + vecs[2 * k] * task_weight
            numerator1 = numerator1 + vecs[2 * k + 1] * task_weight
    normalizer = cutlass.Float32(0.0)
    if denominator != cutlass.Float32(0.0):
        normalizer = output_scale / denominator
    out_row = row * Int32(LATENT_DIM) + band_base
    gOut_arr.store(
        (numerator0 * normalizer).to(sLat.element_type),
        idx=out_row,
        vector_size=columns_per_lane,
        alignment=2 * columns_per_lane,
    )
    gOut_arr.store(
        (numerator1 * normalizer).to(sLat.element_type),
        idx=out_row + Int32(128),
        vector_size=columns_per_lane,
        alignment=2 * columns_per_lane,
    )
    if half == Int32(0) and lane == Int32(0):
        # Helix combine convention: natural-log-domain scaled max and the sum relative to it.
        stats_max = cutlass.Float32(-cutlass.Float32.inf)
        if denominator != cutlass.Float32(0.0):
            stats_max = global_row_max * cutlass.Float32(1.0 / LOG2_E)
        sMax[row] = stats_max
        sSum[row] = denominator


class K3HelixTailKernel:
    """Fold + V up-projection + exchange + combine of one Helix MLA layer (module docstring).

    Args:
        cp_size: ``P`` (at most 8, or 16).
        num_heads: ``H``, MLA heads every rank decodes (a multiple of ``2 * P``).
        fold: The decode left FP32 split partials (fold them); else every row was stored
            directly as BF16 with its statistics.
        has_mask: Apply a per-token ``zero_kv_mask`` (uint8) on the sender side.
        use_pdl: Launch with programmatic dependent launch.
        mx_out: The receivers also store the o_proj operand: the MXFP8 form (E4M3 codes and one
            UE8M0 exponent per 1x32 group, R128c4 scale layout) of ``bf16(out * gate)``, bitwise
            the gated low-M GEMM's in-kernel quantization of it.
    """

    def __init__(
        self,
        cp_size: int,
        num_heads: int,
        fold: bool,
        has_mask: bool,
        use_pdl: bool,
        mx_out: bool,
    ):
        if not can_run(cp_size, num_heads, LATENT_DIM, V_HEAD_DIM):
            raise ValueError(f"cp_size={cp_size}, num_heads={num_heads}")
        self.P = cp_size
        self.H = num_heads
        self.H_LOC = num_heads // cp_size
        self.fold = fold
        self.has_mask = has_mask
        self.use_pdl = use_pdl
        self.mx_out = mx_out
        self.threads = THREADS
        # 16-byte vectors of a (token, source) record: the BF16 values of the H / P heads, then
        # their FP32 (max, sum) pairs (two heads per vector).
        self.DV = self.H_LOC * V_HEAD_DIM // 8
        self.SV = self.H_LOC // 2
        self.RV = self.DV + self.SV
        # Columns (16-byte vectors) per head; a 16-lane group of a warp combines one head.
        self.CPH = V_HEAD_DIM // 8
        # A CTA combines a token's heads in passes of THREADS columns.
        self.passes = _ceil_div(self.DV, THREADS)

    @cute.jit
    def __call__(
        self,
        partial_ptr: cute.Pointer,
        maxima_ptr: cute.Pointer,
        sums_ptr: cute.Pointer,
        plan_ptr: cute.Pointer,
        q_scale_ptr: cute.Pointer,
        kv_scale_ptr: cute.Pointer,
        rows_addr: Int64,
        stats_addr: Int64,
        w_addr: Int64,
        mask_addr: Int64,
        out_addr: Int64,
        ws_addr: Int64,
        state_addr: Int64,
        gate_addr: Int64,
        xq_addr: Int64,
        xs_addr: Int64,
        rank_stride: Int64,
        half_bytes: Int64,
        softmax_scale_log2: Float32,
        output_scale: Float32,
        num_tokens: Int32,
        query_len: Int32,
        physical_m: Int32,
        num_tasks: Int32,
        plan_len: Int32,
        gate_ld: Int32,
        rank: Int32,
        tiles_per_cta: Int32,
        n_ctas: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            half: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 2], 16]
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 2], 16]
            smax: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, ROWS], 16]
            ssum: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, ROWS], 16]
            lat: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, ROWS * LATENT_DIM], 128]
            cst: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, ROWS * _C_ROW_BYTES], 128]
            wv: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, V_HEAD_DIM * _W_ROW_BYTES], 128
            ]

        self.shared_storage = SharedStorage
        mPartial = cute.make_tensor(
            partial_ptr,
            cute.make_layout(
                (num_tasks, physical_m, LATENT_DIM),
                stride=(cutlass.Int64(physical_m) * LATENT_DIM, LATENT_DIM, 1),
            ),
        )
        row_state = cute.make_layout((num_tasks, physical_m), stride=(physical_m, 1))
        mRowMax = cute.make_tensor(maxima_ptr, row_state)
        mRowSum = cute.make_tensor(sums_ptr, row_state)
        mPlan = cute.make_tensor(plan_ptr, cute.make_layout((plan_len,), stride=(1,)))
        mQScale = cute.make_tensor(q_scale_ptr, cute.make_layout((1,), stride=(1,)))
        mKvScale = cute.make_tensor(kv_scale_ptr, cute.make_layout((1,), stride=(1,)))
        self.kernel(
            mPartial,
            mRowMax,
            mRowSum,
            mPlan,
            mQScale,
            mKvScale,
            rows_addr,
            stats_addr,
            w_addr,
            mask_addr,
            out_addr,
            ws_addr,
            state_addr,
            gate_addr,
            xq_addr,
            xs_addr,
            rank_stride,
            half_bytes,
            softmax_scale_log2,
            output_scale,
            num_tokens,
            query_len,
            physical_m,
            gate_ld,
            rank,
            tiles_per_cta,
            n_ctas,
        ).launch(
            grid=(n_ctas, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mPartial: cute.Tensor,
        mRowMax: cute.Tensor,
        mRowSum: cute.Tensor,
        mPlan: cute.Tensor,
        mQScale: cute.Tensor,
        mKvScale: cute.Tensor,
        rows_addr: Int64,
        stats_addr: Int64,
        w_addr: Int64,
        mask_addr: Int64,
        out_addr: Int64,
        ws_addr: Int64,
        state_addr: Int64,
        gate_addr: Int64,
        xq_addr: Int64,
        xs_addr: Int64,
        rank_stride: Int64,
        half_bytes: Int64,
        softmax_scale_log2: Float32,
        output_scale: Float32,
        num_tokens: Int32,
        query_len: Int32,
        physical_m: Int32,
        gate_ld: Int32,
        rank: Int32,
        tiles_per_cta: Int32,
        n_ctas: Int32,
    ):
        H = self.H
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        half_slot = storage.half.data_ptr().toint()
        mbar = storage.mbar.data_ptr().toint()
        lat_base = storage.lat.data_ptr().toint()
        smax_base = storage.smax.data_ptr().toint()
        ssum_base = storage.ssum.data_ptr().toint()
        wv_base = storage.wv.data_ptr().toint()
        cst_base = storage.cst.data_ptr().toint()
        sLat = storage.lat.get_tensor(cute.make_layout((ROWS, LATENT_DIM), stride=(LATENT_DIM, 1)))
        sMax = storage.smax.get_tensor(cute.make_layout((ROWS,), stride=(1,)))
        sSum = storage.ssum.get_tensor(cute.make_layout((ROWS,), stride=(1,)))
        head = bidx % Int32(H)
        cta_tile = bidx // Int32(H)
        # W_UV of the head does not depend on the previous grid: its rows go into shared memory
        # (padded stride) while this grid waits. Lane 0 of each of the last _W_ISSUE_WARPS warps
        # issues an equal share of the row copies (the copies of one thread are issued one after
        # another; thread 0 takes the ticket below).
        if warp >= Int32(_WARPS - _W_ISSUE_WARPS):
            if warp == Int32(_WARPS - 1):
                if lane == 0:
                    _mbar_init(mbar, 1)
                    _mbar_expect_tx(mbar, _W_HEAD_BYTES)
            cute.arch.barrier(barrier_id=1, number_of_threads=32 * _W_ISSUE_WARPS)
            if lane == 0:
                w_head = w_addr + Int64(head) * _W_HEAD_BYTES
                rows_per_warp = V_HEAD_DIM // _W_ISSUE_WARPS
                first = (warp - Int32(_WARPS - _W_ISSUE_WARPS)) * rows_per_warp
                for r in cutlass.range(rows_per_warp, unroll=8):
                    n = first + r
                    _bulk_g2s(
                        wv_base + n * Int32(_W_ROW_BYTES),
                        w_head + Int64(n) * _LAT_ROW_BYTES,
                        _LAT_ROW_BYTES,
                        mbar,
                    )
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        # The ticket is taken after the wait (every earlier launch on the stream has completed);
        # its result is first needed after the first fold.
        ticket = Int64(0)
        if tidx == 0:
            ticket = _atom_add_u64(state_addr, Int64(1))
        scale = Float32(0.0)
        if cutlass.const_expr(self.fold):
            # The packed reduction's scale expression (same operations, same result).
            scale = softmax_scale_log2 / (mQScale[0] * mKvScale[0])
        for tt in cutlass.range(tiles_per_cta, unroll=1):
            token0 = (cta_tile * tiles_per_cta + tt) * Int32(ROWS)
            if token0 < num_tokens:
                # Zero-KV flag of the token this thread pushes (row tidx / 16), loaded ahead.
                zero_kv = Int32(0)
                if cutlass.const_expr(self.has_mask):
                    push_t = token0 + tidx // Int32(16)
                    if tidx < Int32(ROWS * 16) and push_t < num_tokens:
                        zero_kv = _ld_nc_u8(mask_addr + Int64(push_t))
                self.fold_tile(
                    warp, lane, head, token0, sLat, sMax, sSum, lat_base, smax_base, ssum_base,
                    mPartial, mRowMax, mRowSum, mPlan, rows_addr, stats_addr, scale,
                    output_scale, num_tokens, query_len, physical_m,
                )  # fmt: skip
                if tt == 0:
                    if tidx == 0:
                        if (ticket & Int64(0xFFFFFFFF)) == Int64(n_ctas) - Int64(1):
                            # Last arrival: clear the arrivals and advance the launch number.
                            _atom_add_u64(state_addr, Int64(1 << 32) - Int64(n_ctas))
                        _sts_b64(half_slot, ((ticket >> 32) & Int64(1)) * half_bytes)
                cute.arch.barrier()
                if tt == 0:
                    landed = Int32(0)
                    while landed == 0:
                        landed = _mbar_try_wait(mbar, Int32(0))
                self.project_and_push(
                    tidx, warp, lane, head, token0, lat_base, smax_base, ssum_base, wv_base,
                    cst_base, zero_kv, ws_addr, rank_stride, _lds_b64(half_slot), num_tokens, rank,
                )  # fmt: skip
        # The dependent grid (o_proj) is released once this CTA's pushes are issued: its CTAs take
        # the SMs of exiting senders and stream their weights while the receivers wait for peers.
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_launch_dependents()
        self.receive(
            tidx,
            bidx,
            out_addr,
            ws_addr + Int64(rank) * rank_stride + _lds_b64(half_slot),
            num_tokens,
            n_ctas,
            gate_addr,
            gate_ld,
            xq_addr,
            xs_addr,
        )

    @cute.jit
    def fold_tile(
        self,
        warp: Int32,
        lane: Int32,
        head: Int32,
        token0: Int32,
        sLat: cute.Tensor,
        sMax: cute.Tensor,
        sSum: cute.Tensor,
        lat_base: Int32,
        smax_base: Int32,
        ssum_base: Int32,
        mPartial: cute.Tensor,
        mRowMax: cute.Tensor,
        mRowSum: cute.Tensor,
        mPlan: cute.Tensor,
        rows_addr: Int64,
        stats_addr: Int64,
        scale: Float32,
        output_scale: Float32,
        num_tokens: Int32,
        query_len: Int32,
        physical_m: Int32,
    ):
        """Latent rows (BF16) and (max, sum) statistics of tokens ``token0 .. token0 + 15`` of
        ``head`` into shared memory; rows past the batch are zeros."""
        H = self.H
        half = warp % Int32(2)
        for slot in cutlass.range_constexpr(2):
            i = warp // Int32(2) + Int32(slot * _WARPS // 2)
            t = token0 + i
            # This lane's 16 bytes of the 512-byte half row.
            lat_vec = lat_base + i * Int32(_LAT_ROW_BYTES) + half * Int32(_LAT_ROW_BYTES // 2)
            lat_vec = lat_vec + lane * Int32(16)
            if t < num_tokens:
                row = Int64(t) * H + Int64(head)
                folded = Int32(0)
                if cutlass.const_expr(self.fold):
                    request = t // query_len
                    local_row = (t - request * query_len) * Int32(H) + head
                    task_begin = mPlan[Int32(2) + request]
                    request_tasks = mPlan[Int32(3) + request] - task_begin
                    if request_tasks > Int32(1):
                        folded = Int32(1)
                        if request_tasks <= Int32(FOLD_TASKS):
                            _fold_half(
                                mPartial, mRowMax, mRowSum, sLat, sMax, sSum, lane, i, local_row,
                                half, task_begin, request_tasks, physical_m, scale, output_scale,
                            )  # fmt: skip
                        else:
                            _reduce_packed_row(
                                mPartial, mRowMax, mRowSum, sLat, sMax, sSum, lane, i, local_row,
                                half, task_begin, request_tasks, physical_m, scale, output_scale,
                                256, True,
                            )  # fmt: skip
                    else:
                        if half == Int32(0) and lane == Int32(0):
                            # The packed reduction's statistics of a single-task request.
                            row_sum = mRowSum[task_begin, local_row]
                            stats_max = cutlass.Float32(-cutlass.Float32.inf)
                            if row_sum != cutlass.Float32(0.0):
                                stats_max = (
                                    mRowMax[task_begin, local_row]
                                    * scale
                                    * cutlass.Float32(1.0 / LOG2_E)
                                )
                            sMax[i] = stats_max
                            sSum[i] = row_sum
                else:
                    if half == Int32(0) and lane == Int32(0):
                        st = _ld_nc_v2(stats_addr + row * 8)
                        _sts_b32(smax_base + i * 4, st[0])
                        _sts_b32(ssum_base + i * 4, st[1])
                if folded == 0:
                    # The decode stored this row itself: copy this lane's 16 bytes of the half.
                    src = rows_addr + row * _LAT_ROW_BYTES + Int64(half * (_LAT_ROW_BYTES // 2))
                    _sts_v4(lat_vec, *_ld_nc_v4(src + Int64(lane) * 16))
            else:
                zero = Int32(0)
                _sts_v4(lat_vec, zero, zero, zero, zero)

    @cute.jit
    def project_and_push(
        self,
        tidx: Int32,
        warp: Int32,
        lane: Int32,
        head: Int32,
        token0: Int32,
        lat_base: Int32,
        smax_base: Int32,
        ssum_base: Int32,
        wv_base: Int32,
        cst_base: Int32,
        zero_kv: Int32,
        ws_addr: Int64,
        rank_stride: Int64,
        half_off: Int64,
        num_tokens: Int32,
        rank: Int32,
    ):
        """V up-projection of the tile (warp ``w``: output columns ``8 w .. 8 w + 7``) and the
        push of its rows and statistics into the head owner's receive buffer (``zero_kv``: the
        pushed token owns no KV on this rank)."""
        g = lane // 4
        q = lane % 4
        # Two accumulators (the two k16 steps of every 32-column block) halve the MMA dependency
        # chain; they are added at the end.
        accs = [[Float32(0.0)] * 4 for _ in range(2)]
        w_row = wv_base + (warp * 8 + g) * Int32(_W_ROW_BYTES)
        for u in cutlass.range_constexpr(LATENT_DIM // 32):
            col2 = (Int32(u * 32) + q * 8) * 2
            b = _lds_v4(w_row + col2)
            a_lo = _lds_v4(lat_base + g * Int32(_LAT_ROW_BYTES) + col2)
            a_hi = _lds_v4(lat_base + (g + 8) * Int32(_LAT_ROW_BYTES) + col2)
            for v in cutlass.range_constexpr(2):
                accs[v] = list(
                    _mma_bf16(
                        (a_lo[2 * v], a_hi[2 * v], a_lo[2 * v + 1], a_hi[2 * v + 1]),
                        b[2 * v],
                        b[2 * v + 1],
                        accs[v],
                    )
                )
        acc = [accs[0][c] + accs[1][c] for c in range(4)]
        # The BF16 outputs are staged token-major in their own buffer, so the next tile's fold
        # (after the barrier below) cannot overwrite what this tile's pushes still read; the
        # statistics this thread pushes are read before that barrier for the same reason.
        n = warp * 8 + q * 2
        _sts_b32(cst_base + g * Int32(_C_ROW_BYTES) + n * 2, _pack_bf16x2(acc[0], acc[1]))
        _sts_b32(cst_base + (g + 8) * Int32(_C_ROW_BYTES) + n * 2, _pack_bf16x2(acc[2], acc[3]))
        i = tidx // 16
        vec = tidx % 16
        mx = _lds_b32(smax_base + (i % Int32(ROWS)) * 4)
        sm = _lds_b32(ssum_base + (i % Int32(ROWS)) * 4)
        cute.arch.barrier()
        # Row i's 16 vectors (and its statistics) into record (t, this rank) of the head owner.
        if tidx < Int32(ROWS * 16):
            t = token0 + i
            if t < num_tokens:
                words = _lds_v4(cst_base + i * Int32(_C_ROW_BYTES) + vec * 16)
                dst_rank = head // Int32(self.H_LOC)
                local_head = head - dst_rank * Int32(self.H_LOC)
                sent = _prep4(words[0], words[1], words[2], words[3], zero_kv, Int32(0))
                st = _prep4(mx, sm, Int32(0), Int32(0), zero_kv, Int32(1))
                record = (
                    ws_addr
                    + Int64(dst_rank) * rank_stride
                    + half_off
                    + Int64(t * self.P + rank) * (self.RV * 16)
                )
                _st_v4(record + Int64(local_head) * (V_HEAD_DIM * 2) + Int64(vec) * 16, *sent)
                if vec == 0:
                    _st_v2(record + Int64(self.DV * 16) + Int64(local_head) * 8, st[0], st[1])

    @cute.jit
    def receive(
        self,
        tidx: Int32,
        rcta: Int32,
        out_addr: Int64,
        recv_base: Int64,
        num_tokens: Int32,
        n_recv: Int32,
        gate_addr: Int64,
        gate_ld: Int32,
        xq_addr: Int64,
        xs_addr: Int64,
    ):
        """Combine the records of this CTA's tokens over ``passes`` groups of THREADS columns,
        with the instruction sequences of ``helix_post_process_native`` (bitwise). Thread ``c``
        of a pass owns one 16-byte column (8 output values) of one head; lane ``l`` of a 16-lane
        head group polls the statistics of source rank ``l``."""
        P, RV, DV, CPH = self.P, self.RV, self.DV, self.CPH
        lane = tidx % 32
        group = lane - lane % 16
        for t in cutlass.range(rcta, num_tokens, n_recv, unroll=1):
            rec0 = recv_base + Int64(t * (P * RV)) * 16
            for p in cutlass.range_constexpr(self.passes):
                c = tidx + Int32(p * THREADS)
                h = c // CPH
                col = c - h * CPH
                src_l = col % 16
                valid = Int32(c < DV)
                has_stats = valid * Int32(src_l < P)
                # With 16 columns per head (V_HEAD_DIM 128) every group owns its head's statistics.
                owns_stats = has_stats * Int32(col < 16)
                st_addr = rec0 + Int64(src_l * RV + DV) * 16 + Int64(h * 8)
                d_addr = rec0 + Int64(c) * 16
                stats = cute.make_rmem_tensor((2,), Int32)
                stats[0] = Int32(_NEG_INF_BITS)
                stats[1] = Int32(0)
                vals = cute.make_rmem_tensor((P * 4,), Int32)
                g = cute.make_rmem_tensor((4,), Int32)
                if cutlass.const_expr(self.mx_out):
                    # The gate is ready before this grid starts: its load overlaps the polls.
                    for e in cutlass.range_constexpr(4):
                        g[e] = Int32(0)
                    if valid != 0:
                        gw = _ld_nc_v4(gate_addr + (Int64(t) * Int64(gate_ld) + Int64(c * 8)) * 2)
                        for e in cutlass.range_constexpr(4):
                            g[e] = gw[e]
                if valid != 0:
                    ready = Int32(0)
                    while ready == 0:
                        ready = Int32(1)
                        if has_stats != 0:
                            sv = _ld_vol_v2(st_addr)
                            stats[0] = sv[0]
                            stats[1] = sv[1]
                            ready = _all_valid2(sv[0], sv[1])
                        for s in cutlass.range_constexpr(P):
                            w = _ld_vol_v4(d_addr + Int64(s * RV * 16))
                            for e in cutlass.range_constexpr(4):
                                vals[s * 4 + e] = w[e]
                            ready = ready & _all_valid4(*w)
                mx = _as_f32(stats[0])
                gmax = mx
                for off in (1, 2, 4, 8):
                    gmax = _fmax(gmax, _shfl_bfly(gmax, off))
                corrected = _corrected_exp(mx, gmax, _as_f32(stats[1]))
                total = corrected
                for off in (1, 2, 4, 8):
                    total = _fadd(total, _shfl_bfly(total, off))
                weight = _fmul(corrected, _rcp(total))
                acc = [Float32(0.0)] * 8
                for s in cutlass.range_constexpr(P):
                    ws = _shfl_idx(weight, group + s)
                    for e in cutlass.range_constexpr(4):
                        acc[2 * e], acc[2 * e + 1] = _bf16x2_fma(
                            vals[s * 4 + e], ws, acc[2 * e], acc[2 * e + 1]
                        )
                o = [_pack_bf16x2(acc[2 * e], acc[2 * e + 1]) for e in range(4)]
                if valid != 0:
                    _st_v4(out_addr + Int64(t * DV + c) * 16, *o)
                    for s in cutlass.range_constexpr(P):
                        _st_sentinel_v4(d_addr + Int64(s * RV * 16))
                if owns_stats != 0:
                    _st_sentinel_v2(st_addr)
                if cutlass.const_expr(self.mx_out):
                    self.store_mx(o, g, valid, t, c, xq_addr, xs_addr)

    @cute.jit
    def store_mx(
        self,
        o,
        g: cute.Tensor,
        valid: Int32,
        t: Int32,
        c: Int32,
        xq_addr: Int64,
        xs_addr: Int64,
    ):
        """MXFP8 of ``bf16(out * gate)`` for column ``c`` (8 values ``o``, gate words ``g``) of
        token ``t``: the gate multiply, 1x32 amax (the four threads of a group, by shuffles),
        UE8M0 exponent and E4M3 codes of the gated low-M GEMM's quantization. Every lane of the
        warp runs the shuffles; groups are either all valid or all invalid (``DV`` is a multiple
        of 16)."""
        K = self.DV * 8
        xg = low_m_gemm.gate_mul_bf16x8(o[0], o[1], o[2], o[3], g[0], g[1], g[2], g[3])
        amax = low_m_gemm.amax_bf16x8(xg[0], xg[1], xg[2], xg[3])
        amax = low_m_gemm.amax_pair(amax, _shfl_bfly_b32(amax, 1))
        amax = low_m_gemm.amax_pair(amax, _shfl_bfly_b32(amax, 2))
        e8 = low_m_gemm.ue8m0_of_amax(amax)
        inv_scale = low_m_gemm.bf16x2_inv_scale(e8)
        q0, q1 = low_m_gemm.quant_bf16x8_to_e4m3(xg[0], xg[1], xg[2], xg[3], inv_scale)
        if valid != 0:
            _st_v2(xq_addr + Int64(t) * K + Int64(c * 8), q0, q1)
            if (c % 4) == 0:
                # R128c4: atom (t // 128, group // 4) of 512 bytes, byte
                # (t % 32) * 16 + (t % 128 // 32) * 4 + group % 4.
                group = c // 4
                sf_off = (Int64(t >> 7) * (K // 128) + Int64(group >> 2)) * 512
                sf_off = sf_off + Int64((t & 31) * 16 + ((t & 127) >> 5) * 4 + (group & 3))
                _st_u8(xs_addr + sf_off, e8)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def can_run(cp_size: int, num_heads: int, latent_dim: int, v_head_dim: int) -> bool:
    """Whether :func:`k3_helix_tail` supports this Helix MLA geometry."""
    if latent_dim != LATENT_DIM or v_head_dim != V_HEAD_DIM:
        return False
    if not (1 <= cp_size <= 8 or cp_size == MAX_CP):
        return False
    return num_heads % (2 * cp_size) == 0


def record_bytes(num_heads: int, head_dim: int) -> int:
    """Bytes of one ``(token, source rank)`` record in a receive buffer of ``num_heads`` heads."""
    return num_heads * head_dim * 2 + num_heads * 8


def _grid(num_heads: int, num_tokens: int, max_ctas: int) -> tuple[int, int]:
    """(tiles_per_cta, n_ctas): every head gets the same number of CTAs, at most max_ctas
    in all, each with the fewest tiles that cover the tokens."""
    groups = max_ctas // num_heads
    if groups == 0:
        raise ValueError(f"the Helix tail needs one CTA per head: {num_heads} heads > {max_ctas}")
    n_tiles = _ceil_div(num_tokens, ROWS)
    tiles_per_cta = _ceil_div(n_tiles, groups)
    n_ctas = num_heads * _ceil_div(n_tiles, tiles_per_cta)
    assert n_ctas <= max_ctas
    return tiles_per_cta, n_ctas


def _gmem(dtype, address: int, align: int):
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


def _split_ptrs(address: int, fold: bool):
    """The six split-state pointers (partials, maxima, sums, plan, Q / KV scales)."""
    align = 32 if fold else 4
    f32, i32 = cutlass.Float32, cutlass.Int32
    return (
        _gmem(f32, address, align),
        _gmem(f32, address, align),
        _gmem(f32, address, align),
        _gmem(i32, address, 4),
        _gmem(f32, address, 4),
        _gmem(f32, address, 4),
    )


@functools.cache
def _compiled(
    cp_size: int, num_heads: int, fold: bool, has_mask: bool, use_pdl: bool, mx_out: bool
):
    kernel = K3HelixTailKernel(cp_size, num_heads, fold, has_mask, use_pdl, mx_out)
    a, f, i = Int64(1 << 20), Float32(1.0), Int32(1)
    return cute.compile(
        kernel, *_split_ptrs(1 << 20, fold), *(a,) * 12, f, f, *(i,) * 9,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )  # fmt: skip


def k3_helix_tail(
    rows: torch.Tensor,
    stats: torch.Tensor | None,
    w_uv: torch.Tensor,
    out: torch.Tensor,
    *,
    fold: PackedSplitPartials | None,
    zero_kv_mask: torch.Tensor | None,
    cp_rank: int,
    cp_size: int,
    ws_addr: int,
    rank_stride: int,
    half_bytes: int,
    state: torch.Tensor,
    max_ctas: int,
    use_pdl: bool = True,
    mx_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Launch the Helix tail of ``T`` tokens into ``out``.

    Args:
        rows: BF16 ``[T * H, 512]`` latent rows of the decode (row ``t * H + h``). With ``fold``,
            only the rows of single-task requests are read (the decode stored them).
        stats: FP32 ``[T * H, 2]`` ``(max, sum)`` of ``rows`` (natural-log max); read only
            without ``fold``.
        w_uv: BF16 ``[H, 128, 512]`` V up-projection of every head.
        out: BF16 ``[T, H / P * 128]`` combined output of this rank's heads.
        fold: The packed decode's unreduced split state, or None.
        zero_kv_mask: Optional bool ``[T]``, True where this rank owns no KV of the token.
        cp_rank, cp_size: This rank and ``P``.
        ws_addr, rank_stride, half_bytes: Rank 0's receive buffer in this process's address space
            (rank ``r``'s at ``ws_addr + r * rank_stride``), two halves of ``half_bytes`` each,
            every word armed with the sentinel.
        state: int64 launch counter shared by all launches on the workspace.
        max_ctas: Cap on the CTAs (at most one per SM: every CTA polls after its own pushes, so
            all of them must be resident together). A CTA folds and sends ``tiles_per_cta`` tiles
            of 16 tokens of one head, the fewest that keep the grid within the cap.
        use_pdl: Programmatic dependent launch.
        mx_out: Optional ``(gate, xq, xs)``: the BF16 ``[T, H / P * 128]`` gate (row stride a
            multiple of 8), and outputs for the MXFP8 form of ``bf16(out * gate)``: E4M3 ``xq``
            ``[T, H / P * 128]`` and UE8M0 ``xs`` (uint8, ``ceil(T / 128) * 128 * H / P * 4``
            bytes) in the R128c4 layout of ``fp8_block_scale_low_m_gemm``'s prequantized input.
    """
    num_heads = w_uv.shape[0]
    num_tokens = rows.shape[0] // num_heads
    if num_tokens == 0:
        return
    if num_tokens * cp_size * record_bytes(num_heads // cp_size, V_HEAD_DIM) > half_bytes:
        raise ValueError("the Helix tail receive buffer is too small for this many tokens")
    has_mask = zero_kv_mask is not None
    tiles_per_cta, n_ctas = _grid(num_heads, num_tokens, max_ctas)
    if fold is not None:
        ptrs = (
            _gmem(cutlass.Float32, fold.partial.data_ptr(), 32),
            _gmem(cutlass.Float32, fold.maxima.data_ptr(), 32),
            _gmem(cutlass.Float32, fold.sums.data_ptr(), 32),
            _gmem(cutlass.Int32, fold.plan.data_ptr(), 4),
            _gmem(cutlass.Float32, fold.q_scale.data_ptr(), 4),
            _gmem(cutlass.Float32, fold.kv_scale.data_ptr(), 4),
        )
        query_len = fold.valid_m // num_heads
        dims = (fold.physical_m, fold.partial.shape[0], fold.plan.numel())
        scales = (Float32(fold.sm_scale * LOG2_E), Float32(fold.output_scale))
    else:
        # Never dereferenced without ``fold``; any valid pointers keep one signature.
        ptrs = _split_ptrs(stats.data_ptr(), False)
        query_len, dims = 1, (1, 1, 1)
        scales = (Float32(1.0), Float32(1.0))
    mx = (0, 0, 0) if mx_out is None else tuple(t.data_ptr() for t in mx_out)
    gate_ld = 0 if mx_out is None else mx_out[0].stride(0)
    args = (
        *ptrs,
        Int64(rows.data_ptr()),
        Int64(0 if stats is None else stats.data_ptr()),
        Int64(w_uv.data_ptr()),
        Int64(zero_kv_mask.data_ptr() if has_mask else 0),
        Int64(out.data_ptr()),
        Int64(ws_addr),
        Int64(state.data_ptr()),
        *(Int64(a) for a in mx),
        Int64(rank_stride),
        Int64(half_bytes),
        *scales,
        Int32(num_tokens),
        Int32(query_len),
        *(Int32(d) for d in dims),
        Int32(gate_ld),
        Int32(cp_rank),
        Int32(tiles_per_cta),
        Int32(n_ctas),
        cuda.CUstream(torch.cuda.current_stream(rows.device).cuda_stream),
    )
    _compiled(cp_size, num_heads, fold is not None, has_mask, use_pdl, mx_out is not None)(*args)
