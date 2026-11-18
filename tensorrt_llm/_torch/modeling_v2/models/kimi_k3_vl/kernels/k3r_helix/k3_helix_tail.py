# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 Helix generation tail in one kernel: split-KV fold, V up-projection, exchange, combine.

Under Helix every CP rank decodes all ``H`` MLA heads over its ``1 / P`` slice of the KV cache, and
each rank then needs, for its ``H / P`` heads, the partial outputs and softmax statistics of all
``P`` ranks. With the V up-projection before the exchange, one
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
   statistics. With ``ull2`` (``features.ull2_mla_tail``, and always for the normalized FP16
   partials of ``features.fp4_mla_f16_partials``) the requests' first tasks (plan ``begins``) are
   staged in shared memory before ``griddepcontrol.wait``, rows with more tasks are folded in
   rounds of :data:`FOLD_TASKS` by the same arithmetic, and the latent rows are padded by 16 bytes
   so that step 3's operand loads are free of bank conflicts; the bits are the same.
3. V up-projection. ``mma.sync`` BF16 with FP32 accumulation: warp ``w`` computes output columns
   ``8 w .. 8 w + 7`` of the 16 rows over K = 512. K is permuted inside every 32-column block so
   that each thread's A and B fragments of two consecutive K16 steps are one 16-byte shared load;
   the per-column dot products are unchanged, only their FP32 summation order differs from
   cuBLAS'.
4. Push. The BF16 outputs (staged token-major in shared memory) and the statistics go with
   16-byte (8-byte) stores straight into the head owner's Lamport receive buffer, record
   ``(token, this rank)``, in the record layout of ``helix_a2a_fused``. Rows of tokens this rank
   owns no KV for (``zero_kv_mask``) are sent as the no-op contribution (zeros, ``(-inf, 0)``),
   and no word equal to the sentinel ``0xffffffff`` is sent.
5. Receive. CTA ``b`` combines tokens ``b, b + CTAs, ..`` with the receive loop of
   ``helix_a2a_fused`` (passes of 512 columns): poll the token's ``P`` records, combine them
   bitwise like ``helix_post_process_native`` (same ``expf``, reciprocal, butterflies and FMA
   order), store the BF16 ``[T, H / P * 128]`` output (optionally also the MXFP8 o_proj operand)
   and re-arm the words read. A CTA only polls after its own pushes, and the grid is at most one
   CTA per SM, so every sender is scheduled whatever the receivers wait for. Each CTA releases the
   dependent grid (o_proj) between its pushes and its receive loop.

Receive buffers, halves and the launch ticket are those of ``helix_a2a_fused`` (two halves used by
alternate launches; a word that is not the sentinel has arrived; a launch re-arms what it read
before it exits, and the next store into that half comes two launches later from a rank that has
finished the launch in between). Launches that share a workspace must be ordered on one stream on
every rank, in the same sequence on every rank. Every CTA takes its ticket after
``griddepcontrol.wait``, when every earlier launch on the stream has completed, so tickets of
consecutive launches never interleave; the ticket's latency overlaps the first fold.

Numerics: the fold (latent rows and statistics) and the combine are bitwise those of the separate
kernels; the V up-projection rounds the same FP32 dot products to BF16 after another summation
order than the cuBLAS BMM, so the result agrees with the four-kernel path to BF16 rounding.
"""

from __future__ import annotations

import functools
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from ..k3r_fp4_mla.fp4_mla_mufu16 import LOG2_E
from ..k3r_fp4_mla.fp4_mla_packed_reduce import _reduce_packed_row
from ..k3r_gemm.k3_fp8w_lowm_gemm import (
    _bf16x2_inv_scale,
    _gate_mul_bf16x8,
    _quant_bf16x8_to_e4m3,
    _ue8m0_of_amax,
)
from ..k3r_gemm.k3_fp8w_lowm_mxfp8 import amax_bf16x8_words, amax_pair
from .helix_a2a_fused import (
    _NEG_INF_BITS,
    MAX_CP,
    SENTINEL,
    _all_valid2,
    _all_valid4,
    _as_f32,
    _asm,
    _atom_add_u64,
    _bf16x2_fma,
    _corrected_exp,
    _fadd,
    _fmax,
    _fmul,
    _fv,
    _griddepcontrol_launch_dependents,
    _griddepcontrol_wait,
    _iv,
    _ld_nc_u8,
    _ld_nc_v4,
    _ld_vol_v2,
    _ld_vol_v4,
    _lds_b64,
    _lv,
    _pack_bf16x2,
    _prep4,
    _rcp,
    _shfl_bfly,
    _shfl_idx,
    _st_sentinel_v2,
    _st_sentinel_v4,
    _st_v4,
    _sts_b64,
    _sts_v4,
    _unpack,
    record_bytes,
)

__all__ = [
    "LATENT_DIM",
    "ROWS",
    "SENTINEL",
    "V_HEAD_DIM",
    "FoldInputs",
    "can_run",
    "k3_helix_tail",
    "record_bytes",
]

#: Latent (kv_lora_rank) and value (v_head_dim) widths of a head.
LATENT_DIM = 512
V_HEAD_DIM = 128
#: Tokens per tile (the M of one ``mma.sync`` tile).
ROWS = 16
#: Threads per CTA: 16 warps; a warp folds two 256-column row halves and owns one n8 tile.
THREADS = 512
_WARPS = THREADS // 32
#: Decode tasks of a row whose loads are issued in one round; rows with more use
#: ``_reduce_packed_row`` (rounds of this many tasks with ``ull2``).
FOLD_TASKS = 8
#: Bytes of one head's ``W_UV`` (``[V_HEAD_DIM, LATENT_DIM]`` BF16) and of one latent row.
_W_HEAD_BYTES = V_HEAD_DIM * LATENT_DIM * 2
_LAT_ROW_BYTES = LATENT_DIM * 2
#: Row pitch (elements) of the folded latent rows in shared memory: 16 bytes of padding, so the
#: V up-projection's A fragments (rows g and g + 8 of eight lane groups at the same column) fall
#: in eight different 16-byte bank groups instead of one.
_LAT_PITCH = LATENT_DIM + 8
#: Request entries of the task plan (``begins``) staged in shared memory before the grid wait.
_PLAN_SMEM = 1024
#: Row stride of ``W_UV`` in shared memory (64 bytes of padding against bank conflicts).
_W_ROW_BYTES = _LAT_ROW_BYTES + 64
#: Warps whose lane 0 issues an equal share of ``W_UV``'s row copies: the copies of one warp
#: issue one at a time, and four issuing warps land the 128 KB about 1.5 us sooner than one.
_W_ISSUE_WARPS = 4
#: Row stride of the token-major BF16 output staging (128 values + 16 bytes against bank conflicts).
_C_ROW_BYTES = V_HEAD_DIM * 2 + 16
#: Poll rounds after which a debug launch traps instead of waiting for a record.
_DEBUG_POLL_LIMIT = 1 << 24


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Inline PTX helpers (the rest come from helix_a2a_fused)
# ---------------------------------------------------------------------------


@dsl_user_op
def _lds_v4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _lds_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.b32 $0, [$1];", "=r,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _sts_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_b16(addr, value, *, loc=None, ip=None):
    """Store the low 16 bits of ``value`` to shared memory."""
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.u16.u32 h, $1;\nst.shared.b16 [$0], h;\n}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _ld_nc_v2(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.nc.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 2, loc=loc, ip=ip)


@dsl_user_op
def _st_v2(addr, v0, v1, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(v0, loc=loc, ip=ip), _iv(v1, loc=loc, ip=ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_init(addr, count: int, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        f"mbarrier.init.shared::cta.b64 [$0], {count};\nfence.mbarrier_init.release.cluster;",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_expect_tx(addr, nbytes: int, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {nbytes};",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_g2s(dst, src, nbytes: int, mbar, *, loc=None, ip=None):
    """Bulk copy of ``nbytes`` from global ``src`` to shared ``dst``, completing on ``mbar``."""
    _asm(
        None,
        [_iv(dst, loc=loc, ip=ip), _lv(src, loc=loc, ip=ip), _iv(mbar, loc=loc, ip=ip)],
        f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], {nbytes}, [$2];",
        "r,l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_try_wait(addr, phase, *, loc=None, ip=None):
    """1 once phase ``phase`` of the shared mbarrier at ``addr`` has completed, else 0."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip), _iv(phase, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _opaque_f32(value, *, loc=None, ip=None):
    """``value`` through an opaque move: the compiler cannot merge expressions on it with
    expressions on ``value``."""
    return Float32(
        _asm(T.f32(), [_fv(value, loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,f", loc=loc, ip=ip)
    )


@dsl_user_op
def _mma_bf16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """``d[16x8] = a[16x16] @ b[16x8] + c`` (BF16 inputs, FP32 accumulation)."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_iv(x, loc=loc, ip=ip) for x in (a0, a1, a2, a3, b0, b1)]
        + [_fv(x, loc=loc, ip=ip) for x in (c0, c1, c2, c3)],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _shfl_bfly_b32(value, offset: int, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {offset}, 31, -1;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_u8(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.u8 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _globaltimer(*, loc=None, ip=None):
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc=loc, ip=ip))


@dsl_user_op
def _st_u64(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _lv(value, loc=loc, ip=ip)],
        "st.global.u64 [$0], $1;",
        "l,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _trap(*, loc=None, ip=None):
    _asm(None, [], "trap;", "", loc=loc, ip=ip)


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


@cute.jit
def _fold_round_loads(
    mPartial: cute.Tensor,
    mRowMax: cute.Tensor,
    mRowSum: cute.Tensor,
    lane: Int32,
    local_row: Int32,
    half: Int32,
    task_begin: Int32,
    request_tasks: Int32,
    physical_m: Int32,
):
    """The row sums, row maxima and this lane's partial vectors (two groups of 4 columns of the
    256-column ``half``) of the first :data:`FOLD_TASKS` tasks of a row; slots past
    ``request_tasks`` re-read the last task (zero weight later, as the reduction)."""
    columns_per_lane: cutlass.Constexpr = 4
    partial_bits: cutlass.Constexpr = mPartial.element_type.width
    band_base = half * Int32(256) + lane * Int32(columns_per_lane)
    gPartial_arr = cutlass.make_array_view(mPartial)
    sums = []
    maxima = []
    vecs = []
    for k in cutlass.range_constexpr(FOLD_TASKS):
        task = task_begin + cutlass.min(Int32(k), request_tasks - Int32(1))
        sums.append(mRowSum[task, local_row])
        maxima.append(mRowMax[task, local_row])
        partial_row = (task * physical_m + local_row) * Int32(LATENT_DIM) + band_base
        for group in cutlass.range_constexpr(2):
            vecs.append(
                gPartial_arr.subview(partial_row + Int32(group * 128))
                .data_ptr()
                .load(count=columns_per_lane, alignment=partial_bits // 8 * columns_per_lane)
            )
    return sums, maxima, vecs


@cute.jit
def _fold_round_sums(loaded, request_tasks: Int32, scale: Float32, normalized: cutlass.Constexpr):
    """The reduction's global maximum, denominator and numerators of a row with at most
    :data:`FOLD_TASKS` tasks from :func:`_fold_round_loads`' values, in its task order."""
    sums, maxima, vecs = loaded
    columns_per_lane: cutlass.Constexpr = 4
    global_row_max = cutlass.Float32(-cutlass.Float32.inf)
    numerator0 = cutlass.Vector.from_elements(
        tuple(cutlass.Float32(0.0) for _ in range(columns_per_lane)), dtype=cutlass.Float32
    )
    numerator1 = cutlass.Vector.from_elements(
        tuple(cutlass.Float32(0.0) for _ in range(columns_per_lane)), dtype=cutlass.Float32
    )
    denominator = cutlass.Float32(0.0)
    for k in cutlass.range_constexpr(FOLD_TASKS):
        if Int32(k) < request_tasks:
            if sums[k] != cutlass.Float32(0.0):
                global_row_max = cute.arch.fmax(global_row_max, maxima[k] * scale)
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
                if cutlass.const_expr(normalized):
                    task_weight = task_weight * task_row_sum
        # Empty tasks keep zero weight; their payload is never trusted.
        if task_weight != cutlass.Float32(0.0):
            v0 = vecs[2 * k]
            v1 = vecs[2 * k + 1]
            if cutlass.const_expr(normalized):
                v0 = v0.to(cutlass.Float32)
                v1 = v1.to(cutlass.Float32)
            numerator0 = numerator0 + v0 * task_weight
            numerator1 = numerator1 + v1 * task_weight
    return global_row_max, denominator, numerator0, numerator1


@cute.jit
def _fold_store(
    sLat: cute.Tensor,
    sMax: cute.Tensor,
    sSum: cute.Tensor,
    lane: Int32,
    row: Int32,
    half: Int32,
    global_row_max: Float32,
    denominator: Float32,
    numerator0,
    numerator1,
    output_scale: Float32,
    pitch: cutlass.Constexpr,
) -> None:
    """The folded BF16 row half and (for half 0) its Helix statistics into shared memory."""
    columns_per_lane: cutlass.Constexpr = 4
    gOut_arr = cutlass.make_array_view(sLat)
    normalizer = cutlass.Float32(0.0)
    if denominator != cutlass.Float32(0.0):
        normalizer = output_scale / denominator
    out_row = row * Int32(pitch) + half * Int32(256) + lane * Int32(columns_per_lane)
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


@cute.jit
def _fold_half_ull2(
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
    pitch: cutlass.Constexpr = _LAT_PITCH,
) -> None:
    """``ull2``: ``_reduce_packed_row`` of one 256-column half (``band = half`` of 256 columns)
    of a row, into row ``row`` (rows ``pitch`` elements apart) of the shared-memory latent tile,
    for any task count. With at most :data:`FOLD_TASKS` tasks the statistics and partials of
    every task are loaded before any is consumed (the reduction issues them in three rounds);
    with more, the row maxima are scanned first and the tasks are folded :data:`FOLD_TASKS` at a
    time. The weights, the
    denominator and the numerators are accumulated in the reduction's task order with the same
    operations, so the BF16 row and its statistics are bitwise the reduction's. FP16 partials
    (``features.fp4_mla_f16_partials``) are normalized split outputs, weighted by the max-scaled
    row sums, as in the reduction."""
    band_columns: cutlass.Constexpr = 256
    columns_per_lane: cutlass.Constexpr = 4
    partial_bits: cutlass.Constexpr = mPartial.element_type.width
    normalized: cutlass.Constexpr = partial_bits == 16
    band_base = half * Int32(band_columns) + lane * Int32(columns_per_lane)
    gPartial_arr = cutlass.make_array_view(mPartial)
    numerator0 = cutlass.Vector.from_elements(
        tuple(cutlass.Float32(0.0) for _ in range(columns_per_lane)), dtype=cutlass.Float32
    )
    numerator1 = cutlass.Vector.from_elements(
        tuple(cutlass.Float32(0.0) for _ in range(columns_per_lane)), dtype=cutlass.Float32
    )
    denominator = cutlass.Float32(0.0)
    global_row_max = cutlass.Float32(-cutlass.Float32.inf)
    if request_tasks <= Int32(FOLD_TASKS):
        loaded = _fold_round_loads(
            mPartial, mRowMax, mRowSum, lane, local_row, half, task_begin, request_tasks,
            physical_m,
        )  # fmt: skip
        global_row_max, denominator, numerator0, numerator1 = _fold_round_sums(
            loaded, request_tasks, scale, normalized
        )
    else:
        for task_idx in cutlass.range(0, request_tasks, 1, unroll=1):
            task = task_begin + task_idx
            if mRowSum[task, local_row] != cutlass.Float32(0.0):
                global_row_max = cute.arch.fmax(global_row_max, mRowMax[task, local_row] * scale)
        for chunk_base in cutlass.range(0, request_tasks, FOLD_TASKS, unroll=1):
            sums = []
            maxima = []
            vecs = []
            for k in cutlass.range_constexpr(FOLD_TASKS):
                task = task_begin + cutlass.min(chunk_base + Int32(k), request_tasks - Int32(1))
                sums.append(mRowSum[task, local_row])
                maxima.append(mRowMax[task, local_row])
                partial_row = (task * physical_m + local_row) * Int32(LATENT_DIM) + band_base
                for group in cutlass.range_constexpr(2):
                    vec = (
                        gPartial_arr.subview(partial_row + Int32(group * 128))
                        .data_ptr()
                        .load(
                            count=columns_per_lane,
                            alignment=partial_bits // 8 * columns_per_lane,
                        )
                    )
                    if cutlass.const_expr(normalized):
                        vec = vec.to(cutlass.Float32)
                    vecs.append(vec)
            for k in cutlass.range_constexpr(FOLD_TASKS):
                task_weight = cutlass.Float32(0.0)
                if chunk_base + Int32(k) < request_tasks:
                    task_row_sum = sums[k]
                    if task_row_sum != cutlass.Float32(0.0):
                        task_max = _opaque_f32(maxima[k])
                        task_weight = cute.exp2(task_max * scale - global_row_max, fastmath=True)
                        denominator += task_row_sum * task_weight
                        if cutlass.const_expr(normalized):
                            task_weight = task_weight * task_row_sum
                if task_weight != cutlass.Float32(0.0):
                    numerator0 = numerator0 + vecs[2 * k] * task_weight
                    numerator1 = numerator1 + vecs[2 * k + 1] * task_weight
    _fold_store(sLat, sMax, sSum, lane, row, half, global_row_max, denominator, numerator0,
                numerator1, output_scale, pitch)  # fmt: skip


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
            UE8M0 exponent per 1x32 group) of ``bf16(out * gate)``, bitwise the gated low-M
            GEMM's in-kernel quantization of it.
        debug: Also store the folded latent rows, their statistics and the V up-projection
            outputs, and trap instead of polling forever (tests).
        timing: Thread 0 of every CTA stores ``%globaltimer`` at its phase boundaries into
            ``ts[cta * 8 + phase]`` (microbenchmarks).
        loopback: This rank's values of the heads of rank ``d`` go into its own receive buffer
            as the record of source ``d``: every peer "sends" this rank's partials of its own heads
            (the stand-in of single-GPU model tests).
        ull2: (``features.ull2_mla_tail``; always with FP16 partials) the requests' first tasks
            are staged in shared memory before the grid wait, rows with more than
            :data:`FOLD_TASKS` tasks are folded in rounds of :data:`FOLD_TASKS` by the same code,
            and the latent rows are padded (conflict-free V up-projection operands). Same bits.
    """

    def __init__(
        self,
        cp_size: int,
        num_heads: int,
        fold: bool,
        has_mask: bool,
        use_pdl: bool,
        mx_out: bool = False,
        debug: bool = False,
        timing: bool = False,
        loopback: bool = False,
        mx_atom: bool = False,
        ull2: bool = False,
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
        # The MXFP8 scales in the mid-M FP8 GEMM's layout (one 128-row block of tcgen05 atoms)
        # instead of row-major (see ``mx_atom`` of k3_helix_tail).
        self.mx_atom = mx_atom and mx_out
        self.debug = debug
        self.timing = timing
        self.loopback = loopback
        self.ull2 = ull2
        self.lat_pitch = _LAT_PITCH if ull2 else LATENT_DIM
        self.plan_smem = _PLAN_SMEM if (ull2 and fold) else 1
        # Transposed V up-projection (ull2): K-half partial sums of 8 m16 tiles x 2 n8 tiles.
        self.vsum_floats = (_WARPS // 2) * 2 * 32 * 4 if ull2 else 4
        self.threads = THREADS
        # 16-byte vectors of a (token, source) record: the BF16 values of the H / P heads, then
        # their FP32 (max, sum) pairs (two heads per vector); the layout of helix_a2a_fused.
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
        dbg_lat_addr: Int64,
        dbg_v_addr: Int64,
        dbg_stats_addr: Int64,
        ts_addr: Int64,
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

        # ull2: the staged plan, the padded latent rows and the second K half's V-up sums.
        @cute.struct
        class Ull2SharedStorage:
            half: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 2], 16]
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 2], 16]
            smax: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, ROWS], 16]
            ssum: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, ROWS], 16]
            plan: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.plan_smem], 16]
            lat: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, ROWS * self.lat_pitch], 128
            ]
            vsum: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.vsum_floats], 16]
            cst: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, ROWS * _C_ROW_BYTES], 128]
            wv: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, V_HEAD_DIM * _W_ROW_BYTES], 128
            ]

        self.shared_storage = SharedStorage
        if cutlass.const_expr(self.ull2):
            self.shared_storage = Ull2SharedStorage
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
            dbg_lat_addr,
            dbg_v_addr,
            dbg_stats_addr,
            ts_addr,
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
        dbg_lat_addr: Int64,
        dbg_v_addr: Int64,
        dbg_stats_addr: Int64,
        ts_addr: Int64,
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
        ts = ts_addr + Int64(bidx) * 64
        self.stamp(tidx, ts, 0)
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        half_slot = storage.half.data_ptr().toint()
        mbar = storage.mbar.data_ptr().toint()
        lat_base = storage.lat.data_ptr().toint()
        smax_base = storage.smax.data_ptr().toint()
        ssum_base = storage.ssum.data_ptr().toint()
        wv_base = storage.wv.data_ptr().toint()
        cst_base = storage.cst.data_ptr().toint()
        # ull2-only shared memory; on the other path these names stand for values never read.
        vsum_base = cst_base
        if cutlass.const_expr(self.ull2):
            vsum_base = storage.vsum.data_ptr().toint()
        sLat = storage.lat.get_tensor(
            cute.make_layout((ROWS, LATENT_DIM), stride=(self.lat_pitch, 1))
        )
        sPlan = mPlan
        if cutlass.const_expr(self.ull2):
            sPlan = storage.plan.get_tensor(cute.make_layout((self.plan_smem,), stride=(1,)))
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
        if cutlass.const_expr(self.ull2):
            scale = Float32(0.0)
            if cutlass.const_expr(self.fold):
                # The packed reduction's scale expression (same operations, same result); the
                # global scales are fixed tensors, read before the wait.
                scale = softmax_scale_log2 / (mQScale[0] * mKvScale[0])
                # The requests' first tasks (plan ``begins``): written by the plan kernel, which
                # completed before the decode this grid depends on started, so they are read
                # before the wait.
                num_requests = (num_tokens + query_len - Int32(1)) // query_len
                for r in cutlass.range(tidx, num_requests + Int32(1), THREADS, unroll=1):
                    if r < Int32(_PLAN_SMEM):
                        sPlan[r] = mPlan[Int32(2) + r]
                cute.arch.barrier()
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        # The ticket is taken after the wait (every earlier launch on the stream has completed);
        # its result is first needed after the first fold.
        ticket = Int64(0)
        if tidx == 0:
            ticket = _atom_add_u64(state_addr, Int64(1))
        self.stamp(tidx, ts, 1)
        if cutlass.const_expr(not self.ull2):
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
                    mPartial, mRowMax, mRowSum, mPlan, sPlan, rows_addr, stats_addr, scale,
                    output_scale, num_tokens, query_len, physical_m,
                )  # fmt: skip
                if cutlass.const_expr(not self.ull2):
                    if tt == 0:
                        if tidx == 0:
                            if (ticket & Int64(0xFFFFFFFF)) == Int64(n_ctas) - Int64(1):
                                # Last arrival: clear the arrivals and advance the launch number.
                                _atom_add_u64(state_addr, Int64(1 << 32) - Int64(n_ctas))
                            _sts_b64(half_slot, ((ticket >> 32) & Int64(1)) * half_bytes)
                cute.arch.barrier()
                self.stamp(tidx, ts, 2)
                if cutlass.const_expr(self.debug):
                    self.dump_rows(tidx, head, token0, num_tokens, lat_base, smax_base,
                                   ssum_base, dbg_lat_addr, dbg_stats_addr)  # fmt: skip
                if tt == 0:
                    landed = Int32(0)
                    while landed == 0:
                        landed = _mbar_try_wait(mbar, Int32(0))
                if cutlass.const_expr(self.ull2):
                    first_tile = Int32(0)
                    if tt == 0:
                        first_tile = Int32(1)
                    # The receive-buffer half is read inside, after the V up-projection
                    # (``rank_stride`` stands for the unused ``half_off``).
                    self.project_and_push(
                        tidx, warp, lane, head, token0, lat_base, smax_base, ssum_base, wv_base,
                        cst_base, vsum_base, zero_kv, ws_addr, dbg_v_addr, rank_stride,
                        rank_stride, half_slot, num_tokens, rank, ts, ticket, state_addr, n_ctas,
                        half_bytes, first_tile,
                    )  # fmt: skip
                else:
                    # (``zero_kv`` stands for the unused ``first_tile``.)
                    self.project_and_push(
                        tidx, warp, lane, head, token0, lat_base, smax_base, ssum_base, wv_base,
                        cst_base, vsum_base, zero_kv, ws_addr, dbg_v_addr, rank_stride,
                        _lds_b64(half_slot), half_slot, num_tokens, rank, ts, ticket, state_addr,
                        n_ctas, half_bytes, zero_kv,
                    )  # fmt: skip
        # The dependent grid (o_proj) is released once this CTA's pushes are issued: its CTAs take
        # the SMs of exiting senders and stream their weights while the receivers wait for peers.
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_launch_dependents()
        self.stamp(tidx, ts, 4)
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
            ts,
        )
        self.stamp(tidx, ts, 7)

    @cute.jit
    def take_half(
        self,
        tidx: Int32,
        ticket: Int64,
        state_addr: Int64,
        n_ctas: Int32,
        half_slot: Int32,
        half_bytes: Int64,
    ):
        """Thread 0: the receive-buffer half of this launch from its ticket into shared memory
        (read by every thread after the next CTA barrier); the last arrival advances the launch
        number."""
        if tidx == 0:
            if (ticket & Int64(0xFFFFFFFF)) == Int64(n_ctas) - Int64(1):
                # Last arrival: clear the arrivals and advance the launch number.
                _atom_add_u64(state_addr, Int64(1 << 32) - Int64(n_ctas))
            _sts_b64(half_slot, ((ticket >> 32) & Int64(1)) * half_bytes)

    @cute.jit
    def stamp(self, tidx: Int32, ts: Int64, phase: int):
        """``timing``: thread 0 stores ``%globaltimer`` as phase ``phase`` of this CTA."""
        if cutlass.const_expr(self.timing):
            if tidx == 0:
                _st_u64(ts + Int64(phase * 8), _globaltimer())

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
        sPlan: cute.Tensor,
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
            lat_vec = lat_base + i * Int32(self.lat_pitch * 2) + half * Int32(_LAT_ROW_BYTES // 2)
            lat_vec = lat_vec + lane * Int32(16)
            if t < num_tokens:
                row = Int64(t) * H + Int64(head)
                folded = Int32(0)
                if cutlass.const_expr(self.fold):
                    request = t // query_len
                    local_row = (t - request * query_len) * Int32(H) + head
                    if cutlass.const_expr(self.ull2):
                        task_begin = Int32(0)
                        task_end = Int32(0)
                        if request < Int32(_PLAN_SMEM - 1):
                            task_begin = sPlan[request]
                            task_end = sPlan[request + Int32(1)]
                        else:
                            task_begin = mPlan[Int32(2) + request]
                            task_end = mPlan[Int32(3) + request]
                        request_tasks = task_end - task_begin
                    else:
                        task_begin = mPlan[Int32(2) + request]
                        request_tasks = mPlan[Int32(3) + request] - task_begin
                    if request_tasks > Int32(1):
                        folded = Int32(1)
                        if cutlass.const_expr(self.ull2):
                            _fold_half_ull2(
                                mPartial, mRowMax, mRowSum, sLat, sMax, sSum, lane, i, local_row,
                                half, task_begin, request_tasks, physical_m, scale, output_scale,
                                _LAT_PITCH,
                            )  # fmt: skip
                        else:
                            if request_tasks <= Int32(FOLD_TASKS):
                                _fold_half(
                                    mPartial, mRowMax, mRowSum, sLat, sMax, sSum, lane, i,
                                    local_row, half, task_begin, request_tasks, physical_m, scale,
                                    output_scale,
                                )  # fmt: skip
                            else:
                                _reduce_packed_row(
                                    mPartial, mRowMax, mRowSum, sLat, sMax, sSum, lane, i,
                                    local_row, half, task_begin, request_tasks, physical_m, scale,
                                    output_scale, 256, True,
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
        vsum_base: Int32,
        zero_kv: Int32,
        ws_addr: Int64,
        dbg_v_addr: Int64,
        rank_stride: Int64,
        half_off: Int64,
        half_slot: Int32,
        num_tokens: Int32,
        rank: Int32,
        ts: Int64,
        ticket: Int64,
        state_addr: Int64,
        n_ctas: Int32,
        half_bytes: Int64,
        first_tile: Int32,
    ):
        """V up-projection of the tile (warp ``w``: output columns ``8 w .. 8 w + 7``) and the
        push of its rows and statistics into the head owner's receive buffer (``zero_kv``: the
        pushed token owns no KV on this rank)."""
        H = self.H
        g = lane // 4
        q = lane % 4
        if cutlass.const_expr(self.ull2):
            self.project_transposed(warp, lane, token0, num_tokens, lat_base, wv_base, cst_base,
                                    vsum_base)  # fmt: skip
        else:
            # Two accumulators (the two k16 steps of every 32-column block) halve the MMA
            # dependency chain; they are added at the end.
            accs = [[Float32(0.0)] * 4 for _ in range(2)]
            w_row = wv_base + (warp * 8 + g) * Int32(_W_ROW_BYTES)
            for u in cutlass.range_constexpr(LATENT_DIM // 32):
                col2 = (Int32(u * 32) + q * 8) * 2
                b = _lds_v4(w_row + col2)
                a_lo = _lds_v4(lat_base + g * Int32(self.lat_pitch * 2) + col2)
                a_hi = _lds_v4(lat_base + (g + 8) * Int32(self.lat_pitch * 2) + col2)
                for v in cutlass.range_constexpr(2):
                    accs[v] = list(
                        _mma_bf16(
                            a_lo[2 * v],
                            a_hi[2 * v],
                            a_lo[2 * v + 1],
                            a_hi[2 * v + 1],
                            b[2 * v],
                            b[2 * v + 1],
                            *accs[v],
                        )
                    )
            acc = [accs[0][c] + accs[1][c] for c in range(4)]
            # The BF16 outputs are staged token-major in their own buffer, so the next tile's
            # fold (after the barrier below) cannot overwrite what this tile's pushes still read;
            # the statistics this thread pushes are read before that barrier for the same reason.
            n = warp * 8 + q * 2
            _sts_b32(cst_base + g * Int32(_C_ROW_BYTES) + n * 2, _pack_bf16x2(acc[0], acc[1]))
            _sts_b32(cst_base + (g + 8) * Int32(_C_ROW_BYTES) + n * 2, _pack_bf16x2(acc[2], acc[3]))
        if cutlass.const_expr(self.ull2):
            # The launch ticket is first needed by the pushes below: its atomic's latency
            # overlaps the fold and the V up-projection.
            if first_tile != Int32(0):
                self.take_half(tidx, ticket, state_addr, n_ctas, half_slot, half_bytes)
        i = tidx // 16
        vec = tidx % 16
        mx = _lds_b32(smax_base + (i % Int32(ROWS)) * 4)
        sm = _lds_b32(ssum_base + (i % Int32(ROWS)) * 4)
        cute.arch.barrier()
        self.stamp(tidx, ts, 3)
        if cutlass.const_expr(self.ull2):
            # This launch's receive-buffer half (``take_half`` above, before the barrier).
            half_off = _lds_b64(half_slot)
        # Row i's 16 vectors (and its statistics) into the head owner's record.
        if tidx < Int32(ROWS * 16):
            t = token0 + i
            if t < num_tokens:
                words = _lds_v4(cst_base + i * Int32(_C_ROW_BYTES) + vec * 16)
                if cutlass.const_expr(self.debug):
                    row = Int64(t) * H + Int64(head)
                    _st_v4(dbg_v_addr + row * (V_HEAD_DIM * 2) + Int64(vec) * 16, *words)
                dst_rank = head // Int32(self.H_LOC)
                local_head = head - dst_rank * Int32(self.H_LOC)
                sent = _prep4(words[0], words[1], words[2], words[3], zero_kv, Int32(0))
                st = _prep4(mx, sm, Int32(0), Int32(0), zero_kv, Int32(1))
                # Record (t, this rank) of the head owner's buffer; with loopback, record
                # (t, owner) of this rank's own buffer.
                box_rank = dst_rank
                src_rank = rank
                if cutlass.const_expr(self.loopback):
                    box_rank = rank
                    src_rank = dst_rank
                record = (
                    ws_addr
                    + Int64(box_rank) * rank_stride
                    + half_off
                    + Int64(t * self.P + src_rank) * (self.RV * 16)
                )
                _st_v4(record + Int64(local_head) * (V_HEAD_DIM * 2) + Int64(vec) * 16, *sent)
                if vec == 0:
                    _st_v2(record + Int64(self.DV * 16) + Int64(local_head) * 8, st[0], st[1])

    @cute.jit
    def vup_k_half(self, w_lo: Int32, w_hi: Int32, lat_row: Int32, k_col: Int32):
        """One warp's m16 x n8 tile of ``W_UV @ lat^T`` over 256 K columns starting at
        ``k_col`` (this lane's 8-column offset included): two accumulators, added at the end."""
        accs = [[Float32(0.0)] * 4 for _ in range(2)]
        for u in cutlass.range_constexpr(LATENT_DIM // 64):
            col2 = (k_col + Int32(u * 32)) * 2
            a_lo = _lds_v4(w_lo + col2)
            a_hi = _lds_v4(w_hi + col2)
            b = _lds_v4(lat_row + col2)
            for v in cutlass.range_constexpr(2):
                accs[v] = list(
                    _mma_bf16(
                        a_lo[2 * v],
                        a_hi[2 * v],
                        a_lo[2 * v + 1],
                        a_hi[2 * v + 1],
                        b[2 * v],
                        b[2 * v + 1],
                        *accs[v],
                    )
                )
        return [accs[0][c] + accs[1][c] for c in range(4)]

    @cute.jit
    def project_transposed(
        self,
        warp: Int32,
        lane: Int32,
        token0: Int32,
        num_tokens: Int32,
        lat_base: Int32,
        wv_base: Int32,
        cst_base: Int32,
        vsum_base: Int32,
    ):
        """``ull2`` V up-projection as ``out^T = W_UV @ lat^T``: warp ``w`` runs the m16 tile of
        output columns ``16 (w % 8) ..`` over K half ``w // 8``, for the tile's first 8 tokens and
        (only if the tile has more) its last 8, so a tile of at most 8 tokens issues half the
        MMAs of the token-major form. The second K half's sums go through shared memory and are
        added to the first's; the BF16 results are staged token-major like the token-major form's
        (the same dot products, another FP32 summation order)."""
        g = lane // 4
        q = lane % 4
        m_tile = warp % Int32(_WARPS // 2)
        k_half = warp // Int32(_WARPS // 2)
        pitch_bytes: cutlass.Constexpr = self.lat_pitch * 2
        w_lo = wv_base + (m_tile * 16 + g) * Int32(_W_ROW_BYTES)
        w_hi = w_lo + Int32(8 * _W_ROW_BYTES)
        lat_row = lat_base + g * Int32(pitch_bytes)
        k_col = k_half * Int32(LATENT_DIM // 2) + q * 8
        s0 = self.vup_k_half(w_lo, w_hi, lat_row, k_col)
        # The tile's last 8 tokens only if it has any.
        c0 = Float32(0.0)
        c1 = Float32(0.0)
        c2 = Float32(0.0)
        c3 = Float32(0.0)
        if token0 + Int32(8) < num_tokens:
            r = self.vup_k_half(w_lo, w_hi, lat_row + Int32(8 * pitch_bytes), k_col)
            c0 = r[0]
            c1 = r[1]
            c2 = r[2]
            c3 = r[3]
        s1 = [c0, c1, c2, c3]
        sums = [s0, s1]
        # Second K half: (m tile, n tile, lane, 4) partial sums into shared memory.
        if k_half == Int32(1):
            for n_tile in cutlass.range_constexpr(2):
                slot = vsum_base + (((m_tile * 2 + Int32(n_tile)) * 32 + lane) * 4) * 4
                for c in cutlass.range_constexpr(4):
                    _sts_f32(slot + Int32(c * 4), sums[n_tile][c])
        cute.arch.barrier(barrier_id=2, number_of_threads=THREADS)
        if k_half == Int32(0):
            for n_tile in cutlass.range_constexpr(2):
                slot = vsum_base + (((m_tile * 2 + Int32(n_tile)) * 32 + lane) * 4) * 4
                d = [sums[n_tile][c] + _lds_f32(slot + Int32(c * 4)) for c in range(4)]
                # d0, d1: output column 16 m + g, tokens 2 q, 2 q + 1; d2, d3: column + 8.
                lo = _pack_bf16x2(d[0], d[2])
                hi = _pack_bf16x2(d[1], d[3])
                col = (m_tile * 16 + g) * 2
                t0 = Int32(n_tile * 8) + q * 2
                row0 = cst_base + t0 * Int32(_C_ROW_BYTES) + col
                row1 = row0 + Int32(_C_ROW_BYTES)
                _sts_b16(row0, lo)
                _sts_b16(row0 + Int32(16), lo >> 16)
                _sts_b16(row1, hi)
                _sts_b16(row1 + Int32(16), hi >> 16)

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
        ts: Int64,
    ):
        """Combine the records of this CTA's tokens: ``HelixA2AFusedKernel.receive`` (the same
        instruction sequences, so bitwise ``helix_post_process_native``) over ``passes`` groups
        of THREADS columns. Thread ``c`` of a pass owns one 16-byte column (8 output values) of one
        head; lane ``l`` of a 16-lane head group polls the statistics of source rank ``l``."""
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
                    spins = Int32(0)
                    while ready == 0:
                        if cutlass.const_expr(self.debug):
                            # Tests: a record that never arrives fails the launch, not the job.
                            spins = spins + 1
                            if spins > Int32(_DEBUG_POLL_LIMIT):
                                _trap()
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
                if cutlass.const_expr(self.timing and p == 0):
                    # Thread 0 has seen every source of its token's first column.
                    if tidx == 0:
                        _st_u64(ts + Int64(16 + 8 * 3), _globaltimer())

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
        UE8M0 exponent and E4M3 codes of the gated low-M GEMM's quantization warps. Every lane of
        the warp runs the shuffles; groups are either all valid or all invalid (``DV`` is a
        multiple of 16)."""
        K = self.DV * 8
        xg = _gate_mul_bf16x8(o[0], o[1], o[2], o[3], g[0], g[1], g[2], g[3])
        amax = amax_bf16x8_words(xg[0], xg[1], xg[2], xg[3])
        amax = amax_pair(amax, _shfl_bfly_b32(amax, 1))
        amax = amax_pair(amax, _shfl_bfly_b32(amax, 2))
        e8 = _ue8m0_of_amax(amax)
        q0, q1 = _quant_bf16x8_to_e4m3(xg[0], xg[1], xg[2], xg[3], _bf16x2_inv_scale(e8))
        if valid != 0:
            _st_v2(xq_addr + Int64(t) * K + Int64(c * 8), q0, q1)
            if (c % 4) == 0:
                if cutlass.const_expr(self.mx_atom):
                    # Mid-M scale atom of group g = c / 4 for token t < 128: byte (g / 4) * 512
                    # + (t % 32) * 16 + (t / 32) * 4 + g % 4.
                    grp = c // 4
                    _st_u8(
                        xs_addr
                        + Int64(grp // 4) * 512
                        + Int64(t & 31) * 16
                        + Int64((t >> 5) & 3) * 4
                        + Int64(grp % 4),
                        e8,
                    )
                else:
                    _st_u8(xs_addr + Int64(t) * (K // 32) + Int64(c // 4), e8)

    @cute.jit
    def dump_rows(
        self,
        tidx: Int32,
        head: Int32,
        token0: Int32,
        num_tokens: Int32,
        lat_base: Int32,
        smax_base: Int32,
        ssum_base: Int32,
        dbg_lat_addr: Int64,
        dbg_stats_addr: Int64,
    ):
        """Tests: store the tile's latent rows ``[T * H, 512]`` and statistics ``[T * H, 2]``."""
        H = self.H
        for k in cutlass.range_constexpr(ROWS * _LAT_ROW_BYTES // 16 // THREADS):
            idx = tidx + Int32(k * THREADS)
            i = idx // Int32(_LAT_ROW_BYTES // 16)
            vec = idx % Int32(_LAT_ROW_BYTES // 16)
            t = token0 + i
            if t < num_tokens:
                words = _lds_v4(lat_base + i * Int32(self.lat_pitch * 2) + vec * 16)
                row = Int64(t) * H + Int64(head)
                _st_v4(dbg_lat_addr + row * _LAT_ROW_BYTES + Int64(vec) * 16, *words)
        if tidx < Int32(ROWS):
            t = token0 + tidx
            if t < num_tokens:
                mx = _lds_b32(smax_base + tidx * 4)
                sm = _lds_b32(ssum_base + tidx * 4)
                row = Int64(t) * H + Int64(head)
                _st_v2(dbg_stats_addr + row * 8, mx, sm)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


class FoldInputs:
    """The packed decode's split state that the tail folds instead of the reduction kernel
    (the arguments of ``reduce_packed_splits``).

    Args:
        partial: FP32 ``[tasks, physical_m, 512]`` split partials, or the FP16 normalized split
            outputs of ``features.fp4_mla_f16_partials`` (``output_scale`` 1).
        maxima: FP32 ``[tasks, physical_m]`` raw row maxima.
        sums: FP32 ``[tasks, physical_m]`` row sums relative to them.
        plan: int32 task plan (``plan[2 + r]`` = first task of request ``r``).
        q_scale: FP32 scalar device tensor, the FP4 Q global scale.
        kv_scale: FP32 scalar device tensor, the FP4 KV global scale.
        num_requests: Requests of the packed decode.
        valid_m: ``query_len * H`` rows per request.
        physical_m: ``valid_m`` padded to 128.
        sm_scale: Softmax scale.
        output_scale: Scale of the normalized output.
    """

    def __init__(
        self,
        partial: torch.Tensor,
        maxima: torch.Tensor,
        sums: torch.Tensor,
        plan: torch.Tensor,
        q_scale: torch.Tensor,
        kv_scale: torch.Tensor,
        *,
        num_requests: int,
        valid_m: int,
        physical_m: int,
        sm_scale: float,
        output_scale: float,
    ):
        self.partial = partial
        self.maxima = maxima
        self.sums = sums
        self.plan = plan
        self.q_scale = q_scale
        self.kv_scale = kv_scale
        self.num_requests = num_requests
        self.valid_m = valid_m
        self.physical_m = physical_m
        self.sm_scale = sm_scale
        self.output_scale = output_scale


def can_run(cp_size: int, num_heads: int, latent_dim: int, v_head_dim: int) -> bool:
    """Whether :func:`k3_helix_tail` supports this Helix MLA geometry."""
    if latent_dim != LATENT_DIM or v_head_dim != V_HEAD_DIM:
        return False
    if not (1 <= cp_size <= 8 or cp_size == MAX_CP):
        return False
    return num_heads % (2 * cp_size) == 0


@functools.cache
def _kernel(
    cp_size: int,
    num_heads: int,
    fold: bool,
    has_mask: bool,
    use_pdl: bool,
    mx_out: bool,
    debug: bool,
    timing: bool,
    loopback: bool,
    mx_atom: bool = False,
    ull2: bool = False,
):
    return K3HelixTailKernel(
        cp_size, num_heads, fold, has_mask, use_pdl, mx_out, debug, timing, loopback, mx_atom, ull2
    )


_COMPILED: dict = {}


def _gmem(dtype, tensor: torch.Tensor, align: int):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def k3_helix_tail(
    rows: torch.Tensor,
    stats: torch.Tensor | None,
    w_uv: torch.Tensor,
    out: torch.Tensor,
    *,
    fold: FoldInputs | None,
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
    debug: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    timestamps: torch.Tensor | None = None,
    loopback: bool = False,
    mx_atom: bool = False,
    ull2: bool = False,
) -> None:
    """Launch the Helix tail of ``T`` tokens into ``out``.

    Args:
        rows: BF16 ``[T * H, 512]`` latent rows of the decode (row ``t * H + h``). With ``fold``,
            only the rows of single-task requests are read (the decode stored them).
        stats: FP32 ``[T * H, 2]`` ``(max, sum)`` of ``rows`` (natural-log max); read only
            without ``fold``.
        w_uv: BF16 ``[H, 128, 512]`` V up-projection of every head.
        out: BF16 ``[T, H / P * 128]`` combined output of this rank's heads.
        fold: The packed decode's split state to fold, or None.
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
            ``[T, H / P * 128]`` and UE8M0 (uint8) ``xs`` ``[T, H / P * 4]``.
        debug: Optional ``(latent [T * H, 512] BF16, values [T * H, 128] BF16, stats [T * H, 2]
            FP32)`` outputs of the sender stages (tests).
        timestamps: Optional int64 ``[CTAs, 8]`` (``CTAs`` at most ``max_ctas``):
            ``%globaltimer`` at the phase boundaries of every CTA (0 start, 1 after the wait, 2
            fold, 3 V up-projection, 4 pushes done, 5 first column combined, 7 end; 2 and 3 of
            the CTA's last tile).
        loopback: Send to this rank's own buffer, as source ``d``, the values of rank ``d``'s heads
            (single-GPU model tests).
        mx_atom: (with ``mx_out``, ``T <= 128``) ``xs`` is ``uint8 [(H / P * 128 / 128) *
            512]`` in the mid-M FP8 GEMM's scale layout (``k3_fp8w_lowm_gemm.run_mid_prequantized``
            with one 128-row block); the same exponents.
        ull2: The ``features.ull2_mla_tail`` sender path (:class:`K3HelixTailKernel`); always
            taken for FP16 split partials, which the original fold does not read.
    """
    num_heads = w_uv.shape[0]
    num_tokens = rows.shape[0] // num_heads
    if num_tokens == 0:
        return
    if num_tokens * cp_size * record_bytes(num_heads // cp_size, V_HEAD_DIM) > half_bytes:
        raise ValueError("the Helix tail receive buffer is too small for this many tokens")
    mx_atom = mx_atom and mx_out is not None
    if mx_atom and num_tokens > 128:
        raise ValueError(f"mx_atom covers at most 128 tokens, got {num_tokens}")
    has_mask = zero_kv_mask is not None
    flags = (fold is not None, has_mask, use_pdl, mx_out is not None, debug is not None,
             timestamps is not None, loopback, mx_atom)  # fmt: skip
    fold16 = fold is not None and fold.partial.dtype == torch.float16
    ull2 = ull2 or fold16
    kernel = _kernel(cp_size, num_heads, *flags, ull2)
    n_tiles = _ceil_div(num_tokens, ROWS)
    tiles_per_cta = _ceil_div(num_heads * n_tiles, max(max_ctas, num_heads))
    n_ctas = num_heads * _ceil_div(n_tiles, tiles_per_cta)
    if fold is not None:
        ptrs = (
            _gmem(cutlass.Float16 if fold16 else cutlass.Float32, fold.partial, 32),
            _gmem(cutlass.Float32, fold.maxima, 32),
            _gmem(cutlass.Float32, fold.sums, 32),
            _gmem(cutlass.Int32, fold.plan, 4),
            _gmem(cutlass.Float32, fold.q_scale, 4),
            _gmem(cutlass.Float32, fold.kv_scale, 4),
        )
        query_len = fold.valid_m // num_heads
        dims = (fold.physical_m, fold.partial.shape[0], fold.plan.numel())
        scales = (Float32(fold.sm_scale * LOG2_E), Float32(fold.output_scale))
    else:
        # Never dereferenced without ``fold``; any valid pointers keep one signature.
        ptrs = (
            _gmem(cutlass.Float32, stats, 4),
            _gmem(cutlass.Float32, stats, 4),
            _gmem(cutlass.Float32, stats, 4),
            _gmem(cutlass.Int32, stats, 4),
            _gmem(cutlass.Float32, stats, 4),
            _gmem(cutlass.Float32, stats, 4),
        )
        query_len, dims = 1, (1, 1, 1)
        scales = (Float32(1.0), Float32(1.0))
    dbg = (0, 0, 0) if debug is None else tuple(t.data_ptr() for t in debug)
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
        *(Int64(a) for a in dbg),
        Int64(0 if timestamps is None else timestamps.data_ptr()),
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
    key = (cp_size, num_heads, *flags, ull2, fold16)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(kernel, *args)
        _COMPILED[key] = compiled
    compiled(*args)


def reference_v_up(latent: torch.Tensor, w_uv: torch.Tensor) -> torch.Tensor:
    """FP64 V up-projection of BF16 latent rows ``[T * H, 512]`` with ``w_uv`` ``[H, 128, 512]``,
    rounded to BF16 (tests)."""
    num_heads = w_uv.shape[0]
    lat = latent.view(-1, num_heads, LATENT_DIM).double()
    out = torch.einsum("tjk,jnk->tjn", lat, w_uv.double())
    return out.reshape(-1, V_HEAD_DIM).to(torch.bfloat16)


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of BF16 values at ``x`` (tests)."""
    mag = x.float().abs().clamp_min(2.0**-126)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7)


assert math.isclose(LOG2_E, math.log2(math.e))
