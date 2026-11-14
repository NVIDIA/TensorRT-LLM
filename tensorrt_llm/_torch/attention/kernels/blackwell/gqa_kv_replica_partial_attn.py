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
"""KV-replica split partial decode attention, CuTe DSL (SM100 / SM103): one launch per call.

One member (CP rank) of a KV-replica group (cp TP ranks share one KV head; every rank holds the whole KV cache) reads
only the pages it owns (page p belongs to rank p % cp) for every q row of the group: rows = q_len tokens x gh group
heads (row n = t * gh + g, <= 128). It writes one partial (O normalized bf16, base-2 LSE fp32) per (token, group
head) in the layout alltoall_merge takes: out_o [cp, b * q_len, hq, D], out_lse [cp, b * q_len, hq], group head
g = d * hq + h going to member d. Contract: kv_replica_fused_partial_ref in the unit test (row (r, t) sees keys
j < L_r - q_len + t + 1); a rank that sees no key of a row writes O = 0 and LSE = -inf.

Unlike gqa_split_kv_decode_attn.py (keys on MMA M, <= 32 rows on N), the q rows are the MMA M dimension here (FA style,
softmax thread = row = TMEM lane), which keeps row max / sums thread-local at 112-128 rows and makes the partial
epilogue row-contiguous:

* grid (n_ctas, 1), one CTA per SM, all co-resident (see "Split" below). Request r's owned pages (owned index
  i -> global page cp_rank + i * cp) form 128-key tiles of two pages; each CTA takes a contiguous block of one
  request's tiles.
* warp 0, TMA: block-table entries -> SMEM, then per tile 4 boxes of K and 4 of V (64 keys x 128 B, SW128,
  EVICT_FIRST) into a 2-stage ring. Pages >= max((L - q_len) // 64 - fresh_margin_pages, 0) are requested only after
  griddepcontrol.wait (new tokens and positions MTP passes wrote earlier in the step). Warp 3 loads q
  (128 x 256 FP8 box, rows >= q_len * gh zero-filled) after the wait.
* warp 1, MMA (tcgen05 kind::f8f6f4, FP32 in TMEM, 512 columns): GEMM1 S[128 rows, 128 keys] = Q . K^T into one of
  two S buffers; GEMM2 O[128, 256] += P . V as two N = 128 MMAs (one per d half), one tile behind GEMM1.
* warps 4-7, softmax (thread = row): lazy rescale as in gqa_split_kv_decode_attn.py (a warp rescales its O rows in
  TMEM when any of its rows moved); P -> SMEM as E4M3 (K-major SW128). Warps whose rows are all padding skip the
  softmax; their P rows are zero.
* epilogue: O / l -> FP16 -> the CTA's chunk-major workspace slots, LSE per (row, CTA); grid barrier per request
  (64-bit epoch counter); CTA j of the request then merges chunk j of all its CTAs' partials, applies bmm2 and writes
  bf16 O / fp32 LSE. Dependents are released after a CTA barrier that follows the last global store.

PDL: the only global accesses before griddepcontrol.wait are reads of seq_lens, the block tables and prefix KV pages
(TMA); q, the scales, the epoch counter, the workspace and the outputs are touched after it.

Grid-barrier assumptions (counter row ``counters[b]``, see gqa_kv_replica_fused.py), as in gqa_split_kv_decode_attn.py:
(a) launches on one counter row never overlap (every kernel between two launches is a plain launch or waits before it
    exits); (b) no other kernel holds SMs while it waits for work ordered after this launch; (c) every launch on a row
    uses the same n_ctas.

Split: n_ctas = SMs for every batch, shared among the requests by their owned KV, so a ragged batch costs about its
total KV rather than its longest request:

* plan (every CTA, warp 2, from seq_lens before the dependency wait; graph-safe, all CTAs derive the same plan):
  request r owns t_r tiles on this rank; T = the smallest tile count with sum_r max(1, ceil(t_r / T)) <= n_ctas;
  request r gets c_r = max(1, ceil(t_r / T)) consecutive CTAs (the longest one also the spares), each taking a
  near-equal block of its tiles. A lane of the plan warp plans one request up to 32 requests and two (r, r + 32) with
  ``wide_plan`` (33..64); both compute the same plan, and every request needs a CTA (batch < n_ctas).
* partials: chunk-major in the workspace slots of the request's CTAs (cp_partial_balanced_geometry).
* grid barrier per request: the request's first CTA adds n_ctas - c_r + 1 and every other one 1, so every launch adds
  n_ctas to each counter of its row whatever the plan (assumption (c)).
* combine: CTA j of the request finalizes chunk j; the merge loop is dispatched to a compile-time lanes-per-granule
  variant.

Two-slot plan (``two_slot``, kernel_two, 65..128 requests): the plans above give every request at least one CTA, so
past ~n_ctas / 2 uniform requests a CTA runs a whole request while others idle.
Here T = the smallest tile count with sum_r max(t_r, T) <= n_ctas * T; request r takes the virtual tile range
[V_r, V_r + max(t_r, T)) (requests in order) and CTA i the range [i * T, (i + 1) * T), so a CTA computes at most T
tiles, of at most two requests (slot 0, then slot 1, through one K / V ring and one O accumulator). Each request keeps
the layout above with c_r = the CTAs meeting its tiles; its workspace slots start at the exclusive prefix sum of c_r
(sum <= 2 n_ctas, twice the slots); the grid barrier and the combine run once per slot.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

try:
    from cutlass.utils import SmemAllocator
except ImportError:  # pragma: no cover - older CuTe DSL
    from cutlass.memory import SmemAllocator

FP8 = cutlass.Float8E4M3FN
PAGE = 64  # tokens per KV page
HEAD_DIM = 256
TILE_KEYS = 128  # keys per tile (two pages) = MMA N of GEMM1 and MMA K of GEMM2
M_ROWS = 128  # MMA M: q rows (q_len x group heads), zero-padded
SW_BYTES = 128  # bytes per swizzle row (one TMA box column)
KV_STAGES = 2
EVICT_FIRST = 0x12F0000000000000  # CUTLASS TMA::CacheHintSm90::EVICT_FIRST
RESCALE_THRESHOLD = 8.0  # log2 units: P <= 2^8 = 256 < 448 (E4M3 max) without a P shift
P_TOP = 8.75  # log2 of the largest P (2^8.75 = 430 < 448): a P shift s lowers the lazy-rescale threshold to 8.75 - s
NUM_THREADS = (
    256  # warp 0 TMA, warp 1 MMA, warp 2 scalars, warp 3 q, warps 4-7 softmax (thread = row)
)
SOFTMAX_WARP0 = 4
TAB_MAX = 512  # block-table entries per split staged in SMEM (K and V each)
STAGE_ROW = (
    HEAD_DIM * 2 + 16
)  # bytes per row of the FP16 partial staging (16-byte pad: conflict-free v4 stores)
KV_REGION = (
    4 * KV_STAGES * 16384
)  # bytes of the K + V rings; reused for the staging and the combine
TMEM_COLS = 512  # 2 S buffers (2 x 128) + O (2 x 128)
COMB_LSE_OFF = 96 * 1024  # byte offset of the combine's LSE buffer in the K / V ring region
PLAN_WORDS = (
    8  # per-CTA plan in SMEM (request, index, CTAs, first tile, tiles, first CTA, L, valid)
)


def cp_partial_balanced_geometry(rows: int, n_ctas: int, slots: int = 1) -> dict:
    """Workspace and combine bounds for ``rows`` q rows and ``n_ctas`` CTAs (host and kernel).

    A request with c CTAs splits its row-major [rows][256] partials into chunks of chunk(c) elements (CTA j of the
    request finalizes chunk j); its partials [n_chunks(c)][c][chunk_pad(c)] FP16 occupy the O slots of its c
    consecutive CTAs (slot_o_bytes >= n_chunks(c) * chunk_pad(c) * 2 for every c), its LSEs [rows][lse_stride(c)] FP32
    the LSE slots (16 * rows bytes per CTA >= rows * lse_stride(c) * 4 / c). The combine stages c x chunk_pad FP16
    (<= COMB_LSE_OFF bytes) and nrows x lse_stride FP32. ``slots`` = 2 (the two-slot plan, where a CTA works for up
    to two requests) gives every CTA two slots: the sum of c is <= 2 x n_ctas there."""
    total = rows * HEAD_DIM
    slot_o, max_o, max_l = 0, 0, 0
    for c in range(1, n_ctas + 1):
        chunk = 8 * ((total + 8 * c - 1) // (8 * c))
        # an odd number of 16-byte granules per partial row: lanes reading one granule of consecutive partials hit
        # different SMEM bank groups
        chunk_pad = chunk + (8 if (chunk // 8) % 2 == 0 else 0)
        n_chunks = (total + chunk - 1) // chunk
        lse_stride = 4 * ((c + 3) // 4)
        nrows = min((chunk + HEAD_DIM - 2) // HEAD_DIM + 1, rows)  # rows one chunk can touch
        slot_o = max(slot_o, n_chunks * chunk_pad * 2)
        max_o = max(max_o, c * chunk_pad * 2)
        max_l = max(max_l, nrows * lse_stride * 4)
    slot_o = 16 * ((slot_o + 15) // 16)
    n_slots = slots * n_ctas
    return dict(
        slot_o_bytes=slot_o,
        lse_area_off=n_slots * slot_o,
        lse_cta_bytes=16 * rows,
        ws_bytes=n_slots * (slot_o + 16 * rows),
        max_comb_o_bytes=max_o,
        max_comb_lse_bytes=max_l,
    )


def _asm(ret, args, text, cons, side_effects, loc, ip):
    return llvm.inline_asm(
        ret,
        args,
        text,
        cons,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ex2(x, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _lg2(x, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _pack_e4m3x4(a, b, c, d, *, loc=None, ip=None) -> Uint32:
    """Four FP32 -> four E4M3 (cvt.rn.satfinite), ``a`` in the low byte."""
    out = _asm(
        T.i32(),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
        "{\n\t.reg .b16 lo, hi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\t"
        "mov.b32 $0, {lo, hi};\n\t}\n",
        "=r,f,f,f,f",
        False,
        loc,
        ip,
    )
    return Uint32(out)


@dsl_user_op
def _pack_f16x2(lo, hi, *, loc=None, ip=None) -> Uint32:
    """Two FP32 -> FP16x2 (round to nearest), ``lo`` in the low half."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
            "cvt.rn.f16x2.f32 $0, $1, $2;",
            "=r,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _pack_bf16x2(lo, hi, *, loc=None, ip=None) -> Uint32:
    """Two FP32 -> BF16x2 (round to nearest), ``lo`` in the low half."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _unpack_f16x2(w, *, loc=None, ip=None):
    """FP16x2 -> (low, high) FP32."""
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Uint32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, $2;\n\tcvt.f32.f16 $0, l;\n\tcvt.f32.f16 $1, h;\n\t}\n",
        "=f,=f,r",
        False,
        loc,
        ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def _keep_lt(x, j, cnt, *, loc=None, ip=None) -> Float32:
    """x if j < cnt else -inf (one setp + selp; straight-line masking without data-dependent branches)."""
    return Float32(
        _asm(
            T.f32(),
            [
                Float32(x).ir_value(loc=loc, ip=ip),
                Int32(j).ir_value(loc=loc, ip=ip),
                Int32(cnt).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred p;\n\tsetp.lt.s32 p, $2, $3;\n\tselp.f32 $0, $1, 0fFF800000, p;\n\t}\n",
            "=f,f,r,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _vote_any(pred, *, loc=None, ip=None) -> Int32:
    """Warp-wide OR of a predicate (all 32 lanes must execute it)."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(pred).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred pi, po;\n\tsetp.ne.s32 pi, $1, 0;\n\t"
            "vote.sync.any.pred po, pi, 0xffffffff;\n\tselp.s32 $0, 1, 0, po;\n\t}\n",
            "=r,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _redux_add_s32(x, *, loc=None, ip=None) -> Int32:
    """Warp-wide sum (all 32 lanes must execute it)."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(x).ir_value(loc=loc, ip=ip)],
            "redux.sync.add.s32 $0, $1, 0xffffffff;",
            "=r,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _redux_max_s32(x, *, loc=None, ip=None) -> Int32:
    """Warp-wide max (all 32 lanes must execute it)."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(x).ir_value(loc=loc, ip=ip)],
            "redux.sync.max.s32 $0, $1, 0xffffffff;",
            "=r,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ballot_top(pred, *, loc=None, ip=None) -> Int32:
    """Highest lane whose predicate is set (-1 if none; all 32 lanes must execute it)."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(pred).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred pi;\n\t.reg .b32 bal;\n\tsetp.ne.s32 pi, $1, 0;\n\t"
            "vote.sync.ballot.b32 bal, pi, 0xffffffff;\n\tbfind.u32 $0, bal;\n\t}\n",
            "=r,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _shfl_up_s32(x, off, *, loc=None, ip=None) -> Int32:
    """Value of lane - off (own value for lanes < off; all 32 lanes must execute it)."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(x).ir_value(loc=loc, ip=ip), Int32(off).ir_value(loc=loc, ip=ip)],
            "shfl.sync.up.b32 $0, $1, $2, 0, 0xffffffff;",
            "=r,r,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _bfind_u32(x, *, loc=None, ip=None) -> Int32:
    """Index of the most significant set bit (-1 for 0)."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(x).ir_value(loc=loc, ip=ip)],
            "bfind.u32 $0, $1;",
            "=r,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _div_floor_s32(a, b, *, loc=None, ip=None) -> Int32:
    """floor(a / b) for 0 <= a < 2^24, 1 <= b < 2^24: FP32 reciprocal estimate (within one of the quotient there)
    corrected by one step either way; a short dependency chain instead of the integer division sequence."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .f32 fa, fb, fq;\n\t.reg .s32 q, p, r;\n\t.reg .pred hi, lo;\n\t"
            "cvt.rn.f32.s32 fa, $1;\n\tcvt.rn.f32.s32 fb, $2;\n\trcp.approx.ftz.f32 fb, fb;\n\t"
            "mul.rn.f32 fq, fa, fb;\n\tcvt.rzi.s32.f32 q, fq;\n\tmul.lo.s32 p, q, $2;\n\t"
            "setp.gt.s32 hi, p, $1;\n\t@hi sub.s32 q, q, 1;\n\t@hi sub.s32 p, p, $2;\n\t"
            "sub.s32 r, $1, p;\n\tsetp.ge.s32 lo, r, $2;\n\t@lo add.s32 q, q, 1;\n\tmov.b32 $0, q;\n\t}\n",
            "=r,r,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_shared_v4(addr, a, b, c, d, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _ld_shared_v4_u32(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        True,
        loc,
        ip,
    )
    return tuple(Uint32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _ld_shared_f32(addr, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_global_v4_u32(addr, a, b, c, d, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _st_global_u32(addr, a, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(a).ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _st_global_f32(addr, v, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _griddep_wait(*, loc=None, ip=None) -> None:
    _asm(None, [], "griddepcontrol.wait;", "~{memory}", True, loc, ip)


@dsl_user_op
def _griddep_launch(*, loc=None, ip=None) -> None:
    _asm(None, [], "griddepcontrol.launch_dependents;", "~{memory}", True, loc, ip)


@dsl_user_op
def _red_release_add_u64(addr, val, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(val).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.u64 [$0], $1;",
        "l,l",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _ld_acquire_gpu_u64(addr, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.u64 $0, [$1];",
            "=l,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ld_relaxed_gpu_u64(addr, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.u64 $0, [$1];",
            "=l,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _bulk_g2s(dst_smem, src_gmem, nbytes, mbar_smem, *, loc=None, ip=None) -> None:
    """cp.async.bulk global -> shared (16-byte aligned, size a multiple of 16), completion on an mbarrier."""
    _asm(
        None,
        [
            Int32(dst_smem).ir_value(loc=loc, ip=ip),
            Int64(src_gmem).ir_value(loc=loc, ip=ip),
            Int32(nbytes).ir_value(loc=loc, ip=ip),
            Int32(mbar_smem).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _nanosleep(ns, *, loc=None, ip=None) -> None:
    _asm(None, [Int32(ns).ir_value(loc=loc, ip=ip)], "nanosleep.u32 $0;", "r", True, loc, ip)


class GqaKvReplicaPartialAttnKernel:
    """See the module docstring. Compile-time parameters:

    q_len, gh, hql, cp_size: tokens per request, group heads (= cp_size * hql), q heads per rank, CP group size;
                     rows = q_len * gh <= 128.
    n_ctas:          CTAs of the grid (batch < n_ctas <= SMs, batch <= 32 or the variants below), shared among the
                     requests by the in-kernel plan (module docstring).
    wide_plan:       33..64 requests: the plan warp handles requests lane and lane + 32 (_plan_wide, the plan _plan
                     would compute with 64 lanes); the host compiles it only for batch > 32.
    two_slot:        the two-slot plan (module docstring; kernel_two, _plan_two, up to 128 requests): the host's
                     choice above 64 requests.
    bmm1_from_ptr:   read the log2-domain QK scale from ``bmm1_ptr[1]`` (trtllm-gen [raw, raw * log2e] layout) after the
                     dependency wait, else use ``bmm1_log2``. Same for ``bmm2`` (``bmm2_ptr[0]``).
    o_vec16:         out_o rows are 16-byte aligned (8-element vector stores), else 4-byte stores.
    use_pdl:         launched as a programmatic dependent (waits, releases its dependents at the end); the prefix
                     pages below the fresh margin are requested before griddepcontrol.wait.
    fresh_margin_pages: prefix pages just below the new tokens' first page that are read only after the wait.
    p_shift:         E4M3 P shift (log2 units, see gqa_split_kv_decode_attn.py).
    """

    def __init__(
        self,
        *,
        q_len: int,
        gh: int,
        hql: int,
        cp_size: int,
        n_ctas: int,
        bmm1_from_ptr: bool,
        bmm2_from_ptr: bool,
        o_vec16: bool,
        use_pdl: bool = True,
        fresh_margin_pages: int = 1,
        evict_first: bool = True,
        p_shift: float = 5.0,
        wide_plan: bool = False,
        two_slot: bool = False,
    ):
        assert gh == cp_size * hql, "group heads = cp_size x q heads per rank"
        assert not (wide_plan and two_slot), "one plan variant"
        self.wide_plan = bool(wide_plan)
        self.two_slot = bool(two_slot)
        self.q_len = q_len
        self.gh = gh
        self.hql = hql
        self.cp = cp_size
        self.rows = q_len * gh
        assert 1 <= self.rows <= M_ROWS, (
            f"q_len * group heads = {self.rows} must be in [1, {M_ROWS}]"
        )
        self.act_warps = (self.rows + 31) // 32  # softmax warps that hold real rows
        self.n_ctas = n_ctas
        self.bmm1_from_ptr = bmm1_from_ptr
        self.bmm2_from_ptr = bmm2_from_ptr
        self.o_vec16 = o_vec16
        self.use_pdl = use_pdl
        self.fresh_margin_pages = fresh_margin_pages
        self.evict_first = evict_first
        # P = exp2(x - m_ref + p_shift) before the E4M3 conversion (see gqa_split_kv_decode_attn.py): the reference
        # max maps to 2^p_shift; l carries the same factor (the LSE subtracts it); the lazy rescale threshold is
        # P_TOP - p_shift.
        self.p_shift = float(p_shift)
        self.rescale_threshold = min(RESCALE_THRESHOLD, P_TOP - self.p_shift)
        assert 32 * self.act_warps * STAGE_ROW <= KV_REGION
        geo = cp_partial_balanced_geometry(self.rows, n_ctas, 2 if self.two_slot else 1)
        self.ws_bytes = geo["ws_bytes"]
        self.slot_o_bytes = geo["slot_o_bytes"]
        self.lse_area_off = geo["lse_area_off"]
        self.lse_cta_bytes = geo["lse_cta_bytes"]
        assert geo["max_comb_o_bytes"] <= COMB_LSE_OFF, "combine: O partials exceed their buffer"
        assert geo["max_comb_lse_bytes"] <= KV_REGION - COMB_LSE_OFF, (
            "combine: LSEs exceed their buffer"
        )

    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,
        kv_ptr: cute.Pointer,
        n_kv_rows: Int32,
        tab_ptr: cute.Pointer,
        tab_stride_b: Int32,
        tab_stride_kv: Int32,
        seqlen_ptr: cute.Pointer,
        cp_rank: Int32,
        bmm1_ptr: cute.Pointer,
        bmm1_log2: Float32,
        bmm2_ptr: cute.Pointer,
        bmm2: Float32,
        o_ptr: cute.Pointer,
        o_sd: Int32,
        o_st: Int32,
        o_sh: Int32,
        lse_ptr: cute.Pointer,
        l_sd: Int32,
        l_st: Int32,
        l_sh: Int32,
        ws_ptr: cute.Pointer,
        cnt_ptr: cute.Pointer,
        batch: Int32,
        stream,
    ):
        R = self.rows
        NS = self.n_ctas
        # q of request r: rows [r * R, (r + 1) * R) of the all-gather buffer viewed as [b * R, 256] (row t * gh + g)
        mQ = cute.make_tensor(
            q_ptr, cute.make_layout((R, HEAD_DIM, batch), stride=(HEAD_DIM, 1, R * HEAD_DIM))
        )
        mKV = cute.make_tensor(
            kv_ptr, cute.make_layout((n_kv_rows, HEAD_DIM), stride=(HEAD_DIM, 1))
        )
        mKVt = cute.make_tensor(
            kv_ptr, cute.make_layout((HEAD_DIM, n_kv_rows), stride=(1, HEAD_DIM))
        )
        mTab = cute.make_tensor(
            tab_ptr,
            cute.make_layout(
                (batch, 2, 1 << 24),  # page extent unused (no bounds)
                stride=(tab_stride_b, tab_stride_kv, 1),
            ),
        )
        mSeq = cute.make_tensor(seqlen_ptr, cute.make_layout((batch,)))
        mBmm1 = cute.make_tensor(bmm1_ptr, cute.make_layout((2,)))
        mBmm2 = cute.make_tensor(bmm2_ptr, cute.make_layout((1,)))
        # outputs / workspace are addressed explicitly (vector stores, bulk copies)
        mO = cute.make_tensor(o_ptr, cute.make_layout((1,)))
        mLse = cute.make_tensor(lse_ptr, cute.make_layout((1,)))
        mWs = cute.make_tensor(ws_ptr, cute.make_layout((1,)))
        mCnt = cute.make_tensor(cnt_ptr, cute.make_layout((8,)))  # grid-barrier epoch per request

        mma_qk = sm100_utils.make_trivial_tiled_mma(
            FP8,
            FP8,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (M_ROWS, TILE_KEYS),
        )
        mma_pv = sm100_utils.make_trivial_tiled_mma(
            FP8,
            FP8,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            Float32,
            tcgen05.CtaGroup.ONE,
            (M_ROWS, 128),
        )
        qk_tiler = (M_ROWS, TILE_KEYS, HEAD_DIM)
        pv_tiler = (M_ROWS, 128, TILE_KEYS)  # one d half per MMA
        sQ_l = sm100_utils.make_smem_layout_a(mma_qk, qk_tiler, FP8, 1)
        sK_l = sm100_utils.make_smem_layout_b(mma_qk, qk_tiler, FP8, KV_STAGES)
        sP_l = sm100_utils.make_smem_layout_a(mma_pv, pv_tiler, FP8, 2)
        # V: (d half h of stage s) = MMA stage 2 s + h; physically the [stage][d half][key atoms] layout of the TMA view
        sV_l = sm100_utils.make_smem_layout_b(mma_pv, pv_tiler, FP8, 2 * KV_STAGES)
        # the same buffers seen by the TMA (one box = one page x 128 B) and by the softmax P stores ((row, key, buffer))
        sK_tma_l = cute.tiled_divide(
            sm100_utils.make_smem_layout(OperandMajorMode.K, (TILE_KEYS, HEAD_DIM), FP8, KV_STAGES),
            (PAGE, SW_BYTES),
        )
        sV_tma_l = cute.tiled_divide(
            sm100_utils.make_smem_layout(
                OperandMajorMode.MN, (HEAD_DIM, TILE_KEYS), FP8, KV_STAGES
            ),
            (SW_BYTES, PAGE),
        )
        sP_mk_l = sm100_utils.make_smem_layout(OperandMajorMode.K, (M_ROWS, TILE_KEYS), FP8, 2)
        box_k = sm100_utils.make_smem_layout(OperandMajorMode.K, (PAGE, SW_BYTES), FP8, 1)
        box_v = sm100_utils.make_smem_layout(OperandMajorMode.MN, (SW_BYTES, PAGE), FP8, 1)

        op = cpasync.CopyBulkTensorTileG2SOp()
        tma_k, gK_tma = cpasync.make_tiled_tma_atom(op, mKV, box_k, (PAGE, SW_BYTES))
        tma_v, gV_tma = cpasync.make_tiled_tma_atom(op, mKVt, box_v, (SW_BYTES, PAGE))
        cluster_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma_qk.thr_id.shape,))
        sQ_one = cute.slice_(sQ_l, (None, None, None, 0))
        tma_q, gQ_tma = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A((1, 1), mma_qk.thr_id),
            mQ,
            sQ_one,
            qk_tiler,
            mma_qk,
            cluster_vmnk.shape,
        )
        q_bytes = cute.size_in_bytes(FP8, sQ_one)

        grid = [NS, 1, 1]
        if const_expr(self.two_slot):
            self.kernel_two(
                mma_qk,
                mma_pv,
                tma_k,
                gK_tma,
                tma_v,
                gV_tma,
                tma_q,
                gQ_tma,
                mTab,
                mSeq,
                cp_rank,
                mBmm1,
                bmm1_log2,
                mBmm2,
                bmm2,
                mO,
                o_sd,
                o_st,
                o_sh,
                mLse,
                l_sd,
                l_st,
                l_sh,
                mWs,
                mCnt,
                sQ_l,
                sK_l,
                sV_l,
                sP_l,
                sK_tma_l,
                sV_tma_l,
                sP_mk_l,
                q_bytes,
            ).launch(
                grid=grid,
                block=[NUM_THREADS, 1, 1],
                stream=stream,
                use_pdl=self.use_pdl,
                min_blocks_per_mp=1,
            )
        else:
            self.kernel(
                mma_qk,
                mma_pv,
                tma_k,
                gK_tma,
                tma_v,
                gV_tma,
                tma_q,
                gQ_tma,
                mTab,
                mSeq,
                cp_rank,
                mBmm1,
                bmm1_log2,
                mBmm2,
                bmm2,
                mO,
                o_sd,
                o_st,
                o_sh,
                mLse,
                l_sd,
                l_st,
                l_sh,
                mWs,
                mCnt,
                sQ_l,
                sK_l,
                sV_l,
                sP_l,
                sK_tma_l,
                sV_tma_l,
                sP_mk_l,
                q_bytes,
            ).launch(
                grid=grid,
                block=[NUM_THREADS, 1, 1],
                stream=stream,
                use_pdl=self.use_pdl,
                min_blocks_per_mp=1,
            )

    @cute.jit
    def _produce(
        self,
        i,
        kb0,
        kb1,
        vb0,
        vb1,
        tma_k: cute.CopyAtom,
        tKg: cute.Tensor,
        tKs: cute.Tensor,
        tma_v: cute.CopyAtom,
        tVg: cute.Tensor,
        tVs: cute.Tensor,
        full_k,
        empty_k,
        full_v,
        empty_v,
    ):
        """TMA of tile i of the ring: pages (slot 0, slot 1) of K (blocks kb0, kb1) and V (vb0, vb1)."""
        S: cutlass.Constexpr = KV_STAGES
        s = i % S
        ph = (i // S) & 1
        kbs = (kb0, kb1)
        vbs = (vb0, vb1)
        cute.arch.mbarrier_wait(empty_k + s, ph ^ 1)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(full_k + s, 2 * PAGE * HEAD_DIM)
        for p in cutlass.range_constexpr(2):
            for kb in cutlass.range_constexpr(2):
                if const_expr(self.evict_first):
                    cute.copy(
                        tma_k,
                        tKg[(None, kbs[p], kb)],
                        tKs[(None, p, kb, s)],
                        tma_bar_ptr=full_k + s,
                        cache_policy=Int64(EVICT_FIRST),
                    )
                else:
                    cute.copy(
                        tma_k,
                        tKg[(None, kbs[p], kb)],
                        tKs[(None, p, kb, s)],
                        tma_bar_ptr=full_k + s,
                    )
        cute.arch.mbarrier_wait(empty_v + s, ph ^ 1)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(full_v + s, 2 * PAGE * HEAD_DIM)
        for p in cutlass.range_constexpr(2):
            for h in cutlass.range_constexpr(2):
                if const_expr(self.evict_first):
                    cute.copy(
                        tma_v,
                        tVg[(None, h, vbs[p])],
                        tVs[(None, h, p, s)],
                        tma_bar_ptr=full_v + s,
                        cache_policy=Int64(EVICT_FIRST),
                    )
                else:
                    cute.copy(
                        tma_v, tVg[(None, h, vbs[p])], tVs[(None, h, p, s)], tma_bar_ptr=full_v + s
                    )

    @cute.jit
    def _gemm2(
        self,
        j,
        mma_pv: cute.TiledMma,
        tOtO: cute.Tensor,
        tCrP: cute.Tensor,
        tCrV: cute.Tensor,
        full_v,
        p_full,
        empty_v,
        p_empty,
        o_done,
        j_loc=None,
    ):
        """O += P(tile j) . V(tile j) (two d halves); releases the V stage and the P buffer, signals o_done. Returns
        mma_pv: its ACCUMULATE field is updated here, and the caller must rebind it. j counts the CTA's tiles (ring
        and barrier phases); j_loc (kernel_two) is the tile's index in its request: O restarts at j_loc 0."""
        S: cutlass.Constexpr = KV_STAGES
        sj = j % S
        pb = j % 2
        jl = j if j_loc is None else j_loc
        cute.arch.mbarrier_wait(full_v + sj, (j // S) & 1)
        cute.arch.mbarrier_wait(p_full + pb, (j // 2) & 1)
        for h in cutlass.range_constexpr(2):
            for kk in cutlass.range_constexpr(TILE_KEYS // 32):
                mma_pv.set(tcgen05.Field.ACCUMULATE, (jl > 0) or (kk != 0))
                cute.gemm(
                    mma_pv,
                    tOtO[(None, 0, h)],
                    tCrP[(None, 0, kk, pb)],
                    tCrV[(None, 0, kk, 2 * sj + h)],
                    tOtO[(None, 0, h)],
                )
        with cute.arch.elect_one():
            tcgen05.commit(empty_v + sj)
            tcgen05.commit(p_empty + pb)
            tcgen05.commit(o_done)
        return mma_pv

    @cute.jit
    def _softmax_tile(
        self,
        i,
        xs,
        ob64,
        stt,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        t2r,
        thr_ld,
        t2r64,
        thr_ld64,
        r2t64,
        thr_st64,
        bars_sm,
        sP_base,
        sP_mk_l: cute.ComposedLayout,
        st,
        cnt_a,
        cnt_b,
        scale,
        apply_mask: cutlass.Constexpr,
        i_loc=None,
    ):
        """Softmax of tile i (CTA-local) for row st (TMEM lane st): S -> x, mask (apply_mask: keys of slot 0 / 1 visible
        iff key < cnt_a / cnt_b), lazy rescale of (m, l, O row), P -> SMEM as E4M3. stt = [m_ref, l_acc].
        The O rescale of a lazy-rescale tile runs after P(i) is in SMEM (by then GEMM2(i - 1) has normally completed,
        and the S registers are free for 64-column TMEM round trips), before P(i) is released to GEMM2(i).
        i_loc (kernel_two): the tile's index in its request (O holds nothing of the request before i_loc 1)."""
        s_full, s_empty, p_full, p_empty, o_done = bars_sm
        il = i if i_loc is None else i_loc
        sb = i % 2
        cute.arch.mbarrier_wait(s_full + sb, (i // 2) & 1)
        accS = tStS[(None, None), 0, 0, sb]
        for c in cutlass.range_constexpr(4):
            cute.copy(t2r, thr_ld.partition_S(cute.local_tile(accS, (M_ROWS, 32), (0, c))), xs[c])
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(s_empty + sb)
        # Mask (tiles with the new tokens, the request tail or a duplicate second slot; straight-line)
        if const_expr(apply_mask):
            for c in cutlass.range_constexpr(4):
                for v in cutlass.range_constexpr(32):
                    if const_expr(32 * c + v < PAGE):
                        xs[c][v] = _keep_lt(xs[c][v], 32 * c + v, cnt_a)
                    else:
                        xs[c][v] = _keep_lt(xs[c][v], 32 * c + v - PAGE, cnt_b)
        # Row max of the raw scores (scale > 0)
        m0 = xs[0][0]
        m1 = xs[1][0]
        m2 = xs[2][0]
        m3 = xs[3][0]
        for v in cutlass.range_constexpr(1, 32):
            m0 = cute.arch.fmax(m0, xs[0][v])
            m1 = cute.arch.fmax(m1, xs[1][v])
            m2 = cute.arch.fmax(m2, xs[2][v])
            m3 = cute.arch.fmax(m3, xs[3][v])
        mx = cute.arch.fmax(cute.arch.fmax(m0, m1), cute.arch.fmax(m2, m3)) * scale
        m_ref = stt[0]
        need = Int32(0)
        if mx > m_ref + self.rescale_threshold:
            need = Int32(1)
        resc = _vote_any(need)
        alpha = Float32(1.0)
        if resc != 0:
            # slow path (warp-uniform): move the reference max of the rows that exceeded it and rescale l now; the O
            # rows are rescaled below, once P(i) is written
            m_new = m_ref
            if need != 0:
                m_new = mx
            alpha = Float32(0.0)
            if m_ref != -Float32.inf:
                alpha = _ex2(m_ref - m_new)
            stt[1] = stt[1] * alpha
            stt[0] = m_new
        # P = exp2(s * scale - m), row sum (FP32), E4M3 P row -> SMEM (K-major SW128)
        m_use = stt[0]
        if m_use == -Float32.inf:
            m_use = Float32(0.0)  # no visible key yet: every x is -inf, P = 0
        neg_m = self.p_shift - m_use
        la = Float32(0.0)
        lb = Float32(0.0)
        lc = Float32(0.0)
        ld = Float32(0.0)
        pw = cute.make_rmem_tensor((32,), Uint32)
        for c in cutlass.range_constexpr(4):
            for q4 in cutlass.range_constexpr(8):
                p0 = _ex2(_fma(xs[c][4 * q4], scale, neg_m))
                p1 = _ex2(_fma(xs[c][4 * q4 + 1], scale, neg_m))
                p2 = _ex2(_fma(xs[c][4 * q4 + 2], scale, neg_m))
                p3 = _ex2(_fma(xs[c][4 * q4 + 3], scale, neg_m))
                la = la + p0
                lb = lb + p1
                lc = lc + p2
                ld = ld + p3
                pw[8 * c + q4] = _pack_e4m3x4(p0, p1, p2, p3)
        stt[1] = stt[1] + ((la + lb) + (lc + ld))
        pb = i % 2
        cute.arch.mbarrier_wait(p_empty + pb, ((i // 2) & 1) ^ 1)
        for c16 in cutlass.range_constexpr(TILE_KEYS // 16):
            off = cute.crd2idx((st, 16 * c16, pb), sP_mk_l)
            _st_shared_v4(
                sP_base + off, pw[4 * c16], pw[4 * c16 + 1], pw[4 * c16 + 2], pw[4 * c16 + 3]
            )
        if resc != 0:
            if il > 0:
                # O holds GEMM2 of tiles < i once GEMM2(i - 1) completed; GEMM2(i) waits for P(i) (released below).
                cute.arch.mbarrier_wait(o_done, (i - 1) & 1)
                for h in cutlass.range_constexpr(2):
                    accO = tOtO[(None, None), 0, h]
                    for c in cutlass.range_constexpr(2):
                        tO = cute.local_tile(accO, (M_ROWS, 64), (0, c))
                        cute.copy(t2r64, thr_ld64.partition_S(tO), ob64)
                        cute.arch.fence_view_async_tmem_load()
                        for v in cutlass.range_constexpr(64):
                            ob64[v] = ob64[v] * alpha
                        cute.copy(r2t64, ob64, thr_st64.partition_D(tO))
                cute.arch.fence_view_async_tmem_store()
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.mbarrier_arrive(p_full + pb)

    def _out_coords(self, row, req):
        """(member d, token, local head h) of q row ``row`` (= t * gh + g, g = d * hq + h) of request ``req``
        (trace-time helper, inlined)."""
        g = row % self.gh
        return g // self.hql, req * self.q_len + row // self.gh, g % self.hql

    @cute.jit
    def _plan(self, mSeq: cute.Tensor, cp_rank: Int32, cta: Int32, lane: Int32, sPlan: cute.Tensor):
        """Split plan, one warp (lane r = request r; batch <= 32): this CTA's (request, index in the request,
        CTAs of the request, first tile, tiles, request's first CTA, L, CTAs with work) -> sPlan. Every CTA of the grid
        derives the same plan from seq_lens (see the module docstring)."""
        CP: cutlass.Constexpr = self.cp
        NC: cutlass.Constexpr = self.n_ctas
        batch = mSeq.shape[0]
        L = Int32(0)
        if lane < batch:
            L = mSeq[lane]
        # owned tiles of request `lane` on this rank (0 for lanes >= batch: L = 0)
        t = ((((L + PAGE - 1) // PAGE) - cp_rank + CP - 1) // CP + 1) // 2
        total = _redux_add_s32(t)
        tmax = _redux_max_s32(t)
        # smallest T (tiles per CTA) with sum max(1, ceil(t / T)) <= NC; ceil(total / (NC - batch)) and max t fit
        # (tile counts stay far below 2^24: _div_floor_s32's range)
        lo = cutlass.max((total + NC - 1) // NC, 1)
        hi = cutlass.max(cutlass.min(tmax, _div_floor_s32(total + NC - batch - 1, NC - batch)), lo)
        while lo < hi:
            mid = (lo + hi) // 2
            need = Int32(0)
            if lane < batch:
                need = cutlass.max(_div_floor_s32(t + mid - 1, mid), 1)
            if _redux_add_s32(need) <= NC:
                hi = mid
            else:
                lo = mid + 1
        c = Int32(0)
        if lane < batch:
            c = cutlass.max(_div_floor_s32(t + lo - 1, lo), 1)
        spare = NC - _redux_add_s32(c)
        top = Int32(0)
        if lane < batch:
            if t == tmax:
                top = Int32(1)
        if lane == _ballot_top(top):
            c = c + spare  # the spare CTAs go to (one of) the longest requests
        incl = c
        for k in cutlass.range_constexpr(5):
            up = _shfl_up_s32(incl, 1 << k)
            if lane >= (1 << k):
                incl = incl + up
        first = incl - c
        if lane < batch:
            if first <= cta:
                if cta < incl:
                    j = cta - first
                    base = _div_floor_s32(t, c)
                    rem = t - base * c
                    n_t = base
                    if j < rem:
                        n_t = base + 1
                    sPlan[0] = lane
                    sPlan[1] = j
                    sPlan[2] = c
                    sPlan[3] = j * base + cutlass.min(j, rem)
                    sPlan[4] = n_t
                    sPlan[5] = first
                    sPlan[6] = L
                    sPlan[7] = cutlass.min(c, t)

    @cute.jit
    def _plan_wide(
        self, mSeq: cute.Tensor, cp_rank: Int32, cta: Int32, lane: Int32, sPlan: cute.Tensor
    ):
        """_plan for up to 64 requests (wide_plan): lane r plans request r (lower half) and request r + 32 (upper
        half). Same plan as _plan with 64 lanes: T by
        the same binary search, the spare CTAs to the highest-numbered longest request, CTAs in request order."""
        CP: cutlass.Constexpr = self.cp
        NC: cutlass.Constexpr = self.n_ctas
        batch = mSeq.shape[0]
        lane_b = lane + 32
        La = Int32(0)
        Lb = Int32(0)
        if lane < batch:
            La = mSeq[lane]
        if lane_b < batch:
            Lb = mSeq[lane_b]
        # owned tiles on this rank of requests `lane` / `lane + 32` (0 past the batch: L = 0)
        ta = ((((La + PAGE - 1) // PAGE) - cp_rank + CP - 1) // CP + 1) // 2
        tb = ((((Lb + PAGE - 1) // PAGE) - cp_rank + CP - 1) // CP + 1) // 2
        total = _redux_add_s32(ta + tb)
        tmax = _redux_max_s32(cutlass.max(ta, tb))
        lo = cutlass.max((total + NC - 1) // NC, 1)
        hi = cutlass.max(cutlass.min(tmax, _div_floor_s32(total + NC - batch - 1, NC - batch)), lo)
        while lo < hi:
            mid = (lo + hi) // 2
            need = Int32(0)
            if lane < batch:
                need = cutlass.max(_div_floor_s32(ta + mid - 1, mid), 1)
            if lane_b < batch:
                need = need + cutlass.max(_div_floor_s32(tb + mid - 1, mid), 1)
            if _redux_add_s32(need) <= NC:
                hi = mid
            else:
                lo = mid + 1
        ca = Int32(0)
        cb = Int32(0)
        if lane < batch:
            ca = cutlass.max(_div_floor_s32(ta + lo - 1, lo), 1)
        if lane_b < batch:
            cb = cutlass.max(_div_floor_s32(tb + lo - 1, lo), 1)
        spare = NC - _redux_add_s32(ca + cb)
        top_a = Int32(0)
        top_b = Int32(0)
        if lane < batch:
            if ta == tmax:
                top_a = Int32(1)
        if lane_b < batch:
            if tb == tmax:
                top_b = Int32(1)
        # the spare CTAs go to the highest-numbered longest request (an upper-half one when there is one)
        hb = _ballot_top(top_b)
        ha = _ballot_top(top_a)
        if hb >= 0:
            if lane == hb:
                cb = cb + spare
        else:
            if lane == ha:
                ca = ca + spare
        # first CTA of each request: requests 0..31 (lower half) take CTAs first, then 32..63
        incl_a = ca
        incl_b = cb
        for k in cutlass.range_constexpr(5):
            up_a = _shfl_up_s32(incl_a, 1 << k)
            up_b = _shfl_up_s32(incl_b, 1 << k)
            if lane >= (1 << k):
                incl_a = incl_a + up_a
                incl_b = incl_b + up_b
        incl_b = incl_b + _redux_add_s32(ca)
        first_a = incl_a - ca
        first_b = incl_b - cb
        if lane < batch:
            if first_a <= cta:
                if cta < incl_a:
                    self._plan_store(sPlan, lane, cta - first_a, ta, ca, first_a, La)
        if lane_b < batch:
            if first_b <= cta:
                if cta < incl_b:
                    self._plan_store(sPlan, lane_b, cta - first_b, tb, cb, first_b, Lb)

    @cute.jit
    def _plan_two(
        self, mSeq: cute.Tensor, cp_rank: Int32, cta: Int32, lane: Int32, sPlan: cute.Tensor
    ):
        """Two-slot plan (two_slot, up to 128 requests; lanes plan requests r, r + 32, r + 64, r + 96). T = the smallest
        tile count with sum_r max(t_r, T) <= NC * T. Request r occupies the virtual tile range [V_r, V_r + max(t_r, T))
        (requests in order) and CTA i the range [i * T, (i + 1) * T): a CTA meets at most two requests (every range is
        at least T long), slot 0 = the request whose range holds the CTA's first position, slot 1 = the request that
        starts inside it. The CTAs of request r are those meeting its real tiles [V_r, V_r + t_r) (the one holding V_r
        when t_r = 0; it writes O = 0 / LSE = -inf). sPlan[8 k + w], slot k: (request or -1 for none, index among the
        request's CTAs, its CTAs, first tile, tiles, first workspace slot, L, CTAs with work)."""
        CP: cutlass.Constexpr = self.cp
        NC: cutlass.Constexpr = self.n_ctas
        batch = mSeq.shape[0]
        if lane < 2 * PLAN_WORDS:
            sPlan[lane] = Int32(0)
            if lane % PLAN_WORDS == 0:
                sPlan[lane] = Int32(-1)  # no request in the slot unless one is stored below
        cute.arch.sync_warp()
        lane_b = lane + 32
        lane_c = lane + 64
        lane_d = lane + 96
        La = Int32(0)
        Lb = Int32(0)
        Lc = Int32(0)
        Ld = Int32(0)
        if lane < batch:
            La = mSeq[lane]
        if lane_b < batch:
            Lb = mSeq[lane_b]
        if lane_c < batch:
            Lc = mSeq[lane_c]
        if lane_d < batch:
            Ld = mSeq[lane_d]
        ta = ((((La + PAGE - 1) // PAGE) - cp_rank + CP - 1) // CP + 1) // 2
        tb = ((((Lb + PAGE - 1) // PAGE) - cp_rank + CP - 1) // CP + 1) // 2
        tc = ((((Lc + PAGE - 1) // PAGE) - cp_rank + CP - 1) // CP + 1) // 2
        td = ((((Ld + PAGE - 1) // PAGE) - cp_rank + CP - 1) // CP + 1) // 2
        total = _redux_add_s32(ta + tb + tc + td)
        tmax = _redux_max_s32(cutlass.max(cutlass.max(ta, tb), cutlass.max(tc, td)))
        # smallest T with sum max(t_r, T) <= NC * T (the left side grows by at most batch < NC per step of T)
        lo = cutlass.max((total + NC - 1) // NC, 1)
        hi = cutlass.max(tmax, lo)
        while lo < hi:
            mid = (lo + hi) // 2
            need = Int32(0)
            if lane < batch:
                need = cutlass.max(ta, mid)
            if lane_b < batch:
                need = need + cutlass.max(tb, mid)
            if lane_c < batch:
                need = need + cutlass.max(tc, mid)
            if lane_d < batch:
                need = need + cutlass.max(td, mid)
            if _redux_add_s32(need) <= NC * mid:
                hi = mid
            else:
                lo = mid + 1
        T = lo
        # virtual lengths, then their exclusive prefix sums V in request order (quarters a, b, c, d)
        va = Int32(0)
        vb = Int32(0)
        vc = Int32(0)
        vd = Int32(0)
        if lane < batch:
            va = cutlass.max(ta, T)
        if lane_b < batch:
            vb = cutlass.max(tb, T)
        if lane_c < batch:
            vc = cutlass.max(tc, T)
        if lane_d < batch:
            vd = cutlass.max(td, T)
        iva = va
        ivb = vb
        ivc = vc
        ivd = vd
        for k in cutlass.range_constexpr(5):
            ua = _shfl_up_s32(iva, 1 << k)
            ub = _shfl_up_s32(ivb, 1 << k)
            uc = _shfl_up_s32(ivc, 1 << k)
            ud = _shfl_up_s32(ivd, 1 << k)
            if lane >= (1 << k):
                iva = iva + ua
                ivb = ivb + ub
                ivc = ivc + uc
                ivd = ivd + ud
        sva = _redux_add_s32(va)
        svb = _redux_add_s32(vb)
        svc = _redux_add_s32(vc)
        Va = iva - va
        Vb = ivb + sva - vb
        Vc = ivc + sva + svb - vc
        Vd = ivd + sva + svb + svc - vd
        # first / last CTA meeting the real tiles (one position when t = 0) and the request's CTA count
        fa = _div_floor_s32(Va, T)
        fb = _div_floor_s32(Vb, T)
        fc = _div_floor_s32(Vc, T)
        fd = _div_floor_s32(Vd, T)
        ca = Int32(0)
        cb = Int32(0)
        cc = Int32(0)
        cd = Int32(0)
        if lane < batch:
            ca = _div_floor_s32(Va + cutlass.max(ta, 1) - 1, T) - fa + 1
        if lane_b < batch:
            cb = _div_floor_s32(Vb + cutlass.max(tb, 1) - 1, T) - fb + 1
        if lane_c < batch:
            cc = _div_floor_s32(Vc + cutlass.max(tc, 1) - 1, T) - fc + 1
        if lane_d < batch:
            cd = _div_floor_s32(Vd + cutlass.max(td, 1) - 1, T) - fd + 1
        # first workspace slot of each request: exclusive prefix sums of the CTA counts
        ica = ca
        icb = cb
        icc = cc
        icd = cd
        for k in cutlass.range_constexpr(5):
            ua = _shfl_up_s32(ica, 1 << k)
            ub = _shfl_up_s32(icb, 1 << k)
            uc = _shfl_up_s32(icc, 1 << k)
            ud = _shfl_up_s32(icd, 1 << k)
            if lane >= (1 << k):
                ica = ica + ua
                icb = icb + ub
                icc = icc + uc
                icd = icd + ud
        sca = _redux_add_s32(ca)
        scb = _redux_add_s32(cb)
        scc = _redux_add_s32(cc)
        Sa = ica - ca
        Sb = icb + sca - cb
        Sc = icc + sca + scb - cc
        Sd = icd + sca + scb + scc - cd
        if lane < batch:
            self._plan_two_store(sPlan, cta, T, lane, Va, ta, fa, ca, Sa, La)
        if lane_b < batch:
            self._plan_two_store(sPlan, cta, T, lane_b, Vb, tb, fb, cb, Sb, Lb)
        if lane_c < batch:
            self._plan_two_store(sPlan, cta, T, lane_c, Vc, tc, fc, cc, Sc, Lc)
        if lane_d < batch:
            self._plan_two_store(sPlan, cta, T, lane_d, Vd, td, fd, cd, Sd, Ld)

    @cute.jit
    def _plan_two_store(
        self,
        sPlan: cute.Tensor,
        cta: Int32,
        T: Int32,
        r: Int32,
        V: Int32,
        t: Int32,
        f: Int32,
        c: Int32,
        s0: Int32,
        L: Int32,
    ):
        """This CTA's slot for request r (if the CTA meets its real tiles): see _plan_two."""
        if f <= cta:
            if cta < f + c:
                lo_t = cutlass.max(V, cta * T) - V
                hi_t = cutlass.min(V + cutlass.max(t, 1), (cta + 1) * T) - V
                n_t = hi_t - lo_t
                n_valid = c
                if t == 0:
                    lo_t = Int32(0)
                    n_t = Int32(0)
                    n_valid = Int32(0)
                base = Int32(8)
                if V <= cta * T:
                    base = Int32(0)
                sPlan[base] = r
                sPlan[base + 1] = cta - f
                sPlan[base + 2] = c
                sPlan[base + 3] = lo_t
                sPlan[base + 4] = n_t
                sPlan[base + 5] = s0
                sPlan[base + 6] = L
                sPlan[base + 7] = n_valid

    @cute.jit
    def _plan_store(
        self, sPlan: cute.Tensor, r: Int32, j: Int32, t: Int32, c: Int32, first: Int32, L: Int32
    ):
        """This CTA's plan words (as in _plan): CTA j of request r's c CTAs takes the j-th of c near-equal blocks of
        its t tiles."""
        base = _div_floor_s32(t, c)
        rem = t - base * c
        n_t = base
        if j < rem:
            n_t = base + 1
        sPlan[0] = r
        sPlan[1] = j
        sPlan[2] = c
        sPlan[3] = j * base + cutlass.min(j, rem)
        sPlan[4] = n_t
        sPlan[5] = first
        sPlan[6] = L
        sPlan[7] = cutlass.min(c, t)

    @cute.jit
    def _combine(
        self,
        tidx: Int32,
        warp: Int32,
        lane: Int32,
        req: Int32,
        split: Int32,
        c_req: Int32,
        n_valid: Int32,
        chunk: Int32,
        chunk_pad: Int32,
        lse_stride: Int32,
        ws_req,
        ws_lse,
        kv_base,
        comb_bar,
        o_base,
        o_sd: Int32,
        o_st: Int32,
        o_sh: Int32,
        l_base,
        l_sd: Int32,
        l_st: Int32,
        l_sh: Int32,
        sScal: cute.Tensor,
        comb_phase=0,
    ):
        """CTA ``split`` of the request's ``c_req`` CTAs finalizes elements [o_lo, o_hi) of the request with the
        request's chunk geometry (run time); its lanes per granule are dispatched to a compile-time variant (_merge).

        gps lanes (a power of two inside one warp) share one 8-element granule g of row r and take partials j = jg,
        jg + gps, ...: one pass for the row max of the partial LSEs, one for the weights 2^(lse_j - M), the weighted
        sum of the FP16 partials and the denominator, then butterflies over the gps lanes (no weights phase, no CTA
        barrier). The granule holding (r, d = 0) also writes the row's LSE. comb_phase: comb_bar's phase (kernel_two's
        second combine after a first one that loaded partials uses 1)."""
        TOT: cutlass.Constexpr = self.rows * HEAD_DIM
        o_lo = split * chunk
        if o_lo < TOT:
            o_hi = cutlass.min(o_lo + chunk, TOT)
            row0 = o_lo // HEAD_DIM
            nrows = (o_hi - 1) // HEAD_DIM - row0 + 1
            n_gran = (o_hi - o_lo) // 8
            s_comb = kv_base  # [n_valid][chunk_pad] FP16
            s_lse = kv_base + COMB_LSE_OFF  # [nrows][lse_stride] FP32
            if n_valid > 0:
                if warp == 0:
                    cute.arch.fence_proxy(
                        "async.global"
                    )  # acquire (generic proxy) -> bulk reads (async proxy)
                    if lane == 0:
                        nb_o = n_valid * (chunk_pad * 2)
                        nb_l = nrows * (lse_stride * 4)
                        cute.arch.mbarrier_arrive_and_expect_tx(comb_bar, nb_o + nb_l)
                        _bulk_g2s(
                            s_comb, ws_req + (split * c_req * chunk_pad) * 2, nb_o, comb_bar.toint()
                        )
                        _bulk_g2s(s_lse, ws_lse + (row0 * lse_stride) * 4, nb_l, comb_bar.toint())
                # spin (no suspend hint): every lane of a warp must be awake for the shuffles below
                landed = cute.arch.mbarrier_try_wait(comb_bar, comb_phase)
                while not landed:
                    landed = cute.arch.mbarrier_try_wait(comb_bar, comb_phase)
            # lanes per granule: the largest power of two <= 32 with gps * (chunk // 8) <= NUM_THREADS
            lg_gps = cutlass.min(
                _bfind_u32(cutlass.max(_div_floor_s32(NUM_THREADS, chunk // 8), 1)), 5
            )
            args = (
                tidx,
                req,
                o_lo,
                row0,
                n_gran,
                n_valid,
                chunk_pad,
                lse_stride,
                s_comb,
                s_lse,
                o_base,
                o_sd,
                o_st,
                o_sh,
                l_base,
                l_sd,
                l_st,
                l_sh,
                sScal,
            )
            if lg_gps == 0:
                self._merge(1, *args)
            else:
                if lg_gps == 1:
                    self._merge(2, *args)
                else:
                    if lg_gps == 2:
                        self._merge(4, *args)
                    else:
                        if lg_gps == 3:
                            self._merge(8, *args)
                        else:
                            if lg_gps == 4:
                                self._merge(16, *args)
                            else:
                                self._merge(32, *args)

    @cute.jit
    def _merge(
        self,
        lanes: cutlass.Constexpr,
        tidx: Int32,
        req: Int32,
        o_lo: Int32,
        row0: Int32,
        n_gran: Int32,
        n_valid: Int32,
        chunk_pad: Int32,
        lse_stride: Int32,
        s_comb,
        s_lse,
        o_base,
        o_sd: Int32,
        o_st: Int32,
        o_sh: Int32,
        l_base,
        l_sd: Int32,
        l_st: Int32,
        l_sh: Int32,
        sScal: cute.Tensor,
    ):
        """Granule merge of the combine for ``lanes`` (compile time) lanes per granule."""
        gps: cutlass.Constexpr = (
            lanes  # annotated: stays a Python constant inside the dynamic loops below
        )
        n_bfly: cutlass.Constexpr = lanes.bit_length() - 1
        per_pass: cutlass.Constexpr = NUM_THREADS // lanes
        jg = tidx % gps
        for gi in cutlass.range((n_gran + per_pass - 1) // per_pass):
            g = gi * per_pass + tidx // gps
            gc = cutlass.min(g, n_gran - 1)
            e0 = o_lo + gc * 8
            orow = e0 // HEAD_DIM
            dd, tok, hh = self._out_coords(orow, req)
            addr = o_base + (dd * o_sd + tok * o_st + hh * o_sh + e0 % HEAD_DIM) * 2
            if n_valid > 0:
                lrow = s_lse + (orow - row0) * (lse_stride * 4)
                mrow = -Float32.inf
                for j in cutlass.range(jg, n_valid, gps):
                    mrow = cute.arch.fmax(mrow, _ld_shared_f32(lrow + j * 4))
                for lg in cutlass.range_constexpr(n_bfly):
                    mrow = cute.arch.fmax(
                        mrow, cute.arch.shuffle_sync_bfly(mrow, offset=gps >> (lg + 1))
                    )
                m_use = mrow
                if mrow == -Float32.inf:
                    m_use = Float32(
                        0.0
                    )  # no visible key in any split: every weight is exp2(-inf) = 0
                den = Float32(0.0)
                a0 = Float32(0.0)
                a1 = Float32(0.0)
                a2 = Float32(0.0)
                a3 = Float32(0.0)
                a4 = Float32(0.0)
                a5 = Float32(0.0)
                a6 = Float32(0.0)
                a7 = Float32(0.0)
                for j in cutlass.range(jg, n_valid, gps):
                    wj = _ex2(_ld_shared_f32(lrow + j * 4) - m_use)
                    den = den + wj
                    u4 = _ld_shared_v4_u32(s_comb + (j * chunk_pad + gc * 8) * 2)
                    f0, f1 = _unpack_f16x2(u4[0])
                    f2, f3 = _unpack_f16x2(u4[1])
                    f4, f5 = _unpack_f16x2(u4[2])
                    f6, f7 = _unpack_f16x2(u4[3])
                    a0 = _fma(wj, f0, a0)
                    a1 = _fma(wj, f1, a1)
                    a2 = _fma(wj, f2, a2)
                    a3 = _fma(wj, f3, a3)
                    a4 = _fma(wj, f4, a4)
                    a5 = _fma(wj, f5, a5)
                    a6 = _fma(wj, f6, a6)
                    a7 = _fma(wj, f7, a7)
                for lg in cutlass.range_constexpr(n_bfly):
                    sh_off = gps >> (lg + 1)
                    den = den + cute.arch.shuffle_sync_bfly(den, offset=sh_off)
                    a0 = a0 + cute.arch.shuffle_sync_bfly(a0, offset=sh_off)
                    a1 = a1 + cute.arch.shuffle_sync_bfly(a1, offset=sh_off)
                    a2 = a2 + cute.arch.shuffle_sync_bfly(a2, offset=sh_off)
                    a3 = a3 + cute.arch.shuffle_sync_bfly(a3, offset=sh_off)
                    a4 = a4 + cute.arch.shuffle_sync_bfly(a4, offset=sh_off)
                    a5 = a5 + cute.arch.shuffle_sync_bfly(a5, offset=sh_off)
                    a6 = a6 + cute.arch.shuffle_sync_bfly(a6, offset=sh_off)
                    a7 = a7 + cute.arch.shuffle_sync_bfly(a7, offset=sh_off)
                if jg == 0:
                    if g < n_gran:
                        scl = Float32(0.0)  # a row without any visible key gives 0, not NaN
                        if den > Float32(0.0):
                            scl = sScal[0] / den
                        w0 = _pack_bf16x2(a0 * scl, a1 * scl)
                        w1 = _pack_bf16x2(a2 * scl, a3 * scl)
                        w2 = _pack_bf16x2(a4 * scl, a5 * scl)
                        w3 = _pack_bf16x2(a6 * scl, a7 * scl)
                        if const_expr(self.o_vec16):
                            _st_global_v4_u32(addr, w0, w1, w2, w3)
                        else:
                            _st_global_u32(addr, w0)
                            _st_global_u32(addr + 4, w1)
                            _st_global_u32(addr + 8, w2)
                            _st_global_u32(addr + 12, w3)
                        if e0 % HEAD_DIM == 0:
                            lse_out = -Float32.inf
                            if den > Float32(0.0):
                                lse_out = mrow + _lg2(den)
                            _st_global_f32(
                                l_base + (dd * l_sd + tok * l_st + hh * l_sh) * 4, lse_out
                            )
            else:
                # this rank sees no key of the request: O = 0, LSE = -inf
                if jg == 0:
                    if g < n_gran:
                        if const_expr(self.o_vec16):
                            _st_global_v4_u32(addr, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                        else:
                            _st_global_u32(addr, Uint32(0))
                            _st_global_u32(addr + 4, Uint32(0))
                            _st_global_u32(addr + 8, Uint32(0))
                            _st_global_u32(addr + 12, Uint32(0))
                        if e0 % HEAD_DIM == 0:
                            _st_global_f32(
                                l_base + (dd * l_sd + tok * l_st + hh * l_sh) * 4, -Float32.inf
                            )

    @cute.kernel
    def kernel(
        self,
        mma_qk: cute.TiledMma,
        mma_pv: cute.TiledMma,
        tma_k: cute.CopyAtom,
        gK_tma: cute.Tensor,
        tma_v: cute.CopyAtom,
        gV_tma: cute.Tensor,
        tma_q: cute.CopyAtom,
        gQ_tma: cute.Tensor,
        mTab: cute.Tensor,
        mSeq: cute.Tensor,
        cp_rank: Int32,
        mBmm1: cute.Tensor,
        bmm1_log2: Float32,
        mBmm2: cute.Tensor,
        bmm2: Float32,
        mO: cute.Tensor,
        o_sd: Int32,
        o_st: Int32,
        o_sh: Int32,
        mLse: cute.Tensor,
        l_sd: Int32,
        l_st: Int32,
        l_sh: Int32,
        mWs: cute.Tensor,
        mCnt: cute.Tensor,
        sQ_l: cute.ComposedLayout,
        sK_l: cute.ComposedLayout,
        sV_l: cute.ComposedLayout,
        sP_l: cute.ComposedLayout,
        sK_tma_l: cute.ComposedLayout,
        sV_tma_l: cute.ComposedLayout,
        sP_mk_l: cute.ComposedLayout,
        q_bytes: cutlass.Constexpr,
    ):
        S: cutlass.Constexpr = KV_STAGES
        R: cutlass.Constexpr = self.rows
        QL: cutlass.Constexpr = self.q_len
        GH: cutlass.Constexpr = self.gh
        CP: cutlass.Constexpr = self.cp
        NS: cutlass.Constexpr = self.n_ctas
        AW: cutlass.Constexpr = self.act_warps
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tidx % 32
        cta, _, _ = (
            cute.arch.block_idx()
        )  # this CTA's request / index in the request come from the plan
        o_base = mO.iterator.toint()
        l_base = mLse.iterator.toint()

        # Shared memory (K ring then V ring first: [kv_base, kv_base + KV_REGION) is one contiguous region)
        smem = SmemAllocator()
        sK = smem.allocate_tensor(FP8, sK_l.outer, 1024, sK_l.inner)
        sV = smem.allocate_tensor(FP8, sV_l.outer, 1024, sV_l.inner)
        sQ = smem.allocate_tensor(FP8, sQ_l.outer, 1024, sQ_l.inner)
        sP = smem.allocate_tensor(FP8, sP_l.outer, 1024, sP_l.inner)
        bars = smem.allocate_array(Int64, 4 * S + 12)
        full_k = bars
        empty_k = bars + S
        full_v = bars + 2 * S
        empty_v = bars + 3 * S
        q_full = bars + 4 * S
        s_full = bars + 4 * S + 1  # [2]
        s_empty = bars + 4 * S + 3  # [2]
        p_full = bars + 4 * S + 5  # [2]
        p_empty = bars + 4 * S + 7  # [2]
        o_done = bars + 4 * S + 9
        o_full = bars + 4 * S + 10
        comb_bar = bars + 4 * S + 11
        tmem_holder = smem.allocate_array(Int32, 4)
        sTab = smem.allocate_tensor(Int32, cute.make_layout((2, TAB_MAX), stride=(TAB_MAX, 1)), 16)
        sTarget = smem.allocate_tensor(Int64, cute.make_layout((2,)), 16)
        sScal = smem.allocate_tensor(Float32, cute.make_layout((4,)), 16)
        sK_tma = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sK_tma_l.inner, dtype=FP8), sK_tma_l.outer
        )
        sV_tma = cute.make_tensor(
            cute.recast_ptr(sV.iterator, sV_tma_l.inner, dtype=FP8), sV_tma_l.outer
        )
        kv_base = sK.iterator.toint()
        sP_base = sP.iterator.toint()
        # this CTA's share of the batch (warp 2, published by the CTA barrier after the TMEM allocation); seq_lens and
        # the block tables are host-uploaded or written before the q all-gather, so they are read before the wait
        sPlan = smem.allocate_tensor(Int32, cute.make_layout((PLAN_WORDS,)), 16)
        if warp == 2:
            if const_expr(self.wide_plan):
                self._plan_wide(mSeq, cp_rank, cta, lane, sPlan)
            else:
                self._plan(mSeq, cp_rank, cta, lane, sPlan)

        if warp == 0:
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_v)
            cpasync.prefetch_descriptor(tma_q)
        if tidx == 0:
            for i in cutlass.range_constexpr(4 * S + 1):
                cute.arch.mbarrier_init(bars + i, 1)
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(s_full + i, 1)
                cute.arch.mbarrier_init(s_empty + i, 32 * AW)
                cute.arch.mbarrier_init(p_full + i, 32 * AW)
                cute.arch.mbarrier_init(p_empty + i, 1)
            cute.arch.mbarrier_init(o_done, 1)
            cute.arch.mbarrier_init(o_full, 1)
            cute.arch.mbarrier_init(comb_bar, 1)
        cute.arch.mbarrier_init_fence()
        if const_expr(AW < 4):
            # P rows of the softmax warps without real rows: zero once (rows 8a..8a+7 = bytes [1024a, 1024a + 1024)
            # of a 16 KB buffer, any swizzle)
            n_zero: cutlass.Constexpr = (
                (M_ROWS - 32 * AW) * TILE_KEYS // 16
            )  # 16-byte words per buffer
            for z in cutlass.range(tidx, 2 * n_zero, NUM_THREADS):
                zb = z // n_zero
                zw = z - zb * n_zero
                _st_shared_v4(
                    sP_base + zb * (M_ROWS * TILE_KEYS) + 32 * AW * TILE_KEYS + zw * 16,
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                )
            cute.arch.fence_proxy("async.shared", space="cta")
        if warp == 1:
            cute.arch.alloc_tmem(TMEM_COLS, tmem_holder)
        cute.arch.sync_threads()
        req = cute.arch.make_warp_uniform(sPlan[0])
        split = cute.arch.make_warp_uniform(sPlan[1])  # index of this CTA among its request's CTAs
        c_req = cute.arch.make_warp_uniform(sPlan[2])  # CTAs of the request
        k_lo = cute.arch.make_warp_uniform(sPlan[3])
        n_t = cute.arch.make_warp_uniform(sPlan[4])
        first = cute.arch.make_warp_uniform(
            sPlan[5]
        )  # the request's first CTA (= its first workspace slot)
        L = cute.arch.make_warp_uniform(sPlan[6])
        n_valid = cute.arch.make_warp_uniform(
            sPlan[7]
        )  # CTAs with work: the first n_valid of the request
        n_pages = (L + PAGE - 1) // PAGE
        n_own = (n_pages - cp_rank + CP - 1) // CP
        prefix = L - QL
        fresh_page = cutlass.max(prefix // PAGE - self.fresh_margin_pages, 0)
        # the request's combine geometry (cp_partial_balanced_geometry with c_req CTAs) and workspace: partials
        # [n_chunks][c_req][chunk_pad] FP16 in the slots of its CTAs, LSEs [R][lse_stride] FP32
        bchunk = 8 * _div_floor_s32(R * HEAD_DIM + 8 * c_req - 1, 8 * c_req)
        bchunk_g = bchunk // 8
        bchunk_pad = bchunk
        if bchunk_g % 2 == 0:
            bchunk_pad = (
                bchunk + 8
            )  # odd granules per row: consecutive partials of one granule in other banks
        blse_stride = 4 * ((c_req + 3) // 4)
        ws_req = mWs.iterator.toint() + first * self.slot_o_bytes
        ws_lse = mWs.iterator.toint() + self.lse_area_off + first * self.lse_cta_bytes
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, tmem_holder)
        tStS = mma_qk.make_fragment_C(mma_qk.partition_shape_C((M_ROWS, TILE_KEYS, 2)))
        tStS = cute.make_tensor(tmem_ptr, tStS.layout)
        s_cols = tcgen05.find_tmem_tensor_col_offset(tStS)
        tOtO = mma_pv.make_fragment_C(mma_pv.partition_shape_C((M_ROWS, HEAD_DIM)))
        tOtO = cute.make_tensor(tmem_ptr + s_cols, tOtO.layout)

        if warp == 0:
            # this CTA's block-table entries -> SMEM: sTab[kv][j] = owned page 2 k_lo + j (a duplicate second slot
            # re-reads the last owned page), TAB_MAX entries at a time (refilled below)
            n_stage = cutlass.min(2 * n_t, TAB_MAX)
            for j in cutlass.range(lane, n_stage, 32):
                oi = cutlass.min(2 * k_lo + j, n_own - 1)
                pg = cp_rank + oi * CP
                sTab[0, j] = mTab[req, 0, pg]
                sTab[1, j] = mTab[req, 1, pg]
            cute.arch.sync_warp()
            gK = cute.tiled_divide(gK_tma, (PAGE, SW_BYTES))  # ((64, 128), blocks, 2)
            gV = cute.tiled_divide(gV_tma, (SW_BYTES, PAGE))  # ((128, 64), 2, blocks)
            tKs, tKg = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), sK_tma, gK)
            tVs, tVg = cpasync.tma_partition(tma_v, 0, cute.make_layout(1), sV_tma, gV)
            waited = Int32(0)
            for i in cutlass.range(n_t):
                if waited == 0:
                    if cp_rank + cutlass.min(2 * (k_lo + i) + 1, n_own - 1) * CP >= fresh_page:
                        # this tile may hold positions written earlier in the step (QKV append, MTP passes)
                        if const_expr(self.use_pdl):
                            _griddep_wait()
                        waited = Int32(1)
                ti = i % (TAB_MAX // 2)
                if ti == 0:
                    if i > 0:
                        # next TAB_MAX table entries (tiles beyond TAB_MAX / 2 per CTA: > ~490k-token contexts)
                        cute.arch.sync_warp()
                        for j in cutlass.range(lane, cutlass.min(2 * (n_t - i), TAB_MAX), 32):
                            oi = cutlass.min(2 * (k_lo + i) + j, n_own - 1)
                            pg = cp_rank + oi * CP
                            sTab[0, j] = mTab[req, 0, pg]
                            sTab[1, j] = mTab[req, 1, pg]
                        cute.arch.sync_warp()
                self._produce(
                    i,
                    sTab[0, 2 * ti],
                    sTab[0, 2 * ti + 1],
                    sTab[1, 2 * ti],
                    sTab[1, 2 * ti + 1],
                    tma_k,
                    tKg,
                    tKs,
                    tma_v,
                    tVg,
                    tVs,
                    full_k,
                    empty_k,
                    full_v,
                    empty_v,
                )
        elif warp == 1:
            tCrQ = mma_qk.make_fragment_A(sQ)
            tCrK = mma_qk.make_fragment_B(sK)
            tCrP = mma_pv.make_fragment_A(sP)
            tCrV = mma_pv.make_fragment_B(sV)
            if n_t > 0:
                cute.arch.mbarrier_wait(q_full, 0)
            for i in cutlass.range(n_t + 1):
                if i < n_t:
                    s = i % S
                    sb = i % 2
                    cute.arch.mbarrier_wait(full_k + s, (i // S) & 1)
                    cute.arch.mbarrier_wait(s_empty + sb, ((i // 2) & 1) ^ 1)
                    for kk in cutlass.range_constexpr(HEAD_DIM // 32):
                        mma_qk.set(tcgen05.Field.ACCUMULATE, kk != 0)
                        cute.gemm(
                            mma_qk,
                            tStS[(None, 0, 0, sb)],
                            tCrQ[(None, 0, kk, 0)],
                            tCrK[(None, 0, kk, s)],
                            tStS[(None, 0, 0, sb)],
                        )
                    with cute.arch.elect_one():
                        tcgen05.commit(s_full + sb)
                        tcgen05.commit(empty_k + s)
                if i >= 1:
                    mma_pv = self._gemm2(
                        i - 1, mma_pv, tOtO, tCrP, tCrV, full_v, p_full, empty_v, p_empty, o_done
                    )
            with cute.arch.elect_one():
                tcgen05.commit(o_full)
        elif warp >= SOFTMAX_WARP0:
            # Softmax (thread st = q row st = TMEM lane st)
            st = tidx - SOFTMAX_WARP0 * 32
            sw = warp - SOFTMAX_WARP0
            if const_expr(self.use_pdl):
                _griddep_wait()  # the scales may come from the preceding kernels
            if sw < AW:
                scale = bmm1_log2
                if const_expr(self.bmm1_from_ptr):
                    scale = mBmm1[1]
                atom_ld = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), Float32)
                tmpl = cute.local_tile(tStS[(None, None), 0, 0, 0], (M_ROWS, 32), (0, 0))
                t2r = tcgen05.make_tmem_copy(atom_ld, tmpl)
                thr_ld = t2r.get_slice(st)
                frag = thr_ld.partition_D(cute.make_identity_tensor((M_ROWS, 32)))
                ob = cute.make_rmem_tensor_like(frag, Float32)  # epilogue chunk (32 columns)
                # 64-column TMEM round trips for the lazy O rescale
                atom_ld64 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(64)), Float32)
                atom_st64 = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(64)), Float32)
                tmpl64 = cute.local_tile(tOtO[(None, None), 0, 0], (M_ROWS, 64), (0, 0))
                t2r64 = tcgen05.make_tmem_copy(atom_ld64, tmpl64)
                r2t64 = tcgen05.make_tmem_copy(atom_st64, tmpl64)
                thr_ld64 = t2r64.get_slice(st)
                thr_st64 = r2t64.get_slice(st)
                ob64 = cute.make_rmem_tensor_like(
                    thr_ld64.partition_D(cute.make_identity_tensor((M_ROWS, 64))), Float32
                )
                x0 = cute.make_rmem_tensor_like(frag, Float32)
                x1 = cute.make_rmem_tensor_like(frag, Float32)
                x2 = cute.make_rmem_tensor_like(frag, Float32)
                x3 = cute.make_rmem_tensor_like(frag, Float32)
                xs = (x0, x1, x2, x3)
                stt = cute.make_rmem_tensor((2,), Float32)
                stt[0] = -Float32.inf
                stt[1] = Float32(0.0)
                lim = prefix + cutlass.min(
                    st // GH, QL - 1
                )  # last visible key position of this row
                bars_sm = (s_full, s_empty, p_full, p_empty, o_done)
                for i in cutlass.range(n_t):
                    # tile k = owned pages 2k (global pg_a) and 2k + 1 (pg_b, or a duplicate of pg_a past the end)
                    oa = 2 * (k_lo + i)
                    pg_a = cp_rank + oa * CP
                    pg_b = pg_a + CP
                    cnt_a = cutlass.min(cutlass.max(lim - pg_a * PAGE + 1, 0), PAGE)
                    cnt_b = cutlass.min(cutlass.max(lim - pg_b * PAGE + 1, 0), PAGE)
                    masked = Int32(0)
                    if (pg_b + 1) * PAGE > prefix:
                        masked = Int32(1)  # holds new tokens or the request tail
                    if oa + 1 >= n_own:
                        masked = Int32(1)
                        cnt_b = Int32(0)  # duplicate second slot
                    if masked != 0:
                        self._softmax_tile(
                            i,
                            xs,
                            ob64,
                            stt,
                            tStS,
                            tOtO,
                            t2r,
                            thr_ld,
                            t2r64,
                            thr_ld64,
                            r2t64,
                            thr_st64,
                            bars_sm,
                            sP_base,
                            sP_mk_l,
                            st,
                            cnt_a,
                            cnt_b,
                            scale,
                            True,
                        )
                    else:
                        self._softmax_tile(
                            i,
                            xs,
                            ob64,
                            stt,
                            tStS,
                            tOtO,
                            t2r,
                            thr_ld,
                            t2r64,
                            thr_ld64,
                            r2t64,
                            thr_st64,
                            bars_sm,
                            sP_base,
                            sP_mk_l,
                            st,
                            cnt_a,
                            cnt_b,
                            scale,
                            False,
                        )
                if n_t > 0:
                    # Normalized FP16 partial row -> staging (the K / V rings are free once o_full fired)
                    l_tot = stt[1]
                    inv_l = Float32(0.0)
                    lse = -Float32.inf
                    if l_tot > Float32(0.0):
                        inv_l = Float32(1.0) / l_tot
                        lse = stt[0] + _lg2(l_tot) - self.p_shift
                    cute.arch.mbarrier_wait(o_full, 0)
                    stage = kv_base + st * STAGE_ROW
                    for h in cutlass.range_constexpr(2):
                        accO = tOtO[(None, None), 0, h]
                        for c in cutlass.range_constexpr(4):
                            cute.copy(
                                t2r,
                                thr_ld.partition_S(cute.local_tile(accO, (M_ROWS, 32), (0, c))),
                                ob,
                            )
                            cute.arch.fence_view_async_tmem_load()
                            for q8 in cutlass.range_constexpr(4):
                                _st_shared_v4(
                                    stage + (h * 128 + c * 32 + 8 * q8) * 2,
                                    _pack_f16x2(ob[8 * q8] * inv_l, ob[8 * q8 + 1] * inv_l),
                                    _pack_f16x2(ob[8 * q8 + 2] * inv_l, ob[8 * q8 + 3] * inv_l),
                                    _pack_f16x2(ob[8 * q8 + 4] * inv_l, ob[8 * q8 + 5] * inv_l),
                                    _pack_f16x2(ob[8 * q8 + 6] * inv_l, ob[8 * q8 + 7] * inv_l),
                                )
                    if st < R:
                        _st_global_f32(ws_lse + (st * blse_stride + split) * 4, lse)
        elif warp == 2:
            if const_expr(self.use_pdl):
                _griddep_wait()  # the epoch counter of the previous launch, bmm2
            # epoch target of the grid barrier: the counter grows by NS per launch and never resets; fewer than NS CTAs
            # of this launch can have arrived yet, so floor(c / NS) is the previous launches' count.
            if lane == 0:
                cval = _ld_relaxed_gpu_u64(mCnt.iterator.toint() + req * 8)
                ns64 = Int64(NS)
                sTarget[0] = (cval // ns64 + 1) * ns64
                b2v = bmm2
                if const_expr(self.bmm2_from_ptr):
                    b2v = mBmm2[0]
                sScal[0] = b2v
        elif warp == 3:
            # q is the all-gather output: requested right after the dependency wait
            if const_expr(self.use_pdl):
                _griddep_wait()
            if n_t > 0:
                gQ = cute.local_tile(
                    gQ_tma,
                    cute.slice_((M_ROWS, TILE_KEYS, HEAD_DIM), (None, 0, None)),
                    (None, None, None),
                )
                tCgQ = mma_qk.get_slice(0).partition_A(gQ)
                tQs, tQg = cpasync.tma_partition(
                    tma_q,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sQ, 0, 3),
                    cute.group_modes(tCgQ, 0, 3),
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(q_full, q_bytes)
                cute.copy(tma_q, tQg[(None, 0, 0, req)], tQs[(None, 0)], tma_bar_ptr=q_full)

        # Partial -> workspace (coalesced 16-byte stores from the staging rows)
        cute.arch.sync_threads()
        if n_t > 0:
            for g in cutlass.range(tidx, R * (HEAD_DIM // 8), NUM_THREADS):
                srow = g // (HEAD_DIM // 8)
                sgr = g - srow * (HEAD_DIM // 8)
                w4 = _ld_shared_v4_u32(kv_base + srow * STAGE_ROW + sgr * 16)
                ch = _div_floor_s32(g, bchunk_g)
                _st_global_v4_u32(
                    ws_req + ((ch * c_req + split) * bchunk_pad + (g - ch * bchunk_g) * 8) * 2,
                    w4[0],
                    w4[1],
                    w4[2],
                    w4[3],
                )
        # Grid barrier over the CTAs of this request (epoch counter)
        cute.arch.sync_threads()  # partial written; epoch target and bmm2 in SMEM
        if tidx == 0:
            cnt = mCnt.iterator.toint() + req * 8
            # red.release is cumulative over the CTA's partial stores ordered before it by the bar.sync above.
            # Every launch adds NS per counter: the request's first CTA also adds the NS - c_req absent CTAs.
            add = Int64(1)
            if split == 0:
                add = Int64(NS + 1) - Int64(c_req)
            _red_release_add_u64(cnt, add)
            target = sTarget[0]
            seen = _ld_acquire_gpu_u64(cnt)
            while seen < target:
                _nanosleep(32)
                seen = _ld_acquire_gpu_u64(cnt)
        cute.arch.sync_threads()

        # Combine: CTA `split` finalizes elements [o_lo, o_hi) of the request
        self._combine(
            tidx,
            warp,
            lane,
            req,
            split,
            c_req,
            n_valid,
            bchunk,
            bchunk_pad,
            blse_stride,
            ws_req,
            ws_lse,
            kv_base,
            comb_bar,
            o_base,
            o_sd,
            o_st,
            o_sh,
            l_base,
            l_sd,
            l_st,
            l_sh,
            sScal,
        )
        # Release the dependents: after a CTA barrier that follows the last global store
        cute.arch.sync_threads()
        if const_expr(self.use_pdl):
            if tidx == 0:
                _griddep_launch()
        if warp == 1:
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, TMEM_COLS)

    # ---------------------------------------------------------------- device kernel, two-slot plan
    @cute.kernel
    def kernel_two(
        self,
        mma_qk: cute.TiledMma,
        mma_pv: cute.TiledMma,
        tma_k: cute.CopyAtom,
        gK_tma: cute.Tensor,
        tma_v: cute.CopyAtom,
        gV_tma: cute.Tensor,
        tma_q: cute.CopyAtom,
        gQ_tma: cute.Tensor,
        mTab: cute.Tensor,
        mSeq: cute.Tensor,
        cp_rank: Int32,
        mBmm1: cute.Tensor,
        bmm1_log2: Float32,
        mBmm2: cute.Tensor,
        bmm2: Float32,
        mO: cute.Tensor,
        o_sd: Int32,
        o_st: Int32,
        o_sh: Int32,
        mLse: cute.Tensor,
        l_sd: Int32,
        l_st: Int32,
        l_sh: Int32,
        mWs: cute.Tensor,
        mCnt: cute.Tensor,
        sQ_l: cute.ComposedLayout,
        sK_l: cute.ComposedLayout,
        sV_l: cute.ComposedLayout,
        sP_l: cute.ComposedLayout,
        sK_tma_l: cute.ComposedLayout,
        sV_tma_l: cute.ComposedLayout,
        sP_mk_l: cute.ComposedLayout,
        q_bytes: cutlass.Constexpr,
    ):
        """``kernel`` on the two-slot plan (_plan_two): the CTA computes up to two requests back to back (slot 0, then
        slot 1) through one K / V ring and one O accumulator, then runs the grid barrier and the combine of each slot's
        request. Ring and barrier phases count the CTA's tiles across both slots; the q buffer (q_empty: MMA -> q
        loader, after slot 0's last GEMM1) and O (o_empty: softmax -> MMA) change hands between the slots. The K / V
        ring the one-slot kernel stages the FP16 partials through holds slot 1's tiles by then, so the softmax rows
        store theirs straight to the workspace."""
        S: cutlass.Constexpr = KV_STAGES
        R: cutlass.Constexpr = self.rows
        QL: cutlass.Constexpr = self.q_len
        GH: cutlass.Constexpr = self.gh
        CP: cutlass.Constexpr = self.cp
        NS: cutlass.Constexpr = self.n_ctas
        AW: cutlass.Constexpr = self.act_warps
        TOT: cutlass.Constexpr = self.rows * HEAD_DIM
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tidx % 32
        cta, _, _ = cute.arch.block_idx()
        o_base = mO.iterator.toint()
        l_base = mLse.iterator.toint()

        # ---- shared memory (as ``kernel``, + q_empty / o_empty and a second plan) ----
        smem = SmemAllocator()
        sK = smem.allocate_tensor(FP8, sK_l.outer, 1024, sK_l.inner)
        sV = smem.allocate_tensor(FP8, sV_l.outer, 1024, sV_l.inner)
        sQ = smem.allocate_tensor(FP8, sQ_l.outer, 1024, sQ_l.inner)
        sP = smem.allocate_tensor(FP8, sP_l.outer, 1024, sP_l.inner)
        bars = smem.allocate_array(Int64, 4 * S + 14)
        full_k = bars
        empty_k = bars + S
        full_v = bars + 2 * S
        empty_v = bars + 3 * S
        q_full = bars + 4 * S
        s_full = bars + 4 * S + 1  # [2]
        s_empty = bars + 4 * S + 3  # [2]
        p_full = bars + 4 * S + 5  # [2]
        p_empty = bars + 4 * S + 7  # [2]
        o_done = bars + 4 * S + 9
        o_full = bars + 4 * S + 10
        comb_bar = bars + 4 * S + 11
        q_empty = bars + 4 * S + 12  # slot 0's GEMM1s done: q of slot 1 may overwrite sQ
        o_empty = (
            bars + 4 * S + 13
        )  # slot 0's O read by the softmax rows: slot 1's first GEMM2 may overwrite it
        tmem_holder = smem.allocate_array(Int32, 4)
        sTab = smem.allocate_tensor(Int32, cute.make_layout((2, TAB_MAX), stride=(TAB_MAX, 1)), 16)
        sTarget = smem.allocate_tensor(Int64, cute.make_layout((2,)), 16)
        sScal = smem.allocate_tensor(Float32, cute.make_layout((4,)), 16)
        sK_tma = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sK_tma_l.inner, dtype=FP8), sK_tma_l.outer
        )
        sV_tma = cute.make_tensor(
            cute.recast_ptr(sV.iterator, sV_tma_l.inner, dtype=FP8), sV_tma_l.outer
        )
        kv_base = sK.iterator.toint()
        sP_base = sP.iterator.toint()
        sPlan = smem.allocate_tensor(Int32, cute.make_layout((2 * PLAN_WORDS,)), 16)
        if warp == 2:
            self._plan_two(mSeq, cp_rank, cta, lane, sPlan)

        if warp == 0:
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_v)
            cpasync.prefetch_descriptor(tma_q)
        if tidx == 0:
            for i in cutlass.range_constexpr(4 * S + 1):
                cute.arch.mbarrier_init(bars + i, 1)
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(s_full + i, 1)
                cute.arch.mbarrier_init(s_empty + i, 32 * AW)
                cute.arch.mbarrier_init(p_full + i, 32 * AW)
                cute.arch.mbarrier_init(p_empty + i, 1)
            cute.arch.mbarrier_init(o_done, 1)
            cute.arch.mbarrier_init(o_full, 1)
            cute.arch.mbarrier_init(comb_bar, 1)
            cute.arch.mbarrier_init(q_empty, 1)
            cute.arch.mbarrier_init(o_empty, 32 * AW)
        cute.arch.mbarrier_init_fence()
        if const_expr(AW < 4):
            n_zero: cutlass.Constexpr = (
                (M_ROWS - 32 * AW) * TILE_KEYS // 16
            )  # 16-byte words per buffer
            for z in cutlass.range(tidx, 2 * n_zero, NUM_THREADS):
                zb = z // n_zero
                zw = z - zb * n_zero
                _st_shared_v4(
                    sP_base + zb * (M_ROWS * TILE_KEYS) + 32 * AW * TILE_KEYS + zw * 16,
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                )
            cute.arch.fence_proxy("async.shared", space="cta")
        if warp == 1:
            cute.arch.alloc_tmem(TMEM_COLS, tmem_holder)
        cute.arch.sync_threads()

        # ---- the two slots (request -1: no slot; tiles 0 then) ----
        req0 = cute.arch.make_warp_uniform(sPlan[0])
        split0 = cute.arch.make_warp_uniform(sPlan[1])
        c0 = cute.arch.make_warp_uniform(sPlan[2])
        klo0 = cute.arch.make_warp_uniform(sPlan[3])
        nt0 = cute.arch.make_warp_uniform(sPlan[4])
        first0 = cute.arch.make_warp_uniform(sPlan[5])
        L0 = cute.arch.make_warp_uniform(sPlan[6])
        nv0 = cute.arch.make_warp_uniform(sPlan[7])
        req1 = cute.arch.make_warp_uniform(sPlan[8])
        split1 = cute.arch.make_warp_uniform(sPlan[9])
        c1 = cute.arch.make_warp_uniform(sPlan[10])
        klo1 = cute.arch.make_warp_uniform(sPlan[11])
        nt1 = cute.arch.make_warp_uniform(sPlan[12])
        first1 = cute.arch.make_warp_uniform(sPlan[13])
        L1 = cute.arch.make_warp_uniform(sPlan[14])
        nv1 = cute.arch.make_warp_uniform(sPlan[15])
        a0 = Int32(
            0
        )  # slot 0 has tiles (loads q, runs the MMA / softmax pipeline, writes a partial)
        if nt0 > 0:
            a0 = Int32(1)
        own0 = (((L0 + PAGE - 1) // PAGE) - cp_rank + CP - 1) // CP
        own1 = (((L1 + PAGE - 1) // PAGE) - cp_rank + CP - 1) // CP
        prefix0 = L0 - QL
        prefix1 = L1 - QL
        fresh0 = cutlass.max(prefix0 // PAGE - self.fresh_margin_pages, 0)
        fresh1 = cutlass.max(prefix1 // PAGE - self.fresh_margin_pages, 0)
        # each request's combine geometry (cp_partial_geometry with its CTA count) and workspace slots
        cc0 = cutlass.max(c0, 1)
        cc1 = cutlass.max(c1, 1)
        bch0 = 8 * _div_floor_s32(R * HEAD_DIM + 8 * cc0 - 1, 8 * cc0)
        bch1 = 8 * _div_floor_s32(R * HEAD_DIM + 8 * cc1 - 1, 8 * cc1)
        bpad0 = bch0
        if (bch0 // 8) % 2 == 0:
            bpad0 = bch0 + 8
        bpad1 = bch1
        if (bch1 // 8) % 2 == 0:
            bpad1 = bch1 + 8
        blse0 = 4 * ((cc0 + 3) // 4)
        blse1 = 4 * ((cc1 + 3) // 4)
        ws_req0 = mWs.iterator.toint() + first0 * self.slot_o_bytes
        ws_req1 = mWs.iterator.toint() + first1 * self.slot_o_bytes
        ws_lse0 = mWs.iterator.toint() + self.lse_area_off + first0 * self.lse_cta_bytes
        ws_lse1 = mWs.iterator.toint() + self.lse_area_off + first1 * self.lse_cta_bytes
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, tmem_holder)
        tStS = mma_qk.make_fragment_C(mma_qk.partition_shape_C((M_ROWS, TILE_KEYS, 2)))
        tStS = cute.make_tensor(tmem_ptr, tStS.layout)
        s_cols = tcgen05.find_tmem_tensor_col_offset(tStS)
        tOtO = mma_pv.make_fragment_C(mma_pv.partition_shape_C((M_ROWS, HEAD_DIM)))
        tOtO = cute.make_tensor(tmem_ptr + s_cols, tOtO.layout)

        if warp == 0:
            # ======================= TMA producer (tile index i of the CTA runs on across the slots) ============
            gK = cute.tiled_divide(gK_tma, (PAGE, SW_BYTES))
            gV = cute.tiled_divide(gV_tma, (SW_BYTES, PAGE))
            tKs, tKg = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), sK_tma, gK)
            tVs, tVg = cpasync.tma_partition(tma_v, 0, cute.make_layout(1), sV_tma, gV)
            waited = Int32(0)
            for k in cutlass.range_constexpr(2):
                rq = (req0, req1)[k]
                klo = (klo0, klo1)[k]
                nt = (nt0, nt1)[k]
                n_own = (own0, own1)[k]
                fresh = (fresh0, fresh1)[k]
                gb = (Int32(0), nt0)[k]  # tiles of the slots before this one
                if nt > 0:
                    cute.arch.sync_warp()  # every lane is past the previous slot's table reads
                    for j in cutlass.range(lane, cutlass.min(2 * nt, TAB_MAX), 32):
                        oi = cutlass.min(2 * klo + j, n_own - 1)
                        pg = cp_rank + oi * CP
                        sTab[0, j] = mTab[rq, 0, pg]
                        sTab[1, j] = mTab[rq, 1, pg]
                    cute.arch.sync_warp()
                    for i in cutlass.range(nt):
                        if waited == 0:
                            if cp_rank + cutlass.min(2 * (klo + i) + 1, n_own - 1) * CP >= fresh:
                                if const_expr(self.use_pdl):
                                    _griddep_wait()
                                waited = Int32(1)
                        ti = i % (TAB_MAX // 2)
                        if ti == 0:
                            if i > 0:
                                cute.arch.sync_warp()
                                for j in cutlass.range(
                                    lane, cutlass.min(2 * (nt - i), TAB_MAX), 32
                                ):
                                    oi = cutlass.min(2 * (klo + i) + j, n_own - 1)
                                    pg = cp_rank + oi * CP
                                    sTab[0, j] = mTab[rq, 0, pg]
                                    sTab[1, j] = mTab[rq, 1, pg]
                                cute.arch.sync_warp()
                        self._produce(
                            gb + i,
                            sTab[0, 2 * ti],
                            sTab[0, 2 * ti + 1],
                            sTab[1, 2 * ti],
                            sTab[1, 2 * ti + 1],
                            tma_k,
                            tKg,
                            tKs,
                            tma_v,
                            tVg,
                            tVs,
                            full_k,
                            empty_k,
                            full_v,
                            empty_v,
                        )
        elif warp == 1:
            # ======================= MMA issuer =======================
            tCrQ = mma_qk.make_fragment_A(sQ)
            tCrK = mma_qk.make_fragment_B(sK)
            tCrP = mma_pv.make_fragment_A(sP)
            tCrV = mma_pv.make_fragment_B(sV)
            for k in cutlass.range_constexpr(2):
                nt = (nt0, nt1)[k]
                gb = (Int32(0), nt0)[k]
                qph = (Int32(0), a0)[k]  # q loads before this slot's
                if nt > 0:
                    cute.arch.mbarrier_wait(q_full, qph)
                    for i in cutlass.range(nt + 1):
                        if i < nt:
                            g = gb + i
                            s = g % S
                            sb = g % 2
                            cute.arch.mbarrier_wait(full_k + s, (g // S) & 1)
                            cute.arch.mbarrier_wait(s_empty + sb, ((g // 2) & 1) ^ 1)
                            for kk in cutlass.range_constexpr(HEAD_DIM // 32):
                                mma_qk.set(tcgen05.Field.ACCUMULATE, kk != 0)
                                cute.gemm(
                                    mma_qk,
                                    tStS[(None, 0, 0, sb)],
                                    tCrQ[(None, 0, kk, 0)],
                                    tCrK[(None, 0, kk, s)],
                                    tStS[(None, 0, 0, sb)],
                                )
                            with cute.arch.elect_one():
                                tcgen05.commit(s_full + sb)
                                tcgen05.commit(empty_k + s)
                                if const_expr(k == 0):
                                    if i == nt - 1:
                                        tcgen05.commit(
                                            q_empty
                                        )  # slot 0's last GEMM1: sQ may take slot 1's q
                        if i >= 1:
                            if const_expr(k == 1):
                                if i == 1:
                                    if a0 != 0:
                                        cute.arch.mbarrier_wait(
                                            o_empty, 0
                                        )  # slot 0's O has been read out
                            mma_pv = self._gemm2(
                                gb + i - 1,
                                mma_pv,
                                tOtO,
                                tCrP,
                                tCrV,
                                full_v,
                                p_full,
                                empty_v,
                                p_empty,
                                o_done,
                                j_loc=i - 1,
                            )
                    with cute.arch.elect_one():
                        tcgen05.commit(o_full)
        elif warp >= SOFTMAX_WARP0:
            # ======================= softmax (thread st = q row st = TMEM lane st) =======================
            st = tidx - SOFTMAX_WARP0 * 32
            sw = warp - SOFTMAX_WARP0
            if const_expr(self.use_pdl):
                _griddep_wait()  # the scales may come from the preceding kernels
            if sw < AW:
                scale = bmm1_log2
                if const_expr(self.bmm1_from_ptr):
                    scale = mBmm1[1]
                atom_ld = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), Float32)
                tmpl = cute.local_tile(tStS[(None, None), 0, 0, 0], (M_ROWS, 32), (0, 0))
                t2r = tcgen05.make_tmem_copy(atom_ld, tmpl)
                thr_ld = t2r.get_slice(st)
                frag = thr_ld.partition_D(cute.make_identity_tensor((M_ROWS, 32)))
                ob = cute.make_rmem_tensor_like(frag, Float32)
                atom_ld64 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(64)), Float32)
                atom_st64 = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(64)), Float32)
                tmpl64 = cute.local_tile(tOtO[(None, None), 0, 0], (M_ROWS, 64), (0, 0))
                t2r64 = tcgen05.make_tmem_copy(atom_ld64, tmpl64)
                r2t64 = tcgen05.make_tmem_copy(atom_st64, tmpl64)
                thr_ld64 = t2r64.get_slice(st)
                thr_st64 = r2t64.get_slice(st)
                ob64 = cute.make_rmem_tensor_like(
                    thr_ld64.partition_D(cute.make_identity_tensor((M_ROWS, 64))), Float32
                )
                x0 = cute.make_rmem_tensor_like(frag, Float32)
                x1 = cute.make_rmem_tensor_like(frag, Float32)
                x2 = cute.make_rmem_tensor_like(frag, Float32)
                x3 = cute.make_rmem_tensor_like(frag, Float32)
                xs = (x0, x1, x2, x3)
                stt = cute.make_rmem_tensor((2,), Float32)
                bars_sm = (s_full, s_empty, p_full, p_empty, o_done)
                for k in cutlass.range_constexpr(2):
                    nt = (nt0, nt1)[k]
                    klo = (klo0, klo1)[k]
                    n_own = (own0, own1)[k]
                    prefix = (prefix0, prefix1)[k]
                    gb = (Int32(0), nt0)[k]
                    oph = (Int32(0), a0)[k]  # o_full completions before this slot's
                    split_k = (split0, split1)[k]
                    c_k = (cc0, cc1)[k]
                    bch = (bch0, bch1)[k]
                    bpad = (bpad0, bpad1)[k]
                    blse = (blse0, blse1)[k]
                    ws_req = (ws_req0, ws_req1)[k]
                    ws_lse = (ws_lse0, ws_lse1)[k]
                    if nt > 0:
                        stt[0] = -Float32.inf
                        stt[1] = Float32(0.0)
                        lim = prefix + cutlass.min(
                            st // GH, QL - 1
                        )  # last visible key position of this row
                        for i in cutlass.range(nt):
                            oa = 2 * (klo + i)
                            pg_a = cp_rank + oa * CP
                            pg_b = pg_a + CP
                            cnt_a = cutlass.min(cutlass.max(lim - pg_a * PAGE + 1, 0), PAGE)
                            cnt_b = cutlass.min(cutlass.max(lim - pg_b * PAGE + 1, 0), PAGE)
                            masked = Int32(0)
                            if (pg_b + 1) * PAGE > prefix:
                                masked = Int32(1)  # holds new tokens or the request tail
                            if oa + 1 >= n_own:
                                masked = Int32(1)
                                cnt_b = Int32(0)  # duplicate second slot
                            if masked != 0:
                                self._softmax_tile(
                                    gb + i,
                                    xs,
                                    ob64,
                                    stt,
                                    tStS,
                                    tOtO,
                                    t2r,
                                    thr_ld,
                                    t2r64,
                                    thr_ld64,
                                    r2t64,
                                    thr_st64,
                                    bars_sm,
                                    sP_base,
                                    sP_mk_l,
                                    st,
                                    cnt_a,
                                    cnt_b,
                                    scale,
                                    True,
                                    i_loc=i,
                                )
                            else:
                                self._softmax_tile(
                                    gb + i,
                                    xs,
                                    ob64,
                                    stt,
                                    tStS,
                                    tOtO,
                                    t2r,
                                    thr_ld,
                                    t2r64,
                                    thr_ld64,
                                    r2t64,
                                    thr_st64,
                                    bars_sm,
                                    sP_base,
                                    sP_mk_l,
                                    st,
                                    cnt_a,
                                    cnt_b,
                                    scale,
                                    False,
                                    i_loc=i,
                                )
                        # ---- normalized FP16 partial row -> the workspace (chunk-major, see _combine) ----
                        l_tot = stt[1]
                        inv_l = Float32(0.0)
                        lse = -Float32.inf
                        if l_tot > Float32(0.0):
                            inv_l = Float32(1.0) / l_tot
                            lse = stt[0] + _lg2(l_tot) - self.p_shift
                        cute.arch.mbarrier_wait(o_full, oph)
                        for h in cutlass.range_constexpr(2):
                            accO = tOtO[(None, None), 0, h]
                            for c in cutlass.range_constexpr(4):
                                cute.copy(
                                    t2r,
                                    thr_ld.partition_S(cute.local_tile(accO, (M_ROWS, 32), (0, c))),
                                    ob,
                                )
                                cute.arch.fence_view_async_tmem_load()
                                if st < R:
                                    for q8 in cutlass.range_constexpr(4):
                                        e = st * HEAD_DIM + (h * 128 + c * 32 + 8 * q8)
                                        ch = _div_floor_s32(e, bch)
                                        _st_global_v4_u32(
                                            ws_req
                                            + ((ch * c_k + split_k) * bpad + e - ch * bch) * 2,
                                            _pack_f16x2(ob[8 * q8] * inv_l, ob[8 * q8 + 1] * inv_l),
                                            _pack_f16x2(
                                                ob[8 * q8 + 2] * inv_l, ob[8 * q8 + 3] * inv_l
                                            ),
                                            _pack_f16x2(
                                                ob[8 * q8 + 4] * inv_l, ob[8 * q8 + 5] * inv_l
                                            ),
                                            _pack_f16x2(
                                                ob[8 * q8 + 6] * inv_l, ob[8 * q8 + 7] * inv_l
                                            ),
                                        )
                        if st < R:
                            _st_global_f32(ws_lse + (st * blse + split_k) * 4, lse)
                        if const_expr(k == 0):
                            cute.arch.mbarrier_arrive(o_empty)  # this row's share of O is read out
        elif warp == 2:
            if const_expr(self.use_pdl):
                _griddep_wait()  # the epoch counters of the previous launch, bmm2
            if lane == 0:
                ns64 = Int64(NS)
                if req0 >= 0:
                    cv0 = _ld_relaxed_gpu_u64(mCnt.iterator.toint() + req0 * 8)
                    sTarget[0] = (cv0 // ns64 + 1) * ns64
                if req1 >= 0:
                    cv1 = _ld_relaxed_gpu_u64(mCnt.iterator.toint() + req1 * 8)
                    sTarget[1] = (cv1 // ns64 + 1) * ns64
                b2v = bmm2
                if const_expr(self.bmm2_from_ptr):
                    b2v = mBmm2[0]
                sScal[0] = b2v
        elif warp == 3:
            # q of each slot's request after the dependency wait; slot 1's once slot 0's GEMM1s are done with sQ
            if const_expr(self.use_pdl):
                _griddep_wait()
            if nt0 + nt1 > 0:
                gQ = cute.local_tile(
                    gQ_tma,
                    cute.slice_((M_ROWS, TILE_KEYS, HEAD_DIM), (None, 0, None)),
                    (None, None, None),
                )
                tCgQ = mma_qk.get_slice(0).partition_A(gQ)
                tQs, tQg = cpasync.tma_partition(
                    tma_q,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sQ, 0, 3),
                    cute.group_modes(tCgQ, 0, 3),
                )
                if nt0 > 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(q_full, q_bytes)
                    cute.copy(tma_q, tQg[(None, 0, 0, req0)], tQs[(None, 0)], tma_bar_ptr=q_full)
                if nt1 > 0:
                    if nt0 > 0:
                        cute.arch.mbarrier_wait(q_empty, 0)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(q_full, q_bytes)
                    cute.copy(tma_q, tQg[(None, 0, 0, req1)], tQs[(None, 0)], tma_bar_ptr=q_full)

        # ======================= grid barrier and combine of each slot's request =======================
        cute.arch.sync_threads()  # both partials written (softmax rows); epoch targets and bmm2 in SMEM
        if tidx == 0:
            # red.release is cumulative over the CTA's partial stores ordered before it by the bar.sync above; every
            # launch adds NS per counter: the request's first CTA also adds the NS - c absent CTAs
            if req0 >= 0:
                add0 = Int64(1)
                if split0 == 0:
                    add0 = Int64(NS + 1) - Int64(c0)
                _red_release_add_u64(mCnt.iterator.toint() + req0 * 8, add0)
            if req1 >= 0:
                add1 = Int64(1)
                if split1 == 0:
                    add1 = Int64(NS + 1) - Int64(c1)
                _red_release_add_u64(mCnt.iterator.toint() + req1 * 8, add1)
        cph1 = Int32(0)  # comb_bar phase of slot 1: 1 when slot 0's combine loaded partials
        if req0 >= 0:
            if nv0 > 0:
                if split0 * bch0 < TOT:
                    cph1 = Int32(1)
        for k in cutlass.range_constexpr(2):
            rq = (req0, req1)[k]
            if rq >= 0:
                if tidx == 0:
                    cnt = mCnt.iterator.toint() + rq * 8
                    target = sTarget[k]
                    seen = _ld_acquire_gpu_u64(cnt)
                    while seen < target:
                        _nanosleep(32)
                        seen = _ld_acquire_gpu_u64(cnt)
                cute.arch.sync_threads()
                self._combine(
                    tidx,
                    warp,
                    lane,
                    rq,
                    (split0, split1)[k],
                    (cc0, cc1)[k],
                    (nv0, nv1)[k],
                    (bch0, bch1)[k],
                    (bpad0, bpad1)[k],
                    (blse0, blse1)[k],
                    (ws_req0, ws_req1)[k],
                    (ws_lse0, ws_lse1)[k],
                    kv_base,
                    comb_bar,
                    o_base,
                    o_sd,
                    o_st,
                    o_sh,
                    l_base,
                    l_sd,
                    l_st,
                    l_sh,
                    sScal,
                    comb_phase=(Int32(0), cph1)[k],
                )
        # ======================= release the dependents: after a CTA barrier that follows the last global store =====
        cute.arch.sync_threads()
        if const_expr(self.use_pdl):
            if tidx == 0:
                _griddep_launch()
        if warp == 1:
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, TMEM_COLS)
