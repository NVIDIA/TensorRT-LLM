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
"""Kimi K3 KDA fused projection y = x W^T (TP16 rank slice: W bf16 [3208, 7168], x bf16 [T <= 8, 7168]) streamed in
three priority phases, so that a consumer of the projection can start on the rows it needs first.

Rows of W (= columns of y): q [0, 768) | k [768, 1536) | v [1536, 2304) | og [2304, 3072) | f_a [3072, 3200) |
b [3200, 3206) | pad [3206, 3208).

Grid: 26 clusters of 4 CTAs, 256 threads. Cluster k, rank r:
  phase 1 (q | k | f_a, 26 tiles of 64 rows): tile k, K chunks [28 r, 28 r + 28) (split-K 4 over the cluster);
  phase 2 (v, 12 tiles, then b): cluster k < 24: v tile k // 2, K half k % 2, chunks 56 (k % 2) + [14 r, 14 r + 14)
    (split-K 8 = two clusters per tile); clusters 24 and 25: the b tile (8 rows), K half k - 24, likewise;
  phase 3 (og, 12 tiles): clusters k < 24 as phase 2; clusters 24 and 25 have no phase 3.
Warps: 0 weight TMA (issued from launch, before the grid dependency, EVICT_FIRST), 1 activation TMA (after it),
2 TMEM allocation and the M = 64, N = 8 MMAs (weight = A, activation = B; each phase accumulates into its own 8
TMEM columns so a phase's epilogue runs beside the next phase's MMAs), 3 idle, 4-7 epilogue: warp 4 + w loads TMEM
rows [16 w, 16 w + 16) of the tile (16x256b), which rank w of the cluster owns. The other ranks st.async them into
the owner's mailbox; the owner adds the four partials in rank order and publishes them Lamport-style (the data words
are the flags: no fence, no counter, so nothing waits for the stores to be acknowledged under the weight stream):
  phase 1: bf16 bits into ``p1`` [3 buffers][8 tokens][1664 rows (q | k | f_a)];
  phases 2 and 3: the cluster's fp32 partial bits into ``part`` [3 buffers][region (v, og, b)][half][8][768]; the
    projection is bf16(half 0 + half 1), the consumer's job.
Every word of a buffer holds the sentinel (all ones: a NaN no finite GEMV produces; a computed all-ones word is
stored as the canonical NaN instead) until its producer writes it, so a consumer polls the words it needs until none
is the sentinel. A launch reads e = ``epoch[cta]`` (the CTA's buffer index, its launch count mod 3) after the grid
dependency, writes buffer e and stores the sentinel into the same words of buffer (e + 1) % 3, which the next launch
writes and the launch before last read, then writes (e + 1) % 3 back at its end. The index stays in 0..2: a raw
launch count would turn negative after 2^31 launches and its signed remainder would index before the buffers.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.base_dsl.array
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

try:
    from cutlass.memory.smem import SmemAllocator
except ImportError:  # older DSL layout
    from cutlass.utils import SmemAllocator

from ..k3_kda_verify.k3_kda_verify_kernel import (
    _bf16,
    _butterfly,
    _st_async_f32,
    _store8,
    _test_wait_cluster,
)

K_IN = 7168
HK = 768  # 6 local heads x 128
PROJ_ROWS = 4 * HK + 128 + 8  # 3208
V_ROW = 2 * HK
OG_ROW = 3 * HK
FA_ROW = 4 * HK
B_ROW = FA_ROW + 128
TILE = 64  # weight rows per tile = the MMA's M
MMA_N = 8
MMA_K = 16
BOX_K = 64  # bf16 per 128-byte swizzled row
BOX_CH = 2  # 64-column chunks per weight box
BOX_ELEMS = TILE * BOX_K * BOX_CH  # 16 KB
X_ELEMS = MMA_N * BOX_K * BOX_CH  # 2 KB
STREAM_STAGES = 10  # weight ring depth of the projection alone (B1)
FUSED_STAGES = 12  # B2's stream CTAs: the verify role's buffers alias the same bytes (_SmemCarver)
P3_WINDOW = 3  # phase-3 weight boxes in flight per stream CTA
CLUSTER = 4
FUSED_RAW_BYTES = (
    FUSED_STAGES * (BOX_ELEMS + X_ELEMS) * 2 + 3 * CLUSTER * 32 * 4 * 4
)  # the stream role's carve; the verify role's fits inside
STREAM_CLUSTERS = 26
P1_BOXES = K_IN // (BOX_K * BOX_CH) // CLUSTER  # 14
P23_BOXES = P1_BOXES // 2  # 7
THREADS = 256
TMEM_COLS = 32
PART_ROWS = HK
P1_ROWS = 2 * HK + 128  # q | k | f_a
P1_BUF = 8 * P1_ROWS
PART_BUF = 3 * 2 * 8 * PART_ROWS
BUFFERS = 3
SENTINEL = -1  # all-ones bits
CANON_NAN16 = 0x7FC0
CANON_NAN32 = 0x7FC00000
EVICT_FIRST = 0x12F0000000000000  # createpolicy.fractional.L2::evict_first, fraction 1.0

# KDA verify role (stage B2): clusters 26-31 = local heads 0-5, cluster rank = V quarter (32 V rows, 4 per warp).
HD = 128  # key and value head dim
H_LOCAL = HK // HD
CONV_W = 4
NT = 8  # verify tokens: 1 golden + 7 drafts (one request)
NUM_SPEC = NT - 1
S_COLS = CONV_W - 1 + NUM_SPEC  # conv-cache columns
ROWS_U = CONV_W - 1 + NT  # raw conv inputs by position: 3 before token 0, then the tokens
V_CTA = HD // CLUSTER  # 32
REC_ROWS = V_CTA // 8  # V rows per warp in the recurrence
VEC = HD // 32  # keys per lane
# The recurrence's registers (one rmem array, static indices): the state [REC_ROWS][VEC], then the previous token's
# per-lane q-dot partials [REC_ROWS], then the current token's operands: v rows [REC_ROWS], q, decay * k, beta * k and
# decay [VEC] each.
R_Q = REC_ROWS * VEC
R_OP = R_Q + REC_ROWS
REC_REGS = R_OP + REC_ROWS + 4 * VEC
# The drafts' records, in each slot's per-token state region (fp32 words from the slot's start) in place of their
# full states: the row innovations vn [NUM_SPEC][H][V], then beta * k and the decay [NUM_SPEC][H][K]. The next launch
# rebuilds the state after the accepted drafts from the pool (the golden token's state) with the update's own
# arithmetic, S = fma(decay, S, vn * (beta * k)) draft by draft, so it is bit-identical to the one the drafts reached.
# k3_kda_verify writes and reads the same records.
CT_VN = 0
CT_WB = NUM_SPEC * H_LOCAL * HD
CT_WD = CT_WB + NUM_SPEC * H_LOCAL * HD
CT_SLOT = NUM_SPEC * H_LOCAL * HD * HD  # a slot's per-token state region
REC_CTA = (
    V_CTA + 2 * HD
)  # one draft's records a verify CTA replays: vn of its 32 rows, beta * k and decay of 128 keys
PEND_WORDS = (
    4  # pending counts a warp loads beside the slot index: pools up to 32 * PEND_WORDS slots
)
CP_CG = cutlass.base_dsl.array.LoadCacheModifier.CG  # cp.async 16 B through L2 only
HEAD_CLUSTERS = H_LOCAL
FB_TMEM_COLS = 32

# Shared-memory descriptor units (16 bytes).
W_STAGE_U = (BOX_ELEMS * 2) >> 4
W_CHUNK_U = (TILE * BOX_K * 2) >> 4
X_STAGE_U = (X_ELEMS * 2) >> 4
X_CHUNK_U = (MMA_N * BOX_K * 2) >> 4
K_STEP_U = (MMA_K * 2) >> 4


@dsl_user_op
def _mapa_u32(smem_ptr, peer, *, loc=None, ip=None):
    """The shared::cluster address of this CTA's shared-memory location in cluster CTA ``peer``."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [smem_ptr.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(peer).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;", "=r,r,r", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _st_async_v4_f32(dst, a, b, c, d, mbar, *, loc=None, ip=None):
    """st.async of four fp32 to a shared::cluster address, completing ``mbar`` (shared::cluster) by 16 bytes."""
    _llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip), cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip), cutlass.Float32(c).ir_value(loc=loc, ip=ip),
         cutlass.Float32(d).ir_value(loc=loc, ip=ip), cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [$0], {$1, $2, $3, $4}, [$5];", "r,f,f,f,f,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _bf16_bits(x, *, loc=None, ip=None):
    """The bf16 rounding (round to nearest even) of an fp32, as its 16 bits in the low half of an int32."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.u32.u16 $0, h; }", "=r,f", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _f32_bits(x, *, loc=None, ip=None):
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=r,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _st_b16(ptr, bits, *, loc=None, ip=None):
    """st.global.b16 of the low 16 bits of an int32 at a global address (int64)."""
    _llvm.inline_asm(
        None, [cutlass.Int64(ptr).ir_value(loc=loc, ip=ip), cutlass.Int32(bits).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 h; cvt.u16.u32 h, $1; st.global.b16 [$0], h; }", "l,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _st_f32(ptr, value, *, loc=None, ip=None):
    """st.global.f32 of an fp32 at a global address (int64)."""
    _llvm.inline_asm(
        None, [cutlass.Int64(ptr).ir_value(loc=loc, ip=ip), cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;", "l,f", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _st_f32_if(ptr, value, pred, *, loc=None, ip=None):
    """st.global.f32 of an fp32 at a global address (int64) where pred (int32) is non-zero."""
    _llvm.inline_asm(
        None, [cutlass.Int64(ptr).ir_value(loc=loc, ip=ip), cutlass.Float32(value).ir_value(loc=loc, ip=ip),
               cutlass.Int32(pred).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.s32 p, $2, 0; @p st.global.f32 [$0], $1; }", "l,f,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


def _lamport16(bits):
    """bf16 bits with the sentinel pattern replaced by the canonical NaN."""
    return cutlass.select_(bits == cutlass.Int32(0xFFFF), cutlass.Int32(CANON_NAN16), bits)


def _lamport32(bits):
    return cutlass.select_(bits == cutlass.Int32(SENTINEL), cutlass.Int32(CANON_NAN32), bits)


@cute.jit
def _publish(
    p: cutlass.Constexpr[int],
    mbox: cutlass.Array,
    mbox_bar: cutlass.Array,
    p1: cutlass.Array,
    part: cutlass.Array,
    a0,
    a1,
    a2,
    a3,  # this rank's partials: (r0, t0), (r0, t0 + 1), (r0 + 8, t0), (r0 + 8, t0 + 1)
    rank,
    lane,
    cid,
    half,
    has_p3,
    buf,
    nbuf,
):
    """Owner warp of rows [16 rank, 16 rank + 16) of the tile in phase p: wait for the three peers' partials, add
    the four in rank order, and store them Lamport-style into buffer ``buf`` (phase 1: bf16 rows; phases 2-3: the
    cluster's fp32 partial), the sentinel into the same words of buffer ``nbuf``. Clusters without rows in the phase
    (24-25 in phase 3) store nothing."""
    while not _test_wait_cluster(mbox_bar.subview(p).data_ptr(), 0):
        pass
    s0 = cutlass.Float32(0.0)
    s1 = cutlass.Float32(0.0)
    s2 = cutlass.Float32(0.0)
    s3 = cutlass.Float32(0.0)
    for src in cutlass.range_constexpr(CLUSTER):
        peer4 = mbox.load(idx=((p * CLUSTER + src) * 32 + lane) * 4, vector_size=4, alignment=16)
        own = rank == cutlass.Int32(src)
        s0 = s0 + cutlass.select_(own, a0, cutlass.Float32(peer4[0]))
        s1 = s1 + cutlass.select_(own, a1, cutlass.Float32(peer4[1]))
        s2 = s2 + cutlass.select_(own, a2, cutlass.Float32(peer4[2]))
        s3 = s3 + cutlass.select_(own, a3, cutlass.Float32(peer4[3]))
    r0 = rank * 16 + lane // 4
    t0 = (lane % 4) * 2
    if cutlass.const_expr(p == 0):
        # p1 [buffer][token][q | k | f_a]: the tile's rows map to 64 cid + r (cid < 24: q, k; 24-25: f_a).
        row_p = cid * cutlass.Int32(TILE) + r0
        cur = buf * cutlass.Int32(P1_BUF) + row_p
        nxt = nbuf * cutlass.Int32(P1_BUF) + row_p
        o00 = t0 * P1_ROWS
        o10 = (t0 + 1) * P1_ROWS
        _st_b16(p1.subview(cur + o00).data_ptr().toint(), _lamport16(_bf16_bits(s0)))
        _st_b16(p1.subview(cur + o10).data_ptr().toint(), _lamport16(_bf16_bits(s1)))
        _st_b16(p1.subview(cur + o00 + 8).data_ptr().toint(), _lamport16(_bf16_bits(s2)))
        _st_b16(p1.subview(cur + o10 + 8).data_ptr().toint(), _lamport16(_bf16_bits(s3)))
        _st_b16(p1.subview(nxt + o00).data_ptr().toint(), cutlass.Int32(0xFFFF))
        _st_b16(p1.subview(nxt + o10).data_ptr().toint(), cutlass.Int32(0xFFFF))
        _st_b16(p1.subview(nxt + o00 + 8).data_ptr().toint(), cutlass.Int32(0xFFFF))
        _st_b16(p1.subview(nxt + o10 + 8).data_ptr().toint(), cutlass.Int32(0xFFFF))
    else:
        # Region 0 = v (phase 2, clusters < 24), 1 = og (phase 3), 2 = b (phase 2, clusters 24-25; rows 0-7).
        if cutlass.const_expr(p == 1):
            region = cutlass.select_(has_p3, cutlass.Int32(0), cutlass.Int32(2))
            lo_ok = has_p3 | (r0 < cutlass.Int32(8))
        else:
            region = cutlass.Int32(1)
            lo_ok = has_p3
        row_r = cutlass.select_(has_p3, (cid // cutlass.Int32(2)) * cutlass.Int32(TILE) + r0, r0)
        slot = (region * cutlass.Int32(2) + half) * cutlass.Int32(8 * PART_ROWS) + row_r
        cur_p = buf * cutlass.Int32(PART_BUF) + slot
        nxt_p = nbuf * cutlass.Int32(PART_BUF) + slot
        if lo_ok:
            part.store(_lamport32(_f32_bits(s0)), idx=cur_p + t0 * PART_ROWS)
            part.store(_lamport32(_f32_bits(s1)), idx=cur_p + (t0 + 1) * PART_ROWS)
            part.store(cutlass.Int32(SENTINEL), idx=nxt_p + t0 * PART_ROWS)
            part.store(cutlass.Int32(SENTINEL), idx=nxt_p + (t0 + 1) * PART_ROWS)
        if has_p3:
            part.store(_lamport32(_f32_bits(s2)), idx=cur_p + t0 * PART_ROWS + 8)
            part.store(_lamport32(_f32_bits(s3)), idx=cur_p + (t0 + 1) * PART_ROWS + 8)
            part.store(cutlass.Int32(SENTINEL), idx=nxt_p + t0 * PART_ROWS + 8)
            part.store(cutlass.Int32(SENTINEL), idx=nxt_p + (t0 + 1) * PART_ROWS + 8)


@cute.jit
def _stream_role(
    tma_w,
    tma_x,
    p1: cutlass.Array,
    part: cutlass.Array,
    epoch: cutlass.Array,
    ring_w: cutlass.Array,
    ring_x: cutlass.Array,
    full: cutlass.Array,
    empty: cutlass.Array,
    acc_done: cutlass.Array,
    mbox_bar: cutlass.Array,
    mbox: cutlass.Array,
    tmem_holder: cutlass.Array,
    USE_PDL: cutlass.Constexpr[bool],
    STAGES: cutlass.Constexpr[int],
    lbx=None,
):
    """One CTA of a stream cluster (cluster ids 0-25): the three projection phases of its tile rows. ``lbx`` is the
    CTA's logical index in the fused grid (the block index when None)."""
    tidx, _, _ = cute.arch.thread_idx()
    bx = cute.arch.block_idx()[0] if lbx is None else lbx
    lane = tidx % 32
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    rank = cute.arch.block_idx_in_cluster()
    # The f_a / b clusters (logical 24-25) take the grid's first cluster slots, so they are resident with the
    # first CTAs even when a predecessor still holds SMs: every verify CTA needs their phase 1 (f_a).
    cid = (bx // cutlass.Int32(CLUSTER) + cutlass.Int32(STREAM_CLUSTERS - 2)) % cutlass.Int32(
        STREAM_CLUSTERS
    )
    half = cid % cutlass.Int32(2)
    has_p3 = cid < cutlass.Int32(2 * (HK // TILE))
    p1_row = cutlass.select_(
        cid < cutlass.Int32(2 * HK // TILE),
        cid * cutlass.Int32(TILE),
        cutlass.Int32(FA_ROW) + (cid - cutlass.Int32(2 * HK // TILE)) * cutlass.Int32(TILE),
    )
    p2_row = cutlass.select_(
        has_p3,
        cutlass.Int32(V_ROW) + (cid // cutlass.Int32(2)) * cutlass.Int32(TILE),
        cutlass.Int32(B_ROW),
    )
    p3_row = cutlass.Int32(OG_ROW) + (cid // cutlass.Int32(2)) * cutlass.Int32(TILE)
    p1_chunk = rank * cutlass.Int32(P1_BOXES * BOX_CH)
    p23_chunk = half * cutlass.Int32(CLUSTER * P23_BOXES * BOX_CH) + rank * cutlass.Int32(
        P23_BOXES * BOX_CH
    )

    tma_ptr_w = tma_w.get_ptr()
    tma_ptr_x = tma_x.get_ptr()
    if warp == 0:
        prims.prefetch_tensormap(tma_ptr_w)
        prims.prefetch_tensormap(tma_ptr_x)
        if prims.elect_sync():
            for s in cutlass.range_constexpr(STAGES):
                prims.mbarrier_init(full.subview(s), 2)  # the weight and the activation TMA
                prims.mbarrier_init(empty.subview(s), 1)
            for p in cutlass.range_constexpr(3):
                prims.mbarrier_init(acc_done.subview(p), 1)
    elif warp == 2:
        prims.tcgen05_alloc(tmem_holder, TMEM_COLS)
        prims.tcgen05_relinquish_alloc_permit()
    elif warp == 3:
        if prims.elect_sync():
            for p in cutlass.range_constexpr(3):
                prims.mbarrier_init(mbox_bar.subview(p), 1)
                prims.mbarrier_arrive_expect_tx(mbox_bar.subview(p), (CLUSTER - 1) * 32 * 16)
    prims.fence_mbarrier_init()
    # Cluster formation: the peers' mailboxes and their barriers are initialized and addressable from here on.
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)

    if warp == 0:
        # Weight boxes: phase 1 (14), phase 2 (7), phase 3 (7, clusters 0-23). The first STAGES go out at launch.
        if prims.elect_sync():
            for i in cutlass.range_constexpr(P1_BOXES + P23_BOXES):
                stage_w = i % STAGES
                if cutlass.const_expr(i == STAGES and STAGES < P1_BOXES):
                    # Phase 1's boxes beyond the ring go to L2 now, before the grid dependency, so that their loads
                    # after it (once the first stages are consumed) hit L2 instead of HBM.
                    for i_pf in cutlass.range_constexpr(STAGES, P1_BOXES):
                        prims.cp_async_bulk_tensor_prefetch(
                            tma_ptr_w,
                            [cutlass.Int32(0), p1_row, p1_chunk + cutlass.Int32(i_pf * BOX_CH), cutlass.Int32(0),
                             cutlass.Int32(0)],
                            [],
                        )  # fmt: skip
                if cutlass.const_expr(i >= STAGES):
                    while not cute.arch.mbarrier_test_wait(
                        empty.subview(stage_w).data_ptr(), (i // STAGES + 1) % 2
                    ):
                        pass
                if cutlass.const_expr(i < P1_BOXES):
                    row_w = p1_row
                    chunk_w = p1_chunk + cutlass.Int32(i * BOX_CH)
                else:
                    row_w = p2_row
                    chunk_w = p23_chunk + cutlass.Int32((i - P1_BOXES) * BOX_CH)
                prims.mbarrier_arrive_expect_tx(full.subview(stage_w), BOX_ELEMS * 2)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    ring_w.subview(stage_w * BOX_ELEMS), tma_ptr_w,
                    (cutlass.Int32(0), row_w, chunk_w, cutlass.Int32(0), cutlass.Int32(0)), full.subview(stage_w),
                    l2_cache_hint=EVICT_FIRST,
                )  # fmt: skip
            if has_p3:
                for j in cutlass.range_constexpr(P23_BOXES):
                    i3 = P1_BOXES + P23_BOXES + j
                    stage_w3 = i3 % STAGES
                    # Phase 3 (the output gate, needed only after the recurrence) keeps at most P3_WINDOW boxes in
                    # flight: box i3 waits for box i3 - P3_WINDOW to be consumed, which also frees its own stage. A
                    # shallower HBM queue keeps the verify CTAs' phase-2 polls short while phase 3 streams.
                    jw = i3 - P3_WINDOW
                    while not cute.arch.mbarrier_test_wait(
                        empty.subview(jw % STAGES).data_ptr(), (jw // STAGES) % 2
                    ):
                        pass
                    prims.mbarrier_arrive_expect_tx(full.subview(stage_w3), BOX_ELEMS * 2)
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        ring_w.subview(stage_w3 * BOX_ELEMS), tma_ptr_w,
                        (cutlass.Int32(0), p3_row, p23_chunk + cutlass.Int32(j * BOX_CH), cutlass.Int32(0),
                         cutlass.Int32(0)),
                        full.subview(stage_w3), l2_cache_hint=EVICT_FIRST,
                    )  # fmt: skip
            if cutlass.const_expr(USE_PDL):
                # Dependents launch once the last weight box is in flight: they cannot take HBM bandwidth from the
                # boxes this grid still needs.
                prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    elif warp == 1:
        if cutlass.const_expr(USE_PDL):
            prims.griddepcontrol(prims.GridDepAction.WAIT)
        if prims.elect_sync():
            for i in cutlass.range_constexpr(P1_BOXES + P23_BOXES):
                stage_x = i % STAGES
                if cutlass.const_expr(i >= STAGES):
                    while not cute.arch.mbarrier_test_wait(
                        empty.subview(stage_x).data_ptr(), (i // STAGES + 1) % 2
                    ):
                        pass
                if cutlass.const_expr(i < P1_BOXES):
                    chunk_x = p1_chunk + cutlass.Int32(i * BOX_CH)
                else:
                    chunk_x = p23_chunk + cutlass.Int32((i - P1_BOXES) * BOX_CH)
                prims.mbarrier_arrive_expect_tx(full.subview(stage_x), X_ELEMS * 2)
                for c in cutlass.range_constexpr(BOX_CH):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        ring_x.subview(stage_x * X_ELEMS + c * MMA_N * BOX_K), tma_ptr_x,
                        ((chunk_x + cutlass.Int32(c)) * cutlass.Int32(BOX_K), cutlass.Int32(0)), full.subview(stage_x),
                    )  # fmt: skip
            if has_p3:
                for j in cutlass.range_constexpr(P23_BOXES):
                    i3x = P1_BOXES + P23_BOXES + j
                    stage_x3 = i3x % STAGES
                    while not cute.arch.mbarrier_test_wait(
                        empty.subview(stage_x3).data_ptr(), (i3x // STAGES + 1) % 2
                    ):
                        pass
                    prims.mbarrier_arrive_expect_tx(full.subview(stage_x3), X_ELEMS * 2)
                    for c in cutlass.range_constexpr(BOX_CH):
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            ring_x.subview(stage_x3 * X_ELEMS + c * MMA_N * BOX_K), tma_ptr_x,
                            ((p23_chunk + cutlass.Int32(j * BOX_CH + c)) * cutlass.Int32(BOX_K), cutlass.Int32(0)),
                            full.subview(stage_x3),
                        )  # fmt: skip
    elif warp == 2:
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=cutlass.BFloat16,
            b_dtype=cutlass.BFloat16,
            n_dim=MMA_N,
            m_dim=TILE,
        )
        desc_w = prims.Tcgen05SmemDesc.build(
            start_address=ring_w, leading_byte_offset=16, stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        desc_x = prims.Tcgen05SmemDesc.build(
            start_address=ring_x, leading_byte_offset=16, stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        tmem_raw = tmem_holder.load()
        for i in cutlass.range_constexpr(P1_BOXES + P23_BOXES):
            stage_m = i % STAGES
            phase_m = 0 if i < P1_BOXES else 1
            first_m = i == 0 or i == P1_BOXES
            while not cute.arch.mbarrier_test_wait(
                full.subview(stage_m).data_ptr(), (i // STAGES) % 2
            ):
                pass
            for kb in cutlass.range_constexpr(BOX_CH * (BOX_K // MMA_K)):
                c = kb // (BOX_K // MMA_K)
                kk = kb % (BOX_K // MMA_K)
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1,
                        cutlass.inttoptr(tmem_raw + cutlass.Int32(MMA_N * phase_m), 6, cutlass.Int32),
                        desc_w + (stage_m * W_STAGE_U + c * W_CHUNK_U + kk * K_STEP_U),
                        desc_x + (stage_m * X_STAGE_U + c * X_CHUNK_U + kk * K_STEP_U),
                        idesc, not (first_m and kb == 0),
                    )  # fmt: skip
            if prims.elect_sync():
                prims.tcgen05_commit(empty.subview(stage_m))
                if cutlass.const_expr(i == P1_BOXES - 1 or i == P1_BOXES + P23_BOXES - 1):
                    prims.tcgen05_commit(acc_done.subview(phase_m))
        if has_p3:
            for j in cutlass.range_constexpr(P23_BOXES):
                i3m = P1_BOXES + P23_BOXES + j
                stage_m3 = i3m % STAGES
                while not cute.arch.mbarrier_test_wait(
                    full.subview(stage_m3).data_ptr(), (i3m // STAGES) % 2
                ):
                    pass
                for kb3 in cutlass.range_constexpr(BOX_CH * (BOX_K // MMA_K)):
                    c3 = kb3 // (BOX_K // MMA_K)
                    kk3 = kb3 % (BOX_K // MMA_K)
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1,
                            cutlass.inttoptr(tmem_raw + cutlass.Int32(2 * MMA_N), 6, cutlass.Int32),
                            desc_w + (stage_m3 * W_STAGE_U + c3 * W_CHUNK_U + kk3 * K_STEP_U),
                            desc_x + (stage_m3 * X_STAGE_U + c3 * X_CHUNK_U + kk3 * K_STEP_U),
                            idesc, not (j == 0 and kb3 == 0),
                        )  # fmt: skip
                if prims.elect_sync():
                    prims.tcgen05_commit(empty.subview(stage_m3))
                    if cutlass.const_expr(j == P23_BOXES - 1):
                        prims.tcgen05_commit(acc_done.subview(2))
        else:
            # No phase-3 rows: the phase completes empty (its epilogue reduces unused columns and stores nothing).
            if prims.elect_sync():
                prims.tcgen05_commit(acc_done.subview(2))
    elif warp >= 4:
        w = (
            warp - 4
        )  # TMEM lanes 32 w .. 32 w + 31 = tile rows 16 w .. 16 w + 15, owned by cluster rank w
        if cutlass.const_expr(USE_PDL):
            prims.griddepcontrol(prims.GridDepAction.WAIT)
        e = epoch.load(idx=bx)
        buf = e % cutlass.Int32(BUFFERS)
        nbuf = (e + cutlass.Int32(1)) % cutlass.Int32(BUFFERS)
        tmem_raw_e = tmem_holder.load()
        lane_base = (tmem_raw_e >> 16) + w * 32
        col_base = tmem_raw_e & 0xFFFF
        for p in cutlass.range_constexpr(3):
            while not cute.arch.mbarrier_test_wait(acc_done.subview(p).data_ptr(), 0):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            acc = prims.tcgen05_ld(
                "16x256b",
                cutlass.inttoptr((lane_base << 16) | (col_base + p * MMA_N), 6, cutlass.Float32),
                num=1,
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            if w != rank:
                _st_async_v4_f32(
                    _mapa_u32(mbox.subview(((p * CLUSTER + rank) * 32 + lane) * 4).data_ptr(), w),
                    cutlass.Float32(acc[0]), cutlass.Float32(acc[1]), cutlass.Float32(acc[2]), cutlass.Float32(acc[3]),
                    _mapa_u32(mbox_bar.subview(p).data_ptr(), w),
                )  # fmt: skip
            else:
                _publish(p, mbox, mbox_bar, p1, part, cutlass.Float32(acc[0]), cutlass.Float32(acc[1]),
                         cutlass.Float32(acc[2]), cutlass.Float32(acc[3]), rank, lane, cid, half, has_p3, buf,
                         nbuf)  # fmt: skip
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(1, thread_count=128)
        if warp == 4:
            prims.tcgen05_dealloc(cutlass.inttoptr(tmem_raw_e, 6, cutlass.Int32), TMEM_COLS)
            if lane == 0:
                epoch.store(nbuf, idx=bx)


def _rows_out(shq, lane):
    """The warp's 4 row outputs from each lane's q-dot partials: two reduce-scatter rounds (bits 4, 3) then
    butterflies (bits 2, 1, 0), the butterfly's pairs in its order (bit-identical); lanes 8 m hold row
    2 (m >> 1) + (m & 1)."""
    up4 = (lane & cutlass.Int32(16)) != cutlass.Int32(0)
    w0 = cutlass.select_(up4, shq[2], shq[0]) + cute.arch.shuffle_sync_bfly(
        cutlass.select_(up4, shq[0], shq[2]), offset=16, mask=-1, mask_and_clamp=31
    )
    w1 = cutlass.select_(up4, shq[3], shq[1]) + cute.arch.shuffle_sync_bfly(
        cutlass.select_(up4, shq[1], shq[3]), offset=16, mask=-1, mask_and_clamp=31
    )
    up3 = (lane & cutlass.Int32(8)) != cutlass.Int32(0)
    xq = cutlass.select_(up3, w1, w0) + cute.arch.shuffle_sync_bfly(
        cutlass.select_(up3, w0, w1), offset=8, mask=-1, mask_and_clamp=31
    )
    for offset in [4, 2, 1]:
        xq = xq + cute.arch.shuffle_sync_bfly(xq, offset=offset, mask=-1, mask_and_clamp=31)
    return xq


def _load_rec_operands(r_st: cutlass.Array, t, warp, lane, s_v, s_q, s_kd, s_bk, s_dec):
    """Token t's recurrence operands into ``r_st[R_OP:]`` (v of the warp's rows, then the lane's keys of q,
    decay * k, beta * k and decay)."""
    v4 = s_v.load(idx=t * V_CTA + warp * REC_ROWS, vector_size=4, alignment=16)
    for j in range(REC_ROWS):
        r_st.store(cutlass.Float32(v4[j]), idx=R_OP + j)
    for i in range(VEC):
        c = t * HD + i * 32 + lane
        r_st.store(s_q.load(idx=c), idx=R_OP + REC_ROWS + i)
        r_st.store(s_kd.load(idx=c), idx=R_OP + REC_ROWS + VEC + i)
        r_st.store(s_bk.load(idx=c), idx=R_OP + REC_ROWS + 2 * VEC + i)
        r_st.store(s_dec.load(idx=c), idx=R_OP + REC_ROWS + 3 * VEC + i)


def _sent16(word):
    """Whether either bf16 in an int32 word is the Lamport sentinel (all ones)."""
    return ((word & cutlass.Int32(0xFFFF)) == cutlass.Int32(0xFFFF)) | (
        ((word >> 16) & cutlass.Int32(0xFFFF)) == cutlass.Int32(0xFFFF)
    )


def _sent32(word):
    return word == cutlass.Int32(SENTINEL)


@cute.jit
def _poll_partials(part: cutlass.Array, idx, dst: cutlass.Array, dst_idx):
    """Spin (relaxed, gpu scope, no sleep) until the four fp32 partial words at part[idx] are published, then store
    them into dst[dst_idx : dst_idx + 4]."""
    _poll_partials_from(part, idx, dst, dst_idx, cutlass.Int32(SENTINEL), cutlass.Int32(SENTINEL),
                        cutlass.Int32(SENTINEL), cutlass.Int32(SENTINEL))  # fmt: skip


@cute.jit
def _poll_partials_from(part: cutlass.Array, idx, dst: cutlass.Array, dst_idx, w0, w1, w2, w3):
    """``_poll_partials`` starting from the words of an earlier read of part[idx]: re-reads only while one of them
    is still the sentinel."""
    while _sent32(w0) | _sent32(w1) | _sent32(w2) | _sent32(w3):
        vals = prims.load_ext(
            part.subview(idx), dtype=cutlass.Int32, count=4, order="relaxed", scope="gpu"
        )
        w0 = cutlass.Int32(vals[0])
        w1 = cutlass.Int32(vals[1])
        w2 = cutlass.Int32(vals[2])
        w3 = cutlass.Int32(vals[3])
    dst.store((w0.bitcast(cutlass.Float32), w1.bitcast(cutlass.Float32), w2.bitcast(cutlass.Float32),
               w3.bitcast(cutlass.Float32)), idx=dst_idx, alignment=16)  # fmt: skip


@cute.jit
def _head_role(
    tma_wfb,
    p1w: cutlass.Array,
    part: cutlass.Array,
    w_q: cutlass.Array,
    w_k: cutlass.Array,
    w_v: cutlass.Array,
    a_log: cutlass.Array,
    dt_bias: cutlass.Array,
    onorm_w: cutlass.Array,
    cs_q: cutlass.Array,
    cs_k: cutlass.Array,
    cs_v: cutlass.Array,
    ssm: cutlass.Array,
    state_tok: cutlass.Array,
    slots: cutlass.Array,
    pending: cutlass.Array,
    out: cutlass.Array,
    epoch: cutlass.Array,
    smem_a: cutlass.Array,
    smem_b: cutlass.Array,
    bars: cutlass.Array,
    tmem_holder: cutlass.Array,
    s_uq: cutlass.Array,
    s_uk: cutlass.Array,
    s_uv: cutlass.Array,
    s_wq: cutlass.Array,
    s_wk: cutlass.Array,
    s_wv: cutlass.Array,
    s_dtb: cutlass.Array,
    s_onw: cutlass.Array,
    s_gr: cutlass.Array,
    s_braw: cutlass.Array,
    s_og: cutlass.Array,
    s_q: cutlass.Array,
    s_k: cutlass.Array,
    s_dec: cutlass.Array,
    s_kd: cutlass.Array,
    s_bk: cutlass.Array,
    s_beta: cutlass.Array,
    s_v: cutlass.Array,
    s_o: cutlass.Array,
    s_ss: cutlass.Array,
    s_rs: cutlass.Array,
    s_vp: cutlass.Array,
    s_ogp: cutlass.Array,
    s_bp: cutlass.Array,
    s_rec: cutlass.Array,
    r_st: cutlass.Array,
    ssm_stride,
    pool_n,
    lower_bound: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    eps: cutlass.Constexpr[float],
    USE_PDL: cutlass.Constexpr[bool],
    lbx,
):
    """One CTA of a KDA verify cluster (cluster ids 26-31): stage A's verify for local head h = cid - 26 and V rows
    [32 q, 32 q + 32) (q = cluster rank), fed by the stream clusters' Lamport buffers of this launch: q, k and f_a
    (phase 1), v and b (phase 2), the output gate (phase 3). Arithmetic, rounding and state contract as
    ``k3_kda_verify`` (V split 4 instead of 8; the output norm adds the same eight 16-row partial sums in the same
    order)."""
    tidx, _, _ = cute.arch.thread_idx()
    bx = lbx
    lane = tidx % 32
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    vq = cute.arch.block_idx_in_cluster()
    h = bx // cutlass.Int32(CLUSTER) - cutlass.Int32(STREAM_CLUSTERS)
    v0 = vq * V_CTA
    ch0 = h * HD
    w_full = bars.subview(0)
    acc_done = bars.subview(1)
    ss_ready = bars.subview(2)

    tma_ptr_w = tma_wfb.get_ptr()
    if warp == 0:
        prims.prefetch_tensormap(tma_ptr_w)
        if prims.elect_sync():
            prims.mbarrier_init(w_full, 1)
            prims.mbarrier_init(acc_done, 1)
    elif warp == 2:
        prims.tcgen05_alloc(tmem_holder, FB_TMEM_COLS)
        prims.tcgen05_relinquish_alloc_permit()
    elif warp == 3:
        if prims.elect_sync():
            prims.mbarrier_init(ss_ready, 1)
            prims.mbarrier_arrive_expect_tx(ss_ready, CLUSTER * 2 * NT * 4)
    prims.fence_mbarrier_init()
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    if warp == 0:
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(w_full, HD * HD * 2)
            prims.cp_async_bulk_tensor_shared_cta_global(
                smem_a, tma_ptr_w, (cutlass.Int32(0), ch0, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)),
                w_full,
            )  # fmt: skip

    # ---- Before the grid dependency: nothing here is written by this launch's stream clusters or predecessors.
    # Two round trips: first the slot index, every slot's accepted-draft count and what needs neither (conv weights,
    # a_log, the norm and gate constants); then, as asynchronous copies into shared memory, everything the slot and
    # its count select (the state, the conv caches' last positions and, below, the drafts' records).
    slot = slots.load(idx=0)
    pend_w = []
    for i_pw in cutlass.range_constexpr(PEND_WORDS):
        pw_i = lane + cutlass.Int32(32 * i_pw)
        pend_w.append(
            cutlass.Int32(pending.load(idx=cutlass.select_(pw_i < pool_n, pw_i, cutlass.Int32(0))))
        )
    a_raw = a_log.load(idx=h)
    QH = 3 * HD // 4
    VH = 3 * V_CTA // 4
    if tidx >= 2 * QH + VH + V_CTA:
        c4_n = (tidx - (2 * QH + VH + V_CTA)) * 4
        prims.cp_async_shared_global(s_onw.subview(c4_n), onorm_w.subview(v0 + c4_n), 16, CP_CG)
    if tidx < HD // 4:
        prims.cp_async_shared_global(
            s_dtb.subview(tidx * 4), dt_bias.subview(ch0 + tidx * 4), 16, CP_CG
        )
    if tidx < HD:
        wq4 = w_q.load(idx=(ch0 + tidx) * CONV_W, vector_size=CONV_W, alignment=16)
        for w in cutlass.range_constexpr(CONV_W):
            s_wq.store(cutlass.Float32(wq4[w]), idx=w * HD + tidx)
    else:
        ck_w = tidx - HD
        wk4 = w_k.load(idx=(ch0 + ck_w) * CONV_W, vector_size=CONV_W, alignment=16)
        for w in cutlass.range_constexpr(CONV_W):
            s_wk.store(cutlass.Float32(wk4[w]), idx=w * HD + ck_w)
    if tidx >= 2 * QH + VH:
        if tidx < 2 * QH + VH + V_CTA:
            cv_w = tidx - (2 * QH + VH)
            wv4 = w_v.load(idx=(ch0 + v0 + cv_w) * CONV_W, vector_size=CONV_W, alignment=16)
            for w in cutlass.range_constexpr(CONV_W):
                s_wv.store(cutlass.Float32(wv4[w]), idx=w * V_CTA + cv_w)
    pend_word = pend_w[PEND_WORDS - 1]
    for i_pw in cutlass.range_constexpr(PEND_WORDS - 2, -1, -1):
        pend_word = cutlass.Int32(
            cutlass.select_(slot < cutlass.Int32(32 * (i_pw + 1)), pend_w[i_pw], pend_word)
        )
    pend = cutlass.Int32(cute.arch.shuffle_sync(pend_word, offset=slot % cutlass.Int32(32)))
    if pool_n > cutlass.Int32(32 * PEND_WORDS):
        pend = cutlass.Int32(pending.load(idx=slot))
    pend = cutlass.Int32(
        cutlass.select_(pend > cutlass.Int32(NUM_SPEC), cutlass.Int32(NUM_SPEC), pend)
    )
    exp_a = cute.math.exp(a_raw, fastmath=True)
    row_w = v0 + warp * REC_ROWS
    # The slot's pool state and per-token region from their first element: the slot offset in 64 bits, once.
    pool = ssm.subview(cutlass.Int64(slot) * ssm_stride)
    tok = state_tok.subview(cutlass.Int64(slot) * CT_SLOT)
    st_base = (h * HD + row_w) * HD
    for r in cutlass.range_constexpr(REC_ROWS):
        for i in cutlass.range_constexpr(VEC):
            r_st.store(pool.load(idx=st_base + r * HD + i * 32 + lane), idx=r * VEC + i)
    if tidx < QH:
        m_q = tidx // (HD // 4)
        c4_q = (tidx % (HD // 4)) * 4
        prims.cp_async_shared_global(
            s_uq.subview(m_q * HD + c4_q),
            cs_q.subview((slot * S_COLS + pend + m_q) * HK + ch0 + c4_q),
            16,
            CP_CG,
        )
    elif tidx < 2 * QH:
        m_k = (tidx - QH) // (HD // 4)
        c4_k = ((tidx - QH) % (HD // 4)) * 4
        prims.cp_async_shared_global(
            s_uk.subview(m_k * HD + c4_k),
            cs_k.subview((slot * S_COLS + pend + m_k) * HK + ch0 + c4_k),
            16,
            CP_CG,
        )
    elif tidx < 2 * QH + VH:
        m_v = (tidx - 2 * QH) // (V_CTA // 4)
        c4_v = ((tidx - 2 * QH) % (V_CTA // 4)) * 4
        prims.cp_async_shared_global(
            s_uv.subview(m_v * V_CTA + c4_v),
            cs_v.subview((slot * S_COLS + pend + m_v) * HK + ch0 + v0 + c4_v),
            16,
            CP_CG,
        )
    # The drafts the sampler accepted, replayed from their records onto the golden token's state. Every record this
    # CTA may need (all NUM_SPEC drafts) comes into shared memory by one round trip first; a draft per iteration from
    # global memory was one dependent round trip per accepted draft.
    for it_rc in cutlass.range_constexpr((NUM_SPEC * REC_CTA // 4 + THREADS - 1) // THREADS):
        q_rc = tidx + it_rc * THREADS
        if q_rc < NUM_SPEC * REC_CTA // 4:
            t_rc = q_rc // (REC_CTA // 4)
            u_rc = (q_rc % (REC_CTA // 4)) * 4
            rec_rc = (t_rc * H_LOCAL + h) * HD
            src_rc = cutlass.Int32(
                cutlass.select_(
                    u_rc < V_CTA,
                    rec_rc + CT_VN + v0 + u_rc,
                    cutlass.select_(
                        u_rc < V_CTA + HD,
                        rec_rc + CT_WB + (u_rc - V_CTA),
                        rec_rc + CT_WD + (u_rc - V_CTA - HD),
                    ),
                )
            )
            prims.cp_async_shared_global(
                s_rec.subview(t_rc * REC_CTA + u_rc), tok.subview(src_rc), 16, CP_CG
            )
    prims.cp_async_commit_group()
    prims.cp_async_wait_group(0)
    prims.barrier_cta_sync(0)
    for t_acc in cutlass.range(pend, unroll=1):
        rec_s = t_acc * REC_CTA
        vn4_acc = s_rec.load(idx=rec_s + warp * REC_ROWS, vector_size=4, alignment=16)
        vns_acc = [cutlass.Float32(vn4_acc[r]) for r in range(REC_ROWS)]
        wbs_acc = [s_rec.load(idx=rec_s + V_CTA + i * 32 + lane) for i in range(VEC)]
        wds_acc = [s_rec.load(idx=rec_s + V_CTA + HD + i * 32 + lane) for i in range(VEC)]
        sts_acc = [r_st.load(idx=j) for j in range(REC_ROWS * VEC)]
        for r in cutlass.range_constexpr(REC_ROWS):
            for _pi in cutlass.range_constexpr(VEC // 2):
                _p = _pi * 2
                vb0_acc, vb1_acc = cute.arch.mul_packed_f32x2(
                    (vns_acc[r], vns_acc[r]), (wbs_acc[_p], wbs_acc[_p + 1])
                )
                sts_acc[r * VEC + _p], sts_acc[r * VEC + _p + 1] = cute.arch.fma_packed_f32x2(
                    src_a=(wds_acc[_p], wds_acc[_p + 1]), src_b=(sts_acc[r * VEC + _p], sts_acc[r * VEC + _p + 1]),
                    src_c=(vb0_acc, vb1_acc),
                )  # fmt: skip
        for j in cutlass.range_constexpr(REC_ROWS * VEC):
            r_st.store(sts_acc[j], idx=j)
    # Every CTA of the head has read the head's q / k conv caches and per-key records once all four arrive here
    # (release; the acquiring wait follows phase 1): from then on each CTA rewrites its quarter of both, in the idle
    # and pre-recurrence slots instead of after the output norm.
    prims.barrier_cluster_arrive()

    if cutlass.const_expr(USE_PDL):
        prims.griddepcontrol(prims.GridDepAction.WAIT)
    e = epoch.load(idx=bx)
    buf = e % cutlass.Int32(BUFFERS)
    if cutlass.const_expr(USE_PDL):
        # The dependents may launch once the stream CTAs have also triggered (after their last weight box): they
        # then take the SMs the stream leaves and run their pre-wait work beside the verify instead of after it.
        if tidx == 0:
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)

    # ---- Phase 1 (q, k, f_a of this launch): thread i polls 16-byte chunk i of the q | k rows, threads < 128 also
    # chunk i of the f_a rows; q and k go to the conv inputs as fp32, f_a to the MMA's B operand (128-byte swizzle).
    st_qk = tidx // HD
    t_qk = (tidx % HD) // 16
    j_qk = tidx % 16
    a0 = cutlass.Int32(SENTINEL)
    a1 = cutlass.Int32(SENTINEL)
    a2 = cutlass.Int32(SENTINEL)
    a3 = cutlass.Int32(SENTINEL)
    qk_idx = ((buf * NT + t_qk) * P1_ROWS + st_qk * HK + ch0 + j_qk * 8) // 2
    while _sent16(a0) | _sent16(a1) | _sent16(a2) | _sent16(a3):
        vqk = prims.load_ext(
            p1w.subview(qk_idx), dtype=cutlass.Int32, count=4, order="relaxed", scope="gpu"
        )
        a0 = cutlass.Int32(vqk[0])
        a1 = cutlass.Int32(vqk[1])
        a2 = cutlass.Int32(vqk[2])
        a3 = cutlass.Int32(vqk[3])
    if st_qk == 0:
        _store8(s_uq, (a0, a1, a2, a3), (CONV_W - 1 + t_qk) * HD + j_qk * 8)
    else:
        _store8(s_uk, (a0, a1, a2, a3), (CONV_W - 1 + t_qk) * HD + j_qk * 8)
    if tidx < HD:
        t_fa = tidx // 16
        j_fa = tidx % 16
        f0 = cutlass.Int32(SENTINEL)
        f1 = cutlass.Int32(SENTINEL)
        f2 = cutlass.Int32(SENTINEL)
        f3 = cutlass.Int32(SENTINEL)
        fa_idx = ((buf * NT + t_fa) * P1_ROWS + 2 * HK + j_fa * 8) // 2
        while _sent16(f0) | _sent16(f1) | _sent16(f2) | _sent16(f3):
            vfa = prims.load_ext(
                p1w.subview(fa_idx), dtype=cutlass.Int32, count=4, order="relaxed", scope="gpu"
            )
            f0 = cutlass.Int32(vfa[0])
            f1 = cutlass.Int32(vfa[1])
            f2 = cutlass.Int32(vfa[2])
            f3 = cutlass.Int32(vfa[3])
        # Row t_fa, 16-byte unit u of 64-column chunk c: byte 1024 c + 128 t_fa + 16 (u ^ t_fa).
        sw_word = ((j_fa // 8) * 1024 + t_fa * 128 + ((j_fa % 8) ^ t_fa) * 16) // 4
        smem_b.store((f0, f1, f2, f3), idx=sw_word, alignment=16)
        prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
    prims.barrier_cta_sync(0)
    prims.barrier_cluster_wait()

    # ---- f_b on the tensor cores (warp 2 issues, warps 4-7 read TMEM), then the q/k pre-compute, one token per warp.
    if warp == 2:
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=cutlass.BFloat16,
            b_dtype=cutlass.BFloat16,
            n_dim=MMA_N,
            m_dim=HD,
        )
        desc_a_base = prims.Tcgen05SmemDesc.build(
            start_address=smem_a, leading_byte_offset=16, stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        desc_b_base = prims.Tcgen05SmemDesc.build(
            start_address=smem_b, leading_byte_offset=16, stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        tmem_acc = cutlass.inttoptr(tmem_holder.load(), 6, cutlass.Int32)
        while not cute.arch.mbarrier_test_wait(w_full.data_ptr(), 0):
            pass
        prims.tcgen05_fence(
            prims.Tcgen05Fence.AFTER_THREAD_SYNC
        )  # f_a came from the other threads' smem stores
        for kb in cutlass.range_constexpr(2 * (BOX_K // MMA_K)):
            box = kb // (BOX_K // MMA_K)
            within = kb % (BOX_K // MMA_K)
            if prims.elect_sync():
                prims.tcgen05_mma(
                    prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, tmem_acc,
                    desc_a_base + (box * ((HD * BOX_K * 2) >> 4) + within * K_STEP_U),
                    desc_b_base + (box * ((MMA_N * BOX_K * 2) >> 4) + within * K_STEP_U),
                    idesc, kb != 0,
                )  # fmt: skip
        if prims.elect_sync():
            prims.tcgen05_commit(acc_done)
    if warp >= 4:
        while not cute.arch.mbarrier_test_wait(acc_done.data_ptr(), 0):
            pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        acc = prims.tcgen05_ld(
            "32x32b", cutlass.inttoptr(tmem_holder.load(), 6, cutlass.Float32), num=MMA_N
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        c_fb = (warp - 4) * 32 + lane
        for t_fb in cutlass.range_constexpr(NT):
            s_gr.store(_bf16(cutlass.Float32(acc[t_fb])), idx=t_fb * HD + c_fb)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(1, thread_count=128)
        if warp == 4:
            prims.tcgen05_dealloc(
                cutlass.inttoptr(tmem_holder.load(), 6, cutlass.Int32), FB_TMEM_COLS
            )
    # q and k of token w, one token per warp (the arithmetic of kda_mtp_decode's pre-compute warps).
    tk = warp
    pq = [cutlass.Float32(0.0)] * VEC
    for i in cutlass.range_constexpr(VEC):
        c = i * 32 + lane
        conv = cutlass.Float32(0.0)
        for w in cutlass.range_constexpr(CONV_W - 1):
            conv += s_uq.load(idx=(tk + w) * HD + c) * s_wq.load(idx=w * HD + c)
        conv += s_uq.load(idx=(tk + CONV_W - 1) * HD + c) * s_wq.load(idx=(CONV_W - 1) * HD + c)
        ex = cute.math.exp(-conv, fastmath=True)
        pq[i] = conv * cute.arch.rcp_approx(cutlass.Float32(1.0) + ex)
    sum_q = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(VEC):
        sum_q += pq[i] * pq[i]
    sum_q = _butterfly(sum_q)
    rnorm_q = cute.math.rsqrt(sum_q + 1e-06, fastmath=True) * scale
    for i in cutlass.range_constexpr(VEC):
        s_q.store(pq[i] * rnorm_q, idx=tk * HD + i * 32 + lane)
    pk = [cutlass.Float32(0.0)] * VEC
    for i in cutlass.range_constexpr(VEC):
        c = i * 32 + lane
        conv = s_uk.load(idx=tk * HD + c) * s_wk.load(idx=c)
        for w in cutlass.range_constexpr(1, CONV_W - 1):
            conv += s_uk.load(idx=(tk + w) * HD + c) * s_wk.load(idx=w * HD + c)
        conv += s_uk.load(idx=(tk + CONV_W - 1) * HD + c) * s_wk.load(idx=(CONV_W - 1) * HD + c)
        pk[i] = conv * cute.arch.rcp_approx(
            cutlass.Float32(1.0) + cute.math.exp(-conv, fastmath=True)
        )
    sum_k = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(VEC):
        sum_k += pk[i] * pk[i]
    sum_k = _butterfly(sum_k)
    rnorm_k = cute.math.rsqrt(sum_k + 1e-06, fastmath=True)
    for i in cutlass.range_constexpr(VEC):
        s_k.store(pk[i] * rnorm_k, idx=tk * HD + i * 32 + lane)
    if warp < 4:
        # Conv caches for the next round, q and k, this CTA's 32 channels: column s = position s - 2 = row s + 1 of
        # the position-indexed inputs.
        t_cq = tidx
        for it_qk in cutlass.range_constexpr((2 * S_COLS * V_CTA + 127) // 128):
            item_qk = t_cq + it_qk * 128
            if item_qk < 2 * S_COLS * V_CTA:
                sc_qk = (item_qk % (S_COLS * V_CTA)) // V_CTA
                ch_qk = vq * V_CTA + item_qk % V_CTA
                dst_qk = (slot * S_COLS + sc_qk) * HK + ch0 + ch_qk
                if item_qk < S_COLS * V_CTA:
                    cs_q.store(s_uq.load(idx=(sc_qk + 1) * HD + ch_qk), idx=dst_qk)
                else:
                    cs_k.store(s_uk.load(idx=(sc_qk + 1) * HD + ch_qk), idx=dst_qk)
    prims.barrier_cta_sync(0)

    # ---- Phase 2 (v, b): the two K-half partials of this CTA's 32 v rows and of this head's b; the projection is
    # their bf16-rounded sum.
    if tidx < 2 * NT * (V_CTA // 4):
        half_v = tidx // (NT * (V_CTA // 4))
        t_v = (tidx // (V_CTA // 4)) % NT
        j_v = tidx % (V_CTA // 4)
        _poll_partials(
            part,
            ((buf * 3 + 0) * 2 + half_v) * (8 * PART_ROWS) + t_v * PART_ROWS + ch0 + v0 + j_v * 4,
            s_vp,
            (half_v * NT + t_v) * V_CTA + j_v * 4,
        )
    elif tidx < 2 * NT * (V_CTA // 4) + 2 * NT:
        ib = tidx - 2 * NT * (V_CTA // 4)
        half_b = ib // NT
        t_b = ib % NT
        wb = cutlass.Int32(SENTINEL)
        while _sent32(wb):
            wb = prims.load_ext(
                part.subview(((buf * 3 + 2) * 2 + half_b) * (8 * PART_ROWS) + t_b * PART_ROWS + h),
                dtype=cutlass.Int32,
                order="relaxed",
                scope="gpu",
            )
        s_bp.store(wb.bitcast(cutlass.Float32), idx=half_b * NT + t_b)
    prims.barrier_cta_sync(0)
    t_cv = tidx // V_CTA
    c_cv = tidx % V_CTA
    s_uv.store(_bf16(s_vp.load(idx=t_cv * V_CTA + c_cv) + s_vp.load(idx=(NT + t_cv) * V_CTA + c_cv)),
               idx=(CONV_W - 1 + t_cv) * V_CTA + c_cv)  # fmt: skip
    if tidx < NT:
        s_braw.store(_bf16(s_bp.load(idx=tidx) + s_bp.load(idx=NT + tidx)), idx=tidx)
    prims.barrier_cta_sync(0)
    for it_cs in cutlass.range_constexpr((S_COLS * V_CTA + THREADS - 1) // THREADS):
        item_cs = tidx + it_cs * THREADS
        if item_cs < S_COLS * V_CTA:
            s_cs = item_cs // V_CTA
            v_cs = item_cs % V_CTA
            cs_v.store(
                s_uv.load(idx=(s_cs + 1) * V_CTA + v_cs),
                idx=(slot * S_COLS + s_cs) * HK + ch0 + v0 + v_cs,
            )
    # v (this CTA's 32 channels, one (token, channel) per thread) and beta.
    conv_v = cutlass.Float32(0.0)
    for w in cutlass.range_constexpr(CONV_W):
        conv_v += s_uv.load(idx=(t_cv + w) * V_CTA + c_cv) * s_wv.load(idx=w * V_CTA + c_cv)
    conv_v = conv_v * cute.arch.rcp_approx(
        cutlass.Float32(1.0) + cute.math.exp(-conv_v, fastmath=True)
    )
    s_v.store(conv_v, idx=t_cv * V_CTA + c_cv)
    if tidx < NT:
        b_pre = s_braw.load(idx=tidx)
        s_beta.store(
            cute.arch.rcp_approx(cutlass.Float32(1.0) + cute.math.exp(-b_pre, fastmath=True)),
            idx=tidx,
        )
    prims.barrier_cta_sync(0)

    # ---- The recurrence's per-token operands: warp t, token t: decay = exp(gate), decay * k, beta * k.
    tg = warp
    r_beta_g = s_beta.load(idx=tg)
    for i_pair in cutlass.range_constexpr(VEC // 2):
        dks = []
        for i in (i_pair * 2, i_pair * 2 + 1):
            c = i * 32 + lane
            g_raw = s_gr.load(idx=tg * HD + c) + s_dtb.load(idx=c)
            xg = exp_a * g_raw
            sig = cute.arch.rcp_approx(cutlass.Float32(1.0) + cute.math.exp(-xg, fastmath=True))
            dks.append(
                (cute.math.exp(lower_bound * sig, fastmath=True), s_k.load(idx=tg * HD + c), c)
            )
        (d0, k0v, c0), (d1, k1v, c1) = dks
        bk0, bk1 = cute.arch.mul_packed_f32x2((r_beta_g, r_beta_g), (k0v, k1v))
        kd0, kd1 = cute.arch.mul_packed_f32x2((d0, d1), (k0v, k1v))
        s_dec.store(d0, idx=tg * HD + c0)
        s_dec.store(d1, idx=tg * HD + c1)
        s_kd.store(kd0, idx=tg * HD + c0)
        s_kd.store(kd1, idx=tg * HD + c1)
        s_bk.store(bk0, idx=tg * HD + c0)
        s_bk.store(bk1, idx=tg * HD + c1)
    prims.barrier_cta_sync(0)
    # The drafts' per-key records (beta * k and the decay of tokens 1-7), this CTA's 32 keys.
    if tidx < NUM_SPEC * V_CTA:
        t_rq = tidx // V_CTA
        key_rq = vq * V_CTA + tidx % V_CTA
        rec_q = h * HD + t_rq * H_LOCAL * HD + key_rq
        tok.store(s_bk.load(idx=(t_rq + 1) * HD + key_rq), idx=rec_q + CT_WB)
        tok.store(s_dec.load(idx=(t_rq + 1) * HD + key_rq), idx=rec_q + CT_WD)

    # ---- Recurrence over the verify tokens: warp w owns V rows row_w .. row_w + 3, lane l keys 32 i + l. A rolled
    # loop: in the model the MoE's stream has evicted this kernel's code from L2, and the 8 tokens fully unrolled
    # (~1,100 instructions, run once per launch) stall on instruction fetch; one token's body is fetched once. The
    # loop is software-pipelined so that consecutive tokens overlap: iteration tr reduces token tr - 1's q-dot
    # partials over the warp beside token tr's k-dot butterflies (the chain that carries the state) and loads token
    # tr + 1's operands. Every sum is formed from the same values in the same order as before (bit-identical).
    # This lane's first state element in the pool (the golden token's state) and in the first draft's per-token
    # state; element (r, i) sits (r HD + 32 i) floats past either.
    pool_ptr = pool.subview(st_base + lane).data_ptr().toint()
    # Lanes 0-3 store the warp's row innovations, one row each (a lane's address beyond them is never stored to).
    vn_ptr = tok.subview(CT_VN + h * HD + row_w + lane).data_ptr().toint()
    _load_rec_operands(r_st, cutlass.Int32(0), warp, lane, s_v, s_q, s_kd, s_bk, s_dec)
    for r in cutlass.range_constexpr(REC_ROWS):
        r_st.store(cutlass.Float32(0.0), idx=R_Q + r)
    for tr in cutlass.range(NT, unroll=1):
        vrows = [r_st.load(idx=R_OP + j) for j in range(REC_ROWS)]
        wq = [r_st.load(idx=R_OP + REC_ROWS + i) for i in range(VEC)]
        wk = [r_st.load(idx=R_OP + REC_ROWS + VEC + i) for i in range(VEC)]
        wb = [r_st.load(idx=R_OP + REC_ROWS + 2 * VEC + i) for i in range(VEC)]
        wd = [r_st.load(idx=R_OP + REC_ROWS + 3 * VEC + i) for i in range(VEC)]
        shq_prev = [r_st.load(idx=R_Q + r) for r in range(REC_ROWS)]
        # Token tr + 1's operands (the last iteration reloads token NT - 1's and does not use them).
        t_next = cutlass.Int32(cutlass.select_(tr + 1 < NT, tr + 1, cutlass.Int32(NT - 1)))
        _load_rec_operands(r_st, t_next, warp, lane, s_v, s_q, s_kd, s_bk, s_dec)
        stw = [r_st.load(idx=j) for j in range(REC_ROWS * VEC)]
        xq_prev = _rows_out(shq_prev, lane)
        shk = []
        for r in cutlass.range_constexpr(REC_ROWS):
            p1a = cutlass.Float32(0.0)
            p2a = cutlass.Float32(0.0)
            for _pi in cutlass.range_constexpr(VEC // 2):
                _p = _pi * 2
                p1a, p2a = cute.arch.fma_packed_f32x2(
                    src_a=(stw[r * VEC + _p], stw[r * VEC + _p + 1]),
                    src_b=(wk[_p], wk[_p + 1]),
                    src_c=(p1a, p2a),
                )
            shk.append(p1a + p2a)
        for offset in [16, 8, 4, 2, 1]:
            for r in cutlass.range_constexpr(REC_ROWS):
                shk[r] += cute.arch.shuffle_sync_bfly(
                    shk[r], offset=offset, mask=-1, mask_and_clamp=31
                )
        shq = []
        vns = []
        for r in cutlass.range_constexpr(REC_ROWS):
            vn = vrows[r] - shk[r]
            vns.append(vn)
            q1 = cutlass.Float32(0.0)
            q2 = cutlass.Float32(0.0)
            for _pi in cutlass.range_constexpr(VEC // 2):
                _p = _pi * 2
                vb0, vb1 = cute.arch.mul_packed_f32x2((vn, vn), (wb[_p], wb[_p + 1]))
                stw[r * VEC + _p], stw[r * VEC + _p + 1] = cute.arch.fma_packed_f32x2(
                    src_a=(wd[_p], wd[_p + 1]), src_b=(stw[r * VEC + _p], stw[r * VEC + _p + 1]), src_c=(vb0, vb1),
                )  # fmt: skip
                q1, q2 = cute.arch.fma_packed_f32x2(
                    src_a=(stw[r * VEC + _p], stw[r * VEC + _p + 1]),
                    src_b=(wq[_p], wq[_p + 1]),
                    src_c=(q1, q2),
                )
            shq.append(q1 + q2)
        # The golden token's state to the pool (tr = 0); a draft's record: this warp's row innovations (lanes 0-3).
        # Predicated stores, no branch: they issue between the update's FMAs.
        first = cutlass.Int32(cutlass.select_(tr == 0, cutlass.Int32(1), cutlass.Int32(0)))
        for r in cutlass.range_constexpr(REC_ROWS):
            for i in cutlass.range_constexpr(VEC):
                _st_f32_if(pool_ptr + cutlass.Int64((r * HD + i * 32) * 4), stw[r * VEC + i], first)
        vn_lane = cutlass.Float32(
            cutlass.select_(
                lane == 0,
                vns[0],
                cutlass.select_(lane == 1, vns[1], cutlass.select_(lane == 2, vns[2], vns[3])),
            )
        )
        draft_rec = cutlass.Int32(
            cutlass.select_((tr > 0) & (lane < 4), cutlass.Int32(1), cutlass.Int32(0))
        )
        _st_f32_if(
            vn_ptr + cutlass.Int64(tr - 1) * cutlass.Int64(H_LOCAL * HD * 4), vn_lane, draft_rec
        )
        for j in cutlass.range_constexpr(REC_ROWS * VEC):
            r_st.store(stw[j], idx=j)
        for r in cutlass.range_constexpr(REC_ROWS):
            r_st.store(shq[r], idx=R_Q + r)
        # Token tr - 1's outputs, from the first lane of each group of 8 (which all hold the same sum); at tr = 0 a
        # placeholder in token 0's slot, which iteration 1 overwrites from the same lanes.
        o_tok = cutlass.Int32(cutlass.select_(tr > 0, tr - 1, cutlass.Int32(0)))
        if (lane & cutlass.Int32(7)) == cutlass.Int32(0):
            s_o.store(
                _bf16(xq_prev),
                idx=o_tok * V_CTA + warp * REC_ROWS + (lane >> 4) * 2 + ((lane >> 3) & 1),
            )
    xq_last = _rows_out([r_st.load(idx=R_Q + r) for r in range(REC_ROWS)], lane)
    if (lane & cutlass.Int32(7)) == cutlass.Int32(0):
        s_o.store(
            _bf16(xq_last),
            idx=(NT - 1) * V_CTA + warp * REC_ROWS + (lane >> 4) * 2 + ((lane >> 3) & 1),
        )

    # ---- Phase 3 (the output gate), then the gated RMSNorm over V: every CTA sends its two 16-row sums of squares
    # per token to all four CTAs (the eight partials of V split 8, added in the same order). Thread (token t, row v)
    # reads its own gate's two K-half partials first, so their round trip overlaps the norm exchange.
    t_out = tidx // V_CTA
    v_y = tidx % V_CTA
    og_at = (buf * 3 + 1) * 2 * (8 * PART_ROWS) + t_out * PART_ROWS + ch0 + v0 + v_y
    og_h0 = cutlass.Int32(
        prims.load_ext(part.subview(og_at), dtype=cutlass.Int32, order="relaxed", scope="gpu")
    )
    og_h1 = cutlass.Int32(prims.load_ext(part.subview(og_at + 8 * PART_ROWS), dtype=cutlass.Int32, order="relaxed",
                                         scope="gpu"))  # fmt: skip
    prims.barrier_cta_sync(0)
    x_o = s_o.load(idx=tidx)
    ss = x_o * x_o
    for off_ss in [8, 4, 2, 1]:
        ss = ss + cute.arch.shuffle_sync_bfly(ss, offset=off_ss, mask=-1, mask_and_clamp=31)
    if lane % 16 == 0:
        src_slot = vq * 2 + lane // 16
        for r in cutlass.range_constexpr(CLUSTER):
            _st_async_f32(
                _mapa_u32(s_ss.subview(src_slot * NT + t_out).data_ptr(), r),
                ss,
                _mapa_u32(ss_ready.data_ptr(), r),
            )
    while not _test_wait_cluster(ss_ready.data_ptr(), 0):
        pass
    while _sent32(og_h0) | _sent32(og_h1):
        og_h0 = cutlass.Int32(
            prims.load_ext(part.subview(og_at), dtype=cutlass.Int32, order="relaxed", scope="gpu")
        )
        og_h1 = cutlass.Int32(prims.load_ext(part.subview(og_at + 8 * PART_ROWS), dtype=cutlass.Int32,
                                             order="relaxed", scope="gpu"))  # fmt: skip
    if tidx < NT:
        total = cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(2 * CLUSTER):
            total = total + s_ss.load(idx=r * NT + tidx)
        s_rs.store(cute.math.rsqrt(total / HD + eps), idx=tidx)
    prims.barrier_cta_sync(0)
    z = _bf16(og_h0.bitcast(cutlass.Float32) + og_h1.bitcast(cutlass.Float32))
    gate = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-z, fastmath=True))
    y = s_o.load(idx=tidx) * s_rs.load(idx=t_out) * s_onw.load(idx=v_y) * gate
    out.store(cutlass.BFloat16(y), idx=(t_out * H_LOCAL + h) * HD + v0 + v_y)

    if tidx == 0:
        epoch.store((e + cutlass.Int32(1)) % cutlass.Int32(BUFFERS), idx=bx)


@cute.kernel
def k3_kda_attn_kernel(
    tma_w: cutlass.GridConstant[
        cuda.TensorMap
    ],  # W [3208, 7168] bf16, 5-D, box 64 cols x 64 rows x 2 chunks
    tma_x: cutlass.GridConstant[
        cuda.TensorMap
    ],  # x [T, 7168] bf16, box 64 cols x 8 rows (rows >= T read as 0)
    p1: cutlass.Array,  # int16 bits of bf16 [3][8][1664]: phase-1 rows (Lamport)
    part: cutlass.Array,  # int32 bits of fp32 [3][3][2][8][768]: v, og and b cluster partials (Lamport)
    epoch: cutlass.Array,  # int32 [CTAs]: each CTA's buffer index (launches completed mod 3)
    USE_PDL: cutlass.Constexpr[bool],
):
    ring_w = cutlass.Array(
        cutlass.BFloat16, STREAM_STAGES * BOX_ELEMS, space=cutlass.AddressSpace.smem, alignment=1024
    )
    ring_x = cutlass.Array(
        cutlass.BFloat16, STREAM_STAGES * X_ELEMS, space=cutlass.AddressSpace.smem, alignment=1024
    )
    full = cutlass.Array(cutlass.Int64, STREAM_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    empty = cutlass.Array(
        cutlass.Int64, STREAM_STAGES, space=cutlass.AddressSpace.smem, alignment=8
    )
    acc_done = cutlass.Array(cutlass.Int64, 3, space=cutlass.AddressSpace.smem, alignment=8)
    mbox_bar = cutlass.Array(cutlass.Int64, 3, space=cutlass.AddressSpace.smem, alignment=8)
    # [phase][source rank][lane][4]: the partials of the 16 rows this rank owns (its own slot stays unused).
    mbox = cutlass.Array(
        cutlass.Float32, 3 * CLUSTER * 32 * 4, space=cutlass.AddressSpace.smem, alignment=16
    )
    tmem_holder = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    _stream_role(
        tma_w,
        tma_x,
        p1,
        part,
        epoch,
        ring_w,
        ring_x,
        full,
        empty,
        acc_done,
        mbox_bar,
        mbox,
        tmem_holder,
        USE_PDL,
        STREAM_STAGES,
    )


@cute.jit
def k3_kda_qkvg(
    w: cute.Tensor,  # bf16 [3208, 7168]
    x: cute.Tensor,  # bf16 [T, 7168]
    p1: cute.Tensor,  # int16 [3 * 8 * 1664]
    part: cute.Tensor,  # int32 [3 * 3 * 2 * 8 * 768]
    epoch: cute.Tensor,  # int32 [104]
    num_tokens: cutlass.Int32,
    USE_PDL: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """One launch of the three-phase projection (the stream role alone)."""
    tma_w = cuda.create_tensor_map_tiled(
        global_address=w.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[BOX_K, PROJ_ROWS, K_IN // BOX_K, 1, 1],
        global_strides=[(K_IN * 2) // 16, (BOX_K * 2) // 16, (PROJ_ROWS * K_IN * 2) // 16,
                        (PROJ_ROWS * K_IN * 2) // 16],
        box_dims=[BOX_K, TILE, BOX_CH, 1, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )  # fmt: skip
    tma_x = cuda.create_tensor_map_tiled(
        global_address=x.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[K_IN, num_tokens],
        global_strides=[(K_IN * 2) // 16],
        box_dims=[BOX_K, MMA_N],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    k3_kda_attn_kernel(tma_w, tma_x, p1, part, epoch, USE_PDL).launch(
        grid=(STREAM_CLUSTERS * CLUSTER, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(CLUSTER, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
    )


class _SmemCarver:
    """Typed views laid out one after another inside one raw shared-memory buffer: a CTA runs one role for its whole
    life and every CTA of a cluster runs the same role, so the two roles share the bytes and a DSMEM address always
    names the same view in the peer."""

    def __init__(self, raw, raw_bytes: int):
        self.raw = raw
        self.cap = raw_bytes
        self.off = 0

    def view(self, dtype, n: int, align: int = 16) -> cutlass.Array:
        self.off = -(-self.off // align) * align
        base = self.raw if self.off == 0 else self.raw + self.off
        ptr = cute.recast_ptr(base, dtype=dtype)
        self.off += n * dtype.width // 8
        assert self.off <= self.cap, "a role's arrays exceed the shared raw buffer"
        return cutlass.Array(
            ptr,
            shape=(n,),
            dtype=dtype,
            bounds_check=False,
            addrspace=cutlass.AddressSpace.smem.value,
            alignment=align,
        )


@cute.kernel
def k3_kda_attn_fused_kernel(
    tma_w: cutlass.GridConstant[
        cuda.TensorMap
    ],  # W [3208, 7168] bf16, 5-D, box 64 cols x 64 rows x 2 chunks
    tma_x: cutlass.GridConstant[cuda.TensorMap],  # x [8, 7168] bf16, box 64 cols x 8 rows
    tma_wfb: cutlass.GridConstant[
        cuda.TensorMap
    ],  # W_fb [768, 128] bf16, a head's 128 rows by one call
    p1: cutlass.Array,  # int16 bits of bf16 [3][8][1664] (Lamport, written by the stream role)
    p1w: cutlass.Array,  # the same memory as int32 words (polled by the verify role)
    part: cutlass.Array,  # int32 bits of fp32 [3][3][2][8][768] (Lamport)
    epoch: cutlass.Array,  # int32 [128]: each CTA's buffer index (launches completed mod 3)
    w_q: cutlass.Array,  # fp32 [768, 4]
    w_k: cutlass.Array,
    w_v: cutlass.Array,
    a_log: cutlass.Array,  # fp32 [6]
    dt_bias: cutlass.Array,  # fp32 [768]
    onorm_w: cutlass.Array,  # fp32 [128]
    cs_q: cutlass.Array,  # fp32 [pool][10][768]: raw conv inputs, column s = position s - 2 from the golden token
    cs_k: cutlass.Array,
    cs_v: cutlass.Array,
    ssm: cutlass.Array,  # fp32 [pool][6][128][128]: the state after the last golden token
    state_tok: cutlass.Array,  # fp32 [pool][7][6][128][128]: the state after each draft of the last round
    slots: cutlass.Array,  # int32 [1]
    pending: cutlass.Array,  # int32 [pool]: drafts the sampler accepted last round
    out: cutlass.Array,  # bf16 [8][6][128]
    ssm_stride: cutlass.Int64,  # fp32 elements between the pool's slots
    pool_n: cutlass.Int32,  # the pool's slots (entries of pending)
    lower_bound: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    eps: cutlass.Constexpr[float],
    USE_PDL: cutlass.Constexpr[bool],
):
    # The stream CTAs (clusters 0-25) and the verify CTAs (26-31) never share a cluster, so their large arrays alias
    # one raw buffer and the stream's weight ring takes the bytes the verify buffers need elsewhere.
    raw = SmemAllocator().allocate(FUSED_RAW_BYTES, byte_alignment=1024)
    stream_carve = _SmemCarver(raw, FUSED_RAW_BYTES)
    ring_w = stream_carve.view(cutlass.BFloat16, FUSED_STAGES * BOX_ELEMS, 1024)
    ring_x = stream_carve.view(cutlass.BFloat16, FUSED_STAGES * X_ELEMS, 1024)
    mbox = stream_carve.view(cutlass.Float32, 3 * CLUSTER * 32 * 4, 16)
    verify_carve = _SmemCarver(raw, FUSED_RAW_BYTES)
    smem_a = verify_carve.view(cutlass.BFloat16, HD * HD, 1024)
    smem_b = verify_carve.view(cutlass.Int32, MMA_N * HD // 2, 1024)
    s_uq = verify_carve.view(cutlass.Float32, ROWS_U * HD)
    s_uk = verify_carve.view(cutlass.Float32, ROWS_U * HD)
    s_uv = verify_carve.view(cutlass.Float32, ROWS_U * V_CTA)
    s_wq = verify_carve.view(cutlass.Float32, CONV_W * HD)
    s_wk = verify_carve.view(cutlass.Float32, CONV_W * HD)
    s_wv = verify_carve.view(cutlass.Float32, CONV_W * V_CTA)
    s_dtb = verify_carve.view(cutlass.Float32, HD)
    s_onw = verify_carve.view(cutlass.Float32, V_CTA)
    s_gr = verify_carve.view(cutlass.Float32, NT * HD)
    s_braw = verify_carve.view(cutlass.Float32, NT)
    s_og = verify_carve.view(cutlass.Float32, NT * V_CTA)
    s_q = verify_carve.view(cutlass.Float32, NT * HD)
    s_k = verify_carve.view(cutlass.Float32, NT * HD)
    s_dec = verify_carve.view(cutlass.Float32, NT * HD)
    s_kd = verify_carve.view(cutlass.Float32, NT * HD)
    s_bk = verify_carve.view(cutlass.Float32, NT * HD)
    s_beta = verify_carve.view(cutlass.Float32, NT)
    s_v = verify_carve.view(cutlass.Float32, NT * V_CTA)
    s_o = verify_carve.view(cutlass.Float32, NT * V_CTA)
    s_ss = verify_carve.view(cutlass.Float32, 2 * CLUSTER * NT)
    s_rs = verify_carve.view(cutlass.Float32, NT)
    s_vp = verify_carve.view(cutlass.Float32, 2 * NT * V_CTA)
    s_ogp = verify_carve.view(cutlass.Float32, 2 * NT * V_CTA)
    s_bp = verify_carve.view(cutlass.Float32, 2 * NT)
    s_rec = verify_carve.view(cutlass.Float32, NUM_SPEC * REC_CTA)
    # Barriers and the TMEM holder stay static (both roles initialize their own).
    full = cutlass.Array(cutlass.Int64, FUSED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    empty = cutlass.Array(cutlass.Int64, FUSED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    acc_done = cutlass.Array(cutlass.Int64, 3, space=cutlass.AddressSpace.smem, alignment=8)
    mbox_bar = cutlass.Array(cutlass.Int64, 3, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_holder = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    bars = cutlass.Array(cutlass.Int64, 3, space=cutlass.AddressSpace.smem, alignment=8)
    r_st = cutlass.Array(cutlass.Float32, REC_REGS, space=cutlass.AddressSpace.rmem)

    # Launch order by cluster: the f_a / b stream clusters, the verify clusters, then the q/k/v/og stream clusters.
    # The CTAs of the first clusters become resident while the predecessor still holds SMs, so every verify CTA
    # runs its pre-wait prologue (state, records, the drafts' replay) before the grid dependency instead of after
    # it; the stream clusters that come last load their boxes after it. Roles use the logical index:
    # stream CTAs 0-103 (the f_a / b clusters first), verify CTAs 104-127.
    pbx, _, _ = cute.arch.block_idx()
    lbx_head = pbx + cutlass.Int32((STREAM_CLUSTERS - 2) * CLUSTER)
    lbx_qkv = pbx - cutlass.Int32(HEAD_CLUSTERS * CLUSTER)
    lbx = cutlass.Int32(
        cutlass.select_(
            pbx < cutlass.Int32(2 * CLUSTER),
            pbx,
            cutlass.select_(pbx < cutlass.Int32((2 + HEAD_CLUSTERS) * CLUSTER), lbx_head, lbx_qkv),
        )
    )
    if lbx < cutlass.Int32(STREAM_CLUSTERS * CLUSTER):
        _stream_role(tma_w, tma_x, p1, part, epoch, ring_w, ring_x, full, empty, acc_done, mbox_bar, mbox, tmem_holder,
                     USE_PDL, FUSED_STAGES, lbx)  # fmt: skip
    else:
        _head_role(tma_wfb, p1w, part, w_q, w_k, w_v, a_log, dt_bias, onorm_w, cs_q, cs_k, cs_v, ssm, state_tok,
                   slots, pending, out, epoch, smem_a, smem_b, bars, tmem_holder, s_uq, s_uk, s_uv, s_wq, s_wk, s_wv,
                   s_dtb, s_onw, s_gr, s_braw, s_og, s_q, s_k, s_dec, s_kd, s_bk, s_beta, s_v, s_o, s_ss, s_rs, s_vp,
                   s_ogp, s_bp, s_rec, r_st, ssm_stride, pool_n, lower_bound, scale, eps, USE_PDL, lbx)  # fmt: skip


@cute.jit
def k3_kda_attn(
    w: cute.Tensor,  # bf16 [3208, 7168]
    x: cute.Tensor,  # bf16 [8, 7168]
    w_fb: cute.Tensor,  # bf16 [768, 128]
    p1: cute.Tensor,  # int16 [3 * 8 * 1664]
    p1w: cute.Tensor,  # the same memory as int32
    part: cute.Tensor,  # int32 [3 * 3 * 2 * 8 * 768]
    epoch: cute.Tensor,  # int32 [128]
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    onorm_w: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    ssm: cute.Tensor,
    state_tok: cute.Tensor,
    slots: cute.Tensor,
    pending: cute.Tensor,
    out: cute.Tensor,
    ssm_stride: cutlass.Int64,
    lower_bound: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    eps: cutlass.Constexpr[float],
    USE_PDL: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """One launch of the fused KDA projection + verify for one request of 8 verify tokens (stage B2)."""
    tma_w = cuda.create_tensor_map_tiled(
        global_address=w.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[BOX_K, PROJ_ROWS, K_IN // BOX_K, 1, 1],
        global_strides=[(K_IN * 2) // 16, (BOX_K * 2) // 16, (PROJ_ROWS * K_IN * 2) // 16,
                        (PROJ_ROWS * K_IN * 2) // 16],
        box_dims=[BOX_K, TILE, BOX_CH, 1, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )  # fmt: skip
    tma_x = cuda.create_tensor_map_tiled(
        global_address=x.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[K_IN, NT],
        global_strides=[(K_IN * 2) // 16],
        box_dims=[BOX_K, MMA_N],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_wfb = cuda.create_tensor_map_tiled(
        global_address=w_fb.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[BOX_K, HK, HD // BOX_K, 1, 1],
        global_strides=[(HD * 2) // 16, (BOX_K * 2) // 16, (HK * HD * 2) // 16, (HK * HD * 2) // 16],
        box_dims=[BOX_K, HD, HD // BOX_K, 1, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )  # fmt: skip
    k3_kda_attn_fused_kernel(
        tma_w, tma_x, tma_wfb, p1, p1w, part, epoch, w_q, w_k, w_v, a_log, dt_bias, onorm_w, cs_q, cs_k, cs_v, ssm,
        state_tok, slots, pending, out, ssm_stride, cutlass.Int32(cute.size(pending)), lower_bound, scale, eps, USE_PDL,
    ).launch(
        grid=((STREAM_CLUSTERS + HEAD_CLUSTERS) * CLUSTER, 1, 1), block=(THREADS, 1, 1), cluster=(CLUSTER, 1, 1),
        stream=stream, use_pdl=USE_PDL,
        # One CTA per SM (the shared-memory carve); without it ptxas may pick an occupancy-driven register target.
        min_blocks_per_mp=1,
    )  # fmt: skip
