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
# =============================================================================
# Kimi K3 MoE front for decode, one CTM (prims/cute) kernel per MoE layer, M <= 8 tokens:
#
#   head   = x @ [latent-down slice; router slice]^T       (fp32; this rank's 3584/W latent columns, 896/W logits)
#   gather = every rank's head slice, through the multicast mapping of the head workspace (Lamport buffers)
#   route  = top-16 of sigmoid(logits) + bias per token; MXFP8 latent + UE8M0 scales per token
#   shared = SiTU(bf16(x @ gate^T)) * SiTU_lin(bf16(x @ up^T))  (the shared experts' gate_up and activation)
#
# One weight W [N, K = 7168] bf16 in 128-row tiles: the head rows (latent slice, then router slice, then zero rows
# up to a whole tile), then the shared gate_up rows re-ordered so that every 32-row slice holds 16 gate rows and
# the 16 up rows of the same columns (`front_weight`).
#
# Geometry (the long-K CTM decode GEMV): clusters of SPLIT = 8 CTAs; a GEMV cluster owns one 128-row tile per round
# (at most three rounds), rank r streaming k-tiles r, r + 8, ... through a RING-stage ring filled, with every later
# k-tile of every round prefetched into L2, before any wait; x's k-tiles of the rank stay resident. tcgen05 MMA
# M 128, N 8, K 16 into a TMEM accumulator per round. Split-K: rank w < 4 owns rows [32 w, 32 w + 32) of the tile;
# the other ranks write their fp32 partials of those rows into slot [rank] of the owner's round mailbox by st.async,
# which completes the mailbox barrier by bytes; the owner, which expected the bytes at setup, spins on test_wait (a
# warp suspended in try_wait is woken late by remote completions) and adds the 8 partials in rank order. Then, by
# tile kind:
#   head latent rows -> bf16 pairs (cvt.rn.bf16x2.f32, -0.0 halves -> +0.0), 16-byte vectors pushed into slot
#       [buffer][token][rank] of every rank's head buffer (the head all-gather's packing: see k3_route_quant_ag.py);
#   head router rows skip the owner: every rank of the cluster pushes its own fp32 partial (-0.0 -> +0.0) into
#       [buffer][token][rank][q = cluster rank] of the partials region after the buffers, and the route CTAs add the
#       8 partials in rank order from +0.0 (the owner's sum, bit for bit); no fence after the pushes, the readers poll
#       the words themselves;
#   shared rows -> gate and up of one column in lanes l and l + 16: bf16(SiTU(bf16 gate) * SiTU_lin(bf16 up)).
# Role clusters, the first two of the grid (resident before any GEMV cluster): CTA t < M routes token t, CTA M + t
# quantizes it, k3_route_quant_ag's device code minus its push (the GEMV epilogues push), the route CTA selecting
# with all its warps (_top16_cta) instead of top16_warp's one warp. Every CTA that reads the
# buffer index (the 2 M role CTAs and every GEMV CTA) signs in on flags[3] right after the read; thread 0 of token 0's
# quantization CTA waits for all of them and flips the index for the next call, long before the data it polls arrives,
# so no count sits on the role CTAs' output path. With `publish`, per-token ready words as route_quant_ag's.
#
# Warps of a GEMV CTA: 0 weight TMA (+ early dependent trigger), 1 grid-dependency wait, head workspace flags,
# activation TMA, 2 TMEM allocation + MMA, 3 idle, 4-7 epilogue, 8-13 exit after setup. Role CTAs: all 14 warps
# (the quantization CTA holds one 8-element vector per thread).
# =============================================================================
"""Kimi K3 MoE front (head GEMV + all-gather + routing + MXFP8, shared gate_up + SiTU) as one kernel."""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass import dsl_user_op
from cutlass.experimental import primitives as prims

from tensorrt_llm._torch.cute_dsl_kernels.k3_route_quant import k3_route_quant_kernel as _rq

from . import k3_route_quant_ag as _ag

NUM_EXPERTS = 896
TOP_K = 16
HIDDEN_SIZE = 3584  # the routed latent
MODEL_HIDDEN = 7168  # K
SF_VEC_SIZE = 32
M_MAX = 8
EMPTY_WORD = _ag.EMPTY_WORD

CTA_M = 128
HALF_M = 64  # a head half-tile's rows (one round, e.g. TP16: every head k-tile of the CTA on chip before the wait)
MAX_HALF_TILES = 8  # k-tiles a head half-tile CTA can hold (its full-barrier count)
MMA_N = 8
CTA_K = 128
MMA_K = 16
TMA_K_BOX = 64
TMA_COPY_ITERS = CTA_K // TMA_K_BOX
K_BLOCKS_PER_HALF = TMA_K_BOX // MMA_K
SPLIT = 8
assert SPLIT == _ag.FRONT_SPLIT  # the router partials region is sized by the all-gather's layout
OWNERS = 4  # ranks owning 32-row slices of a tile (epilogue warps 4-7 hold TMEM lanes 32 w ..)
ROWS_PER_WARP = 32
MAX_ROUNDS = 3
ROLE_CLUSTERS = 2
THREADS = HIDDEN_SIZE // 8  # 448: one 8-element latent vector per thread in a quantization CTA
TMEM_COLS = 32  # one 8-column accumulator per round (3 x 8 columns used)
# Head half-tile CTAs (head_half) copy every A k-tile into TMEM before the grid wait and run their MMAs from there
# (A from shared memory costs ~4 KB of smem reads per M 128 K 16 MMA): k-tile i at columns A_TMEM_BASE + 64 i,
# 8 columns per K = 16 step, all of TMEM.
A_TMEM_BASE = 64
A_TMEM_COLS = CTA_K // 2
A_KSTEP_COLS = MMA_K // 2
HALF_TMEM_COLS = 512
ELEM_BYTES = 2
EVICT_FIRST = 0x12F0000000000000
# The route CTA's top-16 (_top16_cta): candidate slots warp 0 ranks (more candidates, from heavy ties, take
# _rq.top16_warp), an empty slot (below every candidate), the id complement that makes lower ids rank first.
CAND_SLOTS = 32
CAND_EMPTY = -(2**63)
NO_ID = 0x7FFFFFFF
KEY_MAX = 2**31 - 1

LEADING = 16
STRIDE = 8 * TMA_K_BOX * ELEM_BYTES
A_HALF_ELEMS = CTA_M * TMA_K_BOX
B_HALF_ELEMS = MMA_N * TMA_K_BOX
STEP = (MMA_K * ELEM_BYTES) >> 4
A_BOX = A_HALF_ELEMS >> 3
B_BOX = B_HALF_ELEMS >> 3
STAGE_A = (CTA_M * CTA_K * ELEM_BYTES) >> 4
STAGE_AH = (HALF_M * CTA_K * ELEM_BYTES) >> 4  # a head half-tile's k-tile (16 KB)
A_BOX_H = (HALF_M * TMA_K_BOX) >> 3  # its 64-element K box (8 KB)
STAGE_B = (MMA_N * CTA_K * ELEM_BYTES) >> 4

io_dtype = cutlass.BFloat16


def head_rows(world: int) -> int:
    return HIDDEN_SIZE // world + NUM_EXPERTS // world


def head_tiles(world: int) -> int:
    return (head_rows(world) + CTA_M - 1) // CTA_M


def geometry(world: int, shared_cols: int, max_clusters: int) -> tuple[int, int, int]:
    """(head tiles, shared tiles, GEMV clusters) for this TP world and shared activation width.

    ``max_clusters``: how many clusters of SPLIT CTAs (one CTA per SM) the GPU holds at once. The role clusters and
    every GEMV cluster must fit together: the role clusters stay until the routing is done, so a GEMV cluster that
    does not fit waits for them (a 152-SM GB200 holds 15, not the 16 its 8 GPCs suggest).
    """
    ht = head_tiles(world)
    st = shared_cols // 64
    tiles = ht + st
    clusters = min(tiles, max(1, max_clusters - ROLE_CLUSTERS))
    return ht, st, clusters


def half_geometry(world: int, shared_cols: int, max_clusters: int, k_in: int, ring: int):
    """(head half-tiles, shared tiles, GEMV clusters) when the head fits as 64-row half-tiles in one round next to the
    shared tiles, each head CTA holding all its k-tiles in shared memory (the ring's space); else None."""
    hh = (head_rows(world) + HALF_M - 1) // HALF_M
    st = shared_cols // 64
    my_tiles = k_in // CTA_K // SPLIT
    fits = (
        hh + st <= max_clusters - ROLE_CLUSTERS
        and my_tiles * HALF_M * CTA_K + HALF_M * TMA_K_BOX <= ring * CTA_M * CTA_K
        and my_tiles <= MAX_HALF_TILES
        and A_TMEM_BASE + my_tiles * A_TMEM_COLS <= HALF_TMEM_COLS
    )
    return (hh, st, hh + st) if fits else None


def _weight_rows(world: int, n_head_tiles: int, n_tiles: int, head_half: bool) -> int:
    """Rows of the front weight: the head padded to 128-row tiles, then the shared tiles."""
    if not head_half:
        return n_tiles * CTA_M
    return (head_tiles(world) + n_tiles - n_head_tiles) * CTA_M


def _half_plan(
    head_half: bool, world: int, n_head_tiles: int, ring: int, my_tiles: int
) -> tuple[int, int]:
    """(weight-row shift of the shared tiles, weight full/empty barriers) for the kernel's tile mode."""
    if not head_half:
        return 0, ring
    return (head_tiles(world) - n_head_tiles) * CTA_M, max(ring, my_tiles)


def supports(world: int, shared_cols: int, max_clusters: int, k_in: int, ring: int) -> bool:
    ht, st, clusters = geometry(world, shared_cols, max_clusters)
    k_tiles = k_in // CTA_K
    return (
        world in (4, 8, 16)
        and k_in % (CTA_K * SPLIT) == 0
        and shared_cols % 64 == 0
        and ht + st <= MAX_ROUNDS * clusters
        and 1 <= ring <= k_tiles // SPLIT
        and ring * CTA_M * CTA_K * ELEM_BYTES + (k_tiles // SPLIT) * MMA_N * CTA_K * ELEM_BYTES
        <= 176 * 1024
    )


def _situ(gate, up, gate_cap: float, linear_cap: float):
    """SiTU(gate) * SiTU_lin(up) in fp32 (the fused MoE's device functions)."""
    log2e = 1.4426950408889634

    def tanh_f32(v):
        e = cute.math.exp2(cute.math.abs(v) * cutlass.Float32(-2.0 * log2e), fastmath=True)
        t = (cutlass.Float32(1.0) - e) * cute.arch.rcp_approx(cutlass.Float32(1.0) + e)
        return cutlass.select_(v < cutlass.Float32(0.0), -t, t)

    sig = cute.arch.rcp_approx(
        cutlass.Float32(1.0) + cute.math.exp2(gate * cutlass.Float32(-log2e), fastmath=True)
    )
    g = cutlass.Float32(gate_cap) * tanh_f32(gate * cutlass.Float32(1.0 / gate_cap)) * sig
    u = cutlass.Float32(linear_cap) * tanh_f32(up * cutlass.Float32(1.0 / linear_cap))
    return g * u


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
def _st_async_v4_f32(dst, a, b, c, d, mbar, *, loc=None, ip=None):
    """st.async of four fp32 to a shared::cluster address, completing ``mbar`` (shared::cluster) by 16 bytes."""
    from cutlass._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip), cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip), cutlass.Float32(c).ir_value(loc=loc, ip=ip),
         cutlass.Float32(d).ir_value(loc=loc, ip=ip), cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [$0], {$1, $2, $3, $4}, [$5];", "r,f,f,f,f,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _test_wait_cluster(mbar, parity, *, loc=None, ip=None):
    """mbarrier.test_wait.parity.acquire.cluster on a barrier of this CTA (``mbar``, its shared-memory pointer): whether
    phase ``parity`` has completed, acquiring at cluster scope. The barrier is completed by other CTAs' st.async, whose
    complete_tx releases at cluster scope."""
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


@cute.jit
def _poll_logit_parts(arr, idx, stride: cutlass.Constexpr[int]):
    """Spin (no back-off) until none of the SPLIT 4-word vectors at idx + q * stride is empty, every sweep loading all
    of them at once. Returns the 4 logits, each the fp32 sum of its partials in rank order from +0.0 (the owner's
    order; a -0.0 partial pushed as +0.0 cannot change a sum that starts at +0.0), -0.0 as +0.0 (the owner's push)."""
    empty = cutlass.Int32(EMPTY_WORD)
    s0 = empty
    s1 = empty
    s2 = empty
    s3 = empty
    pending = cutlass.Boolean(True)
    while pending:
        vecs = []
        for q in cutlass.range_constexpr(SPLIT):
            vecs.append(
                arr.load(
                    idx=idx + cutlass.Int32(q * stride),
                    vector_size=4,
                    alignment=16,
                    is_volatile=True,
                )
            )
        missing = cutlass.Boolean(False)
        totals = [cutlass.Float32(0.0)] * 4
        for q in cutlass.range_constexpr(SPLIT):
            for j in cutlass.range_constexpr(4):
                word = cutlass.Int32(vecs[q][j])
                missing = missing | (word == empty)
                totals[j] = totals[j] + word.bitcast(cutlass.Float32)
        s0 = _ag._sanitize_f32(totals[0].bitcast(cutlass.Int32))
        s1 = _ag._sanitize_f32(totals[1].bitcast(cutlass.Int32))
        s2 = _ag._sanitize_f32(totals[2].bitcast(cutlass.Int32))
        s3 = _ag._sanitize_f32(totals[3].bitcast(cutlass.Int32))
        pending = missing
    return s0, s1, s2, s3


def _bf16_rn(v):
    """fp32 -> bf16 precision (round to nearest even), kept as fp32, in the integer domain."""
    u = v.bitcast(cutlass.Int32)
    u = u + (((u >> 16) & 1) + 0x7FFF)
    u = (u >> 16) << 16
    return u.bitcast(cutlass.Float32)


@dsl_user_op
def _atom_add_shared(addr_u32, val, *, loc=None, ip=None):
    """atom.shared.add.u32; returns the old value."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Int32(addr_u32).ir_value(loc=loc, ip=ip), cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
            "atom.shared.add.u32 $0, [$1], $2;", "=r,r,r", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _add_opaque(a, b, *, loc=None, ip=None):
    """add.s32 in inline PTX: the compiler cannot re-associate it, so a tree of them stays a tree (LLVM turns a
    tree of plain adds of 0/1 selects into one dependent chain of conditional increments)."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Int32(a).ir_value(loc=loc, ip=ip), cutlass.Int32(b).ir_value(loc=loc, ip=ip)],
            "add.s32 $0, $1, $2;", "=r,r,r", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


def _tree_sum(vals):
    """Sum of a list of Int32 as a balanced tree of independent adds."""
    while len(vals) > 1:
        vals = [_add_opaque(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)] + (
            [vals[-1]] if len(vals) % 2 else []
        )
    return vals[0]


@cute.jit
def _top16_cta(
    s_key, s_sigmoid, s_top2, s_cnt, s_cand, s_csig, s_win, s_wsig, tx, routed_scaling_factor
):
    """Top-16 of one token by all THREADS threads of the route CTA: the experts, their order and the weight bits of
    ``_rq.top16_warp`` (key descending, ties to the lower expert id; the weights from the same fp32 sum and fp64
    division), in two CTA barriers instead of one warp's 16 dependent redux rounds.

    1. Thread t holds the keys of experts t and t + THREADS; each warp takes its two largest keys (two experts).
    2. Every warp computes B, the 16th largest of those 28 keys: at least 16 experts have a key >= B, so the 16th
       largest key of all is >= B and every winner has a key >= B.
    3. The C experts with a key >= B (C >= 16; ~16-30 for router keys) go to s_cand as (key << 32) | (NO_ID - id),
       whose signed order is the selection order, and their sigmoids to s_csig; one shared atomic per warp hands out
       the slots.
    4. Warp 0 ranks the candidates (lane i: how many beat candidate i). Every expert that beats a winner is a
       candidate, so a winner's rank is exact; the rank-r winner goes to slot r. With C > CAND_SLOTS (heavy ties),
       warp 0 runs ``_rq.top16_warp`` instead.
    Returns (expert id, weight bf16 bits) in lanes 0-15 of warp 0 (the rank-lane expert); other lanes hold garbage.
    Shared memory: s_top2 [32] (warp w's keys at w and 16 + w), s_cnt [1], s_cand [CAND_SLOTS] (Int64), s_csig
    [CAND_SLOTS], s_win and s_wsig [16]; every word is written and read between this call's barriers and the caller's
    barriers around it. Counts are balanced trees of adds: a chain of 28 or 32 dependent adds cost ~0.25 us each."""
    lane = tx % cutlass.Int32(32)
    warp = tx // cutlass.Int32(32)
    key_a = s_key.load(idx=tx)
    key_b = s_key.load(idx=tx + cutlass.Int32(THREADS))
    sig_a = s_sigmoid.load(idx=tx)
    sig_b = s_sigmoid.load(idx=tx + cutlass.Int32(THREADS))
    a_high = key_a > key_b
    hi = cutlass.Int32(cutlass.select_(a_high, key_a, key_b))
    lo = cutlass.Int32(cutlass.select_(a_high, key_b, key_a))
    top1 = prims.redux_sync(hi, prims.ReductionKind.MAX, _rq.FULL_MASK)
    holders = prims.vote_sync(_rq.FULL_MASK, hi == top1, prims.VoteSync.BALLOT)
    # The lowest lane holding the largest key gives it up for its other key.
    gives = (cutlass.Int32(1) << lane) == (holders & (cutlass.Int32(0) - holders))
    top2 = prims.redux_sync(
        cutlass.Int32(cutlass.select_(gives, lo, hi)), prims.ReductionKind.MAX, _rq.FULL_MASK
    )
    if lane == cutlass.Int32(0):
        s_top2.store(top1, idx=warp)
        s_top2.store(top2, idx=warp + cutlass.Int32(16))
    if tx == cutlass.Int32(0):
        s_cnt.store(cutlass.Int32(0), idx=0)
    if tx < cutlass.Int32(CAND_SLOTS):
        s_cand.store(cutlass.Int64(CAND_EMPTY), idx=tx)
    cute.arch.barrier()

    # B: the smallest of the 28 keys with at most 15 of them above it.
    valid = (lane % cutlass.Int32(16)) < cutlass.Int32(THREADS // 32)
    mine = cutlass.Int32(cutlass.select_(valid, s_top2.load(idx=lane), cutlass.Int32(KEY_MAX)))
    greater = []
    for i in cutlass.range_constexpr(8):
        quad = s_top2.load(idx=cutlass.Int32(4 * i), vector_size=4, alignment=16)
        for q in cutlass.range_constexpr(4):
            if cutlass.const_expr((4 * i + q) % 16 < THREADS // 32):
                greater.append(cutlass.Int32(cutlass.select_(cutlass.Int32(quad[q]) > mine, 1, 0)))
    above = _tree_sum(greater)
    bound = prims.redux_sync(
        cutlass.Int32(
            cutlass.select_(above <= cutlass.Int32(TOP_K - 1), mine, cutlass.Int32(KEY_MAX))
        ),
        prims.ReductionKind.MIN,
        _rq.FULL_MASK,
    )
    in_a = key_a >= bound
    in_b = key_b >= bound
    ballot_a = prims.vote_sync(_rq.FULL_MASK, in_a, prims.VoteSync.BALLOT)
    ballot_b = prims.vote_sync(_rq.FULL_MASK, in_b, prims.VoteSync.BALLOT)
    count_a = cute.arch.popc(ballot_a)
    count_w = count_a + cute.arch.popc(ballot_b)
    first_slot = cutlass.Int32(0)
    if lane == cutlass.Int32(0):
        if count_w > cutlass.Int32(0):
            first_slot = _atom_add_shared(s_cnt.data_ptr().toint(), count_w)
    first_slot = cute.arch.shuffle_sync(first_slot, 0)
    below = cutlass.Int32(cute.arch.lanemask_lt())
    if in_a:
        slot_a = first_slot + cute.arch.popc(ballot_a & below)
        if slot_a < cutlass.Int32(CAND_SLOTS):
            s_cand.store(
                (cutlass.Int64(key_a) << cutlass.Int64(32))
                | cutlass.Int64(cutlass.Int32(NO_ID) - tx),
                idx=slot_a,
            )
            s_csig.store(sig_a, idx=slot_a)
    if in_b:
        slot_b = first_slot + count_a + cute.arch.popc(ballot_b & below)
        if slot_b < cutlass.Int32(CAND_SLOTS):
            s_cand.store(
                (cutlass.Int64(key_b) << cutlass.Int64(32))
                | cutlass.Int64(cutlass.Int32(NO_ID) - tx - cutlass.Int32(THREADS)),
                idx=slot_b,
            )
            s_csig.store(sig_b, idx=slot_b)
    cute.arch.barrier()

    n_cand = s_cnt.load(idx=0)
    expert = cutlass.Int32(0)
    weight_bits = cutlass.Int16(0)
    if n_cand <= cutlass.Int32(CAND_SLOTS):
        if warp == cutlass.Int32(0):
            cand = s_cand.load(idx=lane)
            cand_id = cutlass.Int32(NO_ID) - cutlass.Int32(cand & cutlass.Int64(NO_ID))
            # An empty slot (id out of range, sigmoid stale) ranks >= 16, so neither is ever used.
            cand_sig = s_csig.load(idx=lane)
            beaten = []
            for i in cutlass.range_constexpr(CAND_SLOTS // 2):
                pair = s_cand.load(idx=cutlass.Int32(2 * i), vector_size=2, alignment=16)
                for q in cutlass.range_constexpr(2):
                    beaten.append(
                        cutlass.Int32(cutlass.select_(cutlass.Int64(pair[q]) > cand, 1, 0))
                    )
            rank = _tree_sum(beaten)
            if rank < cutlass.Int32(TOP_K):
                s_win.store(cand_id, idx=rank)
                s_wsig.store(cand_sig, idx=rank)
            cute.arch.sync_warp()
            r = lane % cutlass.Int32(TOP_K)
            expert = s_win.load(idx=r)
            # top16_warp's sum: an xor butterfly over the warp (offsets 16, 8, 4, 2, 1) with lanes 16-31 at 0, which
            # leaves every lane with this tree (the partners' adds are the same fp32 adds).
            sig = []
            for i in cutlass.range_constexpr(TOP_K // 4):
                quad_s = s_wsig.load(idx=cutlass.Int32(4 * i), vector_size=4, alignment=16)
                for q in cutlass.range_constexpr(4):
                    sig.append(cutlass.Float32(quad_s[q]) + cutlass.Float32(0.0))
            sum8 = [sig[j] + sig[j + 8] for j in range(8)]
            sum4 = [sum8[j] + sum8[j + 4] for j in range(4)]
            sum2 = [sum4[j] + sum4[j + 2] for j in range(2)]
            weight = (cutlass.Float64(s_wsig.load(idx=r)) * routed_scaling_factor) / (
                cutlass.Float64(sum2[0] + sum2[1]) + cutlass.Float64(1e-20)
            )
            weight_bits = _rq.cvt_rn_bf16_f64(weight)
    else:
        if warp == cutlass.Int32(0):
            expert, weight_bits = _rq.top16_warp(s_key, s_sigmoid, lane, routed_scaling_factor)
    return expert, weight_bits


@cute.kernel
def k3_moe_front_kernel(
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],  # W [N, K] bf16, 5-D, one call per k-tile
    tma_desc_w64: cutlass.GridConstant[
        cuda.TensorMap
    ],  # the same W with a 64-row box (head half-tiles)
    tma_desc_x: cutlass.GridConstant[cuda.TensorMap],  # x [M, K] bf16, box 64 x 8
    bias: cutlass.Array,  # fp32 [896]
    buf_uc: cutlass.Array,  # int32 words of this rank's head buffers
    buf_mc: cutlass.Array,  # int32 words of their multicast mapping
    flags: cutlass.Array,  # int32 [0] buffer of this call, [1] unused (0), [2] epoch, [3] CTAs that read [0]
    ready: cutlass.Array,  # int32 [16] per-token ready words (publish)
    topk_ids: cutlass.Array,  # int32 [M * 16]
    topk_weight_bits: cutlass.Array,  # int16 view of bf16 [M * 16]
    quant_words: cutlass.Array,  # int32 view of e4m3 [M, 3584]
    scales: cutlass.Array,  # uint8 [M * 112]
    shared_bits: cutlass.Array,  # int16 view of bf16 [M, shared_cols]
    num_tokens: cutlass.Int32,
    tp_rank: cutlass.Int32,
    routed_scaling_factor: cutlass.Float64,
    world: cutlass.Constexpr[int],
    shared_cols: cutlass.Constexpr[int],
    n_head_tiles: cutlass.Constexpr[int],
    n_tiles: cutlass.Constexpr[int],
    gemv_clusters: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    gate_cap: cutlass.Constexpr[float],
    linear_cap: cutlass.Constexpr[float],
    publish: cutlass.Constexpr[bool],
    head_half: cutlass.Constexpr[bool],
):
    WL = HIDDEN_SIZE // world
    WE = NUM_EXPERTS // world
    LV = WL // 8
    EV = WE // 4
    SLOT = (LV + EV) * 4
    BUF = M_MAX * world * SLOT
    PBASE = (
        _ag.BUFFERS * BUF
    )  # the router logits' split-K partials, after the buffers: [buffer][token][rank][q][WE]
    k_tiles = k_in // CTA_K
    my_tiles = k_tiles // SPLIT
    rounds_max = (n_tiles + gemv_clusters - 1) // gemv_clusters
    # head_half (one round): clusters < n_head_tiles each own a 64-row head half-tile and hold all its k-tiles; the
    # shared tiles' weight rows start after the 128-row-padded head, hence the row shift for tiles >= n_head_tiles.
    row_shift, nbar = _half_plan(head_half, world, n_head_tiles, ring, my_tiles)

    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp_id = cute.arch.warp_idx()
    lane = tx % 32
    crank = cute.arch.block_idx_in_cluster()
    # The role clusters come first, so they are resident before any GEMV cluster (they spin; the GEMV CTAs never
    # wait on them).
    role_or_gemv = bx // cutlass.Int32(SPLIT)
    is_gemv = role_or_gemv >= cutlass.Int32(ROLE_CLUSTERS)
    cluster = role_or_gemv - cutlass.Int32(ROLE_CLUSTERS)
    tma_ptr_w = tma_desc_w.get_ptr()
    tma_ptr_w64 = tma_desc_w64.get_ptr()
    # A GEMV CTA of a head half-tile (head_half: one round, the head in 64-row tiles on the first clusters).
    head_cta = cutlass.Boolean(False)
    stream_cta = cutlass.Boolean(True)  # its k-tiles stream through the ring
    if cutlass.const_expr(head_half):
        head_cta = cluster < cutlass.Int32(n_head_tiles)
        stream_cta = cluster >= cutlass.Int32(n_head_tiles)
    tma_ptr_x = tma_desc_x.get_ptr()

    smem_a = cutlass.Array(
        io_dtype, ring * CTA_M * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_b = cutlass.Array(
        io_dtype, my_tiles * MMA_N * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    tma_full = cutlass.Array(cutlass.Int64, nbar, space=cutlass.AddressSpace.smem, alignment=8)
    mma_done = cutlass.Array(cutlass.Int64, nbar, space=cutlass.AddressSpace.smem, alignment=8)
    act_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    flags_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    acc_done = cutlass.Array(
        cutlass.Int64, MAX_ROUNDS, space=cutlass.AddressSpace.smem, alignment=8
    )
    mail_full = cutlass.Array(
        cutlass.Int64, MAX_ROUNDS, space=cutlass.AddressSpace.smem, alignment=8
    )
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    # [round][source rank][row of the owned slice][token] fp32 partials.
    mailbox = cutlass.Array(
        cutlass.Float32,
        MAX_ROUNDS * SPLIT * ROWS_PER_WARP * MMA_N,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    s_flags = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    # The owner warp's reduced rows [row][token], for the vector pushes.
    s_rows = cutlass.Array(
        cutlass.Float32,
        OWNERS * ROWS_PER_WARP * MMA_N,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    s_key = cutlass.Array(cutlass.Int32, NUM_EXPERTS, space=cutlass.AddressSpace.smem, alignment=16)
    s_sigmoid = cutlass.Array(
        cutlass.Float32, NUM_EXPERTS, space=cutlass.AddressSpace.smem, alignment=16
    )
    # _top16_cta's scratch (route CTAs).
    s_top2 = cutlass.Array(cutlass.Int32, 32, space=cutlass.AddressSpace.smem, alignment=16)
    s_cnt = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    s_cand = cutlass.Array(cutlass.Int64, CAND_SLOTS, space=cutlass.AddressSpace.smem, alignment=16)
    s_csig = cutlass.Array(
        cutlass.Float32, CAND_SLOTS, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_win = cutlass.Array(cutlass.Int32, TOP_K, space=cutlass.AddressSpace.smem, alignment=16)
    s_wsig = cutlass.Array(cutlass.Float32, TOP_K, space=cutlass.AddressSpace.smem, alignment=16)

    if is_gemv:
        if warp_id == 0:
            prims.prefetch_tensormap(tma_ptr_w)
            if cutlass.const_expr(head_half):
                prims.prefetch_tensormap(tma_ptr_w64)
            prims.prefetch_tensormap(tma_ptr_x)
            if prims.elect_sync():
                for s in cutlass.range_constexpr(nbar):
                    prims.mbarrier_init(tma_full.subview(s), 1)
                    prims.mbarrier_init(mma_done.subview(s), 1)
                prims.mbarrier_init(act_full, 1)
                prims.mbarrier_init(flags_ready, 1)
                for r in cutlass.range_constexpr(MAX_ROUNDS):
                    prims.mbarrier_init(acc_done.subview(r), 1)
                    # Owner ranks (< 4): the other 7 ranks' partials of the owned 32 rows x 8 tokens arrive by
                    # st.async; the phase completes on their bytes (expected here, before the cluster forms).
                    prims.mbarrier_init(mail_full.subview(r), 1)
                    prims.mbarrier_arrive_expect_tx(
                        mail_full.subview(r), (SPLIT - 1) * ROWS_PER_WARP * MMA_N * 4
                    )
        if warp_id == 2:
            prims.tcgen05_alloc(tmem_ptr_i32, HALF_TMEM_COLS if head_half else TMEM_COLS)
            prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    # Cluster formation: the peers' shared memory and barriers are addressable from here on.
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    if is_gemv:
        tmem_ptr = cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32)
        if warp_id == 0:
            # =================================================================
            # Weight TMA: fill the ring, prefetch every later k-tile of both
            # rounds into L2, all before any wait; then refill as MMAs finish.
            # =================================================================
            if prims.elect_sync():
                if head_cta:
                    # A head half-tile: every k-tile of the rank, 64-row boxes 16 KB apart; nothing loads later.
                    mh = cluster * cutlass.Int32(HALF_M)
                    for hi in cutlass.range_constexpr(my_tiles):
                        kh = crank + cutlass.Int32(hi * SPLIT)
                        prims.mbarrier_arrive_expect_tx(
                            tma_full.subview(hi), HALF_M * CTA_K * ELEM_BYTES
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_a.subview(hi * HALF_M * CTA_K),
                            tma_ptr_w64,
                            (cutlass.Int32(0), mh, kh * cutlass.Int32(TMA_COPY_ITERS), cutlass.Int32(0),
                             cutlass.Int32(0)),
                            tma_full.subview(hi),
                            l2_cache_hint=EVICT_FIRST,
                        )  # fmt: skip
                else:
                    m0 = cluster * cutlass.Int32(CTA_M) + cutlass.Int32(row_shift)
                    for i in cutlass.range_constexpr(ring):
                        k = crank + cutlass.Int32(i * SPLIT)
                        prims.mbarrier_arrive_expect_tx(
                            tma_full.subview(i), CTA_M * CTA_K * ELEM_BYTES
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_a.subview(i * CTA_M * CTA_K),
                            tma_ptr_w,
                            (
                                cutlass.Int32(0),
                                m0,
                                k * cutlass.Int32(TMA_COPY_ITERS),
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                            ),
                            tma_full.subview(i),
                            l2_cache_hint=EVICT_FIRST,
                        )
                    for rd in cutlass.range_constexpr(rounds_max):
                        tile = cluster + cutlass.Int32(rd * gemv_clusters)
                        if tile < cutlass.Int32(n_tiles):
                            first = ring if rd == 0 else 0
                            for i in range(first, my_tiles):
                                k = crank + i * cutlass.Int32(SPLIT)
                                prims.cp_async_bulk_tensor_prefetch(
                                    tma_ptr_w,
                                    [cutlass.Int32(0), tile * cutlass.Int32(CTA_M) + cutlass.Int32(row_shift),
                                     k * cutlass.Int32(TMA_COPY_ITERS), cutlass.Int32(0), cutlass.Int32(0)],
                                    [],
                                )  # fmt: skip
            # Dependents may launch now; they wait for this whole grid before reading its outputs.
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
            if prims.elect_sync():
                if stream_cta:
                    stage = cutlass.Int32(0)
                    phase = cutlass.Int32(0)
                    seq = cutlass.Int32(0)  # k-tiles issued so far, over the rounds
                    for rd in cutlass.range_constexpr(rounds_max):
                        tile = cluster + cutlass.Int32(rd * gemv_clusters)
                        if tile < cutlass.Int32(n_tiles):
                            for i in range(my_tiles):
                                if seq >= cutlass.Int32(ring):
                                    while not cute.arch.mbarrier_try_wait(
                                        mma_done.subview(stage).data_ptr(), phase
                                    ):
                                        pass
                                    k = crank + i * cutlass.Int32(SPLIT)
                                    prims.mbarrier_arrive_expect_tx(
                                        tma_full.subview(stage), CTA_M * CTA_K * ELEM_BYTES
                                    )
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        smem_a.subview(stage * cutlass.Int32(CTA_M * CTA_K)),
                                        tma_ptr_w,
                                        (cutlass.Int32(0), tile * cutlass.Int32(CTA_M) + cutlass.Int32(row_shift),
                                         k * cutlass.Int32(TMA_COPY_ITERS),
                                         cutlass.Int32(0), cutlass.Int32(0)),
                                        tma_full.subview(stage),
                                        l2_cache_hint=EVICT_FIRST,
                                    )  # fmt: skip
                                seq = seq + cutlass.Int32(1)
                                stage = stage + cutlass.Int32(1)
                                if stage == cutlass.Int32(ring):
                                    stage = cutlass.Int32(0)
                                    if seq > cutlass.Int32(ring):
                                        phase = phase ^ cutlass.Int32(1)
        elif warp_id == 1:
            # =================================================================
            # Grid dependency, the head workspace's flags, then x's k-tiles.
            # =================================================================
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(act_full, my_tiles * MMA_N * CTA_K * ELEM_BYTES)
                for i in range(my_tiles):
                    k = crank + i * cutlass.Int32(SPLIT)
                    for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_b.subview(
                                i * cutlass.Int32(MMA_N * CTA_K)
                                + cutlass.Int32(half * B_HALF_ELEMS)
                            ),
                            tma_ptr_x,
                            (
                                k * cutlass.Int32(CTA_K) + cutlass.Int32(half * TMA_K_BOX),
                                cutlass.Int32(0),
                            ),
                            act_full,
                        )
                # The buffer index is for the epilogue's pushes, microseconds later: read it once x's loads are
                # issued, so they do not wait for its round trip.
                s_flags.store(flags.load(idx=0, is_volatile=True), idx=0)
                _red_add_release(flags.data_ptr(3).toint(), cutlass.Int32(1))
                prims.mbarrier_arrive(flags_ready)
        elif warp_id == 2:
            # =================================================================
            # MMA: per round, the rank's k-tiles into that round's accumulator.
            # =================================================================
            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=cutlass.Float32,
                a_dtype=io_dtype,
                b_dtype=io_dtype,
                n_dim=MMA_N,
                m_dim=CTA_M,
            )
            desc_a_base = prims.Tcgen05SmemDesc.build(
                start_address=smem_a, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )  # fmt: skip
            desc_b_base = prims.Tcgen05SmemDesc.build(
                start_address=smem_b, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )  # fmt: skip
            if head_cta:
                # Before the grid wait: every A k-tile of the head half-tile into TMEM, one 128 x 16 copy per K = 16
                # step with the MMA's own A descriptor as the source (rows 64-127 alias the next box, as in smem).
                for hc in cutlass.range(my_tiles, unroll=1):
                    while not cute.arch.mbarrier_try_wait(tma_full.subview(hc).data_ptr(), 0):
                        pass
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    for kc in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                        desc_c = desc_a_base + (
                            hc * cutlass.Int32(STAGE_AH)
                            + cutlass.Int32(
                                (kc // K_BLOCKS_PER_HALF) * A_BOX_H
                                + (kc % K_BLOCKS_PER_HALF) * STEP
                            )
                        )
                        a_dst = cutlass.inttoptr(
                            tmem_ptr_i32.load()
                            + cutlass.Int32(A_TMEM_BASE + kc * A_KSTEP_COLS)
                            + hc * cutlass.Int32(A_TMEM_COLS),
                            6,
                            cutlass.Int32,
                        )
                        if prims.elect_sync():
                            prims.tcgen05_cp(prims.Tcgen05CpShape.SHAPE_128X256B, a_dst, desc_c)
            while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
                pass
            if head_cta:
                # A head half-tile: A from TMEM (staged before the wait), B (x) from shared memory. The MMA is M 128:
                # rows 64-127 are the aliased garbage and land in accumulator rows nobody reads.
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                acc_h = cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32)
                for hi in cutlass.range(my_tiles, unroll=1):
                    for kh in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                        a_src = cutlass.inttoptr(
                            tmem_ptr_i32.load()
                            + cutlass.Int32(A_TMEM_BASE + kh * A_KSTEP_COLS)
                            + hi * cutlass.Int32(A_TMEM_COLS),
                            6,
                            cutlass.Int32,
                        )
                        desc_bh = desc_b_base + (
                            hi * cutlass.Int32(STAGE_B)
                            + cutlass.Int32(
                                (kh // K_BLOCKS_PER_HALF) * B_BOX + (kh % K_BLOCKS_PER_HALF) * STEP
                            )
                        )
                        accumulate_h = cutlass.Boolean(True)
                        if cutlass.const_expr(kh == 0):
                            accumulate_h = hi > cutlass.Int32(0)
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, acc_h, a_src, desc_bh, idesc,
                                accumulate_h,
                            )  # fmt: skip
                if prims.elect_sync():
                    prims.tcgen05_commit(acc_done.subview(0))
            else:
                stage = cutlass.Int32(0)
                phase = cutlass.Int32(0)
                for rd in cutlass.range_constexpr(rounds_max):
                    tile = cluster + cutlass.Int32(rd * gemv_clusters)
                    if tile < cutlass.Int32(n_tiles):
                        acc = cutlass.inttoptr(
                            tmem_ptr_i32.load() + cutlass.Int32(rd * MMA_N), 6, cutlass.Int32
                        )
                        for i in range(my_tiles):
                            while not cute.arch.mbarrier_try_wait(
                                tma_full.subview(stage).data_ptr(), phase
                            ):
                                pass
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            for kb in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                                box = kb // K_BLOCKS_PER_HALF
                                within = kb % K_BLOCKS_PER_HALF
                                desc_a = desc_a_base + (
                                    stage * cutlass.Int32(STAGE_A)
                                    + cutlass.Int32(box * A_BOX + within * STEP)
                                )
                                desc_b = desc_b_base + (
                                    i * cutlass.Int32(STAGE_B)
                                    + cutlass.Int32(box * B_BOX + within * STEP)
                                )
                                accumulate = cutlass.Boolean(True)
                                if cutlass.const_expr(kb == 0):
                                    accumulate = i > cutlass.Int32(0)
                                if prims.elect_sync():
                                    prims.tcgen05_mma(
                                        prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, acc, desc_a, desc_b, idesc,
                                        accumulate,
                                    )  # fmt: skip
                            if prims.elect_sync():
                                prims.tcgen05_commit(mma_done.subview(stage))
                            stage = stage + cutlass.Int32(1)
                            if stage == cutlass.Int32(ring):
                                stage = cutlass.Int32(0)
                                phase = phase ^ cutlass.Int32(1)
                        if prims.elect_sync():
                            prims.tcgen05_commit(acc_done.subview(rd))
        elif warp_id >= 4 and warp_id < 8:
            # =================================================================
            # Epilogue: per round, TMEM -> registers, push to / reduce at the
            # row owner; the owner pushes the head slice or writes the shared
            # activation.
            # =================================================================
            w = warp_id - 4  # TMEM lanes 32 w ..: tile rows owned by rank w
            while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
                pass
            # flags_ready orders warp 1's store of the buffer index before this read.
            while not cute.arch.mbarrier_try_wait(flags_ready.data_ptr(), 0):
                pass
            b = s_flags.load(idx=0)
            slot_row = lane * cutlass.Int32(MMA_N)
            # A head half-tile's rows are TMEM lanes 0-63 (warps 4-5); warps 6-7 hold garbage rows and sit out.
            live = cutlass.Boolean(True)
            if cutlass.const_expr(head_half):
                live = stream_cta | (w < cutlass.Int32(2))
            for rd in cutlass.range_constexpr(rounds_max):
                tile = cluster + cutlass.Int32(rd * gemv_clusters)
                if (tile < cutlass.Int32(n_tiles)) & live:
                    while not cute.arch.mbarrier_try_wait(acc_done.subview(rd).data_ptr(), 0):
                        pass
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    acc = prims.tcgen05_ld(
                        "32x32b",
                        cutlass.inttoptr(
                            tmem_ptr_i32.load() + cutlass.Int32(rd * MMA_N), 6, cutlass.Float32
                        ),
                        num=MMA_N,
                    )
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    box_base = cutlass.Int32(rd * SPLIT * ROWS_PER_WARP * MMA_N)
                    tile_row = cutlass.select_(
                        head_cta, tile * cutlass.Int32(HALF_M), tile * cutlass.Int32(CTA_M)
                    )
                    row0 = cutlass.Int32(tile_row) + w * cutlass.Int32(ROWS_PER_WARP)
                    srow0 = w * cutlass.Int32(ROWS_PER_WARP)
                    if (tile < cutlass.Int32(n_head_tiles)) & (row0 >= cutlass.Int32(WL)):
                        # Router rows: every rank of the cluster pushes its own split-K partial (fp32, -0.0 as +0.0)
                        # into slot [crank] of the partials region; the route CTAs add the 8 in rank order.
                        pe0 = row0 - cutlass.Int32(WL)
                        if pe0 < cutlass.Int32(WE):
                            for tt in cutlass.range_constexpr(MMA_N):
                                s_rows.store(
                                    cutlass.Float32(acc[tt]),
                                    idx=(srow0 + lane) * cutlass.Int32(MMA_N) + cutlass.Int32(tt),
                                )
                            cute.arch.sync_warp()
                            # 8 router vectors (4 logits each) per token: two (token, vector) pairs per lane.
                            for ph in cutlass.range_constexpr(2):
                                ppair = lane + cutlass.Int32(ph * 32)
                                ptok = ppair // cutlass.Int32(8)
                                pvec = ppair % cutlass.Int32(8)
                                pe = pe0 + pvec * cutlass.Int32(4)
                                if ptok < num_tokens:
                                    if pe < cutlass.Int32(WE):
                                        pwords = []
                                        for pp in cutlass.range_constexpr(4):
                                            pval = cutlass.Float32(
                                                s_rows.load(
                                                    idx=(
                                                        srow0
                                                        + pvec * cutlass.Int32(4)
                                                        + cutlass.Int32(pp)
                                                    )
                                                    * cutlass.Int32(MMA_N)
                                                    + ptok
                                                )
                                            )
                                            pwords.append(
                                                _ag._sanitize_f32(pval.bitcast(cutlass.Int32))
                                            )
                                        pdst = (
                                            cutlass.Int32(PBASE)
                                            + (
                                                (b * cutlass.Int32(M_MAX) + ptok)
                                                * cutlass.Int32(world)
                                                + tp_rank
                                            )
                                            * cutlass.Int32(SPLIT * WE)
                                            + crank * cutlass.Int32(WE)
                                            + pe
                                        )
                                        buf_mc.store(
                                            (pwords[0], pwords[1], pwords[2], pwords[3]),
                                            idx=pdst,
                                            alignment=16,
                                        )
                    elif w != crank:
                        base = box_base + crank * cutlass.Int32(ROWS_PER_WARP * MMA_N) + slot_row
                        bar = _mapa_u32(mail_full.subview(rd).data_ptr(), w)
                        for h in cutlass.range_constexpr(MMA_N // 4):
                            _st_async_v4_f32(
                                _mapa_u32(mailbox.subview(base + cutlass.Int32(4 * h)).data_ptr(), w),
                                cutlass.Float32(acc[4 * h]), cutlass.Float32(acc[4 * h + 1]),
                                cutlass.Float32(acc[4 * h + 2]), cutlass.Float32(acc[4 * h + 3]), bar,
                            )  # fmt: skip
                    else:
                        # Completed by the peers' st.async: acquire at cluster scope.
                        while not _test_wait_cluster(mail_full.subview(rd).data_ptr(), 0):
                            pass
                        tot = [cutlass.Float32(0.0)] * MMA_N
                        for t in cutlass.range_constexpr(MMA_N):
                            total = cutlass.Float32(0.0)
                            for q in cutlass.range_constexpr(SPLIT):
                                part = mailbox.load(
                                    idx=box_base
                                    + cutlass.Int32(q * ROWS_PER_WARP * MMA_N)
                                    + slot_row
                                    + cutlass.Int32(t)
                                )
                                total = total + cutlass.Float32(
                                    cutlass.select_(
                                        crank == cutlass.Int32(q), cutlass.Float32(acc[t]), part
                                    )
                                )
                            tot[t] = total
                        if tile < cutlass.Int32(n_head_tiles):
                            # Head: stage the warp's 32 rows x 8 tokens, then 16-byte vectors to every rank.
                            for t in cutlass.range_constexpr(MMA_N):
                                s_rows.store(
                                    tot[t],
                                    idx=(w * cutlass.Int32(ROWS_PER_WARP) + lane)
                                    * cutlass.Int32(MMA_N)
                                    + cutlass.Int32(t),
                                )
                            cute.arch.sync_warp()
                            # Latent rows (router rows push their partials above).
                            # 4 latent vectors (8 columns each) per token: lane = token * 4 + vector.
                            t = lane // cutlass.Int32(4)
                            v = lane % cutlass.Int32(4)
                            if t < num_tokens:
                                words = []
                                for p in cutlass.range_constexpr(4):
                                    c0 = srow0 + v * cutlass.Int32(8) + cutlass.Int32(2 * p)
                                    lo = cutlass.Float32(
                                        s_rows.load(idx=c0 * cutlass.Int32(MMA_N) + t)
                                    )
                                    hi = cutlass.Float32(
                                        s_rows.load(
                                            idx=(c0 + cutlass.Int32(1)) * cutlass.Int32(MMA_N) + t
                                        )
                                    )
                                    words.append(_ag._sanitize_bf16x2(_ag._pack_bf16x2(hi, lo)))
                                dst = b * cutlass.Int32(BUF) + (
                                    t * cutlass.Int32(world) + tp_rank
                                ) * cutlass.Int32(SLOT)
                                dst = dst + ((row0 // cutlass.Int32(8)) + v) * cutlass.Int32(4)
                                buf_mc.store(
                                    (words[0], words[1], words[2], words[3]),
                                    idx=dst,
                                    alignment=16,
                                )
                        else:
                            # Shared: lane l < 16 holds gate column j, lane l + 16 its up row.
                            col = (
                                (tile - cutlass.Int32(n_head_tiles)) * cutlass.Int32(64)
                                + w * cutlass.Int32(16)
                                + lane
                            )
                            for t in cutlass.range_constexpr(MMA_N):
                                up = cutlass.Float32(cute.arch.shuffle_sync_bfly(tot[t], offset=16))
                                if lane < cutlass.Int32(16):
                                    if cutlass.Int32(t) < num_tokens:
                                        out = _bf16_rn(
                                            _situ(
                                                _bf16_rn(tot[t]), _bf16_rn(up), gate_cap, linear_cap
                                            )
                                        )
                                        shared_bits.store(
                                            cutlass.Int16(out.bitcast(cutlass.Int32) >> 16),
                                            idx=cutlass.Int32(t * shared_cols) + col,
                                        )
            prims.barrier_cta_sync(1, thread_count=128)
            if warp_id == 4:
                prims.tcgen05_dealloc(tmem_ptr, HALF_TMEM_COLS if head_half else TMEM_COLS)
    else:
        # =====================================================================
        # Role CTAs: route (CTA t < M) or quantize (CTA M + t) token t from
        # every rank's pushed slot, as k3_route_quant_ag after its push.
        # =====================================================================
        role = role_or_gemv * cutlass.Int32(SPLIT) + crank
        if role < num_tokens * cutlass.Int32(2):
            routes = role < num_tokens
            tok = cutlass.select_(routes, role, role - num_tokens)
            polls_logits = tx < cutlass.Int32(NUM_EXPERTS // 4)
            e0 = cutlass.select_(polls_logits, tx * cutlass.Int32(4), cutlass.Int32(0))
            bias4 = bias.load(idx=e0, vector_size=4, alignment=16)
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            if cutlass.const_expr(not publish):
                prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
            if tx == 0:
                s_flags.store(flags.load(idx=0, is_volatile=True), idx=0)
                s_flags.store(flags.load(idx=2, is_volatile=True), idx=1)
            cute.arch.barrier()
            if cutlass.const_expr(publish):
                # The dependent k3_moe (head_flags) advances the epoch at its last claim, so the epoch is read before
                # this CTA's trigger, as in k3_route_quant_ag.
                prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
            b = s_flags.load(idx=0)
            epoch = s_flags.load(idx=1)
            # A role CTA signs in as soon as it has read the buffer index (it reads nothing of flags after this).
            # Token 0's quantization CTA hands the buffer over once every reader is in; its polls wait for the pushes,
            # which come microseconds later.
            if tx == 0:
                _red_add_release(flags.data_ptr(3).toint(), cutlass.Int32(1))
                if role == num_tokens:
                    _flip(
                        flags,
                        b,
                        num_tokens * cutlass.Int32(2) + cutlass.Int32(gemv_clusters * SPLIT),
                    )
            tok_base = b * cutlass.Int32(BUF) + tok * cutlass.Int32(world * SLOT)
            empty = cutlass.Int32(EMPTY_WORD)
            if routes:
                # The keys and the selection run once per layer, so their code is cold in the SM's instruction cache.
                # Pass 0 runs the same code (one copy: a dynamic loop) on synthetic logits while the pushes are still
                # on their way, its outputs discarded; pass 1 polls the pushes and runs it for real, from warm code.
                for p in cutlass.range(2, unroll=1):
                    real = p == cutlass.Int32(1)
                    if polls_logits:
                        # Pass 0's logits: distinct values in [-4.5, 4.5), a permutation of the experts (e * 37 mod
                        # 896) that spreads the 16 largest over 8 warps as router logits spread, so pass 0 takes
                        # _top16_cta's common path (18 candidates; the bias is left out of pass 0's keys).
                        dry = []
                        for q in cutlass.range_constexpr(4):
                            perm = ((e0 + cutlass.Int32(q)) * cutlass.Int32(37)) % cutlass.Int32(
                                NUM_EXPERTS
                            )
                            dry.append(
                                (
                                    cutlass.Float32(perm) * cutlass.Float32(0.01)
                                    - cutlass.Float32(4.5)
                                ).bitcast(cutlass.Int32)
                            )
                        w0, w1, w2, w3 = dry
                        if real:
                            # The 8 split-K partials of these 4 logits, pushed by the pushing rank's GEMV CTAs.
                            addr = (
                                cutlass.Int32(PBASE)
                                + (
                                    (b * cutlass.Int32(M_MAX) + tok) * cutlass.Int32(world)
                                    + tx // cutlass.Int32(EV)
                                )
                                * cutlass.Int32(SPLIT * WE)
                                + (tx % cutlass.Int32(EV)) * cutlass.Int32(4)
                            )
                            w0, w1, w2, w3 = _poll_logit_parts(buf_uc, addr, WE)
                            for q in cutlass.range_constexpr(SPLIT):
                                buf_uc.store(
                                    (empty, empty, empty, empty),
                                    idx=addr + cutlass.Int32(q * WE),
                                    alignment=16,
                                )
                        words = [w0, w1, w2, w3]
                        for q in cutlass.range_constexpr(4):
                            sig = _rq.sigmoid_accurate(words[q].bitcast(cutlass.Float32))
                            s_sigmoid.store(sig, idx=e0 + q)
                            key_bias = cutlass.Float32(
                                cutlass.select_(real, bias4[q], cutlass.Float32(0.0))
                            )
                            s_key.store(_rq.selection_key(sig + key_bias), idx=e0 + q)
                    cute.arch.barrier()
                    expert, weight_bits = _top16_cta(
                        s_key,
                        s_sigmoid,
                        s_top2,
                        s_cnt,
                        s_cand,
                        s_csig,
                        s_win,
                        s_wsig,
                        tx,
                        routed_scaling_factor,
                    )
                    if tx < cutlass.Int32(32):
                        if real:
                            if tx < cutlass.Int32(TOP_K):
                                out = tok * cutlass.Int32(TOP_K) + tx
                                topk_ids.store(expert, idx=out)
                                topk_weight_bits.store(weight_bits, idx=out)
                                if cutlass.const_expr(publish):
                                    cute.arch.fence_acq_rel_gpu()
                            if cutlass.const_expr(publish):
                                cute.arch.sync_warp()
                                if tx == cutlass.Int32(0):
                                    _ag._store_release(
                                        ready.data_ptr(tok).toint(), epoch + cutlass.Int32(1)
                                    )
                    # Pass 0's readers of the keys and of the selection's scratch are done before pass 1 rewrites them.
                    cute.arch.barrier()
            else:
                addr = (
                    tok_base
                    + (tx // cutlass.Int32(LV)) * cutlass.Int32(SLOT)
                    + (tx % cutlass.Int32(LV)) * cutlass.Int32(4)
                )
                w0, w1, w2, w3 = _ag._poll4(buf_uc, addr)
                buf_uc.store((empty, empty, empty, empty), idx=addr, alignment=16)
                q_lo, q_hi, sf_byte = _rq.mxfp8_quant_vec8([w0, w1, w2, w3])
                quant_words.store(
                    (q_lo, q_hi),
                    idx=tok * cutlass.Int32(HIDDEN_SIZE // 4) + tx * cutlass.Int32(2),
                    alignment=8,
                )
                if tx % cutlass.Int32(SF_VEC_SIZE // 8) == cutlass.Int32(0):
                    scales.store(
                        cutlass.Uint8(sf_byte),
                        idx=tok * cutlass.Int32(HIDDEN_SIZE // SF_VEC_SIZE)
                        + tx // cutlass.Int32(SF_VEC_SIZE // 8),
                    )
                if cutlass.const_expr(publish):
                    prims.fence_proxy("async_global")
                    cute.arch.fence_acq_rel_gpu()
                    cute.arch.barrier()
                    if tx == cutlass.Int32(0):
                        _ag._store_release(
                            ready.data_ptr(tok + cutlass.Int32(M_MAX)).toint(),
                            epoch + cutlass.Int32(1),
                        )
            cute.arch.barrier()
        else:
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


@dsl_user_op
def _red_add_release(addr_i64, val, *, loc=None, ip=None):
    """red.release.gpu.global.add.u32 (no round trip)."""
    from cutlass._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(
        None, [addr_i64.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.u32 [$0], $1;", "l,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _load_acquire(addr_i64, *, loc=None, ip=None):
    """ld.acquire.gpu.global.u32."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [addr_i64.ir_value(loc=loc, ip=ip)], "ld.acquire.gpu.global.u32 $0, [$1];", "=r,l",
            has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@cute.jit
def _flip(flags, b, readers):
    """Hands the other buffer to the next call once all ``readers`` CTAs of this call that read the buffer index have
    signed in (flags[3], each with a release right after its read), so none reads it any more. The next call reads the
    index only after this grid has completed."""
    while _load_acquire(flags.data_ptr(3).toint()) < readers:
        pass
    flags.store(cutlass.Int32(0), idx=3)
    flags.store(b ^ cutlass.Int32(1), idx=0)


def _weight_tensor_map(w, n_out, k_in, box_rows=CTA_M):
    """W as five TMA dimensions (64-element column chunk, row, 64-element chunk index, 1, 1), box_rows rows a call."""
    return cuda.create_tensor_map_tiled(
        global_address=w.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[TMA_K_BOX, n_out, k_in // TMA_K_BOX, 1, 1],
        global_strides=[
            (k_in * ELEM_BYTES) // 16,
            (TMA_K_BOX * ELEM_BYTES) // 16,
            (n_out * k_in * ELEM_BYTES) // 16,
            (n_out * k_in * ELEM_BYTES) // 16,
        ],
        box_dims=[TMA_K_BOX, box_rows, TMA_COPY_ITERS, 1, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


def _activation_tensor_map(x, cols, num_tokens):
    """x [M, cols] as (cols, M) with an 8-row box: rows past num_tokens arrive as zeros."""
    return cuda.create_tensor_map_tiled(
        global_address=x.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[cols, num_tokens],
        global_strides=[(cols * ELEM_BYTES) // 16],
        box_dims=[TMA_K_BOX, MMA_N],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


@cute.jit
def k3_moe_front(
    w: cute.Tensor,  # [N, K] bf16
    x: cute.Tensor,  # [M, K] bf16
    bias: cute.Tensor,
    buf_uc: cute.Tensor,
    buf_mc: cute.Tensor,
    flags: cute.Tensor,
    ready: cute.Tensor,
    topk_ids: cute.Tensor,
    topk_weight_bits: cute.Tensor,
    quant_words: cute.Tensor,
    scales: cute.Tensor,
    shared_bits: cute.Tensor,
    num_tokens: cutlass.Int32,
    tp_rank: cutlass.Int32,
    routed_scaling_factor: cutlass.Float64,
    world: cutlass.Constexpr[int],
    shared_cols: cutlass.Constexpr[int],
    n_head_tiles: cutlass.Constexpr[int],
    n_tiles: cutlass.Constexpr[int],
    gemv_clusters: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    gate_cap: cutlass.Constexpr[float],
    linear_cap: cutlass.Constexpr[float],
    publish: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    head_half: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    rows = _weight_rows(world, n_head_tiles, n_tiles, head_half)
    tma_desc_w = _weight_tensor_map(w, rows, k_in)
    tma_desc_w64 = _weight_tensor_map(w, rows, k_in, box_rows=HALF_M)
    tma_desc_x = _activation_tensor_map(x, k_in, num_tokens)
    k3_moe_front_kernel(
        tma_desc_w, tma_desc_w64, tma_desc_x, bias, buf_uc, buf_mc, flags, ready, topk_ids, topk_weight_bits,
        quant_words, scales, shared_bits, num_tokens, tp_rank, routed_scaling_factor, world, shared_cols,
        n_head_tiles, n_tiles, gemv_clusters, k_in, ring, gate_cap, linear_cap, publish, head_half,
    ).launch(
        grid=((gemv_clusters + ROLE_CLUSTERS) * SPLIT, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(SPLIT, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
        # One CTA per SM (shared memory): without it ptxas may cap registers for two and spill (bias4 did).
        min_blocks_per_mp=1,
    )  # fmt: skip
