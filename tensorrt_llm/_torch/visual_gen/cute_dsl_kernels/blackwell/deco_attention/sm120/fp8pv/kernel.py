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
# O is the output accumulator and l the row sum of P.
# ruff: noqa: E741
"""NVFP4 QK + FP8 PV attention for sm_120a (RTX PRO 6000), built on the warp-level MMAs (mma.sync).

A CTA takes 128 query rows of one (batch, head) (8 warps x 16 rows) over tiles of 128 keys: S = Q' K'^T over the 192
channels of Q' and K' on the block-scaled NVFP4 MMA (kind::mxf4nvf4, m16n8k64, UE4M3 scales); the online softmax of
softmax.py (log2 units, lazy rescale at T = 4, I-ExpCast into E4M3 P, the row sum of the quantized P summed in the
softmax); O += P V on the FP8 MMA (m16n8k32), V being the E4M3 V minus its means of every 128 keys; and the means
term, the block sums of P times v_mean on BF16 MMAs. o = (P V + S M) * v8_scale / l. v_mean holds one row per 16
keys, a tile's mean in all 8 of its rows, and the means MMA weights the rows by the 16-key block sums of P, which
add up to the tile's.

Schedule. The warps are paired (w, w + 4); the two warps of a pair alternate in issuing MMAs under a named barrier of
64 threads: one issues its MMA phase k = [PV(k - 1), means, QK(k)] while the other runs the softmax of its scores; the
four pairs run independently. The means of the step pair (2m, 2m + 1) take phase 2m + 2, those of an unpaired last
step the last phase.

Ring. Stage k % 3 of the TMA ring holds exactly what phase k reads (K' and its scale atoms of step k, V of step
k - 1, and for an even phase the v_mean rows of its step pair in one of two pair buffers) under one full barrier,
which a warp waits for before its first fragment loads and its turn, so that a phase starts only when all of its
operands are in shared memory. The softmax keeps the scores in registers, so a warp's reads of a stage end with its
MMA phase: it counts its release then, and the eighth warp to do so refills the stage with K' of step k + 3 and V of
step k + 2. No warp waits for the others.

Key tiles run diagonal first: the query tile's own key tile, then onward, wrapping, and the partial last tile last.

Zero-P skipping is exact: a warp-uniform mask of the 32-key quarters with a nonzero packed P byte picks the dense
phase, the QK-only phase or, for a mixed tile, one indexed branch per quarter around its V loads and MMAs. A scan of V
and v_mean before the attention (scan.py) flags the heads with a non-finite value, which take every quarter (0 * NaN
must propagate).

Split tail. One launch holds the whole units (query tile, head, batch) followed by the key-range pieces of the last
units % nsm + nsm // 2 units of the grid (nsm: the SM count of the device; launch.split_plan); a merge launch combines
the FP32 partials of the pieces (epilogue.py)."""

import math

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32, Int64

from ...common.dsl import BarrierRedux, barrier_cta_red, barrier_cta_sync
from ...common.mem import load_invariant, sptr
from ...common.state import State
from ..ptx.fp8pv import (
    atom_add_shared_u32_lane0,
    lane_zero,
    mbar_wait,
    packed_pv_needed,
    stage_next,
)
from ..ptx.shared import mbar_init_imm
from .epilogue import inv_row_sums, store_partials, store_rows
from .layout import (
    BAR_OFF,
    CNT_OFF,
    EP,
    K_BYTES,
    KT,
    NTHR,
    SF_BYTES,
    SF_OFF,
    STAGE_BYTES,
    STAGES,
    V_BYTES,
    V_OFF,
    VM_BYTES,
    VMP_OFF,
    lane_offsets,
    make_tma_views,
)
from .mma import (
    DIST,
    load_q,
    means_last,
    means_pair_guarded,
    mma_phase,
    pv_sparse_quarters,
    store64,
)
from .scan import head_v_nan
from .softmax import softmax_tile

EARLY = 4  # MMA steps before the end of a phase at which the warp hands the turn to its partner (tuning constant)


# ------------------------------------------------------------------------------------------------ CTA state
# The steps of a CTA body share one State st (common/state.py), built by cta_state and completed by start_ring and
# load_operands; no step assigns a field after that. Its fields:
#   tidx, warp, lane, g, q, grp     the thread: its rows are g = lane / 4 and g + 8 of the warp, q = lane % 4 its place
#                                   in the quad; grp 0: warps 0-3 (rows 0-63), 1: warps 4-7 (rows 64-127)
#   qt, hh, bb                      the unit: query tile, head, batch
#   bid_wait, bid_pass              the turn barriers of the warp pair (w, w + 4): this warp's and its partner's
#   turn, turn_held                 the turn of an MMA phase as mma_phase takes it: (take, pass, EARLY)
#   sp, smem, fbar                  shared memory: its pointer, its address, the address of the stages' full barriers
#   dist                            the ldmatrix lookahead of the MMA phases
#   tma, vnan                       start_ring: the TMA views, the non-finite flag of the head
#   sc, vsc, lz                     load_operands: the score scale, v8_scale, the per-lane zero of the release count
#   mma, voff, vmoff                load_operands: the thread's operands of mma_phase in its order (the Q' fragments
#                                   and scale words, the per-lane offsets of the K' rows, K' scale words and V row,
#                                   the lookahead), and the per-lane offsets of the V row and the v_mean rows
#   O, P                            load_operands: the register tensors of the O accumulators and of the packed P with
#                                   its block sums and quarter mask (softmax.py)


def pass_turn(st):
    """Hand the turn of the warp pair to the partner warp: an arrival on its turn barrier. The turn barriers use the
    non-aligned form because the warps of an MMA phase reach them from two code paths (the dense and the zero-P
    branches)."""
    cute.arch.barrier_arrive(barrier_id=st.bid_pass, number_of_threads=64, aligned=False)


def take_turn(st, sw):
    """The start of an MMA phase without lookahead loads: wait for the stage's full barrier sw = (barrier, parity),
    then for this warp's turn (the pair's 64 threads: this warp's sync and its partner's pass_turn)."""
    mbar_wait(sw[0], sw[1])
    barrier_cta_sync(st.bid_wait, thread_count=64)


def cta_state(tidx, qt, hh, bb, dist):
    """The state of thread tidx of the CTA of unit (query tile qt, head hh, batch bb): the thread coordinates, the
    turn of its warp pair and the shared memory. dist: the lookahead of its MMA phases."""
    warp = tidx >> 5
    lane = tidx & 31
    g = lane >> 2
    q = lane & 3
    grp = warp >> 2  # 0: warps 0-3 (rows 0-63), 1: warps 4-7 (rows 64-127)
    pair = warp & 3  # warps w and w + 4 form a pair and alternate in issuing MMAs
    st = State(tidx=tidx, qt=qt, hh=hh, bb=bb, warp=warp, lane=lane, g=g, q=q, grp=grp, dist=dist)
    st.bid_wait = 1 + pair + 4 * grp  # this warp's turn barrier (64 threads: the pair)
    st.bid_pass = 5 + pair - 4 * grp  # its partner's
    # the turn of an MMA phase (mma_phase): take it, pass it EARLY steps before the end; turn_held: already taken
    st.turn = (lambda: barrier_cta_sync(st.bid_wait, thread_count=64), lambda: pass_turn(st), EARLY)
    st.turn_held = (lambda: None, lambda: pass_turn(st), EARLY)
    st.sp = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024)
    st.smem = Int32(st.sp.toint())
    st.fbar = st.smem + BAR_OFF
    return st


@cute.jit
def start_ring(st, ops, H: cutlass.Constexpr, nT: cutlass.Constexpr):
    """The ring of the CTA before its first load: the TMA views of the unit's (batch, head) (st.tma), the stages'
    full barriers and release counters, which thread 0 initializes, and st.vnan, the OR of the head's scan flags
    over the CTA. The barrier of that reduction is the one behind which every thread sees the initialized barriers and
    counters."""
    _, _, _, flags_ptr, tiles = ops
    vflags = flags_ptr.toint() + Int64((st.bb * H + st.hh) * nT) * 4
    st.tma = make_tma_views(st.sp, *tiles, st.hh, st.bb)
    if st.tidx == 0:
        for stage in cutlass.range_constexpr(STAGES):
            mbar_init_imm(st.fbar + stage * 8, 1)
            cute.arch.store(sptr(Int32, st.smem + CNT_OFF + stage * 4), Int32(0))
        cute.arch.mbarrier_init_fence()
    # a non-finite V or v_mean anywhere in this (batch, head) disables zero-P skipping for the head
    st.vnan = cute.arch.make_warp_uniform(
        Int32(
            barrier_cta_red(
                head_v_nan(vflags, st.tidx, nT) != 0, 0, BarrierRedux.OR, thread_count=NTHR
            )
        )
    )


def load_operands(st, ops, S, H, nT, vs_ptr=None):
    """What a thread keeps in registers over the key sequence: the Q' fragments and scale words of its rows (load_q),
    the score scale, v8_scale (with vs_ptr: the body that writes the output rows), the per-lane zero of the release
    count, the per-lane offsets of the B fragments (lane_offsets), the O accumulators (zero) and the P words (zero
    block sums)."""
    q_ptr, qsf_ptr, ss_ptr, _, _ = ops
    qa, sfq = load_q(
        q_ptr.toint(), qsf_ptr, st.qt, st.warp, st.lane, st.g, st.q, st.bb, st.hh, S, H, nT
    )
    st.sc = load_invariant(Float32, ss_ptr.toint())
    if vs_ptr is not None:
        st.vsc = load_invariant(Float32, vs_ptr.toint())
    st.lz = lane_zero(st.sc, st.lane)
    koff_e, koff_o, sfoff, st.voff, st.vmoff = lane_offsets(st.lane)
    st.mma = (qa, sfq, koff_e, koff_o, sfoff, st.voff, st.dist)
    st.O = cute.make_rmem_tensor(cute.make_layout(64), Float32)
    st.O.fill(0.0)
    st.P = cute.make_rmem_tensor(cute.make_layout(21), Int32)
    st.P[16] = Int32(0)
    st.P[17] = Int32(0)


# ------------------------------------------------------------------------------------------------ stage ring


def release_count(st, stage):
    """Count this warp's release of stage `stage` (its last read of the stage is behind it): the counter before the
    increment, on every lane."""
    cute.arch.sync_warp()
    return cute.arch.shuffle_sync(
        atom_add_shared_u32_lane0(st.smem + CNT_OFF + stage * 4 + st.lz * 4, st.lane), 0
    )


@cute.jit
def rot_tile(k, rot, nm1: cutlass.Constexpr):
    """Key tile of step k < nm1 in the diagonal-first order: (k + rot) mod nm1 (the partial last tile stays last)."""
    t = k + rot
    if t >= nm1:
        t = t - nm1
    return t


def step_tile(j, rot, nT):
    """Key tile of step j of a whole unit, j known at trace time (a step before the first counts as step 0): rot_tile
    for the full tiles, the partial last tile nT - 1 for the last step."""
    return Int32(nT - 1) if j >= nT - 1 else rot_tile(Int32(max(j, 0)), rot, nT - 1)


def pair_buffer(st, k):
    """The pair buffer with the v_mean rows that phase k reads (an even k: those of the step pair (k - 2, k - 1))."""
    return st.smem + VMP_OFF + (((k - 1) >> 1) & 1) * (2 * VM_BYTES)


@cute.jit
def issue_tile(st, stage, ktile, vtile, has_k, has_v, vma, vmb, nvm, vbuf):
    """Stage `stage` of phase j holds what the phase reads: K' (three atom columns and the three scale atoms) of step
    j for QK(j) and V of step j - 1 for PV(j - 1); for an even phase j the same full barrier also covers the v_mean
    rows of the step pair (j - 2, j - 1) (tiles vma, vmb; nvm = 2) in pair buffer vbuf (read by phase j), and for an
    odd last phase those of the unpaired last step (tile vma; nvm = 1) in the stage's unused K' region. The expected
    bytes are those issued."""
    (
        tma_atom_k,
        tKgK,
        tKsK,
        tma_atom_v,
        tVgV,
        tVsV,
        tma_atom_s,
        tSgS,
        tSsS,
        tma_atom_m,
        tMgM,
        tMsM,
        tM1sM,
        bar_ptr,
    ) = st.tma
    if st.lane == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(
            sptr(Int64, st.fbar + stage * 8),
            has_k * (K_BYTES + SF_BYTES) + has_v * V_BYTES + nvm * VM_BYTES,
        )
    if has_k != 0:
        cute.copy(tma_atom_k, tKgK[(None, ktile)], tKsK[(None, stage)], tma_bar_ptr=bar_ptr + stage)
        cute.copy(tma_atom_s, tSgS[(None, ktile)], tSsS[(None, stage)], tma_bar_ptr=bar_ptr + stage)
    if has_v != 0:
        cute.copy(tma_atom_v, tVgV[(None, vtile)], tVsV[(None, stage)], tma_bar_ptr=bar_ptr + stage)
    if nvm == 2:
        cute.copy(tma_atom_m, tMgM[(None, vma)], tMsM[(None, 0, vbuf)], tma_bar_ptr=bar_ptr + stage)
        cute.copy(tma_atom_m, tMgM[(None, vmb)], tMsM[(None, 1, vbuf)], tma_bar_ptr=bar_ptr + stage)
    if nvm == 1:
        # the unpaired last step (a phase without K'): into the stage's K' region
        cute.copy(tma_atom_m, tMgM[(None, vma)], tM1sM[(None, stage)], tma_bar_ptr=bar_ptr + stage)


def first_stages_full(st, rot, nT):
    """The first loads of a whole unit, issued by warp 4. Stage j (phases 0 .. 2): K' of step j and V of step j - 1;
    phase 2 takes the v_mean rows of the step pair (0, 1), a last odd phase 1 (nT == 1) those of step 0."""
    for tl in range(min(STAGES, nT + 1)):
        kt = step_tile(tl, rot, nT)
        vt = step_tile(tl - 1, rot, nT)
        nvm = 2 if (tl % 2 == 0 and tl >= 2) else (1 if (tl % 2 == 1 and tl - 1 == nT - 1) else 0)
        vma = step_tile(tl - 2, rot, nT) if nvm == 2 else vt
        issue_tile(
            st,
            tl,
            kt,
            vt,
            Int32(1 if tl < nT else 0),
            Int32(1 if tl >= 1 else 0),
            vma,
            vt,
            Int32(nvm),
            Int32(((tl - 1) >> 1) & 1 if tl >= 1 else 0),
        )


@cute.jit
def first_stages_part(st, t0, nK):
    """The first loads of a key-range piece (tiles t0 .. t0 + nK - 1, nK >= 2), issued by warp 4. Stage j (phases
    0 .. 2): K' of tile t0 + j (stage 2: when the piece has a third tile) and V of tile t0 + j - 1."""
    for tl in cutlass.range_constexpr(STAGES):
        hk = Int32(1)
        if cutlass.const_expr(tl == 2):
            hk = Int32(0)
            if tl < nK:
                hk = Int32(1)
        # phase 2 takes the v_mean rows of the step pair (0, 1) (pieces have nK >= 2 tiles)
        issue_tile(
            st,
            tl,
            t0 + tl,
            t0 + max(tl - 1, 0),
            hk,
            Int32(1 if tl >= 1 else 0),
            t0,
            t0 + 1,
            Int32(2 if tl == 2 else 0),
            Int32(0),
        )


@cute.jit
def refill_if_last(st, old, ktile, vtile, has_k, vma, vmb, nvm):
    """After softmax_tile counted this warp's release of stage 0 (old: the counter before it): the eighth warp
    refills the stage for phase 3, with the tiles and v_mean rows issue_tile takes (a V tile, pair buffer 1)."""
    if (old & 7) == 7:
        cute.arch.fence_proxy("async.shared", space="cta")
        issue_tile(st, 0, ktile, vtile, has_k, Int32(1), vma, vmb, nvm, Int32(1))


def refill_full0(st, old, rot, nT, vt2):
    """The refill of stage 0 for phase 3 of a whole unit (nT > 2): K' of step 3, V of step 2 (tile vt2) and, for an
    unpaired last step 2 (nT == 3), its v_mean rows."""
    refill_if_last(
        st,
        old,
        step_tile(3, rot, nT),
        vt2,
        Int32(1 if 3 < nT else 0),
        vt2,
        vt2,
        Int32(1 if nT == 3 else 0),
    )


@cute.jit
def refill_loop_full(st, old, k, stage, rot, nT: cutlass.Constexpr):
    """refill_if_last for a loop phase k of the full body with a phase k + 3 to load (k + 2 < nT), the tiles of the
    refill computed by the refilling warp only: K' of step k + 3, V of step k + 2 and, for odd k, the v_mean rows of
    the pair (k + 1, k + 2) (an unpaired last step k + 2 = nT - 1 for even k)."""
    if k + 2 < nT:
        if (old & 7) == 7:
            kt = Int32(nT - 1)
            if k + 3 < nT - 1:
                kt = rot_tile(k + 3, rot, nT - 1)
            vt = Int32(nT - 1)
            if k + 2 < nT - 1:
                vt = rot_tile(k + 2, rot, nT - 1)
            hk = Int32(0)
            if k + 3 < nT:
                hk = Int32(1)
            nvm = Int32(0)
            vma = vt
            if (k & 1) == 1:
                nvm = Int32(2)
                vma = rot_tile(k + 1, rot, nT - 1)
            else:
                if k + 2 == nT - 1:
                    nvm = Int32(1)
            cute.arch.fence_proxy("async.shared", space="cta")
            issue_tile(st, stage, kt, vt, hk, Int32(1), vma, vt, nvm, ((k + 2) >> 1) & 1)


@cute.jit
def refill_loop_part(st, old, k, stage, t0, nK):
    """refill_loop_full for the key-range pieces (tiles t0 + step; a phase k + 3 to load: k + 2 < nK)."""
    if k + 2 < nK:
        if (old & 7) == 7:
            hk = Int32(0)
            if k + 3 < nK:
                hk = Int32(1)
            nvm = Int32(0)
            if (k & 1) == 1:
                nvm = Int32(2)
            else:
                if k + 2 == nK - 1:
                    nvm = Int32(1)
            cute.arch.fence_proxy("async.shared", space="cta")
            issue_tile(
                st,
                stage,
                t0 + k + 3,
                t0 + k + 2,
                hk,
                Int32(1),
                t0 + k + 2 - (k & 1),
                t0 + k + 2,
                nvm,
                ((k + 2) >> 1) & 1,
            )


@cute.jit
def refill_part0(st, old, t0, nK):
    """The refill of stage 0 for phase 3 of a piece (nK > 2)."""
    if 2 < nK:
        hk0 = Int32(0)
        if 3 < nK:
            hk0 = Int32(1)
        nvm0 = Int32(0)
        if nK == 3:
            nvm0 = Int32(1)
        refill_if_last(st, old, t0 + 3, t0 + 2, hk0, t0 + 2, t0 + 2, nvm0)


# ------------------------------------------------------------------------------------------------ phases
# MMA phase k = [PV(k - 1), means, QK(k)], k = 0 .. n (n: the key tiles of the CTA); the warps of a pair alternate (w
# first). The means of the step pair (2m, 2m + 1) take phase 2m + 2: 16 BF16 m16n8k16 (8 channel tiles x 2 query
# n-tiles), their v_mean rows in pair buffer m % 2; an unpaired last step's take phase n. Stage k % 3 holds what phase
# k reads (V of step k - 1, K' of step k; for even k the pair's v_mean rows); the eighth warp to release it after
# phase k refills it for phase k + 3.


def qk_phase(st, base, kwait, turn):
    """The MMA phase [QK] of this thread on the stage at shared address base: S = Q' K'^T of the stage's K', returned
    as 16 x 4 accumulators. kwait: the stage's (full barrier, parity) to wait for before the K' scale words, or None
    when the warp is past it; turn: st.turn, or st.turn_held when the warp holds the turn (mma_phase)."""
    return mma_phase(st.P, st.O, None, base, base + SF_OFF, *st.mma, kwait=kwait, turn=turn)


def pv_qk_phase(st, base, swait, means=None):
    """The dense MMA phase [PV, means, QK] of this thread on the stage at shared address base: O += P V of the
    stage's V, the means MMAs of the pair buffer `means` (None: an odd phase, without), then the QK of the stage's K',
    returned as 16 x 4 accumulators. swait: the stage's (full barrier, parity), waited for before the first fragment
    loads and the turn (mma_phase)."""
    return mma_phase(
        st.P,
        st.O,
        base + V_OFF,
        base,
        base + SF_OFF,
        *st.mma,
        turn=st.turn,
        swait=swait,
        vmoff=st.vmoff,
        means=means,
    )


def softmax_init():
    """The softmax state of a thread's rows g and g + 8 before the first tile, as the steps pass it on: the anchors
    (-inf), the quad's partial row sums (0) and the landing path (1: both -inf anchors require the safe I-ExpCast
    path)."""
    return Float32(-math.inf), Float32(-math.inf), Float32(0.0), Float32(0.0), Int32(1)


def softmax_step(st, acc, sm, lim, mask_tail=None, stage=None, refill=None):
    """softmax_tile of the scores acc of this thread's tile from the softmax state sm (softmax_init): P and its block
    sums into st.P, a rescale into st.O; returns the next state. lim, mask_tail: the tile's valid keys and the
    run-time switch of their mask. stage: the stage whose release by this warp the softmax counts (release_count)
    before the landing, the count going to refill after it; None: the phase releases no stage."""
    release = (lambda: release_count(st, stage)) if stage is not None else None
    return softmax_tile(acc, st.O, st.P, *sm, st.sc, st.q, lim, mask_tail, release, refill)


@cute.jit
def loop_phase(st, k, stage, ph, nzp, sm, refill):
    """Phase k of 1 .. n - 2 on stage `stage` = k % 3 (ph: the parity of its full barrier): PV(k - 1), for an even k
    the means of the step pair (k - 2, k - 1), and QK(k), in the form the quarter mask of P picks; then the softmax of
    tile k, which counts the warp's release of the stage and lets the eighth warp refill it (refill(count)). nzp:
    step k - 2's P had a nonzero byte (or V / v_mean a non-finite value). Returns that flag of step k - 1 and the
    softmax state."""
    P = st.P
    O = st.O
    stk = st.smem + stage * STAGE_BYTES
    sw = (st.fbar + stage * 8, ph)
    acc2 = [[Float32(0.0)] * 4 for _ in range(16)]
    pv_mask = P[20] | (st.vnan * 15)
    nz = Int32(0)
    if pv_mask != 0:
        nz = Int32(1)
    # even phase: the means of the step pair (k - 2, k - 1), skipped when both steps' P are zero
    mpar = k & 1
    mcond = (mpar ^ 1) & (nz | nzp)
    mbase = pair_buffer(st, k)
    if pv_mask == 15:
        # the parity branch sits ahead of the stage wait and the turn; the even copy pipelines the pair's means
        # steps between its PV and QK steps
        if mpar == 0:
            acc2 = pv_qk_phase(st, stk, sw, mbase)
        else:
            acc2 = pv_qk_phase(st, stk, sw)
    else:
        # the nested branches write the QK accumulators to the register tensor A2, which holds them across the join
        A2 = cute.make_rmem_tensor(cute.make_layout(64), Float32)
        if (pv_mask | mcond) == 0:
            # no PV and no pending means: the stage wait, scale words and first K' fragments ahead of the turn
            store64(A2, qk_phase(st, stk, sw, st.turn))
        else:
            # the order of the dense phase: stage wait, turn, PV (of the nonzero quarters only) and means, QK,
            # early pass
            take_turn(st, sw)
            if pv_mask != 0:
                pv_sparse_quarters(P, O, stk + V_OFF, st.voff, pv_mask)
            means_pair_guarded(mcond, P, O, mbase, st.vmoff)
            store64(A2, qk_phase(st, stk, None, st.turn_held))
        acc2 = [[A2[4 * n + e] for e in range(4)] for n in range(16)]
    # this warp is past its reads of stage k % 3 (its MMA phase): the softmax counts the release and the last of
    # the eight refills it with K' of step k + 3, V of step k + 2 and, for odd k, the v_mean rows of the pair
    # (k + 1, k + 2) (or of an unpaired last step k + 2)
    sm = softmax_step(st, acc2, sm, KT, stage=stage, refill=refill)
    return nz, sm


@cute.jit
def last_qk_phase_full(st, kl: cutlass.Constexpr, nzp, sm, lim: cutlass.Constexpr):
    """Phase kl = nT - 1 >= 1 of a whole unit: PV(kl - 1), for an even kl the means of the step pair (kl - 2, kl - 1)
    (behind their branch on nzp when P is zero), and QK(kl); then the softmax of the last tile, its keys from lim on
    masked. No phase reuses the stage, so the softmax counts no release. Returns the nonzero-P flag of step kl - 1
    and the softmax state."""
    stk = st.smem + (kl % STAGES) * STAGE_BYTES
    sw = (st.fbar + (kl % STAGES) * 8, Int32((kl // STAGES) & 1))
    acc3 = [[Float32(0.0)] * 4 for _ in range(16)]
    pvn = packed_pv_needed(st.P, st.vnan)
    mbase3 = pair_buffer(st, kl)
    if pvn != 0:
        acc3 = pv_qk_phase(st, stk, sw, mbase3 if kl % 2 == 0 else None)
    else:
        take_turn(st, sw)
        if cutlass.const_expr(kl % 2 == 0):
            means_pair_guarded(nzp, st.P, st.O, mbase3, st.vmoff)
        acc3 = qk_phase(st, stk, None, st.turn_held)
    sm = softmax_step(st, acc3, sm, lim)
    return pvn, sm


@cute.jit
def last_qk_phase_part(st, kl, nzp, sm, lim: cutlass.Constexpr, piece, split: cutlass.Constexpr):
    """Phase kl = nK - 1 >= 1 of a key-range piece, kl known at run time or at trace time: as last_qk_phase_full, the
    parity of kl behind a branch. The softmax masks the keys from lim on in the last piece (piece == split - 1),
    whose last tile is the partial one."""
    P = st.P
    O = st.O
    stk = st.smem + (kl % STAGES) * STAGE_BYTES
    sw = (st.fbar + (kl % STAGES) * 8, (kl // STAGES) & 1)
    acc3 = [[Float32(0.0)] * 4 for _ in range(16)]
    pvn = packed_pv_needed(P, st.vnan)
    mpar3 = kl & 1
    mcond3 = (mpar3 ^ 1) & (pvn | nzp)
    mbase3 = pair_buffer(st, kl)
    if pvn != 0:
        if mpar3 == 0:
            acc3 = pv_qk_phase(st, stk, sw, mbase3)
        else:
            acc3 = pv_qk_phase(st, stk, sw)
    else:
        take_turn(st, sw)
        means_pair_guarded(mcond3, P, O, mbase3, st.vmoff)
        acc3 = qk_phase(st, stk, None, st.turn_held)
    sm = softmax_step(st, acc3, sm, lim, mask_tail=Int32(piece == split - 1))
    return pvn, sm


@cute.jit
def last_pv_phase(st, n, nzp):
    """Phase n, the last of a CTA with n key tiles (n known at run time or at trace time): PV(n - 1) from stage n % 3,
    skipped for a zero P, and the means of the last step pair (even n; nzp: the nonzero-P flag of its first step) or
    of the unpaired last step, whose v_mean rows are in the stage's K' region. The warp waits for the stage and for
    its turn first."""
    take_turn(st, (st.fbar + (n % STAGES) * 8, (n // STAGES) & 1))
    pvl = packed_pv_needed(st.P, st.vnan)
    if pvl != 0:
        # the MMA phase [PV] on the stage: O += P V of its V
        mma_phase(st.P, st.O, st.smem + (n % STAGES) * STAGE_BYTES + V_OFF, None, None, *st.mma)
    means_last(
        (n & 1) ^ 1,
        pvl,
        nzp,
        st.P,
        st.O,
        pair_buffer(st, n),
        st.smem + (n % STAGES) * STAGE_BYTES,
        st.vmoff,
    )


@cute.jit
def store_unit_rows(
    st, f, o_ptr, nT: cutlass.Constexpr, S: cutlass.Constexpr, H: cutlass.Constexpr
):
    """store_rows of this warp's 16 rows of a whole unit (f: inv_row_sums), staged in a stage that no warp reads any
    more (every thread is past the CTA barrier of the body): the warps 0-3 stage their rows in the stage of tile
    nT - 2, the warps 4-7 in that of tile nT - 3."""
    epi_a = st.smem + ((nT - 2) % STAGES if nT >= 2 else 1) * STAGE_BYTES
    epi_b = st.smem + (nT % STAGES if nT >= 3 else 2) * STAGE_BYTES
    ep = epi_a
    if st.grp == 1:
        ep = epi_b
    store_rows(
        st.O,
        f,
        st.vsc,
        ep + (st.warp & 3) * (16 * EP),
        st.g,
        st.q,
        st.lane,
        st.warp,
        st.qt,
        o_ptr.toint(),
        st.bb,
        st.hh,
        S,
        H,
    )


# ------------------------------------------------------------------------------------------------ CTA bodies
@cute.jit
def full_body(ops, vs_ptr, o_ptr, S: cutlass.Constexpr, H: cutlass.Constexpr, unit):
    """Query tile `unit` (linear over (tile, head, batch)) over all key tiles: the final bf16 rows. ops: the operands
    of attn_kernel that both bodies read."""
    nT = (S + KT - 1) // KT
    LIM = S - (nT - 1) * KT  # valid keys of the last tile
    tidx = Int32(cute.arch.thread_idx()[0])
    st = cta_state(tidx, unit % nT, (unit // nT) % H, unit // (nT * H), DIST)
    # diagonal-first key order: steps 0 .. nT - 2 take full tiles qt, qt + 1, ... (mod nT - 1), the last step the
    # partial tile nT - 1
    rot = st.qt % (nT - 1) if nT >= 2 else Int32(0)
    start_ring(st, ops, H, nT)
    if st.warp == 4:
        first_stages_full(st, rot, nT)
    if st.grp == 1:
        # warp w takes the first turn of its pair
        pass_turn(st)
    load_operands(st, ops, S, H, nT, vs_ptr)
    sm = softmax_init()
    nzp = Int32(0)  # step k - 2's P had a nonzero byte (or V / v_mean a non-finite value)

    # ---- phase 0: QK(0)
    acc = qk_phase(st, st.smem, (st.fbar, Int32(0)), st.turn)
    if cutlass.const_expr(nT > 2):
        # the softmax counts this warp's release of stage 0 (the scores stay in registers); the eighth warp refills it
        vt2 = step_tile(2, rot, nT)
        sm = softmax_step(
            st, acc, sm, KT, stage=0, refill=lambda old: refill_full0(st, old, rot, nT, vt2)
        )
    else:
        # no phase 3: stage 0 is not refilled (a single key tile is a full one: launch.run sends S < KT to small.py)
        sm = softmax_step(st, acc, sm, KT)
    sk_i = Int32(1)  # stage of step k (loop-carried: no modulo in the loop)
    ph = Int32(0)  # its full-barrier parity, (k // STAGES) & 1
    # ---- phases k = 1 .. nT - 2
    for k in cutlass.range(1, nT - 1, unroll=1):
        nzp, sm = loop_phase(
            st, k, sk_i, ph, nzp, sm, lambda old: refill_loop_full(st, old, k, sk_i, rot, nT)
        )
        sk_i, ph = stage_next(sk_i, ph, STAGES)
    # ---- phase nT - 1: PV(nT - 2), means(nT - 2), QK(nT - 1) with the last tile's mask
    if cutlass.const_expr(nT >= 2):
        nzp, sm = last_qk_phase_full(st, nT - 1, nzp, sm, LIM)
    # ---- phase nT: PV(nT - 1) from stage nT % 3 and the means of the last step pair (or unpaired last step)
    # (every phase waits for its stage, so no copy is in flight after it)
    last_pv_phase(st, nT, nzp)
    if st.grp == 0:
        # warp w + 4 takes the last turn of its pair
        pass_turn(st)

    # ---- epilogue: row sums over the quad; o = O * v8_scale / l, staged through a free stage as bf16 rows
    _, _, l0, l1, _ = sm
    f = inv_row_sums(l0, l1, st.q)
    # the warp pairs run independently: every warp must be past its last stage read before any stage is reused for
    # the output rows
    barrier_cta_sync(Int32(0), thread_count=NTHR)
    store_unit_rows(st, f, o_ptr, nT, S, H)


@cute.jit
def partial_body(
    ops,
    part_ptr,
    S: cutlass.Constexpr,
    H: cutlass.Constexpr,
    split: cutlass.Constexpr,
    unit0: cutlass.Constexpr,
    bid,
):
    """Piece bid % split of the key tiles of query tile unit0 + bid / split: its FP32 partials for the merge launch
    (epilogue.merge_partial_kernel). ops: the operands of attn_kernel that both bodies read."""
    nT = (S + KT - 1) // KT
    LIM = S - (nT - 1) * KT  # valid keys of the last tile
    tidx = Int32(cute.arch.thread_idx()[0])
    unit = unit0 + bid // split
    piece = bid % split
    qt = unit % nT
    hh = (unit // nT) % H
    bb = unit // (nT * H)
    # the piece's key tiles t0 .. t0 + nK - 1, in this order
    if cutlass.const_expr(nT % split == 0):
        t0 = piece * (nT // split)
        nK = nT // split
    else:
        t0 = piece * nT // split
        t1 = (piece + 1) * nT // split
        nK = t1 - t0
    st = cta_state(tidx, qt, hh, bb, min(DIST, nT // split))
    start_ring(st, ops, H, nT)
    if st.warp == 4:
        first_stages_part(st, t0, nK)
    if st.grp == 1:
        # warp w takes the first turn of its pair
        pass_turn(st)
    load_operands(st, ops, S, H, nT)
    sm = softmax_init()
    nzp = Int32(0)  # step k - 2's P had a nonzero byte (or V / v_mean a non-finite value)

    # ---- phase 0: QK(0)
    acc = qk_phase(st, st.smem, (st.fbar, Int32(0)), st.turn)
    sm = softmax_step(st, acc, sm, KT, stage=0, refill=lambda old: refill_part0(st, old, t0, nK))
    sk_i = Int32(1)  # stage of step k (loop-carried: no modulo in the loop)
    ph = Int32(0)  # its full-barrier parity, (k // STAGES) & 1
    # ---- phases k = 1 .. nK - 2
    for k in cutlass.range(1, nK - 1, unroll=1):
        nzp, sm = loop_phase(
            st, k, sk_i, ph, nzp, sm, lambda old: refill_loop_part(st, old, k, sk_i, t0, nK)
        )
        sk_i, ph = stage_next(sk_i, ph, STAGES)
    # ---- phase nK - 1: PV(nK - 2), means(nK - 2), QK(nK - 1) with the last tile's mask
    nzp, sm = last_qk_phase_part(st, nK - 1, nzp, sm, LIM, piece, split)
    # ---- phase nK: PV(nK - 1) from stage nK % 3 and the means of the last pair (or unpaired last step)
    last_pv_phase(st, nK, nzp)
    if st.grp == 0:
        # warp w + 4 takes the last turn of its pair
        pass_turn(st)

    # O with its means term
    anc0, anc1, l0, l1, _ = sm
    store_partials(part_ptr, bid, st.tidx, st.O, anc0, anc1, l0, l1)


# ------------------------------------------------------------------------------------------------ kernel
@cute.kernel
def attn_kernel(
    q_ptr: cute.Pointer,
    qsf_ptr: cute.Pointer,
    ss_ptr: cute.Pointer,
    vs_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    part_ptr: cute.Pointer,
    flags_ptr: cute.Pointer,
    tma_atom_k: cute.CopyAtom,
    tma_k: cute.Tensor,
    tma_atom_v: cute.CopyAtom,
    tma_v: cute.Tensor,
    tma_atom_s: cute.CopyAtom,
    tma_s: cute.Tensor,
    tma_atom_m: cute.CopyAtom,
    tma_m: cute.Tensor,
    S: cutlass.Constexpr,
    H: cutlass.Constexpr,
    n_full: cutlass.Constexpr,
    n_tail: cutlass.Constexpr,
    split: cutlass.Constexpr,
):
    """One launch: CTAs 0 .. n_full - 1 take whole query tiles; the CTAs after them take the key-range pieces (split
    per unit) of the last n_tail units of the grid (launch.split_plan). ss_ptr, vs_ptr: the score scale and v8_scale;
    part_ptr: the pieces' partials; flags_ptr: the per-tile flags of the scan launch."""
    bid_, _, _ = cute.arch.block_idx()
    bid = Int32(bid_)
    # what both bodies read: Q', its scales, the score scale, the scan flags, and the TMA atoms and tensors of K', V,
    # the K' scales and v_mean
    tiles = (tma_atom_k, tma_k, tma_atom_v, tma_v, tma_atom_s, tma_s, tma_atom_m, tma_m)
    ops = (q_ptr, qsf_ptr, ss_ptr, flags_ptr, tiles)
    if cutlass.const_expr(n_tail == 0):
        full_body(ops, vs_ptr, o_ptr, S, H, bid)
    elif cutlass.const_expr(n_full == 0):
        partial_body(ops, part_ptr, S, H, split, 0, bid)
    else:
        if bid < n_full:
            full_body(ops, vs_ptr, o_ptr, S, H, bid)
        else:
            partial_body(ops, part_ptr, S, H, split, n_full, bid - n_full)
