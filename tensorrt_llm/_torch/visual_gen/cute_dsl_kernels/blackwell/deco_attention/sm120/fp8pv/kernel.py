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

o = (P V + S M) * v8_scale / l. A CTA takes 128 query rows of one (batch, head) (8 warps x 16 rows) over tiles of 128
keys; per key tile:
    * QK: S = Q' K'^T over the 192 channels of Q' and K' on the block-scaled NVFP4 MMA (kind::mxf4nvf4, m16n8k64,
      UE4M3 scales).
    * Softmax (softmax.py): online, log2 units, lazy rescale at T = 4, I-ExpCast into E4M3 P, row sum of quantized P.
    * PV: O += P V on the FP8 MMA (m16n8k32), V being the E4M3 V minus its means of every 128 keys.
    * Means term: the block sums of P times v_mean on BF16 MMAs. v_mean holds one row per 16 keys, a tile's mean in
      all 8 of its rows; the means MMA weights the rows by the 16-key block sums of P, which add up to the tile's.
Schedule:
    * MMA phase k = [PV(k - 1), means, QK(k)], k = 0 .. n (n: the key tiles of the CTA).
    * The warps are paired (w, w + 4) and the four pairs run independently. The two warps of a pair alternate in
      issuing MMAs under a named barrier of 64 threads, w first: one issues its MMA phase while the other runs the
      softmax of its scores.
    * The means of the step pair (2m, 2m + 1) take phase 2m + 2: 16 BF16 m16n8k16 (8 channel tiles x 2 query
      n-tiles), their v_mean rows in pair buffer m % 2. Those of an unpaired last step take the last phase, n.
    * Key tiles run diagonal first: the query tile's own key tile, then onward, wrapping; the partial last tile last.
Ring:
    * Stage k % 3 of the TMA ring holds exactly what phase k reads, under one full barrier: K' and its scale atoms of
      step k, V of step k - 1 and, for an even phase, the v_mean rows of its step pair in one of two pair buffers.
    * A warp waits for the barrier before its first fragment loads and its turn, so that a phase starts only when all
      of its operands are in shared memory. No warp waits for the others.
    * The softmax keeps the scores in registers, so the reads of a stage by a warp end with its MMA phase: it counts
      its release then, and the eighth warp to do so refills the stage with K' of step k + 3 and V of step k + 2.
Zero-P skipping, exact:
    * A warp-uniform mask of the 32-key quarters with a nonzero packed P byte picks the dense phase, the QK-only
      phase or, for a mixed tile, one indexed branch per quarter around its V loads and MMAs.
    * A scan of V and v_mean before the attention (scan.py) flags the heads with a non-finite value, which take
      every quarter (0 * NaN must propagate).
Split tail:
    * One launch holds the whole units (query tile, head, batch) followed by the key-range pieces of the last
      units % nsm + nsm // 2 units of the grid (nsm: the SM count of the device; launch.split_plan).
    * A merge launch combines the FP32 partials of the pieces (epilogue.py).
"""

import math
from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32, Int64

from ...common.dsl import BarrierRedux, barrier_cta_red, barrier_cta_sync
from ...common.mem import load_invariant, mbar_wait, sptr
from ...common.state import State
from ..ptx.fp8pv import atom_add_shared_u32_lane0, lane_zero, packed_pv_needed, stage_next
from ..valid_len import valid_len
from .epilogue import inv_row_sums, store_partials, store_rows, store_zero_rows
from .layout import (
    BAR_OFF,
    CNT_OFF,
    EP,
    K_BYTES,
    KT,
    LEN_OFF,
    NTHR,
    QT,
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


def pass_turn(st):
    """Hand the turn of the warp pair to the partner warp: an arrival on its turn barrier."""
    # non-aligned: the warps of an MMA phase reach the turn barriers from two code paths (dense and zero-P branches)
    cute.arch.barrier_arrive(barrier_id=st.bid_pass, number_of_threads=64, aligned=False)


def take_turn(st, sw):
    """Start an MMA phase without lookahead loads: wait for the full barrier sw = (barrier, parity), then the turn."""
    mbar_wait(sw[0], sw[1])
    barrier_cta_sync(st.bid_wait, thread_count=64)


def cta_state(tidx, qt, hh, bb, dist):
    """Return the state st of thread tidx of the CTA of unit (query tile qt, head hh, batch bb); dist: its lookahead.

    The steps of a CTA body share st; start_ring and load_operands complete it, then no step assigns a field. Fields:
        tidx, warp, lane, g, q, grp   the thread: its rows are g = lane / 4 and g + 8 of the warp, q = lane % 4 its
                                      place in the quad; grp 0: warps 0-3 (rows 0-63), 1: warps 4-7 (rows 64-127)
        qt, hh, bb, dist              the unit (query tile, head, batch) and the ldmatrix lookahead of the MMA phases
        bid_wait, bid_pass            the turn barriers of the warp pair (w, w + 4): this warp's and its partner's
        turn, turn_held               the turn of an MMA phase as mma_phase takes it: (take, pass, EARLY)
        sp, smem, fbar                shared memory: its pointer, its address, the address of the stages' full barriers
        tma, vnan                     start_ring: the TMA views, the non-finite flag of the head
        sc, vsc, lz                   load_operands: the score scale, v8_scale, the per-lane zero of the release count
        mma, voff, vmoff              load_operands: the thread's operands of mma_phase in its order (the Q' fragments
                                      and scale words, the per-lane offsets of the K' rows, K' scale words and V row,
                                      the lookahead), and the per-lane offsets of the V row and the v_mean rows
        O, P                          load_operands: the register tensors of the O accumulators and of the packed P
                                      with its block sums and quarter mask (softmax.py)
    """
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
    """Set up the ring of the CTA before its first load: st.tma, the stage barriers and counters (thread 0), st.vnan."""
    _, _, _, flags_ptr, tiles = ops
    vflags = flags_ptr.toint() + Int64((st.bb * H + st.hh) * nT) * 4
    st.tma = make_tma_views(st.sp, *tiles, st.hh, st.bb)
    if st.tidx == 0:
        for stage in cutlass.range_constexpr(STAGES):
            cute.arch.mbarrier_init(sptr(Int64, st.fbar + stage * 8), 1)
            cute.arch.store(sptr(Int32, st.smem + CNT_OFF + stage * 4), Int32(0))
        cute.arch.mbarrier_init_fence()
    # a non-finite V or v_mean anywhere in this (batch, head) disables zero-P skipping for the head; behind the barrier
    # of this reduction every thread sees the initialized barriers and counters
    st.vnan = cute.arch.make_warp_uniform(
        Int32(
            barrier_cta_red(
                head_v_nan(vflags, st.tidx, nT) != 0, 0, BarrierRedux.OR, thread_count=NTHR
            )
        )
    )


def load_operands(st, ops, S, H, nT, vs_ptr=None):
    """Load what a thread keeps in registers over the key sequence; v8_scale only in the body that writes the rows."""
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
    """Count the release of `stage` by this warp (its last read is behind it); return the old count on every lane."""
    cute.arch.sync_warp()
    return cute.arch.shuffle_sync(
        atom_add_shared_u32_lane0(st.smem + CNT_OFF + stage * 4 + st.lz * 4, st.lane), 0
    )


@cute.jit
def rot_tile(k, rot, nm1: cutlass.Constexpr):
    """Return the key tile of step k < nm1, diagonal first: (k + rot) mod nm1; the partial last tile stays last."""
    t = k + rot
    if t >= nm1:
        t = t - nm1
    return t


def step_tile(j, rot, nT):
    """Return the key tile of step j of a whole unit (j at trace time; below 0 counts as 0): rot_tile, then nT - 1."""
    return Int32(nT - 1) if j >= nT - 1 else rot_tile(Int32(max(j, 0)), rot, nT - 1)


def pair_buffer(st, k):
    """Return the pair buffer of the v_mean rows that phase k reads (even k: those of the step pair (k - 2, k - 1))."""
    return st.smem + VMP_OFF + (((k - 1) >> 1) & 1) * (2 * VM_BYTES)


@cute.jit
def issue_tile(st, stage, ktile, vtile, has_k, has_v, vma, vmb, nvm, vbuf):
    """Issue the TMA copies of what phase j reads into its stage, under the full barrier of the stage.

    * K' (three atom columns and the three scale atoms) of step j for QK(j) and V of step j - 1 for PV(j - 1).
    * nvm = 2, an even phase j: the v_mean rows of the step pair (j - 2, j - 1) (tiles vma, vmb) into pair buffer vbuf.
    * nvm = 1, an odd last phase: those of the unpaired last step (tile vma) into the unused K' region of the stage.
    """
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


@cute.jit
def first_stages(st, tile, n, n_min: cutlass.Constexpr):
    """Issue the first loads of a CTA over n key tiles (warp 4); tile(j): the key tile of step j.

    * Stage j of the phases 0 .. 2, as far as the CTA has a phase j: K' of step j (j < n) and V of step j - 1; phase 2
      takes the v_mean rows of the step pair (0, 1), a last phase 1 (n == 1) those of step 0.
    * n is known at trace time (a whole unit) or at run time with the trace-time lower bound n_min (a key-range piece:
      2; a range with lengths: 1): what the bound decides is decided at trace time.
    """
    static = isinstance(n, int)
    for tl in cutlass.range_constexpr(STAGES):
        # the CTA has a phase tl; step tl has a tile; a single step brings its own v_mean rows in stage 1
        phase = (tl <= n) if static else (True if tl <= n_min else n >= tl)
        has_k = Int32(1 if tl < n else 0) if static else (Int32(1) if tl < n_min else Int32(tl < n))
        nvm = Int32(2 if tl == 2 else 0)
        if cutlass.const_expr(tl == 1 and (n == 1 if static else n_min < 2)):
            nvm = Int32(1) if static else Int32(n == 1)
        vt = tile(max(tl - 1, 0))
        if phase:
            issue_tile(
                st,
                tl,
                tile(tl),
                vt,
                has_k,
                Int32(1 if tl >= 1 else 0),
                tile(0) if tl == 2 else vt,
                vt,
                nvm,
                Int32(0),
            )


@cute.jit
def refill_if_last(st, old, ktile, vtile, has_k, vma, vmb, nvm):
    """Refill stage 0 for phase 3 (issue_tile with a V tile and pair buffer 1) when the eighth warp has released it."""
    if (old & 7) == 7:
        cute.arch.fence_proxy("async.shared", space="cta")
        issue_tile(st, 0, ktile, vtile, has_k, Int32(1), vma, vmb, nvm, Int32(1))


def refill_full0(st, old, rot, nT, vt2):
    """Refill stage 0 for phase 3 of a whole unit: K' of step 3, V of step 2, the v_mean rows of an unpaired step 2."""
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
    """Refill the stage of loop phase k of the full body for phase k + 3 (k + 2 < nT), as refill_if_last.
    The refilling warp alone computes the tiles: K' of step k + 3, V of step k + 2 and the v_mean rows of the pair
    (k + 1, k + 2) for odd k, of an unpaired last step k + 2 = nT - 1 for even k."""
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
    """Refill as refill_loop_full, for a key-range piece: tiles t0 + step; a phase k + 3 to load when k + 2 < nK."""
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
    """Refill stage 0 for phase 3 of a piece (nK > 2)."""
    if 2 < nK:
        hk0 = Int32(0)
        if 3 < nK:
            hk0 = Int32(1)
        nvm0 = Int32(0)
        if nK == 3:
            nvm0 = Int32(1)
        refill_if_last(st, old, t0 + 3, t0 + 2, hk0, t0 + 2, t0 + 2, nvm0)


# The ring of a range of key tiles of a call with lengths. The range is rng = (t0, n, rot, lim), four functions of no
# argument: the n() >= 1 tiles from t0(), step j taking tile lens_tile(j) in the order of a whole unit (diagonal
# first with rot(), the last tile last) or, with rot 0, of a key-range piece; lim(): the valid keys of the last tile.


def valid_keys(st):
    """Return the valid tokens of the batch element (with lengths), stored by thread 0 before the start_ring barrier."""
    return cute.arch.load(sptr(Int32, st.smem + LEN_OFF), Int32)


@cute.jit
def lens_tile(j, rng):
    """Return the key tile of step j of the range rng; a step past the last counts as the last."""
    n = rng[1]()
    t = n - 1
    if j < n - 1:
        t = rot_tile(j, rng[2](), n - 1)
    return rng[0]() + t


@cute.jit
def refill_lens0(st, old, rng):
    """Refill stage 0 for phase 3 of the range (n > 2), as refill_full0."""
    if (old & 7) == 7:
        n = rng[1]()
        if 2 < n:
            vt2 = lens_tile(2, rng)
            cute.arch.fence_proxy("async.shared", space="cta")
            issue_tile(
                st,
                0,
                lens_tile(3, rng),
                vt2,
                Int32(3 < n),
                Int32(1),
                vt2,
                vt2,
                Int32(n == 3),
                Int32(1),
            )


@cute.jit
def refill_loop_lens(st, old, k, stage, rng):
    """Refill as refill_loop_full, for the range: a phase k + 3 to load when k + 2 < n."""
    if (old & 7) == 7:
        n = rng[1]()
        if k + 2 < n:
            vt = lens_tile(k + 2, rng)
            nvm = Int32(0)
            vma = vt
            if (k & 1) == 1:
                nvm = Int32(2)
                vma = lens_tile(k + 1, rng)
            else:
                if k + 2 == n - 1:
                    nvm = Int32(1)
            cute.arch.fence_proxy("async.shared", space="cta")
            issue_tile(
                st,
                stage,
                lens_tile(k + 3, rng),
                vt,
                Int32(k + 3 < n),
                Int32(1),
                vma,
                vt,
                nvm,
                ((k + 2) >> 1) & 1,
            )


# ------------------------------------------------------------------------------------------------ phases


def qk_phase(st, base, kwait, turn):
    """Run the MMA phase [QK] of this thread on the stage at base: S = Q' K'^T of its K', as 16 x 4 accumulators.

    * kwait: the stage's (full barrier, parity) to wait for before the K' scale words; None when the warp is past it.
    * turn: st.turn, or st.turn_held when the warp holds the turn (mma_phase).
    """
    return mma_phase(st.P, st.O, None, base, base + SF_OFF, *st.mma, kwait=kwait, turn=turn)


def pv_qk_phase(st, base, swait, means=None):
    """Run the dense MMA phase [PV, means, QK] of this thread on the stage at base; return S as 16 x 4 accumulators.

    * swait: the stage's (full barrier, parity), waited for before the first fragment loads and the turn (mma_phase).
    * means: the pair buffer of the means MMAs; None in an odd phase, which has none.
    """
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
    """Return the initial softmax state: anchors -inf, row sums 0, path 1 (-inf anchors need the safe landing path)."""
    return Float32(-math.inf), Float32(-math.inf), Float32(0.0), Float32(0.0), Int32(1)


def softmax_step(st, acc, sm, lim, mask_tail=None, stage=None, refill=None):
    """Run softmax_tile on the scores of the tile from the state sm (softmax_init); return the next state.

    * lim, mask_tail: the valid keys of the tile and the run-time switch of their mask.
    * stage: the softmax counts its release by this warp (release_count), the count going to refill; None: no release.
    """
    release = (lambda: release_count(st, stage)) if stage is not None else None
    return softmax_tile(acc, st.O, st.P, *sm, st.sc, st.q, lim, mask_tail, release, refill)


@cute.jit
def loop_phase(st, k, stage, ph, nzp, sm, refill):
    """Run phase k of 1 .. n - 2 on its stage, k % 3 (ph: the parity of its full barrier), then the softmax of tile k.

    * PV(k - 1), for an even k the means of the step pair (k - 2, k - 1), and QK(k), in the form the quarter mask of P
      picks; the softmax counts the release of the stage and lets the eighth warp refill it (refill(count)).
    * nzp: the P of step k - 2 had a nonzero byte (or V / v_mean a non-finite value); returned for step k - 1 with
      the softmax state.
    """
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
    sm = softmax_step(st, acc2, sm, KT, stage=stage, refill=refill)
    return nz, sm


@cute.jit
def last_qk_phase_full(st, kl: cutlass.Constexpr, nzp, sm, lim: cutlass.Constexpr):
    """Run phase kl = nT - 1 >= 1 of a whole unit, then the softmax of the last tile with its keys from lim on masked.
    PV(kl - 1), for an even kl the means of the step pair (kl - 2, kl - 1) (behind their branch on nzp when P is zero),
    and QK(kl). No phase reuses the stage, so the softmax counts no release. Returns as loop_phase."""
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
    """Run phase kl = nK - 1 >= 1 of a key-range piece as last_qk_phase_full, the parity of kl behind a branch. The
    softmax masks the keys from lim on in the last piece (piece == split - 1), whose last tile is the partial one."""
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
    """Run phase n, the last of a CTA with n key tiles, after the wait for stage n % 3 and for the turn.
    PV(n - 1) from the stage, skipped for a zero P, and the means of the last step pair (even n; nzp: the nonzero-P
    flag of its first step) or of the unpaired last step, whose v_mean rows are in the K' region of the stage."""
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
    """Store the 16 rows of this warp of a whole unit (store_rows; f: inv_row_sums) through a stage no warp still reads.
    Every thread is past the CTA barrier: warps 0-3 use the stage of tile nT - 2, warps 4-7 that of tile nT - 3."""
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
# The key tiles of a CTA, for run_phases: n tiles (n_min: its lower bound at trace time), tile(j) of step j; lim0():
# the valid keys of tile 0; refill0(): the refill of stage 0 (None: no phase 3); refill(count, k, stage): that of
# loop phase k; last_qk(nzp, sm): phase n - 1. lim0 and refill0 are functions: they emit where run_phases calls them.


def whole_tiles(st, rot, nT, lim):
    """All nT key tiles for a whole unit: diagonal first by rot, the partial last tile (lim valid keys) last. A single
    key tile is a full one: launch.run sends S < KT to small.py."""

    def refill0():
        if nT <= 2:
            # no phase 3: stage 0 is not refilled
            return None
        vt2 = step_tile(2, rot, nT)
        return lambda old: refill_full0(st, old, rot, nT, vt2)

    return SimpleNamespace(
        n=nT,
        n_min=nT,
        tile=lambda j: step_tile(j, rot, nT),
        lim0=lambda: KT,
        refill0=refill0,
        refill=lambda old, k, stage: refill_loop_full(st, old, k, stage, rot, nT),
        last_qk=lambda nzp, sm: last_qk_phase_full(st, nT - 1, nzp, sm, lim),
    )


def piece_tiles(st, t0, nK, lim, piece, split):
    """The nK >= 2 key tiles from t0 of piece `piece` of `split`; the last piece ends with the partial tile (lim)."""
    return SimpleNamespace(
        n=nK,
        n_min=2,
        tile=lambda j: t0 + j,
        lim0=lambda: KT,
        refill0=lambda: lambda old: refill_part0(st, old, t0, nK),
        refill=lambda old, k, stage: refill_loop_part(st, old, k, stage, t0, nK),
        last_qk=lambda nzp, sm: last_qk_phase_part(st, nK - 1, nzp, sm, lim, piece, split),
    )


def lens_tiles(st, rng):
    """The key tiles of the range rng of a call with lengths; the last one holds rng[3]() valid keys."""
    n = rng[1]()
    return SimpleNamespace(
        n=n,
        n_min=1,
        tile=lambda j: lens_tile(j, rng),
        lim0=lambda: Int32(cutlass.select_(n == 1, rng[3](), Int32(KT))),
        refill0=lambda: lambda old: refill_lens0(st, old, rng),
        refill=lambda old, k, stage: refill_loop_lens(st, old, k, stage, rng),
        last_qk=lambda nzp, sm: last_qk_phase_part(st, n - 1, nzp, sm, rng[3](), Int32(0), 1),
    )


@cute.jit
def run_phases(st, ops, seq, S: cutlass.Constexpr, H: cutlass.Constexpr, vs_ptr=None):
    """Run the phases 0 .. n of a CTA over its key tiles seq (whole_tiles, piece_tiles, lens_tiles); return the softmax
    state. A single tile goes from its QK straight to the last phase."""
    nT = (S + KT - 1) // KT
    n, refill, last_qk = seq.n, seq.refill, seq.last_qk
    start_ring(st, ops, H, nT)
    if st.warp == 4:
        first_stages(st, seq.tile, n, seq.n_min)
    if st.grp == 1:
        # warp w takes the first turn of its pair
        pass_turn(st)
    load_operands(st, ops, S, H, nT, vs_ptr)
    sm = softmax_init()
    nzp = Int32(0)  # step k - 2's P had a nonzero byte (or V / v_mean a non-finite value)
    # ---- phase 0: QK(0)
    acc = qk_phase(st, st.smem, (st.fbar, Int32(0)), st.turn)
    lim0 = seq.lim0()
    refill0 = seq.refill0()
    sm = softmax_step(st, acc, sm, lim0, stage=None if refill0 is None else 0, refill=refill0)
    sk_i = Int32(1)  # stage of step k (loop-carried: no modulo in the loop)
    ph = Int32(0)  # its full-barrier parity, (k // STAGES) & 1
    # ---- phases k = 1 .. n - 2
    for k in cutlass.range(1, n - 1, unroll=1):
        nzp, sm = loop_phase(st, k, sk_i, ph, nzp, sm, lambda old: refill(old, k, sk_i))
        sk_i, ph = stage_next(sk_i, ph, STAGES)
    # ---- phase n - 1 of n >= 2: PV(n - 2), means(n - 2), QK(n - 1) with the last tile's mask
    if cutlass.const_expr(seq.n_min >= 2):
        nzp, sm = last_qk(nzp, sm)
    elif cutlass.const_expr(not isinstance(n, int)):
        if n >= 2:
            nzp, sm = last_qk(nzp, sm)
    # ---- phase n: PV(n - 1) from stage n % 3 and the means of the last step pair (or unpaired last step)
    # (every phase waits for its stage, so no copy is in flight after it)
    last_pv_phase(st, n, nzp)
    if st.grp == 0:
        # warp w + 4 takes the last turn of its pair
        pass_turn(st)
    return sm


@cute.jit
def full_body(ops, vs_ptr, o_ptr, S: cutlass.Constexpr, H: cutlass.Constexpr, unit):
    """Run query tile `unit` (linear over (tile, head, batch)) over all key tiles and store its final bf16 rows."""
    nT = (S + KT - 1) // KT
    LIM = S - (nT - 1) * KT  # valid keys of the last tile
    tidx = Int32(cute.arch.thread_idx()[0])
    st = cta_state(tidx, unit % nT, (unit // nT) % H, unit // (nT * H), DIST)
    # diagonal-first key order: steps 0 .. nT - 2 take full tiles qt, qt + 1, ... (mod nT - 1), the last step the
    # partial tile nT - 1
    rot = st.qt % (nT - 1) if nT >= 2 else Int32(0)
    sm = run_phases(st, ops, whole_tiles(st, rot, nT, LIM), S, H, vs_ptr)

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
    """Run piece bid % split of the key tiles of query tile unit0 + bid / split; its FP32 partials go to the merge."""
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
    sm = run_phases(st, ops, piece_tiles(st, t0, nK, LIM, piece, split), S, H)
    anc0, anc1, l0, l1, _ = sm
    store_partials(part_ptr, bid, st.tidx, st.O, anc0, anc1, l0, l1)


def _picker(whole):
    """Return pick(a, b): a for a whole unit and b for a piece, by a select when whole is known at run time only."""
    if isinstance(whole, bool):
        return lambda a, b: a if whole else b
    return lambda a, b: Int32(cutlass.select_(whole, a, b))


@cute.jit
def body_lens(
    ops,
    lens_ptr,
    vs_ptr,
    o_ptr,
    part_ptr,
    S: cutlass.Constexpr,
    H: cutlass.Constexpr,
    split: cutlass.Constexpr,
    n_full: cutlass.Constexpr,
    n_tail: cutlass.Constexpr,
    bid,
):
    """Run the CTA body of a call with lengths: a whole unit (bid < n_full) or a key-range piece.

    The key tiles of the batch element of the unit end with the one that holds its last valid key.
    * A whole unit runs them in its order and stores its rows, those from the length on as zeros.
    * A piece runs those of its tiles among them and leaves its partials, or an anchor of -inf and zeros without any.
    * Without a valid query row a whole unit stores zeros without the phases, a piece returns (the merge stores zeros).
    """
    nT = (S + KT - 1) // KT
    tidx = Int32(cute.arch.thread_idx()[0])
    # whole: this CTA takes a whole unit, known at trace time for a launch without pieces or without whole units
    whole = True if n_tail == 0 else (False if n_full == 0 else bid < n_full)

    pick = _picker(whole)
    unit = pick(bid, n_full + (bid - n_full) // split)
    st = cta_state(tidx, unit % nT, (unit // nT) % H, unit // (nT * H), DIST)
    # a whole unit counts as the only piece of one
    piece = pick(Int32(0), (bid - n_full) % split)
    parts = pick(Int32(1), Int32(split))
    nv = valid_len(lens_ptr.toint(), st.bb, S)
    if st.tidx == 0:
        cute.arch.store(sptr(Int32, st.smem + LEN_OFF), nv)
    # the range of lens_tiles: the tiles of this CTA up to the last valid one, diagonal first for a whole unit
    r0 = piece * nT // parts
    kv = (nv + KT - 1) // KT
    r1 = cutlass.min((piece + 1) * nT // parts, kv) - r0
    r2 = pick(st.qt % cutlass.max(kv - 1, Int32(1)), Int32(0))
    r3 = Int32(cutlass.select_(r0 + r1 == kv, (nv - 1) % KT + 1, Int32(KT)))
    rng = (lambda: r0, lambda: r1, lambda: r2, lambda: r3)
    if st.qt * QT < nv:
        if r0 < kv:
            anc0, anc1, l0, l1, _ = run_phases(st, ops, lens_tiles(st, rng), S, H, vs_ptr)
            if whole:
                f = inv_row_sums(l0, l1, st.q)
                # every warp must be past its last stage read before stages 0 and 1 are reused for the output rows
                barrier_cta_sync(Int32(0), thread_count=NTHR)
                store_rows(
                    st.O,
                    f,
                    st.vsc,
                    st.smem + st.grp * STAGE_BYTES + (st.warp & 3) * (16 * EP),
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
                    valid_keys(st),
                )
            else:
                store_partials(part_ptr, bid - n_full, st.tidx, st.O, anc0, anc1, l0, l1)
        else:
            zero = cute.make_rmem_tensor(cute.make_layout(64), Float32)
            zero.fill(0.0)
            ninf = Float32(-math.inf)
            store_partials(
                part_ptr, bid - n_full, st.tidx, zero, ninf, ninf, Float32(0.0), Float32(0.0)
            )
    else:
        if whole:
            store_zero_rows(st.lane, st.warp, st.qt, o_ptr.toint(), st.bb, st.hh, S, H)


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
    lens_ptr,
):
    """Run the attention: CTAs 0 .. n_full - 1 take whole query tiles, the others key-range pieces of the split tail.

    * The pieces: split per unit, of the last n_tail units of the grid (launch.split_plan); part_ptr: their partials.
    * ss_ptr, vs_ptr: the score scale and v8_scale; flags_ptr: the per-tile flags of the scan launch.
    * lens_ptr: None, or the valid prefix lengths, which select the bodies with lengths.
    """
    bid_, _, _ = cute.arch.block_idx()
    bid = Int32(bid_)
    # what both bodies read: Q', its scales, the score scale, the scan flags, and the TMA atoms and tensors of K', V,
    # the K' scales and v_mean
    tiles = (tma_atom_k, tma_k, tma_atom_v, tma_v, tma_atom_s, tma_s, tma_atom_m, tma_m)
    ops = (q_ptr, qsf_ptr, ss_ptr, flags_ptr, tiles)
    if cutlass.const_expr(lens_ptr is not None):
        # a unit of a full batch element takes the bodies without lengths, any other the body with lengths
        nT = (S + KT - 1) // KT
        unit = bid
        if cutlass.const_expr(n_tail > 0):
            if bid >= n_full:
                unit = n_full + (bid - n_full) // split
        if valid_len(lens_ptr.toint(), unit // (nT * H), S) == S:
            if cutlass.const_expr(n_tail == 0):
                full_body(ops, vs_ptr, o_ptr, S, H, bid)
            elif cutlass.const_expr(n_full == 0):
                partial_body(ops, part_ptr, S, H, split, 0, bid)
            else:
                if bid < n_full:
                    full_body(ops, vs_ptr, o_ptr, S, H, bid)
                else:
                    partial_body(ops, part_ptr, S, H, split, n_full, bid - n_full)
        else:
            body_lens(ops, lens_ptr, vs_ptr, o_ptr, part_ptr, S, H, split, n_full, n_tail, bid)
    elif cutlass.const_expr(n_tail == 0):
        full_body(ops, vs_ptr, o_ptr, S, H, bid)
    elif cutlass.const_expr(n_full == 0):
        partial_body(ops, part_ptr, S, H, split, 0, bid)
    else:
        if bid < n_full:
            full_body(ops, vs_ptr, o_ptr, S, H, bid)
        else:
            partial_body(ops, part_ptr, S, H, split, n_full, bid - n_full)
