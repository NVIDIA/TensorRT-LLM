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
"""NVFP4 QK + NVFP4 PV attention for sm_120a (RTX PRO 6000), built on the warp-level MMAs (mma.sync).

One CTA per work unit (128 query rows of one (batch, head)) runs the whole key sequence, a tile of 128 keys at a time:
S = Q' K'^T over the 192 channels of Q' and K' on the block-scaled NVFP4 MMA (kind::mxf4nvf4, m16n8k64); the online
softmax of softmax.py (log2 units, lazy rescale at T = 4, I-ExpCast, NVFP4 P coded against its decoded block scales);
O += P V' on the same MMA, with V' = V minus its 16-key means in V's own 128 channels (key-contiguous E2M1 codes,
UE4M3 scales per channel and 16 keys); and the means term, the 16-key block sums of the unquantized P times v_mean on
BF16 MMAs (m16n8k16, one k per block of a tile pair). o = (P V' + means) * v4_scale / (row sum).

Schedule. Eight query warps of 16 rows keep Q', P, the scores and O in registers. The warps w and w + 4 form a pair
and alternate their MMA bursts [QK(kt), PV(kt - 1), means] under a turn barrier, so that each warp runs its softmax
while its partner issues MMAs. The key tiles come from the packed images of layout.py through a ring of NST = 4
shared-memory stages (K' and V' codes and scales) and two means slots (one per tile pair), filled by TMA bulk copies:
a full mbarrier per stage signals the landing, and an empty mbarrier with a release counter lets the last warp that
leaves a stage refill it, so no warp waits for the others. The tile loop is unrolled over the four stages (static
stage addresses, one phase toggle of the barriers per cycle). An unpaired last tile shares its means slot with the
pair that ends three tiles before it: its means copy is deferred until the last reader of that pair has left, and
completes the same full-barrier phase as its codes.

The last tile is masked past the sequence. The score scale selects one of two copies of the tile loop at run time:
the folded P landing, or the two-step landing for score scales below 2^-10 or not exact on the subnormal grid. The
last units % nsm units (query tile, head, batch) of the grid (nsm: the SM count of the device) run as two CTAs over
half of the keys each, and the half that finishes second merges them in key order (split_plan, epilogue_partial).
"""

import math

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32, Int64, Uint32

from ...common.expmath import bf16x2, f2u, u2f
from ...common.mem import gptr, load_invariant, load_vector, mbar_wait, sptr, store_vector
from ...common.ptx import ld_global_cg_v4
from ...common.state import State
from ..ptx.nvfp4pv import (
    atom_add_global_u32,
    atom_add_shared_u32_elect,
    barrier_turn,
    bulk_complete,
    ld_global_nc_b32_pred,
)
from ..ptx.shared import (
    ex2_approx_f32,
    ldsm_b16_x4,
    ldsm_b16_x4_trans,
    mbar_init_imm,
    mma_bf16,
    mma_nvf4,
    mul_rn_f32,
)
from .layout import IMG_NOVM, SK_OFF, SKSF_OFF, STAGE, SV_OFF, SVM_OFF, SVSF_OFF
from .softmax import softmax_phase

NWARPS = 8
NTHREADS = 32 * NWARPS
# shared memory (bytes): NST stages of a tile's K' and V' codes and scales, two means slots (one per tile pair), then
# the full mbarriers (8 B each), the empty mbarriers and the release counters (4 B each)
NST = 4
CODE_STAGE = IMG_NOVM
MEANS_OFF = NST * CODE_STAGE
MEANS_STAGE = STAGE - SVM_OFF
BAR_OFF = MEANS_OFF + 2 * MEANS_STAGE
EBAR_OFF = BAR_OFF + 8 * NST
CNT_OFF = EBAR_OFF + 8 * NST
SMEM_BYTES = BAR_OFF + 128
PART_BYTES = (
    272  # a thread's record of a split half: its 64 O values, two row sums and two anchors (f32)
)
assert SMEM_BYTES <= 99 * 1024


# ------------------------------------------------------------------------------------------------ tile ring
@cute.jit
def load_tile(st, kt, stage: cutlass.Constexpr, initial: cutlass.Constexpr):
    """Issue the TMA copies of tile kt into stage `stage` and arm its full barrier: the codes and scales and, for an
    odd tile (the means of its pair) or the unpaired last tile, the means rows into the means slot of the stage pair.
    initial: one of the first loads of the range, into an empty ring. The barrier's byte count includes a deferred
    means copy."""
    dst = st.smem + stage * CODE_STAGE
    mb = st.bar + 8 * stage
    src = st.img + Int64(kt) * STAGE
    vm = st.smem + MEANS_OFF + (stage // 2) * MEANS_STAGE
    if cutlass.const_expr(stage % 2 == 1):
        cute.arch.mbarrier_arrive_and_expect_tx(sptr(Int64, mb), IMG_NOVM + MEANS_STAGE)
        bulk_complete(dst, src, IMG_NOVM, mb)
        bulk_complete(vm, src + IMG_NOVM, MEANS_STAGE, mb)
    else:
        if kt == st.nT - 1:
            # The barrier counts the means bytes too. In a short initial range there is no older pair in the
            # means slot and the means are copied here; otherwise the slot still contains the means of odd tile
            # kt - 3, and its last releaser contributes the means bytes.
            cute.arch.mbarrier_arrive_and_expect_tx(sptr(Int64, mb), IMG_NOVM + MEANS_STAGE)
            bulk_complete(dst, src, IMG_NOVM, mb)
            if cutlass.const_expr(initial):
                bulk_complete(vm, src + IMG_NOVM, MEANS_STAGE, mb)
        else:
            cute.arch.mbarrier_arrive_and_expect_tx(sptr(Int64, mb), IMG_NOVM)
            bulk_complete(dst, src, IMG_NOVM, mb)


@cute.jit
def release_tile(st, kt, kend: cutlass.Constexpr, stage: cutlass.Constexpr, phase):
    """This warp is done with tile kt's stage `stage` (empty-barrier phase parity `phase`): every thread fences its
    reads into the async proxy and arrives on the stage's empty mbarrier; the last of the 8 warps (elected by a shared
    counter) waits for that phase, which has then completed, and refills the stage with tile kt + NST (below kend). No
    warp waits for the others here."""
    s = Int32(stage)
    ph = Int32(phase)
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.mbarrier_arrive(sptr(Int64, st.ebar + 8 * s))
    cute.arch.sync_warp()
    old = atom_add_shared_u32_elect(st.cnt + 4 * s)
    nxt = kt + NST
    if (old & 7) == 7:
        if nxt < kend:
            mbar_wait(st.ebar + 8 * s, ph)
            cute.arch.fence_proxy("async.shared", space="cta")
            load_tile(st, nxt, stage, False)
        elif cutlass.const_expr(kend == st.nT and st.nT % 2 == 1 and stage == (st.nT - 4) % NST):
            if kt == st.nT - 4:
                # The means of the tail tile j = nT - 1 take the slot of those of tile j - 3 = kt. All reads of
                # the latter finish before the tail's bytes complete tile j's full barrier.
                mbar_wait(st.ebar + 8 * s, ph)
                cute.arch.fence_proxy("async.shared", space="cta")
                tail_stage = (st.nT - 1) % NST
                vm = st.smem + MEANS_OFF + (tail_stage // 2) * MEANS_STAGE
                bulk_complete(
                    vm,
                    st.img + Int64(st.nT - 1) * STAGE + SVM_OFF,
                    MEANS_STAGE,
                    st.bar + 8 * tail_stage,
                )


# ------------------------------------------------------------------------------------------------ MMA bursts
@cute.jit
def qk_phase(st, masked: cutlass.Constexpr, pre):
    """S = Q' K'^T of the tile at pre["sbq"] (its first B fragments preloaded in pre["bq"]); masked: the last tile."""
    sb = pre["sbq"]
    # The K' scale words are loaded after the turn barrier.
    ksw = [load_vector(sptr(Int32, sb + st.ksf_lane + 512 * a, 16), 4) for a in range(3)]
    Sacc = [[Float32(0.0)] * 4 for _ in range(16)]
    for a in cutlass.range_constexpr(3):
        for jp in cutlass.range_constexpr(8):
            if cutlass.const_expr(a == 0 and jp < len(pre["bq"])):
                bfr = pre["bq"][jp]
            else:
                bfr = ldsm_b16_x4(sb + SK_OFF + a * 4096 + jp * 256 + st.ldsm_off)
            for e in cutlass.range_constexpr(2):
                j = 2 * jp + e
                sidx = 2 * (j >> 3) + ((j >> 2) & 1)
                Sacc[j] = mma_nvf4(
                    Sacc[j], st.qa[a], [bfr[2 * e], bfr[2 * e + 1]], st.qsf[a], ksw[a][sidx], j & 3
                )
    if cutlass.const_expr(masked):
        for j in cutlass.range_constexpr(16):
            s_, h_, i_ = j >> 3, (j >> 2) & 1, j & 3
            for e in cutlass.range_constexpr(2):
                kc = 64 * s_ + 32 * h_ + 2 * i_ + e  # key = kc + 8 t
                if cutlass.const_expr(kc >= st.nvalid_last):
                    Sacc[j][e] = st.ninf
                    Sacc[j][2 + e] = st.ninf
                elif cutlass.const_expr(kc + 24 >= st.nvalid_last):
                    bad = Int32(kc + 8 * st.t >= st.nvalid_last)
                    Sacc[j][e] = Float32(cutlass.select_(bad != 0, st.ninf, Sacc[j][e]))
                    Sacc[j][2 + e] = Float32(cutlass.select_(bad != 0, st.ninf, Sacc[j][2 + e]))
    return Sacc


# straight-line emitters (plain Python: unrolled at trace time)
def emit_pv_kstep(st, sb, Pold, vsw, s_, pre_b=None):
    """O += P(k-step s_) V'(k-step s_), block-scaled FP4, 16 channel tiles (the first B fragments preloaded when
    pre_b is given)."""
    O = st.O
    pa = [Pold[4 * s_ + i] for i in range(4)]
    psf = Pold[8 + s_]
    for jp in range(8):
        if pre_b is not None and jp < len(pre_b):
            bfr = pre_b[jp]
        else:
            bfr = ldsm_b16_x4(sb + SV_OFF + s_ * 4096 + jp * 256 + st.ldsm_off)
        for e in range(2):
            jc = 2 * jp + e
            acc = mma_nvf4(
                [O[4 * jc + x] for x in range(4)],
                pa,
                [bfr[2 * e], bfr[2 * e + 1]],
                psf,
                vsw[jc >> 2][s_],
                jc & 3,
            )
            for x in range(4):
                O[4 * jc + x] = acc[x]


def emit_means(st, sb, lo, hi):
    """O += block sums of the tile pair x v_mean (bf16 m16n8k16: k 0..7 the even tile's blocks in k order, k 8..15
    the odd tile's)."""
    O = st.O
    a = [lo[0], lo[1], hi[0], hi[1]]
    for jp in range(8):
        bvm = ldsm_b16_x4_trans(sb + st.vm_lane + ((((2 * jp) + st.vm_mh) ^ st.vm_pb) << 4))
        for e in range(2):
            jc = 2 * jp + e
            acc = mma_bf16([O[4 * jc + x] for x in range(4)], a, [bvm[2 * e], bvm[2 * e + 1]])
            for x in range(4):
                O[4 * jc + x] = acc[x]


NPRE = 4  # B fragments (ldmatrix.x4) of the QK and of the PV loaded before the turn


def preload(st, has_qk, has_pv, stage):
    """Shared-memory operands of the start of a burst whose QK tile is in stage `stage` (its PV tile in the stage
    before), loaded before the turn barrier, ahead of the burst's first MMAs."""
    pre = {}
    if has_qk:
        sb = st.smem + stage * CODE_STAGE
        pre["sbq"] = sb
        pre["bq"] = [ldsm_b16_x4(sb + SK_OFF + jp * 256 + st.ldsm_off) for jp in range(NPRE)]
    if has_pv:
        sbp = st.smem + ((stage - 1) % NST) * CODE_STAGE
        pre["sbp"] = sbp
        mean_slot = ((stage - 1) % NST) // 2
        pre["mbp"] = st.smem + MEANS_OFF + mean_slot * MEANS_STAGE
        pre["bv"] = [ldsm_b16_x4(sbp + SV_OFF + jp * 256 + st.ldsm_off) for jp in range(NPRE)]
    return pre


@cute.jit
def burst(
    st,
    masked: cutlass.Constexpr,
    has_qk: cutlass.Constexpr,
    has_pv: cutlass.Constexpr,
    prev_odd: cutlass.Constexpr,
    prev_unpaired: cutlass.Constexpr,
    pre,
):
    """The MMA burst of iteration kt (its operands preloaded in pre): QK(kt), then PV(kt - 1) and the means MMAs of
    the pair ending at kt - 1 (or of an unpaired last tile with zeros), back to back."""
    Sacc = None
    if cutlass.const_expr(has_qk):
        Sacc = qk_phase(st, masked, pre)
    if cutlass.const_expr(has_pv):
        sbp = pre["sbp"]
        Pold = [st.Pst[i] for i in range(12)]
        # The V' scale words are loaded after the turn barrier; the first code fragments come preloaded (pre["bv"]).
        vv = [load_vector(sptr(Int32, sbp + st.vsf_lane + 512 * u, 16), 4) for u in range(2)]
        vsw = [[vv[u >> 1][2 * (u & 1) + s] for s in range(2)] for u in range(4)]
        emit_pv_kstep(st, sbp, Pold, vsw, 0, pre["bv"])
        emit_pv_kstep(st, sbp, Pold, vsw, 1)
        if cutlass.const_expr(prev_odd):
            emit_means(st, pre["mbp"], [st.Mlo[0], st.Mlo[1]], [Pold[10], Pold[11]])
        if cutlass.const_expr(prev_unpaired):
            emit_means(st, pre["mbp"], [st.Mlo[0], st.Mlo[1]], [Int32(0), Int32(0)])
    return Sacc


@cute.jit
def iter_step(
    st,
    kt,
    masked: cutlass.Constexpr,
    cur_odd: cutlass.Constexpr,
    kend: cutlass.Constexpr,
    two_step: cutlass.Constexpr,
    stage: cutlass.Constexpr,
    phase,
):
    """Iteration kt (its tile in stage `stage`, full-barrier phase parity `phase`): wait for its tile, take the turn of
    the warp pair, the burst [QK(kt), PV(kt - 1)], hand the turn over, release tile kt - 1, then the softmax of tile kt
    (while the partner warp runs its burst)."""
    mbar_wait(st.bar + 8 * stage, Int32(phase))
    pre = preload(st, True, True, stage)
    cute.arch.barrier(barrier_id=st.turn_wait, number_of_threads=64)
    Sacc = burst(st, masked, True, True, not cur_odd, False, pre)
    cute.arch.barrier_arrive(barrier_id=st.turn_post, number_of_threads=64)
    release_tile(st, kt - 1, kend, (stage - 1) % NST, phase ^ (1 if stage == 0 else 0))
    softmax_phase(st, Sacc, masked, cur_odd, two_step)


@cute.jit
def run_range(st, k0: cutlass.Constexpr, k1: cutlass.Constexpr, two_step: cutlass.Constexpr):
    """The key tiles k0..k1 - 1 (k0 a multiple of 4; all initial full barriers begin in relative phase zero).
    Iteration kt runs the burst [QK(kt), PV(kt - 1) (+ means of the pair kt - 2, kt - 1 for an odd kt - 1)] in its
    turn, then the softmax of kt; the two warps of a pair (w, w + 4) alternate their bursts (warp w first), so that
    each warp runs its softmax while its partner issues MMAs. The last tile of the sequence (nT - 1) is masked."""
    nT = st.nT
    n = k1 - k0
    if st.tidx == 0:
        for s in cutlass.range_constexpr(NST):
            if cutlass.const_expr(s < n):
                load_tile(st, Int32(k0 + s), s, True)
    m0 = n == 1 and k1 == nT
    # iteration k0: the QK burst only (warp w takes the first turn)
    mbar_wait(st.bar, Int32(0))
    pre0 = preload(st, True, False, 0)
    barrier_turn(st.turn_wait, st.isB, "sync")
    Sacc = burst(st, m0, True, False, False, False, pre0)
    cute.arch.barrier_arrive(barrier_id=st.turn_post, number_of_threads=64)
    softmax_phase(st, Sacc, m0, False, two_step)
    # iterations k0 + 1 .. k1 - 1: unmasked ones in pairs (odd kt, even kt + 1), the masked last one apart
    m = n - 1
    u = m - (1 if (k1 == nT and m > 0) else 0)
    ncycles = u // 4
    phase = Int32(0)
    for mp in cutlass.range(ncycles, unroll=1):
        kt = k0 + 1 + 4 * mp
        next_phase = phase ^ 1
        for j in cutlass.range_constexpr(4):
            ph = next_phase if cutlass.const_expr(j == 3) else phase
            iter_step(st, kt + j, False, j % 2 == 0, k1, two_step, (1 + j) % NST, ph)
        phase = next_phase
    for j in cutlass.range_constexpr(u % 4):
        kk = k0 + 1 + 4 * ncycles + j
        iter_step(st, Int32(kk), False, kk % 2 == 1, k1, two_step, kk % NST, ((kk - k0) // NST) & 1)
    if cutlass.const_expr(m > u):
        kk = k1 - 1
        iter_step(st, Int32(kk), True, kk % 2 == 1, k1, two_step, kk % NST, ((kk - k0) // NST) & 1)
    # the last burst: PV(k1 - 1) and its means MMAs (warp w + 4 hands no turn over after it)
    pre1 = preload(st, False, True, k1 % NST)
    cute.arch.barrier(barrier_id=st.turn_wait, number_of_threads=64)
    burst(st, False, False, True, (k1 - 1) % 2 == 1, (k1 - 1) % 2 == 0, pre1)
    barrier_turn(st.turn_post, Int32(1) - st.isB, "arrive")


# ------------------------------------------------------------------------------------------------ attention kernel
@cute.kernel
def attn_kernel(
    q_ptr: cute.Pointer,
    qsf_ptr: cute.Pointer,
    score_scale_ptr: cute.Pointer,
    vs_ptr: cute.Pointer,
    img_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    ws_ptr: cute.Pointer,
    S: cutlass.Constexpr,
    H: cutlass.Constexpr,
    NFULL: cutlass.Constexpr,
    RSPLIT: cutlass.Constexpr,
    KMID: cutlass.Constexpr,
):
    """One work unit per CTA, NTHREADS threads: thread (warp, g, t) holds the query rows 16 warp + g and + 8 of the
    unit's tile, as the warp-level MMAs lay out their fragments. NFULL, RSPLIT, KMID: split_plan of the grid; ws_ptr:
    the split units' partials and arrival counters."""
    nT = (S + 127) // 128
    nvalid_last = S - (nT - 1) * 128

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    # work unit (query tile, head, batch) of this CTA; the last RSPLIT units run as two key halves (part 1, 2)
    unit = bidx
    part = Int32(0)
    if cutlass.const_expr(RSPLIT > 0):
        if bidx >= NFULL:
            j = bidx - NFULL
            unit = NFULL + (j >> 1)
            part = 1 + (j & 1)
    qt = unit % nT
    hh = (unit // nT) % H
    bb = unit // (nT * H)
    warp = tidx // 32
    lane = tidx % 32
    g = lane // 4
    t = lane % 4
    par = t & 1
    smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024).toint()
    bar = smem + BAR_OFF
    ebar = smem + EBAR_OFF
    cnt = smem + CNT_OFF
    bh = bb * H + hh
    img = img_ptr.toint() + Int64(bh * nT) * STAGE

    if tidx == 0:
        for s in cutlass.range_constexpr(NST):
            mbar_init_imm(bar + 8 * s, 1)
            mbar_init_imm(ebar + 8 * s, NTHREADS)
            cute.arch.store(sptr(Int32, cnt + 4 * s), Int32(0))
        cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    # ---------------------------------------------------------------- per-thread fragment addressing
    m4 = lane >> 3
    r8 = lane & 7
    # ldmatrix.x4 over a pair of 8-row blocks of two 16-byte chunk arrays: matrix m -> (block m >> 1, array m & 1)
    ldsm_off = (m4 & 1) * 2048 + ((m4 >> 1) * 8 + r8) * 16
    # K' scale words of n-tiles i = t: (r % 32) * 4 + r // 32 for the keys 8 (g // 2) + 2 t + g % 2 + 32 (2 s + h)
    ksf_lane = SKSF_OFF + 16 * (8 * (g >> 1) + 2 * t + (g & 1))
    # V' scale words of channel tiles jc = t + 4 u
    vsf_lane = SVSF_OFF + 16 * lane
    # v_mean rows in k order (k position r8 -> block perm[r8] = 2 (r8 % 4) + r8 // 4), chunks swizzled by the block
    # (matrices m: k half m & 1, channel tile 2 jp + (m >> 1)); chunk jc of row r stored at jc ^ (r % 8)
    vm_pb = 2 * (r8 & 3) + (r8 >> 2)
    vm_mh = m4 >> 1
    vm_lane = ((m4 & 1) * 8 + vm_pb) * 256

    # ---------------------------------------------------------------- Q' fragments and scales (registers)
    row_g = qt * 128 + warp * 16 + g
    row_g8 = row_g + 8
    qa = []
    for a in cutlass.range_constexpr(3):
        ag = q_ptr.toint() + Int64((bb * S + row_g) * H + hh) * 128 + 32 * a + 4 * t
        ag8 = q_ptr.toint() + Int64((bb * S + row_g8) * H + hh) * 128 + 32 * a + 4 * t
        vg = Int32(row_g < S)
        vg8 = Int32(row_g8 < S)
        qa.append(
            [
                ld_global_nc_b32_pred(ag, vg),
                ld_global_nc_b32_pred(ag8, vg8),
                ld_global_nc_b32_pred(ag + 16, vg),
                ld_global_nc_b32_pred(ag8 + 16, vg8),
            ]
        )
    qr = warp * 16 + g + 8 * par
    qsf = []
    for a in cutlass.range_constexpr(3):
        qsf.append(
            ld_global_nc_b32_pred(
                qsf_ptr.toint()
                + Int64((bh * nT + qt) * 3 + a) * 512
                + 4 * ((qr % 32) * 4 + qr // 32),
                Int32(1),
            )
        )

    c = load_invariant(Float32, score_scale_ptr.toint())
    vscale = load_invariant(Float32, vs_ptr.toint())
    M = mul_rn_f32(
        c, u2f(0x400)
    )  # c * 2^-139: a subnormal multiplier (exact for a power-of-two c >= 2^-10)
    ninf = Float32(-math.inf)

    # ---------------------------------------------------------------- accumulators and softmax state (registers)
    # O: 16 channel tiles x 4 accumulators; per row (g, g + 8) the anchor, the row sum and the raw threshold of an
    # anchor move; the warp's fold-safety flag; Pst: the P of the last tile (8 code words, 2 scale words, the means
    # operand of an odd tile); Mlo: the means operand of the even tile of a pair
    O = cute.make_rmem_tensor((64,), Float32)
    for i in cutlass.range_constexpr(64):
        O[i] = Float32(0.0)
    anchor = cute.make_rmem_tensor((2,), Float32)
    lsum = cute.make_rmem_tensor((2,), Float32)
    thr = cute.make_rmem_tensor((2,), Float32)
    safe = cute.make_rmem_tensor((1,), Int32)
    for r in cutlass.range_constexpr(2):
        anchor[r] = ninf
        lsum[r] = Float32(0.0)
        thr[r] = ninf
    safe[0] = Int32(0)
    Pst = cute.make_rmem_tensor((12,), Int32)
    Mlo = cute.make_rmem_tensor((2,), Int32)

    # the turn of the warp pair (w, w + 4): named barriers 1 + w and 5 + w, each warp syncs on its own and arrives on
    # the other's
    isB = warp >> 2
    pp = warp & 3
    turn_wait = 1 + pp + 4 * isB
    turn_post = 5 + pp - 4 * isB
    st = State(
        smem=smem,
        bar=bar,
        ebar=ebar,
        cnt=cnt,
        isB=isB,
        turn_wait=turn_wait,
        turn_post=turn_post,
        img=img,
        nT=nT,
        S=S,
        H=H,
        bb=bb,
        hh=hh,
        row_g=row_g,
        row_g8=row_g8,
        vscale=vscale,
        o_base=o_ptr.toint(),
        ws=ws_ptr.toint(),
        cnt_g=ws_ptr.toint() + RSPLIT * 2 * NTHREADS * PART_BYTES,
        tidx=tidx,
        t=t,
        par=par,
        ldsm_off=ldsm_off,
        ksf_lane=ksf_lane,
        vsf_lane=vsf_lane,
        vm_lane=vm_lane,
        vm_pb=vm_pb,
        vm_mh=vm_mh,
        qa=qa,
        qsf=qsf,
        nvalid_last=nvalid_last,
        ninf=ninf,
        c=c,
        M=M,
        anchor=anchor,
        O=O,
        lsum=lsum,
        Pst=Pst,
        Mlo=Mlo,
        thr=thr,
        safe=safe,
    )

    # The folded landing needs c * 2^-139 exact: detect rounded-away bits here, without reading device data on the
    # host, and take the two-step copy of the tile loop for them.
    restored_c = mul_rn_f32(mul_rn_f32(M, u2f(0x7F000000)), Float32(4096.0))
    if (c < Float32(2.0**-10)) | (restored_c != c):
        run_dispatch(st, part, unit, NFULL, RSPLIT, KMID, True)
    else:
        run_dispatch(st, part, unit, NFULL, RSPLIT, KMID, False)


@cute.jit
def run_dispatch(
    st,
    part,
    unit,
    NFULL: cutlass.Constexpr,
    RSPLIT: cutlass.Constexpr,
    KMID: cutlass.Constexpr,
    two_step: cutlass.Constexpr,
):
    """The key range of this CTA and its epilogue: all tiles of a full unit (part 0), or one half of a split unit
    (part 1: the tiles below KMID, part 2: the others)."""
    nT = st.nT
    if cutlass.const_expr(RSPLIT == 0):
        run_range(st, 0, nT, two_step)
        epilogue_final(st)
    else:
        if part == 0:
            run_range(st, 0, nT, two_step)
            epilogue_final(st)
        else:
            if part == 1:
                run_range(st, 0, KMID, two_step)
            else:
                run_range(st, KMID, nT, two_step)
            epilogue_partial(st, unit - NFULL, part)


# ------------------------------------------------------------------------------------------------ epilogues
@cute.jit
def epilogue_final(st):
    """o = O * v4_scale / l, l the row sum over the four lanes of the row."""
    O = st.O
    inv = []
    for rr in cutlass.range_constexpr(2):
        l = st.lsum[rr]
        l = l + cute.arch.shuffle_sync_bfly(l, offset=1)
        l = l + cute.arch.shuffle_sync_bfly(l, offset=2)
        inv.append(st.vscale * cute.arch.rcp_approx(l))
    rows = [st.row_g, st.row_g8]
    for rr in cutlass.range_constexpr(2):
        if rows[rr] < st.S:
            ob = st.o_base + Int64((st.bb * st.S + rows[rr]) * st.H + st.hh) * 256 + 4 * st.t
            for jc in cutlass.range_constexpr(16):
                cute.arch.store(
                    gptr(Uint32, ob + 16 * jc),
                    bf16x2(O[4 * jc + 2 * rr] * inv[rr], O[4 * jc + 2 * rr + 1] * inv[rr]),
                )


@cute.jit
def epilogue_partial(st, slot, part):
    """One key half of a split unit: write O, l and the anchors (per thread) to its record; the half that finishes
    second (a device-scope counter per unit) merges both halves, in key order, to the new common anchor and stores o."""
    O = st.O
    anchor = st.anchor
    ls = []
    for rr in cutlass.range_constexpr(2):
        l = st.lsum[rr]
        l = l + cute.arch.shuffle_sync_bfly(l, offset=1)
        l = l + cute.arch.shuffle_sync_bfly(l, offset=2)
        ls.append(l)
    mine = st.ws + Int64((slot * 2 + (part - 1)) * NTHREADS + st.tidx) * PART_BYTES
    for i in cutlass.range_constexpr(16):
        store_vector(gptr(Uint32, mine + 16 * i, 16), [f2u(O[4 * i + x]) for x in range(4)])
    store_vector(
        gptr(Uint32, mine + 256, 16),
        [f2u(ls[0]), f2u(ls[1]), f2u(anchor[0]), f2u(anchor[1])],
    )
    cute.arch.fence_acq_rel_gpu()
    cute.arch.sync_threads()
    if st.tidx == 0:
        old = atom_add_global_u32(st.cnt_g + 4 * slot, Int32(1))
        cute.arch.store(sptr(Int32, st.cnt + 4 * NST), old)
    cute.arch.sync_threads()
    old2 = cute.arch.load(sptr(Int32, st.cnt + 4 * NST), Int32)
    if old2 == 1:
        cute.arch.fence_acq_rel_gpu()
        other = st.ws + Int64((slot * 2 + (2 - part)) * NTHREADS + st.tidx) * PART_BYTES
        w = [ld_global_cg_v4(other + 16 * i, Int32) for i in range(17)]
        # the halves combine in key order (part 1, then part 2) whichever of them merges: the products contract into
        # FMAs, which round differently with the halves swapped, so an order that followed which half finishes second
        # would make the output depend on it
        is_part1 = Int32(2) - part
        f1 = []
        f2 = []
        inv = []
        for rr in cutlass.range_constexpr(2):
            ao = u2f(w[16][2 + rr])
            lo = u2f(w[16][rr])
            a1 = Float32(cutlass.select_(is_part1 != 0, anchor[rr], ao))
            a2 = Float32(cutlass.select_(is_part1 != 0, ao, anchor[rr]))
            am = cute.arch.fmax(a1, a2)
            e1 = ex2_approx_f32(a1 - am)
            e2 = ex2_approx_f32(a2 - am)
            f1.append(e1)
            f2.append(e2)
            inv.append(
                st.vscale
                * cute.arch.rcp_approx(
                    Float32(cutlass.select_(is_part1 != 0, ls[rr], lo)) * e1
                    + Float32(cutlass.select_(is_part1 != 0, lo, ls[rr])) * e2
                )
            )
        rows = [st.row_g, st.row_g8]
        for rr in cutlass.range_constexpr(2):
            if rows[rr] < st.S:
                ob = st.o_base + Int64((st.bb * st.S + rows[rr]) * st.H + st.hh) * 256 + 4 * st.t
                for jc in cutlass.range_constexpr(16):
                    p0 = O[4 * jc + 2 * rr]
                    p1 = O[4 * jc + 2 * rr + 1]
                    o0 = u2f(w[jc][2 * rr])
                    o1 = u2f(w[jc][2 * rr + 1])
                    x0 = (
                        Float32(cutlass.select_(is_part1 != 0, p0, o0)) * f1[rr]
                        + Float32(cutlass.select_(is_part1 != 0, o0, p0)) * f2[rr]
                    ) * inv[rr]
                    x1 = (
                        Float32(cutlass.select_(is_part1 != 0, p1, o1)) * f1[rr]
                        + Float32(cutlass.select_(is_part1 != 0, o1, p1)) * f2[rr]
                    ) * inv[rr]
                    cute.arch.store(gptr(Uint32, ob + 16 * jc), bf16x2(x0, x1))
        if st.tidx == 0:
            cute.arch.store(gptr(Int32, st.cnt_g + 4 * slot), Int32(0))
