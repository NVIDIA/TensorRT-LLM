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
"""Compensation basis of the DeCo preparation: the clear launch and the eigensolver launch.

* The clear launch (one CTA) zeroes the per-call words of gmax, in which the later launches accumulate their maxima.
* The eigensolver launch (288-thread CTAs) accumulates the Gram matrices Gq0 and Gk0 of the sampled q rows and of the
  sampled k rows (the tokens of the two row arguments, in 256-row chunks, summed in parts of GRAM_PART_ROWS rows) with
  tcgen05 MMAs and normalizes each by its trace.
* A CTA writes the eigenvectors V of the DIRECTIONS largest eigenvalues (32, 64 or 128: eig_directions) of
  2 w Gq0 / tr(Gq0) + 2 (1 - w) Gk0 / tr(Gk0) of the unrotated tokens as U^T, the leading direction last.
    - Shared basis: one CTA per (batch, head), w the Gram weight; the compensation terms take its leading directions.
    - Separate bases: two CTAs per (batch, head): w = 0 (Gk alone) for the Q-residual term, then w = 1 (Gq alone) for
      the K-residual term, U^T [2, B, H, DIRECTIONS, 128].
* U = Hd^T V / sqrt(128): G = Hd^T G0 Hd / 128 has the eigenvectors Hd^T V / sqrt(128) when G0 has V (Hd Hd^T = 128 I).
* The eigensolver CTA (_eig_body) has eight row warps and a pivot warp: the Householder tridiagonalization of the whole
  matrix, then, for the DIRECTIONS largest eigenvalues with 256 / DIRECTIONS threads each, the Sturm multisection,
  the eigenvectors by twisted factorization, the cluster repair of nearly repeated eigenvalues, the
  orthonormalization of the vectors in eigenvalue order and the back-transform.
* The rows of U^T are orthonormal, and the last r of them span the space of the r largest eigenvalues for every r;
  inside a run of repeated eigenvalues they are some orthonormal basis of its space.

Per-call words of gmax (float32, cleared by the clear launch):
    0..2, 3..5  maxima of the three scale groups of Q' and of K' (head channels, Q-residual term, K-residual term)
    6           maximum of |V - mean| of the V scan
    7           the units of the V scan that its CTAs have taken (a count)
"""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64, Uint32

from ...common.dsl import Tcgen05Fence, Tcgen05Wait, tcgen05_fence, tcgen05_wait
from ...common.expmath import RN, f2u, hash01, rcp_or_zero, u2f, warp_sum
from ...common.mem import (
    cp_async16,
    gptr,
    load_invariant,
    load_vector,
    mbar_wait,
    sptr,
    store_vector,
)
from ...common.state import State
from ...constants import INV_SQRT_HEAD_DIM, MIN_GRAM_TRACE
from ..ptx import idesc_bf16, smem_desc_mn_sw128
from ..tmem import ld_tmem_f32_x32_wait, st_tmem_f32, tc_commit, tc_mma
from ..valid_len import valid_len

EIG_THREADS = 288  # eigensolver CTA: eight row warps and the pivot warp
# the directions an eigensolver CTA is built for: 256 / DIRECTIONS threads per eigenvalue
EIG_DIRECTIONS = (32, 64, 128)
# Per-call words of gmax (module docstring): count; first Q' and K' maximum, V maximum, the V scan's unit count
GMAX_WORDS = 8
GMAX_Q, GMAX_K, GMAX_V, GMAX_UNIT = 0, 3, 6, 7


@functools.cache
def eig_directions(rank):
    """Return the directions of the eigensolver build of a rank (r_q, r_k): the smallest of EIG_DIRECTIONS that holds
    the larger compensation term; 0 without compensation channels."""
    return next((d for d in EIG_DIRECTIONS if d >= max(rank)), 0) if any(rank) else 0


# ---------------------------------------------------------------- clear launch


@cute.kernel
def clear_kernel(gmax: cute.Tensor):
    """Zero the per-call words of gmax, by one warp. The launch is the programmatic primary of the eigensolver launch,
    which it releases at once: that launch waits for this grid."""
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < GMAX_WORDS:
        cute.arch.store(gptr(Uint32, Int64(gmax.iterator.toint()) + Int64(tidx * 4)), Uint32(0))
    cute.arch.griddepcontrol_launch_dependents()


# ---------------------------------------------------------------- Gram matrices

# Gram: the rows of a part; the MMAs sum the chunks of a part, and a part joins the sum of the parts when it ends
GRAM_PART_ROWS = 1536


@cute.jit
def _sampled_token(rowsb, n_rows, sample, batch, S, lens):
    """Return the token that sample `sample` of a row argument (n_rows rows at the global address rowsb) stands for
    in batch element `batch`; lens: as in _gram."""
    sample = cutlass.min(sample, n_rows - 1)
    if cutlass.const_expr(lens is None):
        r = Int32(load_invariant(Uint32, rowsb + Int64(sample) * Int64(4)))
        tok = cutlass.max(cutlass.min(r, S - 1), Int32(0))
    else:
        # with lengths: the row of the batch element, mapped into its valid prefix or kept below its end
        if cutlass.const_expr(lens[1]):
            sample = batch * n_rows + sample
        r = Int32(load_invariant(Uint32, rowsb + Int64(sample) * Int64(4)))
        n = valid_len(lens[0], batch, S)
        if cutlass.const_expr(lens[2]):
            tok = Int32((Int64(r) * Int64(n)) // Int64(S))
        else:
            tok = cutlass.max(cutlass.min(r, n - 1), Int32(0))
    return Uint32(tok)


@cute.jit
def _close_part(st, tm: Int32, first: Int32):
    """Add the part of the two Gram matrices that the MMAs summed (TMEM columns 0 .. 255 at tm) to the sum of the
    parts (columns 256 .. 511), which the first part starts (first != 0); by the four warps that read TMEM rows."""
    if st.tidx < 128:
        tl = (st.warp * 32) << 16
        for c32 in cutlass.range_constexpr(8):
            part = ld_tmem_f32_x32_wait(tm + tl + 32 * c32)
            total = ld_tmem_f32_x32_wait(tm + tl + 256 + 32 * c32)
            total = [
                Float32(cutlass.select_(first != 0, part[e], total[e] + part[e])) for e in range(32)
            ]
            st_tmem_f32(tm + tl + 256 + 32 * c32, total)
        tcgen05_wait(Tcgen05Wait.STORE)


@cute.jit
def _gram(
    st,
    qtens: cute.Tensor,
    ktens: cute.Tensor,
    rows_q: cute.Tensor,
    rows_k,
    bh: Int32,
    S: Int32,
    H: Int32,
    QP: Int32,
    KP: Int32,
    NQ: Int32,
    NK: Int32,
    gram_weight: Float32,
    lens,
):
    """Stage in VH the weighted sum 2 w Gq / tr(Gq) + 2 (1 - w) Gk / tr(Gk) (w = gram_weight) of the
    trace-normalized Gram matrices of the sampled q rows and of the sampled k rows of bh, 256 rows per chunk.

    * The MMAs sum parts of GRAM_PART_ROWS rows; with more rows than one part, a second pair of TMEM matrices takes
      the sum of the parts (_close_part).

    * QP, KP: the bytes between consecutive tokens of q and of k (256 H for a contiguous tensor).
    * rows_q (NQ rows) gives the tokens of Gq; rows_k (the global address of NK rows) those of Gk, or None: the
      tokens of rows_q.
    * All 288 threads participate (eight warps stage, four read TMEM rows); the step's last barrier publishes VH.
    * lens: None, or (the global address of the valid prefix lengths, rows per sample, mapped rows). With rows per
      sample a row argument is [B, rows] and the batch element uses its own row; a row r of a batch element of n valid
      tokens stands for token floor(r n / S) with mapped rows and for token min(r, n - 1) otherwise (token 0 when n
      is 0).
    """
    sRed = st.sRed
    if st.tidx == 0:
        cute.arch.mbarrier_init(sptr(Int64, st.mbar), 1)
        cute.arch.mbarrier_init_fence()
    if st.warp == 0:
        cute.arch.alloc_tmem(512, sptr(Uint32, st.hold))
        cute.arch.relinquish_tmem_alloc_permit()
    tcgen05_fence(Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.sync_threads()
    tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
    tm = Int32(cute.arch.load(sptr(Uint32, st.hold), Uint32))
    idesc = Int32(idesc_bf16(128, 128) | (1 << 15) | (1 << 16))
    batch = bh // H
    head = bh - batch * H
    qb = Int64(qtens.iterator.toint())
    kb = Int64(ktens.iterator.toint())
    # the token indices of a chunk: those of q in rb, those of k (with rows of its own) in tw
    tok_k = st.rbb
    if cutlass.const_expr(rows_k is not None):
        tok_k = st.twb
    NCQ = (cutlass.max(NQ, NK) + 255) >> 8
    part_chunks = GRAM_PART_ROWS // 256
    for part in cutlass.range((NCQ + part_chunks - 1) // part_chunks, unroll=1):
        ch0 = part * part_chunks
        # a part ends: its sum joins the sum of the parts before the MMAs start the next one
        if part > 0:
            mbar_wait(st.mbar, (ch0 - 1) & 1)
            tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
            _close_part(st, tm, Int32(part == 1))
        for ch in cutlass.range(ch0, cutlass.min(ch0 + part_chunks, NCQ), unroll=1):
            if ch > ch0:
                mbar_wait(st.mbar, (ch - 1) & 1)
            if st.tidx < 256:
                sample = ch * 256 + st.tidx
                cute.arch.store(
                    sptr(Uint32, st.rbb + st.tidx * 4),
                    _sampled_token(Int64(rows_q.iterator.toint()), NQ, sample, batch, S, lens),
                )
                if cutlass.const_expr(rows_k is not None):
                    cute.arch.store(
                        sptr(Uint32, tok_k + st.tidx * 4),
                        _sampled_token(rows_k, NK, sample, batch, S, lens),
                    )
            cute.arch.sync_threads()
            if st.tidx < 256:
                # a batch element without valid tokens copies zeros: its rows stand for no token
                alive = Int32(1)
                if cutlass.const_expr(lens is not None):
                    alive = Int32(valid_len(lens[0], batch, S) > 0)
                for j in cutlass.range_constexpr(16):
                    it = st.tidx + 256 * j
                    tt = it >> 4
                    cc = it & 15
                    tok = Int32(cute.arch.load(sptr(Uint32, st.rbb + tt * 4), Uint32))
                    nb = Int32(16) * Int32((ch * 256 + tt) < NQ) * alive
                    row = Int64(batch) * Int64(S) + Int64(tok)
                    off = Int64(head * 256 + cc * 16)
                    dst = (
                        (tt >> 3) * 2048
                        + (cc >> 3) * 1024
                        + (tt & 7) * 128
                        + (((cc & 7) ^ (tt & 7)) << 4)
                    )
                    cp_async16(st.dmb + dst, qb + row * Int64(QP) + off, nb)
                    if cutlass.const_expr(rows_k is not None):
                        tok = Int32(cute.arch.load(sptr(Uint32, tok_k + tt * 4), Uint32))
                        nb = Int32(16) * Int32((ch * 256 + tt) < NK) * alive
                        row = Int64(batch) * Int64(S) + Int64(tok)
                    cp_async16(st.zbb + dst, kb + row * Int64(KP) + off, nb)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tcgen05_fence(Tcgen05Fence.BEFORE_THREAD_SYNC)
            cute.arch.sync_threads()
            tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
            if st.warp == 0:
                acc0 = Int32(ch > ch0)
                for j in cutlass.range_constexpr(16):
                    da = smem_desc_mn_sw128(st.dmb + j * 4096)
                    dl = smem_desc_mn_sw128(st.zbb + j * 4096)
                    acc = acc0 if j == 0 else Int32(1)
                    tc_mma(tm, da, da, idesc, acc, kind="f16", a_tmem=False)
                    tc_mma(tm + 128, dl, dl, idesc, acc, kind="f16", a_tmem=False)
                tc_commit(st.mbar)
    mbar_wait(st.mbar, (NCQ - 1) & 1)
    tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
    # the matrices: the only part, or the sum of the parts once the last part has joined it
    gm = tm
    if NCQ > part_chunks:
        _close_part(st, tm, Int32(0))
        gm = tm + 256
    # Only the first four warps use trace scratch.
    tl = (st.warp * 32) << 16
    if st.tidx < 128:
        dga = Float32(0.0)
        dgb = Float32(0.0)
        ga_diag = ld_tmem_f32_x32_wait(gm + tl + 32 * st.warp)
        for e in cutlass.range_constexpr(32):
            dga = Float32(cutlass.select_(st.lane == e, ga_diag[e], dga))
        gk_diag = ld_tmem_f32_x32_wait(gm + tl + 128 + 32 * st.warp)
        for e in cutlass.range_constexpr(32):
            dgb = Float32(cutlass.select_(st.lane == e, gk_diag[e], dgb))
        sa = warp_sum(dga)
        sb = warp_sum(dgb)
        if st.lane == 0:
            sRed[st.warp] = sa
            sRed[8 + st.warp] = sb
    cute.arch.sync_threads()
    tra = (sRed[0] + sRed[1]) + (sRed[2] + sRed[3])
    trb = (sRed[8] + sRed[9]) + (sRed[10] + sRed[11])
    # a trace below MIN_GRAM_TRACE: its operand counts as zero, like zero rows
    tra = Float32(cutlass.select_(tra < Float32(MIN_GRAM_TRACE), Float32(0.0), tra))
    trb = Float32(cutlass.select_(trb < Float32(MIN_GRAM_TRACE), Float32(0.0), trb))
    ia = rcp_or_zero(tra) * (Float32(2.0) * gram_weight)
    ib = rcp_or_zero(trb) * (Float32(2.0) - Float32(2.0) * gram_weight)
    # VH is unused until the first Householder pivot. The normalized matrix is staged there with a 16-byte column XOR
    # swizzle, which the paired-row loads undo before VH takes its reflector layout.
    if st.tidx < 128:
        for c32 in cutlass.range_constexpr(4):
            ga = ld_tmem_f32_x32_wait(gm + tl + 32 * c32)
            gk = ld_tmem_f32_x32_wait(gm + tl + 128 + 32 * c32)
            for c4 in cutlass.range_constexpr(8):
                dst = st.vhb + (st.tidx * 32 + ((c32 * 8 + c4) ^ (st.tidx & 7))) * 16
                words = tuple(f2u(ga[c4 * 4 + e] * ia + gk[c4 * 4 + e] * ib) for e in range(4))
                store_vector(sptr(Uint32, dst, 16), words)
    tcgen05_fence(Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.sync_threads()
    if st.warp == 0:
        cute.arch.dealloc_tmem(
            cute.make_ptr(Float32, tm, cute.AddressSpace.tmem, assumed_align=16), 512
        )


# ---------------------------------------------------------------- eigensolver

_ZS = 130  # row stride (floats) of the padded L / z buffer (zbb); the D- buffer has rows of 128


def _eig_state(tidx, DIRECTIONS):
    """Return the state that the steps of the eigensolver CTA share: the coordinates of the thread, the trace-time
    constants of the build and the addresses of the shared-memory arrays."""
    # threads per eigenvalue and per eigenvector of the 256 that search and back-transform them, and the row groups
    # of an eigenvector in the back-transform
    tpe = 256 // DIRECTIONS
    lane = tidx % 32
    warp = tidx // 32

    smem = utils.SmemAllocator()
    # vh: Householder vectors [k][128]; before the first reflector, the matrix that _gram stages
    vh_p = smem.allocate_array(Float32, 128 * 128, byte_alignment=16)
    # zb: L, then z [i][slot] (padded rows); dm: D- [i][slot], later Hd [j][128] as FP32. The two hold the aligned
    # operand images of the Gram MMAs first
    zb_p = smem.allocate_array(Float32, 128 * _ZS, byte_alignment=1024)
    dm_p = smem.allocate_array(Float32, 128 * 128, byte_alignment=1024)
    # pb: p = tau A v of a Householder step [128]; rb: the two rows published for the pivot warp [2][128], and the
    # token indices of a chunk in _gram; part: the warp partials of v . p
    pb_p = smem.allocate_array(Float32, 128, byte_alignment=16)
    rb_p = smem.allocate_array(Float32, 256, byte_alignment=16)
    part_p = smem.allocate_array(Float32, 16, byte_alignment=16)
    # the tridiagonal matrix and its eigenvalues: tau [k], d [i], e [i], e^2 [i] (in the cluster repair, the flags
    # of the replaced slots), lam [slot] and -max(e_{i-1}^2, 1e-15) [i]
    tau_p = smem.allocate_array(Float32, 128, byte_alignment=16)
    dd_p = smem.allocate_array(Float32, 128, byte_alignment=16)
    ee_p = smem.allocate_array(Float32, 128, byte_alignment=16)
    e2_p = smem.allocate_array(Float32, 128, byte_alignment=16)
    lam_p = smem.allocate_array(Float32, 128, byte_alignment=16)
    e2n_p = smem.allocate_array(Float32, 128, byte_alignment=16)
    # cnt: the Sturm counts of the grid [512]; in the cluster repair, the chain starts (0..), ends (256..) and their
    # number (508); in the back-transform, the reflector pair products [j] as FP32
    cnt_p = smem.allocate_array(Int32, 512, byte_alignment=16)
    # red: warp reductions: the trace partials of Gq (0..3) and Gk (8..11); in the cluster repair, per warp the last
    # chain with a replaced member (48..)
    red_p = smem.allocate_array(Float32, 64, byte_alignment=16)
    # tw: in _gram, the token indices of a chunk of k rows; the norm partials of a reflector [lane]; in the twisted
    # factorization, [side][slot] each, the smallest gamma (0..), its row as Uint32 (256..) and the norm half (512..)
    tw_p = smem.allocate_array(Float32, 768, byte_alignment=16)
    # the mbarrier of the Gram MMAs and the word that receives their TMEM address
    gram_mb_p = smem.allocate_array(Int64, 2, byte_alignment=8)
    gram_hold_p = smem.allocate_array(Int32, 4, byte_alignment=16)
    return State(
        tidx=tidx,
        lane=lane,
        warp=warp,
        DIRECTIONS=DIRECTIONS,
        tpe=tpe,
        tpe_log2=tpe.bit_length() - 1,
        ng=32 // tpe,
        vhb=Int32(vh_p.toint()),
        zbb=Int32(zb_p.toint()),
        dmb=Int32(dm_p.toint()),
        pbb=Int32(pb_p.toint()),
        rbb=Int32(rb_p.toint()),
        partb=Int32(part_p.toint()),
        ddb=Int32(dd_p.toint()),
        eeb=Int32(ee_p.toint()),
        e2b=Int32(e2_p.toint()),
        e2nb=Int32(e2n_p.toint()),
        cntb=Int32(cnt_p.toint()),
        twb=Int32(tw_p.toint()),
        mbar=Int32(gram_mb_p.toint()),
        hold=Int32(gram_hold_p.toint()),
        # typed views for the scalar loads and stores; the 8- and 16-byte vector ones take the addresses above
        sTau=cute.make_tensor(tau_p, cute.make_layout(128)),
        sD=cute.make_tensor(dd_p, cute.make_layout(128)),
        sE=cute.make_tensor(ee_p, cute.make_layout(128)),
        sE2=cute.make_tensor(e2_p, cute.make_layout(128)),
        sE2n=cute.make_tensor(e2n_p, cute.make_layout(128)),
        sLam=cute.make_tensor(lam_p, cute.make_layout(128)),
        sRed=cute.make_tensor(red_p, cute.make_layout(64)),
        sZ=cute.make_tensor(zb_p, cute.make_layout((128, _ZS), stride=(_ZS, 1))),
        sDm=cute.make_tensor(dm_p, cute.make_layout((128, 128), stride=(128, 1))),
        sCoef=cute.make_tensor(dm_p, cute.make_layout(256)),
        sGamma=cute.make_tensor(tw_p, cute.make_layout((2, 128), stride=(128, 1))),
        sTwistRow=cute.make_tensor(
            cute.recast_ptr(tw_p + 256, dtype=Uint32), cute.make_layout((2, 128), stride=(128, 1))
        ),
        sNorm2=cute.make_tensor(tw_p + 512, cute.make_layout((2, 128), stride=(128, 1))),
    )


# ---------------------------------------------------------------- eigensolver: Householder tridiagonalization


def _sel4(e, x):
    """Return x[e] for a runtime e in 0..3."""
    return Float32(
        cutlass.select_(
            e == 0,
            x[0],
            Float32(cutlass.select_(e == 1, x[1], Float32(cutlass.select_(e == 2, x[2], x[3])))),
        )
    )


def _pivot(st, kr, x):
    """Pivot warp: reflector kr from row kr (x = elements 4 lane .. 4 lane + 3 of row kr of A^(kr)): v (support > kr)
    goes to VH[kr]; returns tau, d_kr = x[kr] and e_kr (alpha)."""
    j0 = st.lane * 4
    sg = Float32(0.0)
    for e in range(4):
        masked = Float32(cutlass.select_((j0 + e) > kr, x[e], Float32(0.0)))
        sg = sg + masked * masked
    sigma = warp_sum(sg)
    kr1 = kr + 1
    x0 = cute.arch.shuffle_sync(_sel4(kr1 & 3, x), kr1 >> 2)
    dval = cute.arch.shuffle_sync(_sel4(kr & 3, x), kr >> 2)
    rest = sigma - x0 * x0
    nrm = cute.math.sqrt(sigma, approx=True)
    alpha = Float32(cutlass.select_(x0 >= Float32(0.0), -nrm, nrm))
    do_it = Int32(rest > sigma * Float32(1e-30))
    tau = Float32(
        cutlass.select_(do_it != 0, cute.arch.rcp_approx(sigma - x0 * alpha), Float32(0.0))
    )
    alpha = Float32(cutlass.select_(do_it != 0, alpha, x0))
    vv = [
        Float32(
            cutlass.select_((j0 + e) > kr, x[e] - alpha * Float32((j0 + e) == kr1), Float32(0.0))
        )
        for e in range(4)
    ]
    store_vector(
        sptr(Uint32, st.vhb + kr * 512 + st.lane * 16, 16),
        (f2u(vv[0]), f2u(vv[1]), f2u(vv[2]), f2u(vv[3])),
    )
    return tau, dval, alpha


@cute.jit
def _load_row_pairs(st):
    """Row warps: load the thread's row pair of the matrix that _gram staged in VH, undoing the staging swizzle.

    * Thread (rp, q), rp = 8 warp + lane / 4, takes the 4-column blocks 4 m + q (columns 16 m + 4 q + e) of the rows
      r0 = 2 rp and r1 = 2 rp + 1; the threads of rp = 0 publish rows 0 and 1 in rb for the pivot warp.
    * The update rereads the reflector from shared memory instead of holding its 32 values across the step barrier.
      VH takes its reflector layout only after the barrier that follows, when every thread has loaded its tiles.
    """
    a0 = cute.make_rmem_tensor((32,), Float32)
    a1 = cute.make_rmem_tensor((32,), Float32)
    a0.fill(0.0)
    a1.fill(0.0)
    rp = st.tidx >> 2
    q = st.lane & 3
    r0 = rp * 2
    qoff = q * 16
    if st.tidx < 256:
        for m in cutlass.range_constexpr(8):
            off0 = (r0 * 32 + ((4 * m + q) ^ (r0 & 7))) * 16
            off1 = ((r0 + 1) * 32 + ((4 * m + q) ^ ((r0 + 1) & 7))) * 16
            g0 = load_vector(sptr(Uint32, st.vhb + off0, 16), 4)
            g1 = load_vector(sptr(Uint32, st.vhb + off1, 16), 4)
            for e in cutlass.range_constexpr(4):
                a0[4 * m + e] = u2f(g0[e])
                a1[4 * m + e] = u2f(g1[e])
        if rp == 0:
            for m in cutlass.range_constexpr(8):
                store_vector(
                    sptr(Uint32, st.rbb + 512 + 64 * m + qoff, 16),
                    (f2u(a0[4 * m]), f2u(a0[4 * m + 1]), f2u(a0[4 * m + 2]), f2u(a0[4 * m + 3])),
                )
                store_vector(
                    sptr(Uint32, st.rbb + 64 * m + qoff, 16),
                    (f2u(a1[4 * m]), f2u(a1[4 * m + 1]), f2u(a1[4 * m + 2]), f2u(a1[4 * m + 3])),
                )
    return State(a0=a0, a1=a1, rp=rp, q=q, r0=r0, qoff=qoff)


@cute.jit
def _first_reflector(st):
    """Pivot warp: reflector 0 from row 0 of A, which the barrier before published in rb -> VH[0], tau_0, d_0, e_0."""
    sTau, sD, sE = st.sTau, st.sD, st.sE
    if st.tidx >= 256:
        x4 = load_vector(sptr(Uint32, st.rbb + 512 + st.lane * 16, 16), 4)
        tau0, d0, al0 = _pivot(st, Int32(0), [u2f(x4[0]), u2f(x4[1]), u2f(x4[2]), u2f(x4[3])])
        if st.lane == 0:
            sTau[0] = tau0
            sD[0] = d0
            sE[0] = al0


@cute.jit
def _row_pair_products(st, pair, k, ko: cutlass.Constexpr):
    """Row warps >= ko, step k (the barrier before published reflector k in VH and tau_k): thread (rp, q) sums its
    column blocks of (A v)[r0] and (A v)[r0 + 1], the four threads of the row pair combine them, p = tau A v of the two
    rows goes to pb and the warp's partial of v . p to part (zero from an idle warp). Returns the thread's p[r0],
    p[r0 + 1], v[r0], v[r0 + 1] and tau_k for the update after the barrier."""
    a0, a1 = pair.a0, pair.a1
    r0, q, qoff = pair.r0, pair.q, pair.qoff
    vk = st.vhb + k * 512
    # only the active row warps consume these within this step
    p0 = Float32(0.0)
    p1 = Float32(0.0)
    vr0 = Float32(0.0)
    vr1 = Float32(0.0)
    tk = Float32(0.0)
    if st.tidx < 256:
        if st.tidx >= 32 * ko:
            tk = st.sTau[k]
            ac = [Float32(0.0)] * 8
            for m in cutlass.range_constexpr(ko, 8):
                w4 = load_vector(sptr(Uint32, vk + 64 * m + qoff, 16), 4)
                for e2 in cutlass.range_constexpr(2):
                    vm2 = (u2f(w4[2 * e2]), u2f(w4[2 * e2 + 1]))
                    fa = cute.arch.fma_packed_f32x2(
                        (a0[4 * m + 2 * e2], a0[4 * m + 2 * e2 + 1]),
                        vm2,
                        (ac[2 * e2], ac[2 * e2 + 1]),
                    )
                    fb = cute.arch.fma_packed_f32x2(
                        (a1[4 * m + 2 * e2], a1[4 * m + 2 * e2 + 1]),
                        vm2,
                        (ac[4 + 2 * e2], ac[4 + 2 * e2 + 1]),
                    )
                    ac[2 * e2] = fa[0]
                    ac[2 * e2 + 1] = fa[1]
                    ac[4 + 2 * e2] = fb[0]
                    ac[4 + 2 * e2 + 1] = fb[1]
            s0 = (ac[0] + ac[1]) + (ac[2] + ac[3])
            s1 = (ac[4] + ac[5]) + (ac[6] + ac[7])
            for o in (1, 2):
                s0 = s0 + cute.arch.shuffle_sync_bfly(s0, offset=Int32(o))
                s1 = s1 + cute.arch.shuffle_sync_bfly(s1, offset=Int32(o))
            p0 = s0 * tk
            p1 = s1 * tk
            vr0 = cute.arch.load(sptr(Float32, vk + r0 * 4), Float32)
            vr1 = cute.arch.load(sptr(Float32, vk + r0 * 4 + 4), Float32)
            if q == 0:
                store_vector(sptr(Uint32, st.pbb + r0 * 4, 8), (f2u(p0), f2u(p1)))
            t = (p0 * vr0 + p1 * vr1) * Float32(q == 0)
            for o in (4, 8, 16):
                t = t + cute.arch.shuffle_sync_bfly(t, offset=Int32(o))
            if st.lane == 0:
                cute.arch.store(sptr(Float32, st.partb + (st.tidx >> 3)), t)
        else:
            if st.lane == 0:
                cute.arch.store(sptr(Float32, st.partb + (st.tidx >> 3)), Float32(0.0))
    return p0, p1, vr0, vr1, tk


@cute.jit
def _reflector_dot(st):
    """Every thread: v . p of the step, the sum of the eight warp partials that the barrier before published in part."""
    pr0 = load_vector(sptr(Uint32, st.partb, 16), 4)
    pr1 = load_vector(sptr(Uint32, st.partb + 16, 16), 4)
    return ((u2f(pr0[0]) + u2f(pr0[1])) + (u2f(pr0[2]) + u2f(pr0[3]))) + (
        (u2f(pr1[0]) + u2f(pr1[1])) + (u2f(pr1[2]) + u2f(pr1[3]))
    )


@cute.jit
def _row_pair_update(st, pair, prods, ksum, k, ko: cutlass.Constexpr):
    """Row warps >= ko, step k, after the barrier that publishes p (pb) and v . p (ksum): the rank-2 update
    A <- A - v p^T - c v^T with c = p - tau (v . p) v on the thread's row pair (prods: what _row_pair_products
    returned); the threads of row k + 2 publish it in rb."""
    a0, a1 = pair.a0, pair.a1
    qoff = pair.qoff
    p0, p1, vr0, vr1, tk = prods
    vk = st.vhb + k * 512
    if st.tidx >= 32 * ko:
        kc = ksum * tk * Float32(0.5)
        nv0 = -vr0
        nv1 = -vr1
        nc0 = Float32(2.0) * kc * vr0 - p0
        nc1 = Float32(2.0) * kc * vr1 - p1
        for m in cutlass.range_constexpr(ko, 8):
            p4 = load_vector(sptr(Uint32, st.pbb + 64 * m + qoff, 16), 4)
            w4 = load_vector(sptr(Uint32, vk + 64 * m + qoff, 16), 4)
            for e2 in cutlass.range_constexpr(2):
                pp = (u2f(p4[2 * e2]), u2f(p4[2 * e2 + 1]))
                vp = (u2f(w4[2 * e2]), u2f(w4[2 * e2 + 1]))
                x0 = cute.arch.fma_packed_f32x2(
                    (nv0, nv0), pp, (a0[4 * m + 2 * e2], a0[4 * m + 2 * e2 + 1])
                )
                x0 = cute.arch.fma_packed_f32x2((nc0, nc0), vp, (x0[0], x0[1]))
                x1 = cute.arch.fma_packed_f32x2(
                    (nv1, nv1), pp, (a1[4 * m + 2 * e2], a1[4 * m + 2 * e2 + 1])
                )
                x1 = cute.arch.fma_packed_f32x2((nc1, nc1), vp, (x1[0], x1[1]))
                a0[4 * m + 2 * e2] = x0[0]
                a0[4 * m + 2 * e2 + 1] = x0[1]
                a1[4 * m + 2 * e2] = x1[0]
                a1[4 * m + 2 * e2 + 1] = x1[1]
        # publish the live column blocks of row k + 2 (of A^(k+1)) for the pivot of the next step;
        # older words of blocks m < ko are masked to zero by the pivot before any use
        if pair.rp == ((k + 2) >> 1):
            rbn = st.rbb + ((k + 1) & 1) * 512
            if (k & 1) == 0:
                for m in cutlass.range_constexpr(ko, 8):
                    words = tuple(f2u(a0[4 * m + e]) for e in range(4))
                    store_vector(sptr(Uint32, rbn + 64 * m + qoff, 16), words)
            else:
                for m in cutlass.range_constexpr(ko, 8):
                    words = tuple(f2u(a1[4 * m + e]) for e in range(4))
                    store_vector(sptr(Uint32, rbn + 64 * m + qoff, 16), words)


@cute.jit
def _next_reflector(st, ksum, k):
    """Pivot warp, step k, after the barrier that publishes p (pb) and v . p (ksum): row k + 1 of A^(k+1) =
    R - v_{k+1} p - c_{k+1} v from the row R published in rb, then reflector k + 1 -> VH[k + 1], tau_{k+1}, d_{k+1}
    and e_{k+1}."""
    sTau, sD, sE = st.sTau, st.sD, st.sE
    vk = st.vhb + k * 512
    tkp = sTau[k]
    kcp = ksum * tkp * Float32(0.5)
    vk1 = cute.arch.load(sptr(Float32, vk + (k + 1) * 4), Float32)
    pk1 = cute.arch.load(sptr(Float32, st.pbb + (k + 1) * 4), Float32)
    c1 = pk1 - Float32(2.0) * kcp * vk1
    r4 = load_vector(sptr(Uint32, st.rbb + (k & 1) * 512 + st.lane * 16, 16), 4)
    p4 = load_vector(sptr(Uint32, st.pbb + st.lane * 16, 16), 4)
    v4 = load_vector(sptr(Uint32, vk + st.lane * 16, 16), 4)
    xs = [u2f(r4[e]) - vk1 * u2f(p4[e]) - c1 * u2f(v4[e]) for e in range(4)]
    # Publish disjoint four-element segments and their norms; lane 0 finishes the reflector without a shuffle chain.
    kr = k + 1
    j0 = st.lane * 4
    xm = [Float32(cutlass.select_(j0 + e > kr, xs[e], Float32(0.0))) for e in range(4)]
    store_vector(
        sptr(Uint32, st.vhb + kr * 512 + st.lane * 16, 16),
        (f2u(xm[0]), f2u(xm[1]), f2u(xm[2]), f2u(xm[3])),
    )
    cute.arch.store(
        sptr(Float32, st.twb + st.lane * 4),
        (xm[0] * xm[0] + xm[1] * xm[1]) + (xm[2] * xm[2] + xm[3] * xm[3]),
    )
    if st.lane == (kr >> 2):
        sD[kr] = _sel4(kr & 3, xs)
    cute.arch.sync_warp()
    if st.lane == 0:
        pq = [load_vector(sptr(Uint32, st.twb + 16 * c8, 16), 4) for c8 in range(8)]
        x0p = cute.arch.load(sptr(Float32, st.vhb + kr * 512 + (kr + 1) * 4), Float32)
        s4 = [
            (u2f(pq[c8][0]) + u2f(pq[c8][1])) + (u2f(pq[c8][2]) + u2f(pq[c8][3])) for c8 in range(8)
        ]
        sigma = ((s4[0] + s4[1]) + (s4[2] + s4[3])) + ((s4[4] + s4[5]) + (s4[6] + s4[7]))
        rest = sigma - x0p * x0p
        nrm = cute.math.sqrt(sigma, approx=True)
        alpha = Float32(cutlass.select_(x0p >= Float32(0.0), -nrm, nrm))
        # row 126 has one element after the diagonal: no reflector (the back-transform applies H_0 .. H_125), also
        # when the rounding of the two squares leaves a positive rest
        do_it = Int32((rest > sigma * Float32(1e-30)) & (kr < 126))
        tau1 = Float32(
            cutlass.select_(do_it != 0, cute.arch.rcp_approx(sigma - x0p * alpha), Float32(0.0))
        )
        alpha = Float32(cutlass.select_(do_it != 0, alpha, x0p))
        cute.arch.store(sptr(Float32, st.vhb + kr * 512 + (kr + 1) * 4), x0p - alpha)
        sTau[kr] = tau1
        sE[kr] = alpha


@cute.jit
def _store_last_diagonal(st, pair):
    """Thread 255, after the last step: d_127 = A[127][127] from its registers and e_127 = 0."""
    sD, sE = st.sD, st.sE
    if st.tidx == 255:
        sD[127] = pair.a1[31]
        sE[127] = Float32(0.0)


@cute.jit
def _square_offdiagonal(st):
    """Threads < 128, after the barrier that publishes the last d and e: e2[i] = e_i^2 for the twisted factorization
    and e2n[i] = -max(e_{i-1}^2, 1e-15) for the Sturm counts."""
    sE2, sE2n = st.sE2, st.sE2n
    if st.tidx < 128:
        ev = st.sE[st.tidx]
        sE2[st.tidx] = ev * ev
        evp = st.sE[cutlass.max(st.tidx - 1, Int32(0))]
        sE2n[st.tidx] = -cute.arch.fmax(evp * evp, Float32(1e-15))


# ---------------------------------------------------------------- eigensolver: eigenvalues


def _sturm_rounds(tpe):
    """Return the rounds of multisection (2 tpe + 1 sections each) that refine an interval 17^4 times or more: an
    eigenvalue is then known to about an ulp of the Gershgorin span."""
    rounds = 1
    while (2 * tpe + 1) ** rounds < 17**4:
        rounds += 1
    return rounds


def _pow2_inv_scale(a, b):
    """Return 2^(126 - e), e the larger biased exponent of the f32 a and b: it brings max(|a|, |b|) to [0.5, 1)."""
    e = cutlass.max(f2u(a) & Uint32(0x7F800000), f2u(b) & Uint32(0x7F800000))
    return u2f(Uint32(0x7E800000) - e)


@cute.jit
def _sturm2(st, xa: Float32, xb: Float32):
    """Return the eigenvalue counts below xa and below xb (packed pair), from the Sturm sequence of T.

    * A count is the number of sign changes of P_{i+1} = (d_i - x) P_i - e2_{i-1} P_{i-1}, P_0 = 1, P_{-1} = 0; a sign
      change is the sign bit of P_{i+1} * P_i (IEEE keeps the sign of a zero or flushed product).
    * Division-free: every fourth step the pair (P_i, P_{i-1}) is rescaled by a power of two so the larger lies in
      [0.5, 1); e2n holds -max(e_{i-1}^2, 1e-15), which keeps the pair away from underflow over four steps.
    """
    nxa = -xa
    nxb = -xb
    pc0 = Float32(1.0)
    pc1 = Float32(1.0)
    pp0 = Float32(0.0)
    pp1 = Float32(0.0)
    ca = Uint32(0)
    cb = Uint32(0)
    for i4 in cutlass.range(32, unroll=2):
        d4 = load_vector(sptr(Uint32, st.ddb + i4 * 16, 16), 4)
        e4 = load_vector(sptr(Uint32, st.e2nb + i4 * 16, 16), 4)
        for j in cutlass.range_constexpr(4):
            dj = u2f(d4[j])
            ej = u2f(e4[j])
            t = cute.arch.add_packed_f32x2((dj, dj), (nxa, nxb))
            mj = cute.arch.mul_packed_f32x2((ej, ej), (pp0, pp1))
            pn = cute.arch.fma_packed_f32x2(t, (pc0, pc1), mj)
            sg = cute.arch.mul_packed_f32x2(pn, (pc0, pc1))
            ca = ca + (f2u(sg[0]) >> Uint32(31))
            cb = cb + (f2u(sg[1]) >> Uint32(31))
            pp0 = pc0
            pp1 = pc1
            pc0 = pn[0]
            pc1 = pn[1]
        sa = _pow2_inv_scale(pc0, pp0)
        sb = _pow2_inv_scale(pc1, pp1)
        rc = cute.arch.mul_packed_f32x2((pc0, pc1), (sa, sb))
        rp = cute.arch.mul_packed_f32x2((pp0, pp1), (sa, sb))
        pc0 = rc[0]
        pc1 = rc[1]
        pp0 = rp[0]
        pp1 = rp[1]
    return Int32(ca), Int32(cb)


@cute.jit
def _gershgorin(st):
    """Every thread, after the barrier that publishes d and e: the Gershgorin interval of T, four rows per lane and a
    warp reduction. Returns the interval widened by 1e-6 span (glo, ghi), span = the largest |bound|, and pivmin, the
    floor of the pivots of the twisted factorization."""
    glo = Float32(1e30)
    ghi = Float32(-1e30)
    for c4 in cutlass.range_constexpr(4):
        ii = st.lane * 4 + c4
        di = st.sD[ii]
        eprev = cute.math.abs(st.sE[cutlass.max(ii - 1, Int32(0))]) * Float32(ii > 0)
        enext = cute.math.abs(st.sE[ii]) * Float32(ii < 127)
        glo = cute.arch.fmin(glo, di - eprev - enext)
        ghi = cute.arch.fmax(ghi, di + eprev + enext)
    for o in (16, 8, 4, 2, 1):
        glo = cute.arch.fmin(glo, cute.arch.shuffle_sync_bfly(glo, offset=Int32(o)))
        ghi = cute.arch.fmax(ghi, cute.arch.shuffle_sync_bfly(ghi, offset=Int32(o)))
    span = cute.arch.fmax(cute.math.abs(glo), cute.math.abs(ghi))
    glo = glo - span * Float32(1e-6) - Float32(1e-30)
    ghi = ghi + span * Float32(1e-6) + Float32(1e-30)
    pivmin = span * Float32(1e-30) + Float32(1e-37)
    return glo, ghi, span, pivmin


@cute.jit
def _search_threads(st, glo, ghi):
    """The search of a thread: tpe threads per eigenvalue (slot t / tpe, eigenvalue index m_idx = 128 - DIRECTIONS +
    slot, sub the thread within the slot) and the 513-point grid glo + stp0 i over the Gershgorin interval. Each round
    cuts an interval into 2 tpe + 1 sections, and the rounds together refine an interval of the grid 17^4 times or
    more."""
    slot = st.tidx // st.tpe
    m_idx = (128 - st.DIRECTIONS) + slot
    sub = st.tidx % Int32(st.tpe)
    stp0 = (ghi - glo) / Float32(513)
    return State(glo=glo, stp0=stp0, slot=slot, m_idx=m_idx, sub=sub)


@cute.jit
def _grid_counts(st, search):
    """Threads < 256: the Sturm counts at the inner grid points 2 t + 1 and 2 t + 2 -> cnt[2 t], cnt[2 t + 1]."""
    if st.tidx < 256:
        gx0 = search.glo + search.stp0 * Float32(st.tidx * 2 + 1)
        gx1 = search.glo + search.stp0 * Float32(st.tidx * 2 + 2)
        gc0, gc1 = _sturm2(st, gx0, gx1)
        store_vector(sptr(Uint32, st.cntb + st.tidx * 8, 8), (Uint32(gc0), Uint32(gc1)))


@cute.jit
def _refine_eigenvalues(st, search):
    """Threads < 256, after the barrier that publishes the grid counts: a binary search in cnt brackets eigenvalue
    m_idx between two grid points; in each multisection round the tpe threads of the slot count at two points each
    and combine the bounds by shuffles; thread 0 of the slot stores the midpoint in lam[slot]."""
    sLam = st.sLam
    glo, stp0 = search.glo, search.stp0
    m_idx, sub = search.m_idx, search.sub
    npt = Int32(st.tpe)
    nit = Int32(_sturm_rounds(st.tpe))
    nsec = npt * 2 + 1
    if st.tidx < 256:
        bl = Int32(-1)
        br = Int32(512)
        for _ in cutlass.range_constexpr(10):
            mid = (bl + br) >> 1
            cm = Int32(
                cute.arch.load(sptr(Uint32, st.cntb + cutlass.max(mid, Int32(0)) * 4), Uint32)
            )
            if mid >= 0:
                if cm <= m_idx:
                    bl = mid
                else:
                    br = mid
        lo = glo + stp0 * Float32(bl + 1)
        hi = glo + stp0 * Float32(br + 1)
        for _ in cutlass.range(nit):
            stp = (hi - lo) / Float32(nsec)
            x0 = lo + stp * Float32(sub * 2 + 1)
            x1 = lo + stp * Float32(sub * 2 + 2)
            c0, c1 = _sturm2(st, x0, x1)
            below = lo
            above = hi
            below = Float32(cutlass.select_(c0 <= m_idx, x0, below))
            below = Float32(cutlass.select_(c1 <= m_idx, x1, below))
            above = Float32(cutlass.select_(c1 > m_idx, x1, above))
            above = Float32(cutlass.select_(c0 > m_idx, x0, above))
            for o in (1, 2, 4):
                bo = cute.arch.shuffle_sync_bfly(below, offset=Int32(o))
                ao = cute.arch.shuffle_sync_bfly(above, offset=Int32(o))
                below = Float32(cutlass.select_(o < npt, cute.arch.fmax(below, bo), below))
                above = Float32(cutlass.select_(o < npt, cute.arch.fmin(above, ao), above))
            lo = below
            hi = above
        lam = (lo + hi) * Float32(0.5)
        if sub == 0:
            sLam[search.slot] = lam


# ---------------------------------------------------------------- eigensolver: eigenvectors of T


@cute.jit
def _twist_sides(st):
    """The twisted factorization of a thread: the two sides of an eigenvalue sit in different warps (side =
    tidx / DIRECTIONS, slot tslot, active below 2 DIRECTIONS): side 0 runs the backward (D-) sweep, side 1 the forward
    (L) sweep; the twist index and the norm are combined through shared memory."""
    nslot = Int32(st.DIRECTIONS)
    side = st.tidx // nslot
    tslot = st.tidx - side * nslot
    act = st.tidx < 2 * nslot
    return State(side=side, tslot=tslot, act=act)


@cute.jit
def _twisted_sweeps(st, twist, pivmin):
    """Active threads, after the barrier that publishes lam: the factorizations of T - lam I of the thread's slot,
    side 0 the backward one (D-_i -> dm [i][slot]), side 1 the forward one (L_i -> zb [i][slot]); a pivot below pivmin
    in magnitude becomes -pivmin."""
    sZ, sDm = st.sZ, st.sDm
    side, tslot = twist.side, twist.tslot
    lj = st.sLam[tslot]
    if twist.act:
        if side == 0:
            # Load the next four recurrence operands before current stores.
            rdm = Float32(0.0)
            nd4 = load_vector(sptr(Uint32, st.ddb + 124 * 4, 16), 4)
            ne4 = load_vector(sptr(Uint32, st.e2b + 124 * 4, 16), 4)
            for i4 in cutlass.range(31, -1, -1):
                cd4 = nd4
                ce4 = ne4
                nb = cutlass.max(i4 - 1, Int32(0)) * 16
                nd4 = load_vector(sptr(Uint32, st.ddb + nb, 16), 4)
                ne4 = load_vector(sptr(Uint32, st.e2b + nb, 16), 4)
                for q4 in cutlass.range_constexpr(3, -1, -1):
                    dm = (u2f(cd4[q4]) - lj) - u2f(ce4[q4]) * rdm
                    dm = Float32(cutlass.select_(cute.math.abs(dm) < pivmin, -pivmin, dm))
                    sDm[i4 * 4 + q4, tslot] = dm
                    rdm = cute.arch.rcp_approx(dm)
        else:
            dp = st.sD[0] - lj
            dp = Float32(cutlass.select_(cute.math.abs(dp) < pivmin, -pivmin, dp))
            rdp = cute.arch.rcp_approx(dp)
            nd4 = load_vector(sptr(Uint32, st.ddb, 16), 4)
            ne4 = load_vector(sptr(Uint32, st.eeb, 16), 4)
            for i4 in cutlass.range(0, 32):
                cd4 = nd4
                ce4 = ne4
                nb = cutlass.min(i4 + 1, Int32(31)) * 16
                nd4 = load_vector(sptr(Uint32, st.ddb + nb, 16), 4)
                ne4 = load_vector(sptr(Uint32, st.eeb + nb, 16), 4)
                for q4 in cutlass.range_constexpr(4):
                    if i4 * 4 + q4 < 127:
                        ei = u2f(ce4[q4])
                        dn = u2f(cd4[q4 + 1]) if q4 < 3 else u2f(nd4[0])
                        li = ei * rdp
                        sZ[i4 * 4 + q4, tslot] = li
                        dp = (dn - lj) - li * ei
                        dp = Float32(cutlass.select_(cute.math.abs(dp) < pivmin, -pivmin, dp))
                        rdp = cute.arch.rcp_approx(dp)


@cute.jit
def _twist_candidates(st, twist):
    """Active threads, after the barrier that publishes D- and L: the smallest gamma of a thread's half of the rows and
    its row r -> tw. gamma_0 = |D-_0|, gamma_{ii+1} = |D-_{ii+1} - L_ii e_ii|; side 0 scans r < 64, side 1 r >= 64."""
    sGamma, sTwistRow = st.sGamma, st.sTwistRow
    side, tslot = twist.side, twist.tslot
    best = Float32(0.0)
    r = Int32(0)
    if twist.act:
        if side == 0:
            best = cute.math.abs(st.sDm[0, tslot])
            for ii in cutlass.range(0, 63, unroll=4):
                gam = cute.math.abs(st.sDm[ii + 1, tslot] - st.sZ[ii, tslot] * st.sE[ii])
                better = gam < best
                best = Float32(cutlass.select_(Int32(better) != 0, gam, best))
                r = Int32(cutlass.select_(Int32(better) != 0, ii + 1, r))
        else:
            best = cute.math.abs(st.sDm[64, tslot] - st.sZ[63, tslot] * st.sE[63])
            r = Int32(64)
            for ii in cutlass.range(64, 127, unroll=4):
                gam = cute.math.abs(st.sDm[ii + 1, tslot] - st.sZ[ii, tslot] * st.sE[ii])
                better = gam < best
                best = Float32(cutlass.select_(Int32(better) != 0, gam, best))
                r = Int32(cutlass.select_(Int32(better) != 0, ii + 1, r))
        sGamma[side, tslot] = best
        sTwistRow[side, tslot] = Uint32(r)


@cute.jit
def _twisted_solve(st, twist):
    """Active threads, after the barrier that publishes the candidates of both sides: the twist row r is that of the
    smaller gamma; z_r = 1 and the thread's half of z from r (zb [i][slot]), with the half's sum of squares -> tw."""
    sZ, sNorm2 = st.sZ, st.sNorm2
    side, tslot = twist.side, twist.tslot
    nrm2 = Float32(0.0)
    if twist.act:
        b0 = st.sGamma[0, tslot]
        b1 = st.sGamma[1, tslot]
        row0 = Int32(st.sTwistRow[0, tslot])
        row1 = Int32(st.sTwistRow[1, tslot])
        r = Int32(cutlass.select_(b1 < b0, row1, row0))
        # solve: side 0 upwards from r (L), side 1 downwards (e / D-)
        zz = Float32(1.0)
        if side == 0:
            nrm2 = Float32(1.0)
            sZ[r, tslot] = zz
            nl = sZ[cutlass.max(r - 1, Int32(0)), tslot]
            for ii in cutlass.range(r - 1, -1, -1):
                cl = nl
                nl = sZ[cutlass.max(ii - 1, Int32(0)), tslot]
                zz = -cl * zz
                sZ[ii, tslot] = zz
                nrm2 = nrm2 + zz * zz
        else:
            i1 = cutlass.min(r + 1, Int32(127))
            nf = -(st.sE[i1 - 1] * cute.arch.rcp_approx(st.sDm[i1, tslot]))
            for ii in cutlass.range(r + 1, 128):
                cf = nf
                i2 = cutlass.min(ii + 1, Int32(127))
                nf = -(st.sE[i2 - 1] * cute.arch.rcp_approx(st.sDm[i2, tslot]))
                zz = cf * zz
                sZ[ii, tslot] = zz
                nrm2 = nrm2 + zz * zz
        sNorm2[side, tslot] = nrm2


@cute.jit
def _normalize_halves(st, twist):
    """Active threads, after the barrier that publishes the two sums of squares: each side scales its 64 rows of z
    to unit norm of the whole vector."""
    sZ = st.sZ
    side, tslot = twist.side, twist.tslot
    if twist.act:
        inv = Float32(1.0) / cute.math.sqrt(
            st.sNorm2[0, tslot] + st.sNorm2[1, tslot],
            rounding=RN,
        )
        for ii in cutlass.range(side * 64, side * 64 + 64, unroll=8):
            sZ[ii, tslot] = sZ[ii, tslot] * inv


# ---------------------------------------------------------------- eigensolver: eigencluster repair


@cute.jit
def _degen_pivot_floor(value: Float32, floor: Float32) -> Float32:
    """Floor |value| at floor; the pivot sign is preserved in the indefinite shifted tridiagonal solve."""
    safe = Float32(cutlass.select_(value < Float32(0.0), -floor, floor))
    return Float32(cutlass.select_(cute.math.abs(value) < floor, safe, value))


@cute.jit
def _degen_filter(st, slot: Int32, shift: Float32, pmin: Float32):
    """Run one independent shifted LDL/Thomas solve on column `slot` of z and normalize it. Each thread owns one
    column of both scratch matrices; no inter-thread synchronization is needed."""
    sZ, sDm = st.sZ, st.sDm
    dp = _degen_pivot_floor(st.sD[0] - shift, pmin)
    rhs = sZ[0, slot]
    for ii in cutlass.range(127, unroll=4):
        ei = st.sE[ii]
        rinv = cute.math.rcp(dp, approx=True)
        li = ei * rinv
        zi = rhs * rinv
        # Load the next RHS before its location is overwritten later.
        bn = sZ[ii + 1, slot]
        sDm[ii, slot] = li
        sZ[ii, slot] = zi
        dp = _degen_pivot_floor(
            cute.fma(-li, ei, st.sD[ii + 1] - shift),
            pmin,
        )
        rhs = cute.fma(-li, rhs, bn)
    xx = rhs * cute.math.rcp(dp, approx=True)
    sZ[127, slot] = xx
    amax = cute.math.abs(xx)
    for ii in cutlass.range(126, -1, -1, unroll=4):
        xx = cute.fma(
            -sDm[ii, slot],
            xx,
            sZ[ii, slot],
        )
        sZ[ii, slot] = xx
        amax = cute.arch.fmax(amax, cute.math.abs(xx))
    # Scale by the largest component before summing squares. This remains
    # finite even when a zero Gram requires a very small absolute pivot floor.
    invmax = rcp_or_zero(amax)
    n2 = Float32(0.0)
    for ii in cutlass.range(128, unroll=4):
        zz = sZ[ii, slot] * invmax
        n2 = cute.fma(zz, zz, n2)
    invnorm = invmax / cute.math.sqrt(cute.arch.fmax(n2, Float32(1e-30)), rounding=RN)
    for ii in cutlass.range(128, unroll=4):
        sZ[ii, slot] = sZ[ii, slot] * invnorm


@cute.jit
def _orthogonalize_replaced(st, REPS: cutlass.Constexpr = 2, KEPT: cutlass.Constexpr = False):
    """Orthogonalize the replaced slots (flagged in e2) by classical Gram-Schmidt, largest eigenvalue first.

    * Each slot is projected REPS times against every kept slot and every replaced slot before it, then normalized.
      KEPT takes the kept slots instead, each against the kept slots before it.
    * One thread per slot forms the coefficients and one thread per row applies them, two partial sums each; the D-
      buffer holds the coefficients and the norm partials. The step has barriers of its own: one follows each phase.
    """
    sZ, sCoef = st.sZ, st.sCoef
    nslot = Int32(st.DIRECTIONS)
    mine = Int32(0)
    pos = Int32(0)
    if st.tidx < nslot:
        mine = Int32(st.sE2[st.tidx] > Float32(0.5))
        pos = nslot - 1 - st.tidx
    zrow = st.zbb + st.tidx * (_ZS * 4)
    for jj in cutlass.range(0, nslot):
        sj = nslot - 1 - jj
        # every thread reads the same flag, so the barriers below are reached by the whole CTA
        if Int32(st.sE2[sj] > Float32(0.5)) != Int32(1 if KEPT else 0):
            for rep in cutlass.range_constexpr(REPS):
                if st.tidx < nslot:
                    d0 = Float32(0.0)
                    d1 = Float32(0.0)
                    if cutlass.const_expr(KEPT):
                        use = (mine ^ 1) & Int32(pos < jj)
                    else:
                        use = (mine ^ 1) | Int32(pos < jj)
                    if use != 0:
                        for ii in cutlass.range(0, 128, 2, unroll=4):
                            d0 = cute.fma(
                                sZ[ii, st.tidx],
                                sZ[ii, sj],
                                d0,
                            )
                            d1 = cute.fma(
                                sZ[ii + 1, st.tidx],
                                sZ[ii + 1, sj],
                                d1,
                            )
                    sCoef[st.tidx] = d0 + d1
                cute.arch.sync_threads()
                if st.tidx < 128:
                    # pairs of slots, read by 8-byte loads
                    z0 = sZ[st.tidx, sj]
                    z1 = Float32(0.0)
                    for sp in cutlass.range(0, nslot, 2, unroll=4):
                        c2 = load_vector(sptr(Uint32, st.dmb + sp * 4, 8), 2)
                        y2 = load_vector(sptr(Uint32, zrow + sp * 4, 8), 2)
                        z0 = cute.fma(-u2f(c2[0]), u2f(y2[0]), z0)
                        z1 = cute.fma(-u2f(c2[1]), u2f(y2[1]), z1)
                    zz = z0 + z1
                    sZ[st.tidx, sj] = zz
                    if cutlass.const_expr(rep == REPS - 1):
                        n2 = warp_sum(zz * zz)
                        if st.lane == 0:
                            sCoef[128 + st.warp] = n2
                cute.arch.sync_threads()
            if st.tidx < 128:
                n2 = (sCoef[128] + sCoef[129]) + (sCoef[130] + sCoef[131])
                sZ[st.tidx, sj] = sZ[st.tidx, sj] * (
                    Float32(1.0) / cute.math.sqrt(cute.arch.fmax(n2, Float32(1e-30)), rounding=RN)
                )
            cute.arch.sync_threads()


@cute.jit
def _cluster_repair(st, span: Float32):
    """Repair the eigenvectors of nearly repeated eigenvalues, after the barrier that publishes the normalized z.

    * Clusters (chains of consecutive gaps <= 2e-5 span) are found with ballots. Cluster MGS with one warp per cluster:
      each warp orthogonalizes the members of its clusters in slot order (twice) and flags degenerate members in e2.
    * A chain with a bitwise repeated eigenvalue would get identical twisted vectors, so it is replaced whole, without
      its MGS. Replaced members become deterministic seeds.
    * The kept vectors are first orthonormalized among themselves in eigenvalue order: those of close eigenvalues are
      not orthogonal enough for the last seeds, which fill what is left of the complement.
    * Four rounds follow, each a shifted solve (the inverse iteration that refines the seeds) and then the
      orthogonalization against the kept vectors. The step has barriers of its own; its last one publishes z.
    """
    sE2, sRed, sZ = st.sE2, st.sRed, st.sZ
    nslot = Int32(st.DIRECTIONS)
    tolc = span * Float32(2e-5)
    if st.tidx < 128:
        sE2[st.tidx] = Float32(0.0)
    if st.warp == 0:
        nst = Int32(0)
        nen = Int32(0)
        below = (Uint32(1) << Uint32(st.lane)) - Uint32(1)
        for e in cutlass.range_constexpr(4):
            s = st.lane + 32 * e
            le = Int32(0)
            le1 = Int32(0)
            if s >= 1 and s < nslot:
                le = Int32(cute.math.abs(st.sLam[s] - st.sLam[s - 1]) <= tolc)
            if s + 1 < nslot:
                le1 = Int32(cute.math.abs(st.sLam[s + 1] - st.sLam[s]) <= tolc)
            isst = Int32((le == 0) & (le1 != 0))
            isen = Int32((le != 0) & (le1 == 0))
            bst = Uint32(cute.arch.vote_ballot_sync(isst != 0))
            ben = Uint32(cute.arch.vote_ballot_sync(isen != 0))
            if isst != 0:
                cute.arch.store(
                    sptr(Uint32, st.cntb + (nst + Int32(cute.arch.popc(bst & below))) * 4),
                    Uint32(s),
                )
            if isen != 0:
                cute.arch.store(
                    sptr(Uint32, st.cntb + (256 + nen + Int32(cute.arch.popc(ben & below))) * 4),
                    Uint32(s),
                )
            nst = nst + Int32(cute.arch.popc(bst))
            nen = nen + Int32(cute.arch.popc(ben))
        if st.lane == 0:
            cute.arch.store(sptr(Uint32, st.cntb + 508 * 4), Uint32(nst))
    cute.arch.sync_threads()
    ncl = Int32(cute.arch.load(sptr(Uint32, st.cntb + 508 * 4), Uint32))
    # per warp, the last chain with a replaced member (-1 when none)
    cmax = Int32(-1)
    for ci in cutlass.range(st.warp, ncl, EIG_THREADS // 32):
        cstart = Int32(cute.arch.load(sptr(Uint32, st.cntb + ci * 4), Uint32))
        cend = Int32(cute.arch.load(sptr(Uint32, st.cntb + (256 + ci) * 4), Uint32))
        dup = Int32(0)
        for s in cutlass.range(cstart + 1 + st.lane, cend + 1, 32):
            dup = dup | Int32(st.sLam[s] == st.sLam[s - 1])
        dup = Int32(cute.arch.vote_ballot_sync(dup != 0) != 0)
        hit = dup
        # a chain with a repeated eigenvalue has an empty range here: no MGS
        for sj in cutlass.range(cstart + 1, cend + 1 - (cend - cstart) * dup):
            degen = Int32(0)
            for rep in cutlass.range_constexpr(2):
                for sp in cutlass.range(cstart, sj):
                    dt = Float32(0.0)
                    for e in cutlass.range_constexpr(4):
                        ii = st.lane + 32 * e
                        dt = dt + sZ[ii, sj] * sZ[ii, sp]
                    dt = warp_sum(dt)
                    for e in cutlass.range_constexpr(4):
                        ii = st.lane + 32 * e
                        sZ[ii, sj] = sZ[ii, sj] - dt * sZ[ii, sp]
                    cute.arch.sync_warp()
                n2 = Float32(0.0)
                for e in cutlass.range_constexpr(4):
                    ii = st.lane + 32 * e
                    zv = sZ[ii, sj]
                    n2 = n2 + zv * zv
                n2 = warp_sum(n2)
                if cutlass.const_expr(rep == 0):
                    degen = Int32(n2 < Float32(1e-4))
                inv_norm = Float32(1.0) / cute.math.sqrt(
                    cute.arch.fmax(n2, Float32(1e-30)), rounding=RN
                )
                for e in cutlass.range_constexpr(4):
                    ii = st.lane + 32 * e
                    sZ[ii, sj] = sZ[ii, sj] * inv_norm
                cute.arch.sync_warp()
            if st.lane == 0:
                sE2[sj] = Float32(degen)
            hit = hit | degen
        if dup != 0:
            for s in cutlass.range(cstart + st.lane, cend + 1, 32):
                sE2[s] = Float32(1.0)
        if hit != 0:
            cmax = cutlass.max(cmax, ci)
    if st.lane == 0:
        sRed[48 + st.warp] = Float32(cmax)
    cute.arch.sync_threads()
    gmax = Float32(-1.0)
    for w in cutlass.range_constexpr(EIG_THREADS // 32):
        gmax = cute.arch.fmax(gmax, sRed[48 + w])
    if gmax > Float32(-0.5):
        if st.tidx < nslot:
            if sE2[st.tidx] > Float32(0.5):
                for ii in cutlass.range(128, unroll=4):
                    sZ[ii, st.tidx] = hash01(ii, st.tidx + 1)
        # one projection: the kept vectors are far from parallel; the seed columns it reads only meet zero coefficients
        _orthogonalize_replaced(st, REPS=1, KEPT=True)
        pmin = cute.arch.fmax(span * Float32(1e-9), Float32(1e-30))
        for _ in cutlass.range(4):
            if st.tidx < nslot:
                if sE2[st.tidx] > Float32(0.5):
                    _degen_filter(
                        st,
                        st.tidx,
                        st.sLam[st.tidx],
                        pmin,
                    )
            # the filter is column-parallel; the orthogonalization reads whole columns
            cute.arch.sync_threads()
            _orthogonalize_replaced(st)


@cute.jit
def _orthonormalize(st, PASSES: cutlass.Constexpr = 1):
    """Orthonormalize the vectors in eigenvalue order, all slots at once, after the barrier that publishes z:
    z_j <- z_j - sum over the slots i above j of (z_i . z_j) z_i, then unit norm; PASSES times.

    * The twisted vectors of close eigenvalues are not orthogonal to each other. A slot and the slots above it
      span what they spanned before, so the space of every leading run of directions is kept.
    * Products: slot a forms those with the DIRECTIONS / 2 slots that follow it cyclically, shared by its
      256 / DIRECTIONS threads -> the D- buffer [lower slot][higher slot]; the slot's own product and the three
      before it are stored as zero.
    * Rows: thread i < 128 updates row i of z in ascending slot order, four slots per step from the group of
      slot j + 1 on, so a slot reads the slots above it before they change. Norms: one thread per slot (-> the
      norm halves of the twist), then one thread per row.
    * The step has barriers of its own; the last one publishes z.
    """
    sZ, sNorm2 = st.sZ, st.sNorm2
    nslot = st.DIRECTIONS
    slot = st.tidx >> st.tpe_log2
    sub = st.tidx & (st.tpe - 1)
    for _ in cutlass.range_constexpr(PASSES):
        if st.tidx < 256:
            for o in cutlass.range(1 + sub, nslot // 2 + 1, st.tpe):
                other = (slot + o) & (nslot - 1)
                d0 = Float32(0.0)
                d1 = Float32(0.0)
                for ii in cutlass.range(0, 128, 2, unroll=4):
                    pair = cute.arch.fma_packed_f32x2(
                        (sZ[ii, slot], sZ[ii + 1, slot]),
                        (sZ[ii, other], sZ[ii + 1, other]),
                        (d0, d1),
                    )
                    d0 = pair[0]
                    d1 = pair[1]
                at = cutlass.min(slot, other) * 128 + cutlass.max(slot, other)
                cute.arch.store(sptr(Float32, st.dmb + at * 4), d0 + d1)
            if sub == 0:
                for back in cutlass.range_constexpr(4):
                    own = slot * 128 + cutlass.max(slot - back, Int32(0))
                    cute.arch.store(sptr(Float32, st.dmb + own * 4), Float32(0.0))
        cute.arch.sync_threads()
        if st.tidx < 128:
            zrow = st.zbb + st.tidx * (_ZS * 4)
            for j in cutlass.range(0, nslot - 1):
                crow = st.dmb + j * 512
                z0 = Float32(sZ[st.tidx, j])
                z1 = Float32(0.0)
                z2 = Float32(0.0)
                z3 = Float32(0.0)
                for g in cutlass.range((j + 1) >> 2, nslot >> 2):
                    c4 = load_vector(sptr(Uint32, crow + g * 16, 16), 4)
                    ya = load_vector(sptr(Uint32, zrow + g * 16, 8), 2)
                    yb = load_vector(sptr(Uint32, zrow + g * 16 + 8, 8), 2)
                    pa = cute.arch.fma_packed_f32x2(
                        (-u2f(c4[0]), -u2f(c4[1])), (u2f(ya[0]), u2f(ya[1])), (z0, z1)
                    )
                    pb = cute.arch.fma_packed_f32x2(
                        (-u2f(c4[2]), -u2f(c4[3])), (u2f(yb[0]), u2f(yb[1])), (z2, z3)
                    )
                    z0 = pa[0]
                    z1 = pa[1]
                    z2 = pb[0]
                    z3 = pb[1]
                sZ[st.tidx, j] = (z0 + z1) + (z2 + z3)
        cute.arch.sync_threads()
        if st.tidx < nslot:
            n0 = Float32(0.0)
            n1 = Float32(0.0)
            for ii in cutlass.range(0, 128, 2, unroll=4):
                n0 = cute.fma(sZ[ii, st.tidx], sZ[ii, st.tidx], n0)
                n1 = cute.fma(sZ[ii + 1, st.tidx], sZ[ii + 1, st.tidx], n1)
            sNorm2[0, st.tidx] = Float32(1.0) / cute.math.sqrt(
                cute.arch.fmax(n0 + n1, Float32(1e-30)), rounding=RN
            )
        cute.arch.sync_threads()
        if st.tidx < 128:
            for j in cutlass.range(0, nslot, unroll=4):
                sZ[st.tidx, j] = sZ[st.tidx, j] * sNorm2[0, j]
        cute.arch.sync_threads()


# ---------------------------------------------------------------- eigensolver: back-transform and U


@cute.jit
def _stage_hadamard_f32(st, hadamard: cute.Tensor):
    """Every thread: Hd (BF16, 2048 16-byte chunks) as FP32 into the D- buffer, which is free after the cluster
    repair; the barrier after the reflector pair products publishes it."""
    hb = Int64(hadamard.iterator.toint())
    for it in cutlass.range_constexpr(8):
        ch = st.tidx + EIG_THREADS * it
        if ch < 2048:
            w = load_invariant(Uint32, hb + Int64(ch * 16), 4)
            store_vector(
                sptr(Uint32, st.dmb + ch * 32, 16),
                (
                    w[0] << Uint32(16),
                    w[0] & Uint32(0xFFFF0000),
                    w[1] << Uint32(16),
                    w[1] & Uint32(0xFFFF0000),
                ),
            )
            store_vector(
                sptr(Uint32, st.dmb + ch * 32 + 16, 16),
                (
                    w[2] << Uint32(16),
                    w[2] & Uint32(0xFFFF0000),
                    w[3] << Uint32(16),
                    w[3] & Uint32(0xFFFF0000),
                ),
            )


@cute.jit
def _reflector_pair_products(st):
    """One warp per reflector pair (2 j + 1, 2 j): c_j = v_{2j} . v_{2j+1} -> cnt[j] as FP32."""
    for jp in cutlass.range(st.warp, 63, EIG_THREADS // 32):
        x4 = load_vector(sptr(Uint32, st.vhb + (2 * jp) * 512 + st.lane * 16, 16), 4)
        y4 = load_vector(sptr(Uint32, st.vhb + (2 * jp + 1) * 512 + st.lane * 16, 16), 4)
        dc = (u2f(x4[0]) * u2f(y4[0]) + u2f(x4[1]) * u2f(y4[1])) + (
            u2f(x4[2]) * u2f(y4[2]) + u2f(x4[3]) * u2f(y4[3])
        )
        dc = warp_sum(dc)
        if st.lane == 0:
            cute.arch.store(sptr(Float32, st.cntb + jp * 4), dc)


@cute.jit
def _apply_reflector_pair(st, vec, jp, mo: cutlass.Constexpr):
    """A thread of the back-transform, reflector pair jp of segment mo: z <- H_{2jp} H_{2jp+1} z on its rows of the
    row groups >= mo, both reflectors in one pass over the rows (vec: the thread's registers and its place qt among
    the tpe threads of its vector)."""
    z, vs0, vs1, qt = vec.z, vec.vs0, vec.vs1, vec.qt
    tpe, ng = st.tpe, st.ng
    tq1 = st.sTau[2 * jp + 1]
    tq0 = st.sTau[2 * jp]
    cpj = cute.arch.load(sptr(Float32, st.cntb + jp * 4), Float32)
    va0 = st.vhb + (2 * jp) * 512 + qt * 16
    va1 = va0 + 512
    # d0 = v_{2jp} . z and d1 = v_{2jp+1} . z: the thread's rows, then the tpe threads of the vector
    da = [Float32(0.0)] * 8
    for m in cutlass.range_constexpr(mo, ng):
        w0 = load_vector(sptr(Uint32, va0 + 16 * tpe * m, 16), 4)
        w1 = load_vector(sptr(Uint32, va1 + 16 * tpe * m, 16), 4)
        for e in cutlass.range_constexpr(4):
            vs0[4 * m + e] = u2f(w0[e])
            vs1[4 * m + e] = u2f(w1[e])
        for e2 in cutlass.range_constexpr(2):
            zp = (z[4 * m + 2 * e2], z[4 * m + 2 * e2 + 1])
            fa = cute.arch.fma_packed_f32x2(
                (vs0[4 * m + 2 * e2], vs0[4 * m + 2 * e2 + 1]),
                zp,
                (da[2 * e2], da[2 * e2 + 1]),
            )
            fb = cute.arch.fma_packed_f32x2(
                (vs1[4 * m + 2 * e2], vs1[4 * m + 2 * e2 + 1]),
                zp,
                (da[4 + 2 * e2], da[4 + 2 * e2 + 1]),
            )
            da[2 * e2] = fa[0]
            da[2 * e2 + 1] = fa[1]
            da[4 + 2 * e2] = fb[0]
            da[4 + 2 * e2 + 1] = fb[1]
    d0 = (da[0] + da[1]) + (da[2] + da[3])
    d1 = (da[4] + da[5]) + (da[6] + da[7])
    for o in (1, 2, 4)[: st.tpe_log2]:
        d0 = d0 + cute.arch.shuffle_sync_bfly(d0, offset=Int32(o))
        d1 = d1 + cute.arch.shuffle_sync_bfly(d1, offset=Int32(o))
    # H_{2jp+1} z = z - cf1 v_{2jp+1}, and v_{2jp} . (H_{2jp+1} z) = d0 - cf1 c_jp gives the coefficient of v_{2jp}
    cf1 = d1 * tq1
    cf0 = (d0 - cf1 * cpj) * tq0
    n1 = -cf1
    n0 = -cf0
    for m in cutlass.range_constexpr(mo, ng):
        for e2 in cutlass.range_constexpr(2):
            fa = cute.arch.fma_packed_f32x2(
                (n1, n1),
                (vs1[4 * m + 2 * e2], vs1[4 * m + 2 * e2 + 1]),
                (z[4 * m + 2 * e2], z[4 * m + 2 * e2 + 1]),
            )
            fb = cute.arch.fma_packed_f32x2(
                (n0, n0), (vs0[4 * m + 2 * e2], vs0[4 * m + 2 * e2 + 1]), (fa[0], fa[1])
            )
            z[4 * m + 2 * e2] = fb[0]
            z[4 * m + 2 * e2 + 1] = fb[1]


@cute.jit
def _back_transform(st):
    """Threads < 256, after the barrier that publishes the c_j: z <- H_0 ... H_125 z in registers and back to zb. tpe
    threads per vector, each holding the rows 4 tpe m + 4 q + e of the ng row groups m; two reflectors per round, in
    segments of 2 tpe pairs: the reflectors of segment mo are zero on the row groups before mo."""
    sZ = st.sZ
    tpe, ng = st.tpe, st.ng
    if st.tidx < 256:
        cq = st.warp * (32 // tpe) + (st.lane >> st.tpe_log2)
        qt = st.lane & (tpe - 1)
        z = cute.make_rmem_tensor((4 * ng,), Float32)
        vs0 = cute.make_rmem_tensor((4 * ng,), Float32)
        vs1 = cute.make_rmem_tensor((4 * ng,), Float32)
        vs0.fill(0.0)
        vs1.fill(0.0)
        for m in cutlass.range_constexpr(ng):
            for e in cutlass.range_constexpr(4):
                z[4 * m + e] = sZ[4 * tpe * m + 4 * qt + e, cq]
        vec = State(z=z, vs0=vs0, vs1=vs1, qt=qt)
        for mo in cutlass.range_constexpr(ng - 1, -1, -1):
            for jp in cutlass.range(min(2 * tpe * mo + 2 * tpe - 1, 62), 2 * tpe * mo - 1, -1):
                _apply_reflector_pair(st, vec, jp, mo)
        for m in cutlass.range_constexpr(ng):
            for e in cutlass.range_constexpr(4):
                sZ[4 * tpe * m + 4 * qt + e, cq] = z[4 * m + e]


@cute.jit
def _write_basis(st, uT: cute.Tensor, bh: Int32):
    """Threads < 256, after the barrier that publishes the back-transformed z: write U = Hd^T Z / sqrt(128) to basis bh
    of uT (Hd is in the D- buffer); a thread takes 16 rotated channels of one basis vector out of every 32. The
    entries of Hd are +-1, so a term z Hd is exact and the sums add the terms in the order of the rows of z."""
    if st.tidx < 256:
        for part in cutlass.range_constexpr(st.DIRECTIONS // 32):
            urow = st.lane  # basis vector (U^T row)
            if cutlass.const_expr(part > 0):
                urow = urow + 32 * part
            k0 = st.warp * 16  # 16 rotated channels, the same for every lane of the warp
            accu = cute.make_rmem_tensor((16,), Float32)
            accu.fill(0.0)
            for j in cutlass.range(128, unroll=2):
                zj = st.sZ[j, urow]
                for c4 in cutlass.range_constexpr(4):
                    h4 = load_vector(sptr(Uint32, st.dmb + (j * 128 + k0 + 4 * c4) * 4, 16), 4)
                    for e in cutlass.range_constexpr(0, 4, 2):
                        pair = cute.arch.fma_packed_f32x2(
                            (zj, zj),
                            (u2f(h4[e]), u2f(h4[e + 1])),
                            (accu[4 * c4 + e], accu[4 * c4 + e + 1]),
                        )
                        accu[4 * c4 + e] = pair[0]
                        accu[4 * c4 + e + 1] = pair[1]
            s128 = Float32(INV_SQRT_HEAD_DIM)
            ub = (
                Int64(uT.iterator.toint())
                + Int64(bh) * Int64(st.DIRECTIONS * 512)
                + Int64((urow * 128 + k0) * 4)
            )
            for c4 in cutlass.range_constexpr(4):
                dst = ub + Int64(16 * c4)
                words = tuple(f2u(accu[4 * c4 + e] * s128) for e in range(4))
                store_vector(gptr(Uint32, dst, 16), words)


# ---------------------------------------------------------------- eigensolver launch: the CTA body


@cute.jit
def _eig_body(
    qtens,
    ktens,
    rows_q,
    rows_k,
    hadamard,
    uT,
    S,
    H,
    QP,
    KP,
    NQ,
    NK,
    gram_weight,
    DIRECTIONS,
    lens,
    SEPARATE,
):
    """Solve one eigenproblem per CTA: CTA c writes basis c of uT, U^T [B, H, DIRECTIONS, 128] of the weighted Gram
    sum of (batch, head) c, or with SEPARATE U^T [2, B, H, DIRECTIONS, 128]: the basis of Gk, then that of Gq (module
    docstring).

    * The launch is a programmatic dependent of the clear launch and releases its own dependent at once: the V pass
      launched beside it touches only the original V, its own outputs and its words of gmax.
    * The body is the schedule of the CTA: each step runs on the threads its docstring names; each sync_threads is the
      CTA barrier that publishes what the steps since the barrier before it wrote to shared memory (_gram and
      _cluster_repair have barriers of their own, the last of which publishes their result).
    * st is the state that every step shares; pair, search and twist hold the registers and coordinates of the thread
      in one phase, built where that phase begins. lens: None, or the lengths argument of _gram.
    """
    # the preceding launch on this stream clears the per-call words that the launches after this one read
    cute.arch.griddepcontrol_wait()
    tidx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    cute.arch.griddepcontrol_launch_dependents()
    st = _eig_state(tidx, DIRECTIONS)
    bh = cta
    if cutlass.const_expr(SEPARATE):
        gdim, _, _ = cute.arch.grid_dim()
        second = Int32(cta >= (gdim >> 1))
        bh = cta - second * (gdim >> 1)
        gram_weight = Float32(second)
    # dm and zb hold the aligned Gram operand images until the MMAs complete. The normalized matrix is then staged in
    # VH, whose first reflector is written only after all paired-row math threads have loaded their register tiles.
    _gram(st, qtens, ktens, rows_q, rows_k, bh, S, H, QP, KP, NQ, NK, gram_weight, lens)

    # Householder tridiagonalization: the reflectors in VH, tau, and the tridiagonal matrix (d, e)
    pair = _load_row_pairs(st)
    cute.arch.sync_threads()
    _first_reflector(st)
    cute.arch.sync_threads()
    # 8 static segments of 16 steps: in segment ko, column groups m < ko and warps < ko are idle (no predication).
    for ko in cutlass.range_constexpr(8):
        for k in cutlass.range(16 * ko, min(16 * ko + 16, 126)):
            prods = _row_pair_products(st, pair, k, ko)
            cute.arch.sync_threads()
            ksum = _reflector_dot(st)
            if tidx < 256:
                _row_pair_update(st, pair, prods, ksum, k, ko)
            else:
                _next_reflector(st, ksum, k)
            cute.arch.sync_threads()
    _store_last_diagonal(st, pair)
    cute.arch.sync_threads()
    _square_offdiagonal(st)
    cute.arch.sync_threads()

    # eigenvalues: Sturm multisection, 2 points/thread
    glo, ghi, span, pivmin = _gershgorin(st)
    search = _search_threads(st, glo, ghi)
    _grid_counts(st, search)
    cute.arch.sync_threads()
    _refine_eigenvalues(st, search)
    cute.arch.sync_threads()

    # eigenvectors of T: twisted factorization
    twist = _twist_sides(st)
    _twisted_sweeps(st, twist, pivmin)
    cute.arch.sync_threads()
    _twist_candidates(st, twist)
    cute.arch.sync_threads()
    _twisted_solve(st, twist)
    cute.arch.sync_threads()
    _normalize_halves(st, twist)
    cute.arch.sync_threads()
    _cluster_repair(st, span)
    _orthonormalize(st)

    # back-transform: z <- H_0 ... H_125 z, then U = Hd^T Z / sqrt(128)
    _stage_hadamard_f32(st, hadamard)
    _reflector_pair_products(st)
    cute.arch.sync_threads()
    _back_transform(st)
    cute.arch.sync_threads()
    _write_basis(st, uT, cta)


@cute.kernel
def eig_kernel(
    qtens: cute.Tensor,
    ktens: cute.Tensor,
    rows_q: cute.Tensor,
    rows_k: Int64,
    hadamard: cute.Tensor,
    uT: cute.Tensor,
    S: Int32,
    H: Int32,
    QP: Int32,
    KP: Int32,
    NQ: Int32,
    NK: Int32,
    gram_weight: Float32,
    lens: Int64,
    DIRECTIONS: cutlass.Constexpr,
    ROWS_MODE: cutlass.Constexpr,
    TWO_ROW_SETS: cutlass.Constexpr,
    SEPARATE: cutlass.Constexpr,
):
    """Run the eigensolver CTA (_eig_body) of DIRECTIONS directions.

    * ROWS_MODE: None for a call whose batch elements are all full (lens is not read); else (rows per sample, mapped
      rows) for valid prefix lengths, int32 [B] at the address lens: a row argument is [B, rows] or [rows], and
      with mapped rows _gram maps them from the padded sequence into each valid prefix.
    * TWO_ROW_SETS: Gk takes the NK rows at the address rows_k; without it the rows of rows_q (rows_k and NK are
      not read).
    * SEPARATE: the grid holds two CTAs per (batch, head) and gram_weight is not read.
    """
    lens_arg = (lens, ROWS_MODE[0], ROWS_MODE[1]) if ROWS_MODE else None
    _eig_body(
        qtens,
        ktens,
        rows_q,
        rows_k if TWO_ROW_SETS else None,
        hadamard,
        uT,
        S,
        H,
        QP,
        KP,
        NQ,
        NQ if not TWO_ROW_SETS else NK,
        gram_weight,
        DIRECTIONS,
        lens_arg,
        SEPARATE,
    )
