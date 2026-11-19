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
# l is the elimination multiplier of the inverse iteration.
# ruff: noqa: E741
"""Basis launch of the sm_120 preparation: one CTA per (batch, head) eigenproblem, the compensation basis of Q' and K'
(Gram of the sampled q, k tokens, Householder tridiagonalization, multisection bisection of the top 32 eigenvalues,
inverse iteration, MGS, back transform, U^T = (Hd^T V / sqrt(128))^T in BF16); the other CTAs scan V for the range
of what the kernel's PV format casts (the maxima that set its tensor scale). When the launch has room for it (FUSED),
the scan CTAs then run the kernel's V pass on the tiles a counter hands out, joined by the eigensolver CTAs as
basis_cta describes; the V pass is a launch of its own otherwise. The scan and the pass are those of the kernel
package (<kernel package>/v_pass.py), whose basis kernel is basis_cta with them; the helpers both V passes share
are here (bf16_f32, group16, stage_rows)."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int16, Int32, Int64, Uint16, Uint32
from cutlass.utils import SmemAllocator

from ...common.dsl import MMALayout, ldmatrix, nanosleep
from ...common.expmath import bf16x2, f2u, hash01, u2f, warp_max, warp_sum
from ...common.mem import cp_async16, gptr, load_invariant, load_vector, sptr, store_vector
from ...common.ptx import ld_global_acquire_b32, vote_ballot
from ...common.state import State
from ..ptx.shared import ld_global_nc_b32_hint_pred, ld_shared_f32_v4_pred, mma_bf16

EIG_THREADS = 512
NIT_BIS = 6
NIT_INV = 2

# shared memory map (bytes)
SM_UHH = 0  # packed Householder vectors (8440 floats)
SM_WORK = 33792  # 32 KB: Gram staging / G half staging / Z [128][32] + PIV [128][32]
SM_UBUF = 66560  # u of the Householder step in flight (128 floats)
SM_PBUF = 67072  # p = A u of that step (128 floats)
SM_DD = 67584  # diagonal of the tridiagonal matrix (128 floats)
SM_EE = 68096  # its off-diagonal (127 floats)
SM_E2 = 68608  # the squares of the off-diagonal, floored (128 floats)
SM_LAM = 69120  # the 32 eigenvalues
SM_ZC = 69248  # the MGS column in flight: two buffers of 128 floats, used in turn
SM_RED = 70272  # per-warp parts of the sums and bounds taken over the CTA; the join flag of basis_cta at 224
SM_SCL = 70976  # 32 inverse-iteration scales
SM_IDX2 = 71168  # up to 512 sampled token indices (Gram)
SM_VSTG = (
    73728  # V pass (after the eigensolver / scan): input tiles at 0 (2 x 32 KB), output stages here
)
SM_TOTAL = 92160
# The eigensolver also uses the start of the buffer twice: for the two cp.async buffers of the Gram rounds, 26112 B
# each (2 x 48 rows at a 272-byte stride) at 0 and 26112, and at the end for the Hadamard matrix (32 KB at SM_UHH).


def ldsm_x4_t(addr):
    """Four 8x8 16-bit matrices by one transposed ldmatrix (matrix m from the row addresses of lanes 8m .. 8m + 7)."""
    words = ldmatrix(sptr(Int16, addr, 16).llvm_ptr, 4, MMALayout.COL)
    return [words[i] for i in range(4)]


def pow2_inv_of_max(a, b):
    """2^-e with 2^e <= max(|a|, |b|) < 2^(e+1) (a power-of-two rescale factor)."""
    m = cute.arch.fmax(cute.math.abs(a), cute.math.abs(b))
    return (Int32(0x7F000000) - (m.bitcast(Int32) & 0x7F800000)).bitcast(Float32)


@cute.jit
def _uhh_off(k: Int32) -> Int32:
    a = k >> 2
    b = k & 3
    return 512 * a - 8 * a * (a - 1) + b * (128 - 4 * a)


# ------------------------------------------------------------------------------------------------ eigensolver steps
# The steps of _eig, in the order it runs them. st is the state of the thread (State) that _eig builds: the thread
# coordinates, the addresses of the shared-memory regions and the typed views of the regular arrays among them (sIdx,
# sRed, sG, sU, sP, sD, sE, sE2, sLam, sZ, sPiv, sScl, sHd), by name. No step contains a CTA barrier: _eig places
# them between the steps, and a step reads from shared memory what the steps before the last barrier wrote. acc, A,
# uu, za and zbv are the register tensors of _eig.
@cute.jit
def preload_tokens(st):
    """Thread tid < NQK copies sampled token index tid from rows_p to SM_IDX2: the first 512 indices, which the Gram
    copies read from there (those beyond are read from rows_p per copy)."""
    sIdx = st.sIdx
    if st.tid < st.NQK:
        sIdx[st.tid] = load_invariant(Uint32, st.rows_p + Int64(st.tid) * 4)


@cute.jit
def round_chunk(st, it: cutlass.Constexpr):
    """A Gram round is 48 sampled rows of q, then 48 of k, as 1536 16-byte chunks of which the thread copies three:
    (mat, row, chk) of its copy `it`, chunk chk of row `row` (0..47) of q (mat 0) or k (mat 1)."""
    cidx = it * EIG_THREADS + st.tid
    mat = cidx // 768
    rem = cidx - mat * 768
    return mat, rem >> 4, rem & 15


@cute.jit
def preloaded_token(st, sidx):
    """The token index of sample sidx from SM_IDX2 (the index clamped to the last sample and to the 512 preloaded)."""
    return Int32(st.sIdx[cutlass.min(cutlass.min(sidx, st.NQK - 1), 511)])


@cute.jit
def chunk_src(st, mat, tok, chk):
    """The global address of 16-byte chunk chk of the row of token tok of q (mat 0) or k (mat 1) of this (batch,
    head)."""
    return (
        Int64(cutlass.select_(mat != 0, st.k_p, st.q_p))
        + ((st.rowoff + Int64(tok)) * Int64(st.H) + Int64(st.h)) * 256
        + Int64(chk * 16)
    )


@cute.jit
def stage_first_rounds(st):
    """cp.async of the thread's chunks of rounds 0 and 1 into the buffers at 0 and 26112 (row r of a round at
    r * 272), one group per round; a sample past NQK copies zeros. Token indices: preload_tokens, barrier."""
    for pre in cutlass.range_constexpr(2):
        for it in cutlass.range_constexpr(3):
            mat, sm, chk = round_chunk(st, it)
            sidx = pre * 48 + sm
            ok = Int32(sidx < st.NQK)
            tok = preloaded_token(st, sidx)
            src = chunk_src(st, mat, tok, chk)
            cp_async16(
                st.sb + pre * 26112 + (mat * 48 + sm) * 272 + chk * 16,
                src,
                Int32(Uint32(cutlass.select_(ok != 0, Uint32(16), Uint32(0)))),
            )
        cute.arch.cp_async_commit_group()


@cute.jit
def gram_round(st, acc, buf):
    """Adds the 48 rows of q and of k staged at buf (their cp.async group waited for, barrier) to the warp's blocks
    of Xq^T Xq and Xk^T Xk, rows 16 mi .. 16 mi + 15, columns 8 nb .. 8 nb + 63: per 16 staged rows a 16 x 16 A
    fragment and eight 16 x 8 B fragments by transposed ldmatrix, BF16 mma.sync into f32. The thread's entries:
    acc[32 side + 4 n + 2 hh + e] = row 16 mi + g + 8 hh, column 8 (nb + n) + 2 t + e (side 0: q, 1: k)."""
    for ks in cutlass.range_constexpr(3):
        for side in cutlass.range_constexpr(2):
            xb = buf + side * 48 * 272
            a = ldsm_x4_t(
                xb + (16 * ks + (st.jm >> 1) * 8 + st.r8) * 272 + (16 * st.mi + (st.jm & 1) * 8) * 2
            )
            for np2 in cutlass.range_constexpr(4):
                bq = ldsm_x4_t(
                    xb
                    + (16 * ks + (st.jm & 1) * 8 + st.r8) * 272
                    + (8 * (st.nb + 2 * np2 + (st.jm >> 1))) * 2
                )
                for hn in cutlass.range_constexpr(2):
                    base = side * 32 + (2 * np2 + hn) * 4
                    r = mma_bf16(
                        [acc[base], acc[base + 1], acc[base + 2], acc[base + 3]],
                        a,
                        [bq[2 * hn], bq[2 * hn + 1]],
                    )
                    for e in cutlass.range_constexpr(4):
                        acc[base + e] = r[e]


@cute.jit
def refill(st, buf, rd):
    """cp.async of the thread's chunks of round rd + 2 into the buffer at buf, which round rd has left (every warp
    is done with it: barrier), as one group; past the last round the group is empty, which keeps the group
    accounting of the round loop uniform. A token index beyond the 512 preloaded is read from rows_p."""
    for it in cutlass.range_constexpr(3):
        mat, sm, chk = round_chunk(st, it)
        sidx = (rd + 2) * 48 + sm
        ok = Int32(sidx < st.NQK)
        tok = preloaded_token(st, sidx)
        if sidx >= 512:
            if sidx < st.NQK:
                tok = Int32(load_invariant(Uint32, st.rows_p + Int64(sidx) * 4))
        src = chunk_src(st, mat, tok, chk)
        cp_async16(
            buf + (mat * 48 + sm) * 272 + chk * 16,
            src,
            Int32(Uint32(cutlass.select_(ok != 0, Uint32(16), Uint32(0)))),
        )
    cute.arch.cp_async_commit_group()


@cute.jit
def publish_traces(st, acc):
    """The warp's part of the traces of Xq^T Xq and Xk^T Xk (the diagonal entries among its accumulators): lane 0
    writes them to SM_RED [warp] and [16 + warp]."""
    sRed = st.sRed
    trq = Float32(0.0)
    trk = Float32(0.0)
    for n in cutlass.range_constexpr(8):
        for hh2 in cutlass.range_constexpr(2):
            for e in cutlass.range_constexpr(2):
                on = Int32(16 * st.mi + st.g + 8 * hh2 == 8 * (st.nb + n) + 2 * st.t + e)
                trq = trq + Float32(
                    cutlass.select_(on != 0, acc[n * 4 + 2 * hh2 + e], Float32(0.0))
                )
                trk = trk + Float32(
                    cutlass.select_(on != 0, acc[32 + n * 4 + 2 * hh2 + e], Float32(0.0))
                )
    trq = warp_sum(trq)
    trk = warp_sum(trk)
    if st.lane == 0:
        sRed[st.warp] = trq
        sRed[16 + st.warp] = trk


@cute.jit
def normalize_gram(st, acc):
    """acc[0:32] = the thread's entries of G = Xq^T Xq / tr(Xq^T Xq) + Xk^T Xk / tr(Xk^T Xk) (a matrix of zero trace
    is left out), with the traces summed from the 16 warp parts (publish_traces, barrier)."""
    tq = Float32(0.0)
    tk2 = Float32(0.0)
    for wi in cutlass.range_constexpr(16):
        tq = tq + st.sRed[wi]
        tk2 = tk2 + st.sRed[16 + wi]
    iq = Float32(cutlass.select_(tq > Float32(0.0), cute.math.rcp(tq), Float32(0.0)))
    ik = Float32(cutlass.select_(tk2 > Float32(0.0), cute.math.rcp(tk2), Float32(0.0)))
    for e in cutlass.range_constexpr(32):
        acc[e] = acc[e] * iq + acc[32 + e] * ik


@cute.jit
def stage_gram_half(st, acc, half: cutlass.Constexpr):
    """The warps that hold rows 64 half .. 64 half + 63 of G write their entries to SM_WORK as f32 [64][128]. The
    barrier before it: every thread is done with what SM_WORK held."""
    sG = st.sG
    if (st.mi >> 2) == half:
        for n in cutlass.range_constexpr(8):
            col = 8 * (st.nb + n) + 2 * st.t
            row0 = 16 * st.mi + st.g - 64 * half
            sG[row0, col] = acc[n * 4 + 0]
            sG[row0, col + 1] = acc[n * 4 + 1]
            sG[row0 + 8, col] = acc[n * 4 + 2]
            sG[row0 + 8, col + 1] = acc[n * 4 + 3]


@cute.jit
def load_rows(st, A, half: cutlass.Constexpr):
    """The threads of rows 64 half .. 64 half + 63 take their part of row rr from the staged half of G
    (stage_gram_half, barrier): A[4 m + e] = G[rr][16 m + 4 qq + e]."""
    if (st.rr >> 6) == half:
        for m in cutlass.range_constexpr(8):
            v4 = load_vector(
                sptr(Float32, st.work + ((st.rr - 64 * half) * 128 + 16 * m + 4 * st.qq) * 4, 16), 4
            )
            for e in cutlass.range_constexpr(4):
                A[4 * m + e] = v4[e]


@cute.jit
def row_sums(st, A, k, ko: cutlass.Constexpr):
    """(|x|^2, x_0, d) of the row of A that the thread holds with the three other threads of the row, from column
    16 ko: x its entries right of column k, x_0 the entry of column k + 1, d the entry of column k."""
    sg = Float32(0.0)
    x0c = Float32(0.0)
    dc = Float32(0.0)
    for m in cutlass.range_constexpr(ko, 8):
        for e in cutlass.range_constexpr(4):
            j = 16 * m + 4 * st.qq + e
            xv = A[4 * m + e]
            if m == ko:
                msk = Float32(cutlass.select_(j > k, xv, Float32(0.0)))
                sg = sg + msk * msk
                dc = dc + Float32(cutlass.select_(j == k, xv, Float32(0.0)))
            else:
                sg = sg + xv * xv
            if m <= ko + 1:
                x0c = x0c + Float32(cutlass.select_(j == k + 1, xv, Float32(0.0)))
    sg = sg + cute.arch.shuffle_sync_bfly(sg, offset=1)
    x0c = x0c + cute.arch.shuffle_sync_bfly(x0c, offset=1)
    dc = dc + cute.arch.shuffle_sync_bfly(dc, offset=1)
    sg = sg + cute.arch.shuffle_sync_bfly(sg, offset=2)
    x0c = x0c + cute.arch.shuffle_sync_bfly(x0c, offset=2)
    dc = dc + cute.arch.shuffle_sync_bfly(dc, offset=2)
    return sg, x0c, dc


@cute.jit
def write_reflector(st, A, k, ko: cutlass.Constexpr, alpha, uscale):
    """The four threads of row k: u = (x - alpha e_1) uscale from their entries of the row, to SM_UBUF (columns from
    16 ko, zeros up to column k) and, from column k & ~3, to the packed place of step k in SM_UHH."""
    start = (k >> 2) << 2
    offk = _uhh_off(k)
    for m in cutlass.range_constexpr(ko, 8):
        ov = [Float32(0.0)] * 4
        for e in cutlass.range_constexpr(4):
            j = 16 * m + 4 * st.qq + e
            xv = A[4 * m + e]
            if m <= ko + 1:
                xv = xv - Float32(cutlass.select_(j == k + 1, alpha, Float32(0.0)))
            if m == ko:
                ov[e] = Float32(cutlass.select_(j > k, xv * uscale, Float32(0.0)))
            else:
                ov[e] = xv * uscale
        jb = 16 * m + 4 * st.qq
        store_vector(sptr(Float32, st.ubuf + jb * 4, 16), (ov[0], ov[1], ov[2], ov[3]))
        if m == ko:
            if jb >= start:
                store_vector(
                    sptr(Float32, st.uhh + (offk + jb - start) * 4, 16),
                    (ov[0], ov[1], ov[2], ov[3]),
                )
        else:
            store_vector(
                sptr(Float32, st.uhh + (offk + jb - start) * 4, 16),
                (ov[0], ov[1], ov[2], ov[3]),
            )


@cute.jit
def publish_reflector(st, A, k, ko: cutlass.Constexpr):
    """Step k (segment ko) in the warp of row k: with x the part of row k right of the diagonal, the reflector
    H = I - u u^T that maps x to alpha e_1, alpha = -sign(x_0) |x|, u = (x - alpha e_1) / sqrt(|x|^2 - alpha x_0)
    (u = 0 and alpha = x_0 when x has no entry but x_0). The threads of row k write u (write_reflector); its thread
    qq 0 writes d_k to SM_DD and e_k = alpha to SM_EE."""
    sD, sE = st.sD, st.sE
    if st.warp == (k >> 3):
        sg, x0c, dc = row_sums(st, A, k, ko)
        nrm = cute.math.sqrt(sg, approx=True, ftz=True)
        alpha = Float32(cutlass.select_(x0c >= Float32(0.0), -nrm, nrm))
        doit = Int32(sg - x0c * x0c > sg * Float32(1e-30))
        uscale = Float32(
            cutlass.select_(
                doit != 0,
                cute.math.rsqrt(sg - x0c * alpha, approx=True, ftz=True),
                Float32(0.0),
            )
        )
        alpha = Float32(cutlass.select_(doit != 0, alpha, x0c))
        if st.rr == k:
            write_reflector(st, A, k, ko, alpha, uscale)
            if st.qq == 0:
                sD[k] = dc
                sE[k] = alpha


@cute.jit
def reflector_products(st, A, uu, k, ko: cutlass.Constexpr):
    """Rows 16 ko and up (warps 2 ko and up): u from SM_UBUF (publish_reflector, barrier) into uu; p = (A u)[rr] for
    rr > k, 0 otherwise, summed over the four threads of the row, thread qq 0 writes it to SM_PBUF; lane 0 writes
    the warp's part of u . p to SM_RED [warp]. Returns (p, u[rr]); zeros in the idle warps."""
    sRed, sP = st.sRed, st.sP
    p = Float32(0.0)
    ur = Float32(0.0)
    if st.warp >= 2 * ko:
        s0a = Float32(0.0)
        s1a = Float32(0.0)
        for m in cutlass.range_constexpr(ko, 8):
            u4 = load_vector(sptr(Float32, st.ubuf + (16 * m + 4 * st.qq) * 4, 16), 4)
            for e in cutlass.range_constexpr(4):
                uu[4 * m + e] = u4[e]
            s0a = s0a + A[4 * m] * u4[0] + A[4 * m + 2] * u4[2]
            s1a = s1a + A[4 * m + 1] * u4[1] + A[4 * m + 3] * u4[3]
        sp = s0a + s1a
        sp = sp + cute.arch.shuffle_sync_bfly(sp, offset=1)
        sp = sp + cute.arch.shuffle_sync_bfly(sp, offset=2)
        p = Float32(cutlass.select_(st.rr > k, sp, Float32(0.0)))
        ur = st.sU[st.rr]
        part = Float32(cutlass.select_(st.qq == 0, ur * p, Float32(0.0)))
        part = part + cute.arch.shuffle_sync_bfly(part, offset=4)
        part = part + cute.arch.shuffle_sync_bfly(part, offset=8)
        part = part + cute.arch.shuffle_sync_bfly(part, offset=16)
        if st.lane == 0:
            sRed[st.warp] = part
        if st.qq == 0:
            sP[st.rr] = p
    return p, ur


@cute.jit
def rank2_update(st, A, uu, p, ur, ko: cutlass.Constexpr):
    """Rows 16 ko and up: A <- A - u w^T - w u^T, w = p - (u . p / 2) u, on the thread's entries of row rr from
    column 16 ko; p from SM_PBUF and the warp parts of u . p from SM_RED (reflector_products, barrier), u in uu,
    p[rr] and u[rr] as that step returned them."""
    if st.warp >= 2 * ko:
        kk = Float32(0.0)
        for wi in cutlass.range_constexpr(2 * ko, 16):
            kk = kk + st.sRed[wi]
        kk = kk * Float32(0.5)
        wr = p - kk * ur
        for m in cutlass.range_constexpr(ko, 8):
            p4 = load_vector(sptr(Float32, st.pbuf + (16 * m + 4 * st.qq) * 4, 16), 4)
            for e in cutlass.range_constexpr(4):
                wj = p4[e] - kk * uu[4 * m + e]
                A[4 * m + e] = A[4 * m + e] - ur * wj - wr * uu[4 * m + e]


@cute.jit
def store_last_block(st, A):
    """The 2 x 2 block the 126 steps leave: the threads of rows 126 and 127 that hold them write d_126 and d_127 to
    SM_DD and e_126 to SM_EE."""
    sD, sE = st.sD, st.sE
    if st.rr >= 126:
        for m in cutlass.range_constexpr(8):
            for e in cutlass.range_constexpr(4):
                j = 16 * m + 4 * st.qq + e
                if j >= 126:
                    if j == st.rr:
                        sD[st.rr] = A[4 * m + e]
                if j == 127:
                    if st.rr == 126:
                        sE[126] = A[4 * m + e]


@cute.jit
def publish_bounds(st):
    """Thread i < 128, on the tridiagonal matrix d (SM_DD), e (SM_EE) that the barrier after store_last_block
    completes: e2_i = max(e_{i-1}^2, 1e-30) to SM_E2 (e2_0 = 0), and the Gershgorin interval of row i, reduced over
    the warp: lane 0 writes the lower bound to SM_RED [warp] and the upper bound to [4 + warp]."""
    sE2, sRed = st.sE2, st.sRed
    if st.tid < 128:
        di = st.sD[st.tid]
        ea = Float32(0.0)
        if st.tid < 127:
            ea = cute.math.abs(st.sE[st.tid])
        eb = Float32(0.0)
        if st.tid > 0:
            eb = cute.math.abs(st.sE[st.tid - 1])
            sE2[st.tid] = cute.arch.fmax(eb * eb, Float32(1e-30))
        else:
            sE2[0] = Float32(0.0)
        glo = di - ea - eb
        ghi = di + ea + eb
        glo = -warp_max(-glo)
        ghi = warp_max(ghi)
        if st.lane == 0:
            sRed[st.warp] = glo
            sRed[4 + st.warp] = ghi


@cute.jit
def spectrum_bounds(st):
    """(lo, hi): the Gershgorin bounds of the spectrum, from those of warps 0..3 (publish_bounds, barrier)."""
    sRed = st.sRed
    lo0 = cute.arch.fmin(cute.arch.fmin(sRed[0], sRed[1]), cute.arch.fmin(sRed[2], sRed[3]))
    hi0 = cute.arch.fmax(cute.arch.fmax(sRed[4], sRed[5]), cute.arch.fmax(sRed[6], sRed[7]))
    return lo0, hi0


@cute.jit
def sturm_count(st, x):
    """The number of eigenvalues of the tridiagonal matrix below x: the sign changes of the Sturm sequence
    p_i = (d_i - x) p_{i-1} - e2_i p_{i-2} over SM_DD and SM_E2 (publish_bounds, barrier), without a division; a
    power-of-two rescale of (p_i, p_{i-1}) every four terms keeps them in range."""
    pc = Float32(1.0)
    pp = Float32(0.0)
    cnt = Uint32(0)
    for i4 in cutlass.range(32, unroll=2):
        d4 = load_vector(sptr(Float32, st.dd + i4 * 16, 16), 4)
        e4 = load_vector(sptr(Float32, st.e2b + i4 * 16, 16), 4)
        for jj in cutlass.range_constexpr(4):
            pn = (d4[jj] - x) * pc - e4[jj] * pp
            cnt = cnt + ((f2u(pn) ^ f2u(pc)) >> Uint32(31))
            pp = pc
            pc = pn
        scl = pow2_inv_of_max(pc, pp)
        pc = pc * scl
        pp = pp * scl
    return cnt


@cute.jit
def multisect(st, lo0, hi0, span):
    """The bracket (lo, hi) of eigenvalue jv (the jv-th largest) after NIT_BIS rounds of multisection by its 16
    threads (a half warp), from the spectrum bounds widened by span * 1e-6: 16 points cut the bracket into 17
    sections, thread sv counts the eigenvalues below point sv + 1 (sturm_count), and a ballot of the half warp
    picks the section that holds the eigenvalue."""
    mtarget = 127 - st.jv
    lo = lo0 - span * Float32(1e-6) - Float32(1e-30)
    hi = hi0 + span * Float32(1e-6) + Float32(1e-30)
    for it in cutlass.range_constexpr(NIT_BIS):
        wdt = (hi - lo) * Float32(1.0 / 17.0)
        x = lo + wdt * Float32(st.sv + 1)
        cnt = sturm_count(st, x)
        bal = vote_ballot(Int32(Int32(cnt) >= mtarget + 1))
        bits = (bal >> Uint32((st.lane >> 4) * 16)) & Uint32(0xFFFF)
        first = Int32(cute.arch.popc((~bits) & (bits - Uint32(1))))
        nlo = Float32(cutlass.select_(first > 0, lo + wdt * Float32(first), lo))
        nhi = lo + wdt * Float32(first + 1)
        isnone = Int32(bits == Uint32(0))
        lo = Float32(cutlass.select_(isnone != 0, lo + wdt * Float32(16.0), nlo))
        hi = Float32(cutlass.select_(isnone != 0, hi, nhi))
    return lo, hi


@cute.jit
def publish_eigenvalue(st, lo, hi):
    """Thread sv 0 of eigenvalue jv writes the middle of its bracket to SM_LAM [jv]."""
    sLam = st.sLam
    if st.sv == 0:
        sLam[st.jv] = (lo + hi) * Float32(0.5)


@cute.jit
def floored_pivot(piv, pfl):
    """piv, or pfl with the sign of piv when |piv| < pfl."""
    return Float32(
        cutlass.select_(
            cute.math.abs(piv) < pfl,
            Float32(cutlass.select_(piv < Float32(0.0), -pfl, pfl)),
            piv,
        )
    )


@cute.jit
def inverse_iteration(st, span):
    """Warp 0, lane c: the eigenvector of eigenvalue c (SM_LAM: publish_eigenvalue, barrier) by NIT_INV inverse
    iterations from a hashed start vector, in place in column c of Z (f32 [128][32] at SM_WORK): elimination without
    pivoting of T - lambda I (SM_DD, SM_EE), its pivots floored in magnitude at span * 1e-9 and kept behind Z, then
    back substitution. Each iteration starts from the last iterate times 1 / max |x|; the last such scale goes to
    SM_SCL [c]."""
    sD, sE, sZ, sPiv, sScl = st.sD, st.sE, st.sZ, st.sPiv, st.sScl
    if st.warp == 0:
        c = st.lane
        lam = st.sLam[c]
        pfl = span * Float32(1e-9) + Float32(1e-30)
        for i in cutlass.range(128, unroll=4):
            sZ[i, c] = hash01(i, c + 7)
        scale = Float32(1.0)
        for itv in cutlass.range_constexpr(NIT_INV):
            piv = sD[0] - lam
            piv = floored_pivot(piv, pfl)
            sPiv[0, c] = piv
            y = sZ[0, c] * scale
            sZ[0, c] = y
            for i in cutlass.range(1, 128, unroll=4):
                em = sE[i - 1]
                bi = sZ[i, c]
                di = sD[i]
                l = em * cute.arch.rcp_approx(piv)
                piv = (di - lam) - l * em
                piv = floored_pivot(piv, pfl)
                y = bi * scale - l * y
                sPiv[i, c] = piv
                sZ[i, c] = y
            xx = y * cute.arch.rcp_approx(piv)
            sZ[127, c] = xx
            amax = cute.math.abs(xx)
            for i2 in cutlass.range(127, unroll=4):
                i = 126 - i2
                yi = sZ[i, c]
                ei = sE[i]
                ri = cute.arch.rcp_approx(sPiv[i, c])
                xx = (yi - ei * xx) * ri
                sZ[i, c] = xx
                amax = cute.arch.fmax(amax, cute.math.abs(xx))
            scale = Float32(
                cutlass.select_(amax > Float32(0.0), cute.arch.rcp_approx(amax), Float32(1.0))
            )
        sScl[c] = scale


@cute.jit
def load_columns(st, za, zbv):
    """za, zbv = the thread's rows 4 lane .. 4 lane + 3 of columns 2 warp and 2 warp + 1 of Z, times their scales of
    SM_SCL (inverse_iteration, barrier)."""
    sca = st.sScl[2 * st.warp]
    scb = st.sScl[2 * st.warp + 1]
    for e in cutlass.range_constexpr(4):
        za[e] = st.sZ[4 * st.lane + e, 2 * st.warp] * sca
        zbv[e] = st.sZ[4 * st.lane + e, 2 * st.warp + 1] * scb


@cute.jit
def publish_column(st, za, zbv, c, buf):
    """Column c of the MGS, in the warp that holds it (in za or zbv): normalizes it and writes it to buf, the lane
    its four rows."""
    if st.warp == (c >> 1):
        odd = c & 1
        cv = [Float32(cutlass.select_(odd != 0, zbv[e], za[e])) for e in range(4)]
        n2 = warp_sum(cv[0] * cv[0] + cv[1] * cv[1] + cv[2] * cv[2] + cv[3] * cv[3])
        inv = cute.math.rsqrt(cute.arch.fmax(n2, Float32(1e-30)), approx=True, ftz=True)
        for e in cutlass.range_constexpr(4):
            cv[e] = cv[e] * inv
            za[e] = Float32(cutlass.select_(odd != 0, za[e], cv[e]))
            zbv[e] = Float32(cutlass.select_(odd != 0, cv[e], zbv[e]))
        store_vector(sptr(Float32, buf + st.lane * 16, 16), (cv[0], cv[1], cv[2], cv[3]))


@cute.jit
def project_out(st, za, zbv, c, buf):
    """Every warp removes from its columns behind c their component along column c, read from buf (publish_column,
    barrier)."""
    z4 = load_vector(sptr(Float32, buf + st.lane * 16, 16), 4)
    da = warp_sum(z4[0] * za[0] + z4[1] * za[1] + z4[2] * za[2] + z4[3] * za[3])
    db = warp_sum(z4[0] * zbv[0] + z4[1] * zbv[1] + z4[2] * zbv[2] + z4[3] * zbv[3])
    fa = Float32(cutlass.select_(2 * st.warp > c, da, Float32(0.0)))
    fb = Float32(cutlass.select_(2 * st.warp + 1 > c, db, Float32(0.0)))
    for e in cutlass.range_constexpr(4):
        za[e] = za[e] - fa * z4[e]
        zbv[e] = zbv[e] - fb * z4[e]


@cute.jit
def back_transform(st, za, zbv):
    """v = H_0 ... H_125 z on the warp's two columns: for k = 125 .. 0, z <- z - u_k (u_k . z), with u_k from its
    packed place in SM_UHH (complete since the tridiagonalization) by a predicated load: a lane whose rows lie
    before the packed part takes zeros."""
    for k2 in cutlass.range(126, unroll=1):
        k = 125 - k2
        start = (k >> 2) << 2
        u4 = ld_shared_f32_v4_pred(
            st.uhh + (_uhh_off(k) + 4 * st.lane - start) * 4, Int32(4 * st.lane >= start)
        )
        da = warp_sum(u4[0] * za[0] + u4[1] * za[1] + u4[2] * za[2] + u4[3] * za[3])
        db = warp_sum(u4[0] * zbv[0] + u4[1] * zbv[1] + u4[2] * zbv[2] + u4[3] * zbv[3])
        for e in cutlass.range_constexpr(4):
            za[e] = za[e] - da * u4[e]
            zbv[e] = zbv[e] - db * u4[e]


@cute.jit
def store_columns(st, za, zbv):
    """Writes the thread's rows of its two eigenvectors back to columns 2 warp and 2 warp + 1 of Z (SM_WORK)."""
    sZ = st.sZ
    for e in cutlass.range_constexpr(4):
        sZ[4 * st.lane + e, 2 * st.warp] = za[e]
        sZ[4 * st.lane + e, 2 * st.warp + 1] = zbv[e]


@cute.jit
def stage_hadamard(st, had_p):
    """Copies the Hadamard matrix Hd (BF16 [128][128] at had_p) over the packed Householder vectors (SM_UHH), which
    every warp has applied (back_transform, barrier): the thread four 16-byte parts."""
    for it in cutlass.range_constexpr(4):
        ci = it * EIG_THREADS + st.tid
        w4 = load_invariant(Uint32, had_p + Int64(ci) * 16, 4)
        store_vector(sptr(Uint32, st.hds + ci * 16, 16), (w4[0], w4[1], w4[2], w4[3]))


@cute.jit
def project_basis(st, ut_p, bh):
    """U = Hd^T V / sqrt(128) from Hd in SM_UHH and the eigenvectors V in Z (stage_hadamard, store_columns, barrier):
    thread (d = tid & 127, rg = tid >> 7) sums column d of Hd against the eigenvectors r = 8 rg .. 8 rg + 7 and
    stores U^T[r][d] in BF16 at ut_p [bh][r][d]."""
    dcol = st.tid & 127
    rg = st.tid >> 7
    uo = cute.make_rmem_tensor((8,), Float32)
    uo.fill(0.0)
    for j in cutlass.range(128, unroll=4):
        hv = u2f(st.sHd[j, dcol].to(Uint32) << Uint32(16))
        v0 = load_vector(sptr(Float32, st.zb + (j * 32 + 8 * rg) * 4, 16), 4)
        v1 = load_vector(sptr(Float32, st.zb + (j * 32 + 8 * rg + 4) * 4, 16), 4)
        for e in cutlass.range_constexpr(4):
            uo[e] = uo[e] + hv * v0[e]
            uo[4 + e] = uo[4 + e] + hv * v1[e]
    sc = Float32(0.08838834764831845)
    for e in cutlass.range_constexpr(8):
        r = 8 * rg + e
        hb = bf16x2(uo[e] * sc, Float32(0.0))
        cute.arch.store(
            gptr(Uint16, ut_p + (Int64(bh) * 4096 + Int64(r * 128 + dcol)) * 2), hb.to(Uint16)
        )


# ------------------------------------------------------------------------------------------------ eigensolver
@cute.jit
def _eig(
    q_p: Int64,
    k_p: Int64,
    rows_p: Int64,
    had_p: Int64,
    ut_p: Int64,
    bh: Int32,
    S: Int32,
    H: Int32,
    NQK: Int32,
    tid: Int32,
    sb: Int32,
):
    """The eigensolver CTA of (batch, head) bh (thread tid of EIG_THREADS, shared memory at sb with the map SM_*): the
    compensation basis U^T [32][128] in BF16 at ut_p [bh], from the NQK tokens of q and k that rows_p samples and the
    Hadamard matrix at had_p. The body is the sequence of the steps above, by phase. Every CTA barrier of the
    eigensolver is in this body, with what it publishes, and every thread reaches it: none is under a condition, and
    the loops around them have the same trip count in all threads. The state st gets the coordinates, the region
    addresses and the typed views of a phase where the phase begins. A view (s...) covers a region that is a regular
    array of one scalar type, for the scalar loads and stores that index it; the 16-byte vector loads and stores of
    the same arrays, the asynchronous copies, ldmatrix and the predicated loads take the region addresses."""
    st = State(tid=tid, sb=sb, NQK=NQK, H=H, q_p=q_p, k_p=k_p, rows_p=rows_p)
    st.lane = tid & 31
    st.warp = tid >> 5
    # the lane's entries of an mma.sync accumulator fragment: rows g and g + 8, columns 2 t and 2 t + 1
    st.g = st.lane >> 2
    st.t = st.lane & 3
    b = bh // H
    st.h = bh - b * H  # head, of batch b
    st.work = sb + SM_WORK
    st.zb = st.work  # Z [128][32], once G has left SM_WORK
    # the 16 warp parts of two sums or bounds (the second set at 16; at 4 for the bounds of four warps)
    st.sRed = cute.make_tensor(sptr(Float32, sb + SM_RED), cute.make_layout(32))

    # ------------------------------------------------ Gram matrices: rounds of 48 sampled rows in two cp.async
    # buffers (one round in flight while the previous one is multiplied; ldmatrix.trans, bf16 mma.sync, fp32
    # accumulation); rows staged at 272-byte stride; token indices preloaded (beyond 512: read per copy)
    # the warp's blocks of the two matrices: rows 16 mi .. 16 mi + 15, columns 8 nb .. 8 nb + 63
    st.mi = st.warp & 7
    st.nb = (st.warp >> 3) * 8
    nrounds = (NQK + 47) // 48
    st.rowoff = Int64(b) * Int64(S)  # the first token row of the batch
    st.sIdx = cute.make_tensor(sptr(Uint32, sb + SM_IDX2), cute.make_layout(512))
    acc = cute.make_rmem_tensor((64,), Float32)
    acc.fill(0.0)
    preload_tokens(st)
    cute.arch.sync_threads()  # publishes the token indices
    stage_first_rounds(st)
    # the row address a lane gives a transposed ldmatrix: row r8 of matrix jm
    st.jm = st.lane >> 3
    st.r8 = st.lane & 7
    for rd in cutlass.range(nrounds, unroll=1):
        cute.arch.cp_async_wait_group(1)
        cute.arch.sync_threads()  # publishes round rd: every thread has waited for its copies
        buf = sb + (rd & 1) * 26112
        gram_round(st, acc, buf)
        cute.arch.sync_threads()  # every warp is done with the buffer
        refill(st, buf, rd)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    publish_traces(st, acc)
    cute.arch.sync_threads()  # publishes the warp parts of the two traces
    normalize_gram(st, acc)

    # ------------------------------------------------ redistribute G (thread: row rr, columns 16m + 4qq + e)
    st.rr = tid >> 2
    st.qq = tid & 3
    # the staged half of G, f32 [64][128] in SM_WORK
    st.sG = cute.make_tensor(sptr(Float32, st.work), cute.make_layout((64, 128), stride=(128, 1)))
    A = cute.make_rmem_tensor((32,), Float32)
    A.fill(0.0)
    for half in cutlass.range_constexpr(2):
        # every thread is done with what SM_WORK held (for half 1: half 0)
        cute.arch.sync_threads()
        stage_gram_half(st, acc, half)
        cute.arch.sync_threads()  # publishes rows 64 half .. 64 half + 63 of G
        load_rows(st, A, half)

    # ------------------------------------------------ Householder tridiagonalization, 8 segments of 16 steps:
    # in segment ko the column chunks m < ko and the warps < 2 ko are idle
    st.ubuf = sb + SM_UBUF
    st.pbuf = sb + SM_PBUF
    st.dd = sb + SM_DD
    st.uhh = sb + SM_UHH
    st.hds = st.uhh  # Hd, once the Householder vectors have been applied
    st.sU = cute.make_tensor(sptr(Float32, st.ubuf), cute.make_layout(128))
    st.sP = cute.make_tensor(sptr(Float32, st.pbuf), cute.make_layout(128))
    st.sD = cute.make_tensor(sptr(Float32, st.dd), cute.make_layout(128))
    st.sE = cute.make_tensor(sptr(Float32, sb + SM_EE), cute.make_layout(128))
    uu = cute.make_rmem_tensor((32,), Float32)
    uu.fill(0.0)
    for ko in cutlass.range_constexpr(8):
        for k in cutlass.range(16 * ko, min(16 * ko + 16, 126), unroll=1):
            publish_reflector(st, A, k, ko)
            cute.arch.sync_threads()  # publishes u, d_k, e_k
            p, ur = reflector_products(st, A, uu, k, ko)
            cute.arch.sync_threads()  # publishes p and the warp parts of u . p
            rank2_update(st, A, uu, p, ur, ko)
    store_last_block(st, A)
    cute.arch.sync_threads()  # publishes the tridiagonal matrix d, e

    # ------------------------------------------------ e^2 (floored), Gershgorin bounds
    st.e2b = sb + SM_E2
    st.sE2 = cute.make_tensor(sptr(Float32, st.e2b), cute.make_layout(128))
    publish_bounds(st)
    cute.arch.sync_threads()  # publishes e^2 and the warp bounds
    lo0, hi0 = spectrum_bounds(st)
    span = hi0 - lo0

    # ------------------------------------------------ multisection bisection (division-free Sturm counts):
    # thread = section point sv of eigenvalue jv
    st.jv = tid >> 4
    st.sv = tid & 15
    lo, hi = multisect(st, lo0, hi0, span)
    st.sLam = cute.make_tensor(sptr(Float32, sb + SM_LAM), cute.make_layout(32))
    publish_eigenvalue(st, lo, hi)
    cute.arch.sync_threads()  # publishes the 32 eigenvalues

    # ------------------------------------------------ inverse iteration (warp 0, one eigenvector per lane)
    st.sZ = cute.make_tensor(sptr(Float32, st.zb), cute.make_layout((128, 32), stride=(32, 1)))
    # the pivots [128][32], behind Z
    st.sPiv = cute.make_tensor(
        sptr(Float32, st.work + 16384), cute.make_layout((128, 32), stride=(32, 1))
    )
    st.sScl = cute.make_tensor(sptr(Float32, sb + SM_SCL), cute.make_layout(32))
    inverse_iteration(st, span)
    cute.arch.sync_threads()  # publishes Z and the scales

    # ------------------------------------------------ MGS (one pass, normalizing); warp w holds columns 2w, 2w+1
    za = cute.make_rmem_tensor((4,), Float32)
    zbv = cute.make_rmem_tensor((4,), Float32)
    load_columns(st, za, zbv)
    zc = sb + SM_ZC
    for c in cutlass.range(32, unroll=1):
        # two buffers in turn: column c + 1 is written while a warp may still read column c
        buf = zc + (c & 1) * 512
        publish_column(st, za, zbv, c, buf)
        cute.arch.sync_threads()  # publishes the normalized column c
        project_out(st, za, zbv, c, buf)

    # ------------------------------------------------ back transform: v = H_0 ... H_125 z
    back_transform(st, za, zbv)
    cute.arch.sync_threads()  # every warp has applied the Householder vectors: their area is free
    store_columns(st, za, zbv)
    stage_hadamard(st, had_p)
    cute.arch.sync_threads()  # publishes the eigenvectors and Hd

    # ------------------------------------------------ U = Hd^T V / sqrt(128), stored as U^T bf16 [32][128]
    st.sHd = cute.make_tensor(sptr(Uint16, st.hds), cute.make_layout((128, 128), stride=(128, 1)))
    project_basis(st, ut_p, bh)


# ------------------------------------------------------------------------------------------------ V helpers
# What the V scans and V passes of the kernel packages share.
def bf16_f32(w, j):
    """BF16 half j (0: low, 1: high) of the word w as f32."""
    return u2f(w << Uint32(16)) if j == 0 else u2f(w & Uint32(0xFFFF0000))


def group16(base, rstride, cnt):
    """A lane's 8-channel sums, maxima and minima of V over the cnt valid keys of its 16-key group (rows rstride bytes
    apart from base)."""
    sums = [Float32(0.0)] * 8
    vmx = [Float32(-3.0e38)] * 8
    vmn = [Float32(3.0e38)] * 8
    for kk in range(16):
        ok = Int32(kk < cnt)
        w = ld_global_nc_b32_hint_pred(base + rstride * kk, ok, "evict_last", 4)
        for e in range(4):
            lo_ = u2f(w[e] << Uint32(16))
            hi_ = u2f(w[e] & Uint32(0xFFFF0000))
            sums[2 * e] = sums[2 * e] + lo_
            sums[2 * e + 1] = sums[2 * e + 1] + hi_
            vmx[2 * e] = cute.arch.fmax(
                vmx[2 * e], Float32(cutlass.select_(ok != 0, lo_, Float32(-3.0e38)))
            )
            vmx[2 * e + 1] = cute.arch.fmax(
                vmx[2 * e + 1], Float32(cutlass.select_(ok != 0, hi_, Float32(-3.0e38)))
            )
            vmn[2 * e] = cute.arch.fmin(
                vmn[2 * e], Float32(cutlass.select_(ok != 0, lo_, Float32(3.0e38)))
            )
            vmn[2 * e + 1] = cute.arch.fmin(
                vmn[2 * e + 1], Float32(cutlass.select_(ok != 0, hi_, Float32(3.0e38)))
            )
    return sums, vmx, vmn


def stage_rows(v_p, vin, st, tl, S, H, T):
    """cp.async of the 128 x 256 B rows of V tile tl (flat over (batch, head, tile)) into the shared-memory buffer
    vin by the 256 threads of a V-pass half, thread st eight of the 16-byte parts: row `key` at key * 256, its part p
    at (p ^ (key >> 4)) % 16 (zeros for the keys past S)."""
    bh = tl // T
    tt = tl - bh * T
    bb = bh // H
    hh = bh - bb * H
    for it in range(8):
        c = it * 256 + st
        key = c >> 4
        part = c & 15
        kg = tt * 128 + key
        src = (
            v_p
            + ((Int64(bb) * Int64(S) + Int64(cutlass.min(kg, S - 1))) * Int64(H) + Int64(hh)) * 256
            + Int64(part * 16)
        )
        cp_async16(
            vin + key * 256 + ((part ^ (key >> 4)) & 15) * 16,
            src,
            Int32(Uint32(cutlass.select_(kg < S, Uint32(16), Uint32(0)))),
        )


# ------------------------------------------------------------------------------------------------ launch body
@cute.jit
def basis_cta(
    q_p: Int64,
    k_p: Int64,
    rows_p: Int64,
    had_p: Int64,
    ut_p: Int64,
    done_p: Int64,
    S: Int32,
    H: Int32,
    T: Int32,
    NQK: Int32,
    BH: Int32,
    nspare: Int32,
    eigwait: Int32,
    FUSED: cutlass.Constexpr,
    vscan_gs,
    vscan,
    vpass,
):
    """The CTA of a basis launch (grid BH + nspare, EIG_THREADS threads): the CTAs below BH solve one eigenproblem
    each, the nspare others scan V with the scan of the kernel's PV format (FUSED: vscan(lo, hi, slot, tid, sb) over
    a range of tiles each, then its V pass vpass(ntiles, pid, tid, sb); otherwise vscan_gs(cta, tid, sb), grid-stride
    over all of V). sb: the CTA's shared memory."""
    tid, _, _ = cute.arch.thread_idx()
    pid, _, _ = cute.arch.block_idx()
    smem = SmemAllocator()
    buf = smem.allocate_array(Uint32, SM_TOTAL // 4, byte_alignment=128)
    sb = Int32(buf.toint())
    ntiles = BH * T
    if cutlass.const_expr(not FUSED):
        # eigenproblems, and the V maxima in the other CTAs (the V pass is a separate launch)
        if pid < BH:
            _eig(q_p, k_p, rows_p, had_p, ut_p, pid, S, H, NQK, tid, sb)
        else:
            vscan_gs(pid - BH, tid, sb)
    else:
        if pid < BH:
            _eig(q_p, k_p, rows_p, had_p, ut_p, pid, S, H, NQK, tid, sb)
        else:
            sp = pid - BH
            lo = (sp * ntiles) // nspare
            hi = ((sp + 1) * ntiles) // nspare
            vscan(lo, hi, sp, tid, sb)
            if tid == 0:
                cute.arch.fence_acq_rel_gpu()
                cute.arch.atomic_add(gptr(Uint32, done_p), Uint32(1), sem="release", scope="gpu")
        # The scan CTAs wait for each other, then take V tiles; an eigensolver CTA joins them when it is done: it waits
        # for the scan when eigwait is set (the host sets it when the grid has at most as many CTAs as the device has
        # SMs), otherwise it joins only if the scan has already finished. Nothing ever waits on an eigensolver CTA.
        flag = sb + SM_RED + 224
        if tid == 0:
            jn = Uint32(1)
            if pid < BH:
                jn = Uint32(eigwait != 0) | Uint32(ld_global_acquire_b32(done_p) >= Uint32(nspare))
            cute.arch.store(sptr(Uint32, flag), jn)
        cute.arch.sync_threads()
        join = Int32(cute.arch.load(sptr(Uint32, flag), Uint32))
        if join != 0:
            if tid == 0:
                seen = ld_global_acquire_b32(done_p)
                while seen < Uint32(nspare):
                    nanosleep(Int32(200))
                    seen = ld_global_acquire_b32(done_p)
                cute.arch.fence_acq_rel_gpu()
            cute.arch.sync_threads()
            vpass(ntiles, pid, tid, sb)
