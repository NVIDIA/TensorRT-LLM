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
"""Basis launch of the sm_120 preparation: the compensation basis of Q' and K', and the scan of V.

* One CTA per eigenproblem: Gram of the sampled q, k tokens, Householder tridiagonalization, multisection bisection
  of the top eigenvalues, inverse iteration, modified Gram-Schmidt (MGS), back transform,
  U^T = (Hd^T V / sqrt(128))^T in FP32.
* A problem solves for 32 or 64 directions (basis_plan) and stores the leading ones that its basis holds, the
  leading direction last. One problem per (batch, head) weighs the two Gram matrices; of the two problems of
  bases "separate" the first takes the Gram of k alone, for the basis of the Q-residual term, and the second
  that of q, for the basis of the K-residual term.
* The other CTAs scan V for the range of what the kernel's PV format casts (the maxima that set its tensor scale).
  When the launch has room for it (FUSED), they scan the tiles a counter hands out and then run the kernel's V pass
  on the tiles another counter hands out, joined by the eigensolver CTAs as basis_cta describes; the V pass is a
  launch of its own otherwise.
* The scan and the pass are those of the kernel package (<kernel package>/v_pass.py), whose basis kernel is basis_cta
  with them; the helpers both V passes share are here (bf16_f32, group16, stage_rows).
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int16, Int32, Int64, Uint16, Uint32
from cutlass.utils import SmemAllocator

from ...common.dsl import MMALayout, ldmatrix, nanosleep
from ...common.expmath import f2u, hash01, u2f, warp_max, warp_sum
from ...common.mem import cp_async16, gptr, load_invariant, load_vector, sptr, store_vector
from ...common.state import State
from ...constants import INV_SQRT_HEAD_DIM, MIN_GRAM_TRACE
from ..ptx.shared import ld_global_nc_b32_hint_pred, ld_shared_f32_v4_pred, mma_bf16
from ..valid_len import valid_len

EIG_THREADS = 512
NIT_INV = 2
# Gram: the rows of a part; the sums of a part are added to the accumulators of the two matrices when it ends
GRAM_PART_ROWS = 6144
# Householder: a row takes no reflection when the squared sum of its entries behind the first is at most this (the
# matrix has a trace of at most 2)
REFLECT_FLOOR = 1e-30
# MGS: at most this many sweeps; a sweep runs when a column of the sweep before kept less than half of its squared
# norm
MGS_SWEEPS = 2
# directions of a problem -> (threads of an eigenvalue in the multisection, its rounds)
BISECTION = {32: (16, 6), 64: (8, 8)}

# shared memory map (bytes)
SM_UHH = 0  # packed Householder vectors (8440 floats)
SM_WORK = 33792  # 32 KB: Gram staging / G half staging / Z [128][directions], PIV [128][32] behind 32 directions
SM_UBUF = 66560  # u of the Householder step in flight (128 floats)
SM_PBUF = 67072  # p = A u of that step (128 floats)
SM_DD = 67584  # diagonal of the tridiagonal matrix (128 floats)
SM_EE = 68096  # its off-diagonal (127 floats)
SM_E2 = 68608  # the squares of the off-diagonal, floored (128 floats)
SM_ZC = 69248  # the MGS column in flight: two buffers of 128 floats, used in turn
SM_RED = 70272  # per-warp parts of the sums and bounds taken over the CTA; the words of the MGS sweeps at RED_SWEEP
RED_SWEEP = 160  # one word per MGS sweep but the last: not zero when a column lost half of its squared norm in it
SM_IDX2 = 71168  # up to 512 sampled token indices of q (Gram); then the eigenvalues and the scales
SM_LAM = SM_IDX2  # up to 64 eigenvalues
SM_SCL = SM_IDX2 + 256  # up to 64 inverse-iteration scales
# V pass (after the eigensolver / scan): input tiles at 0 (2 x 32 KB), output stages here. The eigensolver keeps
# there up to 512 sampled token indices of k (Gram) and, with 64 directions, PIV [128][32]
SM_VSTG = 73728
SM_TOTAL = 92160
IDX = (SM_IDX2, SM_VSTG)  # the sampled token indices of q and of k
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


def basis_plan(rank: tuple[int, int], separate: bool) -> tuple[int, int]:
    """Return (problems per (batch, head), directions a problem solves for) of the terms rank = (r_q, r_k).

    * Problems: none without a term; one with bases "shared"; two with "separate".
    * Directions: 32 up to max(r_q, r_k) = 32, else 64.
    """
    top = max(rank)
    if top == 0:
        return 0, 0
    return 2 if separate else 1, 32 if top <= 32 else 64


def of_side(dtype, pair, mat):
    """pair[mat] as a dtype value, for mat 0 (q) or 1 (k) at trace time or at run time."""
    if isinstance(mat, int):
        return dtype(pair[mat])
    return dtype(cutlass.select_(mat != 0, dtype(pair[1]), dtype(pair[0])))


@cute.jit
def _uhh_off(k: Int32) -> Int32:
    a = k >> 2
    b = k & 3
    return 512 * a - 8 * a * (a - 1) + b * (128 - 4 * a)


# ------------------------------------------------------------------------------------------------ eigensolver steps
# The steps of _eig, in the order it runs them, on its thread state st and its register tensors part, acc, A, uu,
# z, zn. No step contains a CTA barrier: _eig places them between the steps, and a step reads from shared memory
# what the steps before the last barrier wrote.
@cute.jit
def sampled_token(st, mat, sidx):
    """Return the token of sample sidx of q (mat 0) or k (mat 1) of this batch element.

    * From the rows of that operand (st.rows_p, st.N of them); without lengths a row is clamped to the sequence.
    * With lengths (st.lens_p is their address) the batch element has st.nv valid tokens, and the rows mode of
      the operand decides (st.rows_mode holds it once per operand: that of q in bits 0..1, that of k in bits
      2..3):
        - 0, the default draw over the padded sequence: row r stands for token floor(r nv / S).
        - 1 and 2, the caller's rows, shared by the batch [N] or per batch element [B, N]: taken as given
          below nv, and as the last valid token from nv on.
    """
    rows_p = of_side(Int64, st.rows_p, mat)
    if cutlass.const_expr(st.lens_p is None):
        r = Int32(load_invariant(Uint32, rows_p + Int64(sidx) * 4))
        return Uint32(cutlass.max(cutlass.min(r, st.S - 1), Int32(0)))
    mode = of_side(Int32, (st.rows_mode & 3, st.rows_mode >> 2), mat)
    own = Int64(cutlass.select_(mode == 2, st.b * of_side(Int32, st.N, mat), Int32(0)))
    r = Int32(load_invariant(Uint32, rows_p + (own + Int64(sidx)) * 4))
    mapped = Int32((Int64(r) * Int64(st.nv)) // Int64(st.S))
    tok = Int32(cutlass.select_(mode == 0, mapped, cutlass.min(r, st.nv - 1)))
    return Uint32(cutlass.max(tok, Int32(0)))


@cute.jit
def preload_tokens(st):
    """Thread tid < N copies sampled token index tid of q to SM_IDX2 and of k to SM_VSTG: the first 512 indices of each
    operand, which the Gram copies read from there (those beyond are read from the rows per copy)."""
    for mat in cutlass.range_constexpr(2):
        sIdx = cute.make_tensor(sptr(Uint32, st.sb + IDX[mat]), cute.make_layout(512))
        if st.tid < st.N[mat]:
            sIdx[st.tid] = sampled_token(st, mat, st.tid)


@cute.jit
def round_chunk(st, it: cutlass.Constexpr):
    """A Gram round is 48 sampled rows of q, then 48 of k, as 1536 16-byte chunks of which the thread copies three:
    (mat, row, chk) of its copy `it`, chunk chk of row `row` (0..47) of q (mat 0) or k (mat 1)."""
    cidx = it * EIG_THREADS + st.tid
    mat = cidx // 768
    rem = cidx - mat * 768
    return mat, rem >> 4, rem & 15


@cute.jit
def staged_chunk(st, mat, sidx, chk, late: cutlass.Constexpr):
    """Return (source address, bytes) of the copy of 16-byte chunk chk of sample sidx of q (mat 0) or k (mat 1).

    * Source: the row of the sample's preloaded token, the index clamped to the last sample and to the 512
      preloaded. late: sidx may be 512 or more, and such a token is read from the rows.
    * No bytes, so zeros: a sample past the rows of the operand; an operand that has no weight in the Gram of
      this problem; with lengths, a batch element without a valid token.
    """
    n = of_side(Int32, st.N, mat)
    ok = Int32(sidx < n) & of_side(Int32, st.used, mat)
    if cutlass.const_expr(st.lens_p is not None):
        ok = ok & Int32(st.nv > 0)
    slot = cutlass.max(cutlass.min(cutlass.min(sidx, n - 1), 511), 0)
    idx = st.sb + of_side(Int32, IDX, mat) + slot * 4
    tok = Int32(cute.arch.load(sptr(Uint32, idx), Uint32))
    if cutlass.const_expr(late):
        if sidx >= 512:
            if sidx < n:
                tok = Int32(sampled_token(st, mat, sidx))
    return chunk_src(st, mat, tok, chk), Int32(
        Uint32(cutlass.select_(ok != 0, Uint32(16), Uint32(0)))
    )


@cute.jit
def chunk_src(st, mat, tok, chk):
    """Global address of 16-byte chunk chk of the row of token tok of q (mat 0) or k (mat 1) of this (batch, head)."""
    return (
        Int64(cutlass.select_(mat != 0, st.k_p, st.q_p))
        + (st.rowoff + Int64(tok)) * Int64(cutlass.select_(mat != 0, st.KP, st.QP))
        + Int64(st.h * 256 + chk * 16)
    )


@cute.jit
def stage_first_rounds(st):
    """cp.async of the thread's chunks of rounds 0 and 1 into the buffers at 0 and 26112 (row r of a round at
    r * 272), one group per round (staged_chunk). Token indices: preload_tokens, barrier."""
    for pre in cutlass.range_constexpr(2):
        for it in cutlass.range_constexpr(3):
            mat, sm, chk = round_chunk(st, it)
            src, nbytes = staged_chunk(st, mat, pre * 48 + sm, chk, False)
            cp_async16(st.sb + pre * 26112 + (mat * 48 + sm) * 272 + chk * 16, src, nbytes)
        cute.arch.cp_async_commit_group()


@cute.jit
def gram_round(st, part, buf):
    """Adds the 48 rows of q and of k staged at buf (their cp.async group waited for, barrier) to the warp's blocks
    of Xq^T Xq and Xk^T Xk, rows 16 mi .. 16 mi + 15, columns 8 nb .. 8 nb + 63: per 16 staged rows a 16 x 16 A
    fragment and eight 16 x 8 B fragments by transposed ldmatrix, BF16 mma.sync into f32. The thread's entries:
    part[32 side + 4 n + 2 hh + e] = row 16 mi + g + 8 hh, column 8 (nb + n) + 2 t + e (side 0: q, 1: k)."""
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
                        [part[base], part[base + 1], part[base + 2], part[base + 3]],
                        a,
                        [bq[2 * hn], bq[2 * hn + 1]],
                    )
                    for e in cutlass.range_constexpr(4):
                        part[base + e] = r[e]


@cute.jit
def close_part(part, acc):
    """acc += part, part = 0: the sums of a part of the Gram rounds join the accumulators of Xq^T Xq and Xk^T Xk."""
    for e in cutlass.range_constexpr(64):
        acc[e] = acc[e] + part[e]
        part[e] = Float32(0.0)


@cute.jit
def refill(st, buf, rd):
    """cp.async of the thread's chunks of round rd + 2 into the buffer at buf, which round rd has left (every warp
    is done with it: barrier), as one group (staged_chunk); past the last round the group is empty, which keeps the
    group accounting of the round loop uniform."""
    for it in cutlass.range_constexpr(3):
        mat, sm, chk = round_chunk(st, it)
        src, nbytes = staged_chunk(st, mat, (rd + 2) * 48 + sm, chk, True)
        cp_async16(buf + (mat * 48 + sm) * 272 + chk * 16, src, nbytes)
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
    """acc[0:32] = the thread's entries of G = weight_q Xq^T Xq / tr(Xq^T Xq) + weight_k Xk^T Xk / tr(Xk^T Xk),
    (weight_q, weight_k) = st.weights (a matrix whose trace is below MIN_GRAM_TRACE counts as zero), with the traces
    summed from the 16 warp parts (publish_traces, barrier)."""
    tq = Float32(0.0)
    tk2 = Float32(0.0)
    for wi in cutlass.range_constexpr(16):
        tq = tq + st.sRed[wi]
        tk2 = tk2 + st.sRed[16 + wi]
    weight_q, weight_k = st.weights
    iq = Float32(
        cutlass.select_(tq >= Float32(MIN_GRAM_TRACE), cute.math.rcp(tq) * weight_q, Float32(0.0))
    )
    ik = Float32(
        cutlass.select_(tk2 >= Float32(MIN_GRAM_TRACE), cute.math.rcp(tk2) * weight_k, Float32(0.0))
    )
    for e in cutlass.range_constexpr(32):
        acc[e] = acc[e] * iq + acc[32 + e] * ik


@cute.jit
def stage_gram_half(st, acc, half: cutlass.Constexpr):
    """The warps that hold rows 64 half .. 64 half + 63 of G write their entries to SM_WORK as f32 [64][128]."""
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
    H = I - u u^T that maps x to alpha e_1, alpha = -sign(x_0) |x|, u = (x - alpha e_1) / sqrt(|x|^2 - alpha x_0).
    No reflection (u = 0 and alpha = x_0) when the entries of x behind x_0 have a squared sum of at most
    REFLECT_FLOOR, or of at most 1e-30 of |x|^2. The threads of row k write u (write_reflector); its thread qq 0
    writes d_k to SM_DD and e_k = alpha to SM_EE."""
    sD, sE = st.sD, st.sE
    if st.warp == (k >> 3):
        sg, x0c, dc = row_sums(st, A, k, ko)
        nrm = cute.math.sqrt(sg, approx=True, ftz=True)
        alpha = Float32(cutlass.select_(x0c >= Float32(0.0), -nrm, nrm))
        doit = Int32(sg - x0c * x0c > cute.arch.fmax(sg * Float32(1e-30), Float32(REFLECT_FLOOR)))
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
    """The bracket (lo, hi) of eigenvalue jv (the jv-th largest) after the rounds of multisection by its n threads
    (BISECTION; consecutive lanes of a warp), from the spectrum bounds widened by span * 1e-6: n points cut the
    bracket into n + 1 sections, thread sv counts the eigenvalues below point sv + 1 (sturm_count), and a ballot of
    the n threads picks the section that holds the eigenvalue."""
    n, rounds = BISECTION[st.directions]
    mtarget = 127 - st.jv
    lo = lo0 - span * Float32(1e-6) - Float32(1e-30)
    hi = hi0 + span * Float32(1e-6) + Float32(1e-30)
    for _ in cutlass.range_constexpr(rounds):
        wdt = (hi - lo) * Float32(1.0 / (n + 1))
        x = lo + wdt * Float32(st.sv + 1)
        cnt = sturm_count(st, x)
        bal = Uint32(cute.arch.vote_ballot_sync(Int32(cnt) >= mtarget + 1))
        bits = (bal >> Uint32((st.lane // n) * n)) & Uint32((1 << n) - 1)
        first = Int32(cute.arch.popc((~bits) & (bits - Uint32(1))))
        nlo = Float32(cutlass.select_(first > 0, lo + wdt * Float32(first), lo))
        nhi = lo + wdt * Float32(first + 1)
        isnone = Int32(bits == Uint32(0))
        lo = Float32(cutlass.select_(isnone != 0, lo + wdt * Float32(n), nlo))
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
def inverse_iteration(st, span, first: cutlass.Constexpr):
    """Warp 0, lane c - first: the eigenvector of eigenvalue c (SM_LAM: publish_eigenvalue, barrier) by NIT_INV
    inverse iterations from a hashed start vector, in place in column c of Z (f32 [128][directions] at SM_WORK):
    elimination without pivoting of T - lambda I (SM_DD, SM_EE), its pivots floored in magnitude at span * 1e-9 and
    kept in PIV, then back substitution. Each iteration starts from the last iterate times 1 / max |x|; the last
    such scale goes to SM_SCL [c]."""
    sD, sE, sZ, sPiv, sScl = st.sD, st.sE, st.sZ, st.sPiv, st.sScl
    if st.warp == 0:
        c = first + st.lane
        lam = st.sLam[c]
        pfl = span * Float32(1e-9) + Float32(1e-30)
        for i in cutlass.range(128, unroll=4):
            sZ[i, c] = hash01(i, c + 7)
        scale = Float32(1.0)
        for _ in cutlass.range_constexpr(NIT_INV):
            piv = sD[0] - lam
            piv = floored_pivot(piv, pfl)
            sPiv[0, st.lane] = piv
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
                sPiv[i, st.lane] = piv
                sZ[i, c] = y
            xx = y * cute.arch.rcp_approx(piv)
            sZ[127, c] = xx
            amax = cute.math.abs(xx)
            for i2 in cutlass.range(127, unroll=4):
                i = 126 - i2
                yi = sZ[i, c]
                ei = sE[i]
                ri = cute.arch.rcp_approx(sPiv[i, st.lane])
                xx = (yi - ei * xx) * ri
                sZ[i, c] = xx
                amax = cute.arch.fmax(amax, cute.math.abs(xx))
            scale = Float32(
                cutlass.select_(amax > Float32(0.0), cute.arch.rcp_approx(amax), Float32(1.0))
            )
        sScl[c] = scale


@cute.jit
def load_columns(st, z, zn):
    """z[4 j + e] = the thread's rows 4 lane .. 4 lane + 3 of column st.cols warp + j of Z, the warp's column j, times
    its scale of SM_SCL (inverse_iteration, barrier); zn[j] = the squared norm of that column."""
    for j in cutlass.range_constexpr(st.cols):
        col = st.cols * st.warp + j
        sc = st.sScl[col]
        for e in cutlass.range_constexpr(4):
            z[4 * j + e] = st.sZ[4 * st.lane + e, col] * sc
        zn[j] = warp_sum(
            z[4 * j] * z[4 * j]
            + z[4 * j + 1] * z[4 * j + 1]
            + z[4 * j + 2] * z[4 * j + 2]
            + z[4 * j + 3] * z[4 * j + 3]
        )


@cute.jit
def publish_column(st, z, zn, c, buf, lost_p):
    """Column c of an MGS sweep, in the warp that holds it: normalizes it and writes it to buf, the lane its four rows.

    * zn: the squared norms that the warp's columns had before the sweep; that of column c becomes 1.
    * lost_p: None, or the shared-memory word that lane 0 sets when the column kept less than half of that squared
      norm: the projections of the sweep took the rest.
    """
    if st.warp == c // st.cols:
        own = c % st.cols
        cv = [z[e] for e in range(4)]
        before = zn[0]
        for j in cutlass.range_constexpr(1, st.cols):
            before = Float32(cutlass.select_(own == j, zn[j], before))
            for e in cutlass.range_constexpr(4):
                cv[e] = Float32(cutlass.select_(own == j, z[4 * j + e], cv[e]))
        n2 = warp_sum(cv[0] * cv[0] + cv[1] * cv[1] + cv[2] * cv[2] + cv[3] * cv[3])
        if cutlass.const_expr(lost_p is not None):
            if st.lane == 0:
                if n2 < Float32(0.5) * before:
                    cute.arch.store(sptr(Uint32, lost_p), Uint32(1))
        for j in cutlass.range_constexpr(st.cols):
            zn[j] = Float32(cutlass.select_(own == j, Float32(1.0), zn[j]))
        inv = cute.math.rsqrt(cute.arch.fmax(n2, Float32(1e-30)), approx=True, ftz=True)
        for e in cutlass.range_constexpr(4):
            cv[e] = cv[e] * inv
            for j in cutlass.range_constexpr(st.cols):
                z[4 * j + e] = Float32(cutlass.select_(own == j, cv[e], z[4 * j + e]))
        store_vector(sptr(Float32, buf + st.lane * 16, 16), (cv[0], cv[1], cv[2], cv[3]))


@cute.jit
def project_out(st, z, c, buf):
    """Every warp removes from its columns behind c their component along column c, read from buf (publish_column,
    barrier)."""
    z4 = load_vector(sptr(Float32, buf + st.lane * 16, 16), 4)
    for j in cutlass.range_constexpr(st.cols):
        d = warp_sum(
            z4[0] * z[4 * j] + z4[1] * z[4 * j + 1] + z4[2] * z[4 * j + 2] + z4[3] * z[4 * j + 3]
        )
        f = Float32(cutlass.select_(st.cols * st.warp + j > c, d, Float32(0.0)))
        for e in cutlass.range_constexpr(4):
            z[4 * j + e] = z[4 * j + e] - f * z4[e]


@cute.jit
def back_transform(st, z):
    """v = H_0 ... H_125 z on the warp's columns: for k = 125 .. 0, z <- z - u_k (u_k . z), with u_k from its packed
    place in SM_UHH (complete since the tridiagonalization) by a predicated load: a lane whose rows lie before the
    packed part takes zeros."""
    for k2 in cutlass.range(126, unroll=1):
        k = 125 - k2
        start = (k >> 2) << 2
        u4 = ld_shared_f32_v4_pred(
            st.uhh + (_uhh_off(k) + 4 * st.lane - start) * 4, Int32(4 * st.lane >= start)
        )
        for j in cutlass.range_constexpr(st.cols):
            d = warp_sum(
                u4[0] * z[4 * j]
                + u4[1] * z[4 * j + 1]
                + u4[2] * z[4 * j + 2]
                + u4[3] * z[4 * j + 3]
            )
            for e in cutlass.range_constexpr(4):
                z[4 * j + e] = z[4 * j + e] - d * u4[e]


@cute.jit
def store_columns(st, z):
    """Writes the thread's rows of the warp's eigenvectors back to their columns of Z (SM_WORK)."""
    sZ = st.sZ
    for j in cutlass.range_constexpr(st.cols):
        for e in cutlass.range_constexpr(4):
            sZ[4 * st.lane + e, st.cols * st.warp + j] = z[4 * j + e]


@cute.jit
def stage_hadamard(st, had_p):
    """Copies the Hadamard matrix Hd (BF16 [128][128] at had_p) over the packed Householder vectors (SM_UHH), which
    every warp has applied (back_transform, barrier): the thread four 16-byte parts."""
    for it in cutlass.range_constexpr(4):
        ci = it * EIG_THREADS + st.tid
        w4 = load_invariant(Uint32, had_p + Int64(ci) * 16, 4)
        store_vector(sptr(Uint32, st.uhh + ci * 16, 16), (w4[0], w4[1], w4[2], w4[3]))


@cute.jit
def project_basis(st, ut_p, bh, keep):
    """U = Hd^T V / sqrt(128) from Hd in SM_UHH and the eigenvectors V in Z (stage_hadamard, store_columns, barrier):
    thread (d = tid & 127, rg = tid >> 7) sums, per 32 directions, column d of Hd against the eigenvectors
    r = 8 rg .. 8 rg + 7 of them and stores U^T[r][d] for r < keep in FP32 at ut_p [bh][keep - 1 - r][d]: the basis
    [keep][128] of (batch, head) bh, the leading direction last."""
    dcol = st.tid & 127
    rg = st.tid >> 7
    whole = isinstance(keep, int) and keep == st.directions
    for first in cutlass.range_constexpr(0, st.directions, 32):
        uo = cute.make_rmem_tensor((8,), Float32)
        uo.fill(0.0)
        for j in cutlass.range(128, unroll=4):
            hv = u2f(st.sHd[j, dcol].to(Uint32) << Uint32(16))
            v0 = load_vector(
                sptr(Float32, st.work + (j * st.directions + first + 8 * rg) * 4, 16), 4
            )
            v1 = load_vector(
                sptr(Float32, st.work + (j * st.directions + first + 8 * rg + 4) * 4, 16), 4
            )
            for e in cutlass.range_constexpr(4):
                uo[e] = uo[e] + hv * v0[e]
                uo[4 + e] = uo[4 + e] + hv * v1[e]
        sc = Float32(INV_SQRT_HEAD_DIM)
        for e in cutlass.range_constexpr(8):
            r = first + 8 * rg + e
            kept = cutlass.const_expr(whole) or r < keep
            if kept:
                row = Int64(bh) * Int64(keep) + Int64(keep - 1 - r)
                cute.arch.store(gptr(Float32, ut_p + (row * 128 + dcol) * 4), uo[e] * sc)


# ------------------------------------------------------------------------------------------------ eigensolver
@cute.jit
def _eig(
    q_p: Int64,
    k_p: Int64,
    rows_p,
    had_p: Int64,
    ut_addrs,
    problem: Int32,
    S: Int32,
    H: Int32,
    QP: Int32,
    KP: Int32,
    N,
    weights,
    tid: Int32,
    sb: Int32,
    lens_p,
    rows_mode,
    rank: cutlass.Constexpr,
    separate: cutlass.Constexpr,
):
    """Run the eigensolver CTA of a problem (thread tid of EIG_THREADS, shared memory at sb with the map SM_*).

    * rank, separate: the channels (r_q, r_k) of the two compensation terms and whether each has its own basis;
      basis_plan gives the problems per (batch, head) and the directions a problem solves for. Problem `problem` is
      that of (batch, head) problem // problems.
    * One problem per (batch, head): the Gram matrices of q and k weighed with `weights`; its max(r_q, r_k) leading
      directions are the shared basis, at the first address of ut_addrs. Two: the even one takes the Gram of k
      alone, r_q directions to the basis of the Q-residual term, the odd one that of q alone, r_k to the basis
      of the K-residual term.
    * Result: U^T in FP32, [B H][kept directions][128], at ut_addrs = (basis of the Q-residual term, basis of the
      K-residual term; the shared basis at the first), from the tokens of q and k that rows_p samples (the
      pairs rows_p, N: the rows of q and of k and their counts) and the Hadamard matrix at had_p.
    * The body is the sequence of the steps above, by phase. Every CTA barrier of the eigensolver is in this body,
      with what it publishes, and every thread reaches it: the loops around the barriers have the same trip count
      in all threads, and the one condition around barriers, that of a further MGS sweep, is a shared-memory word
      that every thread reads behind a barrier.
    * st gets the coordinates, the region addresses and the typed views of a phase where the phase begins. A view
      (s...) covers a regular array of one scalar type, for the scalar loads and stores that index it; the 16-byte
      vector loads and stores, the asynchronous copies, ldmatrix and the predicated loads take the region addresses.
    * QP, KP: the bytes between consecutive tokens of q and of k.
    * lens_p, rows_mode: None, or the global address of the valid prefix lengths and the rows mode of
      sampled_token: the sampled tokens then lie in the valid prefix of the batch element.
    """
    problems, directions = basis_plan(rank, separate)
    st = State(
        tid=tid, sb=sb, N=N, H=H, QP=QP, KP=KP, q_p=q_p, k_p=k_p, rows_p=rows_p, lens_p=lens_p
    )
    st.rows_mode = rows_mode
    st.directions = directions
    st.cols = directions // 16  # MGS columns of a warp
    st.lane = tid & 31
    st.warp = tid >> 5
    # the lane's entries of an mma.sync accumulator fragment: rows g and g + 8, columns 2 t and 2 t + 1
    st.g = st.lane >> 2
    st.t = st.lane & 3
    bh = problem // problems
    basis_p = ut_addrs[0]
    keep = max(rank)
    if cutlass.const_expr(problems == 2):
        second = problem - bh * 2
        weight_q = Float32(2 * second)
        weights = (weight_q, Float32(2.0) - weight_q)
        basis_p = of_side(Int64, ut_addrs, second)
        keep = of_side(Int32, rank, second)
    st.weights = weights
    # the tokens of an operand without weight are not copied (staged_chunk); its rows are still preloaded
    st.used = (Int32(weights[0] != Float32(0.0)), Int32(weights[1] != Float32(0.0)))
    b = bh // H
    st.h = bh - b * H  # head, of batch b
    st.S = S
    if cutlass.const_expr(lens_p is not None):
        st.b = b
        st.nv = valid_len(lens_p, b, S)  # the valid tokens of the batch element
    st.work = sb + SM_WORK  # also Z [128][directions], once G has left it
    # the 16 warp parts of two sums or bounds (the second set at 16; at 4 for the bounds of four warps)
    st.sRed = cute.make_tensor(sptr(Float32, sb + SM_RED), cute.make_layout(32))

    # ------------------------------------------------ Gram matrices: rounds of 48 sampled rows in two cp.async
    # buffers, one round in flight while the previous one is multiplied (stage_first_rounds, gram_round, refill)
    # the warp's blocks of the two matrices: rows 16 mi .. 16 mi + 15, columns 8 nb .. 8 nb + 63
    st.mi = st.warp & 7
    st.nb = (st.warp >> 3) * 8
    nrounds = (cutlass.max(N[0] * st.used[0], N[1] * st.used[1]) + 47) // 48
    st.rowoff = Int64(b) * Int64(S)  # the first token row of the batch
    part = cute.make_rmem_tensor((64,), Float32)
    part.fill(0.0)
    acc = cute.make_rmem_tensor((64,), Float32)
    acc.fill(0.0)
    preload_tokens(st)
    cute.arch.sync_threads()  # publishes the token indices
    stage_first_rounds(st)
    # the row address a lane gives a transposed ldmatrix: row r8 of matrix jm
    st.jm = st.lane >> 3
    st.r8 = st.lane & 7
    # the first part, then the further parts (rows beyond GRAM_PART_ROWS)
    part_rounds = GRAM_PART_ROWS // 48
    for rd in cutlass.range(cutlass.min(nrounds, part_rounds), unroll=1):
        cute.arch.cp_async_wait_group(1)
        cute.arch.sync_threads()  # publishes round rd: every thread has waited for its copies
        buf = sb + (rd & 1) * 26112
        gram_round(st, part, buf)
        cute.arch.sync_threads()  # every warp is done with the buffer
        refill(st, buf, rd)
    close_part(part, acc)
    for first in cutlass.range(part_rounds, nrounds, part_rounds, unroll=1):
        for rd in cutlass.range(first, cutlass.min(first + part_rounds, nrounds), unroll=1):
            cute.arch.cp_async_wait_group(1)
            cute.arch.sync_threads()  # publishes round rd
            buf = sb + (rd & 1) * 26112
            gram_round(st, part, buf)
            cute.arch.sync_threads()  # every warp is done with the buffer
            refill(st, buf, rd)
        close_part(part, acc)
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
    st.uhh = sb + SM_UHH  # also Hd, once the Householder vectors have been applied
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
    st.jv = tid // BISECTION[directions][0]
    st.sv = tid % BISECTION[directions][0]
    lo, hi = multisect(st, lo0, hi0, span)
    st.sLam = cute.make_tensor(sptr(Float32, sb + SM_LAM), cute.make_layout(directions))
    publish_eigenvalue(st, lo, hi)
    cute.arch.sync_threads()  # publishes the eigenvalues

    # ------------------------------------------------ inverse iteration (warp 0, one eigenvector per lane, 32 at a
    # time)
    st.sZ = cute.make_tensor(
        sptr(Float32, st.work), cute.make_layout((128, directions), stride=(directions, 1))
    )
    # the pivots [128][32] of the 32 eigenvectors in flight: behind Z, or in the V stage area when Z fills SM_WORK
    piv = st.work + 16384 if directions == 32 else sb + SM_VSTG
    st.sPiv = cute.make_tensor(sptr(Float32, piv), cute.make_layout((128, 32), stride=(32, 1)))
    st.sScl = cute.make_tensor(sptr(Float32, sb + SM_SCL), cute.make_layout(directions))
    for first in cutlass.range_constexpr(0, directions, 32):
        inverse_iteration(st, span, first)
    lost = sb + SM_RED + RED_SWEEP
    if tid == 0:
        for sweep in cutlass.range_constexpr(MGS_SWEEPS - 1):
            cute.arch.store(sptr(Uint32, lost + 4 * sweep), Uint32(0))
    cute.arch.sync_threads()  # publishes Z, the scales and the cleared words of the MGS sweeps

    # ------------------------------------------------ MGS, normalizing; warp w holds st.cols columns from column
    # st.cols w
    # * a further sweep runs while a column kept less than half of its squared norm, MGS_SWEEPS at most
    # * the word of a sweep is shared, so every thread takes the same branch around the barriers
    z = cute.make_rmem_tensor((4 * st.cols,), Float32)
    zn = cute.make_rmem_tensor((st.cols,), Float32)
    load_columns(st, z, zn)
    zc = sb + SM_ZC
    for sweep in cutlass.range_constexpr(MGS_SWEEPS):
        run = Uint32(1)
        if cutlass.const_expr(sweep > 0):
            run = cute.arch.load(sptr(Uint32, lost + 4 * (sweep - 1)), Uint32)
        if run != Uint32(0):
            for c in cutlass.range(directions, unroll=1):
                # two buffers in turn: column c + 1 is written while a warp may still read column c
                buf = zc + (c & 1) * 512
                lost_p = lost + 4 * sweep if sweep < MGS_SWEEPS - 1 else None
                publish_column(st, z, zn, c, buf, lost_p)
                cute.arch.sync_threads()  # publishes the normalized column c and whether it lost half
                project_out(st, z, c, buf)

    # ------------------------------------------------ back transform: v = H_0 ... H_125 z
    back_transform(st, z)
    cute.arch.sync_threads()  # every warp has applied the Householder vectors: their area is free
    store_columns(st, z)
    stage_hadamard(st, had_p)
    cute.arch.sync_threads()  # publishes the eigenvectors and Hd

    # ------------------------------------------------ U = Hd^T V / sqrt(128), stored as U^T f32 [keep][128]
    st.sHd = cute.make_tensor(sptr(Uint16, st.uhh), cute.make_layout((128, 128), stride=(128, 1)))
    project_basis(st, basis_p, bh, keep)


# ------------------------------------------------------------------------------------------------ V helpers
# What the V scans and V passes of the kernel packages share.
# V pass: the input rows of a half (128 keys x 256 B, the two halves from the start of shared memory), the offset in
# the reduction area of the word of a half (the tile it took from the counter), and the byte offset of the tile
# counter in the counters of the launch (the scan counter is first)
VIN_BYTES = 32768
RED_TILE = 192
# the words of the done buffer (zero at the start of a call): the V-scan tiles published, the V-pass tile counter, the
# V-scan tile claim counter, the bits of the V-scan maximum (a non-negative float orders as its unsigned bits)
DONE_SCAN = 0
DONE_TILES = 4
DONE_CLAIM = 8
DONE_MAX = 12
RED_SCAN = 200  # the smem word that shows a CTA's claimed scan tiles


@cute.jit
def scan_max(
    vmax_p: Int64,
    nspare: Int32,
    tid: Int32,
    red: Int32,
    cop: cutlass.Constexpr,
    NT: cutlass.Constexpr,
    SYNC: cutlass.Constexpr,
):
    """Return max|v - vbar| over all nspare scan slots (loaded with cache operator cop), reduced by the NT threads of
    the CTA through red. SYNC: wait for the CTA's earlier readers of red first."""
    c = Float32(0.0)
    for i in cutlass.range(tid, nspare, NT, unroll=1):
        c = cute.arch.fmax(
            c, u2f(cute.arch.load(gptr(Uint32, vmax_p + Int64(i) * 4), Uint32, cop=cop))
        )
    c = warp_max(c)
    if cutlass.const_expr(SYNC):
        cute.arch.sync_threads()
    if (tid & 31) == 0:
        cute.arch.store(sptr(Float32, red + (tid >> 5) * 4), c)
    cute.arch.sync_threads()
    c = Float32(0.0)
    for wi in cutlass.range_constexpr(NT // 32):
        c = cute.arch.fmax(c, cute.arch.load(sptr(Float32, red + wi * 4), Float32))
    return c


@cute.jit
def next_scan_tiles(sb: Int32, done_p: Int64, tid: Int32):
    """Return the first of the next four scan tiles of a fused scan CTA: thread 0 takes them from the claim counter,
    the CTA's smem word RED_SCAN shows them."""
    cute.arch.sync_threads()  # the CTA is done with the word's last value
    if tid == 0:
        cute.arch.store(
            sptr(Uint32, sb + SM_RED + RED_SCAN),
            cute.arch.atomic_add(
                gptr(Uint32, done_p + DONE_CLAIM), Uint32(4), sem="relaxed", scope="gpu"
            ),
        )
    cute.arch.sync_threads()
    return Int32(cute.arch.load(sptr(Uint32, sb + SM_RED + RED_SCAN), Uint32))


@cute.jit
def publish_scan(mx: Float32, scanned: Int32, tid: Int32, sb: Int32, done_p: Int64):
    """Fold the CTA's max|v - vbar| (mx: the thread's part) into the maximum word of the done buffer, then count the
    CTA's scanned tiles as published; the V pass waits for all tiles and reads the maximum after them."""
    mx = warp_max(mx)
    red = sb + SM_RED
    if (tid & 31) == 0:
        cute.arch.store(sptr(Float32, red + (tid >> 5) * 4), mx)
    cute.arch.sync_threads()
    if tid == 0:
        c2 = Float32(0.0)
        for wi in cutlass.range_constexpr(EIG_THREADS // 32):
            c2 = cute.arch.fmax(c2, cute.arch.load(sptr(Float32, red + wi * 4), Float32))
        cute.arch.red(
            gptr(Uint32, done_p + DONE_MAX),
            f2u(c2),
            op="max",
            dtype="u32",
            sem="relaxed",
            scope="gpu",
        )
        cute.arch.fence_acq_rel_gpu()
        cute.arch.atomic_add(
            gptr(Uint32, done_p + DONE_SCAN), Uint32(scanned), sem="release", scope="gpu"
        )


@cute.jit
def scan_max_word(done_p: Int64):
    """Return the maximum of a fused scan, after the wait of basis_cta."""
    return u2f(cute.arch.load(gptr(Uint32, done_p + DONE_MAX), Uint32, sem="acquire", scope="gpu"))


@cute.jit
def store_slot(mx: Float32, tid: Int32, sb: Int32, vmax_p: Int64, slot: Int32):
    """Store the max|v - vbar| of a scan CTA (mx: the thread's part) into V-scan slot `slot`."""
    mx = warp_max(mx)
    red = sb + SM_RED
    if (tid & 31) == 0:
        cute.arch.store(sptr(Float32, red + (tid >> 5) * 4), mx)
    cute.arch.sync_threads()
    if tid == 0:
        c2 = Float32(0.0)
        for wi in cutlass.range_constexpr(EIG_THREADS // 32):
            c2 = cute.arch.fmax(c2, cute.arch.load(sptr(Float32, red + wi * 4), Float32))
        cute.arch.store(gptr(Float32, vmax_p + Int64(slot) * 4), c2)


@cute.jit
def next_tile(word, ctr, tid_h, half):
    """Return the next tile of a V-pass half: thread 0 takes it from the counter at ctr, the half's barrier shows it."""
    if tid_h == 0:
        cute.arch.store(
            sptr(Uint32, word),
            cute.arch.atomic_add(gptr(Uint32, ctr), Uint32(1), sem="relaxed", scope="gpu"),
        )
    cute.arch.barrier(barrier_id=1 + half, number_of_threads=256)
    return Int32(cute.arch.load(sptr(Uint32, word), Uint32))


def key_count(k0, n):
    """The keys of the 16-key group from key k0 that lie below n."""
    return cutlass.max(cutlass.min(n - k0, 16), 0)


def tile_live(tl, S, H, T, lens_p):
    """Whether V tile tl (flat over (batch, head, tile)) holds a valid key: every tile without lens_p."""
    if lens_p is None:
        return True
    bh = tl // T
    return (tl - bh * T) * 128 < valid_len(lens_p, bh // H, S)


def tok_addr(p, bb, row, hh, S, pitch):
    """Global address of the row of token `row` of (batch bb, head hh) of a raw [B, S, H, 128] BF16 tensor at p whose
    tokens are `pitch` bytes apart (256 H for a contiguous tensor, the row of the fused projection for a view)."""
    return p + (Int64(bb) * Int64(S) + Int64(row)) * Int64(pitch) + Int64(hh) * 256


def row_addr(p, bb, row, hh, S, H, pitch):
    """Global address of row `row` of (batch bb, head hh) of a [B, S, H] tensor of `pitch` bytes per row at p."""
    return p + ((Int64(bb) * Int64(S) + Int64(row)) * Int64(H) + Int64(hh)) * pitch


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
            lo_ = bf16_f32(w[e], 0)
            hi_ = bf16_f32(w[e], 1)
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


def stage_rows(v_p, vin, tid_h, tl, S, H, VP, T, lens_p=None):
    """cp.async of the 128 x 256 B rows of V tile tl (flat over (batch, head, tile)) into the shared-memory buffer
    vin by the 256 threads of a V-pass half, thread tid_h eight of the 16-byte parts: row `key` at key * 256, its part p
    at (p ^ (key >> 4)) % 16. The keys past S are zeros, and with lens_p (the global address of the valid prefix
    lengths) those from the length of the batch element on. VP: the bytes between consecutive tokens of V."""
    bh = tl // T
    tt = tl - bh * T
    bb = bh // H
    hh = bh - bb * H
    nk = valid_len(lens_p, bb, S)
    for it in range(8):
        c = it * 256 + tid_h
        key = c >> 4
        part = c & 15
        kg = tt * 128 + key
        src = tok_addr(v_p, bb, cutlass.min(kg, S - 1), hh, S, VP) + Int64(part * 16)
        cp_async16(
            vin + key * 256 + ((part ^ (key >> 4)) & 15) * 16,
            src,
            Int32(Uint32(cutlass.select_(kg < nk, Uint32(16), Uint32(0)))),
        )


@cute.jit
def vscan_tiles(v_p, done_p, S, H, VP, T, ntiles, tid, sb, lens_p, lane_map, fold):
    """Scan the 128-key tiles that the claim counter of the done buffer hands out (flat over (batch, head, tile)) for
    max|v - vbar|, four tiles at a time. lane_map(u): the (16-key group, 8-channel chunk) of thread u of a tile;
    fold(sums, vmx, vmn, cnt, mx): mx raised by the group of the lane (group16), by the mean of the PV format. The
    CTA's maximum and its tile count go to the done buffer (publish_scan)."""
    grp, cc = lane_map(tid & 127)
    sub4 = tid >> 7
    rstride = Int64(VP)
    mx = Float32(0.0)
    scanned = Int32(0)
    t4 = next_scan_tiles(sb, done_p, tid)
    while t4 < ntiles:
        tile = t4 + sub4
        tv = Int32(tile < ntiles)
        tile = cutlass.min(tile, ntiles - 1)
        bh = tile // T
        tt = tile - bh * T
        bb = bh // H
        hh = bh - bb * H
        k0 = tt * 128 + 16 * grp
        nk = valid_len(lens_p, bb, S)
        cnt = key_count(k0, nk) * tv
        base = tok_addr(v_p, bb, cutlass.min(k0, S - 1), hh, S, VP) + Int64(cc * 16)
        # with lengths: a tile without a valid key is skipped
        live = cutlass.const_expr(lens_p is None) or tt * 128 < nk
        if live:
            sums, vmx, vmn = group16(base, rstride, cnt)
            mx = fold(sums, vmx, vmn, cnt, mx)
        scanned = scanned + cutlass.min(Int32(4), ntiles - t4)
        t4 = next_scan_tiles(sb, done_p, tid)
    publish_scan(mx, scanned, tid, sb, done_p)


@cute.jit
def vpass_tiles(v_p, done_p, S, H, VP, T, ntiles, tid, sb, lens_p, encode):
    """Run the tile loop of a V pass in the basis launch: a counter hands out the tiles in descending order to two
    256-thread halves per CTA; thread = (16-key group, 4-channel quad), the 8 groups of a quad in lanes 8 j .. 8 j + 7.
    A tile with a valid key goes through the buffer of its half (stage_rows; the next tile is staged while this one
    is encoded), then encode(wv, sv, tile, bh, tt, k0, cnt, nk, tid_h) of the PV format works on the thread's 16 x 2
    words wv and their four sums sv; k0, cnt: the first key of its group and its keys below nk, the valid keys of the
    batch element."""
    half = tid >> 8
    tid_h = tid & 255
    grp = tid_h & 7
    cq = tid_h >> 3
    vin = sb + half * VIN_BYTES  # next-tile input, rows of 256 B, 16-byte part ^ (row >> 4)
    word = sb + SM_RED + RED_TILE + half * 4
    ctr = done_p + DONE_TILES
    tix = next_tile(word, ctr, tid_h, half)
    if tix < ntiles:
        tl = ntiles - 1 - tix
        if tile_live(tl, S, H, T, lens_p):
            stage_rows(v_p, vin, tid_h, tl, S, H, VP, T, lens_p)
    cute.arch.cp_async_commit_group()
    cute.arch.barrier(barrier_id=1 + half, number_of_threads=256)
    nix = next_tile(word, ctr, tid_h, half)
    while tix < ntiles:
        tile = ntiles - 1 - tix
        bh = tile // T
        tt = tile - bh * T
        k0 = tt * 128 + 16 * grp
        nk = valid_len(lens_p, bh // H, S)
        cnt = key_count(k0, nk)
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier(barrier_id=1 + half, number_of_threads=256)
        live = cutlass.const_expr(lens_p is None) or tt * 128 < nk
        wv = cute.make_rmem_tensor((32,), Uint32)
        sv = cute.make_rmem_tensor((4,), Float32)
        if live:
            sums = [Float32(0.0)] * 4
            for kk in cutlass.range_constexpr(16):
                w = load_vector(
                    sptr(
                        Uint32,
                        vin + (16 * grp + kk) * 256 + (((cq >> 1) ^ grp) * 16) + (cq & 1) * 8,
                        8,
                    ),
                    2,
                )
                for e in cutlass.range_constexpr(2):
                    wv[kk * 2 + e] = w[e]
                    sums[2 * e] = sums[2 * e] + bf16_f32(w[e], 0)
                    sums[2 * e + 1] = sums[2 * e + 1] + bf16_f32(w[e], 1)
            for e in cutlass.range_constexpr(4):
                sv[e] = sums[e]
        cute.arch.barrier(barrier_id=1 + half, number_of_threads=256)
        if nix < ntiles:
            tl = ntiles - 1 - nix
            if tile_live(tl, S, H, T, lens_p):
                stage_rows(v_p, vin, tid_h, tl, S, H, VP, T, lens_p)
        cute.arch.cp_async_commit_group()
        if live:
            encode(wv, sv, tile, bh, tt, k0, cnt, nk, tid_h)
        tix = nix
        nix = next_tile(word, ctr, tid_h, half)


# ------------------------------------------------------------------------------------------------ launch body
@cute.jit
def basis_cta(
    q_p: Int64,
    k_p: Int64,
    had_p: Int64,
    done_p: Int64,
    S: Int32,
    H: Int32,
    QP: Int32,
    KP: Int32,
    T: Int32,
    BH: Int32,
    nspare: Int32,
    eig,
    FUSED: cutlass.Constexpr,
    vscan_gs,
    vscan,
    vpass,
    lens_p,
    rank: cutlass.Constexpr,
    separate: cutlass.Constexpr,
):
    """Run the CTA of a basis launch (grid neig + nspare, EIG_THREADS threads).

    * The CTAs below neig solve one eigenproblem each; the nspare others scan V with the scan of the kernel's PV
      format.
    * FUSED: vscan(ntiles, tid, sb) over the tiles a counter hands out, then the V pass
      vpass(ntiles, pid, tid, sb); otherwise vscan_gs(cta, tid, sb), grid-stride over all of V. sb: the CTA's
      shared memory.
    * eig, the run-time arguments of the eigenproblems: (the addresses of the basis of the Q-residual term and
      of that of the K-residual term, the rows of q and of k, their counts NQ and NK, neig, the weights weight_q and
      weight_k, the rows mode). neig is 0 when the caller gave the basis.
    * QP, KP: the token pitches of q and k; lens_p, rank, separate: as in _eig.
    """
    ut_q_p, ut_k_p, rows_q_p, rows_k_p, NQ, NK, neig, weight_q, weight_k, rows_mode = eig
    tid, _, _ = cute.arch.thread_idx()
    pid, _, _ = cute.arch.block_idx()
    smem = SmemAllocator()
    buf = smem.allocate_array(Uint32, SM_TOTAL // 4, byte_alignment=128)
    sb = Int32(buf.toint())
    ntiles = BH * T
    # eigenproblems, and the V maxima in the other CTAs
    if pid < neig:
        if cutlass.const_expr(max(rank) > 0):
            _eig(
                q_p,
                k_p,
                (rows_q_p, rows_k_p),
                had_p,
                (ut_q_p, ut_k_p),
                pid,
                S,
                H,
                QP,
                KP,
                (NQ, NK),
                (weight_q, weight_k),
                tid,
                sb,
                lens_p,
                rows_mode,
                rank,
                separate,
            )
    elif cutlass.const_expr(not FUSED):
        vscan_gs(pid - neig, tid, sb)
    else:
        vscan(ntiles, tid, sb)
    if cutlass.const_expr(FUSED):
        # The V pass needs the maximum of the whole scan. The scan CTAs claim their tiles from a counter and publish
        # the tiles they finished, so the wait is for tiles, every one of which a running CTA holds: a CTA that is not
        # scheduled yet claims nothing, and no CTA waits on it. An eigensolver CTA joins the V pass when it is done
        # and the scan has finished; nothing ever waits on an eigensolver CTA.
        if tid == 0:
            seen = cute.arch.load(
                gptr(Uint32, done_p + DONE_SCAN), Uint32, sem="acquire", scope="gpu"
            )
            while seen < Uint32(ntiles):
                nanosleep(Int32(200))
                seen = cute.arch.load(
                    gptr(Uint32, done_p + DONE_SCAN), Uint32, sem="acquire", scope="gpu"
                )
            cute.arch.fence_acq_rel_gpu()
        cute.arch.sync_threads()
        vpass(ntiles, pid, tid, sb)
