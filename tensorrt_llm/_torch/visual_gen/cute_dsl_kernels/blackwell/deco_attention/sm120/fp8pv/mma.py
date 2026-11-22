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
"""MMA phases of the sm_120 FP8-PV attention kernel: per warp and phase, O += P V of a key tile on the FP8 MMA, the
means term O += S M on the BF16 MMA, and S = Q' K'^T of the next tile on the block-scaled NVFP4 MMA.

The PV and the means MMA run transposed, so that the S accumulators of the QK, landed and packed by the softmax, are
their B fragments without a shuffle: O^T += V^T P^T takes A = V^T by ldmatrix.trans of the staged V (m16n8k32 E4M3:
16 channels x 8 queries per MMA), and O^T += M^T S^T takes the v_mean rows as A and the 16-key block sums of P as B
(BF16 m16n8k16 over the 16 blocks of a step pair, m16n8k8 for an unpaired last step). A thread holds O^T as 16 x 4
accumulators: O[4 (2 ct + h) + k] for channel tile ct (16 channels) and query n-tile h.

The fragments of a phase are loaded DIST steps ahead of the MMAs that consume them (mma_phase). A tile whose P is zero
in some 32-key quarters skips the V loads and MMAs of those quarters (pv_sparse_quarters), and a step pair with zero P
its means MMAs (means_pair_guarded), both behind warp-uniform indexed branches (brx.idx.uni) that skip the loads and
MMAs."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp
from cutlass.cute.typing import BFloat16, Float8E4M3FN, Float32, Int32, Int64

from ...common.mem import load_invariant, load_vector, mbar_wait, sptr
from ..ptx.fp8pv import ldsm_b8_x2_trans, mma_means_branch, mma_pv_quarters_branch
from ..ptx.shared import ldsm_b16_x4, ldsm_b16_x4_trans, mma_bf16, mma_nvf4
from .layout import K_STEPS, KA_BYTES, QT, SF_BYTES

DIST = 3  # ldmatrix lookahead in MMA steps (tuning constant)


def warp_mma(op, a, b, c):
    """D = A B + C by the warp-level MMA op: a and b the 32-bit fragment words of A and B, c the four f32 of C."""
    atom = cute.make_mma_atom(op)
    frag_a = cute.make_rmem_tensor(
        cute.make_layout((len(a) * 32 // op.ab_dtype.width, 1, 1)), op.ab_dtype
    )
    frag_b = cute.make_rmem_tensor(
        cute.make_layout((len(b) * 32 // op.ab_dtype.width, 1, 1)), op.ab_dtype
    )
    frag_c = cute.make_rmem_tensor(cute.make_layout((4, 1, 1)), Float32)
    words_a = cute.recast_tensor(frag_a, Int32)
    words_b = cute.recast_tensor(frag_b, Int32)
    for i, word in enumerate(a):
        words_a[i] = Int32(word)
    for i, word in enumerate(b):
        words_b[i] = Int32(word)
    for i in range(4):
        frag_c[i] = Float32(c[i])
    cute.gemm(atom, frag_c, frag_a, frag_b, frag_c)
    return [frag_c[i] for i in range(4)]


def mma_fp8(a, b0, b1, c):
    """D = A B + C, m16n8k32 E4M3 x E4M3 with f32 accumulation."""
    return warp_mma(warp.MmaFP8Op(Float8E4M3FN, Float32, (16, 8, 32)), a, [b0, b1], c)


def mma_bf16_k8(a0, a1, b, c):
    """D = A B + C, m16n8k8 BF16 x BF16 with f32 accumulation."""
    return warp_mma(warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 8)), [a0, a1], [b], c)


def mma_phase(
    P,
    O,
    vst,
    kst,
    sfst,
    qa,
    sfq,
    koff_e,
    koff_o,
    sfoff,
    voff,
    dist,
    kwait=None,
    turn=None,
    swait=None,
    vmoff=None,
    means=None,
):
    """One MMA phase of a warp: O += P V of the current tile (if vst is not None: the stage's V), O += S M, the means
    term of a step pair (if means is not None: the pair's v_mean buffer), then S = Q' K'^T of the next tile (if kst is
    not None: the stage's K', its scale atoms at sfst), returned as 16 x 4 accumulators. The fragments are loaded dist
    steps ahead of the MMAs that consume them. swait, kwait = (full barrier, parity): the stage's barrier to wait for
    before the first load, or before the K' scale words; turn = (take, pass, early): the first fragments are loaded
    before take() waits for this warp's turn to issue MMAs, and pass() hands the turn over early steps before the
    end."""
    steps = []
    if vst is not None:
        for i in range(4):
            for cg in range(8):
                steps.append(("pv", i, cg))
    if means is not None:
        # per 16-channel tile 4 hf + j one ldmatrix.x4.trans (the A fragment) and 2 BF16 m16n8k16
        mb = means + vmoff[2]
        for hf in range(2):
            for j in range(4):
                steps.append(("m16", hf, j))
    if kst is not None:
        for a in range(K_STEPS):
            for p in range(8):
                steps.append(("qk", a, p))
    acc = [[Float32(0.0)] * 4 for _ in range(16)]
    n_pv = sum(1 for st in steps if st[0] != "qk")
    w = None
    vrow = vst + voff if vst is not None else None

    def load(st):
        if st[0] == "pv":
            # vrow: the lane's V row address (stage base + row offset + its swizzle chunk); channel group cg flips
            # the chunk bits by XOR, the quarter is an immediate
            return ldsm_b8_x2_trans(vrow, st[2] * 16, st[1] * 32 * 128)
        if st[0] == "m16":
            # channel tile 4 hf + j of both steps of the pair
            return ldsm_b16_x4_trans(mb, st[2] * 32, st[1] * 1024)
        koff = koff_o if st[2] % 2 == 1 else koff_e
        return ldsm_b16_x4(kst + koff + (st[2] // 2) * 32 * 32 + st[1] * KA_BYTES)

    def k_ready():
        # the next tile's K' and scales have landed (its full barrier); then its scale words
        if kwait is not None:
            mbar_wait(kwait[0], kwait[1])
        return [load_vector(sptr(Int32, sfst + a * 512 + sfoff, 16), 4) for a in range(K_STEPS)]

    frags = [None] * len(steps)
    if swait is not None:
        # the stage's full barrier (its V, v_mean and K'), before the first fragment loads and the turn
        mbar_wait(swait[0], swait[1])
    if n_pv == 0 and kst is not None:
        w = k_ready()
    for n in range(dist):
        frags[n] = load(steps[n])
    if turn is not None:
        turn[0]()
    for n, st in enumerate(steps):
        if turn is not None and n == len(steps) - turn[2]:
            turn[1]()
        if n + dist == n_pv and kst is not None:
            w = k_ready()
        if n + dist < len(steps):
            frags[n + dist] = load(steps[n + dist])
        b = frags[n]
        frags[n] = None
        if st[0] == "pv":
            # A = V^T of channels 16 ct .. + 15 (the ldmatrix.trans quad in order), B = P^T of queries 8 h .. + 7
            i, ct = st[1], st[2]
            for h in range(2):
                c = 2 * ct + h
                d = mma_fp8(
                    b,
                    P[4 * i + 2 * h],
                    P[4 * i + 2 * h + 1],
                    [O[4 * c], O[4 * c + 1], O[4 * c + 2], O[4 * c + 3]],
                )
                for k in range(4):
                    O[4 * c + k] = d[k]
        elif st[0] == "m16":
            # A = v_mean^T of channels 16 t .. + 15 (the pair's 16 blocks), B = the pair's BF16 block sums
            hf, j = st[1], st[2]
            for h in range(2):
                cc = 2 * (4 * hf + j) + h
                d = mma_bf16(
                    [O[4 * cc], O[4 * cc + 1], O[4 * cc + 2], O[4 * cc + 3]],
                    b,
                    [P[18 + h], P[16 + h]],
                )
                for k in range(4):
                    O[4 * cc + k] = d[k]
        else:
            a, p = st[1], st[2]
            for h in range(2):
                nt = 2 * p + h
                acc[nt] = mma_nvf4(
                    acc[nt], qa[a], [b[2 * h], b[2 * h + 1]], sfq[a], w[a][nt // 4], nt % 4
                )
    return acc


@cute.jit
def pv_sparse_quarters(P, O, vst, voff, mask):
    """PV of the 32-key quarters whose bit is set in mask (the tiles with zero and nonzero quarters)."""
    vrow = vst + voff
    vaddrs = [vrow ^ (ct * 16) for ct in range(8)]
    out = mma_pv_quarters_branch([O[j] for j in range(64)], [P[j] for j in range(16)], vaddrs, mask)
    for j in cutlass.range_constexpr(64):
        O[j] = out[j]


def means_pair_body(P, O, vmb, vmoff):
    """The means term of a step pair: O += M^T S^T as 16 BF16 m16n8k16 (channel tiles x query n-tiles), A = the v_mean
    rows of both steps (pair buffer vmb: slot 0 the first step, K slots 0-7; slot 1 the second, K slots 8-15) by one
    ldmatrix.x4.trans per channel tile, B = their block sums (P[18 + h]: first step, P[16 + h]: second)."""
    mb = vmb + vmoff[2]
    frags = [ldsm_b16_x4_trans(mb, (t & 3) * 32, (t >> 2) * 1024) for t in range(2)]
    for t in range(8):
        if t + 2 < 8:
            frags.append(ldsm_b16_x4_trans(mb, ((t + 2) & 3) * 32, ((t + 2) >> 2) * 1024))
        for h in range(2):
            c = 2 * t + h
            d = mma_bf16(
                [O[4 * c], O[4 * c + 1], O[4 * c + 2], O[4 * c + 3]],
                frags[t],
                [P[18 + h], P[16 + h]],
            )
            for k in range(4):
                O[4 * c + k] = d[k]


def means_pair_guarded(cond, P, O, vmb, vmoff):
    """The means MMAs of a step pair (block sums P[18..19] first, P[16..17] second) behind one indexed branch, taken
    when cond != 0."""
    out = mma_means_branch(
        [O[j] for j in range(64)], [P[18], P[16], P[19], P[17]], vmb + vmoff[2], cond
    )
    for j in range(64):
        O[j] = out[j]


@cute.jit
def means_pair(cond, P, O, vmb, vmoff):
    """means_pair_body when cond != 0 (an even phase whose step pair has a nonzero P, or a non-finite V / v_mean)."""
    if cond != 0:
        means_pair_body(P, O, vmb, vmoff)


@cute.jit
def means_single(cond, P, O, vmb, vmoff):
    """The means term of the unpaired last step (its block sums in P[16 + h], v_mean in slot 0 of vmb)."""
    if cond != 0:
        means_only(P, O, vmb, vmoff)


@cute.jit
def means_last(pair, pvl, nzp, P, O, vmb, stb, vmoff):
    """The means MMAs of the last phase: of the last step pair (pair != 0) or of the unpaired last step (its v_mean
    rows in the stage's K' region stb). pvl, nzp: nonzero P (or non-finite V / v_mean) of the last two steps."""
    if pair != 0:
        means_pair(pvl | nzp, P, O, vmb, vmoff)
    else:
        means_single(pvl, P, O, stb, vmoff)


def means_only(P, O, vmst, vmoff):
    """The means term of a tile alone (the unpaired last step): 4 ldmatrix.x4.trans of v_mean, 16 BF16 m16n8k8."""
    frags = [ldsm_b16_x4_trans(vmst + vmoff[cp & 1] + (cp >> 1) * 1024) for cp in range(2)]
    for cp in range(4):
        if cp + 2 < 4:
            frags.append(ldsm_b16_x4_trans(vmst + vmoff[(cp + 2) & 1] + ((cp + 2) >> 1) * 1024))
        b = frags[cp]
        for half in range(2):
            for h in range(2):
                c = 2 * (2 * cp + half) + h
                d = mma_bf16_k8(
                    b[2 * half],
                    b[2 * half + 1],
                    P[16 + h],
                    [O[4 * c], O[4 * c + 1], O[4 * c + 2], O[4 * c + 3]],
                )
                for k in range(4):
                    O[4 * c + k] = d[k]


def store64(A, acc):
    """The 64 QK accumulator words into the rmem tensor A."""
    for i in range(64):
        A[i] = acc[i // 4][i % 4]


@cute.jit
def load_q(
    qb,
    qsf_ptr,
    qt,
    warp,
    lane,
    g,
    q,
    bb,
    hh,
    S: cutlass.Constexpr,
    H: cutlass.Constexpr,
    nT: cutlass.Constexpr,
):
    """The Q' A fragments of the warp's rows g, g + 8 in query tile qt (clamped to the last query) and their scale
    words, loaded once."""
    r0 = warp * 16 + g
    qi0 = qt * QT + r0
    qi1 = qi0 + 8
    if qi0 > S - 1:
        qi0 = Int32(S - 1)
    if qi1 > S - 1:
        qi1 = Int32(S - 1)
    qrow0 = qb + Int64((bb * S + qi0) * H + hh) * 128 + q * 4
    qrow1 = qb + Int64((bb * S + qi1) * H + hh) * 128 + q * 4
    qa = []
    for a in cutlass.range_constexpr(K_STEPS):
        qa.append(
            [
                load_invariant(Int32, qrow0 + a * 32),
                load_invariant(Int32, qrow1 + a * 32),
                load_invariant(Int32, qrow0 + a * 32 + 16),
                load_invariant(Int32, qrow1 + a * 32 + 16),
            ]
        )
    rs = r0 + (lane & 1) * 8
    qsf_addr = (
        qsf_ptr.toint()
        + Int64((bb * H + hh) * nT + qt) * SF_BYTES
        + (rs % 32) * 16
        + (rs // 32) * 4
    )
    return qa, [load_invariant(Int32, qsf_addr + a * 512) for a in range(K_STEPS)]
