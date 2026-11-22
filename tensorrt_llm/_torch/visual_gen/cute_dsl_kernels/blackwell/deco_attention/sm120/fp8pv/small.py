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
"""The sm_120 FP8-PV attention for fewer than 128 keys: one key tile, one CTA per (batch, head).

Two groups of 4 warps (16 query rows per warp) take turns in issuing MMAs under the named barriers 1 and 2: the
QK of the tile, its softmax, then the PV with the means term, a group running its softmax while the other
issues its MMAs. The second group copies K', V, the K' scale atoms and the tile's 8 v_mean rows into shared
memory by cp.async (keys past S zero-filled) and arrives on the tile's mbarrier when the copies have landed. The key
order, the transposed PV and means MMAs and the output rows are those of the main kernel (layout.py, mma.py,
epilogue.py). With one tile the anchor of a row is its tile max, so the landing needs no lazy rescale:
expcast_transpose4 (../ptx/fp8pv.py) forms c s - anchor in one FMA and casts it on the bit grid of FP32."""

import math

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32, Int64

from ...common.dsl import barrier_cta_sync
from ...common.mem import cp_async16, load_invariant, load_vector, mbar_wait, sptr
from ..ptx.fp8pv import block_sums_of_p, expcast_transpose4, ldsm_b8_x2_trans
from ..ptx.shared import ldsm_b16_x4, ldsm_b16_x4_trans, mma_nvf4
from ..valid_len import valid_len
from .epilogue import inv_row_sums, store_rows
from .layout import EP, K_STEPS, KT, NTHR, SF_BYTES, V_BYTES, VM_BYTES, key_grp, key_rot
from .mma import DIST, load_q, mma_bf16_k8, mma_fp8
from .softmax import tile_max

# Shared memory of the one key tile: K' at a 112-byte row pitch (96 used bytes; 7 x 16 keeps ldmatrix conflict-free),
# V (token-major, 16-byte chunks swizzled by key % 8), the K' scale atoms, the tile's 8 v_mean rows (128-byte
# halves, chunks swizzled by row), the staging rows of the output and the tile's mbarrier
KP = 112
SK_BYTES = KT * KP
SV_OFF = SK_BYTES
SSF_OFF = SV_OFF + V_BYTES
SVM_OFF = SSF_OFF + SF_BYTES
STILE_BYTES = SVM_OFF + VM_BYTES
SEPI_OFF = STILE_BYTES
SBAR_OFF = SEPI_OFF + 8 * 16 * EP
SMALL_SMEM_BYTES = SBAR_OFF + 8


# ------------------------------------------------------------------------------------------------ tile copy, MMA phases
@cute.jit
def small_issue_tile(
    sbase, kb, vb, kvsrc, kvsrc0, ksfb, t, S: cutlass.Constexpr, H: cutlass.Constexpr, vmb
):
    """Thread t of the second group: cp.async of the key tile into shared memory at sbase as one commit group, 16
    rows per instruction step: thread t copies chunk t % 8 of rows 16 it + t / 8 (K' rows: their used chunks only),
    so each warp instruction covers four whole 128-byte rows; then the K' scale chunk t (32 per atom) and the 16-byte
    chunk t of the tile's 8 v_mean rows (at vmb), after the scale atoms. Keys >= S are zero-filled."""
    c = t & 7
    r16 = t >> 3
    for it in cutlass.range_constexpr(KT // 16):
        row = it * 16 + r16
        nb = Int32(16)
        src = kvsrc + Int64(it * 16) * (H * 128)
        if row >= S:
            nb = Int32(0)
            src = kvsrc0
        cp_async16(sbase + SV_OFF + row * 128 + ((c ^ (r16 & 7)) * 16), vb + src, nb)
        if c < 2 * K_STEPS:
            cp_async16(sbase + row * KP + c * 16, kb + src, nb)
    if t < 32 * K_STEPS:
        cp_async16(sbase + SSF_OFF + t * 16, ksfb + t * 16, Int32(16))
    # v_mean row t / 16, bytes 16 (t % 16): half (t / 8) % 2, chunk t % 8 swizzled by the row
    mr = t >> 4
    cp_async16(
        sbase + SVM_OFF + ((t >> 3) & 1) * 1024 + mr * 128 + ((c ^ mr) * 16),
        vmb + mr * 256 + (t & 15) * 16,
        Int32(16),
    )
    cute.arch.cp_async_commit_group()


def small_mma_phase(
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
    vsw,
    kwait=None,
    vmst=None,
    vmoff=None,
):
    """One MMA phase of a warp: O += P V of the tile (if vst is not None), then the means term O += S M (if vmst is
    not None: the tile's v_mean rows at vmst, its block sums in P[16..17]), then S = Q' K'^T (if kst is not None).
    The fragments are loaded DIST steps ahead of the MMAs that consume them (each step is one ldmatrix and two MMAs, a
    means step four)."""
    steps = []
    if vst is not None:
        for i in range(4):
            for cg in range(8):
                steps.append(("pv", i, cg))
    if vmst is not None:
        for cp in range(4):
            steps.append(("vm", cp))
    if kst is not None:
        for a in range(K_STEPS):
            for p in range(8):
                steps.append(("qk", a, p))
    acc = [[Float32(0.0)] * 4 for _ in range(16)]
    n_pv = sum(1 for st in steps if st[0] != "qk")
    w = None

    def load(st):
        if st[0] == "pv":
            return ldsm_b8_x2_trans(vst + st[1] * 32 * 128 + voff + ((st[2] * 16) ^ vsw))
        if st[0] == "vm":
            return ldsm_b16_x4_trans(vmst + vmoff[st[1] & 1] + (st[1] >> 1) * 1024)
        koff = koff_o if st[2] % 2 == 1 else koff_e
        return ldsm_b16_x4(kst + koff + (st[2] // 2) * 32 * KP + st[1] * 32)

    def k_ready():
        # the next tile's K' and scales have landed (its full barrier); then its scale words
        if kwait is not None:
            mbar_wait(kwait[0], kwait[1])
        return [load_vector(sptr(Int32, sfst + a * 512 + sfoff, 16), 4) for a in range(K_STEPS)]

    frags = [None] * len(steps)
    if n_pv == 0 and kst is not None:
        w = k_ready()
    for n in range(DIST):
        frags[n] = load(steps[n])
    for n, st in enumerate(steps):
        if n + DIST == n_pv and kst is not None:
            w = k_ready()
        if n + DIST < len(steps):
            frags[n + DIST] = load(steps[n + DIST])
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
        elif st[0] == "vm":
            # A = v_mean^T of channels 32 cp .. + 31 (two m16n8k8 A fragments), B = the block sums of query n-tile h
            cp = st[1]
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
        else:
            a, p = st[1], st[2]
            for h in range(2):
                nt = 2 * p + h
                acc[nt] = mma_nvf4(
                    acc[nt], qa[a], [b[2 * h], b[2 * h + 1]], sfq[a], w[a][nt // 4], nt % 4
                )
    return acc


@cute.jit
def small_softmax(acc, P, sc, q, lim: cutlass.Constexpr):
    """Softmax of the single tile (rows g and g + 8 of the warp): tile max as the anchor (keys >= lim masked, lim <
    KT, or a run-time count with lengths), P = E4M3(2^(s - anchor + 4)) into P[4 i + r], the row sums of the
    quantized P and its block sums (P[16], P[17])."""
    for nt in cutlass.range_constexpr(16):
        s = nt % 4
        i = nt // 4
        for e in cutlass.range_constexpr(2):
            kk = 32 * i + 8 * key_grp(s) + ((2 * q + e) ^ key_rot(s))
            ok = Int32(1)
            if cutlass.const_expr(not isinstance(lim, int)):
                ok = Int32(kk < lim)
            elif kk >= lim:
                ok = Int32(0)
            acc[nt][e] = Float32(cutlass.select_(ok != 0, acc[nt][e], Float32(-math.inf)))
            acc[nt][2 + e] = Float32(cutlass.select_(ok != 0, acc[nt][2 + e], Float32(-math.inf)))
    # the anchors of the rows: the tile max in log2 units
    na0, na1 = tile_max(acc, sc)
    s0 = Float32(0.0)
    s1 = Float32(0.0)
    for i in cutlass.range_constexpr(4):
        packed, s0, s1 = expcast_transpose4(acc, 4 * i, sc, na0, na1, s0, s1)
        for e in cutlass.range_constexpr(4):
            P[4 * i + e] = packed[e]
    sb = block_sums_of_p(P, q)
    P[16] = sb[0]
    P[17] = sb[1]
    return s0, s1


# ------------------------------------------------------------------------------------------------ kernel
@cute.kernel
def small_kernel(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    qsf_ptr: cute.Pointer,
    ksf_ptr: cute.Pointer,
    ss_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    vs_ptr: cute.Pointer,
    vm_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    S: cutlass.Constexpr,
    H: cutlass.Constexpr,
    lens_ptr,
):
    """Grid (1, H, B), NTHR threads: the S < 128 queries of one (batch, head) against its S keys. ss_ptr, vs_ptr: the
    score scale and v8_scale. lens_ptr: None, or the valid prefix lengths: the keys from the length of the batch
    element on are masked and its query rows from there on are zero in o."""
    tidx_, _, _ = cute.arch.thread_idx()
    _, hh_, bb_ = cute.arch.block_idx()
    tidx = Int32(tidx_)
    hh = Int32(hh_)
    bb = Int32(bb_)
    warp = tidx >> 5
    lane = tidx & 31
    g = lane >> 2
    q = lane & 3
    grp = warp >> 2
    bid_wait = grp + 1
    bid_pass = 2 - grp
    smem = Int32(cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024).toint())
    fbar = smem + SBAR_OFF

    qb = q_ptr.toint()
    ob = o_ptr.toint()
    t = tidx & 127
    kvsrc0 = Int64((bb * S) * H + hh) * 128 + (t & 7) * 16
    kvsrc = Int64((bb * S + (t >> 3)) * H + hh) * 128 + (t & 7) * 16
    ksfb = ksf_ptr.toint() + Int64(bb * H + hh) * SF_BYTES
    vmb = vm_ptr.toint() + Int64(bb * H + hh) * (8 * 256)

    if tidx == 0:
        cute.arch.mbarrier_init(sptr(Int64, fbar), 128)
        cute.arch.mbarrier_init_fence()
    barrier_cta_sync(Int32(0), thread_count=NTHR)
    if grp == 1:
        small_issue_tile(smem, k_ptr.toint(), v_ptr.toint(), kvsrc, kvsrc0, ksfb, t, S, H, vmb)
        cute.arch.cp_async_wait_group(0)
        cute.arch.mbarrier_arrive(sptr(Int64, fbar))
        # the first group takes the first turn
        cute.arch.barrier_arrive(barrier_id=Int32(1), number_of_threads=NTHR, aligned=False)

    qa, sfq = load_q(qb, qsf_ptr, Int32(0), warp, lane, g, q, bb, hh, S, H, 1)
    sc = load_invariant(Float32, ss_ptr.toint())
    vsc = load_invariant(Float32, vs_ptr.toint())

    # per-lane offsets of the fragment loads, as layout.lane_offsets with this tile's K' pitch and V chunks
    m = lane >> 3
    gg = lane & 7
    s0 = m >> 1
    koff_e = (16 * s0 + (gg ^ key_rot(s0))) * KP + (m & 1) * 16
    koff_o = (16 * s0 + 8 + (gg ^ key_rot(s0 + 2))) * KP + (m & 1) * 16
    sfoff = (8 * key_grp(q) + (g ^ key_rot(q))) * 16
    ts = lane & 3
    vrow = 8 * key_grp(ts) + ((2 * ((lane & 15) >> 2) + (lane >> 4)) ^ key_rot(ts))
    vsw = (vrow & 7) * 16
    voff = vrow * 128
    vr = lane & 7
    vm0 = vr * 128 + ((m ^ vr) * 16)
    vmoff = [vm0, vm0 ^ 64]

    O = cute.make_rmem_tensor(cute.make_layout(64), Float32)
    O.fill(0.0)
    P = cute.make_rmem_tensor(cute.make_layout(18), Int32)

    # ---- phase 0: QK
    barrier_cta_sync(bid_wait, thread_count=NTHR)
    acc = small_mma_phase(
        P,
        O,
        None,
        smem,
        smem + SSF_OFF,
        qa,
        sfq,
        koff_e,
        koff_o,
        sfoff,
        voff,
        vsw,
        (fbar, Int32(0)),
    )
    cute.arch.barrier_arrive(barrier_id=bid_pass, number_of_threads=NTHR, aligned=False)
    nv = None if cutlass.const_expr(lens_ptr is None) else valid_len(lens_ptr.toint(), bb, S)
    l0, l1 = small_softmax(
        [list(r) for r in acc], P, sc, q, S if cutlass.const_expr(nv is None) else nv
    )
    # ---- phase 1: PV and its means term
    barrier_cta_sync(bid_wait, thread_count=NTHR)
    small_mma_phase(
        P,
        O,
        smem + SV_OFF,
        None,
        None,
        qa,
        sfq,
        koff_e,
        koff_o,
        sfoff,
        voff,
        vsw,
        vmst=smem + SVM_OFF,
        vmoff=vmoff,
    )
    if grp == 0:
        cute.arch.barrier_arrive(barrier_id=bid_pass, number_of_threads=NTHR, aligned=False)

    # ---- epilogue: o = O * v8_scale / l through the warp's own staging rows
    f = inv_row_sums(l0, l1, q)
    store_rows(
        O,
        f,
        vsc,
        smem + SEPI_OFF + warp * (16 * EP),
        g,
        q,
        lane,
        warp,
        Int32(0),
        ob,
        bb,
        hh,
        S,
        H,
        nv,
    )
