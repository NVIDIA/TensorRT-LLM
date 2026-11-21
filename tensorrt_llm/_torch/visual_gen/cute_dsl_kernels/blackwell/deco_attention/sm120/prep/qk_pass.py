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
"""Q'/K' pass of the sm_120 preparation: per 16-token warp tile, the Hadamard rotation, NVFP4 of the 8 body blocks,
the projections of q1 (the dequantized body) and dq (its quantization residual) on the basis U (their A fragments
come from the accumulators of the rotation), NVFP4 of the 4 compensation blocks; codes through a per-warp smem stage,
provisional (unencoded) block scales as BF16 in the final atom layout, per-CTA group maxima (the encode launch turns
them into the group scales and the UE4M3 bytes).

A fragments come straight from 8-byte global loads: k-step a (16 channels 16a..16a+15) maps logical k = 2t + e to
channel 16a + 4t + e and k = 2t + 8 + e to channel 16a + 4t + 2 + e, for both the general rotation (B fragments of
Hd from smem) and the fast path. Fast path (Hd verified on device to be diag(s) x Sylvester H128 = H8 (x) H16): one
MMA per 16-channel block against diag(s_a) H16 (= Hd[16a.., 0..15]), then an 8-point FWHT across the blocks in
registers (every thread holds the same in-block positions of all eight blocks)."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint16, Uint32
from cutlass.utils import SmemAllocator

from ...common.expmath import bf16x2, f2u, round_m3, warp_max
from ...common.mem import gptr, load_invariant, load_vector, sptr, store_vector
from ...constants import QPRE
from ..ptx.shared import (
    e2m1x2_to_f32x2,
    f32x4_to_e2m1x4,
    ld_global_nc_b32_hint_pred,
    mma_bf16,
    st_global_b32_v2_hint,
)
from ..valid_len import valid_len

QK_THREADS = 256

# shared memory map (bytes): Hd^T, U^T, the per-warp code and scale stages, the reduction area
SQ_HDT = 0  # 128 rows x 256 B, 8-byte slots, slot ^ 4 (row & 3)
SQ_UTS = 32768  # 32 rows x 272 B
SQ_STG = SQ_UTS + 8704  # per warp: codes 16 x 144 B, scales 16 x 24 B
STG_W = 16 * 144 + 16 * 24  # 2688
SQ_RED = SQ_STG + 8 * STG_W  # 64 floats + flag
SQ_TOTAL = SQ_RED + 512


@cute.jit
def _quant_block(x0: Float32, x1: Float32, x2: Float32, x3: Float32):
    """One 16-channel NVFP4 block spread over a quad (4 values per thread).
    Returns (code word: low byte = (x1,x0), high byte = (x3,x2); scale; dequantized x0..x3)."""
    a0, a1, a2, a3 = (cute.math.abs(x) for x in (x0, x1, x2, x3))
    am = cute.arch.fmax(cute.arch.fmax(a0, a1), cute.arch.fmax(a2, a3))
    am = cute.arch.fmax(Float32(0.0), am)
    am = cute.arch.fmax(am, cute.arch.shuffle_sync_bfly(am, offset=1))
    am = cute.arch.fmax(am, cute.arch.shuffle_sync_bfly(am, offset=2))
    sc = round_m3(am * Float32(1.0 / 6.0))
    inv = Float32(cutlass.select_(sc > Float32(1e-30), cute.arch.rcp_approx(sc), Float32(0.0)))
    cw = f32x4_to_e2m1x4(x0 * inv, x1 * inv, x2 * inv, x3 * inv)
    d0, d1 = e2m1x2_to_f32x2(cw & Uint32(0xFF))
    d2, d3 = e2m1x2_to_f32x2(cw >> Uint32(8))
    return cw, sc, d0 * sc, d1 * sc, d2 * sc, d3 * sc


@cute.jit
def _emit(
    stg: Int32,
    sst: Int32,
    g: Int32,
    t: Int32,
    blk: Int32,
    cw0: Uint32,
    cw1: Uint32,
    sc0: Float32,
    sc1: Float32,
):
    """codes of rows g / g+8 for block blk (bytes 8 blk + t, 8 blk + 4 + t), scales by thread blk % 4."""
    cute.arch.store(sptr(Uint8, stg + g * 144 + 8 * blk + t), cw0.to(Uint8))
    cute.arch.store(sptr(Uint8, stg + g * 144 + 8 * blk + 4 + t), (cw0 >> Uint32(8)).to(Uint8))
    cute.arch.store(sptr(Uint8, stg + (g + 8) * 144 + 8 * blk + t), cw1.to(Uint8))
    cute.arch.store(
        sptr(Uint8, stg + (g + 8) * 144 + 8 * blk + 4 + t), (cw1 >> Uint32(8)).to(Uint8)
    )
    if t == (blk & 3):
        cute.arch.store(sptr(Uint16, sst + (g * 12 + blk) * 2), (f2u(sc0) >> Uint32(16)).to(Uint16))
        cute.arch.store(
            sptr(Uint16, sst + ((g + 8) * 12 + blk) * 2), (f2u(sc1) >> Uint32(16)).to(Uint16)
        )


def _body_block(bo, x, pq, pd, uts, stg, sst, g, t):
    """Trace-time helper: quantize body block bo (x[0..7]: n-tile 0 c0..c3, n-tile 1 c0..c3), emit it, and
    accumulate its projections q1 U -> pq, dq U -> pd. Returns the larger of its two block scales."""
    cw0, sc0, d00, d01, d02, d03 = _quant_block(x[0], x[1], x[4], x[5])
    cw1, sc1, d10, d11, d12, d13 = _quant_block(x[2], x[3], x[6], x[7])
    _emit(stg, sst, g, t, Int32(bo), cw0, cw1, sc0, sc1)
    aq = [bf16x2(d00, d01), bf16x2(d10, d11), bf16x2(d02, d03), bf16x2(d12, d13)]
    ad = [
        bf16x2(x[0] - d00, x[1] - d01),
        bf16x2(x[2] - d10, x[3] - d11),
        bf16x2(x[4] - d02, x[5] - d03),
        bf16x2(x[6] - d12, x[7] - d13),
    ]
    for n in range(4):
        ub = uts + (8 * n + g) * 272 + (16 * bo + 2 * t) * 2
        bv = [
            cute.arch.load(sptr(Uint32, ub), Uint32),
            cute.arch.load(sptr(Uint32, ub + 16), Uint32),
        ]
        r1 = mma_bf16([pq[4 * n], pq[4 * n + 1], pq[4 * n + 2], pq[4 * n + 3]], aq, bv)
        r2 = mma_bf16([pd[4 * n], pd[4 * n + 1], pd[4 * n + 2], pd[4 * n + 3]], ad, bv)
        for e in range(4):
            pq[4 * n + e] = r1[e]
            pd[4 * n + e] = r2[e]
    return cute.arch.fmax(sc0, sc1)


@cute.kernel
def qk_kernel(
    q_p: Int64,
    k_p: Int64,
    had_p: Int64,
    ut_p: Int64,
    q4_p: Int64,
    k4_p: Int64,
    prov_p: Int64,
    gmax_p: Int64,
    B: Int32,
    S: Int32,
    H: Int32,
    T: Int32,
    nitems: Int32,
    ncta: Int32,
    lens_p,
):
    """The Q'/K' pass of the CTA's share of the nitems items (side, batch, head, tile). lens_p: None, or the global
    address of the valid prefix lengths: the tokens from the length of their batch element on are then read as zeros
    (zero codes and block scales, outside the group maxima), and a tile without a valid token is left as it is."""
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    lane = tid & 31
    warp = tid >> 5
    g = lane >> 2
    t = lane & 3
    smem = SmemAllocator()
    buf = smem.allocate_array(Uint32, SQ_TOTAL // 4, byte_alignment=128)
    sb = Int32(buf.toint())
    hdt = sb + SQ_HDT
    uts = sb + SQ_UTS
    stg = sb + SQ_STG + warp * STG_W
    sst = stg + 16 * 144
    red = sb + SQ_RED
    flag = red + 256

    if tid == 0:
        cute.arch.store(sptr(Uint32, flag), Uint32(1))
    # Hd^T into smem: row j (output channel), 8-byte slot c/4 of input channels (c = input channel), slot ^ 4 (j & 3);
    # check Hd == diag(s) x Sylvester (s = column 0)
    ok = Uint32(1)
    for it in cutlass.range_constexpr(8):
        idx = it * QK_THREADS + tid  # 2048 chunks of 8 output channels
        c = idx >> 4
        j0 = (idx & 15) * 8
        w = load_invariant(Uint32, had_p + Int64(c * 128 + j0) * 2, 4)
        s0 = Uint32(load_invariant(Uint16, had_p + Int64(c * 128) * 2))
        ok = ok & Uint32((s0 == Uint32(0x3F80)) | (s0 == Uint32(0xBF80)))
        for e in cutlass.range_constexpr(8):
            j = j0 + e
            slot = (c >> 2) ^ ((j & 3) << 2)
            hv = (w[e // 2] if e % 2 == 0 else w[e // 2] >> Uint32(16)) & Uint32(0xFFFF)
            cute.arch.store(sptr(Uint16, hdt + j * 256 + slot * 8 + (c & 3) * 2), hv.to(Uint16))
            par = Uint32(cute.arch.popc(Uint32(c & j)) & 1)
            ok = ok & Uint32(hv == (s0 ^ (par << Uint32(15))))
    # zero the code stage (bytes 96..127 of every row stay zero)
    zz = Uint32(0)
    for it in cutlass.range_constexpr(5):
        off = it * 512 + lane * 16
        if off < 16 * 144:
            store_vector(sptr(Uint32, stg + off, 16), (zz, zz, zz, zz))
    cute.arch.sync_threads()
    if ok == Uint32(0):
        cute.arch.store(sptr(Uint32, flag), Uint32(0))
    cute.arch.sync_threads()
    fast = Int32(cute.arch.load(sptr(Uint32, flag), Uint32))
    # fast-path B fragments: block a, n-tile n -> Hd[16a + 4t .. +3][8n + g]
    hb = cute.make_rmem_tensor((32,), Uint32)
    for a in cutlass.range_constexpr(8):
        for n in cutlass.range_constexpr(2):
            row = 8 * n + g
            bv = load_vector(
                sptr(Uint32, hdt + row * 256 + ((4 * a + t) ^ ((row & 3) << 2)) * 8, 8), 2
            )
            hb[a * 4 + n * 2] = bv[0]
            hb[a * 4 + n * 2 + 1] = bv[1]

    gm = cute.make_rmem_tensor((6,), Float32)
    gm.fill(0.0)
    item_lo = (cta * nitems) // ncta
    item_hi = ((cta + 1) * nitems) // ncta
    cur = Int32(-1)
    rstride = Int64(H) * 256
    BHn = B * H
    af = cute.make_rmem_tensor((32,), Uint32)
    afn = cute.make_rmem_tensor((32,), Uint32)
    # af[4a + 0/1] = row g channels 16a + 4t + 0..1 / 2..3, af[4a + 2/3] = row g + 8
    if item_lo < item_hi:
        bhs0 = item_lo // T
        tile0 = item_lo - bhs0 * T
        side0 = bhs0 // BHn
        bh0 = bhs0 - side0 * BHn
        bb0 = bh0 // H
        hh0 = bh0 - bb0 * H
        rg0 = tile0 * 128 + warp * 16 + g
        n0 = S if cutlass.const_expr(lens_p is None) else valid_len(lens_p, bb0, S)
        bg0 = (
            Int64(cutlass.select_(side0 != 0, k_p, q_p))
            + ((Int64(bb0) * Int64(S) + Int64(rg0)) * Int64(H) + Int64(hh0)) * 256
            + t * 8
        )
        for a in cutlass.range_constexpr(8):
            wa = ld_global_nc_b32_hint_pred(bg0 + a * 32, Int32(rg0 < n0), "evict_first", 2)
            wb = ld_global_nc_b32_hint_pred(
                bg0 + rstride * 8 + a * 32, Int32(rg0 + 8 < n0), "evict_first", 2
            )
            afn[4 * a] = wa[0]
            afn[4 * a + 1] = wa[1]
            afn[4 * a + 2] = wb[0]
            afn[4 * a + 3] = wb[1]
    for item in cutlass.range(item_lo, item_hi, unroll=1):
        for i in cutlass.range_constexpr(32):
            af[i] = afn[i]
        bhs = item // T
        tile = item - bhs * T
        side = bhs // BHn
        bh = bhs - side * BHn
        bb = bh // H
        hh = bh - bb * H
        # prefetch the next item's rows (zeros past the range / past S)
        nitem = item + 1
        nbhs = nitem // T
        ntile = nitem - nbhs * T
        nside = nbhs // BHn
        nbh = nbhs - nside * BHn
        nbb = nbh // H
        nhh = nbh - nbb * H
        nrg = ntile * 128 + warp * 16 + g
        nok = Int32(nitem < item_hi)
        nn = (
            S
            if cutlass.const_expr(lens_p is None)
            else valid_len(lens_p, cutlass.min(nbb, B - 1), S)
        )
        nbg = (
            Int64(cutlass.select_(nside != 0, k_p, q_p))
            + ((Int64(nbb) * Int64(S) + Int64(nrg)) * Int64(H) + Int64(nhh)) * 256
            + t * 8
        )
        for a in cutlass.range_constexpr(8):
            wa = ld_global_nc_b32_hint_pred(nbg + a * 32, nok & Int32(nrg < nn), "evict_first", 2)
            wb = ld_global_nc_b32_hint_pred(
                nbg + rstride * 8 + a * 32, nok & Int32(nrg + 8 < nn), "evict_first", 2
            )
            afn[4 * a] = wa[0]
            afn[4 * a + 1] = wa[1]
            afn[4 * a + 2] = wb[0]
            afn[4 * a + 3] = wb[1]
        # with lengths: a tile whose tokens are all past the length of its batch element is skipped
        live = cutlass.const_expr(lens_p is None) or tile * 128 < valid_len(lens_p, bb, S)
        if live:
            if bhs != cur:
                cute.arch.sync_threads()
                for it in cutlass.range_constexpr(2):
                    idx = it * QK_THREADS + tid  # 512 chunks of 16 B
                    r = idx >> 4
                    cc = idx & 15
                    w = load_invariant(
                        Uint32, ut_p + (Int64(bh) * 4096 + Int64(r * 128 + cc * 8)) * 2, 4
                    )
                    store_vector(
                        sptr(Uint32, uts + r * 272 + cc * 16, 16), (w[0], w[1], w[2], w[3])
                    )
                cute.arch.sync_threads()
                cur = bhs
            dst = Int64(cutlass.select_(side != 0, k4_p, q4_p))
            pre = Float32(cutlass.select_(side != 0, Float32(1.0), Float32(QPRE)))
            tok0 = tile * 128 + warp * 16
            pq = cute.make_rmem_tensor((16,), Float32)
            pd = cute.make_rmem_tensor((16,), Float32)
            pq.fill(0.0)
            pd.fill(0.0)
            gb = Float32(0.0)
            if fast != 0:
                # stage 1: z_a = x_a (diag(s_a) H16), 16-channel blocks
                yv = cute.make_rmem_tensor((64,), Float32)
                for a in cutlass.range_constexpr(8):
                    aa = [af[4 * a], af[4 * a + 2], af[4 * a + 1], af[4 * a + 3]]
                    for n in cutlass.range_constexpr(2):
                        r = mma_bf16(
                            [Float32(0.0)] * 4, aa, [hb[a * 4 + n * 2], hb[a * 4 + n * 2 + 1]]
                        )
                        for e in cutlass.range_constexpr(4):
                            yv[a * 8 + n * 4 + e] = r[e]
                # stage 2: 8-point FWHT across the blocks, per in-block position
                for pos in cutlass.range_constexpr(8):
                    for hstep in (1, 2, 4):
                        for a in cutlass.range_constexpr(8):
                            if (a & hstep) == 0:
                                u0 = yv[a * 8 + pos]
                                u1 = yv[(a + hstep) * 8 + pos]
                                yv[a * 8 + pos] = u0 + u1
                                yv[(a + hstep) * 8 + pos] = u0 - u1
                for bo in cutlass.range_constexpr(8):
                    x = [yv[bo * 8 + i] * pre for i in range(8)]
                    gb = cute.arch.fmax(gb, _body_block(bo, x, pq, pd, uts, stg, sst, g, t))
            else:
                for bo in cutlass.range_constexpr(8):
                    acc = [Float32(0.0)] * 8
                    for s in cutlass.range_constexpr(8):
                        aa = [af[4 * s], af[4 * s + 2], af[4 * s + 1], af[4 * s + 3]]
                        for n in cutlass.range_constexpr(2):
                            row = 8 * (2 * bo + n) + g
                            bv = load_vector(
                                sptr(
                                    Uint32,
                                    hdt + row * 256 + ((4 * s + t) ^ ((row & 3) << 2)) * 8,
                                    8,
                                ),
                                2,
                            )
                            r = mma_bf16(
                                [acc[4 * n], acc[4 * n + 1], acc[4 * n + 2], acc[4 * n + 3]], aa, bv
                            )
                            for e in cutlass.range_constexpr(4):
                                acc[4 * n + e] = r[e]
                    x = [acc[i] * pre for i in range(8)]
                    gb = cute.arch.fmax(gb, _body_block(bo, x, pq, pd, uts, stg, sst, g, t))
            # compensation channels: Q [dq U | q1 U], K [k1 U | dk U]
            g2 = Float32(0.0)
            g3 = Float32(0.0)
            for ci in cutlass.range_constexpr(2):
                for bp in cutlass.range_constexpr(2):
                    n0 = 2 * bp
                    v = [Float32(0.0)] * 8
                    for e in cutlass.range_constexpr(4):
                        pqa = pq[4 * n0 + e]
                        pqb = pq[4 * (n0 + 1) + e]
                        pda = pd[4 * n0 + e]
                        pdb = pd[4 * (n0 + 1) + e]
                        if ci == 0:
                            v[e] = Float32(cutlass.select_(side != 0, pqa, pda))
                            v[4 + e] = Float32(cutlass.select_(side != 0, pqb, pdb))
                        else:
                            v[e] = Float32(cutlass.select_(side != 0, pda, pqa))
                            v[4 + e] = Float32(cutlass.select_(side != 0, pdb, pqb))
                    cw0, sc0, d0, d1, d2, d3 = _quant_block(v[0], v[1], v[4], v[5])
                    cw1, sc1, e0, e1, e2, e3 = _quant_block(v[2], v[3], v[6], v[7])
                    if ci == 0:
                        g2 = cute.arch.fmax(g2, cute.arch.fmax(sc0, sc1))
                    else:
                        g3 = cute.arch.fmax(g3, cute.arch.fmax(sc0, sc1))
                    _emit(stg, sst, g, t, Int32(8 + 2 * ci + bp), cw0, cw1, sc0, sc1)
            isk = Int32(side == 1)
            gm[0] = cute.arch.fmax(gm[0], Float32(cutlass.select_(isk != 0, Float32(0.0), gb)))
            gm[1] = cute.arch.fmax(gm[1], Float32(cutlass.select_(isk != 0, Float32(0.0), g2)))
            gm[2] = cute.arch.fmax(gm[2], Float32(cutlass.select_(isk != 0, Float32(0.0), g3)))
            gm[3] = cute.arch.fmax(gm[3], Float32(cutlass.select_(isk != 0, gb, Float32(0.0))))
            gm[4] = cute.arch.fmax(gm[4], Float32(cutlass.select_(isk != 0, g2, Float32(0.0))))
            gm[5] = cute.arch.fmax(gm[5], Float32(cutlass.select_(isk != 0, g3, Float32(0.0))))
            cute.arch.sync_warp()
            # codes: 16 rows x 128 B
            for m in cutlass.range_constexpr(4):
                idx = lane + 32 * m
                rl = idx >> 3
                chn = idx & 7
                wv = load_vector(sptr(Uint32, stg + rl * 144 + chn * 16, 16), 4)
                tok = tok0 + rl
                if tok < S:
                    row_dst = (
                        dst
                        + ((Int64(bb) * Int64(S) + Int64(tok)) * Int64(H) + Int64(hh)) * 128
                        + chn * 16
                    )
                    store_vector(gptr(Uint32, row_dst, 16), wv, cop="cs")
            # provisional scales: 16 rows x 3 atoms x 8 B
            pbase = (
                prov_p + ((Int64(side) * Int64(B * H) + Int64(bh)) * Int64(T) + Int64(tile)) * 3072
            )
            for m in cutlass.range_constexpr(2):
                idx = lane + 32 * m
                if idx < 48:
                    rl = idx // 3
                    at = idx - rl * 3
                    sv = load_vector(sptr(Uint32, sst + rl * 24 + at * 8, 8), 2)
                    rw = warp * 16 + rl
                    st_global_b32_v2_hint(
                        pbase + (at * 512 + (rw & 31) * 16 + (rw >> 5) * 4) * 2,
                        sv[0],
                        sv[1],
                        "evict_last",
                    )
            cute.arch.sync_warp()
    # group maxima of this CTA
    for i in cutlass.range_constexpr(6):
        mv = warp_max(gm[i])
        if lane == 0:
            cute.arch.store(sptr(Float32, red + (warp * 8 + i) * 4), mv)
    cute.arch.sync_threads()
    if tid < 6:
        mv = Float32(0.0)
        for wi in cutlass.range_constexpr(8):
            mv = cute.arch.fmax(
                mv, cute.arch.load(sptr(Float32, red + (wi * 8 + tid) * 4), Float32)
            )
        cute.arch.store(gptr(Float32, gmax_p + (Int64(cta) * 8 + Int64(tid)) * 4), mv)
