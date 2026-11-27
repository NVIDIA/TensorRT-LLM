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
"""Q'/K' pass of the sm_120 preparation: rotation, NVFP4 body and compensation channels, per 16-token warp tile.

* Per tile: the Hadamard rotation, NVFP4 of the 8 body blocks, the projections of q1 (the dequantized body) and dq
  (its quantization residual) on the compensation basis (their A fragments come from the accumulators of the
  rotation), NVFP4 of the compensation blocks.
* Block scale (_quant_block): a body block takes it by the scale rule of the call (constants.SCALE_RULES), and q1
  and dq are those of the scale it takes; a compensation block takes that of "max".
* Compensation channels, rank = (r_q, r_k) directions: the blocks of the Q-residual term (Q: dq B_q, K: k1 B_q),
  the blocks of the K-residual term (Q: q1 B_k, K: dk B_k), then zero blocks up to COMP_CHANNELS. B_q and B_k
  are the leading r_q and r_k directions of the basis of each term (chain.basis_rows): of the one basis
  with bases "shared", of a basis of its own with "separate".
* Projections: on the min(r_q, r_k) leading directions of both terms the body and the residual are each projected,
  and the two results change places with the side; the further directions of the wider term take one projection, of
  the operand that the term has on the side of the tile. One MMA per 8 directions of a term and body block.
* Outputs: codes through a per-warp smem stage, provisional (unencoded) block scales as BF16 in the final atom
  layout, per-CTA group maxima (the encode launch turns them into the group scales and the UE4M3 bytes).
* A fragments come straight from 8-byte global loads: k-step a (16 channels 16a..16a+15) maps logical k = 2t + e to
  channel 16a + 4t + e and k = 2t + 8 + e to channel 16a + 4t + 2 + e, for both the general rotation (B fragments of
  Hd: _hd_fragment) and the fast path.
* Fast path (Hd verified on device to be diag(s) x Sylvester H128 = H8 (x) H16): one MMA per 16-channel block against
  diag(s_a) H16 (= Hd[16a.., 0..15]), then an 8-point FWHT across the blocks in registers (every thread holds the
  same in-block positions of all eight blocks).
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint16, Uint32
from cutlass.utils import SmemAllocator

from ...common.expmath import bf16x2, f2u, round_m3, warp_max
from ...common.mem import gptr, load_invariant, load_vector, sptr, store_vector
from ...constants import (
    COMP_CHANNELS,
    E2M1_MAX,
    QK_CHANNELS,
    QK_SCALE_ATOMS,
    QPRE,
    SCALE_BAND_HIGH,
    SCALE_BAND_LOW,
    WIDE_SCALE_MAX,
)
from ..ptx.shared import (
    e2m1x2_to_f32x2,
    f32x4_to_e2m1x4,
    ld_global_nc_b32_hint_pred,
    mma_bf16,
    st_global_b32_v2_hint,
)
from ..valid_len import valid_len
from .basis import row_addr, tok_addr

QK_THREADS = 256
# block scales of a key tile of Q' or K': one per token and 16 channels, in the atom layout of the QK MMA
TILE_SCALES = QK_SCALE_ATOMS * 512
K_CHUNKS = QK_CHANNELS // 32
K_ARRAY = 128 * 16

# shared memory map (bytes): Hd^T from 0 in rows of 256 B (8-byte slots, slot ^ 4 (row & 3)), the U stage (the
# directions of the two terms in BF16, rows of 272 B), the per-warp code and scale stages, the reduction area
STG_W = 16 * 144 + 16 * 24  # per warp: codes 16 x 144 B, scales 16 x 24 B


def _smem_map(stage_rows):
    """Return (rows of Hd^T, offset of the U stage, of the per-warp stages, of the reduction area, bytes) for a U
    stage of stage_rows rows: all of Hd^T beside up to 32 rows of directions; beside more, 64 rows of directions and
    16 rows of Hd^T: those that the fast path reads, or in the general rotation those of the block in work."""
    hd_rows, u_rows = (128, 32) if stage_rows <= 32 else (16, 64)
    uts = hd_rows * 256
    stg = uts + u_rows * 272
    red = stg + 8 * STG_W  # 64 floats + flag
    return hd_rows, uts, stg, red, red + 512


def _hd_slot(hdt, stage_row, c, j):
    """The shared address of Hd[c][j] in the Hd^T stage, whose row stage_row holds output channel j."""
    return sptr(Uint16, hdt + stage_row * 256 + ((c >> 2) ^ ((j & 3) << 2)) * 8 + (c & 3) * 2)


def _hd_fragment(hdt, stage_row, j, s, t):
    """The B fragment of a rotation MMA, Hd[16 s + 4 t .. + 3][j], from its slot of the Hd^T stage, whose row
    stage_row holds output channel j."""
    return load_vector(
        sptr(Uint32, hdt + stage_row * 256 + ((4 * s + t) ^ ((j & 3) << 2)) * 8, 8), 2
    )


@cute.jit
def _quant_block(
    x0: Float32, x1: Float32, x2: Float32, x3: Float32, rule: cutlass.Constexpr, quad=None
):
    """One 16-channel NVFP4 block spread over a quad (4 values per thread).

    * rule: one of constants.SCALE_RULES. With m the largest magnitude of the block, its scale, rounded to
      three mantissa bits, is
        - "max": m / E2M1_MAX.
        - "band": m / WIDE_SCALE_MAX when a magnitude of the block is at least m * SCALE_BAND_LOW and below
          m * SCALE_BAND_HIGH, the FP32 products; else m / E2M1_MAX.
        - "search": of m / E2M1_MAX and m / WIDE_SCALE_MAX the one with the smaller sum of
          (x - code * scale)^2 over the 16 values, m / E2M1_MAX when the sums are equal.
    * quad: the first lane of the quad, for "band": the threads of the warp vote whether they hold a magnitude
      in the band, and a block takes the four votes of its quad.
    * Returns (code word: low byte = (x1,x0), high byte = (x3,x2); scale; dequantized x0..x3).
    """
    a0, a1, a2, a3 = (cute.math.abs(x) for x in (x0, x1, x2, x3))
    am = cute.arch.fmax(cute.arch.fmax(a0, a1), cute.arch.fmax(a2, a3))
    am = cute.arch.fmax(Float32(0.0), am)
    am = cute.arch.fmax(am, cute.arch.shuffle_sync_bfly(am, offset=1))
    am = cute.arch.fmax(am, cute.arch.shuffle_sync_bfly(am, offset=2))
    xs = (x0, x1, x2, x3)
    f_max, f_wide = Float32(1.0 / E2M1_MAX), Float32(1.0 / WIDE_SCALE_MAX)
    if cutlass.const_expr(rule == "band"):
        lo = am * Float32(SCALE_BAND_LOW)
        hi = am * Float32(SCALE_BAND_HIGH)
        hit = Int32(0)
        for a in (a0, a1, a2, a3):
            hit = hit | (Int32(a >= lo) & Int32(a < hi))
        votes = Uint32(cute.arch.vote_ballot_sync(hit != 0))
        in_band = ((votes >> Uint32(quad)) & Uint32(0xF)) != Uint32(0)
        factors = (Float32(cutlass.select_(in_band, f_wide, f_max)),)
    elif cutlass.const_expr(rule == "search"):
        factors = (f_max, f_wide)
    else:
        factors = (f_max,)
    for i in cutlass.range_constexpr(len(factors)):
        sc_i = round_m3(am * factors[i])
        inv = Float32(
            cutlass.select_(sc_i > Float32(1e-30), cute.arch.rcp_approx(sc_i), Float32(0.0))
        )
        cw_i = f32x4_to_e2m1x4(x0 * inv, x1 * inv, x2 * inv, x3 * inv)
        v0, v1 = e2m1x2_to_f32x2(cw_i & Uint32(0xFF))
        v2, v3 = e2m1x2_to_f32x2(cw_i >> Uint32(8))
        d_i = (v0 * sc_i, v1 * sc_i, v2 * sc_i, v3 * sc_i)
        if cutlass.const_expr(len(factors) > 1):
            # the squared error of the block: the thread's four values, then the quad
            sq = [(xs[e] - d_i[e]) * (xs[e] - d_i[e]) for e in range(4)]
            err_i = (sq[0] + sq[1]) + (sq[2] + sq[3])
            err_i = err_i + cute.arch.shuffle_sync_bfly(err_i, offset=1)
            err_i = err_i + cute.arch.shuffle_sync_bfly(err_i, offset=2)
        if cutlass.const_expr(i == 0):
            cw, sc, d = cw_i, sc_i, d_i
            if cutlass.const_expr(len(factors) > 1):
                err = err_i
        else:
            take = err_i < err
            cw = Uint32(cutlass.select_(take, cw_i, cw))
            sc = Float32(cutlass.select_(take, sc_i, sc))
            d = tuple(Float32(cutlass.select_(take, d_i[e], d[e])) for e in range(4))
            err = Float32(cutlass.select_(take, err_i, err))
    return cw, sc, d[0], d[1], d[2], d[3]


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


def _stage_rows(rank, separate):
    """Return the first rows in the U stage of the directions that the residual and the wider term's further
    projection take, and the rows of the stage; the directions of the body start at row 0.

    * Bases "shared": the one basis is staged once, the leading direction first, and all three read it from their
      first direction.
    * Bases "separate": staged by use: the min(r_q, r_k) leading directions of the basis that the body takes on
      the side of the tile, those of the basis of the residual, then the further directions of the wider term.
    """
    both, top = min(rank), max(rank)
    return (both, 2 * both, both + top) if separate else (0, both, top)


def _body_block(bo, x, side, proj, uts, stg, sst, g, t, rule):
    """Quantize and emit body block bo and accumulate its projections; return the larger of its two block scales.

    * A trace-time helper. x[0..7]: n-tile 0 c0..c3, n-tile 1 c0..c3. rule: that of _quant_block.
    * proj = (body, residual, further), each (accumulators, directions, first row in the U stage).
    * Body and residual: the dequantized body and the residual, on the leading directions that both terms have.
    * Further (its fourth entry: the index of the wider term): the operand that the wider term has on this side,
      on that term's further directions.
    """
    cw0, sc0, d00, d01, d02, d03 = _quant_block(x[0], x[1], x[4], x[5], rule, 4 * g)
    cw1, sc1, d10, d11, d12, d13 = _quant_block(x[2], x[3], x[6], x[7], rule, 4 * g)
    _emit(stg, sst, g, t, Int32(bo), cw0, cw1, sc0, sc1)
    aq = [bf16x2(d00, d01), bf16x2(d10, d11), bf16x2(d02, d03), bf16x2(d12, d13)]
    ad = [
        bf16x2(x[0] - d00, x[1] - d01),
        bf16x2(x[2] - d10, x[3] - d11),
        bf16x2(x[4] - d02, x[5] - d03),
        bf16x2(x[6] - d12, x[7] - d13),
    ]
    (pb, both, rb), (pr, _, rr), (pw, further, rw, wide) = proj

    def fragment(row):
        ub = uts + (row + g) * 272 + (16 * bo + 2 * t) * 2
        return [
            cute.arch.load(sptr(Uint32, ub), Uint32),
            cute.arch.load(sptr(Uint32, ub + 16), Uint32),
        ]

    def project(acc, n, a, bv):
        r = mma_bf16([acc[4 * n + e] for e in range(4)], a, bv)
        for e in range(4):
            acc[4 * n + e] = r[e]

    for n in range(both // 8):
        bv = fragment(rb + 8 * n)
        project(pb, n, aq, bv)
        project(pr, n, ad, bv if rr == rb else fragment(rr + 8 * n))
    if further:
        # the Q-residual term has the residual of a Q tile and the body of a K tile, the K-residual term the
        # other one
        of_q, of_k = (aq, ad) if wide else (ad, aq)
        aw = [Uint32(cutlass.select_(side != 0, of_k[i], of_q[i])) for i in range(4)]
        for n in range(further // 8):
            project(pw, n, aw, fragment(rw + 8 * n))
    return cute.arch.fmax(sc0, sc1)


def _item_coords(item, T, BH, H):
    bhs = item // T
    tile = item - bhs * T
    side = bhs // BH
    bh = bhs - side * BH
    bb = bh // H
    return bhs, tile, side, bh, bb, bh - bb * H


def _side_pitch(pitches, side):
    """The token pitch (bytes) of the side's tensor: pitches = (that of q, that of k)."""
    return Int64(cutlass.select_(side != 0, pitches[1], pitches[0]))


def _rows_base(qk, pitches, side, bb, hh, rg, S, t):
    base = Int64(cutlass.select_(side != 0, qk[1], qk[0]))
    return tok_addr(base, bb, rg, hh, S, _side_pitch(pitches, side)) + t * 8


def _load_rows(afn, base, rstride, rg, nv, live=None):
    """A fragments of rows rg, rg + 8 into afn[4a + 0/1], afn[4a + 2/3] (channels 16a + 4t + 0..3); zero from row nv."""

    def ok(row):
        return Int32(row < nv) if live is None else live & Int32(row < nv)

    for a in range(8):
        wa = ld_global_nc_b32_hint_pred(base + a * 32, ok(rg), "evict_first", 2)
        wb = ld_global_nc_b32_hint_pred(base + rstride * 8 + a * 32, ok(rg + 8), "evict_first", 2)
        afn[4 * a] = wa[0]
        afn[4 * a + 1] = wa[1]
        afn[4 * a + 2] = wb[0]
        afn[4 * a + 3] = wb[1]


@cute.jit
def _store_rows(stg, lane, tok0, dst, bb, hh, S, H):
    for m in cutlass.range_constexpr(4):
        idx = lane + 32 * m
        rl = idx >> 3
        chn = idx & 7
        wv = load_vector(sptr(Uint32, stg + rl * 144 + chn * 16, 16), 4)
        tok = tok0 + rl
        if tok < S:
            row_dst = row_addr(dst, bb, tok, hh, S, H, 128) + chn * 16
            store_vector(gptr(Uint32, row_dst, 16), wv, cop="cs")


def ntile_row(r):
    rp = r & 31
    return 8 * (8 * (r >> 6) + 4 * ((r >> 5) & 1) + ((rp & 7) >> 1)) + 2 * (rp >> 3) + (rp & 1)


def _store_tile(stg, lane, warp, tok0, kimg, S):
    """Store the warp's staged K' rows in the tile image at kimg: K_CHUNKS arrays of 128 16-byte rows, n-tile order."""
    rl = lane & 15
    chn = lane >> 4
    src = stg + rl * 144 + chn * 16
    key_dst = kimg + (chn * K_ARRAY + ntile_row(warp * 16 + rl) * 16)
    for m in range(K_CHUNKS // 2):
        wv = load_vector(sptr(Uint32, src + 32 * m, 16), 4)
        kv = [Uint32(cutlass.select_(tok0 + rl < S, w, Uint32(0))) for w in wv]
        store_vector(gptr(Uint32, key_dst + 2 * K_ARRAY * m, 16), kv, cop="cs")


@cute.kernel
def qk_kernel(
    q_p: Int64,
    k_p: Int64,
    had_p: Int64,
    ut_q_p: Int64,
    ut_k_p: Int64,
    q4_p: Int64,
    k4_p: Int64,
    prov_p: Int64,
    gmax_p: Int64,
    B: Int32,
    S: Int32,
    H: Int32,
    QP: Int32,
    KP: Int32,
    T: Int32,
    nitems: Int32,
    ncta: Int32,
    lens_p,
    *,
    tile_pitch: cutlass.Constexpr = None,
    rank: cutlass.Constexpr,
    separate: cutlass.Constexpr,
    scale_rule: cutlass.Constexpr,
):
    """Run the Q'/K' pass of the CTA's share of the nitems items (side, batch, head, tile).

    * QP, KP: the bytes between consecutive tokens of q and of k.
    * lens_p: None, or the global address of the valid prefix lengths: the tokens from the length of their batch
      element on are then read as zeros (zero codes and block scales, outside the group maxima), and a tile
      without a valid token is left as it is.
    * tile_pitch: None, or K' goes to tile images.
    * rank, separate: the channels of the two compensation terms and whether each has a basis of its own.
    * scale_rule: the rule for the scale of a body block, one of constants.SCALE_RULES (_quant_block); a
      compensation block takes that of "max".
    * ut_q_p, ut_k_p: the basis of the Q-residual term and that of the K-residual term, FP32
      [B H][directions][128] with the leading direction last; with bases "shared" the one address, twice.
    """
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    lane = tid & 31
    warp = tid >> 5
    g = lane >> 2
    t = lane & 3
    res_row, fur_row, stage_rows = _stage_rows(rank, separate)
    both, wide = min(rank), int(rank[1] > rank[0])
    hd_rows, sq_uts, sq_stg, sq_red, sq_bytes = _smem_map(stage_rows)
    smem = SmemAllocator()
    buf = smem.allocate_array(Uint32, sq_bytes // 4, byte_alignment=128)
    sb = Int32(buf.toint())
    hdt = sb
    uts = sb + sq_uts
    stg = sb + sq_stg + warp * STG_W
    sst = stg + 16 * 144
    red = sb + sq_red
    flag = red + 256

    if tid == 0:
        cute.arch.store(sptr(Uint32, flag), Uint32(1))
    # Hd^T into smem (row j < hd_rows = output channel, _hd_slot); check Hd == diag(s) x H128
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
            hv = (w[e // 2] if e % 2 == 0 else w[e // 2] >> Uint32(16)) & Uint32(0xFFFF)
            staged = cutlass.const_expr(hd_rows == 128) or j < hd_rows
            if staged:
                cute.arch.store(_hd_slot(hdt, j, c, j), hv.to(Uint16))
            par = Uint32(cute.arch.popc(Uint32(c & j)) & 1)
            ok = ok & Uint32(hv == (s0 ^ (par << Uint32(15))))
    # zero the code stage (bytes 96..127 of every row stay zero), and with zero blocks the scale stage behind it
    stage_bytes = 16 * 144 if sum(rank) == COMP_CHANNELS else STG_W
    zz = Uint32(0)
    for it in cutlass.range_constexpr((stage_bytes + 511) // 512):
        off = it * 512 + lane * 16
        if off < stage_bytes:
            store_vector(sptr(Uint32, stg + off, 16), (zz, zz, zz, zz))
    cute.arch.sync_threads()
    if ok == Uint32(0):
        cute.arch.store(sptr(Uint32, flag), Uint32(0))
    cute.arch.sync_threads()
    fast = Int32(cute.arch.load(sptr(Uint32, flag), Uint32))
    gm = cute.make_rmem_tensor((6,), Float32)
    gm.fill(0.0)
    item_lo = (cta * nitems) // ncta
    item_hi = ((cta + 1) * nitems) // ncta
    cur = Int32(-1)
    pitches = (QP, KP)
    BH = B * H
    af = cute.make_rmem_tensor((32,), Uint32)
    afn = cute.make_rmem_tensor((32,), Uint32)

    if item_lo < item_hi:
        _, tile0, side0, _, bb0, hh0 = _item_coords(item_lo, T, BH, H)
        rg0 = tile0 * 128 + warp * 16 + g
        nv0 = valid_len(lens_p, bb0, S)
        _load_rows(
            afn,
            _rows_base((q_p, k_p), pitches, side0, bb0, hh0, rg0, S, t),
            _side_pitch(pitches, side0),
            rg0,
            nv0,
        )
    for item in cutlass.range(item_lo, item_hi, unroll=1):
        for i in cutlass.range_constexpr(32):
            af[i] = afn[i]
        bhs, tile, side, bh, bb, hh = _item_coords(item, T, BH, H)
        # prefetch the next item's rows (zeros past the range / past S)
        nitem = item + 1
        _, ntile, nside, _, nbb, nhh = _item_coords(nitem, T, BH, H)
        nrg = ntile * 128 + warp * 16 + g
        nok = Int32(nitem < item_hi)
        nvn = valid_len(lens_p, cutlass.min(nbb, B - 1), S)
        _load_rows(
            afn,
            _rows_base((q_p, k_p), pitches, nside, nbb, nhh, nrg, S, t),
            _side_pitch(pitches, nside),
            nrg,
            nvn,
            nok,
        )
        live = cutlass.const_expr(lens_p is None) or tile * 128 < valid_len(lens_p, bb, S)
        if live:
            if bhs != cur:
                cute.arch.sync_threads()
                for it in cutlass.range_constexpr(stage_rows // 8):
                    idx = it * QK_THREADS + tid  # 8 rows of 32 chunks of four directions' channels
                    r = idx >> 5
                    cc = idx & 31
                    # stage row r holds direction r - first of a basis that has `directions` of them (_stage_rows):
                    #   * bases "shared": the one basis
                    #   * rows of the body: the basis of the K-residual term on a Q tile, of the Q-residual term
                    #     on a K tile; rows of the residual: that of the other term
                    #   * further rows: the basis of the wider term, from its direction `both` on
                    first, directions, basis_p = 0, stage_rows, ut_q_p
                    if cutlass.const_expr(separate and 8 * it >= fur_row):
                        first, directions, basis_p = both, rank[wide], (ut_q_p, ut_k_p)[wide]
                    elif cutlass.const_expr(separate):
                        first = res_row if 8 * it >= res_row else 0
                        q_term = (side != 0) if first == 0 else (side == 0)
                        directions = Int64(cutlass.select_(q_term, Int64(rank[0]), Int64(rank[1])))
                        basis_p = Int64(cutlass.select_(q_term, ut_q_p, ut_k_p))
                    src = Int64(bh) * directions + Int64(directions - 1 - (r - first))
                    u4 = load_invariant(Float32, basis_p + (src * 128 + Int64(cc * 4)) * 4, 4)
                    store_vector(
                        sptr(Uint32, uts + r * 272 + cc * 8, 8),
                        (bf16x2(u4[0], u4[1]), bf16x2(u4[2], u4[3])),
                    )
                cute.arch.sync_threads()
                cur = bhs
            if cutlass.const_expr(tile_pitch is None):
                dst = Int64(cutlass.select_(side != 0, k4_p, q4_p))
            pre = Float32(cutlass.select_(side != 0, Float32(1.0), Float32(QPRE)))
            tok0 = tile * 128 + warp * 16
            # the projections (module docstring), 4 accumulators per 8 directions: the body and the residual on the
            # directions both terms have, and the wider term's further directions, from their rows of the stage
            pb = cute.make_rmem_tensor((COMP_CHANNELS // 2,), Float32)
            pr = cute.make_rmem_tensor((COMP_CHANNELS // 2,), Float32)
            pw = cute.make_rmem_tensor((COMP_CHANNELS // 2,), Float32)
            pb.fill(0.0)
            pr.fill(0.0)
            pw.fill(0.0)
            proj = (
                (pb, both, 0),
                (pr, both, res_row),
                (pw, max(rank) - both, fur_row, wide),
            )
            gb = Float32(0.0)
            if fast != 0:
                # stage 1: z_a = x_a (diag(s_a) H16), 16-channel blocks
                yv = cute.make_rmem_tensor((64,), Float32)
                for a in cutlass.range_constexpr(8):
                    aa = [af[4 * a], af[4 * a + 2], af[4 * a + 1], af[4 * a + 3]]
                    for n in cutlass.range_constexpr(2):
                        # B fragment of block a, n-tile n: Hd[16a + 4t .. +3][8n + g]
                        hb = _hd_fragment(hdt, 8 * n + g, 8 * n + g, a, t)
                        r = mma_bf16([Float32(0.0)] * 4, aa, hb)
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
                    gb = cute.arch.fmax(
                        gb, _body_block(bo, x, side, proj, uts, stg, sst, g, t, scale_rule)
                    )
            else:
                for bo in cutlass.range_constexpr(8):
                    if cutlass.const_expr(hd_rows < 128):
                        # the stage holds the 16 rows of Hd^T of one block at a time: thread = (input channel,
                        # 8 output channels of the block)
                        cute.arch.sync_threads()
                        hc = tid >> 1
                        hj = 16 * bo + (tid & 1) * 8
                        hw = load_invariant(Uint32, had_p + Int64(hc * 128 + hj) * 2, 4)
                        for e in cutlass.range_constexpr(8):
                            hv = (hw[e // 2] if e % 2 == 0 else hw[e // 2] >> Uint32(16)) & Uint32(
                                0xFFFF
                            )
                            cute.arch.store(
                                _hd_slot(hdt, hj + e - 16 * bo, hc, hj + e), hv.to(Uint16)
                            )
                        cute.arch.sync_threads()
                    acc = [Float32(0.0)] * 8
                    for s in cutlass.range_constexpr(8):
                        aa = [af[4 * s], af[4 * s + 2], af[4 * s + 1], af[4 * s + 3]]
                        for n in cutlass.range_constexpr(2):
                            row = 8 * (2 * bo + n) + g
                            bv = _hd_fragment(
                                hdt, row - (16 * bo if hd_rows < 128 else 0), row, s, t
                            )
                            r = mma_bf16(
                                [acc[4 * n], acc[4 * n + 1], acc[4 * n + 2], acc[4 * n + 3]], aa, bv
                            )
                            for e in cutlass.range_constexpr(4):
                                acc[4 * n + e] = r[e]
                    x = [acc[i] * pre for i in range(8)]
                    gb = cute.arch.fmax(
                        gb, _body_block(bo, x, side, proj, uts, stg, sst, g, t, scale_rule)
                    )
            # the projections of the two terms: the Q-residual term has the residual of a Q tile and the body of a K
            # tile, the K-residual term the other one; then the further directions of the wider term
            term = [
                [Float32(cutlass.select_(side != 0, pb[i], pr[i])) for i in range(both // 2)],
                [Float32(cutlass.select_(side != 0, pr[i], pb[i])) for i in range(both // 2)],
            ]
            term[wide] = term[wide] + [pw[i] for i in range((max(rank) - both) // 2)]
            # compensation channels: the 16-channel blocks of the Q-residual term, then of the K-residual term (the
            # stage holds zeros behind them); gc: the larger block scale of each term
            gc = [Float32(0.0)] * 2
            for cb in cutlass.range_constexpr(sum(rank) // 16):
                ti = int(cb >= rank[0] // 16)
                nb = 2 * cb - ti * rank[0] // 8
                v = term[ti][4 * nb : 4 * nb + 8]
                cw0, sc0, *_ = _quant_block(v[0], v[1], v[4], v[5], "max")
                cw1, sc1, *_ = _quant_block(v[2], v[3], v[6], v[7], "max")
                gc[ti] = cute.arch.fmax(gc[ti], cute.arch.fmax(sc0, sc1))
                _emit(stg, sst, g, t, Int32(8 + cb), cw0, cw1, sc0, sc1)
            isk = Int32(side == 1)
            groups = (gb, gc[0], gc[1])
            for j in cutlass.range_constexpr(3):
                gm[j] = cute.arch.fmax(
                    gm[j], Float32(cutlass.select_(isk != 0, Float32(0.0), groups[j]))
                )
            for j in cutlass.range_constexpr(3):
                gm[3 + j] = cute.arch.fmax(
                    gm[3 + j], Float32(cutlass.select_(isk != 0, groups[j], Float32(0.0)))
                )
            cute.arch.sync_warp()
            # codes: 16 rows x 128 B (tile_pitch: K' to the tile images that many bytes apart from k4_p, zero past S)
            if cutlass.const_expr(tile_pitch is None):
                _store_rows(stg, lane, tok0, dst, bb, hh, S, H)
            elif side == 0:
                _store_rows(stg, lane, tok0, q4_p, bb, hh, S, H)
            else:
                _store_tile(stg, lane, warp, tok0, k4_p + Int64(bh * T + tile) * tile_pitch, S)
            # provisional scales: 16 rows x 3 atoms x 8 B
            pbase = prov_p + ((Int64(side) * Int64(BH) + Int64(bh)) * Int64(T) + Int64(tile)) * (
                2 * TILE_SCALES
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
        else:
            # a tile past the valid length writes no codes, and zero provisional scales: the encode kernel reads them
            pbase = prov_p + ((Int64(side) * Int64(BH) + Int64(bh)) * Int64(T) + Int64(tile)) * (
                2 * TILE_SCALES
            )
            if tid < TILE_SCALES // 8:
                store_vector(
                    gptr(Uint32, pbase + Int64(tid) * 16, 16),
                    (Uint32(0), Uint32(0), Uint32(0), Uint32(0)),
                )
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
