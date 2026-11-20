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
"""Online softmax of the sm_120 attention kernel, in log2 units, one 128-key tile at a time: from the raw scores of a
tile to the NVFP4 P of the PV MMA.

Every thread holds 8 scores of each of 8 blocks of 16 keys (4 blocks x its rows g and g + 8); the XOR-1 lane holds the
other 8 scores of the same blocks. With c the score scale (the multiplier of the raw scores in log2 units), per row:

- anchor: the running row max of c s, moved only when a tile's max exceeds it by more than T = RESCALE_THRESHOLD
  (lazy rescale: a move rescales O and the row sum by 2^(old - new), anchor_update); the P block scales carry T
  binades of headroom.
- block scale: a = UE4M3(448 * 2^(c bm - anchor - T)) with bm the exact max of the block's 16 scores, encoded and
  read back (block_scales2). The decoded scale is what the MMA multiplies by, so the elements are coded against it.
- elements (I-ExpCast): E2M1(6 * 2^(c (s - bm) + d)), d the log2 of the unrounded scale over the decoded one. The
  exponential is a cast: z = c (s - bm) + d lands on the f32 subnormal grid as K0 + z * 2^10 ulps, K0 the f16 bit
  pattern of 6, so the low 16 bits are the f16 pattern of a piecewise-linear 6 * 2^z (the exponent field counts the
  binades, the mantissa interpolates); a saturating pack flushes everything below the grid to +0 and
  cvt.rn.satfinite.e2m1x2.f16x2 codes the elements (land32_sum).
- sums: the 16-key block sums of the unquantized P (the f16 element values times the decoded block scale), which are
  the operand of the means MMA and add up to the row sum.

The landing has three forms. Folded (the default): one FMA per score pair, s * M + (K0 - bm M + d 2^-139) with the
subnormal multiplier M = c 2^-139; it needs the block offsets on the fine subnormal grid, which the warp's fold-safety
flag guarantees (anchors within +-7900 and c <= 1/16, anchor_update). Subtract first (flag clear): the scores minus
their block max, then the same FMA with the offset K0 + d 2^-139 (prepare_land64_flag picks between the two behind a
warp-uniform branch). Two-step (the copy of the tile loop the kernel takes for a score scale below 2^-10 or with
c 2^-139 inexact): a normal-domain FMA c (s - bm) + (17.5 + d), then the multiply by 2^-139 onto the grid.

The PTX blocks (need_vote, anchor_update, rescale64_loop, block_scales2, prepare_land64_flag, land32_sum) are in
../ptx/nvfp4pv.py; this module holds the flow of a tile around them.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32

from ...common.expmath import bf16x2, scale_bf16x2, u2f
from ..ptx.nvfp4pv import (
    anchor_update,
    block_scales2,
    land32_sum,
    need_vote,
    prepare_land64_flag,
    rescale64_loop,
)
from ..ptx.shared import fma_rn_f32

# The score-scale floor of this family: the preparation raises the score scale to a power of two of at least
# 2^MIN_SCORE_LOG2 (its encode launch grows the Q group scales up to it), inside the range of the folded landing
MIN_SCORE_LOG2 = -6
# the P block scales carry T = RESCALE_THRESHOLD binades of headroom, UE4M3(448 * 2^(c bm - m - T)): the anchor m
# (running max) only moves, and O is only rescaled, when a tile max exceeds it by more than T


# ------------------------------------------------------------------------------------------------ softmax of one tile
# A thread's scores of a tile: Sacc[8 s + 4 h + i] = the four accumulators of n-tile i of the 32-key group (s, h),
# elements 2 rr + e of row g + 8 rr, key 64 s + 32 h + 8 t + 2 i + e of the tile. A 16-key block is (s, h, t // 2):
# the lanes t and t ^ 1 hold its two halves. Straight-line emitters (plain Python, unrolled at trace time).
def emit_block_max(Sacc, masked, floor=None):
    """16-key block maxima of the raw scores: bm[s][h] = (row g, row g + 8); the pair (t, t ^ 1) holds one block.
    masked: the tile has keys past the sequence (scores -inf), whose blocks keep a finite max. floor: with lengths,
    the run-time lower bound of the maxima (-1e30 for the last tile of the keys, else -inf)."""
    bm = [[None, None], [None, None]]
    for s_ in range(2):
        for h_ in range(2):
            pr = []
            for rr in range(2):
                vals = []
                for i_ in range(4):
                    vals.append(Sacc[8 * s_ + 4 * h_ + i_][2 * rr])
                    vals.append(Sacc[8 * s_ + 4 * h_ + i_][2 * rr + 1])
                m01 = cute.arch.fmax(
                    cute.arch.fmax(vals[0], vals[1]), cute.arch.fmax(vals[2], vals[3])
                )
                m23 = cute.arch.fmax(
                    cute.arch.fmax(vals[4], vals[5]), cute.arch.fmax(vals[6], vals[7])
                )
                mm = cute.arch.fmax(m01, m23)
                mm = cute.arch.fmax(mm, cute.arch.shuffle_sync_bfly(mm, offset=1))
                if masked:
                    mm = cute.arch.fmax(mm, Float32(-1e30))
                if floor is not None:
                    mm = cute.arch.fmax(mm, floor)
                pr.append(mm)
            bm[s_][h_] = pr
    return bm


def emit_p(st, Sacc, bm, ma_dst, ma_off, two_step):
    """NVFP4 P of the tile into st.Pst (codes [4 s + 2 h + rr], scale words [8 + s]), the block sums of the
    unquantized P as the BF16 operand of the means MMA into ma_dst[ma_off] and ma_dst[ma_off + 1], and the row-sum
    update."""
    par = st.par
    P = st.Pst
    lsum = st.lsum
    pm = st.c if two_step else st.M
    sch = [None, None]
    e4 = [None, None]
    # dsh[s][h][rr]: the log2 of block (s, h, rr)'s unrounded scale over its decoded one, this lane's for s = par and
    # the XOR-1 lane's for the other
    dsh = [[None, None], [None, None]]
    # The XOR-1 lanes share their block maxima. Each lane owns the blocks s = par for the block sums: it computes
    # their two scales (rows g and g + 8) per h, and the lanes exchange the shifts and the UE4M3 bytes.
    for h_ in range(2):
        b0 = Float32(cutlass.select_(par != 0, bm[1][h_][0], bm[0][h_][0]))
        b1 = Float32(cutlass.select_(par != 0, bm[1][h_][1], bm[0][h_][1]))
        sch[h_], e4[h_], d0, d1 = block_scales2(b0, b1, st.c, st.anchor[0], st.anchor[1])
        own = [d0, d1]
        peer = [
            cute.arch.shuffle_sync_bfly(d0, offset=1),
            cute.arch.shuffle_sync_bfly(d1, offset=1),
        ]
        dsh[0][h_] = [Float32(cutlass.select_(par != 0, peer[rr], own[rr])) for rr in range(2)]
        dsh[1][h_] = [Float32(cutlass.select_(par != 0, own[rr], peer[rr])) for rr in range(2)]
    prepared = None
    if not two_step:
        all_scores = []
        all_maxima = []
        for s_ in range(2):
            for h_ in range(2):
                for rr in range(2):
                    all_maxima.append(bm[s_][h_][rr])
                    for i_ in range(4):
                        all_scores.append(Sacc[8 * s_ + 4 * h_ + i_][2 * rr])
                        all_scores.append(Sacc[8 * s_ + 4 * h_ + i_][2 * rr + 1])
        prepared = prepare_land64_flag(all_scores, all_maxima, st.M, st.safe[0])
    xs = []
    for h_ in range(2):
        vals = []
        offsets = []
        kofs = [] if two_step else None
        for s_ in range(2):
            for rr in range(2):
                if two_step:
                    for i_ in range(4):
                        vals.append(Sacc[8 * s_ + 4 * h_ + i_][2 * rr])
                        vals.append(Sacc[8 * s_ + 4 * h_ + i_][2 * rr + 1])
                    offsets.append(bm[s_][h_][rr])
                    kofs.append(dsh[s_][h_][rr] + Float32(17.5))
                else:
                    gi = 4 * s_ + 2 * h_ + rr
                    vals.extend(prepared[8 * gi : 8 * gi + 8])
                    # the shift of the decoded scale, on the subnormal grid of the folded offset (K0 - bm M, or K0
                    # after the subtraction)
                    offsets.append(fma_rn_f32(dsh[s_][h_][rr], u2f(0x400), prepared[64 + gi]))
        words, sums = land32_sum(vals, offsets, pm, sch[h_], par, kofs)
        for s_ in range(2):
            for rr in range(2):
                P[4 * s_ + 2 * h_ + rr] = words[2 * s_ + rr]
        xs.append(sums)
    lsum[0] = lsum[0] + (xs[0][0] + xs[1][0])
    lsum[1] = lsum[1] + (xs[0][1] + xs[1][1])
    ma_dst[ma_off] = Int32(bf16x2(xs[0][0], xs[1][0]))
    ma_dst[ma_off + 1] = Int32(bf16x2(xs[0][1], xs[1][1]))
    # P scale words (lanes 4g: row g, 4g + 1: row g + 8): blocks 4s + 2h from this pair, 4s + 2h + 1 from t ^ 2
    wa = e4[0] | (e4[1] << 16)
    wb = cute.arch.shuffle_sync_bfly(wa, offset=1)
    dsel = Int32(cutlass.select_(par != 0, Int32(0x3175), Int32(0x6420)))
    dw = Int32(cute.arch.prmt(wa, wb, dsel))
    rw = cute.arch.shuffle_sync_bfly(dw, offset=2)
    P[8] = Int32(cute.arch.prmt(dw, rw, Int32(0x5140)))
    P[9] = Int32(cute.arch.prmt(dw, rw, Int32(0x7362)))


@cute.jit
def softmax_phase(
    st, Sacc, masked: cutlass.Constexpr, cur_odd: cutlass.Constexpr, two_step: cutlass.Constexpr
):
    """Softmax of the current tile (scores Sacc), without an MMA: NVFP4 P into st.Pst, its means operand into st.Mlo
    (even tile) or st.Pst[10..11] (odd tile), the row sums; an anchor move rescales O (after the burst's PV, behind a
    warp-uniform branch) and the pending means operand of the even tile. masked, cur_odd "lens": a tile of a range
    that ends at a run-time tile, with the floor of its block maxima in st.mask[1] and its parity in st.odd."""
    if cutlass.const_expr(masked == "lens"):
        bm = emit_block_max(Sacc, False, st.mask[1])
    else:
        bm = emit_block_max(Sacc, masked)
    bml = [bm[s_][h_][rr] for s_ in range(2) for h_ in range(2) for rr in range(2)]
    anchor, thr, safe, lsum = st.anchor, st.thr, st.safe, st.lsum
    # Raw thresholds round down: no skipped anchor move even for a non-power-of-two c.
    # Full row reductions, correction exp2s and guard votes run only on a possible move.
    vneed = need_vote(bml, thr[0], thr[1])
    a0, a1, f0, f1, t0, t1, s0, s1, sf = anchor_update(
        vneed, bml, anchor[0], anchor[1], thr[0], thr[1], lsum[0], lsum[1], safe[0], st.c
    )
    anchor[0] = a0
    anchor[1] = a1
    thr[0] = t0
    thr[1] = t1
    lsum[0] = s0
    lsum[1] = s1
    safe[0] = sf
    O = st.O
    vals = rescale64_loop([O[i] for i in range(64)], f0, f1, vneed)
    for i in cutlass.range_constexpr(64):
        O[i] = vals[i]
    Mlo = st.Mlo
    if cutlass.const_expr(cur_odd == "lens"):
        if (vneed & st.odd) != 0:
            Mlo[0] = scale_bf16x2(Mlo[0], f0)
            Mlo[1] = scale_bf16x2(Mlo[1], f1)
        # the means operand of the tile, then into the place of its parity
        ma = cute.make_rmem_tensor((2,), Int32)
        emit_p(st, Sacc, bm, ma, 0, two_step)
        for i in cutlass.range_constexpr(2):
            st.Pst[10 + i] = Int32(cutlass.select_(st.odd != 0, ma[i], st.Pst[10 + i]))
            Mlo[i] = Int32(cutlass.select_(st.odd != 0, Mlo[i], ma[i]))
    else:
        if cutlass.const_expr(cur_odd):
            if vneed != 0:
                # the even tile's means operand waits for its partner: bring it to the new anchor
                Mlo[0] = scale_bf16x2(Mlo[0], f0)
                Mlo[1] = scale_bf16x2(Mlo[1], f1)
        if cutlass.const_expr(cur_odd):
            emit_p(st, Sacc, bm, st.Pst, 10, two_step)
        else:
            emit_p(st, Sacc, bm, Mlo, 0, two_step)
