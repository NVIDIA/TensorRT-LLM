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
"""Online softmax of the sm_120 FP8-PV attention kernel, in log2 units, one 128-key tile at a time: from the raw
scores of a warp's tile to the E4M3 P of the PV MMA, its row sums and its 16-key block sums.

With c the score scale (the multiplier of the raw scores in log2 units), a the anchor of a row (the running maximum
of c s) and T = RESCALE_THRESHOLD, P = E4M3(2^(c s - a + T)):

- landing (I-ExpCast, land_pack): one FMA per score pair puts t = c s - a + T on the f32 subnormal grid as
  K0 + (c s - a) 2^10 ulps, K0 the f16 bit pattern of 2^T, so the low 16 bits are the f16 pattern of a
  piecewise-linear 2^t; a saturating pack flushes everything below the grid (and -inf) to +0.
- lazy rescale (softmax_tile): a tile is landed with the current anchors first. A max tree over the landed patterns
  tells whether some score exceeds an anchor by more than T or is not finite (landed_flags); only then does the warp
  take the exact tile max, move the anchors, rescale O, the row sums and the pending block sums by 2^(old - new),
  and land the tile again.
- P, sums (land_finish): cvt.rn.satfinite.e4m3x2.f16x2 codes P. The row sum is that of the quantized P, the P the PV
  multiplies: the E4M3 pairs go back to f16x2 (exact) into f16x2 trees per row, grouped by 16-key block. A quad
  reduce-scatter of the per-block nodes gives every lane the sums of two blocks as a BF16 pair, the B fragment of
  the means MMA. A 4-bit warp-uniform mask of the 32-key quarters with a nonzero P byte lets the PV skip the others.

The landing has two paths behind a warp-uniform index (expcast_guard): one FMA per pair with the subnormal multiplier
c 2^-139, which needs it exact on the grid (c a multiple of 2^-10: the preparation's power-of-two scales from
2^MIN_SCORE_LOG2 up) and finite anchors; otherwise c s - a in the normal domain first, then a second FMA onto the
grid.

Per thread, P[4 i + e] holds the packed E4M3 word of quarter i at element e, P[16..17] the block sums of the step
for its rows g and g + 8, P[18..19] those of the step before (the first of a pair) and P[20] the quarter mask.

The PTX blocks (expcast_guard, land_pack, landed_flags, land_finish) are in ../ptx/fp8pv.py; this module holds the
flow of a tile around them."""

import math

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32

from ...common.expmath import scale_bf16x2
from ...constants import RESCALE_THRESHOLD
from ..ptx.fp8pv import compare, expcast_guard, land_finish, land_pack, landed_flags
from ..ptx.shared import ex2_approx_f32, mul_rn_f32
from .layout import KT, key_grp, key_rot

# The score-scale floor of this kernel: the preparation raises the score scale to a power of two of at least
# 2^MIN_SCORE_LOG2, from which the one-FMA landing's multiplier is exact on the subnormal grid
MIN_SCORE_LOG2 = -10


# ------------------------------------------------------------------------------------------------ softmax of one tile
def mask_tile(acc, q, lim, mask_tail):
    """Keys >= lim of the tile's scores to -inf (mask_tail: runtime Int32 that enables it, or None for always); n-tile
    s of a 32-key quarter covers keys 8 key_grp(s) .. 8 key_grp(s) + 7."""
    for nt in range(16):
        s = nt % 4
        i = nt // 4
        for e in range(2):
            kk = 32 * i + 8 * key_grp(s) + ((2 * q + e) ^ key_rot(s))
            ok = compare(kk, Int32(lim), "lt", Int32)
            if mask_tail is not None:
                ok = ok | (Int32(1) - mask_tail)
            acc[nt][e] = Float32(cutlass.select_(ok != 0, acc[nt][e], Float32(-math.inf)))
            acc[nt][2 + e] = Float32(cutlass.select_(ok != 0, acc[nt][2 + e], Float32(-math.inf)))


def tile_max(acc, sc):
    """The exact max of the warp's S tile per row (g, g + 8; over its n-tiles and the quad) in log2 units."""
    m0 = acc[0][0]
    m1 = acc[0][2]
    for nt in range(16):
        for e in range(2):
            if nt > 0 or e > 0:
                m0 = cute.arch.fmax(m0, acc[nt][e])
                m1 = cute.arch.fmax(m1, acc[nt][2 + e])
    m0 = cute.arch.fmax(m0, cute.arch.shuffle_sync_bfly(m0, offset=1))
    m1 = cute.arch.fmax(m1, cute.arch.shuffle_sync_bfly(m1, offset=1))
    m0 = cute.arch.fmax(m0, cute.arch.shuffle_sync_bfly(m0, offset=2))
    m1 = cute.arch.fmax(m1, cute.arch.shuffle_sync_bfly(m1, offset=2))
    # The maxima are multiplied by sc, so the anchors are in log2 units for every positive scale.
    return mul_rn_f32(m0, sc), mul_rn_f32(m1, sc)


def rescale_o(O, al0, al1, q):
    """O times the factors al0, al1 of rows g, g + 8: the O^T fragment columns are queries 2q, 2q + 1 (row g of quads
    2q, 2q + 1) and 8 + 2q, 9 + 2q (row g + 8)."""
    fa = cute.arch.shuffle_sync(al0, 8 * q)
    fb = cute.arch.shuffle_sync(al0, 8 * q + 4)
    fc = cute.arch.shuffle_sync(al1, 8 * q)
    fd = cute.arch.shuffle_sync(al1, 8 * q + 4)
    for ct in range(8):
        O[8 * ct] = O[8 * ct] * fa
        O[8 * ct + 1] = O[8 * ct + 1] * fb
        O[8 * ct + 2] = O[8 * ct + 2] * fa
        O[8 * ct + 3] = O[8 * ct + 3] * fb
        O[8 * ct + 4] = O[8 * ct + 4] * fc
        O[8 * ct + 5] = O[8 * ct + 5] * fd
        O[8 * ct + 6] = O[8 * ct + 6] * fc
        O[8 * ct + 7] = O[8 * ct + 7] * fd


@cute.jit
def softmax_tile(
    acc,
    O,
    P,
    anc0,
    anc1,
    l0,
    l1,
    exp_path,
    sc,
    q,
    lim: cutlass.Constexpr,
    mask_tail,
    release,
    refill,
):
    """Online softmax of one S tile (rows g and g + 8 of the warp): P = E4M3(2^(s - anchor + 4)) into the PV fragments
    P[4 i + r], the row sums of the quantized P and its 16-key block sums (P[16], P[17]: the means-MMA B fragments),
    landed with the current anchors. Only when some score exceeds an anchor by more than T (or is non-finite) does the
    warp take the exact tile max of the scores (kept in registers), move the anchors, rescale O and the sums, and
    re-land. Keys at or past lim (< 128) are masked: always (mask_tail None) or where the runtime mask_tail is 1.
    release() counts the warp's release of its stage and refill(count) refills the stage when the count says so (the
    stage ring of kernel.py; None: the phase releases nothing). Returns the anchors, the row sums and the landing
    path."""
    acc = [list(r) for r in acc]
    if cutlass.const_expr(lim < KT):
        mask_tile(acc, q, lim, mask_tail)
    na0 = anc0
    na1 = anc1
    # the scores stay in registers until the overflow decision, so the warp's reads of the stage ended with its MMA
    # phase: count the release before the landing; the eighth warp refills the stage after it
    old = Int32(0)
    if cutlass.const_expr(release is not None):
        old = release()
    xs = land_pack(acc, sc, anc0, anc1, exp_path)
    if cutlass.const_expr(refill is not None):
        refill(old)
    fl = landed_flags(xs)
    if (fl & 1) != 0:
        m0, m1 = tile_max(acc, sc)
        # skip correction: the anchor moves only when the tile max exceeds it by more than T
        mv0 = compare(m0, anc0 + RESCALE_THRESHOLD, "gt", Float32)
        mv1 = compare(m1, anc1 + RESCALE_THRESHOLD, "gt", Float32)
        na0 = Float32(cutlass.select_(mv0 != 0, m0, anc0))
        na1 = Float32(cutlass.select_(mv1 != 0, m1, anc1))
        exp_path = expcast_guard(sc, na0, na1)
        al0 = Float32(cutlass.select_(mv0 != 0, ex2_approx_f32(anc0 - na0), Float32(1.0)))
        al1 = Float32(cutlass.select_(mv1 != 0, ex2_approx_f32(anc1 - na1), Float32(1.0)))
        l0 = l0 * al0
        l1 = l1 * al1
        rescale_o(O, al0, al1, q)
        # block sums still waiting for their means MMA (the first step of a pair) are not in O yet: same rescale
        P[16] = scale_bf16x2(P[16], al0)
        P[17] = scale_bf16x2(P[17], al1)
        xs = land_pack(acc, sc, na0, na1, exp_path)
    # the block sums of the last two steps: P[18..19] (k - 1, the first of a pair), P[16..17] (this step)
    P[18] = P[16]
    P[19] = P[17]
    if fl == 2:
        # every landed value rounds to an E4M3 zero (no re-landing happened): zero P bytes and block sums, row sums
        # unchanged
        for e in cutlass.range_constexpr(18):
            P[e] = Int32(0)
        P[20] = Int32(0)
    else:
        packed, l0, l1, sb, pmask = land_finish(xs, l0, l1, q)
        for e in cutlass.range_constexpr(16):
            P[e] = packed[e]
        P[16] = sb[0]
        P[17] = sb[1]
        P[20] = pmask
    return na0, na1, l0, l1, exp_path
