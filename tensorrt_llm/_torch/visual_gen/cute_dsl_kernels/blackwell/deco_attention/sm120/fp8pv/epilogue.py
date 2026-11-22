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
"""Epilogues of the sm_120 FP8-PV attention: the output rows of a whole unit, the FP32 partials of a key-range piece
of the split tail, and the merge launch that combines the pieces of a unit.

A thread holds O transposed (mma.py): its accumulators are those of the queries 2q, 2q + 1 (row g of the warp's quads
2q, 2q + 1) and 8 + 2q, 9 + 2q (row g + 8), while its anchors are those of its own rows g and g + 8 and its row
sums the quad's partial sums of those rows. So the factors of a row (1 / l, or the rescale of a merge) are reduced
over the quad and fetched from the lanes that own the rows."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import BFloat16, Float32, Int32, Int64

from ...common.mem import gptr, load_invariant, load_vector, sptr, store_vector
from ..valid_len import valid_len
from .layout import EP, NTHR, QT

RN = cute.math.RoundingMode.NEAREST_EVEN  # multiplies and FMAs that keep subnormal results

PART_N = 68  # FP32 words per thread of a piece's partials: 64 of O, 2 anchors, 2 row sums


def inv_row_sums(l0, l1, q):
    """1 / l of the O^T fragment columns (as in rescale_o), l the row sums of rows g, g + 8 from the quad's partial
    sums l0, l1: every lane divides for its own rows, then fetches the reciprocals of its columns."""
    l0 = l0 + cute.arch.shuffle_sync_bfly(l0, offset=1)
    l1 = l1 + cute.arch.shuffle_sync_bfly(l1, offset=1)
    l0 = l0 + cute.arch.shuffle_sync_bfly(l0, offset=2)
    l1 = l1 + cute.arch.shuffle_sync_bfly(l1, offset=2)
    r0 = Float32(1.0) / l0
    r1 = Float32(1.0) / l1
    return (
        cute.arch.shuffle_sync(r0, 8 * q),
        cute.arch.shuffle_sync(r0, 8 * q + 4),
        cute.arch.shuffle_sync(r1, 8 * q),
        cute.arch.shuffle_sync(r1, 8 * q + 4),
    )


@cute.jit
def store_rows(
    O,
    f,
    vsc,
    ep,
    g,
    q,
    lane,
    warp,
    qt,
    ob,
    bb,
    hh,
    S: cutlass.Constexpr,
    H: cutlass.Constexpr,
    nv=None,
):
    """The warp's 16 output rows o = O f v8_scale as BF16 (f = inv_row_sums: normalized first, as v8_scale / l may
    underflow where (O / l) v8_scale does not), staged in shared memory at ep (rows EP bytes apart), then 16-byte
    global stores of the rows below S (query QT qt + 16 warp + row). nv: with lengths, the valid tokens of the batch
    element: the rows from there on are stored as zeros."""
    fa, fb, fc, fd = f
    for ct in cutlass.range_constexpr(8):
        for h in cutlass.range_constexpr(2):
            ch = ep + (16 * ct + 8 * h + g) * 2
            sptr(BFloat16, ch + (2 * q) * EP).store(
                cute.math.mul(
                    cute.math.mul(O[8 * ct + 2 * h], fa, rounding=RN), vsc, rounding=RN
                ).to(BFloat16)
            )
            sptr(BFloat16, ch + (2 * q + 1) * EP).store(
                cute.math.mul(
                    cute.math.mul(O[8 * ct + 2 * h + 1], fb, rounding=RN), vsc, rounding=RN
                ).to(BFloat16)
            )
            sptr(BFloat16, ch + (8 + 2 * q) * EP).store(
                cute.math.mul(
                    cute.math.mul(O[8 * ct + 4 + 2 * h], fc, rounding=RN), vsc, rounding=RN
                ).to(BFloat16)
            )
            sptr(BFloat16, ch + (9 + 2 * q) * EP).store(
                cute.math.mul(
                    cute.math.mul(O[8 * ct + 4 + 2 * h + 1], fd, rounding=RN), vsc, rounding=RN
                ).to(BFloat16)
            )
    cute.arch.sync_warp()
    for it in cutlass.range_constexpr(8):
        row = 2 * it + (lane >> 4)
        c16 = lane & 15
        w = load_vector(sptr(Int32, ep + row * EP + c16 * 16, 16), 4)
        oq = qt * QT + warp * 16 + row
        if cutlass.const_expr(nv is not None):
            w = [Int32(cutlass.select_(oq < nv, x, Int32(0))) for x in w]
        if oq < S:
            store_vector(gptr(Int32, ob + Int64((bb * S + oq) * H + hh) * 256 + c16 * 16, 16), w)


@cute.jit
def store_zero_rows(lane, warp, qt, ob, bb, hh, S: cutlass.Constexpr, H: cutlass.Constexpr):
    """The warp's 16 output rows of a unit without a valid query row: zeros, in the global stores of store_rows."""
    for it in cutlass.range_constexpr(8):
        oq = qt * QT + warp * 16 + 2 * it + (lane >> 4)
        if oq < S:
            dst = ob + Int64((bb * S + oq) * H + hh) * 256 + (lane & 15) * 16
            store_vector(gptr(Int32, dst, 16), [Int32(0)] * 4)


def store_partials(part_ptr, bid, tidx, O, anc0, anc1, l0, l1):
    """Piece bid's FP32 state for merge_partial_kernel, element-major for coalesced writes: O in its transposed
    fragment mapping, the anchors and the row sums (partial over the quad)."""
    pbase = part_ptr.toint() + Int64(bid * PART_N) * (NTHR * 4) + tidx * 4
    for i in range(64):
        dst = pbase + i * (NTHR * 4)
        o = O[i]
        cute.arch.store(gptr(Float32, dst), o)
    for i, x in enumerate((anc0, anc1, l0, l1)):
        cute.arch.store(gptr(Float32, pbase + (64 + i) * (NTHR * 4)), x)


@cute.jit
def _merge_rows(
    part_ptr,
    vs_ptr,
    ob,
    bid,
    tidx,
    warp,
    lane,
    g,
    q,
    qt,
    bb,
    hh,
    split: cutlass.Constexpr,
    S: cutlass.Constexpr,
    H: cutlass.Constexpr,
    nv,
):
    """The merge of the pieces of one unit (merge_partial_kernel): the partials brought to their common anchor and
    summed, the rows normalized and stored as in a whole unit, those from nv on as zeros."""
    pb = [
        part_ptr.toint() + Int64((bid * split + p) * PART_N) * (NTHR * 4) + tidx * 4
        for p in range(split)
    ]
    an0 = [load_invariant(Float32, pb[p] + 64 * (NTHR * 4)) for p in range(split)]
    an1 = [load_invariant(Float32, pb[p] + 65 * (NTHR * 4)) for p in range(split)]
    m0 = an0[0]
    m1 = an1[0]
    for p in cutlass.range_constexpr(1, split):
        m0 = cute.arch.fmax(m0, an0[p])
        m1 = cute.arch.fmax(m1, an1[p])
    O = cute.make_rmem_tensor(cute.make_layout(64), Float32)
    O.fill(0.0)
    l0 = Float32(0.0)
    l1 = Float32(0.0)
    for p in cutlass.range_constexpr(split):
        # the anchors are in log2 units already: the score scale is not applied again
        a0 = cute.arch.exp2(an0[p] - m0)
        a1 = cute.arch.exp2(an1[p] - m1)
        l0 = cute.math.fma(load_invariant(Float32, pb[p] + 66 * (NTHR * 4)), a0, l0, rounding=RN)
        l1 = cute.math.fma(load_invariant(Float32, pb[p] + 67 * (NTHR * 4)), a1, l1, rounding=RN)
        # the factors belong to the rows g, g + 8 of a quad; the transposed O holds the queries 2q, 2q + 1, 8 + 2q,
        # 9 + 2q
        fa = cute.arch.shuffle_sync(a0, 8 * q)
        fb = cute.arch.shuffle_sync(a0, 8 * q + 4)
        fc = cute.arch.shuffle_sync(a1, 8 * q)
        fd = cute.arch.shuffle_sync(a1, 8 * q + 4)
        for i in cutlass.range_constexpr(64):
            if cutlass.const_expr(i % 8 < 4):
                factor = fa if i % 2 == 0 else fb
            else:
                factor = fc if i % 2 == 0 else fd
            O[i] = cute.math.fma(
                load_invariant(Float32, pb[p] + i * (NTHR * 4)), factor, O[i], rounding=RN
            )
    f = inv_row_sums(l0, l1, q)
    vsc = load_invariant(Float32, vs_ptr.toint())
    smem = Int32(cute.arch.get_dyn_smem(cutlass.Uint8, alignment=16).toint())
    store_rows(O, f, vsc, smem + warp * (16 * EP), g, q, lane, warp, qt, ob, bb, hh, S, H, nv)


@cute.kernel
def merge_partial_kernel(
    part_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    vs_ptr: cute.Pointer,
    S: cutlass.Constexpr,
    H: cutlass.Constexpr,
    nT: cutlass.Constexpr,
    split: cutlass.Constexpr,
    unit0: cutlass.Constexpr,
    lens_ptr,
):
    """The merge of the split key-range pieces of unit unit0 + CTA index, in the thread mapping of the attention: the
    partials of the pieces are brought to their common anchor (the anchors are in log2 units already) and summed, then
    the rows are normalized and stored as in a whole unit. lens_ptr: None, or the valid prefix lengths: the rows
    from the length of the batch element on are zeros (a piece without a valid key tile has left an anchor of -inf
    and zeros; a unit without a valid query row has no partials and stores zeros without reading any)."""
    bid_, _, _ = cute.arch.block_idx()
    tidx_, _, _ = cute.arch.thread_idx()
    bid = Int32(bid_)
    tidx = Int32(tidx_)
    unit = unit0 + bid
    qt = unit % nT
    hh = (unit // nT) % H
    bb = unit // (nT * H)
    warp = tidx >> 5
    lane = tidx & 31
    g = lane >> 2
    q = lane & 3
    ob = o_ptr.toint()
    if cutlass.const_expr(lens_ptr is None):
        _merge_rows(
            part_ptr, vs_ptr, ob, bid, tidx, warp, lane, g, q, qt, bb, hh, split, S, H, None
        )
    else:
        nv = valid_len(lens_ptr.toint(), bb, S)
        if qt * QT >= nv:
            # the unit has no valid query row: its pieces wrote no partials, its rows are zeros
            store_zero_rows(lane, warp, qt, ob, bb, hh, S, H)
        else:
            _merge_rows(
                part_ptr, vs_ptr, ob, bid, tidx, warp, lane, g, q, qt, bb, hh, split, S, H, nv
            )
