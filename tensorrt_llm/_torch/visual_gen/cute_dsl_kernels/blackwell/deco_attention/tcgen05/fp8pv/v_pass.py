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
"""V pass of the DeCo preparation: E4M3 V with smoothing over V_BLOCK keys.

* Every block of V_BLOCK keys of a (batch, head) loses its mean (rounded to BF16) before the E4M3 cast, so the cast
  spends its range on the variation inside a block; a BF16 MMA of the attention over the block sums of P adds it back.
* Two launches of one grid over the same chunks per thread: the scan writes the BF16 means and folds the largest
  deviation from them into gmax, the encode reads that maximum and writes the codes and the scaled means. The scan is
  a programmatic dependent of the eigensolver launch and runs beside it; the encode follows both in stream order. No
  CTA waits for another CTA of its grid, so neither launch needs its whole grid resident.
* A unit of work is a block of one batch element. A CTA of either launch takes every gdim-th unit from its index
  on, so the encode of a unit runs where its scan ran. A scan that shares its SMs with the eigensolver CTAs takes
  its work from a counter of the call instead (counted), a unit in pieces of one chunk per thread.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64, Uint32

from ...common.expmath import bf16x2, e4m3x4, f2u, pow2_ceil_448, u2f, warp_max
from ...common.mem import gptr, load_invariant, load_vector, sptr, store_vector
from ...constants import KEY_TILE, V_MEAN_HEADROOM
from ..prep.basis import GMAX_UNIT, GMAX_V
from ..ptx import max_bf16x2, min_bf16x2
from ..valid_len import valid_len

V_BLOCK = KEY_TILE  # keys per mean: one KV tile of the attention
V_THREADS = 256
# CTAs per SM that the launches request (min_blocks_per_mp) and that size their grid
V_CTAS_PER_SM = 4


@cute.jit
def _grid_max(x: Float32, maxima: Int32, gb: Int64, tidx: Int32, lane: Int32, warp: Int32):
    """Fold the CTA's maximum of x into word GMAX_V of gmax (gb): the warp maxima meet in maxima (shared memory), thread
    0 reduces them and adds the result with an atomic maximum."""
    wm = warp_max(x)
    if lane == 0:
        cute.arch.store(sptr(Float32, maxima + warp * 4), wm)
    cute.arch.sync_threads()
    if tidx == 0:
        block_max = cute.arch.load(sptr(Float32, maxima), Float32)
        for w in cutlass.range_constexpr(1, V_THREADS // 32):
            block_max = cute.arch.fmax(
                block_max, cute.arch.load(sptr(Float32, maxima + w * 4), Float32)
            )
        # the maxima are non-negative floats: the unsigned maximum of their bits orders them
        word = gb + Int64(4 * GMAX_V)
        bits = f2u(block_max)
        cute.arch.red(gptr(Uint32, word), bits, op="max", dtype="u32", sem="relaxed", scope="gpu")


@cute.jit
def _take_piece(units: Int32, taken: Int32, gb: Int64, tidx: Int32, pieces: Int32, counted: Int32):
    """Return the next piece of the scan of the CTA, which has taken `taken` pieces; a unit has `pieces` pieces.

    * counted == 0: the pieces of every gdim-th unit from the CTA's index on, in order.
    * Otherwise thread 0 takes the piece from the counter of the call (word GMAX_UNIT of gmax at gb, an atomic add)
      and publishes it in shared memory, which every thread reads after the CTA barrier. units: the shared address
      of two words, used in turn.
    """
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    done = taken // pieces
    piece = (bidx + done * gdim) * pieces + (taken - done * pieces)
    if counted != 0:
        slot = units + (taken & 1) * 4
        if tidx == 0:
            unit = cute.arch.atomic_add(
                gptr(Int32, gb + Int64(4 * GMAX_UNIT)), Int32(1), sem="relaxed", scope="gpu"
            )
            cute.arch.store(sptr(Int32, slot), Int32(unit))
        cute.arch.sync_threads()
        piece = cute.arch.load(sptr(Int32, slot), Int32)
    return piece


@cute.jit
def _v_scan_body(v, v_mean, gmax, S, H, VP, NB, nunits, counted, lens):
    """The scan: per chunk (8 channels of a head) of every block the sum, minimum and maximum over the block's valid
    keys, the BF16 mean mu into v_mean (row = block) and max(max - mu, mu - min) into the grid's maximum (_grid_max).
    VP: the bytes between consecutive tokens of v. lens: None, or the global address of the valid prefix lengths: the
    valid keys of a block are then those below the length of its batch element, and a block without valid keys gets
    a zero mean. A piece of work (_take_piece) is V_THREADS chunks of a unit, one per thread; counted: nonzero when
    the CTAs take their pieces from the counter of the call."""
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx & 31
    warp = tidx >> 5
    smem = utils.SmemAllocator()
    maxima_p = smem.allocate_array(Float32, V_THREADS // 32, byte_alignment=16)
    maxima = Int32(maxima_p.toint())
    units = Int32(smem.allocate_array(Int32, 2, byte_alignment=8).toint())
    vb = Int64(v.iterator.toint())
    mb = Int64(v_mean.iterator.toint())
    gb = Int64(gmax.iterator.toint())
    nblk = (S + Int32(V_BLOCK - 1)) // Int32(V_BLOCK)
    nchunk = H * Int32(16)
    row_bytes = Int64(VP)
    acc = Float32(0.0)

    pieces = (nchunk + Int32(V_THREADS - 1)) // Int32(V_THREADS)  # of a unit
    taken = Int32(0)
    piece = _take_piece(units, taken, gb, tidx, pieces, counted)
    while piece < nunits * pieces:
        u = piece // pieces
        b = u // nblk
        j = u - b * nblk
        s0 = j * Int32(V_BLOCK)
        nvalid = cutlass.min(Int32(V_BLOCK), S - s0)
        rcp_n = Float32(1.0) / Float32(nvalid)
        if cutlass.const_expr(lens is not None):
            nvalid = cutlass.max(cutlass.min(nvalid, valid_len(lens, b, S) - s0), Int32(0))
            rcp_n = Float32(1.0) / Float32(cutlass.max(nvalid, Int32(1)))
        q = tidx + (piece - u * pieces) * Int32(V_THREADS)
        if q < nchunk:
            hh = q >> 4
            cg = q & 15
            base = (
                (Int64(b) * Int64(S) + Int64(s0)) * row_bytes
                + Int64(hh) * Int64(256)
                + Int64(cg) * Int64(16)
            )
            # the sums as FP32; the minima and maxima as BF16 pairs in the words of the keys, from +inf and -inf
            sums = cute.make_rmem_tensor((8,), Float32)
            minw = cute.make_rmem_tensor((4,), Uint32)
            maxw = cute.make_rmem_tensor((4,), Uint32)
            for e in cutlass.range_constexpr(8):
                sums[e] = Float32(0.0)
            for e in cutlass.range_constexpr(4):
                minw[e] = Uint32(0x7F807F80)
                maxw[e] = Uint32(0xFF80FF80)
            for r in cutlass.range(nvalid, unroll=8):
                w = load_invariant(Uint32, vb + base + Int64(r) * row_bytes, 4)
                for e in cutlass.range_constexpr(4):
                    lo = u2f(w[e] << Uint32(16))
                    hi = u2f(w[e] & Uint32(0xFFFF0000))
                    pair = cute.arch.add_packed_f32x2((sums[2 * e], sums[2 * e + 1]), (lo, hi))
                    sums[2 * e] = pair[0]
                    sums[2 * e + 1] = pair[1]
                    minw[e] = min_bf16x2(minw[e], w[e])
                    maxw[e] = max_bf16x2(maxw[e], w[e])
            words = []
            for e in cutlass.range_constexpr(4):
                wbits = bf16x2(sums[2 * e] * rcp_n, sums[2 * e + 1] * rcp_n)
                words.append(wbits)
                mlo = u2f(wbits << Uint32(16))
                mhi = u2f(wbits & Uint32(0xFFFF0000))
                # the extremes of the valid keys as FP32, within +-3e38 (those of a block without valid keys)
                bound = Float32(3.0e38)
                min_lo = cute.arch.fmin(bound, u2f(minw[e] << Uint32(16)))
                min_hi = cute.arch.fmin(bound, u2f(minw[e] & Uint32(0xFFFF0000)))
                max_lo = cute.arch.fmax(-bound, u2f(maxw[e] << Uint32(16)))
                max_hi = cute.arch.fmax(-bound, u2f(maxw[e] & Uint32(0xFFFF0000)))
                acc = cute.arch.fmax(acc, cute.arch.fmax(max_lo - mlo, mlo - min_lo))
                acc = cute.arch.fmax(acc, cute.arch.fmax(max_hi - mhi, mhi - min_hi))
                # the scaled means stay finite (V_MEAN_HEADROOM)
                acc = cute.arch.fmax(acc, cute.arch.fmax(mlo, -mlo) * Float32(V_MEAN_HEADROOM))
                acc = cute.arch.fmax(acc, cute.arch.fmax(mhi, -mhi) * Float32(V_MEAN_HEADROOM))
            mrow = (Int64(b) * Int64(H) + Int64(hh)) * Int64(NB) + Int64(j)
            store_vector(gptr(Uint32, mb + mrow * Int64(256) + Int64(cg) * Int64(16), 16), words)
        taken = taken + 1
        piece = _take_piece(units, taken, gb, tidx, pieces, counted)

    _grid_max(acc, maxima, gb, tidx, lane, warp)
    # every thread waits before the grid retires, so the next plain launch cannot race the eigensolver grid
    cute.arch.griddepcontrol_wait()


@cute.jit
def _v_encode_body(v, v8, v_mean, gmax, S, H, VP, NB, nunits, lens):
    """The encode: v8 = E4M3((V - mu) / s), s the power-of-two scale of the maximum of the scan (the encode launch's
    v8_scale), and v_mean = mu / s (exact). lens: as for _v_scan_body; the keys from the length on get zero codes."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    vb = Int64(v.iterator.toint())
    ob = Int64(v8.iterator.toint())
    mb = Int64(v_mean.iterator.toint())
    gb = Int64(gmax.iterator.toint())
    nblk = (S + Int32(V_BLOCK - 1)) // Int32(V_BLOCK)
    nchunk = H * Int32(16)
    row_bytes = Int64(VP)
    inv8 = Float32(1.0) / pow2_ceil_448(load_invariant(Float32, gb + Int64(4 * GMAX_V)))

    u = bidx
    while u < nunits:
        b = u // nblk
        j = u - b * nblk
        s0 = j * Int32(V_BLOCK)
        nrows = cutlass.min(Int32(V_BLOCK), S - s0)
        nvalid = nrows
        if cutlass.const_expr(lens is not None):
            nvalid = cutlass.max(cutlass.min(nrows, valid_len(lens, b, S) - s0), Int32(0))
        q = tidx
        while q < nchunk:
            hh = q >> 4
            cg = q & 15
            mrow = (Int64(b) * Int64(H) + Int64(hh)) * Int64(NB) + Int64(j)
            maddr = mb + mrow * Int64(256) + Int64(cg) * Int64(16)
            mw = load_vector(gptr(Uint32, maddr, 16), 4)
            mu = []
            for e in cutlass.range_constexpr(4):
                mu += [u2f(mw[e] << Uint32(16)), u2f(mw[e] & Uint32(0xFFFF0000))]
            base = (
                (Int64(b) * Int64(S) + Int64(s0)) * row_bytes
                + Int64(hh) * Int64(256)
                + Int64(cg) * Int64(16)
            )
            obase = ((Int64(b) * Int64(S) + Int64(s0)) * Int64(H) + Int64(hh)) * Int64(128) + Int64(
                cg
            ) * Int64(8)
            for r in cutlass.range(nvalid, unroll=8):
                w = load_invariant(Uint32, vb + base + Int64(r) * row_bytes, 4)
                f = []
                for e in cutlass.range_constexpr(4):
                    pr = cute.arch.mul_packed_f32x2(
                        (
                            u2f(w[e] << Uint32(16)) - mu[2 * e],
                            u2f(w[e] & Uint32(0xFFFF0000)) - mu[2 * e + 1],
                        ),
                        (inv8, inv8),
                    )
                    f += [pr[0], pr[1]]
                dst = ob + obase + Int64(r) * Int64(H) * Int64(128)
                codes = (e4m3x4(*f[0:4]), e4m3x4(*f[4:8]))
                store_vector(gptr(Uint32, dst, 8), codes)
            if cutlass.const_expr(lens is not None):
                for r in cutlass.range(nvalid, nrows, unroll=1):
                    dst = ob + obase + Int64(r) * Int64(H) * Int64(128)
                    store_vector(gptr(Uint32, dst, 8), (Uint32(0), Uint32(0)))
            means = tuple(bf16x2(mu[2 * e] * inv8, mu[2 * e + 1] * inv8) for e in range(4))
            store_vector(gptr(Uint32, maddr, 16), means)
            q = q + Int32(V_THREADS)
        u = u + gdim


@cute.kernel
def v_scan_kernel(
    v: cute.Tensor,
    v_mean: cute.Tensor,
    gmax: cute.Tensor,
    S: Int32,
    H: Int32,
    VP: Int32,
    NB: Int32,
    nunits: Int32,
    counted: Int32,
    lens: Int64,
    LENS: cutlass.Constexpr,
):
    """The scan (_v_scan_body); counted: nonzero when its CTAs take their work from the counter of the call; with
    LENS the call has valid prefix lengths, int32 [B] at the address lens."""
    _v_scan_body(v, v_mean, gmax, S, H, VP, NB, nunits, counted, lens if LENS else None)


@cute.kernel
def v_encode_kernel(
    v: cute.Tensor,
    v8: cute.Tensor,
    v_mean: cute.Tensor,
    gmax: cute.Tensor,
    S: Int32,
    H: Int32,
    VP: Int32,
    NB: Int32,
    nunits: Int32,
    lens: Int64,
    LENS: cutlass.Constexpr,
):
    """The encode (_v_encode_body); with LENS the call has valid prefix lengths, int32 [B] at the address lens."""
    _v_encode_body(v, v8, v_mean, gmax, S, H, VP, NB, nunits, lens if LENS else None)
