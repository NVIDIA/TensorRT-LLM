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
"""V pass of the DeCo preparation: E4M3 V with smoothing over V_BLOCK keys. Every block of V_BLOCK keys of a (batch,
head) loses its mean (rounded to BF16) before the E4M3 cast, so the cast spends its range on the variation inside a
block; the attention adds the means back through a BF16 MMA over the block sums of P. The kernel writes v8 = E4M3((V -
mean) / s) and v_mean = mean / s, with s the power-of-two scale of the largest |V - mean| of the call (the encode
launch writes it to v8_scale). The launch is a programmatic dependent of the eigensolver launch and runs beside it."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64, Uint32

from ...common.dsl import nanosleep
from ...common.expmath import bf16x2, e4m3x4, f2u, pow2_ceil_448, u2f, warp_max
from ...common.mem import gptr, load_invariant, load_vector, sptr, store_vector
from ..valid_len import valid_len

V_BLOCK = 128  # keys per mean: one 128-key tile of the attention
V_THREADS = 256
# CTAs per SM that the launch requests (min_blocks_per_mp); the grid is sized so that all its CTAs run at once, which
# the grid-wide maximum requires
V_CTAS_PER_SM = 4


@cute.jit
def _grid_inv8(
    x: Float32, maxima: Int32, gb: Int64, tidx: Int32, lane: Int32, warp: Int32, gdim: Int32
):
    """1 / the power-of-two E4M3 scale of the grid-wide maximum of x, for a grid of 256-thread CTAs that all run at
    once (V_CTAS_PER_SM): the warp maxima meet in maxima (shared memory), thread 0 folds the CTA's into byte 24 of
    gmax (gb), counts the CTA at byte 32 and polls the count until every CTA has added its maximum."""
    wm = warp_max(x)
    if lane == 0:
        cute.arch.store(sptr(Float32, maxima + warp * 4), wm)
    cute.arch.sync_threads()
    if tidx == 0:
        block_max = cute.arch.load(sptr(Float32, maxima), Float32)
        for w in cutlass.range_constexpr(1, 8):
            block_max = cute.arch.fmax(
                block_max, cute.arch.load(sptr(Float32, maxima + w * 4), Float32)
            )
        # the maxima are non-negative floats: the unsigned maximum of their bits orders them
        word = gb + Int64(24)
        bits = f2u(block_max)
        cute.arch.red(gptr(Uint32, word), bits, op="max", dtype="u32", sem="relaxed", scope="gpu")
        # the release part of the add orders this thread's max update before it; the acquiring loads of the poll
        # observe every CTA's maximum before the cast begins
        seen = cute.arch.atomic_add(
            gptr(Uint32, gb + Int64(32)), Uint32(1), sem="acq_rel", scope="gpu"
        ) + Uint32(1)
        while seen < Uint32(gdim):
            nanosleep(256)
            seen = cute.arch.load(gptr(Uint32, gb + Int64(32)), Uint32, sem="acquire", scope="gpu")
    cute.arch.sync_threads()
    vmax = u2f(cute.arch.load(gptr(Uint32, gb + Int64(24)), Uint32, sem="acquire", scope="gpu"))
    return Float32(1.0) / pow2_ceil_448(vmax)


@cute.jit
def _v_body(v, v8, v_mean, gmax, S, H, NB, nunits, lens):
    """E4M3 V minus its BF16 means of every V_BLOCK keys. Unit u = (batch u // nblk, block u % nblk of V_BLOCK keys)
    over every head; a CTA takes units bidx, bidx + gdim, ... and its threads the same 16-byte chunks (8 channels of
    a head) of a unit in both passes. Pass 1: per chunk the sum, minimum and maximum over the block's valid keys, the
    BF16 mean mu into v_mean (row = block) and max(max - mu, mu - min) into the grid's maximum (byte 24 of gmax,
    behind the counter at byte 32: all CTAs of the grid run at once). Pass 2: E4M3((V - mu) / s), s the power-of-two
    scale of that maximum (the encode launch's v8_scale), and mu / s into v_mean (exact). v_mean is [B, H, NB, 128].
    lens: None, or the global address of the valid prefix lengths: the valid keys of a block are then those below the
    length of its batch element, the keys after them get zero codes and a block without valid keys a zero mean."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    lane = tidx & 31
    warp = tidx >> 5
    smem = utils.SmemAllocator()
    maxima_p = smem.allocate_array(Float32, 8, byte_alignment=16)
    maxima = Int32(maxima_p.toint())
    vb = Int64(v.iterator.toint())
    ob = Int64(v8.iterator.toint())
    mb = Int64(v_mean.iterator.toint())
    gb = Int64(gmax.iterator.toint())
    nblk = (S + Int32(V_BLOCK - 1)) // Int32(V_BLOCK)
    nchunk = H * Int32(16)
    row_bytes = Int64(H) * Int64(256)
    acc = Float32(0.0)

    u = bidx
    while u < nunits:
        b = u // nblk
        j = u - b * nblk
        s0 = j * Int32(V_BLOCK)
        nvalid = cutlass.min(Int32(V_BLOCK), S - s0)
        rcp_n = Float32(1.0) / Float32(nvalid)
        if cutlass.const_expr(lens is not None):
            nvalid = cutlass.max(cutlass.min(nvalid, valid_len(lens, b, S) - s0), Int32(0))
            rcp_n = Float32(1.0) / Float32(cutlass.max(nvalid, Int32(1)))
        q = tidx
        while q < nchunk:
            hh = q >> 4
            cg = q & 15
            base = (
                (Int64(b) * Int64(S) + Int64(s0)) * row_bytes
                + Int64(hh) * Int64(256)
                + Int64(cg) * Int64(16)
            )
            sums = cute.make_rmem_tensor((8,), Float32)
            mins = cute.make_rmem_tensor((8,), Float32)
            maxs = cute.make_rmem_tensor((8,), Float32)
            for e in cutlass.range_constexpr(8):
                sums[e] = Float32(0.0)
                mins[e] = Float32(3.0e38)
                maxs[e] = Float32(-3.0e38)
            for r in cutlass.range(nvalid, unroll=8):
                w = load_invariant(Uint32, vb + base + Int64(r) * row_bytes, 4)
                for e in cutlass.range_constexpr(4):
                    lo = u2f(w[e] << Uint32(16))
                    hi = u2f(w[e] & Uint32(0xFFFF0000))
                    sums[2 * e] = sums[2 * e] + lo
                    sums[2 * e + 1] = sums[2 * e + 1] + hi
                    mins[2 * e] = cute.arch.fmin(mins[2 * e], lo)
                    mins[2 * e + 1] = cute.arch.fmin(mins[2 * e + 1], hi)
                    maxs[2 * e] = cute.arch.fmax(maxs[2 * e], lo)
                    maxs[2 * e + 1] = cute.arch.fmax(maxs[2 * e + 1], hi)
            words = []
            for e in cutlass.range_constexpr(4):
                wbits = bf16x2(sums[2 * e] * rcp_n, sums[2 * e + 1] * rcp_n)
                words.append(wbits)
                mlo = u2f(wbits << Uint32(16))
                mhi = u2f(wbits & Uint32(0xFFFF0000))
                acc = cute.arch.fmax(acc, cute.arch.fmax(maxs[2 * e] - mlo, mlo - mins[2 * e]))
                acc = cute.arch.fmax(
                    acc, cute.arch.fmax(maxs[2 * e + 1] - mhi, mhi - mins[2 * e + 1])
                )
            mrow = (Int64(b) * Int64(H) + Int64(hh)) * Int64(NB) + Int64(j)
            store_vector(gptr(Uint32, mb + mrow * Int64(256) + Int64(cg) * Int64(16), 16), words)
            q = q + Int32(256)
        u = u + gdim

    inv8 = _grid_inv8(acc, maxima, gb, tidx, lane, warp, gdim)

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
            # this thread's own pass-1 write: a coherent load
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
            q = q + Int32(256)
        u = u + gdim

    # every thread waits before the grid retires, so the next plain launch cannot race the eigensolver grid
    cute.arch.griddepcontrol_wait()


@cute.kernel
def v_kernel(
    v: cute.Tensor,
    v8: cute.Tensor,
    v_mean: cute.Tensor,
    gmax: cute.Tensor,
    S: Int32,
    H: Int32,
    NB: Int32,
    nunits: Int32,
):
    """The V pass (_v_body) of a call whose batch elements are all full."""
    _v_body(v, v8, v_mean, gmax, S, H, NB, nunits, None)


@cute.kernel
def v_kernel_lens(
    v: cute.Tensor,
    v8: cute.Tensor,
    v_mean: cute.Tensor,
    gmax: cute.Tensor,
    lens: cute.Tensor,
    S: Int32,
    H: Int32,
    NB: Int32,
    nunits: Int32,
):
    """The V pass (_v_body) of a call with valid prefix lengths lens [B] (int32)."""
    _v_body(v, v8, v_mean, gmax, S, H, NB, nunits, Int64(lens.iterator.toint()))
