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
"""Q'/K' pass of the DeCo preparation: rotation, NVFP4 quantization and compensation channels of every 128-token tile.

* A tile of q or k is rotated by the Hadamard matrix (BF16 MMA) and quantized to NVFP4 in 16-channel blocks; the
  quantization residual dq and the body x - dq are projected on the directions of the two compensation terms (TF32
  MMAs) and quantized the same way into the compensation channels.
* RANK = (r_q, r_k): Q' = [x - dq | dq B_q | (x - dq) B_k] and K' = [x - dk | (x - dk) B_q | dk B_k], B_q the r_q
  directions of the Q-residual term and B_k the r_k directions of the K-residual term: Q' K'^T adds the two
  first-order error terms dq (k - dk)^T and (q - dq) dk^T, projected on B_q and on B_k, to the product of the
  quantized bodies. A term of 0 directions is absent.
* Basis: the leading r_q and r_k directions of one basis (the last rows of its U^T), or with SEPARATE each term's
  directions of its own basis.
* Out: E2M1 codes to q4 / k4 (the head channels, the two terms, then zero codes up to the channels of the QK build,
  constants.qk_channels); block scales, as BF16, in a write-coalesced tile layout that the Q/K scale encoding
  (encode.py) converts to UE4M3; the maxima of each tensor's three scale groups (head channels, Q-residual term,
  K-residual term) to gmax.
* QPRE, the score prescale log2(e) / sqrt(128), multiplies the rotated Q before its quantization: the scores arrive
  in the log2 units the attention exponentiates while the score scale stays a power of two.

TMEM columns of a CTA:
    * Two rotated tiles (2 x 128), the residual dq as the A operand of the projections (128) and the projections of a
      tile in the order of its compensation channels (r_q + r_k columns per buffer).
    * Up to 64 projected channels: two projection buffers alternate, so the MMA warp projects a tile while the
      compensation warps still read the previous one. Above: one buffer in the 512 columns; the MMA warp reuses it
      once those warps loaded it.

Kernel body (qk_kernel): the setup of the CTA and one role per warp, each a function of this module.
    * A CTA takes a contiguous range of tiles in the order (batch, head), 128-token block, q then k; every role walks
      that range with its own copy of the tile iterator (_first_tile, _next_tile).
    * With valid prefix lengths the CTAs share the live tiles in the same order (_live_tiles), and the load warp
      stores zero block scales for the other tiles (_zero_scales).
    * The state object holds what the roles share: the thread's warp and lane, the shared-memory and TMEM addresses,
      the tile range and the trace-time constants.

Mbarriers, on which the roles meet. A full barrier completes when its producer has published the data, an empty barrier
when its consumers have released it; the arrivals of a phase are in parentheses.
    XFULL, XEMPTY    per X ring stage: the tile is copied (the TMA bytes); the rotation that read it is done (MMA warp)
    XRFULL, XREMPTY  per rotated tile buffer: the rotation is done (MMA warp); the projections that read the buffer are
                     done (MMA warp)
    AFULL, AEMPTY    the residual dq, and the prescaled values of a Q tile, are in TMEM (the 16 math warps); the
                     projections that read them are done (MMA warp)
    DFULL, DEMPTY    per projection buffer: the projections are done (MMA warp); their last channels are in registers
                     (the 4 compensation warps)
    UFULL, UEMPTY    per U operand stage: the basis of a (batch, head) is staged (load warp); the last projection that
                     reads the stage is done (MMA warp)
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint16, Uint32
from cutlass.cute.nvgpu import cpasync

from ... import target
from ...common.dsl import Tcgen05Fence, tcgen05_fence
from ...common.expmath import f2u, warp_max
from ...common.mem import gptr, load_invariant, sptr, store_vector
from ...common.state import State
from ...constants import HEAD_DIM, QK_BLOCK, QPRE, qk_channels
from ..ptx import idesc_bf16, idesc_tf32, mbar_wait_bounded, smem_desc_sw128
from ..tmem import (
    ld_tmem_16x32bx2_f32_x32_absmax_wait,
    ld_tmem_f32_x32_wait,
    st_tmem_16x32bx2_f32_x32_wait,
    tc_commit,
    tc_mma,
)
from ..valid_len import valid_len
from .basis import GMAX_K, GMAX_Q, eig_directions
from .encode import sfv_tile_bytes
from .quant import quant16, quant16_dq_knownmax

QK_THREADS = 22 * 32  # 16 math warps, load warp 16, MMA warp 17, compensation warps 18-21
X_STAGES = 3  # input tile ring depth
# Rows of the U operand up to which two stages of it fit the shared memory of a CTA beside the X ring and the rotation
U_TWO_STAGE_ROWS = 96
TM_XR, TM_DQ, TM_D = 0, 256, 384
# mbarrier slots (module docstring: Mbarriers)
B_XFULL, B_XEMPTY, B_XRFULL, B_XREMPTY = 0, X_STAGES, 2 * X_STAGES, 2 * X_STAGES + 2
B_AFULL, B_AEMPTY, B_DFULL, B_DEMPTY = (
    2 * X_STAGES + 4,
    2 * X_STAGES + 5,
    2 * X_STAGES + 6,
    2 * X_STAGES + 8,
)
B_UFULL, B_UEMPTY = 2 * X_STAGES + 10, 2 * X_STAGES + 12
QK_NBARS = 2 * X_STAGES + 14


def _d_buffer(parity, stages):
    """Return the projection buffer of a tile of this parity in the ring of `stages` buffers (two or one)."""
    return parity if stages == 2 else 0


def _d_rounds(i, stages):
    """Return the earlier uses of tile i's projection buffer in the ring of `stages` buffers."""
    return i >> 1 if stages == 2 else i


def _u_stage(ucount, stages):
    """Return the U operand stage of the CTA's (batch, head) number ucount in the ring of `stages` stages (two or
    one)."""
    return ucount & 1 if stages == 2 else 0


def _u_rounds(ucount, stages):
    """Return the earlier uses of the U operand stage of (batch, head) number ucount in the ring of `stages` stages."""
    return ucount >> 1 if stages == 2 else ucount


def _u_rows(rank, separate):
    """Return the rows of the U operand of the projections: (rows staged, first row of the Q-residual term, first row
    of the K-residual term, per block of 16 staged rows its (basis, first row counted from the end of the basis)).

    * One basis: its last max(r_q, r_k) rows are staged; a term takes the last r of them.
    * Separate bases: the last r_q rows of the basis of the Q-residual term (0), then the last r_k rows of the basis
      of the K-residual term (1).
    """
    r_q, r_k = rank
    if separate:
        source = [(0, r_q - row) for row in range(0, r_q, QK_BLOCK)]
        source += [(1, r_k - row) for row in range(0, r_k, QK_BLOCK)]
        return r_q + r_k, 0, r_q, source
    rows = max(r_q, r_k)
    return rows, rows - r_q, rows - r_k, [(0, rows - row) for row in range(0, rows, QK_BLOCK)]


# ------------------------------------------------------------------------------------------------ setup
def _init_barriers(bars):
    """Thread 0: every mbarrier of the CTA, with the arrivals of one phase (16 on AFULL, 4 on each DEMPTY, else one)."""
    for i in range(X_STAGES):
        cute.arch.mbarrier_init(sptr(Int64, bars + 8 * (B_XFULL + i)), 1)
        cute.arch.mbarrier_init(sptr(Int64, bars + 8 * (B_XEMPTY + i)), 1)
    for i in range(2):
        cute.arch.mbarrier_init(sptr(Int64, bars + 8 * (B_XRFULL + i)), 1)
        cute.arch.mbarrier_init(sptr(Int64, bars + 8 * (B_XREMPTY + i)), 1)
        cute.arch.mbarrier_init(sptr(Int64, bars + 8 * (B_DFULL + i)), 1)
        cute.arch.mbarrier_init(sptr(Int64, bars + 8 * (B_DEMPTY + i)), 4)
        cute.arch.mbarrier_init(sptr(Int64, bars + 8 * (B_UFULL + i)), 1)
        cute.arch.mbarrier_init(sptr(Int64, bars + 8 * (B_UEMPTY + i)), 1)
    cute.arch.mbarrier_init(sptr(Int64, bars + 8 * B_AFULL), 16)
    cute.arch.mbarrier_init(sptr(Int64, bars + 8 * B_AEMPTY), 1)
    cute.arch.mbarrier_init_fence()


@cute.jit
def _stage_hadamard(hadamard, tidx, ht_base):
    """Every thread: stage its 16-byte chunks (tidx, tidx + QK_THREADS, ...) of the Hadamard matrix as the B operand of
    the rotation at the shared address ht_base: element (k, n) = hadamard[k, n] at row n, column k (K-major, SW128)."""
    had = Int64(hadamard.iterator.toint())
    hws = []
    for it in cutlass.range_constexpr(4):
        ch = cutlass.min(tidx + QK_THREADS * it, 2047)
        hws.append(load_invariant(Uint32, had + Int64(ch * 16), 4))
    for it in cutlass.range_constexpr(4):
        ch = tidx + QK_THREADS * it
        if ch < 2048:
            kk = ch >> 4
            kb = kk & 63
            for e in cutlass.range_constexpr(8):
                nn = (ch & 15) * 8 + e
                hv = hws[it][e // 2] >> Uint32(16 * (e % 2)) & Uint32(0xFFFF)
                dst = (
                    ht_base
                    + (kk >> 6) * 16384
                    + nn * 128
                    + ((((kb >> 3) ^ (nn & 7)) << 4) | ((kb & 7) * 2))
                )
                cute.arch.store(sptr(Uint16, dst), hv.to(Uint16))


# ------------------------------------------------------------------------------------------------ steps of the roles
def _tile_origin(st):
    """Return (bh, rem) of the CTA's first tile t0: its (batch, head) and its position among that bh's 2 NB tiles."""
    bh = st.t0 // st.nb2
    return bh, st.t0 - bh * st.nb2


def _blocks(lens, b, S):
    """With lengths: the 128-token blocks of batch element b that hold a valid token."""
    return (valid_len(lens, b, S) + 127) >> 7


def _live_blocks(st, b):
    return _blocks(st.lens, cutlass.min(b, st.B - 1), st.S)


def _first_tile(st):
    """Return the CTA's first tile as the iterator (kind, sblk, bh, h, b), fastest first: kind 0 for the q tile of a
    block and 1 for its k tile, 128-token block sblk, (batch, head) index bh = b H + h, head h, batch b."""
    if st.lens is not None:
        return st.first
    bh, rem = _tile_origin(st)
    sblk = rem >> 1
    kind = rem & 1
    b = bh // st.H
    h = bh - b * st.H
    return kind, sblk, bh, h, b


@cute.jit
def _next_tile(st, kind, sblk, bh, h, b):
    """Return the tile after (kind, sblk, bh, h, b): the k tile of the block, then the q tile of the next block, of the
    next head, of the next batch. The iterator advances without divisions. With lengths a head ends with the last live
    block of its batch element, and the batch elements without a valid token are passed over."""
    kind = kind ^ 1
    if kind == 0:
        sblk = sblk + 1
        blocks = st.NB
        if cutlass.const_expr(st.lens is not None):
            blocks = _live_blocks(st, b)
        if sblk == blocks:
            sblk = 0
            bh = bh + 1
            h = h + 1
            if h == st.H:
                h = 0
                b = b + 1
                if cutlass.const_expr(st.lens is not None):
                    while (b < st.B) & (_live_blocks(st, b) == 0):
                        b = b + 1
                        bh = bh + st.H
    return kind, sblk, bh, h, b


@cute.jit
def _warp_arrive(st, slot):
    """Arrive once for the warp on mbarrier `slot`: by lane 0, once every lane of the warp has reached this point."""
    cute.arch.sync_warp()
    if st.lane == 0:
        cute.arch.mbarrier_arrive(sptr(Int64, st.bars + 8 * slot))


@cute.jit
def _tile_outputs(st, codes, scales, rstride, kind, sblk, b, h):
    """Return where the results of a tile go: the global addresses of its tensor's code rows and scale tiles (kind 0:
    Q, 1: K; codes: the addresses of q4 and k4, scales: of q_sfv and k_sfv) and the offset of the code row of the tile's
    first token at head h. A code row is the 128 bytes of a (token, head); rstride: the bytes of a token, 128 H."""
    code_base = codes[0]
    scale_base = scales[0]
    if kind == 1:
        code_base = codes[1]
        scale_base = scales[1]
    row0 = (Int64(b) * Int64(st.S) + Int64(sblk * 128)) * rstride + Int64(h * 128)
    return code_base, scale_base, row0


def _scale_tile(st, bh, sblk):
    """Return the index of the first BF16 scale of a tile among the scale tiles of its tensor. A tile holds sfv_tile
    scales: 1024 of the body, (32-channel group, token, 16-channel block), then the 128 tokens of each 16-channel
    block of the compensation channels."""
    return (Int64(bh) * Int64(st.NB) + Int64(sblk)) * Int64(st.sfv_tile)


@cute.jit
def _reduce_maxima(st, gmax, maxima):
    """Reduce each maximum over the warp and, by lane 0, into its word of gmax across the CTAs. maxima: (word of gmax,
    the thread's maximum) pairs; gmax: its global address."""
    reduced = [(word, warp_max(value)) for word, value in maxima]
    if st.lane == 0:
        # the maxima are non-negative floats: the unsigned maximum of their bits orders them
        for word, m in reduced:
            max_dst = gmax + Int64(4 * word)
            max_bits = f2u(m)
            cute.arch.red(
                gptr(Uint32, max_dst),
                max_bits,
                op="max",
                dtype="u32",
                sem="relaxed",
                scope="gpu",
            )


# ------------------------------------------------------------------------------------------------ load warp
def _x_views(st, tma, tiles):
    """Return the views of the TMA copy of a tile into the X ring: (the ring by stage, `tiles` by block and (head,
    batch)); tiles: the 128-token tiles of the tensor of tma (q or k)."""
    return cpasync.tma_partition(
        tma.atom,
        0,
        cute.make_layout(1),
        cute.group_modes(st.sX, 0, 3),
        cute.group_modes(st.thr_mma.partition_A(tiles), 0, 3),
    )


@cute.jit
def _stage_basis(st, ut_addrs, bh, ucount):
    """Load warp: stage the rows of the basis of (batch, head) bh that the projections read (st.u_source; a basis is
    U^T, FP32 [st.directions][128] per (batch, head), at its global address in ut_addrs) into the U stage of ucount
    as the K-major SW128 TF32 B operand: 4 K-boxes of st.urows rows x 128 B. Waits UEMPTY of the stage and arrives on
    its UFULL."""
    us = _u_stage(ucount, st.u_stages)
    mbar_wait_bounded(st.bars + 8 * (B_UEMPTY + us), (_u_rounds(ucount, st.u_stages) + 1) & 1)
    for blk in cutlass.range_constexpr(st.urows // QK_BLOCK):
        basis, from_end = st.u_source[blk]
        src = ut_addrs[basis] + (
            Int64(bh) * Int64(st.directions) + Int64(st.directions - from_end)
        ) * Int64(512)
        uws = []
        for it in cutlass.range_constexpr(16):
            # lane: 16-byte chunk of row it of the block
            uws.append(load_invariant(Uint32, src + Int64(it * 512 + st.lane * 16), 4))
        for it in cutlass.range_constexpr(16):
            nrow = blk * 16 + it
            cc = st.lane
            w = uws[it]
            dst = (
                st.u_base
                + us * st.ustage
                + (cc >> 3) * (st.urows * 128)
                + nrow * 128
                + (((cc & 7) ^ (nrow & 7)) << 4)
            )
            store_vector(sptr(Uint32, dst, 16), (w[0], w[1], w[2], w[3]))
    cute.arch.fence_proxy("async.shared", space="cta")
    _warp_arrive(st, B_UFULL + us)


@cute.jit
def _load_warp(st, ut_addrs, scales):
    """Load warp (16): stage the U operand of each (batch, head) from the basis at ut_addrs; copy tiles to the X ring.

    * _stage_basis waits UEMPTY and arrives on UFULL at the first tile of a (batch, head): with two U stages before
      the copy of that tile, with one stage after it. The one stage is released by the projections of the tile
      before, and the MMA warp rotates the first tile ahead of them: its copy must be on the way.
    * A tile waits XEMPTY of its ring stage (the rotation that read the stage is done); its TMA copy completes XFULL.
    """
    gQ = cute.local_tile(st.tma_q.tma_tensor, (128, 128), (None, 0, None))
    gK = cute.local_tile(st.tma_k.tma_tensor, (128, 128), (None, 0, None))
    tQs, tQg = _x_views(st, st.tma_q, gQ)
    tKs, tKg = _x_views(st, st.tma_k, gK)
    bh_prev = Int32(-1)
    ucount = Int32(-1)
    kind, sblk, bh, h, b = _first_tile(st)
    for i in cutlass.range(st.n):
        if cutlass.const_expr(st.urows > 0 and st.u_stages == 2):
            if bh != bh_prev:
                ucount = ucount + 1
                bh_prev = bh
                _stage_basis(st, ut_addrs, bh, ucount)
        xs = i % X_STAGES
        mbar_wait_bounded(st.bars + 8 * (B_XEMPTY + xs), ((i // X_STAGES) + 1) & 1)
        if st.lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(
                sptr(Int64, st.bars + 8 * (B_XFULL + xs)), 32768
            )
        cute.arch.sync_warp()
        if kind == 0:
            cute.copy(
                st.tma_q.atom,
                tQg[(None, sblk, (h, b))],
                tQs[(None, xs)],
                tma_bar_ptr=st.bars_p + (B_XFULL + xs),
            )
        else:
            cute.copy(
                st.tma_k.atom,
                tKg[(None, sblk, (h, b))],
                tKs[(None, xs)],
                tma_bar_ptr=st.bars_p + (B_XFULL + xs),
            )
        if cutlass.const_expr(st.urows > 0 and st.u_stages == 1):
            if bh != bh_prev:
                ucount = ucount + 1
                bh_prev = bh
                _stage_basis(st, ut_addrs, bh, ucount)
        kind, sblk, bh, h, b = _next_tile(st, kind, sblk, bh, h, b)
    if cutlass.const_expr(st.lens is not None):
        _zero_scales(st, scales)


@cute.jit
def _zero_scales(st, scales):
    """Load warp, with lengths: store zero block scales for the tiles without a valid token among the CTA's range of
    all tiles (t0_all, n_all), which no role visits. scales: the global addresses of (q_sfv, k_sfv)."""
    z = Uint32(0)
    end = st.t0_all + st.n_all
    # per (batch, head) of the range: its tiles from the first block without a valid token on
    for bh in cutlass.range(st.t0_all // st.nb2, (end + st.nb2 - 1) // st.nb2):
        t0 = bh * st.nb2
        for t in cutlass.range(
            cutlass.max(st.t0_all, t0 + 2 * _live_blocks(st, bh // st.H)),
            cutlass.min(end, t0 + st.nb2),
        ):
            rem = t - t0
            base = scales[0]
            if rem & 1 == 1:
                base = scales[1]
            base = base + _scale_tile(st, bh, rem >> 1) * Int64(2)
            for c in cutlass.range(st.lane, st.sfv_tile // 8, 32):
                store_vector(gptr(Uint32, base + Int64(c * 16), 16), (z, z, z, z))


# ------------------------------------------------------------------------------------------------ MMA warp
def _qk_rotation(st, j, idesc):
    """MMA warp: rotate tile j, xr[j & 1] = X[j % X_STAGES] * Hd. Waits XFULL of the ring stage and XREMPTY of the
    rotated tile buffer; commits XRFULL of the buffer and XEMPTY of the stage."""
    xs = j % X_STAGES
    s = j & 1
    mbar_wait_bounded(st.bars + 8 * (B_XFULL + xs), (j // X_STAGES) & 1)
    mbar_wait_bounded(st.bars + 8 * (B_XREMPTY + s), ((j >> 1) + 1) & 1)
    tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
    xa = st.x_base + xs * 32768
    for kk in range(8):
        off = (kk >> 2) * 16384 + (kk & 3) * 32
        tc_mma(
            st.tm + TM_XR + s * 128,
            smem_desc_sw128(xa + off),
            smem_desc_sw128(st.ht_base + off),
            idesc,
            Int32(kk),
            kind="f16",
            a_tmem=False,
        )
    tc_commit(st.bars + 8 * (B_XRFULL + s))
    tc_commit(st.bars + 8 * (B_XEMPTY + xs))


def _by_kind(q_tile, for_q, for_k):
    """Return for_q for a Q tile and for_k for a K tile: a trace-time value when both are equal."""
    return for_q if for_q == for_k else Int32(cutlass.select_(q_tile, Int32(for_q), Int32(for_k)))


def _project(st, d, a, ua, row, idesc, accumulate):
    """MMA warp: the TMEM columns at d = (or, with accumulate, +=) the 128 TMEM columns at a times the rows of the U
    operand at the shared address ua from byte offset `row` on, as 16 MMAs of kind::tf32 (K = 8 each) of the
    descriptor idesc, which holds the number of columns."""
    for kk in range(16):
        bd = smem_desc_sw128(ua + (kk >> 2) * (st.urows * 128) + row + (kk & 3) * 32)
        tc_mma(d, a + 8 * kk, bd, idesc, Int32(1 if accumulate else kk), kind="tf32", a_tmem=True)


@cute.jit
def _qk_projections(st, kind, p, dp, us):
    """MMA warp: the projections of the tile in xr[p] (kind 0: a Q tile, 1: a K tile) into projection buffer dp, with
    the basis in U stage us, at the columns of the tile's compensation channels: the Q-residual term (r_q columns),
    then the K-residual term (r_k columns).

    * Q tile: the Q-residual term is the residual dq B_q, the K-residual term the body (x - dq) B_k.
    * K tile: the Q-residual term is the body (x - dk) B_q, the K-residual term the residual dk B_k.
    * The residual is FP32 in the A region; a body is x B + (-residual) B with x still in xr[p], the negated product
      by the descriptor's negate-A bit. A term of 0 columns has no MMA.
    * A tile of either kind takes the same MMAs, the residual projection first; the rows of the U operand, the
      descriptors and the columns of the two projections follow the kind as values.
    """
    r_q, r_k = st.RANK
    ua = st.u_base + us * st.ustage
    d = st.tm + TM_D + dp * st.comp
    dq = st.tm + TM_DQ
    x = st.tm + p * 128
    rows = (st.u_row_q * 128, st.u_row_k * 128)
    idesc = (idesc_tf32(128, r_q), idesc_tf32(128, r_k))
    neg = 1 << 13
    q_tile = kind == 0
    if cutlass.const_expr(r_q > 0 and r_k > 0):
        row_res = _by_kind(q_tile, rows[0], rows[1])
        row_body = _by_kind(q_tile, rows[1], rows[0])
        d_res = d + _by_kind(q_tile, 0, r_q)
        d_body = d + _by_kind(q_tile, r_q, 0)
        _project(st, d_res, dq, ua, row_res, Int32(_by_kind(q_tile, idesc[0], idesc[1])), False)
        _project(st, d_body, x, ua, row_body, Int32(_by_kind(q_tile, idesc[1], idesc[0])), False)
        _project(
            st,
            d_body,
            dq,
            ua,
            row_body,
            Int32(_by_kind(q_tile, idesc[1] | neg, idesc[0] | neg)),
            True,
        )
    elif cutlass.const_expr(r_q > 0):
        # the Q-residual term alone: the residual of a Q tile, the body of a K tile
        if q_tile:
            _project(st, d, dq, ua, rows[0], Int32(idesc[0]), False)
        else:
            _project(st, d, x, ua, rows[0], Int32(idesc[0]), False)
            _project(st, d, dq, ua, rows[0], Int32(idesc[0] | neg), True)
    elif cutlass.const_expr(r_k > 0):
        # the K-residual term alone: the body of a Q tile, the residual of a K tile
        if q_tile:
            _project(st, d, x, ua, rows[1], Int32(idesc[1]), False)
            _project(st, d, dq, ua, rows[1], Int32(idesc[1] | neg), True)
        else:
            _project(st, d, dq, ua, rows[1], Int32(idesc[1]), False)


@cute.jit
def _mma_warp(st):
    """MMA warp (17): rotate every tile and project it on the basis.

    * The rotation (_qk_rotation) runs one tile ahead: the math warps work on tile i + 1, the compensation warps on i.
    * The projections of tile i wait UFULL at the first tile of a (batch, head), AFULL (the math warps have stored the
      residual) and DEMPTY of the tile's projection buffer; they commit DFULL, AEMPTY and XREMPTY, and UEMPTY after
      the last tile of the (batch, head) or of the CTA.
    """
    idesc1 = Int32(idesc_bf16(128, 128))

    if st.n > 0:
        _qk_rotation(st, Int32(0), idesc1)
    bh_prev = Int32(-1)
    ucount = Int32(-1)
    if cutlass.const_expr(st.lens is not None):
        kind, sblk, bh, h, b = _first_tile(st)
        rem = Int32(0)
    else:
        bh, rem = _tile_origin(st)
        kind = rem & 1
    for i in cutlass.range(st.n):
        if i + 1 < st.n:
            _qk_rotation(st, i + 1, idesc1)
        if bh != bh_prev:
            ucount = ucount + 1
            bh_prev = bh
            if cutlass.const_expr(st.urows > 0):
                mbar_wait_bounded(
                    st.bars + 8 * (B_UFULL + _u_stage(ucount, st.u_stages)),
                    _u_rounds(ucount, st.u_stages) & 1,
                )
        us = _u_stage(ucount, st.u_stages)
        p = i & 1
        dp = _d_buffer(p, st.d_stages)
        mbar_wait_bounded(st.bars + 8 * B_AFULL, i & 1)
        mbar_wait_bounded(st.bars + 8 * (B_DEMPTY + dp), (_d_rounds(i, st.d_stages) + 1) & 1)
        tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
        _qk_projections(st, kind, p, dp, us)
        tc_commit(st.bars + 8 * (B_DFULL + dp))
        tc_commit(st.bars + 8 * B_AEMPTY)
        tc_commit(st.bars + 8 * (B_XREMPTY + p))
        # the tile's position in its (batch, head): the U stage is released after the last one
        if cutlass.const_expr(st.lens is not None):
            # with lengths the last one is the k tile of the last live block of the batch element
            if cutlass.const_expr(st.urows > 0):
                if (i + 1 == st.n) or ((kind == 1) & (sblk + 1 == _live_blocks(st, b))):
                    tc_commit(st.bars + 8 * (B_UEMPTY + us))
            kind, sblk, bh, h, b = _next_tile(st, kind, sblk, bh, h, b)
        else:
            rem = rem + 1
            if rem == st.nb2:
                rem = 0
            if cutlass.const_expr(st.urows > 0):
                if (i + 1 == st.n) or (rem == 0):
                    tc_commit(st.bars + 8 * (B_UEMPTY + us))
            if rem == 0:
                bh = bh + 1
            kind = kind ^ 1


# ------------------------------------------------------------------------------------------------ compensation warps
@cute.jit
def _compensation_warps(st, codes, scales, gmax):
    """Compensation warps (18-21), one per TMEM lane quadrant: quantize the projected channels of every tile to NVFP4.

    * A thread takes the full row of the projected channels of its token (the tile's compensation channels in D),
      loaded 32 channels at a time, and stores their NVFP4 codes and block scales; the maxima of the scales of each
      term go to gmax. codes, scales, gmax: the global addresses of (q4, k4), (q_sfv, k_sfv) and gmax.
    * The compensation channels after the two terms get zero codes; their scales are zero in the scale encoding.
    * A tile waits DFULL of its projection buffer; the warp arrives on DEMPTY once its last channels are in registers.
    """
    r_q = st.RANK[0]
    blocks = st.comp // QK_BLOCK  # 16-channel blocks of the two terms
    parts = (blocks + 1) // 2  # 32-channel loads that hold them
    lq = st.warp & 3
    row = lq * 32 + st.lane
    tl = (lq * 32) << 16
    mq1 = Float32(0.0)
    mq2 = Float32(0.0)
    mk1 = Float32(0.0)
    mk2 = Float32(0.0)
    kind, sblk, bh, h, b = _first_tile(st)
    rstride = Int64(st.H) * Int64(128)
    thr_off = Int64(row) * rstride
    for i in cutlass.range(st.n):
        dp = _d_buffer(i & 1, st.d_stages)
        mbar_wait_bounded(st.bars + 8 * (B_DFULL + dp), _d_rounds(i, st.d_stages) & 1)
        tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
        tok = sblk * 128 + row
        if cutlass.const_expr(st.lens is not None):
            n_tok = valid_len(st.lens, b, st.S)
        code_base, scale_base, row0 = _tile_outputs(st, codes, scales, rstride, kind, sblk, b, h)
        soff = _scale_tile(st, bh, sblk) + Int64(1024 + lq * 32 + st.lane)
        if cutlass.const_expr(parts == 0):
            _warp_arrive(st, B_DEMPTY + dp)
        for part in cutlass.range_constexpr(parts):
            dv = ld_tmem_f32_x32_wait(st.tm + tl + TM_D + dp * st.comp + part * 32)
            if cutlass.const_expr(part == parts - 1):
                # the last channels are in registers: release the D buffer to the MMA warp
                tcgen05_fence(Tcgen05Fence.BEFORE_THREAD_SYNC)
                _warp_arrive(st, B_DEMPTY + dp)
            ws = []
            for grp in cutlass.range_constexpr(2):
                j = part * 2 + grp
                if cutlass.const_expr(j < blocks):
                    sg, w0, w1 = quant16(dv[16 * grp : 16 * grp + 16])
                    if cutlass.const_expr(st.lens is not None):
                        sg = Float32(cutlass.select_(tok < n_tok, sg, Float32(0.0)))
                    ws.append(w0)
                    ws.append(w1)
                    sg_dst = scale_base + Int64((soff + Int64(j * 128)) * 2)
                    sg_bits = sg.to(BFloat16).bitcast(Uint16)
                    cute.arch.store(gptr(Uint16, sg_dst), sg_bits)
                    if kind == 1:
                        if cutlass.const_expr(j * QK_BLOCK < r_q):
                            mk1 = cute.arch.fmax(mk1, sg)
                        else:
                            mk2 = cute.arch.fmax(mk2, sg)
                    else:
                        if cutlass.const_expr(j * QK_BLOCK < r_q):
                            mq1 = cute.arch.fmax(mq1, sg)
                        else:
                            mq2 = cute.arch.fmax(mq2, sg)
            # the blocks of the load: 8 bytes each in the token row, after the 64 bytes of the head channels
            if tok < st.S:
                store_vector(
                    gptr(Uint32, code_base + row0 + thr_off + Int64(64 + part * 16), 4 * len(ws)),
                    ws,
                )
        # zero codes in the blocks after the two terms, 16 bytes at a time from an even block on
        if cutlass.const_expr(blocks < st.comp_blocks):
            if tok < st.S:
                z = Uint32(0)
                rest = code_base + row0 + thr_off + Int64(64)
                if cutlass.const_expr(blocks % 2 == 1):
                    store_vector(gptr(Uint32, rest + Int64(blocks * 8), 8), (z, z))
                for zb in cutlass.range_constexpr(blocks + blocks % 2, st.comp_blocks, 2):
                    store_vector(gptr(Uint32, rest + Int64(zb * 8), 16), (z, z, z, z))
        kind, sblk, bh, h, b = _next_tile(st, kind, sblk, bh, h, b)
    _reduce_maxima(
        st, gmax, ((GMAX_Q + 1, mq1), (GMAX_Q + 2, mq2), (GMAX_K + 1, mk1), (GMAX_K + 2, mk2))
    )


# ------------------------------------------------------------------------------------------------ math warps
def _prescaled(x, am0, am1):
    """Return the 32 values x and the maxima of their two blocks, each times the score prescale QPRE."""
    pre = Float32(QPRE)
    xs = []
    for j in range(0, 32, 2):
        ps = cute.arch.mul_packed_f32x2((x[j], x[j + 1]), (pre, pre))
        xs += [ps[0], ps[1]]
    return xs, am0 * pre, am1 * pre


@cute.jit
def _math_warps(st, codes, scales, gmax):
    """Math warps (0-15): quantize the rotated tiles to NVFP4 and write the residual for the projections.

    * A thread takes 32 channels of a token of the rotated tile, quantizes them to NVFP4 in two 16-channel blocks,
      writes the quantization residual to the A columns of the projections and stores the codes and the block scales;
      the maxima of the scales go to gmax. codes, scales, gmax: as in _compensation_warps.
    * A tile waits XRFULL of its rotated buffer, and AEMPTY (the projections of the tile before are done) before the
      residual is overwritten; the warp arrives on AFULL once its residual and a Q tile's prescaled values are in TMEM.
    """
    lq = st.warp & 3
    row_half = (st.warp >> 2) & 1
    n_pair = st.warp >> 3
    # 16x32bx2 TMEM shape: lanes 0..15 and 16..31 read the same 16 rows at two column bases, so a thread keeps
    # contiguous 16-channel blocks and code stores. row_half: the first or second 16 rows of the warp's 32 TMEM lanes.
    row_lane = row_half * 16 + (st.lane & 15)
    row = lq * 32 + row_lane
    cq = n_pair * 2 + (st.lane >> 4)
    tl = (lq * 32 + row_half * 16) << 16
    mq0 = Float32(0.0)
    mk0 = Float32(0.0)
    kind, sblk, bh, h, b = _first_tile(st)
    rstride = Int64(st.H) * Int64(128)
    thr_off = Int64(row) * rstride
    for i in cutlass.range(st.n):
        s = i & 1
        mbar_wait_bounded(st.bars + 8 * (B_XRFULL + s), (i >> 1) & 1)
        tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
        xr = st.tm + tl + TM_XR + s * 128 + n_pair * 64
        x, am0, am1 = ld_tmem_16x32bx2_f32_x32_absmax_wait(xr, target.has_tmem_ld_red())
        # Q tiles carry the score prescale: the rotated values are scaled before the quantization, and the scaled tile
        # goes back to TMEM, where the compensation MMAs read it (the AFULL arrive below orders the store before them)
        if kind == 0:
            x, am0, am1 = _prescaled(x, am0, am1)
            st_tmem_16x32bx2_f32_x32_wait(xr, x)
        sc0, w0, w1, dq0 = quant16_dq_knownmax(x[0:16], am0)
        sc1, w2, w3, dq1 = quant16_dq_knownmax(x[16:32], am1)
        mbar_wait_bounded(st.bars + 8 * B_AEMPTY, (i + 1) & 1)
        tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
        st_tmem_16x32bx2_f32_x32_wait(st.tm + tl + TM_DQ + n_pair * 64, dq0 + dq1)
        tcgen05_fence(Tcgen05Fence.BEFORE_THREAD_SYNC)
        _warp_arrive(st, B_AFULL)
        tok = sblk * 128 + row
        if cutlass.const_expr(st.lens is not None):
            # a token past the valid prefix of its batch element: zero block scales, outside the maxima
            live = tok < valid_len(st.lens, b, st.S)
            sc0 = Float32(cutlass.select_(live, sc0, Float32(0.0)))
            sc1 = Float32(cutlass.select_(live, sc1, Float32(0.0)))
        code_base, scale_base, row0 = _tile_outputs(st, codes, scales, rstride, kind, sblk, b, h)
        if tok < st.S:
            store_vector(
                gptr(Uint32, code_base + row0 + thr_off + Int64(cq * 16), 16), (w0, w1, w2, w3)
            )
        # provisional scales, write-coalesced tile layout (the Q/K scale encoding permutes them)
        soff = _scale_tile(st, bh, sblk) + Int64(((cq * 4 + lq) * 32 + row_lane) * 2)
        cute.arch.store(
            gptr(Uint32, scale_base + Int64(soff * 2)),
            sc0.to(BFloat16).bitcast(Uint16).to(Uint32)
            | (sc1.to(BFloat16).bitcast(Uint16).to(Uint32) << Uint32(16)),
        )
        m01 = cute.arch.fmax(sc0, sc1)
        if kind == 1:
            mk0 = cute.arch.fmax(mk0, m01)
        else:
            mq0 = cute.arch.fmax(mq0, m01)
        kind, sblk, bh, h, b = _next_tile(st, kind, sblk, bh, h, b)
    _reduce_maxima(st, gmax, ((GMAX_Q, mq0), (GMAX_K, mk0)))


# ------------------------------------------------------------------------------------------------ kernel
@cute.jit
def _qk_body(
    tiled_mma,
    tma_q,
    tma_k,
    hadamard,
    ut_addrs,
    q4,
    k4,
    q_sfv,
    k_sfv,
    gmax,
    S,
    H,
    NB,
    ntiles,
    RANK,
    lens,
):
    """Run the Q'/K' pass of the CTA's share of the ntiles tiles: the setup, the role of the thread's warp, the release.

    * RANK: (r_q, r_k), the channels of the two compensation terms. ut_addrs: the global address of the basis U^T
      [B, H, directions, 128] of both terms, directions = eig_directions(RANK), or of two of them: the basis of the
      Q-residual term and that of the K-residual term.
    * lens: None, or the global address of the valid prefix lengths: the block scales of the tokens past the length of
      their batch element are then zero and stay out of the maxima, and the CTAs share the live tiles (_live_tiles).
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32

    smem = utils.SmemAllocator()
    bars_p = smem.allocate_array(Int64, QK_NBARS, byte_alignment=8)
    hold_p = smem.allocate_array(Int32, 4, byte_alignment=16)
    sX = smem.allocate_tensor(
        BFloat16, tma_q.smem_layout.outer, byte_alignment=1024, swizzle=tma_q.smem_layout.inner
    )
    sHT = smem.allocate_array(Uint8, 32768, byte_alignment=1024)
    # the U operand (urows rows of 128 FP32 each per stage), and the projection buffers of comp columns after TM_D
    # that the 512 TMEM columns hold
    comp = RANK[0] + RANK[1]
    urows, u_row_q, u_row_k, u_source = _u_rows(RANK, len(ut_addrs) == 2)
    ustage = urows * 512
    u_stages = 2 if urows <= U_TWO_STAGE_ROWS else 1
    d_stages = 2 if TM_D + 2 * comp <= 512 else 1
    # BF16 block scales of a tile in the write-coalesced layout, and the 16-channel blocks of the compensation
    # channels of the QK build
    channels = qk_channels(RANK)
    sfv_tile = sfv_tile_bytes(channels) // 2
    comp_blocks = (channels - HEAD_DIM) // QK_BLOCK
    bars = Int32(bars_p.toint())
    hold = Int32(hold_p.toint())
    x_base = Int32(sX.iterator.toint())
    ht_base = Int32(sHT.toint())
    u_base = Int32(0)
    if cutlass.const_expr(urows > 0):
        sU = smem.allocate_array(Uint8, u_stages * ustage, byte_alignment=1024)
        u_base = Int32(sU.toint())

    # the tiles of this CTA: n tiles from t0
    t0 = (bidx * ntiles) // gdim
    n = ((bidx + 1) * ntiles) // gdim - t0
    nb2 = NB * 2

    if tidx == 0:
        _init_barriers(bars)
    if warp == 0:
        cute.arch.alloc_tmem(512, sptr(Uint32, hold))
        cute.arch.relinquish_tmem_alloc_permit()
    _stage_hadamard(hadamard, tidx, ht_base)
    # the barriers, the TMEM address and the Hadamard operand are set for every thread after this barrier
    cute.arch.fence_proxy("async.shared", space="cta")
    tcgen05_fence(Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.sync_threads()
    tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
    tm = Int32(cute.arch.load(sptr(Uint32, hold), Uint32))

    thr_mma = tiled_mma.get_slice(0)
    t0_all, n_all, first, batch = t0, n, None, None
    if cutlass.const_expr(lens is not None):
        batch = ntiles // (nb2 * H)
        n, first = _live_tiles(lens, S, H, batch, bidx, gdim)
    st = State(
        warp=warp,
        lane=lane,
        # shared memory: the mbarrier array (address, and pointer for the TMA copies), the X ring (tensor and
        # address), the Hadamard operand and the U operand stages; the TMEM base
        bars=bars,
        bars_p=bars_p,
        sX=sX,
        x_base=x_base,
        ht_base=ht_base,
        u_base=u_base,
        tm=tm,
        # the tiles of the CTA and the shape of the problem
        t0=t0,
        n=n,
        nb2=nb2,
        S=S,
        H=H,
        NB=NB,
        # with lengths: n counts the live tiles of the CTA from the tile `first`, (t0_all, n_all) is its range of
        # all tiles and B the batch size
        lens=lens,
        first=first,
        t0_all=t0_all,
        n_all=n_all,
        B=batch,
        # trace-time constants
        RANK=RANK,
        comp=comp,
        comp_blocks=comp_blocks,
        directions=eig_directions(RANK),
        urows=urows,
        u_row_q=u_row_q,
        u_row_k=u_row_k,
        u_source=u_source,
        ustage=ustage,
        u_stages=u_stages,
        d_stages=d_stages,
        sfv_tile=sfv_tile,
        # the TMA of q and k and the MMA slice that partitions their tiles
        tma_q=tma_q,
        tma_k=tma_k,
        thr_mma=thr_mma,
    )
    # One role per warp; the global addresses a role reads or writes are read from the tensors in its branch.
    if warp == 16:
        _load_warp(
            st,
            ut_addrs,
            (Int64(q_sfv.iterator.toint()), Int64(k_sfv.iterator.toint())),
        )
    elif warp == 17:
        _mma_warp(st)
    elif warp >= 18:
        _compensation_warps(
            st,
            codes=(Int64(q4.iterator.toint()), Int64(k4.iterator.toint())),
            scales=(Int64(q_sfv.iterator.toint()), Int64(k_sfv.iterator.toint())),
            gmax=Int64(gmax.iterator.toint()),
        )
    else:
        _math_warps(
            st,
            codes=(Int64(q4.iterator.toint()), Int64(k4.iterator.toint())),
            scales=(Int64(q_sfv.iterator.toint()), Int64(k_sfv.iterator.toint())),
            gmax=Int64(gmax.iterator.toint()),
        )
    # every role has finished its n tiles: warp 0 releases the TMEM columns
    tcgen05_fence(Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.sync_threads()
    tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
    if warp == 0:
        cute.arch.dealloc_tmem(
            cute.make_ptr(Float32, tm, cute.AddressSpace.tmem, assumed_align=16), 512
        )


@cute.jit
def _live_tiles(lens, S, H, batch, bidx, gdim):
    """With lengths: return (the number of live tiles of CTA bidx of gdim, its first one as the tile iterator). The
    live tiles, in the order of all tiles, are the q and k tiles of the blocks that hold a valid token, in every head
    of their batch element; the CTAs take contiguous ranges of them of the same size."""
    total = Int32(0)
    for b in cutlass.range(batch):
        total = total + H * 2 * _blocks(lens, b, S)
    l0 = (bidx * total) // gdim
    n = ((bidx + 1) * total) // gdim - l0
    # the batch element that holds live tile l0, and the live tiles before it
    b = Int32(0)
    before = Int32(0)
    per_head = 2 * _blocks(lens, cutlass.min(b, batch - 1), S)
    while (b < batch) & (l0 >= before + H * per_head):
        before = before + H * per_head
        b = b + 1
        per_head = 2 * _blocks(lens, cutlass.min(b, batch - 1), S)
    local = l0 - before
    h = local // cutlass.max(per_head, Int32(1))
    rem = local - h * per_head
    return n, (rem & 1, rem >> 1, b * H + h, h, b)


@cute.kernel
def qk_kernel(
    tiled_mma: cute.TiledMma,
    tma_q: cpasync.TmaInfo,
    tma_k: cpasync.TmaInfo,
    hadamard: cute.Tensor,
    ut_q: cute.Tensor,
    ut_k: Int64,
    q4: cute.Tensor,
    k4: cute.Tensor,
    q_sfv: cute.Tensor,
    k_sfv: cute.Tensor,
    gmax: cute.Tensor,
    S: Int32,
    H: Int32,
    NB: Int32,
    ntiles: Int32,
    lens: Int64,
    RANK: cutlass.Constexpr,
    LENS: cutlass.Constexpr,
    SEPARATE: cutlass.Constexpr,
):
    """Run the Q'/K' pass (_qk_body) of RANK = (r_q, r_k).

    * ut_q: the basis U^T of both terms; with SEPARATE that of the Q-residual term, and ut_k the address of that of
      the K-residual term (else ut_k is not read).
    * With LENS the call has valid prefix lengths, int32 [B] at the address lens.
    """
    _qk_body(
        tiled_mma,
        tma_q,
        tma_k,
        hadamard,
        (Int64(ut_q.iterator.toint()), ut_k) if SEPARATE else (Int64(ut_q.iterator.toint()),),
        q4,
        k4,
        q_sfv,
        k_sfv,
        gmax,
        S,
        H,
        NB,
        ntiles,
        RANK,
        lens if LENS else None,
    )
