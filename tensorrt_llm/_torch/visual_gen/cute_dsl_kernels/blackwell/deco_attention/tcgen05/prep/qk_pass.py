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
"""Q'/K' pass of the DeCo preparation. Every 128-token tile of q and k is rotated by the Hadamard matrix (BF16 MMA) and
quantized to NVFP4 in 16-channel blocks; the quantization residual dq and the body x - dq are projected on the
compensation basis U of RANK directions (TF32 MMAs) and quantized the same way into the 2 x RANK compensation
channels: Q' = [x - dq | dq U | (x - dq) U] and K' = [x - dq | (x - dq) U | dq U], so that Q' K'^T adds the two
first-order error terms dq (k - dk)^T and (q - dq) dk^T, projected on U, to the product of the quantized bodies. The
E2M1 codes go to q4 / k4 (128 + 2 RANK channels per token); the block scales go, as BF16, to a write-coalesced tile
layout that the Q/K scale encoding (encode.py) converts to UE4M3, and the maxima of the three scale groups of each
tensor to gmax. QPRE, the score prescale log2(e) / sqrt(128), multiplies the rotated Q before its quantization, so the
scores arrive in the log2 units the attention exponentiates while the score scale stays a power of two.

The TMEM columns of a CTA: two rotated tiles (2 x 128), the residual dq as the A operand of the projections (128) and
the projections [dq U | (x - dq) U] (2 RANK columns per buffer). At rank 32 two projection buffers alternate, so the
MMA warp projects a tile while the compensation warps still read the previous one; the 512 columns hold one buffer at
rank 64, which the MMA warp reuses once the compensation warps have loaded it.

The kernel body (qk_kernel) is the setup of the CTA and one role per warp, each a function of this module: the load
warp, the MMA warp, the compensation warps and the math warps. A CTA takes a contiguous range of tiles in the order
(batch, head), 128-token block, q then k; every role walks that range with its own copy of the tile iterator
(_first_tile, _next_tile), and the roles meet on the mbarriers listed below. With valid prefix lengths the CTAs share
the live tiles, those that hold a token below the length of their batch element, in the same order: the iterator walks
only them, and the load warp stores zero block scales for the other tiles of the CTA's range of all tiles. The state
object of the kernel holds what the roles share: the thread's warp and lane, the shared-memory and TMEM addresses, the
tile range and the trace-time constants."""

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
from ...constants import QPRE, qk_scale_atoms
from ..ptx import idesc_bf16, idesc_tf32, mbar_wait_bounded, smem_desc_sw128
from ..tmem import (
    ld_tmem_16x32bx2_f32_x32_absmax_wait,
    ld_tmem_f32_x32_wait,
    st_tmem_16x32bx2_f32_x32_wait,
    tc_commit,
    tc_mma,
)
from ..valid_len import valid_len
from .quant import quant16, quant16_dq_knownmax

QK_THREADS = 22 * 32  # 16 math warps, load warp 16, MMA warp 17, compensation warps 18-21
X_STAGES = 3  # input tile ring depth
TM_XR, TM_AD1, TM_D = 0, 256, 384
# mbarrier slots. A full barrier completes when its producer has published the data, an empty barrier when its
# consumers have released it; the arrivals of a phase are in parentheses.
#   XFULL, XEMPTY    per X ring stage: the tile is copied (the TMA bytes); the rotation that read it is done (MMA warp)
#   XRFULL, XREMPTY  per rotated tile buffer: the rotation is done (MMA warp); the projections that read the buffer are
#                    done (MMA warp)
#   AFULL, AEMPTY    the residual dq, and the prescaled values of a Q tile, are in TMEM (the 16 math warps); the
#                    projections that read them are done (MMA warp)
#   DFULL, DEMPTY    per projection buffer: the projections are done (MMA warp); their last channels are in registers
#                    (the 4 compensation warps)
#   UFULL, UEMPTY    per U operand stage: the basis of a (batch, head) is staged (load warp); the last projection that
#                    reads the stage is done (MMA warp)
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
    """The projection buffer of a tile of this parity in the ring of `stages` buffers (two or one)."""
    return parity if stages == 2 else 0


def _d_rounds(i, stages):
    """The earlier uses of tile i's projection buffer in the ring of `stages` buffers."""
    return i >> 1 if stages == 2 else i


# ------------------------------------------------------------------------------------------------ setup
def _init_barriers(bars):
    """Thread 0: every mbarrier of the CTA, with the arrivals of one phase (16 math warps on AFULL, 4 compensation
    warps on each DEMPTY, one arrival on the others)."""
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
    """Every thread: its 16-byte chunks (tidx, tidx + QK_THREADS, ... of the 2048) of the Hadamard matrix into the B
    operand of the rotation at the shared address ht_base: element (k, n) = hadamard[k, n] at row n, column k (K-major,
    SW128)."""
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
    """(bh, rem) of the CTA's first tile t0: its (batch, head) and its position among the 2 NB tiles of that (batch,
    head)."""
    bh = st.t0 // st.nb2
    return bh, st.t0 - bh * st.nb2


def _live_blocks(st, b):
    """With lengths: the 128-token blocks of batch element b that hold a valid token (those of the last element for
    a b past it)."""
    return (valid_len(st.lens, cutlass.min(b, st.B - 1), st.S) + 127) >> 7


def _first_tile(st):
    """The CTA's first tile as the iterator (kind, sblk, bh, h, b), fastest first: kind 0 for the q tile of a block and
    1 for its k tile, 128-token block sblk, (batch, head) index bh = b H + h, head h, batch b."""
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
    """The tile after (kind, sblk, bh, h, b): the k tile of the block, then the q tile of the next block, of the next
    head, of the next batch. The iterator advances without divisions. With lengths a head ends with the last live block
    of its batch element, and the batch elements without a valid token are passed over."""
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
    """One arrival of the warp on mbarrier `slot`, by lane 0 once every lane of the warp has reached this point."""
    cute.arch.sync_warp()
    if st.lane == 0:
        cute.arch.mbarrier_arrive(sptr(Int64, st.bars + 8 * slot))


@cute.jit
def _tile_outputs(st, codes, scales, rstride, kind, sblk, b, h):
    """Where the results of a tile go: the global addresses of its tensor's code rows and scale tiles (kind 0: Q, 1:
    K; codes: the addresses of q4 and k4, scales: of q_sfv and k_sfv) and the offset of the code row of the tile's
    first token at head h. A code row is the 128 bytes of a (token, head); rstride: the bytes of a token, 128 H."""
    code_base = codes[0]
    scale_base = scales[0]
    if kind == 1:
        code_base = codes[1]
        scale_base = scales[1]
    row0 = (Int64(b) * Int64(st.S) + Int64(sblk * 128)) * rstride + Int64(h * 128)
    return code_base, scale_base, row0


def _scale_tile(st, bh, sblk):
    """The index of the first BF16 scale of a tile among the scale tiles of its tensor. A tile holds sfv_tile scales:
    1024 of the body, (32-channel group, token, 16-channel block), then the 128 tokens of each 16-channel block of the
    compensation channels."""
    return (Int64(bh) * Int64(st.NB) + Int64(sblk)) * Int64(st.sfv_tile)


@cute.jit
def _reduce_maxima(st, gmax, maxima):
    """maxima: (word of gmax, the thread's maximum) pairs. Each maximum is reduced over the warp and, by lane 0, into
    its word of gmax (gmax: its global address) across the CTAs."""
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
    """The views of the TMA copy of a tile into the X ring: (the ring by stage, `tiles` by block and (head, batch));
    tiles: the 128-token tiles of the tensor of tma (q or k)."""
    return cpasync.tma_partition(
        tma.atom,
        0,
        cute.make_layout(1),
        cute.group_modes(st.sX, 0, 3),
        cute.group_modes(st.thr_mma.partition_A(tiles), 0, 3),
    )


@cute.jit
def _stage_basis(st, ut_base, bh, ucount):
    """Load warp: the basis of (batch, head) bh, U^T (FP32 [RANK][128]) at the global address ut_base, into U stage
    ucount & 1 as the K-major SW128 TF32 B operand of the projections: 4 K-boxes of RANK rows x 128 B. Waits UEMPTY of
    the stage and arrives on its UFULL."""
    us = ucount & 1
    mbar_wait_bounded(st.bars + 8 * (B_UEMPTY + us), ((ucount >> 1) + 1) & 1)
    for half_ in cutlass.range_constexpr(st.RANK // 16):
        uws = []
        for it in cutlass.range_constexpr(16):
            qq = st.lane + 32 * (half_ * 16 + it)  # 16-byte chunk of the operand
            nrow = qq >> 5
            cc = qq & 31
            uws.append(
                load_invariant(
                    Uint32,
                    ut_base
                    + ((Int64(bh) * Int64(st.RANK) + Int64(nrow)) * Int64(512) + Int64(cc * 16)),
                    4,
                )
            )
        for it in cutlass.range_constexpr(16):
            qq = st.lane + 32 * (half_ * 16 + it)
            nrow = qq >> 5
            cc = qq & 31
            w = uws[it]
            dst = (
                st.u_base
                + us * st.ustage
                + (cc >> 3) * (st.RANK * 128)
                + nrow * 128
                + (((cc & 7) ^ (nrow & 7)) << 4)
            )
            store_vector(sptr(Uint32, dst, 16), (w[0], w[1], w[2], w[3]))
    cute.arch.fence_proxy("async.shared", space="cta")
    _warp_arrive(st, B_UFULL + us)


@cute.jit
def _load_warp(st, ut_base, scales):
    """Load warp (16): stages the basis of each (batch, head) of the CTA's tiles as the U operand (ut_base: the global
    address of the basis uT) and copies the q and k tiles into the X ring.

    Handshake: _stage_basis waits UEMPTY and arrives on UFULL at the first tile of a (batch, head). A tile waits XEMPTY
    of its ring stage (the rotation that read the stage is done) and its TMA copy completes XFULL of the stage."""
    gQ = cute.local_tile(st.tma_q.tma_tensor, (128, 128), (None, 0, None))
    gK = cute.local_tile(st.tma_k.tma_tensor, (128, 128), (None, 0, None))
    tQs, tQg = _x_views(st, st.tma_q, gQ)
    tKs, tKg = _x_views(st, st.tma_k, gK)
    bh_prev = Int32(-1)
    ucount = Int32(-1)
    kind, sblk, bh, h, b = _first_tile(st)
    for i in cutlass.range(st.n):
        if bh != bh_prev:
            ucount = ucount + 1
            bh_prev = bh
            _stage_basis(st, ut_base, bh, ucount)
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
        kind, sblk, bh, h, b = _next_tile(st, kind, sblk, bh, h, b)
    if cutlass.const_expr(st.lens is not None):
        _zero_scales(st, scales)


@cute.jit
def _zero_scales(st, scales):
    """Load warp, with lengths: zero block scales for the tiles without a valid token among the CTA's range of all
    tiles (t0_all, n_all), which no role visits. scales: the global addresses of (q_sfv, k_sfv); a lane stores its
    share of the 16-byte chunks of a tile's BF16 scales."""
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
    """MMA warp: rotation of tile j, xr[j & 1] = X[j % X_STAGES] * Hd. Waits XFULL of the ring stage and XREMPTY of the
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


def _project(st, d, a, ua, idesc, accumulate):
    """MMA warp: the RANK TMEM columns at d = (or, with accumulate, +=) the 128 TMEM columns at a times the U operand
    at the shared address ua, as 16 MMAs of kind::tf32 (K = 8 each)."""
    for kk in range(16):
        bd = smem_desc_sw128(ua + (kk >> 2) * (st.RANK * 128) + (kk & 3) * 32)
        tc_mma(d, a + 8 * kk, bd, idesc, Int32(1 if accumulate else kk), kind="tf32", a_tmem=True)


def _qk_projections(st, p, dp, us, idesc, idesc_neg):
    """MMA warp: the projections of the tile in xr[p] into projection buffer dp, with the basis in U stage us: the
    residual dq U (dq FP32 in the A region) into the first half and the body (x - dq) U = x U + (-dq) U (x still in
    xr[p]) into the second; the negated product uses the descriptor's negate-A bit (idesc_neg)."""
    ua = st.u_base + us * st.ustage
    d = st.tm + TM_D + dp * (2 * st.RANK)
    dq = st.tm + TM_AD1
    _project(st, d, dq, ua, idesc, False)
    _project(st, d + st.RANK, st.tm + p * 128, ua, idesc, False)
    _project(st, d + st.RANK, dq, ua, idesc_neg, True)


@cute.jit
def _mma_warp(st):
    """MMA warp (17): the rotation of every tile, issued one tile ahead of the projections, so the math warps work on
    tile i + 1 while the compensation warps work on tile i.

    Handshake: _qk_rotation waits XFULL and XREMPTY and commits XRFULL and XEMPTY. The projections of tile i wait
    UFULL at the first tile of a (batch, head), AFULL (the math warps have stored the residual) and DEMPTY of the
    tile's projection buffer; they commit DFULL, AEMPTY and XREMPTY, and UEMPTY after the last tile of the (batch,
    head) or of the CTA."""
    idesc1 = Int32(idesc_bf16(128, 128))
    idesc2 = Int32(idesc_tf32(128, st.RANK))
    idesc2n = Int32(idesc_tf32(128, st.RANK) | (1 << 13))

    if st.n > 0:
        _qk_rotation(st, Int32(0), idesc1)
    bh_prev = Int32(-1)
    ucount = Int32(-1)
    if cutlass.const_expr(st.lens is not None):
        kind, sblk, bh, h, b = _first_tile(st)
        rem = Int32(0)
    else:
        bh, rem = _tile_origin(st)
    for i in cutlass.range(st.n):
        if i + 1 < st.n:
            _qk_rotation(st, i + 1, idesc1)
        if bh != bh_prev:
            ucount = ucount + 1
            bh_prev = bh
            mbar_wait_bounded(st.bars + 8 * (B_UFULL + (ucount & 1)), (ucount >> 1) & 1)
        p = i & 1
        dp = _d_buffer(p, st.d_stages)
        mbar_wait_bounded(st.bars + 8 * B_AFULL, i & 1)
        mbar_wait_bounded(st.bars + 8 * (B_DEMPTY + dp), (_d_rounds(i, st.d_stages) + 1) & 1)
        tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
        _qk_projections(st, p, dp, ucount & 1, idesc2, idesc2n)
        tc_commit(st.bars + 8 * (B_DFULL + dp))
        tc_commit(st.bars + 8 * B_AEMPTY)
        tc_commit(st.bars + 8 * (B_XREMPTY + p))
        # the tile's position in its (batch, head): the U stage is released after the last one
        if cutlass.const_expr(st.lens is not None):
            # with lengths the last one is the k tile of the last live block of the batch element
            if (i + 1 == st.n) or ((kind == 1) & (sblk + 1 == _live_blocks(st, b))):
                tc_commit(st.bars + 8 * (B_UEMPTY + (ucount & 1)))
            kind, sblk, bh, h, b = _next_tile(st, kind, sblk, bh, h, b)
        else:
            rem = rem + 1
            if rem == st.nb2:
                rem = 0
            if (i + 1 == st.n) or (rem == 0):
                tc_commit(st.bars + 8 * (B_UEMPTY + (ucount & 1)))
            if rem == 0:
                bh = bh + 1


# ------------------------------------------------------------------------------------------------ compensation warps
@cute.jit
def _compensation_warps(st, codes, scales, gmax):
    """Compensation warps (18-21), one per TMEM lane quadrant: a thread takes the full row of the 2 RANK projected
    channels of its token ([dq U | (x - dq) U] in D; K stores the halves swapped), loaded 32 channels at a time, and
    stores their NVFP4 codes and block scales; the maxima of the scales of each half go to gmax. codes, scales, gmax:
    the global addresses of (q4, k4), of (q_sfv, k_sfv) and of gmax.

    Handshake: a tile waits DFULL of its projection buffer; the warp arrives on DEMPTY of the buffer once the tile's
    last channels are in registers."""
    q2 = st.warp & 3
    row = q2 * 32 + st.lane
    tl = (q2 * 32) << 16
    mq1 = Float32(0.0)
    mq2 = Float32(0.0)
    mk1 = Float32(0.0)
    mk2 = Float32(0.0)
    kind, sblk, bh, h, b = _first_tile(st)
    rstride = Int64(st.H) * Int64(128)
    thr_off = Int64(row) * rstride
    for j in cutlass.range(st.n):
        p = _d_buffer(j & 1, st.d_stages)
        mbar_wait_bounded(st.bars + 8 * (B_DFULL + p), _d_rounds(j, st.d_stages) & 1)
        tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
        tok = sblk * 128 + row
        if cutlass.const_expr(st.lens is not None):
            n_valid = valid_len(st.lens, b, st.S)
        code_base, scale_base, row0 = _tile_outputs(st, codes, scales, rstride, kind, sblk, b, h)
        soff = _scale_tile(st, bh, sblk) + Int64(1024 + q2 * 32 + st.lane)
        for part in cutlass.range_constexpr(st.RANK // 16):
            half = part // (st.RANK // 32)
            dv = ld_tmem_f32_x32_wait(st.tm + tl + TM_D + p * (2 * st.RANK) + part * 32)
            if cutlass.const_expr(part == st.RANK // 16 - 1):
                # the last channels are in registers: release the D buffer to the MMA warp
                tcgen05_fence(Tcgen05Fence.BEFORE_THREAD_SYNC)
                _warp_arrive(st, B_DEMPTY + p)
            ws = []
            for grp in cutlass.range_constexpr(2):
                cqi = part * 2 + grp
                sg, w0, w1 = quant16(dv[16 * grp : 16 * grp + 16])
                if cutlass.const_expr(st.lens is not None):
                    sg = Float32(cutlass.select_(tok < n_valid, sg, Float32(0.0)))
                ws.append(w0)
                ws.append(w1)
                sg_dst = scale_base + Int64((soff + Int64(cqi * 128)) * 2)
                sg_bits = sg.to(BFloat16).bitcast(Uint16)
                cute.arch.store(gptr(Uint16, sg_dst), sg_bits)
                if kind == 1:
                    if cutlass.const_expr(half == 1):
                        mk1 = cute.arch.fmax(mk1, sg)
                    else:
                        mk2 = cute.arch.fmax(mk2, sg)
                else:
                    if cutlass.const_expr(half == 0):
                        mq1 = cute.arch.fmax(mq1, sg)
                    else:
                        mq2 = cute.arch.fmax(mq2, sg)
            # the half's RANK / 2 bytes of the token row: the first after the head channels' 64 for dq U of Q
            # and (x - dq) U of K
            slot = Int32(half) ^ kind
            if tok < st.S:
                store_vector(
                    gptr(
                        Uint32,
                        code_base
                        + row0
                        + thr_off
                        + Int64(64 + part % (st.RANK // 32) * 16)
                        + Int64(slot * (st.RANK // 2)),
                        16,
                    ),
                    ws,
                )
        kind, sblk, bh, h, b = _next_tile(st, kind, sblk, bh, h, b)
    _reduce_maxima(st, gmax, ((1, mq1), (2, mq2), (4, mk1), (5, mk2)))


# ------------------------------------------------------------------------------------------------ math warps
def _prescaled(x, am0, am1):
    """The 32 values x and the maxima of their two blocks, each times the score prescale QPRE."""
    pre = Float32(QPRE)
    xs = []
    for j in range(0, 32, 2):
        ps = cute.arch.mul_packed_f32x2((x[j], x[j + 1]), (pre, pre))
        xs += [ps[0], ps[1]]
    return xs, am0 * pre, am1 * pre


@cute.jit
def _math_warps(st, codes, scales, gmax):
    """Math warps (0-15): a thread takes 32 channels of a token of the rotated tile, quantizes them to NVFP4 in two
    16-channel blocks, writes the quantization residual to the A columns of the projections and stores the codes and
    the block scales; the maxima of the scales go to gmax. codes, scales, gmax: the global addresses of (q4, k4), of
    (q_sfv, k_sfv) and of gmax.

    Handshake: a tile waits XRFULL of its rotated tile buffer, and AEMPTY (the projections of the tile before are
    done) before the residual is overwritten; the warp arrives on AFULL once its residual, and the prescaled values of
    a Q tile, are in TMEM."""
    lq = st.warp & 3
    row_half = (st.warp >> 2) & 1
    n_pair = st.warp >> 3
    # TMEM accesses use the 16x32bx2 shape: lanes 0..15 and 16..31 read the same 16 rows at two column bases, so
    # every thread keeps contiguous 16-channel blocks, so its code stores are contiguous. Each warp stays inside its own
    # 32 TMEM lanes (row_half selects the first or second 16 rows).
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
        # Q tiles carry the score prescale: the rotated values are scaled before the quantization and the scaled
        # tile goes back to TMEM, where the compensation MMAs read it (the AFULL arrive below orders the store
        # before them)
        if kind == 0:
            x, am0, am1 = _prescaled(x, am0, am1)
            st_tmem_16x32bx2_f32_x32_wait(xr, x)
        sc0, w0, w1, dq0 = quant16_dq_knownmax(x[0:16], am0)
        sc1, w2, w3, dq1 = quant16_dq_knownmax(x[16:32], am1)
        mbar_wait_bounded(st.bars + 8 * B_AEMPTY, (i + 1) & 1)
        tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
        st_tmem_16x32bx2_f32_x32_wait(st.tm + tl + TM_AD1 + n_pair * 64, dq0 + dq1)
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
    _reduce_maxima(st, gmax, ((0, mq0), (3, mk0)))


# ------------------------------------------------------------------------------------------------ kernel
@cute.jit
def _qk_body(
    tiled_mma, tma_q, tma_k, hadamard, uT, q4, k4, q_sfv, k_sfv, gmax, S, H, NB, ntiles, RANK, lens
):
    """The Q'/K' pass of the CTA's share of the ntiles tiles: the setup of the CTA, the role of the thread's warp, then
    the release of the TMEM columns. RANK: the directions of the basis uT [B, H, RANK, 128]. lens: None, or the global
    address of the valid prefix lengths: the block scales of the tokens past the length of their batch element are
    then zero and stay out of the maxima, and the CTAs share the tiles that hold a valid token (_live_tiles)."""
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
    # two stages of the U operand (RANK rows of 128 FP32 each), and the projection buffers after TM_D that the
    # 512 TMEM columns hold
    ustage = RANK * 512
    d_stages = 2 if TM_D + 4 * RANK <= 512 else 1
    # BF16 block scales of a tile in the write-coalesced layout
    sfv_tile = 512 * qk_scale_atoms(RANK)
    sU = smem.allocate_array(Uint8, 2 * ustage, byte_alignment=1024)
    bars = Int32(bars_p.toint())
    hold = Int32(hold_p.toint())
    x_base = Int32(sX.iterator.toint())
    ht_base = Int32(sHT.toint())
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
        ustage=ustage,
        d_stages=d_stages,
        sfv_tile=sfv_tile,
        # the TMA of q and k and the MMA slice that partitions their tiles
        tma_q=tma_q,
        tma_k=tma_k,
        thr_mma=thr_mma,
    )
    # One role per warp. A role takes the global addresses it reads or writes; each is read from its tensor here, in
    # the branch of the role.
    if warp == 16:
        _load_warp(
            st,
            Int64(uT.iterator.toint()),
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
    """With lengths: (the number of live tiles of CTA bidx of gdim, its first one as the tile iterator). The live
    tiles, in the order of all tiles, are the q and k tiles of the blocks that hold a valid token, in every head of
    their batch element; the CTAs take contiguous ranges of them of the same size."""
    total = Int32(0)
    for b in cutlass.range(batch):
        total = total + H * 2 * ((valid_len(lens, b, S) + 127) >> 7)
    l0 = (bidx * total) // gdim
    n = ((bidx + 1) * total) // gdim - l0
    # the batch element that holds live tile l0, and the live tiles before it
    b = Int32(0)
    before = Int32(0)
    per_head = 2 * ((valid_len(lens, cutlass.min(b, batch - 1), S) + 127) >> 7)
    while (b < batch) & (l0 >= before + H * per_head):
        before = before + H * per_head
        b = b + 1
        per_head = 2 * ((valid_len(lens, cutlass.min(b, batch - 1), S) + 127) >> 7)
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
    uT: cute.Tensor,
    q4: cute.Tensor,
    k4: cute.Tensor,
    q_sfv: cute.Tensor,
    k_sfv: cute.Tensor,
    gmax: cute.Tensor,
    S: Int32,
    H: Int32,
    NB: Int32,
    ntiles: Int32,
    RANK: cutlass.Constexpr,
):
    """The Q'/K' pass (_qk_body) of a call whose batch elements are all full."""
    _qk_body(
        tiled_mma,
        tma_q,
        tma_k,
        hadamard,
        uT,
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
        None,
    )


@cute.kernel
def qk_kernel_lens(
    tiled_mma: cute.TiledMma,
    tma_q: cpasync.TmaInfo,
    tma_k: cpasync.TmaInfo,
    hadamard: cute.Tensor,
    uT: cute.Tensor,
    q4: cute.Tensor,
    k4: cute.Tensor,
    q_sfv: cute.Tensor,
    k_sfv: cute.Tensor,
    gmax: cute.Tensor,
    lens: cute.Tensor,
    S: Int32,
    H: Int32,
    NB: Int32,
    ntiles: Int32,
    RANK: cutlass.Constexpr,
):
    """The Q'/K' pass (_qk_body) of a call with valid prefix lengths lens [B] (int32)."""
    _qk_body(
        tiled_mma,
        tma_q,
        tma_k,
        hadamard,
        uT,
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
        Int64(lens.iterator.toint()),
    )
