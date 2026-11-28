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
"""Scale encoding of Q' and K' in the DeCo preparation: the encode launch, a plain launch after the Q/K pass.

* In (left by the Q/K pass): the block scales of every 128-token tile as BF16 values in a write-coalesced tile layout
  (q_sfv, k_sfv) and the maxima of the three scale groups of each tensor (gmax).
* Out: the UE4M3 scale-factor atoms the attention's QK reads (q_sf, k_sf), with one power-of-two group scale per
  tensor and scale group folded in; score_scale, the product of the group scales over 128 (the multiplier of the raw
  scores in log2 units); v8_scale, the power-of-two scale of the E4M3 V.
* Atoms: [B, H, tiles, atoms, 512], one byte per 16 channels and token; an atom per 64 channels, three in the QK
  build of 192 channels and four in that of 256. Atoms 0 and 1 hold the head channels (scale group 0), the atoms
  after them the compensation channels: the r_q / 16 blocks of the Q-residual term (scale group 1), the r_k / 16
  blocks of the K-residual term (scale group 2), then blocks of scale zero.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64, Uint8, Uint16, Uint32

from ...common.expmath import e4m3x4, f2u, pow2_ceil_448, u2f
from ...common.mem import gptr, load_invariant, sptr, store_vector
from ...constants import E4M3_MAX, HEAD_DIM, QK_BLOCK, qk_channels, qk_scale_atoms
from .basis import GMAX_Q, GMAX_V

ENC_G = 4  # Q/K scale tiles per staging round


def encode_pad128(byte_offset):
    """Return the byte offset in the staged Q/K scale tiles, whose 128-byte rows are padded to 160 bytes."""
    return (byte_offset & 127) + (byte_offset >> 7) * 160


def _gmax(gb, word):
    """Return float word `word` of gmax (at address gb): written by the earlier launches and only read by this one."""
    return load_invariant(Float32, gb + Int64(4 * word))


def sfv_tile_bytes(channels):
    """Return the bytes of a tile's BF16 block scales in the write-coalesced layout of the Q/K pass, for a QK build of
    `channels` channels: 1024 per atom."""
    return 1024 * qk_scale_atoms(channels)


def encode_threads(channels):
    """Return the threads of the Q/K scale encoding: one per 16-byte chunk of a tile's BF16 scales."""
    return sfv_tile_bytes(channels) // 16


def staged_tile_bytes(channels):
    """Return the shared-memory bytes of one staged tile of BF16 scales (its 128-byte rows padded)."""
    return encode_pad128(sfv_tile_bytes(channels))


def encode_stage_bytes(channels):
    """Return the shared-memory bytes of the Q/K scale encoding: ENC_G staged tiles of Q' scales and of K' scales."""
    return ENC_G * 2 * staged_tile_bytes(channels)


@cute.kernel
def encode_kernel(
    q_sfv: cute.Tensor,
    k_sfv: cute.Tensor,
    q_sf: cute.Tensor,
    k_sf: cute.Tensor,
    gmax: cute.Tensor,
    score_scale: cute.Tensor,
    ntile: Int32,
    v8_scale: cute.Tensor,
    MIN_PROD: cutlass.Constexpr,
    RANK: cutlass.Constexpr,
):
    """Encode the Q/K scales of the ntile tiles of RANK = (r_q, r_k) on the encode_threads threads of a CTA of its QK
    build, ENC_G tiles per round.

    * MIN_PROD is 128 times the smallest score scale the attention takes: the score scale (prod / 128) must be a power
      of two of at least MIN_PROD / 128, the floor of its P landing.
    * Both are powers of two: the Q group scales grow by the square root (rounded up to a power of two) of the ratio,
      so the Q and the K UE4M3 scales each shrink by about its square root and neither side underflows.
    """
    et, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    smem = utils.SmemAllocator()
    r_q, r_k = RANK
    channels = qk_channels(RANK)
    stb = Int32(smem.allocate_array(Uint8, encode_stage_bytes(channels), byte_alignment=16).toint())
    atoms = qk_scale_atoms(channels)
    threads = encode_threads(channels)  # a tile has twice as many words of UE4M3 scales
    sfv_bytes = sfv_tile_bytes(channels)
    staged = staged_tile_bytes(channels)
    # the scale group of each 16-channel block of the compensation channels: 1 (the Q-residual term), 2 (the
    # K-residual term), or 0 after the terms
    groups = [
        1 if c * QK_BLOCK < r_q else 2 if c * QK_BLOCK < r_q + r_k else 0
        for c in range((channels - HEAD_DIM) // QK_BLOCK)
    ]
    gb = Int64(gmax.iterator.toint())
    gq0, gq1, gq2, gk0, gk1, gk2 = (
        pow2_ceil_448(_gmax(gb, word)) for word in range(GMAX_Q, GMAX_V)
    )
    prod = cute.arch.fmax(cute.arch.fmax(gq0 * gk0, gq1 * gk1), gq2 * gk2)
    if prod < Float32(MIN_PROD):
        e_up = (Int32(f2u(Float32(MIN_PROD) / prod) >> Uint32(23)) - Int32(127) + Int32(1)) >> 1
        fup = u2f(Uint32(e_up + Int32(127)) << Uint32(23))
        gq0 = gq0 * fup
        gq1 = gq1 * fup
        gq2 = gq2 * fup
        prod = Float32(MIN_PROD)
    iq0 = Float32(1.0) / gq0
    iq1 = Float32(1.0) / gq1
    iq2 = Float32(1.0) / gq2
    inv_prod = Float32(1.0) / prod
    qsb = Int64(q_sfv.iterator.toint())
    ksb = Int64(k_sfv.iterator.toint())
    qfb = Int64(q_sf.iterator.toint())
    kfb = Int64(k_sf.iterator.toint())
    lim = Float32(E4M3_MAX)
    for tb0 in cutlass.range(bidx * ENC_G, ntile, gdim * ENC_G):
        wq = []
        wk = []
        for g in cutlass.range_constexpr(ENC_G):
            tbc = cutlass.min(tb0 + g, ntile - 1)
            wq.append(
                load_invariant(Uint32, qsb + Int64(tbc) * Int64(sfv_bytes) + Int64(et * 16), 4)
            )
            wk.append(
                load_invariant(Uint32, ksb + Int64(tbc) * Int64(sfv_bytes) + Int64(et * 16), 4)
            )
        for g in cutlass.range_constexpr(ENC_G):
            q_words = wq[g]
            k_words = wk[g]
            store_vector(
                sptr(Uint32, stb + g * (2 * staged) + encode_pad128(et * 16), 16),
                (q_words[0], q_words[1], q_words[2], q_words[3]),
            )
            store_vector(
                sptr(Uint32, stb + g * (2 * staged) + staged + encode_pad128(et * 16), 16),
                (k_words[0], k_words[1], k_words[2], k_words[3]),
            )
        cute.arch.sync_threads()
        for g in cutlass.range_constexpr(ENC_G):
            tb = tb0 + g
            if tb < ntile:
                sg = stb + g * (2 * staged)
                for hh in cutlass.range_constexpr(2):
                    # row rm of the tile's 512-byte atoms; its atom selects the group scales (module docstring)
                    rm = et + threads * hh
                    atom = rm >> 7
                    rl = rm >> 2 & 31
                    lqv = rm & 3
                    # per block c of the atom: the factor of its Q' scales and of its K' scales (the group scale
                    # of the head channels, or of the block's term) and whether the block holds channels of a term
                    head = Float32(atom < 2)
                    fac_q = [iq0 * head] * 4
                    fac_k = [gq0 * head] * 4
                    live = [None] * 4
                    for a in cutlass.range_constexpr(2, atoms):
                        is_a = Float32(atom == a)
                        for c in cutlass.range_constexpr(4):
                            group = groups[(a - 2) * 4 + c]
                            if cutlass.const_expr(group != 0):
                                fac_q[c] = fac_q[c] + (iq1, iq2)[group - 1] * is_a
                                fac_k[c] = fac_k[c] + (gq1, gq2)[group - 1] * is_a
                                live[c] = is_a if live[c] is None else live[c] + is_a
                    fac_k = [f * inv_prod for f in fac_k]
                    all_live = all(groups)
                    fq = cute.make_rmem_tensor((4,), Float32)
                    fk = cute.make_rmem_tensor((4,), Float32)
                    if atom < 2:
                        o0 = ((2 * atom * 4 + lqv) * 32 + rl) * 4
                        o1 = (((2 * atom + 1) * 4 + lqv) * 32 + rl) * 4
                        wq0 = cute.arch.load(sptr(Uint32, sg + encode_pad128(o0)), Uint32)
                        wq1 = cute.arch.load(sptr(Uint32, sg + encode_pad128(o1)), Uint32)
                        wk0 = cute.arch.load(sptr(Uint32, sg + staged + encode_pad128(o0)), Uint32)
                        wk1 = cute.arch.load(sptr(Uint32, sg + staged + encode_pad128(o1)), Uint32)
                        fq[0] = u2f(wq0 << Uint32(16))
                        fq[1] = u2f(wq0 & Uint32(0xFFFF0000))
                        fq[2] = u2f(wq1 << Uint32(16))
                        fq[3] = u2f(wq1 & Uint32(0xFFFF0000))
                        fk[0] = u2f(wk0 << Uint32(16))
                        fk[1] = u2f(wk0 & Uint32(0xFFFF0000))
                        fk[2] = u2f(wk1 << Uint32(16))
                        fk[3] = u2f(wk1 & Uint32(0xFFFF0000))
                    else:
                        # the atom's four blocks of compensation channels, which the Q/K pass stored in the
                        # channel order of each tensor; a block after the terms was not stored and has scale zero
                        for c in cutlass.range_constexpr(4):
                            cq = c
                            if cutlass.const_expr(atoms > 3):
                                cq = (atom - 2) * 4 + c
                            off = encode_pad128((1024 + (cq * 4 + lqv) * 32 + rl) * 2)
                            bits_q = cute.arch.load(sptr(Uint16, sg + off), Uint16).to(Uint32)
                            bits_k = cute.arch.load(sptr(Uint16, sg + staged + off), Uint16).to(
                                Uint32
                            )
                            if cutlass.const_expr(not all_live):
                                if cutlass.const_expr(live[c] is None):
                                    bits_q = Uint32(0)
                                    bits_k = Uint32(0)
                                else:
                                    stored = live[c] > Float32(0.5)
                                    bits_q = Uint32(cutlass.select_(stored, bits_q, Uint32(0)))
                                    bits_k = Uint32(cutlass.select_(stored, bits_k, Uint32(0)))
                            fq[c] = u2f(bits_q << Uint32(16))
                            fk[c] = u2f(bits_k << Uint32(16))
                    bq = e4m3x4(*(cute.arch.fmin(fq[c] * fac_q[c], lim) for c in range(4)))
                    bk = e4m3x4(*(cute.arch.fmin(fk[c] * fac_k[c], lim) for c in range(4)))
                    cute.arch.store(
                        gptr(Uint32, qfb + Int64(tb) * Int64(512 * atoms) + Int64(rm * 4)), bq
                    )
                    cute.arch.store(
                        gptr(Uint32, kfb + Int64(tb) * Int64(512 * atoms) + Int64(rm * 4)), bk
                    )
        cute.arch.sync_threads()
    if bidx == 0 and et == 0:
        cute.arch.store(
            gptr(Uint32, Int64(score_scale.iterator.toint())), f2u(prod / Float32(128.0))
        )
        cute.arch.store(
            gptr(Uint32, Int64(v8_scale.iterator.toint())), f2u(pow2_ceil_448(_gmax(gb, GMAX_V)))
        )
