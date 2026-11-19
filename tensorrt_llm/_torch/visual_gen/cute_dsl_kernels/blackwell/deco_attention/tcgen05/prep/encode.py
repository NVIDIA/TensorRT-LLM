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
"""Scale encoding of Q' and K' in the DeCo preparation, after the Q/K pass. The Q/K pass leaves the block scales of
every 128-token tile as BF16 values in a write-coalesced tile layout (q_sfv, k_sfv) and the maxima of the three scale
groups of each tensor in gmax. encode_kernel turns them into the UE4M3 scale-factor atoms the attention's QK reads
(q_sf, k_sf: [B, H, tiles, atoms, 512], one byte per 16 channels and token; an atom per 64 channels: three at rank 32,
four at rank 64), with one power-of-two group scale per tensor and scale group folded in, and writes score_scale, the
product of the group scales over 128 (the multiplier of the raw scores in log2 units). It runs on the
encode_threads(rank) threads of every CTA of the encode launch, a plain launch after the Q/K pass that also writes
v8_scale, the power-of-two scale of the E4M3 V."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64, Uint8, Uint16, Uint32

from ...common.expmath import e4m3x4, f2u, pow2_ceil_448, u2f
from ...common.mem import gptr, load_invariant, sptr, store_vector
from ...constants import qk_scale_atoms

ENC_G = 4  # Q/K scale tiles per staging round


def encode_pad128(byte_offset):
    """Byte offset in the staged Q/K scale tiles, whose 128-byte rows are padded to 160 bytes."""
    return (byte_offset & 127) + (byte_offset >> 7) * 160


def _gmax(gb, word):
    """Float word `word` of gmax (at address gb): written by the earlier launches and only read by this one."""
    return load_invariant(Float32, gb + Int64(4 * word))


def sfv_tile_bytes(rank):
    """Bytes of a tile's BF16 block scales in the write-coalesced layout of the Q/K pass: 1024 per atom."""
    return 1024 * qk_scale_atoms(rank)


def encode_threads(rank):
    """Threads of the Q/K scale encoding: one per 16-byte chunk of a tile's BF16 scales."""
    return sfv_tile_bytes(rank) // 16


def staged_tile_bytes(rank):
    """Shared-memory bytes of one staged tile of BF16 scales (its 128-byte rows padded)."""
    return encode_pad128(sfv_tile_bytes(rank))


def encode_stage_bytes(rank):
    """Shared-memory bytes of the Q/K scale encoding: ENC_G staged tiles of Q' scales and of K' scales."""
    return ENC_G * 2 * staged_tile_bytes(rank)


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
    """The Q/K scale encoding by the encode_threads(RANK) threads of a CTA (ENC_G tiles per round, staged in
    encode_stage_bytes(RANK) bytes of shared memory), once the Q/K pass has finished: the final Q/K UE4M3 atoms of
    the ntile tiles with the group scales folded in (q_sf, k_sf), score_scale and v8_scale. MIN_PROD: 128 times the
    smallest score scale the attention takes."""
    et, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    smem = utils.SmemAllocator()
    stb = Int32(smem.allocate_array(Uint8, encode_stage_bytes(RANK), byte_alignment=16).toint())
    atoms = qk_scale_atoms(RANK)
    threads = encode_threads(RANK)  # a tile has twice as many words of UE4M3 scales
    sfv_bytes = sfv_tile_bytes(RANK)
    staged = staged_tile_bytes(RANK)
    comp_blocks = RANK // 8  # 16-channel blocks of the compensation channels
    gb = Int64(gmax.iterator.toint())
    gq0, gq1, gq2, gk0, gk1, gk2 = (pow2_ceil_448(_gmax(gb, word)) for word in range(6))
    prod = cute.arch.fmax(cute.arch.fmax(gq0 * gk0, gq1 * gk1), gq2 * gk2)
    # the attention needs the score scale (prod / 128) to be a power of two of at least MIN_PROD / 128 (the floor of
    # its P landing). Both are powers of two: the Q group scales grow by the square root (rounded up to a power of
    # two) of the ratio, so the Q and the K UE4M3 scales each shrink by about its square root and neither side
    # underflows
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
    ip_ = Float32(1.0) / prod
    qsb = Int64(q_sfv.iterator.toint())
    ksb = Int64(k_sfv.iterator.toint())
    qfb = Int64(q_sf.iterator.toint())
    kfb = Int64(k_sf.iterator.toint())
    lim = Float32(448.0)
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
            a = wq[g]
            b2 = wk[g]
            store_vector(
                sptr(Uint32, stb + g * (2 * staged) + encode_pad128(et * 16), 16),
                (a[0], a[1], a[2], a[3]),
            )
            store_vector(
                sptr(Uint32, stb + g * (2 * staged) + staged + encode_pad128(et * 16), 16),
                (b2[0], b2[1], b2[2], b2[3]),
            )
        cute.arch.sync_threads()
        for g in cutlass.range_constexpr(ENC_G):
            tb = tb0 + g
            if tb < ntile:
                sg = stb + g * (2 * staged)
                for hh in cutlass.range_constexpr(2):
                    # row rm of the tile's 512-byte atoms: atoms 0 and 1 hold the head channels (group scale 0), the
                    # atoms after them the two compensation halves (group scales 1 and 2): at rank 32 the low and
                    # the high block pair of atom 2, at rank 64 the atoms 2 and 3
                    rm = et + threads * hh
                    atom = rm >> 7
                    rl = rm >> 2 & 31
                    lqv = rm & 3
                    is2 = Float32(atom == 2)
                    if cutlass.const_expr(atoms == 3):
                        ilo = iq0 * (Float32(1.0) - is2) + iq1 * is2
                        ihi = iq0 * (Float32(1.0) - is2) + iq2 * is2
                        glo = (gq0 * (Float32(1.0) - is2) + gq1 * is2) * ip_
                        ghi = (gq0 * (Float32(1.0) - is2) + gq2 * is2) * ip_
                    else:
                        is3 = Float32(atom == 3)
                        ilo = iq0 * (Float32(1.0) - is2 - is3) + iq1 * is2 + iq2 * is3
                        ihi = ilo
                        glo = (gq0 * (Float32(1.0) - is2 - is3) + gq1 * is2 + gq2 * is3) * ip_
                        ghi = glo
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
                        # the atom's four blocks of compensation channels: the Q/K pass stored the scales of both
                        # tensors in the block order of Q', and K' holds the two halves swapped
                        for c_ in cutlass.range_constexpr(4):
                            cq = c_
                            if cutlass.const_expr(atoms > 3):
                                cq = (atom - 2) * 4 + c_
                            cqk = (cq + comp_blocks // 2) % comp_blocks
                            fq[c_] = u2f(
                                cute.arch.load(
                                    sptr(
                                        Uint16,
                                        sg + encode_pad128((1024 + (cq * 4 + lqv) * 32 + rl) * 2),
                                    ),
                                    Uint16,
                                ).to(Uint32)
                                << Uint32(16)
                            )
                            fk[c_] = u2f(
                                cute.arch.load(
                                    sptr(
                                        Uint16,
                                        sg
                                        + staged
                                        + encode_pad128((1024 + (cqk * 4 + lqv) * 32 + rl) * 2),
                                    ),
                                    Uint16,
                                ).to(Uint32)
                                << Uint32(16)
                            )
                    bq = e4m3x4(
                        cute.arch.fmin(fq[0] * ilo, lim),
                        cute.arch.fmin(fq[1] * ilo, lim),
                        cute.arch.fmin(fq[2] * ihi, lim),
                        cute.arch.fmin(fq[3] * ihi, lim),
                    )
                    bk = e4m3x4(
                        cute.arch.fmin(fk[0] * glo, lim),
                        cute.arch.fmin(fk[1] * glo, lim),
                        cute.arch.fmin(fk[2] * ghi, lim),
                        cute.arch.fmin(fk[3] * ghi, lim),
                    )
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
            gptr(Uint32, Int64(v8_scale.iterator.toint())), f2u(pow2_ceil_448(_gmax(gb, 6)))
        )
