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
"""Shared-memory layout of the sm_120 FP8-PV attention kernel: the stages of its TMA ring, the key order of a tile and
the per-lane addresses of the B fragments.

A stage holds what one MMA phase reads: K' of a key tile as three 32-byte atom columns (64 channels each, the three
k-steps of the QK MMA over the 192 channels of Q' and K'; 32-byte TMA swizzle), V of the tile before it (128 keys x
128 bytes, 128-byte swizzle) and the three scale atoms of K'. After the stages come two buffers of v_mean rows, one
per pair of steps (two tiles of 8 rows x 256 bytes, 128-byte swizzle within each 1 KB half); the rows of an unpaired
last step go to the K' region of its stage, which that phase leaves unread. The swizzles keep every ldmatrix and
scale-word read free of bank conflicts.

Key order. The MMAs take the 128 keys of a tile as four quarters of 32 keys, and n-tile s of a quarter covers the
keys 8 key_grp(s) + (column ^ key_rot(s)): n-tiles 0 and 2 hold the quarter's first 16-key block and 1 and 3 its
second, so every f16x2 pair of P that the softmax packs (n-tiles 2j, 2j + 1) holds one key of either block, and the
block sums of the means MMA fall out of the row-sum tree. The same permutation addresses the K' rows and scale words
of the QK and the V rows of the PV, so the S fragments become PV fragments without shuffles."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync

from ...constants import HEAD_DIM, QK_CHANNELS

KT = 128  # keys per tile
QT = 128  # query rows per CTA
NTHR = 256  # 8 warps x 16 rows
STAGES = 3
K_STEPS = (
    QK_CHANNELS // 64
)  # k-steps of the QK MMA: one 32-byte column of K' and one scale atom per 64 channels
V_BYTES = KT * HEAD_DIM
SF_BYTES = K_STEPS * 512
EP = 272  # epilogue row pitch (bf16 rows of 128 channels + 16 bytes)

KA_BYTES = KT * 32  # one 32-byte K' atom column (64 channels) of a tile, 32-byte swizzle
K_BYTES = K_STEPS * KA_BYTES
V_OFF = K_BYTES  # V: 128 keys x 128 bytes, 128-byte swizzle (1 KB aligned)
SF_OFF = V_OFF + V_BYTES  # the K' scale atoms
STAGE_BYTES = 30720  # 30208 bytes used, 1 KB aligned for the swizzles
VM_BYTES = 8 * 256  # a tile's 8 v_mean rows: two 128-byte halves of 8 rows, 128-byte swizzle
VMP_OFF = (
    STAGES * STAGE_BYTES
)  # two buffers of a step pair's v_mean rows (slot 0: the pair's first step)
VMP_BYTES = 2 * 2 * VM_BYTES
BAR_OFF = VMP_OFF + VMP_BYTES  # the stages' full mbarriers
CNT_OFF = (
    BAR_OFF + 8 * STAGES
)  # per-stage release counters: one increment per warp and use of the stage
SMEM_BYTES = CNT_OFF + 4 * STAGES


def key_rot(s):
    """The key rotation of row group s of a 32-key step, 0, 3, 4, 7: column n of QK n-tile (s, i) is key
    32 i + 8 key_grp(s) + (n ^ key_rot(s)), which keeps the ldmatrix and scale-word reads bank-conflict free."""
    return (s & 1) * 3 + (s >> 1) * 4


def key_grp(s):
    """The 8-key group of n-tile s within its 32-key quarter, 0, 2, 1, 3: n-tiles 0 and 2 hold the quarter's first
    16-key block, 1 and 3 its second, so each f16x2 pair of P (n-tiles 2j, 2j + 1) holds one key of either block."""
    return (s & 1) * 2 + (s >> 1)


def lane_offsets(lane):
    """Per-lane shared-memory offsets of the B-fragment loads (keys of n-tile s of a quarter: 8 key_grp(s) +
    (column ^ key_rot(s))): the K' rows of the even and odd n-tile pairs, the K' scale words, the V row, and the
    v_mean rows [of the two channel halves of an unpaired step, of a step pair]."""
    q = lane & 3
    g = lane >> 2
    # K' ldmatrix.x4 of n-tile pair p: matrix m = lane / 8 is n-tile 2p + m / 2, k-chunk m % 2; row = lane % 8; each
    # 32-byte atom column at a 32-byte pitch with the 32-byte swizzle (chunk ^= row bit 2)
    m = lane >> 3
    gg = lane & 7
    s0 = m >> 1
    s1 = s0 + 2
    rl_e = gg ^ key_rot(s0)
    rl_o = gg ^ key_rot(s1)
    koff_e = (16 * s0 + rl_e) * 32 + ((m & 1) ^ ((rl_e >> 2) & 1)) * 16
    koff_o = (16 * s0 + 8 + rl_o) * 32 + ((m & 1) ^ ((rl_o >> 2) & 1)) * 16
    # K' scale words: lane 4g + s reads the 16-byte chunk of the key of column g of n-tile s (word i: key + 32 i)
    sfoff = (8 * key_grp(q) + (g ^ key_rot(q))) * 16
    # V ldmatrix.trans: lane T holds the address of the key of logical k = T of the 32-key step (n-tile T % 4, column
    # 2 ((T % 16) / 4) + T / 16)
    ts = lane & 3
    vrow = 8 * key_grp(ts) + ((2 * ((lane & 15) >> 2) + (lane >> 4)) ^ key_rot(ts))
    # the offset of channel group 0 (row and swizzle chunk); group cg is at (stage base + voff) ^ (16 cg)
    voff = vrow * 128 + (vrow & 7) * 16
    # v_mean ldmatrix.x4.trans: matrix m = lane / 8 is channels 32 cp + 8 m .. + 7 of block row lane % 8 (128-byte
    # swizzle within each 1 KB half: chunk ^= row)
    vr = lane & 7
    vm0 = vr * 128 + ((m ^ vr) * 16)
    # the pair layout: matrix m = lane / 8 is slot m / 2 (the pair's first or second step), channels 8 (m % 2) .. + 7
    # of the tile, block row lane % 8: one ldmatrix.x4.trans is the m16n8k16 A fragment of a 16-channel tile (tile
    # 4 hf + j at (buffer + vmp) ^ 32 j + 1024 hf)
    vmp = (m >> 1) * VM_BYTES + vr * 128 + (((m & 1) ^ vr) * 16)
    vmoff = [vm0, vm0 ^ 64, vmp]
    return koff_e, koff_o, sfoff, voff, vmoff


def make_tma_views(
    sp, tma_atom_k, tma_k, tma_atom_v, tma_v, tma_atom_s, tma_s, tma_atom_m, tma_m, hh, bb
):
    """TMA views of a CTA: the K', V, K' scale and v_mean tiles of its (batch, head) in global memory and their
    places in the shared memory at sp (the stages, the pair buffers of v_mean and, for an unpaired last step, the K'
    region of a stage), with the stages' full barriers: the tma tuple the ring's copies take."""
    sK = cute.make_tensor(
        sp,
        cute.make_composed_layout(
            cute.make_swizzle(1, 4, 3),
            0,
            cute.make_layout((32, KT, K_STEPS, STAGES), stride=(1, 32, KA_BYTES, STAGE_BYTES)),
        ),
    )
    gK = cute.local_tile(tma_k[None, None, None, hh, bb], (32, KT, K_STEPS), (0, None, 0))
    tKsK, tKgK = cpasync.tma_partition(
        tma_atom_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(gK, 0, 3)
    )
    sV = cute.make_tensor(
        sp + V_OFF,
        cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((KT, 128, STAGES), stride=(128, 1, STAGE_BYTES)),
        ),
    )
    gV = cute.local_tile(tma_v[None, None, hh, bb], (KT, 128), (None, 0))
    tVsV, tVgV = cpasync.tma_partition(
        tma_atom_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2)
    )
    sS = cute.make_tensor(
        sp + SF_OFF, cute.make_layout((32 * K_STEPS, 16, STAGES), stride=(16, 1, STAGE_BYTES))
    )
    gS = cute.local_tile(tma_s[None, None, hh, bb], (32 * K_STEPS, 16), (None, 0))
    tSsS, tSgS = cpasync.tma_partition(
        tma_atom_s, 0, cute.make_layout(1), cute.group_modes(sS, 0, 2), cute.group_modes(gS, 0, 2)
    )
    sM = cute.make_tensor(
        sp + VMP_OFF,
        cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((128, 8, 2, 2, 2), stride=(1, 128, 1024, VM_BYTES, 2 * VM_BYTES)),
        ),
    )
    gM = cute.local_tile(tma_m[None, None, None, hh, bb], (128, 8, 2), (0, None, 0))
    tMsM, tMgM = cpasync.tma_partition(
        tma_atom_m, 0, cute.make_layout(1), cute.group_modes(sM, 0, 3), cute.group_modes(gM, 0, 3)
    )
    sM1 = cute.make_tensor(
        sp,
        cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((128, 8, 2, STAGES), stride=(1, 128, 1024, STAGE_BYTES)),
        ),
    )
    tM1sM, _ = cpasync.tma_partition(
        tma_atom_m, 0, cute.make_layout(1), cute.group_modes(sM1, 0, 3), cute.group_modes(gM, 0, 3)
    )
    return (
        tma_atom_k,
        tKgK,
        tKsK,
        tma_atom_v,
        tVgV,
        tVsV,
        tma_atom_s,
        tSgS,
        tSsS,
        tma_atom_m,
        tMgM,
        tMsM,
        tM1sM,
        cute.recast_ptr(sp + BAR_OFF, dtype=cutlass.Int64),
    )
