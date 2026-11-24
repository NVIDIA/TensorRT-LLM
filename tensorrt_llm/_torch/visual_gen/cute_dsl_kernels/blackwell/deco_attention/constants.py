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
"""Constants of the DeCo-Attention kernels that the host code and the kernel families share. What one family alone
uses (thread counts, ring depths, tilers, schedule constants) stays in its modules."""

# The releases of nvidia-cutlass-dsl the kernels are validated with
VALIDATED_DSL_VERSIONS = ("4.8.0", "4.8.1")
# The alignment the launches assume of their pointers: of the address of a workspace and of every buffer offset in it
ALIGN = 256
# The default draw of the rows of the compensation basis takes one token in SAMPLE_FRAC, and at least twice the rank
SAMPLE_FRAC = 256

HEAD_DIM = 128  # channels of q, k, v and o per head
KEY_TILE = 128  # keys of a KV tile of the attention, and of a block of the preparation passes
INV_SQRT_HEAD_DIM = HEAD_DIM**-0.5
# The ranks of the compensation basis a kernel may be built for; a family's ranks() names those a target offers. Q'
# and K' carry 2 x rank compensation channels after the rotated head channels
RANKS = (32, 64)
# The rank every target has, and the default of the rank arguments
DEFAULT_RANK = 32
QK_ROW_BYTES = 128  # a packed token row of Q' / K': room for 256 E2M1 codes, the first qk_channels(rank) of them used


def key_tiles(s: int) -> int:
    return (s + KEY_TILE - 1) // KEY_TILE


def qk_channels(rank: int) -> int:
    """Return the channels of Q' and K' at a rank: the rotated head channels, then the compensation channels."""
    return HEAD_DIM + 2 * rank


def qk_scale_atoms(rank: int) -> int:
    """Return the 512-byte atoms of the Q' / K' scale factors of a tile at a rank, one per 64 channels."""
    return qk_channels(rank) // 64


# The channels and the scale atoms at the default rank: the kernels of the sm120 family have that rank only
QK_CHANNELS = qk_channels(DEFAULT_RANK)
QK_SCALE_ATOMS = qk_scale_atoms(DEFAULT_RANK)
# T: the anchor of a softmax row (its running maximum) only moves, and O is only rescaled, when the maximum of a key
# tile exceeds it by more than T binades
RESCALE_THRESHOLD = 4.0
E4M3_MAX = 448.0  # the largest finite E4M3 value
E2M1_MAX = 6.0  # the largest E2M1 value
# the V scale is a power of two at or above the largest deviation from the block means over the code range; the block
# means are stored divided by that scale too, so the deviation maximum also takes |mean| times this factor, which
# keeps the scaled mean of a block of large equal values (deviation zero) below 2^102 in FP32, with room for the block
# sums of P (at most 128 keys at the P prescale) that the means MMA multiplies it by
V_MEAN_HEADROOM = 2.0**-90
# f32 subnormal ulps per log2 unit of the exp2 landing: the landed bits are the f16 pattern of 2^t
SUBNORMAL_ULPS = 1024
# log2(e) / sqrt(HEAD_DIM), the score prescale folded into Q': the scores arrive in the log2 units the kernels
# exponentiate, and the score scale stays a power of two
QPRE = 1.4426950408889634 / HEAD_DIM**0.5
