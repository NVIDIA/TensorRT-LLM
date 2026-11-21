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
"""Constants of the DeCo-Attention kernels that the host code and the kernel families share."""

HEAD_DIM = 128  # channels of q, k, v and o per head
# Rank of the compensation basis: Q' and K' carry 2 x rank compensation channels after the rotated head channels.
# RANKS are the ranks a kernel may be built for; a family's ranks() names those a target offers. DEFAULT_RANK is the
# rank every target has, and the default of the rank arguments
RANKS = (32, 64)
DEFAULT_RANK = 32
QK_ROW_BYTES = 128  # a packed token row of Q' / K': room for 256 E2M1 codes, the first qk_channels(rank) of them used


def qk_channels(rank):
    """Channels of Q' and K' at a rank: the rotated head channels, then the compensation channels."""
    return HEAD_DIM + 2 * rank


def qk_scale_atoms(rank):
    """512-byte atoms of the Q' / K' scale factors of a tile at a rank, one per 64 channels."""
    return qk_channels(rank) // 64


# The channels and the scale atoms at the default rank, for the sm120 family, whose kernels have that rank only (the
# tcgen05 kernels take both per rank from the functions above)
QK_CHANNELS = qk_channels(DEFAULT_RANK)
QK_SCALE_ATOMS = qk_scale_atoms(DEFAULT_RANK)
# T: the anchor of a softmax row (its running maximum) only moves, and O is only rescaled, when the maximum of a key
# tile exceeds it by more than T binades
RESCALE_THRESHOLD = 4.0
# f32 subnormal ulps per log2 unit of the exp2 landing: the landed bits are the f16 pattern of 2^t
SUBNORMAL_ULPS = 1024
# log2(e) / sqrt(HEAD_DIM), the score prescale folded into Q': the scores arrive in the log2 units the kernels
# exponentiate, and the score scale stays a power of two
QPRE = 1.4426950408889634 / HEAD_DIM**0.5
