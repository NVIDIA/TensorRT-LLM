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

import math

# The releases of nvidia-cutlass-dsl the kernels are validated with
VALIDATED_DSL_VERSIONS = ("4.8.0", "4.8.1")
# The alignment the launches assume of their pointers: of the address of a workspace and of every buffer offset in it
ALIGN = 256
# The default draw of the rows of a compensation basis takes one token in SAMPLE_FRAC, and at least twice the
# channels of the larger compensation term (never fewer than MIN_SAMPLE_ROWS)
SAMPLE_FRAC = 256
MIN_SAMPLE_ROWS = 64
# A Gram matrix of the rows of a compensation basis whose trace is below this counts as zero, like the one of zero
# rows: the squares of its tokens are too close to the smallest FP32 to hold its directions
MIN_GRAM_TRACE = 2.0**-100

HEAD_DIM = 128  # channels of q, k, v and o per head
KEY_TILE = 128  # keys of a KV tile of the attention, and of a block of the preparation passes
INV_SQRT_HEAD_DIM = HEAD_DIM**-0.5
QK_BLOCK = 16  # channels of an NVFP4 block of Q' / K', which has one scale: a compensation term fills whole blocks
# Compensation: Q' and K' carry, after the rotated head channels, the r_q channels of the Q-residual term, the r_k
# channels of the K-residual term (multiples of QK_BLOCK; the rank of a call is the pair (r_q, r_k)) and zero blocks
# up to the compensation channels of the QK build: COMP_CHANNELS, or twice as many where a family's
# max_comp_channels() offers them and r_q + r_k is above COMP_CHANNELS
COMP_CHANNELS = 64
QK_ROW_BYTES = 128  # a packed token row of Q' / K': room for 256 E2M1 codes, the first qk_channels(rank) of them used
# The bases kinds of the two compensation terms: "shared", the leading eigenvectors of one weighted sum of the Gram
# matrices of Q and K; "separate", each term on the leading eigenvectors of the other operand's Gram matrix
BASES = ("shared", "separate")


def key_tiles(s: int) -> int:
    return (s + KEY_TILE - 1) // KEY_TILE


def qk_channels(rank: tuple[int, int]) -> int:
    """Return the channels of Q' and K' of the QK build that holds the compensation terms of a rank (r_q, r_k): the
    rotated head channels, then COMP_CHANNELS compensation channels, or twice as many for terms above them."""
    wide = rank[0] + rank[1] > COMP_CHANNELS
    return HEAD_DIM + (2 * COMP_CHANNELS if wide else COMP_CHANNELS)


def qk_scale_atoms(channels: int) -> int:
    """Return the 512-byte atoms of the Q' / K' scale factors of a tile of a QK build of that many channels, one per
    64 channels."""
    return channels // 64


# The channels and the scale atoms of the QK build of COMP_CHANNELS compensation channels, the one build of the sm120
# family
QK_CHANNELS = HEAD_DIM + COMP_CHANNELS
QK_SCALE_ATOMS = qk_scale_atoms(QK_CHANNELS)
# T: the anchor of a softmax row (its running maximum) only moves, and O is only rescaled, when the maximum of a key
# tile exceeds it by more than T binades. The default of the attention's rescale threshold
RESCALE_THRESHOLD = 4.0
E4M3_MAX = 448.0  # the largest finite E4M3 value
# The rescale thresholds that the attention offers per PV format, the default first:
#   * "fp8": the thresholds up to the default. Its E4M3 P reaches 2^-(18 - T) of a row's anchor: a threshold above
#     the default would cut weights that the default keeps
#   * "nvfp4": thresholds above the default as well. Its P has a scale per block of keys
RESCALE_THRESHOLDS = {
    "fp8": (RESCALE_THRESHOLD, 0.0, 2.0),
    "nvfp4": (RESCALE_THRESHOLD, 0.0, 2.0, 6.0, 8.0),
}


def p_prescale_log2(rescale_threshold: float) -> int:
    """Return log2 of the E4M3 P of a score at the anchor of its row, floor(log2(448) - T) at the rescale threshold T:
    the P of a score T binades above the anchor is then within the E4M3 range. ValueError, while a kernel is traced,
    unless 0 <= T <= log2(448)."""
    if not 0.0 <= rescale_threshold <= math.log2(E4M3_MAX):
        raise ValueError(
            f"an E4M3 P takes a rescale threshold from 0 to log2({E4M3_MAX:g}), got {rescale_threshold!r}"
        )
    return int(math.log2(E4M3_MAX) - rescale_threshold)


E2M1_MAX = 6.0  # the largest E2M1 value
# The rules for the scale of an NVFP4 block of QK_BLOCK values of the head channels of Q' and K' (prepare:
# scale_rule). m is the largest magnitude of the block, and a scale is rounded to three mantissa bits:
#   "max"     m / E2M1_MAX: the largest magnitude lands on the largest code
#   "band"    m / WIDE_SCALE_MAX where a value of the block has lo <= |x| < hi, with lo = m * SCALE_BAND_LOW and
#             hi = m * SCALE_BAND_HIGH as rounded FP32 products; else m / E2M1_MAX
#   "search"  the block is quantized with m / E2M1_MAX and with m / WIDE_SCALE_MAX and keeps the scale with the
#             smaller FP32 sum of (code * scale - x)^2, m / E2M1_MAX when the sums are equal
# The compensation terms take the residual of the kept scale; a block of a compensation term takes m / E2M1_MAX
# under every rule
SCALE_RULES = ("max", "band", "search")
# The default of scale_rule
DEFAULT_SCALE_RULE = "band"
# The code of the largest magnitude of a block under the wider scale of "band" and "search"
WIDE_SCALE_MAX = 4.0
# The band of "band", in units of the largest magnitude of the block
SCALE_BAND_LOW, SCALE_BAND_HIGH = 0.76, 0.85
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
