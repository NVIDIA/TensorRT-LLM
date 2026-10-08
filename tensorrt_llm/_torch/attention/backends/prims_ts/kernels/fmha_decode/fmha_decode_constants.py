# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants shared by the FMHA decode TS implementation.

Keep non-obvious constants here with their rationale so config, resource, and
reduction code can use named values without duplicating comments.
"""

import math

# B200 has 148 SMs. Use this only when the runtime SM query is unavailable,
# so auto split-KV selection remains deterministic in offline/test flows.
FALLBACK_SM_COUNT_B200 = 148

# Shared-memory budget constants are in KiB because profile sizing is based on
# the hardware SMEM carveout. KV staging is capped below the full
# 218 KiB budget so Q staging, page-offset staging, and scratch can coexist.
TOTAL_SMEM_BUDGET_KIB = 218
MAX_KV_STAGE_SMEM_KIB = 144
BYTES_PER_KIB = 1024
# Dynamic SMEM one CTA can use on the SM100 family: 228 KiB per SM minus the
# 1 KiB the system reserves. The complete decode layout (pipelines, barriers,
# metadata and scratch) must fit it; TOTAL_SMEM_BUDGET_KIB sizes the Q and
# K/V pipelines below it.
SMEM_CAPACITY_KIB = 227

# The M64N256 profile stages one 256-row K or V tile per shared-ring slot. A
# 16-bit tile is 64 KiB, so three stages, the 16-KiB Q stage, the tail
# exchange and the metadata fill the SM100 carveout. A byte-wide tile is
# 32 KiB; with four stages every K load takes the slot the previous QK freed
# instead of waiting for a V slot that its tile's PV still holds. Keep both
# exact-profile depths separate from the conservative, topology-independent
# MAX_KV_STAGE_SMEM_KIB inference above.
KV_TILE_256_SHARED_FIFO_STAGES = 3
KV_TILE_256_BYTE_WIDE_SHARED_FIFO_STAGES = 4

# The four semantic K64 atoms of a KV256 tile are staged in these physical K
# slots. On the WS 2x2 datapath each TMEM lane half computes 128 contiguous QK
# columns, while PV pairs KV64 blocks 2j and 2j+1 per K step; staging K as
# (0, 2, 1, 3) lets P alias S unchanged with V in natural order. The score
# column mapping undoes the permutation for masking and Sage scales.
KV_TILE_256_K_SLOT_FOR_SEMANTIC_ATOM = (0, 2, 1, 3)

# Keep the old maximum as the exponent reference while a new maximum is at
# most eight log2 units larger. This avoids an output-correction round without
# letting an intermediate probability exceed 2**8; the softmax identity is
# unchanged apart from normal finite-precision rounding. As in the
# FlashInfer/TRT-LLM policy, this assumes normal model logits rather than
# adversarial values outside the qualified probability bound. Streamed KV256
# and block-sparse Keeps profiles apply it; see
# ``FmhaDecodeConfig.defers_softmax_anchor_updates``.
SOFTMAX_RESCALE_THRESHOLD_LOG2 = 8.0

# A launch bound makes ptxas honor warpgroup ``setmaxnreg`` allocations, but
# the resulting register hand-off has a fixed cost. Paired B200 measurements
# show that it is amortized once a Q64/KV256 CTA processes at least 32 tiles
# (8K dense KV tokens); shorter loops are as fast or faster without it.
KV_TILE_256_REGISTER_REALLOCATION_MIN_TILES = 32

# TMA-swizzled Q rows are padded to 128 B before computing how many KV stages
# fit in the remaining SMEM budget.
Q_ROW_ALIGNMENT_BYTES = 128
BITS_PER_BYTE = 8

# Conservative per-CTA budget for cluster distributed-SMEM reduction leader
# staging. The reducer-owner CTA holds every split's partial O/stats for its
# row band; keep staging below this cap to avoid dynamic SMEM overflow.
CLUSTER_PARTIAL_SMEM_LIMIT_KIB = 96
MAX_CLUSTER_PARTIAL_SMEM_BYTES = CLUSTER_PARTIAL_SMEM_LIMIT_KIB * BYTES_PER_KIB

# Fused GMEM/cluster partial O is staged in 16-bit elements, and each row's stats
# are a float2 pair (max, sum). The separate-GMEM workspace contract instead
# uses one FP32 log2-LSE scalar and normalized 16-bit O.
PARTIAL_O_ELEMENT_BYTES = 2
PARTIAL_STATS_VALUES_PER_ROW = 2
SEPARATE_REDUCTION_LSE_VALUES_PER_ROW = 1
FP32_BYTES = 4

# Hardware clusterDim.x limit for this decode reduction layout.
MAX_CLUSTER_DIM_X = 16

# One split-KV CTA should cover at least two loop iterations so reduction
# overhead does not dominate tiny per-split K ranges.
MIN_LOOP_ITERS_PER_SPLIT = 2

# Grouped sparse attention may add two producer-only groups to the common
# four-warpgroup decode topology.
MAX_WARP_GROUPS = 6

# Grouped sparse routes keep one membership byte per selected page in a table
# separate from the plain Int32 locators. Four bytes share one Int32 word. Q1 does
# not consume membership; Q2--Q8 use the packed table for their per-query
# softmax mask. A byte supports all eight query positions without changing the ABI.
Q_TOKEN_KV_BLOCK_SPARSE_PAGE_MEMBERSHIP_BITS = 8
Q_TOKEN_KV_BLOCK_SPARSE_PAGE_MEMBERSHIP_MASK = (
    1 << Q_TOKEN_KV_BLOCK_SPARSE_PAGE_MEMBERSHIP_BITS
) - 1
Q_TOKEN_KV_BLOCK_SPARSE_PAGE_MEMBERSHIPS_PER_WORD = (
    32 // Q_TOKEN_KV_BLOCK_SPARSE_PAGE_MEMBERSHIP_BITS
)
# Holding a complete locator window removes duplicate K/V metadata traffic,
# but a graph-safe Q2/Q4 worst-case bound can otherwise reserve 8 KiB or more
# of SMEM for rows that commonly use only three KV tiles. Beyond this crossover,
# stage locators one tile at a time while retaining the compact packed
# membership row separately.
Q_TOKEN_KV_BLOCK_SPARSE_HELD_LOCATOR_MAX_TILES = 8

# Default used only when the launch helper has no resolved decode config.
# Config-aware FMHA and block-sparse callers pass their selected KV tile.
AUTO_LAUNCH_TILE_SIZE_KV = 128

# Split-KV is worthwhile below one static SM wave only when each CTA still owns
# enough K/V work to amortize GMEM reduction. The B200-qualified
# crossover is 2,048 tokens: 16 KV128 tiles or 8 KV256 tiles.
SPLIT_KV_MIN_TOKENS_PER_CTA = 2_048

# Two interleaved K/V instances form the decode cadence: instance 0 is the
# first K/P/V stream in each loop iteration and instance 1 is the second. These
# integer tags are passed through TS work calls and used in constexpr branches.
KV_INST0 = 0
KV_INST1 = 1

# Compact K/V selector used by shared SMEM resources. Keep this as an integer
# contract because the JIT work-call plumbing expects constexpr scalar values.
KV_KIND_K = 0
KV_KIND_V = 1

# One hardware warp. Barrier participant counts are expressed as warps * lanes.
WARP_THREADS = 32

# TMEM column layout for staged SwapsMmaAb O. A TMEM row holds 256 columns, and
# the tcgen05 descriptor encodes a 16-row jump in the high 16 bits.
TMEM_COLUMNS_PER_ROW = 256
TMEM_MAX_ALLOCATION_COLUMNS = 512
TMEM_ROW_STRIDE = 16 << 16

# Number of scalar softmax/output values packed in one register for the
# supported element widths.
PACKED_REGISTER_BYTES = 4
FP8_VALUES_PER_REG = 4
FP16_VALUES_PER_REG = 2

# Bytes of the K-major operand row one tcgen05 MMA instruction consumes: 32
# one-byte or 16 two-byte K elements.
MMA_K_STEP_BYTES = 32

# FP8 probabilities are quantized as 448 * p (the E4M3 maximum); row sums and
# sink terms share the scale, and the output normalization divides it out.
FP8_P_QUANT_SCALE = 448.0
FP8_P_QUANT_LOG2_SCALE = math.log2(FP8_P_QUANT_SCALE)

# tcgen05 SMEM descriptor geometry: address offsets count 16-byte units and a
# 128-byte swizzle atom spans one 128-byte row per K or MN index.
SMEM_DESC_UNIT_BYTES = 16
SWIZZLE_128B_ROW_BYTES = 128

# Per-lane register ownership denominators for packed output fragments.
FP8_OUTPUT_ELEMENTS_PER_REG_GROUP = 512
FP16_OUTPUT_ELEMENTS_PER_REG_GROUP = 256

# SwapsMmaAb maps up to eight Q heads into one q-repetition group.
Q_REPETITION_GROUP_HEADS = 8

# Packed P register count per q-repetition in the SwapsMmaAb path.
FP8_P_PACKED_REGS_PER_Q_REPEAT = 2
FP16_P_PACKED_REGS_PER_Q_REPEAT = 4

# Standalone reducer CTA shape. Each thread owns a 16-byte vector, so one CTA
# reduces one contiguous 8 KiB slice of the partial-O buffer.
REDUCTION_THREADS_PER_CTA = 512
REDUCTION_BYTES_PER_THREAD = 16
REDUCTION_BYTES_PER_SLICE = REDUCTION_THREADS_PER_CTA * REDUCTION_BYTES_PER_THREAD

# Clustered standalone reducer shape. One 128-thread CTA covers a contiguous
# 2 KiB partial-O slice with one 16-byte vector per thread. Each cluster rank
# owns a compile-time 2, 4, or 8 split slots. Loads are batched in groups of at
# most four; padded split slots remain neutral and never form GMEM pointers.
PARALLEL_REDUCTION_THREADS_PER_CTA = 128
PARALLEL_REDUCTION_BYTES_PER_SLICE = (
    PARALLEL_REDUCTION_THREADS_PER_CTA * REDUCTION_BYTES_PER_THREAD
)
PARALLEL_REDUCTION_LOAD_BATCH = 4
PARALLEL_REDUCTION_FINAL_REDUCERS = 4

# Each reduction thread produces an 8-element O vector backed by four packed
# 16-bit registers.
OUTPUT_VALUES_PER_THREAD = 8
FP8_PACKED_OUTPUT_REGS_PER_THREAD = 2
PACKED_OUTPUT_REGS_PER_THREAD = 4

# INT8 Q/K scores accumulate onto the bit pattern of ``1.5 * 2**23``. Integers
# in ``[2**23, 2**24)`` are FP32 numbers with unit spacing and every D = 128
# INT8 dot product has ``|score| <= 2**21``, so the INT32 accumulator is the
# exact FP32 number ``12582912.0 + score`` and the softmax reads it without a
# conversion, removing the bias once per scale group.
INT32_SCORE_BIAS = 12582912.0
# One ``kind::f16`` MMA step writes the bias before the INT8 K steps: both
# operands read one BF16 tile whose K = 16 rows repeat
# ``[1024, 1024, 1024, 0]``, so the twelve products ``2**20`` sum exactly to
# ``1.5 * 2**23``. The tile is unswizzled K-major 32-byte rows with 128-byte
# LBO and 256-byte SBO core-matrix strides.
INT32_SCORE_SEED_MMA_K = 16
INT32_SCORE_SEED_TILE_WORDS = (0x44804480, 0x00004480)
INT32_SCORE_SEED_TILE_LBO = 128
INT32_SCORE_SEED_TILE_SBO = 256
