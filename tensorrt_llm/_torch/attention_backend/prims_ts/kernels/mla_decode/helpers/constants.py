# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Shared instruction and layout constants for MLA decode TS examples."""

# CUDA warp geometry used by lane ownership, butterfly reductions, and TMEM row
# addressing.  Keep masks and shifts named so instruction-level code does not
# hide the warp contract behind raw bit constants.
WARP_LANES = 32
WARP_LANE_MASK = WARP_LANES - 1
WARP_LANE_SHIFT = 5
HALF_WARP_LANES = 16
HALF_WARP_MASK = HALF_WARP_LANES - 1
QUAD_LANES = 4
QUAD_LANE_MASK = QUAD_LANES - 1
QUAD_LANE_SHIFT = 2
OCTET_LANES = 8
OCTET_LANE_MASK = OCTET_LANES - 1
WARP_REDUCTION_BFLY_DISTANCES = (16, 8, 4, 2, 1)

# One warpgroup is four warps.  Several softmax/P/O protocols synchronize one
# full warpgroup through a named CTA barrier.
WARPGROUP_WARPS = 4
WARPGROUP_THREADS = WARPGROUP_WARPS * WARP_LANES

# Kernel-level TMEM lifecycle synchronization.  Barrier ID 15 is reserved for
# alloc/dealloc phases in the 1CTA path; the 2CTA path uses config-owned IDs.
TMEM_LIFECYCLE_BARRIER_ID = 15
TMEM_DEALLOC_MBAR_THREADS = WARP_LANES

# STSM and vector-copy helpers share the tcgen05 128B-row, 16B-column swizzled
# SMEM layout.  The shift constants are the log2 form of the byte units used in
# address calculations.
SMEM_ROW_BYTES = 128
SMEM_VECTOR_BYTES = 16
SMEM_WORD_BYTES = 4
SMEM_ROW_BYTE_SHIFT = 7
SMEM_VECTOR_BYTE_SHIFT = 4
SMEM_WORD_BYTE_SHIFT = 2
STSM_MATRIX_LANES = OCTET_LANES
STSM_MATRIX_LANE_SHIFT = 3
STSM_X4_REG_COUNT = 4
STSM_MATRICES_PER_WARP = QUAD_LANES
STSM_MATRICES_PER_WARP_SHIFT = 2
STSM_WARPS_PER_SLICE = 2
STSM_WARPS_PER_SLICE_SHIFT = 1
STSM_ROW_BLOCK_ROWS = 16
O_STAGE_COPY_SEGMENT_BYTES = 2048
SWIZZLE_ROW_MASK = OCTET_LANE_MASK

# tcgen05 TMEM load/store instruction geometry used by MLA softmax, correction,
# and epilogue paths.
TCGEN05_32B_SHAPE = "32x32b"
TCGEN05_16X256B_SHAPE = "16x256b"
TCGEN05_32B_REGS_PER_LOAD = 32
TCGEN05_16X256B_REGS_PER_LOAD = 4
TCGEN05_16X32BX2_BF16_P_STRIDE = 32
TCGEN05_16X32BX2_FP8_P_STRIDE = 16
TCGEN05_SECOND_PANEL_ADDR_OFFSET = 16 << 16

# Softmax scratch stores the max state first and the sum state in a second
# fixed-size panel.  The offset is in Uint32 scratch words.
SOFTMAX_SCRATCH_SUM_WORD_OFFSET = 384
SCORE_ROWS_PER_Q_PAIR = 2
SCORE_TOKENS_PER_QK_GROUP = 8

# CTA-local split-reduction barrier used after the scale-writing warp publishes
# per-split LSE rescale factors in SMEM.
SPLIT_REDUCTION_SCALE_BARRIER_ID = 4

# SmemPResource uses adjacent CTA barriers for the two FP8 P producer instances
# after byte-transposed STSM stores have published their SMEM payloads.
SMEM_P_FP8_STORE_BARRIER_BASE_ID = 4

# Page-offset entries are Int32 page IDs staged through cp.async.
PAGE_OFFSET_BYTES = 4
CP_ASYNC_CACHE_CA = "ca"

# Dense MLA kernels specialize paged-KV addressing for this explicit ABI set.
# Each listed size exactly partitions a 128-token KV tile; other page sizes are
# outside the current kernel ABI.
SUPPORTED_MLA_PAGE_SIZES = (16, 32, 64, 128)

# Throughput 2CTA epilogue maps 128 local threads onto two 64-row groups and
# 128-column output halves for vectorized GMEM publication.
EPILOGUE_THREAD_TILE_THREADS = 128
EPILOGUE_THREAD_TILE_MASK = EPILOGUE_THREAD_TILE_THREADS - 1
EPILOGUE_ROW_THREADS = 64
EPILOGUE_ROW_MASK = EPILOGUE_ROW_THREADS - 1
EPILOGUE_COLUMN_GROUP_SHIFT = 6
BF16_OUTPUT_VECTOR_ELEMENTS = 8
FP8_OUTPUT_VECTOR_ELEMENTS = 16
PACKED_FP8_OUTPUT_REGS = 4

# tcgen05 SMEM descriptors are advanced in 16B units.  The normal Q/K next-K
# increment is two units; the wrapped K descriptor moves backward across the
# descriptor ring to the next logical 128-wide head-dim block.
TCGEN05_DESC_NEXT_K_BLOCK_UNITS = 2
TCGEN05_DESC_WRAPPED_K_BLOCK_UNITS = 1018

# Tensor-map descriptors and workspace allocations use 1024B alignment because
# the backing STensor/TMA views are 1 KiB aligned in these kernels.
TMA_DESCRIPTOR_ALIGNMENT_BYTES = 1024
