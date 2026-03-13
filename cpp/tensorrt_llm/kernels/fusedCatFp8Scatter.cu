/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fusedCatFp8Scatter.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int HEAD_DIM = 128;
constexpr int WARP_SIZE = 32;
constexpr int ELEMS_PER_THREAD = 4; // 128 / 32
constexpr int ROWS_PER_BLOCK = 8;
constexpr float INV_FP8_E4M3_MAX = 1.0f / 448.0f;
constexpr float MIN_AMAX = 1.0e-12f;
constexpr int32_t SCALE_SIZE = 4;
constexpr int32_t PER_TOKEN_SIZE = HEAD_DIM + SCALE_SIZE; // 132

/// Warp-wide max reduction
__device__ __forceinline__ float warpReduceMax(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/// Helper union for vectorized BF16 loads (4 BF16 values = 8 bytes).
union BF16x4
{
    int2 vec;
    __nv_bfloat162 bf16x2[2];
};

/// Helper union for vectorized FP8 stores (4 FP8 values = 4 bytes).
union FP8x4
{
    uint32_t u32;
    __nv_fp8_e4m3 fp8[4];
};

/**
 * Unravel a flat element index into a byte offset for the indexer K cache.
 * Specialized for the DeepSeek-V3.2 layout: d2=1, d3=132 (compile-time constants).
 * This enables the compiler to use multiplication-based integer division (4 cycles)
 * instead of general-purpose int64 division (20+ cycles) for the d3 unravel.
 */
__device__ __forceinline__ int64_t flatIndexToMemoryOffset(
    int64_t flat_idx, int32_t d1_mask, int32_t d1_shift, int64_t s0, int64_t s1, int64_t s3)
{
    // d3 = PER_TOKEN_SIZE = 132 (compile-time constant → fast multiply-based reduction)
    int32_t i3 = flat_idx % PER_TOKEN_SIZE;
    flat_idx /= PER_TOKEN_SIZE;
    // d2 = 1: skip (always 0)
    // d1 is power of 2 → bitwise AND/shift instead of integer division
    int32_t i1 = static_cast<int32_t>(flat_idx) & d1_mask;
    int32_t i0 = static_cast<int32_t>(flat_idx) >> d1_shift;
    return i0 * s0 + i1 * s1 + i3 * s3;
}

/// Fused kernel: cat + FP8 quantization + scatter to paged K cache.
///
/// Grid: (ceil(M / ROWS_PER_BLOCK),)
/// Block: (WARP_SIZE * ROWS_PER_BLOCK,)   i.e., (256,)
///
/// Each warp handles one row:
///   - Loads from pe/nope, quantizes to FP8
///   - Writes FP8 data to contiguous output AND to paged K cache
///   - Lane 0 writes scale to contiguous output AND to paged K cache
///
/// Optimizations:
///   1. Compile-time d3=132, d2=1: fast integer division via multiplication reduction.
///   2. __ldg for read-only inputs (pe, nope, slot_mappings).
///   3. All threads compute own cache offset in SIMT lockstep (no shuffle overhead).
template <bool UseUe8m0>
__global__ __launch_bounds__(WARP_SIZE* ROWS_PER_BLOCK) void fusedCatFp8ScatterKernel(
    __nv_fp8_e4m3* __restrict__ fp8_out, float* __restrict__ scale_out, uint8_t* __restrict__ k_cache,
    __nv_bfloat16 const* __restrict__ pe, __nv_bfloat16 const* __restrict__ nope,
    int64_t const* __restrict__ slot_mapping_fp8, int64_t const* __restrict__ slot_mapping_scale, int32_t M,
    int32_t pe_dim, int32_t nope_dim, int32_t pe_row_stride, int32_t nope_row_stride, int64_t cache_stride_0,
    int64_t cache_stride_1, int64_t cache_stride_3, int32_t d1_mask, int32_t d1_shift)
{
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * ROWS_PER_BLOCK + warp_in_block;

    if (row >= M)
    {
        return;
    }

    // ---- Stage 1: Load + Concat (vectorized 8-byte loads) ----
    float v0, v1, v2, v3;
    {
        int base = lane * ELEMS_PER_THREAD;
        __nv_bfloat16 const* pe_row = pe + static_cast<int64_t>(row) * pe_row_stride;
        __nv_bfloat16 const* nope_row = nope + static_cast<int64_t>(row) * nope_row_stride;

        bool from_pe = (base < pe_dim);
        __nv_bfloat16 const* src = from_pe ? pe_row : nope_row;
        int col = from_pe ? base : (base - pe_dim);

        BF16x4 loaded;
        loaded.vec = __ldg(reinterpret_cast<int2 const*>(src + col));

        float2 f0 = __bfloat1622float2(loaded.bf16x2[0]);
        float2 f1 = __bfloat1622float2(loaded.bf16x2[1]);
        v0 = f0.x;
        v1 = f0.y;
        v2 = f1.x;
        v3 = f1.y;
    }

    // ---- Stage 2: FP8 Quantization ----
    float local_max = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    float amax = warpReduceMax(local_max);
    amax = fmaxf(amax, MIN_AMAX);

    float scale;
    if constexpr (UseUe8m0)
    {
        float ratio = amax * INV_FP8_E4M3_MAX;
        uint32_t bits = __float_as_uint(ratio);
        uint32_t mantissa = bits & 0x007FFFFFu;
        uint32_t exp_bits = bits & 0x7F800000u;
        if (mantissa != 0u)
        {
            exp_bits += 0x00800000u;
        }
        scale = __uint_as_float(exp_bits);
    }
    else
    {
        scale = amax * INV_FP8_E4M3_MAX;
    }

    float inv_scale;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_scale) : "f"(scale));

    auto quantize = [&](float val) -> __nv_fp8_e4m3
    {
        float scaled = val * inv_scale;
        return __nv_fp8_e4m3(scaled);
    };

    // ---- Stage 3: Store to contiguous output ----
    FP8x4 packed;
    packed.fp8[0] = quantize(v0);
    packed.fp8[1] = quantize(v1);
    packed.fp8[2] = quantize(v2);
    packed.fp8[3] = quantize(v3);

    int base_out = row * HEAD_DIM + lane * ELEMS_PER_THREAD;
    *reinterpret_cast<uint32_t*>(fp8_out + base_out) = packed.u32;

    if (lane == 0)
    {
        scale_out[row] = scale;
    }

    // ---- Stage 4: Scatter to paged K cache ----
    // All threads compute their own offset in SIMT lockstep (no shuffle overhead).
    // Uses compile-time d3=132 and bitwise d1 for fast integer operations.
    int64_t flat_idx_fp8_base = __ldg(&slot_mapping_fp8[row]);
    if (flat_idx_fp8_base >= 0)
    {
        int32_t head_dim_idx = lane * ELEMS_PER_THREAD;
        int64_t flat_idx = flat_idx_fp8_base + head_dim_idx;

        int64_t dst_offset
            = flatIndexToMemoryOffset(flat_idx, d1_mask, d1_shift, cache_stride_0, cache_stride_1, cache_stride_3);

        *reinterpret_cast<uint32_t*>(&k_cache[dst_offset]) = packed.u32;

        if (lane == 0)
        {
            int64_t flat_idx_scale_base = __ldg(&slot_mapping_scale[row]);
            int64_t dst_offset_scale = flatIndexToMemoryOffset(
                flat_idx_scale_base, d1_mask, d1_shift, cache_stride_0, cache_stride_1, cache_stride_3);

            *reinterpret_cast<float*>(&k_cache[dst_offset_scale]) = scale;
        }
    }
}

} // anonymous namespace

void invokeFusedCatFp8Scatter(__nv_fp8_e4m3* fp8_out, float* scale_out, uint8_t* k_cache, __nv_bfloat16 const* pe,
    __nv_bfloat16 const* nope, int64_t const* slot_mapping_fp8, int64_t const* slot_mapping_scale, int32_t M,
    int32_t pe_dim, int32_t nope_dim, int32_t head_dim, int32_t pe_row_stride, int32_t nope_row_stride, bool use_ue8m0,
    int32_t cache_dim_0, int32_t cache_dim_1, int32_t cache_dim_2, int32_t cache_dim_3, int64_t cache_stride_0,
    int64_t cache_stride_1, int64_t cache_stride_2, int64_t cache_stride_3, cudaStream_t stream)
{
    if (M == 0)
    {
        return;
    }

    TLLM_CHECK_WITH_INFO(head_dim == HEAD_DIM, "fusedCatFp8Scatter: head_dim must be 128, got %d", head_dim);
    TLLM_CHECK_WITH_INFO(pe_dim + nope_dim == head_dim,
        "fusedCatFp8Scatter: pe_dim (%d) + nope_dim (%d) != head_dim (%d)", pe_dim, nope_dim, head_dim);
    TLLM_CHECK_WITH_INFO((head_dim & (head_dim - 1)) == 0, "fusedCatFp8Scatter: head_dim must be power of 2");
    TLLM_CHECK_WITH_INFO(pe_dim % ELEMS_PER_THREAD == 0, "fusedCatFp8Scatter: pe_dim (%d) must be a multiple of %d",
        pe_dim, ELEMS_PER_THREAD);
    TLLM_CHECK_WITH_INFO(pe_row_stride >= pe_dim, "fusedCatFp8Scatter: pe_row_stride (%d) must be >= pe_dim (%d)",
        pe_row_stride, pe_dim);
    TLLM_CHECK_WITH_INFO(nope_row_stride >= nope_dim,
        "fusedCatFp8Scatter: nope_row_stride (%d) must be >= nope_dim (%d)", nope_row_stride, nope_dim);
    TLLM_CHECK_WITH_INFO(pe_row_stride % ELEMS_PER_THREAD == 0,
        "fusedCatFp8Scatter: pe_row_stride (%d) must be a multiple of %d", pe_row_stride, ELEMS_PER_THREAD);
    TLLM_CHECK_WITH_INFO(nope_row_stride % ELEMS_PER_THREAD == 0,
        "fusedCatFp8Scatter: nope_row_stride (%d) must be a multiple of %d", nope_row_stride, ELEMS_PER_THREAD);
    TLLM_CHECK_WITH_INFO(cache_dim_2 == 1, "fusedCatFp8Scatter: cache_dim_2 must be 1 (got %d)", cache_dim_2);
    TLLM_CHECK_WITH_INFO(cache_dim_3 == PER_TOKEN_SIZE,
        "fusedCatFp8Scatter: cache_dim_3 must be %d (head_dim + scale_size, got %d)", PER_TOKEN_SIZE, cache_dim_3);
    TLLM_CHECK_WITH_INFO((cache_dim_1 & (cache_dim_1 - 1)) == 0,
        "fusedCatFp8Scatter: cache_dim_1 (block_size=%d) must be a power of 2", cache_dim_1);

    // Pre-compute mask and shift for power-of-2 block_size division
    int32_t d1_mask = cache_dim_1 - 1;
    int32_t d1_shift = __builtin_ctz(cache_dim_1);

    int num_blocks = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    dim3 grid(num_blocks);
    dim3 block(WARP_SIZE * ROWS_PER_BLOCK); // 256 threads per block

    if (use_ue8m0)
    {
        fusedCatFp8ScatterKernel<true><<<grid, block, 0, stream>>>(fp8_out, scale_out, k_cache, pe, nope,
            slot_mapping_fp8, slot_mapping_scale, M, pe_dim, nope_dim, pe_row_stride, nope_row_stride, cache_stride_0,
            cache_stride_1, cache_stride_3, d1_mask, d1_shift);
    }
    else
    {
        fusedCatFp8ScatterKernel<false><<<grid, block, 0, stream>>>(fp8_out, scale_out, k_cache, pe, nope,
            slot_mapping_fp8, slot_mapping_scale, M, pe_dim, nope_dim, pe_row_stride, nope_row_stride, cache_stride_0,
            cache_stride_1, cache_stride_3, d1_mask, d1_shift);
    }

    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END
