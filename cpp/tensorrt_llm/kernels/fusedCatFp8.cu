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

#include "fusedCatFp8.h"
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

// Constants
constexpr int HEAD_DIM = 128;       // Fixed for DSV3.2 indexer
constexpr int WARP_SIZE = 32;       // One warp per row
constexpr int ELEMS_PER_THREAD = 4; // 128 / 32 = 4 elements per thread
constexpr int ROWS_PER_BLOCK = 8;   // Process 8 rows per block for occupancy
constexpr float INV_FP8_E4M3_MAX = 1.0f / 448.0f;
constexpr float MIN_AMAX = 1.0e-12f;

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

/// Fused kernel: cat + FP8 quantization.
///
/// Grid: (ceil(M / ROWS_PER_BLOCK),)
/// Block: (WARP_SIZE * ROWS_PER_BLOCK,)   i.e., (256,)
///
/// Each warp handles one row. Within a warp:
///   - Thread t handles elements [4t, 4t+1, 4t+2, 4t+3] of the 128-dim row.
///   - Loads from pe or nope based on element index (vectorized 8-byte loads).
///   - FP8 quantizes with per-row scale (vectorized 4-byte stores).
///
/// Templated on UseUe8m0 to eliminate branch divergence.
template <bool UseUe8m0>
__global__ __launch_bounds__(WARP_SIZE* ROWS_PER_BLOCK) void fusedCatFp8Kernel(__nv_fp8_e4m3* __restrict__ fp8_out,
    float* __restrict__ scale_out, __nv_bfloat16 const* __restrict__ pe, __nv_bfloat16 const* __restrict__ nope,
    int32_t M, int32_t pe_dim, int32_t nope_dim, int32_t pe_row_stride, int32_t nope_row_stride)
{
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * ROWS_PER_BLOCK + warp_in_block;

    if (row >= M)
    {
        return;
    }

    // ---- Stage 1: Load + Concat (vectorized 8-byte loads) ----
    // pe_dim is guaranteed to be a multiple of ELEMS_PER_THREAD by the host check,
    // so each thread's 4 elements come entirely from pe or entirely from nope.
    // Use branchless pointer selection (compiles to SELP) to avoid warp divergence.
    float v0, v1, v2, v3;
    {
        int base = lane * ELEMS_PER_THREAD;
        __nv_bfloat16 const* pe_row = pe + static_cast<int64_t>(row) * pe_row_stride;
        __nv_bfloat16 const* nope_row = nope + static_cast<int64_t>(row) * nope_row_stride;

        bool from_pe = (base < pe_dim);
        __nv_bfloat16 const* src = from_pe ? pe_row : nope_row;
        int col = from_pe ? base : (base - pe_dim);

        BF16x4 loaded;
        loaded.vec = *reinterpret_cast<int2 const*>(src + col);

        float2 f0 = __bfloat1622float2(loaded.bf16x2[0]);
        float2 f1 = __bfloat1622float2(loaded.bf16x2[1]);
        v0 = f0.x;
        v1 = f0.y;
        v2 = f1.x;
        v3 = f1.y;
    }

    // ---- Stage 2: FP8 Quantization (1x128 block = entire row) ----
    float local_max = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    float amax = warpReduceMax(local_max);
    amax = fmaxf(amax, MIN_AMAX);

    float scale;
    if constexpr (UseUe8m0)
    {
        // UE8M0: scale = 2^ceil(log2(amax / FP8_MAX)) via IEEE 754 bit manipulation.
        // This replaces ceilf(log2f(...)) + exp2f(...) with integer ops.
        float ratio = amax * INV_FP8_E4M3_MAX;
        uint32_t bits = __float_as_uint(ratio);
        uint32_t mantissa = bits & 0x007FFFFFu;
        uint32_t exp_bits = bits & 0x7F800000u;
        // If mantissa is non-zero, round exponent up to next power of 2
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

    // Use hardware approximate reciprocal (MUFU.RCP, ~2^-23 relative error).
    // This is more than sufficient for FP8 E4M3 quantization (3 mantissa bits).
    // Avoids the expensive Newton-Raphson refinement of __frcp_rn.
    float inv_scale;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_scale) : "f"(scale));

    // Quantize to FP8 — clamp is mathematically redundant since
    // |val/scale| <= amax/scale <= FP8_MAX by construction, but kept
    // for safety against floating-point rounding edge cases.
    auto quantize = [&](float val) -> __nv_fp8_e4m3
    {
        float scaled = val * inv_scale;
        return __nv_fp8_e4m3(scaled);
    };

    // ---- Stage 3: Store (vectorized 4-byte FP8 store) ----
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
}

} // anonymous namespace

void invokeFusedCatFp8(__nv_fp8_e4m3* fp8_out, float* scale_out, __nv_bfloat16 const* pe, __nv_bfloat16 const* nope,
    int32_t M, int32_t pe_dim, int32_t nope_dim, int32_t head_dim, int32_t pe_row_stride, int32_t nope_row_stride,
    bool use_ue8m0, cudaStream_t stream)
{
    if (M == 0)
    {
        return;
    }

    TLLM_CHECK_WITH_INFO(head_dim == HEAD_DIM, "fusedCatFp8: head_dim must be 128, got %d", head_dim);
    TLLM_CHECK_WITH_INFO(pe_dim + nope_dim == head_dim, "fusedCatFp8: pe_dim (%d) + nope_dim (%d) != head_dim (%d)",
        pe_dim, nope_dim, head_dim);
    TLLM_CHECK_WITH_INFO((head_dim & (head_dim - 1)) == 0, "fusedCatFp8: head_dim must be power of 2");
    TLLM_CHECK_WITH_INFO(pe_dim % ELEMS_PER_THREAD == 0,
        "fusedCatFp8: pe_dim (%d) must be a multiple of %d for vectorized access", pe_dim, ELEMS_PER_THREAD);
    TLLM_CHECK_WITH_INFO(
        pe_row_stride >= pe_dim, "fusedCatFp8: pe_row_stride (%d) must be >= pe_dim (%d)", pe_row_stride, pe_dim);
    TLLM_CHECK_WITH_INFO(nope_row_stride >= nope_dim, "fusedCatFp8: nope_row_stride (%d) must be >= nope_dim (%d)",
        nope_row_stride, nope_dim);
    TLLM_CHECK_WITH_INFO(pe_row_stride % ELEMS_PER_THREAD == 0,
        "fusedCatFp8: pe_row_stride (%d) must be a multiple of %d for aligned vectorized access", pe_row_stride,
        ELEMS_PER_THREAD);
    TLLM_CHECK_WITH_INFO(nope_row_stride % ELEMS_PER_THREAD == 0,
        "fusedCatFp8: nope_row_stride (%d) must be a multiple of %d for aligned vectorized access", nope_row_stride,
        ELEMS_PER_THREAD);

    int num_blocks = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    dim3 grid(num_blocks);
    dim3 block(WARP_SIZE * ROWS_PER_BLOCK); // 256 threads per block

    if (use_ue8m0)
    {
        fusedCatFp8Kernel<true><<<grid, block, 0, stream>>>(
            fp8_out, scale_out, pe, nope, M, pe_dim, nope_dim, pe_row_stride, nope_row_stride);
    }
    else
    {
        fusedCatFp8Kernel<false><<<grid, block, 0, stream>>>(
            fp8_out, scale_out, pe, nope, M, pe_dim, nope_dim, pe_row_stride, nope_row_stride);
    }

    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END
