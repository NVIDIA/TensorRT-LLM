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

#include "fusedCatHadamardFp8.h"
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
constexpr float FP8_E4M3_MAX = 448.0f;

/// Butterfly Hadamard step: XOR shuffle across lanes.
__device__ __forceinline__ void hadamardButterflyShfl(float& v0, float& v1, float& v2, float& v3, int xor_mask)
{
    float o0 = __shfl_xor_sync(0xFFFFFFFF, v0, xor_mask);
    float o1 = __shfl_xor_sync(0xFFFFFFFF, v1, xor_mask);
    float o2 = __shfl_xor_sync(0xFFFFFFFF, v2, xor_mask);
    float o3 = __shfl_xor_sync(0xFFFFFFFF, v3, xor_mask);

    // Standard Hadamard butterfly: for element index i with stride s,
    //   if (i & s) == 0: new_val = old_val + old_partner
    //   if (i & s) != 0: new_val = old_partner - old_val
    // This is: new_val = sign * old_val + old_partner, where sign = ((lane & xor_mask) ? -1 : 1).
    int lane = threadIdx.x % WARP_SIZE;
    float sign = (lane & xor_mask) ? -1.0f : 1.0f;
    v0 = sign * v0 + o0;
    v1 = sign * v1 + o1;
    v2 = sign * v2 + o2;
    v3 = sign * v3 + o3;
}

/// Butterfly Hadamard step: within a single thread's registers.
__device__ __forceinline__ void hadamardButterflyLocal(float& a, float& b)
{
    float tmp = a;
    a = a + b;
    b = tmp - b;
}

/// Compute UE8M0 scale: 2^ceil(log2(amax / FP8_MAX))
/// This ensures amax / scale <= FP8_MAX, with scale being a power of 2.
__device__ __forceinline__ float computeUe8m0Scale(float amax)
{
    // Minimum value to avoid log2(0)
    constexpr float MIN_AMAX = 1.0e-12f;
    amax = fmaxf(amax, MIN_AMAX);
    float log2_scale = ceilf(log2f(amax / FP8_E4M3_MAX));
    return exp2f(log2_scale);
}

/// Compute standard (non-UE8M0) scale: amax / FP8_MAX
__device__ __forceinline__ float computeStdScale(float amax)
{
    constexpr float MIN_AMAX = 1.0e-12f;
    amax = fmaxf(amax, MIN_AMAX);
    return amax / FP8_E4M3_MAX;
}

/// Warp-wide max reduction
__device__ __forceinline__ float warpReduceMax(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/// Fused kernel: cat + Hadamard transform + FP8 quantization.
///
/// Grid: (ceil(M / ROWS_PER_BLOCK),)
/// Block: (WARP_SIZE * ROWS_PER_BLOCK,)   i.e., (256,)
///
/// Each warp handles one row. Within a warp:
///   - Thread t handles elements [4t, 4t+1, 4t+2, 4t+3] of the 128-dim row.
///   - Loads from pe or nope based on element index.
///   - Performs 7-stage Walsh-Hadamard butterfly (stages 0-1 local, stages 2-6 via shfl_xor).
///   - FP8 quantizes with per-row scale.
__global__ void fusedCatHadamardFp8Kernel(__nv_fp8_e4m3* __restrict__ fp8_out, float* __restrict__ scale_out,
    __nv_bfloat16 const* __restrict__ pe, __nv_bfloat16 const* __restrict__ nope, int32_t M, int32_t pe_dim,
    int32_t nope_dim, bool use_ue8m0)
{
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * ROWS_PER_BLOCK + warp_in_block;

    if (row >= M)
    {
        return;
    }

    // ---- Stage 1: Load + Concat ----
    // Thread t handles elements [4t, 4t+1, 4t+2, 4t+3].
    // Element i: if i < pe_dim, load from pe; otherwise load from nope[i - pe_dim].
    float v0, v1, v2, v3;
    {
        int base = lane * ELEMS_PER_THREAD;
        __nv_bfloat16 const* pe_row = pe + static_cast<int64_t>(row) * pe_dim;
        __nv_bfloat16 const* nope_row = nope + static_cast<int64_t>(row) * nope_dim;

        auto load_elem = [&](int idx) -> float
        {
            if (idx < pe_dim)
            {
                return __bfloat162float(pe_row[idx]);
            }
            else
            {
                return __bfloat162float(nope_row[idx - pe_dim]);
            }
        };

        v0 = load_elem(base + 0);
        v1 = load_elem(base + 1);
        v2 = load_elem(base + 2);
        v3 = load_elem(base + 3);
    }

    // ---- Stage 2: Walsh-Hadamard Transform (7 stages for dim=128) ----
    // The full Hadamard matrix H_128 = H_2 ⊗ H_2 ⊗ ... (7 times).
    // We decompose this into butterfly stages with increasing stride.
    //
    // Stages 0-1: Within-thread (stride 1, 2 within the 4-element group)
    // Stage 0: stride 1 - pairs (v0,v1) and (v2,v3)
    hadamardButterflyLocal(v0, v1);
    hadamardButterflyLocal(v2, v3);

    // Stage 1: stride 2 - pairs (v0,v2) and (v1,v3)
    hadamardButterflyLocal(v0, v2);
    hadamardButterflyLocal(v1, v3);

    // Stages 2-6: Cross-thread via __shfl_xor_sync
    // stride 4  -> XOR mask 1  (swap between lanes 0↔1, 2↔3, ...)
    // stride 8  -> XOR mask 2
    // stride 16 -> XOR mask 4
    // stride 32 -> XOR mask 8
    // stride 64 -> XOR mask 16
    hadamardButterflyShfl(v0, v1, v2, v3, 1);
    hadamardButterflyShfl(v0, v1, v2, v3, 2);
    hadamardButterflyShfl(v0, v1, v2, v3, 4);
    hadamardButterflyShfl(v0, v1, v2, v3, 8);
    hadamardButterflyShfl(v0, v1, v2, v3, 16);

    // Apply normalization: scale by head_dim^{-0.5} = 128^{-0.5} = 1/sqrt(128)
    constexpr float HADAMARD_SCALE = 0.088388347648318440f; // 1.0 / sqrt(128)
    v0 *= HADAMARD_SCALE;
    v1 *= HADAMARD_SCALE;
    v2 *= HADAMARD_SCALE;
    v3 *= HADAMARD_SCALE;

    // ---- Stage 3: FP8 Quantization (1x128 block = entire row) ----
    // Find per-row amax
    float local_max = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    float amax = warpReduceMax(local_max);

    // Compute scale
    float scale;
    if (use_ue8m0)
    {
        scale = computeUe8m0Scale(amax);
    }
    else
    {
        scale = computeStdScale(amax);
    }

    float inv_scale = 1.0f / scale;

    // Quantize to FP8
    auto quantize = [&](float val) -> __nv_fp8_e4m3
    {
        float scaled = fminf(fmaxf(val * inv_scale, -FP8_E4M3_MAX), FP8_E4M3_MAX);
        return __nv_fp8_e4m3(scaled);
    };

    __nv_fp8_e4m3 q0 = quantize(v0);
    __nv_fp8_e4m3 q1 = quantize(v1);
    __nv_fp8_e4m3 q2 = quantize(v2);
    __nv_fp8_e4m3 q3 = quantize(v3);

    // ---- Stage 4: Store ----
    int base_out = row * HEAD_DIM + lane * ELEMS_PER_THREAD;
    fp8_out[base_out + 0] = q0;
    fp8_out[base_out + 1] = q1;
    fp8_out[base_out + 2] = q2;
    fp8_out[base_out + 3] = q3;

    // Only lane 0 writes the scale for this row
    if (lane == 0)
    {
        scale_out[row] = scale;
    }
}

} // anonymous namespace

void invokeFusedCatHadamardFp8(__nv_fp8_e4m3* fp8_out, float* scale_out, __nv_bfloat16 const* pe,
    __nv_bfloat16 const* nope, int32_t M, int32_t pe_dim, int32_t nope_dim, int32_t head_dim, bool use_ue8m0,
    cudaStream_t stream)
{
    if (M == 0)
    {
        return;
    }

    TLLM_CHECK_WITH_INFO(head_dim == HEAD_DIM, "fusedCatHadamardFp8: head_dim must be 128, got %d", head_dim);
    TLLM_CHECK_WITH_INFO(pe_dim + nope_dim == head_dim,
        "fusedCatHadamardFp8: pe_dim (%d) + nope_dim (%d) != head_dim (%d)", pe_dim, nope_dim, head_dim);
    TLLM_CHECK_WITH_INFO((head_dim & (head_dim - 1)) == 0, "fusedCatHadamardFp8: head_dim must be power of 2");

    int num_blocks = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    dim3 grid(num_blocks);
    dim3 block(WARP_SIZE * ROWS_PER_BLOCK); // 256 threads per block

    fusedCatHadamardFp8Kernel<<<grid, block, 0, stream>>>(fp8_out, scale_out, pe, nope, M, pe_dim, nope_dim, use_ue8m0);

    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END
