/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/glmMoeDsa/glmRouterGemm.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{

namespace
{

// Convert 8 bfloat16 values from a uint4 to float array - optimized conversion
template <int VPT>
__device__ __forceinline__ void bf16_uint4_to_float8(uint4 const& vec, float* dst)
{
    __nv_bfloat16* bf16_ptr = reinterpret_cast<__nv_bfloat16*>(const_cast<uint4*>(&vec));

#pragma unroll
    for (int i = 0; i < VPT; i++)
    {
        dst[i] = __bfloat162float(bf16_ptr[i]);
    }
}

// mat_a: the activations [kNumTokens, kHiddenDim]; mat_b: the weight [kNumExperts, kHiddenDim]; out: the logits
// [kNumTokens, kNumExperts]. Before the grid-dependency wait each CTA prefetches its weight row into L2 (no kernel of
// the step writes the weight), so the weight loads after the wait hit L2; the dependent grid may launch right after the
// wait instead of after the logits are written, and must wait on its own grid dependency before reading them.
template <typename T, int kBlockSize, int VPT, int kNumTokens, int kNumExperts, int kHiddenDim>
__global__ __launch_bounds__(128, 1) void router_gemm_kernel(float* out, T const* mat_a, T const* mat_b)
{
    // Each block handles one expert column
    int const n_idx = blockIdx.x;
    int const tid = threadIdx.x;
    constexpr int kWarpSize = 32;
    constexpr int kNumWarps = kBlockSize / kWarpSize;
    // Constants for this kernel
    constexpr int k_elems_per_k_iteration = VPT * kBlockSize;
    constexpr int k_iterations = kHiddenDim / k_elems_per_k_iteration; // Total K iterations

    // Initialize accumulators for all M rows
    float acc[kNumTokens] = {};

    // Shared memory for warp-level reduction
    __shared__ float sm_reduction[kNumTokens][kNumWarps]; // kNumWarps

    // The weight row of expert n_idx (a column of B in column-major order)
    T const* b_col = mat_b + n_idx * kHiddenDim;

    // Pre-compute k_base values for each iteration to help compiler optimize
    int k_bases[k_iterations];
#pragma unroll
    for (int ki = 0; ki < k_iterations; ki++)
    {
        k_bases[ki] = ki * k_elems_per_k_iteration + tid * VPT;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    {
        // The primary grid does not write the weights, so this CTA's column can stream into L2 while the primary
        // runs; the weight loads below then hit L2. What is loaded and the accumulation order are unchanged.
        constexpr int kLineBytes = 128;
        constexpr int kColumnBytes = kHiddenDim * static_cast<int>(sizeof(T));
        static_assert(kColumnBytes % kLineBytes == 0, "the weight column must span whole 128-byte lines");
        char const* b_col_bytes = reinterpret_cast<char const*>(b_col);
        for (int offset = tid * kLineBytes; offset < kColumnBytes; offset += kBlockSize * kLineBytes)
        {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(b_col_bytes + offset) : "memory");
        }
    }
    cudaGridDependencySynchronize();
    // Every dependent waits on its own grid dependency before it reads the logits, so it may launch now and overlap
    // its prologue with this GEMV.
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    // Process the GEMM in chunks
    for (int ki = 0; ki < k_iterations; ki++)
    {
        int const k_base = k_bases[ki];

        // Load B matrix values using vector load (8 bf16 values)
        uint4 b_vec = *reinterpret_cast<uint4 const*>(b_col + k_base);

        // Convert B values to float
        float b_float[VPT];
        bf16_uint4_to_float8<VPT>(b_vec, b_float);

// Process each token
#pragma unroll
        for (int m_idx = 0; m_idx < kNumTokens; m_idx++)
        {
            // Load both rows of A matrix using vector loads
            uint4 a_vec = *reinterpret_cast<uint4 const*>(mat_a + (m_idx * kHiddenDim) + k_base);

            // Convert A values to float
            float a_float[VPT];
            bf16_uint4_to_float8<VPT>(a_vec, a_float);

// Process elements in this chunk
#pragma unroll
            for (int k = 0; k < VPT; k++)
            {
                float a = a_float[k];
                float b = b_float[k];
                acc[m_idx] += a * b;
            }
        }
    }

    // Perform warp-level reduction
    int const warpSize = 32;
    int const warpId = tid / warpSize;
    int const laneId = tid % warpSize;

    // Register for warp-level reduction results
    float warp_result[kNumTokens];

#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; m_idx++)
    {
        warp_result[m_idx] = acc[m_idx];
    }

// Perform warp-level reduction using optimized butterfly pattern
#pragma unroll
    for (int m = 0; m < kNumTokens; m++)
    {
        float sum = warp_result[m];

        // Butterfly reduction pattern
        sum += __shfl_xor_sync(0xffffffff, sum, 16);
        sum += __shfl_xor_sync(0xffffffff, sum, 8);
        sum += __shfl_xor_sync(0xffffffff, sum, 4);
        sum += __shfl_xor_sync(0xffffffff, sum, 2);
        sum += __shfl_xor_sync(0xffffffff, sum, 1);

        // Only the first thread in each warp stores to shared memory
        if (laneId == 0)
        {
            sm_reduction[m][warpId] = sum;
        }
    }

    __syncthreads();

    // Final reduction across warps (only first thread)
    if (tid == 0)
    {
#pragma unroll
        for (int m = 0; m < kNumTokens; m++)
        {
            float final_sum = 0.0f;

// Sum across the kNumWarps
#pragma unroll
            for (int w = 0; w < kNumWarps; w++)
            {
                final_sum += sm_reduction[m][w];
            }

            // Write final result
            out[m * kNumExperts + n_idx] = final_sum;
        }
    }
}

template <int kNumTokens>
void launchRouterGemm(
    float* logits, __nv_bfloat16 const* hiddenStates, __nv_bfloat16 const* weight, cudaStream_t stream)
{
    using T = __nv_bfloat16;
    constexpr int VPT = 16 / sizeof(T);
    constexpr int kBlockSize = 128;
    cudaLaunchConfig_t config;
    config.gridDim = kRouterNumExperts;
    config.blockDim = kBlockSize;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config,
        router_gemm_kernel<T, kBlockSize, VPT, kNumTokens, kRouterNumExperts, kRouterHiddenSize>, logits, hiddenStates,
        weight));
}

// Launches the instance of numTokens, one of kNumTokens..kRouterMaxTokens.
template <int kNumTokens>
void dispatchRouterGemm(
    float* logits, __nv_bfloat16 const* hiddenStates, __nv_bfloat16 const* weight, int numTokens, cudaStream_t stream)
{
    if constexpr (kNumTokens <= kRouterMaxTokens)
    {
        if (numTokens == kNumTokens)
        {
            launchRouterGemm<kNumTokens>(logits, hiddenStates, weight, stream);
        }
        else
        {
            dispatchRouterGemm<kNumTokens + 1>(logits, hiddenStates, weight, numTokens, stream);
        }
    }
    else
    {
        TLLM_THROW("The GLM router GEMV takes 1 to %d tokens, got %d", kRouterMaxTokens, numTokens);
    }
}

} // namespace

void invokeRouterGemm(
    float* logits, __nv_bfloat16 const* hiddenStates, __nv_bfloat16 const* weight, int numTokens, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(
        numTokens >= 1, "The GLM router GEMV takes 1 to %d tokens, got %d", kRouterMaxTokens, numTokens);
    dispatchRouterGemm<1>(logits, hiddenStates, weight, numTokens, stream);
}

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
