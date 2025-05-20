/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>

#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/groupRmsNormKernels/groupRmsNormKernels.h"

namespace tensorrt_llm::kernels::group_rms_norm
{
// Helper function to calculate the number of warps to launch for GroupRMSNormBase
template <typename DType, int n>
uint32_t calculateNumWarpsBase(GroupRMSParams<n> const& params)
{
    constexpr uint32_t kPackedSize = sizeof(float4) / sizeof(DType);
    uint32_t input_chunk_per_warp = 32 * kPackedSize;

    // Calculate rounded input dimensions and total input length
    int rounded_input_dims[n];
    for (uint32_t i = 0; i < params.num_inputs; i++)
    {
        // Make rounded_input_dims[i] a multiple of 32 * kPackedSize
        rounded_input_dims[i]
            = (params.input_last_dims[i] + input_chunk_per_warp - 1) / input_chunk_per_warp * input_chunk_per_warp;
    }

    // Calculate total warps needed
    uint32_t total_input_length = std::accumulate(rounded_input_dims, rounded_input_dims + params.num_inputs, 0);
    uint32_t total_warps_needed = total_input_length / input_chunk_per_warp;
    return std::min<uint32_t>(32, total_warps_needed);
}

template <typename DType, int n>
struct LargeBatchWarpsInfo
{
    uint32_t num_warps_to_launch;   // Total warps to launch
    uint32_t num_warps_to_launch_0; // Warps for first input
    uint32_t num_warps_to_launch_1; // Warps for second input
    uint32_t rounds_0;              // Rounds for first input
    uint32_t rounds_1;              // Rounds for second input
};

// Helper function to calculate the number of warps to launch for GroupRMSNormKernelLargeBatch
template <typename DType, int n>
LargeBatchWarpsInfo<DType, n> calculateNumWarpsLargeBatch(GroupRMSParams<n> const& params)
{
    constexpr uint32_t kPackedSize = sizeof(float4) / sizeof(DType);
    uint32_t input_chunk_per_warp = 32 * kPackedSize;

    // Calculate warps needed for each input
    uint32_t warps_needed_0 = (params.input_last_dims[0] + input_chunk_per_warp - 1) / input_chunk_per_warp;
    uint32_t warps_needed_1 = (params.input_last_dims[1] + input_chunk_per_warp - 1) / input_chunk_per_warp;

    LargeBatchWarpsInfo<DType, n> info;
    info.num_warps_to_launch_0 = std::min((uint32_t) 32, warps_needed_0);
    info.num_warps_to_launch_1 = std::min((uint32_t) 32, warps_needed_1);

    // Use the maximum of the two for the final warps to launch
    info.num_warps_to_launch = std::max(info.num_warps_to_launch_0, info.num_warps_to_launch_1);

    // Calculate rounds needed for each input
    info.rounds_0 = (warps_needed_0 + info.num_warps_to_launch_0 - 1) / info.num_warps_to_launch_0;
    info.rounds_1 = (warps_needed_1 + info.num_warps_to_launch_1 - 1) / info.num_warps_to_launch_1;

    return info;
}

// Allocate more warps to deal with the second input
template <typename DType, typename PackedType, int n, bool EnableWeights, bool MultiRounds>
__global__ void GroupRMSNormBaseKernel(GroupRMSParams<n> params, int rounds)
{
    const uint32_t batch_idx = blockIdx.x; // Maps to batch size
    constexpr uint32_t warp_size = 32;
    const uint32_t warp_idx = threadIdx.y;
    const uint32_t lane_idx = threadIdx.x;

    static constexpr int kPackedSize = sizeof(PackedType) / sizeof(DType);

    // Each thread calculates its own partial sum
    alignas(128) __shared__ float smem_rsqrts[32];
    alignas(128) __shared__ uint32_t smem_input_mask[32];
    alignas(128) __shared__ float smem_warp_sum_sqs[32];
    if (warp_idx == 0)
    {
        smem_rsqrts[lane_idx] = 0.0f;
        // Initialize input mask to 33 to indicate no input
        smem_input_mask[lane_idx] = 33;
        smem_warp_sum_sqs[lane_idx] = 0.0f;
    }
    float warp_acc = 0.0f;
    PackedType const* __restrict__ weight_ptr = nullptr;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    // Find which input current warp operates on
    const uint32_t input_idx = params.warp_input_idx[warp_idx]; // Maps to input index
    const uint32_t warp_start = params.warp_prefix_sum[input_idx];
    const uint32_t local_warp_idx = warp_idx - warp_start;

    PackedType const* __restrict__ input_ptr = params.inputs[input_idx];
    PackedType* __restrict__ output_ptr = params.outputs[input_idx];

    if constexpr (EnableWeights)
    {
        weight_ptr = params.weights[input_idx];
    }

    uint32_t block_offset = batch_idx * params.input_strides[input_idx];
    uint32_t round_offset = warp_size * kPackedSize;
    const uint32_t input_dim = params.input_last_dims[input_idx];

    uint32_t idx_round0 = block_offset + local_warp_idx * round_offset * rounds + lane_idx * kPackedSize;

    // Store the first round of data as local variable to reduce memory access
    PackedType input_cache;
    PackedType weight_cache;
    if (idx_round0 < input_dim + block_offset)
    {
        input_cache = input_ptr[idx_round0 / kPackedSize];
        if constexpr (EnableWeights)
        {
            weight_cache = weight_ptr[(idx_round0 - block_offset) / kPackedSize];
        }

#pragma unroll
        for (uint32_t j = 0; j < kPackedSize; j++)
        {
            float v = static_cast<float>(reinterpret_cast<DType*>(&input_cache)[j]);
            warp_acc += v * v;
        }
    }

    // Process round1+
    // If input dtype is fp16, round1+ is needed when input_dim > 8192, which is uncommon
    if constexpr (MultiRounds)
    {
        for (uint32_t i = 1; i < rounds; i++)
        {
            uint32_t idx
                = block_offset + local_warp_idx * round_offset * rounds + i * round_offset + lane_idx * kPackedSize;
            if (idx < input_dim + block_offset)
            {
                PackedType packed_data = input_ptr[idx / kPackedSize];
#pragma unroll
                for (uint32_t j = 0; j < kPackedSize; j++)
                {
                    float v = static_cast<float>(reinterpret_cast<DType*>(&packed_data)[j]);
                    warp_acc += v * v;
                }
            }
        }
    }
    float warp_sum = tensorrt_llm::common::warpReduceSum(warp_acc);
    if (lane_idx == 0)
    {
        smem_warp_sum_sqs[warp_idx] = warp_sum;
    }

// Extra _syncwarp for sm < 900, needed to avoid race condition on smem_input_mask write.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
    __syncwarp();
#endif
    if (lane_idx == 0)
    {
        smem_input_mask[warp_idx] = input_idx;
    }

    __syncthreads();

    warp_acc = 0.0f;
    // Cross wrap reduction on all inputs
    if (warp_idx < n)
    {
        // Each warp sums one input
        if (warp_idx == smem_input_mask[lane_idx])
        {
            warp_acc = smem_warp_sum_sqs[lane_idx];
        }
        float sum_sq = tensorrt_llm::common::warpReduceSum(warp_acc);
        if (lane_idx == 0)
        {
            smem_rsqrts[warp_idx] = rsqrtf(sum_sq / params.input_last_dims[warp_idx] + params.eps);
        }
    }

    __syncthreads();

    // Apply normalization
    if (idx_round0 < input_dim + block_offset)
    {
        PackedType packed_output;
#pragma unroll
        for (uint32_t j = 0; j < kPackedSize; j++)
        {
            if constexpr (EnableWeights)
            {
                reinterpret_cast<DType*>(&packed_output)[j] = static_cast<DType>(
                    static_cast<float>(reinterpret_cast<DType*>(&input_cache)[j]) * smem_rsqrts[input_idx]
                    * (static_cast<float>(reinterpret_cast<DType*>(&weight_cache)[j]) + params.weight_bias));
            }
            else
            {
                reinterpret_cast<DType*>(&packed_output)[j] = static_cast<DType>(
                    static_cast<float>(reinterpret_cast<DType*>(&input_cache)[j]) * smem_rsqrts[input_idx]);
            }
        }
        output_ptr[idx_round0 / kPackedSize] = packed_output;
    }

    if constexpr (MultiRounds)
    {
        for (uint32_t i = 1; i < rounds; i++)
        {
            uint32_t idx
                = block_offset + local_warp_idx * round_offset * rounds + i * round_offset + lane_idx * kPackedSize;
            if (idx < input_dim + block_offset)
            {
                PackedType packed_input = input_ptr[idx / kPackedSize];
                PackedType packed_output;

#pragma unroll
                for (uint32_t j = 0; j < kPackedSize; j++)
                {
                    if constexpr (EnableWeights)
                    {
                        reinterpret_cast<DType*>(&packed_output)[j]
                            = static_cast<float>(reinterpret_cast<DType*>(&packed_input)[j]) * smem_rsqrts[input_idx]
                            * (static_cast<float>(
                                   reinterpret_cast<DType const*>(&weight_ptr[(idx - block_offset) / kPackedSize])[j])
                                + params.weight_bias);
                    }
                    else
                    {
                        reinterpret_cast<DType*>(&packed_output)[j]
                            = static_cast<float>(reinterpret_cast<DType*>(&packed_input)[j]) * smem_rsqrts[input_idx];
                    }
                }
                output_ptr[idx / kPackedSize] = packed_output;
            }
        }
    }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// This kernel is optimized for large batch sizes with 2 inputs
// Some warps process both inputs
// Fewer warps are launched allowing for more blocks to be scheduled on one SM
template <typename DType, typename PackedType, int n, bool EnableWeights, bool MultiRounds_0, bool MultiRounds_1>
__global__ void GroupRMSNormKernelLargeBatch(
    GroupRMSParams<n> params, int rounds_0, int rounds_1, int warp_size_0, int warp_size_1)
{
    const uint32_t batch_idx = blockIdx.x; // Maps to batch size
    constexpr uint32_t warp_size = 32;
    const uint32_t warp_idx = threadIdx.y;
    const uint32_t lane_idx = threadIdx.x;
    static constexpr int kPackedSize = sizeof(PackedType) / sizeof(DType);

    alignas(128) __shared__ float smem_rsqrts[n][32];

    if (warp_idx < n)
    {
        smem_rsqrts[warp_idx][lane_idx] = 0.0f;
    }

    const uint32_t round_offset = warp_size * kPackedSize;

    float sum_sq_0 = 0.0f;
    float sum_sq_1 = 0.0f;

    // Cache for prefetching
    PackedType input_0_cache;
    PackedType input_1_cache;
    PackedType weight_0_cache;
    PackedType weight_1_cache;

    PackedType const* __restrict__ weight_ptr_0 = nullptr;
    PackedType const* __restrict__ weight_ptr_1 = nullptr;

    bool process_input_0 = warp_idx < warp_size_0;
    bool process_input_1 = warp_idx < warp_size_1;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    // Get input pointers
    PackedType const* __restrict__ input_ptr_0 = params.inputs[0];
    PackedType* __restrict__ output_ptr_0 = params.outputs[0];
    PackedType const* __restrict__ input_ptr_1 = params.inputs[1];
    PackedType* __restrict__ output_ptr_1 = params.outputs[1];

    if constexpr (EnableWeights)
    {
        weight_ptr_0 = params.weights[0];
        weight_ptr_1 = params.weights[1];
    }

    const uint32_t block_offset_0 = batch_idx * params.input_strides[0];
    const uint32_t block_offset_1 = batch_idx * params.input_strides[1];

    const uint32_t input_dim_0 = params.input_last_dims[0];
    const uint32_t input_dim_1 = params.input_last_dims[1];

    uint32_t idx_0 = block_offset_0 + warp_idx * round_offset * rounds_0 + lane_idx * kPackedSize;
    uint32_t idx_1 = block_offset_1 + warp_idx * round_offset * rounds_1 + lane_idx * kPackedSize;

    // Prefetch inputs for round0
    if (idx_0 < block_offset_0 + input_dim_0)
    {
        input_0_cache = input_ptr_0[idx_0 / kPackedSize];
        if constexpr (EnableWeights)
        {
            weight_0_cache = weight_ptr_0[(idx_0 - block_offset_0) / kPackedSize];
        }
    }
    if (idx_1 < block_offset_1 + input_dim_1)
    {
        input_1_cache = input_ptr_1[idx_1 / kPackedSize];
        if constexpr (EnableWeights)
        {
            weight_1_cache = weight_ptr_1[(idx_1 - block_offset_1) / kPackedSize];
        }
    }

    // Process round0
    if (idx_0 < block_offset_0 + input_dim_0)
    {
#pragma unroll
        for (uint32_t j = 0; j < kPackedSize; j++)
        {
            float val = static_cast<float>(reinterpret_cast<DType*>(&input_0_cache)[j]);
            sum_sq_0 += val * val;
        }
    }

    // Process round1+
    // If input dtype is fp16, round1+ is needed when input_dim > 8192, which is uncommon
    if constexpr (MultiRounds_0)
    {
        for (uint32_t i = 1; i < rounds_0; i++)
        {
            uint32_t idx
                = block_offset_0 + warp_idx * round_offset * rounds_0 + i * round_offset + lane_idx * kPackedSize;
            if (idx < block_offset_0 + input_dim_0)
            {
                PackedType packed_data = input_ptr_0[idx / kPackedSize];
#pragma unroll
                for (uint32_t j = 0; j < kPackedSize; j++)
                {
                    float val = static_cast<float>(reinterpret_cast<DType*>(&packed_data)[j]);
                    sum_sq_0 += val * val;
                }
            }
        }
    }

    if (process_input_0)
    {
        float warp_sum = tensorrt_llm::common::warpReduceSum(sum_sq_0);
        if (lane_idx == 0)
        {
            smem_rsqrts[0][warp_idx] = warp_sum;
        }
    }

    // Process round0
    if (idx_1 < block_offset_1 + input_dim_1)
    {
#pragma unroll
        for (uint32_t j = 0; j < kPackedSize; j++)
        {
            float val = static_cast<float>(reinterpret_cast<DType*>(&input_1_cache)[j]);
            sum_sq_1 += val * val;
        }
    }

    // Process round1+
    if constexpr (MultiRounds_1)
    {
        for (uint32_t i = 1; i < rounds_1; i++)
        {
            uint32_t idx
                = block_offset_1 + warp_idx * round_offset * rounds_1 + i * round_offset + lane_idx * kPackedSize;
            if (idx < block_offset_1 + input_dim_1)
            {
                PackedType packed_data = input_ptr_1[idx / kPackedSize];
#pragma unroll
                for (uint32_t j = 0; j < kPackedSize; j++)
                {
                    float val = static_cast<float>(reinterpret_cast<DType*>(&packed_data)[j]);
                    sum_sq_1 += val * val;
                }
            }
        }
    }

    // Store warp reduction to shared memory
    if (process_input_1)
    {
        float warp_sum = tensorrt_llm::common::warpReduceSum(sum_sq_1);
        if (lane_idx == 0)
        {
            smem_rsqrts[1][warp_idx] = warp_sum;
        }
    }

    __syncthreads();

    // The if-elseif code block is faster than if (warp_idx < 2)
    if (warp_idx == 0)
    {
        // Final reduction across warps
        float block_sum = tensorrt_llm::common::warpReduceSum(smem_rsqrts[0][lane_idx]);

        // Compute rsqrt
        if (lane_idx == 0)
        {
            smem_rsqrts[0][0] = rsqrtf(block_sum / input_dim_0 + params.eps);
        }
    }
    else if (warp_idx == 1)
    {
        float block_sum = tensorrt_llm::common::warpReduceSum(smem_rsqrts[1][lane_idx]);
        // Compute rsqrt
        if (lane_idx == 0)
        {
            smem_rsqrts[1][0] = rsqrtf(block_sum / input_dim_1 + params.eps);
        }
    }

    __syncthreads();

    // Apply normalization
    if (idx_0 < block_offset_0 + input_dim_0)
    {
        PackedType packed_output;
#pragma unroll
        for (uint32_t j = 0; j < kPackedSize; j++)
        {
            if constexpr (EnableWeights)
            {
                reinterpret_cast<DType*>(&packed_output)[j] = static_cast<DType>(
                    static_cast<float>(reinterpret_cast<DType*>(&input_0_cache)[j]) * smem_rsqrts[0][0]
                    * (static_cast<float>(reinterpret_cast<DType*>(&weight_0_cache)[j]) + params.weight_bias));
            }
            else
            {
                reinterpret_cast<DType*>(&packed_output)[j] = static_cast<DType>(
                    static_cast<float>(reinterpret_cast<DType*>(&input_0_cache)[j]) * smem_rsqrts[0][0]);
            }
        }
        output_ptr_0[idx_0 / kPackedSize] = packed_output;
    }

    if constexpr (MultiRounds_0)
    {
        for (uint32_t i = 1; i < rounds_0; i++)
        {
            uint32_t idx
                = block_offset_0 + warp_idx * round_offset * rounds_0 + i * round_offset + lane_idx * kPackedSize;
            if (idx < block_offset_0 + input_dim_0)
            {
                PackedType packed_input = input_ptr_0[idx / kPackedSize];
                PackedType packed_output;

#pragma unroll
                for (uint32_t j = 0; j < kPackedSize; j++)
                {
                    if constexpr (EnableWeights)
                    {
                        reinterpret_cast<DType*>(&packed_output)[j] = static_cast<DType>(
                            static_cast<float>(reinterpret_cast<DType*>(&packed_input)[j]) * smem_rsqrts[0][0]
                            * (static_cast<float>(reinterpret_cast<DType const*>(
                                   &weight_ptr_0[(idx - block_offset_0) / kPackedSize])[j])
                                + params.weight_bias));
                    }
                    else
                    {
                        reinterpret_cast<DType*>(&packed_output)[j] = static_cast<DType>(
                            static_cast<float>(reinterpret_cast<DType*>(&packed_input)[j]) * smem_rsqrts[0][0]);
                    }
                }
                output_ptr_0[idx / kPackedSize] = packed_output;
            }
        }
    }

    if (idx_1 < block_offset_1 + input_dim_1)
    {
        PackedType packed_output;
#pragma unroll
        for (uint32_t j = 0; j < kPackedSize; j++)
        {
            if constexpr (EnableWeights)
            {
                reinterpret_cast<DType*>(&packed_output)[j] = static_cast<DType>(
                    static_cast<float>(reinterpret_cast<DType*>(&input_1_cache)[j]) * smem_rsqrts[1][0]
                    * (static_cast<float>(reinterpret_cast<DType*>(&weight_1_cache)[j]) + params.weight_bias));
            }
            else
            {
                reinterpret_cast<DType*>(&packed_output)[j] = static_cast<DType>(
                    static_cast<float>(reinterpret_cast<DType*>(&input_1_cache)[j]) * smem_rsqrts[1][0]);
            }
        }
        output_ptr_1[idx_1 / kPackedSize] = packed_output;
    }

    if constexpr (MultiRounds_1)
    {
        for (uint32_t i = 1; i < rounds_1; i++)
        {
            uint32_t idx
                = block_offset_1 + warp_idx * round_offset * rounds_1 + i * round_offset + lane_idx * kPackedSize;
            if (idx < block_offset_1 + input_dim_1)
            {
                PackedType packed_input = input_ptr_1[idx / kPackedSize];
                PackedType packed_output;

#pragma unroll
                for (uint32_t j = 0; j < kPackedSize; j++)
                {
                    if constexpr (EnableWeights)
                    {
                        reinterpret_cast<DType*>(&packed_output)[j] = static_cast<DType>(
                            static_cast<float>(reinterpret_cast<DType*>(&packed_input)[j]) * smem_rsqrts[1][0]
                            * (static_cast<float>(reinterpret_cast<DType const*>(
                                   &weight_ptr_1[(idx - block_offset_1) / kPackedSize])[j])
                                + params.weight_bias));
                    }
                    else
                    {
                        reinterpret_cast<DType*>(&packed_output)[j] = static_cast<DType>(
                            static_cast<float>(reinterpret_cast<DType*>(&packed_input)[j]) * smem_rsqrts[1][0]);
                    }
                }
                output_ptr_1[idx / kPackedSize] = packed_output;
            }
        }
    }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename DType, int n, bool EnableWeights>
void GroupRMSNormBaseKernel(GroupRMSParams<n>& params)
{
    // Kernel assertions
    constexpr uint32_t kPackedSize = sizeof(float4) / sizeof(DType);
    TLLM_CHECK_WITH_INFO(params.num_inputs <= 2,
        "Only up to 2 inputs are supported with performance guarantees. Kernels with more than 2 inputs can be "
        "instantiated.");
    int rounded_input_dims[n];
    uint32_t input_chunk_per_warp = 32 * kPackedSize;
    for (uint32_t i = 0; i < params.num_inputs; i++)
    {
        TLLM_CHECK_WITH_INFO(
            params.input_last_dims[i] % 32 == 0, "The last dimension of input must be divisible by 32.");
        TLLM_CHECK_WITH_INFO(params.input_last_dims[i] % kPackedSize == 0,
            "Input[%u] dimension %u is not divisible by %u (128b / sizeof(dype)). Finer granularity is not "
            "supported yet.",
            i, params.input_last_dims[i], kPackedSize);
        // Make rounded_input_dims[i] a multiple of 32 * kPackedSize
        rounded_input_dims[i]
            = (params.input_last_dims[i] + input_chunk_per_warp - 1) / input_chunk_per_warp * input_chunk_per_warp;
    }

    // Calculate total warps to launch and rounds needed
    uint32_t total_input_length = std::accumulate(rounded_input_dims, rounded_input_dims + params.num_inputs, 0);
    uint32_t total_warps_needed = total_input_length / input_chunk_per_warp;
    uint32_t num_warps_to_launch = calculateNumWarpsBase<DType, n>(params);
    uint32_t rounds = (total_warps_needed + num_warps_to_launch - 1) / num_warps_to_launch; // ceil_div

    // Calculate warp_prefix_sum
    float warps_per_token = float(num_warps_to_launch) / total_input_length;
    std::vector<int> warps_per_array(params.num_inputs);
    int warp_prefix = 0;

    for (int i = 0; i < params.num_inputs; ++i)
    {
        params.warp_prefix_sum[i] = warp_prefix;
        warps_per_array[i] = std::max(1, int(round(rounded_input_dims[i] * warps_per_token)));
        for (int j = warp_prefix; j < warp_prefix + warps_per_array[i]; ++j)
        {
            params.warp_input_idx[j] = i;
        }
        warp_prefix += warps_per_array[i];
    }
    params.warp_prefix_sum[params.num_inputs] = warp_prefix;

    dim3 grid_dim(params.batch_size);
    dim3 block_dim(32, num_warps_to_launch);

    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attribute[1];
    cfg.gridDim = grid_dim;
    cfg.blockDim = block_dim;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = params.stream;
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    cfg.attrs = attribute;
    cfg.numAttrs = 1;
    if (rounds > 1)
    {
        TLLM_CUDA_CHECK(
            cudaLaunchKernelEx(&cfg, GroupRMSNormBaseKernel<DType, float4, n, EnableWeights, true>, params, rounds));
    }
    else
    {
        TLLM_CUDA_CHECK(
            cudaLaunchKernelEx(&cfg, GroupRMSNormBaseKernel<DType, float4, n, EnableWeights, false>, params, rounds));
    }
}

template <int n>
void GroupRMSNormBaseKernelLauncher(GroupRMSParams<n>& params)
{
#define GROUP_RMS_NORM_DISPATCH(DTYPE)                                                                                 \
    if (params.enable_weights)                                                                                         \
    {                                                                                                                  \
        return GroupRMSNormBaseKernel<DTYPE, n, true>(params);                                                         \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        return GroupRMSNormBaseKernel<DTYPE, n, false>(params);                                                        \
    }

    switch (params.dtype)
    {
    case nvinfer1::DataType::kHALF: GROUP_RMS_NORM_DISPATCH(half); break;
    case nvinfer1::DataType::kBF16: GROUP_RMS_NORM_DISPATCH(__nv_bfloat16); break;
    case nvinfer1::DataType::kFLOAT: GROUP_RMS_NORM_DISPATCH(float); break;
    default: TLLM_CHECK_WITH_INFO(false, "Unsupported data type for GroupRMSNorm");
    }

#undef GROUP_RMS_NORM_DISPATCH
}

#define INSTANTIATE_GROUP_RMS_NORM_BASE(n) template void GroupRMSNormBaseKernelLauncher<n>(GroupRMSParams<n> & params);

INSTANTIATE_GROUP_RMS_NORM_BASE(1)
INSTANTIATE_GROUP_RMS_NORM_BASE(2)

template <typename DType, int n, bool EnableWeights>
void GroupRMSNormKernelLargeBatch(GroupRMSParams<n>& params)
{
    // Kernel assertions
    constexpr uint32_t kPackedSize = sizeof(float4) / sizeof(DType);
    TLLM_CHECK_WITH_INFO(params.num_inputs == 2, "GroupRMSNormKernelLargeBatch only supports exactly 2 inputs.");

    for (uint32_t i = 0; i < params.num_inputs; i++)
    {
        TLLM_CHECK_WITH_INFO(
            params.input_last_dims[i] % 32 == 0, "The last dimension of input must be divisible by 32.");
        TLLM_CHECK_WITH_INFO(params.input_last_dims[i] % kPackedSize == 0,
            "Input[%u] dimension %u is not divisible by %u (128b / sizeof(dype)). Finer granularity is not "
            "supported yet.",
            i, params.input_last_dims[i], kPackedSize);
    }

    // Calculate warps information
    auto warpInfo = calculateNumWarpsLargeBatch<DType, n>(params);
    uint32_t num_warps_to_launch = warpInfo.num_warps_to_launch;
    uint32_t rounds_0 = warpInfo.rounds_0;
    uint32_t rounds_1 = warpInfo.rounds_1;

    dim3 grid_dim(params.batch_size);
    dim3 block_dim(32, num_warps_to_launch);

    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attribute[1];
    cfg.gridDim = grid_dim;
    cfg.blockDim = block_dim;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = params.stream;
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    cfg.attrs = attribute;
    cfg.numAttrs = 1;
    bool MultiRounds_0 = rounds_0 > 1;
    bool MultiRounds_1 = rounds_1 > 1;

    // Choose kernel based on whether weights are enabled and rounds needed
    if (MultiRounds_0 && MultiRounds_1)
    {
        TLLM_CUDA_CHECK(
            cudaLaunchKernelEx(&cfg, GroupRMSNormKernelLargeBatch<DType, float4, n, EnableWeights, true, true>, params,
                rounds_0, rounds_1, warpInfo.num_warps_to_launch_0, warpInfo.num_warps_to_launch_1));
    }
    else if (MultiRounds_0 && !MultiRounds_1)
    {
        TLLM_CUDA_CHECK(
            cudaLaunchKernelEx(&cfg, GroupRMSNormKernelLargeBatch<DType, float4, n, EnableWeights, true, false>, params,
                rounds_0, rounds_1, warpInfo.num_warps_to_launch_0, warpInfo.num_warps_to_launch_1));
    }
    else if (!MultiRounds_0 && MultiRounds_1)
    {
        TLLM_CUDA_CHECK(
            cudaLaunchKernelEx(&cfg, GroupRMSNormKernelLargeBatch<DType, float4, n, EnableWeights, false, true>, params,
                rounds_0, rounds_1, warpInfo.num_warps_to_launch_0, warpInfo.num_warps_to_launch_1));
    }
    else
    {
        TLLM_CUDA_CHECK(
            cudaLaunchKernelEx(&cfg, GroupRMSNormKernelLargeBatch<DType, float4, n, EnableWeights, false, false>,
                params, rounds_0, rounds_1, warpInfo.num_warps_to_launch_0, warpInfo.num_warps_to_launch_1));
    }
}

template <int n>
void GroupRMSNormKernelLargeBatchLauncher(GroupRMSParams<n>& params)
{
#define GROUP_RMS_NORM_LARGE_BATCH_DISPATCH(DTYPE)                                                                     \
    if (params.enable_weights)                                                                                         \
    {                                                                                                                  \
        return GroupRMSNormKernelLargeBatch<DTYPE, n, true>(params);                                                   \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        return GroupRMSNormKernelLargeBatch<DTYPE, n, false>(params);                                                  \
    }

    switch (params.dtype)
    {
    case nvinfer1::DataType::kHALF: GROUP_RMS_NORM_LARGE_BATCH_DISPATCH(half); break;
    case nvinfer1::DataType::kBF16: GROUP_RMS_NORM_LARGE_BATCH_DISPATCH(__nv_bfloat16); break;
    case nvinfer1::DataType::kFLOAT: GROUP_RMS_NORM_LARGE_BATCH_DISPATCH(float); break;
    default: TLLM_CHECK_WITH_INFO(false, "Unsupported data type for GroupRMSNormV2");
    }

#undef GROUP_RMS_NORM_LARGE_BATCH_DISPATCH
}

#define INSTANTIATE_GROUP_RMS_NORM_LARGE_BATCH(n)                                                                      \
    template void GroupRMSNormKernelLargeBatchLauncher<n>(GroupRMSParams<n> & params);

INSTANTIATE_GROUP_RMS_NORM_LARGE_BATCH(2)

int getComputeCapabilityMajor()
{
    int device;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    TLLM_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop.major;
}

bool prefer_base_kernel(int batch, int base_warps, float scheduling_efficiency_ratio)
{
    int sm_major = getComputeCapabilityMajor();
    bool found_match = false;
    for (auto const& [known_model, model] : gpu_models)
    {
        if (sm_major == known_model)
        {
            float p = model.batch_size * batch + model.base_warps * base_warps
                + model.scheduling_efficiency_ratio * scheduling_efficiency_ratio + model.intercept;
            p = 1.0f / (1.0f + std::exp(-p));
            found_match = true;
            return p > 0.5f;
        }
    }
    if (!found_match)
    {
        TLLM_LOG_INFO(
            "GroupRMSNorm: Failed to find heuristic for GPU compute capability %d. Falling back to the base kernel.",
            sm_major);
    }
    return true;
}

template <int n>
void GroupRMSNormKernelLauncherWithHeuristic(GroupRMSParams<n>& params)
{
    if (params.num_inputs == 1)
    {
        GroupRMSNormBaseKernelLauncher<n>(params);
    }
    else if (params.num_inputs == 2)
    {
        int num_warps_per_sm = 64;
        uint32_t base_warps;
        uint32_t large_batch_warps;

        // Choose the appropriate DType
        switch (params.dtype)
        {
        case nvinfer1::DataType::kHALF:
            base_warps = calculateNumWarpsBase<half, n>(params);
            large_batch_warps = calculateNumWarpsLargeBatch<half, n>(params).num_warps_to_launch;
            break;
        case nvinfer1::DataType::kBF16:
            base_warps = calculateNumWarpsBase<__nv_bfloat16, n>(params);
            large_batch_warps = calculateNumWarpsLargeBatch<__nv_bfloat16, n>(params).num_warps_to_launch;
            break;
        case nvinfer1::DataType::kFLOAT:
            base_warps = calculateNumWarpsBase<float, n>(params);
            large_batch_warps = calculateNumWarpsLargeBatch<float, n>(params).num_warps_to_launch;
            break;
        default: TLLM_CHECK_WITH_INFO(false, "Unsupported data type for GroupRMSNorm"); return;
        }

        int concurrent_block_per_sm_base = std::floor(num_warps_per_sm / base_warps);
        int concurrent_block_per_sm_large_batch = std::floor(num_warps_per_sm / large_batch_warps);

        /*
         * Kernel Selection Logic:
         * We use trained Logistic Regression models to determine which kernel variant to use based on performance
         * characteristics:
         *
         * - base_warps: Proportional to the sum of last dimensions of inputs
         * - large_batch_warps: Proportional to the max of last dimensions of inputs
         *
         * Trade-offs:
         * - With equal concurrent blocks per SM, base_warps achieves better compute efficiency
         * - However, large_batch_warps allows more concurrent blocks to be scheduled:
         *   - concurrent_block_per_sm_base: Maximum blocks of base kernel schedulable per SM
         *   - concurrent_block_per_sm_large_batch: Maximum blocks of large batch kernel schedulable per SM
         *
         * The large batch kernel is preferred when the scheduling efficiency advantage outweighs
         * the compute efficiency advantage of the base kernel, particularly at larger batch sizes.
         */
        if (concurrent_block_per_sm_large_batch > concurrent_block_per_sm_base)
        {
            float scheduling_efficiency_ratio
                = float(concurrent_block_per_sm_large_batch) / float(concurrent_block_per_sm_base);
            if (prefer_base_kernel(params.batch_size, base_warps, scheduling_efficiency_ratio))
            {
                GroupRMSNormBaseKernelLauncher<n>(params);
            }
            else
            {
                GroupRMSNormKernelLargeBatchLauncher<n>(params);
            }
        }
        else
        {
            GroupRMSNormBaseKernelLauncher<n>(params);
        }
    }
    else
    {
        // Unsupported number of inputs
        TLLM_CHECK_WITH_INFO(false, "Unsupported number of inputs for GroupRMSNorm");
    }
}

#define INSTANTIATE_GROUP_RMS_NORM_WITH_HEURISTIC(n)                                                                   \
    template void GroupRMSNormKernelLauncherWithHeuristic<n>(GroupRMSParams<n> & params);

INSTANTIATE_GROUP_RMS_NORM_WITH_HEURISTIC(1)
INSTANTIATE_GROUP_RMS_NORM_WITH_HEURISTIC(2)

} // namespace tensorrt_llm::kernels::group_rms_norm
