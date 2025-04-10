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
#include <algorithm> // For std::gcd, std::max, std::min
#include <cmath>     // For round
#include <cstdint>   // For uint32_t
#include <numeric>   // For std::gcd

#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"

using tensorrt_llm::common::deviceMalloc;
using tensorrt_llm::common::cudaAutoCpy;
using tensorrt_llm::common::deviceFree;

// DSV3:
// kv_lora_rank (int, optional, defaults to 512) — Rank of the LoRA matrices for key and value projections.
// q_lora_rank (int, optional, defaults to 1536) — Rank of the LoRA matrices for query projections.

// The hidden size seems too small to take advantage of the cluster group based reduction.
// Below code is inspired by flashinfer's implementation.

// LLamaV4 uses L2Norm which calls RMS norm underneath.
// Hidden size for q = 5120
// Hidden size for kv = 8 (config.num_key_value_heads) * 128 (config.head_dim) = 1024

// Base on the hidden_size in interest, a cluster_group based implementation might be more performant.

// Target usage:
// q, kv, k_pe = self.fused_a(
//                 hidden_states).split([
//                     self.dim_q + self.dim_kv, self.qk_rope_head_dim
//                 ], -1)
// q, kv = groupNmsrNorm([q, kv])

// TODOs:
// - Check if it works for 1 input case
// - Check if it works for 2 input case

namespace tensorrt_llm::kernels::group_rms_norm
{

uint32_t ceil_div(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

// binary search over warp_prefix_sum to find which array this warp belongs to
__device__ __forceinline__ uint32_t find_input_idx_for_warp(
    uint32_t const* __restrict__ warp_prefix_sum, const uint32_t warp_idx, const uint32_t num_inputs)
{
    uint32_t lo = 0, hi = num_inputs;
    while (lo < hi)
    {
        uint32_t mid = (lo + hi) / 2;
        if (warp_prefix_sum[mid + 1] <= warp_idx)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// This kernel only supports 2 inputs for now
// Will add support for more inputs in the future

// sizeof(PackedType) = 128b
template <typename DType, typename PackedType>
__global__ void GroupRMSNormKernelWithoutWeights(PackedType** __restrict__ inputs, PackedType** __restrict__ outputs,
    uint32_t const* __restrict__ input_dims, uint32_t const* __restrict__ input_strides,
    uint32_t const* __restrict__ output_strides,
    // Starting warp idx for each input
    uint32_t const* __restrict__ warp_prefix_sum,
    // Base on the LLama and DSV3 implementation, 1 round is enough for Half[8192].
    // Maybe we should have a kernel without round loop to further reduce runtime.
    const uint32_t __restrict__ rounds, const uint32_t __restrict__ num_inputs, float const eps)
{
    const uint32_t bx = blockIdx.x; // Maps to batch size
    constexpr uint32_t warp_size = 32;
    const uint32_t num_warps = blockDim.y;
    const uint32_t warp_idx = threadIdx.y;
    const uint32_t lane_idx = threadIdx.x;
    const uint32_t thread_idx = warp_idx * warp_size + lane_idx;

    static constexpr int kPackedSize = sizeof(PackedType) / sizeof(DType);

    // Each thread calculates its own partial sum
    float sum_sq_q = 0.f;
    float sum_sq_kv = 0.f;

    __shared__ float smem_rsqrts[32];
    float sum_sqs[32];
#pragma unroll
    for (int i = 0; i < 32; ++i)
    {
        sum_sqs[i] = 0.0f;
    }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    // Find which input this warp operates on
    const uint32_t input_idx = find_input_idx_for_warp(warp_prefix_sum, warp_idx, num_inputs);
    const uint32_t warp_start = warp_prefix_sum[input_idx];
    const uint32_t local_warp_idx = warp_idx - warp_start;
    PackedType const* input_ptr = inputs[input_idx];
    PackedType* output_ptr = outputs[input_idx];

    // Offset for the next batch
    uint32_t block_offset = bx * input_strides[input_idx];
    // Offset for the next warp, is same for all inputs
    uint32_t warp_offset = input_dims[0] / warp_prefix_sum[1];
    uint32_t round_offset = 32 * kPackedSize;

    // TODO: assert input_dims are multiple of 32 to avoid boundary check
    // Calculate sum of squares for each thread
    for (uint32_t i = 0; i < rounds; i++)
    {
        uint32_t idx = block_offset + local_warp_idx * warp_offset + i * round_offset + lane_idx;
#pragma unroll
        for (uint32_t j = 0; j < kPackedSize; j++)
        {
            if (idx + j < input_dims[input_idx])
            {
                float v = static_cast<float>(reinterpret_cast<DType const*>(&input_ptr)[idx + j]);
                sum_sqs[input_idx] += v * v;
            }
        }
    }

    // Reduce sum of squares across all threads in the block
    for (uint32_t i = 0; i < num_inputs; i++)
    {
        sum_sqs[i] = tensorrt_llm::common::blockReduceSum(sum_sqs[i]);
    }

    // Calculate normalization factor for each input
    // Use thread_idx as input_idx here.
    if (thread_idx < num_inputs)
    {
        smem_rsqrts[thread_idx] = rsqrtf(sum_sqs[thread_idx] / input_dims[thread_idx] + eps);
    }

    // Second synchronization point to ensure normalization factors are available
    __syncthreads();

    // Apply normalization for both Q and KV in parallel
    for (uint32_t i = 0; i < rounds; i++)
    {
        uint32_t idx = block_offset + local_warp_idx * warp_offset + i * round_offset + lane_idx;
// Load input in unit of 128b and convert to an array of Dtype
// Should have cache hit.
#pragma unroll
        for (uint32_t j = 0; j < kPackedSize; j++)
        {
            if (idx + j < input_dims[input_idx])
            {
                reinterpret_cast<DType*>(&output_ptr)[idx + j] = static_cast<DType>(
                    static_cast<float>(reinterpret_cast<DType const*>(&input_ptr)[idx + j]) * smem_rsqrts[input_idx]);
            }
        }
    }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename DType>
void GroupRMSNormKernel(DType** inputs, DType** weights, DType** outputs, uint32_t const* input_dims,
    uint32_t const* input_strides, uint32_t const* output_strides, const uint32_t batch_size, const uint32_t num_inputs,
    float const eps, bool enable_weights, cudaStream_t stream)
{
    // Kernel assertions
    constexpr uint32_t kPackedSize = sizeof(float4) / sizeof(DType);
    TLLM_CHECK_WITH_INFO(!enable_weights, "Weights are not supported yet. Will add in the future.");
    TLLM_CHECK_WITH_INFO(weights[0] == nullptr, "Weights are not supported yet. Will add in the future.");
    TLLM_CHECK_WITH_INFO(
        num_inputs <= 32, "Only up to 32 inputs are supported for now. Will add support for more in the future.");
    for (uint32_t i = 0; i < num_inputs; i++)
    {
        TLLM_CHECK_WITH_INFO(input_dims[i] % 32 == 0,
            "Input dimension must be divisible by 32. Padding support will be added in the future.");
    }

    // Calculate total warps to launch and rounds needed
    uint32_t total_input_length = std::accumulate(input_dims, input_dims + num_inputs, 0);
    uint32_t input_chunk_per_warp = 32 * kPackedSize;
    uint32_t total_warps_needed = ceil_div(total_input_length, input_chunk_per_warp);
    uint32_t num_warps_to_launch = std::min<uint32_t>(32, total_warps_needed);
    uint32_t rounds = ceil_div(total_warps_needed, num_warps_to_launch);

    // Calculate warp_prefix_sum
    float warps_per_token = float(num_warps_to_launch) / total_input_length;
    std::vector<int> warps_per_array(num_inputs);
    std::vector<uint32_t> warp_prefix_sum(num_inputs + 1);
    warp_prefix_sum[0] = 0;

    for (int i = 0; i < num_inputs; ++i)
    {
        warps_per_array[i] = std::max(1, int(round(input_dims[i] * warps_per_token)));
        warp_prefix_sum[i + 1] = warp_prefix_sum[i] + warps_per_array[i];
    }

    dim3 grid_dim(batch_size);
    dim3 block_dim(32, num_warps_to_launch);

    // Prepare inputs and outputs for the kernel
    float4** d_inputs;
    float4** d_outputs;
    uint32_t* d_input_dims;
    uint32_t* d_input_strides;
    uint32_t* d_output_strides;
    uint32_t* d_warp_prefix_sum;

    // Device memory allocation
    deviceMalloc(&d_inputs, num_inputs * sizeof(float4*));
    deviceMalloc(&d_outputs, num_inputs * sizeof(float4*));
    deviceMalloc(&d_input_dims, num_inputs * sizeof(uint32_t));
    deviceMalloc(&d_input_strides, num_inputs * sizeof(uint32_t));
    deviceMalloc(&d_output_strides, num_inputs * sizeof(uint32_t));
    deviceMalloc(&d_warp_prefix_sum, (num_inputs + 1) * sizeof(uint32_t));

    // Prepare host arrays
    float4* h_inputs[num_inputs];
    float4* h_outputs[num_inputs];
    for (uint32_t i = 0; i < num_inputs; i++)
    {
        h_inputs[i] = reinterpret_cast<float4*>(inputs[i]);
        h_outputs[i] = reinterpret_cast<float4*>(outputs[i]);
    }

    // Copy data to device
    cudaAutoCpy(d_inputs, h_inputs, num_inputs * sizeof(float4*), stream);
    cudaAutoCpy(d_outputs, h_outputs, num_inputs * sizeof(float4*), stream);
    cudaAutoCpy(d_input_dims, input_dims, num_inputs * sizeof(uint32_t), stream);
    cudaAutoCpy(d_input_strides, input_strides, num_inputs * sizeof(uint32_t), stream);
    cudaAutoCpy(d_output_strides, output_strides, num_inputs * sizeof(uint32_t), stream);
    cudaAutoCpy(d_warp_prefix_sum, warp_prefix_sum.data(), (num_inputs + 1) * sizeof(uint32_t), stream);

    // Shared memory size: used for the rsqrt values if each input, max 32 inputs.
    const uint32_t smem_size = 32 * sizeof(float);
    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attribute[1];
    cfg.gridDim = grid_dim;
    cfg.blockDim = block_dim;
    cfg.dynamicSmemBytes = smem_size;
    cfg.stream = stream;
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    cfg.attrs = attribute;
    cfg.numAttrs = 1;

    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, GroupRMSNormKernelWithoutWeights<DType, float4>, d_inputs, d_outputs,
        d_input_dims, d_input_strides, d_output_strides, d_warp_prefix_sum, rounds, num_inputs, eps));

    // Free device memory
    deviceFree(d_inputs);
    deviceFree(d_outputs);
    deviceFree(d_input_dims);
    deviceFree(d_input_strides);
    deviceFree(d_output_strides);
    deviceFree(d_warp_prefix_sum);
}
} // namespace tensorrt_llm::kernels::group_rms_norm
