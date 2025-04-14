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
#include "tensorrt_llm/kernels/groupRmsNormKernels.h"

using tensorrt_llm::common::deviceMalloc;
using tensorrt_llm::common::cudaAutoCpy;
using tensorrt_llm::common::deviceFree;

namespace tensorrt_llm::kernels::group_rms_norm
{

// binary search over warp_prefix_sum to find which array this warp belongs to
__device__ __forceinline__ uint32_t find_input_idx_for_warp(
    uint32_t const* __restrict__ warp_prefix_sum, const uint32_t warp_idx, const uint32_t num_inputs)
{
    uint32_t lo = 0, hi = num_inputs;
    while (lo < hi)
    {
        uint32_t mid = (lo + hi) / 2;
        if (warp_prefix_sum[mid + 1] <= warp_idx)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

template <typename DType, typename PackedType>
__global__ void GroupRMSNormKernelWithWeights(PackedType** __restrict__ inputs, PackedType** __restrict__ outputs,
    PackedType** __restrict__ weights, uint32_t const* __restrict__ input_dims,
    uint32_t const* __restrict__ input_strides, uint32_t const* __restrict__ output_strides,
    // Starting warp idx for each input
    uint32_t const* __restrict__ warp_prefix_sum, const uint32_t __restrict__ rounds,
    const uint32_t __restrict__ num_inputs, float const eps, float const weight_bias)
{
    const uint32_t bx = blockIdx.x; // Maps to batch size
    constexpr uint32_t warp_size = 32;
    const uint32_t warp_idx = threadIdx.y;
    const uint32_t lane_idx = threadIdx.x;

    static constexpr int kPackedSize = sizeof(PackedType) / sizeof(DType);
    float warp_acc = 0.0f;

    // Helper variables for block reduction of all inputs.
    // smem_input_mask[warp_idx] is the input index which the warp is processing.
    __shared__ uint32_t smem_input_mask[32];
    // smem_warp_sum_sqs[warp_idx] is the warp-level sum of squares of the corresponding input to the warp.
    __shared__ float smem_warp_sum_sqs[32];

    // smem_rsqrts[input_idx] is the rsqrt of the sum of squares of the input.
    __shared__ float smem_rsqrts[32];
#pragma unroll
    for (int i = 0; i < 32; ++i)
    {
        smem_input_mask[i] = 0;
        smem_warp_sum_sqs[i] = 0.0f;
        smem_rsqrts[i] = 0.0f;
    }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    // Phase 1: Calculate the warp-level sum of squares of the corresponding input to the warp.
    // Find which input this warp operates on
    const uint32_t input_idx = find_input_idx_for_warp(warp_prefix_sum, warp_idx, num_inputs);
    const uint32_t warp_start = warp_prefix_sum[input_idx];
    const uint32_t local_warp_idx = warp_idx - warp_start;
    PackedType const* input_ptr = inputs[input_idx];
    PackedType* output_ptr = outputs[input_idx];

    // Offset for the next batch
    uint32_t block_offset = bx * input_strides[input_idx];
    // Offset for the next round
    uint32_t round_offset = warp_size * kPackedSize;
    // Offset for the next warp
    uint32_t warp_offset = round_offset * rounds;

    // Calculate sum of squares for each thread
    for (uint32_t i = 0; i < rounds; i++)
    {
        uint32_t idx = block_offset + local_warp_idx * warp_offset + i * round_offset + lane_idx * kPackedSize;
        if (idx < input_dims[input_idx] + block_offset)
        {
            // Vectorized load of the input data, 128b at once
            PackedType packed_data = input_ptr[idx / kPackedSize];
            DType* packed_input_ptr = reinterpret_cast<DType*>(&packed_data);
#pragma unroll
            // No boundary check here as we assert the input dims are divisible by 32 * kPackedSize
            for (uint32_t j = 0; j < kPackedSize; j++)
            {
                float v = static_cast<float>(packed_input_ptr[j]);
                warp_acc += v * v;
            }
        }
    }
    // For each warp, store the input index and the warp-level sum of squares for corresponding input to
    smem_input_mask[warp_idx] = input_idx;
    smem_warp_sum_sqs[warp_idx] = tensorrt_llm::common::warpReduceSum(warp_acc);

    __syncthreads();

    // Phase 2: Cross wrap reduction on all inputs and calculate the normalization factor for each input
    warp_acc = 0.0f;
    if (warp_idx < num_inputs)
    {
        // Each warp sums one input
        if (warp_idx == smem_input_mask[lane_idx])
        {
            warp_acc = smem_warp_sum_sqs[lane_idx];
        }
        smem_rsqrts[warp_idx] = tensorrt_llm::common::warpReduceSum(warp_acc);
        smem_rsqrts[warp_idx] = rsqrtf(smem_rsqrts[warp_idx] / input_dims[warp_idx] + eps);
    }

    __syncthreads();

    // Phase 3: Apply normalization for inputs
    for (uint32_t i = 0; i < rounds; i++)
    {
        uint32_t idx = block_offset + local_warp_idx * warp_offset + i * round_offset + lane_idx * kPackedSize;
        if (idx < input_dims[input_idx] + block_offset)
        {
            // Vectorized load of the input data, 128b at once
            PackedType packed_input = input_ptr[idx / kPackedSize];
            DType* packed_input_ptr = reinterpret_cast<DType*>(&packed_input);
            PackedType packed_weight = weights[input_idx][(idx - block_offset) / kPackedSize];
            DType* packed_weight_ptr = reinterpret_cast<DType*>(&packed_weight);

            PackedType packed_output;
            DType* packed_output_ptr = reinterpret_cast<DType*>(&packed_output);
#pragma unroll
            for (uint32_t j = 0; j < kPackedSize; j++)
            {
                packed_output_ptr[j] = static_cast<DType>(static_cast<float>(packed_input_ptr[j])
                    * smem_rsqrts[input_idx] * (static_cast<float>(packed_weight_ptr[j]) + weight_bias));
            }
            // Vectorized store of the output data, 128b at once
            output_ptr[idx / kPackedSize] = packed_output;
        }
    }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename DType, typename PackedType>
__global__ void GroupRMSNormKernelWithoutWeights(PackedType** __restrict__ inputs, PackedType** __restrict__ outputs,
    uint32_t const* __restrict__ input_dims, uint32_t const* __restrict__ input_strides,
    uint32_t const* __restrict__ output_strides,
    // Starting warp idx for each input
    uint32_t const* __restrict__ warp_prefix_sum, const uint32_t __restrict__ rounds,
    const uint32_t __restrict__ num_inputs, float const eps)
{
    const uint32_t bx = blockIdx.x; // Maps to batch size
    constexpr uint32_t warp_size = 32;
    const uint32_t warp_idx = threadIdx.y;
    const uint32_t lane_idx = threadIdx.x;

    static constexpr int kPackedSize = sizeof(PackedType) / sizeof(DType);
    float warp_acc = 0.0f;

    // Helper variables for block reduction of all inputs.
    // smem_input_mask[warp_idx] is the input index which the warp is processing.
    __shared__ uint32_t smem_input_mask[32];
    // smem_warp_sum_sqs[warp_idx] is the warp-level sum of squares of the corresponding input to the warp.
    __shared__ float smem_warp_sum_sqs[32];

    // smem_rsqrts[input_idx] is the rsqrt of the sum of squares of the input.
    __shared__ float smem_rsqrts[32];
#pragma unroll
    for (int i = 0; i < 32; ++i)
    {
        smem_input_mask[i] = 0;
        smem_warp_sum_sqs[i] = 0.0f;
        smem_rsqrts[i] = 0.0f;
    }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    // Phase 1: Calculate the warp-level sum of squares of the corresponding input to the warp.
    // Find which input this warp operates on
    const uint32_t input_idx = find_input_idx_for_warp(warp_prefix_sum, warp_idx, num_inputs);
    const uint32_t warp_start = warp_prefix_sum[input_idx];
    const uint32_t local_warp_idx = warp_idx - warp_start;
    PackedType const* input_ptr = inputs[input_idx];
    PackedType* output_ptr = outputs[input_idx];

    // Offset for the next batch
    uint32_t block_offset = bx * input_strides[input_idx];
    // Offset for the next round
    uint32_t round_offset = warp_size * kPackedSize;
    // Offset for the next warp
    uint32_t warp_offset = round_offset * rounds;

    // Calculate sum of squares for each thread
    for (uint32_t i = 0; i < rounds; i++)
    {
        uint32_t idx = block_offset + local_warp_idx * warp_offset + i * round_offset + lane_idx * kPackedSize;
        if (idx < input_dims[input_idx] + block_offset)
        {
            // Vectorized load of the input data, 128b at once
            PackedType packed_data = input_ptr[idx / kPackedSize];
            DType* packed_input_ptr = reinterpret_cast<DType*>(&packed_data);
#pragma unroll
            // No boundary check here as we assert the input dims are divisible by 32 * kPackedSize
            for (uint32_t j = 0; j < kPackedSize; j++)
            {
                float v = static_cast<float>(packed_input_ptr[j]);
                warp_acc += v * v;
            }
        }
    }
    // For each warp, store the input index and the warp-level sum of squares for corresponding input to
    smem_input_mask[warp_idx] = input_idx;
    smem_warp_sum_sqs[warp_idx] = tensorrt_llm::common::warpReduceSum(warp_acc);

    __syncthreads();

    // Phase 2: Cross wrap reduction on all inputs and calculate the normalization factor for each input
    warp_acc = 0.0f;
    if (warp_idx < num_inputs)
    {
        // Each warp sums one input
        if (warp_idx == smem_input_mask[lane_idx])
        {
            warp_acc = smem_warp_sum_sqs[lane_idx];
        }
        smem_rsqrts[warp_idx] = tensorrt_llm::common::warpReduceSum(warp_acc);
        smem_rsqrts[warp_idx] = rsqrtf(smem_rsqrts[warp_idx] / input_dims[warp_idx] + eps);
    }

    __syncthreads();

    // Phase 3: Apply normalization for inputs
    for (uint32_t i = 0; i < rounds; i++)
    {
        uint32_t idx = block_offset + local_warp_idx * warp_offset + i * round_offset + lane_idx * kPackedSize;
        if (idx < input_dims[input_idx] + block_offset)
        {
            // Vectorized load of the input data, 128b at once
            PackedType packed_input = input_ptr[idx / kPackedSize];
            DType* packed_input_ptr = reinterpret_cast<DType*>(&packed_input);

            PackedType packed_output;
            DType* packed_output_ptr = reinterpret_cast<DType*>(&packed_output);
#pragma unroll
            for (uint32_t j = 0; j < kPackedSize; j++)
            {
                packed_output_ptr[j]
                    = static_cast<DType>(static_cast<float>(packed_input_ptr[j]) * smem_rsqrts[input_idx]);
            }
            // Vectorized store of the output data, 128b at once
            output_ptr[idx / kPackedSize] = packed_output;
        }
    }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename DType>
void GroupRMSNormKernel(DType** inputs, DType** weights, DType** outputs, uint32_t const* input_dims,
    uint32_t const* input_strides, uint32_t const* output_strides, const uint32_t batch_size, const uint32_t num_inputs,
    float const eps, float const weight_bias, bool enable_weights, cudaStream_t stream)
{
    // Kernel assertions
    constexpr uint32_t kPackedSize = sizeof(float4) / sizeof(DType);
    TLLM_CHECK_WITH_INFO(num_inputs <= 32, "Only up to 32 inputs are supported.");
    for (uint32_t i = 0; i < num_inputs; i++)
    {
        TLLM_CHECK_WITH_INFO(input_dims[i] % 32 == 0, "Input dimension must be divisible by 32.");
        // Each wrap process 32 * kPackedSize tokens per round.
        // Need to be divisible by 32*kPackedSize
        TLLM_CHECK_WITH_INFO(input_dims[i] % (32 * kPackedSize) == 0,
            "Input[%u] dimension %u is not divisible by %u (32 * (128b / sizeof(dype))). Finer granularity is not "
            "supported yet.",
            i, input_dims[i], 32 * kPackedSize);
    }

    // Allocate memory for inputs, weights, and outputs and cast them to float4*
    std::vector<float4*> inputs_float4(num_inputs);
    std::vector<float4*> outputs_float4(num_inputs);
    std::vector<float4*> weights_float4(num_inputs);
    for (uint32_t i = 0; i < num_inputs; ++i)
    {
        inputs_float4[i] = reinterpret_cast<float4*>(inputs[i]); // Safe if aligned
        outputs_float4[i] = reinterpret_cast<float4*>(outputs[i]);
        if (enable_weights)
        {
            weights_float4[i] = reinterpret_cast<float4*>(weights[i]);
        }
    }
    float4** d_inputs;
    float4** d_outputs;
    float4** d_weights;

    cudaMalloc(&d_inputs, num_inputs * sizeof(float4*));
    cudaMemcpy(d_inputs, inputs_float4.data(), num_inputs * sizeof(float4*), cudaMemcpyHostToDevice);

    cudaMalloc(&d_outputs, num_inputs * sizeof(float4*));
    cudaMemcpy(d_outputs, outputs_float4.data(), num_inputs * sizeof(float4*), cudaMemcpyHostToDevice);

    if (enable_weights)
    {
        cudaMalloc(&d_weights, num_inputs * sizeof(float4*));
        cudaMemcpy(d_weights, weights_float4.data(), num_inputs * sizeof(float4*), cudaMemcpyHostToDevice);
    }

    // Calculate total warps to launch and rounds needed
    uint32_t total_input_length = std::accumulate(input_dims, input_dims + num_inputs, 0);
    uint32_t input_chunk_per_warp = 32 * kPackedSize;
    uint32_t total_warps_needed = (total_input_length + input_chunk_per_warp - 1) / input_chunk_per_warp; // ceil_div
    uint32_t num_warps_to_launch = std::min<uint32_t>(32, total_warps_needed);
    uint32_t rounds = (total_warps_needed + num_warps_to_launch - 1) / num_warps_to_launch;

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

    // Prepare inputs parameters for the kernel
    uint32_t* d_input_dims;
    uint32_t* d_input_strides;
    uint32_t* d_output_strides;
    uint32_t* d_warp_prefix_sum;

    // Device memory allocation
    deviceMalloc(&d_input_dims, num_inputs * sizeof(uint32_t));
    deviceMalloc(&d_input_strides, num_inputs * sizeof(uint32_t));
    deviceMalloc(&d_output_strides, num_inputs * sizeof(uint32_t));
    deviceMalloc(&d_warp_prefix_sum, (num_inputs + 1) * sizeof(uint32_t));

    cudaAutoCpy(d_input_dims, input_dims, num_inputs, stream);
    cudaAutoCpy(d_input_strides, input_strides, num_inputs, stream);
    cudaAutoCpy(d_output_strides, output_strides, num_inputs, stream);
    cudaAutoCpy(d_warp_prefix_sum, warp_prefix_sum.data(), num_inputs + 1, stream);

    TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Shared memory size: used for the rsqrt values if each input, max 32 inputs.
    const uint32_t smem_size = 3 * 32 * sizeof(float);
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

    if (enable_weights)
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, GroupRMSNormKernelWithWeights<DType, float4>, d_inputs, d_outputs,
            d_weights, d_input_dims, d_input_strides, d_output_strides, d_warp_prefix_sum, rounds, num_inputs, eps,
            weight_bias));
    }
    else
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, GroupRMSNormKernelWithoutWeights<DType, float4>, d_inputs, d_outputs,
            d_input_dims, d_input_strides, d_output_strides, d_warp_prefix_sum, rounds, num_inputs, eps));
    }

    TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Cleanup device memory
    TLLM_CUDA_CHECK(cudaFree(d_inputs));
    TLLM_CUDA_CHECK(cudaFree(d_outputs));
    if (enable_weights)
    {
        TLLM_CUDA_CHECK(cudaFree(d_weights));
    }
    deviceFree(d_input_dims);
    deviceFree(d_input_strides);
    deviceFree(d_output_strides);
    deviceFree(d_warp_prefix_sum);
}

template void tensorrt_llm::kernels::group_rms_norm::GroupRMSNormKernel<half>(half**, half**, half**, uint32_t const*,
    uint32_t const*, uint32_t const*, uint32_t, uint32_t, float, float, bool, cudaStream_t);

#ifdef ENABLE_BF16
template void tensorrt_llm::kernels::group_rms_norm::GroupRMSNormKernel<__nv_bfloat16>(__nv_bfloat16**, __nv_bfloat16**,
    __nv_bfloat16**, uint32_t const*, uint32_t const*, uint32_t const*, uint32_t, uint32_t, float, float, bool,
    cudaStream_t);
#endif

template void tensorrt_llm::kernels::group_rms_norm::GroupRMSNormKernel<float>(float**, float**, float**,
    uint32_t const*, uint32_t const*, uint32_t const*, uint32_t, uint32_t, float, float, bool, cudaStream_t);
} // namespace tensorrt_llm::kernels::group_rms_norm
