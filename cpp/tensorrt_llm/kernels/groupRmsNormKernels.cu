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
#include <nvtx3/nvToolsExt.h>

#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/groupRmsNormKernels.h"

using tensorrt_llm::common::deviceMalloc;
using tensorrt_llm::common::cudaAutoCpy;
using tensorrt_llm::common::deviceFree;

namespace tensorrt_llm::kernels::group_rms_norm
{

template <typename DType, typename PackedType, int n>
__global__ void GroupRMSNormKernelWithWeights(GroupRMSParams<n> params, int rounds)
{
    const uint32_t batch_idx = blockIdx.x; // Maps to batch size
    const uint32_t input_idx = blockIdx.y; // Maps to input index
    constexpr uint32_t warp_size = 32;
    const uint32_t warp_idx = threadIdx.y;
    const uint32_t lane_idx = threadIdx.x;

    static constexpr int kPackedSize = sizeof(PackedType) / sizeof(DType);
    float acc = 0.0f;

    // smem_rsqrts[input_idx] is the rsqrt of the sum of squares of the input.
    __shared__ float smem_rsqrts[32];
    if (warp_idx == 0)
    {
        smem_rsqrts[lane_idx] = 0.0f;
    }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    // Phase 1: Calculate the warp-level sum of squares of the corresponding input to the warp.
    // Find which input this warp operates on
    PackedType const* input_ptr = params.inputs[input_idx];
    PackedType const* weight_ptr = params.weights[input_idx];
    PackedType* output_ptr = params.outputs[input_idx];

    // Offset for the next batch
    uint32_t block_offset = batch_idx * params.input_strides[input_idx];
    // Offset for the next round
    uint32_t round_offset = warp_size * kPackedSize;
    // Offset for the next warp
    uint32_t warp_offset = round_offset * rounds;

    // Calculate sum of squares for each thread
    for (uint32_t i = 0; i < rounds; i++)
    {
        uint32_t idx = block_offset + warp_idx * warp_offset + i * round_offset + lane_idx * kPackedSize;
        if (idx < params.input_dims[input_idx] + block_offset)
        {
            // Vectorized load of the input data, 128b at once
            PackedType packed_data = input_ptr[idx / kPackedSize];
            DType* packed_input_ptr = reinterpret_cast<DType*>(&packed_data);
#pragma unroll
            // No boundary check here as we assert the input dims are divisible by 32 * kPackedSize
            for (uint32_t j = 0; j < kPackedSize; j++)
            {
                float v = static_cast<float>(packed_input_ptr[j]);
                acc += v * v;
            }
        }
    }
    // For each warp, store the input index and the warp-level sum of squares for corresponding input to
    smem_rsqrts[warp_idx] = tensorrt_llm::common::warpReduceSum(acc);
    __syncthreads();
    if (warp_idx == 0)
    {
        smem_rsqrts[0] = tensorrt_llm::common::warpReduceSum(smem_rsqrts[lane_idx]);
    }
    __syncthreads();

    float block_rsqrt = rsqrtf(smem_rsqrts[0] / params.input_dims[input_idx] + params.eps);

    // Phase 3: Apply normalization for inputs
    for (uint32_t i = 0; i < rounds; i++)
    {
        uint32_t idx = block_offset + warp_idx * warp_offset + i * round_offset + lane_idx * kPackedSize;
        if (idx < params.input_dims[input_idx] + block_offset)
        {
            // Vectorized load of the input data, 128b at once
            PackedType packed_input = input_ptr[idx / kPackedSize];
            DType* packed_input_ptr = reinterpret_cast<DType*>(&packed_input);

            PackedType packed_weight = weight_ptr[(idx - block_offset) / kPackedSize];
            DType* packed_weight_ptr = reinterpret_cast<DType*>(&packed_weight);

            PackedType packed_output;
            DType* packed_output_ptr = reinterpret_cast<DType*>(&packed_output);
#pragma unroll
            for (uint32_t j = 0; j < kPackedSize; j++)
            {
                packed_output_ptr[j] = static_cast<DType>(static_cast<float>(packed_input_ptr[j]) * block_rsqrt
                    * (static_cast<float>(packed_weight_ptr[j]) + params.weight_bias));
            }
            // Vectorized store of the output data, 128b at once
            output_ptr[idx / kPackedSize] = packed_output;
        }
    }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename DType, typename PackedType, int n>
__global__ void GroupRMSNormKernelWithoutWeights(GroupRMSParams<n> params, int rounds)
{
    const uint32_t batch_idx = blockIdx.x; // Maps to batch size
    const uint32_t input_idx = blockIdx.y; // Maps to input index
    constexpr uint32_t warp_size = 32;
    const uint32_t warp_idx = threadIdx.y;
    const uint32_t lane_idx = threadIdx.x;

    static constexpr int kPackedSize = sizeof(PackedType) / sizeof(DType);
    float acc = 0.0f;

    // smem_rsqrts[input_idx] is the rsqrt of the sum of squares of the input.
    __shared__ float smem_rsqrts[32];
    if (warp_idx == 0)
    {
        smem_rsqrts[lane_idx] = 0.0f;
    }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    // Phase 1: Calculate the warp-level sum of squares of the corresponding input to the warp.
    // Find which input this warp operates on
    PackedType const* input_ptr = params.inputs[input_idx];
    PackedType* output_ptr = params.outputs[input_idx];

    // Offset for the next batch
    uint32_t block_offset = batch_idx * params.input_strides[input_idx];
    // Offset for the next round
    uint32_t round_offset = warp_size * kPackedSize;
    // Offset for the next warp
    uint32_t warp_offset = round_offset * rounds;

    // Calculate sum of squares for each thread
    for (uint32_t i = 0; i < rounds; i++)
    {
        uint32_t idx = block_offset + warp_idx * warp_offset + i * round_offset + lane_idx * kPackedSize;
        if (idx < params.input_dims[input_idx] + block_offset)
        {
            // Vectorized load of the input data, 128b at once
            PackedType packed_data = input_ptr[idx / kPackedSize];
            DType* packed_input_ptr = reinterpret_cast<DType*>(&packed_data);
#pragma unroll
            // No boundary check here as we assert the input dims are divisible by 32 * kPackedSize
            for (uint32_t j = 0; j < kPackedSize; j++)
            {
                float v = static_cast<float>(packed_input_ptr[j]);
                acc += v * v;
            }
        }
    }
    // For each warp, store the input index and the warp-level sum of squares for corresponding input to
    smem_rsqrts[warp_idx] = tensorrt_llm::common::warpReduceSum(acc);
    __syncthreads();
    if (warp_idx == 0)
    {
        smem_rsqrts[0] = tensorrt_llm::common::warpReduceSum(smem_rsqrts[lane_idx]);
    }
    __syncthreads();
    float block_rsqrt = rsqrtf(smem_rsqrts[0] / params.input_dims[input_idx] + params.eps);

    // Phase 3: Apply normalization for inputs
    for (uint32_t i = 0; i < rounds; i++)
    {
        uint32_t idx = block_offset + warp_idx * warp_offset + i * round_offset + lane_idx * kPackedSize;
        if (idx < params.input_dims[input_idx] + block_offset)
        {
            // Vectorized load of the input data, 128b at once
            PackedType packed_input = input_ptr[idx / kPackedSize];
            DType* packed_input_ptr = reinterpret_cast<DType*>(&packed_input);

            PackedType packed_output;
            DType* packed_output_ptr = reinterpret_cast<DType*>(&packed_output);
#pragma unroll
            for (uint32_t j = 0; j < kPackedSize; j++)
            {
                packed_output_ptr[j] = static_cast<DType>(static_cast<float>(packed_input_ptr[j]) * block_rsqrt);
            }
            // Vectorized store of the output data, 128b at once
            output_ptr[idx / kPackedSize] = packed_output;
        }
    }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename DType, int n>
void GroupRMSNormKernel(GroupRMSParams<n> params)
{
    // Kernel assertions
    constexpr uint32_t kPackedSize = sizeof(float4) / sizeof(DType);
    for (int i = 0; i < params.num_inputs; i++)
    {
        TLLM_CHECK_WITH_INFO(params.input_dims[i] % 32 == 0, "Input dimension must be divisible by 32.");
        // Each wrap process 32 * kPackedSize tokens per round.
        // Need to be divisible by 32*kPackedSize
        TLLM_CHECK_WITH_INFO(params.input_dims[i] % kPackedSize == 0,
            "Input[%u] dimension %u is not divisible by %u (128b / sizeof(dype)). Finer granularity is not "
            "supported yet.",
            i, params.input_dims[i], kPackedSize);
    }

    // Calculate total warps to launch and rounds needed
    int max_input_length = *std::max_element(params.input_dims, params.input_dims + params.num_inputs);
    uint32_t input_chunk_per_warp = 32 * kPackedSize;
    uint32_t max_warps_needed = (max_input_length + input_chunk_per_warp - 1) / input_chunk_per_warp; // ceil_div
    uint32_t num_warps_to_launch = std::min<uint32_t>(32, max_warps_needed);
    uint32_t rounds = (max_warps_needed + num_warps_to_launch - 1) / num_warps_to_launch;

    dim3 grid_dim(params.batch_size, params.num_inputs);
    dim3 block_dim(32, num_warps_to_launch);

    // Shared memory size: used for the rsqrt values if each input, max 32 inputs.
    const uint32_t smem_size = 32 * sizeof(float);
    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attribute[1];
    cfg.gridDim = grid_dim;
    cfg.blockDim = block_dim;
    cfg.dynamicSmemBytes = smem_size;
    cfg.stream = params.stream;
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    cfg.attrs = attribute;
    cfg.numAttrs = 1;

    if (params.enable_weights)
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(
            &cfg, (void (*)(GroupRMSParams<n>, int)) GroupRMSNormKernelWithWeights<DType, float4, n>, params, rounds));
    }
    else
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg,
            (void (*)(GroupRMSParams<n>, int)) GroupRMSNormKernelWithoutWeights<DType, float4, n>, params, rounds));
    }
}

template <int n>
void GroupRMSNormKernelLauncher(GroupRMSParams<n> params)
{
#define GROUP_RMS_NORM_DISPATCH1(DTYPE) return GroupRMSNormKernel<DTYPE, n>(params);

    switch (params.dtype)
    {
    case nvinfer1::DataType::kHALF: GROUP_RMS_NORM_DISPATCH1(half); break;
    case nvinfer1::DataType::kBF16: GROUP_RMS_NORM_DISPATCH1(__nv_bfloat16); break;
    case nvinfer1::DataType::kFLOAT: GROUP_RMS_NORM_DISPATCH1(float); break;
    default: TLLM_CHECK_WITH_INFO(false, "Unsupported data type for GroupRMSNorm");
    }
}

// Explicit instantiations for n=1 to 32
#define INSTANTIATE_GROUP_RMS_NORM(n) template void GroupRMSNormKernelLauncher<n>(GroupRMSParams<n> params);

INSTANTIATE_GROUP_RMS_NORM(1)
INSTANTIATE_GROUP_RMS_NORM(2)
INSTANTIATE_GROUP_RMS_NORM(3)
INSTANTIATE_GROUP_RMS_NORM(4)
INSTANTIATE_GROUP_RMS_NORM(5)
INSTANTIATE_GROUP_RMS_NORM(6)
INSTANTIATE_GROUP_RMS_NORM(7)
INSTANTIATE_GROUP_RMS_NORM(8)

} // namespace tensorrt_llm::kernels::group_rms_norm
