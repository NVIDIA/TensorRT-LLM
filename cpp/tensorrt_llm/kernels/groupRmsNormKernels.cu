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
__global__ void GroupRMSNormKernelWithoutWeights(PackedType** __restrict__ inputs, PackedType** __restrict__ outputs,
    uint32_t const* __restrict__ input_dims, uint32_t const* __restrict__ input_strides,
    uint32_t const* __restrict__ output_strides,
    // Starting warp idx for each input
    uint32_t const* __restrict__ warp_prefix_sum, const uint32_t __restrict__ rounds,
    const uint32_t __restrict__ num_inputs, float const eps)
{
    // Add prints here to log the values of the inputs and outputs
    // Debug prints for thread 0 in first warp
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
    {
        printf("---------------GroupRMSNormKernelWithoutWeights-----------------\n");
        printf("[DEBUG] Kernel launch parameters:\n");
        printf("num_inputs=%u\n", num_inputs);
        printf("rounds=%u\n", rounds);
        printf("eps=%f\n", eps);

        // Print input metadata
        for (uint32_t i = 0; i < num_inputs; ++i)
        {
            printf("Input %u:\n", i);
            printf("  dim=%u\n", input_dims[i]);
            printf("  in_stride=%u\n", input_strides[i]);
            printf("  out_stride=%u\n", output_strides[i]);
            printf("  warp_prefix=%u\n", warp_prefix_sum[i]);
            // printf("  ptr=%p\n", reinterpret_cast<const void*>(inputs[i]));
        }
        printf("Total warp_prefix=%u\n", warp_prefix_sum[num_inputs]);
    }
    __syncthreads();

    const uint32_t bx = blockIdx.x; // Maps to batch size
    constexpr uint32_t warp_size = 32;
    const uint32_t warp_idx = threadIdx.y;
    const uint32_t lane_idx = threadIdx.x;
    const uint32_t thread_idx = warp_idx * warp_size + lane_idx;

    static constexpr int kPackedSize = sizeof(PackedType) / sizeof(DType);
    float warp_acc = 0.0f;

    // Each thread calculates its own partial sum

    __shared__ float smem_rsqrts[32];
    __shared__ uint32_t smem_input_mask[32];
    __shared__ float smem_warp_sum_sqs[32];
#pragma unroll
    for (int i = 0; i < 32; ++i)
    {
        smem_input_mask[i] = 0;
        smem_warp_sum_sqs[i] = 0.0f;
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
    uint32_t round_offset = 32 * kPackedSize;
    uint32_t warp_offset = round_offset * rounds;
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        printf("Block[%u] block_offset: %u\n", bx, block_offset);
        printf("Block[%u] warp_offset: %u\n", bx, warp_offset);
        printf("Block[%u] round_offset: %u\n", bx, round_offset);
    }
    __syncthreads();
    if (lane_idx == 0)
    {
        printf("Block[%u] Warp[%u] processing input %u\n", bx, warp_idx, input_idx);
    }
    __syncthreads();
    // Calculate sum of squares for each thread
    for (uint32_t i = 0; i < rounds; i++)
    {
        uint32_t idx = block_offset + local_warp_idx * warp_offset + i * round_offset + lane_idx * kPackedSize;
        if (idx < input_dims[input_idx] + block_offset)
        {
            if (lane_idx % 16 == 0)
            {
                printf("Thread[%u, %u, %u] idx = %u; total warp offset: %u; round_offset: %u; lane_offset: %u\n", bx,
                    threadIdx.x, threadIdx.y, idx, local_warp_idx * warp_offset, i * round_offset,
                    lane_idx * kPackedSize);
            }
            PackedType packed_data = input_ptr[idx / kPackedSize];
            DType* packed_input_ptr = reinterpret_cast<DType*>(&packed_data);
#pragma unroll
            for (uint32_t j = 0; j < kPackedSize; j++)
            {
                float v = static_cast<float>(packed_input_ptr[j]);
                warp_acc += v * v;
            }
            if (lane_idx == 0)
            {
                printf("Thread[%u, %u, %u] idx = %u; input_idx %u; warp_acc = %f; Loaded packed: [%f, %f, %f, %f]\n",
                    bx, threadIdx.x, threadIdx.y, idx, input_idx, warp_acc, static_cast<float>(packed_input_ptr[0]),
                    static_cast<float>(packed_input_ptr[1]), static_cast<float>(packed_input_ptr[2]),
                    static_cast<float>(packed_input_ptr[3]));
            }
        }
        else
        {
            if (lane_idx % 16 == 0)
            {
                printf("Thread[%u, %u, %u] idx = %u; input %u out of bounds\n", bx, threadIdx.x, threadIdx.y, idx,
                    input_idx);
            }
        }
    }

    __syncthreads();
    smem_input_mask[warp_idx] = input_idx;
    smem_warp_sum_sqs[warp_idx] = tensorrt_llm::common::warpReduceSum(warp_acc);
    if (lane_idx == 0)
    {
        printf("Block [%u] Warp[%u] input_mask %u: wrap_level reduction = %f\n", bx, warp_idx,
            smem_input_mask[warp_idx], smem_warp_sum_sqs[warp_idx]);
    }

    __syncthreads();

    warp_acc = 0.0f;
    // // Cross wrap reduction on all inputs
    if (warp_idx < num_inputs)
    {
        // Each warp sums one input
        if (warp_idx == smem_input_mask[lane_idx])
        {
            warp_acc = smem_warp_sum_sqs[lane_idx];
        }
        smem_rsqrts[warp_idx] = tensorrt_llm::common::warpReduceSum(warp_acc);
    }

    __syncthreads();
    if (warp_idx < num_inputs && warp_idx < 2)
    {
        printf("Block [%u] Warp[%u] lane[%u] target input %u: warp_acc = %f; rsqrt = %f\n", bx, warp_idx, lane_idx,
            smem_input_mask[lane_idx], warp_acc, smem_rsqrts[warp_idx]);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        printf("After block reduce sum_sqs: ");
        for (uint32_t i = 0; i < num_inputs; i++)
        {
            printf("Block[%u] reduce sum_sqs for input %u: %f ", bx, i, smem_rsqrts[i]);
        }
        printf("\n");
    }

    // Calculate normalization factor for each input
    // Use thread_idx as input_idx here.
    if (warp_idx == 0 && thread_idx < num_inputs)
    {
        smem_rsqrts[thread_idx] = rsqrtf(smem_rsqrts[thread_idx] / input_dims[thread_idx] + eps);
    }

    // Second synchronization point to ensure normalization factors are available
    __syncthreads();

    // Apply normalization for inputs in parallel
    for (uint32_t i = 0; i < rounds; i++)
    {
        uint32_t idx = block_offset + local_warp_idx * warp_offset + i * round_offset + lane_idx * kPackedSize;
        if (lane_idx % 16 == 0)
        {
            printf("Thread[%u, %u, %u] idx = %u; total warp offset: %u; round_offset: %u; lane_offset: %u\n", bx,
                threadIdx.x, threadIdx.y, idx, local_warp_idx * warp_offset, i * round_offset, lane_idx * kPackedSize);
        }
        if (idx < input_dims[input_idx] + block_offset)
        {
            PackedType packed_input = input_ptr[idx / kPackedSize];
            DType* packed_input_ptr = reinterpret_cast<DType*>(&packed_input);

            PackedType packed_output = output_ptr[idx / kPackedSize];
            DType* packed_output_ptr = reinterpret_cast<DType*>(&packed_output);
#pragma unroll
            for (uint32_t j = 0; j < kPackedSize; j++)
            {
                packed_output_ptr[j]
                    = static_cast<DType>(static_cast<float>(packed_input_ptr[j]) * smem_rsqrts[input_idx]);
                output_ptr[idx / kPackedSize] = packed_output;

                if (j == kPackedSize - 1 && lane_idx == 0)
                {
                    printf(
                        "Thread[%u, %u, %u] idx = %u; smem_rsqrts[%u] = %f; input packed: [%f, %f, %f, %f]; output "
                        "packed: [%f, %f, %f, %f]\n",
                        bx, threadIdx.x, threadIdx.y, idx, input_idx, smem_rsqrts[input_idx],
                        static_cast<float>(packed_input_ptr[0]), static_cast<float>(packed_input_ptr[1]),
                        static_cast<float>(packed_input_ptr[2]), static_cast<float>(packed_input_ptr[3]),
                        static_cast<float>(packed_output_ptr[0]), static_cast<float>(packed_output_ptr[1]),
                        static_cast<float>(packed_output_ptr[2]), static_cast<float>(packed_output_ptr[3]));
                }
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

    cudaStreamSynchronize(stream);
    // Log input parameters
    printf("---------------GroupRMSNormKernel-----------------\n");
    printf("inputs: %p\n", inputs);
    printf("weights: %p\n", weights);
    printf("outputs: %p\n", outputs);
    for (uint32_t i = 0; i < num_inputs; i++)
    {
        printf("input %u: %p\n", i, inputs[i]);
        printf("weight %u: %p\n", i, weights[i]);
        printf("output %u: %p\n", i, outputs[i]);
    }
    printf("batch_size: %u\n", batch_size);
    printf("num_inputs: %u\n", num_inputs);
    printf("input_dims: ");
    for (uint32_t i = 0; i < num_inputs; i++)
    {
        printf("%u ", input_dims[i]);
    }
    printf("\n");

    // Allocate memory for inputs, weights, and outputs and cast them to float4*
    std::vector<float4*> inputs_float4(num_inputs);
    std::vector<float4*> outputs_float4(num_inputs);
    std::vector<float4*> weights_float4(num_inputs);
    for (uint32_t i = 0; i < num_inputs; ++i)
    {
        inputs_float4[i] = reinterpret_cast<float4*>(inputs[i]); // Safe if aligned
        outputs_float4[i] = reinterpret_cast<float4*>(outputs[i]);
        weights_float4[i] = reinterpret_cast<float4*>(weights[i]);
    }
    float4** d_inputs;
    float4** d_outputs;
    float4** d_weights;

    cudaMalloc(&d_inputs, num_inputs * sizeof(float4*));
    cudaMemcpy(d_inputs, inputs_float4.data(), num_inputs * sizeof(float4*), cudaMemcpyHostToDevice);

    cudaMalloc(&d_outputs, num_inputs * sizeof(float4*));
    cudaMemcpy(d_outputs, outputs_float4.data(), num_inputs * sizeof(float4*), cudaMemcpyHostToDevice);

    cudaMalloc(&d_weights, num_inputs * sizeof(float4*));
    cudaMemcpy(d_weights, weights_float4.data(), num_inputs * sizeof(float4*), cudaMemcpyHostToDevice);

    // Kernel assertions
    constexpr uint32_t kPackedSize = sizeof(float4) / sizeof(DType);
    // TLLM_CHECK_WITH_INFO(!enable_weights, "Weights are not supported yet. Will add in the future.");
    // TLLM_CHECK_WITH_INFO(weights[0] == nullptr, "Weights are not supported yet. Will add in the future.");
    TLLM_CHECK_WITH_INFO(num_inputs <= 32, "Only up to 32 inputs are supported.");
    for (uint32_t i = 0; i < num_inputs; i++)
    {
        TLLM_CHECK_WITH_INFO(input_dims[i] % 32 == 0, "Input dimension must be divisible by 32.");
        // Each wrap process 32*kPackedSize tokens per round.
        // Need to be divisible by 32*kPackedSize
        TLLM_CHECK_WITH_INFO(input_dims[i] % (32 * kPackedSize) == 0,
            "Input[%u] dimension %u is not divisible by %u. Finer granularity is not supported yet.", i, input_dims[i],
            32 * kPackedSize);
    }

    // Calculate total warps to launch and rounds needed
    uint32_t total_input_length = std::accumulate(input_dims, input_dims + num_inputs, 0);
    uint32_t input_chunk_per_warp = 32 * kPackedSize;
    uint32_t total_warps_needed = (total_input_length + input_chunk_per_warp - 1) / input_chunk_per_warp; // ceil_div
    uint32_t num_warps_to_launch = std::min<uint32_t>(32, total_warps_needed);
    uint32_t rounds = (total_warps_needed + num_warps_to_launch - 1) / num_warps_to_launch;
    printf("kPackedSize: %u\n", kPackedSize);
    printf("rounds: %u\n", rounds);

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

    printf("warp_prefix_sum: ");
    for (int i = 0; i < num_inputs + 1; ++i)
    {
        printf("%u ", warp_prefix_sum[i]);
    }
    printf("\n");

    dim3 grid_dim(batch_size);
    dim3 block_dim(32, num_warps_to_launch);
    printf("grid_dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("block_dim: %d, %d, %d\n", block_dim.x, block_dim.y, block_dim.z);

    // Prepare inputs and outputs for the kernel
    uint32_t* d_input_dims;
    uint32_t* d_input_strides;
    uint32_t* d_output_strides;
    uint32_t* d_warp_prefix_sum;

    // Device memory allocation
    TLLM_CUDA_CHECK(cudaMalloc((void**) &d_input_dims, num_inputs * sizeof(uint32_t)));
    TLLM_CUDA_CHECK(cudaMalloc((void**) &d_input_strides, num_inputs * sizeof(uint32_t)));
    TLLM_CUDA_CHECK(cudaMalloc((void**) &d_output_strides, num_inputs * sizeof(uint32_t)));
    TLLM_CUDA_CHECK(cudaMalloc((void**) &d_warp_prefix_sum, (num_inputs + 1) * sizeof(uint32_t)));

    // Remove all host->device copies for inputs/outputs
    // Keep only metadata copies:
    TLLM_CUDA_CHECK(
        cudaMemcpyAsync(d_input_dims, input_dims, num_inputs * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    TLLM_CUDA_CHECK(
        cudaMemcpyAsync(d_input_strides, input_strides, num_inputs * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    TLLM_CUDA_CHECK(cudaMemcpyAsync(
        d_output_strides, output_strides, num_inputs * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    TLLM_CUDA_CHECK(cudaMemcpyAsync(d_warp_prefix_sum, warp_prefix_sum.data(), (num_inputs + 1) * sizeof(uint32_t),
        cudaMemcpyHostToDevice, stream));

    // Wait for async copies to complete before kernel launch
    TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify device pointers
    for (uint32_t i = 0; i < num_inputs; ++i)
    {
        TLLM_CHECK_WITH_INFO(inputs[i] != nullptr, "Input pointer is null for index %d", i);
        TLLM_CHECK_WITH_INFO(outputs[i] != nullptr, "Output pointer is null for index %d", i);
    }

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

    // Check pointer alignment
    assert(reinterpret_cast<uintptr_t>(inputs[0]) % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(outputs[0]) % 16 == 0);

    // TLLM_CUDA_CHECK(
    cudaLaunchKernelEx(&cfg, GroupRMSNormKernelWithoutWeights<DType, float4>, d_inputs, d_outputs, d_input_dims,
        d_input_strides, d_output_strides, d_warp_prefix_sum, rounds, num_inputs, eps);
    // );

    // Add synchronization for debugging
    // TLLM_CUDA_CHECK(
    cudaStreamSynchronize(stream);
    // );

    // Cleanup device memory
    TLLM_CUDA_CHECK(cudaFree(d_inputs));
    TLLM_CUDA_CHECK(cudaFree(d_outputs));
    TLLM_CUDA_CHECK(cudaFree(d_weights));
    TLLM_CUDA_CHECK(cudaFree(d_input_dims));
    TLLM_CUDA_CHECK(cudaFree(d_input_strides));
    TLLM_CUDA_CHECK(cudaFree(d_output_strides));
    TLLM_CUDA_CHECK(cudaFree(d_warp_prefix_sum));
}

template void tensorrt_llm::kernels::group_rms_norm::GroupRMSNormKernel<half>(half**, half**, half**, uint32_t const*,
    uint32_t const*, uint32_t const*, uint32_t, uint32_t, float, bool, cudaStream_t);

#ifdef ENABLE_BF16
template void tensorrt_llm::kernels::group_rms_norm::GroupRMSNormKernel<__nv_bfloat16>(__nv_bfloat16**, __nv_bfloat16**,
    __nv_bfloat16**, uint32_t const*, uint32_t const*, uint32_t const*, uint32_t, uint32_t, float, bool, cudaStream_t);
#endif

template void tensorrt_llm::kernels::group_rms_norm::GroupRMSNormKernel<float>(float**, float**, float**,
    uint32_t const*, uint32_t const*, uint32_t const*, uint32_t, uint32_t, float, bool, cudaStream_t);
} // namespace tensorrt_llm::kernels::group_rms_norm
