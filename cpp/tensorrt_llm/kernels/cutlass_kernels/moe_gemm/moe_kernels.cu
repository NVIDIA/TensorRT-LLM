/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/workspace.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>

// Ignore CUTLASS warnings about type punning
#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"

#include "cutlass/array.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#include "cutlass_extensions/epilogue/thread/fused_activations.h"

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif

#include "../include/moe_kernels.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"
#include "tensorrt_llm/kernels/preQuantScaleKernel.h"
#include "tensorrt_llm/kernels/quantization.cuh"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#else
#include "3rdparty/cub/cub.cuh"
#include "3rdparty/cub/device/device_radix_sort.cuh"
#include "3rdparty/cub/util_type.cuh"
#endif

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels::cutlass_kernels
{
/**
 * Takes the input maps and prepares the expanded maps for min latency
 * @param num_active_experts_per_node: Number of active experts on current node
 * @param experts_to_token_scores: The score of each token for each activated expert. 0 if the expert is not chosen by
 * the token. Only the first num_active_experts_per_ rows are valid
 * @param active_expert_global_ids: The global expert id for each activated expert
 * Only the first num_active_experts_per_ values are valid
 * @param expert_first_token_offset: Store the first token offset for each expert
 */
template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ void initTensor(T* value, int const tid, int const total_num, T const init_value)
{
    for (int i = tid; i < total_num; i += BLOCK_SIZE)
    {
        value[i] = init_value;
    }
}

template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ void setLocalExperts(int* s_local_experts, T const* token_selected_experts,
    int const total_num_experts, int const tid, int const start_expert, int const end_expert)
{
    for (int i = tid; i < total_num_experts; i += BLOCK_SIZE)
    {
        int const expert = token_selected_experts[i];

        // If expert is in the current node, subtract start_expert to shift the range to [0, num_experts_per_node)
        bool is_valid_expert = expert >= start_expert && expert < end_expert;
        if (is_valid_expert)
        {
            int local_expert_id = expert - start_expert;
            if (s_local_experts[local_expert_id] == 0)
            {
                s_local_experts[local_expert_id] = 1; // @TODO: Make sure that we allow duplicated write here
            }
        }
    }
    __syncthreads();
}

template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ void prefixSum(T* out, T* in, int const num, int const tid)
{
    typedef cub::BlockScan<T, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage tempStorage;

    T threadData = 0;
    if (tid < num)
    {
        threadData = in[tid];
    }

    BlockScan(tempStorage).InclusiveSum(threadData, threadData);
    __syncthreads();

    if (tid < num)
    {
        out[tid] = threadData;
    }
    __syncthreads();
}

__device__ __forceinline__ void setActiveNum(int& num_active, int& num_active_offset_start, int& num_active_offset_end,
    int const cluster_size, int const cluster_rank)
{
    int num_remainder = num_active % cluster_size;
    int num_active_per_node = max(0, num_active - 1) / cluster_size; // num_active_per_node shouldn't be neg
    if (cluster_rank < num_remainder)
    {
        num_active = num_active_per_node + 1;
        num_active_offset_start = cluster_rank * num_active;
    }
    else
    {
        num_active = num_active_per_node;
        num_active_offset_start = cluster_rank * num_active_per_node + num_remainder;
    }
    num_active_offset_end = num_active_offset_start + num_active;
}

template <int BLOCK_SIZE>
__global__ void buildMinLatencyActiveExpertMapsKernel(int* num_active_experts_per_node, float* experts_to_token_scores,
    int* active_expert_global_ids, int64_t* expert_first_token_offset, int const* token_selected_experts,
    float const* token_final_scales, int64_t const num_tokens, int const num_experts_per_token, int const start_expert,
    int const end_expert, int const num_experts_per_node, bool const smart_routing, int const cluster_rank,
    int const cluster_size, int const num_experts_smem)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    // Use one block to process the min latency case
    int tid = threadIdx.x;
    // 0. init the global memory experts_to_token_scores [num_experts_per_node, num_token]
    int const total_local_scales = num_experts_per_node * num_tokens;
    initTensor<float, BLOCK_SIZE>(experts_to_token_scores, tid, total_local_scales, 0.0f);
    initTensor<int, BLOCK_SIZE>(active_expert_global_ids, tid, num_experts_per_node, -1);

    __threadfence(); //@Todo: check do I need this fence for previous zero setting

    // 1. mask for the active expert: 1 stands for active
    extern __shared__ int s_local_experts[];
    int* s_store_experts = s_local_experts + num_experts_smem;
    initTensor<int, BLOCK_SIZE>(s_local_experts, tid, num_experts_smem, 0);
    __syncthreads();

    // 2. set the shared array s_local_experts[]
    int const total_num_experts = num_tokens * num_experts_per_token;
    setLocalExperts<int, BLOCK_SIZE>(
        s_local_experts, token_selected_experts, total_num_experts, tid, start_expert, end_expert);

    // 3. perform prefix sum to acquire the store position and total active experts
    //@TODO: Use cub first, might need to change it to self-defined api
    prefixSum<int, BLOCK_SIZE>(s_store_experts, s_local_experts, num_experts_smem, tid);

    // 4. store the num of active experts
    int num_active = s_store_experts[num_experts_smem - 1];
    int num_active_offset_start = 0;
    int num_active_offset_end = 0;

    if (smart_routing)
    {
        setActiveNum(num_active, num_active_offset_start, num_active_offset_end, cluster_size, cluster_rank);
    }

    if (tid == 0)
    {
        *num_active_experts_per_node = num_active;
    }

    // 5. store the global expert id for each expert
    if (smart_routing)
    {
        for (int i = tid; i < num_experts_smem; i += BLOCK_SIZE)
        {
            if (s_local_experts[i])
            {
                int offset = s_store_experts[i] - 1;
                if (offset >= num_active_offset_start && offset < num_active_offset_end)
                {
                    active_expert_global_ids[offset - num_active_offset_start] = i;
                }
                else
                {
                    s_local_experts[i] = 0;
                }
            }
        }
        __syncthreads(); // Need sync to update the s_local_experts
    }
    else
    {
        for (int i = tid; i < num_experts_smem; i += BLOCK_SIZE)
        {
            if (s_local_experts[i])
            {
                int offset = s_store_experts[i] - 1;
                active_expert_global_ids[offset] = i + start_expert;
            }
        }
    }

    // 6. store the scale values
    __threadfence(); //@Todo: check do I need this fence for previous zero setting
    for (int i = tid; i < total_num_experts; i += BLOCK_SIZE)
    {
        int const expert = token_selected_experts[i];

        // If expert is not in the current node, set it to num_experts_per_node
        // If expert is in the current node, subtract start_expert to shift the range to [0, num_experts_per_node)
        bool is_valid_expert
            = smart_routing ? s_local_experts[expert] : (expert >= start_expert && expert < end_expert);

        if (is_valid_expert)
        {
            int token = i / num_experts_per_token;
            float const scale = token_final_scales[i];
            int offset = s_store_experts[expert - start_expert] - 1 - num_active_offset_start;
            experts_to_token_scores[offset * num_tokens + token] = scale;
        }
    }
    // 7. set default value for redundant memory
    for (int i_exp = num_active + tid; i_exp < num_experts_per_node; i_exp += BLOCK_SIZE)
    {
        active_expert_global_ids[i_exp] = -1;
    }
    // 8. set expert_first_token_offset
    for (int i_exp = tid; i_exp < num_experts_per_node + 1; i_exp += BLOCK_SIZE)
    {
        if (i_exp < num_active)
        {
            expert_first_token_offset[i_exp] = i_exp * num_tokens;
        }
        else
        {
            expert_first_token_offset[i_exp] = num_active * num_tokens;
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void buildMinLatencyActiveExpertMaps(int* num_active_experts_per_node, float* experts_to_token_scores,
    int* active_expert_global_ids, int64_t* expert_first_token_offset, int const* token_selected_experts,
    float const* token_final_scales, int64_t const num_tokens, int const experts_per_token, int const start_expert,
    int const end_expert, int const num_experts_per_node, int const cluster_rank, int const cluster_size,
    int const num_experts_smem, cudaStream_t const stream)
{
    TLLM_CHECK_WITH_INFO(num_experts_per_node == (end_expert - start_expert),
        "num_experts_per_node must be equal to end_expert - start_expert");

    TLLM_CHECK_WITH_INFO(num_experts_per_node <= 256, "don't support num_experts_per_node > 256 cases");

    int const threads = 256;
    int const blocks = 1;
    bool const smart_routing = cluster_size > 1;

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = threads;
    config.dynamicSmemBytes = num_experts_smem * sizeof(int) * 2;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, buildMinLatencyActiveExpertMapsKernel<threads>, num_active_experts_per_node,
        experts_to_token_scores, active_expert_global_ids, expert_first_token_offset, token_selected_experts,
        token_final_scales, num_tokens, experts_per_token, start_expert, end_expert, num_experts_per_node,
        smart_routing, cluster_rank, cluster_size, num_experts_smem);
}

template <int BLOCK_SIZE, int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
__global__ void fusedBuildExpertMapsSortFirstTokenKernel(int const* const token_selected_experts,
    int* const unpermuted_token_selected_experts, int* const permuted_source_token_ids,
    int64_t* const expert_first_token_offset, int64_t const num_tokens, int const experts_per_token,
    int const start_expert, int const end_expert, int const num_experts_per_node)
{
    // Only using block wise collective so we can only have one block
    assert(gridDim.x == 1);

    assert(start_expert <= end_expert);
    assert(num_experts_per_node == (end_expert - start_expert));
    assert(num_experts_per_node <= (1 << LOG2_NUM_EXPERTS));

    int const token = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    bool is_valid_token = token < num_tokens;

    // This is the masked expert id for this token
    int local_token_selected_experts[EXPERTS_PER_TOKEN];
    // This is the final permuted rank of this token (ranked by selected expert)
    int local_token_permuted_indices[EXPERTS_PER_TOKEN];

    // Wait PDL before reading token_selected_experts
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

// build expert map
// we need to populate expert ids for all threads, even if there are
// fewer tokens
#pragma unroll
    for (int i = 0; i < EXPERTS_PER_TOKEN; i++)
    {
        int const expert
            = is_valid_token ? token_selected_experts[token * EXPERTS_PER_TOKEN + i] : num_experts_per_node;

        // If the token is not valid, set the expert id to num_experts_per_node + 1
        // If expert is not in the current node, set it to num_experts_per_node
        // If expert is in the current node, subtract start_expert to shift the range to [0, num_experts_per_node)
        bool is_valid_expert = expert >= start_expert && expert < end_expert;
        local_token_selected_experts[i] = !is_valid_token ? num_experts_per_node + 1
            : is_valid_expert                             ? (expert - start_expert)
                                                          : num_experts_per_node;
    }

    // TODO: decompose cub's sort to expose the bucket starts, and just return
    // that to elide the binary search

    // sort the expert map
    using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
    extern __shared__ unsigned char temp_storage[];
    auto& sort_temp = *reinterpret_cast<typename BlockRadixRank::TempStorage*>(temp_storage);

    // Sanity check that the number of bins do correspond to the number of experts
    static_assert(BlockRadixRank::BINS_TRACKED_PER_THREAD * BLOCK_SIZE >= (1 << LOG2_NUM_EXPERTS));
    assert(BlockRadixRank::BINS_TRACKED_PER_THREAD * BLOCK_SIZE >= num_experts_per_node);

    int local_expert_first_token_offset[BlockRadixRank::BINS_TRACKED_PER_THREAD];

    cub::BFEDigitExtractor<int> extractor(0, LOG2_NUM_EXPERTS);
    BlockRadixRank(sort_temp).RankKeys(
        local_token_selected_experts, local_token_permuted_indices, extractor, local_expert_first_token_offset);

// We are done with compute, launch the dependent kernels while the stores are in flight
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif

    // write to shared memory and global memory
    if (is_valid_token)
    {
#pragma unroll
        for (int i = 0; i < EXPERTS_PER_TOKEN; i++)
        {
            unpermuted_token_selected_experts[token * EXPERTS_PER_TOKEN + i] = local_token_selected_experts[i];
            permuted_source_token_ids[local_token_permuted_indices[i]] = i * num_tokens + token;
        }
    }

#pragma unroll
    for (int expert_id = 0; expert_id < BlockRadixRank::BINS_TRACKED_PER_THREAD; expert_id++)
    {
        int out_expert_id = expert_id + token * BlockRadixRank::BINS_TRACKED_PER_THREAD;
        if (out_expert_id < num_experts_per_node + 1)
        {
            expert_first_token_offset[out_expert_id] = local_expert_first_token_offset[expert_id];
        }
    }
}

template <int BLOCK_SIZE, int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenDispatch(int const* token_selected_experts,
    int* unpermuted_token_selected_experts, int* permuted_source_token_ids, int64_t* expert_first_token_offset,
    int64_t const num_tokens, int const num_experts_per_node, int const experts_per_token, int const start_expert,
    int const end_expert, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(num_experts_per_node == (end_expert - start_expert),
        "num_experts_per_node must be equal to end_expert - start_expert");
    int const threads = BLOCK_SIZE;
    int const blocks = (num_tokens + threads - 1) / threads;
    TLLM_CHECK_WITH_INFO(blocks == 1, "Current implementation requires single block");

    using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
    size_t shared_size = sizeof(typename BlockRadixRank::TempStorage);

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = threads;
    config.dynamicSmemBytes = shared_size;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;

    auto kernel = &fusedBuildExpertMapsSortFirstTokenKernel<BLOCK_SIZE, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;

    int device = 0;
    int max_smem_per_block = 0;
    check_cuda_error(cudaGetDevice(&device));
    check_cuda_error(cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (shared_size >= static_cast<size_t>(max_smem_per_block))
    {
        // This should mean that
        // cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)
        // wouldn't work.
        return false;
    }

    check_cuda_error(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));
    check_cuda_error(cudaLaunchKernelEx(&config, kernel, token_selected_experts, unpermuted_token_selected_experts,
        permuted_source_token_ids, expert_first_token_offset, num_tokens, experts_per_token, start_expert, end_expert,
        num_experts_per_node));

    return true;
}

template <int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenBlockSize(int const* token_selected_experts,
    int* unpermuted_token_selected_experts, int* permuted_source_token_ids, int64_t* expert_first_token_offset,
    int64_t const num_tokens, int const num_experts_per_node, int const experts_per_token, int const start_expert,
    int const end_expert, cudaStream_t stream)
{
    int const block_size = num_tokens;
    if (num_tokens > 256)
    {
        TLLM_LOG_TRACE(
            "Number of tokens %d is greater than 256, which is not supported for fused moe prologues", num_tokens);
        return false;
    }

    auto func = &fusedBuildExpertMapsSortFirstTokenDispatch<32, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
    if (block_size > 32 && block_size <= 64)
    {
        func = &fusedBuildExpertMapsSortFirstTokenDispatch<64, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
    }
    else if (block_size > 64 && block_size <= 128)
    {
        func = &fusedBuildExpertMapsSortFirstTokenDispatch<128, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
    }
    else if (block_size > 128 && block_size <= 256)
    {
        func = &fusedBuildExpertMapsSortFirstTokenDispatch<256, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
    }

    return func(token_selected_experts, unpermuted_token_selected_experts, permuted_source_token_ids,
        expert_first_token_offset, num_tokens, num_experts_per_node, experts_per_token, start_expert, end_expert,
        stream);
}

template <int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenBlockSize(int const* token_selected_experts,
    int* unpermuted_token_selected_experts, int* permuted_source_token_ids, int64_t* expert_first_token_offset,
    int64_t const num_tokens, int const num_experts_per_node, int const experts_per_token, int const start_expert,
    int const end_expert, cudaStream_t stream)
{
    auto func = &fusedBuildExpertMapsSortFirstTokenBlockSize<1, LOG2_NUM_EXPERTS>;
    switch (experts_per_token)
    {
    case 1:
    {
        func = &fusedBuildExpertMapsSortFirstTokenBlockSize<1, LOG2_NUM_EXPERTS>;
        break;
    }
    case 2:
    {
        func = &fusedBuildExpertMapsSortFirstTokenBlockSize<2, LOG2_NUM_EXPERTS>;
        break;
    }
    case 4:
    {
        func = &fusedBuildExpertMapsSortFirstTokenBlockSize<4, LOG2_NUM_EXPERTS>;
        break;
    }
    case 6:
    {
        func = &fusedBuildExpertMapsSortFirstTokenBlockSize<6, LOG2_NUM_EXPERTS>;
        break;
    }
    case 8:
    {
        func = &fusedBuildExpertMapsSortFirstTokenBlockSize<8, LOG2_NUM_EXPERTS>;
        break;
    }
    default:
    {
        TLLM_LOG_TRACE("Top-K value %d does not have supported fused moe prologues", experts_per_token);
        return false;
    }
    }
    return func(token_selected_experts, unpermuted_token_selected_experts, permuted_source_token_ids,
        expert_first_token_offset, num_tokens, num_experts_per_node, experts_per_token, start_expert, end_expert,
        stream);
}

bool fusedBuildExpertMapsSortFirstToken(int const* token_selected_experts, int* unpermuted_token_selected_experts,
    int* permuted_source_token_ids, int64_t* expert_first_token_offset, int64_t const num_tokens,
    int const num_experts_per_node, int const experts_per_token, int const start_expert, int const end_expert,
    cudaStream_t stream)
{
    // We need enough bits to represent [0, num_experts_per_node+1] (inclusive) i.e. num_experts_per_node + 2 values
    // This is floor(log2(num_experts_per_node+1)) + 1
    int expert_log = static_cast<int>(log2(num_experts_per_node + 1)) + 1;
    if (expert_log <= 9)
    {
        auto funcs = std::array{&fusedBuildExpertMapsSortFirstTokenBlockSize<1>,
            &fusedBuildExpertMapsSortFirstTokenBlockSize<2>, &fusedBuildExpertMapsSortFirstTokenBlockSize<3>,
            &fusedBuildExpertMapsSortFirstTokenBlockSize<4>, &fusedBuildExpertMapsSortFirstTokenBlockSize<5>,
            &fusedBuildExpertMapsSortFirstTokenBlockSize<6>, &fusedBuildExpertMapsSortFirstTokenBlockSize<7>,
            &fusedBuildExpertMapsSortFirstTokenBlockSize<8>, &fusedBuildExpertMapsSortFirstTokenBlockSize<9>};

        return funcs[expert_log - 1](token_selected_experts, unpermuted_token_selected_experts,
            permuted_source_token_ids, expert_first_token_offset, num_tokens, num_experts_per_node, experts_per_token,
            start_expert, end_expert, stream);
    }
    TLLM_LOG_TRACE("Experts per node %d does not have supported fused moe prologues", num_experts_per_node);
    return false;
}

/**
 * Takes the input maps and prepares the expanded maps for the sort step
 * @param unpermuted_token_selected_experts: Buffer of transformed expert ids masked for the current node, used as the
 * keys for the sort
 * @param unpermuted_source_token_ids: Buffer of unpermuted token ids that will be used to identify the source row for
 * each expanded token, used as the values for the sort
 */
__global__ void buildExpertMapsKernel(int const* token_selected_experts, int* unpermuted_token_selected_experts,
    int* unpermuted_source_token_ids, int64_t const num_tokens, int const experts_per_token, int const start_expert,
    int const end_expert, int const num_experts_per_node)
{
    int const token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens)
    {
        return;
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    for (int i = 0; i < experts_per_token; i++)
    {
        int const expert = token_selected_experts[token * experts_per_token + i];
        // If expert is not in the current node, set it to num_experts_per_node
        // If expert is in the current node, subtract start_expert to shift the range to [0, num_experts_per_node)
        bool is_valid_expert = expert >= start_expert && expert < end_expert;
        unpermuted_token_selected_experts[token * experts_per_token + i]
            = is_valid_expert ? (expert - start_expert) : num_experts_per_node;
        unpermuted_source_token_ids[token * experts_per_token + i] = i * num_tokens + token;
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void buildExpertMaps(int const* token_selected_experts, int* unpermuted_token_selected_experts,
    int* unpermuted_source_token_ids, int64_t const num_tokens, int const num_experts_per_node,
    int const experts_per_token, int const start_expert, int const end_expert, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(num_experts_per_node == (end_expert - start_expert),
        "num_experts_per_node must be equal to end_expert - start_expert");
    int const threads = std::min(int64_t(1024), num_tokens);
    int const blocks = (num_tokens + threads - 1) / threads;

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = threads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, buildExpertMapsKernel, token_selected_experts, unpermuted_token_selected_experts,
        unpermuted_source_token_ids, num_tokens, experts_per_token, start_expert, end_expert, num_experts_per_node);
}

// ========================== CUB Sorting things ====================================
CubKeyValueSorter::CubKeyValueSorter()
    : num_experts_(0)
    , num_bits_(sizeof(int) * 8)
{
}

int CubKeyValueSorter::expertsToBits(int num_experts)
{
    // Max value we represent is V = num_experts + (num_experts - 1) = 2 * num_experts - 1
    // The maximum number of bits is therefore floor(log2(V)) + 1
    return static_cast<int>(log2(2 * num_experts - 1)) + 1;
}

CubKeyValueSorter::CubKeyValueSorter(int const num_experts)
    : num_experts_(num_experts)
    , num_bits_(expertsToBits(num_experts))
{
}

void CubKeyValueSorter::updateNumExperts(int const num_experts)
{
    num_experts_ = num_experts;
    num_bits_ = expertsToBits(num_experts);
}

size_t CubKeyValueSorter::getWorkspaceSize(size_t const num_key_value_pairs, int const num_experts)
{
    int num_bits = expertsToBits(num_experts);
    size_t required_storage = 0;
    int* null_int = nullptr;
    cub::DeviceRadixSort::SortPairs(
        nullptr, required_storage, null_int, null_int, null_int, null_int, num_key_value_pairs, 0, num_bits);

    // TODO: fix DeviceRadixSort
    //   when num_key_value_pairs, num_experts, num_bits, required_storage = 64, 4, 3, 0
    //   The required_storage seems to vary between 0 and 1 for the same inputs
    if (required_storage == 0)
    {
        required_storage = 1;
    }
    return required_storage;
}

void CubKeyValueSorter::run(void* workspace, size_t const workspace_size, int const* keys_in, int* keys_out,
    int const* values_in, int* values_out, size_t const num_key_value_pairs, cudaStream_t stream)
{
    size_t expected_ws_size = getWorkspaceSize(num_key_value_pairs, num_experts_);
    size_t actual_ws_size = workspace_size;

    TLLM_CHECK_WITH_INFO(expected_ws_size <= workspace_size,
        "[CubKeyValueSorter::run] The allocated workspace is too small to run this problem.");
    cub::DeviceRadixSort::SortPairs(
        workspace, actual_ws_size, keys_in, keys_out, values_in, values_out, num_key_value_pairs, 0, num_bits_, stream);
}

// ============================== Infer GEMM sizes =================================
// TODO Could linear search be better for small # experts
template <class T>
__device__ inline int64_t findTotalEltsLessThanTarget(T const* sorted_indices, int64_t const arr_length, T const target)
{
    int64_t low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high)
    {
        int64_t mid = (low + high) / 2;

        if (sorted_indices[mid] >= target)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

// Calculates the start offset of the tokens for a given expert. The last element is the total number of valid tokens
__global__ void computeExpertFirstTokenOffsetKernel(int const* sorted_experts, int64_t const sorted_experts_len,
    int64_t const num_experts_per_node, int64_t* expert_first_token_offset)
{
    // First, compute the global tid. We only need 1 thread per expert.
    int const expert = blockIdx.x * blockDim.x + threadIdx.x;

    // Note that expert goes [0, num_experts] (inclusive) because we want a count for the total number of active tokens
    // at the end of the scan.
    if (expert >= num_experts_per_node + 1)
    {
        return;
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    expert_first_token_offset[expert] = findTotalEltsLessThanTarget(sorted_experts, sorted_experts_len, expert);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void computeExpertFirstTokenOffset(int const* sorted_indices, int const total_indices, int const num_experts_per_node,
    int64_t* expert_first_token_offset, cudaStream_t stream)
{
    int const num_entries = num_experts_per_node + 1;
    int const threads = std::min(1024, num_entries);
    int const blocks = (num_entries + threads - 1) / threads;

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = threads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, computeExpertFirstTokenOffsetKernel, sorted_indices, total_indices,
        num_experts_per_node, expert_first_token_offset);
}

template <class T>
using sizeof_bits = cutlass::sizeof_bits<typename cutlass_kernels::TllmToCutlassTypeAdapter<std::remove_cv_t<T>>::type>;

// Function to safely offset an pointer that may contain sub-byte types (FP4/INT4)
template <class T>
__host__ __device__ constexpr T* safe_inc_ptr(T* ptr, size_t offset)
{
    constexpr int adjustment = (sizeof_bits<T>::value < 8) ? (8 / sizeof_bits<T>::value) : 1;
    assert(offset % adjustment == 0 && "Attempt to offset index to sub-byte");
    return ptr + offset / adjustment;
}

__host__ __device__ constexpr int64_t getOffsetWeightSF(int64_t expert_id, int64_t gemm_n, int64_t gemm_k)
{
    auto min_alignment = TmaWarpSpecializedGroupedGemmInput::MinNumRowsAlignmentFP4;
    int64_t rounded_gemm_n = cute::ceil_div(gemm_n, min_alignment) * min_alignment;
    assert(gemm_k % TmaWarpSpecializedGroupedGemmInput::BlockScaleVectorSize == 0);
    return expert_id * rounded_gemm_n * gemm_k / TmaWarpSpecializedGroupedGemmInput::BlockScaleVectorSize;
}

__host__ __device__ constexpr int64_t getOffsetActivationSF(int64_t expert_id, int64_t token_offset, int64_t gemm_k)
{
    auto min_alignment = TmaWarpSpecializedGroupedGemmInput::MinNumRowsAlignmentFP4;
    // This formulation ensures that sf_offset[i + 1] - sf_offset[i] >= token_offset[i + 1] - token_offset[i].
    int64_t sf_offset = (token_offset + expert_id * (min_alignment - 1)) / min_alignment * min_alignment;
    assert(gemm_k % TmaWarpSpecializedGroupedGemmInput::BlockScaleVectorSize == 0);
    return sf_offset * gemm_k / TmaWarpSpecializedGroupedGemmInput::BlockScaleVectorSize;
}

constexpr static int NVFP4_VEC_SIZE = 16;

template <class GemmOutputType, class ComputeElem>
__device__ uint32_t quantizePackedFP4Value(ComputeElem& post_act_val, float global_scale_val,
    int64_t num_tokens_before_expert, int64_t expert_id, int64_t token_id, int64_t elem_idx, int64_t num_cols,
    int64_t max_tokens_per_expert, TmaWarpSpecializedGroupedGemmInput::ElementSF* act_sf_flat)
{
    static constexpr int CVT_FP4_NUM_THREADS_PER_SF = NVFP4_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;
    // Quantize the input to FP4
    static_assert(std::is_same_v<GemmOutputType, __nv_bfloat16> || std::is_same_v<GemmOutputType, half>);
    static_assert(ComputeElem::kElements == CVT_FP4_ELTS_PER_THREAD);
    PackedVec<GemmOutputType> packed_vec{};
    for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++)
    {
        packed_vec.elts[i].x = static_cast<GemmOutputType>(post_act_val[i * 2 + 0]);
        packed_vec.elts[i].y = static_cast<GemmOutputType>(post_act_val[i * 2 + 1]);
    }

    // We need to offset into the scaling factors for just this expert
    auto act_sf_expert = act_sf_flat + getOffsetActivationSF(expert_id, num_tokens_before_expert, num_cols);

    // Use `token - num_tokens_before_expert` because we want this to be relative to the start of this expert
    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF,
        CVT_FP4_NUM_THREADS_PER_SF, NVFP4_VEC_SIZE>(std::nullopt /* batchIdx */, token_id - num_tokens_before_expert,
        elem_idx, std::nullopt /* numRows */, num_cols, act_sf_expert, FP4QuantizationSFLayout::SWIZZLED);

    // Do the conversion and set the output and scaling factor
    constexpr bool UE8M0 = false;
    auto res = cvt_warp_fp16_to_fp4<GemmOutputType, NVFP4_VEC_SIZE, UE8M0>(packed_vec, global_scale_val, sf_out);
    return res;
}

__device__ void writeSF(int64_t num_tokens_before_expert, int64_t expert_id, int64_t source_token_id, int64_t token_id,
    int64_t elem_idx, int64_t num_cols, int64_t max_tokens_per_expert,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* act_sf_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf)
{
    static constexpr int CVT_FP4_NUM_THREADS_PER_SF = NVFP4_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;

    // We need to offset into the scaling factors for just this expert
    auto act_sf_expert = act_sf_flat + getOffsetActivationSF(expert_id, num_tokens_before_expert, num_cols);

    // Use `token - num_tokens_before_expert` because we want this to be relative to the start of this expert
    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF,
        CVT_FP4_NUM_THREADS_PER_SF, NVFP4_VEC_SIZE>(std::nullopt /* batchIdx */, token_id - num_tokens_before_expert,
        elem_idx, std::nullopt /* numRows */, num_cols, act_sf_expert, FP4QuantizationSFLayout::SWIZZLED);
    if (sf_out)
    {
        auto const sf_in = cvt_quant_to_fp4_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF,
            CVT_FP4_NUM_THREADS_PER_SF, NVFP4_VEC_SIZE>(std::nullopt /* batchIdx */, source_token_id, elem_idx,
            std::nullopt /* numRows */, num_cols, const_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(input_sf),
            FP4QuantizationSFLayout::SWIZZLED);
        *sf_out = *sf_in;
    }
}

// ====================== Compute FP8 dequant scale only ===============================
__global__ void computeFP8DequantScaleKernel(
    float const** alpha_scale_ptr_array, int64_t const num_experts_per_node, float const* fp8_dequant)
{
    // First, compute the global tid. We only need 1 thread per expert.
    int const expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts_per_node)
    {
        return;
    }

    assert(fp8_dequant != nullptr);
    alpha_scale_ptr_array[expert] = fp8_dequant + expert;
}

float const** computeFP8DequantScale(
    float const** alpha_scale_ptr_array, int const num_experts_per_node, float const* fp8_dequant, cudaStream_t stream)
{
    if (!fp8_dequant)
    {
        return nullptr;
    }

    int const threads = std::min(1024, num_experts_per_node);
    int const blocks = (num_experts_per_node + threads - 1) / threads;

    computeFP8DequantScaleKernel<<<blocks, threads, 0, stream>>>(
        alpha_scale_ptr_array, num_experts_per_node, fp8_dequant);

    return alpha_scale_ptr_array;
}

template <class LayoutInfo>
__device__ void setupFP4BlockScalingFactors(LayoutInfo& layout_info, int expert, int gemm_m, int gemm_n, int gemm_k,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* weight_block_scale, int64_t num_tokens_before_expert)
{
    assert(layout_info.fp4_block_scaling_factors_stride_A);
    assert(layout_info.fp4_block_scaling_factors_stride_B);

    // tile_atom_to_shape_SFB & tile_atom_to_shape_SFA swapped for transpose
    layout_info.fp4_block_scaling_factors_stride_A[expert] = TmaWarpSpecializedGroupedGemmInput::tile_atom_to_shape_SFB(
        cute::make_shape((int) gemm_n, (int) gemm_m, (int) gemm_k, 1));
    layout_info.fp4_block_scaling_factors_stride_B[expert] = TmaWarpSpecializedGroupedGemmInput::tile_atom_to_shape_SFA(
        cute::make_shape((int) gemm_n, (int) gemm_m, (int) gemm_k, 1));

    // This assert validates our current assumption that A&B can be safely transposed without needing to modify
    assert(TmaWarpSpecializedGroupedGemmInput::tile_atom_to_shape_SFB(
               cute::make_shape((int) gemm_n, (int) gemm_m, (int) gemm_k, 1))
        == TmaWarpSpecializedGroupedGemmInput::tile_atom_to_shape_SFA(
            cute::make_shape((int) gemm_m, (int) gemm_n, (int) gemm_k, 1)));

    layout_info.fp4_block_scaling_factors_A[expert]
        = fp4_act_flat + getOffsetActivationSF(expert, num_tokens_before_expert, gemm_k);

    layout_info.fp4_block_scaling_factors_B[expert] = weight_block_scale + getOffsetWeightSF(expert, gemm_n, gemm_k);
}

__device__ void computeTmaWarpSpecializedInputStrides(
    TmaWarpSpecializedGroupedGemmInput layout_info, int gemm_m, int gemm_n, int gemm_k, int64_t out_idx)
{
    layout_info.stride_a[out_idx] = cutlass::make_cute_packed_stride(
        TmaWarpSpecializedGroupedGemmInput::StrideA{}, cute::make_shape(gemm_m, gemm_k, 1));
    layout_info.stride_b[out_idx] = cutlass::make_cute_packed_stride(
        TmaWarpSpecializedGroupedGemmInput::StrideB{}, cute::make_shape(gemm_n, gemm_k, 1));
    if (layout_info.stride_c)
    {
        assert(false && "CUTLASS does not support a 1xN bias");
        //        layout_info.stride_c[out_idx] = cute::make_stride(0, cute::Int<1>{}, 0);
        layout_info.stride_c[out_idx] = cutlass::make_cute_packed_stride(
            TmaWarpSpecializedGroupedGemmInput::StrideC{}, cute::make_shape(1, gemm_n, 1));
    }
    if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE)
    {
        layout_info.default_epilogue.stride_d[out_idx] = cutlass::make_cute_packed_stride(
            TmaWarpSpecializedGroupedGemmInput::DefaultEpilogue::StrideD{}, cute::make_shape(gemm_n, gemm_m, 1));
    }
    if (layout_info.int4_groupwise_params.enabled)
    {
        layout_info.int4_groupwise_params.stride_s_a[out_idx]
            = cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::StrideSFA{},
                cute::make_shape(gemm_n, gemm_k / 128, 1));
    }
}

template <class T, class WeightType, class OutputType, class ScaleBiasType>
__device__ void computeTmaWarpSpecializedInputPointers(TmaWarpSpecializedGroupedGemmInput layout_info, int64_t gemm_m,
    int64_t gemm_n, int64_t gemm_k, int num_tokens_before_expert, int64_t expert, T const* in,
    WeightType const* weights, TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA const* w4a8_weight_scale,
    ScaleBiasType const* bias, OutputType* output, int64_t const out_idx)
{
    // The input prior to this contains K elements per token, with `num_tokens_before_expert` tokens
    layout_info.ptr_a[out_idx] = safe_inc_ptr(in, num_tokens_before_expert * gemm_k);

    // Each expert's weight matrix is a constant size NxK, get the matrix at index `expert`
    layout_info.ptr_b[out_idx] = safe_inc_ptr(weights, expert * (gemm_n * gemm_k));

    if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE)
    {
        // The output prior to this contains N elements per token, with `num_tokens_before_expert` tokens
        layout_info.default_epilogue.ptr_d[out_idx] = safe_inc_ptr(output, num_tokens_before_expert * gemm_n);
    }
    if (layout_info.int4_groupwise_params.enabled)
    {
        layout_info.int4_groupwise_params.ptr_s_a[out_idx]
            = safe_inc_ptr(w4a8_weight_scale, expert * (gemm_n * gemm_k / 128));
    }
}

// TODO Some of this setup could be cached
template <class T, class WeightType, class OutputType, class ScaleBiasType>
__global__ void computeStridesTmaWarpSpecializedKernel(int64_t const* expert_first_token_offset,
    TmaWarpSpecializedGroupedGemmInput layout_info1, TmaWarpSpecializedGroupedGemmInput layout_info2,
    int64_t num_tokens, int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n, int64_t gemm2_k,
    int64_t const num_experts_per_node, T const* gemm1_in, T const* gemm2_in, WeightType const* weights1,
    WeightType const* weights2, float const* alpha_scale_flat1, float const* alpha_scale_flat2,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2, QuantParams quant_params,
    ScaleBiasType const* bias1, ScaleBiasType const* bias2, OutputType* gemm1_output, OutputType* gemm2_output)
{
    // First, compute the global tid. We only need 1 thread per expert.
    int const expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts_per_node)
    {
        return;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    // Both gemms use the same token offset
    auto const num_tokens_before_expert = expert_first_token_offset[expert];
    auto const num_tokens_including_expert = expert_first_token_offset[expert + 1];
    auto const num_tokens_to_expert = num_tokens_including_expert - num_tokens_before_expert;
    auto const gemm_m = num_tokens_to_expert;

    // M and N transposed since we are using the #tokens as the N dimension
    layout_info1.shape_info.problem_shapes[expert]
        = TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(gemm1_n, gemm_m, gemm1_k);
    layout_info2.shape_info.problem_shapes[expert]
        = TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(gemm2_n, gemm_m, gemm2_k);

    if (layout_info1.int4_groupwise_params.enabled)
    {
        layout_info1.int4_groupwise_params.shape.problem_shapes[expert]
            = TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::ProblemShapeInt::UnderlyingProblemShape(
                gemm1_n, gemm_m, gemm1_k);
    }

    if (layout_info2.int4_groupwise_params.enabled)
    {
        layout_info2.int4_groupwise_params.shape.problem_shapes[expert]
            = TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::ProblemShapeInt::UnderlyingProblemShape(
                gemm2_n, gemm_m, gemm2_k);
    }

    if (alpha_scale_flat1 && alpha_scale_flat2)
    {
        layout_info1.alpha_scale_ptr_array[expert] = alpha_scale_flat1 + expert;
        layout_info2.alpha_scale_ptr_array[expert] = alpha_scale_flat2 + expert;
    }

    if (quant_params.fp4.fc1.weight_block_scale)
    {
        setupFP4BlockScalingFactors(layout_info1, expert, gemm_m, gemm1_n, gemm1_k, fp4_act_flat1,
            quant_params.fp4.fc1.weight_block_scale, num_tokens_before_expert);
    }

    if (quant_params.fp4.fc2.weight_block_scale)
    {
        setupFP4BlockScalingFactors(layout_info2, expert, gemm_m, gemm2_n, gemm2_k, fp4_act_flat2,
            quant_params.fp4.fc2.weight_block_scale, num_tokens_before_expert);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    assert(gemm_m <= INT32_MAX);
    assert(gemm1_n > 0 && gemm1_n <= INT32_MAX);
    assert(gemm1_k > 0 && gemm1_k <= INT32_MAX);
    assert(gemm2_n > 0 && gemm2_n <= INT32_MAX);
    assert(gemm2_k > 0 && gemm2_k <= INT32_MAX);
    computeTmaWarpSpecializedInputStrides(layout_info1, gemm_m, gemm1_n, gemm1_k, expert);
    computeTmaWarpSpecializedInputStrides(layout_info2, gemm_m, gemm2_n, gemm2_k, expert);

    computeTmaWarpSpecializedInputPointers(layout_info1, gemm_m, gemm1_n, gemm1_k, num_tokens_before_expert, expert,
        gemm1_in, weights1,
        reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA const*>(
            quant_params.groupwise.fc1.weight_scales),
        bias1, gemm1_output, expert);
    computeTmaWarpSpecializedInputPointers(layout_info2, gemm_m, gemm2_n, gemm2_k, num_tokens_before_expert, expert,
        gemm2_in, weights2,
        reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA const*>(
            quant_params.groupwise.fc2.weight_scales),
        bias2, gemm2_output, expert);
}

template <class T, class WeightType, class OutputType, class ScaleBiasType>
__global__ void computeStridesTmaWarpSpecializedLowLatencyKernel(TmaWarpSpecializedGroupedGemmInput layout_info1,
    TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens, int64_t gemm1_n, int64_t gemm1_k,
    int64_t gemm2_n, int64_t gemm2_k, int64_t const num_experts_per_node, T const* in1, T const* in2,
    WeightType const* weights1, WeightType const* weights2, float const* alpha_scale_flat1,
    float const* alpha_scale_flat2, TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2, QuantParams quant_params,
    ScaleBiasType const* bias1, ScaleBiasType const* bias2, OutputType* output1, OutputType* output2,
    int const* num_active_experts_per, int const* active_expert_global_ids, int start_expert)
{
    // First, compute the global tid. We only need 1 thread per expert.
    int const expert = blockIdx.x * blockDim.x + threadIdx.x;

    if (expert >= num_experts_per_node)
    {
        return;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    // Note: expert is used to calculate the offset of the input and output
    // local_expert is used to calculate the offset of the weight
    auto const num_tokens_before_expert = expert * num_tokens;
    bool const is_active_expert = expert < *num_active_experts_per;
    int const local_expert = is_active_expert ? active_expert_global_ids[expert] - start_expert : -1;
    auto const gemm_m = is_active_expert ? num_tokens : 0;

    // M and N transposed since we are using the #tokens as the N dimension
    layout_info1.shape_info.problem_shapes[expert]
        = TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(gemm1_n, gemm_m, gemm1_k);
    layout_info2.shape_info.problem_shapes[expert]
        = TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(gemm2_n, gemm_m, gemm2_k);

    if (alpha_scale_flat1)
    {
        assert(alpha_scale_flat2);
        if (is_active_expert)
        {
            layout_info1.alpha_scale_ptr_array[expert] = alpha_scale_flat1 + local_expert;
            layout_info2.alpha_scale_ptr_array[expert] = alpha_scale_flat2 + local_expert;
        }
        else
        {
            layout_info1.alpha_scale_ptr_array[expert] = nullptr;
            layout_info2.alpha_scale_ptr_array[expert] = nullptr;
        }
    }

    if (quant_params.fp4.fc1.weight_block_scale)
    {
        setupFP4BlockScalingFactors(layout_info1, expert, gemm_m, gemm1_n, gemm1_k, fp4_act_flat1,
            quant_params.fp4.fc1.weight_block_scale, num_tokens_before_expert);

        // Override the scaling factors, fc1 uses the same A input for all experts and the scaling factor B offsets from
        // the local expert index
        if (is_active_expert)
        {
            layout_info1.fp4_block_scaling_factors_A[expert] = fp4_act_flat1;
            layout_info1.fp4_block_scaling_factors_B[expert]
                = quant_params.fp4.fc1.weight_block_scale + getOffsetWeightSF(local_expert, gemm1_n, gemm1_k);
        }
        else
        {
            layout_info1.fp4_block_scaling_factors_A[expert] = nullptr;
            layout_info1.fp4_block_scaling_factors_B[expert] = nullptr;
        }
    }

    if (quant_params.fp4.fc2.weight_block_scale)
    {
        setupFP4BlockScalingFactors(layout_info2, expert, gemm_m, gemm2_n, gemm2_k, fp4_act_flat2,
            quant_params.fp4.fc2.weight_block_scale, num_tokens_before_expert);

        // Override the scaling factors, fc2 scaling factor B offsets by the local expert index
        if (is_active_expert)
        {
            layout_info2.fp4_block_scaling_factors_B[expert]
                = quant_params.fp4.fc2.weight_block_scale + getOffsetWeightSF(local_expert, gemm2_n, gemm2_k);
        }
        else
        {
            layout_info2.fp4_block_scaling_factors_A[expert] = nullptr;
            layout_info2.fp4_block_scaling_factors_B[expert] = nullptr;
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif

    assert(gemm_m <= INT32_MAX);
    assert(gemm1_n > 0 && gemm1_n <= INT32_MAX);
    assert(gemm1_k > 0 && gemm1_k <= INT32_MAX);
    assert(gemm2_n > 0 && gemm2_n <= INT32_MAX);
    assert(gemm2_k > 0 && gemm2_k <= INT32_MAX);
    computeTmaWarpSpecializedInputStrides(layout_info1, gemm_m, gemm1_n, gemm1_k, expert);
    computeTmaWarpSpecializedInputStrides(layout_info2, gemm_m, gemm2_n, gemm2_k, expert);

    if (is_active_expert)
    {
        // Note: under low latency mode, we use the same input for all experts
        // so for gemm1, the inputs are the same,
        // for gemm2, we use the input generated by gemm1
        layout_info1.ptr_a[expert] = in1;
        layout_info2.ptr_a[expert] = safe_inc_ptr(in2, expert * num_tokens * gemm2_k);

        // Each expert's weight matrix is a constant size NxK, get the matrix at index `expert`
        layout_info1.ptr_b[expert] = safe_inc_ptr(weights1, local_expert * (gemm1_n * gemm2_k));
        layout_info2.ptr_b[expert] = safe_inc_ptr(weights2, local_expert * (gemm1_n * gemm2_k));

        assert(layout_info1.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE);
        layout_info1.default_epilogue.ptr_d[expert] = safe_inc_ptr(output1, expert * num_tokens * gemm1_n);

        if (layout_info2.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE)
        {
            // The output prior to this contains N elements per token, with `num_tokens` tokens
            layout_info2.default_epilogue.ptr_d[expert] = safe_inc_ptr(output2, expert * num_tokens * gemm2_n);
        }
    }
    else
    {
        layout_info1.ptr_a[expert] = nullptr;
        layout_info2.ptr_a[expert] = nullptr;
        layout_info1.ptr_b[expert] = nullptr;
        layout_info2.ptr_b[expert] = nullptr;

        layout_info1.default_epilogue.ptr_d[expert] = nullptr;
        if (layout_info2.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE)
        {
            layout_info2.default_epilogue.ptr_d[expert] = nullptr;
        }
    }
}

// ========================== Permutation things =======================================

template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input)
{
    using Type = typename U::Element;
    static_assert(T::kElements == U::kElements);
    U u;
#pragma unroll
    for (int i = 0; i < U::kElements; i++)
    {
        u[i] = static_cast<Type>(input[i]);
    }
    return u;
}

// Duplicated and permutes rows for MoE. In addition, reverse the permutation map to help with finalizing routing.

// "expanded_x_row" simply means that the number of values is num_rows x k. It is "expanded" since we will have to
// duplicate some rows in the input matrix to match the dimensions. Duplicates will always get routed to separate
// experts in the end.

// Note that the expanded_dest_row_to_expanded_source_row map referred to here has indices in the range (0,
// k*rows_in_input - 1). However, it is set up so that index 0, rows_in_input, 2*rows_in_input ... (k-1)*rows_in_input
// all map to row 0 in the original matrix. Thus, to know where to read in the source matrix, we simply take the modulus
// of the expanded index.

constexpr static int EXPAND_THREADS_PER_BLOCK = 256;

template <class InputActivationsType, class ExpandedActivationsType>
__global__ void expandInputRowsKernel(InputActivationsType const* unpermuted_input,
    ExpandedActivationsType* permuted_output, float const* unpermuted_scales, float* permuted_scales,
    int const* expanded_dest_row_to_expanded_source_row, int* expanded_source_row_to_expanded_dest_row,
    int64_t const num_rows, int64_t const cols, int64_t const k, float const* fc1_act_global_scale,
    int64_t const* expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, int64_t const num_experts_per_node)
{
#ifdef ENABLE_FP4
    constexpr bool is_fp4 = std::is_same_v<ExpandedActivationsType, __nv_fp4_e2m1>;
    constexpr bool is_fp4_input = is_fp4 && std::is_same_v<InputActivationsType, __nv_fp4_e2m1>;
    constexpr bool need_fp4_quant = is_fp4 && !std::is_same_v<InputActivationsType, __nv_fp4_e2m1>;
#else
    constexpr bool is_fp4 = false;
    constexpr bool is_fp4_input = false;
    constexpr bool need_fp4_quant = false;
#endif

    static_assert(need_fp4_quant || std::is_same_v<InputActivationsType, ExpandedActivationsType>,
        "Only FP4 quantization supports outputting a different format as part of the expansion");

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node];

    for (int64_t expanded_dest_row = blockIdx.x; expanded_dest_row < num_valid_tokens; expanded_dest_row += gridDim.x)
    {
        // Reverse permutation map.
        // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need
        // the reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in
        // MoE. 1 thread block will be responsible for all k summations.
        int64_t const expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
        if (threadIdx.x == 0)
        {
            assert(expanded_dest_row <= INT32_MAX);
            expanded_source_row_to_expanded_dest_row[expanded_source_row] = static_cast<int>(expanded_dest_row);
        }

        // Load 128-bits per thread
        constexpr int64_t ELEM_PER_THREAD
            = is_fp4 ? CVT_FP4_ELTS_PER_THREAD : (128 / sizeof_bits<InputActivationsType>::value);
        constexpr int64_t ELEM_PER_BYTE = is_fp4_input ? 2 : 1;
        using DataElem
            = std::conditional_t<is_fp4_input, uint32_t, cutlass::Array<InputActivationsType, ELEM_PER_THREAD>>;
        using OutputElem = std::conditional_t<is_fp4, uint32_t, DataElem>;

        // Duplicate and permute rows
        int64_t const source_k_rank = expanded_source_row / num_rows;
        int64_t const source_row = expanded_source_row % num_rows;

        auto const* source_row_ptr
            = reinterpret_cast<DataElem const*>(unpermuted_input + source_row * cols / ELEM_PER_BYTE);
        // Cast first to handle when this is FP4
        auto* dest_row_ptr
            = reinterpret_cast<OutputElem*>(permuted_output) + expanded_dest_row * cols / ELEM_PER_THREAD;

        int64_t const start_offset = threadIdx.x;
        int64_t const stride = EXPAND_THREADS_PER_BLOCK;
        int64_t const num_elems_in_col = cols / ELEM_PER_THREAD;
        assert(cols % ELEM_PER_THREAD == 0);

        if constexpr (is_fp4)
        {
            int64_t expert = findTotalEltsLessThanTarget(
                                 expert_first_token_offset, num_experts_per_node, (int64_t) expanded_dest_row + 1)
                - 1;
            float global_scale_val = fc1_act_global_scale ? *fc1_act_global_scale : 1.0f;
            int64_t num_tokens_before_expert = expert_first_token_offset[expert];

            for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
            {
                auto in_vec = source_row_ptr[elem_index];
                if constexpr (need_fp4_quant)
                {
                    auto res = quantizePackedFP4Value<InputActivationsType, DataElem>(in_vec, global_scale_val,
                        num_tokens_before_expert, expert, expanded_dest_row, elem_index, cols, num_rows,
                        fc1_act_sf_flat);
                    dest_row_ptr[elem_index] = res;
                }
                else
                {
                    writeSF(num_tokens_before_expert, expert, source_row, expanded_dest_row, elem_index, cols, num_rows,
                        fc1_act_sf_flat, input_sf);
                    dest_row_ptr[elem_index] = in_vec;
                }
            }
        }
        else
        {
            for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
            {
                dest_row_ptr[elem_index] = source_row_ptr[elem_index];
            }
        }

        if (permuted_scales && threadIdx.x == 0)
        {
            int64_t const source_k_idx = source_row * k + source_k_rank;
            permuted_scales[expanded_dest_row] = unpermuted_scales ? unpermuted_scales[source_k_idx] : 1.0f;
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <class InputActivationsType, class ExpandedActivationsType>
void expandInputRowsKernelLauncher(InputActivationsType const* unpermuted_input,
    ExpandedActivationsType* permuted_output, float const* unpermuted_scales, float* permuted_scales,
    int const* expanded_dest_row_to_expanded_source_row, int* expanded_source_row_to_expanded_dest_row,
    int64_t const num_rows, int64_t const cols, int const k, int const num_experts_per_node,
    float const* fc1_act_global_scale, int64_t* expert_first_token_offset,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, cudaStream_t stream)
{
    if (fc1_act_sf_flat)
    {
        size_t num_elems
            = getOffsetActivationSF(num_experts_per_node, num_rows * std::min(k, num_experts_per_node), cols);
        check_cuda_error(cudaMemsetAsync(
            fc1_act_sf_flat, 0x0, num_elems * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF), stream));
    }

    static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
    // Note: Launching 8 blocks per SM can fully leverage the memory bandwidth (tested on B200).
    int64_t const blocks = smCount * 8;
    int64_t const threads = EXPAND_THREADS_PER_BLOCK;
    auto func = expandInputRowsKernel<InputActivationsType, ExpandedActivationsType>;

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = threads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, func, unpermuted_input, permuted_output, unpermuted_scales, permuted_scales,
        expanded_dest_row_to_expanded_source_row, expanded_source_row_to_expanded_dest_row, num_rows, cols, k,
        fc1_act_global_scale, expert_first_token_offset, fc1_act_sf_flat, input_sf, num_experts_per_node);
}

enum class ScaleMode : int
{
    NO_SCALE = 0,
    DEFAULT = 1,
};

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
template <typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE>
__global__ void finalizeMoeRoutingKernel(GemmOutputType const* expanded_permuted_rows,
    OutputType* reduced_unpermuted_output, ScaleBiasType const* bias, float const* scales,
    int const* expanded_source_row_to_expanded_dest_row, int const* expert_for_source_row, int64_t const orig_cols,
    int64_t const experts_per_token, int const num_experts_per_node)
{
    assert(orig_cols % 4 == 0);
    int64_t const original_row = blockIdx.x;
    int64_t const num_rows = gridDim.x;
    auto const offset = original_row * orig_cols;
    OutputType* reduced_row_ptr = reduced_unpermuted_output + offset;

    // Load 128-bits per thread, according to the smallest data type we read/write
    constexpr int64_t FINALIZE_ELEM_PER_THREAD
        = 128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
    int64_t const num_elems_in_col = orig_cols / FINALIZE_ELEM_PER_THREAD;

    using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
    using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
    auto const* bias_v = reinterpret_cast<BiasElem const*>(bias);
    auto const* expanded_permuted_rows_v = reinterpret_cast<InputElem const*>(expanded_permuted_rows);
    auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll
    for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        ComputeElem thread_output;
        thread_output.fill(0);
        for (int k_idx = 0; k_idx < experts_per_token; ++k_idx)
        {
            int64_t const k_offset = original_row * experts_per_token + k_idx;
            int64_t const expert_idx = expert_for_source_row[k_offset];
            if (expert_idx >= num_experts_per_node)
            {
                continue;
            }

            int64_t const expanded_original_row = original_row + k_idx * num_rows;
            int64_t const expanded_permuted_row = expanded_source_row_to_expanded_dest_row[expanded_original_row];

            float const row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];

            auto const* expanded_permuted_rows_row_ptr
                = expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_col;

            ComputeElem expert_result
                = arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);

            if (bias)
            {
                auto const* bias_ptr = bias_v + expert_idx * num_elems_in_col;
                expert_result = expert_result + arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
            }

            thread_output = thread_output + row_scale * expert_result;
        }

        OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
        reduced_row_ptr_v[elem_index] = output_elem;
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
template <typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE>
__global__ void finalizeMoeRoutingNoFillingKernel(GemmOutputType const* expanded_permuted_rows,
    OutputType* reduced_unpermuted_output, ScaleBiasType const* bias, float const* scales,
    int const* const expanded_source_row_to_expanded_dest_row, int const* expanded_dest_row_to_expanded_source_row,
    int const* expert_for_source_row, int64_t const* expert_first_token_offset, int64_t const num_rows,
    int64_t const orig_cols, int64_t const experts_per_token, int const num_experts_per_node)
{
    assert(orig_cols % 4 == 0);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node];
    for (int64_t expanded_permuted_row = blockIdx.x; expanded_permuted_row < num_valid_tokens;
         expanded_permuted_row += gridDim.x)
    {
        int64_t expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_permuted_row];

        // Duplicate and permute rows
        int64_t const source_k_rank = expanded_source_row / num_rows;
        int64_t const source_row = expanded_source_row % num_rows;

        // If the expert is the first selected (valid) one of the corresponding token on the current EP rank, do
        // reduction; otherwise, skip.
        bool is_first_selected_expert = true;
        for (int k_idx = 0; k_idx < source_k_rank; ++k_idx)
        {
            if (expert_for_source_row[source_row * experts_per_token + k_idx] < num_experts_per_node)
            {
                is_first_selected_expert = false;
                break;
            }
        }
        if (!is_first_selected_expert)
        {
            continue;
        }

        OutputType* reduced_row_ptr = reduced_unpermuted_output + source_row * orig_cols;

        // Load 128-bits per thread, according to the smallest data type we read/write
        constexpr int64_t FINALIZE_ELEM_PER_THREAD
            = 128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);

        int64_t const start_offset = threadIdx.x;
        int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
        int64_t const num_elems_in_col = orig_cols / FINALIZE_ELEM_PER_THREAD;

        using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
        using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
        using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
        using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
        auto const* bias_v = reinterpret_cast<BiasElem const*>(bias);
        auto const* expanded_permuted_rows_v = reinterpret_cast<InputElem const*>(expanded_permuted_rows);
        auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

        for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
        {
            ComputeElem thread_output;
            thread_output.fill(0);
            for (int k_idx = 0; k_idx < experts_per_token; ++k_idx)
            {
                int64_t const k_offset = source_row * experts_per_token + k_idx;
                int64_t const expert_idx = expert_for_source_row[k_offset];
                if (expert_idx >= num_experts_per_node)
                {
                    continue;
                }

                int64_t const expanded_permuted_row_from_k_idx
                    = expanded_source_row_to_expanded_dest_row[source_row + k_idx * num_rows];

                float const row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];

                auto const* expanded_permuted_rows_row_ptr
                    = expanded_permuted_rows_v + expanded_permuted_row_from_k_idx * num_elems_in_col;

                ComputeElem expert_result
                    = arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);

                if (bias)
                {
                    auto const* bias_ptr = bias_v + expert_idx * num_elems_in_col;
                    expert_result = expert_result + arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
                }

                thread_output = thread_output + row_scale * expert_result;
            }
            OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
            reduced_row_ptr_v[elem_index] = output_elem;
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <class OutputType, class GemmOutputType, class ScaleBiasType>
void finalizeMoeRoutingKernelLauncher(GemmOutputType const* expanded_permuted_rows,
    OutputType* reduced_unpermuted_output, ScaleBiasType const* bias, float const* final_scales,
    int const* expanded_source_row_to_expanded_dest_row, int const* expanded_dest_row_to_expanded_source_row,
    int const* expert_for_source_row, int64_t const* expert_first_token_offset, int64_t const num_rows,
    int64_t const cols, int64_t const experts_per_token, int const num_experts_per_node,
    MOEParallelismConfig parallelism_config, bool const enable_alltoall, cudaStream_t stream)
{
    // Only add bias on rank 0 for tensor parallelism
    bool const is_rank_0 = parallelism_config.tp_rank == 0;
    ScaleBiasType const* bias_ptr = is_rank_0 ? bias : nullptr;

    cudaLaunchConfig_t config;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;

    if (parallelism_config.ep_size > 1 && enable_alltoall)
    {
        // If all-to-all comm is enabled, finalizeMoeRouting doesn't need to fill the invalid output tokens with zeros.
        static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
        // Note: Launching 8 blocks per SM can fully leverage the memory bandwidth (tested on B200).
        int64_t const blocks = smCount * 8;
        int64_t const threads = FINALIZE_THREADS_PER_BLOCK;
        config.gridDim = blocks;
        config.blockDim = threads;
        auto func = final_scales
            ? &finalizeMoeRoutingNoFillingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT>
            : &finalizeMoeRoutingNoFillingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::NO_SCALE>;
        cudaLaunchKernelEx(&config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr, final_scales,
            expanded_source_row_to_expanded_dest_row, expanded_dest_row_to_expanded_source_row, expert_for_source_row,
            expert_first_token_offset, num_rows, cols, experts_per_token, num_experts_per_node);
    }
    else
    {
        // If all-gather reduce-scatter is used, finalizeMoeRouting must fill invalid output tokens with zeros.
        int64_t const blocks = num_rows;
        int64_t const threads = FINALIZE_THREADS_PER_BLOCK;
        config.gridDim = blocks;
        config.blockDim = threads;
        auto func = final_scales
            ? &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT>
            : &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::NO_SCALE>;
        cudaLaunchKernelEx(&config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr, final_scales,
            expanded_source_row_to_expanded_dest_row, expert_for_source_row, cols, experts_per_token,
            num_experts_per_node);
    }
}

// ============================== Gated Activation =================================
constexpr static int ACTIVATION_THREADS_PER_BLOCK = 256;

template <class ActivationOutputType, class GemmOutputType, template <class> class ActFn>
__global__ void doGatedActivationKernel(ActivationOutputType* output, GemmOutputType const* gemm_result,
    int64_t const* num_valid_tokens_ptr, int64_t inter_size)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
    {
        return;
    }

    output = output + token * inter_size;
    gemm_result = gemm_result + token * inter_size * 2;

    constexpr int64_t ACTIVATION_ELEM_PER_THREAD = 128 / sizeof_bits<ActivationOutputType>::value;

    using OutputElem = cutlass::Array<ActivationOutputType, ACTIVATION_ELEM_PER_THREAD>;
    using GemmResultElem = cutlass::Array<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
    auto gemm_result_vec = reinterpret_cast<GemmResultElem const*>(gemm_result);
    auto output_vec = reinterpret_cast<OutputElem*>(output);
    int64_t const start_offset = tid;
    int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
    assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
    int64_t const inter_size_vec = inter_size / ACTIVATION_ELEM_PER_THREAD;

    ActFn<ComputeElem> fn{};
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto fc1_value = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index]);
        // BF16 isn't supported, use FP32 for activation function
        auto gate_value = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index + inter_size_vec]);
        auto gate_act = fn(gate_value);
        output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(fc1_value * gate_act);
    }
}

template <typename ActivationOutputType, typename GemmOutputType>
void doGatedActivation(ActivationOutputType* output, GemmOutputType const* gemm_result,
    int64_t const* num_valid_tokens_ptr, int64_t inter_size, int64_t num_tokens, ActivationType activation_type,
    cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = ACTIVATION_THREADS_PER_BLOCK;

    auto* fn = activation_type == ActivationType::Swiglu
        ? &doGatedActivationKernel<ActivationOutputType, GemmOutputType, cutlass::epilogue::thread::SiLu>
        : &doGatedActivationKernel<ActivationOutputType, GemmOutputType, cutlass::epilogue::thread::GELU>;
    fn<<<blocks, threads, 0, stream>>>(output, gemm_result, num_valid_tokens_ptr, inter_size);
}

// ============================== Activation =================================

template <class T, class GemmOutputType, class ScaleBiasType, template <class> class ActFn>
__global__ void doActivationKernel(T* output, GemmOutputType const* gemm_result, float const* fp8_quant,
    ScaleBiasType const* bias_ptr, bool bias_is_broadcast, int64_t const* expert_first_token_offset,
    int num_experts_per_node, int64_t inter_size, int64_t max_tokens_per_expert, bool gated,
    float const* fc2_act_global_scale, TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_act_sf_flat)
{
#ifdef ENABLE_FP4
    constexpr bool IsFP4 = std::is_same_v<T, __nv_fp4_e2m1>;
#else
    constexpr bool IsFP4 = cute::dependent_false<T>;
#endif

    int64_t const tid = threadIdx.x;
    size_t const gated_size_mul = gated ? 2 : 1;
    size_t const gated_off = gated ? inter_size : 0;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node];

    for (int64_t token = blockIdx.x; token < num_valid_tokens; token += gridDim.x)
    {
        size_t gemm_result_offset = token * inter_size * gated_size_mul;
        size_t output_offset = token * inter_size;

        int64_t expert = 0;
        if (bias_ptr || IsFP4)
        {
            // TODO this is almost certainly faster as a linear scan
            expert = findTotalEltsLessThanTarget(expert_first_token_offset, num_experts_per_node, token + 1) - 1;
        }

        float const quant_scale = fp8_quant ? *fp8_quant : 1.f;

        // Some globals for FP4
        float global_scale_val = fc2_act_global_scale ? *fc2_act_global_scale : 1.0f;
        int64_t num_tokens_before_expert = IsFP4 ? expert_first_token_offset[expert] : 0;

        size_t bias_offset = 0;
        if (bias_ptr)
        {
            bias_offset = (bias_is_broadcast ? expert * inter_size * gated_size_mul : gemm_result_offset);
        }

        // Load 128-bits per thread, according to the smallest data type we read/write
        constexpr int64_t ACTIVATION_ELEM_PER_THREAD = IsFP4
            ? CVT_FP4_ELTS_PER_THREAD
            : (128 / std::min(sizeof_bits<T>::value, sizeof_bits<GemmOutputType>::value));

        using BiasElem = cutlass::Array<ScaleBiasType, ACTIVATION_ELEM_PER_THREAD>;
        using GemmResultElem = cutlass::Array<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
        using OutputElem = std::conditional_t<IsFP4, uint32_t, cutlass::Array<T, ACTIVATION_ELEM_PER_THREAD>>;
        using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
        // Aliases gemm_result for non-gated, non-fp8 cases
        auto gemm_result_vec = reinterpret_cast<GemmResultElem const*>(gemm_result + gemm_result_offset);
        auto output_vec = reinterpret_cast<OutputElem*>(safe_inc_ptr(output, output_offset));
        auto bias_ptr_vec = reinterpret_cast<BiasElem const*>(bias_ptr + bias_offset);
        int64_t const start_offset = tid;
        int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
        assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
        int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
        assert(gated_off % ACTIVATION_ELEM_PER_THREAD == 0);
        int64_t const gated_off_vec = gated_off / ACTIVATION_ELEM_PER_THREAD;

        ActFn<ComputeElem> fn{};
        for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
        {
            auto fc1_value = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index + gated_off_vec]);
            if (bias_ptr)
            {
                fc1_value = fc1_value + arrayConvert<BiasElem, ComputeElem>(bias_ptr_vec[elem_index + gated_off_vec]);
            }

            auto gate_act = fn(fc1_value);

            if (gated)
            {
                auto gate_mul = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index]);
                if (bias_ptr_vec)
                {
                    gate_mul = gate_mul + arrayConvert<BiasElem, ComputeElem>(bias_ptr_vec[elem_index]);
                }
                gate_act = gate_act * gate_mul;
            }

            auto post_act_val = gate_act * quant_scale;

            if constexpr (IsFP4)
            {
                // We use GemmOutputType as the intermediate compute type as that should always be unquantized
                auto res = quantizePackedFP4Value<GemmOutputType, ComputeElem>(post_act_val, global_scale_val,
                    num_tokens_before_expert, expert, token, elem_index, inter_size, max_tokens_per_expert,
                    fc2_act_sf_flat);
                output_vec[elem_index] = res;
            }
            else
            {
                output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(post_act_val);
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <class T, class GemmOutputType, class ScaleBiasType>
void doActivation(T* output, GemmOutputType const* gemm_result, float const* fp8_quant, ScaleBiasType const* bias,
    bool bias_is_broadcast, int64_t const* expert_first_token_offset, int num_experts_per_node, int64_t inter_size,
    int64_t num_tokens, int64_t expanded_num_tokens, ActivationType activation_type, float const* fc2_act_global_scale,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_act_sf_flat, cudaStream_t stream)
{
    static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
    // Note: Launching 8 blocks per SM can fully leverage the memory bandwidth (tested on B200).
    int64_t const blocks = smCount * 8;
    int64_t const threads = ACTIVATION_THREADS_PER_BLOCK;

    auto fn_list = std::array{
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::GELU>,    // Gelu
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::ReLu>,    // Relu
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::SiLu>,    // Silu
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::SiLu>,    // Swiglu
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::GELU>,    // Geglu
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::Identity> // Identity
    };
    auto fn = fn_list[static_cast<int>(activation_type)];

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = threads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, fn, output, gemm_result, fp8_quant, bias, bias_is_broadcast, expert_first_token_offset,
        num_experts_per_node, inter_size, num_tokens, isGatedActivation(activation_type), fc2_act_global_scale,
        fc2_act_sf_flat);
}

// ============================== Lora Add Bias =================================
constexpr static int LORA_KERNELS_THREADS_PER_BLOCK = 256;

template <class ScaleBiasType, class LoraType, bool IsGated>
__global__ void loraAddBiasKernel(ScaleBiasType* output, LoraType const* lora_result, ScaleBiasType const* bias,
    int64_t const* num_valid_tokens_ptr, int* permuted_token_selected_experts, int64_t inter_size)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    int64_t const num_tokens = gridDim.x;
    if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
    {
        return;
    }

    LoraType const* lora_result_1 = lora_result + token * inter_size;
    int expert_id = permuted_token_selected_experts[token];
    if constexpr (IsGated)
    {
        output = output + token * inter_size * 2;
        bias = bias + expert_id * inter_size * 2;
    }
    else
    {
        output = output + token * inter_size;
        bias = bias + expert_id * inter_size;
    }

    constexpr int64_t LORA_ADD_BIAS_ELEM_PER_THREAD = 128 / sizeof_bits<LoraType>::value;

    using DataElem = cutlass::Array<LoraType, LORA_ADD_BIAS_ELEM_PER_THREAD>;
    using BiasElem = cutlass::Array<ScaleBiasType, LORA_ADD_BIAS_ELEM_PER_THREAD>;
    auto lora_result_1_vec = reinterpret_cast<DataElem const*>(lora_result_1);
    auto bias_vec = reinterpret_cast<BiasElem const*>(bias);
    auto output_vec = reinterpret_cast<BiasElem*>(output);

    int64_t const start_offset = tid;
    int64_t const stride = LORA_KERNELS_THREADS_PER_BLOCK;
    assert(inter_size % LORA_ADD_BIAS_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / LORA_ADD_BIAS_ELEM_PER_THREAD;

    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto lora_value = lora_result_1_vec[elem_index];
        auto bias_value = bias_vec[elem_index];
        output_vec[elem_index] = bias_value + arrayConvert<DataElem, BiasElem>(lora_value);
    }

    if constexpr (IsGated)
    {
        auto lora_result_2_vec = reinterpret_cast<DataElem const*>(lora_result_1 + num_tokens * inter_size);
        int64_t const inter_size_vec = inter_size / LORA_ADD_BIAS_ELEM_PER_THREAD;
        for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
        {
            auto lora_value = lora_result_2_vec[elem_index];
            auto bias_value = bias_vec[elem_index + inter_size_vec];
            output_vec[elem_index + inter_size_vec] = bias_value + arrayConvert<DataElem, BiasElem>(lora_value);
        }
    }
}

template <class ScaleBiasType, class LoraType>
void loraAddBias(ScaleBiasType* output, LoraType const* lora_result, ScaleBiasType const* bias,
    int64_t const* num_valid_tokens_ptr, int64_t inter_size, int* permuted_token_selected_experts, int64_t num_tokens,
    bool is_gated_activation, cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = LORA_KERNELS_THREADS_PER_BLOCK;

    auto selected_fn = is_gated_activation ? loraAddBiasKernel<ScaleBiasType, LoraType, true>
                                           : loraAddBiasKernel<ScaleBiasType, LoraType, false>;
    selected_fn<<<blocks, threads, 0, stream>>>(
        output, lora_result, bias, num_valid_tokens_ptr, permuted_token_selected_experts, inter_size);
}

template <class T>
__global__ void loraReorderKernel(
    T* output, T const* lora_result, int64_t const* num_valid_tokens_ptr, int64_t inter_size)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    int64_t const num_tokens = gridDim.x;
    if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
    {
        return;
    }

    T const* lora_result_1 = lora_result + token * inter_size;
    output = output + token * inter_size * 2;

    constexpr int64_t LORA_REORDER_ELEM_PER_THREAD = 128 / sizeof_bits<T>::value;

    using DataElem = cutlass::Array<T, LORA_REORDER_ELEM_PER_THREAD>;
    auto lora_result_1_vec = reinterpret_cast<DataElem const*>(lora_result_1);
    auto output_vec = reinterpret_cast<DataElem*>(output);

    int64_t const start_offset = tid;
    int64_t const stride = LORA_KERNELS_THREADS_PER_BLOCK;
    assert(inter_size % LORA_REORDER_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / LORA_REORDER_ELEM_PER_THREAD;

    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto lora_value = lora_result_1_vec[elem_index];
        output_vec[elem_index] = lora_value;
    }

    auto lora_result_2_vec = reinterpret_cast<DataElem const*>(lora_result_1 + num_tokens * inter_size);
    int64_t const inter_size_vec = inter_size / LORA_REORDER_ELEM_PER_THREAD;
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto lora_value = lora_result_2_vec[elem_index];
        output_vec[elem_index + inter_size_vec] = lora_value;
    }
}

template <class T>
void loraReorder(T* output, T const* lora_result, int64_t const* num_valid_tokens_ptr, int64_t inter_size,
    int64_t num_tokens, cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = LORA_KERNELS_THREADS_PER_BLOCK;

    loraReorderKernel<T><<<blocks, threads, 0, stream>>>(output, lora_result, num_valid_tokens_ptr, inter_size);
}

// ============================== DEQUANT_FP8 =================================
constexpr static int DEQUANT_KERNELS_THREADS_PER_BLOCK = 256;

template <class OutputType, class InputType>
__global__ void dequantFP8Kernel(OutputType* output, InputType const* input, int64_t const* num_valid_tokens_ptr,
    int64_t inter_size, float const* scale, bool scale_is_dequant)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
    {
        return;
    }

    output = output + token * inter_size;
    input = input + token * inter_size;

    constexpr int64_t DEQUANT_ELEM_PER_THREAD = 128 / sizeof_bits<InputType>::value;

    using DataElem = cutlass::Array<InputType, DEQUANT_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<OutputType, DEQUANT_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, DEQUANT_ELEM_PER_THREAD>;
    auto input_vec = reinterpret_cast<DataElem const*>(input);
    auto output_vec = reinterpret_cast<OutputElem*>(output);

    int64_t const start_offset = tid;
    int64_t const stride = DEQUANT_KERNELS_THREADS_PER_BLOCK;
    assert(inter_size % DEQUANT_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / DEQUANT_ELEM_PER_THREAD;

    ComputeElem deqaunt_scale_value;
    float dequant_scale = scale[0];
    if (!scale_is_dequant)
    {
        dequant_scale = 1.f / dequant_scale;
    }
    deqaunt_scale_value.fill(dequant_scale);

    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto input_value = arrayConvert<DataElem, ComputeElem>(input_vec[elem_index]);
        output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(input_value * deqaunt_scale_value);
    }
}

template <class OutputType, class InputType>
void dequantFP8(OutputType* output, InputType const* input, int64_t const* num_valid_tokens_ptr, int64_t inter_size,
    int64_t num_tokens, float const* scale, bool scale_is_dequant, cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = DEQUANT_KERNELS_THREADS_PER_BLOCK;

    dequantFP8Kernel<OutputType, InputType>
        <<<blocks, threads, 0, stream>>>(output, input, num_valid_tokens_ptr, inter_size, scale, scale_is_dequant);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::CutlassMoeFCRunner()
    : blockscale_gemm_runner_{std::make_unique<
        kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>>()}
{
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
std::map<std::string, std::pair<size_t, size_t>>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::getWorkspaceDeviceBufferSizes(
    int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node,
    int const experts_per_token, ActivationType activation_type, bool use_lora, bool use_fp8_block_scaling,
    bool min_latency_mode, bool use_awq)
{
    size_t num_moe_inputs = min_latency_mode ? num_experts_per_node * num_rows : experts_per_token * num_rows;
    size_t const permuted_elems = num_moe_inputs * hidden_size;
    size_t const interbuf_elems = num_moe_inputs * inter_size;
    size_t glu_inter_elems = 0;
    bool is_gated_activation = isGatedActivation(activation_type);
    if (is_gated_activation)
    {
        glu_inter_elems = interbuf_elems * 2;
    }
    else if (mayHaveDifferentGEMMOutputType())
    {
        // In this case we are using activation quantization, and some intermediate buffers will be unquantized
        // We need to have separate memory for these as we can no longer alias the output buffer for reuse
        glu_inter_elems = interbuf_elems;
    }

    bool using_tma_ws = moe_gemm_runner_.supportsTmaWarpSpecialized();

    size_t const gemm_output_dtype = sizeof(UnfusedGemmOutputType);

    constexpr float dtype_size = use_fp4 ? 0.5f : (use_w4afp8 ? 2.0f : sizeof(T));

    size_t const unpermuted_token_selected_experts_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
    size_t const unpermuted_source_token_ids_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
    size_t const permuted_source_token_ids_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
    size_t const permuted_token_selected_experts_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
    size_t const permuted_data_size = permuted_elems * dtype_size;
    size_t const expert_first_token_offset_size = (num_experts_per_node + 1) * sizeof(int64_t);
    size_t const permuted_token_final_scales_size = mayHaveFinalizeFused() ? num_moe_inputs * sizeof(float) : 0;
    size_t const glu_inter_size = glu_inter_elems * gemm_output_dtype; // May be an intermediate type for quantization
    size_t const fc1_result_size = interbuf_elems * dtype_size;        // Activation quantizes so back to dtype_size
    size_t const sorter_ws_size
        = min_latency_mode ? 0 : CubKeyValueSorter::getWorkspaceSize(num_rows, num_experts_per_node);
    size_t const fc2_result_size = min_latency_mode
        ? 0
        : num_moe_inputs * hidden_size * gemm_output_dtype; // May be an intermediate type for quantization

    auto act_sf_rows = min_latency_mode
        ? num_moe_inputs
        : std::min(num_moe_inputs, static_cast<size_t>(num_rows * num_experts_per_node));
    size_t const fc1_fp4_act_scale_size = use_fp4
        ? getOffsetActivationSF(num_experts_per_node, act_sf_rows, hidden_size)
            * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF)
        : 0;
    size_t const fc2_fp4_act_scale_size = use_fp4 ? getOffsetActivationSF(num_experts_per_node, act_sf_rows, inter_size)
            * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF)
                                                  : 0;
    size_t const fp4_act_scale_size = std::max(fc1_fp4_act_scale_size, fc2_fp4_act_scale_size);

    size_t const tma_ws_size
        = using_tma_ws ? TmaWarpSpecializedGroupedGemmInput::workspaceSize(num_experts_per_node) : 0;

    size_t const gemm_workspace_size = moe_gemm_runner_.getMaxWorkspaceSize(num_experts_per_node);

    // lora related
    size_t const lora_input_size
        = (use_lora && use_fp8) ? std::max(permuted_elems, interbuf_elems) * sizeof(ScaleBiasType) : 0;
    size_t const lora_fc1_result_size = use_lora
        ? (is_gated_activation ? 2 * interbuf_elems * sizeof(ScaleBiasType) : interbuf_elems * sizeof(ScaleBiasType))
        : 0;
    size_t const lora_add_bias_size = use_lora ? lora_fc1_result_size : 0;
    size_t const lora_fc2_result_size = use_lora ? permuted_elems * sizeof(ScaleBiasType) : 0;

    // We do some overlapping of the large workspace buffers. Although we could overlap some of the other buffers, they
    // are small enough (i.e no factor of hidden size) they will only be a couple MiB at most, so we don't bother
    // in the case of fused activation we overlap permuted_data and fc2_result
    // in the case of unfused activation we overlap permuted_data and fc1_result
    // we need to calculate the max possible size, so use the max of all three
    size_t overlapped_gemm1_gemm2_inputs_size = std::max(permuted_data_size, fc2_result_size);
    // When glu_inter_elems is 0 we are always fused, otherwise we may need the un-fused case
    if (glu_inter_elems > 0)
    {
        overlapped_gemm1_gemm2_inputs_size = std::max(overlapped_gemm1_gemm2_inputs_size, fc1_result_size);
    }

    size_t const alpha_scale_ptr_array_size = num_experts_per_node * sizeof(float*);

    // if we have glu_inter we overlap it with fc2_result, otherwise we use fc1_result by itself
    size_t overlapped_gemm1_gemm2_outputs_size = fc1_result_size;
    if (glu_inter_elems > 0)
    {
        overlapped_gemm1_gemm2_outputs_size
            = std::max(std::max(glu_inter_size, fc2_result_size), overlapped_gemm1_gemm2_outputs_size);
    }

    size_t smoothed_act_size = use_awq ? std::max(permuted_elems, interbuf_elems) * sizeof(T) * 2
                                       : 0; // Extra workspace required by AWQ for smoothing activations
    size_t deepseek_fc_workspace_size = 0;
    if (use_fp8_block_scaling)
    {
        size_t factor = is_gated_activation ? 2 : 1;
        size_t blockscale_fc1_output_size = factor * interbuf_elems * gemm_output_dtype;
        size_t blockscale_fc2_output_size = permuted_elems * gemm_output_dtype;
        overlapped_gemm1_gemm2_inputs_size
            = std::max(std::max(permuted_data_size, fc1_result_size), blockscale_fc2_output_size);
        overlapped_gemm1_gemm2_outputs_size = blockscale_fc1_output_size;

        auto* blockscale_gemm_runner = getBlockScaleGemmRunner();
        TLLM_CHECK(blockscale_gemm_runner != nullptr);
        auto deepseek_fc1_workspace_size = blockscale_gemm_runner->getWorkspaceSize(
            num_rows, factor * inter_size, hidden_size, experts_per_token, num_experts_per_node);
        auto deepseek_fc2_workspace_size = blockscale_gemm_runner->getWorkspaceSize(
            num_rows, hidden_size, inter_size, experts_per_token, num_experts_per_node);
        deepseek_fc_workspace_size = std::max(deepseek_fc1_workspace_size, deepseek_fc2_workspace_size);
    }

    size_t map_offset = 0;
    std::map<std::string, std::pair<size_t, size_t>> out_map;

#define ADD_NAME(name, size)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        auto aligned_size = tensorrt_llm::common::alignSize(size, tensorrt_llm::common::kCudaMemAlign);                \
        out_map[#name] = std::pair{aligned_size, map_offset};                                                          \
        map_offset += aligned_size;                                                                                    \
    } while (false)
#define ADD(name) ADD_NAME(name, name##_size)

    ADD(unpermuted_source_token_ids);
    ADD(unpermuted_token_selected_experts);
    ADD(permuted_source_token_ids);
    ADD(permuted_token_selected_experts);
    ADD(expert_first_token_offset);
    ADD(permuted_token_final_scales);
    ADD(sorter_ws);
    ADD(overlapped_gemm1_gemm2_inputs);
    ADD(overlapped_gemm1_gemm2_outputs);
    ADD_NAME(alpha_scale_ptr_array_fc1, alpha_scale_ptr_array_size);
    ADD_NAME(alpha_scale_ptr_array_fc2, alpha_scale_ptr_array_size);
    ADD(fp4_act_scale);
    ADD_NAME(tma_ws_gemm1_workspace, tma_ws_size);
    ADD_NAME(tma_ws_gemm2_workspace, tma_ws_size);
    ADD(gemm_workspace);
    ADD(lora_input);
    ADD(lora_fc1_result);
    ADD(lora_add_bias);
    ADD(lora_fc2_result);
    ADD(deepseek_fc_workspace);
    ADD(smoothed_act);

    return out_map;

#undef ADD_NAME
#undef ADD
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
size_t CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::getWorkspaceSize(
    int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size, int const num_experts,
    int const experts_per_token, ActivationType activation_type, MOEParallelismConfig parallelism_config, bool use_lora,
    bool use_fp8_block_scaling, bool min_latency_mode, bool use_awq)
{
    int const ep_size = parallelism_config.ep_size;
    TLLM_CHECK_WITH_INFO(num_experts % ep_size == 0, "Number of experts must be a multiple of ep size");
    auto sizes_map = getWorkspaceDeviceBufferSizes(num_rows, hidden_size, inter_size, num_experts / ep_size,
        experts_per_token, activation_type, use_lora, use_fp8_block_scaling, min_latency_mode, use_awq);
    std::vector<size_t> sizes(sizes_map.size());
    std::transform(sizes_map.begin(), sizes_map.end(), sizes.begin(), [](auto& v) { return v.second.first; });
    size_t size = tensorrt_llm::common::calculateTotalWorkspaceSize(sizes.data(), sizes.size());
    TLLM_LOG_TRACE("Mixture Of Experts Plugin requires workspace of %2f MiB", size / 1024.f / 1024.f);
    return size;
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::configureWsPtrs(char* ws_ptr,
    int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node,
    int const experts_per_token, ActivationType activation_type, MOEParallelismConfig parallelism_config, bool use_lora,
    bool use_fp8_block_scaling, bool min_latency_mode, bool use_awq)
{
    auto workspaces = getWorkspaceDeviceBufferSizes(num_rows, hidden_size, inter_size, num_experts_per_node,
        experts_per_token, activation_type, use_lora, use_fp8_block_scaling, min_latency_mode, use_awq);

    auto getWsPtr = [&](auto type, std::string const& name)
    {
        return workspaces.at(name).first ? reinterpret_cast<decltype(type)*>(ws_ptr + workspaces.at(name).second)
                                         : nullptr;
    };

    unpermuted_source_token_ids_ = getWsPtr(int{}, "unpermuted_source_token_ids");
    unpermuted_token_selected_experts_ = getWsPtr(int{}, "unpermuted_token_selected_experts");
    permuted_source_token_ids_ = getWsPtr(int{}, "permuted_source_token_ids");
    permuted_token_selected_experts_ = getWsPtr(int{}, "permuted_token_selected_experts");

    expert_first_token_offset_ = getWsPtr(int64_t{}, "expert_first_token_offset");

    // We check if the provided config uses fused finalize and disable it if it does not
    bool const gemm2_using_tma_ws = moe_gemm_runner_.isTmaWarpSpecialized(*gemm2_config_);
    permuted_token_final_scales_
        = (gemm2_using_tma_ws && mayHaveFinalizeFused()) ? getWsPtr(float{}, "permuted_token_final_scales") : nullptr;

    sorter_ws_ = getWsPtr(char{}, "sorter_ws");

    bool const is_gated_activation = isGatedActivation(activation_type);
    bool const gemm1_using_fused_moe
        = moe_gemm_runner_.isFusedGatedActivation(*gemm1_config_, is_gated_activation, inter_size, hidden_size);
    bool const gemm1_using_tma_ws = moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_);
    bool const tma_ws_has_glu = gemm1_using_tma_ws && (mayHaveDifferentGEMMOutputType() || is_gated_activation);
    // We always use fused path if we can
    bool const non_tma_ws_has_glu = !gemm1_using_fused_moe && is_gated_activation;
    bool const has_glu_inter_result = tma_ws_has_glu || non_tma_ws_has_glu || use_fp8;

    // Always same value, but overlapped with either fc1_result_ or fc2_result_
    permuted_data_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
    // Always same value, ignored if not needed
    glu_inter_result_ = has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs") : nullptr;

    // fc1 and fc2 alias one of the above pointers, but it depends on if actfn is fused/unfused which is overlapped
    // NOTE: It is important to get the overlapped pointers correct as the wrong order will cause the buffer to be used
    // as an input and output for the same gemm, which will cause corruption
    fc1_result_ = has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs")
                                       : getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs");
    fc2_result_ = has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs")
                                       : getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");

    if (use_fp8_block_scaling)
    {
        permuted_data_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
        fc1_result_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
        glu_inter_result_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs");
        fc2_result_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
    }

    alpha_scale_ptr_array_fc1_ = getWsPtr((float const*) (nullptr), "alpha_scale_ptr_array_fc1");
    alpha_scale_ptr_array_fc2_ = getWsPtr((float const*) (nullptr), "alpha_scale_ptr_array_fc2");

    // NOTE: We alias these, but if we fuse the quantization for GEMM2 into GEMM1 they will need separated
    fc1_fp4_act_scale_ = nullptr;
    fc2_fp4_act_scale_ = nullptr;
    if (use_fp4)
    {
        fc1_fp4_act_scale_ = getWsPtr(TmaWarpSpecializedGroupedGemmInput::ElementSF{}, "fp4_act_scale");
        fc2_fp4_act_scale_ = getWsPtr(TmaWarpSpecializedGroupedGemmInput::ElementSF{}, "fp4_act_scale");
        TLLM_CHECK(fc1_fp4_act_scale_ != nullptr);
        TLLM_CHECK(fc2_fp4_act_scale_ != nullptr);
    }

    tma_ws_grouped_gemm1_input_ = {};
    tma_ws_grouped_gemm2_input_ = {};
    if (moe_gemm_runner_.supportsTmaWarpSpecialized())
    {
        tma_ws_grouped_gemm1_input_.configureWorkspace(getWsPtr(int8_t{}, "tma_ws_gemm1_workspace"),
            num_experts_per_node, getWsPtr(int8_t{}, "gemm_workspace"), workspaces.at("gemm_workspace").first);
        tma_ws_grouped_gemm2_input_.configureWorkspace(getWsPtr(int8_t{}, "tma_ws_gemm2_workspace"),
            num_experts_per_node, getWsPtr(int8_t{}, "gemm_workspace"), workspaces.at("gemm_workspace").first);
    }

    lora_fc1_result_ = {};
    lora_add_bias_ = {};
    lora_fc2_result_ = {};

    if (use_lora)
    {
        lora_input_ = getWsPtr(ScaleBiasType{}, "lora_input");
        lora_fc1_result_ = getWsPtr(ScaleBiasType{}, "lora_fc1_result");
        lora_add_bias_ = getWsPtr(ScaleBiasType{}, "lora_add_bias");
        lora_fc2_result_ = getWsPtr(ScaleBiasType{}, "lora_fc2_result");
        TLLM_CHECK_WITH_INFO(!use_fp8 || lora_input_ != nullptr, "LoRA input must not be nullptr if FP8 is enabled");
        TLLM_CHECK(lora_fc1_result_ != nullptr);
        TLLM_CHECK(lora_add_bias_ != nullptr);
        TLLM_CHECK(lora_fc2_result_ != nullptr);
    }

    if (use_fp8_block_scaling)
    {
        auto* blockscale_gemm_runner = getBlockScaleGemmRunner();
        TLLM_CHECK(blockscale_gemm_runner != nullptr);
        blockscale_gemm_runner->configureWorkspace(getWsPtr(char{}, "deepseek_fc_workspace"));
    }

    if (use_awq)
    {
        smoothed_act_ = getWsPtr(int8_t{}, "smoothed_act");
    }
}

void generateTokenPermutation(int const* unpermuted_token_selected_experts, int const* unpermuted_source_token_ids,
    int* permuted_token_selected_experts, int* permuted_source_token_ids, int64_t* expert_first_token_offset,
    int64_t num_rows, int64_t num_experts_per_node, int64_t k, CubKeyValueSorter& sorter, void* sorter_ws,
    cudaStream_t stream)
{
    int64_t const expanded_num_rows = k * num_rows;
    sorter.updateNumExperts(num_experts_per_node);
    size_t const sorter_ws_size_bytes
        = pad_to_multiple_of_16(sorter.getWorkspaceSize(expanded_num_rows, num_experts_per_node));
    sorter.run((void*) sorter_ws, sorter_ws_size_bytes, unpermuted_token_selected_experts,
        permuted_token_selected_experts, unpermuted_source_token_ids, permuted_source_token_ids, expanded_num_rows,
        stream);

    sync_check_cuda_error(stream);

    // Upper bound on number of expanded rows
    computeExpertFirstTokenOffset(
        permuted_token_selected_experts, expanded_num_rows, num_experts_per_node, expert_first_token_offset, stream);
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunnerInterface*
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::getBlockScaleGemmRunner() const
{
    TLLM_CHECK_WITH_INFO((std::is_same_v<T, __nv_bfloat16> && std::is_same_v<OutputType, __nv_bfloat16>),
        "Block scale GEMM runner only supports BF16 A/output");
    TLLM_CHECK_WITH_INFO(
        (std::is_same_v<WeightType, __nv_fp8_e4m3>), "Block scale GEMM runner only supports FP8 weights.");
    return blockscale_gemm_runner_.get();
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::BlockScaleFC1(
    BlockScaleGemmRunner& gemm_runner, T const* const input, T* const output, void* const gemm_output,
    int64_t const* const expert_first_token_offset, WeightType const* const fc1_expert_weights,
    ScaleBiasType const* const fc1_expert_biases, float const* const fc2_fp8_quant, int64_t const num_rows,
    int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const num_experts_per_node, ActivationType fc1_activation_type, QuantParams& quant_params, cudaStream_t stream)
{
    bool const is_gated_activation = isGatedActivation(fc1_activation_type);

    int shape_n = is_gated_activation ? inter_size * 2 : inter_size;
    int shape_k = hidden_size;

    // NOTE: we assume gemm_runner.configureWorkspace has already been called.
    gemm_runner.moeGemm(gemm_output, input, fc1_expert_weights, expert_first_token_offset, num_experts_per_node,
        shape_n, shape_k, stream, nullptr, quant_params.fp8_block_scaling.fc1_scales_ptrs);

    sync_check_cuda_error(stream);
    constexpr bool bias_is_broadcast = true;
    doActivation<T, UnfusedGemmOutputType>(output, static_cast<UnfusedGemmOutputType const*>(gemm_output),
        fc2_fp8_quant, fc1_expert_biases, bias_is_broadcast, expert_first_token_offset, num_experts_per_node,
        inter_size, num_rows, expanded_num_rows, fc1_activation_type, nullptr, nullptr, stream);

    sync_check_cuda_error(stream);
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::BlockScaleFC2(
    BlockScaleGemmRunner& gemm_runner, T const* const input, void* const gemm_output, OutputType* const final_output,
    int64_t const* const expert_first_token_offset, WeightType const* const fc2_expert_weights,
    ScaleBiasType const* const fc2_expert_biases, float const* const unpermuted_final_scales,
    int const* const expanded_source_row_to_expanded_dest_row,
    int const* const expanded_dest_row_to_expanded_source_row, int const* const expert_for_source_row,
    int64_t const* const num_valid_tokens_ptr, int64_t const num_rows, int64_t const expanded_num_rows,
    int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node, int64_t const k,
    MOEParallelismConfig parallelism_config, bool const enable_alltoall, QuantParams& quant_params, cudaStream_t stream)
{
    int shape_n = hidden_size;
    int shape_k = inter_size;

    // NOTE: we assume gemm_runner.configureWorkspace has already been called.
    gemm_runner.moeGemm(gemm_output, input, fc2_expert_weights, expert_first_token_offset, num_experts_per_node,
        shape_n, shape_k, stream, nullptr, quant_params.fp8_block_scaling.fc2_scales_ptrs);

    sync_check_cuda_error(stream);

    finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(
        static_cast<UnfusedGemmOutputType const*>(gemm_output), final_output, fc2_expert_biases,
        unpermuted_final_scales, expanded_source_row_to_expanded_dest_row, expanded_dest_row_to_expanded_source_row,
        expert_for_source_row, expert_first_token_offset, num_rows, hidden_size, k, num_experts_per_node,
        parallelism_config, enable_alltoall, stream);
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
T const* CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::applyPrequantScale(
    void* smoothed_act, void const* permuted_data, void const* prequant_scales, int64_t const* num_valid_tokens_ptr,
    int64_t const expanded_num_rows, int64_t const seq_len, bool const use_awq, cudaStream_t stream)
{
    T const* gemm_input;
    bool use_prequant_scale_kernel = use_awq && !std::is_same_v<T, WeightType>;
    if (use_prequant_scale_kernel)
    {
        TLLM_CHECK_WITH_INFO(
            (!std::is_same_v<T, WeightType>), "Prequant scales are only used for different weight/activation type!");
        if constexpr (!std::is_same_v<T, WeightType>)
        {
            tensorrt_llm::kernels::apply_per_channel_scale_kernel_launcher<UnfusedGemmOutputType, T>(
                reinterpret_cast<T*>(smoothed_act), reinterpret_cast<UnfusedGemmOutputType const*>(permuted_data),
                reinterpret_cast<UnfusedGemmOutputType const*>(prequant_scales), expanded_num_rows, seq_len,
                num_valid_tokens_ptr, stream);
        }
        gemm_input = reinterpret_cast<T const*>(smoothed_act);
    }
    else
    {
        gemm_input = reinterpret_cast<T const*>(permuted_data);
    }
    sync_check_cuda_error(stream);
    return gemm_input;
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::gemm1(
    MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner,
    BlockScaleGemmRunner* fp8_blockscale_gemm_runner, T const* const input, T* const output,
    void* const intermediate_result, int64_t const* const expert_first_token_offset,
    TmaWarpSpecializedGroupedGemmInput const tma_ws_input_template, WeightType const* const fc1_expert_weights,
    ScaleBiasType const* const fc1_expert_biases, int64_t const* const num_valid_tokens_ptr,
    ScaleBiasType const* const fc1_int_scales, float const* const fc1_fp8_dequant, float const* const fc2_fp8_quant,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_flat, QuantParams quant_params, int64_t const num_rows,
    int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const num_experts_per_node, ActivationType fc1_activation_type, float const** alpha_scale_ptr_array,
    bool bias_is_broadcast, cudaStream_t stream, cutlass_extensions::CutlassGemmConfig config, bool min_latency_mode,
    int* num_active_experts_per, int* active_expert_global_ids, int start_expert)
{

    if (fp8_blockscale_gemm_runner)
    {
        TLLM_CHECK(!min_latency_mode);
        Self::BlockScaleFC1(*fp8_blockscale_gemm_runner, input, output, intermediate_result, expert_first_token_offset,
            fc1_expert_weights, fc1_expert_biases, fc2_fp8_quant, num_rows, expanded_num_rows, hidden_size, inter_size,
            num_experts_per_node, fc1_activation_type, quant_params, stream);
        return;
    }

    bool const using_tma_ws_gemm1 = gemm_runner.isTmaWarpSpecialized(config);
    bool const is_gated_activation = isGatedActivation(fc1_activation_type);
    bool const use_ampere_activation_fusion
        = gemm_runner.isFusedGatedActivation(config, is_gated_activation, inter_size, hidden_size);
    size_t const fc1_out_size = ((!use_ampere_activation_fusion) && is_gated_activation) ? inter_size * 2 : inter_size;

    int64_t const* total_tokens_including_expert = expert_first_token_offset + 1;

    if (min_latency_mode)
    {
        TLLM_CHECK_WITH_INFO(using_tma_ws_gemm1, "Only TMA warp specialized GEMM is supported in min latency mode.");
        // TODO: as for bias, need to get the correct expert id according to the active expert global ids
        TLLM_CHECK_WITH_INFO(fc1_expert_biases == nullptr, "Min latency mode does not support bias.");
        TLLM_CHECK(use_fp4);
    }

    if (using_tma_ws_gemm1)
    {
        TLLM_CHECK(config.is_tma_warp_specialized);
        TLLM_CHECK(!use_ampere_activation_fusion);

        TLLM_CHECK(!use_fp4 || fc1_fp4_act_flat);
        TLLM_CHECK(!use_fp4 || fc2_fp4_act_flat);

        bool has_different_gemm_output_type = using_tma_ws_gemm1 && !std::is_same_v<T, OutputType>;
        bool const has_intermediate = has_different_gemm_output_type || is_gated_activation;
        TLLM_CHECK_WITH_INFO(has_intermediate || input != output, "Input and output buffers are overlapping");
        auto* gemm_output = has_intermediate ? intermediate_result : static_cast<void*>(output);

        auto tma_ws_input = tma_ws_input_template;

        if (use_w4afp8)
        {
            alpha_scale_ptr_array = computeFP8DequantScale(
                alpha_scale_ptr_array, num_experts_per_node, quant_params.groupwise.fc1.alpha, stream);
        }

        auto universal_input
            = GroupedGemmInput<T, WeightType, OutputType, OutputType>{input, total_tokens_including_expert,
                /*weights*/ nullptr, /*scales*/ nullptr, /*zeros*/ nullptr, /*biases*/ nullptr, /*C*/ nullptr,
                alpha_scale_ptr_array, /*occupancy*/ nullptr, fc1_activation_type, num_rows,
                /*N*/ int64_t(fc1_out_size),
                /*K*/ hidden_size, num_experts_per_node, quant_params.groupwise.group_size, /*bias_is_broadcast*/ true,
                /*use_fused_moe*/ false, stream, config};
        gemm_runner.moeGemm(universal_input, tma_ws_input);

        sync_check_cuda_error(stream);

        if (fc2_fp4_act_flat)
        {
            auto act_sf_rows
                = min_latency_mode ? expanded_num_rows : std::min(expanded_num_rows, num_rows * num_experts_per_node);
            size_t num_elems = getOffsetActivationSF(num_experts_per_node, act_sf_rows, inter_size);
            check_cuda_error(cudaMemsetAsync(
                fc2_fp4_act_flat, 0x0, num_elems * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF), stream));
        }

        // TODO: when bias_is_broadcast is false, fuse bias to gemm
        using GatedActOutputType = std::conditional_t<use_w4afp8, BackBoneType, T>;
        doActivation<GatedActOutputType, UnfusedGemmOutputType>(reinterpret_cast<GatedActOutputType*>(output),
            static_cast<UnfusedGemmOutputType const*>(gemm_output), fc2_fp8_quant, fc1_expert_biases, bias_is_broadcast,
            expert_first_token_offset, num_experts_per_node, inter_size, num_rows, expanded_num_rows,
            fc1_activation_type, quant_params.fp4.fc2.act_global_scale, fc2_fp4_act_flat, stream);

        sync_check_cuda_error(stream);
    }
    else if (use_fp8)
    {
        TLLM_CHECK(!use_ampere_activation_fusion);
        TLLM_CHECK(!config.is_tma_warp_specialized);
        TLLM_CHECK(!use_fp4);

        alpha_scale_ptr_array
            = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node, quant_params.fp8.dequant_fc1, stream);

        auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{input,
            total_tokens_including_expert, fc1_expert_weights, /*scales*/ nullptr, /*zeros*/ nullptr,
            /*biases*/ nullptr, reinterpret_cast<UnfusedGemmOutputType*>(intermediate_result), alpha_scale_ptr_array,
            /*occupancy*/ nullptr, fc1_activation_type, expanded_num_rows, /*N*/ int64_t(fc1_out_size),
            /*K*/ hidden_size, num_experts_per_node, quant_params.groupwise.group_size, /*bias_is_broadcast*/ true,
            /*use_fused_moe*/ false, stream, config};
        gemm_runner.moeGemm(universal_input, TmaWarpSpecializedGroupedGemmInput{});

        doActivation<T, UnfusedGemmOutputType>(output, static_cast<UnfusedGemmOutputType const*>(intermediate_result),
            fc2_fp8_quant, fc1_expert_biases, bias_is_broadcast, expert_first_token_offset, num_experts_per_node,
            inter_size, num_rows, expanded_num_rows, fc1_activation_type, nullptr, nullptr, stream);

        sync_check_cuda_error(stream);
    }
    else if (!is_gated_activation)
    {
        TLLM_CHECK(!use_ampere_activation_fusion);
        TLLM_CHECK(!config.is_tma_warp_specialized);
        TLLM_CHECK(!use_fp4);
        if (use_w4afp8)
        {
            alpha_scale_ptr_array = computeFP8DequantScale(
                alpha_scale_ptr_array, num_experts_per_node, quant_params.groupwise.fc1.alpha, stream);
        }
        auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{input,
            total_tokens_including_expert, fc1_expert_weights,
            /*scales*/ quant_params.groupwise.group_size > 0
                ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_scales)
                : fc1_int_scales,
            /*zeros*/ quant_params.groupwise.group_size > 0
                ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_zeros)
                : nullptr,
            fc1_expert_biases, reinterpret_cast<OutputType*>(output), alpha_scale_ptr_array, /*occupancy*/ nullptr,
            fc1_activation_type, expanded_num_rows, /*N*/ int64_t(fc1_out_size),
            /*K*/ hidden_size, num_experts_per_node, quant_params.groupwise.group_size, bias_is_broadcast,
            /*use_fused_moe*/ false, stream, config};
        gemm_runner.moeGemmBiasAct(universal_input, TmaWarpSpecializedGroupedGemmInput{});

        sync_check_cuda_error(stream);
    }
    else
    {
        TLLM_CHECK(!config.is_tma_warp_specialized);
        TLLM_CHECK(is_gated_activation);
        TLLM_CHECK_WITH_INFO(
            !use_ampere_activation_fusion || input != output, "Input and output buffers are overlapping");
        TLLM_CHECK(!use_fp4);
        if (use_w4afp8)
        {
            alpha_scale_ptr_array = computeFP8DequantScale(
                alpha_scale_ptr_array, num_experts_per_node, quant_params.groupwise.fc1.alpha, stream);
        }
        // Run the GEMM with activation function overridden with `Identity`, we do the activation separately
        auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{input,
            total_tokens_including_expert, fc1_expert_weights,
            /*scales*/ quant_params.groupwise.group_size > 0
                ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_scales)
                : fc1_int_scales,
            /*zeros*/ quant_params.groupwise.group_size > 0
                ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_zeros)
                : nullptr,
            fc1_expert_biases, static_cast<OutputType*>(use_ampere_activation_fusion ? output : intermediate_result),
            alpha_scale_ptr_array, /*occupancy*/ nullptr,
            use_ampere_activation_fusion ? fc1_activation_type : ActivationType::Identity, expanded_num_rows,
            /*N*/ int64_t(fc1_out_size),
            /*K*/ hidden_size, num_experts_per_node, quant_params.groupwise.group_size, bias_is_broadcast,
            use_ampere_activation_fusion, stream, config};
        gemm_runner.moeGemmBiasAct(universal_input, TmaWarpSpecializedGroupedGemmInput{});

        sync_check_cuda_error(stream);

        if (!use_ampere_activation_fusion)
        {
            using GatedActOutputType = std::conditional_t<use_w4afp8, BackBoneType, T>;
            doGatedActivation<GatedActOutputType, UnfusedGemmOutputType>(reinterpret_cast<GatedActOutputType*>(output),
                static_cast<UnfusedGemmOutputType const*>(intermediate_result), num_valid_tokens_ptr, inter_size,
                expanded_num_rows, fc1_activation_type, stream);

            sync_check_cuda_error(stream);
        }
    }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::gemm2(
    MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner,
    BlockScaleGemmRunner* fp8_blockscale_gemm_runner, T const* const input, void* const gemm_output,
    OutputType* const final_output, int64_t const* const expert_first_token_offset,
    TmaWarpSpecializedGroupedGemmInput const tma_ws_input_template, WeightType const* const fc2_expert_weights,
    ScaleBiasType const* const fc2_expert_biases, ScaleBiasType const* const fc2_int_scales,
    float const* const fc2_fp8_dequant, TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat,
    QuantParams quant_params, float const* const unpermuted_final_scales, float const* const permuted_final_scales,
    int const* const expanded_source_row_to_expanded_dest_row, int const* expanded_dest_row_to_expanded_source_row,
    int const* const expert_for_source_row, int64_t const* const num_valid_tokens_ptr, int64_t const num_rows,
    int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const num_experts_per_node, int64_t const k, float const** alpha_scale_ptr_array, bool use_lora, void* fc2_lora,
    cudaStream_t stream, MOEParallelismConfig parallelism_config, bool const enable_alltoall,
    cutlass_extensions::CutlassGemmConfig config, bool min_latency_mode, int* num_active_experts_per,
    int* active_expert_global_ids, int start_expert)
{
    int64_t const* total_tokens_including_expert = expert_first_token_offset + 1;

    bool const using_tma_ws_gemm2 = gemm_runner.isTmaWarpSpecialized(config);

    if (min_latency_mode)
    {
        TLLM_CHECK_WITH_INFO(using_tma_ws_gemm2, "Only TMA warp specialized GEMM is supported in min latency mode.");
        TLLM_CHECK(use_fp4);
    }

    if (fp8_blockscale_gemm_runner)
    {
        Self::BlockScaleFC2(*fp8_blockscale_gemm_runner, input, gemm_output, final_output, expert_first_token_offset,
            fc2_expert_weights, fc2_expert_biases, unpermuted_final_scales, expanded_source_row_to_expanded_dest_row,
            expanded_dest_row_to_expanded_source_row, expert_for_source_row, num_valid_tokens_ptr, num_rows,
            expanded_num_rows, hidden_size, inter_size, num_experts_per_node, k, parallelism_config, enable_alltoall,
            quant_params, stream);
        return;
    }

    TmaWarpSpecializedGroupedGemmInput tma_ws_input{};
    if (using_tma_ws_gemm2)
    {
        tma_ws_input = tma_ws_input_template;
        if (tma_ws_input.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE)
        {
            // TODO For some reason this has to be done here, it should not overlap with anything else, but
            // doing it in setupTmaWarpSpecializedInputs gives a different result. Ideally, we want this to run on a
            // second stream and overlap with everything else
            //
            // This also means it is included in the timing for the profiler, which is probably more representative
            // until we can overlap it
            check_cuda_error(cudaMemsetAsync(final_output, 0x0, sizeof(OutputType) * num_rows * hidden_size, stream));
        }
    }
    else if (use_fp8)
    {
        alpha_scale_ptr_array
            = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node, fc2_fp8_dequant, stream);
    }
    if (use_w4afp8)
    {
        alpha_scale_ptr_array = computeFP8DequantScale(
            alpha_scale_ptr_array, num_experts_per_node, quant_params.groupwise.fc2.alpha, stream);
    }

    bool fuse_lora_bias = use_lora && !(use_fp8 || using_tma_ws_gemm2);
    // Note: expanded_num_rows, to check this value, it's greater than num_rows * num_experts_per_node
    auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{input, total_tokens_including_expert,
        fc2_expert_weights,
        quant_params.groupwise.group_size > 0
            ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc2.weight_scales)
            : fc2_int_scales,
        quant_params.groupwise.group_size > 0
            ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc2.weight_zeros)
            : nullptr,
        fuse_lora_bias ? static_cast<ScaleBiasType*>(fc2_lora) : nullptr, static_cast<OutputType*>(gemm_output),
        alpha_scale_ptr_array, /*occupancy*/ nullptr, ActivationType::Identity, expanded_num_rows,
        /*N*/ hidden_size,
        /*K*/ inter_size, num_experts_per_node, quant_params.groupwise.group_size, /*bias_is_broadcast*/ false,
        /*use_fused_moe*/ false, stream, config};
    gemm_runner.moeGemmBiasAct(universal_input, tma_ws_input);
    sync_check_cuda_error(stream);

    if (min_latency_mode)
        return;

    if (use_lora && !fuse_lora_bias)
    {
        auto loraBiasApplyFunc = doActivation<UnfusedGemmOutputType, UnfusedGemmOutputType, ScaleBiasType>;
        loraBiasApplyFunc(static_cast<UnfusedGemmOutputType*>(gemm_output),
            static_cast<UnfusedGemmOutputType const*>(gemm_output), nullptr,
            static_cast<ScaleBiasType const*>(fc2_lora), false, expert_first_token_offset, num_experts_per_node,
            hidden_size, num_rows, expanded_num_rows, ActivationType::Identity, nullptr, nullptr, stream);
        sync_check_cuda_error(stream);
    }

    bool has_different_output_type_ampere = (use_w4afp8 || use_fp8) && !using_tma_ws_gemm2;
    bool using_hopper_fused_finalize
        = tma_ws_input.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
    bool has_different_output_type_tma_ws = !using_hopper_fused_finalize && using_tma_ws_gemm2;

    if (has_different_output_type_ampere || has_different_output_type_tma_ws)
    {
        finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(
            static_cast<UnfusedGemmOutputType const*>(gemm_output), final_output, fc2_expert_biases,
            unpermuted_final_scales, expanded_source_row_to_expanded_dest_row, expanded_dest_row_to_expanded_source_row,
            expert_for_source_row, expert_first_token_offset, num_rows, hidden_size, k, num_experts_per_node,
            parallelism_config, enable_alltoall, stream);
    }
    else if (!using_tma_ws_gemm2)
    {
        finalizeMoeRoutingKernelLauncher<OutputType, T>(static_cast<T const*>(gemm_output), final_output,
            fc2_expert_biases, unpermuted_final_scales, expanded_source_row_to_expanded_dest_row,
            expanded_dest_row_to_expanded_source_row, expert_for_source_row, expert_first_token_offset, num_rows,
            hidden_size, k, num_experts_per_node, parallelism_config, enable_alltoall, stream);
    }
    sync_check_cuda_error(stream);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
bool CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::setupLoraWorkspace(
    int64_t expanded_num_rows, int64_t num_rows, int64_t inter_size, int64_t hidden_size, int start_expert,
    bool is_gated_activation, int num_experts_per_node, bool needs_num_valid, LoraParams& lora_params,
    cudaStream_t stream)
{
    std::vector<int>& host_permuted_rows = host_lora_workspace_.host_permuted_rows;
    std::vector<void const*>& host_permuted_fc1_weight_ptrs = host_lora_workspace_.host_permuted_fc1_weight_ptrs;
    std::vector<void const*>& host_permuted_fc2_weight_ptrs = host_lora_workspace_.host_permuted_fc2_weight_ptrs;
    std::vector<void const*>& host_permuted_gated_weight_ptrs = host_lora_workspace_.host_permuted_gated_weight_ptrs;

    std::vector<int32_t>& host_permuted_fc1_lora_ranks = host_lora_workspace_.host_permuted_fc1_lora_ranks;
    std::vector<int32_t>& host_permuted_fc2_lora_ranks = host_lora_workspace_.host_permuted_fc2_lora_ranks;
    std::vector<int32_t>& host_permuted_gated_lora_ranks = host_lora_workspace_.host_permuted_gated_lora_ranks;
    std::vector<int64_t>& host_expert_first_token_offset = host_lora_workspace_.host_expert_first_token_offset;

    bool all_token_without_lora = true;

    host_permuted_fc1_weight_ptrs.resize(expanded_num_rows * 2);
    host_permuted_fc1_lora_ranks.resize(expanded_num_rows);
    host_permuted_fc2_weight_ptrs.resize(expanded_num_rows * 2);
    host_permuted_fc2_lora_ranks.resize(expanded_num_rows);

    if (is_gated_activation)
    {
        host_permuted_gated_weight_ptrs.resize(expanded_num_rows * 2);
        host_permuted_gated_lora_ranks.resize(expanded_num_rows);
    }

    TLLM_CUDA_CHECK(cudaEventSynchronize(*(lora_params.memcpy_event_ptr)));

    size_t num_valid_tokens
        = needs_num_valid ? host_expert_first_token_offset[num_experts_per_node] : expanded_num_rows;

    for (int expert_idx = 0; expert_idx < num_experts_per_node; ++expert_idx)
    {
        int weight_index = expert_idx + start_expert;
        for (size_t i = host_expert_first_token_offset[expert_idx]; i < host_expert_first_token_offset[expert_idx + 1];
             ++i)
        {
            int source_index = host_permuted_rows[i] % num_rows;
            int32_t lora_rank = lora_params.fc1_lora_ranks[source_index];
            host_permuted_fc1_weight_ptrs[i * 2]
                = reinterpret_cast<ScaleBiasType const*>(lora_params.fc1_lora_weight_ptrs[source_index * 2])
                + weight_index * hidden_size * lora_rank;
            host_permuted_fc1_weight_ptrs[i * 2 + 1]
                = reinterpret_cast<ScaleBiasType const*>(lora_params.fc1_lora_weight_ptrs[source_index * 2 + 1])
                + weight_index * lora_rank * inter_size;
            host_permuted_fc1_lora_ranks[i] = lora_rank;

            lora_rank = lora_params.fc2_lora_ranks[source_index];
            host_permuted_fc2_weight_ptrs[i * 2]
                = reinterpret_cast<ScaleBiasType const*>(lora_params.fc2_lora_weight_ptrs[source_index * 2])
                + weight_index * inter_size * lora_rank;
            host_permuted_fc2_weight_ptrs[i * 2 + 1]
                = reinterpret_cast<ScaleBiasType const*>(lora_params.fc2_lora_weight_ptrs[source_index * 2 + 1])
                + weight_index * lora_rank * hidden_size;
            host_permuted_fc2_lora_ranks[i] = lora_rank;

            if (host_permuted_fc1_lora_ranks[i] || host_permuted_fc2_lora_ranks[i])
            {
                all_token_without_lora = false;
            }

            if (is_gated_activation)
            {
                lora_rank = lora_params.gated_lora_ranks[source_index];
                host_permuted_gated_weight_ptrs[i * 2]
                    = reinterpret_cast<ScaleBiasType const*>(lora_params.gated_lora_weight_ptrs[source_index * 2])
                    + weight_index * hidden_size * lora_rank;
                host_permuted_gated_weight_ptrs[i * 2 + 1]
                    = reinterpret_cast<ScaleBiasType const*>(lora_params.gated_lora_weight_ptrs[source_index * 2 + 1])
                    + weight_index * lora_rank * inter_size;
                host_permuted_gated_lora_ranks[i] = lora_rank;

                if (host_permuted_gated_lora_ranks[i])
                {
                    all_token_without_lora = false;
                }
            }
        }
    }
    return all_token_without_lora;
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
auto CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::loraFC1(int64_t expanded_num_rows,
    int64_t inter_size, int64_t hidden_size, int num_experts_per_node, int start_expert,
    int64_t const* num_valid_tokens_ptr, bool is_gated_activation, ScaleBiasType const* fc1_expert_biases,
    LoraParams& lora_params, float const* input_fp8_dequant, cudaStream_t stream) -> ScaleBiasType const*
{
    TLLM_CHECK_WITH_INFO(!use_fp4, "LoRA does not support FP4 activations");
    std::vector<void const*>& host_permuted_fc1_weight_ptrs = host_lora_workspace_.host_permuted_fc1_weight_ptrs;
    std::vector<void const*>& host_permuted_gated_weight_ptrs = host_lora_workspace_.host_permuted_gated_weight_ptrs;

    std::vector<int32_t>& host_permuted_fc1_lora_ranks = host_lora_workspace_.host_permuted_fc1_lora_ranks;
    std::vector<int32_t>& host_permuted_gated_lora_ranks = host_lora_workspace_.host_permuted_gated_lora_ranks;
    std::vector<int64_t>& host_expert_first_token_offset = host_lora_workspace_.host_expert_first_token_offset;

    auto fc1_lora_impl = lora_params.fc1_lora_impl;
    int num_reqs = lora_params.num_reqs;

    ScaleBiasType *lora_gated_out = nullptr, *lora_fc1_result = nullptr;

    if (is_gated_activation)
    {
        lora_gated_out = lora_fc1_result_;
        lora_fc1_result = lora_fc1_result_ + expanded_num_rows * inter_size;
    }
    else
    {
        lora_fc1_result = lora_fc1_result_;
    }

    ScaleBiasType* input{};
    if constexpr (use_fp8)
    {
        TLLM_CHECK(lora_input_);
        bool const scale_is_dequant = true;
        dequantFP8<ScaleBiasType, T>(lora_input_, permuted_data_, num_valid_tokens_ptr, hidden_size, expanded_num_rows,
            input_fp8_dequant, scale_is_dequant, stream);
        sync_check_cuda_error(stream);
        input = lora_input_;
    }
    else if constexpr (!use_fp4)
    {
        TLLM_CHECK(!lora_input_);
        input = reinterpret_cast<ScaleBiasType*>(permuted_data_);
    }

    void* lora_workspace = lora_params.workspace;
    void* tmp_lora_fc_result = static_cast<void*>(lora_fc1_result);
    int64_t num_valid_tokens = host_expert_first_token_offset[num_experts_per_node];
    int64_t num_reqs_lora = std::min(num_valid_tokens, static_cast<int64_t>(num_reqs * num_experts_per_node));

    ::tensorrt_llm::kernels::Lora_run(fc1_lora_impl.get(), num_valid_tokens, num_reqs_lora, input,
        host_permuted_fc1_lora_ranks.data(), host_permuted_fc1_weight_ptrs.data(), 0, &tmp_lora_fc_result,
        lora_workspace, stream);

    if (is_gated_activation)
    {
        void* tmp_lora_gated_result = static_cast<void*>(lora_gated_out);
        ::tensorrt_llm::kernels::Lora_run(fc1_lora_impl.get(), num_valid_tokens, num_reqs_lora, input,
            host_permuted_gated_lora_ranks.data(), host_permuted_gated_weight_ptrs.data(), 0, &tmp_lora_gated_result,
            lora_workspace, stream);
    }

    // add bias and reorder
    if (fc1_expert_biases != nullptr)
    {
        loraAddBias(lora_add_bias_, lora_fc1_result_, fc1_expert_biases, num_valid_tokens_ptr, inter_size,
            permuted_token_selected_experts_, expanded_num_rows, is_gated_activation, stream);
        return lora_add_bias_;
    }
    else if (is_gated_activation)
    {
        loraReorder(lora_add_bias_, lora_fc1_result_, num_valid_tokens_ptr, inter_size, expanded_num_rows, stream);
        return lora_add_bias_;
    }
    else
    {
        return lora_fc1_result_;
    }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::loraFC2(int64_t inter_size,
    int64_t hidden_size, int num_experts_per_node, int start_expert, int64_t const* num_valid_tokens_ptr,
    int64_t num_tokens, LoraParams& lora_params, float const* fc2_fp8_quant, cudaStream_t stream)
{
    std::vector<void const*>& host_permuted_fc2_weight_ptrs = host_lora_workspace_.host_permuted_fc2_weight_ptrs;
    std::vector<int32_t>& host_permuted_fc2_lora_ranks = host_lora_workspace_.host_permuted_fc2_lora_ranks;
    std::vector<int64_t>& host_expert_first_token_offset = host_lora_workspace_.host_expert_first_token_offset;
    auto fc2_lora_impl = lora_params.fc2_lora_impl;
    int num_reqs = lora_params.num_reqs;

    ScaleBiasType* input{};
    if constexpr (use_fp8)
    {
        TLLM_CHECK(lora_input_);
        bool const scale_is_dequant = false;
        dequantFP8(lora_input_, fc1_result_, num_valid_tokens_ptr, inter_size, num_tokens, fc2_fp8_quant,
            scale_is_dequant, stream);
        sync_check_cuda_error(stream);
        input = lora_input_;
    }
    else if constexpr (!use_fp4)
    {
        TLLM_CHECK(!lora_input_);
        input = reinterpret_cast<ScaleBiasType*>(fc1_result_);
    }

    void* lora_workspace = lora_params.workspace;
    int64_t num_valid_tokens = host_expert_first_token_offset[num_experts_per_node];
    void* tmp_lora_fc_result = static_cast<void*>(lora_fc2_result_);
    int64_t num_reqs_lora = std::min(num_valid_tokens, static_cast<int64_t>(num_reqs * num_experts_per_node));

    ::tensorrt_llm::kernels::Lora_run(fc2_lora_impl.get(), num_valid_tokens, num_reqs_lora, input,
        host_permuted_fc2_lora_ranks.data(), host_permuted_fc2_weight_ptrs.data(), 0, &tmp_lora_fc_result,
        lora_workspace, stream);
    sync_check_cuda_error(stream);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::runMoe(
    void const* input_activations_void, void const* input_sf_void, int const* token_selected_experts,
    float const* token_final_scales, void const* fc1_expert_weights_void, void const* fc1_expert_biases_void,
    ActivationType fc1_activation_type, void const* fc2_expert_weights_void, void const* fc2_expert_biases_void,
    QuantParams quant_params, int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const full_num_experts, int const experts_per_token, char* workspace_ptr, void* final_output_void,
    int* expanded_source_row_to_expanded_dest_row, MOEParallelismConfig parallelism_config, bool const enable_alltoall,
    bool use_lora, LoraParams& lora_params, bool use_fp8_block_scaling, bool min_latency_mode,
    MoeMinLatencyParams& min_latency_params, cudaStream_t stream)
{
    static constexpr bool int_scales_required
        = std::is_same<WeightType, uint8_t>::value || std::is_same<WeightType, cutlass::uint4b_t>::value;
    static constexpr bool fp8_scales_required
        = std::is_same<WeightType, __nv_fp8_e4m3>::value || std::is_same<WeightType, __nv_fp8_e5m2>::value;

    auto const* input_activations = static_cast<InputType const*>(input_activations_void);
    auto const* input_sf = input_sf_void
        ? reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF const*>(input_sf_void)
        : nullptr;
    auto const* fc1_expert_weights = static_cast<WeightType const*>(fc1_expert_weights_void);
    auto const* fc1_expert_biases = reinterpret_cast<ScaleBiasType const*>(fc1_expert_biases_void);
    auto const* fc2_expert_weights = static_cast<WeightType const*>(fc2_expert_weights_void);
    auto const* fc1_int_scales = reinterpret_cast<ScaleBiasType const*>(quant_params.wo.fc1_weight_scales);
    auto const* fc2_int_scales = reinterpret_cast<ScaleBiasType const*>(quant_params.wo.fc2_weight_scales);

    auto const* fc1_fp8_dequant = quant_params.fp8.dequant_fc1;
    auto const* fc2_fp8_quant = quant_params.fp8.quant_fc2;
    auto const* fc2_fp8_dequant = quant_params.fp8.dequant_fc2;
    auto const* input_fp8_dequant = quant_params.fp8.dequant_input;
    auto const* fc2_expert_biases = reinterpret_cast<ScaleBiasType const*>(fc2_expert_biases_void);
    auto* final_output = static_cast<OutputType*>(final_output_void);
    float const* token_topk_unpermuted_scales = token_final_scales;
    // Note: getBlockScaleGemmRunner will do a sanity check on our template parameters.
    auto* blockscale_gemm_runner = use_fp8_block_scaling ? getBlockScaleGemmRunner() : nullptr;

    TLLM_CHECK(input_activations);
    TLLM_CHECK(token_selected_experts);
    TLLM_CHECK(fc1_expert_weights);
    TLLM_CHECK(fc2_expert_weights);
    TLLM_CHECK(workspace_ptr);
    // TLLM_CHECK(token_topk_unpermuted_scales);
    TLLM_CHECK(expanded_source_row_to_expanded_dest_row);
    TLLM_CHECK(full_num_experts % parallelism_config.ep_size == 0);
    TLLM_CHECK(full_num_experts % parallelism_config.cluster_size == 0);
    TLLM_CHECK_WITH_INFO(hidden_size % (128 / sizeof_bits<WeightType>::value) == 0,
        "Hidden size does not meet minimum alignment requirements for MOE GEMM");
    // Require at least 64 bytes of alignment for MOE GEMM
    TLLM_CHECK_WITH_INFO(inter_size % (128 / sizeof_bits<WeightType>::value) == 0,
        "Inter size does not meet minimum alignment requirements for MOE GEMM");
    if (use_fp4)
    {
        TLLM_CHECK_WITH_INFO(
            hidden_size % 128 == 0, "Inter size does not meet minimum alignment requirements for MOE GEMM");
        TLLM_CHECK_WITH_INFO(
            inter_size % 128 == 0, "Inter size does not meet minimum alignment requirements for MOE GEMM");
    }

    // These values must fit into an int for building the source maps
    TLLM_CHECK_WITH_INFO(num_rows <= std::numeric_limits<int>::max(), "Number of rows is too large");
    TLLM_CHECK_WITH_INFO(
        num_rows * full_num_experts <= std::numeric_limits<int>::max(), "Number of rows * num_experts is too large");
    TLLM_CHECK_WITH_INFO(experts_per_token * full_num_experts <= std::numeric_limits<int>::max(),
        "experts_per_token * num_experts is too large");

    TLLM_CHECK_WITH_INFO(gemm1_config_, "MOE GEMM1 Config is not set");
    TLLM_CHECK_WITH_INFO(gemm2_config_, "MOE GEMM2 Config is not set");

    TLLM_CHECK_WITH_INFO(!use_lora || !use_fp4, "MOE does not support LoRA with FP4 model");

    if (int_scales_required)
    {
        if (!(quant_params.groupwise.fc1.weight_scales && quant_params.groupwise.fc2.weight_scales))
        {
            TLLM_CHECK_WITH_INFO(
                fc1_int_scales != nullptr, "Weight scales expected but scale for first matmul is a null pointer");
            TLLM_CHECK_WITH_INFO(
                fc2_int_scales != nullptr, "Weight scales expected but scale for second matmul is a null pointer");
        }
        TLLM_CHECK_WITH_INFO(fc1_fp8_dequant == nullptr && fc2_fp8_quant == nullptr && fc2_fp8_dequant == nullptr,
            "FP8 scales are provided for integer quantization");
    }
    else if (fp8_scales_required && !use_fp8_block_scaling)
    {
        TLLM_CHECK_WITH_INFO(fc1_expert_biases == nullptr, "Bias is not supported with FP8");
        TLLM_CHECK_WITH_INFO(fc2_expert_biases == nullptr, "Bias is not supported with FP8");

        TLLM_CHECK_WITH_INFO(
            fc1_fp8_dequant != nullptr, "FP8 scales expected but dequant scale for FC1 is a null pointer");
        TLLM_CHECK_WITH_INFO(fc2_fp8_quant != nullptr, "FP8 scales expected but quant scale for FC2 is a null pointer");
        TLLM_CHECK_WITH_INFO(
            fc2_fp8_dequant != nullptr, "FP8 scales expected but quant scale for FC2 is a null pointer");

        TLLM_CHECK_WITH_INFO(
            fc1_int_scales == nullptr && fc2_int_scales == nullptr, "Integer scales are provided for FP8 quantization");
    }
    else if (use_lora && use_fp8)
    {
        TLLM_CHECK_WITH_INFO(
            input_fp8_dequant != nullptr, "FP8 scales expected but quant scale for input is a null pointer");
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            fc1_int_scales == nullptr, "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC1");
        TLLM_CHECK_WITH_INFO(
            fc2_int_scales == nullptr, "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC2");
        TLLM_CHECK_WITH_INFO(
            fc1_fp8_dequant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received dequant scale for FC1");
        TLLM_CHECK_WITH_INFO(
            fc2_fp8_quant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
        TLLM_CHECK_WITH_INFO(
            fc2_fp8_dequant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
    }

    bool use_awq = quant_params.groupwise.fc1.act_scales && quant_params.groupwise.fc2.act_scales;
    int const num_experts_per_node = full_num_experts / parallelism_config.ep_size;

    configureWsPtrs(workspace_ptr, num_rows, hidden_size, inter_size, num_experts_per_node, experts_per_token,
        fc1_activation_type, parallelism_config, use_lora, use_fp8_block_scaling, min_latency_mode, use_awq);

    int start_expert = num_experts_per_node * parallelism_config.ep_rank;
    int end_expert = start_expert + num_experts_per_node;

    bool const needs_num_valid = parallelism_config.ep_size > 1;
    int64_t const* num_valid_tokens_ptr = needs_num_valid ? expert_first_token_offset_ + num_experts_per_node : nullptr;

    auto expanded_num_rows = num_rows * experts_per_token;

    if (min_latency_mode)
    {
        TLLM_CHECK(use_lora == false);
        TLLM_CHECK(use_awq == false);
        TLLM_CHECK(use_fp4 == true);

        buildMinLatencyActiveExpertMaps(min_latency_params.num_active_experts_per_node,
            min_latency_params.experts_to_token_score, min_latency_params.active_expert_global_ids,
            expert_first_token_offset_, token_selected_experts, token_final_scales, num_rows, experts_per_token,
            start_expert, end_expert, num_experts_per_node, parallelism_config.cluster_rank,
            parallelism_config.cluster_size, full_num_experts, stream);
        sync_check_cuda_error(stream);

        auto [gemm1_tma_ws_input, gemm2_tma_ws_input] = setupTmaWarpSpecializedInputs(num_rows, expanded_num_rows,
            fc1_activation_type, hidden_size, inter_size, num_experts_per_node, input_activations_void, input_sf,
            final_output, fc1_expert_weights, fc2_expert_weights, quant_params, fc1_expert_biases, fc2_expert_biases,
            min_latency_mode, min_latency_params, use_lora, start_expert, parallelism_config, stream);

        // todo: input_activations_void should be nvfp4, waiting for yuxian's mr ready
        Self::gemm1(moe_gemm_runner_, blockscale_gemm_runner, reinterpret_cast<T const*>(input_activations_void),
            fc1_result_, glu_inter_result_, expert_first_token_offset_, gemm1_tma_ws_input, fc1_expert_weights,
            fc1_expert_biases, num_valid_tokens_ptr, fc1_int_scales, fc1_fp8_dequant, fc2_fp8_quant,
            input_sf /*input fp4 scale or expanded fp4 scale*/, fc2_fp4_act_scale_, quant_params, num_rows,
            expanded_num_rows, hidden_size, inter_size, num_experts_per_node, fc1_activation_type,
            alpha_scale_ptr_array_fc1_, !use_lora, stream, *gemm1_config_, true,
            min_latency_params.num_active_experts_per_node, min_latency_params.active_expert_global_ids, start_expert);
        sync_check_cuda_error(stream);

        auto gemm2_input = applyPrequantScale(smoothed_act_, fc1_result_, quant_params.groupwise.fc2.act_scales,
            num_valid_tokens_ptr, expanded_num_rows, inter_size, use_awq, stream);
        Self::gemm2(moe_gemm_runner_, blockscale_gemm_runner, gemm2_input, final_output, nullptr,
            expert_first_token_offset_, gemm2_tma_ws_input, fc2_expert_weights, fc2_expert_biases, fc2_int_scales,
            fc2_fp8_dequant, fc2_fp4_act_scale_, quant_params, token_topk_unpermuted_scales,
            permuted_token_final_scales_, expanded_source_row_to_expanded_dest_row, permuted_source_token_ids_,
            unpermuted_token_selected_experts_, num_valid_tokens_ptr, num_rows, expanded_num_rows, hidden_size,
            inter_size, num_experts_per_node, experts_per_token, alpha_scale_ptr_array_fc2_, use_lora, lora_fc2_result_,
            stream, parallelism_config, enable_alltoall, *gemm2_config_, true,
            min_latency_params.num_active_experts_per_node, min_latency_params.active_expert_global_ids, start_expert);
        sync_check_cuda_error(stream);
    }
    else
    {
        bool fused_prologue_result = false;
        if (!use_w4afp8)
        {
            // WAR: fusedBuildExpertMapsSortFirstToken kernel will lead to illegal memory access for W4AFP8
            fused_prologue_result = fusedBuildExpertMapsSortFirstToken(token_selected_experts,
                unpermuted_token_selected_experts_, permuted_source_token_ids_, expert_first_token_offset_, num_rows,
                num_experts_per_node, experts_per_token, start_expert, end_expert, stream);
        }
        if (!fused_prologue_result)
        {
            TLLM_LOG_TRACE("Falling back to unfused prologue");
            buildExpertMaps(token_selected_experts, unpermuted_token_selected_experts_, unpermuted_source_token_ids_,
                num_rows, num_experts_per_node, experts_per_token, start_expert, end_expert, stream);

            sync_check_cuda_error(stream);

            generateTokenPermutation(unpermuted_token_selected_experts_, unpermuted_source_token_ids_,
                permuted_token_selected_experts_, permuted_source_token_ids_, expert_first_token_offset_, num_rows,
                num_experts_per_node, experts_per_token, sorter_, static_cast<void*>(sorter_ws_), stream);
        }

        sync_check_cuda_error(stream);

        bool is_gated_activation = isGatedActivation(fc1_activation_type);

        if (use_lora)
        {
            std::vector<int>& host_permuted_rows = host_lora_workspace_.host_permuted_rows;
            std::vector<int64_t>& host_expert_first_token_offset = host_lora_workspace_.host_expert_first_token_offset;
            host_permuted_rows.resize(expanded_num_rows);
            TLLM_CUDA_CHECK(tensorrt_llm::common::cudaMemcpyAsyncSanitized(host_permuted_rows.data(),
                permuted_source_token_ids_, expanded_num_rows * sizeof(int), cudaMemcpyDeviceToHost, stream));
            host_expert_first_token_offset.resize(num_experts_per_node + 1);
            TLLM_CUDA_CHECK(tensorrt_llm::common::cudaMemcpyAsyncSanitized(host_expert_first_token_offset.data(),
                expert_first_token_offset_, (num_experts_per_node + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost,
                stream));
            TLLM_CUDA_CHECK(cudaEventRecord(*(lora_params.memcpy_event_ptr), stream));
        }

        using ExpandedActivationsType = std::conditional_t<use_w4afp8, BackBoneType, T>;
        expandInputRowsKernelLauncher(input_activations, reinterpret_cast<ExpandedActivationsType*>(permuted_data_),
            token_topk_unpermuted_scales, permuted_token_final_scales_, permuted_source_token_ids_,
            expanded_source_row_to_expanded_dest_row, num_rows, hidden_size, experts_per_token, num_experts_per_node,
            quant_params.fp4.fc1.act_global_scale, expert_first_token_offset_, fc1_fp4_act_scale_, input_sf, stream);

        sync_check_cuda_error(stream);

        auto [gemm1_tma_ws_input, gemm2_tma_ws_input] = setupTmaWarpSpecializedInputs(num_rows, expanded_num_rows,
            fc1_activation_type, hidden_size, inter_size, num_experts_per_node, input_activations_void, input_sf,
            final_output, fc1_expert_weights, fc2_expert_weights, quant_params, fc1_expert_biases, fc2_expert_biases,
            min_latency_mode, min_latency_params, use_lora, start_expert, parallelism_config, stream);

        if (use_lora)
        {
            bool all_token_without_lora = setupLoraWorkspace(expanded_num_rows, num_rows, inter_size, hidden_size,
                start_expert, is_gated_activation, num_experts_per_node, needs_num_valid, lora_params, stream);

            if (!all_token_without_lora)
            {
                fc1_expert_biases = loraFC1(expanded_num_rows, inter_size, hidden_size, num_experts_per_node,
                    start_expert, num_valid_tokens_ptr, is_gated_activation, fc1_expert_biases, lora_params,
                    input_fp8_dequant, stream);
                sync_check_cuda_error(stream);
            }
            else
            {
                use_lora = false;
            }
        }

        auto gemm1_input = applyPrequantScale(smoothed_act_, permuted_data_, quant_params.groupwise.fc1.act_scales,
            num_valid_tokens_ptr, expanded_num_rows, hidden_size, use_awq, stream);
        sync_check_cuda_error(stream);
        Self::gemm1(moe_gemm_runner_, blockscale_gemm_runner, gemm1_input, fc1_result_, glu_inter_result_,
            expert_first_token_offset_, gemm1_tma_ws_input, fc1_expert_weights, fc1_expert_biases, num_valid_tokens_ptr,
            fc1_int_scales, fc1_fp8_dequant, fc2_fp8_quant, fc1_fp4_act_scale_, fc2_fp4_act_scale_, quant_params,
            num_rows, expanded_num_rows, hidden_size, inter_size, num_experts_per_node, fc1_activation_type,
            alpha_scale_ptr_array_fc1_, !use_lora, stream, *gemm1_config_, false, nullptr, nullptr, 0);
        sync_check_cuda_error(stream);

        if (use_lora)
        {
            loraFC2(inter_size, hidden_size, num_experts_per_node, start_expert, num_valid_tokens_ptr,
                expanded_num_rows, lora_params, fc2_fp8_quant, stream);
            sync_check_cuda_error(stream);
        }

        auto gemm2_input = applyPrequantScale(smoothed_act_, fc1_result_, quant_params.groupwise.fc2.act_scales,
            num_valid_tokens_ptr, expanded_num_rows, inter_size, use_awq, stream);
        sync_check_cuda_error(stream);
        Self::gemm2(moe_gemm_runner_, blockscale_gemm_runner, gemm2_input, fc2_result_, final_output,
            expert_first_token_offset_, gemm2_tma_ws_input, fc2_expert_weights, fc2_expert_biases, fc2_int_scales,
            fc2_fp8_dequant, fc2_fp4_act_scale_, quant_params, token_topk_unpermuted_scales,
            permuted_token_final_scales_, expanded_source_row_to_expanded_dest_row, permuted_source_token_ids_,
            unpermuted_token_selected_experts_, num_valid_tokens_ptr, num_rows, expanded_num_rows, hidden_size,
            inter_size, num_experts_per_node, experts_per_token, alpha_scale_ptr_array_fc2_, use_lora, lora_fc2_result_,
            stream, parallelism_config, enable_alltoall, *gemm2_config_, false, nullptr, nullptr, 0);
        sync_check_cuda_error(stream);
    }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::computeStridesTmaWarpSpecialized(
    int64_t const* expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput layout_info1,
    TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens, int64_t expanded_num_tokens, int64_t gemm1_n,
    int64_t gemm1_k, int64_t gemm2_n, int64_t gemm2_k, int const num_experts_per_node, T const* gemm1_in,
    T const* gemm2_in, WeightType const* weights1, WeightType const* weights2, float const* fp8_dequant1,
    float const* fp8_dequant2, TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2, QuantParams quant_params,
    ScaleBiasType const* bias1, ScaleBiasType const* bias2, UnfusedGemmOutputType* gemm1_output,
    UnfusedGemmOutputType* gemm2_output, cudaStream_t stream)
{
    // Always nullptr
    layout_info1.ptr_c = nullptr;
    layout_info1.stride_c = nullptr;
    layout_info2.ptr_c = nullptr;
    layout_info2.stride_c = nullptr;

    auto alpha_scale_flat1 = use_fp4 ? quant_params.fp4.fc1.global_scale : fp8_dequant1;
    auto alpha_scale_flat2 = use_fp4 ? quant_params.fp4.fc2.global_scale : fp8_dequant2;
    if (!alpha_scale_flat1 && !alpha_scale_flat2)
    {
        layout_info1.alpha_scale_ptr_array = nullptr;
        layout_info2.alpha_scale_ptr_array = nullptr;
    }

    layout_info1.int4_groupwise_params.enabled = use_w4afp8;
    layout_info2.int4_groupwise_params.enabled = use_w4afp8;

    int const threads = std::min(1024, num_experts_per_node);
    int const blocks = (num_experts_per_node + threads - 1) / threads;

    auto* kernel_instance = &computeStridesTmaWarpSpecializedKernel<T, WeightType, OutputType, ScaleBiasType>;

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = threads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel_instance, expert_first_token_offset, layout_info1, layout_info2, num_tokens,
        expanded_num_tokens, gemm1_n, gemm1_k, gemm2_n, gemm2_k, num_experts_per_node, gemm1_in, gemm2_in, weights1,
        weights2, alpha_scale_flat1, alpha_scale_flat2, fp4_act_flat1, fp4_act_flat2, quant_params, bias1, bias2,
        gemm1_output, gemm2_output);

    return std::make_pair(layout_info1, layout_info2);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType,
    Enable>::computeStridesTmaWarpSpecializedLowLatency(TmaWarpSpecializedGroupedGemmInput layout_info1,
    TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens, int64_t gemm1_n, int64_t gemm1_k,
    int64_t gemm2_n, int64_t gemm2_k, int const num_experts, T const* input1, T const* input2,
    WeightType const* weights1, WeightType const* weights2, float const* fp8_dequant1, float const* fp8_dequant2,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat, QuantParams quant_params,
    ScaleBiasType const* bias1, ScaleBiasType const* bias2, UnfusedGemmOutputType* output1,
    UnfusedGemmOutputType* output2, int const* num_active_experts_per, int const* active_expert_global_ids,
    int start_expert, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(!use_w4afp8, "W4AFP8 is not supported in low latency mode");

    // Always nullptr
    layout_info1.ptr_c = nullptr;
    layout_info1.stride_c = nullptr;
    layout_info2.ptr_c = nullptr;
    layout_info2.stride_c = nullptr;

    auto alpha_scale_flat1 = use_fp4 ? quant_params.fp4.fc1.global_scale : fp8_dequant1;
    auto alpha_scale_flat2 = use_fp4 ? quant_params.fp4.fc2.global_scale : fp8_dequant2;
    if (!alpha_scale_flat1)
    {
        layout_info1.alpha_scale_ptr_array = nullptr;
    }
    if (!alpha_scale_flat2)
    {
        layout_info2.alpha_scale_ptr_array = nullptr;
    }

    layout_info1.int4_groupwise_params.enabled = false;
    layout_info2.int4_groupwise_params.enabled = false;

    int const threads = std::min(1024, num_experts);
    int const blocks = (num_experts + threads - 1) / threads;

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = threads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config,
        computeStridesTmaWarpSpecializedLowLatencyKernel<T, WeightType, OutputType, ScaleBiasType>, layout_info1,
        layout_info2, num_tokens, gemm1_n, gemm1_k, gemm2_n, gemm2_k, num_experts, input1, input2, weights1, weights2,
        alpha_scale_flat1, alpha_scale_flat2, fc1_fp4_act_flat, fc2_fp4_act_flat, quant_params, bias1, bias2, output1,
        output2, num_active_experts_per, active_expert_global_ids, start_expert);

    return std::make_pair(layout_info1, layout_info2);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType, class Enable>
std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, Enable>::setupTmaWarpSpecializedInputs(
    int64_t num_rows, int64_t expanded_num_rows, ActivationType fc1_activation_type, int64_t hidden_size,
    int64_t inter_size, int64_t num_experts_per_node, void const* input_activations_void,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, void* final_output,
    WeightType const* fc1_expert_weights, WeightType const* fc2_expert_weights, QuantParams quant_params,
    ScaleBiasType const* fc1_expert_biases, ScaleBiasType const* fc2_expert_biases, bool min_latency_mode,
    MoeMinLatencyParams& min_latency_params, bool use_lora, int start_expert, MOEParallelismConfig parallelism_config,
    cudaStream_t stream)
{
    auto gemm1_tma_ws_input = tma_ws_grouped_gemm1_input_;
    auto gemm2_tma_ws_input = tma_ws_grouped_gemm2_input_;
    if (!moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_)
        && !moe_gemm_runner_.isTmaWarpSpecialized(*gemm2_config_))
    {
        return std::make_pair(gemm1_tma_ws_input, gemm2_tma_ws_input);
    }

    bool use_awq = quant_params.groupwise.fc1.act_scales && quant_params.groupwise.fc2.act_scales;

    bool is_gated_activation = isGatedActivation(fc1_activation_type);
    int64_t const fc1_out_size = is_gated_activation ? inter_size * 2 : inter_size;

    bool has_different_gemm_output_type = !std::is_same_v<T, OutputType>;
    bool const has_intermediate = has_different_gemm_output_type || is_gated_activation;
    auto* gemm1_output = has_intermediate ? glu_inter_result_ : static_cast<void*>(fc1_result_);

    bool use_prequant_scale_kernel = use_awq && !std::is_same_v<T, WeightType>;
    auto gemm2_input = use_prequant_scale_kernel ? smoothed_act_ : fc1_result_;

    if (min_latency_mode)
    {
        auto gemm1_input = input_activations_void;

        gemm1_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
        gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;

        TLLM_CHECK_WITH_INFO(gemm1_input != gemm1_output, "Input and output buffers are overlapping");
        return Self::computeStridesTmaWarpSpecializedLowLatency(gemm1_tma_ws_input, gemm2_tma_ws_input, num_rows,
            fc1_out_size, hidden_size, hidden_size, inter_size, num_experts_per_node,
            reinterpret_cast<T const*>(gemm1_input), reinterpret_cast<T const*>(gemm2_input), fc1_expert_weights,
            fc2_expert_weights, quant_params.fp8.dequant_fc1, quant_params.fp8.dequant_fc2, input_sf,
            fc2_fp4_act_scale_, quant_params, nullptr, nullptr, reinterpret_cast<UnfusedGemmOutputType*>(gemm1_output),
            reinterpret_cast<UnfusedGemmOutputType*>(fc2_result_), min_latency_params.num_active_experts_per_node,
            min_latency_params.active_expert_global_ids, start_expert, stream);
    }
    else
    {
        auto gemm1_input = use_prequant_scale_kernel ? smoothed_act_ : permuted_data_;

        gemm1_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
        gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;

        bool apply_bias = parallelism_config.tp_rank == 0;
        bool using_hopper_fused_finalize
            = !use_deterministic_hopper_reduce_ && gemm2_config_->sm_version == 90 && !use_w4afp8 && !use_lora;
        if (using_hopper_fused_finalize)
        {
            assert(min_latency_mode == false);
            gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
            gemm2_tma_ws_input.setFinalizeFusionParams(final_output, permuted_token_final_scales_,
                expert_first_token_offset_, permuted_source_token_ids_, apply_bias ? fc2_expert_biases : nullptr,
                hidden_size, num_rows);
        }

        TLLM_CHECK_WITH_INFO(gemm1_input != gemm1_output, "Input and output buffers are overlapping");
        return Self::computeStridesTmaWarpSpecialized(expert_first_token_offset_, gemm1_tma_ws_input,
            gemm2_tma_ws_input, num_rows, expanded_num_rows, fc1_out_size, hidden_size, hidden_size, inter_size,
            num_experts_per_node, reinterpret_cast<T const*>(gemm1_input), reinterpret_cast<T const*>(gemm2_input),
            fc1_expert_weights, fc2_expert_weights, quant_params.fp8.dequant_fc1, quant_params.fp8.dequant_fc2,
            fc1_fp4_act_scale_, fc2_fp4_act_scale_, quant_params, fc1_expert_biases, fc2_expert_biases,
            reinterpret_cast<UnfusedGemmOutputType*>(gemm1_output),
            reinterpret_cast<UnfusedGemmOutputType*>(fc2_result_), stream);
    }
}

// ==================== Helper for getting load balanced routing for profiling ==================================

__global__ void prepareFakeRouterBuffers(int* unpermuted_source_rows, int* unpermuted_expert_selection,
    int64_t num_tokens, int64_t k, int64_t num_experts, int64_t num_experts_per_node)
{
    int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    int64_t sample = blockIdx.y;
    if (tid >= num_tokens)
    {
        return;
    }

    // Offset the buffers to the start of the sample
    unpermuted_source_rows += sample * num_tokens * k;
    unpermuted_expert_selection += sample * num_tokens * k;

    // This is not perf sensitive we just init the state here every time prepare is called
    // This means the first N tokens will always have the same distribution, regardless of num_tokens
    curandStatePhilox4_32_10_t state;
    curand_init(sample, tid, 0, &state);
    for (int k_idx = 0; k_idx < k; k_idx++)
    {
        while (true)
        {
            // curand_uniform includes 1 but not 0, so round up and subtract 1
            int expert = std::ceil(static_cast<float>(num_experts) * curand_uniform(&state)) - 1;

            bool valid = true;
            for (int prev_k = 0; prev_k < k_idx; prev_k++)
            {
                int prev_expert = unpermuted_expert_selection[k * tid + prev_k];
                if (expert == prev_expert)
                {
                    valid = false;
                    break;
                }
            }

            if (valid)
            {
                int64_t const idx = k * tid + k_idx;
                unpermuted_expert_selection[idx] = expert < num_experts_per_node ? expert : num_experts_per_node;
                unpermuted_source_rows[idx] = k_idx * num_tokens + tid;
                break;
            }
        }
    }
}

__global__ void populateRandomBufferKernel(void* buffer_void, size_t size)
{
    int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
    {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(size, tid, 0, &state);

    constexpr int elem_per_thread = 128 / sizeof(uint4);
    auto* buffer = reinterpret_cast<uint4*>(buffer_void);
#pragma unroll
    for (int i = 0; i < elem_per_thread; i++)
        buffer[tid * elem_per_thread + i] = curand4(&state);
}

__global__ void buildReverseMap(int* expanded_source_row_to_expanded_dest_row,
    int const* expanded_dest_row_to_expanded_source_row, int64_t expanded_num_tokens)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < expanded_num_tokens)
    {
        assert(expanded_dest_row_to_expanded_source_row[tid] >= 0);
        assert(expanded_dest_row_to_expanded_source_row[tid] < expanded_num_tokens);
        expanded_source_row_to_expanded_dest_row[expanded_dest_row_to_expanded_source_row[tid]] = tid;
    }
}

template <int BLOCK_SIZE, int NUM_ROUTING_SAMPLES>
__global__ void prepareMinLatencyBuffer(int* num_active_experts_per_node, int* active_expert_global_ids,
    int64_t* expert_first_token_offset, int const num_tokens, int const num_experts_per_token,
    int const num_experts_per_node)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // 0. set offset
    num_active_experts_per_node += bid;
    active_expert_global_ids += bid * num_experts_per_node;
    expert_first_token_offset += bid * (num_experts_per_node + 1);

    // 1. set the num_active_experts_per_node
    int num_active = max(1, (int) (bid * ((float) num_experts_per_node / NUM_ROUTING_SAMPLES)));
    *num_active_experts_per_node = num_active;

    // 2. generate random active experts
    extern __shared__ float s_buf[];
    float* expert_refs = s_buf;
    int* expert_refs_idx = reinterpret_cast<int*>(expert_refs + num_experts_per_node);

    curandState_t local_state;
    curand_init(bid, tid, 0, &local_state);
    for (int i = tid; i < num_experts_per_node; i += BLOCK_SIZE)
    {
        expert_refs[i] = (float) curand_uniform(&local_state);
        expert_refs_idx[i] = (int) i;
    }
    __syncthreads();

    float thread_key[1];
    int thread_value[1];
    thread_key[0] = std::numeric_limits<float>::max();
    thread_value[0] = num_experts_per_node;

    if (tid < num_experts_per_node)
    {
        thread_key[0] = expert_refs[tid];
        thread_value[0] = expert_refs_idx[tid];
    }

    using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_SIZE, 1, int>;
    using BlockRadixSortValue = cub::BlockRadixSort<int, BLOCK_SIZE, 1>;

    union TempStorage
    {
        typename BlockRadixSort::TempStorage key_value;
        typename BlockRadixSortValue::TempStorage value;
    };
    __shared__ union TempStorage temp_storage;

    BlockRadixSort(temp_storage.key_value).Sort(thread_key, thread_value);
    __syncthreads();

    if (tid > num_active)
    {
        thread_value[0] = std::numeric_limits<int>::max();
    }
    BlockRadixSortValue(temp_storage.value).Sort(thread_value);
    __syncthreads();

    // 3. set the active_expert_global_ids and expert_first_token_offset
    for (int i = tid; i < num_experts_per_node; i += BLOCK_SIZE)
    {
        if (i < num_active)
        {
            active_expert_global_ids[i] = thread_value[0];
            expert_first_token_offset[i] = i * num_tokens;
        }
        else
        {
            active_expert_global_ids[i] = -1;
            expert_first_token_offset[i] = num_active * num_tokens;
        }
    }
    if (tid == 0)
    {
        expert_first_token_offset[num_experts_per_node] = num_active * num_tokens;
    }
}

void populateRandomBuffer(void* buffer_void, size_t size, cudaStream_t stream)
{
    // Each thread initialises 128 bytes
    TLLM_CHECK_WITH_INFO(size % 128 == 0, "Unexpected size alignment");
    auto threads = size / 128;
    populateRandomBufferKernel<<<ceilDiv(threads, 128), 128, 0, stream>>>(buffer_void, threads);
}

std::map<std::string, std::pair<size_t, size_t>> GemmProfilerBackend::getProfilerWorkspaces(
    int maxM, bool is_tma_ws_input)
{
    size_t k = mK;
    size_t num_expanded_tokens = mMinLatencyMode ? maxM * mNumExpertsPerNode : maxM * k;

    TLLM_CHECK(mDType != nvinfer1::DataType::kINT4);
    // nvllm still uses int64 because torch doesn't have fp4 yet.
    bool is_4bit_act = mDType == nvinfer1::DataType::kFP4 || mDType == nvinfer1::DataType::kINT64;
    bool is_4bit_weight = mWType == nvinfer1::DataType::kINT4 || mWType == nvinfer1::DataType::kFP4
        || mWType == nvinfer1::DataType::kINT64;
    TLLM_CHECK_WITH_INFO(!is_4bit_act || is_4bit_weight, "Cannot have 4-bit activation with non-4-bit weight");
    float dtype_bytes = is_4bit_act
        ? 0.5f
        : static_cast<float>(mWType == nvinfer1::DataType::kINT4 ? tensorrt_llm::common::getDTypeSize(mOType)
                                                                 : tensorrt_llm::common::getDTypeSize(mDType));
    float weight_bytes = is_4bit_weight ? 0.5f : static_cast<float>(tensorrt_llm::common::getDTypeSize(mWType));
    size_t output_bytes = tensorrt_llm::common::getDTypeSize(mOType);
    size_t gemm_output_bytes = (mOType == nvinfer1::DataType::kFP8)
        ? sizeof(TmaWarpSpecializedGroupedGemmInput::OutputTypeAdaptor_t<__nv_fp8_e4m3>)
        : output_bytes;

    size_t hidden_size = mExpertHiddenSize;
    size_t inter_size = mExpertInterSize; // Already divided by TP
    size_t num_experts_per_node = mNumExpertsPerNode;

    size_t fc1_out_size = inter_size;
    if (isGatedActivation(mActivationType))
    {
        fc1_out_size = inter_size * 2;
    }

    // TODO Needs updated when gather/finalize fusion is integrated
    size_t input_size1 = hidden_size * num_expanded_tokens * dtype_bytes;
    size_t output_size1 = inter_size * num_expanded_tokens * dtype_bytes;

    size_t input_size2 = inter_size * num_expanded_tokens * dtype_bytes;
    size_t output_size2 = hidden_size * output_bytes;

    size_t input_size = mGemmToProfile == GemmToProfile::GEMM_1 ? input_size1 : input_size2;
    size_t output_size = mGemmToProfile == GemmToProfile::GEMM_1 ? output_size1 : output_size2;

    // This may allocate a pointer when not required. That's fine it will be ignored at the cost of some memory
    size_t intermediate_size1 = fc1_out_size * num_expanded_tokens * gemm_output_bytes; // Note gemm_output_bytes
    size_t intermediate_size2 = hidden_size * num_expanded_tokens * gemm_output_bytes;  // Note gemm_output_bytes

    size_t intermediate_size = mGemmToProfile == GemmToProfile::GEMM_1 ? intermediate_size1 : intermediate_size2;

    size_t weights_1 = hidden_size * fc1_out_size * num_experts_per_node * weight_bytes;
    size_t bias_1 = mBias ? fc1_out_size * num_experts_per_node * dtype_bytes : 0;
    if (mUseLora && !is_tma_ws_input)
        bias_1 = output_size1;
    size_t weights_2 = hidden_size * inter_size * num_experts_per_node * weight_bytes;
    size_t bias_2 = mBias ? hidden_size * num_experts_per_node * dtype_bytes : 0;

    size_t weights_size = mNeedWeights ? (mGemmToProfile == GemmToProfile::GEMM_1 ? weights_1 : weights_2) : 0;
    size_t bias_size = mGemmToProfile == GemmToProfile::GEMM_1 ? bias_1 : bias_2;

    // TODO Make quant 2 & 4 bigger for FP8 if we ever change to scaling per expert
    bool is_int_w_quant
        = (mWType == nvinfer1::DataType::kINT8 || mWType == nvinfer1::DataType::kINT4) && mGroupSize <= 0;
    bool is_int_groupwise_w_quant
        = (mWType == nvinfer1::DataType::kINT8 || mWType == nvinfer1::DataType::kINT4) && mGroupSize > 0;
    bool is_fp8_w_quant = mWType == nvinfer1::DataType::kFP8;
    // nvllm still uses int64 because torch doesn't have fp4 yet.
    bool is_fp4_act_quant = mDType == nvinfer1::DataType::kFP4 || mDType == nvinfer1::DataType::kINT64;
    bool is_fp4_w_quant = mWType == nvinfer1::DataType::kFP4 || mWType == nvinfer1::DataType::kINT64;
    bool is_w4afp8_quant = is_int_groupwise_w_quant && mDType == nvinfer1::DataType::kFP8;

    // Int sizes
    size_t quant_1_size = is_int_w_quant ? fc1_out_size * num_experts_per_node * dtype_bytes : 0;
    size_t quant_2_size = is_int_w_quant ? hidden_size * num_experts_per_node * dtype_bytes : 0;
    if (is_int_w_quant)
    {
        quant_1_size = fc1_out_size * num_experts_per_node * dtype_bytes;
        quant_2_size = hidden_size * num_experts_per_node * dtype_bytes;
    }
    else if (is_int_groupwise_w_quant)
    {
        quant_1_size = fc1_out_size * num_experts_per_node * dtype_bytes * hidden_size / mGroupSize;
        quant_2_size = hidden_size * num_experts_per_node * dtype_bytes * inter_size / mGroupSize;
    }

    // FP8 sizes
    quant_1_size = is_fp8_w_quant ? num_experts_per_node * sizeof(float) : quant_1_size;
    quant_2_size = is_fp8_w_quant ? sizeof(float) : quant_2_size;
    size_t quant_3_size = is_fp8_w_quant ? num_experts_per_node * sizeof(float) : 0;
    size_t quant_4_size = 0; // Currently ignored by the GEMM
    if (is_int_groupwise_w_quant)
    {
        quant_3_size = quant_1_size;
        quant_4_size = quant_2_size;
    }

    // FP4 sizes
    quant_1_size = is_fp4_w_quant ? sizeof(float) : quant_1_size;
    quant_2_size = is_fp4_w_quant ? getOffsetWeightSF(num_experts_per_node, inter_size, hidden_size)
            * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF)
                                  : quant_2_size;
    quant_3_size = is_fp4_w_quant ? num_experts_per_node * sizeof(float) : quant_3_size;
    quant_4_size = is_fp4_w_quant ? sizeof(float) : quant_4_size;
    size_t quant_5_size = is_fp4_w_quant ? getOffsetWeightSF(num_experts_per_node, hidden_size, inter_size)
            * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF)
                                         : 0;
    size_t quant_6_size = is_fp4_w_quant ? num_experts_per_node * sizeof(float) : 0;

    size_t tma_ws_input_workspace_size = 0;
    if (is_tma_ws_input)
    {
        tma_ws_input_workspace_size
            = TmaWarpSpecializedGroupedGemmInput::workspaceSize(num_experts_per_node) * (NUM_ROUTING_SAMPLES + 1);

        if (is_w4afp8_quant)
        {
            quant_3_size = 0;
            quant_4_size = 0;
        }
    }

    auto act_sf_rows = mMinLatencyMode
        ? num_expanded_tokens
        : std::min(num_expanded_tokens, static_cast<size_t>(maxM * num_experts_per_node));
    size_t const fc1_fp4_act_scale_size = is_fp4_act_quant
        ? getOffsetActivationSF(num_experts_per_node, act_sf_rows, hidden_size)
            * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF)
        : 0;
    size_t const fc2_fp4_act_scale_size = is_fp4_act_quant
        ? getOffsetActivationSF(num_experts_per_node, act_sf_rows, inter_size)
            * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF)
        : 0;
    size_t const fp4_act_scale_flat_size = std::max(fc1_fp4_act_scale_size, fc2_fp4_act_scale_size);

    size_t w4a8_alpha_size = is_w4afp8_quant ? num_experts_per_node * sizeof(float) : 0;
    size_t alpha_scale_ptr_array_size = num_experts_per_node * sizeof(float**);
    size_t gemm_workspace_size = mInterface->getGemmWorkspaceSize(num_experts_per_node);

    // Routing info
    size_t expert_first_token_offset_size = (num_experts_per_node + 1) * sizeof(int64_t) * NUM_ROUTING_SAMPLES;
    size_t map_size = mMinLatencyMode ? 0 : NUM_ROUTING_SAMPLES * num_expanded_tokens * sizeof(int);
    size_t unpermuted_size = mMinLatencyMode ? 0 : NUM_ROUTING_SAMPLES * num_expanded_tokens * sizeof(int);
    size_t permuted_size = mMinLatencyMode ? 0 : num_expanded_tokens * sizeof(int);
    size_t sorter_ws_size = mMinLatencyMode ? 0 : mSorter.getWorkspaceSize(num_expanded_tokens, mNumExpertsPerNode);
    size_t token_topk_unpermuted_scales_size = mMinLatencyMode ? 0 : num_expanded_tokens * sizeof(float);

    // The follow buffers are used in min_latency_mode
    size_t num_active_experts_per_node_size
        = mMinLatencyMode ? sizeof(int) * NUM_ROUTING_SAMPLES : 0; // smaller than or equal to num_experts_per_node
    size_t active_expert_global_ids_size = mMinLatencyMode ? mNumExpertsPerNode * sizeof(int) * NUM_ROUTING_SAMPLES : 0;

    size_t map_offset = 0;
    std::map<std::string, std::pair<size_t, size_t>> out_map;

#define ADD_NAME(name, size)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        auto aligned_size = tensorrt_llm::common::alignSize(size, tensorrt_llm::common::kCudaMemAlign);                \
        out_map[#name] = std::pair{aligned_size, map_offset};                                                          \
        map_offset += aligned_size;                                                                                    \
    } while (false)
#define ADD(name) ADD_NAME(name, name##_size)

    ADD(expert_first_token_offset);
    ADD_NAME(source_to_dest, map_size);
    ADD_NAME(dest_to_source, map_size);
    ADD_NAME(unpermuted_selected_experts, unpermuted_size);
    ADD_NAME(unpermuted_source_rows, unpermuted_size);
    ADD_NAME(permuted_token_selected_experts, permuted_size);
    ADD(sorter_ws);
    ADD(token_topk_unpermuted_scales);
    ADD(num_active_experts_per_node);
    ADD(active_expert_global_ids);
    ADD(input);
    ADD(output);
    ADD(intermediate);
    ADD(weights);
    ADD(bias);
    ADD(quant_1);
    ADD(quant_2);
    ADD(quant_3);
    ADD(quant_4);
    ADD(quant_5);
    ADD(quant_6);
    ADD(tma_ws_input_workspace);
    ADD(w4a8_alpha);
    ADD(alpha_scale_ptr_array);
    ADD(fp4_act_scale_flat);
    ADD(gemm_workspace);

#undef ADD_NAME
#undef ADD

    return out_map;
}

void GemmProfilerBackend::prepareRouting(int num_tokens, char* workspace_ptr_char, cudaStream_t stream)
{
    auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);
#define GET_WS_PTR_BASE(type, name)                                                                                    \
    auto* name##_base                                                                                                  \
        = (workspaces.at(#name).first ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second)       \
                                      : nullptr)
#define GET_WS_PTR(type, name)                                                                                         \
    auto* name                                                                                                         \
        = (workspaces.at(#name).first ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second)       \
                                      : nullptr)

    GET_WS_PTR_BASE(int64_t*, expert_first_token_offset);
    GET_WS_PTR_BASE(int*, source_to_dest);
    GET_WS_PTR_BASE(int*, dest_to_source);
    GET_WS_PTR_BASE(int*, unpermuted_selected_experts);
    GET_WS_PTR_BASE(int*, unpermuted_source_rows);
    GET_WS_PTR(int*, permuted_token_selected_experts);
    GET_WS_PTR(int*, sorter_ws);
    GET_WS_PTR(int*, num_active_experts_per_node);
    GET_WS_PTR(int*, active_expert_global_ids);

#undef GET_WS_PTR_BASE
#undef GET_WS_PTR

    if (mMinLatencyMode)
    {
        // expert_first_token_offset for each sample
        TLLM_CHECK_WITH_INFO(mNumExpertsPerNode <= 256, "Min latency mode only supports #experts < 256");
        prepareMinLatencyBuffer<256, NUM_ROUTING_SAMPLES>
            <<<NUM_ROUTING_SAMPLES, 256, mNumExpertsPerNode * (sizeof(float) + sizeof(int)), stream>>>(
                num_active_experts_per_node, active_expert_global_ids, expert_first_token_offset_base, num_tokens, mK,
                mNumExpertsPerNode);
    }
    else
    {
        int64_t num_expanded_tokens = num_tokens * mK;
        uint32_t num_threads = 256;
        dim3 grid_dim{(num_tokens + num_threads - 1) / num_threads, NUM_ROUTING_SAMPLES, 1};
        prepareFakeRouterBuffers<<<grid_dim, num_threads, 0, stream>>>(unpermuted_source_rows_base,
            unpermuted_selected_experts_base, num_tokens, mK, mNumExperts, mNumExpertsPerNode);
        sync_check_cuda_error(stream);

        for (int64_t i = 0; i < NUM_ROUTING_SAMPLES; i++)
        {
            int64_t* expert_first_token_offset = expert_first_token_offset_base + i * (mNumExpertsPerNode + 1);
            int* source_to_dest = source_to_dest_base + i * num_expanded_tokens;
            int* dest_to_source = dest_to_source_base + i * num_expanded_tokens;
            int* unpermuted_expert_selection = unpermuted_selected_experts_base + i * num_expanded_tokens;
            int* unpermuted_source_rows = unpermuted_source_rows_base + i * num_expanded_tokens;

            generateTokenPermutation(unpermuted_expert_selection, unpermuted_source_rows,
                permuted_token_selected_experts, dest_to_source, expert_first_token_offset, num_tokens,
                mNumExpertsPerNode, mK, mSorter, sorter_ws, stream);

            sync_check_cuda_error(stream);

            int grid_dim = (num_expanded_tokens + num_threads - 1) / num_threads;
            buildReverseMap<<<grid_dim, num_threads, 0, stream>>>(source_to_dest, dest_to_source, num_expanded_tokens);
        }
    }
}

void GemmProfilerBackend::prepareQuantParams(int num_tokens, char* workspace_ptr_char, cudaStream_t)
{
    auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);
#define GET_WS_PTR(type, name)                                                                                         \
    auto* name                                                                                                         \
        = (workspaces.at(#name).first ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second)       \
                                      : nullptr)
    GET_WS_PTR(void const*, quant_1);
    GET_WS_PTR(void const*, quant_2);
    GET_WS_PTR(void const*, quant_3);
    GET_WS_PTR(void const*, quant_4);
    GET_WS_PTR(void const*, quant_5);
    GET_WS_PTR(void const*, quant_6);
    GET_WS_PTR(float const*, w4a8_alpha);
#undef GET_WS_PTR

    if ((mWType == nvinfer1::DataType::kINT8 || mWType == nvinfer1::DataType::kINT4) && mGroupSize < 0)
    {
        TLLM_CHECK(quant_1 && quant_2);
        mQuantParams = QuantParams::Int(quant_1, quant_2);
    }
    else if (mWType == nvinfer1::DataType::kINT4)
    {
        TLLM_CHECK(quant_1 && quant_2);
        if (mDType == nvinfer1::DataType::kFP8)
        {
            TLLM_CHECK(w4a8_alpha);
            mQuantParams = QuantParams::GroupWise(
                mGroupSize, quant_1, quant_2, nullptr, nullptr, quant_3, quant_4, w4a8_alpha, w4a8_alpha);
        }
        else
        {
            mQuantParams = QuantParams::GroupWise(mGroupSize, quant_1, quant_2, nullptr, nullptr, quant_3, quant_4);
        }
    }
    else if (mWType == nvinfer1::DataType::kFP8)
    {
        TLLM_CHECK(quant_1 && quant_2 && quant_3);
        mQuantParams = QuantParams::FP8(static_cast<float const*>(quant_1), static_cast<float const*>(quant_2),
            static_cast<float const*>(quant_3), static_cast<float const*>(quant_4));
    }
    else if (mWType == nvinfer1::DataType::kFP4 || mWType == nvinfer1::DataType::kINT64)
    {
        // nvllm still uses int64 because torch doesn't have fp4 yet.
        TLLM_CHECK(quant_1 && quant_2 && quant_3 && quant_4 && quant_5 && quant_6);
        mQuantParams = QuantParams::FP4(static_cast<float const*>(quant_1),
            static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF const*>(quant_2),
            static_cast<float const*>(quant_3), static_cast<float const*>(quant_4),
            static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF const*>(quant_5),
            static_cast<float const*>(quant_6));
    }
}

void GemmProfilerBackend::prepareTmaWsInputs(
    int num_tokens, char* workspace_ptr_char, void const* expert_weights, cudaStream_t stream)
{
    if (mSM < 90)
    {
        return;
    }

    auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);

#define GET_WS_PTR(type, name)                                                                                         \
    auto* name                                                                                                         \
        = (workspaces.at(#name).first ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second)       \
                                      : nullptr)

    GET_WS_PTR(int64_t*, expert_first_token_offset);
    int64_t* expert_first_token_offset_base = expert_first_token_offset;
    GET_WS_PTR(int*, dest_to_source);
    int* dest_to_source_base = dest_to_source;
    GET_WS_PTR(void*, input);
    GET_WS_PTR(void*, output);
    GET_WS_PTR(void*, intermediate);
    GET_WS_PTR(void*, weights);
    TLLM_CHECK(mNeedWeights == (expert_weights == nullptr));
    void const* weights_sel = mNeedWeights ? weights : expert_weights;
    GET_WS_PTR(void*, bias);
    GET_WS_PTR(float*, token_topk_unpermuted_scales);
    GET_WS_PTR(int8_t*, tma_ws_input_workspace);
    GET_WS_PTR(void*, gemm_workspace);
    GET_WS_PTR(float*, alpha_scale_ptr_array);
    GET_WS_PTR(TmaWarpSpecializedGroupedGemmInput::ElementSF*, fp4_act_scale_flat);
    GET_WS_PTR(int*, num_active_experts_per_node);
    GET_WS_PTR(int*, active_expert_global_ids);

#undef GET_WS_PTR

    size_t tma_ws_size = TmaWarpSpecializedGroupedGemmInput::workspaceSize(mNumExpertsPerNode);

    TmaWarpSpecializedGroupedGemmInput dummy_tma_ws_input;
    dummy_tma_ws_input.configureWorkspace(
        tma_ws_input_workspace, mNumExpertsPerNode, gemm_workspace, workspaces.at("gemm_workspace").first);
    tma_ws_input_workspace += tma_ws_size;

    size_t num_expanded_tokens = num_tokens * mK;
    for (int64_t i = 0; i < NUM_ROUTING_SAMPLES; i++)
    {
        mTmaInputCache[i].configureWorkspace(
            tma_ws_input_workspace, mNumExpertsPerNode, gemm_workspace, workspaces.at("gemm_workspace").first);
        tma_ws_input_workspace += tma_ws_size;

        int64_t* expert_first_token_offset = expert_first_token_offset_base + i * (mNumExpertsPerNode + 1);
        int* expanded_dest_row_to_expanded_source_row = dest_to_source_base + i * num_expanded_tokens;

        auto& gemm1_tma_ws_input = mGemmToProfile == GemmToProfile::GEMM_1 ? mTmaInputCache[i] : dummy_tma_ws_input;
        auto& gemm2_tma_ws_input = mGemmToProfile == GemmToProfile::GEMM_2 ? mTmaInputCache[i] : dummy_tma_ws_input;
        if (mSM >= 90)
        {
            /* GEMM1 */
            gemm1_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
            gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;

            bool apply_bias = true;
            bool use_w4afp8 = (mDType == nvinfer1::DataType::kFP8 && mWType == nvinfer1::DataType::kINT4);
            bool using_fused_finalize
                = !mInterface->use_deterministic_hopper_reduce_ && mSM == 90 && !mMinLatencyMode && !use_w4afp8;
            if (using_fused_finalize)
            {
                assert(!mMinLatencyMode);
                gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
                gemm2_tma_ws_input.setFinalizeFusionParams(output, token_topk_unpermuted_scales,
                    expert_first_token_offset, expanded_dest_row_to_expanded_source_row, apply_bias ? bias : nullptr,
                    mExpertHiddenSize, num_tokens);
            }

            auto fc1_output_size = isGatedActivation(mActivationType) ? mExpertInterSize * 2 : mExpertInterSize;
            if (mMinLatencyMode)
            {
                std::tie(gemm1_tma_ws_input, gemm2_tma_ws_input)
                    = mInterface->computeStridesTmaWarpSpecializedLowLatencyDispatch(gemm1_tma_ws_input,
                        gemm2_tma_ws_input, num_tokens, fc1_output_size, mExpertHiddenSize, mExpertHiddenSize,
                        mExpertInterSize, mNumExpertsPerNode, input, input, weights_sel, weights_sel,
                        mQuantParams.fp8.dequant_fc1, mQuantParams.fp8.dequant_fc2, fp4_act_scale_flat,
                        fp4_act_scale_flat, mQuantParams, nullptr, nullptr, intermediate, intermediate,
                        num_active_experts_per_node, active_expert_global_ids, 0, stream);
            }
            else
            {
                std::tie(gemm1_tma_ws_input, gemm2_tma_ws_input) = mInterface->computeStridesTmaWarpSpecializedDispatch(
                    expert_first_token_offset, gemm1_tma_ws_input, gemm2_tma_ws_input, num_tokens, num_tokens * mK,
                    fc1_output_size, mExpertHiddenSize, mExpertHiddenSize, mExpertInterSize, mNumExpertsPerNode, input,
                    input, weights_sel, weights_sel, mQuantParams.fp8.dequant_fc1, mQuantParams.fp8.dequant_fc2,
                    fp4_act_scale_flat, fp4_act_scale_flat, mQuantParams, nullptr, nullptr, intermediate, intermediate,
                    stream);
            }
            sync_check_cuda_error(stream);
        }
    }
}

void GemmProfilerBackend::prepare(
    int num_tokens, char* workspace_ptr_char, void const* expert_weights, cudaStream_t stream)
{
    mAllTacticsSaved = mInterface->getTactics();
    mSampleIndex = 0;

    mSorter.updateNumExperts(mNumExpertsPerNode);

    auto workspace_size = getWorkspaceSize(num_tokens);
    populateRandomBuffer(workspace_ptr_char, workspace_size, stream);

    prepareRouting(num_tokens, workspace_ptr_char, stream);
    prepareQuantParams(num_tokens, workspace_ptr_char, stream);
    prepareTmaWsInputs(num_tokens, workspace_ptr_char, expert_weights, stream);
}

size_t GemmProfilerBackend::getWorkspaceSize(int maxM)
{
    auto sizes_map = getProfilerWorkspaces(maxM, mSM >= 90);
    std::vector<size_t> sizes(sizes_map.size());
    std::transform(sizes_map.begin(), sizes_map.end(), sizes.begin(), [](auto& v) { return v.second.first; });
    size_t size = calculateTotalWorkspaceSize(sizes.data(), sizes.size());
    TLLM_LOG_TRACE("MOE profiler workspace size: %zu", size);
    return size;
}

void GemmProfilerBackend::runProfiler(int original_num_tokens, Config const& tactic, char* workspace_ptr_char,
    void const* expert_weights, cudaStream_t const& stream)
{
    int64_t expanded_num_tokens = original_num_tokens * mK;
    int64_t num_experts_per_node = mNumExpertsPerNode;

    mSampleIndex = (mSampleIndex + 1) % NUM_ROUTING_SAMPLES;

    auto workspaces = getProfilerWorkspaces(original_num_tokens, tactic.is_tma_warp_specialized);

#define GET_WS_PTR_OFFSET(type, name, offset)                                                                          \
    auto* name = (workspaces.at(#name).first                                                                           \
            ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) + (offset)                      \
            : nullptr)
#define GET_WS_PTR(type, name)                                                                                         \
    auto* name                                                                                                         \
        = (workspaces.at(#name).first ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second)       \
                                      : nullptr)

    GET_WS_PTR_OFFSET(int64_t const*, expert_first_token_offset, (mSampleIndex * (mNumExpertsPerNode + 1)));
    GET_WS_PTR_OFFSET(int const*, source_to_dest, (mSampleIndex * expanded_num_tokens));
    GET_WS_PTR_OFFSET(int const*, dest_to_source, (mSampleIndex * expanded_num_tokens));
    GET_WS_PTR_OFFSET(int const*, unpermuted_selected_experts, (mSampleIndex * expanded_num_tokens));

    GET_WS_PTR(float const*, token_topk_unpermuted_scales);
    auto const* token_topk_permuted_scales = token_topk_unpermuted_scales;

    GET_WS_PTR_OFFSET(int*, num_active_experts_per_node, mSampleIndex);
    GET_WS_PTR_OFFSET(int*, active_expert_global_ids, (mSampleIndex * mNumExpertsPerNode));
    GET_WS_PTR(void const*, input);
    GET_WS_PTR(void*, output);
    GET_WS_PTR(void*, intermediate);
    GET_WS_PTR(void const*, weights);
    TLLM_CHECK(mNeedWeights == (expert_weights == nullptr));
    void const* weights_sel = mNeedWeights ? weights : expert_weights;
    GET_WS_PTR(void const*, bias);

    GET_WS_PTR(float const**, alpha_scale_ptr_array);
    GET_WS_PTR(TmaWarpSpecializedGroupedGemmInput::ElementSF*, fp4_act_scale_flat);
    GET_WS_PTR(void*, gemm_workspace);

#undef GET_WS_PTR_OFFSET
#undef GET_WS_PTR

    TmaWarpSpecializedGroupedGemmInput tma_ws_input_template;
    if (tactic.is_tma_warp_specialized)
    {
        tma_ws_input_template = mTmaInputCache[mSampleIndex];
    }

    mInterface->is_profiler = true;
    if (mGemmToProfile == GemmToProfile::GEMM_1)
    {
        mInterface->gemm1(input,                              //
            output,                                           //
            intermediate,                                     //
            expert_first_token_offset,                        //
            tma_ws_input_template,                            //
            weights_sel,                                      //
            bias,                                             //
            expert_first_token_offset + num_experts_per_node, //
            mQuantParams.wo.fc1_weight_scales,                //
            mQuantParams.fp8.dequant_fc1,                     //
            mQuantParams.fp8.quant_fc2,                       //
            fp4_act_scale_flat,                               //
            fp4_act_scale_flat,                               //
            mQuantParams,                                     //
            original_num_tokens,                              //
            expanded_num_tokens,                              //
            mExpertHiddenSize,                                //
            mExpertInterSize,                                 //
            num_experts_per_node,                             //
            mActivationType,                                  //
            alpha_scale_ptr_array,                            //
            !mUseLora,                                        //
            /*use_fp8_block_scaling=*/false,                  //
            stream,                                           //
            tactic,                                           //
            mMinLatencyMode,                                  //
            num_active_experts_per_node,                      //
            active_expert_global_ids,                         //
            /*start_expert=*/0);
    }
    else
    {
        TLLM_CHECK(mGemmToProfile == GemmToProfile::GEMM_2);
        mInterface->gemm2(input,                            //
            intermediate,                                   //
            output,                                         //
            expert_first_token_offset,                      //
            tma_ws_input_template,                          //
            weights_sel,                                    //
            bias,                                           //
            mQuantParams.wo.fc2_weight_scales,              //
            mQuantParams.fp8.dequant_fc2,                   //
            fp4_act_scale_flat,                             //
            mQuantParams,                                   //
            token_topk_unpermuted_scales,                   //
            token_topk_permuted_scales,                     //
            source_to_dest,                                 //
            dest_to_source,                                 //
            unpermuted_selected_experts,                    //
            expert_first_token_offset + mNumExpertsPerNode, //
            original_num_tokens,                            //
            expanded_num_tokens,                            //
            mExpertHiddenSize,                              //
            mExpertInterSize,                               //
            num_experts_per_node,                           //
            mK,                                             //
            alpha_scale_ptr_array,                          //
            false,                                          //
            nullptr,                                        //
            /*use_fp8_block_scaling=*/false,                //
            stream,                                         //
            mParallelismConfig,                             //
            mEnableAlltoall,                                //
            tactic,                                         //
            mMinLatencyMode,                                //
            num_active_experts_per_node,                    //
            active_expert_global_ids,                       //
            /*start_expert=*/0);
    }
    mInterface->is_profiler = false;

    sync_check_cuda_error(stream);
}

// ==================== Variable batched GEMM specializations ==================================
template class CutlassMoeFCRunner<float, float>;

#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_bfloat16, uint8_t>;
template class CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>;
#endif

template class CutlassMoeFCRunner<half, half>;
template class CutlassMoeFCRunner<half, uint8_t>;
template class CutlassMoeFCRunner<half, cutlass::uint4b_t>;
#ifdef ENABLE_FP8
// template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, half>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>;
#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>;
#endif
#endif
#ifdef ENABLE_FP4
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, half>;
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, half, half>;
#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>;
#endif
#endif

} // namespace tensorrt_llm::kernels::cutlass_kernels
