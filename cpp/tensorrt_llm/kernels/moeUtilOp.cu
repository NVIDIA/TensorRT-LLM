/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "cutlass_kernels/include/moe_kernels.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"
#include "tensorrt_llm/kernels/moeUtilOp.h"
#include "tensorrt_llm/kernels/quantization.cuh"

#include <cuda_fp16.h>
#include <float.h>

#include <climits> // For INT_MAX
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda/std/limits> // For numeric_limits
#include <math.h>

#include <cutlass/array.h>
#include <cutlass/half.h>
#include <cutlass/numeric_types.h>

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

namespace cg = cooperative_groups;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels
{

// TODO: These kernel implementations are duplicated in moe_kernels.cu. They will be refactored later (tracked by
// https://jirasw.nvidia.com/browse/TRTLLM-708)
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
    assert(end_expert <= num_experts_per_node);
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

__host__ __device__ constexpr int64_t getOffsetActivationSF(int64_t expert_id, int64_t token_offset, int64_t gemm_k,
    cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType scaling_type)
{
    auto function = [=](int64_t min_alignment, int64_t block_size)
    {
        // This formulation ensures that sf_offset[i + 1] - sf_offset[i] >= token_offset[i + 1] - token_offset[i].
        int64_t sf_offset = (token_offset + expert_id * (min_alignment - 1)) / min_alignment * min_alignment;
        assert(gemm_k % block_size == 0);
        return sf_offset * gemm_k / block_size;
    };
    switch (scaling_type)
    {
    case cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX:
        return function(cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::MinNumRowsAlignmentMXFPX,
            cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize);
    case cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4:
        return function(cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::MinNumRowsAlignmentNVFP4,
            cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize);
    case cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE:
        return 0; // No scaling factors, no offset
    }

    assert(false && "Unrecognized scaling type");
    return 0;
}

constexpr static int NVFP4_VEC_SIZE = 16;

template <class GemmOutputType, class ComputeElem>
__device__ uint32_t quantizePackedFP4Value(ComputeElem& post_act_val, float global_scale_val,
    int64_t num_tokens_before_expert, int64_t expert_id, int64_t token_id, int64_t elem_idx, int64_t num_cols,
    int64_t max_tokens_per_expert, cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF* act_sf_flat,
    cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType scaling_type)
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
    auto act_sf_expert
        = act_sf_flat + getOffsetActivationSF(expert_id, num_tokens_before_expert, num_cols, scaling_type);

    // Use `token - num_tokens_before_expert` because we want this to be relative to the start of this expert
    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF,
        CVT_FP4_NUM_THREADS_PER_SF, NVFP4_VEC_SIZE>(std::nullopt /* batchIdx */, token_id - num_tokens_before_expert,
        elem_idx, std::nullopt /* numRows */, num_cols, act_sf_expert, FP4QuantizationSFLayout::SWIZZLED);

    // Do the conversion and set the output and scaling factor
    auto func = (scaling_type == cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4)
        ? &cvt_warp_fp16_to_fp4<GemmOutputType, NVFP4_VEC_SIZE, false>
        : &cvt_warp_fp16_to_fp4<GemmOutputType, NVFP4_VEC_SIZE, true>;
    auto res = func(packed_vec, global_scale_val, sf_out);
    return res;
}

__device__ void writeSF(int64_t num_tokens_before_expert, int64_t expert_id, int64_t source_token_id, int64_t token_id,
    int64_t elem_idx, int64_t num_cols, int64_t max_tokens_per_expert,
    cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF* act_sf_flat,
    cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf)
{
    static constexpr int CVT_FP4_NUM_THREADS_PER_SF = NVFP4_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;

    // We need to offset into the scaling factors for just this expert
    auto act_sf_expert = act_sf_flat
        + getOffsetActivationSF(expert_id, num_tokens_before_expert, num_cols,
            cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4);

    // Use `token - num_tokens_before_expert` because we want this to be relative to the start of this expert
    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF,
        CVT_FP4_NUM_THREADS_PER_SF, NVFP4_VEC_SIZE>(std::nullopt /* batchIdx */, token_id - num_tokens_before_expert,
        elem_idx, std::nullopt /* numRows */, num_cols, act_sf_expert, FP4QuantizationSFLayout::SWIZZLED);
    if (sf_out)
    {
        auto const sf_in
            = cvt_quant_to_fp4_get_sf_out_offset<cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF,
                CVT_FP4_NUM_THREADS_PER_SF, NVFP4_VEC_SIZE>(std::nullopt /* batchIdx */, source_token_id, elem_idx,
                std::nullopt /* numRows */, num_cols,
                const_cast<cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF*>(input_sf),
                FP4QuantizationSFLayout::SWIZZLED);
        *sf_out = *sf_in;
    }
}

void generateTokenPermutation(int const* unpermuted_token_selected_experts, int const* unpermuted_source_token_ids,
    int* permuted_token_selected_experts, int* permuted_source_token_ids, int64_t* expert_first_token_offset,
    int64_t num_rows, int64_t num_experts_per_node, int64_t k, cutlass_kernels::CubKeyValueSorter& sorter,
    void* sorter_ws, cudaStream_t stream)
{
    int64_t const expanded_num_rows = k * num_rows;
    sorter.updateNumExperts(num_experts_per_node);
    size_t const sorter_ws_size_bytes
        = cutlass_kernels::pad_to_multiple_of_16(sorter.getWorkspaceSize(expanded_num_rows, num_experts_per_node));
    sorter.run((void*) sorter_ws, sorter_ws_size_bytes, unpermuted_token_selected_experts,
        permuted_token_selected_experts, unpermuted_source_token_ids, permuted_source_token_ids, expanded_num_rows,
        stream);

    sync_check_cuda_error(stream);

    // Upper bound on number of expanded rows
    computeExpertFirstTokenOffset(
        permuted_token_selected_experts, expanded_num_rows, num_experts_per_node, expert_first_token_offset, stream);
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

// // Duplicated and permutes rows for MoE. In addition, reverse the permutation map to help with finalizing routing.

// // "expanded_x_row" simply means that the number of values is num_rows x k. It is "expanded" since we will have to
// // duplicate some rows in the input matrix to match the dimensions. Duplicates will always get routed to separate
// // experts in the end.

// // Note that the expanded_dest_row_to_expanded_source_row map referred to here has indices in the range (0,
// // k*rows_in_input - 1). However, it is set up so that index 0, rows_in_input, 2*rows_in_input ...
// (k-1)*rows_in_input
// // all map to row 0 in the original matrix. Thus, to know where to read in the source matrix, we simply take the
// modulus
// // of the expanded index.

// constexpr static int EXPAND_THREADS_PER_BLOCK = 256;

// template <class InputActivationsType, class ExpandedActivationsType, bool CHECK_SKIPPED>
// __global__ void expandInputRowsKernel(InputActivationsType const* unpermuted_input,
//     ExpandedActivationsType* permuted_output, float const* unpermuted_scales, float* permuted_scales,
//     int const* expanded_dest_row_to_expanded_source_row, int* expanded_source_row_to_expanded_dest_row,
//     int64_t const num_rows, int64_t const* num_dest_rows, int64_t const cols, int64_t k,
//     float const* fc1_act_global_scale, int64_t* expert_first_token_offset,
//     cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
//     cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, int64_t num_experts_per_node)
// {
// #ifdef ENABLE_FP4
//     constexpr bool is_fp4 = std::is_same_v<ExpandedActivationsType, __nv_fp4_e2m1>;
//     constexpr bool is_fp4_input = is_fp4 && std::is_same_v<InputActivationsType, __nv_fp4_e2m1>;
//     constexpr bool need_fp4_quant = is_fp4 && !std::is_same_v<InputActivationsType, __nv_fp4_e2m1>;
// #else
//     constexpr bool is_fp4 = false;
//     constexpr bool is_fp4_input = false;
//     constexpr bool need_fp4_quant = false;
// #endif

//     static_assert(need_fp4_quant || std::is_same_v<InputActivationsType, ExpandedActivationsType>,
//         "Only FP4 quantization supports outputting a different format as part of the expansion");

//     // Reverse permutation map.
//     // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need
//     the
//     // reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
//     // thread block will be responsible for all k summations.
//     int64_t const expanded_dest_row = blockIdx.x;
// #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
//     asm volatile("griddepcontrol.wait;");
// #endif
//     int64_t const expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
//     if (threadIdx.x == 0)
//     {
//         assert(expanded_dest_row <= INT32_MAX);
//         expanded_source_row_to_expanded_dest_row[expanded_source_row] = static_cast<int>(expanded_dest_row);
//     }

//     if (!CHECK_SKIPPED || blockIdx.x < *num_dest_rows)
//     {
//         // Load 128-bits per thread
//         constexpr int64_t ELEM_PER_THREAD
//             = is_fp4 ? CVT_FP4_ELTS_PER_THREAD : (128 / sizeof_bits<InputActivationsType>::value);
//         constexpr int64_t ELEM_PER_BYTE = is_fp4_input ? 2 : 1;
//         using DataElem
//             = std::conditional_t<is_fp4_input, uint32_t, cutlass::Array<InputActivationsType, ELEM_PER_THREAD>>;
//         using OutputElem = std::conditional_t<is_fp4, uint32_t, DataElem>;

//         // Duplicate and permute rows
//         int64_t const source_k_rank = expanded_source_row / num_rows;
//         int64_t const source_row = expanded_source_row % num_rows;

//         auto const* source_row_ptr
//             = reinterpret_cast<DataElem const*>(unpermuted_input + source_row * cols / ELEM_PER_BYTE);
//         // Cast first to handle when this is FP4
//         auto* dest_row_ptr
//             = reinterpret_cast<OutputElem*>(permuted_output) + expanded_dest_row * cols / ELEM_PER_THREAD;

//         int64_t const start_offset = threadIdx.x;
//         int64_t const stride = EXPAND_THREADS_PER_BLOCK;
//         int64_t const num_elems_in_col = cols / ELEM_PER_THREAD;
//         assert(cols % ELEM_PER_THREAD == 0);

//         if constexpr (is_fp4)
//         {
//             int64_t expert = findTotalEltsLessThanTarget(
//                                  expert_first_token_offset, num_experts_per_node, (int64_t) expanded_dest_row + 1)
//                 - 1;
//             float global_scale_val = fc1_act_global_scale ? *fc1_act_global_scale : 1.0f;
//             int64_t num_tokens_before_expert = expert_first_token_offset[expert];

//             for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
//             {
//                 auto in_vec = source_row_ptr[elem_index];
//                 if constexpr (need_fp4_quant)
//                 {
//                     // auto res = quantizePackedFP4Value<InputActivationsType, DataElem>(in_vec, global_scale_val,
//                     //     num_tokens_before_expert, expert, expanded_dest_row, elem_index, cols, num_rows,
//                     //     fc1_act_sf_flat);
//                     auto res = quantizePackedFP4Value<InputActivationsType, DataElem>(in_vec, global_scale_val,
//                         num_tokens_before_expert, expert, expanded_dest_row, elem_index, cols, num_rows,
//                         fc1_act_sf_flat,
//                         cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4);
//                     dest_row_ptr[elem_index] = res;
//                 }
//                 else
//                 {
//                     writeSF(num_tokens_before_expert, expert, source_row, expanded_dest_row, elem_index, cols,
//                     num_rows,
//                         fc1_act_sf_flat, input_sf);
//                     dest_row_ptr[elem_index] = in_vec;
//                 }
//             }
//         }
//         else
//         {
//             for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
//             {
//                 dest_row_ptr[elem_index] = source_row_ptr[elem_index];
//             }
//         }

//         if (permuted_scales && threadIdx.x == 0)
//         {
//             int64_t const source_k_idx = source_row * k + source_k_rank;
//             permuted_scales[expanded_dest_row] = unpermuted_scales ? unpermuted_scales[source_k_idx] : 1.0f;
//         }
//     }
// #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
//     asm volatile("griddepcontrol.launch_dependents;");
// #endif
// }

// template <class InputActivationsType, class ExpandedActivationsType>
// void expandInputRowsKernelLauncher(InputActivationsType const* unpermuted_input,
//     ExpandedActivationsType* permuted_output, float const* unpermuted_scales, float* permuted_scales,
//     int const* expanded_dest_row_to_expanded_source_row, int* expanded_source_row_to_expanded_dest_row,
//     int64_t const num_rows, int64_t const* num_valid_tokens_ptr, int64_t const cols, int const k,
//     int const num_experts_per_node, float const* fc1_act_global_scale, int64_t* expert_first_token_offset,
//     cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
//     cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, cudaStream_t stream)
// {
//     if (fc1_act_sf_flat)
//     {
//         assert(false && "Not supported, we need to keep the same as moe_kerenls.cu in the future (TODO).");
//     }

//     int64_t const blocks = num_rows * k;
//     int64_t const threads = EXPAND_THREADS_PER_BLOCK;
//     auto func = (num_valid_tokens_ptr != nullptr)
//         ? expandInputRowsKernel<InputActivationsType, ExpandedActivationsType, true>
//         : expandInputRowsKernel<InputActivationsType, ExpandedActivationsType, false>;

//     cudaLaunchConfig_t config;
//     config.gridDim = blocks;
//     config.blockDim = threads;
//     config.dynamicSmemBytes = 0;
//     config.stream = stream;
//     cudaLaunchAttribute attrs[1];
//     attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
//     attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
//     config.numAttrs = 1;
//     config.attrs = attrs;
//     cudaLaunchKernelEx(&config, func, unpermuted_input, permuted_output, unpermuted_scales, permuted_scales,
//         expanded_dest_row_to_expanded_source_row, expanded_source_row_to_expanded_dest_row, num_rows,
//         num_valid_tokens_ptr, cols, k, fc1_act_global_scale, expert_first_token_offset, fc1_act_sf_flat, input_sf,
//         num_experts_per_node);
// }

// #define INSTANTIATE_EXPAND_INPUT_ROWS(InputActivationsType, ExpandedActivationsType)                                   \
//     template void expandInputRowsKernelLauncher<InputActivationsType, ExpandedActivationsType>(                        \
//         InputActivationsType const* unpermuted_input, ExpandedActivationsType* permuted_output,                        \
//         float const* unpermuted_scales, float* permuted_scales, int const* expanded_dest_row_to_expanded_source_row,   \
//         int* expanded_source_row_to_expanded_dest_row, int64_t const num_rows, int64_t const* num_valid_tokens_ptr,    \
//         int64_t const cols, int const k, int const num_experts_per_node, float const* fc1_act_global_scale,            \
//         int64_t* expert_first_token_offset,                                                                            \
//         cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,                               \
//         cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, cudaStream_t stream);

// INSTANTIATE_EXPAND_INPUT_ROWS(half, half);
// INSTANTIATE_EXPAND_INPUT_ROWS(float, float);
// #ifdef ENABLE_BF16
// INSTANTIATE_EXPAND_INPUT_ROWS(__nv_bfloat16, __nv_bfloat16);
// #endif

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
    int64_t const* expert_first_token_offset,
    cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
    cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, int64_t const num_experts_per_node)
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
                        fc1_act_sf_flat,
                        cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4);
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
    cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
    cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, cudaStream_t stream)
{
#ifdef ENABLE_FP4
    // TODO Currently this is a bit hacky because we assume we are in FP8_MXFP4 mode if activations are FP8.
    //   This code is still needed if we add MXFP8_MXFP4 mode.
    // TODO This is also wasteful, we should solve this properly by properly writing the padding in the kernel
    if (fc1_act_sf_flat && std::is_same_v<ExpandedActivationsType, __nv_fp4_e2m1>)
    {
        size_t num_elems = getOffsetActivationSF(num_experts_per_node, num_rows * std::min(k, num_experts_per_node),
            cols, cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4);
        check_cuda_error(cudaMemsetAsync(fc1_act_sf_flat, 0x0,
            num_elems * sizeof(cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF), stream));
    }
#endif

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

#define INSTANTIATE_EXPAND_INPUT_ROWS(InputActivationsType, ExpandedActivationsType)                                   \
    template void expandInputRowsKernelLauncher<InputActivationsType, ExpandedActivationsType>(                        \
        InputActivationsType const* unpermuted_input, ExpandedActivationsType* permuted_output,                        \
        float const* unpermuted_scales, float* permuted_scales, int const* expanded_dest_row_to_expanded_source_row,   \
        int* expanded_source_row_to_expanded_dest_row, int64_t const num_rows, int64_t const cols, int const k,        \
        int const num_experts_per_node, float const* fc1_act_global_scale, int64_t* expert_first_token_offset,         \
        cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,                               \
        cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, cudaStream_t stream);

INSTANTIATE_EXPAND_INPUT_ROWS(half, half);
INSTANTIATE_EXPAND_INPUT_ROWS(float, float);
#ifdef ENABLE_BF16
INSTANTIATE_EXPAND_INPUT_ROWS(__nv_bfloat16, __nv_bfloat16);
#endif

// enum class ScaleMode : int
// {
//     NO_SCALE = 0,
//     DEFAULT = 1,
// };

// constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;

// template <class T>
// using sizeof_bits = cutlass::sizeof_bits<typename
// cutlass_kernels::TllmToCutlassTypeAdapter<std::remove_cv_t<T>>::type>;

// // Final kernel to unpermute and scale
// // This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
// template <typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE, bool CHECK_SKIPPED>
// __global__ void finalizeMoeRoutingKernel(GemmOutputType const* expanded_permuted_rows,
//     OutputType* reduced_unpermuted_output, ScaleBiasType const* bias, float const* scales,
//     int const* expanded_source_row_to_expanded_dest_row, int const* expert_for_source_row, int64_t const orig_cols,
//     int64_t const experts_per_token, int64_t const* num_valid_ptr)
// {
//     assert(orig_cols % 4 == 0);
//     int64_t const original_row = blockIdx.x;
//     int64_t const num_rows = gridDim.x;
//     auto const offset = original_row * orig_cols;
//     OutputType* reduced_row_ptr = reduced_unpermuted_output + offset;

//     // Load 128-bits per thread, according to the smallest data type we read/write
//     constexpr int64_t FINALIZE_ELEM_PER_THREAD
//         = 128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);

//     int64_t const start_offset = threadIdx.x;
//     int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
//     int64_t const num_elems_in_col = orig_cols / FINALIZE_ELEM_PER_THREAD;

//     using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
//     using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
//     using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
//     using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
//     auto const* bias_v = reinterpret_cast<BiasElem const*>(bias);
//     auto const* expanded_permuted_rows_v = reinterpret_cast<InputElem const*>(expanded_permuted_rows);
//     auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

// #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
//     asm volatile("griddepcontrol.wait;");
// #endif
//     int64_t const num_valid = *num_valid_ptr;

// #pragma unroll
//     for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
//     {
//         bool has_valid = false;
//         ComputeElem thread_output;
//         thread_output.fill(0);
//         for (int k_idx = 0; k_idx < experts_per_token; ++k_idx)
//         {
//             int64_t const expanded_original_row = original_row + k_idx * num_rows;
//             int64_t const expanded_permuted_row = expanded_source_row_to_expanded_dest_row[expanded_original_row];

//             int64_t const k_offset = original_row * experts_per_token + k_idx;
//             float const row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];

//             // Check after row_rescale has accumulated
//             if (CHECK_SKIPPED && expanded_permuted_row >= num_valid)
//             {
//                 continue;
//             }

//             auto const* expanded_permuted_rows_row_ptr
//                 = expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_col;

//             int64_t const expert_idx = expert_for_source_row[k_offset];

//             auto const* bias_ptr = bias_v + expert_idx * num_elems_in_col;
//             ComputeElem bias_value;
//             if (bias)
//             {
//                 bias_value = arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
//             }
//             else
//             {
//                 bias_value.fill(0);
//             }

//             ComputeElem expert_result
//                 = arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);
//             thread_output = thread_output + row_scale * (expert_result + bias_value);
//             has_valid = true;
//         }

//         OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
//         reduced_row_ptr_v[elem_index] = output_elem;
//     }
// #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
//     asm volatile("griddepcontrol.launch_dependents;");
// #endif
// }

// template <class OutputType, class GemmOutputType, class ScaleBiasType>
// void finalizeMoeRoutingKernelLauncher(GemmOutputType const* expanded_permuted_rows,
//     OutputType* reduced_unpermuted_output, ScaleBiasType const* bias, float const* final_scales,
//     int const* expanded_source_row_to_expanded_dest_row, int const* expert_for_source_row, int64_t const num_rows,
//     int64_t const cols, int64_t const experts_per_token, int64_t const* num_valid_ptr,
//     cutlass_kernels::MOEParallelismConfig parallelism_config, cudaStream_t stream)
// {
//     int64_t const blocks = num_rows;
//     int64_t const threads = FINALIZE_THREADS_PER_BLOCK;

//     // Only add bias on rank 0 for tensor parallelism
//     bool const is_rank_0 = parallelism_config.tp_rank == 0;
//     ScaleBiasType const* bias_ptr = is_rank_0 ? bias : nullptr;

//     bool const check_skipped = num_valid_ptr != nullptr;

//     ScaleMode scale_mode = final_scales ? ScaleMode::DEFAULT : ScaleMode::NO_SCALE;

//     using FuncPtr
//         = decltype(&finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT, false>);
//     FuncPtr func_map[2][3] = {
//         {
//             &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::NO_SCALE, false>,
//             &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT, false>,
//         },
//         {
//             &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::NO_SCALE, true>,
//             &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT, true>,
//         },
//     };
//     auto* const func = func_map[check_skipped][int(scale_mode)];

//     cudaLaunchConfig_t config;
//     config.gridDim = blocks;
//     config.blockDim = threads;
//     config.dynamicSmemBytes = 0;
//     config.stream = stream;
//     cudaLaunchAttribute attrs[1];
//     attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
//     attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
//     config.numAttrs = 1;
//     config.attrs = attrs;
//     cudaLaunchKernelEx(&config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr, final_scales,
//         expanded_source_row_to_expanded_dest_row, expert_for_source_row, cols, experts_per_token, num_valid_ptr);
// }

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
    cutlass_kernels::MOEParallelismConfig parallelism_config, bool const enable_alltoall, cudaStream_t stream)
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

#define INSTANTIATE_FINALIZE_MOE_ROUTING(OutputT, GemmOutputT, ScaleBiasT)                                             \
    template void finalizeMoeRoutingKernelLauncher<OutputT, GemmOutputT, ScaleBiasT>(                                  \
        GemmOutputT const* expanded_permuted_rows, OutputT* reduced_unpermuted_output, ScaleBiasT const* bias,         \
        float const* final_scales, int const* expanded_source_row_to_expanded_dest_row,                                \
        int const* expanded_dest_row_to_expanded_source_row, int const* expert_for_source_row,                         \
        int64_t const* expert_first_token_offset, int64_t const num_rows, int64_t const cols,                          \
        int64_t const experts_per_token, int const num_experts_per_node,                                               \
        cutlass_kernels::MOEParallelismConfig parallelism_config, bool const enable_alltoall, cudaStream_t stream);

INSTANTIATE_FINALIZE_MOE_ROUTING(half, half, half);
INSTANTIATE_FINALIZE_MOE_ROUTING(float, float, float);
#ifdef ENABLE_BF16
INSTANTIATE_FINALIZE_MOE_ROUTING(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
#endif

} // namespace tensorrt_llm::kernels
