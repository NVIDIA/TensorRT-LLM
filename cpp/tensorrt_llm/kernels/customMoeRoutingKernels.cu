/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "moeTopKFuncs.cuh"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/archCondition.h"
#include "tensorrt_llm/kernels/customMoeRoutingKernels.h"
#include <climits> // For INT_MAX
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda/std/limits> // For numeric_limits
#include <math.h>

namespace cg = cooperative_groups;
using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

static constexpr int WARP_SIZE = 32;
// Default block size for kernels with small MaxNumExperts (<=128).
// Large-expert variants (256/384/512) use a smaller block (see pickBlockSize)
// to reduce register-file pressure and permit higher SM occupancy.
static constexpr int DEFAULT_BLOCK_SIZE = 128;
static constexpr int LARGE_BLOCK_SIZE = 256;

template <int MaxNumExperts>
constexpr int pickBlockSize()
{
    return MaxNumExperts > 128 ? LARGE_BLOCK_SIZE : DEFAULT_BLOCK_SIZE;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
__device__ DataType calcSoftmax(
    cg::thread_block_tile<WARP_SIZE> const& warp, DataType score, int32_t laneIdx, int32_t NumTopExperts)
{
    float maxScore = -INFINITY;
    if (laneIdx < NumTopExperts)
    {
        maxScore = float(score) >= maxScore ? float(score) : maxScore;
    }
    maxScore = cg::reduce(warp, maxScore, cg::greater<float>());

    float sumScore = 0.f;
    float newScore;
    // Get the summation of scores for each token
    if (laneIdx < NumTopExperts)
    {
        newScore = static_cast<float>(score) - static_cast<float>(maxScore);
        newScore = static_cast<float>(exp(newScore));
        sumScore += newScore;
    }
    sumScore = cg::reduce(warp, sumScore, cg::plus<float>());

    if (laneIdx < NumTopExperts)
    {
        score = static_cast<DataType>(newScore / sumScore);
    }

    return score;
}

template <typename DataType, int VecSize>
__device__ void calcSoftmax(cg::thread_block_tile<WARP_SIZE> const& warp, DataType (&scores)[VecSize])
{
    // Compute in float to support half/bfloat16 inputs safely.
    float maxScore = -INFINITY;
    float sumScore = 0.f;
    // Get the max score for each token
#pragma unroll
    for (int i = 0; i < VecSize; ++i)
    {
        float si = static_cast<float>(scores[i]);
        maxScore = si >= maxScore ? si : maxScore;
    }
    maxScore = cg::reduce(warp, maxScore, cg::greater<float>());

    // Get the summation of scores for each token
#pragma unroll
    for (int i = 0; i < VecSize; ++i)
    {
        float si = static_cast<float>(scores[i]);
        float e = expf(si - maxScore);
        scores[i] = static_cast<DataType>(e);
        sumScore += e;
    }
    sumScore = cg::reduce(warp, sumScore, cg::plus<float>());

    // Normalize the scores
#pragma unroll
    for (int i = 0; i < VecSize; ++i)
    {
        float si = static_cast<float>(scores[i]) / sumScore;
        scores[i] = static_cast<DataType>(si);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename InputT, typename OutputT, typename IdxT, int MaxNumExperts, int MaxNumTopExperts,
    bool DoSoftmaxBeforeTopK>
__global__ void __launch_bounds__(pickBlockSize<MaxNumExperts>(), 1) customMoeRoutingKernel(InputT* routerLogits,
    OutputT* topkValues, IdxT* topkIndices, int32_t const numTokens, int32_t const numExperts, int32_t const topK)
{
    using BaseType = std::conditional_t<DoSoftmaxBeforeTopK, float, InputT>;
    constexpr int kBlockSize = pickBlockSize<MaxNumExperts>();
    constexpr int kWarpsPerBlock = kBlockSize / WARP_SIZE;
    uint32_t const blockRank = blockIdx.x;
    uint32_t const tIdx = kBlockSize * blockRank + threadIdx.x;
    uint32_t const warpIdx = tIdx / WARP_SIZE;
    uint32_t const laneIdx = tIdx % WARP_SIZE;
    uint32_t const warpNum = gridDim.x * kWarpsPerBlock;
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);

    BaseType minScore = BaseType{-INFINITY};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    for (uint32_t tokenId = warpIdx; tokenId < numTokens; tokenId += warpNum)
    {
        auto scoreOffset = tokenId * numExperts;
        auto outputOffset = tokenId * topK;

        BaseType inputScore[MaxNumExperts / WARP_SIZE];
        IdxT inputIndex[MaxNumExperts / WARP_SIZE];

        BaseType warpTopKScore[MaxNumTopExperts];
        IdxT warpTopKExpertIdx[MaxNumTopExperts];

        // Load scores and indices for this warp
        for (uint32_t i = 0; i < MaxNumExperts / WARP_SIZE; ++i)
        {
            auto expertIdx = i * WARP_SIZE + laneIdx;
            inputScore[i]
                = expertIdx < numExperts ? static_cast<BaseType>(routerLogits[scoreOffset + expertIdx]) : minScore;
            inputIndex[i] = expertIdx;
        }

        if constexpr (DoSoftmaxBeforeTopK)
        {
            calcSoftmax(warp, inputScore);
        }
        // Reduce topK scores and indices for this warp
        reduce_topk::reduceTopK(warp, warpTopKScore, warpTopKExpertIdx, inputScore, inputIndex, minScore);

        // Normalize the scores
        if constexpr (DoSoftmaxBeforeTopK)
        {
            if (laneIdx < topK)
            {
                topkValues[outputOffset + laneIdx] = static_cast<OutputT>(warpTopKScore[laneIdx]);
                topkIndices[outputOffset + laneIdx] = warpTopKExpertIdx[laneIdx];
            }
        }
        else
        {
            auto softmaxScore = calcSoftmax(warp,
                laneIdx < topK ? static_cast<float>(warpTopKScore[laneIdx]) : static_cast<float>(minScore), laneIdx,
                topK);
            if (laneIdx < topK)
            {
                topkValues[outputOffset + laneIdx] = static_cast<OutputT>(softmaxScore);
                topkIndices[outputOffset + laneIdx] = warpTopKExpertIdx[laneIdx];
            }
        }
    } // end for tokenId

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

int nextPowerOfTwo(int num)
{
    if (num <= 0)
    {
        return 1; // Handle invalid input
    }
    int power = 1;
    while (power < num)
    {
        // Check for overflow before shifting
        if (power > INT_MAX / 2)
        {
            return power;
        }
        power <<= 1;
    }
    return power;
}

#define CASE(MAX_NUM_EXPERTS)                                                                                          \
    case MAX_NUM_EXPERTS:                                                                                              \
        switch (maxNumTopExperts)                                                                                      \
        {                                                                                                              \
        case 1:                                                                                                        \
            kernelInstance = &customMoeRoutingKernel<InputT, OutputT, IdxT, MAX_NUM_EXPERTS, 1, DoSoftmaxBeforeTopK>;  \
            break;                                                                                                     \
        case 2:                                                                                                        \
            kernelInstance = &customMoeRoutingKernel<InputT, OutputT, IdxT, MAX_NUM_EXPERTS, 2, DoSoftmaxBeforeTopK>;  \
            break;                                                                                                     \
        case 4:                                                                                                        \
            kernelInstance = &customMoeRoutingKernel<InputT, OutputT, IdxT, MAX_NUM_EXPERTS, 4, DoSoftmaxBeforeTopK>;  \
            break;                                                                                                     \
        case 8:                                                                                                        \
            kernelInstance = &customMoeRoutingKernel<InputT, OutputT, IdxT, MAX_NUM_EXPERTS, 8, DoSoftmaxBeforeTopK>;  \
            break;                                                                                                     \
        case 16:                                                                                                       \
            kernelInstance = &customMoeRoutingKernel<InputT, OutputT, IdxT, MAX_NUM_EXPERTS, 16, DoSoftmaxBeforeTopK>; \
            break;                                                                                                     \
        default: kernelInstance = nullptr; break;                                                                      \
        }                                                                                                              \
        break;

template <typename InputT, typename OutputT, typename IdxT, bool DoSoftmaxBeforeTopK>
void invokeCustomMoeRouting(InputT* routerLogits, OutputT* topkValues, IdxT* topkIndices, int64_t const numTokens,
    int64_t const numExperts, int64_t const topK, cudaStream_t const stream)
{

    const uint32_t maxNumBlocks = 8192;

    uint32_t maxNumExperts = nextPowerOfTwo(numExperts) < 32 ? 32 : nextPowerOfTwo(numExperts);
    uint32_t maxNumTopExperts = nextPowerOfTwo(topK);

    // Pick block size matching what pickBlockSize<> selects for this MaxNumExperts.
    // Large-expert variants use LARGE_BLOCK_SIZE to reduce register pressure.
    uint32_t blockSize = maxNumExperts > 128 ? LARGE_BLOCK_SIZE : DEFAULT_BLOCK_SIZE;
    uint32_t warpsPerBlock = blockSize / WARP_SIZE;
    const uint32_t numBlocks = std::min(static_cast<uint32_t>((numTokens - 1) / warpsPerBlock + 1), maxNumBlocks);

    auto* kernelInstance = &customMoeRoutingKernel<InputT, OutputT, IdxT, 128, 8, DoSoftmaxBeforeTopK>;

    switch (maxNumExperts)
    {
        CASE(32)
        CASE(64)
        CASE(96)
        CASE(128)
        CASE(256)
        CASE(384)
        CASE(512)
    default: kernelInstance = nullptr; break;
    }

    if (kernelInstance == nullptr)
    {
        TLLM_CHECK_WITH_INFO(kernelInstance != nullptr, "Can not find corresponding kernel instance.");
    }

    dim3 renormMoeRoutingGridDim(numBlocks);
    dim3 renormMoeRoutingBlockDim(blockSize);
    cudaLaunchConfig_t config;
    config.gridDim = renormMoeRoutingGridDim;
    config.blockDim = renormMoeRoutingBlockDim;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernelInstance, routerLogits, topkValues, topkIndices, static_cast<int32_t>(numTokens),
        static_cast<int32_t>(numExperts), static_cast<int32_t>(topK));
    sync_check_cuda_error(stream);
}

#define INSTANTIATE_RENORM_MOE_ROUTING(InputT, OutputT, IdxT, DoSoftmaxBeforeTopK)                                     \
    template void invokeCustomMoeRouting<InputT, OutputT, IdxT, DoSoftmaxBeforeTopK>(InputT * routerLogits,            \
        OutputT * topkValues, IdxT * topkIndices, int64_t const numTokens, int64_t const numExperts,                   \
        int64_t const topK, cudaStream_t const stream);

INSTANTIATE_RENORM_MOE_ROUTING(float, float, int32_t, false);
INSTANTIATE_RENORM_MOE_ROUTING(half, float, int32_t, false);
INSTANTIATE_RENORM_MOE_ROUTING(float, float, int32_t, true);
INSTANTIATE_RENORM_MOE_ROUTING(half, float, int32_t, true);

#ifdef ENABLE_BF16
INSTANTIATE_RENORM_MOE_ROUTING(__nv_bfloat16, float, int32_t, false);
INSTANTIATE_RENORM_MOE_ROUTING(float, __nv_bfloat16, int32_t, false);
INSTANTIATE_RENORM_MOE_ROUTING(half, __nv_bfloat16, int32_t, false);
INSTANTIATE_RENORM_MOE_ROUTING(__nv_bfloat16, __nv_bfloat16, int32_t, false);

INSTANTIATE_RENORM_MOE_ROUTING(__nv_bfloat16, float, int32_t, true);
INSTANTIATE_RENORM_MOE_ROUTING(float, __nv_bfloat16, int32_t, true);
INSTANTIATE_RENORM_MOE_ROUTING(half, __nv_bfloat16, int32_t, true);
INSTANTIATE_RENORM_MOE_ROUTING(__nv_bfloat16, __nv_bfloat16, int32_t, true);
#endif

static constexpr int kTOPK = 6;

// CUDA kernel for gate forward
// Input: pre-computed scores from linear(x, weight) done outside kernel
// Template parameters:
//   nExperts: number of experts
//   topK: number of top experts to select
//   hash: true for hash mode, false for topk mode
// One warp per row (batch element)
template <int nExperts, int topK, bool hash>
__global__ void gate_forward_kernel(
    float const* __restrict__ scores_in, // [batch_size, nExperts] - pre-computed from linear(x, weight)
    float const* __restrict__ bias,      // [nExperts] (only used when hash=false)
    int const* __restrict__ input_ids,   // [batch_size] (only used when hash=true)
    int const* __restrict__ tid2eid,     // [vocab_size, topK] (only used when hash=true)
    float* __restrict__ out_weights,     // [batch_size, topK]
    int* __restrict__ out_indices,       // [batch_size, topK]
    int batch_size, float route_scale)
{
    // Compile-time constants
    constexpr int kExpertsPerThread = nExperts / WARP_SIZE;
    constexpr int kWarpsPerBlock = 4; // Adjust based on occupancy needs

    // Shared memory for original scores (one array per warp in the block)
    __shared__ float smem_scores[kWarpsPerBlock][nExperts];

    // One warp per batch element
    int const global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int const local_warp_id = (threadIdx.x / WARP_SIZE) % kWarpsPerBlock;
    int const lane_id = threadIdx.x % WARP_SIZE;

    if (global_warp_id >= batch_size)
        return;

    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

    // Pointer to this warp's shared memory and input scores
    float* my_smem = smem_scores[local_warp_id];
    float const* scores_row = scores_in + global_warp_id * nExperts;

// Load scores, apply score function (softplus + sqrt), and store to shared memory
#pragma unroll
    for (int e = 0; e < kExpertsPerThread; ++e)
    {
        int expert_id = lane_id + e * WARP_SIZE;
        float s = scores_row[expert_id];
        float sp = log1pf(expf(s));
        float score = sqrtf(sp);
        my_smem[expert_id] = score; // Store original score to shared memory
    }
    __syncwarp();                   // Ensure all scores are written before reading

    // Output: each of first K lanes holds one value
    float my_topk_value = 0.0f;
    int my_topk_index = 0;

    if constexpr (hash)
    {
        // Hash mode: directly read from shared memory
        int token_id = input_ids[global_warp_id];
        int const* expert_ids = tid2eid + token_id * topK;

        if (lane_id < topK)
        {
            int expert_id = expert_ids[lane_id];
            my_topk_index = expert_id;
            my_topk_value = my_smem[expert_id]; // Direct lookup from shared memory
        }
    }
    else
    {
        // Topk mode: load from shared memory, add bias to registers for topk
        float scores[kExpertsPerThread];
        int indices[kExpertsPerThread];

#pragma unroll
        for (int e = 0; e < kExpertsPerThread; ++e)
        {
            int expert_id = lane_id + e * WARP_SIZE;
            indices[e] = expert_id;
            scores[e] = my_smem[expert_id] + bias[expert_id]; // Add bias for topk selection
        }

        // Use reduceTopK to find top-k experts
        float topk_values[topK];
        int32_t topk_indices[topK];
        constexpr float minValue = -1e30f;
        reduce_topk::reduceTopK<topK, float, kExpertsPerThread>(
            warp, topk_values, topk_indices, scores, indices, minValue, topK);

        // Gather original weights (without bias) from shared memory
        if (lane_id < topK)
        {
            int expert_id = topk_indices[lane_id];
            my_topk_index = expert_id;
            my_topk_value = my_smem[expert_id]; // Read original score (no bias)
        }
    }

    // Reduce to get sum (first K lanes have values, others have 0)
    float weight_sum = cg::reduce(warp, my_topk_value, cg::plus<float>{});

    // Normalize weights and write output (first K lanes)
    if (lane_id < topK)
    {
        out_weights[global_warp_id * topK + lane_id] = (my_topk_value / weight_sum) * route_scale;
        out_indices[global_warp_id * topK + lane_id] = my_topk_index;
    }
}

// C++ wrapper function (output tensors passed as parameters)
// All tensors are float32
template <int nExperts, bool hash>
void launch_gate_forward_kernel(float* scores_in, float* bias, int* input_ids, int* tid2eid, float* out_weights,
    int* out_indices, int batch_size, float route_scale, cudaStream_t stream)
{
    constexpr int warps_per_block = 4;
    constexpr int threads_per_block = warps_per_block * WARP_SIZE;
    int const blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    gate_forward_kernel<nExperts, kTOPK, hash><<<blocks, threads_per_block, 0, stream>>>(
        scores_in, bias, input_ids, tid2eid, out_weights, out_indices, batch_size, route_scale);
}

void gate_forward(void* scores_in, // [batch_size, nExperts] - pre-computed from linear(x, weight)
    void* bias,                    // nullptr if hash mode
    void* input_ids,               // nullptr if non-hash mode
    void* tid2eid,                 // nullptr if non-hash mode
    void* out_weights,             // [batch_size, topK] - pre-allocated
    void* out_indices,             // [batch_size, topK] - pre-allocated
    int batch_size, int n_experts, float route_scale, bool is_hash, cudaStream_t stream)
{
    auto* scores = static_cast<float*>(scores_in);
    auto* bias_ptr = static_cast<float*>(bias);
    auto* input_ids_ptr = static_cast<int*>(input_ids);
    auto* tid2eid_ptr = static_cast<int*>(tid2eid);
    auto* weights = static_cast<float*>(out_weights);
    auto* indices = static_cast<int*>(out_indices);

    switch (n_experts)
    {
    case 256:
        if (is_hash)
        {
            launch_gate_forward_kernel<256, true>(
                scores, nullptr, input_ids_ptr, tid2eid_ptr, weights, indices, batch_size, route_scale, stream);
        }
        else
        {
            launch_gate_forward_kernel<256, false>(
                scores, bias_ptr, nullptr, nullptr, weights, indices, batch_size, route_scale, stream);
        }
        break;
    case 384:
        if (is_hash)
        {
            launch_gate_forward_kernel<384, true>(
                scores, nullptr, input_ids_ptr, tid2eid_ptr, weights, indices, batch_size, route_scale, stream);
        }
        else
        {
            launch_gate_forward_kernel<384, false>(
                scores, bias_ptr, nullptr, nullptr, weights, indices, batch_size, route_scale, stream);
        }
        break;
    default: TLLM_CHECK_WITH_INFO(false, "gate_forward only supports n_experts 256 or 384");
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
