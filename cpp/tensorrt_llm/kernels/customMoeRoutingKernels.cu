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
static constexpr int DEFAULT_BLOCK_SIZE = 1024;
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

    const uint32_t maxNumBlocks = 1024;

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

} // namespace kernels

TRTLLM_NAMESPACE_END
