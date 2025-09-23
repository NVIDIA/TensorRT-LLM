/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

namespace tensorrt_llm::kernels
{

static constexpr int BLOCK_SIZE = 1024;
static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

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
__global__ void customMoeRoutingKernel(InputT* routerLogits, OutputT* topkValues, IdxT* topkIndices,
    int32_t const numTokens, int32_t const numExperts, int32_t const topK)
{
    using BaseType = std::conditional_t<DoSoftmaxBeforeTopK, float, InputT>;
    uint32_t const blockRank = blockIdx.x;
    uint32_t const tIdx = BLOCK_SIZE * blockRank + threadIdx.x;
    uint32_t const warpIdx = tIdx / WARP_SIZE;
    uint32_t const laneIdx = tIdx % WARP_SIZE;
    uint32_t const warpNum = gridDim.x * WARPS_PER_BLOCK;
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);

    BaseType minScore = BaseType{-INFINITY};
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
        default: kernelInstance = nullptr; break;                                                                      \
        }                                                                                                              \
        break;

template <typename InputT, typename OutputT, typename IdxT, bool DoSoftmaxBeforeTopK>
void invokeCustomMoeRouting(InputT* routerLogits, OutputT* topkValues, IdxT* topkIndices, int64_t const numTokens,
    int64_t const numExperts, int64_t const topK, cudaStream_t const stream)
{

    const uint32_t maxNumBlocks = 1024;
    const uint32_t numBlocks = std::min(static_cast<uint32_t>((numTokens - 1) / WARPS_PER_BLOCK + 1), maxNumBlocks);

    uint32_t maxNumExperts = nextPowerOfTwo(numExperts) < 32 ? 32 : nextPowerOfTwo(numExperts);
    uint32_t maxNumTopExperts = nextPowerOfTwo(topK);

    auto* kernelInstance = &customMoeRoutingKernel<InputT, OutputT, IdxT, 128, 8, DoSoftmaxBeforeTopK>;

    switch (maxNumExperts)
    {
        CASE(32)
        CASE(64)
        CASE(96)
        CASE(128)
    default: kernelInstance = nullptr; break;
    }

    if (kernelInstance == nullptr)
    {
        TLLM_CHECK_WITH_INFO(kernelInstance != nullptr, "Can not find corresponding kernel instance.");
    }

    dim3 renormMoeRoutingGridDim(numBlocks);
    dim3 renormMoeRoutingBlockDim(BLOCK_SIZE);
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

} // namespace tensorrt_llm::kernels
