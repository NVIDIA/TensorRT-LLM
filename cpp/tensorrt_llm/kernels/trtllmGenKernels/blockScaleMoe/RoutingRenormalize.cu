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
#include "RoutingKernel.cuh"

namespace moe::dev::routing
{
namespace routingRenormalize
{
////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumExpertsLimit = 512;

static constexpr int NumThreads = 1024;
static constexpr int NumWarps = NumThreads / WarpSize;
static constexpr int MaxNumTopExperts = 10;

static constexpr int MaxNumTokensSingleCluster = NumBlocksPerCluster * NumThreads;
static constexpr int MaxNumTokensSingleClusterScores = NumBlocksPerCluster * NumWarps;

static constexpr int BlockKernelMaxNumTokens = 4;

template <typename DataType, typename InputType, int VecSize, bool DoSoftmaxBeforeTopK>
__forceinline__ __device__ void routingTopKExperts(cg::thread_block_tile<WarpSize> const& warp,
    DataType (&score)[VecSize], int32_t (&idx)[VecSize], DataType (&warpTopKScore)[MaxNumTopExperts],
    int32_t (&warpTopKExpertIdx)[MaxNumTopExperts], int32_t const laneIdx, int32_t const numExperts, int32_t topK,
    InputType const* ptrScores, bool const normTopkProb, bool const applySoftmaxAfterTopK = true)
{
    DataType minScore = DataType{-INFINITY};

    for (int i = 0; i < VecSize; i++)
    {
        auto expertIdx = i * WarpSize + laneIdx;
        auto newScore = expertIdx < numExperts ? static_cast<DataType>(ptrScores[expertIdx]) : minScore;
        score[i] = newScore;
        idx[i] = expertIdx;
    }
    if constexpr (DoSoftmaxBeforeTopK)
    {
        calcSoftmax(warp, score);
    }

    // Get the top-k scores and their corresponding expert indices
    topk::reduceTopK(warp, warpTopKScore, warpTopKExpertIdx, score, idx, minScore, topK);

    // Normalize the scores
    if constexpr (DoSoftmaxBeforeTopK)
    {
        float sum = float{1.f};
        if (normTopkProb)
        {
            sum = static_cast<float>(laneIdx < topK ? warpTopKScore[laneIdx] : 0);
            sum = cg::reduce(warp, sum, cg::plus<float>());
        }
        if (laneIdx < topK)
        {
            warpTopKScore[laneIdx] = warpTopKScore[laneIdx] / sum;
        }
    }
    else
    {
        if (applySoftmaxAfterTopK)
        {
            auto softmaxScore = calcSoftmax(warp, laneIdx < topK ? warpTopKScore[laneIdx] : minScore, laneIdx, topK);
            if (laneIdx < topK)
            {
                warpTopKScore[laneIdx] = softmaxScore;
            }
        }
    }
}

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts) routingIndicesBlockKernel(KernelParams params)
{
    // types used in this kernel
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;
    using BaseType = std::conditional_t<KernelParams::DoSoftmaxBeforeTopK, float, InputT>;
    using TypePacked = PackedScoreIdx<BaseType>;
    int constexpr MaxNumExperts = KernelParams::MaxNumExperts;

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    int32_t const laneIdx = cutlass::arch::LaneId();
    int32_t const expert = threadIdx.x;
    auto scoreOffset = warpIdx * params.mNumExperts;
    bool validToken = warpIdx < params.mNumTokens;

    static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;
    static constexpr int totalExpertCounts = BlockKernelMaxNumTokens * MaxNumExperts;
    __shared__ int8_t __attribute((aligned(128))) smemOffset[totalExpertCounts];
    __shared__ int8_t __attribute((aligned(128))) smemKIdx[totalExpertCounts];

    using Scan = cub::BlockScan<int32_t, MaxNumExperts>;
    __shared__ typename Scan::TempStorage tempStorage;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

    for (int i = threadIdx.x; i < totalExpertCounts; i += blockDim.x)
    {
        smemOffset[i] = int8_t{-1};
        smemKIdx[i] = int8_t{-1};
    }
    __syncthreads();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // then wait on primary grid
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif

    if (params.mPtrTopKIds != nullptr)
    {
        if (validToken)
        {
            if (laneIdx < params.mTopK)
            {
                int offset = warpIdx * MaxNumExperts + params.mPtrTopKIds[warpIdx * params.mTopK + laneIdx];
                smemKIdx[offset] = static_cast<int8_t>(laneIdx);
            }
        }
    }
    else if (params.mPtrScores != nullptr)
    {
        // in this case, each warp represents a token
        BaseType score[VecSize];
        int32_t idx[VecSize];

        BaseType warpTopKScore[MaxNumTopExperts];
        int32_t warpTopKExpertIdx[MaxNumTopExperts];

        BaseType minScore = BaseType{-INFINITY};
        if (validToken)
        {
            routingTopKExperts<BaseType, InputT, VecSize, KernelParams::DoSoftmaxBeforeTopK>(warp, score, idx,
                warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
                params.mPtrScores + scoreOffset, params.mNormTopkProb);

            if (laneIdx < params.mTopK)
            {
                int offset = warpIdx * MaxNumExperts + warpTopKExpertIdx[laneIdx];
                smemKIdx[offset] = static_cast<int8_t>(laneIdx);
                if (params.mPtrTopKWeights != nullptr)
                {
                    params.mPtrTopKWeights[warpIdx * params.mTopK + laneIdx] = OutputT{warpTopKScore[laneIdx]};
                }
            }
        } // end if (validToken)
    }
    __syncthreads();

    // set local experts
    auto localExpertIdx = expert - params.mLocalExpertsStartIdx;
    auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < params.mNumLocalExperts
        && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
    // Get the count of each expert and the offset for each token
    int accExpertCount = 0;

    if (isLocalExpert)
    {
        int offset = expert;
        for (int j = 0; j < BlockKernelMaxNumTokens; j++)
        {
            if (smemKIdx[offset] >= 0)
            {
                smemOffset[offset] = static_cast<int8_t>(accExpertCount);
                accExpertCount++;
            }
            offset += MaxNumExperts;
        }
    }
    __syncthreads();
    // Get the number of CTAs and the offset for each CTA
    int32_t numCta;
    if constexpr (KernelParams::isPow2)
    {
        numCta = divUpLog2<int32_t>(accExpertCount, params.mPaddingLog2);
    }
    else
    {
        numCta = divUpTileN<int32_t>(accExpertCount, params.mTileTokensDim);
    }
    int32_t ctaOffset = 0;
    int32_t numNonExitingCtas;
    Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);

    int32_t expertScanCounts = 0;
    int32_t tmpCount;
    if constexpr (KernelParams::isPow2)
    {
        tmpCount = divUpMulLog2<int32_t>(accExpertCount, params.mPaddingLog2);
    }
    else
    {
        tmpCount = divUpMulTileN<int32_t>(accExpertCount, params.mTileTokensDim);
    }
    Scan(tempStorage).ExclusiveSum(tmpCount, expertScanCounts);
    __syncthreads();

    if (isLocalExpert)
    {
        for (int cta = 0; cta < numCta; ++cta)
        {
            const int32_t localExpertIdx = (expert - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
            params.mPtrCtaIdxXyToBatchIdx[ctaOffset + cta] = localExpertIdx;
            int32_t mnLimit1;
            int32_t mnLimit2;
            if constexpr (KernelParams::isPow2)
            {
                mnLimit1 = mulLog2<int32_t>(ctaOffset + cta + 1, params.mPaddingLog2);
                mnLimit2 = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2) + accExpertCount;
            }
            else
            {
                mnLimit1 = mulTileN<int32_t>(ctaOffset + cta + 1, params.mTileTokensDim);
                mnLimit2 = mulTileN<int32_t>(ctaOffset, params.mTileTokensDim) + accExpertCount;
            }
            params.mPtrCtaIdxXyToMnLimit[ctaOffset + cta] = min(mnLimit1, mnLimit2);
        }
    }

    // at this point, we can write out padded count
    if (threadIdx.x == 0)
    {
        int32_t permutedIdxSize;
        if constexpr (KernelParams::isPow2)
        {
            permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
        }
        else
        {
            permutedIdxSize = mulTileN<int32_t>(numNonExitingCtas, params.mTileTokensDim);
        }
        params.mPtrPermutedIdxSize[0] = permutedIdxSize;
        params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // we can trigger the next kernel at this point
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif

    for (int tokenIdx = 0; tokenIdx < params.mNumTokens; tokenIdx++)
    {
        int offset = tokenIdx * MaxNumExperts + threadIdx.x;
        if (smemKIdx[offset] >= 0)
        {
            int const expandedIdx = tokenIdx * params.mTopK + smemKIdx[offset];
            int const offsetWithinExpert = static_cast<int>(smemOffset[offset]);
            int const offsetForExpert = expertScanCounts;
            int const permutedIdx = isLocalExpert ? offsetForExpert + offsetWithinExpert : int32_t{-1};

            params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
            if (isLocalExpert)
            {
                params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
            }
        }
    }
}

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(NumThreads)
    routingIndicesClusterKernel(KernelParams params)
{
    // number of tokens/expanded idx is bounded by total number of warps
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;

    using BaseType = std::conditional_t<KernelParams::DoSoftmaxBeforeTopK, float, InputT>;
    using TypePacked = PackedScoreIdx<BaseType>;

    static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;

    __shared__ TypePacked __attribute((aligned(128))) smemPackedScoreIdx[NumWarps * MaxNumTopExperts];

    uint32_t const clusterBlockRank = blockIdx.x;

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    int32_t const laneIdx = cutlass::arch::LaneId();

    auto warpTokenIdx = clusterBlockRank * NumWarps + warpIdx;
    auto scoreOffset = warpTokenIdx * params.mNumExperts;
    bool validToken = warpTokenIdx < params.mNumTokens;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

    // then wait on primary grid
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }

    if (params.mPtrScores != nullptr)
    {
        // in this case, each warp represents a token
        BaseType score[VecSize];
        int32_t idx[VecSize];

        BaseType warpTopKScore[MaxNumTopExperts];
        int32_t warpTopKExpertIdx[MaxNumTopExperts];

        BaseType minScore = BaseType{-INFINITY};
        if (validToken)
        {
            routingTopKExperts<BaseType, InputT, VecSize, KernelParams::DoSoftmaxBeforeTopK>(warp, score, idx,
                warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
                params.mPtrScores + scoreOffset, params.mNormTopkProb);

            if (laneIdx < params.mTopK)
            {
                smemPackedScoreIdx[warpIdx * params.mTopK + laneIdx]
                    = TypePacked{warpTopKScore[laneIdx], static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
            }
        } // end if (validToken)
    }

    // make packed scores available to all threads in cluster
    __cluster_barrier_arrive();
    __cluster_barrier_wait();

    if (params.mPtrScores != nullptr)
    {
        routingPermutation<KernelParams, BaseType, NumThreads, NumWarps, MaxNumTopExperts,
            /*LoadExpertIdxFromGlobal=*/false>(params, smemPackedScoreIdx, warpIdx, clusterBlockRank);
    }
    else
    {
        routingPermutation<KernelParams, BaseType, NumThreads, NumWarps, MaxNumTopExperts,
            /*LoadExpertIdxFromGlobal=*/true>(params, smemPackedScoreIdx, warpIdx, clusterBlockRank);
    }
}
#else
__global__ void __launch_bounds__(NumThreads) routingIndicesClusterKernel(KernelParams /* params */)
{
    assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
////////////////////////////////////////////////////////////////////////////////////////////////////

// this kernel is needed in case we have scores as input for the histogram kernel
template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts) routingIndicesHistogramScoresKernel(KernelParams params)
{
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;
    using BaseType = std::conditional_t<KernelParams::DoSoftmaxBeforeTopK, float, InputT>;

    static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;

    int32_t const laneIdx = cutlass::arch::LaneId();
    int32_t const warpIdx = threadIdx.x / WarpSize;
    int32_t const globalWarpIdx = blockIdx.x * KernelParams::MaxNumExperts / WarpSize + warpIdx;
    int32_t const globalWarpStride = gridDim.x * KernelParams::MaxNumExperts / WarpSize;
    BaseType minScore = BaseType{-INFINITY};
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // Wait on primary grid.
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

    // initialize the mPtrExpertCounts
    int32_t expertCountsNum = 2 * params.mNumExperts;
    int32_t globalThreadIdx = blockIdx.x * KernelParams::MaxNumExperts + threadIdx.x;
    int32_t globalThreadStride = gridDim.x * KernelParams::MaxNumExperts;
    initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // Trigger secondary kernel.
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

    // in this case, each warp represents a token, and we use a grid-stride loop
    // over all warps/tokens
    BaseType allScores[VecSize];
    int32_t allExpertIdx[VecSize];
    BaseType warpTopKScore[MaxNumTopExperts];
    int32_t warpTopKExpertIdx[MaxNumTopExperts];
    for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens; tokenIdx += globalWarpStride)
    {
        auto scoreOffset = tokenIdx * params.mNumExperts;

        routingTopKExperts<BaseType, InputT, VecSize, KernelParams::DoSoftmaxBeforeTopK>(warp, allScores, allExpertIdx,
            warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
            params.mPtrScores + scoreOffset, params.mNormTopkProb);

        if (laneIdx < params.mTopK)
        {
            PackedScoreIdx<OutputT> packedScore{
                static_cast<OutputT>(warpTopKScore[laneIdx]), static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
            params.mPtrTopKPacked[tokenIdx * params.mTopK + laneIdx] = packedScore;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int32_t constexpr getMaxNumExperts(int32_t numExperts)
{
    if (numExperts <= topk::MaxNumExpertsUnit)
    {
        return topk::MaxNumExpertsUnit;
    }
    else if (numExperts <= NumExpertsLimit)
    {
        return NumExpertsLimit;
    }
    else
    {
        TLLM_LOG_ERROR("Unsupported numExperts");
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define LAUNCH_ROUTING_RENORNALIZE(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1)      \
    if (data.mNumExperts <= topk::MaxNumExpertsUnit)                                                                   \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_NUM_EXPERTS(                                                                               \
            data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1, topk::MaxNumExpertsUnit);   \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumExpertsLimit)                                                                      \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_NUM_EXPERTS(                                                                               \
            data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1, NumExpertsLimit);           \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_LOG_ERROR("Unsupported numExperts");                                                                      \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////
void run(Data const& data, void* stream)
{
    TLLM_CHECK_WITH_INFO(data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr || data.mPtrTopKIds != nullptr,
        "Routing kernel requires at least one input parameter");
    if (data.mPtrTopKIds != nullptr)
    {
        TLLM_CHECK_WITH_INFO(data.mPtrTopKWeights != nullptr,
            "When mPtrTopKIds is provided, mPtrTopKWeights must also be provided for Renormalize routing.");
    }
    TLLM_CHECK_WITH_INFO(data.mPtrPermutedIdxSize != nullptr && data.mPtrCtaIdxXyToBatchIdx != nullptr
            && data.mPtrCtaIdxXyToMnLimit != nullptr && data.mPtrNumNonExitingCtas != nullptr,
        "Llama4 routing kernel expects permuted idx and grouped Gemm launch config buffers");
    TLLM_CHECK_WITH_INFO(data.mTopK <= MaxNumTopExperts, "Routing kernel expects topK experts <= %d, got %d",
        MaxNumTopExperts, data.mTopK);
    TLLM_CHECK_WITH_INFO(data.mNumExperts <= NumExpertsLimit,
        "Routing kernel expects #experts %d to be no more than %d", data.mNumExperts, NumExpertsLimit);
    // static_assert(MaxNumExperts <= NumThreads, "#experts must be bounded by #threads");
    // static_assert(MaxNumExperts <= NumThreadsHist, "#experts must be bounded by #threads"); //@todo: check how to add
    // similar check
    TLLM_CHECK_WITH_INFO(
        data.mNumExperts % 4 == 0, "Routing kernel expects #experts %d to be a multiple of 4.", data.mNumExperts);

    bool const useSingleBlock = data.mNumTokens <= BlockKernelMaxNumTokens;

    bool const useSingleCluster = data.mNumTokens <= ((data.mPtrScores != nullptr || data.mPtrTopKIds != nullptr)
                                          ? MaxNumTokensSingleClusterScores
                                          : MaxNumTokensSingleCluster);

    if (!useSingleCluster && !useSingleBlock)
    {
        TLLM_CHECK_WITH_INFO((data.mPtrTopKPacked != nullptr || data.mPtrTopKIds != nullptr),
            "When #tokens is large, `mPtrTopKPacked` or `mPtrTopKIds` is a required input.");
        TLLM_CHECK_WITH_INFO(
            data.mPtrExpertCounts != nullptr, "When #tokens is large, `mPtrExpertCounts` is a required input.");
    }
    uint32_t const numThreadsHist = getMaxNumExperts(data.mNumExperts);
    if (useSingleBlock)
    {
        //@TODO: For now we use the single block kernel for cases with token number no larger than 4.
        // We will future tune this threshold based on the performance.
        LAUNCH_ROUTING_RENORNALIZE(data, false, routingIndicesBlockKernel, 1, numThreadsHist,
            /*smemSize=*/0, // No dynamic smem
            stream, data.mDoSoftmaxBeforeTopK);
    }
    else if (useSingleCluster)
    {
        LAUNCH_ROUTING_RENORNALIZE(data, false, routingIndicesClusterKernel, NumBlocksPerCluster, NumThreads,
            /*smemSize=*/0, // No dynamic smem
            stream, data.mDoSoftmaxBeforeTopK);
    }
    else
    {
        uint32_t const expandedIdxSize = data.mNumTokens * data.mTopK;
        uint32_t const histogramEltsPerBlock = 8 * numThreadsHist;
        uint32_t const offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * numThreadsHist;

        // Limit grid size (all kernels use a grid-stride loop).
        uint32_t const maxNumBlocks = 1024;

        int const numBlocksHistogram
            = std::min((expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
        int const numBlocksOffsets
            = std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

        if (data.mPtrScores != nullptr && data.mPtrTopKIds == nullptr)
        {
            LAUNCH_ROUTING_RENORNALIZE(data, false, routingIndicesHistogramScoresKernel, maxNumBlocks, numThreadsHist,
                /*smemSize=*/0, // No dynamic smem
                stream, data.mDoSoftmaxBeforeTopK);
        }
        else
        {
            // Reset the global histograms.
            LAUNCH_ROUTING_RENORNALIZE(data, false, routingInitExpertCounts,
                (2 * data.mNumExperts - 1) / numThreadsHist + 1, numThreadsHist,
                /*smemSize=*/0, // No dynamic smem
                stream, data.mDoSoftmaxBeforeTopK);
        }
        LAUNCH_ROUTING_RENORNALIZE(data, false, routingIndicesHistogramKernel, numBlocksHistogram, numThreadsHist,
            /*smemSize=*/0, // No dynamic smem
            stream, data.mDoSoftmaxBeforeTopK);
        LAUNCH_ROUTING_RENORNALIZE(data, false, routingIndicesOffsetsKernel, numBlocksOffsets, numThreadsHist,
            /*smemSize=*/0, // No dynamic smem
            stream, data.mDoSoftmaxBeforeTopK);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingRenormalize
} // namespace moe::dev::routing
