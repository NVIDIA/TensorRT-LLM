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

static constexpr int NumThreads = 1024;
static constexpr int NumWarps = NumThreads / WarpSize;
static constexpr int MaxNumTopExperts = 8;
static constexpr int MaxNumExperts = 128;
static constexpr int MaxNumTokensSingleCluster = NumBlocksPerCluster * NumThreads;
static constexpr int MaxNumTokensSingleClusterScores = NumBlocksPerCluster * NumWarps;

template <typename DataType, typename InputType, int VecSize, bool DoSoftmaxBeforeTopK>
__forceinline__ __device__ void routingTopKExperts(cg::thread_block_tile<WarpSize> const& warp,
    DataType (&score)[VecSize], int32_t (&idx)[VecSize], DataType (&warpTopKScore)[MaxNumTopExperts],
    int32_t (&warpTopKExpertIdx)[MaxNumTopExperts], int32_t const laneIdx, int32_t const numExperts, int32_t topK,
    InputType const* ptrScores, bool const normTopkProb)
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
        auto softmaxScore = calcSoftmax(warp, laneIdx < topK ? warpTopKScore[laneIdx] : minScore, laneIdx, topK);
        if (laneIdx < topK)
        {
            warpTopKScore[laneIdx] = softmaxScore;
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

    static constexpr int VecSize = MaxNumExperts / WarpSize;
    // we assume that #experts is a multiple of 4, so VecSize must be 4.
    static_assert(VecSize == 4);

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

        // make packed scores available to all threads in cluster
        __cluster_barrier_arrive();
        __cluster_barrier_wait();
    }

    routingPermutation<KernelParams, BaseType, NumThreads, NumWarps, MaxNumTopExperts,
        /*LoadExpertIdxFromGlobal=*/false>(params, smemPackedScoreIdx, warpIdx, clusterBlockRank);
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
__global__ void __launch_bounds__(NumThreadsHist) routingIndicesHistogramScoresKernel(KernelParams params)
{
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;
    using BaseType = std::conditional_t<KernelParams::DoSoftmaxBeforeTopK, float, InputT>;

    static constexpr int VecSize = MaxNumExperts / WarpSize;
    // we assume that #experts is a multiple of 4, so VecSize must be 4.
    static_assert(VecSize == 4);

    int32_t const laneIdx = cutlass::arch::LaneId();
    int32_t const warpIdx = threadIdx.x / WarpSize;
    int32_t const globalWarpIdx = blockIdx.x * NumWarpsHist + warpIdx;
    int32_t const globalWarpStride = gridDim.x * NumWarpsHist;
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
    int32_t globalThreadIdx = blockIdx.x * NumThreads + threadIdx.x;
    int32_t globalThreadStride = gridDim.x * NumThreads;
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
            params.mPtrExpertIdx[tokenIdx * params.mTopK + laneIdx] = packedScore;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream)
{
    TLLM_CHECK_WITH_INFO(data.mPtrExpertIdx != nullptr || data.mPtrScores != nullptr,
        "Routing kernel requires at least one input parameter");
    TLLM_CHECK_WITH_INFO(data.mPtrPermutedIdxSize != nullptr && data.mPtrCtaIdxXyToBatchIdx != nullptr
            && data.mPtrCtaIdxXyToMnLimit != nullptr && data.mPtrNumNonExitingCtas != nullptr,
        "Llama4 routing kernel expects permuted idx and grouped Gemm launch config buffers");
    TLLM_CHECK_WITH_INFO(data.mTopK <= MaxNumTopExperts, "Routing kernel expects topK experts <= %d, got %d",
        MaxNumTopExperts, data.mTopK);
    TLLM_CHECK_WITH_INFO(data.mNumExperts <= MaxNumExperts,
        "Routing kernel expects #experts %d to be at most max #experts %d", data.mNumExperts, MaxNumExperts);
    static_assert(MaxNumExperts <= NumThreads, "#experts must be bounded by #threads");
    static_assert(MaxNumExperts <= NumThreadsHist, "#experts must be bounded by #threads");
    TLLM_CHECK_WITH_INFO(
        data.mNumExperts % 4 == 0, "Routing kernel expects #experts %d to be a multiple of 4.", data.mNumExperts);
    TLLM_CHECK_WITH_INFO(data.mPaddingLog2 < 8, "Routing kernel expects padding log2 < 8, got %d", data.mPaddingLog2);

    bool const useSingleCluster
        = data.mNumTokens <= (data.mPtrScores != nullptr ? MaxNumTokensSingleClusterScores : MaxNumTokensSingleCluster);

    if (!useSingleCluster)
    {
        TLLM_CHECK_WITH_INFO(
            data.mPtrExpertIdx != nullptr, "When #tokens is large, `mPtrExpertIdx` is a required input.");
        TLLM_CHECK_WITH_INFO(
            data.mPtrExpertCounts != nullptr, "When #tokens is large, `mPtrExpertCounts` is a required input.");
    }

    if (useSingleCluster)
    {
        LAUNCH_ROUTING_WITH_EXTRA_FLAG(data, false, routingIndicesClusterKernel, NumBlocksPerCluster, NumThreads,
            /*smemSize=*/0, // No dynamic smem
            stream, data.mDoSoftmaxBeforeTopK, /*forceFloatInput=*/false);
    }
    else
    {
        uint32_t const expandedIdxSize = data.mNumTokens * data.mTopK;

        uint32_t const histogramEltsPerBlock = 8 * NumThreadsHist;
        uint32_t const offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * NumThreadsHist;

        // Limit grid size (all kernels use a grid-stride loop).
        uint32_t const maxNumBlocks = 1024;

        int const numBlocksHistogram
            = std::min((expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
        int const numBlocksOffsets
            = std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

        if (data.mPtrScores != nullptr)
        {
            LAUNCH_ROUTING_WITH_EXTRA_FLAG(data, false, routingIndicesHistogramScoresKernel, maxNumBlocks,
                NumThreadsHist,
                /*smemSize=*/0, // No dynamic smem
                stream, data.mDoSoftmaxBeforeTopK, /*forceFloatInput=*/false);
        }
        else
        {
            // Reset the global histograms.
            TLLM_CUDA_CHECK(cudaMemsetAsync(data.mPtrExpertCounts, 0,
                static_cast<size_t>(2 * NumThreads) * sizeof(int32_t), (cudaStream_t) stream));
        }
        LAUNCH_ROUTING_WITH_EXTRA_FLAG(data, false, routingIndicesHistogramKernel, numBlocksHistogram, NumThreadsHist,
            /*smemSize=*/0, // No dynamic smem
            stream, data.mDoSoftmaxBeforeTopK, /*forceFloatInput=*/false);
        LAUNCH_ROUTING_WITH_EXTRA_FLAG(data, false, routingIndicesOffsetsKernel, numBlocksOffsets, NumThreadsHist,
            /*smemSize=*/0, // No dynamic smem
            stream, data.mDoSoftmaxBeforeTopK, /*forceFloatInput=*/false);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingRenormalize
} // namespace moe::dev::routing
