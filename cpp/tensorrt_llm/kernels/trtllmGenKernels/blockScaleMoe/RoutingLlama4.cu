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
namespace routingLlama4
{

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumThreads = 1024;
static constexpr int NumWarps = NumThreads / WarpSize;
static constexpr int MaxNumTopExperts = 1;
static constexpr int MaxNumExperts = 128;
static constexpr int MaxNumTokensSingleCluster = NumBlocksPerCluster * NumThreads;
static constexpr int MaxNumTokensSingleClusterScores = NumBlocksPerCluster * NumWarps;
static constexpr int WarpKernelSmemStride = 33;
// with further optimization to `routingIndicesWarpKernel`, this limit may
// increase. For now, it is a good cut-off point for when the block-wise
// operations are more efficient end-to-end.
static constexpr int WarpKernelMaxNumTokens = 4;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DataType, int VecSize>
__forceinline__ __device__ void routingTopKExperts(cg::thread_block_tile<WarpSize> const& warp,
    DataType (&warpMaxScore)[MaxNumTopExperts], int32_t (&warpMaxExpertIdx)[MaxNumTopExperts], int32_t const laneIdx,
    int32_t const numExperts, DataType const* ptrScores)
{
    DataType minScore = DataType{-INFINITY};
    DataType maxScore = minScore;
    int32_t maxExpertIdx{-1};
    using DataTypeVec = std::conditional_t<sizeof(DataType) == 2, float2, float4>;

    // Non-vectorized loading: directly access ptrScores with expertIdx
    for (int i = 0; i < VecSize; ++i)
    {
        auto expertIdx = i * WarpSize + laneIdx;
        auto newScore = expertIdx < numExperts ? ptrScores[expertIdx] : minScore;
        // note: use `>=` s.t. highest index always wins, just like in `reduceTopK`
        if (newScore > maxScore)
        {
            maxScore = newScore;
            maxExpertIdx = expertIdx;
        }
    }

    topk::reduceTopK(warp, warpMaxScore, warpMaxExpertIdx, maxScore, maxExpertIdx, minScore);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void __launch_bounds__(WarpSize) routingIndicesWarpKernel(KernelParams params)
{
    // types used in this kernel
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;
    using TypePacked = PackedScoreIdx<OutputT>;
    // use the default cub warp-scan, with shfl
    using Scan = cub::WarpScan<int32_t>;
    __shared__ typename Scan::TempStorage tempStorage;

    // each thread encodes 4 experts in one `int32_t`. The assumption is that
    // we don't have more than 127 tokens, but `WarpKernelMaxNumTokens` must be
    // smaller than that because other approaches will be more efficient for
    // 127 tokens.
    static constexpr int ExpertsPerThread = sizeof(int32_t);
    static_assert(WarpKernelMaxNumTokens <= 127);
    // this is a full table of which token is routed to which expert.
    // the assumption here is that there are no more than 128 experts.
    // we use a stride of 33 instead of 32 to avoid shared memory bank conflicts.
    __shared__ int32_t __attribute((aligned(128)))
    smemExpertTokenCountFull[WarpKernelMaxNumTokens][WarpKernelSmemStride];
    static_assert(WarpKernelSmemStride == WarpSize + 1);
    static_assert(MaxNumExperts / sizeof(int32_t) <= WarpSize);

    // values needed for the top-1 reduction, if required
    InputT minScore = InputT{-INFINITY};
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

#pragma unroll
    for (int tokenIdx = 0; tokenIdx < WarpKernelMaxNumTokens; ++tokenIdx)
    {
        // reset full shared memory field to 0
        smemExpertTokenCountFull[tokenIdx][threadIdx.x] = 0;
    }
    __syncwarp();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // then wait on primary grid
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif

    if (params.mPtrScores != nullptr)
    {
        // if we use `mPtrScores` as input, we need to perform the top-1 reduction
        // for each token, we load the scores then use `reduceTopK` for this.
        // each thread works on 4 experts, so a local reduction is done before
        for (int tokenIdx = 0; tokenIdx < params.mNumTokens; ++tokenIdx)
        {
            auto scoreOffset = tokenIdx * params.mNumExperts;
            int32_t warpMaxExpertIdx[MaxNumTopExperts];
            InputT warpMaxScore[MaxNumTopExperts];

            // Use routingTopKExperts function instead of inline logic
            routingTopKExperts<InputT, ExpertsPerThread>(
                warp, warpMaxScore, warpMaxExpertIdx, threadIdx.x, params.mNumExperts, params.mPtrScores + scoreOffset);

            if (cute::elect_one_sync())
            {
                // one thread updates the count linking token to chosen expert
                auto expertTokenCount = 0;
                setBits</* IsZero= */ true>(expertTokenCount, 1, warpMaxExpertIdx[0] % ExpertsPerThread);
                smemExpertTokenCountFull[tokenIdx][warpMaxExpertIdx[0] / ExpertsPerThread] = expertTokenCount;
                // we also compute the final score here and write it out if required
                auto finalScore = OutputT{sigmoid_accurate(float{warpMaxScore[0]})};
                if (params.mPtrExpertWeights != nullptr)
                {
                    params.mPtrExpertWeights[tokenIdx] = finalScore;
                }
            }
        }
    }
    else
    {
        // if we do not have `mPtrScores` as input, we expect that `mPtrExpertWeights`
        // contains the top-1 packed score and index already.
        // Each thread represents a token here, and we extract the relevant score
        // The assumption is that the #tokens is limited by warp-size
        static_assert(WarpKernelMaxNumTokens <= WarpSize);
        TypePacked scoreIdx = threadIdx.x < params.mNumTokens ? params.mPtrExpertIdx[threadIdx.x] : TypePacked{};
        int32_t expertTokenCount = 0;
        setBits</* IsZero= */ true>(expertTokenCount, 1, scoreIdx.idx % ExpertsPerThread);
        if (threadIdx.x < params.mNumTokens)
        {
            smemExpertTokenCountFull[threadIdx.x][scoreIdx.idx / ExpertsPerThread] = expertTokenCount;
        }
        // we also compute the final score here and write it out if required
        auto finalScore = OutputT{sigmoid_accurate(float{scoreIdx.score})};
        if (params.mPtrExpertWeights != nullptr && threadIdx.x < params.mNumTokens)
        {
            params.mPtrExpertWeights[threadIdx.x] = finalScore;
        }
    }

    // make the full table available to all threads
    __syncwarp();

    // at this point, each thread keeps a count of its 4 assigned experts in
    // `expertCount`, as well as the offsets for all tokens w.r.t. these 4 experts
    // in `expertOffset`.
    int32_t expertCount = 0;
    int32_t expertOffset[WarpKernelMaxNumTokens + 1];
#pragma unroll
    for (int tokenIdx = 0; tokenIdx < WarpKernelMaxNumTokens + 1; ++tokenIdx)
    {
        if (tokenIdx > params.mNumTokens)
            break;
        // simple reduction for `expertCount`, and scan for `expertOffset`
        auto expertTokenCount = tokenIdx < params.mNumTokens ? smemExpertTokenCountFull[tokenIdx][threadIdx.x] : 0;
        expertOffset[tokenIdx] = expertCount;
        expertCount += expertTokenCount;
    }

    // at this point, we are ready for the scan across all experts to get the
    // thread-wise offsets across experts
    // first, we need to reduce across our 4 experts into `numCta`
    int32_t numCta = 0;
#pragma unroll
    for (int ii = 0; ii < ExpertsPerThread; ++ii)
    {
        auto count = getBits(expertCount, ii);
        numCta += divUpLog2<int32_t>(count, params.mPaddingLog2);
    }
    // second, we perform the exclusive sum across the warp
    int32_t ctaOffset;
    int32_t numNonExitingCtas;
    Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);

    // finally, we perform a scan across our local experts, starting with the
    // warp-wide scan result (`ctaOffset`)
    auto ctaOffsetExp = ctaOffset;
#pragma unroll
    for (int ii = 0; ii < ExpertsPerThread; ++ii)
    {
        auto count = getBits(expertCount, ii);
        auto finalNumCta = divUpLog2<int32_t>(count, params.mPaddingLog2);
        auto expertIdx = threadIdx.x * ExpertsPerThread + ii;
        // during the scan for expert offsets, we can already write out
        // both `mPtrCtaIdxXyToBatchIdx` and `mPtrCtaIdxXyToMnLimit`
        for (int cta = 0; cta < finalNumCta; ++cta)
        {
            params.mPtrCtaIdxXyToBatchIdx[ctaOffsetExp + cta] = expertIdx;
            params.mPtrCtaIdxXyToMnLimit[ctaOffsetExp + cta]
                = min(mulLog2<int32_t>(ctaOffsetExp + cta + 1, params.mPaddingLog2),
                    mulLog2<int32_t>(ctaOffsetExp, params.mPaddingLog2) + count);
        }
        ctaOffsetExp += finalNumCta;
    }

    // at this point, we can write out padded count from the warp-aggregate
    if (cute::elect_one_sync())
    {
        const int32_t permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
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

    // at this point, all values for offsets are ready, except the final offsets
    // within the padded index (`permutedIdx`)
    // for this, we perform a scan similar to the one directly after the warp-scan:
    // here, we keep the local offset for each of the thread's experts in a field
    // of registers
    auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
    int32_t finalExpertOffset[ExpertsPerThread];
    finalExpertOffset[0] = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2);
#pragma unroll
    for (int ii = 1; ii < ExpertsPerThread; ++ii)
    {
        finalExpertOffset[ii]
            = finalExpertOffset[ii - 1] + divUpMulLog2<int32_t>(getBits(expertCount, ii - 1), params.mPaddingLog2);
    }

#pragma unroll
    for (int tokenIdx = 0; tokenIdx < WarpKernelMaxNumTokens; ++tokenIdx)
    {
        // at this point, we can calculate the final index:
        // we simply loop over all tokens, and all experts assigned to this thread.
        // For each pair, we determine whether that token was routed to that expert
        // based on whether the offset for that token changed.
        // we can then easily compute the final `expertIdx` and `permutedIdx` relative
        // to this token and expert, and write them out.
        if (tokenIdx >= params.mNumTokens)
            break;

#pragma unroll
        for (int ii = 0; ii < ExpertsPerThread; ++ii)
        {
            // determine whether the offset for this expert and token changes
            auto localOffsetToken = getBits(expertOffset[tokenIdx], ii);
            auto isTokenRouted = getBits(expertOffset[tokenIdx + 1], ii) > localOffsetToken;
            // the expert index of this expert
            auto expertIdx = threadIdx.x * ExpertsPerThread + ii;
            auto localExpertIdx = static_cast<int32_t>(expertIdx) - params.mLocalExpertsStartIdx;
            auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
                && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
            // the permuted index: we add the local offset relative to this expert and token
            // to the global offset from the scan for this expert
            auto permutedIdx = isLocalExpert ? finalExpertOffset[ii] + localOffsetToken : int32_t{-1};
            // write out `mPtrExpandedIdxToPermutedIdx` if required
            if (params.mPtrExpandedIdxToPermutedIdx != nullptr && isTokenRouted)
            {
                params.mPtrExpandedIdxToPermutedIdx[tokenIdx] = permutedIdx;
            }
            // write out `mPtrPermutedIdxToTokenIdx` if required
            if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocalExpert && isTokenRouted)
            {
                params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
            }
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(NumThreads)
    routingIndicesClusterKernel(KernelParams params)
{
    // number of tokens/expanded idx is bounded by total number of warps
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;
    using TypePacked = PackedScoreIdx<OutputT>;
    __shared__ TypePacked __attribute((aligned(128))) smemPackedScoreIdx[NumWarps];

    uint32_t const clusterBlockRank = blockIdx.x;
    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    int32_t const laneIdx = cutlass::arch::LaneId();

    // TODO(mjoux): expand to more tokens (possibly)
    auto warpTokenIdx = clusterBlockRank * NumWarps + warpIdx;
    auto scoreOffset = warpTokenIdx * params.mNumExperts;
    bool validToken = warpTokenIdx < params.mNumTokens;
    InputT minScore = InputT{-INFINITY};

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
        // we then exchange all token max scores, s.t. afterwards, each thread
        // represents a token
        InputT warpMaxScore[MaxNumTopExperts];
        int32_t warpMaxExpertIdx[MaxNumTopExperts];

        if (validToken)
        {
            routingTopKExperts<InputT, MaxNumExperts / WarpSize>(
                warp, warpMaxScore, warpMaxExpertIdx, laneIdx, params.mNumExperts, params.mPtrScores + scoreOffset);
            if (cute::elect_one_sync())
            {
                auto finalScore = OutputT{sigmoid_accurate(float{warpMaxScore[0]})};
                TypePacked packedScore{finalScore, static_cast<int16_t>(warpMaxExpertIdx[0])};
                smemPackedScoreIdx[warpIdx] = packedScore;
            }
        }
        // make packed scores available to all threads in cluster
        __cluster_barrier_arrive();
        __cluster_barrier_wait();
    }

    routingPermutation<KernelParams, OutputT, NumThreads, NumWarps, MaxNumTopExperts,
        /*LoadExpertIdxFromGlobal=*/false>(params, smemPackedScoreIdx, warpIdx, clusterBlockRank);
}
#else
__global__ void routingIndicesClusterKernel(KernelParams params)
{
    assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// this kernel is needed in case we have scores as input for the histogram kernel
template <typename KernelParams>
__global__ void __launch_bounds__(NumThreadsHist) routingIndicesHistogramScoresKernel(KernelParams params)
{
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;
    using TypePacked = PackedScoreIdx<OutputT>;
    static constexpr int VecSize = MaxNumExperts / WarpSize;
    //  we assume that #experts is a multiple of 4, so VecSize must be 4.
    static_assert(VecSize == 4);

    int32_t const laneIdx = cutlass::arch::LaneId();
    int32_t const warpIdx = threadIdx.x / WarpSize;
    int32_t const globalWarpIdx = blockIdx.x * NumWarpsHist + warpIdx;
    int32_t const globalWarpStride = gridDim.x * NumWarpsHist;
    InputT minScore = InputT{-INFINITY};
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

    // initialize the mPtrExpertCounts
    int32_t expertCountsNum = 2 * params.mNumExperts;
    int32_t globalThreadIdx = blockIdx.x * NumThreads + threadIdx.x;
    int32_t globalThreadStride = gridDim.x * NumThreads;
    initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // Wait on primary grid and trigger secondary kernel.
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif

    // in this case, each warp represents a token, and we use a grid-stride loop
    // over all warps/tokens
    for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens; tokenIdx += globalWarpStride)
    {
        auto scoreOffset = tokenIdx * params.mNumExperts;
        int32_t warpMaxExpertIdx[MaxNumTopExperts];
        InputT warpMaxScore[MaxNumTopExperts];

        routingTopKExperts<InputT, MaxNumExperts / WarpSize>(
            warp, warpMaxScore, warpMaxExpertIdx, laneIdx, params.mNumExperts, params.mPtrScores + scoreOffset);

        if (cute::elect_one_sync())
        {
            auto finalScore = OutputT{sigmoid_accurate(float{warpMaxScore[0]})};
            TypePacked packedScore{finalScore, static_cast<int16_t>(warpMaxExpertIdx[0])};
            params.mPtrExpertIdx[tokenIdx] = packedScore;
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

    bool const useSingleWarp = (data.mPtrScores == nullptr && data.mNumTokens <= WarpKernelMaxNumTokens)
        || data.mNumTokens < WarpKernelMaxNumTokens;
    bool const useSingleCluster
        = data.mNumTokens <= (data.mPtrScores != nullptr ? MaxNumTokensSingleClusterScores : MaxNumTokensSingleCluster);
    if (!useSingleCluster)
    {
        TLLM_CHECK_WITH_INFO(
            data.mPtrExpertIdx != nullptr, "When #tokens is large, `mPtrExpertIdx` is a required input.");
        TLLM_CHECK_WITH_INFO(
            data.mPtrExpertCounts != nullptr, "When #tokens is large, `mPtrExpertCounts` is a required input.");
    }

    if (useSingleWarp)
    {
        LAUNCH_ROUTING(data,
            /*coopLaunch=*/false, routingIndicesWarpKernel, 1, WarpSize,
            /*smemSize=*/0, // No dynamic smem
            stream);
    }
    else if (useSingleCluster)
    {
        LAUNCH_ROUTING(data,
            /*coopLaunch=*/false, routingIndicesClusterKernel, NumBlocksPerCluster, NumThreads,
            /*smemSize=*/0, // No dynamic smem
            stream);
    }
    else
    {
        const uint32_t expandedIdxSize = data.mNumTokens * data.mTopK;

        const uint32_t histogramEltsPerBlock = 8 * NumThreadsHist;
        const uint32_t offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * NumThreadsHist;

        // Limit grid size (all kernels use a grid-stride loop).
        const uint32_t maxNumBlocks = 1024;

        int const numBlocksHistogram
            = std::min((expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
        int const numBlocksOffsets
            = std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

        if (data.mPtrScores != nullptr)
        {
            LAUNCH_ROUTING(data,
                /*coopLaunch=*/false, routingIndicesHistogramScoresKernel, maxNumBlocks, NumThreadsHist,
                /*smemSize=*/0, // No dynamic smem
                stream);
        }
        else
        {
            // Reset the global histograms.
            TLLM_CUDA_CHECK(cudaMemsetAsync(data.mPtrExpertCounts, 0,
                static_cast<size_t>(2 * NumThreads) * sizeof(int32_t), (cudaStream_t) stream));
        }
        LAUNCH_ROUTING(data,
            /*coopLaunch=*/false, routingIndicesHistogramKernel, numBlocksHistogram, NumThreadsHist,
            /*smemSize=*/0, // No dynamic smem
            stream);
        LAUNCH_ROUTING(data,
            /*coopLaunch=*/false, routingIndicesOffsetsKernel, numBlocksOffsets, NumThreadsHist,
            /*smemSize=*/0, // No dynamic smem
            stream);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingLlama4
} // namespace moe::dev::routing
