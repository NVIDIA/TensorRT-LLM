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

namespace routingDeepSeek
{

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumKimiK2Experts = 384;
static constexpr int NumDeepseekExperts = 256;
static constexpr int NumTopGroupScores = 2;
static constexpr int MaxNumTopExperts = 8;
static constexpr int MaxNumTopGroups = 4;
static constexpr int MaxNumGroups = 8;

template <typename KernelParams>
__global__ void routingMainKernel(KernelParams params)
{
    // declare types
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;

    // declare shared memory structure
    // number of experts is bounded by number of threads
    __shared__ float __attribute((aligned(128))) smemScoreSigmoid[KernelParams::MaxNumExperts];
    __shared__ float __attribute((aligned(128))) smemScoreBias[KernelParams::MaxNumExperts];
    // number of expert groups is bounded by number of warps
    __shared__ float __attribute((aligned(128))) smemGroupScores[MaxNumGroups];

    // needed for warp reduce
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    // for the final reduction of weight norm, only some lanes need to participate
    int32_t laneIdx = threadIdx.x % WarpSize;
    int32_t warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    // warps outside the range of expert groups do not participate
    if constexpr (KernelParams::UseGroups)
    {
        if (warpIdx >= params.mNumExpertGroups)
        {
            return;
        }
    }

    // note that for invalid scores, we simply use a negative value:
    // they work well even with the compacted format used in topK, and
    // sigmoid / bias activated scores cannot be negative
    static constexpr float invalidScoreFloat = -1.F;
    const OutputT invalidScore = OutputT{invalidScoreFloat};

    // load bias already; each warp represents one expert group
    auto threadExpert = threadIdx.x;
    bool expertSelected = threadExpert < params.mNumExperts;
    if constexpr (KernelParams::UseGroups)
    {
        threadExpert = warpIdx * params.mNumExpertsPerGroup + laneIdx;
        expertSelected = laneIdx < params.mNumExpertsPerGroup;
    }
    auto scoreIdx = int64_t{blockIdx.x} * int64_t{params.mNumExperts} + threadExpert;
    auto biasVal = expertSelected ? params.mPtrRoutingBias[threadExpert] : invalidScore;

    // initialize the mPtrExpertCounts
    if (params.mPtrExpertCounts)
    {
        int32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
        int32_t globalThreadStride = gridDim.x * blockDim.x;
        int32_t expertCountsNum = 2 * params.mNumExperts;
        initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);
    }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // trigger the secondary kernel when using PDL, then wait on primary
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
#endif

    if (params.mPtrScores != nullptr)
    {
        // get our assigned thread score; each warp represents one expert group
        float score = expertSelected ? static_cast<float>(params.mPtrScores[scoreIdx]) : invalidScoreFloat;
        // get the sigmoid score
        // note that for invalid values, we simply use a negative value:
        // sigmoig scores are always strictly positive
        auto scoreSigmoid = sigmoid_accurate(score);
        // write the sigmoid score to shared for later use
        if (expertSelected)
        {
            smemScoreSigmoid[threadExpert] = scoreSigmoid;
        }
        // get the score with bias
        // note that with invalid values, because sigmoid is < 1 and bias is -1,
        // we must get a negative value, which is smaller than any valid value
        auto scoreBias = float{scoreSigmoid + float{biasVal}};

        if (expertSelected)
        {
            smemScoreBias[threadExpert] = scoreBias;
        }

        // registers for top group score reduction
        float topExpGroupScores[NumTopGroupScores];
        [[maybe_unused]] int32_t topExpGroupIdx[NumTopGroupScores];
        float topGroups[MaxNumTopGroups]; // bound of params.mNumLimitedGroups
        int32_t topGroupIdx[MaxNumTopGroups];
        float expertScoreGroup[MaxNumTopGroups];
        int32_t expertIdxGroup[MaxNumTopGroups];
        float topScores[MaxNumTopExperts]; // bound of params.mTopK
        int32_t topExperts[MaxNumTopExperts];

        if constexpr (KernelParams::UseGroups)
        {
            topk::reduceTopK(warp, topExpGroupScores, topExpGroupIdx, scoreBias, threadExpert,
                /* minValue */ invalidScoreFloat);
            // get the final group score and write it to shared
            if (cute::elect_one_sync())
            {
                auto groupScore = topExpGroupScores[0] + topExpGroupScores[1];
                smemGroupScores[warpIdx] = groupScore;
            }
        }

        // make group scores available to all warps
        __syncthreads();

        auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
        if constexpr (KernelParams::UseGroups)
        { // a single warp performs the selection of top groups, and goes on to select the final experts
            if (warpIdx == 0)
            {
                float groupScore = laneIdx < params.mNumExpertGroups ? smemGroupScores[laneIdx] : invalidScoreFloat;
                topk::reduceTopK(warp, topGroups, topGroupIdx, groupScore, laneIdx,
                    /* minValue */ invalidScoreFloat);
                // final expert selection: get relevant indexes and scores from shared

#pragma unroll
                for (int ii = 0; ii < MaxNumTopGroups; ++ii)
                { // bound of params.mNumLimitedGroups
                    auto groupIdx = topGroupIdx[ii];
                    expertIdxGroup[ii] = groupIdx * params.mNumExpertsPerGroup + laneIdx;
                    // note: expertSelected implies laneIdx < params.mNumExpertsPerGroup.
                    // we have params.mNumExpertsPerGroup == params.mNumExperts / params.mNumExpertGroups,
                    // thus groupIdx <= params.mNumExpertGroups - 1 =>
                    // groupIdx * params.mNumExpertsPerGroup <= params.mNumExperts - params.mNumExpertsPerGroup
                    // => expertIdxGroup[ii] < params.mNumExperts <= NumThreads,
                    // so the access is safe here
                    expertScoreGroup[ii] = groupIdx < params.mNumExpertGroups && expertSelected
                        ? smemScoreBias[expertIdxGroup[ii]]
                        : invalidScoreFloat;
                }

                topk::reduceTopK(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup,
                    /* minValue */ invalidScoreFloat, params.mTopK);
            }
        }
        else if constexpr (KernelParams::MaxNumExperts > topk::MaxNumExpertsUnit)
        {
            // without groups, each thread just takes `MaxNumTopGroups` experts
            int constexpr NumExpertWarps = (KernelParams::MaxNumExperts - 1) / topk::MaxNumExpertsUnit + 1;
            int constexpr NumInterTopK = NumExpertWarps * MaxNumTopExperts;
            __shared__ float __attribute((aligned(128))) smemInterTopScores[NumInterTopK];
            __shared__ int32_t __attribute((aligned(128))) smemInterTopExperts[NumInterTopK];
            if (warpIdx < NumExpertWarps)
            {
                int offset = warpIdx * WarpSize * MaxNumTopGroups;
#pragma unroll
                for (int ii = 0; ii < MaxNumTopGroups; ++ii)
                {
                    auto expertIdx = ii * WarpSize + laneIdx;
                    expertIdxGroup[ii] = offset + expertIdx;
                    expertScoreGroup[ii] = offset + expertIdx < params.mNumExperts ? smemScoreBias[offset + expertIdx]
                                                                                   : invalidScoreFloat;
                }
                topk::reduceTopK(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup,
                    /* minValue */ invalidScoreFloat, params.mTopK);

                if (laneIdx < params.mTopK)
                {
                    smemInterTopScores[warpIdx * MaxNumTopExperts + laneIdx] = topScores[laneIdx];
                    smemInterTopExperts[warpIdx * MaxNumTopExperts + laneIdx] = topExperts[laneIdx];
                }
            }
            __syncthreads();
            if (warpIdx == 0)
            {
                int constexpr NumInterTopKPerThread = (NumInterTopK * NumExpertWarps - 1) / WarpSize + 1;
                float intermidiateScore[NumInterTopKPerThread];
                int32_t intermidiateExpert[NumInterTopKPerThread];
                for (int i = laneIdx; i < NumInterTopKPerThread * WarpSize; i += WarpSize)
                {
                    int ii = i / WarpSize;
                    if (i < NumInterTopK)
                    {
                        intermidiateScore[ii] = smemInterTopScores[i];
                        intermidiateExpert[ii] = smemInterTopExperts[i];
                    }
                    else
                    {
                        intermidiateScore[ii] = invalidScoreFloat;
                        intermidiateExpert[ii] = KernelParams::MaxNumExperts - 1;
                    }
                }
                topk::reduceTopK(warp, topScores, topExperts, intermidiateScore, intermidiateExpert,
                    /* minValue */ invalidScoreFloat, params.mTopK);
            }
        }
        else
        {
            if (warpIdx == 0)
            {
                // without groups, each thread just takes `MaxNumTopGroups` experts
#pragma unroll
                for (int ii = 0; ii < MaxNumTopGroups; ++ii)
                {
                    auto expertIdx = ii * WarpSize + laneIdx;
                    expertIdxGroup[ii] = expertIdx;
                    expertScoreGroup[ii]
                        = expertIdx < params.mNumExperts ? smemScoreBias[expertIdx] : invalidScoreFloat;
                }
                topk::reduceTopK(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup,
                    /* minValue */ invalidScoreFloat, params.mTopK);
            }
        }

        if (warpIdx == 0)
        {
            // determine our lane's expert index and write to output
            int32_t expertIdx = 0;
#pragma unroll
            for (int ii = 0; ii < params.mTopK; ++ii)
            { // bound of params.mTopK
                expertIdx = laneIdx == ii ? topExperts[ii] : expertIdx;
            }
            // determine whether our expert is local to this GPU
            auto localExpertIdx = expertIdx - params.mLocalExpertsStartIdx;
            auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
                && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;

            float scoreNorm = laneIdx < params.mTopK ? smemScoreSigmoid[expertIdx] : 0.F;
            auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
            auto finalScore = OutputT{scoreNorm * params.mRouteScale / redNorm};

            // write expert idx out already
            auto idxTopK = blockIdx.x * params.mTopK + laneIdx;
            if (laneIdx < params.mTopK && params.mPtrTopKPacked != nullptr)
            {
                PackedScoreIdx<OutputT> packedScore{static_cast<OutputT>(finalScore), static_cast<int16_t>(expertIdx)};
                params.mPtrTopKPacked[idxTopK] = packedScore;
            }

            if (laneIdx < params.mTopK && params.mPtrTopKWeights != nullptr && params.mPtrTopKIds == nullptr)
            {
                params.mPtrTopKWeights[idxTopK] = finalScore;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(KernelParams::MaxNumExperts)
    routingIndicesClusterKernel(KernelParams params)
{
    using OutputT = typename KernelParams::OutputT;

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    int32_t const clusterBlockRank = blockIdx.x;

    //@todo: try to move it into routingPermutation
    // then wait on primary grid
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
    routingPermutation<KernelParams, OutputT, KernelParams::MaxNumExperts, KernelParams::MaxNumExperts / WarpSize,
        MaxNumTopExperts, /*LoadExpertIdxFromGlobal=*/true>(params, nullptr, warpIdx, clusterBlockRank);
}
#else
__global__ void routingIndicesClusterKernel(KernelParams params)
{
    assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __launch_bounds__(KernelParams::MaxNumExperts) routingIndicesCoopKernel(KernelParams params)
{
    // number of experts is bounded by number of threads
    int constexpr NumThreads = KernelParams::MaxNumExperts;
    __shared__ int32_t __attribute((aligned(128))) smemExpertCount[NumThreads];
    __shared__ int32_t __attribute((aligned(128))) smemExpertOffset[NumThreads];
    // needed for the exclusive sum of token offsets
    using Scan = cub::BlockScan<int32_t, NumThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename Scan::TempStorage tempStorage;
    // 64 elements -> 128+ registers. Above that we may start to see spilling to local memory.
    static constexpr int MaxExpandedIdxPerThread = 64;

    // Initialize grid.
    cg::grid_group grid = cg::this_grid();
    // Note: the following is more efficient than grid.block_index() because we don't use y and z.
    int32_t const gridBlockIdx = blockIdx.x;
    int32_t const gridThreadIdx = NumThreads * gridBlockIdx + threadIdx.x;
    int32_t const numBlocks = gridDim.x;
    int32_t const numThreadsPerGrid = numBlocks * NumThreads;

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);

    auto expandedIdxSize = params.mNumTokens * params.mTopK;

    // pre-fill the counts with 0
    smemExpertCount[threadIdx.x] = 0;
    __syncthreads();

    // then wait on primary grid
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }

    // each thread keeps has some number of "expanded indexes" assigned to it
    // for each of these, we keep the associated expert and offset within expert in registers
    int32_t expertIndexes[MaxExpandedIdxPerThread];
    int32_t expertOffsets[MaxExpandedIdxPerThread];
    auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
    // In order to avoid a serialization LDG-ATOMS-LDG-ATOMS-..., we skip multiple iterations at a
    // time, and branch between a fast path without bound checks and a slow path with bound checks.
    int constexpr IterStride = 4;
    static_assert(MaxExpandedIdxPerThread % IterStride == 0);

    // Define a lambda to avoid code duplication in both branches.
    auto loopBody = [&](int ii, int expandedIdx)
    {
        int32_t expertIdx
            = params.mPtrTopKIds != nullptr ? params.mPtrTopKIds[expandedIdx] : params.mPtrTopKPacked[expandedIdx].idx;
        expertIndexes[ii] = expertIdx;
        // check whether this expert is local to our GPU at all and ignore if not
        auto localExpertIdx = expertIdx - params.mLocalExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
            && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
        expertOffsets[ii] = isLocalExpert ? atomicAdd(smemExpertCount + expertIdx, 1) : 0;
    };

#pragma unroll
    for (int32_t ii0 = 0; ii0 < MaxExpandedIdxPerThread; ii0 += IterStride)
    {
        // Whether it's safe to do multiple iterations without bound checks.
        bool const takeFastPath = (ii0 + IterStride) * numThreadsPerGrid <= expandedIdxSize;
        if (takeFastPath)
        {
#pragma unroll
            for (int32_t jj = 0; jj < IterStride; jj++)
            {
                int const ii = ii0 + jj;
                auto expandedIdx = static_cast<int32_t>(gridThreadIdx) + ii * numThreadsPerGrid;
                loopBody(ii, expandedIdx);
            }
        }
        else
        {
            bool doBreak = false;
#pragma unroll
            for (int32_t jj = 0; jj < IterStride; jj++)
            {
                int const ii = ii0 + jj;
                auto expandedIdx = static_cast<int32_t>(gridThreadIdx) + ii * numThreadsPerGrid;
                if (expandedIdx >= expandedIdxSize)
                {
                    doBreak = true;
                    break;
                }
                loopBody(ii, expandedIdx);
            }
            if (doBreak)
            {
                break;
            }
        }
    }

    // Make histogram (token counts per expert) available to all threads in the block.
    __syncthreads();

    //
    // Each thread now represents one expert
    //

    // Add the local bin count to the common bin count and get a per-CTA offset.
    int32_t const localExpertCount = smemExpertCount[threadIdx.x];

    int32_t blockExpertOffset = 0;
    if (threadIdx.x < params.mNumExperts)
    {
        blockExpertOffset = atomicAdd(&params.mPtrExpertCounts[threadIdx.x], localExpertCount);
    }

    // Sync to wait for completion of the histogram reduction.
    grid.sync();

    // Get total count for this expert.
    int32_t count = (threadIdx.x < params.mNumExperts) ? params.mPtrExpertCounts[threadIdx.x] : 0;

    // Note: the scan is redundant in all CTAs, but doing it in only 1 CTA would be worse for latency.

    // Compute the runtime config for projections
    // Whether or not an expert is local is taken into account when smemExpertCount is computed
    // so we do not need to take it into account here.

    int32_t numCta;
    if constexpr (KernelParams::isPow2)
    {
        numCta = divUpLog2<int32_t>(count, params.mPaddingLog2);
    }
    else
    {
        numCta = divUpTileN<int32_t>(count, params.mTileTokensDim);
    }

    int32_t ctaOffset;
    int32_t numNonExitingCtas;
    Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);

    for (int32_t cta = gridBlockIdx; cta < numCta; cta += numBlocks)
    {
        const int32_t localExpertIdx = (threadIdx.x - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
        params.mPtrCtaIdxXyToBatchIdx[ctaOffset + cta] = localExpertIdx;
        int32_t mnLimit1;
        int32_t mnLimit2;
        if constexpr (KernelParams::isPow2)
        {
            mnLimit1 = mulLog2<int32_t>(ctaOffset + cta + 1, params.mPaddingLog2);
            mnLimit2 = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2) + count;
        }
        else
        {
            mnLimit1 = mulTileN<int32_t>(ctaOffset + cta + 1, params.mTileTokensDim);
            mnLimit2 = mulTileN<int32_t>(ctaOffset, params.mTileTokensDim) + count;
        }
        params.mPtrCtaIdxXyToMnLimit[ctaOffset + cta] = min(mnLimit1, mnLimit2);
    }

    // get the padded offset associated with this expert
    int32_t offset;
    if constexpr (KernelParams::isPow2)
    {
        offset = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2);
    }
    else
    {
        offset = mulTileN<int32_t>(ctaOffset, params.mTileTokensDim);
    }
    int32_t permutedIdxSize;
    if constexpr (KernelParams::isPow2)
    {
        permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
    }
    else
    {
        permutedIdxSize = mulTileN<int32_t>(numNonExitingCtas, params.mTileTokensDim);
    }

    // write out padded count
    if (gridBlockIdx == 0 && warpIdx == NumThreads / WarpSize - 1 && cute::elect_one_sync())
    {
        params.mPtrPermutedIdxSize[0] = permutedIdxSize;
        params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
    }

    // write expert offsets to shared
    smemExpertOffset[threadIdx.x] = offset + blockExpertOffset;

    // make expert offsets available to all threads
    __syncthreads();

    // trigger the secondary kernel when using PDL
    // We can't do it earlier because FC1 depends on the mPtrCtaIdxXyToBatchIdx,
    // mPtrCtaIdxXyToMnLimit, mPtrNumNonExitingCtas and mPtrTotalNumPaddedTokens
    // TODO: this is not sufficient to ensure visibility in the next kernel!
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }

// each thread has the same "expanded indexes" assigned to it as above
// at this point, we know the final offsets of experts and the offsets within
// experts, which allows writing the final index values
#pragma unroll
    for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ++ii)
    {
        auto expandedIdx = static_cast<int32_t>(gridThreadIdx) + ii * numThreadsPerGrid;
        if (expandedIdx >= expandedIdxSize)
        {
            break;
        }
        auto expertIdx = expertIndexes[ii];
        // check whether this expert is local to our GPU at all
        auto localExpertIdx = static_cast<int32_t>(expertIdx) - params.mLocalExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
            && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
        auto tokenIdx = expandedIdx / params.mTopK;
        auto permutedIdx = isLocalExpert ? int32_t{smemExpertOffset[expertIdx]} + expertOffsets[ii] : int32_t{-1};
        if (params.mPtrExpandedIdxToPermutedIdx != nullptr)
        {
            params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
        }
        if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocalExpert)
        {
            params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
        }
    }
}
#else
__global__ void routingIndicesCoopKernel(KernelParams params)
{
    assert(false && "routingIndicesCoopKernel is only supported on SM90+ architectures");
}
#endif

int constexpr getMaxNumExperts(int32_t numExperts)
{
    if (numExperts <= topk::MaxNumExpertsUnit)
    {
        return topk::MaxNumExpertsUnit;
    }
    else if (numExperts <= NumDeepseekExperts)
    {
        return NumDeepseekExperts;
    }
    else if (numExperts <= NumKimiK2Experts)
    {
        return NumKimiK2Experts;
    }
    else
    {
        TLLM_LOG_ERROR("Unsupported numExperts");
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LAUNCH_ROUTING_DEEPSEEK(                                                                                       \
    data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1, forceFloatInput)                    \
    if (data.mNumExperts <= topk::MaxNumExpertsUnit)                                                                   \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_NUM_EXPERTS_FORCE_FLOAT_INPUT(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,   \
            stream, extraFlag1, forceFloatInput, topk::MaxNumExpertsUnit);                                             \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumDeepseekExperts)                                                                   \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_NUM_EXPERTS_FORCE_FLOAT_INPUT(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,   \
            stream, extraFlag1, forceFloatInput, NumDeepseekExperts);                                                  \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumKimiK2Experts)                                                                     \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_NUM_EXPERTS_FORCE_FLOAT_INPUT(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,   \
            stream, extraFlag1, forceFloatInput, NumKimiK2Experts);                                                    \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_LOG_ERROR("Unsupported numExperts");                                                                      \
    }

void run(Data& data, void* stream)
{
    TLLM_CHECK_WITH_INFO(data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr || data.mPtrTopKIds != nullptr,
        "Routing kernel requires at least one input parameter");
    if (data.mPtrTopKIds != nullptr)
    {
        TLLM_CHECK_WITH_INFO(data.mPtrTopKWeights != nullptr,
            "When mPtrTopKIds is provided, mPtrTopKWeights must also be provided for DeepSeek routing.");
    }
    if (data.mPtrExpandedIdxToPermutedIdx != nullptr || data.mPtrPermutedIdxToTokenIdx != nullptr)
        TLLM_CHECK_WITH_INFO(
            (data.mPtrTopKPacked != nullptr || data.mPtrTopKIds != nullptr) && data.mPtrPermutedIdxSize,
            "If permuted index is required, `mPtrTopKPacked` or `mPtrTopKIds` is also required");
    TLLM_CHECK_WITH_INFO(!data.mUseRoutingSoftmax, "Routing with softmax not implemented yet");
    TLLM_CHECK_WITH_INFO(data.mNumLimitedGroups <= MaxNumTopGroups, "Routing kernel expects <= %d top groups, got %d",
        MaxNumTopGroups, data.mNumLimitedGroups);
    TLLM_CHECK_WITH_INFO(data.mTopK <= MaxNumTopExperts, "Routing kernel expects topK experts <= %d, got %d",
        MaxNumTopExperts, data.mTopK);
    TLLM_CHECK_WITH_INFO(data.mTopK <= WarpSize, "Routing kernel expects top K <= warp size, got %d", data.mTopK);
    TLLM_CHECK_WITH_INFO(data.mTopK * data.mNumLimitedGroups <= WarpSize,
        "Routing kernel expects top K * top groups <= warp size (for now), got %d * %d", data.mTopK,
        data.mNumLimitedGroups);
    TLLM_CHECK_WITH_INFO(data.mNumExperts >= MaxNumTopExperts, "Routing kernel expects %d to be at most #experts %d",
        MaxNumTopExperts, data.mNumExperts);
    TLLM_CHECK_WITH_INFO(data.mNumExperts <= NumKimiK2Experts, "Routing kernel expects #experts %d  <= #threads %d",
        data.mNumExperts, NumKimiK2Experts);
    TLLM_CHECK_WITH_INFO(data.mNumExpertGroups >= data.mNumLimitedGroups,
        "Routing kernel expects top groups %d to be limited by #expert groups %d", data.mNumLimitedGroups,
        data.mNumExpertGroups);
    if (data.mNumExpertGroups > 1)
    {
        TLLM_CHECK_WITH_INFO(data.mNumExpertGroups <= MaxNumGroups,
            "Routing kernel expects #experts groups %d to be <= #warps %d", data.mNumExpertGroups, MaxNumGroups);
        TLLM_CHECK_WITH_INFO(data.mNumExperts % data.mNumExpertGroups == 0,
            "Routing kernel expects #experts %d to be a multiple of #expert groups %d", data.mNumExperts,
            data.mNumExpertGroups);
        TLLM_CHECK_WITH_INFO(data.mNumExperts / data.mNumExpertGroups <= WarpSize,
            "Routing kernel expects #experts per group <= warp size, got %d, data.mNumExpertGroups %d",
            data.mNumExperts / data.mNumExpertGroups, data.mNumExpertGroups);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(data.mTopK <= topk::MaxNumTopK, "Routing kernel expects top K %d to be <= #warps %d",
            data.mTopK, topk::MaxNumTopK);
    }
    TLLM_CHECK_WITH_INFO(
        data.mNumExperts % 4 == 0, "Routing kernel expects #experts %d to be a multiple of 4.", data.mNumExperts);
    int const numBlocks = data.mNumTokens;
    int const numThreadsHist = getMaxNumExperts(data.mNumExperts);

    bool const useSingleCluster = data.mNumTokens <= 1024;
    if (!useSingleCluster)
    {
        // Reset the global histograms (not used in single-cluster code path).
        // Cover both for the cooperative and two-kernel code paths.
        TLLM_CHECK_WITH_INFO(
            data.mPtrExpertCounts != nullptr, "When #tokens is large, `mPtrExpertCounts` is a required input.");
    }
    else
    {
        data.mPtrExpertCounts = nullptr; // Set it to nullptr for single-cluster code path, as it won't be used
    }

    // Number of blocks we can use in the cooperative kernel
    // The number of blocks must be:
    //   >= ⌈(numTokens * topK) / (MaxExpandedIdxPerThread * NumThreads)⌉
    //   <= numSms, assuming an occupancy of 1 block/SM
    //
    // If too small for the given numTokens, fall back to the less performant two-step method.
    //
    // The upper bound is a strict requirement. The number of blocks should be determined by querying
    // the device properties, or conservatively low.
    // /!\ The following number is not portable!! (but works on H100 and B200)
    int const numBlocksCoop = 128;

    // Maximum number of tokens supported by the kernel using a cooperative launch.
    int const maxTokensCoop = (numBlocksCoop * numThreadsHist * 64) / data.mTopK;
    if (data.mPtrTopKIds == nullptr)
    {
        int const numThreadsMain = data.mNumExperts < NumDeepseekExperts ? NumDeepseekExperts : NumKimiK2Experts;
        LAUNCH_ROUTING_DEEPSEEK(data,
            /*coopLaunch=*/false, routingMainKernel, numBlocks, numThreadsMain,
            /*smemSize=*/0, // No dynamic smem
            stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);
    }
    else
    {
        // Reset the global histograms.
        LAUNCH_ROUTING_DEEPSEEK(data, false, routingInitExpertCounts, (2 * data.mNumExperts - 1) / numThreadsHist + 1,
            numThreadsHist,
            /*smemSize=*/0, // No dynamic smem
            stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/false);
    }

    if (data.mPtrPermutedIdxSize != nullptr)
    {
        if (useSingleCluster)
        {
            LAUNCH_ROUTING_DEEPSEEK(data,
                /*coopLaunch=*/false, routingIndicesClusterKernel, NumBlocksPerCluster, numThreadsHist,
                /*smemSize=*/0, // No dynamic smem
                stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);
        }
        else if (data.mNumTokens <= maxTokensCoop)
        {
            LAUNCH_ROUTING_DEEPSEEK(data,
                /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop, numThreadsHist,
                /*smemSize=*/0, // No dynamic smem
                stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);
        }
        else
        {
            const int32_t expandedIdxSize = data.mNumTokens * data.mTopK;
            const int32_t histogramEltsPerBlock = 8 * numThreadsHist;
            const int32_t offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * numThreadsHist;

            // Limit grid size (both kernels use a grid-stride loop).
            const int32_t maxNumBlocks = 1024;

            int const numBlocksHistogram
                = std::min((expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
            int const numBlocksOffsets
                = std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

            LAUNCH_ROUTING_DEEPSEEK(data,
                /*coopLaunch=*/false, routingIndicesHistogramKernel, numBlocksHistogram, numThreadsHist,
                /*smemSize=*/0, // No dynamic smem
                stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);
            LAUNCH_ROUTING_DEEPSEEK(data,
                /*coopLaunch=*/false, routingIndicesOffsetsKernel, numBlocksOffsets, numThreadsHist,
                /*smemSize=*/0, // No dynamic smem
                stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingDeepSeek
} // namespace moe::dev::routing
