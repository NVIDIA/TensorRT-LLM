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
#include "RoutingDeepSeekCommon.cuh"

namespace moe::dev::routing
{
namespace routingDeepSeek
{

////////////////////////////////////////////////////////////////////////////////////////////////////

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
    static constexpr float invalidScoreFloat = float{-INFINITY};
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
        float topScores[KernelParams::MaxNumTopExperts]; // bound of params.mTopK
        int32_t topExperts[KernelParams::MaxNumTopExperts];

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
                    expertScoreGroup[ii]
                        = (ii < params.mNumLimitedGroups) && (groupIdx < params.mNumExpertGroups) && expertSelected
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
            int constexpr NumInterTopK = NumExpertWarps * KernelParams::MaxNumTopExperts;
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
                    smemInterTopScores[warpIdx * KernelParams::MaxNumTopExperts + laneIdx] = topScores[laneIdx];
                    smemInterTopExperts[warpIdx * KernelParams::MaxNumTopExperts + laneIdx] = topExperts[laneIdx];
                }
                else if (laneIdx >= params.mTopK && laneIdx < KernelParams::MaxNumTopExperts)
                {
                    smemInterTopScores[warpIdx * KernelParams::MaxNumTopExperts + laneIdx] = invalidScoreFloat;
                    smemInterTopExperts[warpIdx * KernelParams::MaxNumTopExperts + laneIdx]
                        = MaxSupportedExpertCount - 1;
                }
            }
            __syncthreads();
            if (warpIdx == 0)
            {
                int constexpr NumInterTopKPerThread = (NumInterTopK - 1) / WarpSize + 1;
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

void launchMainKernel(Data& data, int numBlocks, int numThreadsMain, void* stream)
{
    LAUNCH_ROUTING_DEEPSEEK(data,
        /*coopLaunch=*/false, routingMainKernel, numBlocks, numThreadsMain,
        /*smemSize=*/0, // No dynamic smem
        stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingDeepSeek
} // namespace moe::dev::routing
