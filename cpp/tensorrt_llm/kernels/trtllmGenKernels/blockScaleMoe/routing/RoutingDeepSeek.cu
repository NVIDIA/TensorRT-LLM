/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

// DeepSeek routing: entry point, constants, dispatch macros, kernel definitions, and launch wrappers.
//
// Kernel inventory:
//   1. routingMainKernel               — DeepSeek-specific main kernel (sigmoid + bias + group TopK)
//   2. routingIndicesClusterKernel      — single-cluster fused kernel (SM90+)
//   3. launchCoopKernel                 — delegates to routingCustom's coop implementation
//   4. launchInitExpertCounts           — zero expert counts
//   5. launchHistogramKernel            — histogram from packed TopK
//   6. launchOffsetsKernel              — prefix-scan + permutation

#include "RoutingCustomPolicy.cuh"
#include "RoutingKernel.cuh"

namespace moe::dev::routing
{

// Forward declaration of routingCustom's coop kernel (used by DeepSeek's coop path)
namespace routingCustom
{
void launchCoopKernel(Data const& data, int numBlocksCoop, uint32_t numThreadsHist, void* stream);
} // namespace routingCustom

namespace routingDeepSeek
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Constants and dispatch macros
//
////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumNemotronExperts = 512;
static constexpr int NumKimiK2Experts = 384;
static constexpr int NumDeepseekExperts = 256;
static constexpr int MaxSupportedExpertCount = std::max({NumNemotronExperts, NumKimiK2Experts, NumDeepseekExperts});
static constexpr int NumTopGroupScores = 2;
static constexpr int MaxNumTopGroups = 4;
static constexpr int MaxNumGroups = 8;

static constexpr int NumTop8Experts = 8;
static constexpr int NumTop22Experts = 22;
static constexpr int MaxSupportedTopExperts = 32;

////////////////////////////////////////////////////////////////////////////////////////////////////

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
    else if (numExperts <= NumNemotronExperts)
    {
        return NumNemotronExperts;
    }
    else
    {
        TLLM_LOG_ERROR("Unsupported numExperts");
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper macro: dispatch on topK tier for a given numExperts tier.
#define LAUNCH_DEEPSEEK_WITH_TOPK(                                                                                     \
    data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1, forceFloatInput, numExperts)        \
    if (data.mTopK <= NumTop8Experts)                                                                                  \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_NUM_EXPERTS_FORCE_FLOAT_INPUT(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,   \
            stream, extraFlag1, forceFloatInput, numExperts, NumTop8Experts);                                          \
    }                                                                                                                  \
    else if (data.mTopK <= NumTop22Experts)                                                                            \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_NUM_EXPERTS_FORCE_FLOAT_INPUT(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,   \
            stream, extraFlag1, forceFloatInput, numExperts, NumTop22Experts);                                         \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_NUM_EXPERTS_FORCE_FLOAT_INPUT(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,   \
            stream, extraFlag1, forceFloatInput, numExperts, MaxSupportedTopExperts);                                  \
    }

#define LAUNCH_ROUTING_DEEPSEEK(                                                                                       \
    data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1, forceFloatInput)                    \
    if (data.mNumExperts <= topk::MaxNumExpertsUnit)                                                                   \
    {                                                                                                                  \
        LAUNCH_DEEPSEEK_WITH_TOPK(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1,       \
            forceFloatInput, topk::MaxNumExpertsUnit);                                                                 \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumDeepseekExperts)                                                                   \
    {                                                                                                                  \
        LAUNCH_DEEPSEEK_WITH_TOPK(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1,       \
            forceFloatInput, NumDeepseekExperts);                                                                      \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumKimiK2Experts)                                                                     \
    {                                                                                                                  \
        LAUNCH_DEEPSEEK_WITH_TOPK(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1,       \
            forceFloatInput, NumKimiK2Experts);                                                                        \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumNemotronExperts)                                                                   \
    {                                                                                                                  \
        LAUNCH_DEEPSEEK_WITH_TOPK(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1,       \
            forceFloatInput, NumNemotronExperts);                                                                      \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_LOG_ERROR("Unsupported numExperts");                                                                      \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 1. Main kernel — DeepSeek-specific routing with sigmoid activation, bias, and group TopK.
//    Handles both grouped and non-grouped expert selection.
//
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
        // Inactive warps (warpIdx >= mNumExpertGroups) must NOT return early because they
        // still need to reach the __syncthreads() barriers below.  Setting expertSelected
        // to false is enough to keep them from doing any out-of-bounds reads or smem writes.
        expertSelected = (warpIdx < params.mNumExpertGroups) && (laneIdx < params.mNumExpertsPerGroup);
    }
    auto scoreIdx = int64_t{blockIdx.x} * int64_t{params.mNumExperts} + threadExpert;
    auto biasVal = expertSelected
        ? static_cast<OutputT>(loadScalar(params.mPtrRoutingBias, threadExpert, params.mDtypeBias))
        : invalidScore;

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
    if (params.mUsePdl)
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
                float intermediateScore[NumInterTopKPerThread];
                int32_t intermediateExpert[NumInterTopKPerThread];
                for (int i = laneIdx; i < NumInterTopKPerThread * WarpSize; i += WarpSize)
                {
                    int ii = i / WarpSize;
                    if (i < NumInterTopK)
                    {
                        intermediateScore[ii] = smemInterTopScores[i];
                        intermediateExpert[ii] = smemInterTopExperts[i];
                    }
                    else
                    {
                        intermediateScore[ii] = invalidScoreFloat;
                        intermediateExpert[ii] = KernelParams::MaxNumExperts - 1;
                    }
                }
                topk::reduceTopK(warp, topScores, topExperts, intermediateScore, intermediateExpert,
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
                && (localExpertIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

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

static void launchMainKernel(Data& data, int numBlocks, int numThreadsMain, void* stream)
{
    LAUNCH_ROUTING_DEEPSEEK(data,
        /*coopLaunch=*/false, routingMainKernel, numBlocks, numThreadsMain,
        /*smemSize=*/0, // No dynamic smem
        stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 2. Cluster kernel — single-cluster fused kernel (SM90+).
//
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
    if (params.mUsePdl)
    {
        cudaGridDependencySynchronize();
    }
    routingPermutation<KernelParams, OutputT, KernelParams::MaxNumExperts, KernelParams::MaxNumExperts / WarpSize,
        KernelParams::MaxNumTopExperts, /*LoadExpertIdxFromGlobal=*/true>(params, nullptr, warpIdx, clusterBlockRank);
}
#else
__global__ void routingIndicesClusterKernel(KernelParams params)
{
    assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif

static void launchClusterKernel(Data& data, int numThreadsHist, void* stream)
{
    LAUNCH_ROUTING_DEEPSEEK(data,
        /*coopLaunch=*/false, routingIndicesClusterKernel, NumBlocksPerCluster, numThreadsHist,
        /*smemSize=*/0, // No dynamic smem
        stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 3-6. Launch wrappers for shared kernels.
//      Coop delegates to routingCustom; others use LAUNCH_ROUTING_DEEPSEEK macro.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

static void launchCoopKernel(Data& data, int numBlocksCoop, int /*numThreadsHist*/, void* stream)
{
    // Use routingCustom's coop kernel implementation (they are identical).
    // Convert DeepSeek Data to Custom Data for launching.
    routingCustom::Data customData;
    // Copy base fields
    static_cast<DataBase&>(customData) = static_cast<DataBase const&>(data);
    // Set routingCustom-specific defaults (not needed for coop kernel)
    customData.mDtypeOutput = data.mDtypeOutput;
    // The coop kernel doesn't read routing logits (mPtrInput), only mPtrTopKPacked.
    // Set mDtypeInput = mDtypeOutput so the dispatched template is <OutputT, OutputT>,
    // avoiding an unnecessary mixed-type instantiation.
    customData.mDtypeInput = data.mDtypeOutput;
    customData.mPreprocessType = RoutingPreprocessType::None;
    customData.mPostprocessType = RoutingPostprocessType::Softmax;

    // Recompute numThreadsHist using routingCustom's expert tiers (128, 512, 2048),
    // since the custom coop kernel dispatch selects template parameters based on these tiers.
    // DeepSeek's getMaxNumExperts uses different tiers (256, 384, 512) which would mismatch.
    uint32_t const customNumThreadsHist
        = std::min(1024u, static_cast<uint32_t>(routingCustom::getMaxNumExperts(data.mNumExperts)));
    routingCustom::launchCoopKernel(customData, numBlocksCoop, customNumThreadsHist, stream);
}

static void launchHistogramKernel(Data& data, int numBlocksHistogram, int numThreadsHist, void* stream)
{
    LAUNCH_ROUTING_DEEPSEEK(data,
        /*coopLaunch=*/false, routingIndicesHistogramKernel, numBlocksHistogram, numThreadsHist,
        /*smemSize=*/0, // No dynamic smem
        stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);
}

static void launchOffsetsKernel(Data& data, int numBlocksOffsets, int numThreadsHist, void* stream)
{
    LAUNCH_ROUTING_DEEPSEEK(data,
        /*coopLaunch=*/false, routingIndicesOffsetsKernel, numBlocksOffsets, numThreadsHist,
        /*smemSize=*/0, // No dynamic smem
        stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Entry point
//
////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data& data, void* stream)
{
    TLLM_CHECK_WITH_INFO(data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr || data.mPtrTopKIds != nullptr,
        "Routing kernel requires at least one input parameter");

    // When topK is already computed (mPtrTopKIds or mPtrTopKPacked without scores),
    // delegate to the shared post-topK pipeline which handles all path selection
    // (single-block, single-cluster, coop, multi-kernel) automatically.
    // No routing-method-specific logic needed.
    if (data.mPtrTopKIds != nullptr || (data.mPtrTopKPacked != nullptr && data.mPtrScores == nullptr))
    {
        if (data.mPtrTopKIds != nullptr)
        {
            TLLM_CHECK_WITH_INFO(data.mPtrTopKWeights != nullptr,
                "When mPtrTopKIds is provided, mPtrTopKWeights must also be provided for DeepSeek routing.");
        }
        int const numThreadsHist = getMaxNumExperts(data.mNumExperts);
        runPostTopKPipeline(data, numThreadsHist, stream);
        return;
    }

    // After this point, input is mPtrScores (raw logits that need DeepSeek-specific routing).
    TLLM_CHECK_WITH_INFO(!data.mUseRoutingSoftmax, "Routing with softmax not implemented yet");
    TLLM_CHECK_WITH_INFO(data.mNumExperts >= data.mTopK, "Routing kernel expects topK (%d) to be <= numExperts (%d)",
        data.mTopK, data.mNumExperts);
    TLLM_CHECK_WITH_INFO(data.mNumExperts <= MaxSupportedExpertCount,
        "Routing kernel expects #experts %d  <= #threads %d", data.mNumExperts, MaxSupportedExpertCount);
    TLLM_CHECK_WITH_INFO(data.mTopK <= MaxSupportedTopExperts, "Routing kernel expects topK experts <= %d, got %d",
        MaxSupportedTopExperts, data.mTopK);

    if (data.mPtrExpandedIdxToPermutedIdx != nullptr || data.mPtrPermutedIdxToExpandedIdx != nullptr
        || data.mPtrPermutedIdxToTokenIdx != nullptr)
        TLLM_CHECK_WITH_INFO(data.mPtrTopKPacked != nullptr && data.mPtrPermutedIdxSize,
            "If permuted index is required, `mPtrTopKPacked` is also required");

    // Routing needs to be executed - validate routing kernel constraints
    if (data.mNumExpertGroups > 1)
    {
        TLLM_CHECK_WITH_INFO(data.mNumExpertGroups <= MaxNumGroups,
            "Routing kernel expects #expert groups %d to be <= max groups %d", data.mNumExpertGroups, MaxNumGroups);
        TLLM_CHECK_WITH_INFO(data.mNumExperts % data.mNumExpertGroups == 0,
            "Routing kernel expects #experts %d to be a multiple of #expert groups %d", data.mNumExperts,
            data.mNumExpertGroups);
        TLLM_CHECK_WITH_INFO(data.mNumExperts / data.mNumExpertGroups <= WarpSize,
            "Routing kernel expects #experts per group <= warp size (%d), got %d experts / %d groups = %d experts "
            "per group",
            WarpSize, data.mNumExperts, data.mNumExpertGroups, data.mNumExperts / data.mNumExpertGroups);
        TLLM_CHECK_WITH_INFO(data.mNumLimitedGroups <= MaxNumTopGroups,
            "Routing kernel expects <= %d top groups, got %d", MaxNumTopGroups, data.mNumLimitedGroups);

        TLLM_CHECK_WITH_INFO(data.mNumExpertGroups >= data.mNumLimitedGroups,
            "Routing kernel expects top groups %d to be limited by #expert groups %d", data.mNumLimitedGroups,
            data.mNumExpertGroups);
        TLLM_CHECK_WITH_INFO(
            data.mNumExperts % 4 == 0, "Routing kernel expects #experts %d to be a multiple of 4.", data.mNumExperts);
    }

    int const numBlocks = data.mNumTokens;
    int const numThreadsHist = getMaxNumExperts(data.mNumExperts);
    static int const smMajor = tensorrt_llm::common::getSMVersion() / 10;
    // Step 1: Run DeepSeek-specific topK computation (writes to mPtrTopKPacked)
    int const numThreadsMain = max(data.mNumExpertGroups * WarpSize, getMaxNumExperts(data.mNumExperts));
    launchMainKernel(data, numBlocks, numThreadsMain, stream);

    // Step 2: Permutation pipeline (reads from mPtrTopKPacked written by step 1)
    if (data.mPtrPermutedIdxSize != nullptr)
    {
        TLLM_CHECK_WITH_INFO(data.mPtrCtaIdxXyToBatchIdx != nullptr && data.mPtrCtaIdxXyToMnLimit != nullptr
                && data.mPtrNumNonExitingCtas != nullptr,
            "DeepSeek routing step 2 requires grouped-GEMM launch config buffers "
            "(mPtrCtaIdxXyToBatchIdx, mPtrCtaIdxXyToMnLimit, mPtrNumNonExitingCtas)");

        bool const useSingleCluster = (smMajor >= 9) && (data.mNumTokens <= 1024);
        if (!useSingleCluster)
        {
            TLLM_CHECK_WITH_INFO(
                data.mPtrExpertCounts != nullptr, "When #tokens is large, `mPtrExpertCounts` is a required input.");
        }
        else
        {
            data.mPtrExpertCounts = nullptr; // Set it to nullptr for single-cluster code path, as it won't be used
        }

        // Number of blocks we can use in the cooperative kernel
        static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
        // WAR: Reserve 8 SMs for overlapping kernels.
        int const numBlocksCoop = smCount - 8;
        // Maximum number of tokens supported by the kernel using a cooperative launch.
        int const maxTokensCoop = (numBlocksCoop * numThreadsHist * 64) / data.mTopK;

        // Last routing kernel: disable overlap so GEMM waits via stream serialization.
        bool const savedAllowOverlap = data.mPdlAllowOverlap;

        if (useSingleCluster)
        {
            data.mPdlAllowOverlap = false;
            launchClusterKernel(data, numThreadsHist, stream);
        }
        else if ((smMajor >= 9) && (data.mNumTokens <= maxTokensCoop))
        {
            data.mPdlAllowOverlap = false;
            launchCoopKernel(data, numBlocksCoop, numThreadsHist, stream);
        }
        else
        {
            const int32_t expandedIdxSize = data.mNumTokens * data.mTopK;
            const int32_t histogramEltsPerBlock = 8 * numThreadsHist;
            const int32_t offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * numThreadsHist;
            const int32_t maxNumBlocks = 1024;

            int const numBlocksHistogram
                = std::min((expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
            int const numBlocksOffsets
                = std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

            launchHistogramKernel(data, numBlocksHistogram, numThreadsHist, stream);
            data.mPdlAllowOverlap = false;
            launchOffsetsKernel(data, numBlocksOffsets, numThreadsHist, stream);
        }

        data.mPdlAllowOverlap = savedAllowOverlap;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#undef LAUNCH_DEEPSEEK_WITH_TOPK
#undef LAUNCH_ROUTING_DEEPSEEK

} // namespace routingDeepSeek
} // namespace moe::dev::routing
