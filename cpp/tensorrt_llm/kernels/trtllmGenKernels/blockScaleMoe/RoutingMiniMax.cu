/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 * Contributed by Baseten.co
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
namespace routingMiniMax
{

////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumExpertsLimit = 256;
static constexpr int MaxSupportedTopExperts = 8;

////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void routingMainKernel(KernelParams params)
{
    using OutputT = typename KernelParams::OutputT;

    static constexpr int MaxNumExperts = KernelParams::MaxNumExperts;
    static_assert(MaxNumExperts <= NumExpertsLimit, "MiniMax supports up to 256 experts.");
    // MiniMax is configured for topK=8 (enforced at runtime in run())

    // One token per block
    int32_t const tokenIdx = blockIdx.x;
    int32_t const expertIdx = threadIdx.x;

    // Cooperative groups
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    int32_t const laneIdx = cutlass::arch::LaneId();
    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);

    // Shared memory for per-expert probabilities (DeepSeek style)
    __shared__ float __attribute((aligned(128))) smemProb[MaxNumExperts];
    __shared__ float __attribute((aligned(128))) smemSel[MaxNumExperts];

    // Invalid handling
    static constexpr float kNegInf = float{-INFINITY};

    // Load and compute per-expert scores for this token
    bool const validExpert = expertIdx < params.mNumExperts;
    float prob = 0.f;
    float sel = kNegInf;

    if (validExpert)
    {
        // logit layout: token-major, contiguous experts
        int64_t scoreIndex = int64_t{tokenIdx} * int64_t{params.mNumExperts} + int64_t{expertIdx};
        float logit = static_cast<float>(params.mPtrScores[scoreIndex]);

        // MiniMax: sigmoid (not softmax)
        prob = sigmoid_accurate(logit);

        float bias = static_cast<float>(params.mPtrRoutingBias[expertIdx]);
        sel = prob + bias; // selection score
    }

    // Stage to shared so warp0 can index by expert id after topK
    if (expertIdx < MaxNumExperts)
    {
        smemProb[expertIdx] = validExpert ? prob : 0.f;   // invalid contributes 0 to renorm
        smemSel[expertIdx] = validExpert ? sel : kNegInf; // invalid never selected
    }

    __syncthreads();

    // Only warp0 does the final expert selection (DeepSeek style)
    if (warpIdx == 0)
    {
        // Each lane owns VecSize experts: expert = ii*WarpSize + laneIdx
        static constexpr int VecSize = MaxNumExperts / WarpSize; // 256 -> 8
        static_assert(MaxNumExperts % WarpSize == 0, "MaxNumExperts must be multiple of 32.");

        float laneVals[VecSize];
        int32_t laneIdxs[VecSize];

#pragma unroll
        for (int ii = 0; ii < VecSize; ++ii)
        {
            int e = ii * WarpSize + laneIdx;
            laneIdxs[ii] = e;
            laneVals[ii] = (e < params.mNumExperts) ? smemSel[e] : kNegInf;
        }

        // TopK outputs
        float topScores[MaxSupportedTopExperts];
        int32_t topExperts[MaxSupportedTopExperts];

        // Reduce on selection scores (prob+bias)
        topk::reduceTopK(warp, topScores, topExperts, laneVals, laneIdxs, kNegInf, params.mTopK);

        // Convert selection into final weights:
        // final = prob (unbiased), optionally renormalized over topK
        float w = 0.f;
        int32_t chosenExpert = 0;

#pragma unroll
        for (int ii = 0; ii < MaxSupportedTopExperts; ++ii)
        {
            if (laneIdx == ii)
            {
                chosenExpert = topExperts[ii];
                w = (ii < params.mTopK && chosenExpert >= 0 && chosenExpert < params.mNumExperts)
                    ? smemProb[chosenExpert]
                    : 0.f;
            }
        }

        // Renormalize within topK if requested
        float denom = 1.f;
        if (params.mNormTopkProb)
        {
            float x = (laneIdx < params.mTopK) ? w : 0.f;
            denom = cg::reduce(warp, x, cg::plus<float>{});
            denom += 1e-20f;
        }

        float finalW = (laneIdx < params.mTopK) ? (w / denom) : 0.f;

        // Write outputs
        int32_t out = tokenIdx * params.mTopK + laneIdx;

        if (laneIdx < params.mTopK && params.mPtrTopKPacked != nullptr)
        {
            PackedScoreIdx<OutputT> packed{static_cast<OutputT>(finalW), static_cast<int16_t>(chosenExpert)};
            params.mPtrTopKPacked[out] = packed;
        }
        if (laneIdx < params.mTopK && params.mPtrTopKWeights != nullptr)
        {
            params.mPtrTopKWeights[out] = static_cast<OutputT>(finalW);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////

// Cluster kernel removed for simplification - use histogram path for all token counts

////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts) routingIndicesHistogramScoresKernel(KernelParams params)
{
    using OutputT = typename KernelParams::OutputT;

    int32_t const laneIdx = cutlass::arch::LaneId();
    int32_t const warpIdx = threadIdx.x / WarpSize;
    int32_t const globalWarpIdx = blockIdx.x * (KernelParams::MaxNumExperts / WarpSize) + warpIdx;
    int32_t const globalWarpStride = gridDim.x * (KernelParams::MaxNumExperts / WarpSize);

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

    static constexpr int MaxNumExperts = KernelParams::MaxNumExperts;
    static constexpr int VecSize = MaxNumExperts / WarpSize; // 256->8

    for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens; tokenIdx += globalWarpStride)
    {
        // per-lane candidates
        float laneVals[VecSize];
        int32_t laneIdxs[VecSize];

// each lane covers VecSize experts
#pragma unroll
        for (int ii = 0; ii < VecSize; ++ii)
        {
            int e = ii * WarpSize + laneIdx;
            laneIdxs[ii] = e;

            if (e < params.mNumExperts)
            {
                int64_t scoreIndex = int64_t{tokenIdx} * int64_t{params.mNumExperts} + int64_t{e};
                float logit = static_cast<float>(params.mPtrScores[scoreIndex]);
                float prob = sigmoid_accurate(logit);
                float bias = static_cast<float>(params.mPtrRoutingBias[e]);
                laneVals[ii] = prob + bias; // selection
            }
            else
            {
                laneVals[ii] = float{-INFINITY};
            }
        }

        float topSel[MaxSupportedTopExperts];
        int32_t topExp[MaxSupportedTopExperts];
        topk::reduceTopK(warp, topSel, topExp, laneVals, laneIdxs, float{-INFINITY}, params.mTopK);

        // produce packed output weights from *unbiased prob* (sigmoid only)
        if (laneIdx < params.mTopK)
        {
            int e = topExp[laneIdx];

            float prob = 0.f;
            if (e >= 0 && e < params.mNumExperts)
            {
                int64_t scoreIndex = int64_t{tokenIdx} * int64_t{params.mNumExperts} + int64_t{e};
                float logit = static_cast<float>(params.mPtrScores[scoreIndex]);
                prob = sigmoid_accurate(logit);
            }

            // renorm if requested
            float denom = 1.f;
            if (params.mNormTopkProb)
            {
                float x = (laneIdx < params.mTopK) ? prob : 0.f;
                denom = cg::reduce(warp, x, cg::plus<float>{}) + 1e-20f;
            }
            float finalW = (laneIdx < params.mTopK) ? (prob / denom) : 0.f;

            PackedScoreIdx<OutputT> packed{static_cast<OutputT>(finalW), static_cast<int16_t>(e)};
            if (params.mPtrTopKPacked != nullptr && laneIdx < params.mTopK)
            {
                params.mPtrTopKPacked[tokenIdx * params.mTopK + laneIdx] = packed;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////

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

// MiniMax-specific dispatch: InputT is always float (gate is float32).
// OutputT varies based on mDtypeExpW (bf16 for bias/weights, or float).
// DoSoftmaxBeforeTopK is always false (MiniMax uses sigmoid, not softmax).
#define LAUNCH_ROUTING_MINIMAX_IMPL(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, numExperts)     \
    if (data.mDtypeExpW == tg::Dtype::Fp32)                                                                            \
    {                                                                                                                  \
        LAUNCH_TILEN(data, coopLaunch, LAUNCH_ESC(float, float, numExperts, false), kernel, numBlocks, numThreads,     \
            smemSize, stream);                                                                                         \
    }                                                                                                                  \
    else if (data.mDtypeExpW == tg::Dtype::Bfloat16)                                                                   \
    {                                                                                                                  \
        LAUNCH_TILEN(data, coopLaunch, LAUNCH_ESC(float, __nv_bfloat16, numExperts, false), kernel, numBlocks,         \
            numThreads, smemSize, stream);                                                                             \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_LOG_ERROR("Unsupported dtypeExpW");                                                                       \
    }

#define LAUNCH_ROUTING_MINIMAX(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream)                      \
    if (data.mNumExperts <= topk::MaxNumExpertsUnit)                                                                   \
    {                                                                                                                  \
        LAUNCH_ROUTING_MINIMAX_IMPL(                                                                                   \
            data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, topk::MaxNumExpertsUnit);               \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumExpertsLimit)                                                                      \
    {                                                                                                                  \
        LAUNCH_ROUTING_MINIMAX_IMPL(                                                                                   \
            data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, NumExpertsLimit);                       \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_LOG_ERROR("Unsupported numExperts");                                                                      \
    }

void run(Data const& data, void* stream)
{
    // Create non-const alias for launch macros that may require non-const data
    auto& d = const_cast<Data&>(data);

    TLLM_CHECK_WITH_INFO(d.mPtrTopKPacked != nullptr || d.mPtrScores != nullptr || d.mPtrTopKIds != nullptr,
        "Routing kernel requires at least one input parameter");

    if (d.mPtrTopKIds != nullptr)
    {
        TLLM_CHECK_WITH_INFO(d.mPtrTopKWeights != nullptr,
            "When mPtrTopKIds is provided, mPtrTopKWeights must also be provided for MiniMax routing.");
    }

    // Permutation outputs required by grouped GEMM launch
    TLLM_CHECK_WITH_INFO(d.mPtrPermutedIdxSize != nullptr && d.mPtrCtaIdxXyToBatchIdx != nullptr
            && d.mPtrCtaIdxXyToMnLimit != nullptr && d.mPtrNumNonExitingCtas != nullptr,
        "MiniMax routing expects permuted idx and grouped GEMM launch config buffers");

    TLLM_CHECK_WITH_INFO(d.mTopK == 8, "MiniMax is configured for topK=8, got %d", d.mTopK);
    TLLM_CHECK_WITH_INFO(d.mNumExperts == 256, "MiniMax is configured for 256 experts, got %d", d.mNumExperts);

    TLLM_CHECK_WITH_INFO(d.mNumExperts % 4 == 0, "Routing expects #experts multiple of 4, got %d", d.mNumExperts);

    // This "DeepSeek-like" path uses a block-per-token routingMainKernel when we need routing
    int const numThreadsHist = getMaxNumExperts(d.mNumExperts);

    // Simplified: always use histogram path for permutation building
    TLLM_CHECK_WITH_INFO((d.mPtrTopKPacked != nullptr || d.mPtrTopKIds != nullptr),
        "MiniMax requires `mPtrTopKPacked` or `mPtrTopKIds` for permutation building.");
    TLLM_CHECK_WITH_INFO(
        d.mPtrExpertCounts != nullptr, "MiniMax requires `mPtrExpertCounts` for permutation building.");

    // 1) If TopK not provided, compute it - choose path based on token count
    if (d.mPtrTopKIds == nullptr)
    {
        TLLM_CHECK_WITH_INFO(d.mPtrScores != nullptr, "If mPtrTopKIds is null, mPtrScores must be provided.");
        TLLM_CHECK_WITH_INFO(
            d.mPtrTopKPacked != nullptr, "MiniMax requires mPtrTopKPacked when computing topK from scores.");

        // Choose computation path based on token count to avoid double work
        constexpr int SmallTokenThreshold = 256; // Same as MaxNumTokensSingleClusterScores
        bool const useSmallTokenPath = d.mNumTokens <= SmallTokenThreshold;

        if (useSmallTokenPath)
        {
            // Small token count: use block-per-token routingMainKernel
            LAUNCH_ROUTING_MINIMAX(d,
                /*coopLaunch=*/false, routingMainKernel,
                /*numBlocks=*/d.mNumTokens,
                /*numThreads=*/numThreadsHist,
                /*smemSize=*/0, stream);
        }
        // else: large token count - will use routingIndicesHistogramScoresKernel below
    }

    // 2) Build permutation / CTA schedule (always use histogram path)
    if (d.mPtrPermutedIdxSize != nullptr)
    {
        // Histogram + offsets path
        int32_t const expandedIdxSize = d.mNumTokens * d.mTopK;
        int32_t const histogramEltsPerBlock = 8 * numThreadsHist;
        int32_t const offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * numThreadsHist;
        int32_t const maxNumBlocks = 1024;

        int const numBlocksHistogram
            = std::min((expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
        int const numBlocksOffsets
            = std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

        // Always initialize expert counts first (avoid race conditions)
        LAUNCH_ROUTING_MINIMAX(d,
            /*coopLaunch=*/false, routingInitExpertCounts,
            /*numBlocks=*/(2 * d.mNumExperts - 1) / numThreadsHist + 1,
            /*numThreads=*/numThreadsHist,
            /*smemSize=*/0, stream);

        // Only compute topK from scores if we didn't already do it in routingMainKernel
        constexpr int SmallTokenThreshold = 256;
        bool const usedSmallTokenPath = d.mNumTokens <= SmallTokenThreshold;

        if (d.mPtrScores != nullptr && d.mPtrTopKIds == nullptr && !usedSmallTokenPath)
        {
            // produce mPtrTopKPacked from scores (sigmoid+bias selection) for large token counts
            LAUNCH_ROUTING_MINIMAX(d,
                /*coopLaunch=*/false, routingIndicesHistogramScoresKernel,
                /*numBlocks=*/maxNumBlocks,
                /*numThreads=*/numThreadsHist,
                /*smemSize=*/0, stream);
        }

        LAUNCH_ROUTING_MINIMAX(d,
            /*coopLaunch=*/false, routingIndicesHistogramKernel,
            /*numBlocks=*/numBlocksHistogram,
            /*numThreads=*/numThreadsHist,
            /*smemSize=*/0, stream);

        LAUNCH_ROUTING_MINIMAX(d,
            /*coopLaunch=*/false, routingIndicesOffsetsKernel,
            /*numBlocks=*/numBlocksOffsets,
            /*numThreads=*/numThreadsHist,
            /*smemSize=*/0, stream);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingMiniMax
} // namespace moe::dev::routing