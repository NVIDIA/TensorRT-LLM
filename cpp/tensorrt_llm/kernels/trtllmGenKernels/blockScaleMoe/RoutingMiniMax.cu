/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 * Contibuted by Baseten.co
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
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

//////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumExpertsLimit = 256;

static constexpr int NumThreads = 1024;
static constexpr int NumWarps = NumThreads / WarpSize;
static constexpr int MaxSupportedTopExperts = 10;

static constexpr int MaxNumTokensSingleCluster = NumBlocksPerCluster * NumThreads;
static constexpr int MaxNumTokensSingleClusterScores = NumBlocksPerCluster * NumWarps;

static constexpr int BlockKernelMaxNumTokens = 8;

template <typename DataType, typename InputType, typename BiasType, int VecSize>
__forceinline__ __device__ void routingTopKExperts(cg::thread_block_tile<WarpSize> const& warp,
    DataType (&score)[VecSize], DataType (&scoreWithBias)[VecSize], int32_t (&idx)[VecSize],
    DataType (&warpTopKScore)[MaxSupportedTopExperts], int32_t (&warpTopKExpertIdx)[MaxSupportedTopExperts],
    int32_t const laneIdx, int32_t const numExperts, int32_t topK, InputType const* ptrScores,
    BiasType const* ptrRoutingBias, bool const normTopkProb)
{
    DataType minScore = DataType{-INFINITY};
    
    // Store float scores for accurate shuffle and selection
    float score_f[VecSize];
    float scoreWithBias_f[VecSize];

    // Step 1: Apply sigmoid to logits (NOT softmax - MiniMax2 uses sigmoid)
    for (int i = 0; i < VecSize; i++)
    {
        auto expertIdx = i * WarpSize + laneIdx;
        if (expertIdx < numExperts)
        {
            // Apply sigmoid to get probability (store in float for accuracy)
            float logit = static_cast<float>(ptrScores[expertIdx]);
            score_f[i] = sigmoid_accurate(logit);
            score[i] = static_cast<DataType>(score_f[i]);
            scoreWithBias_f[i] = score_f[i];  // Will add bias below
            scoreWithBias[i] = score[i];  // Keep DataType version for compatibility
        }
        else
        {
            score_f[i] = 0.0f;
            score[i] = DataType{0};  // Invalid experts have 0 probability
            scoreWithBias_f[i] = -INFINITY;
            scoreWithBias[i] = minScore;  // Invalid experts have -inf for selection
        }
        idx[i] = expertIdx;
    }

    // Step 2: Add routing bias for selection (only for valid experts)
    for (int i = 0; i < VecSize; i++)
    {
        auto expertIdx = i * WarpSize + laneIdx;
        if (expertIdx < numExperts)
        {
            float bias = static_cast<float>(ptrRoutingBias[expertIdx]);
            scoreWithBias_f[i] = score_f[i] + bias;
            scoreWithBias[i] = static_cast<DataType>(scoreWithBias_f[i]);
        }
    }

    // Step 3: Top-K selection using bias-added scores (use float for stability)
    topk::reduceTopK(warp, warpTopKScore, warpTopKExpertIdx, scoreWithBias, idx, minScore, topK);

    // Step 4: Gather original sigmoid scores (not bias-added) for selected experts
    // Use warp shuffle to get the unbiased score from the lane that has it
    if (laneIdx < topK)
    {
        int selectedExpertIdx = warpTopKExpertIdx[laneIdx];
        int targetLane = selectedExpertIdx % WarpSize;
        int vecIdx = selectedExpertIdx / WarpSize;
        
        // Select the per-lane value via unrolled loop, then shuffle
        float v = 0.0f;
        #pragma unroll
        for (int i = 0; i < VecSize; ++i)
        {
            if (vecIdx == i) v = score_f[i];
        }
        float originalScore_f = __shfl_sync(0xffffffff, v, targetLane);
        warpTopKScore[laneIdx] = static_cast<DataType>(originalScore_f);
    }

    // Step 5: Renormalize using original sigmoid scores (with epsilon to prevent divide-by-zero)
    float sum = 1.0f;
    if (normTopkProb)
    {
        sum = (laneIdx < topK) ? static_cast<float>(warpTopKScore[laneIdx]) : 0.0f;
        sum = cg::reduce(warp, sum, cg::plus<float>()) + 1e-20f;  // Add epsilon
    }
    if (laneIdx < topK)
    {
        float normalized = static_cast<float>(warpTopKScore[laneIdx]) / sum;
        warpTopKScore[laneIdx] = static_cast<DataType>(normalized);
    }
}

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts) routingIndicesBlockKernel(KernelParams params)
{
    // Enforce kernel constraints
    if (params.mNumTokens > BlockKernelMaxNumTokens) return;
    if (params.mTopK > MaxSupportedTopExperts) return;
    
    // Guard against thread count mismatch
    if (blockDim.x != KernelParams::MaxNumExperts) return;
    
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;
    using BaseType = InputT;
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
                auto expertIdx = params.mPtrTopKIds[warpIdx * params.mTopK + laneIdx];
                if (expertIdx != -1)
                {
                    int offset = warpIdx * MaxNumExperts + expertIdx;
                    smemKIdx[offset] = static_cast<int8_t>(laneIdx);
                }
                else
                {
                    params.mPtrExpandedIdxToPermutedIdx[warpIdx * params.mTopK + laneIdx] = int32_t{-1};
                }
            }
        }
    }
    else if (params.mPtrScores != nullptr)
    {
        BaseType score[VecSize];
        BaseType scoreWithBias[VecSize];
        int32_t idx[VecSize];

        BaseType warpTopKScore[MaxSupportedTopExperts];
        int32_t warpTopKExpertIdx[MaxSupportedTopExperts];

        BaseType minScore = BaseType{-INFINITY};
        if (validToken)
        {
            routingTopKExperts<BaseType, InputT, BaseType, VecSize>(warp, score, scoreWithBias, idx, warpTopKScore,
                warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK, params.mPtrScores + scoreOffset,
                params.mPtrRoutingBias, params.mNormTopkProb);

            if (laneIdx < params.mTopK)
            {
                int offset = warpIdx * MaxNumExperts + warpTopKExpertIdx[laneIdx];
                smemKIdx[offset] = static_cast<int8_t>(laneIdx);
                if (params.mPtrTopKWeights != nullptr)
                {
                    params.mPtrTopKWeights[warpIdx * params.mTopK + laneIdx] = OutputT{warpTopKScore[laneIdx]};
                }
            }
        }
    }
    __syncthreads();

    auto localExpertIdx = expert - params.mLocalExpertsStartIdx;
    int stride = 1 << params.mLocalExpertsStrideLog2;
    auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < (params.mNumLocalExperts * stride)
        && (localExpertIdx % stride) == 0;
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
    __syncthreads();

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

            if (params.mPtrExpandedIdxToPermutedIdx != nullptr)
            {
                params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
            }
            if (params.mPtrPermutedIdxToExpandedIdx != nullptr && isLocalExpert)
            {
                params.mPtrPermutedIdxToExpandedIdx[permutedIdx] = expandedIdx;
            }
            if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocalExpert)
            {
                params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
            }
        }
    }
}

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) 
    __launch_bounds__(KernelParams::MaxNumExperts)
    routingIndicesClusterKernel(KernelParams params)
#else
__global__ void __launch_bounds__(KernelParams::MaxNumExperts) 
    routingIndicesClusterKernel(KernelParams params)
#endif
{
    // Enforce kernel constraints
    if (params.mTopK > MaxSupportedTopExperts) return;
    
    // Guard against thread count mismatch
    if (blockDim.x != KernelParams::MaxNumExperts) return;
    
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;

    using BaseType = InputT;
    using TypePacked = PackedScoreIdx<BaseType>;

    static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;
    // routingPermutation expects NumThreads == MaxNumExperts
    static constexpr int PermThreads = KernelParams::MaxNumExperts;
    static constexpr int PermWarps = PermThreads / WarpSize;

    // Shared memory for topK results - sized for MaxNumExperts threads
    __shared__ TypePacked __attribute((aligned(128))) smemPackedScoreIdx[PermWarps * MaxSupportedTopExperts];

    uint32_t const clusterBlockRank = blockIdx.x;

    int32_t const warpIdx = threadIdx.x / WarpSize;
    int32_t const laneIdx = cutlass::arch::LaneId();

    auto warpTokenIdx = clusterBlockRank * PermWarps + warpIdx;
    auto scoreOffset = warpTokenIdx * params.mNumExperts;
    bool validToken = warpTokenIdx < params.mNumTokens;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif

    if (params.mPtrScores != nullptr)
    {
        BaseType score[VecSize];
        BaseType scoreWithBias[VecSize];
        int32_t idx[VecSize];

        BaseType warpTopKScore[MaxSupportedTopExperts];
        int32_t warpTopKExpertIdx[MaxSupportedTopExperts];

        BaseType minScore = BaseType{-INFINITY};
        if (validToken)
        {
            routingTopKExperts<BaseType, InputT, BaseType, VecSize>(warp, score, scoreWithBias, idx, warpTopKScore,
                warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK, params.mPtrScores + scoreOffset,
                params.mPtrRoutingBias, params.mNormTopkProb);

            if (laneIdx < params.mTopK)
            {
                // Use fixed stride to avoid indexing mistakes
                constexpr int STRIDE = MaxSupportedTopExperts;
                smemPackedScoreIdx[warpIdx * STRIDE + laneIdx]
                    = TypePacked{warpTopKScore[laneIdx], static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
            }
        }
    }

    // Pack topK results into shared memory for all input types
    if (params.mPtrScores != nullptr)
    {
        // Already packed by routingTopKExperts above
    }
    else if (params.mPtrTopKIds != nullptr && params.mPtrTopKWeights != nullptr)
    {
        // Pack from pre-computed TopKIds and TopKWeights
        if (validToken && laneIdx < params.mTopK)
        {
            int id = params.mPtrTopKIds[warpTokenIdx * params.mTopK + laneIdx];
            BaseType w = static_cast<BaseType>(params.mPtrTopKWeights[warpTokenIdx * params.mTopK + laneIdx]);
            // Handle invalid expert IDs explicitly
            if (id == -1) {
                smemPackedScoreIdx[warpIdx * MaxSupportedTopExperts + laneIdx] = 
                    TypePacked{BaseType{0}, int16_t{-1}};
            } else {
                smemPackedScoreIdx[warpIdx * MaxSupportedTopExperts + laneIdx] = 
                    TypePacked{w, static_cast<int16_t>(id)};
            }
        }
    }
    else if (params.mPtrTopKPacked != nullptr)
    {
        // Pack from pre-computed packed format with explicit type casting
        if (validToken && laneIdx < params.mTopK)
        {
            auto p = params.mPtrTopKPacked[warpTokenIdx * params.mTopK + laneIdx];
            BaseType w = static_cast<BaseType>(p.score);
            int16_t id = p.idx;
            smemPackedScoreIdx[warpIdx * MaxSupportedTopExperts + laneIdx] = TypePacked{w, id};
        }
    }
    
    // Initialize unused lanes to prevent stale data
    if (validToken && laneIdx >= params.mTopK && laneIdx < MaxSupportedTopExperts)
    {
        smemPackedScoreIdx[warpIdx * MaxSupportedTopExperts + laneIdx] = 
            TypePacked{BaseType{0}, int16_t{-1}};
    }
    
    // Synchronize before using shared memory
    __syncthreads();
    
    // Use fixed stride for shared memory layout
    // LoadExpertIdxFromGlobal=false: load from shared memory (smemPackedScoreIdx)
    constexpr int STRIDE = MaxSupportedTopExperts;
    routingPermutation<KernelParams, BaseType, PermThreads, PermWarps, STRIDE,
        /*LoadExpertIdxFromGlobal=*/false>(params, smemPackedScoreIdx, warpIdx, clusterBlockRank);
}

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts) routingIndicesHistogramScoresKernel(KernelParams params)
{
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;
    using BaseType = InputT;

    static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;

    int32_t const laneIdx = cutlass::arch::LaneId();
    int32_t const warpIdx = threadIdx.x / WarpSize;
    int32_t const globalWarpIdx = blockIdx.x * KernelParams::MaxNumExperts / WarpSize + warpIdx;
    int32_t const globalWarpStride = gridDim.x * KernelParams::MaxNumExperts / WarpSize;
    BaseType minScore = BaseType{-INFINITY};
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif

    int32_t expertCountsNum = 2 * params.mNumExperts;
    int32_t globalThreadIdx = blockIdx.x * KernelParams::MaxNumExperts + threadIdx.x;
    int32_t globalThreadStride = gridDim.x * KernelParams::MaxNumExperts;
    initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif

    BaseType allScores[VecSize];
    BaseType allScoresWithBias[VecSize];
    int32_t allExpertIdx[VecSize];
    BaseType warpTopKScore[MaxSupportedTopExperts];
    int32_t warpTopKExpertIdx[MaxSupportedTopExperts];
    for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens; tokenIdx += globalWarpStride)
    {
        auto scoreOffset = tokenIdx * params.mNumExperts;

        routingTopKExperts<BaseType, InputT, BaseType, VecSize>(warp, allScores, allScoresWithBias, allExpertIdx,
            warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
            params.mPtrScores + scoreOffset, params.mPtrRoutingBias, params.mNormTopkProb);

        if (laneIdx < params.mTopK)
        {
            PackedScoreIdx<OutputT> packedScore{
                static_cast<OutputT>(warpTopKScore[laneIdx]), static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
            params.mPtrTopKPacked[tokenIdx * params.mTopK + laneIdx] = packedScore;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

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

//////////////////////////////////////////////////////////////////////////////////////////////////

#define LAUNCH_ROUTING_MINIMAX(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream)      \
    if (data.mNumExperts <= topk::MaxNumExpertsUnit)                                                   \
    {                                                                                                  \
        LAUNCH_ROUTING_WITH_NUM_EXPERTS(                                                               \
            data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, false, topk::MaxNumExpertsUnit);   \
    }                                                                                                  \
    else if (data.mNumExperts <= NumExpertsLimit)                                                     \
    {                                                                                                  \
        LAUNCH_ROUTING_WITH_NUM_EXPERTS(                                                               \
            data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, false, NumExpertsLimit);           \
    }                                                                                                  \
    else                                                                                               \
    {                                                                                                  \
        TLLM_LOG_ERROR("Unsupported numExperts");                                                      \
    }

//////////////////////////////////////////////////////////////////////////////////////////////////
void run(Data const& data, void* stream)
{
    TLLM_CHECK_WITH_INFO(data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr || data.mPtrTopKIds != nullptr,
        "Routing kernel requires at least one input parameter");
    if (data.mPtrTopKIds != nullptr)
    {
        TLLM_CHECK_WITH_INFO(data.mPtrTopKWeights != nullptr,
            "When mPtrTopKIds is provided, mPtrTopKWeights must also be provided for MiniMax routing.");
    }
    TLLM_CHECK_WITH_INFO(data.mPtrPermutedIdxSize != nullptr && data.mPtrCtaIdxXyToBatchIdx != nullptr
            && data.mPtrCtaIdxXyToMnLimit != nullptr && data.mPtrNumNonExitingCtas != nullptr,
        "MiniMax routing kernel expects permuted idx and grouped Gemm launch config buffers");
    TLLM_CHECK_WITH_INFO(data.mTopK <= MaxSupportedTopExperts, "Routing kernel expects topK experts <= %d, got %d",
        MaxSupportedTopExperts, data.mTopK);
    TLLM_CHECK_WITH_INFO(data.mNumExperts <= NumExpertsLimit,
        "Routing kernel expects #experts %d to be no more than %d", data.mNumExperts, NumExpertsLimit);
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
        LAUNCH_ROUTING_MINIMAX(data, false, routingIndicesBlockKernel, 1, numThreadsHist,
            0, stream);
    }
    else if (useSingleCluster)
    {
        LAUNCH_ROUTING_MINIMAX(data, false, routingIndicesClusterKernel, NumBlocksPerCluster, numThreadsHist,
            0, stream);
    }
    else
    {
        uint32_t const expandedIdxSize = data.mNumTokens * data.mTopK;
        uint32_t const histogramEltsPerBlock = 8 * numThreadsHist;
        uint32_t const offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * numThreadsHist;

        uint32_t const maxNumBlocks = 1024;

        int const numBlocksHistogram
            = std::min((expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
        int const numBlocksOffsets
            = std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

        if (data.mPtrScores != nullptr && data.mPtrTopKIds == nullptr)
        {
            LAUNCH_ROUTING_MINIMAX(data, false, routingIndicesHistogramScoresKernel, maxNumBlocks, numThreadsHist,
                0, stream);
        }
        else
        {
            LAUNCH_ROUTING_MINIMAX(data, false, routingInitExpertCounts,
                (2 * data.mNumExperts - 1) / numThreadsHist + 1, numThreadsHist,
                0, stream);
        }
        LAUNCH_ROUTING_MINIMAX(data, false, routingIndicesHistogramKernel, numBlocksHistogram, numThreadsHist,
            0, stream);
        LAUNCH_ROUTING_MINIMAX(data, false, routingIndicesOffsetsKernel, numBlocksOffsets, numThreadsHist,
            0, stream);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingMiniMax
} // namespace moe::dev::routing