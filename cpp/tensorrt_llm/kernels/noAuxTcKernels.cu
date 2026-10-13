/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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
#include "tensorrt_llm/kernels/noAuxTcKernels.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdlib>

namespace cg = cooperative_groups;
using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
static constexpr int WARP_SIZE = 32;
static constexpr int NumDeepseekExperts = 256;
static constexpr int MaxSupportedExpertCount = 1024;
static constexpr int NumTopGroupScores = 2;
static constexpr int DefaultMaxNumTopExperts = 8;
static constexpr int MaxSupportedTopExperts = 32;
static constexpr int DefaultMaxNumTopGroups = 4;
static constexpr int LargeMaxNumTopGroups = 8;

static __device__ inline float sigmoid_accurate(float x)
{
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}

// Warp-wide max of a 32-bit unsigned value: one redux.sync on SM100+, shuffle tree elsewhere.
static __device__ __forceinline__ uint32_t warpMaxU32(cg::thread_block_tile<WARP_SIZE> const& warp, uint32_t v)
{
    if constexpr (reduce_topk::kTLLM_GEN_HAS_FAST_REDUX)
    {
        uint32_t r;
        asm("redux.sync.max.u32 %0, %1, 0xffffffff;\n" : "=r"(r) : "r"(v));
        return r;
    }
    else
    {
        return cg::reduce(warp, v, cg::greater<uint32_t>{});
    }
}

// One arg-max round over the warp's key slots (keys as produced by TopKRedType<float>::makeCmpVal:
// high word = order-preserving value bits, low 16 bits = 65535 - expert index, so "larger key" means
// "larger value, then smaller index" — the same total order the iterated reduceTopK used). The winner
// is removed from the slot that held it (slot := 0, below every real key). Two 32-bit warp maxima
// replace the previous 64-bit shuffle reduction.
template <int N>
static __device__ __forceinline__ uint64_t warpArgMaxRound(
    cg::thread_block_tile<WARP_SIZE> const& warp, uint64_t (&slot)[N])
{
    uint64_t best = slot[0];
#pragma unroll
    for (int ii = 1; ii < N; ++ii)
    {
        best = slot[ii] > best ? slot[ii] : best;
    }
    uint32_t const bestHi = static_cast<uint32_t>(best >> 32);
    uint32_t const hi = warpMaxU32(warp, bestHi);
    uint32_t const lo = warpMaxU32(warp, bestHi == hi ? static_cast<uint32_t>(best & 0xFFFFu) : 0u);
    uint64_t const winner = (static_cast<uint64_t>(hi) << 32) | lo;
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        slot[ii] = slot[ii] == winner ? uint64_t{0} : slot[ii];
    }
    return winner;
}

// SmallBatch selects the ungrouped implementation: true = two-level parallel top-k (all warps, redux.sync)
// that minimizes per-CTA latency for decode batches; false = the original single-warp iterated
// reduceTopK, which issues fewer instructions and wins once thousands of CTAs saturate the GPU
// (prefill). Both produce bit-identical results; invokeNoAuxTc picks by token count.
template <typename InputT, typename BiasT, typename OutputT, typename IdxT, int MaxNumExperts, bool UseGroups,
    int MaxNumTopExperts = DefaultMaxNumTopExperts, int MaxNumTopGroups = DefaultMaxNumTopGroups,
    bool SmallBatch = true>
__device__ __forceinline__ void deepseek_v3_topk_block(InputT* scores, OutputT* topkValues, IdxT* topkIndices,
    BiasT* routingBias, int64_t const numTokens, int64_t const numGroup, int64_t const topkGroup, int64_t const topk,
    int64_t const numExperts, int64_t const numExpertsPerGroup, double const routedScalingFactor, int64_t tokenIdx)
{
    __shared__ float __attribute((aligned(128))) smemScoreSigmoid[MaxNumExperts];
    __shared__ float __attribute((aligned(128))) smemScoreBias[MaxNumExperts];

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);

    int32_t laneIdx = threadIdx.x % WARP_SIZE;
    int32_t warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WARP_SIZE, 0);

    static constexpr float invalidScoreFloat = float{-INFINITY};

    topkValues += tokenIdx * topk;
    topkIndices += tokenIdx * topk;

    if constexpr (UseGroups)
    {
        int constexpr NumWarps = MaxNumExperts / WARP_SIZE;
        __shared__ float __attribute((aligned(128))) smemGroupScores[NumWarps];

        auto threadExpert = warpIdx * numExpertsPerGroup + laneIdx;
        bool expertSelected = laneIdx < numExpertsPerGroup;

        auto scoreIdx = tokenIdx * static_cast<int64_t>(numExperts) + threadExpert;
        auto biasVal = expertSelected ? static_cast<float>(routingBias[threadExpert]) : invalidScoreFloat;
        float score = expertSelected ? static_cast<float>(scores[scoreIdx]) : invalidScoreFloat;
        auto scoreSigmoid = sigmoid_accurate(score);
        if (expertSelected)
        {
            smemScoreSigmoid[threadExpert] = scoreSigmoid;
        }
        auto scoreBias = float{scoreSigmoid + float{biasVal}};
        if (expertSelected)
        {
            smemScoreBias[threadExpert] = scoreBias;
        }

        float topExpGroupScores[NumTopGroupScores];
        [[maybe_unused]] int32_t topExpGroupIdx[NumTopGroupScores];
        reduce_topk::reduceTopK(warp, topExpGroupScores, topExpGroupIdx, scoreBias, threadExpert,
            /* minValue */ invalidScoreFloat);

        if (warp.thread_rank() == 0)
        {
            auto groupScore = topExpGroupScores[0] + topExpGroupScores[1];
            smemGroupScores[warpIdx] = groupScore;
        }

        __syncthreads();

        float topScores[MaxNumTopExperts];
        int32_t topExperts[MaxNumTopExperts];

        if (warpIdx == 0)
        {
            float topGroups[MaxNumTopGroups];
            int32_t topGroupIdx[MaxNumTopGroups];
            float groupScore = laneIdx < numGroup ? smemGroupScores[laneIdx] : invalidScoreFloat;
            reduce_topk::reduceTopK(warp, topGroups, topGroupIdx, groupScore, laneIdx,
                /* minValue */ invalidScoreFloat);

            float expertScoreGroup[MaxNumTopGroups];
            int32_t expertIdxGroup[MaxNumTopGroups];
#pragma unroll
            for (int ii = 0; ii < MaxNumTopGroups; ++ii)
            {
                auto groupIdx = topGroupIdx[ii];
                expertIdxGroup[ii] = groupIdx * numExpertsPerGroup + laneIdx;
                expertScoreGroup[ii]
                    = (ii < topkGroup) && expertSelected ? smemScoreBias[expertIdxGroup[ii]] : invalidScoreFloat;
            }

            reduce_topk::reduceTopK(
                warp, topScores, topExperts, expertScoreGroup, expertIdxGroup, /* minValue */ invalidScoreFloat, topk);

            int32_t expertIdx = laneIdx < topk ? topExperts[laneIdx] : MaxNumExperts - 1;
            float scoreNorm = laneIdx < topk ? smemScoreSigmoid[expertIdx] : 0.F;
            auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
            auto finalScore = static_cast<OutputT>(scoreNorm * routedScalingFactor / (redNorm + 1e-20));
            if (laneIdx < topk)
            {
                topkValues[laneIdx] = static_cast<OutputT>(finalScore);
                topkIndices[laneIdx] = expertIdx;
            }
        }
    }
    else if constexpr (!SmallBatch)
    {
        // Original ungrouped implementation (throughput regime): warp 0 runs the iterated reduceTopK.
        for (int e = threadIdx.x; e < numExperts; e += blockDim.x)
        {
            auto scoreIdx = tokenIdx * static_cast<int64_t>(numExperts) + e;
            auto biasVal = static_cast<float>(routingBias[e]);
            float score = static_cast<float>(scores[scoreIdx]);
            auto scoreSigmoid = sigmoid_accurate(score);
            smemScoreSigmoid[e] = scoreSigmoid;
            smemScoreBias[e] = scoreSigmoid + biasVal;
        }

        __syncthreads();

        float topScores[MaxNumTopExperts];
        int32_t topExperts[MaxNumTopExperts];

        if (warpIdx == 0)
        {
            constexpr int NumChunks = (MaxNumExperts + WARP_SIZE - 1) / WARP_SIZE;
            float localScores[NumChunks];
            int32_t localIdx[NumChunks];
#pragma unroll
            for (int ii = 0; ii < NumChunks; ++ii)
            {
                auto expertIdx = ii * WARP_SIZE + laneIdx;
                localIdx[ii] = expertIdx;
                localScores[ii] = expertIdx < numExperts ? smemScoreBias[expertIdx] : invalidScoreFloat;
            }
            reduce_topk::reduceTopK(warp, topScores, topExperts, localScores, localIdx,
                /* minValue */ invalidScoreFloat, topk);

            int32_t expertIdx = laneIdx < topk ? topExperts[laneIdx] : MaxNumExperts - 1;
            float scoreNorm = laneIdx < topk ? smemScoreSigmoid[expertIdx] : 0.F;
            auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
            auto finalScore = static_cast<OutputT>(scoreNorm * routedScalingFactor / (redNorm + 1e-20));
            if (laneIdx < topk)
            {
                topkValues[laneIdx] = static_cast<OutputT>(finalScore);
                topkIndices[laneIdx] = expertIdx;
            }
        }
    }
    else
    {
        // Ungrouped (n_group == 1, e.g. Kimi K3: 896 experts / top-16): two-level parallel top-k.
        //
        // The previous implementation had warp 0 alone run Sort<MaxNumExperts/32> per lane followed by
        // `topk` serial 64-bit shuffle reductions while the other warps idled, and its phase 1 issued
        // the per-thread global loads one dependent iteration at a time -- ~6.8 us per CTA for the
        // 1024-expert tier on B200; with one CTA per token that latency is fully exposed at small
        // decode batches. Here:
        //   phase 1: every thread prefetches all its scores/bias (one round of load latency), then
        //            writes the sigmoid and the packed (value, index) key of reduce_topk (a strict total
        //            order: higher value first, then lower index);
        //   phase 2: each 128-expert batch is handled by one warp: `topk` arg-max rounds, each two
        //            32-bit redux.sync maxima (value word, then index word) instead of a 64-bit shuffle
        //            tree -- all warps work in parallel;
        //   phase 3: warp 0 runs the same rounds over the NumBatches*topk candidates.
        // Every round removes exactly the maximum key under the same total order the old iterated
        // reduceTopK used, so the selected experts and their order are identical (bit-exact).
        using RedType = reduce_topk::TopKRedType<float>;
        using KeyT = typename RedType::TypeCmp;
        static_assert(std::is_same_v<KeyT, uint64_t>, "float keys are 64-bit");
        static constexpr int ChunksPerBatch = 4;
        static constexpr int BatchSize = ChunksPerBatch * WARP_SIZE; // 128 experts per batch
        static_assert(MaxNumExperts % BatchSize == 0, "MaxNumExperts must be a multiple of 128");
        static_assert(BatchSize >= MaxNumTopExperts, "a batch must hold at least topk experts");
        static constexpr int NumBatches = MaxNumExperts / BatchSize;
        static constexpr int MaxCandPerLane = (NumBatches * MaxNumTopExperts + WARP_SIZE - 1) / WARP_SIZE;
        // All launchers use blockDim >= 128 (128/256 threads, 448 in the K3 fused kernel).
        static constexpr int MinBlockDim = 128;
        static constexpr int MaxItersPhase1 = MaxNumExperts / MinBlockDim;
        __shared__ KeyT __attribute((aligned(128))) smemKey[MaxNumExperts];
        __shared__ KeyT __attribute((aligned(128))) smemCand[NumBatches * MaxNumTopExperts];

        int32_t const numWarps = blockDim.x / WARP_SIZE;

        // Phase 1: prefetch, then sigmoid + bias packed into a sortable key. Padding experts get -inf
        // (never selected unless numExperts < topk, matching the previous behaviour).
        float scoreArr[MaxItersPhase1];
        float biasArr[MaxItersPhase1];
#pragma unroll
        for (int ii = 0; ii < MaxItersPhase1; ++ii)
        {
            int const e = threadIdx.x + ii * blockDim.x;
            if (e < numExperts)
            {
                scoreArr[ii] = static_cast<float>(scores[tokenIdx * static_cast<int64_t>(numExperts) + e]);
                biasArr[ii] = static_cast<float>(routingBias[e]);
            }
        }
#pragma unroll
        for (int ii = 0; ii < MaxItersPhase1; ++ii)
        {
            int const e = threadIdx.x + ii * blockDim.x;
            if (e < MaxNumExperts)
            {
                float scoreBias = invalidScoreFloat;
                float scoreSigmoid = 0.F;
                if (e < numExperts)
                {
                    scoreSigmoid = sigmoid_accurate(scoreArr[ii]);
                    scoreBias = scoreSigmoid + biasArr[ii];
                }
                smemScoreSigmoid[e] = scoreSigmoid;
                smemKey[e] = RedType::makeCmpVal(scoreBias, e);
            }
        }
        __syncthreads();

        // Phase 2: per-batch top-k candidates (one warp per batch, warps loop over batches).
        for (int b = warpIdx; b < NumBatches; b += numWarps)
        {
            KeyT slot[ChunksPerBatch];
#pragma unroll
            for (int ii = 0; ii < ChunksPerBatch; ++ii)
            {
                slot[ii] = smemKey[b * BatchSize + ii * WARP_SIZE + laneIdx];
            }
            for (int kk = 0; kk < topk; ++kk)
            {
                KeyT const winner = warpArgMaxRound(warp, slot);
                if (laneIdx == kk)
                {
                    smemCand[b * topk + kk] = winner;
                }
            }
        }
        __syncthreads();

        // Phase 3 + finalize (warp 0): top-k over the NumBatches * topk candidates; lane kk keeps the
        // kk-th largest expert, then the normalization is the same arithmetic as before.
        if (warpIdx == 0)
        {
            int32_t const numCand = NumBatches * static_cast<int32_t>(topk);
            KeyT slot[MaxCandPerLane];
#pragma unroll
            for (int ii = 0; ii < MaxCandPerLane; ++ii)
            {
                int const c = ii * WARP_SIZE + laneIdx;
                slot[ii] = c < numCand ? smemCand[c] : KeyT{0};
            }
            KeyT myOut = 0;
            for (int kk = 0; kk < topk; ++kk)
            {
                KeyT const winner = warpArgMaxRound(warp, slot);
                myOut = laneIdx == kk ? winner : myOut;
            }

            int32_t expertIdx = MaxNumExperts - 1;
            float scoreNorm = 0.F;
            if (laneIdx < topk)
            {
                float unusedScore;
                RedType::unpack(unusedScore, expertIdx, myOut);
                scoreNorm = smemScoreSigmoid[expertIdx];
            }
            auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
            auto finalScore = static_cast<OutputT>(scoreNorm * routedScalingFactor / (redNorm + 1e-20));
            if (laneIdx < topk)
            {
                topkValues[laneIdx] = static_cast<OutputT>(finalScore);
                topkIndices[laneIdx] = expertIdx;
            }
        }
    }
}

template <typename InputT, typename BiasT, typename OutputT, typename IdxT, int MaxNumExperts, bool UseGroups,
    int MaxNumTopExperts = DefaultMaxNumTopExperts, int MaxNumTopGroups = DefaultMaxNumTopGroups,
    bool SmallBatch = true>
__global__ void deepseek_v3_topk_kernel(InputT* scores, OutputT* topkValues, IdxT* topkIndices, BiasT* routingBias,
    int64_t const numTokens, int64_t const numGroup, int64_t const topkGroup, int64_t const topk,
    int64_t const numExperts, int64_t const numExpertsPerGroup, double const routedScalingFactor)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    if constexpr (UseGroups)
    {
        if (threadIdx.x / WARP_SIZE >= numGroup)
        {
            return;
        }
    }
    deepseek_v3_topk_block<InputT, BiasT, OutputT, IdxT, MaxNumExperts, UseGroups, MaxNumTopExperts, MaxNumTopGroups,
        SmallBatch>(scores, topkValues, topkIndices, routingBias, numTokens, numGroup, topkGroup, topk, numExperts,
        numExpertsPerGroup, routedScalingFactor, blockIdx.x);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// K3 decode specialization: CTAs [0, M) route 896 experts to top-16, while
// CTAs [M, 2M) independently quantize one 3584-wide BF16 activation row to
// MXFP8 with linear UE8M0 group-32 scales.
static constexpr int KimiK3NumExperts = 896;
static constexpr int KimiK3TopK = 16;
static constexpr int KimiK3HiddenSize = 3584;
static constexpr int MxFp8SfVecSize = 32;
static constexpr int KimiK3QuantThreads = KimiK3HiddenSize / CVT_ELTS_PER_THREAD;

__global__ __launch_bounds__(KimiK3QuantThreads) void kimi_k3_noaux_tc_mxfp8_quant_kernel(float* scores,
    float* routingBias, __nv_bfloat16* hiddenStates, __nv_bfloat16* topkValues, int32_t* topkIndices,
    int64_t* quantizedHiddenStates, int32_t* hiddenStatesScale, int64_t numTokens, double routedScalingFactor)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    if (blockIdx.x < numTokens)
    {
        deepseek_v3_topk_block<float, float, __nv_bfloat16, int32_t, MaxSupportedExpertCount, false,
            MaxSupportedTopExperts>(scores, topkValues, topkIndices, routingBias, numTokens, 1, 1, KimiK3TopK,
            KimiK3NumExperts, KimiK3NumExperts, routedScalingFactor, blockIdx.x);
    }
    else
    {
        using QuantT = __nv_bfloat16;
        using QuantPackedVec = PackedVec<QuantT>;
        static constexpr int CvtNumThreadsPerSf = MxFp8SfVecSize / CVT_ELTS_PER_THREAD;
        int const rowIdx = blockIdx.x - numTokens;
        int const colIdx = threadIdx.x;
        int const numColThreads = KimiK3HiddenSize / CVT_ELTS_PER_THREAD;

        std::optional<int> optionalNumRows = numTokens;
        auto sfOut = cvt_quant_get_sf_out_offset<uint32_t, CvtNumThreadsPerSf>(std::nullopt, rowIdx, colIdx,
            optionalNumRows, KimiK3HiddenSize / MxFp8SfVecSize, reinterpret_cast<uint32_t*>(hiddenStatesScale),
            QuantizationSFLayout::LINEAR);
        int64_t const offset = static_cast<int64_t>(rowIdx) * numColThreads + colIdx;
        QuantPackedVec inVec = reinterpret_cast<QuantPackedVec const*>(hiddenStates)[offset];
        reinterpret_cast<uint64_t*>(quantizedHiddenStates)[offset]
            = cvt_warp_fp16_to_mxfp8<QuantT, MxFp8SfVecSize>(inVec, sfOut);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    __threadfence();
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void invokeKimiK3NoAuxTcMxFp8Quant(float* scores, float* bias, __nv_bfloat16* hidden_states, __nv_bfloat16* topk_values,
    int32_t* topk_indices, int64_t* quantized_hidden_states, int32_t* hidden_states_scale, int64_t const num_tokens,
    double const routed_scaling_factor, cudaStream_t const stream)
{
    cudaLaunchConfig_t config;
    config.gridDim = 2 * num_tokens;
    config.blockDim = KimiK3QuantThreads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;

    cudaLaunchKernelEx(&config, kimi_k3_noaux_tc_mxfp8_quant_kernel, scores, bias, hidden_states, topk_values,
        topk_indices, quantized_hidden_states, hidden_states_scale, num_tokens, routed_scaling_factor);
    sync_check_cuda_error(stream);
}

template <typename InputT, typename BiasT, typename OutputT, typename IdxT>
void invokeNoAuxTc(InputT* scores, BiasT* bias, OutputT* topk_values, IdxT* topk_indices, int64_t const num_tokens,
    int64_t const num_experts, int64_t const n_group, int64_t const topk_group, int64_t const topk,
    double const routed_scaling_factor, cudaStream_t const stream)
{
    bool const is_single_group
        = (n_group <= 1) && (num_experts <= MaxSupportedExpertCount) && (topk <= MaxSupportedTopExperts);

    int64_t const experts_per_group = num_experts / n_group;
    bool const is_multi_group = (n_group > 1) && (num_experts <= NumDeepseekExperts) && (experts_per_group <= WARP_SIZE)
        && (topk <= DefaultMaxNumTopExperts) && (experts_per_group * topk_group <= LargeMaxNumTopGroups * WARP_SIZE);

    if (is_single_group || is_multi_group)
    {
        cudaLaunchConfig_t config;
        auto* kernel_instance = &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, NumDeepseekExperts, true>;
        int num_threads = NumDeepseekExperts;

        if (is_multi_group)
        {
            if (experts_per_group * topk_group <= DefaultMaxNumTopGroups * WARP_SIZE)
            {
                kernel_instance = &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, NumDeepseekExperts, true>;
            }
            else
            {
                kernel_instance = &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, NumDeepseekExperts, true,
                    DefaultMaxNumTopExperts, LargeMaxNumTopGroups>;
            }
            num_threads = NumDeepseekExperts;
        }
        else if (is_single_group)
        {
            // Small batches (decode) use the latency-optimized two-level top-k; large batches keep the
            // original single-warp implementation (fewer instructions once the GPU is saturated).
            // Measured on B200 (148 SMs), 896 experts / top-16, kernel-only old -> new: 8 tok 6.84 -> 5.12 us,
            // 128 tok 8.23 -> 5.70, 192-256 tok -3 %, 384 tok +9 %, 512 tok 10.9 -> 12.3, 8192 tok 94 -> 109.
            // The break is where CTAs start sharing SMs, so the default crossover is 256 tokens.
            // TLLM_NOAUX_TC_SMALL_BATCH_MAX_TOKENS overrides it (0 = always the original path).
            static int const smallBatchMaxTokens = []
            {
                char const* env = std::getenv("TLLM_NOAUX_TC_SMALL_BATCH_MAX_TOKENS");
                return env != nullptr ? std::atoi(env) : 256;
            }();
            bool const smallBatch = num_tokens <= smallBatchMaxTokens;
            if (num_experts <= 128)
            {
                kernel_instance = smallBatch ? &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, 128, false,
                                      MaxSupportedTopExperts, DefaultMaxNumTopGroups, true>
                                             : &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, 128, false,
                                                 MaxSupportedTopExperts, DefaultMaxNumTopGroups, false>;
                num_threads = 128;
            }
            else if (num_experts <= 256)
            {
                kernel_instance = smallBatch ? &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, 256, false,
                                      MaxSupportedTopExperts, DefaultMaxNumTopGroups, true>
                                             : &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, 256, false,
                                                 MaxSupportedTopExperts, DefaultMaxNumTopGroups, false>;
                num_threads = 256;
            }
            else if (num_experts <= 512)
            {
                kernel_instance = smallBatch ? &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, 512, false,
                                      MaxSupportedTopExperts, DefaultMaxNumTopGroups, true>
                                             : &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, 512, false,
                                                 MaxSupportedTopExperts, DefaultMaxNumTopGroups, false>;
                num_threads = 256;
            }
            else
            {
                kernel_instance = smallBatch ? &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, 1024, false,
                                      MaxSupportedTopExperts, DefaultMaxNumTopGroups, true>
                                             : &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, 1024, false,
                                                 MaxSupportedTopExperts, DefaultMaxNumTopGroups, false>;
                num_threads = 256;
            }
        }

        config.gridDim = num_tokens;
        config.blockDim = num_threads;
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaLaunchKernelEx(&config, kernel_instance, scores, topk_values, topk_indices, bias, num_tokens, n_group,
            topk_group, topk, num_experts, num_experts / n_group, routed_scaling_factor);
        sync_check_cuda_error(stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false,
            "invokeNoAuxTc: unsupported configuration (n_group=%ld, num_experts=%ld, topk_group=%ld, topk=%ld). "
            "Please use original pytorch implementation.",
            n_group, num_experts, topk_group, topk);
    }
}

#define INSTANTIATE_NOAUX_TC(InputT, BiasT, OutputT, IdxT)                                                             \
    template void invokeNoAuxTc<InputT, BiasT, OutputT, IdxT>(InputT * scores, BiasT * bias, OutputT * topk_values,    \
        IdxT * topk_indices, int64_t const num_tokens, int64_t const num_experts, int64_t const n_group,               \
        int64_t const topk_group, int64_t const topk, double const routed_scaling_factor, cudaStream_t const stream);

INSTANTIATE_NOAUX_TC(float, float, float, int32_t);
INSTANTIATE_NOAUX_TC(float, half, float, int32_t);

INSTANTIATE_NOAUX_TC(half, float, half, int32_t);
INSTANTIATE_NOAUX_TC(half, half, half, int32_t);

#ifdef ENABLE_BF16
INSTANTIATE_NOAUX_TC(float, __nv_bfloat16, float, int32_t);
INSTANTIATE_NOAUX_TC(half, __nv_bfloat16, half, int32_t);

INSTANTIATE_NOAUX_TC(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, float, __nv_bfloat16, int32_t);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, half, __nv_bfloat16, int32_t);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END
