/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

#include "RoutingKernel.cuh"

namespace moe::dev::routing
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// Preprocess policies: applied to all expert scores BEFORE topK selection.
//
// Each policy must provide:
//   - template <typename InputT> using BaseType
//       The data type used for intermediate score computation.
//   - template <typename OutputT> struct Params { void set(Data const&); }
//       Policy-specific runtime data, populated from the host-side Data struct.
//       Empty for policies that don't need extra data (zero register cost).
//   - template <typename DataType, int VecSize, typename ParamsT>
//     static void apply(warp, score[VecSize], idx[VecSize], numExperts, params)
//       Transforms scores in-place before topK selection.
////////////////////////////////////////////////////////////////////////////////////////////////////

/// No-op: scores are passed through unchanged.
struct NoOpPreprocess
{
    /// BaseType: when no preprocess is applied, use the input type directly.
    template <typename InputT>
    using BaseType = InputT;

    template <typename OutputT>
    struct Params
    {
        void set(routingCustom::Data const& /*data*/) {}
    };

    template <typename DataType, int VecSize, typename ParamsT>
    __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& /*warp*/,
        DataType (& /*score*/)[VecSize], int32_t const (& /*idx*/)[VecSize], int32_t /*numExperts*/,
        ParamsT const& /*params*/)
    {
    }
};

/// Softmax: applies softmax over all expert scores before topK selection.
struct SoftmaxPreprocess
{
    /// BaseType: softmax is always computed in float for numerical stability.
    template <typename InputT>
    using BaseType = float;

    template <typename OutputT>
    struct Params
    {
        void set(routingCustom::Data const& /*data*/) {}
    };

    template <typename DataType, int VecSize, typename ParamsT>
    __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& warp,
        DataType (&score)[VecSize], int32_t const (& /*idx*/)[VecSize], int32_t /*numExperts*/,
        ParamsT const& /*params*/)
    {
        calcSoftmax(warp, score);
    }
};

/// Sigmoid: applies sigmoid(score) for topK selection (no bias).
/// Used by Cohere-style routing where expert selection is based on raw sigmoid scores.
struct SigmoidPreprocess
{
    /// BaseType: sigmoid is computed in float for numerical stability.
    template <typename InputT>
    using BaseType = float;

    template <typename OutputT>
    struct Params
    {
        void set(routingCustom::Data const& /*data*/) {}
    };

    template <typename DataType, int VecSize, typename ParamsT>
    __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& /*warp*/,
        DataType (&score)[VecSize], int32_t const (&idx)[VecSize], int32_t numExperts, ParamsT const& /*params*/)
    {
#pragma unroll
        for (int i = 0; i < VecSize; i++)
        {
            float s = sigmoid_accurate(static_cast<float>(score[i]));
            score[i] = idx[i] < numExperts ? static_cast<DataType>(s) : DataType{-INFINITY};
        }
    }
};

/// SigmoidBias: applies sigmoid(score) + bias[expertIdx] for topK selection.
/// Used by DeepSeek-style routing where expert selection is based on biased sigmoid scores.
struct SigmoidBiasPreprocess
{
    /// BaseType: sigmoid is computed in float for numerical stability.
    template <typename InputT>
    using BaseType = float;

    template <typename OutputT>
    struct Params
    {
        // Store as void const* to support any bias dtype (float, bfloat16, etc.) without conversion.
        void const* ptrRoutingBias = nullptr;
        batchedGemm::trtllm::gen::Dtype dtypeBias = batchedGemm::trtllm::gen::Dtype::Bfloat16;

        void set(routingCustom::Data const& data)
        {
            ptrRoutingBias = data.mPtrRoutingBias;
            dtypeBias = data.mDtypeBias;
        }
    };

    template <typename DataType, int VecSize, typename ParamsT>
    __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& /*warp*/,
        DataType (&score)[VecSize], int32_t const (&idx)[VecSize], int32_t numExperts, ParamsT const& params)
    {
#pragma unroll
        for (int i = 0; i < VecSize; i++)
        {
            float s = sigmoid_accurate(static_cast<float>(score[i]));
            float bias
                = idx[i] < numExperts ? loadScalar(params.ptrRoutingBias, idx[i], params.dtypeBias) : float{-INFINITY};
            score[i] = static_cast<DataType>(s + bias);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Postprocess policies: applied to the top-K scores AFTER topK selection.
//
// Each policy must provide:
//   - template <typename OutputT> struct Params { void set(Data const&); }
//       Policy-specific runtime data. Empty when not needed.
//   - template <typename DataType, int K, typename ParamsT>
//     static void apply(warp, warpTopKScore[K], warpTopKExpertIdx[K], laneIdx, topK, params)
//       Transforms top-K scores in-place after topK selection.
////////////////////////////////////////////////////////////////////////////////////////////////////

/// No-op: top-K scores are left unchanged.
struct NoOpPostprocess
{
    template <typename OutputT>
    struct Params
    {
        void set(routingCustom::Data const& /*data*/) {}
    };

    template <typename DataType, int K, typename ParamsT>
    __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& /*warp*/,
        DataType (& /*warpTopKScore*/)[K], int32_t const (& /*warpTopKExpertIdx*/)[K], int32_t /*laneIdx*/,
        int32_t /*topK*/, ParamsT const& /*params*/)
    {
    }
};

/// Softmax: applies softmax over the top-K scores.
struct SoftmaxPostprocess
{
    template <typename OutputT>
    struct Params
    {
        void set(routingCustom::Data const& /*data*/) {}
    };

    template <typename DataType, int K, typename ParamsT>
    __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& warp,
        DataType (&warpTopKScore)[K], int32_t const (& /*warpTopKExpertIdx*/)[K], int32_t laneIdx, int32_t topK,
        ParamsT const& /*params*/)
    {
        DataType minScore = DataType{-INFINITY};
        auto softmaxScore = calcSoftmax(warp, laneIdx < topK ? warpTopKScore[laneIdx] : minScore, laneIdx, topK);
        if (laneIdx < topK)
        {
            warpTopKScore[laneIdx] = softmaxScore;
        }
    }
};

/// SumNormalize: divides each top-K score by the sum of all top-K scores.
/// Used when softmax has already been applied before topK selection.
struct SumNormalizePostprocess
{
    template <typename OutputT>
    struct Params
    {
        bool normTopkProb = true;

        void set(routingCustom::Data const& data)
        {
            normTopkProb = data.mNormTopkProb;
        }
    };

    template <typename DataType, int K, typename ParamsT>
    __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& warp,
        DataType (&warpTopKScore)[K], int32_t const (& /*warpTopKExpertIdx*/)[K], int32_t laneIdx, int32_t topK,
        ParamsT const& params)
    {
        float sum = float{1.f};
        if (params.normTopkProb)
        {
            sum = static_cast<float>(laneIdx < topK ? warpTopKScore[laneIdx] : 0);
            sum = cg::reduce(warp, sum, cg::plus<float>());
        }
        if (laneIdx < topK)
        {
            warpTopKScore[laneIdx] = warpTopKScore[laneIdx] / sum;
        }
    }
};

/// ScaledSumNormalize: recovers un-biased sigmoid scores by subtracting per-expert bias from the
/// selection scores (sigmoid + bias), then normalizes by sum and applies routeScale.
/// Used by DeepSeek-style routing: final_weight = sigmoid(raw) * routeScale / (sum + epsilon).
/// DeepSeek uses epsilon=0 (no guard); MiniMax2 uses epsilon=1e-20 to prevent division by zero.
struct ScaledSumNormalizePostprocess
{
    template <typename OutputT>
    struct Params
    {
        // Store as void const* to support any bias dtype (float, bfloat16, etc.) without conversion.
        void const* ptrRoutingBias = nullptr;
        batchedGemm::trtllm::gen::Dtype dtypeBias = batchedGemm::trtllm::gen::Dtype::Bfloat16;
        float routeScale = 1.0f;
        float sumEpsilon = 0.0f;

        void set(routingCustom::Data const& data)
        {
            ptrRoutingBias = data.mPtrRoutingBias;
            dtypeBias = data.mDtypeBias;
            routeScale = data.mRouteScale;
            sumEpsilon = data.mSumEpsilon;
        }
    };

    template <typename DataType, int K, typename ParamsT>
    __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& warp,
        DataType (&warpTopKScore)[K], int32_t const (&warpTopKExpertIdx)[K], int32_t laneIdx, int32_t topK,
        ParamsT const& params)
    {
        // Recover sigmoid score: selection_score = sigmoid(raw) + bias, so sigmoid = score - bias
        float biasVal
            = laneIdx < topK ? loadScalar(params.ptrRoutingBias, warpTopKExpertIdx[laneIdx], params.dtypeBias) : 0.f;
        float sigmoidScore = laneIdx < topK ? (static_cast<float>(warpTopKScore[laneIdx]) - biasVal) : 0.f;
        float sum = cg::reduce(warp, sigmoidScore, cg::plus<float>());
        if (laneIdx < topK)
        {
            warpTopKScore[laneIdx]
                = static_cast<DataType>(sigmoidScore * params.routeScale / (sum + params.sumEpsilon));
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// ExpertSelectPolicy: encapsulates the entire expert selection logic.
//
// Each policy must provide:
//   - template <typename InputT> using BaseType
//       The data type used for intermediate score computation.
//   - template <typename OutputT> struct Params { void set(Data const&); }
//       Policy-specific runtime data, populated from the host-side Data struct.
//       Empty for policies that don't need extra data (zero register cost).
//   - template <typename DataType, typename InputType, int VecSize, int K, typename KP>
//     static void apply(warp, warpTopKScore[K], warpTopKExpertIdx[K], laneIdx, numExperts, topK,
//                        ptrScores, params)
//       Selects the top-K experts and computes their weights.
//
// The default TopKExpertSelect wraps existing PreprocessPolicy + PostprocessPolicy,
// but users can write completely custom policies that bypass the preprocess+topK+postprocess
// pattern (e.g., lookup-table-based expert selection).
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Default ExpertSelectPolicy: preprocess + topK reduction + postprocess.
/// Wraps existing PreprocessPolicy and PostprocessPolicy as internal composition.
template <typename PreprocessPolicy_, typename PostprocessPolicy_>
struct TopKExpertSelect
{
    /// BaseType: delegated to the preprocess policy.
    template <typename InputT>
    using BaseType = typename PreprocessPolicy_::template BaseType<InputT>;

    /// Params: combines preprocess and postprocess runtime parameters.
    template <typename OutputT>
    struct Params
    {
        typename PreprocessPolicy_::template Params<OutputT> mPreprocessParams;
        typename PostprocessPolicy_::template Params<OutputT> mPostprocessParams;

        void set(routingCustom::Data const& data)
        {
            mPreprocessParams.set(data);
            mPostprocessParams.set(data);
        }
    };

    /// Selects top-K experts using preprocess → topK reduction → postprocess.
    template <typename DataType, typename InputType, int VecSize, int K, typename KP>
    __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& warp,
        DataType (&warpTopKScore)[K], int32_t (&warpTopKExpertIdx)[K], int32_t const laneIdx, int32_t const numExperts,
        int32_t topK, InputType const* ptrScores, KP const& params)
    {
        DataType minScore = DataType{-INFINITY};
        DataType score[VecSize];
        int32_t idx[VecSize];

        for (int i = 0; i < VecSize; i++)
        {
            auto expertIdx = i * WarpSize + laneIdx;
            auto newScore = expertIdx < numExperts ? static_cast<DataType>(ptrScores[expertIdx]) : minScore;
            score[i] = newScore;
            idx[i] = expertIdx;
        }

        // Apply preprocess (e.g. softmax over all scores, sigmoid + bias, ...)
        PreprocessPolicy_::apply(warp, score, idx, numExperts, params.mExpertSelectParams.mPreprocessParams);

        // Get the top-k scores and their corresponding expert indices
        topk::reduceTopK(warp, warpTopKScore, warpTopKExpertIdx, score, idx, minScore, topK);

        // Apply postprocess (e.g. renormalize, softmax over top-K, scaled renormalize, ...)
        PostprocessPolicy_::apply(
            warp, warpTopKScore, warpTopKExpertIdx, laneIdx, topK, params.mExpertSelectParams.mPostprocessParams);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace routingCustom
{
////////////////////////////////////////////////////////////////////////////////////////////////////

// Expert-count tiers (must be multiples of WarpSize=32 and of 4).
// Each tier covers all values ≤ the tier constant.
static constexpr int NumExperts128Experts = 128;
static constexpr int NumExperts160Experts = 160;
static constexpr int NumExperts256Experts = 256;
static constexpr int NumExperts384Experts = 384;
static constexpr int NumExperts512Experts = 512;
static constexpr int NumExperts576Experts = 576;
static constexpr int MaxSupportedExperts = 2048;

// TopK tiers (must be ≤ WarpSize=32).
static constexpr int NumTop4Experts = 4;
static constexpr int NumTop8Experts = 8;
static constexpr int NumTop16Experts = 16;
static constexpr int NumTop22Experts = 22;
static constexpr int MaxSupportedTopExperts = 32;

static constexpr int NumThreads = 1024;
static constexpr int NumWarps = NumThreads / WarpSize;

static constexpr int MaxNumTokensSingleCluster = NumBlocksPerCluster * NumThreads;
static constexpr int MaxNumTokensSingleClusterScores = NumBlocksPerCluster * NumWarps;

static constexpr int BlockKernelMaxNumTokens = 4;
static constexpr int DynBlockKernelMaxNumTokens = 16;
static constexpr int DynBlockKernelMaxNumExperts = 512;

////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t constexpr getMaxNumExperts(int32_t numExperts)
{
    if (numExperts <= NumExperts128Experts)
    {
        return NumExperts128Experts;
    }
    else if (numExperts <= NumExperts160Experts)
    {
        return NumExperts160Experts;
    }
    else if (numExperts <= NumExperts256Experts)
    {
        return NumExperts256Experts;
    }
    else if (numExperts <= NumExperts384Experts)
    {
        return NumExperts384Experts;
    }
    else if (numExperts <= NumExperts512Experts)
    {
        return NumExperts512Experts;
    }
    else if (numExperts <= NumExperts576Experts)
    {
        return NumExperts576Experts;
    }
    else if (numExperts <= MaxSupportedExperts)
    {
        return MaxSupportedExperts;
    }
    else
    {
        TLLM_LOG_ERROR("Unsupported numExperts");
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// TIER PAIR TYPES — compile-time (MaxNumExperts, MaxNumTopExperts) configuration.
//
// Each Tier<E, K> declares a supported kernel instantiation.
// TierList<Tier<...>, ...> is an ordered list tried from first to last.
// The dispatch picks the FIRST pair where numExperts ≤ E AND topK ≤ K.
//
// Pairs must be sorted so that tighter tiers come first:
//   - Sort by E ascending, then by K ascending within equal E.
//   - A config (numExperts, topK) always matches the tightest available pair.
//   - If the tightest expert tier doesn't have a topK that covers the runtime topK,
//     the dispatch falls through to the next larger expert tier that does.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int E_, int K_>
struct Tier
{
    static constexpr int kExperts = E_;
    static constexpr int kTopK = K_;
};

template <typename... Tiers>
struct TierList
{
};

// Recursive dispatch: try each tier in order, call `fn` with the first match.
// fn receives (integral_constant<int, E>, integral_constant<int, K>) as compile-time args.
// Base case: empty list — no match.
template <typename Fn, typename Data>
inline bool dispatchTierPairs(TierList<>*, Data const& /*data*/, Fn&& /*fn*/)
{
    return false;
}

// Recursive case: check First, then recurse on Rest...
template <typename First, typename... Rest, typename Fn, typename Data>
inline bool dispatchTierPairs(TierList<First, Rest...>*, Data const& data, Fn&& fn)
{
    if (data.mNumExperts <= First::kExperts && data.mTopK <= First::kTopK)
    {
        fn(std::integral_constant<int, First::kExperts>{}, std::integral_constant<int, First::kTopK>{});
        return true;
    }
    return dispatchTierPairs(static_cast<TierList<Rest...>*>(nullptr), data, std::forward<Fn>(fn));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// POLICY TIER CONFIGURATION
//
// PolicyTraits<PreProc, PostProc>::Pairs declares the supported (expert, topK) pairs.
// Only these pairs are compiled as kernel instantiations.
// To add support for a new model config, add a Tier<E, K> to the appropriate TierList.
//
// THREAD-COUNT SAFETY: LAUNCH_ROUTING_FOR_POLICY automatically clamps the launch thread
// count to at least min(MaxNumExperts, 1024) from the dispatched tier.  This prevents
// mismatches when a policy's smallest tier is larger than getMaxNumExperts() returns for
// the same numExperts (e.g., 72 experts → getMaxNumExperts returns 128, but a policy
// whose smallest tier is 256 would produce MaxNumExperts=256).  See the comment on
// LAUNCH_ROUTING_FOR_POLICY for details.
//
// ┌──────────────────────────────────────────────────────────────────────────────┐
// │  Policy (PreProc + PostProc)           Supported pairs                      │
// ├──────────────────────────────────────────────────────────────────────────────┤
// │  Softmax + None       (Default)        (128,8)                              │
// │  None + Softmax       (Renormalize)    (128,4) (128,8) (160,8) (256,8)     │
// │                                        (256,16) (512,8) (512,16)  │
// │                                        (512,22) (576,8) (2048,32)          │
// │  Sigmoid + SumNorm    (SigmoidRenorm)  (128,8)                              │
// │  SigmoidBias + ScaleS (DS nGroup≤1)    (128,8) (256,8) (384,8) (512,8)    │
// │                                        (512,22)                            │
// │  Softmax + SumNorm    (RenormNaive)    (128,4) (128,8) (256,8) (512,8)    │
// │                                        (2048,8)                             │
// │  None + None          (fallback)       (128,8)                              │
// └──────────────────────────────────────────────────────────────────────────────┘
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Default: fallback for new/unknown policies.
/// Provides K8 (tight for most models) + K32 (catch-all for high topK) at each common expert tier.
/// Omits 160/384/576 — those are model-specific and handled by explicit specializations.
/// If a new policy needs a tighter tier, add a PolicyTraits specialization.
template <typename PreProc, typename PostProc>
struct PolicyTraits
{
    using Pairs = TierList<Tier<128, 8>, Tier<128, 32>, Tier<256, 8>, Tier<256, 32>, Tier<512, 8>, Tier<512, 32>,
        Tier<2048, 8>, Tier<2048, 32>>;
};

/// Softmax + None (Default): single config.
template <>
struct PolicyTraits<SoftmaxPreprocess, NoOpPostprocess>
{
    using Pairs = TierList<Tier<128, 8>>;
};

/// None + Softmax (Renormalize): many model configs.
template <>
struct PolicyTraits<NoOpPreprocess, SoftmaxPostprocess>
{
    using Pairs
        = TierList<Tier<128, 4>, // Mixtral 8x7B (topK=2), Qwen2-MoE (topK=4), Arctic (topK=2), DBRX (topK=4), GPT-OSS
            Tier<128, 8>,        // DeepSeek-V2-Lite (topK=6), Mixtral 8x22B (topK=2)
            Tier<160, 8>,        // Qwen3-Coder-480B
            Tier<256, 8>,        // Mistral Large 3 (topK=8)
            Tier<256, 16>,       // Models with 256 experts and topK 9..16
            Tier<512, 8>,        // Various 512-expert models
            Tier<512, 16>,       // Various 512-expert models with high topK
            Tier<512, 22>,       // Nemotron Super V3 (512 experts, topK=22)
            Tier<576, 8>,        // Customized model with 576 experts
            Tier<2048, 32>       // Large-expert fallback
            >;
};

/// Sigmoid + SumNormalize (SigmoidRenorm, Cohere): single config.
template <>
struct PolicyTraits<SigmoidPreprocess, SumNormalizePostprocess>
{
    using Pairs = TierList<Tier<128, 8>>;
};

/// SigmoidBias + ScaledSumNormalize (DeepSeek nGroup≤1 / MiniMax2 / Kimi-K2 / Nemotron SuperV3).
template <>
struct PolicyTraits<SigmoidBiasPreprocess, ScaledSumNormalizePostprocess>
{
    using Pairs = TierList<Tier<128, 8>, // Small expert counts (≤128 experts, e.g. DeepSeek-V2-Lite)
        Tier<256, 8>,                    // MiniMax M2 (256 experts, topK=6)
        Tier<384, 8>,                    // Kimi K2 (384 experts)
        Tier<512, 8>,                    // DeepSeek nGroup≤1 (256 experts → E512 fallback)
        Tier<512, 22>                    // Nemotron Super V3 (512 experts, topK=22, nGroup≤1)
        >;
};

/// Softmax + SumNormalize (RenormalizeNaive): no specialization needed.
/// At runtime, RenormalizeNaive is always converted to the Renormalize path
/// (None + Softmax) by the runner, so this policy is never dispatched.
/// If it ever is, the default PolicyTraits provides broad fallback coverage.

/// None + None (fallback for unknown preprocess/postprocess in LAUNCH_ROUTING_CUSTOM).
template <>
struct PolicyTraits<NoOpPreprocess, NoOpPostprocess>
{
    using Pairs = TierList<Tier<128, 8>>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// EXAMPLE: Custom ExpertSelectPolicy that bypasses the PreProc→topK→PostProc pattern.
//
// To enable it:
//   1. Uncomment the struct and PolicyTraits below.
//   2. Add an enum value (e.g., RoutingPreprocessType::FirstK) in RoutingKernel.h.
//   3. Add a branch in LAUNCH_ROUTING_CUSTOM that calls LAUNCH_ROUTING_FOR_EXPERT_SELECT.
//   4. Set the enum in runner.cu for the desired routing method type.
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
struct FirstKExpertSelect
{
    template <typename InputT> using BaseType = float;
    template <typename OutputT> struct Params { void set(routingCustom::Data const&) {} };

    template <typename DataType, typename InputType, int VecSize, int K, typename KP>
    __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const&,
        DataType (&warpTopKScore)[K], int32_t (&warpTopKExpertIdx)[K], int32_t const laneIdx,
        int32_t const, int32_t topK, InputType const*, KP const&)
    {
        if (laneIdx < topK)
        {
            warpTopKExpertIdx[laneIdx] = laneIdx;
            warpTopKScore[laneIdx] = static_cast<DataType>(1.0f / topK);
        }
    }
};

template <> struct PolicyTraits<FirstKExpertSelect, void>
{
    using Pairs = TierList<Tier<128, 8>>;
};
*/

////////////////////////////////////////////////////////////////////////////////////////////////////
// GENERIC DISPATCH MACROS
//
// These macros are fixed infrastructure — they never need editing when adding new
// policies or changing tier support.  All configuration lives in PolicyTraits above.
//
// The dispatch iterates PolicyTraits::Pairs (a TierList) via dispatchTierPairs.
// A generic lambda captures the kernel name (macro requirement) and receives
// (expert, topK) as compile-time integral_constants.
////////////////////////////////////////////////////////////////////////////////////////////////////

// Generic per-policy dispatch.  Iterates PolicyTraits<PreProc, PostProc>::Pairs,
// picking the first (expert, topK) pair that covers the runtime values.
//
// IMPORTANT: numThreads is clamped to at least min(MaxNumExperts, 1024) from the dispatched tier.
// Many routing kernels derive their internal NumThreadsBlock from MaxNumExperts and use it for
// grid-stride addressing, initArr strides, and cub::BlockScan.  If the caller's numThreads
// (typically getMaxNumExperts(mNumExperts)) is smaller than the tier's MaxNumExperts, the kernel
// would compute wrong indices, skip initialization, and corrupt memory.  The max() below
// guarantees the launch thread count always matches or exceeds the kernel's NumThreadsBlock:
//   - "derive from tier" kernels: numThreadsHist < MaxNumExperts → bumped to MaxNumExperts  ✓
//   - "fixed 1024" kernels (cluster): numThreads=1024 ≥ MaxNumExperts → unchanged            ✓
#define LAUNCH_ROUTING_FOR_POLICY(                                                                                     \
    data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, PreProc, PostProc)                              \
    [&](auto pt_tag_)                                                                                                  \
    {                                                                                                                  \
        using Pairs_ = typename decltype(pt_tag_)::Pairs;                                                              \
        bool dispatched_ = dispatchTierPairs(static_cast<Pairs_*>(nullptr), data,                                      \
            [&](auto eTag_, auto kTag_)                                                                                \
            {                                                                                                          \
                constexpr int tierMaxExp_ = decltype(eTag_)::value;                                                    \
                constexpr int tierThreads_ = tierMaxExp_ <= 1024 ? tierMaxExp_ : 1024;                                 \
                int const effectiveThreads_ = std::max(static_cast<int>(numThreads), tierThreads_);                    \
                LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, effectiveThreads_, smemSize, stream, \
                    PreProc, PostProc, decltype(eTag_)::value, decltype(kTag_)::value);                                \
            });                                                                                                        \
        if (!dispatched_)                                                                                              \
        {                                                                                                              \
            TLLM_LOG_ERROR("No tier covers numExperts=%d topK=%d", data.mNumExperts, data.mTopK);                      \
        }                                                                                                              \
    }(PolicyTraits<PreProc, PostProc>{})

////////////////////////////////////////////////////////////////////////////////////////////////////
// CUSTOM EXPERT SELECT DISPATCH
////////////////////////////////////////////////////////////////////////////////////////////////////

// Generic dispatch for custom ExpertSelectPolicy. PolicyTraits key is <ExpertSelect, void>.
// Same numThreads clamping as LAUNCH_ROUTING_FOR_POLICY — see comment above.
#define LAUNCH_ROUTING_FOR_EXPERT_SELECT(                                                                              \
    data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, ExpertSelect)                                   \
    [&](auto pt_tag_)                                                                                                  \
    {                                                                                                                  \
        using Pairs_ = typename decltype(pt_tag_)::Pairs;                                                              \
        bool dispatched_ = dispatchTierPairs(static_cast<Pairs_*>(nullptr), data,                                      \
            [&](auto eTag_, auto kTag_)                                                                                \
            {                                                                                                          \
                constexpr int tierMaxExp_ = decltype(eTag_)::value;                                                    \
                constexpr int tierThreads_ = tierMaxExp_ <= 1024 ? tierMaxExp_ : 1024;                                 \
                int const effectiveThreads_ = std::max(static_cast<int>(numThreads), tierThreads_);                    \
                LAUNCH_ROUTING_WITH_EXPERT_SELECT(data, coopLaunch, kernel, numBlocks, effectiveThreads_, smemSize,    \
                    stream, ExpertSelect, decltype(eTag_)::value, decltype(kTag_)::value);                             \
            });                                                                                                        \
        if (!dispatched_)                                                                                              \
        {                                                                                                              \
            TLLM_LOG_ERROR("No tier covers numExperts=%d topK=%d", data.mNumExperts, data.mTopK);                      \
        }                                                                                                              \
    }(PolicyTraits<ExpertSelect, void>{})

////////////////////////////////////////////////////////////////////////////////////////////////////
// PUBLIC DISPATCH MACROS
//
// These are the only macros that call sites use.
////////////////////////////////////////////////////////////////////////////////////////////////////

// Lightweight dispatch for utility kernels (histogram, init-counts, offsets) that do NOT use
// expert select policies, InputT, or MaxNumTopExperts.
// - Always uses NoOp expert select (no policy dispatch).
// - Always uses a fixed NumTop8Experts (no topK-tier dispatch).
// - Dispatches only on expert tiers.
// This is intentionally NOT routed through LAUNCH_ROUTING_FOR_POLICY to avoid
// instantiating all topK tiers — utility kernels don't use topK at all.
#define LAUNCH_ROUTING_CUSTOM_NO_POLICY(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream)             \
    if (data.mNumExperts <= NumExperts128Experts)                                                                      \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream,                \
            NoOpPreprocess, NoOpPostprocess, NumExperts128Experts, NumTop8Experts);                                    \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumExperts160Experts)                                                                 \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream,                \
            NoOpPreprocess, NoOpPostprocess, NumExperts160Experts, NumTop8Experts);                                    \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumExperts256Experts)                                                                 \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream,                \
            NoOpPreprocess, NoOpPostprocess, NumExperts256Experts, NumTop8Experts);                                    \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumExperts384Experts)                                                                 \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream,                \
            NoOpPreprocess, NoOpPostprocess, NumExperts384Experts, NumTop8Experts);                                    \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumExperts512Experts)                                                                 \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream,                \
            NoOpPreprocess, NoOpPostprocess, NumExperts512Experts, NumTop8Experts);                                    \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumExperts576Experts)                                                                 \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream,                \
            NoOpPreprocess, NoOpPostprocess, NumExperts576Experts, NumTop8Experts);                                    \
    }                                                                                                                  \
    else if (data.mNumExperts <= MaxSupportedExperts)                                                                  \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream,                \
            NoOpPreprocess, NoOpPostprocess, MaxSupportedExperts, NumTop8Experts);                                     \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_LOG_ERROR("Unsupported numExperts");                                                                      \
    }

// Top-level dispatch: maps runtime preprocess/postprocess enums to compile-time policy types,
// then delegates to LAUNCH_ROUTING_FOR_POLICY which reads PolicyTraits for tier support.
// Use this ONLY for kernels that call ExpertSelectPolicy::apply (block, cluster, histogramScores).
#define LAUNCH_ROUTING_CUSTOM(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream)                       \
    if (data.mPreprocessType == RoutingPreprocessType::SigmoidBias)                                                    \
    {                                                                                                                  \
        LAUNCH_ROUTING_FOR_POLICY(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream,                   \
            SigmoidBiasPreprocess, ScaledSumNormalizePostprocess);                                                     \
    }                                                                                                                  \
    else if (data.mPreprocessType == RoutingPreprocessType::Sigmoid)                                                   \
    {                                                                                                                  \
        LAUNCH_ROUTING_FOR_POLICY(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream,                   \
            SigmoidPreprocess, SumNormalizePostprocess);                                                               \
    }                                                                                                                  \
    else if (data.mPreprocessType == RoutingPreprocessType::Softmax                                                    \
        && data.mPostprocessType == RoutingPostprocessType::None)                                                      \
    {                                                                                                                  \
        LAUNCH_ROUTING_FOR_POLICY(                                                                                     \
            data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, SoftmaxPreprocess, NoOpPostprocess);    \
    }                                                                                                                  \
    else if (data.mPreprocessType == RoutingPreprocessType::Softmax)                                                   \
    {                                                                                                                  \
        LAUNCH_ROUTING_FOR_POLICY(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream,                   \
            SoftmaxPreprocess, SumNormalizePostprocess);                                                               \
    }                                                                                                                  \
    else if (data.mPostprocessType == RoutingPostprocessType::Softmax)                                                 \
    {                                                                                                                  \
        LAUNCH_ROUTING_FOR_POLICY(                                                                                     \
            data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, NoOpPreprocess, SoftmaxPostprocess);    \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        LAUNCH_ROUTING_FOR_POLICY(                                                                                     \
            data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, NoOpPreprocess, NoOpPostprocess);       \
    }
/* ── Example: hooking a custom ExpertSelectPolicy into the dispatch ──────── *
 *                                                                            *
 *  else if (data.mPreprocessType == RoutingPreprocessType::FirstK)           *
 *  {                                                                         *
 *      LAUNCH_ROUTING_FOR_EXPERT_SELECT(data, coopLaunch, kernel, numBlocks, *
 *          numThreads, smemSize, stream, FirstKExpertSelect);                *
 *  }                                                                         *
 * ────────────────────────────────────────────────────────────────────────── */

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingCustom
} // namespace moe::dev::routing
