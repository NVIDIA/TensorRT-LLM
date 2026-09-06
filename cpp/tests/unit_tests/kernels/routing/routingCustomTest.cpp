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

#include "tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/routing/RoutingCustomSelection.h"
#include "tests/unit_tests/kernels/routing/routingTest.h"

#include <vector>

namespace tk = tensorrt_llm::kernels;
namespace btg = batchedGemm::trtllm::gen;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::tests::kernels::routing;

namespace
{

template <typename T>
class RoutingCustomKernelTest : public RoutingKernelTest<T>
{

protected:
    using RoutingKernelTest<T>::mSeed;
    using RoutingKernelTest<T>::mStream;
    using RoutingKernelTest<T>::mBufferManager;
    using typename RoutingKernelTest<T>::PackedType;

private:
    // Routing bias buffers (used by SigmoidBias preprocess)
    TensorPtr mPtrRoutingBiasHost;
    TensorPtr mPtrRoutingBiasDevice;

    static float sigmoid_accurate(float x)
    {
        return 0.5f * std::tanh(0.5f * x) + 0.5f;
    }

    // Reference implementation for all policy combinations:
    //   1. Softmax     + NoOp               (Default: softmax before topK, raw scores)
    //   2. NoOp        + Softmax             (Renormalize: topK first, then softmax)
    //   3. Softmax     + SumNormalize        (RenormalizeNaive path)
    //   4. SigmoidBias + ScaledSumNormalize  (DeepSeek-style path)
    //   5. Sigmoid     + SumNormalize        (SigmoidRenorm path)
    //   6. NoOp        + NoOp               (raw topK, no transformation)
    void computeTopKExperts(RoutingKernelTestParam const& param) override
    {
        for (int it = 0; it < param.numTokens; ++it)
        {
            std::vector<PackedFloat> expWeightsIdx(param.numExperts);
            std::vector<PackedFloat> expIdx(param.topK);

            // Per-expert sigmoid scores — only populated for SigmoidBias preprocess.
            std::vector<float> sigmoidScores(param.numExperts, 0.f);

            // --- Read raw scores and apply preprocess ---
            for (int ie = 0; ie < param.numExperts; ++ie)
            {
                float score = static_cast<float>(bufferCast<T>(*this->mPtrScoresHost)[it * param.numExperts + ie]);

                if (param.preprocessType == RoutingPreprocessType::Sigmoid)
                {
                    float sig = sigmoid_accurate(score);
                    score = ie < param.numExperts ? sig : -std::numeric_limits<float>::infinity();
                }
                else if (param.preprocessType == RoutingPreprocessType::SigmoidBias)
                {
                    float sig = sigmoid_accurate(score);
                    sigmoidScores[ie] = sig;
                    float bias = static_cast<float>(bufferCast<T>(*mPtrRoutingBiasHost)[ie]);
                    score = sig + bias;
                }

                expWeightsIdx[ie] = PackedFloat{score, static_cast<int16_t>(ie)};
            }

            // Apply softmax preprocess (over all experts) when requested
            if (param.preprocessType == RoutingPreprocessType::Softmax)
            {
                float maxScore = -std::numeric_limits<float>::infinity();
                for (int ie = 0; ie < param.numExperts; ++ie)
                {
                    maxScore = std::max(maxScore, expWeightsIdx[ie].score);
                }
                float sum = 0.f;
                for (int ie = 0; ie < param.numExperts; ++ie)
                {
                    expWeightsIdx[ie].score = std::exp(expWeightsIdx[ie].score - maxScore);
                    sum += expWeightsIdx[ie].score;
                }
                for (int ie = 0; ie < param.numExperts; ++ie)
                {
                    expWeightsIdx[ie].score /= sum;
                }
            }

            // --- TopK selection ---
            std::partial_sort_copy(expWeightsIdx.begin(), expWeightsIdx.end(), expIdx.begin(), expIdx.end(), comp);

            // --- Apply postprocess ---
            if (param.postprocessType == RoutingPostprocessType::Softmax)
            {
                // Softmax over top-K scores
                float maxScore = -std::numeric_limits<float>::infinity();
                for (uint32_t ie = 0; ie < param.topK; ++ie)
                {
                    maxScore = std::max(maxScore, expIdx[ie].score);
                }
                float sum = 0.f;
                for (uint32_t ie = 0; ie < param.topK; ++ie)
                {
                    sum += std::exp(expIdx[ie].score - maxScore);
                }
                for (uint32_t ie = 0; ie < param.topK; ++ie)
                {
                    expIdx[ie].score = std::exp(expIdx[ie].score - maxScore) / sum;
                }
            }
            else if (param.postprocessType == RoutingPostprocessType::SumNormalize)
            {
                // SumNormalize: divide top-K scores by their sum
                if (param.normTopkProb)
                {
                    float sum = 0.f;
                    for (uint32_t ie = 0; ie < param.topK; ++ie)
                    {
                        sum += expIdx[ie].score;
                    }
                    for (uint32_t ie = 0; ie < param.topK; ++ie)
                    {
                        expIdx[ie].score /= sum;
                    }
                }
            }
            else if (param.postprocessType == RoutingPostprocessType::ScaledSumNormalize)
            {
                // Recover sigmoid scores, renormalize by their sum, and scale
                float sumSigmoid = 0.f;
                for (uint32_t ie = 0; ie < param.topK; ++ie)
                {
                    sumSigmoid += sigmoidScores[expIdx[ie].idx];
                }
                for (uint32_t ie = 0; ie < param.topK; ++ie)
                {
                    expIdx[ie].score = sigmoidScores[expIdx[ie].idx] * param.routedScalingFactor / sumSigmoid;
                }
            }
            // For NoOp postprocess: scores are left unchanged.

            // --- Store results ---
            for (uint32_t ie = 0; ie < param.topK; ++ie)
            {
                // Set invalid topk indices for the first half of the topk
                if (param.hasInvalidTopKInput && ie < param.topK / 2 + 1)
                {
                    expIdx[ie].idx = static_cast<int16_t>(param.invalidExpertIdValue);
                }

                PackedType si{static_cast<T>(expIdx[ie].score), expIdx[ie].idx};
                reinterpret_cast<PackedType*>(bufferCast<int8_t>(*this->mPtrTopKPackedHost))[it * param.topK + ie] = si;
                if (param.useTopKAsInput)
                {
                    bufferCast<int32_t>(*this->mPtrTopKIdsHost)[it * param.topK + ie]
                        = static_cast<int32_t>(expIdx[ie].idx);
                    bufferCast<T>(*this->mPtrTopKWeightsHost)[it * param.topK + ie] = static_cast<T>(expIdx[ie].score);
                }
                else if (param.getExpWeights)
                {
                    bufferCast<T>(*this->mPtrTopKWeightsHost)[it * param.topK + ie] = static_cast<T>(expIdx[ie].score);
                }
            }
        }
    }

protected:
    void allocateBuffers(RoutingKernelTestParam const& param) override
    {
        RoutingKernelTest<T>::allocateBuffers(param);
        int64_t scoresSize = param.numTokens * param.numExperts;
        this->mPtrScoresHost = mBufferManager->pinned(ITensor::makeShape({scoresSize}), TRTDataType<T>::value);
        this->mPtrScoresDevice = mBufferManager->gpu(ITensor::makeShape({scoresSize}), TRTDataType<T>::value);

        // Allocate routing bias buffers when needed
        if (param.preprocessType == RoutingPreprocessType::SigmoidBias)
        {
            mPtrRoutingBiasHost = mBufferManager->pinned(ITensor::makeShape({param.numExperts}), TRTDataType<T>::value);
            mPtrRoutingBiasDevice = mBufferManager->gpu(ITensor::makeShape({param.numExperts}), TRTDataType<T>::value);
        }
    }

    void setupBuffers(RoutingKernelTestParam const& param) override
    {
        RoutingKernelTest<T>::setupBuffers(param);

        // Initialize routing bias with small random values
        if (param.preprocessType == RoutingPreprocessType::SigmoidBias)
        {
            T* biasPtr = bufferCast<T>(*mPtrRoutingBiasHost);
            initData(biasPtr, param.numExperts, mSeed + 7);
            mBufferManager->copy(*mPtrRoutingBiasHost, *mPtrRoutingBiasDevice);
        }
    }

    template <typename RoutingData>
    void setParams(RoutingKernelTestParam const& param, RoutingData& routingData)
    {
        RoutingKernelTest<T>::setCommonParams(param, routingData);

        if (sizeof(T) == 4)
        {
            routingData.mDtypeOutput = btg::Dtype::Fp32;
            routingData.mDtypeInput = btg::Dtype::Fp32;
        }
        else
        {
            routingData.mDtypeOutput = btg::Dtype::Bfloat16;
            routingData.mDtypeInput = btg::Dtype::Bfloat16;
        }

        // Set policy types from test param (already derived by build())
        routingData.mPreprocessType = param.preprocessType;
        routingData.mPostprocessType = param.postprocessType;
        routingData.mNormTopkProb = param.normTopkProb;

        // Set routing bias and scale when using SigmoidBias preprocess
        if (param.preprocessType == RoutingPreprocessType::SigmoidBias)
        {
            routingData.mPtrRoutingBias = bufferCast<T>(*mPtrRoutingBiasDevice);
            // Bias dtype matches T (the test's type parameter)
            routingData.mDtypeBias = (sizeof(T) == 4) ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
            routingData.mRouteScale = param.routedScalingFactor;
        }

        if (param.useTopKAsInput)
        {
            routingData.mPtrTopKIds = bufferCast<int32_t>(*this->mPtrTopKIdsDevice);
            routingData.mPtrScores = nullptr;
        }
        else if (param.useTopKPackedAsInput)
        {
            // mPtrTopKPacked is already set by setCommonParams; just clear scores and topKIds
            routingData.mPtrTopKIds = nullptr;
            routingData.mPtrScores = nullptr;
        }
        else
        {
            routingData.mPtrTopKIds = nullptr;
            routingData.mPtrScores = bufferCast<T>(*this->mPtrScoresDevice);
        }
    }

    void callTestedFunction(
        RoutingKernelTestParam const& param, tensorrt_llm::runtime::ITensor::SharedPtr& workspaceDevice) override
    {
        moe::dev::routing::routingCustom::Data routingData;
        setParams(param, routingData);
        moe::dev::routing::routingCustom::run(routingData, mStream->get());
    }
};

TYPED_TEST_SUITE(RoutingCustomKernelTest, FloatAndBf16Types);

TYPED_TEST(RoutingCustomKernelTest, BlockLevelParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(4)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, BlockLevelParallelizationWithExpertParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(4)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withExpertParallelization(2, 1)
                     .withTileTokensDim(192)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, BlockLevelClassicBoundaryE512K16)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(4)
                     .withNumExperts(512)
                     .withTopK(16)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, BlockLevelCooperativeBoundaryE576K8)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(4)
                     .withNumExperts(576)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

// Tier 512 is the only tier whose launcher changes with the token count: one token keeps
// the cooperative kernel, two and up take the classic one. These two straddle that flip.
// Tier 576 has no such flip -- it is cooperative across the whole one-to-four range -- so
// it is covered once, above, at four tokens.
TYPED_TEST(RoutingCustomKernelTest, BlockLevelCooperativeBoundaryE512K16SingleToken)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1)
                     .withNumExperts(512)
                     .withTopK(16)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, BlockLevelClassicBoundaryE512K16TwoTokens)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(2)
                     .withNumExperts(512)
                     .withTopK(16)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

// SigmoidBias keeps the cooperative kernel at every tier, including the ones Renormalize
// gives up. Four tokens is already covered by DeepSeekNoGroupBlockLevel and
// MiniMax2BlockLevel; one token is not covered anywhere else in this file, and it is where
// the cooperative kernel has the least work to amortise its one-thread-per-expert shape.
TYPED_TEST(RoutingCustomKernelTest, BlockLevelSigmoidBiasBoundaryE512K16SingleToken)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::MiniMax2)
                     .withNumTokens(1)
                     .withNumExperts(512)
                     .withTopK(16)
                     .withTileTokensDim(256)
                     .withRoutedScalingFactor(2.5f)
                     .build();
    this->runTest(param);
};

// The boundary tests above validate outputs, which both launchers produce identically,
// so they cannot detect a change in which launcher is selected. This one pins the selection
// table itself. Tiers are the compile-time values queryDispatchedMaxExperts() returns, not
// raw expert counts.
TEST(RoutingCustomSelectionTest, CoopBlockKernelSelectionTable)
{
    using moe::dev::routing::RoutingPostprocessType;
    using moe::dev::routing::RoutingPreprocessType;
    using moe::dev::routing::routingCustom::prefersCoopBlockKernel;

    auto const renormalize = [](int32_t numTokens, int32_t tier)
    { return prefersCoopBlockKernel(RoutingPreprocessType::None, RoutingPostprocessType::Softmax, numTokens, tier); };

    // The tiers below are exactly the ones each policy's PolicyTraits::Pairs can produce,
    // so every cell here is a combination that queryDispatchedMaxExperts() can actually
    // return. Renormalize: 128, 160, 256, 512, 576, 2048.
    //
    // Classic through the 512 tier, cooperative from 576 up.
    for (int32_t numTokens : {2, 3, 4})
    {
        for (int32_t tier : {128, 160, 256, 512})
        {
            EXPECT_FALSE(renormalize(numTokens, tier)) << "tier " << tier << ", " << numTokens << " tokens";
        }
        EXPECT_TRUE(renormalize(numTokens, 576)) << numTokens << " tokens";
    }

    // At a single token the boundary moves down one tier: 512 stays cooperative.
    EXPECT_FALSE(renormalize(1, 128));
    EXPECT_FALSE(renormalize(1, 160));
    EXPECT_FALSE(renormalize(1, 256));
    EXPECT_TRUE(renormalize(1, 512));
    EXPECT_TRUE(renormalize(1, 576));

    // Per-expert preprocessing spills registers in the classic kernel well below 512
    // experts, so the lower bound must not apply to it. SigmoidBias tiers: 128, 256, 384,
    // 512, 1024. Sigmoid has a single tier.
    for (int32_t tier : {128, 256, 384, 512, 1024})
    {
        EXPECT_TRUE(prefersCoopBlockKernel(
            RoutingPreprocessType::SigmoidBias, RoutingPostprocessType::ScaledSumNormalize, 4, tier))
            << "tier " << tier;
    }
    EXPECT_TRUE(prefersCoopBlockKernel(RoutingPreprocessType::Sigmoid, RoutingPostprocessType::SumNormalize, 4, 128));

    // The None + None fallback shares a preprocess with Renormalize but was never measured,
    // so it keeps the cooperative kernel that the parent selected.
    EXPECT_TRUE(prefersCoopBlockKernel(RoutingPreprocessType::None, RoutingPostprocessType::None, 4, 128));

    // Above one CUDA block of experts the cooperative kernel cannot run at all.
    EXPECT_FALSE(renormalize(1, 2048));
    EXPECT_FALSE(prefersCoopBlockKernel(
        RoutingPreprocessType::SigmoidBias, RoutingPostprocessType::ScaledSumNormalize, 1, 2048));

    // Softmax-over-experts is not elementwise, so it never uses the cooperative kernel.
    EXPECT_FALSE(prefersCoopBlockKernel(RoutingPreprocessType::Softmax, RoutingPostprocessType::None, 1, 576));

    // The whole path is gated on the static-block token limit.
    EXPECT_TRUE(
        prefersCoopBlockKernel(RoutingPreprocessType::SigmoidBias, RoutingPostprocessType::ScaledSumNormalize, 4, 512));
    EXPECT_FALSE(
        prefersCoopBlockKernel(RoutingPreprocessType::SigmoidBias, RoutingPostprocessType::ScaledSumNormalize, 5, 512));
};

// TLLM_ROUTING_COOP_BLOCK_MIN_EXPERTS reaches the predicate as an argument, so the two
// directions of the escape hatch are asserted here rather than through the environment.
TEST(RoutingCustomSelectionTest, CoopBlockKernelMinNumExpertsOverride)
{
    using moe::dev::routing::RoutingPostprocessType;
    using moe::dev::routing::RoutingPreprocessType;
    using moe::dev::routing::routingCustom::prefersCoopBlockKernel;

    auto const renormalize = [](int32_t numTokens, int32_t tier, int32_t override)
    {
        return prefersCoopBlockKernel(
            RoutingPreprocessType::None, RoutingPostprocessType::Softmax, numTokens, tier, override);
    };

    // 0 restores the parent selection: every Renormalize tier the cooperative kernel can
    // run at all takes it again, at every token count in the static-block range.
    for (int32_t numTokens : {1, 2, 3, 4})
    {
        for (int32_t tier : {128, 160, 256, 512, 576})
        {
            EXPECT_TRUE(renormalize(numTokens, tier, 0)) << "tier " << tier << ", " << numTokens << " tokens";
        }
    }

    // A bound above every tier forces the classic kernel across the range.
    for (int32_t numTokens : {1, 2, 3, 4})
    {
        for (int32_t tier : {128, 160, 256, 512, 576})
        {
            EXPECT_FALSE(renormalize(numTokens, tier, 4096)) << "tier " << tier << ", " << numTokens << " tokens";
        }
    }

    // A negative value means unset and leaves the built-in bounds in place.
    EXPECT_FALSE(renormalize(4, 512, -1));
    EXPECT_TRUE(renormalize(1, 512, -1));

    // The hard constraints are not negotiable: the override cannot put more than one CUDA
    // block of experts, or more than four tokens, on the cooperative kernel.
    EXPECT_FALSE(renormalize(1, 2048, 0));
    EXPECT_FALSE(renormalize(5, 512, 0));

    // Policies outside the bound are untouched in both directions.
    EXPECT_TRUE(prefersCoopBlockKernel(
        RoutingPreprocessType::SigmoidBias, RoutingPostprocessType::ScaledSumNormalize, 4, 512, 4096));
};

TYPED_TEST(RoutingCustomKernelTest, BlockLevelParallelizationWithInvalidTopKInput)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(4)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withExpertParallelization(2, 1)
                     .withTileTokensDim(192)
                     .withUseTopKAsInput(true)
                     .withHasInvalidTopKInput(true)
                     .build();
    this->runTest(param);
};

// --- Tests for useTopKAsInput (mPtrTopKIds + mPtrTopKWeights as input) ---
// These test the runPostTopKPipeline path at block, cluster, and coop levels.

TYPED_TEST(RoutingCustomKernelTest, BlockLevelTopKAsInput)
{
    // Small token count -> single-block path in runPostTopKPipeline
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(4)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUseTopKAsInput(true)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ClusterLevelTopKAsInput)
{
    // Medium token count -> single-cluster path in runPostTopKPipeline
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(100)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUseTopKAsInput(true)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, CoopLevelTopKAsInput)
{
    // Large token count -> coop path in runPostTopKPipeline
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1000)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUseTopKAsInput(true)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ClusterLevelParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(10)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(192)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ClusterLevelParallelizationWithExpertParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(100)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withExpertParallelization(2, 1)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ClusterLevelParallelizationWithInvalidTopKInput)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(100)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withExpertParallelization(2, 1)
                     .withTileTokensDim(256)
                     .withUseTopKAsInput(true)
                     .withHasInvalidTopKInput(true)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ClusterLevelParallelizationWithRenormalizeNaive)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::RenormalizeNaive)
                     .withNumTokens(10)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

// --- Tests for useTopKPackedAsInput (mPtrTopKPacked without mPtrScores) ---
// These test the runPostTopKPipeline path for the packed input format.

TYPED_TEST(RoutingCustomKernelTest, BlockLevelTopKPackedAsInput)
{
    // Small token count -> single-block path in runPostTopKPipeline
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(4)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUseTopKPackedAsInput(true)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ClusterLevelTopKPackedAsInput)
{
    // Medium token count -> single-cluster path in runPostTopKPipeline
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(100)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUseTopKPackedAsInput(true)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, DeviceLevelTopKPackedAsInput)
{
    // Large token count -> coop or multi-kernel path in runPostTopKPipeline
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1000)
                     .withNumExperts(128)
                     .withTopK(4)
                     .withTileTokensDim(256)
                     .withUseTopKPackedAsInput(true)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, DeviceLevelParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1000)
                     .withNumExperts(512)
                     .withTopK(10)
                     .withTileTokensDim(8)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ClusterLevelParallelizationTop4)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(200)
                     .withNumExperts(128)
                     .withTopK(4)
                     .withTileTokensDim(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ClusterLevelParallelizationWithInvalidTopKInputTop4)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(200)
                     .withNumExperts(128)
                     .withTopK(4)
                     .withTileTokensDim(8)
                     .withUseTopKAsInput(true)
                     .withHasInvalidTopKInput(true)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ClusterLevelParallelizationWithExpertParallelizationTop4)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(256)
                     .withNumExperts(128)
                     .withTopK(4)
                     .withExpertParallelization(2, 1)
                     .withTileTokensDim(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ClusterLevelParallelizationWithRenormalizeNaiveTop4)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::RenormalizeNaive)
                     .withNumTokens(10)
                     .withNumExperts(128)
                     .withTopK(4)
                     .withTileTokensDim(8)
                     .build();
    this->runTest(param);
};

// --- Tests for Default (Softmax + NoOp postprocess) ---

TYPED_TEST(RoutingCustomKernelTest, DefaultBlockLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Default)
                     .withNumTokens(4)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, DefaultClusterLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Default)
                     .withNumTokens(10)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, DefaultDeviceLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Default)
                     .withNumTokens(1000)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, DefaultWithExpertParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Default)
                     .withNumTokens(10)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withExpertParallelization(2, 1)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

// --- Tests for RenormalizeNaive at block and device levels ---

TYPED_TEST(RoutingCustomKernelTest, RenormalizeNaiveBlockLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::RenormalizeNaive)
                     .withNumTokens(4)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, RenormalizeNaiveDeviceLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::RenormalizeNaive)
                     .withNumTokens(1000)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, RenormalizeNaiveWithExpertParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::RenormalizeNaive)
                     .withNumTokens(10)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withExpertParallelization(2, 1)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, DeviceLevelParallelizationTop4)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1000)
                     .withNumExperts(128)
                     .withTopK(4)
                     .withTileTokensDim(256)
                     .withUseTopKAsInput(true)
                     .withHasInvalidTopKInput(true)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, BlockLevelParallelizationLargeN)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(4)
                     .withNumExperts(2048)
                     .withTopK(32)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ClusterLevelParallelizationLargeN)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(100)
                     .withNumExperts(2048)
                     .withTopK(32)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, DeviceLevelParallelizationLargeN)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1000)
                     .withNumExperts(2048)
                     .withTopK(32)
                     .withTileTokensDim(256)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, DeviceLevelParallelizationLargeNWithInvalidTopKInput)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1000)
                     .withNumExperts(2048)
                     .withTopK(32)
                     .withTileTokensDim(256)
                     .withUseTopKAsInput(true)
                     .withHasInvalidTopKInput(true)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests for invalid expertId = numExperts (instead of -1).
// Some frameworks use expertId == numExperts to mark unassigned slots.
// The kernel must handle this without illegal memory access.
// These tests exercise the block, cluster, and device-level paths with topKIds input
// where some expert IDs are set to numExperts.
////////////////////////////////////////////////////////////////////////////////////////////////////

TYPED_TEST(RoutingCustomKernelTest, BlockLevelInvalidExpertIdEqualsNumExperts)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(4)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUseTopKAsInput(true)
                     .withHasInvalidTopKInput(true)
                     .withInvalidExpertIdValue(128) // numExperts as invalid marker
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ClusterLevelInvalidExpertIdEqualsNumExperts)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(100)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUseTopKAsInput(true)
                     .withHasInvalidTopKInput(true)
                     .withInvalidExpertIdValue(128)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, DeviceLevelInvalidExpertIdEqualsNumExperts)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1000)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUseTopKAsInput(true)
                     .withHasInvalidTopKInput(true)
                     .withInvalidExpertIdValue(128)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

// Test with numExperts < MaxNumExperts (fall-through tier) — expertId=numExperts
// passes the `< MaxNumExperts` check but should still be treated as invalid.
TYPED_TEST(RoutingCustomKernelTest, BlockLevelInvalidExpertIdFallThroughTier)
{
    // numExperts=100 → dispatches to E128 tier (MaxNumExperts=128).
    // expertId=100 passes `100 < 128` but is invalid (only 0..99 are valid).
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(4)
                     .withNumExperts(100)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUseTopKAsInput(true)
                     .withHasInvalidTopKInput(true)
                     .withInvalidExpertIdValue(100) // numExperts as invalid marker
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ClusterLevelInvalidExpertIdFallThroughTier)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(100)
                     .withNumExperts(100)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUseTopKAsInput(true)
                     .withHasInvalidTopKInput(true)
                     .withInvalidExpertIdValue(100)
                     .build();
    this->runTest(param);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests for Renormalize with new expert/topK tiers (E160, E576, K22)
////////////////////////////////////////////////////////////////////////////////////////////////////

// --- E576 experts, K22 topK (exercises the new E576 and K22 tiers) ---

TYPED_TEST(RoutingCustomKernelTest, RenormalizeBlockLevelE576K22)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(4)
                     .withNumExperts(576)
                     .withTopK(22)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, RenormalizeClusterLevelE576K22)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(100)
                     .withNumExperts(576)
                     .withTopK(22)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, RenormalizeDeviceLevelE576K22)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1000)
                     .withNumExperts(576)
                     .withTopK(22)
                     .withTileTokensDim(256)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, RenormalizeDeviceLevelE576K22TopKAsInput)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1000)
                     .withNumExperts(576)
                     .withTopK(22)
                     .withTileTokensDim(256)
                     .withUseTopKAsInput(true)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

// --- E160 experts, K8 topK (exercises the new E160 tier) ---

TYPED_TEST(RoutingCustomKernelTest, RenormalizeBlockLevelE160)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(4)
                     .withNumExperts(160)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, RenormalizeClusterLevelE160)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(100)
                     .withNumExperts(160)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, RenormalizeDeviceLevelE160)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1000)
                     .withNumExperts(160)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, RenormalizeClusterLevelE160WithExpertParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(100)
                     .withNumExperts(160)
                     .withTopK(8)
                     .withExpertParallelization(2, 1)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests for 384-expert tier (Renormalize + SigmoidBias policies).
// 384 is in getMaxNumExperts() tiers but was previously missing from some PolicyTraits,
// causing thread-count mismatch bugs.  These tests cover block, cluster, and device paths.
////////////////////////////////////////////////////////////////////////////////////////////////////

// --- Renormalize with E384 (exercises Tier<384,8> in None+Softmax policy) ---

TYPED_TEST(RoutingCustomKernelTest, RenormalizeBlockLevelE384)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1)
                     .withNumExperts(384)
                     .withTopK(8)
                     .withTileTokensDim(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, RenormalizeClusterLevelE384)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(100)
                     .withNumExperts(384)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, RenormalizeDeviceLevelE384)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1000)
                     .withNumExperts(384)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, RenormalizeBlockLevelE384WithEP)
{
    // Mirrors the failing multi-GPU test: e384, topK=8, seq=1, EP=4
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1)
                     .withNumExperts(384)
                     .withTopK(8)
                     .withExpertParallelization(4, 1)
                     .withTileTokensDim(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, RenormalizeBlockLevelE384TopKAsInput)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1)
                     .withNumExperts(384)
                     .withTopK(8)
                     .withTileTokensDim(8)
                     .withUseTopKAsInput(true)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, RenormalizeBlockLevelE384TopKAsInputWithEP)
{
    // Mirrors the failing multi-GPU test with pre-computed topK: e384, topK=8, seq=1, EP=4
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1)
                     .withNumExperts(384)
                     .withTopK(8)
                     .withExpertParallelization(4, 1)
                     .withTileTokensDim(8)
                     .withUseTopKAsInput(true)
                     .build();
    this->runTest(param);
};

// --- SigmoidBias with E384 (exercises Tier<384,8> in SigmoidBias+ScaledSumNormalize policy) ---

TYPED_TEST(RoutingCustomKernelTest, SigmoidBiasBlockLevelE384)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withPreprocessType(RoutingPreprocessType::SigmoidBias)
                     .withPostprocessType(RoutingPostprocessType::ScaledSumNormalize)
                     .withRoutedScalingFactor(2.5f)
                     .withNumTokens(4)
                     .withNumExperts(384)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, SigmoidBiasClusterLevelE384)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withPreprocessType(RoutingPreprocessType::SigmoidBias)
                     .withPostprocessType(RoutingPostprocessType::ScaledSumNormalize)
                     .withRoutedScalingFactor(2.5f)
                     .withNumTokens(100)
                     .withNumExperts(384)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests for scores input + cooperative kernel path (scores→topK kernel + coop histogram+offsets).
// These verify the coop fast-path when input is raw mPtrScores (not pre-computed topK).
// Triggered when numTokens > cluster capacity (256) and within coop capacity.
// Requires SM90+ (coop kernel uses grid-sync).
////////////////////////////////////////////////////////////////////////////////////////////////////

TYPED_TEST(RoutingCustomKernelTest, ScoresCoopRenormalize)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1000)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ScoresCoopRenormalizeE160)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1000)
                     .withNumExperts(160)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ScoresCoopRenormalizeE256K4)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1000)
                     .withNumExperts(256)
                     .withTopK(4)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ScoresCoopRenormalizeE576K22)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(500)
                     .withNumExperts(576)
                     .withTopK(22)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ScoresCoopRenormalizeWithExpertParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(1000)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withExpertParallelization(2, 1)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ScoresCoopDefault)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Default)
                     .withNumTokens(1000)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ScoresCoopRenormalizeNaive)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::RenormalizeNaive)
                     .withNumTokens(1000)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, ScoresCoopSigmoidBias)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withPreprocessType(RoutingPreprocessType::SigmoidBias)
                     .withPostprocessType(RoutingPostprocessType::ScaledSumNormalize)
                     .withRoutedScalingFactor(2.5f)
                     .withNumTokens(1000)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests for SigmoidBias + ScaledSumNormalize (DeepSeek-style routing via routingCustom)
////////////////////////////////////////////////////////////////////////////////////////////////////

// SigmoidBias PolicyTraits: only E512 × K8.

TYPED_TEST(RoutingCustomKernelTest, SigmoidBiasBlockLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withPreprocessType(RoutingPreprocessType::SigmoidBias)
                     .withPostprocessType(RoutingPostprocessType::ScaledSumNormalize)
                     .withRoutedScalingFactor(2.5f)
                     .withNumTokens(4)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, SigmoidBiasClusterLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withPreprocessType(RoutingPreprocessType::SigmoidBias)
                     .withPostprocessType(RoutingPostprocessType::ScaledSumNormalize)
                     .withRoutedScalingFactor(2.5f)
                     .withNumTokens(100)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, SigmoidBiasDeviceLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withPreprocessType(RoutingPreprocessType::SigmoidBias)
                     .withPostprocessType(RoutingPostprocessType::ScaledSumNormalize)
                     .withRoutedScalingFactor(2.5f)
                     .withNumTokens(1000)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withTileTokensDim(8)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, SigmoidBiasWithExpertParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withPreprocessType(RoutingPreprocessType::SigmoidBias)
                     .withPostprocessType(RoutingPostprocessType::ScaledSumNormalize)
                     .withRoutedScalingFactor(2.5f)
                     .withNumTokens(10)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withExpertParallelization(2, 1)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests for MiniMax2 (SigmoidBias + ScaledSumNormalize with routeScale=1.0)
////////////////////////////////////////////////////////////////////////////////////////////////////

// MiniMax2 PolicyTraits: same as SigmoidBias, only E512 × K8.

TYPED_TEST(RoutingCustomKernelTest, MiniMax2BlockLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::MiniMax2)
                     .withNumTokens(4)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, MiniMax2ClusterLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::MiniMax2)
                     .withNumTokens(100)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, MiniMax2DeviceLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::MiniMax2)
                     .withNumTokens(1000)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withTileTokensDim(8)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, MiniMax2WithExpertParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::MiniMax2)
                     .withNumTokens(10)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withExpertParallelization(2, 1)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests for mixed input/bias dtypes (SigmoidBias with float32 scores + bfloat16 bias, and vice versa).
// These test the loadScalar + mDtypeBias dispatch for cross-dtype bias reading.
// The test allocates bias in the "opposite" dtype from T (e.g. T=float → OtherT=bfloat16).
//
// NOTE: This test only verifies that the kernel runs without crashing or producing garbage.
// Strict numerical comparison against the T-typed host reference is intentionally omitted
// because the T→OtherT bias conversion introduces small rounding differences that can
// flip the ranking of experts with nearly identical sigmoid(score)+bias values, leading
// to different topK selections and completely different expert weights/permutations.
// Full numerical correctness for same-dtype bias is covered by MiniMax2BlockLevel/ClusterLevel,
// and for fp32 bias with matched reference by ClusterLevelWithFloat32Bias (DeepSeek).
////////////////////////////////////////////////////////////////////////////////////////////////////

TYPED_TEST(RoutingCustomKernelTest, MiniMax2MixedBiasDtype)
{
    using OtherT = std::conditional_t<std::is_same_v<TypeParam, float>, __nv_bfloat16, float>;

    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::MiniMax2)
                     .withNumTokens(100)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();

    this->allocateBuffers(param);
    this->setupBuffers(param);

    // Allocate bias in the "opposite" dtype from T
    auto otherBiasHost
        = this->mBufferManager->pinned(ITensor::makeShape({param.numExperts}), TRTDataType<OtherT>::value);
    auto otherBiasDevice
        = this->mBufferManager->gpu(ITensor::makeShape({param.numExperts}), TRTDataType<OtherT>::value);
    auto otherBiasPtr = bufferCast<OtherT>(*otherBiasHost);
    for (int i = 0; i < param.numExperts; i++)
    {
        otherBiasPtr[i] = static_cast<OtherT>(0.01f * (i % 100));
    }
    this->mBufferManager->copy(*otherBiasHost, *otherBiasDevice);
    this->mStream->synchronize();

    // Setup routing data with mixed dtypes
    moe::dev::routing::routingCustom::Data routingData;
    this->setCommonParams(param, routingData);
    routingData.mDtypeOutput = (sizeof(TypeParam) == 4) ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
    routingData.mDtypeInput = routingData.mDtypeOutput;
    routingData.mPreprocessType = param.preprocessType;
    routingData.mPostprocessType = param.postprocessType;
    routingData.mNormTopkProb = param.normTopkProb;
    routingData.mPtrScores = bufferCast<TypeParam>(*this->mPtrScoresDevice);
    routingData.mPtrRoutingBias = bufferCast<OtherT>(*otherBiasDevice);
    routingData.mDtypeBias = (sizeof(OtherT) == 4) ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
    routingData.mRouteScale = param.routedScalingFactor;

    // Mixed-precision bias test: verifies the kernel correctly reads OtherT-typed bias
    // via loadScalar + mDtypeBias without crashing or producing garbage.
    // Strict numerical comparison against the T-typed host reference is not possible
    // because the precision difference in bias values can change expert selection.
    moe::dev::routing::routingCustom::run(routingData, this->mStream->get());
    this->mStream->synchronize();
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test that non-multiple-of-4 expert counts work correctly.
// The routing kernel uses compile-time MaxNumExperts (always a multiple of 32) and
// masks invalid experts with numExperts < MaxNumExperts. The % 4 constraint was removed
// because it is not a routing kernel requirement.
////////////////////////////////////////////////////////////////////////////////////////////////////

TYPED_TEST(RoutingCustomKernelTest, BlockLevelNonMultipleOf4Experts7)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(4)
                     .withNumExperts(7)
                     .withTopK(2)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, BlockLevelNonMultipleOf4Experts13)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(4)
                     .withNumExperts(13)
                     .withTopK(3)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests for DeepSeek nGroup=1 via routingCustom (SigmoidBias + ScaledSumNormalize with routeScale != 1.0)
// When nGroup <= 1, DeepSeek routing is equivalent to SigmoidBias + ScaledSumNormalize,
// and production code routes through routingCustom (not routingDeepSeek).
////////////////////////////////////////////////////////////////////////////////////////////////////

// DeepSeek nGroup=1: uses SigmoidBias policy (E512 × K8).

TYPED_TEST(RoutingCustomKernelTest, DeepSeekNoGroupBlockLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::MiniMax2)
                     .withNumTokens(4)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withRoutedScalingFactor(2.5f)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, DeepSeekNoGroupClusterLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::MiniMax2)
                     .withNumTokens(100)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withRoutedScalingFactor(2.5f)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, DeepSeekNoGroupDeviceLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::MiniMax2)
                     .withNumTokens(1000)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withRoutedScalingFactor(2.5f)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests for SigmoidRenorm (Sigmoid + SumNormalize)
////////////////////////////////////////////////////////////////////////////////////////////////////

TYPED_TEST(RoutingCustomKernelTest, SigmoidRenormBlockLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::SigmoidRenorm)
                     .withNumTokens(4)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, SigmoidRenormClusterLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::SigmoidRenorm)
                     .withNumTokens(100)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

// SigmoidRenorm PolicyTraits: only E128 × K8.

TYPED_TEST(RoutingCustomKernelTest, SigmoidRenormDeviceLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::SigmoidRenorm)
                     .withNumTokens(1000)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(8)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, SigmoidRenormWithExpertParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::SigmoidRenorm)
                     .withNumTokens(10)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withExpertParallelization(2, 1)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests for NoOp + NoOp (raw topK, no score transformation)
////////////////////////////////////////////////////////////////////////////////////////////////////

TYPED_TEST(RoutingCustomKernelTest, NoOpBlockLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withPreprocessType(RoutingPreprocessType::None)
                     .withPostprocessType(RoutingPostprocessType::None)
                     .withNumTokens(4)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, NoOpClusterLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withPreprocessType(RoutingPreprocessType::None)
                     .withPostprocessType(RoutingPostprocessType::None)
                     .withNumTokens(100)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

// NoOp PolicyTraits: only E128 × K8.

TYPED_TEST(RoutingCustomKernelTest, NoOpDeviceLevel)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withPreprocessType(RoutingPreprocessType::None)
                     .withPostprocessType(RoutingPostprocessType::None)
                     .withNumTokens(1000)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(8)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, NoOpWithExpertParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withPreprocessType(RoutingPreprocessType::None)
                     .withPostprocessType(RoutingPostprocessType::None)
                     .withNumTokens(10)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withExpertParallelization(2, 1)
                     .withTileTokensDim(256)
                     .build();
    this->runTest(param);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Dynamic-block kernel tests (5-16 tokens, ≤512 experts)
////////////////////////////////////////////////////////////////////////////////////////////////////

TYPED_TEST(RoutingCustomKernelTest, DynBlockBasic)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(8)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUsePdl(true)
                     .withGetExpWeights(true)
                     .withRequiredComputeCapability(9)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, DynBlockMaxTokens)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(16)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUsePdl(true)
                     .withGetExpWeights(true)
                     .withRequiredComputeCapability(9)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, DynBlockWithExpertParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(12)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withExpertParallelization(2, 1)
                     .withTileTokensDim(192)
                     .withUsePdl(true)
                     .withGetExpWeights(true)
                     .withRequiredComputeCapability(9)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, DynBlockWithTopKAsInput)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(8)
                     .withNumExperts(128)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUsePdl(true)
                     .withGetExpWeights(true)
                     .withUseTopKAsInput(true)
                     .withRequiredComputeCapability(9)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, DynBlockWithInvalidTopKInput)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::Renormalize)
                     .withNumTokens(10)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withExpertParallelization(2, 0)
                     .withTileTokensDim(256)
                     .withUsePdl(true)
                     .withGetExpWeights(true)
                     .withUseTopKAsInput(true)
                     .withHasInvalidTopKInput(true)
                     .withRequiredComputeCapability(9)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingCustomKernelTest, DynBlockWithRenormalizeNaive)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::RenormalizeNaive)
                     .withNumTokens(16)
                     .withNumExperts(512)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUsePdl(true)
                     .withGetExpWeights(true)
                     .withRequiredComputeCapability(9)
                     .build();
    this->runTest(param);
};

} // namespace
