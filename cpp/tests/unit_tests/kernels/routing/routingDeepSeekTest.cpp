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

#include "tests/unit_tests/kernels/routing/routingTest.h"

namespace tk = tensorrt_llm::kernels;
namespace btg = batchedGemm::trtllm::gen;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::tests::kernels::routing;

namespace
{

template <typename T>
class RoutingDeepSeekKernelTest : public RoutingKernelTest<T>
{

protected:
    using RoutingKernelTest<T>::mSeed;
    using RoutingKernelTest<T>::mStream;
    using RoutingKernelTest<T>::mBufferManager;
    using typename RoutingKernelTest<T>::PackedType;

    TensorPtr mPtrRoutingBiasHost;
    TensorPtr mPtrRoutingBiasDevice;

private:
    // private methods
    static inline float sigmoid_accurate(float x)
    {
        return 0.5f * tanhf(0.5f * x) + 0.5f;
    }

    void computeTopKExperts(RoutingKernelTestParam const& param) override
    {
        // note that for invalid scores, we simply use a negative value:
        // they work well even with the compacted format used in topK, and
        // sigmoid / bias activated scores cannot be negative
        static constexpr float invalidScoreFloat = float{-INFINITY};
        const T invalidScore = T{invalidScoreFloat};

        float scoreSigmoid[param.numExperts];
        for (int it = 0; it < param.numTokens; ++it)
        {
            PackedFloat expWeightsIdx[param.numExperts];
            // Get the sigmoid score
            float sum = float{0.0f};
            for (int ie = 0; ie < param.numExperts; ++ie)
            {
                float score;
                int16_t newIdx = static_cast<int16_t>(ie);
                score = static_cast<float>(bufferCast<float>(*this->mPtrScoresHost)[it * param.numExperts + ie]);
                score = sigmoid_accurate(score);
                scoreSigmoid[ie] = score;

                auto biasVal = bufferCast<T>(*this->mPtrRoutingBiasHost)[ie];
                auto scoreBias = score + static_cast<float>(biasVal);

                PackedFloat si{static_cast<float>(scoreBias), newIdx};
                expWeightsIdx[ie] = si;
            }

            PackedFloat finalTopkExperts[param.topK];
            if (param.nGroup != 0)
            {
                // Calculate the group score
                int32_t expertsPerGroup = param.numExperts / param.nGroup;
                PackedFloat groupScoresCandidate[param.nGroup][2];
                PackedFloat groupScores[param.nGroup];
                for (int ig = 0; ig < param.nGroup; ++ig)
                {
                    std::partial_sort_copy(expWeightsIdx + ig * expertsPerGroup,
                        expWeightsIdx + (ig + 1) * expertsPerGroup, groupScoresCandidate[ig],
                        groupScoresCandidate[ig] + 2, comp);
                    PackedFloat si{
                        static_cast<float>(groupScoresCandidate[ig][0].score + groupScoresCandidate[ig][1].score),
                        static_cast<int16_t>(ig)};
                    groupScores[ig] = si;
                }

                // Get the topkGroup group score
                PackedFloat topGroupScores[param.topkGroup];
                std::partial_sort_copy(
                    groupScores, groupScores + param.nGroup, topGroupScores, topGroupScores + param.topkGroup, comp);

                // Prepare the data for the final topk experts selection
                PackedFloat topkExpertsCandidate[param.topkGroup * expertsPerGroup];

                for (int ig = 0; ig < param.topkGroup; ++ig)
                {
                    for (int ie = 0; ie < expertsPerGroup; ++ie)
                    {
                        topkExpertsCandidate[ig * expertsPerGroup + ie]
                            = expWeightsIdx[topGroupScores[ig].idx * expertsPerGroup + ie];
                    }
                }

                std::partial_sort_copy(topkExpertsCandidate, topkExpertsCandidate + param.topkGroup * expertsPerGroup,
                    finalTopkExperts, finalTopkExperts + param.topK, comp);
            }
            else
            {
                std::partial_sort_copy(expWeightsIdx, expWeightsIdx + param.numExperts, finalTopkExperts,
                    finalTopkExperts + param.topK, comp);
            }

            // Normalize the score
            float sumScore = 0.0f;
            for (int ie = 0; ie < param.topK; ++ie)
            {
                sumScore += scoreSigmoid[finalTopkExperts[ie].idx];
            }
            for (int ie = 0; ie < param.topK; ++ie)
            {
                float finalScore = scoreSigmoid[finalTopkExperts[ie].idx] * param.routedScalingFactor / sumScore;
                finalTopkExperts[ie].score = finalScore;
            }

            // Convert back to io_dtype and store the topk expert results in hostData.mPtrTopKPacked
            for (int ie = 0; ie < param.topK; ++ie)
            {
                if (param.useTopKAsInput)
                {
                    bufferCast<int32_t>(*this->mPtrTopKIdsHost)[it * param.topK + ie]
                        = static_cast<int32_t>(finalTopkExperts[ie].idx);
                    bufferCast<T>(*this->mPtrTopKWeightsHost)[it * param.topK + ie]
                        = static_cast<T>(finalTopkExperts[ie].score);
                }
                else if (param.getExpWeights)
                {
                    bufferCast<T>(*this->mPtrTopKWeightsHost)[it * param.topK + ie]
                        = static_cast<T>(finalTopkExperts[ie].score);
                }

                PackedType si{static_cast<T>(finalTopkExperts[ie].score), finalTopkExperts[ie].idx};
                reinterpret_cast<PackedType*>(bufferCast<int8_t>(*this->mPtrTopKPackedHost))[it * param.topK + ie] = si;
            }
        }
    }

protected:
    void allocateBuffers(RoutingKernelTestParam const& param)
    {
        RoutingKernelTest<T>::allocateBuffers(param);
        int64_t scoresSize = param.numTokens * param.numExperts;
        this->mPtrScoresHost = mBufferManager->pinned(ITensor::makeShape({scoresSize}), nvinfer1::DataType::kFLOAT);
        this->mPtrScoresDevice = mBufferManager->gpu(ITensor::makeShape({scoresSize}), nvinfer1::DataType::kFLOAT);

        this->mPtrRoutingBiasHost
            = mBufferManager->pinned(ITensor::makeShape({param.numExperts}), TRTDataType<T>::value);
        this->mPtrRoutingBiasDevice
            = mBufferManager->gpu(ITensor::makeShape({param.numExperts}), TRTDataType<T>::value);
    }

    void setupBuffers(RoutingKernelTestParam const& param) override
    {
        float* scoresHostPtr = bufferCast<float>(*this->mPtrScoresHost);
        initData(scoresHostPtr, param.numTokens * param.numExperts, mSeed);
        mBufferManager->copy(*this->mPtrScoresHost, *this->mPtrScoresDevice);

        T* biasHostPtr = bufferCast<T>(*mPtrRoutingBiasHost);
        initData(biasHostPtr, param.numExperts, mSeed);
        mBufferManager->copy(*mPtrRoutingBiasHost, *mPtrRoutingBiasDevice);
    }

    template <typename RoutingData>
    void setParams(RoutingKernelTestParam const& param, RoutingData& routingData)
    {
        RoutingKernelTest<T>::setCommonParams(param, routingData);
        routingData.mDtypeOutput = btg::Dtype::Bfloat16;

        routingData.mPtrRoutingBias = bufferCast<T>(*this->mPtrRoutingBiasDevice);
        // Bias dtype matches T (the test's type parameter)
        routingData.mDtypeBias = (sizeof(T) == 4) ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

        routingData.mNumExpertGroups = param.nGroup;
        routingData.mNumLimitedGroups = param.topkGroup;
        routingData.mRouteScale = param.routedScalingFactor;
        routingData.mUseRoutingSoftmax = false;

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
            routingData.mPtrScores = bufferCast<float>(*this->mPtrScoresDevice);
        }
    }

    void callTestedFunction(
        RoutingKernelTestParam const& param, tensorrt_llm::runtime::ITensor::SharedPtr& workspaceDevice) override
    {
        moe::dev::routing::routingDeepSeek::Data routingData;
        setParams(param, routingData);
        moe::dev::routing::routingDeepSeek::run(routingData, mStream->get());
    }
};

TYPED_TEST_SUITE(RoutingDeepSeekKernelTest, Bf16Types);

TYPED_TEST(RoutingDeepSeekKernelTest, ClusterLevelParallelization32)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(4)
                     .withNumExperts(32)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withNGroup(8)
                     .withTopkGroup(4)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, ClusterLevelParallelization72)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(4)
                     .withNumExperts(72)
                     .withTopK(6)
                     .withTileTokensDim(256)
                     .withNGroup(1)
                     .withTopkGroup(1)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, ClusterLevelParallelization384)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(4)
                     .withNumExperts(384)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withNGroup(1)
                     .withTopkGroup(1)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, ClusterLevelParallelization512)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(4)
                     .withNumExperts(512)
                     .withTopK(22)
                     .withTileTokensDim(256)
                     .withNGroup(1)
                     .withTopkGroup(1)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, ClusterLevelParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(1024)
                     .withNumExperts(256)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withNGroup(8)
                     .withTopkGroup(4)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, BlockLevelTopKAsInput)
{
    // Small token count -> single-block path in runPostTopKPipeline
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(4)
                     .withNumExperts(256)
                     .withTopK(8)
                     .withTileTokensDim(192)
                     .withUseTopKAsInput(true)
                     .withNGroup(8)
                     .withTopkGroup(4)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, ClusterLevelParallelizationWithTopKAsInput)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(1024)
                     .withNumExperts(256)
                     .withTopK(8)
                     .withTileTokensDim(192)
                     .withUseTopKAsInput(true)
                     .withHasInvalidTopKInput(true)
                     .withNGroup(8)
                     .withTopkGroup(4)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, ClusterLevelParallelizationWithTopKAsInput384)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(1024)
                     .withNumExperts(384)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUseTopKAsInput(true)
                     .withNGroup(1)
                     .withTopkGroup(1)
                     .build();
    this->runTest(param);
};

// --- Tests for useTopKPackedAsInput (mPtrTopKPacked without mPtrScores) ---
// These test the runPostTopKPipeline path for the packed input format.

TYPED_TEST(RoutingDeepSeekKernelTest, BlockLevelTopKPackedAsInput)
{
    // Small token count -> single-block path in runPostTopKPipeline
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(4)
                     .withNumExperts(256)
                     .withTopK(8)
                     .withTileTokensDim(192)
                     .withUseTopKPackedAsInput(true)
                     .withNGroup(8)
                     .withTopkGroup(4)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, ClusterLevelTopKPackedAsInput)
{
    // Medium token count -> single-cluster path in runPostTopKPipeline
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(100)
                     .withNumExperts(256)
                     .withTopK(8)
                     .withTileTokensDim(192)
                     .withUseTopKPackedAsInput(true)
                     .withNGroup(8)
                     .withTopkGroup(4)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, DeviceLevelTopKPackedAsInput)
{
    // Large token count -> coop or multi-kernel path in runPostTopKPipeline
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(2048)
                     .withNumExperts(256)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUseTopKPackedAsInput(true)
                     .withNGroup(8)
                     .withTopkGroup(4)
                     .withRequiredComputeCapability(8)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, ClusterLevelParallelizationWithExpertParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(100)
                     .withNumExperts(256)
                     .withTopK(8)
                     .withExpertParallelization(2, 1)
                     .withTileTokensDim(192)
                     .withNGroup(8)
                     .withTopkGroup(4)
                     .build();
    this->runTest(param);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test DeepSeek main kernel with float32 bias (T=bf16 for scores output, but bias is float32).
// This exercises the loadScalar path with mismatched bias dtype.
////////////////////////////////////////////////////////////////////////////////////////////////////

TYPED_TEST(RoutingDeepSeekKernelTest, ClusterLevelWithFloat32Bias)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(100)
                     .withNumExperts(256)
                     .withTopK(8)
                     .withTileTokensDim(192)
                     .withNGroup(8)
                     .withTopkGroup(4)
                     .build();

    this->allocateBuffers(param);

    // Override: allocate bias as float32 instead of T (bf16)
    auto float32BiasHost
        = this->mBufferManager->pinned(ITensor::makeShape({param.numExperts}), nvinfer1::DataType::kFLOAT);
    auto float32BiasDevice
        = this->mBufferManager->gpu(ITensor::makeShape({param.numExperts}), nvinfer1::DataType::kFLOAT);
    auto biasPtr = bufferCast<float>(*float32BiasHost);
    for (int i = 0; i < param.numExperts; i++)
    {
        biasPtr[i] = 0.01f * (i % 100);
    }
    this->mBufferManager->copy(*float32BiasHost, *float32BiasDevice);

    // Setup normal buffers (scores, etc.)
    float* scoresHostPtr = bufferCast<float>(*this->mPtrScoresHost);
    initData(scoresHostPtr, param.numTokens * param.numExperts, 42);
    this->mBufferManager->copy(*this->mPtrScoresHost, *this->mPtrScoresDevice);
    this->mStream->synchronize();

    // Setup routing data with float32 bias
    moe::dev::routing::routingDeepSeek::Data routingData;
    this->setCommonParams(param, routingData);
    routingData.mDtypeOutput = btg::Dtype::Bfloat16;
    routingData.mPtrScores = bufferCast<float>(*this->mPtrScoresDevice);
    routingData.mPtrRoutingBias = bufferCast<float>(*float32BiasDevice);
    routingData.mDtypeBias = btg::Dtype::Fp32; // Float32 bias with bf16 output
    routingData.mNumExpertGroups = param.nGroup;
    routingData.mNumLimitedGroups = param.topkGroup;
    routingData.mRouteScale = param.routedScalingFactor;
    routingData.mUseRoutingSoftmax = false;

    // Run kernel — verifies it doesn't crash with float32 bias
    moe::dev::routing::routingDeepSeek::run(routingData, this->mStream->get());
    this->mStream->synchronize();
};

TYPED_TEST(RoutingDeepSeekKernelTest, CooperativeLevelParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(1030)
                     .withNumExperts(256)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withNGroup(8)
                     .withTopkGroup(4)
                     .withRequiredComputeCapability(10)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, CooperativeLevelParallelization384)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(1030)
                     .withNumExperts(384)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withNGroup(1)
                     .withTopkGroup(1)
                     .withRequiredComputeCapability(10)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, CooperativeLevelParallelization512)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(1030)
                     .withNumExperts(512)
                     .withTopK(22)
                     .withTileTokensDim(256)
                     .withNGroup(1)
                     .withTopkGroup(1)
                     .withRequiredComputeCapability(10)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, DeviceLevelParallelization)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(20300)
                     .withNumExperts(256)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUseTopKAsInput(true)
                     .withHasInvalidTopKInput(true)
                     .withNGroup(8)
                     .withTopkGroup(4)
                     .withRequiredComputeCapability(10)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, DeviceLevelParallelization384)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(20300)
                     .withNumExperts(384)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withNGroup(1)
                     .withTopkGroup(1)
                     .withRequiredComputeCapability(10)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, ClusterLevelParallelizationTop2)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(10)
                     .withNumExperts(256)
                     .withTopK(2)
                     .withTileTokensDim(256)
                     .withNGroup(8)
                     .withTopkGroup(4)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, ClusterLevelParallelizationWithExpertParallelizationTop2)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(100)
                     .withNumExperts(256)
                     .withTopK(2)
                     .withExpertParallelization(2, 1)
                     .withTileTokensDim(192)
                     .withNGroup(8)
                     .withTopkGroup(4)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, CooperativeLevelParallelizationTop2)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(1030)
                     .withNumExperts(256)
                     .withTopK(2)
                     .withTileTokensDim(256)
                     .withNGroup(8)
                     .withTopkGroup(4)
                     .withRequiredComputeCapability(10)
                     .build();
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, CooperativeLevelParallelizationTop8)
{
    auto param = RoutingKernelTestParam()
                     .withRoutingMethod(RoutingMethodType::DeepSeekV3)
                     .withNumTokens(1030)
                     .withNumExperts(32)
                     .withTopK(8)
                     .withTileTokensDim(256)
                     .withUseTopKAsInput(true)
                     .withHasInvalidTopKInput(true)
                     .withNGroup(8)
                     .withTopkGroup(4)
                     .withRequiredComputeCapability(10)
                     .build();
    this->runTest(param);
};
} // namespace
