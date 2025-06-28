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

    // Add this variable because the definition of mPtrExpertIdx is "int32_t*" for Deepseek
    //@TODO: remove this variable after refactoring
    TensorPtr mPtrDeepseekExpertIdxHost;
    TensorPtr mPtrDeepseekExpertIdxDevice;

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
        static constexpr float invalidScoreFloat = -1.F;
        const T invalidScore = T{invalidScoreFloat};
        int32_t* expIdxHostPtr = bufferCast<int32_t>(*mPtrDeepseekExpertIdxHost);

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

            // Calculate the group score
            int32_t expertsPerGroup = param.numExperts / param.nGroup;
            PackedFloat groupScoresCandidate[param.nGroup][2];
            PackedFloat groupScores[param.nGroup];
            for (int ig = 0; ig < param.nGroup; ++ig)
            {
                std::partial_sort_copy(expWeightsIdx + ig * expertsPerGroup, expWeightsIdx + (ig + 1) * expertsPerGroup,
                    groupScoresCandidate[ig], groupScoresCandidate[ig] + 2, comp);
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
            PackedFloat finalTopkExperts[param.topK];

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

            // Convert back to io_dtype and store the topk expert results in hostData.mPtrExpertIdx
            for (int ie = 0; ie < param.topK; ++ie)
            {
                expIdxHostPtr[it * param.topK + ie] = static_cast<int32_t>(finalTopkExperts[ie].idx);
                if (param.getExpWeights)
                {
                    bufferCast<T>(*this->mPtrExpertWeightsHost)[it * param.topK + ie]
                        = static_cast<T>(finalTopkExperts[ie].score);
                }
                PackedType si{static_cast<T>(finalTopkExperts[ie].score), finalTopkExperts[ie].idx};
                reinterpret_cast<PackedType*>(bufferCast<int8_t>(*this->mPtrExpertIdxHost))[it * param.topK + ie] = si;
            }
        }
    }

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

        this->mPtrDeepseekExpertIdxHost
            = mBufferManager->pinned(ITensor::makeShape({param.numTokens * param.topK}), TRTDataType<int32_t>::value);
        this->mPtrDeepseekExpertIdxDevice
            = mBufferManager->gpu(ITensor::makeShape({param.numTokens * param.topK}), TRTDataType<int32_t>::value);
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
        routingData.mDtypeExpW = btg::Dtype::Bfloat16;
        routingData.mPtrScores = bufferCast<float>(*this->mPtrScoresDevice);
        routingData.mPtrRoutingBias = bufferCast<T>(*this->mPtrRoutingBiasDevice);
        //@todo: remove this line after refactoring
        routingData.mPtrExpertIdx = bufferCast<int32_t>(*this->mPtrDeepseekExpertIdxDevice);

        routingData.mNumExpertGroups = param.nGroup;
        routingData.mNumLimitedGroups = param.topkGroup;
        routingData.mRouteScale = param.routedScalingFactor;
        routingData.mUseRoutingSoftmax = false;
    }

    void callTestedFunction(
        RoutingKernelTestParam const& param, tensorrt_llm::runtime::ITensor::SharedPtr& workspaceDevice) override
    {
        moe::dev::routing::Data routingData;
        setParams(param, routingData);
        moe::dev::routing::run(routingData, mStream->get());
    }

    void verifyExpertRoutingIndices(RoutingKernelTestParam const& param)
    {
        // for permuted index, there is non-determinism, thus we check set-equality
        // for this, we go over every expert and retrieve the tokens routed to it
        // we then get the associated indexes and check set equality
        auto const expandedIdxToPermutedIdxHost
            = mBufferManager->copyFrom(*this->mPtrExpandedIdxToPermutedIdxDevice, MemoryType::kCPU);
        auto const hostExpToPermTest = bufferCast<int32_t>(*expandedIdxToPermutedIdxHost);

        auto const permutedIdxToTokenIdxHost
            = mBufferManager->copyFrom(*this->mPtrPermutedIdxToTokenIdxDevice, MemoryType::kCPU);
        auto const hostPermToTokTest = bufferCast<int32_t>(*permutedIdxToTokenIdxHost);
        mStream->synchronize();

        int32_t* expIdxToPermHostptr = bufferCast<int32_t>(*this->mPtrExpandedIdxToPermutedIdxHost);
        int32_t* expIdxHostPtr = bufferCast<int32_t>(*this->mPtrDeepseekExpertIdxHost);

        for (int ie = 0; ie < param.numExperts; ++ie)
        {
            std::set<int32_t> permutedIdx, permutedIdxTest;
            std::set<int32_t> tokenIdx, tokenIdxTest;
            auto localExpertIdx = ie - param.localExpertsStartIdx;
            auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < param.numLocalExperts
                && (localExpertIdx & param.localExpertsStrideLog2) == 0;

            for (int it = 0; it < param.numTokens * param.topK; ++it)
            {
                if (expIdxHostPtr[it] == ie)
                {
                    int const permIdx = isLocalExpert ? expIdxToPermHostptr[it] : int32_t{-1};
                    permutedIdx.insert(permIdx);
                    if (isLocalExpert)
                    {
                        tokenIdx.insert(it / param.topK);
                    }

                    int const permIdxTest = hostExpToPermTest[it];
                    permutedIdxTest.insert(permIdxTest);
                    if (isLocalExpert)
                    {
                        tokenIdxTest.insert(hostPermToTokTest[permIdxTest]);
                    }
                }
            }
            EXPECT_EQ(checkSetEqual(ie, permutedIdx, permutedIdxTest, "permuted idx"), true);
            EXPECT_EQ(checkSetEqual(ie, tokenIdx, tokenIdxTest, "token idx"), true);
        }
    }
};

TYPED_TEST_SUITE(RoutingDeepSeekKernelTest, Bf16Types);

TYPED_TEST(RoutingDeepSeekKernelTest, ClusterLevelParallelization)
{
    RoutingKernelTestParam param(RoutingMethodType::DeepSeekV3, /*numTokens=*/10,
        /*numExperts=*/128, /*topK=*/8,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true,
        /*nGroup*/ 8, /*topkGroup*/ 4, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 9);
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, ClusterLevelParallelizationWithExpertParallelization)
{
    RoutingKernelTestParam param(RoutingMethodType::DeepSeekV3, /*numTokens=*/100,
        /*numExperts=*/128, /*topK=*/8,
        /*expertParallelization=*/2, /*expertParallelizationId=*/1,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true,
        /*nGroup*/ 8, /*topkGroup*/ 4, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 9);
    this->runTest(param);
};

TYPED_TEST(RoutingDeepSeekKernelTest, CooperativeLevelParallelization)
{
    RoutingKernelTestParam param(RoutingMethodType::DeepSeekV3, /*numTokens=*/1030,
        /*numExperts=*/128, /*topK=*/8,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true,
        /*nGroup*/ 8, /*topkGroup*/ 4, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 10);
    this->runTest(param);
};

} // namespace
