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
class RoutingRenormalizeKernelTest : public RoutingKernelTest<T>
{

protected:
    using RoutingKernelTest<T>::mSeed;
    using RoutingKernelTest<T>::mStream;
    using RoutingKernelTest<T>::mBufferManager;
    using typename RoutingKernelTest<T>::PackedType;

private:
    // private methods
    void computeTopKExperts(RoutingKernelTestParam const& param) override
    {
        for (int it = 0; it < param.numTokens; ++it)
        {
            PackedFloat expWeightsIdx[param.numExperts];
            PackedFloat expIdx[param.topK];
            float sum = float{0.0f};
            float maxScore = -std::numeric_limits<float>::infinity();
            for (int ie = 0; ie < param.numExperts; ++ie)
            {
                float score;
                int16_t newIdx = static_cast<int16_t>(ie);
                score = static_cast<float>(bufferCast<T>(*this->mPtrScoresHost)[it * param.numExperts + ie]);

                if (param.doSoftmaxBeforeTopK && score > maxScore)
                {
                    maxScore = score;
                }

                PackedFloat si{static_cast<float>(score), newIdx};
                expWeightsIdx[ie] = si;
            }

            if (param.doSoftmaxBeforeTopK)
            {
                // Run softmax before topk
                for (int ie = 0; ie < param.numExperts; ++ie)
                {
                    expWeightsIdx[ie].score
                        = static_cast<float>(std::exp(static_cast<float>(expWeightsIdx[ie].score) - maxScore));
                    sum += expWeightsIdx[ie].score;
                }

                for (int ie = 0; ie < param.numExperts; ++ie)
                {
                    float score = static_cast<float>(expWeightsIdx[ie].score);
                    score /= sum;
                    expWeightsIdx[ie].score = static_cast<float>(score);
                }
            }

            // Calculate the top-k scores and indices
            std::partial_sort_copy(expWeightsIdx, expWeightsIdx + param.numExperts, expIdx, expIdx + param.topK, comp);

            if (param.doSoftmaxBeforeTopK)
            {
                // Normalize the value after the topk
                if (param.normTopkProb)
                {
                    float sum = float{0.0f};
                    for (int ie = 0; ie < param.topK; ++ie)
                    {
                        sum += static_cast<float>(expIdx[ie].score);
                    }
                    for (int ie = 0; ie < param.topK; ++ie)
                    {
                        float score = static_cast<float>(expIdx[ie].score);
                        score /= sum;
                        expIdx[ie].score = static_cast<float>(score);
                    }
                }
            }
            else
            {
                // Perform softmax after topk
                float sum = float{0.0f};
                float maxScore = -std::numeric_limits<float>::infinity();
                float score;
                for (int ie = 0; ie < param.topK; ++ie)
                {
                    score = static_cast<float>(expIdx[ie].score);
                    maxScore = score >= maxScore ? score : maxScore;
                }
                for (int ie = 0; ie < param.topK; ++ie)
                {
                    score = static_cast<float>(expIdx[ie].score) - maxScore;
                    score = std::exp(score);
                    sum += score;
                }
                for (int ie = 0; ie < param.topK; ++ie)
                {
                    score = static_cast<float>(expIdx[ie].score) - maxScore;
                    score = static_cast<float>(std::exp(score));
                    score /= sum;
                    expIdx[ie].score = static_cast<float>(score);
                }
            }

            // convert back to io_dtype and store the topk expert results in hostData.mPtrTopKPacked
            for (int ie = 0; ie < param.topK; ++ie)
            {
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

    void allocateBuffers(RoutingKernelTestParam const& param) override
    {
        RoutingKernelTest<T>::allocateBuffers(param);
        int64_t scoresSize = param.numTokens * param.numExperts;
        this->mPtrScoresHost = mBufferManager->pinned(ITensor::makeShape({scoresSize}), TRTDataType<T>::value);
        this->mPtrScoresDevice = mBufferManager->gpu(ITensor::makeShape({scoresSize}), TRTDataType<T>::value);
    }

    template <typename RoutingData>
    void setParams(RoutingKernelTestParam const& param, RoutingData& routingData)
    {
        RoutingKernelTest<T>::setCommonParams(param, routingData);

        if (sizeof(T) == 4)
        {
            routingData.mDtypeExpW = btg::Dtype::Fp32;
        }
        else
        {
            routingData.mDtypeExpW = btg::Dtype::Bfloat16;
        }

        // Special case for RenormalizeNaive
        routingData.mDoSoftmaxBeforeTopK = param.routingMethod == RoutingMethodType::RenormalizeNaive;
        routingData.mNormTopkProb = param.routingMethod == RoutingMethodType::RenormalizeNaive;

        if (param.useTopKAsInput)
        {
            routingData.mPtrTopKIds = bufferCast<int32_t>(*this->mPtrTopKIdsDevice);
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
        moe::dev::routing::routingRenormalize::Data routingData;
        setParams(param, routingData);
        moe::dev::routing::routingRenormalize::run(routingData, mStream->get());
    }
};

TYPED_TEST_SUITE(RoutingRenormalizeKernelTest, FloatAndBf16Types);

TYPED_TEST(RoutingRenormalizeKernelTest, BlockLevelParallelization)
{
    RoutingKernelTestParam param(RoutingMethodType::Renormalize, /*numTokens=*/4,
        /*numExperts=*/128, /*topK=*/8,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0, /*tileTokensDim=*/256,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/false,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 9);
    this->runTest(param);
};

TYPED_TEST(RoutingRenormalizeKernelTest, BlockLevelParallelizationWithExpertParallelization)
{
    RoutingKernelTestParam param(RoutingMethodType::Renormalize, /*numTokens=*/4,
        /*numExperts=*/128, /*topK=*/8,
        /*expertParallelization=*/2, /*expertParallelizationId=*/1, /*tileTokensDim=*/192,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/true,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 9);
    this->runTest(param);
};

TYPED_TEST(RoutingRenormalizeKernelTest, ClusterLevelParallelization)
{
    RoutingKernelTestParam param(RoutingMethodType::Renormalize, /*numTokens=*/10,
        /*numExperts=*/128, /*topK=*/8,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0, /*tileTokensDim=*/192,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/false,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 9);
    this->runTest(param);
};

TYPED_TEST(RoutingRenormalizeKernelTest, ClusterLevelParallelizationWithExpertParallelization)
{
    RoutingKernelTestParam param(RoutingMethodType::Renormalize, /*numTokens=*/100,
        /*numExperts=*/128, /*topK=*/8,
        /*expertParallelization=*/2, /*expertParallelizationId=*/1, /*tileTokensDim=*/256,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/false,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 9);
    this->runTest(param);
};

TYPED_TEST(RoutingRenormalizeKernelTest, ClusterLevelParallelizationWithRenormalizeNaive)
{
    RoutingKernelTestParam param(RoutingMethodType::RenormalizeNaive, /*numTokens=*/10,
        /*numExperts=*/128, /*topK=*/8,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0, /*tileTokensDim=*/256,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/false,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 9);
    this->runTest(param);
};

TYPED_TEST(RoutingRenormalizeKernelTest, DeviceLevelParallelization)
{
    RoutingKernelTestParam param(RoutingMethodType::Renormalize, /*numTokens=*/1000,
        /*numExperts=*/128, /*topK=*/8,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0, /*tileTokensDim=*/8,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/false,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 8);
    this->runTest(param);
};

TYPED_TEST(RoutingRenormalizeKernelTest, ClusterLevelParallelizationTop4)
{
    RoutingKernelTestParam param(RoutingMethodType::Renormalize, /*numTokens=*/200,
        /*numExperts=*/128, /*topK=*/4,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0, /*tileTokensDim=*/8,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/true,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 9);
    this->runTest(param);
};

TYPED_TEST(RoutingRenormalizeKernelTest, ClusterLevelParallelizationWithExpertParallelizationTop4)
{
    RoutingKernelTestParam param(RoutingMethodType::Renormalize, /*numTokens=*/256,
        /*numExperts=*/128, /*topK=*/4,
        /*expertParallelization=*/2, /*expertParallelizationId=*/1, /*tileTokensDim=*/8,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/true,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 9);
    this->runTest(param);
};

TYPED_TEST(RoutingRenormalizeKernelTest, ClusterLevelParallelizationWithRenormalizeNaiveTop4)
{
    RoutingKernelTestParam param(RoutingMethodType::RenormalizeNaive, /*numTokens=*/10,
        /*numExperts=*/128, /*topK=*/4,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0, /*tileTokensDim=*/8,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/true,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 9);
    this->runTest(param);
};

TYPED_TEST(RoutingRenormalizeKernelTest, DeviceLevelParallelizationTop4)
{
    RoutingKernelTestParam param(RoutingMethodType::Renormalize, /*numTokens=*/1000,
        /*numExperts=*/128, /*topK=*/4,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0, /*tileTokensDim=*/256,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/true,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 8);
    this->runTest(param);
};

TYPED_TEST(RoutingRenormalizeKernelTest, BlockLevelParallelizationLargeN)
{
    RoutingKernelTestParam param(RoutingMethodType::Renormalize, /*numTokens=*/4,
        /*numExperts=*/512, /*topK=*/10,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0, /*tileTokensDim=*/256,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/false,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 9);
    this->runTest(param);
};

TYPED_TEST(RoutingRenormalizeKernelTest, ClusterLevelParallelizationLargeN)
{
    RoutingKernelTestParam param(RoutingMethodType::Renormalize, /*numTokens=*/100,
        /*numExperts=*/512, /*topK=*/10,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0, /*tileTokensDim=*/256,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/false,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 9);
    this->runTest(param);
};

TYPED_TEST(RoutingRenormalizeKernelTest, DeviceLevelParallelizationLargeN)
{
    RoutingKernelTestParam param(RoutingMethodType::Renormalize, /*numTokens=*/1000,
        /*numExperts=*/512, /*topK=*/10,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0, /*tileTokensDim=*/256,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/false,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 8);
    this->runTest(param);
};
} // end namespace
