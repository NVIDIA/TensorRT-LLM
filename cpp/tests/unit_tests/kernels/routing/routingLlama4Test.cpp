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
class RoutingLlama4KernelTest : public RoutingKernelTest<T>
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

                PackedFloat si{static_cast<float>(score), newIdx};
                expWeightsIdx[ie] = si;
            }

            // Calculate the top-k scores and indices
            std::partial_sort_copy(expWeightsIdx, expWeightsIdx + param.numExperts, expIdx, expIdx + param.topK,
                [](PackedFloat const& a, PackedFloat const& b)
                {
                    return (
                        (a.score > b.score) || (a.score == b.score && a.idx < b.idx)); //@TODO: check if this is correct
                });

            // Apply sigmoid to the top-k scores
            for (int ie = 0; ie < param.topK; ++ie)
            {
                auto finalScore = 1.F / (1.F + std::exp(-expIdx[ie].score));
                expIdx[ie].score = static_cast<float>(finalScore);
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
        routingData.mDtypeExpW = btg::Dtype::Bfloat16;

        routingData.mPtrTopKPacked = reinterpret_cast<PackedType*>(bufferCast<int8_t>(*this->mPtrTopKPackedDevice));
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
        moe::dev::routing::routingLlama4::Data routingData;
        setParams(param, routingData);
        moe::dev::routing::routingLlama4::run(routingData, mStream->get());
    }
};

TYPED_TEST_SUITE(RoutingLlama4KernelTest, Bf16Types);

TYPED_TEST(RoutingLlama4KernelTest, WarpLevelParallelization)
{
    RoutingKernelTestParam param(RoutingMethodType::Llama4, /*numTokens=*/3,
        /*numExperts=*/128, /*topK=*/1,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0, /*tileTokensDim=*/8,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/false,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 0.0f,
        /*requiredComputeCapability*/ 8);
    this->runTest(param);
};

TYPED_TEST(RoutingLlama4KernelTest, ClusterLevelParallelization)
{
    RoutingKernelTestParam param(RoutingMethodType::Llama4, /*numTokens=*/10,
        /*numExperts=*/128, /*topK=*/1,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0, /*tileTokensDim=*/8,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/false,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 9);
    this->runTest(param);
};

TYPED_TEST(RoutingLlama4KernelTest, DeviceLevelParallelization)
{
    RoutingKernelTestParam param(RoutingMethodType::Llama4, /*numTokens=*/300,
        /*numExperts=*/128, /*topK=*/1,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0, /*tileTokensDim=*/8,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/false,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 8);
    this->runTest(param);
};

TYPED_TEST(RoutingLlama4KernelTest, WarpLevelParallelizationTopKAsInput)
{
    RoutingKernelTestParam param(RoutingMethodType::Llama4, /*numTokens=*/3,
        /*numExperts=*/128, /*topK=*/1,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0, /*tileTokensDim=*/8,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/true,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 0.0f,
        /*requiredComputeCapability*/ 8);
    this->runTest(param);
};

TYPED_TEST(RoutingLlama4KernelTest, ClusterLevelParallelizationTopKAsInput)
{
    RoutingKernelTestParam param(RoutingMethodType::Llama4, /*numTokens=*/10,
        /*numExperts=*/128, /*topK=*/1,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0, /*tileTokensDim=*/8,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/true,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 9);
    this->runTest(param);
};

TYPED_TEST(RoutingLlama4KernelTest, DeviceLevelParallelizationTopKAsInput)
{
    RoutingKernelTestParam param(RoutingMethodType::Llama4, /*numTokens=*/300,
        /*numExperts=*/128, /*topK=*/1,
        /*expertParallelization=*/1, /*expertParallelizationId=*/0, /*tileTokensDim=*/8,
        /*paddingLog2=*/3, /*localExpertsStrideLog2=*/0,
        /*usePdl=*/true, /*getExpWeights=*/true, /*useTopKAsInput=*/true,
        /*nGroup*/ 0, /*topkGroup*/ 0, /*routedScalingFactor*/ 1.0f, /*requiredComputeCapability*/ 8);
    this->runTest(param);
};

} // namespace
