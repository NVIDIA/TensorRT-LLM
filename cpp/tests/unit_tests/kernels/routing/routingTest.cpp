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

namespace tensorrt_llm::tests::kernels::routing
{

template <typename T>
void RoutingKernelTest<T>::SetUp()
{
    mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
    mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);

    auto const device = tc::getDevice();
    cudaGetDeviceProperties(&mDeviceProp, device);
}

template <typename T>
void RoutingKernelTest<T>::TearDown()
{
}

template <typename T>
void RoutingKernelTest<T>::allocateBuffers(RoutingKernelTestParam const& param)
{
    auto const numTokens = param.numTokens;
    auto const numExperts = param.numExperts;
    auto const topK = param.topK;
    // auto const paddingLog2 = param.paddingLog2;
    auto const tileTokensDim = param.tileTokensDim;
    auto const localExpertsStartIdx = param.localExpertsStartIdx;
    auto const localExpertsStrideLog2 = param.localExpertsStrideLog2;
    auto const numLocalExperts = param.numLocalExperts;
    auto const usePdl = param.usePdl;
    auto const doSoftmaxBeforeTopK = param.doSoftmaxBeforeTopK;
    auto const normTopkProb = param.normTopkProb;
    auto const useTopKAsInput = param.useTopKAsInput;

    int64_t countsSize = 2 * numExperts;
    if (param.routingMethod == RoutingMethodType::DeepSeekV3)
    {
        countsSize = 2 * 256;
    }
    mPtrExpertCountsHost = mBufferManager->pinned(ITensor::makeShape({countsSize}), nvinfer1::DataType::kINT32);
    mPtrExpertCountsDevice = mBufferManager->gpu(ITensor::makeShape({countsSize}), nvinfer1::DataType::kINT32);

    int64_t permIdxSize = 1;
    mPtrPermutedIdxSizeHost = mBufferManager->pinned(ITensor::makeShape({permIdxSize}), nvinfer1::DataType::kINT32);
    mPtrPermutedIdxSizeDevice = mBufferManager->gpu(ITensor::makeShape({permIdxSize}), nvinfer1::DataType::kINT32);

    int64_t expIdxToPermIdxSize = numTokens * topK;
    mPtrExpandedIdxToPermutedIdxHost
        = mBufferManager->pinned(ITensor::makeShape({expIdxToPermIdxSize}), nvinfer1::DataType::kINT32);
    mPtrExpandedIdxToPermutedIdxDevice
        = mBufferManager->gpu(ITensor::makeShape({expIdxToPermIdxSize}), nvinfer1::DataType::kINT32);

    // int64_t permIdxToTokenIdxSize = (numTokens * topK + (numExperts << paddingLog2) - numExperts);
    int64_t permIdxToTokenIdxSize = (numTokens * topK + (numExperts * tileTokensDim) - numExperts);
    mPtrPermutedIdxToTokenIdxHost
        = mBufferManager->pinned(ITensor::makeShape({permIdxToTokenIdxSize}), nvinfer1::DataType::kINT32);
    mPtrPermutedIdxToTokenIdxDevice
        = mBufferManager->gpu(ITensor::makeShape({permIdxToTokenIdxSize}), nvinfer1::DataType::kINT32);

    int64_t expWeightsSize = numTokens * topK;
    mPtrTopKWeightsHost = mBufferManager->pinned(ITensor::makeShape({expWeightsSize}), TRTDataType<T>::value);
    mPtrTopKWeightsDevice = mBufferManager->gpu(ITensor::makeShape({expWeightsSize}), TRTDataType<T>::value);

    if (useTopKAsInput)
    {
        int64_t topKIdsSize = numTokens * topK;
        mPtrTopKIdsHost = mBufferManager->pinned(ITensor::makeShape({topKIdsSize}), nvinfer1::DataType::kINT32);
        mPtrTopKIdsDevice = mBufferManager->gpu(ITensor::makeShape({topKIdsSize}), nvinfer1::DataType::kINT32);
    }
    else
    {
        mPtrTopKIdsHost = nullptr;
        mPtrTopKIdsDevice = nullptr;
    }

    int64_t ctaIdxSize = numTokens * topK;
    mPtrCtaIdxXyToBatchIdxHost = mBufferManager->pinned(ITensor::makeShape({ctaIdxSize}), nvinfer1::DataType::kINT32);
    mPtrCtaIdxXyToBatchIdxDevice = mBufferManager->gpu(ITensor::makeShape({ctaIdxSize}), nvinfer1::DataType::kINT32);

    mPtrCtaIdxXyToMnLimitHost = mBufferManager->pinned(ITensor::makeShape({ctaIdxSize}), nvinfer1::DataType::kINT32);
    mPtrCtaIdxXyToMnLimitDevice = mBufferManager->gpu(ITensor::makeShape({ctaIdxSize}), nvinfer1::DataType::kINT32);

    int64_t numNonExitingCtasSize = 1;
    mPtrNumNonExitingCtasHost
        = mBufferManager->pinned(ITensor::makeShape({numNonExitingCtasSize}), nvinfer1::DataType::kINT32);
    mPtrNumNonExitingCtasDevice
        = mBufferManager->gpu(ITensor::makeShape({numNonExitingCtasSize}), nvinfer1::DataType::kINT32);

    int64_t idxSize = numTokens * topK * sizeof(PackedType);
    mPtrTopKPackedHost = mBufferManager->pinned(ITensor::makeShape({idxSize}), nvinfer1::DataType::kINT8);
    mPtrTopKPackedDevice = mBufferManager->gpu(ITensor::makeShape({idxSize}), nvinfer1::DataType::kINT8);
    mCurandStatesDevice
        = mBufferManager->gpu(ITensor::makeShape({numTokens, sizeof(curandState_t)}), nvinfer1::DataType::kINT8);
}

template <typename T>
void RoutingKernelTest<T>::setupBuffers(RoutingKernelTestParam const& param)
{
    T* scoresHostPtr = bufferCast<T>(*mPtrScoresHost);
    initData(scoresHostPtr, param.numTokens * param.numExperts, mSeed);
    mBufferManager->copy(*mPtrScoresHost, *mPtrScoresDevice);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
void RoutingKernelTest<T>::computePermutation(RoutingKernelTestParam const& param)
{

    int32_t* expertCountsHostPtr = bufferCast<int32_t>(*this->mPtrExpertCountsHost);
    PackedType* expIdxHostPtr = reinterpret_cast<PackedType*>(bufferCast<int8_t>(*this->mPtrTopKPackedHost));

    auto tokenToExpertHost
        = mBufferManager->pinned(ITensor::makeShape({param.numTokens * param.topK}), nvinfer1::DataType::kINT32);
    auto tokenToExpertHostPtr = bufferCast<int32_t>(*tokenToExpertHost);

    auto tokenToIdxInExpertHost
        = mBufferManager->pinned(ITensor::makeShape({param.numTokens * param.topK}), nvinfer1::DataType::kINT32);
    auto tokenToIdxInExpertHostPtr = bufferCast<int32_t>(*tokenToIdxInExpertHost);

    auto expertScanCountsHost
        = mBufferManager->pinned(ITensor::makeShape({param.numExperts + 1}), nvinfer1::DataType::kINT32);
    auto expertScanCountsHostPtr = bufferCast<int32_t>(*expertScanCountsHost);

    auto ctaScanCountsHost
        = mBufferManager->pinned(ITensor::makeShape({param.numExperts + 1}), nvinfer1::DataType::kINT32);
    auto ctaScanCountsHostPtr = bufferCast<int32_t>(*ctaScanCountsHost);

    for (int ie = 0; ie < param.numExperts + 1; ++ie)
    {
        if (ie < param.numExperts)
        {
            expertCountsHostPtr[ie] = 0;
        }
        expertScanCountsHostPtr[ie] = 0;
        ctaScanCountsHostPtr[ie] = 0;
    }

    for (int it = 0; it < param.numTokens; ++it)
    {
        for (int ie = 0; ie < param.topK; ++ie)
        {
            int32_t index = expIdxHostPtr[it * param.topK + ie].idx;
            tokenToExpertHostPtr[it * param.topK + ie] = index;

            auto localExpertIdx = index - param.localExpertsStartIdx;
            auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < param.numLocalExperts
                && (localExpertIdx & param.localExpertsStrideLog2) == 0;
            tokenToIdxInExpertHostPtr[it * param.topK + ie] = expertCountsHostPtr[index];
            if (isLocalExpert)
            {
                expertCountsHostPtr[index]++;
            }
        }
    }

    // Calculate prefix sum of expert counts, padded to tileTokensDim
    for (int ie = 0; ie < param.numExperts; ++ie)
    {
        int32_t tmp;
        tmp = divUpMulTileN<int32_t>(expertCountsHostPtr[ie], param.tileTokensDim);
        expertScanCountsHostPtr[ie + 1] = expertScanCountsHostPtr[ie] + tmp;
        tmp = divUpTileN(expertCountsHostPtr[ie], param.tileTokensDim);
        ctaScanCountsHostPtr[ie + 1] = ctaScanCountsHostPtr[ie] + tmp;
    }

    // Store total size needed for permuted indices buffer
    bufferCast<int32_t>(*this->mPtrPermutedIdxSizeHost)[0] = expertScanCountsHostPtr[param.numExperts];
    // Store total number of CTAs needed across all experts
    bufferCast<int32_t>(*this->mPtrNumNonExitingCtasHost)[0] = ctaScanCountsHostPtr[param.numExperts];

    auto permutedBufferMaxSize
        = param.numTokens * param.topK + mulTileN(param.numExperts, param.tileTokensDim) - param.numExperts;
    for (int ii = 0; ii < permutedBufferMaxSize; ++ii)
        bufferCast<int32_t>(*this->mPtrPermutedIdxToTokenIdxHost)[ii] = -1;

    for (int tokenIdx = 0; tokenIdx < param.numTokens; tokenIdx++)
    {
        for (int k = 0; k < param.topK; k++)
        {
            int const expandedIdx = tokenIdx * param.topK + k;
            int const expert = tokenToExpertHostPtr[expandedIdx];
            auto localExpertIdx = expert - param.localExpertsStartIdx;
            auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < param.numLocalExperts
                && (localExpertIdx & param.localExpertsStrideLog2) == 0;

            int const offsetWithinExpert = tokenToIdxInExpertHostPtr[expandedIdx];
            int const offsetForExpert = expertScanCountsHostPtr[expert];
            int const permutedIdx = isLocalExpert ? offsetForExpert + offsetWithinExpert : int32_t{-1};
            // int const permutedIdx = offsetForExpert + offsetWithinExpert;
            bufferCast<int32_t>(*this->mPtrExpandedIdxToPermutedIdxHost)[expandedIdx] = permutedIdx;
            if (isLocalExpert)
            {
                bufferCast<int32_t>(*this->mPtrPermutedIdxToTokenIdxHost)[permutedIdx] = tokenIdx;
            }
        }
    }

    for (int ie = 0; ie < param.numExperts; ++ie)
    {
        int m = expertCountsHostPtr[ie];
        // Skip if expert isn't used
        if (m == 0)
        {
            continue;
        }

        // int32_t numCta = divUpLog2(m, param.paddingLog2);
        int32_t numCta = divUpTileN(m, param.tileTokensDim);

        const int32_t localExpertIdx = (ie - param.localExpertsStartIdx) >> param.localExpertsStrideLog2;

        for (int32_t cta = 0; cta < numCta; ++cta)
        {
            // Map CTA index to expert index and compute token range for this CTA
            bufferCast<int32_t>(*this->mPtrCtaIdxXyToBatchIdxHost)[ctaScanCountsHostPtr[ie] + cta] = localExpertIdx;
            bufferCast<int32_t>(*this->mPtrCtaIdxXyToMnLimitHost)[ctaScanCountsHostPtr[ie] + cta]
                = std::min(mulTileN(ctaScanCountsHostPtr[ie] + cta + 1, param.tileTokensDim),
                    mulTileN(ctaScanCountsHostPtr[ie], param.tileTokensDim) + m);
        }
    }
}

template <typename T>
void RoutingKernelTest<T>::callHostFunction(RoutingKernelTestParam const& param)
{
    computeTopKExperts(param);
    computePermutation(param);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
void RoutingKernelTest<T>::verifyExpertRoutingIndices(RoutingKernelTestParam const& param)
{

    // for permuted index, there is non-determinism, thus we check set-equality
    // for this, we go over every expert and retrieve the tokens routed to it
    // we then get the associated indexes and check set equality

    auto const expandedIdxToPermutedIdxHost
        = mBufferManager->copyFrom(*mPtrExpandedIdxToPermutedIdxDevice, MemoryType::kCPU);
    auto const hostExpToPermTest = bufferCast<int32_t>(*expandedIdxToPermutedIdxHost);

    auto const permutedIdxToTokenIdxHost = mBufferManager->copyFrom(*mPtrPermutedIdxToTokenIdxDevice, MemoryType::kCPU);
    auto const hostPermToTokTest = bufferCast<int32_t>(*permutedIdxToTokenIdxHost);

    mStream->synchronize();

    int32_t* expIdxToPermHostptr = bufferCast<int32_t>(*mPtrExpandedIdxToPermutedIdxHost);
    PackedType* expIdxHostPtr = reinterpret_cast<PackedType*>(bufferCast<int8_t>(*mPtrTopKPackedHost));

    for (int ie = 0; ie < param.numExperts; ++ie)
    {
        std::set<int32_t> permutedIdx, permutedIdxTest;
        std::set<int32_t> tokenIdx, tokenIdxTest;
        auto localExpertIdx = ie - param.localExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < param.numLocalExperts
            && (localExpertIdx & param.localExpertsStrideLog2) == 0;

        for (int it = 0; it < param.numTokens * param.topK; ++it)
        {
            if (expIdxHostPtr[it].idx == ie)
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

template <typename T>
void RoutingKernelTest<T>::verifyResult(RoutingKernelTestParam const& param)
{
    auto const expertWeightsHost = mBufferManager->copyFrom(*mPtrTopKWeightsDevice, MemoryType::kCPU);
    auto const expertCountsHost = mBufferManager->copyFrom(*mPtrExpertCountsDevice, MemoryType::kCPU);
    auto const permutedIdxSizeHost = mBufferManager->copyFrom(*mPtrPermutedIdxSizeDevice, MemoryType::kCPU);
    auto const numNonExitingCtasHost = mBufferManager->copyFrom(*mPtrNumNonExitingCtasDevice, MemoryType::kCPU);
    auto const ctaIdxXyToBatchIdxHost = mBufferManager->copyFrom(*mPtrCtaIdxXyToBatchIdxDevice, MemoryType::kCPU);
    auto const ctaIdxXyToMnLimitHost = mBufferManager->copyFrom(*mPtrCtaIdxXyToMnLimitDevice, MemoryType::kCPU);

    auto const expertWeightsPtr = bufferCast<T>(*expertWeightsHost);
    auto const expertCountsPtr = bufferCast<int32_t>(*expertCountsHost);
    auto const permutedIdxSizePtr = bufferCast<int32_t>(*permutedIdxSizeHost);
    auto const numNonExitingCtasPtr = bufferCast<int32_t>(*numNonExitingCtasHost);
    auto const ctaIdxXyToBatchIdxPtr = bufferCast<int32_t>(*ctaIdxXyToBatchIdxHost);
    auto const ctaIdxXyToMnLimitPtr = bufferCast<int32_t>(*ctaIdxXyToMnLimitHost);

    mStream->synchronize();

    if (param.getExpWeights)
    {
        EXPECT_EQ(isClose(bufferCast<T>(*mPtrTopKWeightsHost), expertWeightsPtr, param.numTokens * param.topK,
                      "expert weights"),
            true);
    }
    // expert counts aren't always used, but if tokens > 8 * 1024, we are sure they are used
    if (param.numTokens > param.singleClusterTokenNum)
    { //@Todo: check if this is always true
        assertEqual(bufferCast<int32_t>(*mPtrExpertCountsHost), expertCountsPtr, param.numExperts, "expert counts");
        if (param.routingMethod != RoutingMethodType::DeepSeekV3)
        {
            assertEqual(bufferCast<int32_t>(*mPtrExpertCountsHost), expertCountsPtr + param.numExperts,
                param.numExperts, "expert counts (2)");
        }
    }

    assertEqual(bufferCast<int32_t>(*mPtrPermutedIdxSizeHost), permutedIdxSizePtr, 1, "permuted idx size");
    assertEqual(bufferCast<int32_t>(*mPtrNumNonExitingCtasHost), numNonExitingCtasPtr, 1, "#non exiting CTAs");

    verifyExpertRoutingIndices(param);

    assertEqual(bufferCast<int32_t>(*mPtrCtaIdxXyToBatchIdxHost), ctaIdxXyToBatchIdxPtr,
        bufferCast<int32_t>(*mPtrNumNonExitingCtasHost)[0], "cta idx -> batch idx");
    assertEqual(bufferCast<int32_t>(*mPtrCtaIdxXyToMnLimitHost), ctaIdxXyToMnLimitPtr,
        bufferCast<int32_t>(*mPtrNumNonExitingCtasHost)[0], "cta idx -> M/N limit");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void RoutingKernelTest<T>::runTest(RoutingKernelTestParam const& param)
{
    if (mDeviceProp.major < param.requiredComputeCapability)
    {
        GTEST_SKIP() << "Skip test due to compute capability requirement.";
    }
    // Set seed to time-based seed
    resetToTimeBasedSeed();

    // Allocate buffers
    allocateBuffers(param);
    // Setup buffers
    setupBuffers(param);

    // Call host function
    callHostFunction(param);
    if (param.useTopKAsInput)
    {
        // Set the topk_ids as input
        mBufferManager->copy(*mPtrTopKIdsHost, *mPtrTopKIdsDevice);
        mBufferManager->copy(*mPtrTopKWeightsHost, *mPtrTopKWeightsDevice);
        mStream->synchronize();
    }
    // Retrieve the workspace size of the routing kernel.
    auto const workspaceSize = getDeviceWorkspaceSize(param);
    TensorPtr workspaceDevice
        = mBufferManager->gpu(ITensor::makeShape({static_cast<int64_t>(workspaceSize)}), nvinfer1::DataType::kINT8);

    // Call tested function routing
    callTestedFunction(param, workspaceDevice);
    // Verify results
    verifyResult(param);
}

template class RoutingKernelTest<float>;
template class RoutingKernelTest<__nv_bfloat16>;

} // namespace tensorrt_llm::tests::kernels::routing
