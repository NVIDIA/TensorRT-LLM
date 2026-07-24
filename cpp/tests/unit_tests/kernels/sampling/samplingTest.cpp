/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tests/unit_tests/kernels/sampling/samplingTest.h"

namespace tensorrt_llm::tests::kernels::sampling
{

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
namespace trk = tensorrt_llm::runtime::kernels;

template <typename T>
void SamplingKernelTest<T>::SetUp()
{
    mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
    mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);

    auto const device = tc::getDevice();
    cudaGetDeviceProperties(&mDeviceProp, device);
}

template <typename T>
void SamplingKernelTest<T>::TearDown()
{
}

template <typename T>
void SamplingKernelTest<T>::allocateBuffers(SamplingKernelTestParam const& param)
{
    auto const batchSize = param.batchSize;
    auto const maxBatchSize = 2 * batchSize;
    auto const vocabSize = param.vocabSize;
    auto const maxTokensPerStep = param.maxTokensPerStep;

    auto const dataType = TRTDataType<T>::value;
    auto const ptrType = TRTDataType<T*>::value;

    // Allocate GPU data
    mSeqLengthsHost = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    mSeqLengthsDevice = mBufferManager->gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);

    mFinishedHost = BufferManager::pinned(
        ITensor::makeShape({maxBatchSize}), TRTDataType<tk::FinishedState::UnderlyingType>::value);
    mFinishedDevice = mBufferManager->gpu(
        ITensor::makeShape({maxBatchSize}), TRTDataType<tk::FinishedState::UnderlyingType>::value);

    mOutputIdsHost = BufferManager::pinned(ITensor::makeShape({maxBatchSize, mMaxSeqLen}), nvinfer1::DataType::kINT32);
    mOutputIdsDevice = mBufferManager->gpu(ITensor::makeShape({maxBatchSize, mMaxSeqLen}), nvinfer1::DataType::kINT32);

    mProbsHost = BufferManager::pinned(ITensor::makeShape({batchSize, maxTokensPerStep, vocabSize}), dataType);
    mProbsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize, maxTokensPerStep, vocabSize}), dataType);
    mProbsPtrsDevice
        = BufferManager::pinned(ITensor::makeShape({batchSize, maxTokensPerStep}), nvinfer1::DataType::kINT64);

    mCumLogProbsDevice = mBufferManager->gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);

    if (param.returnAllSelectedTokens)
    {
        SizeType32 maxTopK = param.topK == 0 ? vocabSize : param.topK;
        mOutputLogProbsDevice
            = mBufferManager->gpu(ITensor::makeShape({maxBatchSize, maxTopK}), nvinfer1::DataType::kFLOAT);
    }
    else
    {
        mOutputLogProbsDevice
            = mBufferManager->gpu(ITensor::makeShape({mMaxSeqLen, maxBatchSize}), nvinfer1::DataType::kFLOAT);
    }

    mZeroParentIdsDevice
        = mBufferManager->gpu(ITensor::makeShape({maxBatchSize, maxTokensPerStep}), nvinfer1::DataType::kINT32);

    mLogitsHost = BufferManager::pinned(ITensor::makeShape({batchSize, maxTokensPerStep, vocabSize}), dataType);
    mLogProbsHost = BufferManager::pinned(ITensor::makeShape({batchSize, maxTokensPerStep, vocabSize}), dataType);
    mIdsPtrHost = BufferManager::pinned(ITensor::makeShape({2 * maxBatchSize}), ptrType);

    mEndIdsHost = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    mEndIdsDevice = mBufferManager->gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);

    mTopPsHost = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    mTopPsDevice = mBufferManager->gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);

    mTopKsHost = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    mTopKsDevice = mBufferManager->gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);

    mSkipDecodeHost = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kBOOL);
    mSkipDecodeDevice = mBufferManager->gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kBOOL);

    mTokensPerStep = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);

    mBatchSlots = BufferManager::pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);

    mExpectedCumLogProbsHost = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);

    if (param.returnAllSelectedTokens)
    {
        SizeType32 maxTopK = param.topK == 0 ? vocabSize : param.topK;
        mExpectedLogProbsHost
            = BufferManager::pinned(ITensor::makeShape({maxBatchSize, maxTopK}), nvinfer1::DataType::kFLOAT);
    }
    else
    {
        mExpectedLogProbsHost
            = BufferManager::pinned(ITensor::makeShape({mMaxSeqLen, maxBatchSize}), nvinfer1::DataType::kFLOAT);
    }

    mCurandStatesDevice
        = mBufferManager->gpu(ITensor::makeShape({maxBatchSize, sizeof(curandState_t)}), nvinfer1::DataType::kINT8);
}

template <typename T>
void SamplingKernelTest<T>::setupBuffers(SamplingKernelTestParam const& param)
{
    auto const batchSize = param.batchSize;
    auto const maxBatchSize = 2 * batchSize;
    auto const vocabSize = param.vocabSize;
    auto const maxTokensPerStep = param.maxTokensPerStep;

    auto const topK = param.topK;
    auto const topP = param.topP;
    // TopK == 0 case (TopP kernel)
    auto const topKDistUpperBound = std::max(topK, static_cast<unsigned int>(1));

    std::mt19937 gen(42);

    auto* batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
    auto probsPtr = BufferRange<T*>(*mProbsPtrsDevice);
    auto probsDevicePtr = bufferCast<T>(*mProbsDevice);
    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        batchSlotsPtr[bi] = 2 * bi;
        for (SizeType32 ti = 0; ti < maxTokensPerStep; ++ti)
        {
            probsPtr[bi * maxTokensPerStep + ti] = probsDevicePtr + bi * maxTokensPerStep * vocabSize + ti * vocabSize;
        }
    }

    // Allocate and init curand states
    tk::invokeCurandInitialize(reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStatesDevice)),
        batchSlotsPtr, batchSize, mSeed, mStream->get());

    std::uniform_int_distribution<> endIdsDistr(
        0, vocabSize - 1); // -1 because uniform_int_distribution generates closed interval
    std::uniform_real_distribution<> skipDecodeDist(0, 1);
    std::uniform_real_distribution<> topPDist(0, topP);
    std::uniform_int_distribution<> topKDist(1, topKDistUpperBound);
    std::uniform_int_distribution<> tokensPerStepDist(1, maxTokensPerStep);
    std::uniform_int_distribution<> seqLenDist(0, mMaxSeqLen - maxTokensPerStep);
    std::uniform_real_distribution<> logProbDist(-3.f, 3.f);
    std::uniform_real_distribution<> finishedDist(0, 1);

    // Init by zero.
    trk::invokeFill(*mFinishedDevice, uint8_t{0}, *mStream);
    trk::invokeFill(*mOutputLogProbsDevice, float{0.0f}, *mStream);
    trk::invokeFill(*mZeroParentIdsDevice, int32_t{0}, *mStream);
    trk::invokeFill(*mOutputIdsDevice, int32_t{0}, *mStream);

    // Init topK, topP and endIds for each request in batch
    auto skipDecodeHostPtr = bufferCast<bool>(*mSkipDecodeHost);
    auto topPsHostPtr = bufferCast<float>(*mTopPsHost);
    auto topKsHostPtr = bufferCast<int32_t>(*mTopKsHost);
    auto endIdsHostPtr = bufferCast<int32_t>(*mEndIdsHost);
    auto tokensPerStepPtr = bufferCast<int32_t>(*mTokensPerStep);
    auto finishedHostPtr
        = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedHost));
    for (SizeType32 bi = 0; bi < maxBatchSize; ++bi)
    {
        endIdsHostPtr[bi] = endIdsDistr(gen);
        skipDecodeHostPtr[bi] = skipDecodeDist(gen) > 0.8;
        topPsHostPtr[bi] = topPDist(gen);
        topKsHostPtr[bi] = topK == 0 ? 0 : topKDist(gen);
        tokensPerStepPtr[bi] = tokensPerStepDist(gen);
        finishedHostPtr[bi] = finishedDist(gen) > 0.8 ? tk::FinishedState::finished() : tk::FinishedState::empty();
    }
    mMaxTopK = topK;
    mMaxTopP = topP;

    TLLM_CHECK(mMaxTopK * maxTokensPerStep <= mMaxSeqLen);

    // Setup pointers to output ids for each request in batch
    auto idsPtrHostPtr = BufferRange<void*>(*mIdsPtrHost);
    auto outputIdsDevicePtr = bufferCast<int32_t>(*mOutputIdsDevice);
    auto zeroParentIdsDevicePtr = bufferCast<int32_t>(*mZeroParentIdsDevice);
    auto seqLensHostPtr = bufferCast<int32_t>(*mSeqLengthsHost);
    auto logProbHostPtr = bufferCast<float>(*mExpectedCumLogProbsHost);
    for (SizeType32 bi = 0; bi < maxBatchSize; bi++)
    {
        idsPtrHostPtr[bi] = outputIdsDevicePtr + bi * mMaxSeqLen;
        idsPtrHostPtr[maxBatchSize + bi] = zeroParentIdsDevicePtr + bi * mMaxSeqLen;
    }

    for (SizeType32 bi = 0; bi < maxBatchSize; bi++)
    {
        seqLensHostPtr[bi] = seqLenDist(gen);
        logProbHostPtr[bi] = logProbDist(gen);
    }

    mBufferManager->copy(*mEndIdsHost, *mEndIdsDevice);
    mBufferManager->copy(*mSkipDecodeHost, *mSkipDecodeDevice);
    mBufferManager->copy(*mTopPsHost, *mTopPsDevice);
    mBufferManager->copy(*mTopKsHost, *mTopKsDevice);
    mBufferManager->copy(*mSeqLengthsHost, *mSeqLengthsDevice);
    mBufferManager->copy(*mExpectedCumLogProbsHost, *mCumLogProbsDevice);
    mBufferManager->copy(*mFinishedHost, *mFinishedDevice);

    // Init logits randomly
    auto logitsHostPtr = bufferCast<T>(*mLogitsHost);
    initRandom(logitsHostPtr, batchSize * maxTokensPerStep * vocabSize, -3.0f, 3.0f);
    // Only in greedy search we can guarantee the selected token and stop by condition
    // TopK == 1 for TopK kernel greedy, TopK == 0 for TopP kernels
    if (topK <= 1)
    {
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            for (int32_t ti = 0; ti < maxTokensPerStep; ++ti)
            {
                // Set logit of the endId for the finished request to the value above others
                // NOTE that we can guarantee finish only in greedy search
                logitsHostPtr[(bi * maxTokensPerStep + ti) * vocabSize + endIdsHostPtr[batchSlot]] = 4.0f;
            }
        }
    }

    // Compute probabilities for each token
    computeProb(bufferCast<T>(*mProbsHost), logitsHostPtr, batchSize * maxTokensPerStep, vocabSize);
    mBufferManager->copy(*mProbsHost, *mProbsDevice);
}

template <typename T>
std::vector<SizeType32> SamplingKernelTest<T>::computeTopKTopPVariants(SamplingKernelTestParam const& param, int32_t bi,
    int32_t batchSlot, int32_t ti, int32_t maxTokensPerStep, int32_t vocabSize)
{
    std::vector<SizeType32> allowedTokens;
    auto probsPtr = bufferCast<T>(*mProbsHost) + (bi * maxTokensPerStep + ti) * vocabSize;
    std::vector<SizeType32> indices(vocabSize);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
        [probsPtr](SizeType32 i1, SizeType32 i2) { return probsPtr[i1] > probsPtr[i2]; });

    auto topK = bufferCast<int32_t>(*mTopKsHost)[batchSlot];
    auto topP = bufferCast<float>(*mTopPsHost)[batchSlot];

    if (topK > 0)         // handling top K kernel, top P result based on topK tokens
    {
        float sSum = 0.f; // sSum as in samplingTopKKernels.cu
        for (auto ki = 0; ki < topK; ki++)
        {
            sSum += static_cast<float>(probsPtr[indices[ki]]);
        }
        topP *= sSum; // the adjusted topP in the selected topK distribution
    }

    float totalProb = 0.f;
    SizeType32 idx = 0;
    while (totalProb < topP && idx < vocabSize)
    {
        allowedTokens.push_back(indices[idx]);
        totalProb += static_cast<float>(probsPtr[indices[idx++]]);
        // cuda may selected a different index with same probability in kernel reduce, in test we allow them
        while (idx < vocabSize
            && static_cast<float>(probsPtr[indices[idx]]) == static_cast<float>(probsPtr[indices[idx - 1]]))
        {
            if (param.returnAllSelectedTokens && (totalProb + static_cast<float>(probsPtr[indices[idx]]) >= topP))
            {
                break;
            }
            allowedTokens.push_back(indices[idx]);
            totalProb += static_cast<float>(probsPtr[indices[idx++]]);
        }
    }
    return allowedTokens;
}

template <typename T>
void SamplingKernelTest<T>::verifyResult(SamplingKernelTestParam const& param)
{
    auto const batchSize = param.batchSize;
    auto const maxBatchSize = 2 * batchSize;
    auto const vocabSize = param.vocabSize;
    auto const maxTokensPerStep = param.maxTokensPerStep;

    auto const outputIdsHost = mBufferManager->copyFrom(*mOutputIdsDevice, MemoryType::kCPU);
    auto const seqLenHost = mBufferManager->copyFrom(*mSeqLengthsDevice, MemoryType::kCPU);
    auto const finishedHost = mBufferManager->copyFrom(*mFinishedDevice, MemoryType::kCPU);
    auto const cumLogProbsHost = mBufferManager->copyFrom(*mCumLogProbsDevice, MemoryType::kCPU);
    auto const logProbsHost = mBufferManager->copyFrom(*mOutputLogProbsDevice, MemoryType::kCPU);

    // Synchronize to get valid data on Host
    mStream->synchronize();

    // Compute reference.
    computeLogProb(bufferCast<T>(*mLogProbsHost), bufferCast<T>(*mLogitsHost), batchSize * maxTokensPerStep, vocabSize);

    auto const batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);

    auto const outputIdsHostPtr = bufferCast<int32_t>(*outputIdsHost);
    auto const seqLengthsHostPtr = bufferCast<int32_t>(*seqLenHost);
    auto const finishedHostPtr
        = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*finishedHost));

    auto const outputIdsOrigHostPtr = bufferCast<int32_t>(*mOutputIdsHost);
    auto const seqLengthsOrigHostPtr = bufferCast<int32_t>(*mSeqLengthsHost);
    auto const finishedOrigHostPtr
        = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedHost));

    auto const logProbsHostPtr = bufferCast<T>(*mLogProbsHost);
    auto const endIdsHostPtr = bufferCast<int32_t>(*mEndIdsHost);
    auto const skipDecodeHostPtr = bufferCast<bool>(*mSkipDecodeHost);
    auto const tokensPerStepPtr = bufferCast<int32_t>(*mTokensPerStep);
    auto const expectedCumLogProbsHostPtr = bufferCast<float>(*mExpectedCumLogProbsHost);
    auto const expectedLogProbsHostPtr = bufferCast<float>(*mExpectedLogProbsHost);

    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        auto const batchSlot = batchSlotsPtr[bi];
        auto const tokensPerStep = tokensPerStepPtr[batchSlot];
        for (SizeType32 ti = 0; ti < tokensPerStep; ++ti)
        {
            auto topK = bufferCast<int32_t>(*mTopKsHost)[batchSlot];
            auto kResults = param.returnAllSelectedTokens ? (topK == 0 ? vocabSize : topK) : 1;
            auto topKTopPVariants = computeTopKTopPVariants(param, bi, batchSlot, ti, maxTokensPerStep, vocabSize);
            SizeType32 ki;
            for (ki = 0; ki < kResults && ki < topKTopPVariants.size(); ++ki)
            {
                // Set reference finished state to true if we finished before or at current step
                auto const idsIdx
                    = param.returnAllSelectedTokens ? ti * mMaxTopK + ki : seqLengthsOrigHostPtr[batchSlot] + ti;
                auto const outputId = outputIdsHostPtr[batchSlot * mMaxSeqLen + idsIdx];
                // Check the range of the returned token ([0, vocabSize))
                EXPECT_TRUE((outputId >= 0) && (outputId < vocabSize));
                bool const generatedEOS = outputId == endIdsHostPtr[batchSlot];

                // If decoding for this batch is skipped ignore cumLog calculation
                if (!skipDecodeHostPtr[batchSlot] && !finishedOrigHostPtr[batchSlot].isFinished()
                    && !finishedOrigHostPtr[batchSlot].isSkipDecoding())
                {
                    if (maxTokensPerStep == 1 && !param.returnAllSelectedTokens)
                    {
                        if (generatedEOS)
                        {
                            EXPECT_EQ(seqLengthsHostPtr[batchSlot], seqLengthsOrigHostPtr[batchSlot]);
                            EXPECT_TRUE(finishedHostPtr[batchSlot].isFinished());
                        }
                        else
                        {
                            EXPECT_EQ(seqLengthsHostPtr[batchSlot], seqLengthsOrigHostPtr[batchSlot] + tokensPerStep);
                            EXPECT_EQ(
                                finishedHostPtr[batchSlot].isFinished(), finishedOrigHostPtr[batchSlot].isFinished());
                        }
                    }

                    bool found = false;
                    for (auto const& var : topKTopPVariants)
                    {
                        if (outputId == var)
                        {
                            found = true;
                            break;
                        }
                    }
                    EXPECT_TRUE(found) << "Incorrect output id token";

                    // Compute reference cumLogProb by summing all logProbs up to the stop token
                    expectedCumLogProbsHostPtr[batchSlot]
                        += static_cast<float>(logProbsHostPtr[bi * vocabSize + outputId]);

                    if (param.returnAllSelectedTokens)
                    {
                        // expectedLogProbsHostPtr shape: [maxBatchSize, maxTopK]
                        if (param.topK)
                        {
                            expectedLogProbsHostPtr[batchSlot * mMaxTopK + ki]
                                = static_cast<float>(logProbsHostPtr[bi * vocabSize + outputId]);
                        }
                    }
                    else
                    {
                        auto const curSeqLen = seqLengthsOrigHostPtr[batchSlot];
                        // expectedLogProbsHostPtr shape: [maxSeqLen, maxBatchSize]
                        expectedLogProbsHostPtr[curSeqLen * maxBatchSize + batchSlot]
                            = static_cast<float>(logProbsHostPtr[bi * vocabSize + outputId]);
                    }
                }
                else
                {
                    // Check that tensors are not modified
                    auto const idsIdx = batchSlot * mMaxSeqLen + seqLengthsOrigHostPtr[batchSlot] + ti;
                    EXPECT_EQ(outputId, outputIdsOrigHostPtr[idsIdx]);
                    EXPECT_EQ(seqLengthsHostPtr[batchSlot], seqLengthsOrigHostPtr[batchSlot]);
                    EXPECT_EQ(finishedHostPtr[batchSlot].isFinished(), finishedOrigHostPtr[batchSlot].isFinished());
                }
            }

            // a boundary check for returnAllSelectedTokens in topP kernel and when TopP selected indices < topK in topK
            // kernel.
            if (!skipDecodeHostPtr[batchSlot] && !finishedOrigHostPtr[batchSlot].isFinished()
                && !finishedOrigHostPtr[batchSlot].isSkipDecoding())
            {
                if (param.returnAllSelectedTokens && (topK == 0 || ki != topK))
                {
                    auto const idsIdx = ti * mMaxTopK + ki;
                    auto const outputId = outputIdsHostPtr[batchSlot * mMaxSeqLen + idsIdx];
                    EXPECT_EQ(outputId, -1);
                }
            }
        }
    }

    // Check logProb
    // if (maxTokensPerStep == 1 && !param.returnAllSelectedTokens)
    if (maxTokensPerStep == 1)
    {
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto* batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
            auto const batchSlot = batchSlotsPtr[bi];

            if (!skipDecodeHostPtr[batchSlot] && !finishedOrigHostPtr[batchSlot].isFinished()
                && !finishedOrigHostPtr[batchSlot].isSkipDecoding())
            {
                auto const curSeqLen = seqLengthsOrigHostPtr[batchSlot];
                if (param.returnAllSelectedTokens)
                {
                    if (param.topK)
                    {
                        // Only
                        // logprob shape: [maxBatchSize, maxTopK]
                        auto topK = bufferCast<int32_t>(*mTopKsHost)[batchSlot];
                        auto kResults = param.returnAllSelectedTokens ? (topK == 0 ? vocabSize : topK) : 1;
                        auto topKTopPVariants
                            = computeTopKTopPVariants(param, bi, batchSlot, 1, maxTokensPerStep, vocabSize);

                        for (SizeType32 ki = 0; ki < kResults && ki < topKTopPVariants.size(); ++ki)
                        {
                            auto const logprobValue = *(bufferCast<float>(*logProbsHost) + (batchSlot * mMaxTopK + ki));
                            auto const expectedLogprobValue = expectedLogProbsHostPtr[batchSlot * mMaxTopK + ki];
                            bool passed = checkResult("log probs", &logprobValue, &expectedLogprobValue, 1);
                            EXPECT_TRUE(passed);
                        }
                    }
                }
                else
                {
                    // logprob shape: [maxSeqLen, maxBatchSize]
                    auto const logprobValue
                        = *(bufferCast<float>(*logProbsHost) + (curSeqLen * maxBatchSize + batchSlot));
                    auto const expectedLogprobValue = expectedLogProbsHostPtr[curSeqLen * maxBatchSize + batchSlot];

                    bool passed = checkResult("log probs", &logprobValue, &expectedLogprobValue, 1);
                    EXPECT_TRUE(passed);
                }
            }
        }
    }

    // Cum log probs is not supported for multiple tokens per step or all top K return
    if (maxTokensPerStep == 1 && !param.returnAllSelectedTokens)
    {
        for (int32_t bi = 0; bi < batchSize; ++bi)
        {
            auto* batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
            auto const batchSlot = batchSlotsPtr[bi];
            bool passed = checkResult("cum log probs", bufferCast<float>(*cumLogProbsHost) + batchSlot,
                bufferCast<float>(*mExpectedCumLogProbsHost) + batchSlot, 1);
            EXPECT_TRUE(passed);
        }
    }
}

template <typename T>
void SamplingKernelTest<T>::runTest(SamplingKernelTestParam const& param)
{
    // Allocate buffers
    allocateBuffers(param);

    // Setup buffers
    setupBuffers(param);

    // Retrieve the workspace size of the sampling kernel.
    auto const workspaceSize = getWorkspaceSize(param);
    TensorPtr workspaceDevice
        = mBufferManager->gpu(ITensor::makeShape({static_cast<int32_t>(workspaceSize)}), nvinfer1::DataType::kINT8);

    // Call tested function sampling
    callTestedFunction(param, workspaceDevice);

    // Verify results
    verifyResult(param);
}

template class SamplingKernelTest<float>;
template class SamplingKernelTest<half>;

} // namespace tensorrt_llm::tests::kernels::sampling
