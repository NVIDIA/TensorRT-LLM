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
#include "tests/kernels/sampling/samplingTest.h"

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

    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&mDeviceProp, device);
}

template <typename T>
void SamplingKernelTest<T>::TearDown()
{
}

template <typename T>
void SamplingKernelTest<T>::allocateBuffers(int32_t batchSize, int32_t vocabSize, int32_t maxSeqLen, int32_t outputLen)
{
    // Allocate GPU data
    mSeqLengthsHost = mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
    mSeqLengthsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);

    mFinishedHost = mBufferManager->pinned(
        ITensor::makeShape({batchSize}), TRTDataType<tk::FinishedState::UnderlyingType>::value);
    mFinishedDevice
        = mBufferManager->gpu(ITensor::makeShape({batchSize}), TRTDataType<tk::FinishedState::UnderlyingType>::value);

    mOutputIdsHost = mBufferManager->pinned(ITensor::makeShape({batchSize, maxSeqLen}), nvinfer1::DataType::kINT32);
    mOutputIdsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize, maxSeqLen}), nvinfer1::DataType::kINT32);

    mProbsHost = mBufferManager->pinned(ITensor::makeShape({batchSize, vocabSize}),
        std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);
    mProbsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize, vocabSize}),
        std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);

    mCumLogProbsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    mOutputLogProbsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize, outputLen}), nvinfer1::DataType::kFLOAT);

    mZeroParentIdsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize, maxSeqLen}), nvinfer1::DataType::kINT32);
    mTopPIdValsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize, vocabSize}), nvinfer1::DataType::kINT32);
    mBeginOffsetsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32);
    mEndOffsetsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32);

    mLogitsHost = mBufferManager->pinned(ITensor::makeShape({batchSize, vocabSize}),
        std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);
    mLogProbsHost = mBufferManager->pinned(ITensor::makeShape({batchSize, vocabSize}),
        std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);
    mIdsPtrHost = mBufferManager->pinned(ITensor::makeShape({2 * batchSize}), nvinfer1::DataType::kINT64);

    mEndIdsHost = mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
    mEndIdsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);

    mTopPsHost = mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    mTopPsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);

    mTopKsHost = mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
    mTopKsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);

    mSkipDecodeHost = mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kBOOL);
    mSkipDecodeDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), nvinfer1::DataType::kBOOL);

    mExpectedCumLogProbsHost = mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
}

template <typename T>
void SamplingKernelTest<T>::setupBuffers(int32_t batchSize, int32_t vocabSize, int32_t maxSeqLen, int32_t outputLen,
    int32_t topK, float topP, bool useSkipDecode, bool hasDiffRuntimeArgs, std::mt19937& gen,
    std::uniform_int_distribution<>& endIdsDistr)
{
    // Allocate and init curand states
    cudaMalloc(&mCurandStatesDevice, sizeof(curandState_t) * batchSize);
    tk::invokeCurandInitialize(mCurandStatesDevice, batchSize, mSeed, mStream->get());

    std::uniform_real_distribution<> skipDecodeDist(0, 1); // uniform distribution between 0 and 1
    std::uniform_real_distribution<> topPDist(0, 1);       // uniform distribution between 0 and 1
    std::uniform_int_distribution<> topKDist(1, std::min(1024, vocabSize));

    // Init by zero.
    trk::invokeFill(*mSeqLengthsDevice, int32_t{0}, *mStream);
    trk::invokeFill(*mFinishedDevice, uint8_t{0}, *mStream);
    trk::invokeFill(*mCumLogProbsDevice, float{0.0f}, *mStream);
    trk::invokeFill(*mOutputLogProbsDevice, float{0.0f}, *mStream);
    trk::invokeFill(*mZeroParentIdsDevice, int32_t{0}, *mStream);
    trk::invokeFill(*mOutputIdsDevice, int32_t{0}, *mStream);
    std::fill_n(bufferCast<float>(*mExpectedCumLogProbsHost), batchSize, 0);

    // Init topK, topP and endIds for each request in batch
    auto skipDecodeHostPtr = bufferCast<bool>(*mSkipDecodeHost);
    auto topPsHostPtr = bufferCast<float>(*mTopPsHost);
    auto topKsHostPtr = bufferCast<int32_t>(*mTopKsHost);
    auto endIdsHostPtr = bufferCast<int32_t>(*mEndIdsHost);
    for (SizeType bi = 0; bi < batchSize; ++bi)
    {
        endIdsHostPtr[bi] = endIdsDistr(gen);
        skipDecodeHostPtr[bi] = useSkipDecode ? skipDecodeDist(gen) > 0.8 : false;
        topKsHostPtr[bi] = hasDiffRuntimeArgs ? topKDist(gen) : topK;
        topPsHostPtr[bi] = hasDiffRuntimeArgs ? topPDist(gen) : topP;
    }
    mMaxTopK = *std::max_element(topKsHostPtr, topKsHostPtr + batchSize);
    mMaxTopP = *std::max_element(topPsHostPtr, topPsHostPtr + batchSize);

    // Setup pointers to output ids for each request in batch
    auto idsPtrHostPtr = reinterpret_cast<void**>(bufferCast<int64_t>(*mIdsPtrHost));
    auto outputIdsDevicePtr = bufferCast<int32_t>(*mOutputIdsDevice);
    auto zeroParentIdsDevicePtr = bufferCast<int32_t>(*mZeroParentIdsDevice);
    for (SizeType bi = 0; bi < batchSize; bi++)
    {
        idsPtrHostPtr[bi] = outputIdsDevicePtr + bi * maxSeqLen;
    }
    for (SizeType bi = 0; bi < batchSize; bi++)
    {
        idsPtrHostPtr[batchSize + bi] = zeroParentIdsDevicePtr + bi * maxSeqLen;
    }

    mBufferManager->copy(*mEndIdsHost, *mEndIdsDevice);
    mBufferManager->copy(*mSkipDecodeHost, *mSkipDecodeDevice);
    mBufferManager->copy(*mTopPsHost, *mTopPsDevice);
    mBufferManager->copy(*mTopKsHost, *mTopKsDevice);
}

template <typename T>
void SamplingKernelTest<T>::verifyCurrentStep(int32_t batchSize, int32_t vocabSize, int32_t maxSeqLen, int32_t step,
    bool greedySearch, bool useSkipDecode, bool hasDiffRuntimeArgs, std::vector<tk::FinishedState>& refFinished,
    std::vector<int32_t>& refSeqLength, const std::vector<tk::FinishedState>& finishedCurrentStep)
{
    const auto outputIdsHostPtr = bufferCast<int32_t>(*mOutputIdsHost);
    const auto seqLengthsHostPtr = bufferCast<int32_t>(*mSeqLengthsHost);
    const auto finishedHostPtr
        = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedHost));
    const auto logProbsHostPtr = bufferCast<T>(*mLogProbsHost);
    const auto endIdsHostPtr = bufferCast<int32_t>(*mEndIdsHost);
    const auto skipDecodeHostPtr = bufferCast<bool>(*mSkipDecodeHost);
    auto expectedCumLogProbsHostPtr = bufferCast<float>(*mExpectedCumLogProbsHost);

    for (SizeType bi = 0; bi < batchSize; ++bi)
    {
        // Set reference finished state to true if we finished before or at current step
        const bool generatedEOS = outputIdsHostPtr[bi * maxSeqLen + step] == endIdsHostPtr[bi];
        bool finishedThisStep = finishedCurrentStep[bi].isFinished() || generatedEOS;
        refFinished[bi] = generatedEOS ? tk::FinishedState::finishedEOS() : refFinished[bi];

        if (!refFinished[bi].isFinished())
        {
            // Increase reference seq len excluding the EOS token
            refSeqLength[bi]++;
        }

        // If decoding for this batch is skipped ignore cumLog calculation
        if (!skipDecodeHostPtr[bi])
        {
            // Check seq len correctness
            EXPECT_EQ(seqLengthsHostPtr[bi], refSeqLength[bi]);
            // Only in greedy search we can guarantee the selected token and stop by condition
            if (greedySearch)
            {
                EXPECT_EQ(finishedHostPtr[bi].isFinished(), refFinished[bi].isFinished());
            }

            int idx = bi * vocabSize + outputIdsHostPtr[bi * maxSeqLen + step];
            // Compute reference cumLogProb by summing all logProbs up to the stop token
            expectedCumLogProbsHostPtr[bi]
                += step < refSeqLength[bi] || finishedThisStep ? (float) logProbsHostPtr[idx] : 0.0f;
            // If sequence has just finished at this step
            if (finishedHostPtr[bi].isFinished() && step < seqLengthsHostPtr[bi])
            {
                // Check that finished tokens is endId
                EXPECT_EQ(outputIdsHostPtr[bi * maxSeqLen + step], endIdsHostPtr[bi])
                    << "step: " << step << " b: " << bi << " hasDiffRuntimeArgs: " << hasDiffRuntimeArgs
                    << " useSkipDecode: " << useSkipDecode;
            }
            // TODO(nkorobov): check correctness with K>1
        }
    }
}

template <typename T>
void SamplingKernelTest<T>::runTest(const SamplingKernelTestParam& param, bool hasDiffRuntimeArgs, bool useSkipDecode)
{
    const auto batchSize = param.batchSize;
    const auto vocabSize = param.vocabSize;
    const auto outputLen = param.outputLen;
    const auto maxSeqLen = outputLen;

    const auto topK = param.topK;
    const auto topP = param.topP;

    const bool greedySearch = topK == 1 && hasDiffRuntimeArgs == false && useSkipDecode == false;

    std::mt19937 gen(42);
    std::uniform_real_distribution<> finishedDist(0, 1); // uniform distribution between 0 and 1
    std::uniform_int_distribution<> endIdsDistr(
        0, vocabSize - 1); // -1 because uniform_int_distribution generates closed interval

    // Allocate buffers
    allocateBuffers(batchSize, vocabSize, maxSeqLen, outputLen);

    // Setup buffers
    setupBuffers(
        batchSize, vocabSize, maxSeqLen, outputLen, topK, topP, useSkipDecode, hasDiffRuntimeArgs, gen, endIdsDistr);

    // Allocate internal state holders for reference
    std::vector<int32_t> refSeqLength(batchSize);
    std::vector<tk::FinishedState> refFinished(batchSize, tk::FinishedState::empty());

    // retrieve the workspace size of the sampling kernel.
    const auto workspaceSize = getWorkspaceSize(param);
    TensorPtr workspaceDevice
        = mBufferManager->gpu(ITensor::makeShape({static_cast<int32_t>(workspaceSize)}), nvinfer1::DataType::kINT8);

    for (size_t step = 0; step < outputLen; ++step)
    {
        // Init logits randomly
        auto logitsHostPtr = bufferCast<T>(*mLogitsHost);
        auto endIdsHostPtr = bufferCast<int32_t>(*mEndIdsHost);
        initRandom(logitsHostPtr, batchSize * vocabSize, -3.0f, 3.0f);

        std::vector<tk::FinishedState> finishedCurrentStep(batchSize, tk::FinishedState::empty());
        // Only in greedy search we can guarantee the selected token and stop by condition
        if (greedySearch)
        {
            for (SizeType bi = 0; bi < batchSize; ++bi)
            {
                // Randomly decide if the sequence finishes at current step
                finishedCurrentStep[bi] = (refFinished[bi].isFinished() == false && finishedDist(gen) < 0.1)
                    ? tk::FinishedState::finishedEOS()
                    : tk::FinishedState::empty();

                if (finishedCurrentStep[bi].isFinished())
                {
                    // Set logit of the endId for the finished request to the value above others
                    // NOTE that we can guarantee finish only in greedy search
                    logitsHostPtr[bi * vocabSize + endIdsHostPtr[bi]] = 4.0f;
                }
            }
        }

        // Compute probobilities for each token
        computeProb(bufferCast<T>(*mProbsHost), bufferCast<T>(*mLogitsHost), batchSize, vocabSize);
        mBufferManager->copy(*mProbsHost, *mProbsDevice);

        // Call tested function sampling
        callTestedFunction(param, hasDiffRuntimeArgs, workspaceSize, workspaceDevice);

        mBufferManager->copy(*mOutputIdsDevice, *mOutputIdsHost);
        mBufferManager->copy(*mSeqLengthsDevice, *mSeqLengthsHost);
        mBufferManager->copy(*mFinishedDevice, *mFinishedHost);

        // Synchronize to get valid data on Host
        mStream->synchronize();

        // Compute reference.
        computeLogProb(bufferCast<T>(*mLogProbsHost), bufferCast<T>(*mLogitsHost), batchSize, vocabSize);

        verifyCurrentStep(batchSize, vocabSize, maxSeqLen, step, greedySearch, useSkipDecode, hasDiffRuntimeArgs,
            refFinished, refSeqLength, finishedCurrentStep);
    }
    const auto cumLogProbsHost = mBufferManager->copyFrom(*mCumLogProbsDevice, MemoryType::kCPU);

    // Check if cumulative prob is correct
    const bool passed = checkResult(
        param.toString(), bufferCast<float>(*cumLogProbsHost), bufferCast<float>(*mExpectedCumLogProbsHost), batchSize);
    EXPECT_TRUE(passed);

    cudaFree(mCurandStatesDevice);
}

template <typename T>
void SamplingKernelTest<T>::runTest(const SamplingKernelTestParam& param)
{
    runTest(param, false, false); // Single params, do not skip decoders
    runTest(param, true, false);  // Different params, do not skip decoders
    runTest(param, false, true);  // Single params, skip some decoders
    runTest(param, true, true);   // Different params, skip some decoders
}

template class SamplingKernelTest<float>;
template class SamplingKernelTest<half>;

} // namespace tensorrt_llm::tests::kernels::sampling
