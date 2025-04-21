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

#include "tests/unit_tests/layers/baseSamplingLayerTest.h"

namespace tensorrt_llm::tests::layers::sampling
{

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::layers;
using namespace tensorrt_llm::common;

namespace tk = tensorrt_llm::kernels;
namespace trk = tensorrt_llm::runtime::kernels;

template <typename T>
void BaseSamplingLayerTest<T>::setup(uint64_t seed, TestSamplingParams const& params)
{
    auto const dataType = TRTDataType<T>::value;
    auto const ptrType = TRTDataType<T*>::value;

    // clang-format off

    // logits = (-0.9163, -1.2040, -1.6094, -2.3026) -> prob = (0.4, 0.3, 0.2, 0.1)
    std::vector<T> testLogits = {
                -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -0.9163, -1.2040, -1.6094, -2.3026, // step 0
                -0.9163, -1.2040, -1.6094, -2.3026, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, // step 1
                -FLT_MAX, -FLT_MAX, -0.9163, -1.2040, -1.6094, -2.3026, -FLT_MAX, -FLT_MAX, // step 2
                -0.9163, -1.2040, -1.6094, -2.3026, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX  // step 3
    };

    // clang-format on

    if (params.beamWidth == 1)
    {
        mTestLogitsInit = testLogits;
    }
    else
    {
        for (int step = 0; step < mMaxSeqLen; ++step)
        {
            auto const& logitsBegin = testLogits.begin() + mVocabSize * step;
            auto const& logitsEnd = testLogits.begin() + mVocabSize * (step + 1);
            for (int bm = 0; bm < params.beamWidth; ++bm)
            {
                mTestLogitsInit.insert(mTestLogitsInit.end(), logitsBegin, logitsEnd);
            }
        }
    }

    if (mComputeProbs)
    {
        computeProb(mTestLogitsInit.data(), mTestLogitsInit.data(),
            BaseSamplingLayerTest::mMaxOutputLen * params.beamWidth, mVocabSize);
    }

    mSeqLengthsDevice = mBufferManager->gpu(ITensor::makeShape({maxBatchSize()}), nvinfer1::DataType::kINT32);
    mContextLengthDevice = mBufferManager->gpu(ITensor::makeShape({maxBatchSize()}), nvinfer1::DataType::kINT32);
    mFinishedDevice = params.isExternalDraftTokensLayerTest
        ? mBufferManager->gpu(ITensor::makeShape({mMaxTokensPerEngineStep, maxBatchSize()}),
            TRTDataType<tk::FinishedState::UnderlyingType>::value)
        : mBufferManager->gpu(
            ITensor::makeShape({maxBatchSize()}), TRTDataType<tk::FinishedState::UnderlyingType>::value);
    mOutputIdsDevice
        = mBufferManager->gpu(ITensor::makeShape({maxBatchSize(), mBeamWidth, mMaxSeqLen}), nvinfer1::DataType::kINT32);
    mEndIdsDevice = mBufferManager->gpu(ITensor::makeShape({maxBatchSize()}), nvinfer1::DataType::kINT32);
    mIdsPtrHost = mBufferManager->pinned(ITensor::makeShape({maxBatchSize()}), ptrType);

    mCumLogProbsDevice = mBufferManager->gpu(ITensor::makeShape({maxBatchSize()}), nvinfer1::DataType::kFLOAT);
    mOutputLogProbsDevice
        = mBufferManager->gpu(ITensor::makeShape({maxBatchSize(), mMaxSeqLen}), nvinfer1::DataType::kFLOAT);

    mBatchSlots
        = mBufferManager->pinned(ITensor::makeShape({mBatchSize + mBatchSizeBadPad}), nvinfer1::DataType::kINT32);
    mCurandStatesDevice
        = mBufferManager->gpu(ITensor::makeShape({maxBatchSize(), sizeof(curandState_t)}), nvinfer1::DataType::kINT8);

    auto const workspaceSize = mSamplingLayer->getWorkspaceSize();

    trk::invokeFill(*mSeqLengthsDevice, int32_t{0}, *mStream);
    trk::invokeFill(*mContextLengthDevice, int32_t{0}, *mStream);
    trk::invokeFill(*mFinishedDevice, uint8_t{0}, *mStream);
    trk::invokeFill(*mOutputIdsDevice, int32_t{0}, *mStream);
    trk::invokeFill(*mCumLogProbsDevice, float{0.0f}, *mStream);
    trk::invokeFill(*mOutputLogProbsDevice, float{0.0f}, *mStream);
    trk::invokeFill(*mEndIdsDevice, int32_t{mEndId}, *mStream);
    tk::invokeCurandInitialize(reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStatesDevice)), nullptr,
        maxBatchSize(), seed, mStream->get());

    auto batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
    for (SizeType32 bi = 0; bi < mBatchSize; ++bi)
    {
        batchSlotsPtr[bi] = kDoubleBatchIdx * bi;
    }
    for (SizeType32 bi = 0; bi < mBatchSizeBadPad; ++bi)
    {
        batchSlotsPtr[mBatchSize + bi] = 0xbaadf00d;
    }

    auto idsPtrHostPtr = BufferRange<void*>(*mIdsPtrHost);
    auto outputIdsDevicePtr = bufferCast<int32_t>(*mOutputIdsDevice);
    for (SizeType32 bi = 0; bi < maxBatchSize(); bi++)
    {
        idsPtrHostPtr[bi] = outputIdsDevicePtr + bi * mMaxSeqLen;
    }

    std::shared_ptr<DecodingSetupParams> setupParams;
    if (params.isExternalDraftTokensLayerTest)
    {
        auto externalDraftTokensSetupParams = std::make_shared<ExternalDraftTokensSetupParams>();
        externalDraftTokensSetupParams->randomSeed = std::make_optional<std::vector<uint64_t>>({seed});
        externalDraftTokensSetupParams->runtimeTopK
            = params.topKs.size() ? std::make_optional<std::vector<SizeType32>>(params.topKs) : std::nullopt;
        externalDraftTokensSetupParams->runtimeTopP
            = params.topPs.size() ? std::make_optional<std::vector<float>>(params.topPs) : std::nullopt;

        setupParams = externalDraftTokensSetupParams;
    }
    else if (mBeamWidth == 1)
    {
        auto samplingSetupParams = std::make_shared<SamplingSetupParams>();
        samplingSetupParams->randomSeed = std::make_optional<std::vector<uint64_t>>({seed});
        samplingSetupParams->runtimeTopK
            = params.topKs.size() ? std::make_optional<std::vector<SizeType32>>(params.topKs) : std::nullopt;
        samplingSetupParams->runtimeTopP
            = params.topPs.size() ? std::make_optional<std::vector<float>>(params.topPs) : std::nullopt;
        samplingSetupParams->topPDecay
            = params.decay.size() ? std::make_optional<std::vector<float>>(params.decay) : std::nullopt;
        samplingSetupParams->topPMin
            = params.minTopP.size() ? std::make_optional<std::vector<float>>(params.minTopP) : std::nullopt;
        samplingSetupParams->topPResetIds
            = params.topPResetIds.size() ? std::make_optional<std::vector<int32_t>>(params.topPResetIds) : std::nullopt;

        setupParams = samplingSetupParams;
    }
    else // Beam Search
    {
        auto samplingSetupParams = std::make_shared<BeamSearchSetupParams>();
        setupParams = samplingSetupParams;

        mSrcCacheIndirection = mBufferManager->gpu(
            ITensor::makeShape({maxBatchSize(), mBeamWidth, mMaxSeqLen}), nvinfer1::DataType::kINT32);
        mTgtCacheIndirection = mBufferManager->gpu(
            ITensor::makeShape({maxBatchSize(), mBeamWidth, mMaxSeqLen}), nvinfer1::DataType::kINT32);
        mParentIds = mBufferManager->gpu(
            ITensor::makeShape({maxBatchSize(), mBeamWidth, mMaxSeqLen}), nvinfer1::DataType::kINT32);

        auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
        auto constexpr nvSizeType = TRTDataType<SizeType32>::value;
        auto constexpr nvFloatType = TRTDataType<float>::value;
        auto constexpr nvBoolType = TRTDataType<bool>::value;
        mOutputIdsCBA
            = mBufferManager->gpu(ITensor::makeShape({maxBatchSize(), 2 * mBeamWidth, mMaxSeqLen}), nvTokenIdType);
        mLogProbsCBA
            = mBufferManager->gpu(ITensor::makeShape({maxBatchSize(), 2 * mBeamWidth, mMaxSeqLen}), nvFloatType);
        mSequenceLengthsCBA = mBufferManager->gpu(ITensor::makeShape({maxBatchSize(), 2 * mBeamWidth}), nvSizeType);
        mCumLogProbsCBA = mBufferManager->gpu(ITensor::makeShape({maxBatchSize(), 2 * mBeamWidth}), nvFloatType);
        mNormedScoresCBA = mBufferManager->gpu(ITensor::makeShape({maxBatchSize(), 2 * mBeamWidth}), nvFloatType);
        mNumBeamsCBA = mBufferManager->gpu(ITensor::makeShape({maxBatchSize()}), nvSizeType);
        mMinNormedScoresCBA = mBufferManager->gpu(ITensor::makeShape({maxBatchSize()}), nvFloatType);
        mBatchDones = mBufferManager->gpu(ITensor::makeShape({maxBatchSize()}), nvBoolType);
        mOutputIdsPtr = mBufferManager->pinned(ITensor::makeShape({maxBatchSize()}), ptrType);
        mParentIdsPtr = mBufferManager->pinned(ITensor::makeShape({maxBatchSize()}), ptrType);

        trk::invokeFill(*mSrcCacheIndirection, int32_t{0}, *mStream);
        trk::invokeFill(*mTgtCacheIndirection, int32_t{0}, *mStream);
        trk::invokeFill(*mParentIds, int32_t{0}, *mStream);
        trk::invokeFill(*mOutputIdsCBA, int32_t{0}, *mStream);
        trk::invokeFill(*mLogProbsCBA, float{0}, *mStream);
        trk::invokeFill(*mSequenceLengthsCBA, int32_t{0}, *mStream);
        trk::invokeFill(*mCumLogProbsCBA, float{0}, *mStream);
        trk::invokeFill(*mNormedScoresCBA, float{0}, *mStream);
        trk::invokeFill(*mNumBeamsCBA, int32_t{0}, *mStream);
        trk::invokeFill(*mMinNormedScoresCBA, float{0}, *mStream);
        trk::invokeFill(*mBatchDones, bool{0}, *mStream);

        auto outputIdsPtr = bufferCast<int*>(*mOutputIdsPtr);
        auto parentIdsPtr = bufferCast<int*>(*mParentIdsPtr);
        for (SizeType32 bi = 0; bi < maxBatchSize(); bi++)
        {
            outputIdsPtr[bi] = outputIdsDevicePtr + bi * mMaxSeqLen;
            parentIdsPtr[bi] = outputIdsDevicePtr + bi * mMaxSeqLen;
        }
    }

    mDecodingWorkspace->setDeviceBatchSlots(mBatchSlots);
    mDecodingWorkspace->getDeviceRuntimeLogits()->reshape(ITensor::makeShape({mBatchSize, mBeamWidth, mVocabSize}));
    mSamplingLayer->setup(mBatchSize, mBeamWidth, mBatchSlots, setupParams, mDecodingWorkspace);
    mStream->synchronize();
}

template <typename T>
std::shared_ptr<DecodingInputs> BaseSamplingLayerTest<T>::createInputTensors(int32_t step)
{
    constexpr int32_t ite = 0;

    auto decodeInputTensors = (mBeamWidth > 1)
        ? std::make_shared<DecodingInputs>(mEndIdsDevice, mBatchSlots, step, ite, mBatchSize)
        : std::make_shared<SamplingInputs>(mEndIdsDevice, mBatchSlots, step, ite, mBatchSize);
    decodeInputTensors->logits = mDecodingWorkspace->getDeviceRuntimeLogits();
    decodeInputTensors->inputLengths = mContextLengthDevice;
    decodeInputTensors->finished = mFinishedDevice;

    if (mBeamWidth > 1)
    {
        decodeInputTensors->srcCacheIndirection = mSrcCacheIndirection;
    }
    else
    {
        auto samplingInputTensors = std::dynamic_pointer_cast<SamplingInputs>(decodeInputTensors);
        samplingInputTensors->probsComputed = mComputeProbs;
        samplingInputTensors->curandStates = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStatesDevice));
    }

    return decodeInputTensors;
}

template <typename T>
std::shared_ptr<BaseDecodingOutputs> BaseSamplingLayerTest<T>::createOutputTensors()
{
    // TODO: check log probs and cum_log_probs

    auto decodeOutputs = (mBeamWidth > 1) ? std::make_shared<BeamSearchOutputs>(mOutputIdsDevice)
                                          : std::make_shared<BaseDecodingOutputs>(mOutputIdsDevice);
    decodeOutputs->outputIdsPtr = mIdsPtrHost;
    decodeOutputs->outputIdsPtrHost = mIdsPtrHost;
    decodeOutputs->sequenceLength = mSeqLengthsDevice;
    decodeOutputs->finished = mFinishedDevice;
    decodeOutputs->outputLogProbs = mOutputLogProbsDevice;
    decodeOutputs->cumLogProbs = mCumLogProbsDevice;

    if (mBeamWidth > 1)
    {
        auto beamSearchOutputs = std::dynamic_pointer_cast<BeamSearchOutputs>(decodeOutputs);
        beamSearchOutputs->tgtCacheIndirection = mTgtCacheIndirection;
        beamSearchOutputs->parentIds = mParentIds;
        beamSearchOutputs->parentIdsPtr = mParentIdsPtr;
        beamSearchOutputs->beamHypotheses = std::make_unique<tensorrt_llm::kernels::BeamHypotheses>();
        beamSearchOutputs->beamHypotheses->outputIdsCBA = bufferCast<int>(*mOutputIdsCBA);
        beamSearchOutputs->beamHypotheses->logProbsCBA = bufferCast<float>(*mLogProbsCBA);
        beamSearchOutputs->beamHypotheses->sequenceLengthsCBA = bufferCast<int>(*mSequenceLengthsCBA);
        beamSearchOutputs->beamHypotheses->cumLogProbsCBA = bufferCast<float>(*mCumLogProbsCBA);
        beamSearchOutputs->beamHypotheses->normedScoresCBA = bufferCast<float>(*mNormedScoresCBA);
        beamSearchOutputs->beamHypotheses->numBeamsCBA = bufferCast<int>(*mNumBeamsCBA);
        beamSearchOutputs->beamHypotheses->minNormedScoresCBA = bufferCast<float>(*mMinNormedScoresCBA);
        beamSearchOutputs->beamHypotheses->batchDones = bufferCast<bool>(*mBatchDones);
    }

    return decodeOutputs;
}

template <typename T>
void BaseSamplingLayerTest<T>::batchCopy(int32_t step)
{
    auto const logitsHost = ITensor::wrap(mTestLogitsInit.data() + step * mBeamWidth * mVocabSize,
        TRTDataType<T>::value, ITensor::makeShape({mBeamWidth, mVocabSize}));

    for (int32_t bi = 0; bi < mBatchSize; ++bi)
    {
        auto logitsDeviceView = ITensor::slice(mDecodingWorkspace->getDeviceRuntimeLogits(), bi, 1);
        mBufferManager->copy(*logitsHost, *logitsDeviceView);
    }
}

template <typename T>
bool BaseSamplingLayerTest<T>::checkResult(int32_t const* outputIds, std::vector<std::set<int32_t>> const& expectedIds)
{
    assert(expectedIds.size() == mMaxSeqLen * batchBeam());
    int failures = 0;
    auto* const batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
    for (int32_t i = 0; i < mMaxSeqLen * mBatchSize; ++i)
    {
        int32_t s = i / mBatchSize;
        int32_t b = i % mBatchSize;
        auto const batchSlot = batchSlotsPtr[b];
        std::set<int32_t> expts = expectedIds.at(i);
        auto const outputId = outputIds[batchSlot * mMaxSeqLen + s];
        if (expts.count(outputId) == 0)
        {
            if (failures < 10)
            {
                std::stringstream ss;
                ss << " - Fail "
                   << " (step=" << s << ", batch=" << b << ") "
                   << "actual=" << outputId << ", expected";
                for (auto const& expt : expts)
                {
                    ss << " " << expt;
                }
                TLLM_LOG_DEBUG("%s", ss.str().c_str());
            }
            ++failures;
        }
    }
    TLLM_LOG_DEBUG(
        "check...%6s : failures: %d / %d", failures == 0 ? "....OK" : "FAILED", failures, mMaxSeqLen * batchBeam());
    return failures == 0;
}

template <typename T>
void BaseSamplingLayerTest<T>::runTest(
    std::vector<std::set<int32_t>> const& expectedOutputIds, TestSamplingParams const& params, int32_t endId)
{
    mBatchSize = params.batchSize;
    if (params.beamWidth > 1)
    {
        mBeamWidth = params.beamWidth;
        mMaxSeed = 1;
        mComputeProbs = true;
    }
    initLayer(params);

    auto const decoderDomain
        = tensorrt_llm::layers::DecoderDomain(maxBatchSize(), mBeamWidth, mVocabSize, mVocabSizePadded);
    mDecodingWorkspace = std::make_unique<tensorrt_llm::runtime::DecodingLayerWorkspace>(
        mBufferManager, decoderDomain, TRTDataType<T>::value, mSamplingLayer->getWorkspaceSize());
    mEndId = endId;
    for (uint64_t seed = 0; seed < mMaxSeed; ++seed)
    {
        setup(seed, params);

        int32_t step = mMaxInputLen;
        auto inputTensors = createInputTensors(step);
        auto outputTensors = createOutputTensors();

        for (step = mMaxInputLen; step < mMaxOutputLen; ++step)
        {
            // Reset by the test value since the sampling layer internally updates the logit buffer.
            batchCopy(step);
            if (params.isExternalDraftTokensLayerTest)
            {
                inputTensors = createInputTensors(step);
            }
            else
            {
                inputTensors->step = step;
            }
            mDecodingWorkspace->setDeviceBatchSlots(mBatchSlots);
            mSamplingLayer->forwardAsync(outputTensors, inputTensors, mDecodingWorkspace);
            mStream->synchronize();
        }

        auto const outputIdsHost = mBufferManager->copyFrom(*mOutputIdsDevice, tensorrt_llm::runtime::MemoryType::kCPU);

        mStream->synchronize();

        bool passed = checkResult(bufferCast<int32_t>(*outputIdsHost), expectedOutputIds);
        EXPECT_TRUE(passed) << "Output ids check failed at seed " << seed;
        if (!passed)
        {
            std::stringstream ss;
            ss << "Actual output ids:" << std::endl << *outputIdsHost;
            TLLM_LOG_DEBUG(ss.str());
        }
    }
}

template class BaseSamplingLayerTest<float>;
template class BaseSamplingLayerTest<half>;

} // namespace tensorrt_llm::tests::layers::sampling
