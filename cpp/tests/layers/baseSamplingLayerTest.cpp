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

#include "tests/layers/baseSamplingLayerTest.h"

namespace tensorrt_llm::tests::layers::sampling
{

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::layers;
using namespace tensorrt_llm::common;

namespace tk = tensorrt_llm::kernels;
namespace tcc = tensorrt_llm::common::conversion;
namespace trk = tensorrt_llm::runtime::kernels;

template <typename T>
void BaseSamplingLayerTest<T>::setup(uint64_t seed, SamplingParams const& params)
{
    auto const dataType = TRTDataType<T>::value;
    auto const ptrType = TRTDataType<T*>::value;

    // clang-format off

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    mTestLogitsInit = {
            -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -0.9163, -1.2040, -1.6094, -2.3026, // step 0
            -0.9163, -1.2040, -1.6094, -2.3026, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, // step 1
            -FLT_MAX, -FLT_MAX, -0.9163, -1.2040, -1.6094, -2.3026, -FLT_MAX, -FLT_MAX, // step 2
            -0.9163, -1.2040, -1.6094, -2.3026, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX  // step 3
    };

    // clang-format on
    if (mComputeProbs)
    {
        computeProb(mTestLogitsInit.data(), mTestLogitsInit.data(), 4, mVocabSize);
    }

    mLogitsDevice = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mVocabSize}), dataType);

    mSeqLengthsDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
    mContextLengthDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
    mFinishedDevice = mBufferManager->gpu(
        ITensor::makeShape({mMaxBatchSize}), TRTDataType<tk::FinishedState::UnderlyingType>::value);
    mOutputIdsDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize, mMaxSeqLen}), nvinfer1::DataType::kINT32);
    mEndIdsDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
    mIdsPtrHost = mBufferManager->pinned(ITensor::makeShape({mMaxBatchSize}), ptrType);

    mCumLogProbsDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kFLOAT);

    mBatchSlots = mBufferManager->pinned(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);

    mCurandStatesDevice
        = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize, sizeof(curandState_t)}), nvinfer1::DataType::kINT8);
    auto const workspaceSize = mSamplingLayer->getWorkspaceSize();
    mSamplingWorkspaceDevice = mBufferManager->gpu(workspaceSize, nvinfer1::DataType::kINT8);

    trk::invokeFill(*mSeqLengthsDevice, int32_t{0}, *mStream);
    trk::invokeFill(*mContextLengthDevice, int32_t{0}, *mStream);
    trk::invokeFill(*mFinishedDevice, uint8_t{0}, *mStream);
    trk::invokeFill(*mOutputIdsDevice, int32_t{0}, *mStream);
    trk::invokeFill(*mCumLogProbsDevice, float{0.0f}, *mStream);
    trk::invokeFill(*mEndIdsDevice, int32_t{mEndId}, *mStream);

    auto batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
    for (SizeType bi = 0; bi < mBatchSize; ++bi)
    {
        batchSlotsPtr[bi] = 2 * bi;
    }

    auto idsPtrHostPtr = BufferRange<void*>(*mIdsPtrHost);
    auto outputIdsDevicePtr = bufferCast<int32_t>(*mOutputIdsDevice);
    for (SizeType bi = 0; bi < mMaxBatchSize; bi++)
    {
        idsPtrHostPtr[bi] = outputIdsDevicePtr + bi * mMaxSeqLen;
    }

    typename TopKSamplingLayer<T>::SetupParams setupParams;
    setupParams.randomSeed = std::make_optional<std::vector<uint64_t>>({seed});
    setupParams.runtime_top_k
        = params.topKs.size() ? std::make_optional<std::vector<uint32_t>>(params.topKs) : std::nullopt;
    setupParams.runtime_top_p
        = params.topPs.size() ? std::make_optional<std::vector<float>>(params.topPs) : std::nullopt;
    setupParams.top_p_decay = params.decay.size() ? std::make_optional<std::vector<float>>(params.decay) : std::nullopt;
    setupParams.top_p_min
        = params.minTopP.size() ? std::make_optional<std::vector<float>>(params.minTopP) : std::nullopt;
    setupParams.top_p_reset_ids
        = params.topPResetIds.size() ? std::make_optional<std::vector<int32_t>>(params.topPResetIds) : std::nullopt;

    mSamplingLayer->setup(mBatchSize, batchSlotsPtr, setupParams);

    mStream->synchronize();
}

template <typename T>
typename BaseSamplingLayer<T>::ForwardParams BaseSamplingLayerTest<T>::createInputTensors(int32_t step)
{
    constexpr int32_t ite = 0;
    typename BaseSamplingLayer<T>::ForwardParams decodeInputTensors{
        step, ite, tcc::toTllmTensor(*mLogitsDevice), tcc::toTllmTensor(*mEndIdsDevice), mMaxSeqLen};

    decodeInputTensors.input_lengths = tcc::toTllmTensor(*mContextLengthDevice);

    decodeInputTensors.finished = tcc::toTllmTensor(*mFinishedDevice);

    decodeInputTensors.batch_slots = tcc::toTllmTensor(*mBatchSlots);

    decodeInputTensors.probs_computed = mComputeProbs;

    decodeInputTensors.curand_states = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStatesDevice));

    decodeInputTensors.sampling_workspace = reinterpret_cast<void*>(bufferCast<int8_t>(*mSamplingWorkspaceDevice));

    return decodeInputTensors;
}

template <typename T>
DecodingOutputParams BaseSamplingLayerTest<T>::createOutputTensors()
{
    DecodingOutputParams decodeOutputs(tcc::toTllmTensor(*mOutputIdsDevice));
    decodeOutputs.output_ids_ptr = tcc::toTllmTensor(*mIdsPtrHost);

    decodeOutputs.sequence_length = tcc::toTllmTensor(*mSeqLengthsDevice);

    decodeOutputs.finished = tcc::toTllmTensor(*mFinishedDevice);

    decodeOutputs.cum_log_probs = tcc::toTllmTensor(*mCumLogProbsDevice);

    // TODO(nkorobov): check log probs and cum_log_probs
    return decodeOutputs;
}

template <typename T>
void BaseSamplingLayerTest<T>::batchCopy(int32_t step)
{
    auto const logitsHost = ITensor::wrap(
        mTestLogitsInit.data() + step * mVocabSize, TRTDataType<T>::value, ITensor::makeShape({1, mVocabSize}));
    for (int32_t bi = 0; bi < mBatchSize; ++bi)
    {
        auto logitsDeviceView = ITensor::slice(mLogitsDevice, bi, 1);
        mBufferManager->copy(*logitsHost, *logitsDeviceView);
    }
}

template <typename T>
bool BaseSamplingLayerTest<T>::checkResult(int32_t* outputIds, std::vector<std::set<int32_t>>& expectedIds)
{
    assert(expectedIds.size() == mMaxSeqLen * mBatchBeam);
    int failures = 0;
    auto const batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
    for (int32_t i = 0; i < mMaxSeqLen * mBatchBeam; ++i)
    {
        int32_t s = i / mBatchBeam;
        int32_t b = i % mBatchBeam;
        auto const batchSlot = batchSlotsPtr[b];
        std::set<int32_t> expts = expectedIds.at(i);
        const auto outputId = outputIds[batchSlot * mMaxSeqLen + s];
        if (expts.count(outputId) == 0)
        {
            if (failures < 10)
            {
                std::stringstream ss;
                ss << " - Fail "
                   << " (step=" << s << ", batch=" << b << ") "
                   << "actual=" << outputId << ", expected";
                for (auto& expt : expts)
                {
                    ss << " " << expt;
                }
                TLLM_LOG_DEBUG("%s", ss.str().c_str());
            }
            ++failures;
        }
    }
    TLLM_LOG_DEBUG(
        "check...%6s : failures: %d / %d", failures == 0 ? "....OK" : "FAILED", failures, mMaxSeqLen * mBatchBeam);
    return failures == 0;
}

template <typename T>
void BaseSamplingLayerTest<T>::runTest(
    std::vector<std::set<int32_t>> expectedOutputIds, SamplingParams const& params, int32_t endId)
{
    initLayer(params);

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
            inputTensors.step = step;
            mSamplingLayer->forward(outputTensors, inputTensors);
            mStream->synchronize();
        }

        const auto outputIdsHost = mBufferManager->copyFrom(*mOutputIdsDevice, tensorrt_llm::runtime::MemoryType::kCPU);

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
