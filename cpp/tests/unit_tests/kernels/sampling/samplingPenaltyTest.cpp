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

#include "tensorrt_llm/kernels/penaltyTypes.h"
#include "tests/unit_tests/kernels/sampling/samplingTest.h"

using namespace tensorrt_llm::tests::kernels::sampling;

namespace
{

namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
namespace trk = tensorrt_llm::runtime::kernels;

using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::kernels;

struct TemperatureTestParam
{
    int32_t batchSize;
    int32_t vocabSize;
    TensorPtr temperatures;
    int32_t temperaturesSize;
    int32_t maxTokensPerStep{1};

    TemperatureTestParam& setBatchSize(int32_t bs)
    {
        batchSize = bs;
        return *this;
    }

    TemperatureTestParam& setVocabSize(int32_t vs)
    {
        vocabSize = vs;
        return *this;
    }

    TemperatureTestParam& setTemperaturesSize(int32_t ts)
    {
        temperaturesSize = ts;
        return *this;
    }

    TemperatureTestParam& setTemperatures(TensorPtr temp)
    {
        temperatures = temp;
        return *this;
    }

    TemperatureTestParam& setMaxTokensPerStep(int32_t ts)
    {
        maxTokensPerStep = ts;
        return *this;
    }

    std::string toString() const
    {
        return tc::fmtstr("TemperatureTestParam[batch=%d, vocab=%d, temperatures=%s]", batchSize, vocabSize,
            tc::arr2str(bufferCast<float>(*temperatures), temperaturesSize).c_str());
    }
};

size_t padVocabSize(size_t vocabSize, size_t pad = 8)
{
    return (vocabSize + pad - 1) / pad * pad;
}

template <typename T>
void initLogitsAndBias(T* logits, T* bias, size_t const batchSize, size_t const maxTokensPerStep,
    size_t const vocabSize, size_t const vocabSizePadded)
{
    initRandom(logits, batchSize * maxTokensPerStep * vocabSizePadded, -5.0f, 5.0f);
    if (bias != nullptr)
    {
        initRandom(bias, batchSize * vocabSizePadded, -5.0f, 5.0f);
    }
    bool is_half = sizeof(T) == 2;
    for (size_t i = 0; i < batchSize; ++i)
    {
        for (size_t t = 0; t < maxTokensPerStep; ++t)
        {
            for (size_t j = vocabSize; j < vocabSizePadded; ++j)
            {
                logits[(i * maxTokensPerStep + t) * vocabSizePadded + j]
                    = static_cast<T>(is_half ? -HALF_FLT_MAX : -FLT_MAX);
            }
        }
        for (size_t j = vocabSize; j < vocabSizePadded; ++j)
        {
            if (bias != nullptr && i == 0)
            {
                bias[i * vocabSizePadded + j] = (T) 0.0f;
            }
        }
    }
}

/////////////////////////////////// Tests //////////////////////////////////////////

template <typename T>
class TemperaturePenaltyTest : public SamplingKernelTest<T>
{
protected:
    // Set up test
    int32_t mMaxBatchSize;
    int32_t mBatchSize;
    int32_t mVocabSize;
    int32_t mVocabSizePadded;
    int32_t mMaxTokensPerStep;

    using SamplingKernelTest<T>::mBufferManager;
    using SamplingKernelTest<T>::mStream;
    using SamplingKernelTest<T>::mLogitsHost;

    TensorPtr mLogitsDevice;
    TensorPtr mOutLogitsDevice;
    TensorPtr mLogitsRefHost;
    TensorPtr mLogitsPtrs;
    TensorPtr mPenaltyWorkspaceDevice;
    TensorPtr mTokensPerStep;
    TensorPtr mBiasHost;
    TensorPtr mBiasDevice;
    TensorPtr mTemperaturesDevice;
    TensorPtr mBatchSlots;

    void subsetup(TemperatureTestParam const& param)
    {
        auto const dataType = TRTDataType<T>::value;
        auto const ptrType = TRTDataType<T*>::value;

        mBatchSize = param.batchSize;
        mMaxBatchSize = 2 * mBatchSize;
        mVocabSize = param.vocabSize;
        mVocabSizePadded = padVocabSize(mVocabSize);
        mMaxTokensPerStep = param.maxTokensPerStep;

        mLogitsHost
            = BufferManager::pinned(ITensor::makeShape({mBatchSize, mMaxTokensPerStep, mVocabSizePadded}), dataType);
        mLogitsRefHost
            = BufferManager::pinned(ITensor::makeShape({mBatchSize, mMaxTokensPerStep, mVocabSizePadded}), dataType);
        mLogitsDevice
            = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mMaxTokensPerStep, mVocabSizePadded}), dataType);
        mOutLogitsDevice
            = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mMaxTokensPerStep, mVocabSizePadded}), dataType);
        mLogitsPtrs = BufferManager::pinned(ITensor::makeShape({mBatchSize}), ptrType);

        mPenaltyWorkspaceDevice = mBufferManager->gpu(
            ITensor::makeShape({mMaxBatchSize, mMaxTokensPerStep, mVocabSizePadded}), nvinfer1::DataType::kINT32);

        mTokensPerStep = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);

        mBiasHost = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize, mVocabSizePadded}), dataType);
        mBiasDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize, mVocabSizePadded}), dataType);

        mBatchSlots = BufferManager::pinned(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);

        trk::invokeFill(*mLogitsRefHost, T{0.0f}, *mStream);
        trk::invokeFill(*mOutLogitsDevice, T{0.0f}, *mStream);

        trk::invokeFill(*mLogitsRefHost, T{0.0f}, *mStream);
        trk::invokeFill(*mOutLogitsDevice, T{0.0f}, *mStream);

        auto batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
        for (SizeType32 bi = 0; bi < mBatchSize; ++bi)
        {
            batchSlotsPtr[bi] = 2 * bi;
        }

        initLogitsAndBias(bufferCast<T>(*mLogitsHost), bufferCast<T>(*mBiasHost), mBatchSize, mMaxTokensPerStep,
            mVocabSize, mVocabSizePadded);

        mBufferManager->copy(*mLogitsHost, *mLogitsDevice);

        auto logitsPtrs = BufferRange<T*>(*mLogitsPtrs);
        for (SizeType32 bi = 0; bi < mBatchSize; ++bi)
        {
            logitsPtrs[bi]
                = bufferCast<T>(*mLogitsDevice) + (mBatchSize - bi - 1) * mMaxTokensPerStep * mVocabSizePadded;
        }

        auto tokensPerStepPtr = bufferCast<int32_t>(*mTokensPerStep);
        for (SizeType32 bi = 0; bi < mMaxBatchSize; ++bi)
        {
            tokensPerStepPtr[bi] = (std::rand() % mMaxTokensPerStep) + 1;
        }

        mBufferManager->copy(*mBiasHost, *mBiasDevice);

        ASSERT_EQ(param.temperaturesSize, mMaxBatchSize) << "Invalid test configuration.";
        mTemperaturesDevice
            = mBufferManager->gpu(ITensor::makeShape({param.temperaturesSize}), nvinfer1::DataType::kFLOAT);
        mBufferManager->copy(*param.temperatures, *mTemperaturesDevice);
    }

    void computeReference(T const* const inLogits, T* const outLogits, T const* const bias,
        float const* const temperatures, size_t const temperaturesSize)
    {
        bool const IS_FP16 = std::is_same<T, half>::value;
        T const MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
        auto const batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
        auto const tokensPerStepPtr = bufferCast<int32_t>(*mTokensPerStep);

        for (int32_t bi = 0; bi < mBatchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            for (int32_t ti = 0; ti < tokensPerStepPtr[batchSlot]; ++ti)
            {
                for (int32_t vi = 0; vi < mVocabSizePadded; ++vi)
                {
                    auto const srcIdx = ((mBatchSize - bi - 1) * mMaxTokensPerStep + ti) * mVocabSizePadded + vi;
                    auto const dstIdx = (bi * mMaxTokensPerStep + ti) * mVocabSizePadded + vi;
                    outLogits[dstIdx] = inLogits[srcIdx];
                }
            }
        }

        for (size_t bi = 0; bi < mBatchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            for (int32_t ti = 0; ti < tokensPerStepPtr[batchSlot]; ++ti)
            {
                auto temperature = temperatures[batchSlot];
                ASSERT_GT(temperature, 0.0f) << "temperature should be positive but got " << temperature;
                for (size_t j = 0; j < mVocabSizePadded; ++j)
                {
                    size_t index = (bi * mMaxTokensPerStep + ti) * mVocabSizePadded + j;
                    auto logit = static_cast<float>(outLogits[index]);
                    if (j < mVocabSize && bias != nullptr)
                    {
                        logit += static_cast<float>(bias[batchSlot * mVocabSizePadded + j]);
                    }
                    outLogits[index] = j < mVocabSize ? static_cast<T>(logit / temperature) : -MAX_T_VAL;
                }
            }
        }
    }

public:
    void runTest(TemperatureTestParam param)
    {
        subsetup(param);
        // Do test
        InvokeBatchApplyPenaltyParams<T> penaltyParams{reinterpret_cast<T**>(bufferCast<int64_t>(*mLogitsPtrs)),
            bufferCast<T>(*mOutLogitsDevice), bufferCast<T>(*mBiasDevice),
            bufferCast<int32_t>(*mPenaltyWorkspaceDevice), nullptr, bufferCast<float>(*mTemperaturesDevice), nullptr,
            nullptr, nullptr, mBatchSize, 1, 1, mVocabSize, mVocabSizePadded, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, bufferCast<int32_t>(*mBatchSlots), mMaxTokensPerStep,
            bufferCast<int32_t>(*mTokensPerStep), nullptr, mStream->get()};
        tk::invokeBatchApplyPenalty(penaltyParams);
        auto logitsOutHost = mBufferManager->copyFrom(*mOutLogitsDevice, MemoryType::kCPU);

        mStream->synchronize();

        computeReference(bufferCast<T>(*mLogitsHost), bufferCast<T>(*mLogitsRefHost), bufferCast<T>(*mBiasHost),
            bufferCast<float>(*param.temperatures), param.temperaturesSize);
        bool passed = checkResult(param.toString(), bufferCast<T>(*logitsOutHost), bufferCast<T>(*mLogitsRefHost),
            mBatchSize * mMaxTokensPerStep * mVocabSizePadded);
        EXPECT_TRUE(passed);
    }
};

TYPED_TEST_SUITE(TemperaturePenaltyTest, FloatAndHalfTypes);

TYPED_TEST(TemperaturePenaltyTest, NoPenalty)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr temperaturesHost = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*temperaturesHost)[i] = 1.0f;
    }
    this->runTest(TemperatureTestParam()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setTemperaturesSize(maxBatchSize)
                      .setTemperatures(temperaturesHost));
}

TYPED_TEST(TemperaturePenaltyTest, LessThanOne)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr temperaturesHost = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*temperaturesHost)[i] = 0.53f;
    }
    this->runTest(TemperatureTestParam()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setTemperaturesSize(maxBatchSize)
                      .setTemperatures(temperaturesHost));
}

TYPED_TEST(TemperaturePenaltyTest, GreaterThaneOne)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr temperaturesHost = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*temperaturesHost)[i] = 2.01f;
    }
    this->runTest(TemperatureTestParam()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setTemperaturesSize(maxBatchSize)
                      .setTemperatures(temperaturesHost));
}

TYPED_TEST(TemperaturePenaltyTest, Mixed)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr temperaturesHost = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*temperaturesHost)[i] = 0.53f + 0.2f * i;
    }
    this->runTest(TemperatureTestParam()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setTemperaturesSize(maxBatchSize)
                      .setTemperatures(temperaturesHost));
}

TYPED_TEST(TemperaturePenaltyTest, LargeVocab)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr temperaturesHost = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*temperaturesHost)[i] = 0.53f + 0.2f * i;
    }
    this->runTest(TemperatureTestParam()
                      .setBatchSize(batchSize)
                      .setVocabSize(50001)
                      .setTemperaturesSize(maxBatchSize)
                      .setTemperatures(temperaturesHost));
}

TYPED_TEST(TemperaturePenaltyTest, LargeVocabTokensPerStep)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr temperaturesHost = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*temperaturesHost)[i] = 1.f; // 0.53f + 0.2f * i;
    }
    this->runTest(TemperatureTestParam()
                      .setBatchSize(batchSize)
                      .setVocabSize(8)
                      .setTemperaturesSize(maxBatchSize)
                      .setTemperatures(temperaturesHost)
                      .setMaxTokensPerStep(4));
}

struct RepetitionPenaltyTestCase
{
    int32_t batchSize;
    int32_t vocabSize;
    int32_t maxInputLength;
    TensorPtr repetitionPenalties;
    TensorPtr presencePenalties;
    TensorPtr frequencyPenalties;
    int32_t repetitionPenaltiesSize;
    int32_t presencePenaltiesSize;
    int32_t frequencyPenaltiesSize;
    int32_t maxTokensPerStep{1};

    RepetitionPenaltyTestCase& setBatchSize(int32_t bs)
    {
        batchSize = bs;
        return *this;
    }

    RepetitionPenaltyTestCase& setVocabSize(int32_t vs)
    {
        vocabSize = vs;
        return *this;
    }

    RepetitionPenaltyTestCase& setMaxInputLength(int32_t len)
    {
        maxInputLength = len;
        return *this;
    }

    RepetitionPenaltyTestCase& setRepetitionPenalties(TensorPtr rp)
    {
        repetitionPenalties = rp;
        return *this;
    }

    RepetitionPenaltyTestCase& setPresencePenalties(TensorPtr pp)
    {
        presencePenalties = pp;
        return *this;
    }

    RepetitionPenaltyTestCase& setFrequencyPenalties(TensorPtr fp)
    {
        frequencyPenalties = fp;
        return *this;
    }

    RepetitionPenaltyTestCase& setRepetitionPenaltiesSize(int32_t rps)
    {
        repetitionPenaltiesSize = rps;
        return *this;
    }

    RepetitionPenaltyTestCase& setPresencePenaltiesSize(int32_t pps)
    {
        presencePenaltiesSize = pps;
        return *this;
    }

    RepetitionPenaltyTestCase& setFrequencyPenaltiesSize(int32_t fps)
    {
        frequencyPenaltiesSize = fps;
        return *this;
    }

    RepetitionPenaltyTestCase& setMaxTokensPerStep(int32_t ts)
    {
        maxTokensPerStep = ts;
        return *this;
    }

    std::string toString() const
    {
        return tc::fmtstr(
            "RepetitionPenaltyTestCase[batch=%d, vocab=%d, maxInputLength=%d, "
            "repetitionPenalties=%s, presencePenalties=%s, frequencyPenalties=%s]",
            batchSize, vocabSize, maxInputLength,
            tc::arr2str(bufferCast<float>(*repetitionPenalties), repetitionPenaltiesSize).c_str(),
            tc::arr2str(bufferCast<float>(*presencePenalties), presencePenaltiesSize).c_str(),
            tc::arr2str(bufferCast<float>(*frequencyPenalties), frequencyPenaltiesSize).c_str());
    }
};

template <typename T>
class RepetitionPenaltyTest : public SamplingKernelTest<T>
{
protected:
    // Set up test
    int32_t mBatchSize;
    int32_t mMaxBatchSize;
    int32_t mVocabSize;
    int32_t mVocabSizePadded;
    int32_t mMaxInputLength;
    int32_t mSequenceLength;
    int32_t mMaxTokensPerStep;

    using SamplingKernelTest<T>::mBufferManager;
    using SamplingKernelTest<T>::mStream;
    using SamplingKernelTest<T>::mLogitsHost;

    TensorPtr mLogitsDevice;
    TensorPtr mOutLogitsDevice;
    TensorPtr mLogitsRefHost;
    TensorPtr mLogitsPtrs;
    TensorPtr mPenaltyWorkspaceDevice;

    TensorPtr mTokensPerStep;

    TensorPtr mOutputIdsHost;
    TensorPtr mOutputIdsDevice;

    TensorPtr mContextLengthHost;
    TensorPtr mContextLengthDevice;

    TensorPtr mSeqLengthHost;
    TensorPtr mSeqLengthDevice;

    TensorPtr mIdsPtrHost;
    TensorPtr mIdsPtrDevice;

    TensorPtr mRepetitionPenaltiesDevice;
    TensorPtr mPresencePenaltiesDevice;
    TensorPtr mFrequencyPenaltiesDevice;
    TensorPtr mBatchSlots;

    void subsetup(RepetitionPenaltyTestCase param)
    {
        auto const dataType = TRTDataType<T>::value;
        auto const ptrType = TRTDataType<T*>::value;

        mBatchSize = param.batchSize;
        mMaxBatchSize = 2 * mBatchSize;
        mVocabSize = param.vocabSize;
        mVocabSizePadded = padVocabSize(mVocabSize);
        mMaxInputLength = param.maxInputLength;
        mSequenceLength = 2 * mMaxInputLength; // input + output
        mMaxTokensPerStep = param.maxTokensPerStep;

        mLogitsHost
            = BufferManager::pinned(ITensor::makeShape({mBatchSize, mMaxTokensPerStep, mVocabSizePadded}), dataType);
        mLogitsRefHost
            = BufferManager::pinned(ITensor::makeShape({mBatchSize, mMaxTokensPerStep, mVocabSizePadded}), dataType);
        mLogitsDevice
            = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mMaxTokensPerStep, mVocabSizePadded}), dataType);
        mOutLogitsDevice
            = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mMaxTokensPerStep, mVocabSizePadded}), dataType);
        mLogitsPtrs = BufferManager::pinned(ITensor::makeShape({mBatchSize}), ptrType);

        mPenaltyWorkspaceDevice = mBufferManager->gpu(
            ITensor::makeShape({mBatchSize, mMaxTokensPerStep, mVocabSize}), nvinfer1::DataType::kINT32);

        mTokensPerStep = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);

        mOutputIdsHost
            = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize, mSequenceLength}), nvinfer1::DataType::kINT32);
        mOutputIdsDevice
            = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize, mSequenceLength}), nvinfer1::DataType::kINT32);

        mSeqLengthHost = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        mSeqLengthDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);

        mContextLengthHost = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        mContextLengthDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);

        mIdsPtrHost = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), ptrType);
        mIdsPtrDevice = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), ptrType);

        mBatchSlots = BufferManager::pinned(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);

        auto batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
        for (SizeType32 bi = 0; bi < mBatchSize; ++bi)
        {
            batchSlotsPtr[bi] = 2 * bi;
        }

        initLogitsAndBias(bufferCast<T>(*mLogitsHost), static_cast<T*>(nullptr), mBatchSize, mMaxTokensPerStep,
            mVocabSize, mVocabSizePadded);
        initRandomInt(bufferCast<int32_t>(*mOutputIdsHost), mSequenceLength * mMaxBatchSize, 0, mVocabSize);
        initRandomInt(bufferCast<int32_t>(*mSeqLengthHost), mMaxBatchSize, 1, mSequenceLength);
        for (size_t i = 0; i < mMaxBatchSize; ++i)
        {
            bufferCast<int32_t>(*mContextLengthHost)[i] = bufferCast<int32_t>(*mSeqLengthHost)[i];
        }

        trk::invokeFill(*mLogitsRefHost, T{0.0f}, *mStream);
        trk::invokeFill(*mOutLogitsDevice, T{0.0f}, *mStream);

        auto idsPtrHostPtr = BufferRange<void*>(*mIdsPtrHost);
        auto outputIdsDevicePtr = bufferCast<int32_t>(*mOutputIdsDevice);
        auto tokensPerStepPtr = bufferCast<int32_t>(*mTokensPerStep);
        for (SizeType32 bi = 0; bi < mMaxBatchSize; bi++)
        {
            idsPtrHostPtr[bi] = outputIdsDevicePtr + bi * mSequenceLength;
            tokensPerStepPtr[bi] = (std::rand() % mMaxTokensPerStep) + 1;
        }

        mBufferManager->copy(*mLogitsHost, *mLogitsDevice);
        mBufferManager->copy(*mOutputIdsHost, *mOutputIdsDevice);
        mBufferManager->copy(*mContextLengthHost, *mContextLengthDevice);
        mBufferManager->copy(*mSeqLengthHost, *mSeqLengthDevice);
        mBufferManager->copy(*mIdsPtrHost, *mIdsPtrDevice);

        auto logitsPtrs = BufferRange<T*>(*mLogitsPtrs);
        for (SizeType32 bi = 0; bi < mBatchSize; ++bi)
        {
            logitsPtrs[bi]
                = bufferCast<T>(*mLogitsDevice) + (mBatchSize - bi - 1) * mMaxTokensPerStep * mVocabSizePadded;
        }

        ASSERT_EQ(param.repetitionPenaltiesSize, mMaxBatchSize) << "Invalid test configuration.";
        ASSERT_EQ(param.presencePenaltiesSize, mMaxBatchSize) << "Invalid test configuration.";
        ASSERT_EQ(param.frequencyPenaltiesSize, mMaxBatchSize) << "Invalid test configuration.";
        mRepetitionPenaltiesDevice
            = mBufferManager->gpu(ITensor::makeShape({param.repetitionPenaltiesSize}), nvinfer1::DataType::kFLOAT);
        mPresencePenaltiesDevice
            = mBufferManager->gpu(ITensor::makeShape({param.presencePenaltiesSize}), nvinfer1::DataType::kFLOAT);
        mFrequencyPenaltiesDevice
            = mBufferManager->gpu(ITensor::makeShape({param.frequencyPenaltiesSize}), nvinfer1::DataType::kFLOAT);
        mBufferManager->copy(*param.repetitionPenalties, *mRepetitionPenaltiesDevice);
        mBufferManager->copy(*param.presencePenalties, *mPresencePenaltiesDevice);
        mBufferManager->copy(*param.frequencyPenalties, *mFrequencyPenaltiesDevice);
    }

    void computeReference(T const* const inLogits, T* const outLogits, int32_t const* const outputIds,
        int32_t const* const sequenceLengths, float const* const repetitionPenalties,
        float const* const presencePenalties, float const* const frequencyPenalties,
        int32_t const repetitionPenaltiesSize, int32_t const presencePenaltiesSize,
        int32_t const frequencyPenaltiesSize)
    {
        std::vector<bool> penalized(mVocabSize);
        auto const batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
        auto const tokensPerStepPtr = bufferCast<int32_t>(*mTokensPerStep);

        for (int32_t bi = 0; bi < mBatchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            for (int32_t ti = 0; ti < tokensPerStepPtr[batchSlot]; ++ti)
            {
                for (int32_t vi = 0; vi < mVocabSizePadded; ++vi)
                {
                    auto const srcIdx = ((mBatchSize - bi - 1) * mMaxTokensPerStep + ti) * mVocabSizePadded + vi;
                    auto const dstIdx = (bi * mMaxTokensPerStep + ti) * mVocabSizePadded + vi;
                    outLogits[dstIdx] = inLogits[srcIdx];
                }
            }
        }

        for (int32_t bi = 0; bi < mBatchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            for (int32_t ti = 0; ti < tokensPerStepPtr[batchSlot]; ++ti)
            {
                float repetitionPenalty
                    = repetitionPenaltiesSize > 1 ? repetitionPenalties[batchSlot] : repetitionPenalties[0];
                float presencePenalty = presencePenaltiesSize > 1 ? presencePenalties[batchSlot] : presencePenalties[0];
                float frequencyPenalty
                    = frequencyPenaltiesSize > 1 ? frequencyPenalties[batchSlot] : frequencyPenalties[0];

                std::fill(penalized.begin(), penalized.end(), false);
                size_t offset = (bi * mMaxTokensPerStep + ti) * mVocabSizePadded;
                auto const step = sequenceLengths[batchSlot];
                for (int32_t t = 0; t < step; ++t)
                {
                    auto tokenId = outputIds[batchSlot * mSequenceLength + t];
                    if (!penalized[tokenId])
                    {
                        auto logit = static_cast<float>(outLogits[offset + tokenId]);
                        outLogits[offset + tokenId] = static_cast<T>(
                            (logit < 0.0f ? logit * repetitionPenalty : logit / repetitionPenalty) - presencePenalty);
                        penalized[tokenId] = true;
                    }
                    outLogits[offset + tokenId] -= frequencyPenalty;
                }
            }
        }
    }

public:
    void runTest(RepetitionPenaltyTestCase param)
    {
        subsetup(param);

        InvokeBatchApplyPenaltyParams<T> penaltyParams{reinterpret_cast<T**>(bufferCast<int64_t>(*mLogitsPtrs)),
            bufferCast<T>(*mOutLogitsDevice), nullptr, bufferCast<int32_t>(*mPenaltyWorkspaceDevice), nullptr, nullptr,
            bufferCast<float>(*mRepetitionPenaltiesDevice), bufferCast<float>(*mPresencePenaltiesDevice),
            bufferCast<float>(*mFrequencyPenaltiesDevice), mBatchSize, 1, mSequenceLength, mVocabSize, mVocabSizePadded,
            reinterpret_cast<int32_t const**>(bufferCast<int64_t>(*mIdsPtrDevice)), nullptr,
            bufferCast<int32_t>(*mContextLengthDevice), bufferCast<int32_t>(*mSeqLengthDevice), nullptr, nullptr,
            bufferCast<int32_t>(*mBatchSlots), mMaxTokensPerStep, bufferCast<int32_t>(*mTokensPerStep), nullptr,
            mStream->get()};
        tk::invokeBatchApplyPenalty(penaltyParams);

        auto logitsOutHost = mBufferManager->copyFrom(*mOutLogitsDevice, MemoryType::kCPU);

        computeReference(bufferCast<T>(*mLogitsHost), bufferCast<T>(*mLogitsRefHost),
            bufferCast<int32_t>(*mOutputIdsHost), bufferCast<int32_t>(*mSeqLengthHost),
            bufferCast<float>(*param.repetitionPenalties), bufferCast<float>(*param.presencePenalties),
            bufferCast<float>(*param.frequencyPenalties), param.repetitionPenaltiesSize, param.presencePenaltiesSize,
            param.frequencyPenaltiesSize);

        mStream->synchronize();

        bool passed = checkResult(param.toString(), bufferCast<T>(*logitsOutHost), bufferCast<T>(*mLogitsRefHost),
            mBatchSize * mMaxTokensPerStep * mVocabSizePadded);
        EXPECT_TRUE(passed);
    }
};

TYPED_TEST_SUITE(RepetitionPenaltyTest, FloatAndHalfTypes);

TYPED_TEST(RepetitionPenaltyTest, BatchNoPenalty)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr repetitionPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*repetitionPenaltyHost)[i] = 1.0f;
        bufferCast<float>(*presencePenaltyHost)[i] = 0.0f;
        bufferCast<float>(*frequencyPenaltyHost)[i] = 0.0f;
    }
    this->runTest(RepetitionPenaltyTestCase()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setMaxInputLength(5)
                      .setRepetitionPenalties(repetitionPenaltyHost)
                      .setPresencePenalties(presencePenaltyHost)
                      .setFrequencyPenalties(frequencyPenaltyHost)
                      .setRepetitionPenaltiesSize(maxBatchSize)
                      .setPresencePenaltiesSize(maxBatchSize)
                      .setFrequencyPenaltiesSize(maxBatchSize));
}

TYPED_TEST(RepetitionPenaltyTest, BatchRepetitionLessThanOne)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr repetitionPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*repetitionPenaltyHost)[i] = 0.53f;
        bufferCast<float>(*presencePenaltyHost)[i] = 0.0f;
        bufferCast<float>(*frequencyPenaltyHost)[i] = 0.0f;
    }
    this->runTest(RepetitionPenaltyTestCase()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setMaxInputLength(5)
                      .setRepetitionPenalties(repetitionPenaltyHost)
                      .setPresencePenalties(presencePenaltyHost)
                      .setFrequencyPenalties(frequencyPenaltyHost)
                      .setRepetitionPenaltiesSize(maxBatchSize)
                      .setPresencePenaltiesSize(maxBatchSize)
                      .setFrequencyPenaltiesSize(maxBatchSize));
}

TYPED_TEST(RepetitionPenaltyTest, BatchRepetitionGreaterThaneOne)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr repetitionPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*repetitionPenaltyHost)[i] = 2.01f;
        bufferCast<float>(*presencePenaltyHost)[i] = 0.0f;
        bufferCast<float>(*frequencyPenaltyHost)[i] = 0.0f;
    }
    this->runTest(RepetitionPenaltyTestCase()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setMaxInputLength(5)
                      .setRepetitionPenalties(repetitionPenaltyHost)
                      .setPresencePenalties(presencePenaltyHost)
                      .setFrequencyPenalties(frequencyPenaltyHost)
                      .setRepetitionPenaltiesSize(maxBatchSize)
                      .setPresencePenaltiesSize(maxBatchSize)
                      .setFrequencyPenaltiesSize(maxBatchSize));
}

TYPED_TEST(RepetitionPenaltyTest, BatchRepetitionMixed)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr repetitionPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*repetitionPenaltyHost)[i] = 0.53 + i * 0.2f;
        bufferCast<float>(*presencePenaltyHost)[i] = 0.0f;
        bufferCast<float>(*frequencyPenaltyHost)[i] = 0.0f;
    }
    this->runTest(RepetitionPenaltyTestCase()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setMaxInputLength(5)
                      .setRepetitionPenalties(repetitionPenaltyHost)
                      .setPresencePenalties(presencePenaltyHost)
                      .setFrequencyPenalties(frequencyPenaltyHost)
                      .setRepetitionPenaltiesSize(maxBatchSize)
                      .setPresencePenaltiesSize(maxBatchSize)
                      .setFrequencyPenaltiesSize(maxBatchSize));
}

TYPED_TEST(RepetitionPenaltyTest, BatchPresenceMixed)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr repetitionPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*repetitionPenaltyHost)[i] = 1.0f;
        bufferCast<float>(*presencePenaltyHost)[i] = 0.53 + i * 0.2f;
        bufferCast<float>(*frequencyPenaltyHost)[i] = 0.0f;
    }
    this->runTest(RepetitionPenaltyTestCase()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setMaxInputLength(5)
                      .setRepetitionPenalties(repetitionPenaltyHost)
                      .setPresencePenalties(presencePenaltyHost)
                      .setFrequencyPenalties(frequencyPenaltyHost)
                      .setRepetitionPenaltiesSize(maxBatchSize)
                      .setPresencePenaltiesSize(maxBatchSize)
                      .setFrequencyPenaltiesSize(maxBatchSize));
}

TYPED_TEST(RepetitionPenaltyTest, BatchPresenceHasDefaultValueZero2)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr repetitionPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*repetitionPenaltyHost)[i] = 1.0f;
        bufferCast<float>(*presencePenaltyHost)[i] = i % 2 == 0 ? 1.0f : 0.0f;
        bufferCast<float>(*frequencyPenaltyHost)[i] = 0.0f;
    }
    this->runTest(RepetitionPenaltyTestCase()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setMaxInputLength(5)
                      .setRepetitionPenalties(repetitionPenaltyHost)
                      .setPresencePenalties(presencePenaltyHost)
                      .setFrequencyPenalties(frequencyPenaltyHost)
                      .setRepetitionPenaltiesSize(maxBatchSize)
                      .setPresencePenaltiesSize(maxBatchSize)
                      .setFrequencyPenaltiesSize(maxBatchSize));
}

TYPED_TEST(RepetitionPenaltyTest, BatchFrequencyMixed)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr repetitionPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*repetitionPenaltyHost)[i] = 1.0f;
        bufferCast<float>(*presencePenaltyHost)[i] = 0.0f;
        bufferCast<float>(*frequencyPenaltyHost)[i] = 0.53 + i * 0.2f;
    }
    this->runTest(RepetitionPenaltyTestCase()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setMaxInputLength(5)
                      .setRepetitionPenalties(repetitionPenaltyHost)
                      .setPresencePenalties(presencePenaltyHost)
                      .setFrequencyPenalties(frequencyPenaltyHost)
                      .setRepetitionPenaltiesSize(maxBatchSize)
                      .setPresencePenaltiesSize(maxBatchSize)
                      .setFrequencyPenaltiesSize(maxBatchSize));
}

TYPED_TEST(RepetitionPenaltyTest, BatchFrequencyHasDefaultValueZero2)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr repetitionPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*repetitionPenaltyHost)[i] = 1.0f;
        bufferCast<float>(*presencePenaltyHost)[i] = 0.0f;
        bufferCast<float>(*frequencyPenaltyHost)[i] = i % 2 == 0 ? 1.0f : 0.0f;
    }
    this->runTest(RepetitionPenaltyTestCase()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setMaxInputLength(5)
                      .setRepetitionPenalties(repetitionPenaltyHost)
                      .setPresencePenalties(presencePenaltyHost)
                      .setFrequencyPenalties(frequencyPenaltyHost)
                      .setRepetitionPenaltiesSize(maxBatchSize)
                      .setPresencePenaltiesSize(maxBatchSize)
                      .setFrequencyPenaltiesSize(maxBatchSize));
}

TYPED_TEST(RepetitionPenaltyTest, PenaltyTypeRepetitionPresence)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr repetitionPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*repetitionPenaltyHost)[i] = 0.53 + i * 0.2f;
        bufferCast<float>(*presencePenaltyHost)[i] = 0.53 + i * 0.2f;
        bufferCast<float>(*frequencyPenaltyHost)[i] = 0.0f;
    }
    this->runTest(RepetitionPenaltyTestCase()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setMaxInputLength(5)
                      .setRepetitionPenalties(repetitionPenaltyHost)
                      .setPresencePenalties(presencePenaltyHost)
                      .setFrequencyPenalties(frequencyPenaltyHost)
                      .setRepetitionPenaltiesSize(maxBatchSize)
                      .setPresencePenaltiesSize(maxBatchSize)
                      .setFrequencyPenaltiesSize(maxBatchSize));
}

TYPED_TEST(RepetitionPenaltyTest, PenaltyTypeRepetitionFrequency)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr repetitionPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*repetitionPenaltyHost)[i] = 0.53 + i * 0.2f;
        bufferCast<float>(*presencePenaltyHost)[i] = 0.0f;
        bufferCast<float>(*frequencyPenaltyHost)[i] = 0.53 + i * 0.2f;
    }
    this->runTest(RepetitionPenaltyTestCase()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setMaxInputLength(5)
                      .setRepetitionPenalties(repetitionPenaltyHost)
                      .setPresencePenalties(presencePenaltyHost)
                      .setFrequencyPenalties(frequencyPenaltyHost)
                      .setRepetitionPenaltiesSize(maxBatchSize)
                      .setPresencePenaltiesSize(maxBatchSize)
                      .setFrequencyPenaltiesSize(maxBatchSize));
}

TYPED_TEST(RepetitionPenaltyTest, PenaltyTypePresenceFrequency)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr repetitionPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*repetitionPenaltyHost)[i] = 1.0f;
        bufferCast<float>(*presencePenaltyHost)[i] = 0.53 + i * 0.2f;
        bufferCast<float>(*frequencyPenaltyHost)[i] = 0.53 + i * 0.2f;
    }
    this->runTest(RepetitionPenaltyTestCase()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setMaxInputLength(5)
                      .setRepetitionPenalties(repetitionPenaltyHost)
                      .setPresencePenalties(presencePenaltyHost)
                      .setFrequencyPenalties(frequencyPenaltyHost)
                      .setRepetitionPenaltiesSize(maxBatchSize)
                      .setPresencePenaltiesSize(maxBatchSize)
                      .setFrequencyPenaltiesSize(maxBatchSize));
}

TYPED_TEST(RepetitionPenaltyTest, PenaltyTypeFull)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr repetitionPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*repetitionPenaltyHost)[i] = 0.53 + i * 0.2f;
        bufferCast<float>(*presencePenaltyHost)[i] = 0.53 + i * 0.2f;
        bufferCast<float>(*frequencyPenaltyHost)[i] = 0.53 + i * 0.2f;
    }
    this->runTest(RepetitionPenaltyTestCase()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setMaxInputLength(5)
                      .setRepetitionPenalties(repetitionPenaltyHost)
                      .setPresencePenalties(presencePenaltyHost)
                      .setFrequencyPenalties(frequencyPenaltyHost)
                      .setRepetitionPenaltiesSize(maxBatchSize)
                      .setPresencePenaltiesSize(maxBatchSize)
                      .setFrequencyPenaltiesSize(maxBatchSize));
}

TYPED_TEST(RepetitionPenaltyTest, PenaltyTypeFullTokensPerStep)
{
    int32_t batchSize = 6;
    int32_t maxBatchSize = 2 * batchSize;
    TensorPtr repetitionPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < maxBatchSize; ++i)
    {
        bufferCast<float>(*repetitionPenaltyHost)[i] = 0.53 + i * 0.2f;
        bufferCast<float>(*presencePenaltyHost)[i] = 0.53 + i * 0.2f;
        bufferCast<float>(*frequencyPenaltyHost)[i] = 0.53 + i * 0.2f;
    }
    this->runTest(RepetitionPenaltyTestCase()
                      .setBatchSize(batchSize)
                      .setVocabSize(4)
                      .setMaxInputLength(5)
                      .setRepetitionPenalties(repetitionPenaltyHost)
                      .setPresencePenalties(presencePenaltyHost)
                      .setFrequencyPenalties(frequencyPenaltyHost)
                      .setRepetitionPenaltiesSize(maxBatchSize)
                      .setPresencePenaltiesSize(maxBatchSize)
                      .setFrequencyPenaltiesSize(maxBatchSize)
                      .setMaxTokensPerStep(4));
}

struct MinLengthPenaltyTestParams
{
    int32_t batchSize;
    int32_t vocabSize;
    int32_t maxSeqLength;
    int32_t maxTokensPerStep{1};

    MinLengthPenaltyTestParams& setBatchSize(int32_t bs)
    {
        batchSize = bs;
        return *this;
    }

    MinLengthPenaltyTestParams& setVocabSize(int32_t vs)
    {
        vocabSize = vs;
        return *this;
    }

    MinLengthPenaltyTestParams& setMaxSeqLength(int32_t sl)
    {
        maxSeqLength = sl;
        return *this;
    }

    MinLengthPenaltyTestParams& setMaxTokensPerStep(int32_t ts)
    {
        maxTokensPerStep = ts;
        return *this;
    }

    std::string toString() const
    {
        return tc::fmtstr("MinLengthPenaltyTestParams[batch=%d, vocab=%d, maxSeqLen=%d, maxTokensPerStep=%d]",
            batchSize, vocabSize, maxSeqLength, maxTokensPerStep);
    }
};

template <typename T>
class MinLengthPenaltyTest : public SamplingKernelTest<T>
{
protected:
    // Set up test
    int32_t mBatchSize;
    int32_t mMaxBatchSize;
    int32_t mVocabSize;
    int32_t mVocabSizePadded;
    int32_t mMaxInputLength;
    int32_t mSequenceLength;
    int32_t mMaxTokensPerStep;

    using SamplingKernelTest<T>::mBufferManager;
    using SamplingKernelTest<T>::mStream;
    using SamplingKernelTest<T>::mLogitsHost;

    TensorPtr mLogitsDevice;
    TensorPtr mOutLogitsDevice;
    TensorPtr mLogitsRefHost;
    TensorPtr mLogitsPtrs;
    TensorPtr mPenaltyWorkspaceDevice;

    TensorPtr mTokensPerStep;

    TensorPtr mContextLengthHost;
    TensorPtr mContextLengthDevice;

    TensorPtr mSeqLengthHost;
    TensorPtr mSeqLengthDevice;

    TensorPtr mMinLengthHost;
    TensorPtr mMinLengthDevice;

    TensorPtr mEndIdsHost;
    TensorPtr mEndIdsDevice;

    TensorPtr mBatchSlots;

    void subsetup(MinLengthPenaltyTestParams param)
    {
        auto const dataType = TRTDataType<T>::value;
        auto const ptrType = TRTDataType<T*>::value;

        mBatchSize = param.batchSize;
        mMaxBatchSize = 2 * mBatchSize;
        mVocabSize = param.vocabSize;
        mVocabSizePadded = padVocabSize(mVocabSize);
        mMaxInputLength = param.maxSeqLength;
        mSequenceLength = 2 * mMaxInputLength; // input + output
        mMaxTokensPerStep = param.maxTokensPerStep;

        mLogitsHost
            = BufferManager::pinned(ITensor::makeShape({mBatchSize, mMaxTokensPerStep, mVocabSizePadded}), dataType);
        mLogitsRefHost
            = BufferManager::pinned(ITensor::makeShape({mBatchSize, mMaxTokensPerStep, mVocabSizePadded}), dataType);
        mLogitsDevice
            = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mMaxTokensPerStep, mVocabSizePadded}), dataType);
        mOutLogitsDevice
            = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mMaxTokensPerStep, mVocabSizePadded}), dataType);
        mLogitsPtrs = BufferManager::pinned(ITensor::makeShape({mBatchSize}), ptrType);

        mPenaltyWorkspaceDevice = mBufferManager->gpu(
            ITensor::makeShape({mBatchSize, mMaxTokensPerStep, mVocabSize}), nvinfer1::DataType::kINT32);

        mTokensPerStep = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);

        mSeqLengthHost = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        mSeqLengthDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);

        mContextLengthHost = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        mContextLengthDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);

        mMinLengthHost = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        mMinLengthDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);

        mEndIdsHost = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        mEndIdsDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);

        mBatchSlots = BufferManager::pinned(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);

        auto batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
        for (SizeType32 bi = 0; bi < mBatchSize; ++bi)
        {
            batchSlotsPtr[bi] = 2 * bi;
        }

        initLogitsAndBias(bufferCast<T>(*mLogitsHost), static_cast<T*>(nullptr), mBatchSize, mMaxTokensPerStep,
            mVocabSize, mVocabSizePadded);
        initRandomInt(bufferCast<int32_t>(*mContextLengthHost), mMaxBatchSize, 0, mMaxInputLength);
        initRandomInt(bufferCast<int32_t>(*mMinLengthHost), mMaxBatchSize, 1, mMaxInputLength);
        initRandomInt(bufferCast<int32_t>(*mEndIdsHost), mMaxBatchSize, 0, mVocabSize);

        trk::invokeFill(*mLogitsRefHost, T{0.0f}, *mStream);
        trk::invokeFill(*mOutLogitsDevice, T{0.0f}, *mStream);

        auto seqLengthHostPtr = bufferCast<int32_t>(*mSeqLengthHost);
        auto contextLengthHostPtr = bufferCast<int32_t>(*mContextLengthHost);
        auto minLengthHostPtr = bufferCast<int32_t>(*mMinLengthHost);
        auto tokensPerStepPtr = bufferCast<int32_t>(*mTokensPerStep);
        for (SizeType32 bi = 0; bi < mMaxBatchSize; bi++)
        {
            // Current generated seq len is randomly either smaller than min length or larger
            auto const generatedSeqLen = std::max(0,
                std::min(
                    static_cast<int32_t>(minLengthHostPtr[bi] + 2 * std::pow(-1, std::rand() % 2)), mMaxInputLength));
            seqLengthHostPtr[bi] = contextLengthHostPtr[bi] + generatedSeqLen;
            tokensPerStepPtr[bi] = (std::rand() % mMaxTokensPerStep) + 1;
        }

        mBufferManager->copy(*mLogitsHost, *mLogitsDevice);
        mBufferManager->copy(*mMinLengthHost, *mMinLengthDevice);
        mBufferManager->copy(*mContextLengthHost, *mContextLengthDevice);
        mBufferManager->copy(*mSeqLengthHost, *mSeqLengthDevice);
        mBufferManager->copy(*mEndIdsHost, *mEndIdsDevice);

        auto logitsPtrs = BufferRange<T*>(*mLogitsPtrs);
        for (SizeType32 bi = 0; bi < mBatchSize; ++bi)
        {
            logitsPtrs[bi]
                = bufferCast<T>(*mLogitsDevice) + (mBatchSize - bi - 1) * mMaxTokensPerStep * mVocabSizePadded;
        }
    }

    void computeReference(T const* const inLogits, T* const outLogits, int32_t const* const minSeqLen,
        int32_t const* const endIds, int32_t const* const sequenceLengths, int32_t const* const contextLengths)
    {
        bool const IS_FP16 = std::is_same<T, half>::value;
        T const MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
        auto const batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
        auto const tokensPerStepPtr = bufferCast<int32_t>(*mTokensPerStep);

        for (int32_t bi = 0; bi < mBatchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            for (int32_t ti = 0; ti < tokensPerStepPtr[batchSlot]; ++ti)
            {
                for (int32_t vi = 0; vi < mVocabSizePadded; ++vi)
                {
                    auto const srcIdx = ((mBatchSize - bi - 1) * mMaxTokensPerStep + ti) * mVocabSizePadded + vi;
                    auto const dstIdx = (bi * mMaxTokensPerStep + ti) * mVocabSizePadded + vi;
                    outLogits[dstIdx] = inLogits[srcIdx];
                }
            }
        }

        for (int32_t bi = 0; bi < mBatchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            for (int32_t ti = 0; ti < tokensPerStepPtr[batchSlot]; ++ti)
            {
                auto const generatedSeqLen = sequenceLengths[batchSlot] - contextLengths[batchSlot];
                auto const endId = endIds[batchSlot];
                if (generatedSeqLen < minSeqLen[batchSlot])
                {
                    outLogits[(bi * mMaxTokensPerStep + ti) * mVocabSizePadded + endId] = -MAX_T_VAL;
                }
            }
        }
    }

public:
    void runTest(MinLengthPenaltyTestParams param)
    {
        subsetup(param);

        InvokeBatchApplyPenaltyParams<T> penaltyParams{reinterpret_cast<T**>(bufferCast<int64_t>(*mLogitsPtrs)),
            bufferCast<T>(*mOutLogitsDevice), nullptr, bufferCast<int32_t>(*mPenaltyWorkspaceDevice), nullptr, nullptr,
            nullptr, nullptr, nullptr, mBatchSize, 1, mSequenceLength, mVocabSize, mVocabSizePadded, nullptr, nullptr,
            bufferCast<int32_t>(*mContextLengthDevice), bufferCast<int32_t>(*mSeqLengthDevice),
            bufferCast<int32_t>(*mMinLengthDevice), bufferCast<int32_t>(*mEndIdsDevice),
            bufferCast<int32_t>(*mBatchSlots), mMaxTokensPerStep, bufferCast<int32_t>(*mTokensPerStep), nullptr,
            mStream->get()};
        tk::invokeBatchApplyPenalty(penaltyParams);

        mStream->synchronize();

        computeReference(bufferCast<T>(*mLogitsHost), bufferCast<T>(*mLogitsRefHost),
            bufferCast<int32_t>(*mMinLengthHost), bufferCast<int32_t>(*mEndIdsHost),
            bufferCast<int32_t>(*mSeqLengthHost), bufferCast<int32_t>(*mContextLengthHost));

        auto logitsOutHost = mBufferManager->copyFrom(*mOutLogitsDevice, MemoryType::kCPU);

        mStream->synchronize();

        bool passed = checkResult(param.toString(), bufferCast<T>(*logitsOutHost), bufferCast<T>(*mLogitsRefHost),
            mBatchSize * mMaxTokensPerStep * mVocabSizePadded);
        EXPECT_TRUE(passed);
    }
};

TYPED_TEST_SUITE(MinLengthPenaltyTest, FloatAndHalfTypes);

TYPED_TEST(MinLengthPenaltyTest, BatchMaxSeqLen2)
{
    this->runTest(MinLengthPenaltyTestParams().setBatchSize(16).setVocabSize(10).setMaxSeqLength(2));
}

TYPED_TEST(MinLengthPenaltyTest, BatchMaxSeqLen64)
{
    this->runTest(MinLengthPenaltyTestParams().setBatchSize(16).setVocabSize(51200).setMaxSeqLength(64));
}

TYPED_TEST(MinLengthPenaltyTest, BatchMaxSeqLen64TokensPerStep)
{
    this->runTest(
        MinLengthPenaltyTestParams().setBatchSize(16).setVocabSize(51200).setMaxSeqLength(64).setMaxTokensPerStep(4));
}

// When endId=-1 is not handled properly, minLength penalty kernel accesses out-of-bounds memory just
// prior to mOutLogitsDevice. Input logits are laid out just before mOutLogitsDevice in memory in this test
// to show that last input logit is corrupted when endId=-1 is not handled correctly.
template <typename T>
class MinLengthPenaltyOOBSafetyTest : public SamplingKernelTest<T>
{
protected:
    // Set up test
    int32_t mBatchSize;
    int32_t mMaxBatchSize;
    int32_t mVocabSize;
    int32_t mVocabSizePadded;
    int32_t mMaxInputLength;
    int32_t mSequenceLength;
    int32_t mMaxTokensPerStep;

    using SamplingKernelTest<T>::mBufferManager;
    using SamplingKernelTest<T>::mStream;
    using SamplingKernelTest<T>::mLogitsHost;

    TensorPtr mLogitsDevice;
    TensorPtr mOutLogitsDevice;
    TensorPtr mLogitsRefHost;
    TensorPtr mLogitsPtrs;

    TensorPtr mContextLengthHost;
    TensorPtr mContextLengthDevice;

    TensorPtr mSeqLengthHost;
    TensorPtr mSeqLengthDevice;

    TensorPtr mMinLengthHost;
    TensorPtr mMinLengthDevice;

    TensorPtr mEndIdsHost;
    TensorPtr mEndIdsDevice;

    TensorPtr mBatchSlots;

    void subsetupMemorySafety()
    {
        auto const dataType = TRTDataType<T>::value;
        auto const ptrType = TRTDataType<T*>::value;

        mBatchSize = 1;
        mMaxBatchSize = 1;
        // Set vocabSize to be same as vocabSizePadded so that no mask value is set
        // in output logits because of vocabSize < vocabSizePadded. This means outputs
        // logits are set to input logits as is without any substitution.
        mVocabSize = 16;
        mVocabSizePadded = padVocabSize(mVocabSize);
        mMaxInputLength = 5;
        mSequenceLength = 10; // input + output
        mMaxTokensPerStep = 1;

        // Create a large chunk of memory which hosts both input and output logits contiguously.
        // Goal is to show that last input logit is corrupted when endId=-1 is not handled correctly.
        mLogitsDevice
            = mBufferManager->gpu(ITensor::makeShape({2 * mBatchSize, mMaxTokensPerStep, mVocabSizePadded}), dataType);
        trk::invokeFill(*mLogitsDevice, T{0.1f}, *mStream);
        mOutLogitsDevice = ITensor::slice(mLogitsDevice, mBatchSize);
        trk::invokeFill(*mOutLogitsDevice, T{0.2f}, *mStream);

        mLogitsPtrs = BufferManager::pinned(ITensor::makeShape({mBatchSize}), ptrType);
        auto logitsPtrs = BufferRange<T*>(*mLogitsPtrs);
        for (SizeType32 bi = 0; bi < mBatchSize; ++bi)
        {
            logitsPtrs[bi]
                = bufferCast<T>(*mLogitsDevice) + (mBatchSize - bi - 1) * mMaxTokensPerStep * mVocabSizePadded;
        }

        // Defines currentStep.
        mSeqLengthHost = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        initConstant<int>(bufferCast<int32_t>(*mSeqLengthHost), mMaxBatchSize, 3);
        mSeqLengthDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        mBufferManager->copy(*mSeqLengthHost, *mSeqLengthDevice);

        // Defines inputLength.
        mContextLengthHost = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        initConstant<int>(bufferCast<int32_t>(*mContextLengthHost), mMaxBatchSize, 2);
        mContextLengthDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        mBufferManager->copy(*mContextLengthHost, *mContextLengthDevice);

        // Defines minLength.
        mMinLengthHost = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        initConstant<int>(bufferCast<int32_t>(*mMinLengthHost), mMaxBatchSize, 10);
        mMinLengthDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        mBufferManager->copy(*mMinLengthHost, *mMinLengthDevice);

        // Defines endIds.
        mEndIdsHost = BufferManager::pinned(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        initConstant<int>(bufferCast<int32_t>(*mEndIdsHost), mMaxBatchSize, -1);
        mEndIdsDevice = mBufferManager->gpu(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        mBufferManager->copy(*mEndIdsHost, *mEndIdsDevice);

        mBatchSlots = BufferManager::pinned(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);
        auto batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
        for (SizeType32 bi = 0; bi < mBatchSize; ++bi)
        {
            batchSlotsPtr[bi] = 2 * bi;
        }
    }

public:
    void runMemorySafetyTest()
    {
        subsetupMemorySafety();
        InvokeBatchApplyPenaltyParams<T> penaltyParams{
            /*inputLogits=*/reinterpret_cast<T**>(bufferCast<int64_t>(*mLogitsPtrs)),
            /*outputLogits=*/bufferCast<T>(*mOutLogitsDevice),
            /*biases=*/nullptr,
            /*penaltyWorkspace=*/nullptr,
            /*penaltyWorkspacePrev=*/nullptr,
            /*temperatures=*/nullptr,
            /*repetitionPenalties=*/nullptr,
            /*presencePenalties=*/nullptr,
            /*frequencyPenalties=*/nullptr,
            /*batchSize=*/mBatchSize,
            /*beamWidth=*/1,
            /*maxSeqLen=*/mSequenceLength,
            /*vocabSize=*/mVocabSize,
            /*vocabSizePadded=*/mVocabSizePadded,
            /*outputIdsPtr=*/nullptr,
            /*parentIdsPtr=*/nullptr,
            /*inputLengths=*/bufferCast<int32_t>(*mContextLengthDevice),
            /*sequenceLengths=*/bufferCast<int32_t>(*mSeqLengthDevice),
            /*minLengths=*/bufferCast<int32_t>(*mMinLengthDevice),
            /*endIds=*/bufferCast<int32_t>(*mEndIdsDevice),
            /*batchSlots=*/bufferCast<int32_t>(*mBatchSlots),
            /*maxTokensPerStep=*/mMaxTokensPerStep,
            /*tokensPerStep=*/nullptr,
            /*finished=*/nullptr,
            /*stream=*/mStream->get()};
        tk::invokeBatchApplyPenalty(penaltyParams);
        mStream->synchronize();

        auto logitsOutHost = mBufferManager->copyFrom(*mOutLogitsDevice, MemoryType::kCPU);
        auto logitsInHost = mBufferManager->copyFrom(*mLogitsDevice, MemoryType::kCPU);
        auto minLengthsHost = mBufferManager->copyFrom(*mMinLengthDevice, MemoryType::kCPU);
        auto inputLengthsHost = mBufferManager->copyFrom(*mContextLengthDevice, MemoryType::kCPU);
        auto endIdsHost = mBufferManager->copyFrom(*mEndIdsDevice, MemoryType::kCPU);
        auto seqLengthsHost = mBufferManager->copyFrom(*mSeqLengthDevice, MemoryType::kCPU);
        mStream->synchronize();

        // Verify that output logits were modified to input logits.
        auto logitsOutPtr = bufferCast<T>(*logitsOutHost);
        for (int32_t i = 0; i < mBatchSize * mMaxTokensPerStep * mVocabSizePadded; ++i)
        {
            // All output logits must be updated to be same as input logits.
            // No mask value is set anywhere because vocabSize is same as vocabSizePadded.
            EXPECT_EQ(static_cast<T>(logitsOutPtr[i]), T{0.1f})
                << "Output logits were modified to unexpected value at index " << i;
        }

        // Verify that batch slots were not modified.
        auto batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
        for (SizeType32 bi = 0; bi < mBatchSize; ++bi)
        {
            EXPECT_EQ(batchSlotsPtr[bi], 2 * bi) << "Batch slots were modified at index " << bi;
        }

        // Verify that inLogits were not modified.
        auto logitsInPtr = bufferCast<T>(*logitsInHost);
        for (int32_t i = 0; i < mBatchSize * mMaxTokensPerStep * mVocabSizePadded; ++i)
        {
            EXPECT_EQ(static_cast<T>(logitsInPtr[i]), T{0.1f})
                << "Input logits were modified to unexpected value at index " << i;
        }

        // Verify that minLengths were not modified.
        auto minLengthsPtr = bufferCast<int32_t>(*minLengthsHost);
        for (int32_t i = 0; i < mMaxBatchSize; ++i)
        {
            EXPECT_EQ(minLengthsPtr[i], 10) << "MinLengths were modified at index " << i;
        }

        // Verify that inputLengths were not modified.
        auto inputLengthsPtr = bufferCast<int32_t>(*inputLengthsHost);
        for (int32_t i = 0; i < mMaxBatchSize; ++i)
        {
            EXPECT_EQ(inputLengthsPtr[i], 2) << "InputLengths were modified at index " << i;
        }

        // Verify that endIds were not modified.
        auto endIdsPtr = bufferCast<int32_t>(*endIdsHost);
        for (int32_t i = 0; i < mMaxBatchSize; ++i)
        {
            EXPECT_EQ(endIdsPtr[i], -1) << "EndIds were modified at index " << i;
        }

        // Verify that seqLengths were not modified.
        auto seqLengthsPtr = bufferCast<int32_t>(*seqLengthsHost);
        for (int32_t i = 0; i < mMaxBatchSize; ++i)
        {
            EXPECT_EQ(seqLengthsPtr[i], 3) << "SeqLengths were modified at index " << i;
        }
    }
};

TYPED_TEST_SUITE(MinLengthPenaltyOOBSafetyTest, FloatAndHalfTypes);

TYPED_TEST(MinLengthPenaltyOOBSafetyTest, VerifyMemorySafetyForNegativeEndId)
{
    this->runMemorySafetyTest();
}

} // namespace
