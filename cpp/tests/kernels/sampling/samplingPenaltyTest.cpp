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
#include "tests/kernels/sampling/samplingTest.h"

using namespace tensorrt_llm::tests::kernels::sampling;

namespace
{

namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;

using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::kernels;

struct TemperatureTestParam
{
    int32_t batchSize;
    int32_t vocabSize;
    TensorPtr temperatures;
    int32_t temperaturesSize;

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
void initLogitsAndBias(
    T* logits, T* bias, const size_t batchSize, const size_t vocabSize, const size_t mVocabSizepadded)
{
    initRandom(logits, batchSize * mVocabSizepadded, -5.0f, 5.0f);
    if (bias != nullptr)
    {
        initRandom(bias, vocabSize, -5.0f, 5.0f);
    }
    bool is_half = sizeof(T) == 2;
    for (size_t i = 0; i < batchSize; ++i)
    {
        for (size_t j = 0; j < mVocabSizepadded; ++j)
        {
            if (j >= vocabSize)
            {
                logits[i * mVocabSizepadded + j] = static_cast<T>(is_half ? -65504.f : -FLT_MAX);
                if (bias != nullptr && i == 0)
                {
                    bias[j] = (T) 0.0f;
                }
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
    int32_t mBatchSize;
    int32_t mVocabSize;
    int32_t mVocabSizePadded;

    using SamplingKernelTest<T>::mBufferManager;
    using SamplingKernelTest<T>::mStream;
    using SamplingKernelTest<T>::mLogitsHost;

    TensorPtr mLogitsDevice;
    TensorPtr mPenaltyWorkspaceDevice;
    TensorPtr mBiasHost;
    TensorPtr mBiasDevice;
    TensorPtr mTemperaturesDevice;

    void subsetup(const TemperatureTestParam& param)
    {
        mBatchSize = param.batchSize;
        mVocabSize = param.vocabSize;
        mVocabSizePadded = padVocabSize(mVocabSize);

        mLogitsHost = mBufferManager->pinned(ITensor::makeShape({mBatchSize, mVocabSizePadded}),
            std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);
        mLogitsDevice = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mVocabSizePadded}),
            std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);

        mPenaltyWorkspaceDevice
            = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mVocabSize}), nvinfer1::DataType::kINT32);

        mBiasHost = mBufferManager->pinned(ITensor::makeShape({mVocabSizePadded}),
            std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);
        mBiasDevice = mBufferManager->gpu(ITensor::makeShape({mVocabSizePadded}),
            std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);

        initLogitsAndBias(
            bufferCast<T>(*mLogitsHost), bufferCast<T>(*mBiasHost), mBatchSize, mVocabSize, mVocabSizePadded);

        mBufferManager->copy(*mLogitsHost, *mLogitsDevice);
        mBufferManager->copy(*mBiasHost, *mBiasDevice);

        ASSERT_EQ(param.temperaturesSize, param.batchSize) << "Invalid test configuration.";
        mTemperaturesDevice
            = mBufferManager->gpu(ITensor::makeShape({param.temperaturesSize}), nvinfer1::DataType::kFLOAT);
        mBufferManager->copy(*param.temperatures, *mTemperaturesDevice);
    }

    void computeReference(T* logits, const T* bias, const float* temperatures, const size_t temperaturesSize)
    {
        const bool IS_FP16 = std::is_same<T, half>::value;
        const T MAX_T_VAL = (IS_FP16) ? 65504.F : FLT_MAX;
        for (size_t i = 0; i < mBatchSize; ++i)
        {
            float temperature = temperatures[i];
            ASSERT_GT(temperature, 0.0f) << "temperature should be positive but got " << temperature;
            for (size_t j = 0; j < mVocabSizePadded; ++j)
            {
                size_t index = i * mVocabSizePadded + j;
                float logit = static_cast<float>(logits[index]);
                if (j < mVocabSize && bias != nullptr)
                {
                    logit += static_cast<float>(bias[j]);
                }
                logits[index] = j < mVocabSize ? static_cast<T>(logit / temperature) : -MAX_T_VAL;
            }
        }
    }

public:
    void runTest(TemperatureTestParam param)
    {
        subsetup(param);
        // Do test
        InvokeBatchApplyPenaltyParams<T> penalty_params{bufferCast<T>(*mLogitsDevice), bufferCast<T>(*mBiasDevice),
            bufferCast<int32_t>(*mPenaltyWorkspaceDevice), nullptr, bufferCast<float>(*mTemperaturesDevice), nullptr,
            nullptr, nullptr, false, mBatchSize, 1, 1, mVocabSize, mVocabSizePadded, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, mStream->get()};
        tk::invokeBatchApplyPenalty(penalty_params);
        auto logitsOutHost = mBufferManager->copyFrom(*mLogitsDevice, MemoryType::kCPU);

        mStream->synchronize();

        computeReference(bufferCast<T>(*mLogitsHost), bufferCast<T>(*mBiasHost), bufferCast<float>(*param.temperatures),
            param.temperaturesSize);

        bool passed = checkResult(param.toString(), bufferCast<T>(*logitsOutHost), bufferCast<T>(*mLogitsHost),
            mBatchSize * mVocabSizePadded);
        EXPECT_TRUE(passed);
    }
};

TYPED_TEST_SUITE(TemperaturePenaltyTest, FloatAndHalfTypes);

TYPED_TEST(TemperaturePenaltyTest, NoPenalty)
{
    int32_t batchSize = 6;
    TensorPtr temperaturesHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
    {
        bufferCast<float>(*temperaturesHost)[i] = 1.0f;
    }
    this->runTest(
        TemperatureTestParam().setBatchSize(batchSize).setVocabSize(4).setTemperaturesSize(batchSize).setTemperatures(
            temperaturesHost));
}

TYPED_TEST(TemperaturePenaltyTest, LessThanOne)
{
    int32_t batchSize = 6;
    TensorPtr temperaturesHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
    {
        bufferCast<float>(*temperaturesHost)[i] = 0.53f;
    }
    this->runTest(
        TemperatureTestParam().setBatchSize(batchSize).setVocabSize(4).setTemperaturesSize(batchSize).setTemperatures(
            temperaturesHost));
}

TYPED_TEST(TemperaturePenaltyTest, GreaterThaneOne)
{
    int32_t batchSize = 6;
    TensorPtr temperaturesHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
    {
        bufferCast<float>(*temperaturesHost)[i] = 2.01f;
    }
    this->runTest(
        TemperatureTestParam().setBatchSize(batchSize).setVocabSize(4).setTemperaturesSize(batchSize).setTemperatures(
            temperaturesHost));
}

TYPED_TEST(TemperaturePenaltyTest, Mixed)
{
    int32_t batchSize = 6;
    TensorPtr temperaturesHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
    {
        bufferCast<float>(*temperaturesHost)[i] = 0.53f + 0.2f * i;
    }
    this->runTest(
        TemperatureTestParam().setBatchSize(batchSize).setVocabSize(4).setTemperaturesSize(batchSize).setTemperatures(
            temperaturesHost));
}

TYPED_TEST(TemperaturePenaltyTest, LargeVocab)
{
    int32_t batchSize = 6;
    TensorPtr temperaturesHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
    {
        bufferCast<float>(*temperaturesHost)[i] = 0.53f + 0.2f * i;
    }
    this->runTest(TemperatureTestParam()
                      .setBatchSize(batchSize)
                      .setVocabSize(50001)
                      .setTemperaturesSize(batchSize)
                      .setTemperatures(temperaturesHost));
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
    int32_t mVocabSize;
    int32_t mVocabSizePadded;
    int32_t mMaxInputLength;
    int32_t mSequenceLength;

    using SamplingKernelTest<T>::mBufferManager;
    using SamplingKernelTest<T>::mStream;
    using SamplingKernelTest<T>::mLogitsHost;

    TensorPtr mLogitsDevice;
    TensorPtr mPenaltyWorkspaceDevice;

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

    void subsetup(RepetitionPenaltyTestCase param)
    {
        mBatchSize = param.batchSize;
        mVocabSize = param.vocabSize;
        mVocabSizePadded = padVocabSize(mVocabSize);
        mMaxInputLength = param.maxInputLength;
        mSequenceLength = 2 * mMaxInputLength; // input + output

        mLogitsHost = mBufferManager->pinned(ITensor::makeShape({mBatchSize, mVocabSizePadded}),
            std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);
        mLogitsDevice = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mVocabSizePadded}),
            std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);

        mPenaltyWorkspaceDevice
            = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mVocabSize}), nvinfer1::DataType::kINT32);

        mOutputIdsHost
            = mBufferManager->pinned(ITensor::makeShape({mBatchSize, mSequenceLength}), nvinfer1::DataType::kINT32);
        mOutputIdsDevice
            = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mSequenceLength}), nvinfer1::DataType::kINT32);

        mSeqLengthHost = mBufferManager->pinned(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);
        mSeqLengthDevice = mBufferManager->gpu(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);

        mContextLengthHost = mBufferManager->pinned(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);
        mContextLengthDevice = mBufferManager->gpu(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);

        mIdsPtrHost = mBufferManager->pinned(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT64);
        mIdsPtrDevice = mBufferManager->pinned(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT64);

        initLogitsAndBias(
            bufferCast<T>(*mLogitsHost), static_cast<T*>(nullptr), mBatchSize, mVocabSize, mVocabSizePadded);
        initRandomInt(bufferCast<int32_t>(*mOutputIdsHost), mSequenceLength * mBatchSize, 0, mVocabSize);
        initRandomInt(bufferCast<int32_t>(*mSeqLengthHost), mBatchSize, 1, mSequenceLength);
        for (size_t i = 0; i < mBatchSize; ++i)
        {
            bufferCast<int32_t>(*mContextLengthHost)[i] = bufferCast<int32_t>(*mSeqLengthHost)[i];
        }

        auto idsPtrHostPtr = reinterpret_cast<void**>(bufferCast<int64_t>(*mIdsPtrHost));
        auto outputIdsDevicePtr = bufferCast<int32_t>(*mOutputIdsDevice);
        for (SizeType bi = 0; bi < mBatchSize; bi++)
        {
            idsPtrHostPtr[bi] = outputIdsDevicePtr + bi * mSequenceLength;
        }

        mBufferManager->copy(*mLogitsHost, *mLogitsDevice);
        mBufferManager->copy(*mOutputIdsHost, *mOutputIdsDevice);
        mBufferManager->copy(*mContextLengthHost, *mContextLengthDevice);
        mBufferManager->copy(*mSeqLengthHost, *mSeqLengthDevice);
        mBufferManager->copy(*mIdsPtrHost, *mIdsPtrDevice);

        ASSERT_EQ(param.repetitionPenaltiesSize, param.batchSize) << "Invalid test configuration.";
        ASSERT_EQ(param.presencePenaltiesSize, param.batchSize) << "Invalid test configuration.";
        ASSERT_EQ(param.frequencyPenaltiesSize, param.batchSize) << "Invalid test configuration.";
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

    void computeReference(T* logits, const int* outputIds, const int* sequenceLengths, const float* repetitionPenalties,
        const float* presencePenalties, const float* frequencyPenalties, const int32_t repetitionPenaltiesSize,
        const int32_t presencePenaltiesSize, const int32_t frequencyPenaltiesSize)
    {
        std::vector<bool> penalized(mVocabSize);
        for (int32_t bi = 0; bi < mBatchSize; ++bi)
        {
            float repetitionPenalty = repetitionPenaltiesSize > 1 ? repetitionPenalties[bi] : repetitionPenalties[0];
            float presencePenalty = presencePenaltiesSize > 1 ? presencePenalties[bi] : presencePenalties[0];
            float frequencyPenalty = frequencyPenaltiesSize > 1 ? frequencyPenalties[bi] : frequencyPenalties[0];

            std::fill(penalized.begin(), penalized.end(), false);
            size_t offset = bi * mVocabSizePadded;
            const auto step = sequenceLengths[bi];
            for (int32_t t = 0; t < step; ++t)
            {
                int tokenId = outputIds[bi * mSequenceLength + t];
                if (!penalized[tokenId])
                {
                    float logit = static_cast<float>(logits[offset + tokenId]);
                    logits[offset + tokenId] = static_cast<T>(
                        (logit < 0.0f ? logit * repetitionPenalty : logit / repetitionPenalty) - presencePenalty);
                    penalized[tokenId] = true;
                }
                logits[offset + tokenId] -= frequencyPenalty;
            }
        }
    }

public:
    void runTest(RepetitionPenaltyTestCase param)
    {
        subsetup(param);
        InvokeBatchApplyPenaltyParams<T> penalty_params{bufferCast<T>(*mLogitsDevice), nullptr,
            bufferCast<int32_t>(*mPenaltyWorkspaceDevice), nullptr, nullptr,
            bufferCast<float>(*mRepetitionPenaltiesDevice), bufferCast<float>(*mPresencePenaltiesDevice),
            bufferCast<float>(*mFrequencyPenaltiesDevice), true, mBatchSize, 1, mSequenceLength, mVocabSize,
            mVocabSizePadded, reinterpret_cast<const int32_t**>(bufferCast<int64_t>(*mIdsPtrDevice)), nullptr,
            bufferCast<int32_t>(*mContextLengthDevice), bufferCast<int32_t>(*mSeqLengthDevice), nullptr, nullptr,
            mStream->get()};
        tk::invokeBatchApplyPenalty(penalty_params);

        auto logitsOutHost = mBufferManager->copyFrom(*mLogitsDevice, MemoryType::kCPU);

        computeReference(bufferCast<T>(*mLogitsHost), bufferCast<int32_t>(*mOutputIdsHost),
            bufferCast<int32_t>(*mSeqLengthHost), bufferCast<float>(*param.repetitionPenalties),
            bufferCast<float>(*param.presencePenalties), bufferCast<float>(*param.frequencyPenalties),
            param.repetitionPenaltiesSize, param.presencePenaltiesSize, param.frequencyPenaltiesSize);

        mStream->synchronize();

        bool passed = checkResult(param.toString(), bufferCast<T>(*logitsOutHost), bufferCast<T>(*mLogitsHost),
            mBatchSize * mVocabSizePadded);
        EXPECT_TRUE(passed);
    }
};

TYPED_TEST_SUITE(RepetitionPenaltyTest, FloatAndHalfTypes);

TYPED_TEST(RepetitionPenaltyTest, BatchNoPenalty)
{
    int32_t batchSize = 6;
    TensorPtr repetitionPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
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
                      .setRepetitionPenaltiesSize(batchSize)
                      .setPresencePenaltiesSize(batchSize)
                      .setFrequencyPenaltiesSize(batchSize));
}

TYPED_TEST(RepetitionPenaltyTest, BatchRepetitionLessThanOne)
{
    int32_t batchSize = 6;
    TensorPtr repetitionPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
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
                      .setRepetitionPenaltiesSize(batchSize)
                      .setPresencePenaltiesSize(batchSize)
                      .setFrequencyPenaltiesSize(batchSize));
}

TYPED_TEST(RepetitionPenaltyTest, BatchRepetitionGreaterThaneOne)
{
    int32_t batchSize = 6;
    TensorPtr repetitionPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
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
                      .setRepetitionPenaltiesSize(batchSize)
                      .setPresencePenaltiesSize(batchSize)
                      .setFrequencyPenaltiesSize(batchSize));
}

TYPED_TEST(RepetitionPenaltyTest, BatchRepetitionMixed)
{
    int32_t batchSize = 6;
    TensorPtr repetitionPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
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
                      .setRepetitionPenaltiesSize(batchSize)
                      .setPresencePenaltiesSize(batchSize)
                      .setFrequencyPenaltiesSize(batchSize));
}

TYPED_TEST(RepetitionPenaltyTest, BatchPresenceMixed)
{
    int32_t batchSize = 6;
    TensorPtr repetitionPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
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
                      .setRepetitionPenaltiesSize(batchSize)
                      .setPresencePenaltiesSize(batchSize)
                      .setFrequencyPenaltiesSize(batchSize));
}

TYPED_TEST(RepetitionPenaltyTest, BatchPresenceHasDefaultValueZero2)
{
    int32_t batchSize = 6;
    TensorPtr repetitionPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
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
                      .setRepetitionPenaltiesSize(batchSize)
                      .setPresencePenaltiesSize(batchSize)
                      .setFrequencyPenaltiesSize(batchSize));
}

TYPED_TEST(RepetitionPenaltyTest, BatchFrequencyMixed)
{
    int32_t batchSize = 6;
    TensorPtr repetitionPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
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
                      .setRepetitionPenaltiesSize(batchSize)
                      .setPresencePenaltiesSize(batchSize)
                      .setFrequencyPenaltiesSize(batchSize));
}

TYPED_TEST(RepetitionPenaltyTest, BatchFrequencyHasDefaultValueZero2)
{
    int32_t batchSize = 6;
    TensorPtr repetitionPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
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
                      .setRepetitionPenaltiesSize(batchSize)
                      .setPresencePenaltiesSize(batchSize)
                      .setFrequencyPenaltiesSize(batchSize));
}

TYPED_TEST(RepetitionPenaltyTest, PenaltyTypeRepetitionPresence)
{
    int32_t batchSize = 6;
    TensorPtr repetitionPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
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
                      .setRepetitionPenaltiesSize(batchSize)
                      .setPresencePenaltiesSize(batchSize)
                      .setFrequencyPenaltiesSize(batchSize));
}

TYPED_TEST(RepetitionPenaltyTest, PenaltyTypeRepetitionFrequency)
{
    int32_t batchSize = 6;
    TensorPtr repetitionPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
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
                      .setRepetitionPenaltiesSize(batchSize)
                      .setPresencePenaltiesSize(batchSize)
                      .setFrequencyPenaltiesSize(batchSize));
}

TYPED_TEST(RepetitionPenaltyTest, PenaltyTypePresenceFrequency)
{
    int32_t batchSize = 6;
    TensorPtr repetitionPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
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
                      .setRepetitionPenaltiesSize(batchSize)
                      .setPresencePenaltiesSize(batchSize)
                      .setFrequencyPenaltiesSize(batchSize));
}

TYPED_TEST(RepetitionPenaltyTest, PenaltyTypeFull)
{
    int32_t batchSize = 6;
    TensorPtr repetitionPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr presencePenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    TensorPtr frequencyPenaltyHost
        = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kFLOAT);
    for (int32_t i = 0; i < batchSize; ++i)
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
                      .setRepetitionPenaltiesSize(batchSize)
                      .setPresencePenaltiesSize(batchSize)
                      .setFrequencyPenaltiesSize(batchSize));
}

struct MinLengthPenaltyTestParams
{
    int32_t batchSize;
    int32_t vocabSize;
    int32_t maxSeqLength;

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

    std::string toString() const
    {
        return tc::fmtstr(
            "MinLengthPenaltyTestParams[batch=%d, vocab=%d, maxSeqLen=%d]", batchSize, vocabSize, maxSeqLength);
    }
};

template <typename T>
class MinLengthPenaltyTest : public SamplingKernelTest<T>
{
protected:
    // Set up test
    int32_t mBatchSize;
    int32_t mVocabSize;
    int32_t mVocabSizePadded;
    int32_t mMaxInputLength;
    int32_t mSequenceLength;

    using SamplingKernelTest<T>::mBufferManager;
    using SamplingKernelTest<T>::mStream;
    using SamplingKernelTest<T>::mLogitsHost;

    TensorPtr mLogitsDevice;
    TensorPtr mPenaltyWorkspaceDevice;

    TensorPtr mContextLengthHost;
    TensorPtr mContextLengthDevice;

    TensorPtr mSeqLengthHost;
    TensorPtr mSeqLengthDevice;

    TensorPtr mMinLengthHost;
    TensorPtr mMinLengthDevice;

    TensorPtr mEndIdsHost;
    TensorPtr mEndIdsDevice;

    void subsetup(MinLengthPenaltyTestParams param)
    {
        mBatchSize = param.batchSize;
        mVocabSize = param.vocabSize;
        mVocabSizePadded = padVocabSize(mVocabSize);
        mMaxInputLength = param.maxSeqLength;
        mSequenceLength = 2 * mMaxInputLength; // input + output

        mLogitsHost = mBufferManager->pinned(ITensor::makeShape({mBatchSize, mVocabSizePadded}),
            std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);
        mLogitsDevice = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mVocabSizePadded}),
            std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);

        mPenaltyWorkspaceDevice
            = mBufferManager->gpu(ITensor::makeShape({mBatchSize, mVocabSize}), nvinfer1::DataType::kINT32);

        mSeqLengthHost = mBufferManager->pinned(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);
        mSeqLengthDevice = mBufferManager->gpu(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);

        mContextLengthHost = mBufferManager->pinned(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);
        mContextLengthDevice = mBufferManager->gpu(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);

        mMinLengthHost = mBufferManager->pinned(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);
        mMinLengthDevice = mBufferManager->gpu(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);

        mEndIdsHost = mBufferManager->pinned(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);
        mEndIdsDevice = mBufferManager->gpu(ITensor::makeShape({mBatchSize}), nvinfer1::DataType::kINT32);

        initLogitsAndBias(
            bufferCast<T>(*mLogitsHost), static_cast<T*>(nullptr), mBatchSize, mVocabSize, mVocabSizePadded);
        initRandomInt(bufferCast<int32_t>(*mContextLengthHost), mBatchSize, 0, mMaxInputLength);
        initRandomInt(bufferCast<int32_t>(*mMinLengthHost), mBatchSize, 1, mMaxInputLength);
        initRandomInt(bufferCast<int32_t>(*mEndIdsHost), mBatchSize, 0, mVocabSize);

        auto seqLengthHostPtr = bufferCast<int32_t>(*mSeqLengthHost);
        auto contextLengthHostPtr = bufferCast<int32_t>(*mContextLengthHost);
        auto minLengthHostPtr = bufferCast<int32_t>(*mMinLengthHost);
        for (SizeType bi = 0; bi < mBatchSize; bi++)
        {
            // Current generated seq len is randomly either smaller than min length or larger
            const auto generatedSeqLen = std::max(0,
                std::min(
                    static_cast<int32_t>(minLengthHostPtr[bi] + 2 * std::pow(-1, std::rand() % 2)), mMaxInputLength));
            seqLengthHostPtr[bi] = contextLengthHostPtr[bi] + generatedSeqLen;
        }

        mBufferManager->copy(*mLogitsHost, *mLogitsDevice);
        mBufferManager->copy(*mMinLengthHost, *mMinLengthDevice);
        mBufferManager->copy(*mContextLengthHost, *mContextLengthDevice);
        mBufferManager->copy(*mSeqLengthHost, *mSeqLengthDevice);
        mBufferManager->copy(*mEndIdsHost, *mEndIdsDevice);
    }

    void computeReference(
        T* logits, const int* minSeqLen, const int* endIds, const int* sequenceLengths, const int* contextLengths)
    {
        const bool IS_FP16 = std::is_same<T, half>::value;
        const T MAX_T_VAL = (IS_FP16) ? 65504.F : FLT_MAX;

        for (int32_t bi = 0; bi < mBatchSize; ++bi)
        {
            const auto generatedSeqLen = sequenceLengths[bi] - contextLengths[bi];
            const auto endId = endIds[bi];
            if (generatedSeqLen < minSeqLen[bi])
            {
                logits[bi * mVocabSizePadded + endId] = -MAX_T_VAL;
            }
        }
    }

public:
    void runTest(MinLengthPenaltyTestParams param)
    {
        subsetup(param);
        InvokeBatchApplyPenaltyParams<T> penalty_params{bufferCast<T>(*mLogitsDevice), nullptr,
            bufferCast<int32_t>(*mPenaltyWorkspaceDevice), nullptr, nullptr, nullptr, nullptr, nullptr, false,
            mBatchSize, 1, mSequenceLength, mVocabSize, mVocabSizePadded, nullptr, nullptr,
            bufferCast<int32_t>(*mContextLengthDevice), bufferCast<int32_t>(*mSeqLengthDevice),
            bufferCast<int32_t>(*mMinLengthDevice), bufferCast<int32_t>(*mEndIdsDevice), mStream->get()};
        tk::invokeBatchApplyPenalty(penalty_params);

        mStream->synchronize();

        computeReference(bufferCast<T>(*mLogitsHost), bufferCast<int32_t>(*mMinLengthHost),
            bufferCast<int32_t>(*mEndIdsHost), bufferCast<int32_t>(*mSeqLengthHost),
            bufferCast<int32_t>(*mContextLengthHost));

        auto logitsOutHost = mBufferManager->copyFrom(*mLogitsDevice, MemoryType::kCPU);

        mStream->synchronize();

        bool passed = checkResult(param.toString(), bufferCast<T>(*logitsOutHost), bufferCast<T>(*mLogitsHost),
            mBatchSize * mVocabSizePadded);
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

} // namespace
