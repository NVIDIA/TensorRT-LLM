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
#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include "tests/kernels/sampling/samplingTest.h"
#include <random>

namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
namespace trk = tensorrt_llm::runtime::kernels;

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::tests::kernels::sampling;

namespace
{

template <typename T>
class TopPSamplingKernelTest : public SamplingKernelTest<T>
{

protected:
    const int32_t endId = 0;
    using SamplingKernelTest<T>::mSeed;
    using SamplingKernelTest<T>::mStream;
    using SamplingKernelTest<T>::mBufferManager;

private:
    size_t getWorkspaceSize(const SamplingKernelTestParam& params) override
    {
        size_t workspaceSize;
        size_t cubTempStorageSize;
        tk::invokeBatchTopPSampling<T>(nullptr, // workspace
            workspaceSize, cubTempStorageSize,
            nullptr,                            // output_ids
            nullptr,                            // sequence_length
            nullptr,                            // finished_buffer
            nullptr,                            // finished_buffer
            nullptr,                            // cum_log_probs
            nullptr,                            // output_log_probs
            nullptr,                            // log_probs
            bufferCast<int32_t>(*this->mTopPIdValsDevice), bufferCast<int32_t>(*this->mEndOffsetsDevice),
            bufferCast<int32_t>(*this->mBeginOffsetsDevice), this->mCurandStatesDevice, params.batchSize,
            params.vocabSize, nullptr, this->mMaxTopP, bufferCast<float>(*this->mTopPsDevice), this->mStream->get(),
            nullptr);
        return workspaceSize;
    }

    void callTestedFunction(const SamplingKernelTestParam& params, bool hasDiffRuntimeArgs, size_t workspaceSize,
        tensorrt_llm::runtime::ITensor::SharedPtr& workspaceDevice) override
    {
        size_t cubTempStorageSize;
        tk::invokeBatchTopPSampling<T>(nullptr, // workspace
            workspaceSize, cubTempStorageSize,
            nullptr,                            // output_ids
            nullptr,                            // sequence_length
            nullptr,                            // finished_buffer
            nullptr,                            // finished_buffer
            nullptr,                            // cum_log_probs
            nullptr,                            // output_log_probs
            nullptr,                            // log_probs
            bufferCast<int32_t>(*this->mTopPIdValsDevice), bufferCast<int32_t>(*this->mEndOffsetsDevice),
            bufferCast<int32_t>(*this->mBeginOffsetsDevice), this->mCurandStatesDevice, params.batchSize,
            params.vocabSize, nullptr, this->mMaxTopP, bufferCast<float>(*this->mTopPsDevice), this->mStream->get(),
            nullptr);

        // Perform batched TopP sampling
        tk::invokeTopPInitialize(bufferCast<int32_t>(*this->mTopPIdValsDevice),
            bufferCast<int32_t>(*this->mEndOffsetsDevice), bufferCast<int32_t>(*this->mBeginOffsetsDevice),
            params.batchSize, params.vocabSize, this->mStream->get());

        // Perform batched TopP sampling
        tk::invokeBatchTopPSampling<T>(workspaceDevice->data(), workspaceSize, cubTempStorageSize,
            bufferCast<int*>(*this->mIdsPtrHost), bufferCast<int32_t>(*this->mSeqLengthsDevice),
            reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
                bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(*this->mFinishedDevice)),
            reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
                bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(*this->mFinishedDevice)),
            bufferCast<float>(*this->mCumLogProbsDevice), bufferCast<float>(*this->mOutputLogProbsDevice),
            // Note that the kernel needs vocab probs instead of
            // log-prob if cum_log_probs or output_log_probs are
            // provided. It's because the sampling layer already
            // preprocesses log_prob_buf when those are provided.
            bufferCast<T>(*this->mProbsDevice), bufferCast<int32_t>(*this->mTopPIdValsDevice),
            bufferCast<int32_t>(*this->mEndOffsetsDevice), bufferCast<int32_t>(*this->mBeginOffsetsDevice),
            this->mCurandStatesDevice, params.batchSize, params.vocabSize, bufferCast<int32_t>(*this->mEndIdsDevice),
            this->mMaxTopP, hasDiffRuntimeArgs ? bufferCast<float>(*this->mTopPsDevice) : nullptr, this->mStream->get(),
            bufferCast<bool>(*this->mSkipDecodeDevice));
    }
};

TYPED_TEST_SUITE(TopPSamplingKernelTest, FloatAndHalfTypes);

TYPED_TEST(TopPSamplingKernelTest, CorrectnessSmallP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(0).setTopP(0.2f).setOutputLen(1));
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessLargeP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(0).setTopP(0.9f).setOutputLen(1));
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessAncestral)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(0).setTopP(1.0f).setOutputLen(1));
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessLargeVocabSmallP)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(32).setVocabSize(51200).setTopK(0).setTopP(0.2f).setOutputLen(16));
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessLargeVocabLargeP)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(32).setVocabSize(51200).setTopK(0).setTopP(0.9f).setOutputLen(16));
};

class TopPSamplingKernelUtilsTest : public SamplingKernelTest<float>
{
};

TEST_F(TopPSamplingKernelUtilsTest, invokeTopPInitialize)
{
    const int32_t batchSize = 8;
    const int32_t vocabSize = 256;

    const auto topPIdValsDevice
        = this->mBufferManager->gpu(ITensor::makeShape({batchSize, vocabSize}), nvinfer1::DataType::kINT32);
    const auto beginOffsetsDevice
        = this->mBufferManager->gpu(ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32);
    const auto endOffsetsDevice
        = this->mBufferManager->gpu(ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32);

    tk::invokeTopPInitialize(bufferCast<int32_t>(*topPIdValsDevice), bufferCast<int32_t>(*endOffsetsDevice),
        bufferCast<int32_t>(*beginOffsetsDevice), batchSize, vocabSize, this->mStream->get());

    const auto topPIdValsHost = this->mBufferManager->copyFrom(*topPIdValsDevice, MemoryType::kCPU);
    const auto endOffsetsHost = this->mBufferManager->copyFrom(*endOffsetsDevice, MemoryType::kCPU);
    const auto beginOffsetsHost = this->mBufferManager->copyFrom(*beginOffsetsDevice, MemoryType::kCPU);

    this->mStream->synchronize();

    const auto topPIdValsHostPtr = bufferCast<int32_t>(*topPIdValsHost);
    const auto endOffsetsHostPtr = bufferCast<int32_t>(*endOffsetsHost);
    const auto beginOffsetsHostPtr = bufferCast<int32_t>(*beginOffsetsHost);

    for (int32_t bi = 0; bi < batchSize + 1; ++bi)
    {
        EXPECT_EQ(endOffsetsHostPtr[bi], bi * vocabSize);
        EXPECT_EQ(beginOffsetsHostPtr[bi], bi * vocabSize);
    }
    for (int32_t bi = 0; bi < batchSize; ++bi)
    {
        for (int32_t vi = 0; vi < vocabSize; ++vi)
        {
            EXPECT_EQ(topPIdValsHostPtr[bi * vocabSize + vi], vi);
        }
    }
};

} // end of namespace
