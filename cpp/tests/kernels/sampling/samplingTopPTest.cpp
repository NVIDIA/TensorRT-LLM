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
    size_t getWorkspaceSize(SamplingKernelTestParam const& params) override
    {
        return tensorrt_llm::kernels::getTopPWorkspaceSize<T>(params.batchSize, params.vocabSize);
    }

    void callTestedFunction(
        SamplingKernelTestParam const& params, tensorrt_llm::runtime::ITensor::SharedPtr& workspaceDevice) override
    {
        auto const maxBatchSize = 2 * params.batchSize;

        // Perform batched TopP sampling
        tk::invokeTopPInitialize(bufferCast<int32_t>(*this->mTopPIdValsDevice),
            bufferCast<int32_t>(*this->mEndOffsetsDevice), bufferCast<int32_t>(*this->mBeginOffsetsDevice),
            params.batchSize, params.vocabSize, this->mStream->get());

        // Perform batched TopP sampling
        tk::invokeBatchTopPSampling<T>(workspaceDevice->data(), bufferCast<int*>(*this->mIdsPtrHost),
            bufferCast<int32_t>(*this->mSeqLengthsDevice),
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
            reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*this->mCurandStatesDevice)), params.batchSize,
            maxBatchSize, params.vocabSize, bufferCast<int32_t>(*this->mEndIdsDevice), this->mMaxTopP,
            bufferCast<float>(*this->mTopPsDevice), this->mStream->get(), bufferCast<bool>(*this->mSkipDecodeDevice),
            bufferCast<int32_t>(*this->mBatchSlots));
    }
};

TYPED_TEST_SUITE(TopPSamplingKernelTest, FloatAndHalfTypes);

TYPED_TEST(TopPSamplingKernelTest, CorrectnessSmallP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopP(0.2f));
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessLargeP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopP(0.9f));
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessAncestral)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopP(1.0f));
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessLargeVocabSmallP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(32).setVocabSize(51200).setTopP(0.2f));
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessLargeVocabLargeP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(32).setVocabSize(51200).setTopP(0.9f));
};

class TopPSamplingKernelUtilsTest : public SamplingKernelTest<float>
{
};

TEST_F(TopPSamplingKernelUtilsTest, invokeTopPInitialize)
{
    const int32_t batchSize = 8;
    const int32_t vocabSize = 256;

    auto const topPIdValsDevice
        = this->mBufferManager->gpu(ITensor::makeShape({batchSize, vocabSize}), nvinfer1::DataType::kINT32);
    auto const beginOffsetsDevice
        = this->mBufferManager->gpu(ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32);
    auto const endOffsetsDevice
        = this->mBufferManager->gpu(ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32);

    tk::invokeTopPInitialize(bufferCast<int32_t>(*topPIdValsDevice), bufferCast<int32_t>(*endOffsetsDevice),
        bufferCast<int32_t>(*beginOffsetsDevice), batchSize, vocabSize, this->mStream->get());

    auto const topPIdValsHost = this->mBufferManager->copyFrom(*topPIdValsDevice, MemoryType::kCPU);
    auto const endOffsetsHost = this->mBufferManager->copyFrom(*endOffsetsDevice, MemoryType::kCPU);
    auto const beginOffsetsHost = this->mBufferManager->copyFrom(*beginOffsetsDevice, MemoryType::kCPU);

    this->mStream->synchronize();

    auto const topPIdValsHostPtr = bufferCast<int32_t>(*topPIdValsHost);
    auto const endOffsetsHostPtr = bufferCast<int32_t>(*endOffsetsHost);
    auto const beginOffsetsHostPtr = bufferCast<int32_t>(*beginOffsetsHost);

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
