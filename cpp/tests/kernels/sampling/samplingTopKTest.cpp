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
class TopKSamplingKernelTest : public SamplingKernelTest<T>
{

protected:
    const int32_t endId = 0;
    using SamplingKernelTest<T>::mSeed;
    using SamplingKernelTest<T>::mStream;
    using SamplingKernelTest<T>::mBufferManager;

    size_t getWorkspaceSize(const SamplingKernelTestParam& params) override
    {
        size_t workspaceSize;
        tk::invokeTopKSampling<T>(nullptr, workspaceSize, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, this->mMaxTopK, 1.0f, params.vocabSize, nullptr, this->mStream->get(), params.batchSize, nullptr,
            true);
        return workspaceSize;
    }

    void callTestedFunction(const SamplingKernelTestParam& params, bool hasDiffRuntimeArgs, size_t workspaceSize,
        tensorrt_llm::runtime::ITensor::SharedPtr& workspaceDevice) override
    {
        // Perform batched TopK sampling
        tk::invokeBatchTopKSampling(workspaceDevice->data(), workspaceSize,
            // Note that the kernel needs vocab probs instead of
            // log-prob if cum_log_probs or output_log_probs are
            // provided. It's because the sampling layer already
            // preprocesses log_prob_buf when those are provided.
            bufferCast<T>(*this->mProbsDevice), bufferCast<int*>(*this->mIdsPtrHost),
            bufferCast<int32_t>(*this->mSeqLengthsDevice),
            reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
                bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(*this->mFinishedDevice)),
            reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
                bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(*this->mFinishedDevice)),
            bufferCast<float>(*this->mCumLogProbsDevice), bufferCast<float>(*this->mOutputLogProbsDevice),
            this->mCurandStatesDevice, this->mMaxTopK,
            hasDiffRuntimeArgs ? bufferCast<int32_t>(*this->mTopKsDevice) : nullptr, params.topP,
            hasDiffRuntimeArgs ? bufferCast<float>(*this->mTopPsDevice) : nullptr, params.vocabSize,
            bufferCast<int32_t>(*this->mEndIdsDevice), this->mStream->get(), params.batchSize,
            bufferCast<bool>(*this->mSkipDecodeDevice), params.normalizeLogProbs);
    }
};

TYPED_TEST_SUITE(TopKSamplingKernelTest, FloatAndHalfTypes);

TYPED_TEST(TopKSamplingKernelTest, CorrectnessGreedy)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(1).setTopP(1.0f).setOutputLen(1));
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessGreedyLarge)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(16).setVocabSize(51200).setTopK(1).setTopP(1.0f).setOutputLen(8));
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessAncestral)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(4).setTopP(1.0f).setOutputLen(1));
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessLargeK63)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(16).setVocabSize(51200).setTopK(63).setTopP(1.0f).setOutputLen(8));
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessLargeK1024)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(16).setVocabSize(51200).setTopK(1024).setTopP(1.0f).setOutputLen(8));
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessTopKTopP)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(16).setVocabSize(4000).setTopK(63).setTopP(0.3f).setOutputLen(8));
};

TYPED_TEST(TopKSamplingKernelTest, NotSupportedLargerThanK1024)
{
    EXPECT_THROW(
        this->runTest(
            SamplingKernelTestParam().setBatchSize(16).setVocabSize(4000).setTopK(1025).setTopP(1.0f).setOutputLen(8)),
        std::domain_error);
};

} // end of namespace
