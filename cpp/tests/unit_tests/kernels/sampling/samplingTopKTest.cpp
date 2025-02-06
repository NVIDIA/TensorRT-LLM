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

#include "tensorrt_llm/common/tllmException.h"
#include "tests/unit_tests/kernels/sampling/samplingTest.h"

namespace tk = tensorrt_llm::kernels;

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::tests::kernels::sampling;

namespace
{

template <typename T>
class TopKSamplingKernelTest : public SamplingKernelTest<T>
{

protected:
    int32_t const endId = 0;
    using SamplingKernelTest<T>::mSeed;
    using SamplingKernelTest<T>::mStream;
    using SamplingKernelTest<T>::mBufferManager;

    size_t getWorkspaceSize(SamplingKernelTestParam const& params) override
    {
        return tk::getTopKWorkspaceSize<T>(params.batchSize, params.maxTokensPerStep, this->mMaxTopK, params.vocabSize);
    }

    void callTestedFunction(
        SamplingKernelTestParam const& params, tensorrt_llm::runtime::ITensor::SharedPtr& workspaceDevice) override
    {
        auto const maxBatchSize = 2 * params.batchSize;

        tk::TopKSamplingKernelParams<T> kernelParams;
        kernelParams.logProbs = params.useLogitsPtrs ? nullptr : bufferCast<T>(*this->mProbsDevice);
        kernelParams.logProbsPtrs = params.useLogitsPtrs
            ? reinterpret_cast<T const* const*>(bufferCast<int64_t>(*this->mProbsPtrsDevice))
            : nullptr;
        kernelParams.outputIdsPtrs = bufferCast<int32_t*>(*this->mIdsPtrHost);
        kernelParams.workspace = workspaceDevice->data();
        kernelParams.maxTopP = params.topP;
        kernelParams.topPs = bufferCast<float>(*this->mTopPsDevice);
        kernelParams.maxTopK = this->mMaxTopK;
        kernelParams.topKs = bufferCast<int32_t>(*this->mTopKsDevice);
        kernelParams.sequenceLengths = bufferCast<int32_t>(*this->mSeqLengthsDevice);
        kernelParams.endIds = bufferCast<int32_t>(*this->mEndIdsDevice);
        kernelParams.batchSlots = bufferCast<int32_t>(*this->mBatchSlots);
        kernelParams.finishedInput = reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
            bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(*this->mFinishedDevice));
        kernelParams.finishedOutput = reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
            bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(*this->mFinishedDevice));
        kernelParams.skipDecode = bufferCast<bool>(*this->mSkipDecodeDevice);
        kernelParams.cumLogProbs = params.returnAllSelectedTokens || params.maxTokensPerStep > 1
            ? nullptr
            : bufferCast<float>(*this->mCumLogProbsDevice);
        kernelParams.outputLogProbs
            = params.maxTokensPerStep > 1 ? nullptr : bufferCast<float>(*this->mOutputLogProbsDevice);
        kernelParams.curandState = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*this->mCurandStatesDevice));
        kernelParams.batchSize = params.batchSize;
        kernelParams.maxBatchSize = maxBatchSize;
        kernelParams.maxTokensPerStep = params.maxTokensPerStep;
        kernelParams.tokensPerStep = bufferCast<int32_t>(*this->mTokensPerStep);
        kernelParams.vocabSizePadded = params.vocabSize;
        kernelParams.normalizeLogProbs = params.normalizeLogProbs;
        kernelParams.logitsHasProbs = params.logitsHasProbs;
        kernelParams.returnAllSelectedTokens = params.returnAllSelectedTokens;

        // Perform batched TopK sampling
        tk::invokeBatchTopKSampling(kernelParams, this->mStream->get());
    }
};

TYPED_TEST_SUITE(TopKSamplingKernelTest, FloatAndHalfTypes);

TYPED_TEST(TopKSamplingKernelTest, CorrectnessGreedy)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(1).setTopP(1.0f));
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessGreedyLarge)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(16).setVocabSize(51200).setTopK(1).setTopP(1.0f));
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessAncestral)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(4).setTopP(1.0f));
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessLargeK63)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(16).setVocabSize(51200).setTopK(63).setTopP(1.0f));
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessLargeK1024)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(16).setVocabSize(51200).setTopK(1024).setTopP(1.0f));
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessTopKTopP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(16).setVocabSize(4000).setTopK(63).setTopP(0.3f));
};

TYPED_TEST(TopKSamplingKernelTest, NotSupportedLargerThanK1024)
{
    EXPECT_THROW(
        this->runTest(SamplingKernelTestParam().setBatchSize(16).setVocabSize(4000).setTopK(1025).setTopP(1.0f)),
        tensorrt_llm::common::TllmException);
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessTopKMaxTokensPerStep)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(16).setVocabSize(4000).setTopK(63).setTopP(1.0f).setMaxTokensPerStep(4));
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessReturnAllSelectedTokens)
{
    this->runTest(SamplingKernelTestParam()
                      .setBatchSize(16)
                      .setVocabSize(50)
                      .setTopK(10)
                      .setTopP(1.0f)
                      .setMaxTokensPerStep(4)
                      .setReturnAllSelectedTokens());
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessReturnAllSelectedTokensMaxTokensPerStep1)
{
    this->runTest(SamplingKernelTestParam()
                      .setBatchSize(16)
                      .setVocabSize(50)
                      .setTopK(10)
                      .setTopP(1.0f)
                      .setMaxTokensPerStep(1)
                      .setReturnAllSelectedTokens());
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessReturnAllSelectedTokensSmallP)
{
    this->runTest(SamplingKernelTestParam()
                      .setBatchSize(16)
                      .setVocabSize(50)
                      .setTopK(20)
                      .setTopP(0.3f)
                      .setMaxTokensPerStep(4)
                      .setReturnAllSelectedTokens());
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessLogitsPtrs)
{
    this->runTest(SamplingKernelTestParam()
                      .setBatchSize(16)
                      .setVocabSize(50)
                      .setTopK(10)
                      .setTopP(1.0f)
                      .setMaxTokensPerStep(4)
                      .setUseLogitsPtrs());
};
} // end of namespace
