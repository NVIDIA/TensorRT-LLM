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

namespace tk = tensorrt_llm::kernels;

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::tests::kernels::sampling;

namespace
{

template <typename T>
class AirTopPSamplingKernelTest : public SamplingKernelTest<T>
{

protected:
    const int32_t endId = 0;
    using SamplingKernelTest<T>::mSeed;
    using SamplingKernelTest<T>::mStream;
    using SamplingKernelTest<T>::mBufferManager;

private:
    size_t getWorkspaceSize(SamplingKernelTestParam const& params) override
    {
        return tensorrt_llm::kernels::getAirTopPWorkspaceSize<T>(
            params.batchSize, params.vocabSize, params.isDeterministicTopP);
    }

    void callTestedFunction(
        SamplingKernelTestParam const& params, tensorrt_llm::runtime::ITensor::SharedPtr& workspaceDevice) override
    {
        // Calculate the number of blocks based on the number of multiprocessors, batchSize and vocabSize.
        int dev;
        int smCnt;
        TLLM_CUDA_CHECK(cudaGetDevice(&dev));
        TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&smCnt, cudaDevAttrMultiProcessorCount, dev));
        auto const maxBatchSize = 2 * params.batchSize;

        int blockNum
            = tk::calcAirTopPBlockNum<T>(params.batchSize, params.vocabSize, smCnt, params.isDeterministicTopP);

        tk::TopPSamplingKernelParams<T> kernelParams;
        kernelParams.probs = bufferCast<T>(*this->mProbsDevice);
        kernelParams.outputIdsPtrs = bufferCast<int*>(*this->mIdsPtrHost);
        kernelParams.workspace = workspaceDevice->data();
        kernelParams.topPs = bufferCast<float>(*this->mTopPsDevice);
        kernelParams.sequenceLength = bufferCast<int32_t>(*this->mSeqLengthsDevice);
        kernelParams.endIds = bufferCast<int32_t>(*this->mEndIdsDevice);
        kernelParams.batchSlots = bufferCast<int32_t>(*this->mBatchSlots);
        kernelParams.finishedInput = reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
            bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(*this->mFinishedDevice));
        kernelParams.finishedOutput = reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
            bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(*this->mFinishedDevice));
        kernelParams.skipDecode = bufferCast<bool>(*this->mSkipDecodeDevice);
        kernelParams.cumLogProbs = bufferCast<float>(*this->mCumLogProbsDevice);
        kernelParams.outputLogProbs = bufferCast<float>(*this->mOutputLogProbsDevice);
        kernelParams.curandState = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*this->mCurandStatesDevice));
        kernelParams.batchSize = params.batchSize;
        kernelParams.maxBatchSize = maxBatchSize;
        kernelParams.vocabSizePadded = params.vocabSize;
        kernelParams.blockNum = blockNum;
        kernelParams.isDeterministic = params.isDeterministicTopP;

        // Perform batched TopP sampling
        tk::invokeBatchAirTopPSampling<T>(kernelParams, this->mStream->get());
    }
};

TYPED_TEST_SUITE(AirTopPSamplingKernelTest, FloatAndHalfTypes);

TYPED_TEST(AirTopPSamplingKernelTest, NondeterministicCorrectnessSmallP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(0).setTopP(0.2f));
};

TYPED_TEST(AirTopPSamplingKernelTest, NondeterministicCorrectnessLargeP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(0).setTopP(0.9f));
};

TYPED_TEST(AirTopPSamplingKernelTest, NondeterministicCorrectnessAncestral)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(0).setTopP(1.0f));
};

TYPED_TEST(AirTopPSamplingKernelTest, NondeterministicCorrectnessLargeVocabSmallP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(32).setVocabSize(51200).setTopK(0).setTopP(0.2f));
};

TYPED_TEST(AirTopPSamplingKernelTest, NondeterministicCorrectnessLargeVocabLargeP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(32).setVocabSize(51200).setTopK(0).setTopP(0.9f));
};

TYPED_TEST(AirTopPSamplingKernelTest, DeterministicCorrectnessSmallP)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(0).setTopP(0.2f).setDeterministicTopP(true));
};

TYPED_TEST(AirTopPSamplingKernelTest, DeterministicCorrectnessLargeP)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(0).setTopP(0.9f).setDeterministicTopP(true));
};

TYPED_TEST(AirTopPSamplingKernelTest, DeterministicCorrectnessAncestral)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(0).setTopP(1.0f).setDeterministicTopP(true));
};

TYPED_TEST(AirTopPSamplingKernelTest, DeterministicCorrectnessLargeVocabSmallP)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(32).setVocabSize(51200).setTopK(0).setTopP(0.2f).setDeterministicTopP(
            true));
};

TYPED_TEST(AirTopPSamplingKernelTest, DeterministicCorrectnessLargeVocabLargeP)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(32).setVocabSize(51200).setTopK(0).setTopP(0.9f).setDeterministicTopP(
            true));
};

class AirTopPSamplingKernelUtilsTest : public SamplingKernelTest<float>
{
};

} // end of namespace
