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
        return tensorrt_llm::kernels::getAirTopPWorkspaceSize<T>(params.batchSize, params.vocabSize);
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

        int blockNum = tk::calcAirTopPBlockNum<T, int, float>(params.batchSize, params.vocabSize, smCnt);
        // Perform batched TopP sampling
        tk::invokeBatchAirTopPSampling<T>(workspaceDevice->data(), bufferCast<int*>(*this->mIdsPtrHost),
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
            bufferCast<T>(*this->mProbsDevice),
            reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*this->mCurandStatesDevice)), params.batchSize,
            maxBatchSize, params.vocabSize, bufferCast<int32_t>(*this->mEndIdsDevice), this->mMaxTopP,
            bufferCast<float>(*this->mTopPsDevice), this->mStream->get(), blockNum,
            bufferCast<bool>(*this->mSkipDecodeDevice), bufferCast<int32_t>(*this->mBatchSlots));
    }
};

TYPED_TEST_SUITE(AirTopPSamplingKernelTest, FloatAndHalfTypes);

TYPED_TEST(AirTopPSamplingKernelTest, CorrectnessSmallP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopP(0.2f));
};

TYPED_TEST(AirTopPSamplingKernelTest, CorrectnessLargeP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopP(0.9f));
};

TYPED_TEST(AirTopPSamplingKernelTest, CorrectnessAncestral)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopP(1.0f));
};

TYPED_TEST(AirTopPSamplingKernelTest, CorrectnessLargeVocabSmallP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(32).setVocabSize(51200).setTopP(0.2f));
};

TYPED_TEST(AirTopPSamplingKernelTest, CorrectnessLargeVocabLargeP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(32).setVocabSize(51200).setTopP(0.9f));
};

class AirTopPSamplingKernelUtilsTest : public SamplingKernelTest<float>
{
};

} // end of namespace
