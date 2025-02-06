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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tests/unit_tests/layers/baseSamplingLayerTest.h"

#include <algorithm>

namespace
{

namespace tle = tensorrt_llm::executor;
namespace trk = tensorrt_llm::runtime::kernels;

using namespace tensorrt_llm::tests::layers::sampling;
using namespace tensorrt_llm::layers;
using namespace tensorrt_llm::runtime;

template <typename T>
class ExternalDraftTokensLayerTest : public BaseSamplingLayerTest<T>
{
protected:
    int32_t const mMaxDraftLen = this->mMaxTokensPerEngineStep - 1;

    TensorPtr mDraftLogits;
    TensorPtr mDraftProbs;
    TensorPtr mTargetProbs;
    TensorPtr mNumDraftTokens;
    TensorPtr mNumDraftTokensHost;
    TensorPtr mDraftTokenIds;
    TensorPtr mUseDraftLogits;
    TensorPtr mUseDraftLogitsHost;
    float mConstantThreshold = 1.0f;
    bool mUseRandomAcceptanceThreshold = true;

    std::vector<T>* mTestDraftLogitsInit;
    std::vector<T> mTestDraftLogitsAccept = {
        -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -0.9163, -1.2040, -1.6094, -2.3026, // step 0
        -0.9163, -1.2040, -1.6094, -2.3026, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, // step 1
        -FLT_MAX, -FLT_MAX, -0.9163, -1.2040, -1.6094, -2.3026, -FLT_MAX, -FLT_MAX, // step 2
    };
    std::vector<T> mTestDraftLogitsReject = {
        -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -0.9163, -1.2040, -1.6094, -2.3026, // step 0
        -FLT_MAX, -FLT_MAX, -FLT_MAX, -0.9163, -1.2040, -1.6094, -2.3026, -FLT_MAX, // step 1
        -FLT_MAX, -FLT_MAX, -0.9163, -1.2040, -1.6094, -2.3026, -FLT_MAX, -FLT_MAX, // step 2
    };
    std::vector<std::vector<SizeType32>> mTestDraftTokenIdsInit;

    void SetUp() override
    {
        this->mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        this->mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(this->mStream);
    }

    void initLayer(TestSamplingParams const& params) override
    {
        auto decodingMode = tle::DecodingMode::ExternalDraftTokens();

        auto const decodingDomain
            = tensorrt_llm::layers::DecoderDomain(this->maxBatchSize(), 1, this->mVocabSize, this->mVocabSizePadded);
        this->mSamplingLayer = std::make_shared<tensorrt_llm::layers::ExternalDraftTokensLayer<T>>(
            decodingMode, decodingDomain, this->mBufferManager, true, params.isAirTopPExternalDraftTokensLayer);

        auto const dataType = TRTDataType<T>::value;

        mDraftLogits = this->mBufferManager->gpu(
            ITensor::makeShape({this->maxBatchSize(), mMaxDraftLen, this->mVocabSize}), dataType);
        mDraftProbs = this->mBufferManager->gpu(
            ITensor::makeShape({this->maxBatchSize(), mMaxDraftLen, this->mBeamWidth, this->mVocabSize}), dataType);
        mTargetProbs = this->mBufferManager->gpu(
            ITensor::makeShape(
                {this->maxBatchSize(), this->mMaxTokensPerEngineStep, this->mBeamWidth, this->mVocabSize}),
            dataType);

        mDraftTokenIds = this->mBufferManager->gpu(
            ITensor::makeShape({this->maxBatchSize(), mMaxDraftLen}), nvinfer1::DataType::kINT32);
        mUseDraftLogits
            = this->mBufferManager->gpu(ITensor::makeShape({this->maxBatchSize()}), TRTDataType<bool>::value);
        mUseDraftLogitsHost
            = this->mBufferManager->cpu(ITensor::makeShape({this->maxBatchSize()}), TRTDataType<bool>::value);

        mNumDraftTokens
            = this->mBufferManager->gpu(ITensor::makeShape({this->maxBatchSize()}), TRTDataType<SizeType32>::value);
        mNumDraftTokensHost
            = this->mBufferManager->cpu(ITensor::makeShape({this->maxBatchSize()}), TRTDataType<SizeType32>::value);

        batchCopyDraftTokenIds();
        if (params.useDraftLogits)
        {
            batchCopyDraftLogits();
        }

        batchUseDraftLogits(params.useDraftLogits);
    }

    std::shared_ptr<DecodingInputs> createInputTensors(int32_t step) override
    {
        constexpr int32_t ite = 0;
        auto decodeInputTensors = std::make_shared<ExternalDraftTokensInputs>(
            this->mEndIdsDevice, this->mBatchSlots, step, ite, this->mBatchSize);

        decodeInputTensors->logits = this->mDecodingWorkspace->getDeviceRuntimeLogits();

        decodeInputTensors->inputLengths = this->mContextLengthDevice;

        decodeInputTensors->finished = this->mFinishedDevice;

        decodeInputTensors->probsComputed = this->mComputeProbs;

        decodeInputTensors->curandStates
            = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*this->mCurandStatesDevice));

        decodeInputTensors->draftLogits = mDraftLogits;
        decodeInputTensors->draftProbs = mDraftProbs;
        decodeInputTensors->targetProbs = mTargetProbs;
        decodeInputTensors->numDraftTokens = mNumDraftTokens;
        decodeInputTensors->numDraftTokensHost = mNumDraftTokensHost;
        decodeInputTensors->draftTokenIds = mDraftTokenIds;
        decodeInputTensors->constantThreshold = mConstantThreshold;
        decodeInputTensors->useRandomAcceptanceThreshold = mUseRandomAcceptanceThreshold;
        decodeInputTensors->step = step;
        decodeInputTensors->useDraftLogits = mUseDraftLogits;
        decodeInputTensors->useDraftLogitsHost = mUseDraftLogitsHost;

        return decodeInputTensors;
    }

    void batchCopyDraftLogits();
    void batchCopyDraftTokenIds();
    void batchUseDraftLogits(bool useDraftLogits);
};

template <typename T>
void ExternalDraftTokensLayerTest<T>::batchCopyDraftLogits()
{
    auto const draftLogitsHost = ITensor::wrap(
        mTestDraftLogitsInit->data(), TRTDataType<T>::value, ITensor::makeShape({mMaxDraftLen, this->mVocabSize}));
    TLLM_CHECK(mTestDraftLogitsInit->size() == draftLogitsHost->getSize());

    for (int32_t bi = 0; bi < this->mBatchSize; ++bi)
    {
        auto draftLogitsDeviceView
            = ITensor::slice(mDraftLogits, bi * ExternalDraftTokensLayerTest::kDoubleBatchIdx, 1);
        this->mBufferManager->copy(*draftLogitsHost, *draftLogitsDeviceView);
    }
}

template <typename T>
void ExternalDraftTokensLayerTest<T>::batchCopyDraftTokenIds()
{
    auto numDraftTokensHostRange = BufferRange<SizeType32>(*mNumDraftTokensHost);

    for (int32_t bi = 0; bi < this->mBatchSize; ++bi)
    {
        auto batchSlot = bi * ExternalDraftTokensLayerTest::kDoubleBatchIdx;

        auto const& draftTokenIdsHost = mTestDraftTokenIdsInit.at(bi);
        numDraftTokensHostRange[batchSlot] = draftTokenIdsHost.size();

        auto draftTokenIdsDeviceView = ITensor::at(mDraftTokenIds, {batchSlot});
        TLLM_CHECK(draftTokenIdsDeviceView->getSize() == mMaxDraftLen);

        draftTokenIdsDeviceView->resize(draftTokenIdsHost.size());
        TLLM_CHECK(draftTokenIdsDeviceView->getSize() == draftTokenIdsHost.size());

        this->mBufferManager->copy(draftTokenIdsHost.data(), *draftTokenIdsDeviceView);
    }

    this->mBufferManager->copy(*this->mNumDraftTokensHost, *this->mNumDraftTokens);
}

template <typename T>
void ExternalDraftTokensLayerTest<T>::batchUseDraftLogits(bool useDraftLogits)
{
    auto useDraftLogitsHost = BufferRange<bool>(*this->mUseDraftLogitsHost);
    std::fill(useDraftLogitsHost.begin(), useDraftLogitsHost.end(), useDraftLogits);
    trk::invokeFill(*this->mUseDraftLogits, useDraftLogits, *this->mStream);
}

TYPED_TEST_SUITE(ExternalDraftTokensLayerTest, FloatAndHalfTypes);

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByLogitsTopK)
{
    SizeType32 topK = 2;
    float topP = 0.0f;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;
    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsAccept;

    // step 0, only token 4 and 5 (topK==2) get accepted
    // step 1, only token 0 and 1 gets accepted
    // step 2, only token 2 and 3 gets accepted
    // step 3, bonus step, token 0 and 1 can be sampled
    this->mTestDraftTokenIdsInit = {
        {4, 1, 2}, //
        {5, 0, 3}, //
        {4, 1, 2}, //
        {5, 0, 3}, //
        {4, 1, 2}, //
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {5}, {4}, {5}, {4}, {4, 5},             // step 0
        {1}, {0}, {1}, {0}, {1}, {0},                // step 1
        {2}, {3}, {2}, {3}, {2}, {0},                // step 2
        {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, // step 3
    };

    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByLogitsTopKReject)
{
    SizeType32 topK = 2;
    float topP = 0.0f;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsReject;

    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, // accept, accept, accept, sampled
        {4, 0, 2}, // accept, accept, accept, sampled
        {4, 3, 4}, // accept, reject, 0, 0
        {4, 3, 4}, // accept, reject, 0, 0
        {2, 3, 4}, // reject, 0, 0, 0
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4, 5, 6, 7}, {4, 5},       // step 0
        {0}, {0}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0}, {0}, // step 1
        {2}, {2}, {0}, {0}, {0}, {0},                   // step 2
        {0, 1}, {0, 1}, {0}, {0}, {0}, {0},             // step 3
    };

    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByLogitsTopK1TopP0)
{
    SizeType32 topK = 1;
    float topP = 0.0f;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsAccept;

    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {0}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}, // step 3
    };

    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByLogitsTopK1TopP0Reject)
{
    SizeType32 topK = 1;
    float topP = 0.0f;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsReject;

    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, // accept, accept, accept, sampled
        {4, 0, 3}, // accept, accept, reject, 0
        {4, 1, 2}, // accept, reject, 0, 0
        {4, 1, 2}, // accept, reject, 0, 0
        {5, 0, 2}, // reject, 0, 0, 0
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {0}, {0}, {0}, {0}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}, // step 3
    };

    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByLogitsBatchTopK)
{
    std::vector<SizeType32> topKs = {1, 1, 2, 2, 4, 4};
    float topP = 0.0f;
    TestSamplingParams params;
    params.topKs = {topKs};
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsAccept;

    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4, 5, 6, 7},                // step 0
        {0}, {0}, {0}, {0}, {0}, {0},                         // step 1
        {2}, {2}, {2}, {2}, {2}, {0},                         // step 2
        {0}, {0}, {0, 1}, {0, 1}, {0, 1, 2, 3}, {0, 1, 2, 3}, // step 3
    };

    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByLogitsBatchTopKReject)
{
    std::vector<SizeType32> topKs = {1, 1, 2, 2, 4, 4};
    float topP = 0.0f;
    TestSamplingParams params;
    params.topKs = {topKs};
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsReject;

    this->mTestDraftTokenIdsInit = {
        {},        // no draft tokens, token will be sampled
        {5, 0, 2}, // reject, 0, 0, 0
        {4, 0, 2}, // accept, accept, accept, sampled
        {4, 0, 4}, // accept, accept, reject, 0
        {4, 0, 2}, // accept, accept, accept, sampled
        {4, 5, 2}, // accept, reject, 0, 0
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4, 5, 6, 7}, {4}, {4}, {4}, {4},    // step 0
        {0}, {0}, {0}, {0}, {0}, {0, 1, 2, 3},    // step 1
        {0}, {0}, {2}, {2, 3}, {2}, {0},          // step 2
        {0}, {0}, {0, 1}, {0}, {0, 1, 2, 3}, {0}, // step 3
    };

    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByLogitsTopP)
{
    // Skip topK decode
    float topP = 0.3;
    TestSamplingParams params;
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsAccept;

    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {0}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}, // step 3
    };

    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByLogitsTopPReject)
{
    // Skip topK decode
    float topP = 0.3;
    TestSamplingParams params;
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsReject;

    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, // accept, accept, accept, sampled
        {4, 0, 2}, // accept, accept, reject, 0
        {4, 1, 3}, // accept, reject, 0, 0
        {7, 0, 2}, // reject, 0, 0, 0
        {7, 0, 2}, // reject, 0, 0, 0
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {0}, {0}, {0}, {0}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}, // step 3
    };

    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByLogitsTopKTopP)
{
    SizeType32 topK = 2;
    float topP = 0.3;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsAccept;

    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {0}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}, // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByLogitsTopKTopPReject)
{
    SizeType32 topK = 2;
    float topP = 0.3;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsReject;

    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, // accept, accept, accept, sampled
        {4, 0, 3}, // accept, accept, reject, 0
        {4, 3, 2}, // accept, reject, 0, 0
        {7, 0, 2}, // reject, 0, 0, 0
        {7, 0, 2}, // reject, 0, 0, 0
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {0}, {0}, {0}, {0}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}, // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByLogitsBatchTopKBatchTopP)
{
    std::vector<SizeType32> topKs = {3, 2, 1, 2, 2, 1};
    std::vector<float> topPs = {0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsAccept;

    this->mTestDraftTokenIdsInit = {
        {6, 2, 4}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {5, 1, 3}, //
        {4, 0, 2}, //
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {6}, {4}, {4}, {5}, {4}, {4},          // step 0
        {2}, {0}, {0}, {1}, {0}, {0},          // step 1
        {4}, {2}, {2}, {3}, {2}, {0},          // step 2
        {0, 1, 2}, {0}, {0}, {0, 1}, {0}, {0}, // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByLogitsBatchTopKBatchTopPReject)
{
    std::vector<SizeType32> topKs = {3, 2, 1, 2, 2, 1};
    std::vector<float> topPs = {0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsReject;

    this->mTestDraftTokenIdsInit = {
        {6, 3, 4}, // accept, reject, 0, 0
        {4, 0, 3}, // accept, accept, reject, 0
        {4, 2, 2}, // accept, reject, 0, 0
        {7, 1, 3}, // reject, 0, 0, 0
        {4, 0, 3}, // accept, accept, reject, 0
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {6}, {4}, {4, 5}, {4, 5}, {4}, {4}, // step 0
        {0, 1, 2}, {0}, {0}, {0}, {0}, {0}, // step 1
        {0}, {2}, {0}, {0}, {2}, {0},       // step 2
        {0, 1, 2}, {0}, {0}, {0}, {0}, {0}, // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByLogitsBatchTopK0BatchTopP)
{
    std::vector<SizeType32> topKs = {0, 0, 0, 0, 0, 0};
    std::vector<float> topPs = {1.0, 1.0, 0.5, 0.5, 0.3, 0.3};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsAccept;

    this->mTestDraftTokenIdsInit = {
        {7, 3, 5}, //
        {5, 1, 3}, //
        {5, 1, 3}, //
        {5, 1, 3}, //
        {4, 0, 2}, //
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {7}, {5}, {5}, {5}, {4}, {4},                         // step 0
        {3}, {1}, {1}, {1}, {0}, {0},                         // step 1
        {5}, {3}, {3}, {3}, {2}, {0},                         // step 2
        {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1}, {0, 1}, {0}, {0}, // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByLogitsBatchTopK0BatchTopPReject)
{
    std::vector<SizeType32> topKs = {0, 0, 0, 0, 0, 0};
    std::vector<float> topPs = {1.0, 1.0, 0.5, 0.5, 0.3, 0.3};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsReject;

    this->mTestDraftTokenIdsInit = {
        {7, 4, 5}, // accept, reject, 0, 0
        {5, 5, 3}, // accept, reject, 0, 0
        {6, 1, 3}, // reject, 0, 0, 0
        {5, 1, 3}, // accept, accept, accept, sampled
        {4, 2, 2}, // accept, reject, 0, 0
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {7}, {5}, {4, 5}, {5}, {4}, {4},                // step 0
        {0, 1, 2, 3}, {0, 1, 2, 3}, {0}, {1}, {0}, {0}, // step 1
        {0}, {0}, {0}, {3}, {0}, {0},                   // step 2
        {0}, {0}, {0}, {0, 1}, {0}, {0},                // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByTokenIdsBatchTopKBatchTopP)
{
    std::vector<SizeType32> topKs = {3, 2, 1, 2, 2, 1};
    std::vector<float> topPs = {0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = false;

    // accept by token ids result may different for different seeds
    // therefore there are more possible paths in expectedOutputIds
    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5, 6}, {4}, {4}, {4, 5}, {4}, {4},       // step 0
        {0, 1, 2}, {0}, {0}, {0, 1}, {0}, {0},       // step 1
        {0, 2, 3, 4}, {2}, {2}, {0, 2, 3}, {2}, {0}, // step 2
        {0, 1, 2}, {0}, {0}, {0, 1}, {0}, {0},       // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByTokenIdsBatchTopKBatchTopPReject)
{
    std::vector<SizeType32> topKs = {3, 2, 1, 2, 2, 1};
    std::vector<float> topPs = {0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = false;

    // accept by token ids result may different for different seeds
    // therefore there are more possible paths in expectedOutputIds
    this->mTestDraftTokenIdsInit = {
        {4, 3, 2}, // accept, reject, 0, 0
        {5, 0, 2}, // reject, 0, 0, 0
        {4, 0, 3}, // accept, accept, reject, 0
        {6, 0, 2}, // reject, 0, 0, 0
        {4, 1, 2}, // accept, reject, 0, 0
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5, 6}, {4}, {4}, {4, 5}, {4}, {4}, // step 0
        {0, 1, 2}, {0}, {0}, {0}, {0}, {0},    // step 1
        {0}, {0}, {2}, {0}, {0}, {0},          // step 2
        {0}, {0}, {0}, {0}, {0}, {0},          // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByTokenIdsBatchTopK0BatchTopP)
{
    std::vector<SizeType32> topKs = {0, 0, 0, 0, 0, 0};
    std::vector<float> topPs = {1.0, 1.0, 0.5, 0.5, 0.3, 0.3};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = false;

    // accept by token ids result may different for different seeds
    // therefore there are more possible paths in expectedOutputIds
    this->mTestDraftTokenIdsInit = {
        {7, 3, 5}, //
        {5, 1, 3}, //
        {5, 1, 3}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5, 6, 7}, {4, 5, 6, 7}, {4, 5}, {4, 5}, {4}, {4},             // step 0
        {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1}, {0, 1}, {0}, {0},             // step 1
        {0, 2, 3, 4, 5}, {0, 2, 3, 4, 5}, {0, 2, 3}, {0, 2, 3}, {2}, {0}, // step 2
        {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1}, {0, 1}, {0}, {0},             // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AcceptByTokenIdsBatchTopK0BatchTopPReject)
{
    std::vector<SizeType32> topKs = {0, 0, 0, 0, 0, 0};
    std::vector<float> topPs = {1.0, 1.0, 0.5, 0.5, 0.3, 0.3};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = false;

    // accept by token ids result may different for different seeds
    // therefore there are more possible paths in expectedOutputIds
    this->mTestDraftTokenIdsInit = {
        {7, 4, 5}, // accept, reject, 0, 0
        {5, 1, 6}, // accept/reject, accept/reject, reject, 0
        {6, 1, 3}, // reject, 0, 0, 0
        {4, 0, 2}, // accept/reject, accept/reject, accept, sampled
        {4, 1, 2}, // accept, reject, 0, 0
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5, 6, 7}, {4, 5, 6, 7}, {4, 5}, {4, 5}, {4}, {4}, // step 0
        {0, 1, 2, 3}, {0, 1, 2, 3}, {0}, {0, 1}, {0}, {0},    // step 1
        {0}, {0, 2, 3, 4, 5}, {0}, {0, 2, 3}, {0}, {0},       // step 2
        {0}, {0}, {0}, {0, 1}, {0}, {0},                      // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByLogitsTopK)
{
    SizeType32 topK = 2;
    float topP = 0.0f;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;
    params.isAirTopPExternalDraftTokensLayer = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsAccept;

    // step 0, only token 4 and 5 (topK==2) get accepted
    // step 1, only token 0 and 1 gets accepted
    // step 2, only token 2 and 3 gets accepted
    // step 3, bonus step, token 0 and 1 can be sampled
    this->mTestDraftTokenIdsInit = {
        {4, 1, 2}, //
        {5, 0, 3}, //
        {4, 1, 2}, //
        {5, 0, 3}, //
        {4, 1, 2}, //
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {5}, {4}, {5}, {4}, {4, 5},             // step 0
        {1}, {0}, {1}, {0}, {1}, {0},                // step 1
        {2}, {3}, {2}, {3}, {2}, {0},                // step 2
        {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, // step 3
    };

    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByLogitsTopKReject)
{
    SizeType32 topK = 2;
    float topP = 0.0f;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;
    params.isAirTopPExternalDraftTokensLayer = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsReject;

    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, // accept, accept, accept, sampled
        {4, 0, 2}, // accept, accept, accept, sampled
        {4, 3, 4}, // accept, reject, 0, 0
        {4, 3, 4}, // accept, reject, 0, 0
        {2, 3, 4}, // reject, 0, 0, 0
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4, 5, 6, 7}, {4, 5, 6, 7}, // step 0
        {0}, {0}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0}, {0}, // step 1
        {2}, {2}, {0}, {0}, {0}, {0},                   // step 2
        {0, 1}, {0, 1}, {0}, {0}, {0}, {0},             // step 3
    };

    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByLogitsTopK1TopP0)
{
    SizeType32 topK = 1;
    float topP = 0.0f;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;
    params.isAirTopPExternalDraftTokensLayer = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsAccept;

    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {0}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}, // step 3
    };

    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByLogitsTopK1TopP0Reject)
{
    SizeType32 topK = 1;
    float topP = 0.0f;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;
    params.isAirTopPExternalDraftTokensLayer = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsReject;

    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, // accept, accept, accept, sampled
        {4, 0, 3}, // accept, accept, reject, 0
        {4, 1, 2}, // accept, reject, 0, 0
        {4, 1, 2}, // accept, reject, 0, 0
        {5, 0, 2}, // reject, 0, 0, 0
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {0}, {0}, {0}, {0}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}, // step 3
    };

    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByLogitsBatchTopK)
{
    std::vector<SizeType32> topKs = {1, 1, 2, 2, 4, 4};
    float topP = 0.0f;
    TestSamplingParams params;
    params.topKs = {topKs};
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;
    params.isAirTopPExternalDraftTokensLayer = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsAccept;

    this->mTestDraftTokenIdsInit = {
        {},        // no draft tokens, token will be sampled
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4},                         // step 0
        {0}, {0}, {0}, {0}, {0}, {0},                         // step 1
        {0}, {2}, {2}, {2}, {2}, {2},                         // step 2
        {0}, {0}, {0, 1}, {0, 1}, {0, 1, 2, 3}, {0, 1, 2, 3}, // step 3
    };

    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByLogitsBatchTopKReject)
{
    std::vector<SizeType32> topKs = {1, 1, 2, 2, 4, 4};
    float topP = 0.0f;
    TestSamplingParams params;
    params.topKs = {topKs};
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;
    params.isAirTopPExternalDraftTokensLayer = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsReject;

    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, // accept, accept, accept, sampled
        {5, 0, 2}, // reject, 0, 0, 0
        {},        // no draft tokens, token will be sampled
        {4, 0, 4}, // accept, accept, reject, 0
        {4, 0, 2}, // accept, accept, accept, sampled
        {4, 5, 2}, // accept, reject, 0, 0
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4, 5, 6, 7}, {4, 5}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0, 1, 2, 3},    // step 1
        {2}, {0}, {0}, {2, 3}, {2}, {0},          // step 2
        {0}, {0}, {0}, {0}, {0, 1, 2, 3}, {0},    // step 3
    };

    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByLogitsTopP)
{
    // Skip topK decode
    float topP = 0.3;
    TestSamplingParams params;
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;
    params.isAirTopPExternalDraftTokensLayer = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsAccept;

    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {0}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}, // step 3
    };

    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByLogitsTopPReject)
{
    // Skip topK decode
    float topP = 0.3;
    TestSamplingParams params;
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;
    params.isAirTopPExternalDraftTokensLayer = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsReject;

    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, // accept, accept, accept, sampled
        {4, 0, 2}, // accept, accept, reject, 0
        {4, 1, 3}, // accept, reject, 0, 0
        {7, 0, 2}, // reject, 0, 0, 0
        {7, 0, 2}, // reject, 0, 0, 0
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {0}, {0}, {0}, {0}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}, // step 3
    };

    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByLogitsTopKTopP)
{
    SizeType32 topK = 2;
    float topP = 0.3;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;
    params.isAirTopPExternalDraftTokensLayer = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsAccept;

    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {4, 0, 2}, //
        {},        // no draft tokens, token will be sampled
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {0}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}, // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByLogitsTopKTopPReject)
{
    SizeType32 topK = 2;
    float topP = 0.3;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;
    params.isAirTopPExternalDraftTokensLayer = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsReject;

    this->mTestDraftTokenIdsInit = {
        {4, 0, 2}, // accept, accept, accept, sampled
        {4, 0, 3}, // accept, accept, reject, 0
        {4, 3, 2}, // accept, reject, 0, 0
        {7, 0, 2}, // reject, 0, 0, 0
        {7, 0, 2}, // reject, 0, 0, 0
        {7, 0, 2}, // reject, 0, 0, 0
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {0}, {0}, {0}, {0}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}, // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByLogitsBatchTopKBatchTopP)
{
    std::vector<SizeType32> topKs = {3, 2, 1, 2, 2, 1};
    std::vector<float> topPs = {0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;
    params.isAirTopPExternalDraftTokensLayer = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsAccept;

    this->mTestDraftTokenIdsInit = {
        {6, 2, 4},
        {4, 0, 2},
        {4, 0, 2},
        {5, 1, 3},
        {4, 0, 2},
        {4, 0, 2},
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {6}, {4}, {4}, {5}, {4}, {4},             // step 0
        {2}, {0}, {0}, {1}, {0}, {0},             // step 1
        {4}, {2}, {2}, {3}, {2}, {2},             // step 2
        {0, 1, 2}, {0}, {0}, {0, 1}, {0}, {0, 1}, // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByLogitsBatchTopKBatchTopPReject)
{
    std::vector<SizeType32> topKs = {3, 2, 1, 2, 2, 1};
    std::vector<float> topPs = {0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;
    params.isAirTopPExternalDraftTokensLayer = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsReject;

    this->mTestDraftTokenIdsInit = {
        {6, 3, 4}, // accept, reject, 0, 0
        {4, 0, 3}, // accept, accept, reject, 0
        {6, 0, 2}, // reject, 0, 0, 0
        {7, 1, 3}, // reject, 0, 0, 0
        {4, 0, 3}, // accept, accept, reject, 0
        {4, 2, 2}, // accept, reject, 0, 0
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {6}, {4}, {4, 5}, {4, 5}, {4}, {4},    // step 0
        {0, 1, 2}, {0}, {0}, {0}, {0}, {0, 1}, // step 1
        {0}, {2}, {0}, {0}, {2}, {0},          // step 2
        {0, 1, 2}, {0}, {0}, {0}, {0}, {0},    // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByLogitsBatchTopK0BatchTopP)
{
    std::vector<SizeType32> topKs = {0, 0, 0, 0, 0, 0};
    std::vector<float> topPs = {1.0, 1.0, 0.5, 0.5, 0.3, 0.3};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;
    params.isAirTopPExternalDraftTokensLayer = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsAccept;

    this->mTestDraftTokenIdsInit = {
        {7, 3, 5},
        {5, 1, 3},
        {5, 1, 3},
        {5, 1, 3},
        {4, 0, 2},
        {4, 0, 2},
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {7}, {5}, {5}, {5}, {4}, {4},                         // step 0
        {3}, {1}, {1}, {1}, {0}, {0},                         // step 1
        {5}, {3}, {3}, {3}, {2}, {2},                         // step 2
        {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1}, {0, 1}, {0}, {0}, // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByLogitsBatchTopK0BatchTopPReject)
{
    std::vector<SizeType32> topKs = {0, 0, 0, 0, 0, 0};
    std::vector<float> topPs = {1.0, 1.0, 0.5, 0.5, 0.3, 0.3};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = true;
    params.isAirTopPExternalDraftTokensLayer = true;

    // prob = (0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1)
    this->mTestDraftLogitsInit = &this->mTestDraftLogitsReject;

    this->mTestDraftTokenIdsInit = {
        {7, 4, 5}, // accept, reject, 0, 0
        {5, 5, 3}, // accept, reject, 0, 0
        {6, 1, 3}, // reject, 0, 0, 0
        {5, 1, 3}, // accept, accept, accept, sampled
        {4, 2, 2}, // accept, reject, 0, 0
        {4, 0, 3}, // accept, accept, reject, 0
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {7}, {5}, {4, 5}, {5}, {4}, {4},                // step 0
        {0, 1, 2, 3}, {0, 1, 2, 3}, {0}, {1}, {0}, {0}, // step 1
        {0}, {0}, {0}, {3}, {0}, {2},                   // step 2
        {0}, {0}, {0}, {0, 1}, {0}, {0},                // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByTokenIdsBatchTopKBatchTopP)
{
    std::vector<SizeType32> topKs = {3, 2, 1, 2, 2, 1};
    std::vector<float> topPs = {0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = false;
    params.isAirTopPExternalDraftTokensLayer = true;

    // accept by token ids result may different for different seeds
    // therefore there are more possible paths in expectedOutputIds
    this->mTestDraftTokenIdsInit = {
        {4, 0, 2},
        {4, 0, 2},
        {4, 0, 2},
        {4, 0, 2},
        {4, 0, 2},
        {4, 0, 2},
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5, 6}, {4}, {4}, {4, 5}, {4}, {4},       // step 0
        {0, 1, 2}, {0}, {0}, {0, 1}, {0}, {0},       // step 1
        {0, 2, 3, 4}, {2}, {2}, {0, 2, 3}, {2}, {2}, // step 2
        {0, 1, 2}, {0}, {0}, {0, 1}, {0}, {0},       // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByTokenIdsBatchTopKBatchTopPReject)
{
    std::vector<SizeType32> topKs = {3, 2, 1, 2, 2, 1};
    std::vector<float> topPs = {0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = false;
    params.isAirTopPExternalDraftTokensLayer = true;

    // accept by token ids result may different for different seeds
    // therefore there are more possible paths in expectedOutputIds
    this->mTestDraftTokenIdsInit = {
        {4, 3, 2}, // accept, reject, 0, 0
        {5, 0, 2}, // reject, 0, 0, 0
        {4, 0, 3}, // accept, accept, reject, 0
        {6, 0, 2}, // reject, 0, 0, 0
        {4, 1, 2}, // accept, reject, 0, 0
        {4, 1, 2}, // accept, reject, 0, 0
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5, 6}, {4}, {4}, {4, 5}, {4}, {4}, // step 0
        {0, 1, 2}, {0}, {0}, {0}, {0}, {0},    // step 1
        {0}, {0}, {2}, {0}, {0}, {0},          // step 2
        {0}, {0}, {0}, {0}, {0}, {0},          // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByTokenIdsBatchTopK0BatchTopP)
{
    std::vector<SizeType32> topKs = {0, 0, 0, 0, 0, 0};
    std::vector<float> topPs = {1.0, 1.0, 0.5, 0.5, 0.3, 0.3};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = false;
    params.isAirTopPExternalDraftTokensLayer = true;

    // accept by token ids result may different for different seeds
    // therefore there are more possible paths in expectedOutputIds
    this->mTestDraftTokenIdsInit = {
        {7, 3, 5},
        {5, 1, 3},
        {5, 1, 3},
        {4, 0, 2},
        {4, 0, 2},
        {4, 0, 2},
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5, 6, 7}, {4, 5, 6, 7}, {4, 5}, {4, 5}, {4}, {4},             // step 0
        {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1}, {0, 1}, {0}, {0},             // step 1
        {0, 2, 3, 4, 5}, {0, 2, 3, 4, 5}, {0, 2, 3}, {0, 2, 3}, {2}, {2}, // step 2
        {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1}, {0, 1}, {0}, {0},             // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, AirTopPAcceptByTokenIdsBatchTopK0BatchTopPReject)
{
    std::vector<SizeType32> topKs = {0, 0, 0, 0, 0, 0};
    std::vector<float> topPs = {1.0, 1.0, 0.5, 0.5, 0.3, 0.3};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = false;
    params.isAirTopPExternalDraftTokensLayer = true;

    // accept by token ids result may different for different seeds
    // therefore there are more possible paths in expectedOutputIds
    this->mTestDraftTokenIdsInit = {
        {7, 4, 5}, // accept, reject, 0, 0
        {5, 1, 6}, // accept/reject, accept/reject, reject, 0
        {6, 1, 3}, // reject, 0, 0, 0
        {4, 0, 2}, // accept/reject, accept/reject, accept, sampled
        {4, 1, 2}, // accept, reject, 0, 0
        {4, 0, 3}, // accept, accept, reject, 0
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5, 6, 7}, {4, 5, 6, 7}, {4, 5}, {4, 5}, {4}, {4}, // step 0
        {0, 1, 2, 3}, {0, 1, 2, 3}, {0}, {0, 1}, {0}, {0},    // step 1
        {0}, {0, 2, 3, 4, 5}, {0}, {0, 2, 3}, {0}, {2},       // step 2
        {0}, {0}, {0}, {0, 1}, {0}, {0},                      // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(ExternalDraftTokensLayerTest, BatchTopKBatchTopP)
{
    std::vector<SizeType32> topKs = {3, 2, 1, 2, 2, 1};
    std::vector<float> topPs = {0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    params.isExternalDraftTokensLayerTest = true;
    params.useDraftLogits = false;

    this->mTestDraftTokenIdsInit = {
        {},
        {},
        {},
        {},
        {},
        {},
    };

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5, 6}, {4}, {4}, {4, 5}, {4}, {4}, // step 0
        {0}, {0, 1}, {0}, {0}, {0}, {0},       // step 1
        {0}, {0, 1}, {0}, {0}, {0}, {0},       // step 2
        {0}, {0, 1}, {0}, {0}, {0}, {0},       // step 3
    };
    this->runTest(expectedOutputIds, params);
}

} // namespace
