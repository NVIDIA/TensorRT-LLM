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
#pragma once

#include <gtest/gtest.h>

#include <memory>

#include "tensorrt_llm/layers/dynamicDecodeLayer.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include "tensorrt_llm/kernels/penaltyKernels.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include "tensorrt_llm/common/cudaAllocator.h"
#include "tensorrt_llm/common/tensorConversion.h"
#include "tensorrt_llm/common/tllmException.h"

namespace tensorrt_llm::tests::layers::sampling
{

struct TestSamplingParams
{
    std::vector<runtime::SizeType> topKs;
    std::vector<float> topPs;
    std::vector<float> temperatures;
    std::vector<float> repetitionPenalties;
    std::vector<float> presencePenalties;
    std::vector<float> frequencyPenalties;
    std::vector<runtime::SizeType> minLengths;
    std::vector<float> decay;
    std::vector<float> minTopP;
    std::vector<runtime::TokenIdType> topPResetIds;
    std::vector<std::vector<std::vector<runtime::TokenIdType>>> badWords;
    std::vector<std::vector<std::vector<runtime::TokenIdType>>> stopWords;
    bool useBias{false};

    // Medusa setup
    bool useMedusa{false};
    std::optional<runtime::SizeType> maxNumMedusaHeads{std::nullopt};
    std::optional<std::vector<std::vector<runtime::SizeType>>> topKMedusaHeads{std::nullopt};
    std::optional<std::vector<runtime::SizeType>> tokensPerStep{std::nullopt};
    std::optional<std::vector<std::vector<tensorrt_llm::runtime::SizeType>>> paths;
    std::optional<std::vector<std::vector<tensorrt_llm::runtime::TokenIdType>>> outputIds;
};

template <typename T>
class DynamicDecodeLayerTest : public testing::Test
{
private:
    void SetUp() override;

    using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;
    using BufferPtr = tensorrt_llm::runtime::IBuffer::SharedPtr;

    static const uint64_t mMaxSeed{64};
    runtime::SizeType const mBatchSize{6};
    runtime::SizeType const mMaxBatchSize{2 * mBatchSize};
    runtime::SizeType const mBeamWidth{1};
    runtime::SizeType const mBatchBeam{mBatchSize * mBeamWidth};
    runtime::SizeType const mVocabSize{9};
    runtime::SizeType const mVocabSizePadded{mVocabSize};

    runtime::SizeType const mMaxInputLen{0}; // has no effect.
    runtime::SizeType const mMaxOutputLen{4};
    runtime::SizeType const mMaxSeqLen{mMaxInputLen + mMaxOutputLen};
    runtime::SizeType const mSinkTokenLength{0};
    runtime::TokenIdType mEndId = mVocabSize;
    runtime::SizeType mMaxTokensPerStep{1};
    runtime::SizeType mMaxMedusaHeads{0};

    bool mUseLogitsVec{false};

    TensorPtr mLogitsDevice;
    TensorPtr mRuntimeLogitsHost;
    TensorPtr mLogitsRefHost;
    TensorPtr mContextLengthDevice;
    TensorPtr mSeqLengthsDevice;
    TensorPtr mFinishedDevice;
    TensorPtr mFinishedSumDevice;
    TensorPtr mOutputIdsDevice;
    TensorPtr mNewTokens;
    TensorPtr mEndIdsDevice;
    TensorPtr mBatchSlots;

    TensorPtr mBadWordsLens;
    TensorPtr mBadWords;
    TensorPtr mBadWordsPtrs;

    TensorPtr mStopWordsLens;
    TensorPtr mStopWords;
    TensorPtr mStopWordsPtrs;

    TensorPtr mEmbeddingBiasHost;
    TensorPtr mEmbeddingBiasDevice;

    TensorPtr mRefLogProbsHost;
    TensorPtr mOutputLogProbsDevice;
    TensorPtr mOutputLogProbsTiledDevice;

    TensorPtr mCumLogProbsDevice;

    // Medusa tensors
    TensorPtr mPathsDevice;
    TensorPtr mTreeIdsDevice;
    TensorPtr mAcceptedLengths;
    TensorPtr mAcceptedLengthCumSumDevice;
    TensorPtr mPackedPathsDevice;
    TensorPtr mMedusaLogitsDevice;
    TensorPtr mNextDraftTokensDevice;
    TensorPtr mTokensPerStepDevice;

    std::vector<tensorrt_llm::common::Tensor> mLogitsVec;

    // Order is important because we pass mAllocator to mDecodeLayer and it is used in destructor
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::common::CudaAllocator> mAllocator;
    std::unique_ptr<tensorrt_llm::layers::DynamicDecodeLayer<T>> mDecodeLayer;

    std::vector<T> mTestLogitsInit;

    runtime::SizeType mMaxBadWordsLen{0};
    runtime::SizeType mMaxStopWordsLen{0};

    bool mUseMedusa{false};

private:
    void allocateData(TestSamplingParams const& params);
    void allocateMedusaData(TestSamplingParams const& params);

    void setup(uint64_t seed, TestSamplingParams const& params);

    runtime::SizeType getMaxWordsLen(std::vector<std::vector<std::vector<runtime::TokenIdType>>> const& inputWords);
    void initXWordsTensors(runtime::SizeType* batchSlotsPtr, runtime::TokenIdType* wordsData,
        runtime::TokenIdType** wordsPtr, runtime::SizeType* wordsLenData, runtime::SizeType maxWordsLen,
        std::vector<std::vector<std::vector<runtime::TokenIdType>>> const& inputWords);

    std::shared_ptr<tensorrt_llm::layers::DynamicDecodeInputParams> createInputTensors(runtime::SizeType step);

    std::shared_ptr<tensorrt_llm::layers::DynamicDecodeOutputParams> createOutputTensors();

    void batchCopy(runtime::SizeType step);
    bool checkResult(runtime::TokenIdType* outputIds, std::vector<std::set<runtime::TokenIdType>> const& expectedIds,
        runtime::SizeType* seqLens, runtime::SizeType leadingDim, runtime::SizeType stride, runtime::SizeType step,
        bool outputIdsTransposed = false, runtime::SizeType strideTransposed = 0);

    void runTestImpl(std::vector<std::set<runtime::TokenIdType>> const& expectedOutputIds,
        TestSamplingParams const& params, runtime::TokenIdType endId = -1);

    void fillRefLogits(runtime::SizeType const* seqLenHost,
        std::vector<std::set<runtime::TokenIdType>> const& expectedOutputIds, runtime::SizeType step);

    tensorrt_llm::layers::DynamicDecodeInputParams::MedusaInputs createMedusaInputs();
    tensorrt_llm::layers::DynamicDecodeOutputParams::MedusaOutputs createMedusaOutputs();

public:
    void runTest(std::vector<std::set<runtime::TokenIdType>> const& expectedOutputIds, TestSamplingParams const& params,
        runtime::TokenIdType endId = -1);
};

typedef testing::Types<float, half> FloatAndHalfTypes;

} // namespace tensorrt_llm::tests::layers::sampling
