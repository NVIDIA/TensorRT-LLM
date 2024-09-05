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

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include "tensorrt_llm/executor/types.h"

namespace tensorrt_llm::tests::layers::sampling
{

struct TestSamplingParams
{
    std::vector<runtime::SizeType32> topKs;
    std::vector<float> topPs;
    std::vector<float> temperatures;
    std::vector<float> repetitionPenalties;
    std::vector<float> presencePenalties;
    std::vector<float> frequencyPenalties;
    std::vector<runtime::SizeType32> minLengths;
    std::vector<float> decay;
    std::vector<float> minTopP;
    std::vector<runtime::TokenIdType> topPResetIds;
    std::vector<std::vector<std::vector<runtime::TokenIdType>>> badWords;
    std::vector<std::vector<std::vector<runtime::TokenIdType>>> stopWords;
    std::vector<runtime::SizeType32> repeatNGramSizes;
    bool useBias{false};

    std::optional<executor::DecodingMode> decodingMode;

    // Medusa setup
    std::optional<runtime::SizeType32> maxNumMedusaHeads{std::nullopt};
    std::optional<std::vector<std::vector<runtime::SizeType32>>> topKMedusaHeads{std::nullopt};
    std::optional<std::vector<runtime::SizeType32>> tokensPerStep{std::nullopt};
    std::optional<std::vector<std::vector<tensorrt_llm::runtime::SizeType32>>> paths;
    std::optional<std::vector<std::vector<tensorrt_llm::runtime::TokenIdType>>> outputIds;
};

template <typename T>
class DynamicDecodeLayerTest : public testing::Test
{
private:
    void SetUp() override;

    using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;
    using TensorConstPtr = tensorrt_llm::runtime::ITensor::SharedConstPtr;
    using BufferPtr = tensorrt_llm::runtime::IBuffer::SharedPtr;

    static uint64_t const mMaxSeed{64};
    runtime::SizeType32 const mBatchSize{6};
    runtime::SizeType32 const mMaxBatchSize{2 * mBatchSize};
    runtime::SizeType32 const mBeamWidth{1};
    runtime::SizeType32 const mBatchBeam{mBatchSize * mBeamWidth};
    runtime::SizeType32 const mVocabSize{9};
    runtime::SizeType32 const mVocabSizePadded{mVocabSize};

    runtime::SizeType32 const mMaxInputLen{0}; // has no effect.
    runtime::SizeType32 const mMaxOutputLen{4};
    runtime::SizeType32 const mMaxSeqLen{mMaxInputLen + mMaxOutputLen};
    runtime::SizeType32 const mSinkTokenLength{0};
    runtime::TokenIdType mEndId = mVocabSize;
    runtime::SizeType32 mMaxTokensPerStep{1};
    runtime::SizeType32 mMaxMedusaHeads{0};

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

    std::vector<TensorConstPtr> mLogitsVec;

    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::unique_ptr<tensorrt_llm::layers::DynamicDecodeLayer<T>> mDecodeLayer;
    std::shared_ptr<runtime::DecodingLayerWorkspace> mDecodingWorkspace;

    std::vector<T> mTestLogitsInit;

    runtime::SizeType32 mMaxBadWordsLen{0};
    runtime::SizeType32 mMaxStopWordsLen{0};

    executor::DecodingMode mDecodingMode = executor::DecodingMode::Auto();

private:
    void allocateMedusaData(TestSamplingParams const& params);

    void setup(uint64_t seed, TestSamplingParams const& params);

    runtime::SizeType32 getMaxWordsLen(std::vector<std::vector<std::vector<runtime::TokenIdType>>> const& inputWords);
    void initXWordsTensors(runtime::SizeType32* batchSlotsPtr, runtime::TokenIdType* wordsData,
        runtime::TokenIdType** wordsPtr, runtime::SizeType32* wordsLenData, runtime::SizeType32 maxWordsLen,
        std::vector<std::vector<std::vector<runtime::TokenIdType>>> const& inputWords);

    std::shared_ptr<tensorrt_llm::layers::DecodingInputs> createInputTensors(runtime::SizeType32 step);

    std::shared_ptr<tensorrt_llm::layers::BaseDecodingOutputs> createOutputTensors();

    void batchCopy(runtime::SizeType32 step);
    bool checkResult(runtime::TokenIdType* outputIds, std::vector<std::set<runtime::TokenIdType>> const& expectedIds,
        runtime::SizeType32* seqLens, runtime::SizeType32 leadingDim, runtime::SizeType32 stride,
        runtime::SizeType32 step, bool outputIdsTransposed = false, runtime::SizeType32 strideTransposed = 0);

    void fillRefLogits(runtime::SizeType32 const* seqLenHost,
        std::vector<std::set<runtime::TokenIdType>> const& expectedOutputIds, runtime::SizeType32 step);

    void createMedusaInputs(std::shared_ptr<tensorrt_llm::layers::DecodingInputs>& baseInputs);
    void createMedusaOutputs(std::shared_ptr<tensorrt_llm::layers::BaseDecodingOutputs>& baseOutputs);

public:
    void runTest(std::vector<std::set<runtime::TokenIdType>> const& expectedOutputIds, TestSamplingParams const& params,
        runtime::TokenIdType endId = -1);

    void allocateData(TestSamplingParams const& params, runtime::TokenIdType endId = -1);

    void runTestImpl(std::vector<std::set<runtime::TokenIdType>> const& expectedOutputIds,
        TestSamplingParams const& params, runtime::TokenIdType endId = -1);
};

typedef testing::Types<float, half> FloatAndHalfTypes;

} // namespace tensorrt_llm::tests::layers::sampling
