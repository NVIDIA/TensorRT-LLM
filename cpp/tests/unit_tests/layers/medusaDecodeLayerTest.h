/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/layers/medusaDecodingLayer.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"

namespace tensorrt_llm::tests::layers
{

struct SamplingParams
{
    tensorrt_llm::runtime::SizeType32 batchSize;
    std::vector<tensorrt_llm::runtime::SizeType32> runtimeTopK;
    std::vector<std::vector<tensorrt_llm::runtime::SizeType32>> runtimeHeadsTopK;
    std::vector<std::vector<tensorrt_llm::runtime::TokenIdType>> draftIds;
    std::vector<std::vector<tensorrt_llm::runtime::SizeType32>> paths;
    std::vector<std::vector<tensorrt_llm::runtime::SizeType32>> treeIds;
    std::vector<tensorrt_llm::runtime::SizeType32> tokensPerStep;
    std::vector<tensorrt_llm::runtime::SizeType32> acceptedCumSum;
    std::vector<tensorrt_llm::runtime::SizeType32> packedPaths;
    std::optional<tensorrt_llm::runtime::TokenIdType> endId;
};

template <typename T>
class MedusaDecodingLayerTest : public testing::Test
{
private:
    void SetUp() override;

public:
    using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;
    using BufferPtr = tensorrt_llm::runtime::IBuffer::SharedPtr;
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TokenIdType = tensorrt_llm::runtime::TokenIdType;

private:
    SizeType32 mBatchSize{6};
    SizeType32 mMaxBatchSize{2 * mBatchSize};
    SizeType32 const mVocabSize{9};
    SizeType32 const mVocabSizePadded{mVocabSize};
    SizeType32 const mMaxDecodingTokens{12};
    SizeType32 const mMaxDraftPathLen{4};

    SizeType32 const mMaxSeqLen{mMaxDecodingTokens};
    TokenIdType mEndId{mVocabSize};

    bool mUseLogitsVec{false};

    TensorPtr mTargetLogitsDevice;
    TensorPtr mMedusaLogitsDevice;

    TensorPtr mFinishedDevice;
    TensorPtr mSeqLengthsDevice;
    TensorPtr mAcceptedLengths;
    TensorPtr mOutputIdsDevice;
    TensorPtr mNextDraftTokensDevice;

    TensorPtr mPathsDevice;
    TensorPtr mTreeIdsDevice;
    TensorPtr mAcceptedLengthCumSumDevice;
    TensorPtr mPackedPathsDevice;
    TensorPtr mEndIdsDevice;
    TensorPtr mBatchSlots;

    TensorPtr mTokensPerStepDevice;

    std::vector<TensorPtr> mLogitsVec;

    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::layers::MedusaDecodingLayer<T>> mMedusaDecodingLayer;
    std::shared_ptr<runtime::DecodingLayerWorkspace> mDecodingWorkspace;

private:
    void allocateBuffers();

    void setup(SamplingParams& params);

    std::shared_ptr<tensorrt_llm::layers::MedusaDecodingInputs> createInputTensors();

    std::shared_ptr<tensorrt_llm::layers::SpeculativeDecodingOutputs> createOutputTensors();

    void checkResult(std::vector<std::vector<std::set<TokenIdType>>> const& expectedOutTokens,
        std::vector<std::vector<TokenIdType>> const& expectedDraftTokens, std::vector<bool> const& finished,
        SamplingParams& params);

public:
    void runTest(std::vector<std::vector<std::set<TokenIdType>>> const& expectedOutTokens,
        std::vector<std::vector<TokenIdType>> const& expectedDraftTokens, std::vector<bool> const& finished,
        SamplingParams& params);
};

typedef testing::Types<float, half> FloatAndHalfTypes;

} // namespace tensorrt_llm::tests::layers
