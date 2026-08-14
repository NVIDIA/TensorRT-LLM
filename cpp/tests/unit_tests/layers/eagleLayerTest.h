/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/layers/eagleDecodingLayer.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include <gtest/gtest.h>

#include <memory>

namespace tensorrt_llm::tests::layers
{

class SamplingParams
{
public:
    SamplingParams() {}

    inline void setBatchSize(runtime::SizeType32 batchSize)
    {
        mBatchSize = batchSize;
    }

    inline void setMaxPathLen(runtime::SizeType32 maxPathLen)
    {
        mMaxPathLen = maxPathLen;
    }

    inline void setMaxDecodingTokens(runtime::SizeType32 maxDecodingTokens)
    {
        mMaxDecodingTokens = maxDecodingTokens;
    }

    [[nodiscard]] inline runtime::SizeType32 getBatchSize() const
    {
        return mBatchSize;
    }

    [[nodiscard]] inline runtime::SizeType32 getVocabSize() const
    {
        return mVocabSize;
    }

    [[nodiscard]] inline runtime::SizeType32 getMaxBatchSize() const
    {
        return 2 * getBatchSize();
    }

    [[nodiscard]] inline runtime::SizeType32 getMaxPathLen() const
    {
        return mMaxPathLen;
    }

    [[nodiscard]] inline runtime::SizeType32 getMaxDraftPathLen() const
    {
        return getMaxPathLen() - 1;
    }

    [[nodiscard]] inline runtime::SizeType32 getMaxDecodingTokens() const
    {
        return mMaxDecodingTokens;
    }

    [[nodiscard]] inline runtime::SizeType32 getMaxDecodingDraftTokens() const
    {
        return getMaxDecodingTokens() - 1;
    }

    [[nodiscard]] inline runtime::SizeType32 getMaxSeqLen() const
    {
        return getMaxDecodingTokens() * 2;
    }

    [[nodiscard]] inline runtime::TokenIdType getPadId() const
    {
        return mPadId;
    }

private:
    runtime::SizeType32 mBatchSize{6};
    runtime::SizeType32 mMaxPathLen{4};
    runtime::SizeType32 mMaxDecodingTokens{32};
    runtime::SizeType32 mVocabSize{256};
    runtime::TokenIdType mPadId{-1};
};

using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;
using BufferPtr = tensorrt_llm::runtime::IBuffer::SharedPtr;
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using TokenIdType = tensorrt_llm::runtime::TokenIdType;

using TokensVec = std::vector<TokenIdType>;
using DraftLettersVec = std::vector<std::string>;
using DraftTokensVec = std::vector<TokensVec>;
using DraftPath = std::vector<std::vector<SizeType32>>;
using DraftPaths = std::vector<DraftPath>;

class EagleDummyNetwork
{
public:
    void forward(SamplingParams const& params, std::vector<std::string> const& prompts,
        std::vector<std::vector<std::string>> const& predictionLetters,
        std::vector<DraftLettersVec> const& nextDraftLetters, std::vector<DraftLettersVec> const& lastDraftLetters);

    TokensVec tokenize(std::string const& letters) const;

    std::string detokenize(TokensVec const& tokens) const;

    SizeType32 longestCommonPrefixLength(TokensVec const& a, TokensVec const& b) const;

    DraftTokensVec draftLettersToTokens(DraftLettersVec const& draftLetters) const;

    DraftPath pathFromDraftTokens(
        DraftTokensVec const& tokens, SizeType32 maxDecodingTokens, SizeType32 maxPathLen) const;

    TokensVec flattenTokens(DraftTokensVec const& tokens, DraftPath const& path, bool isDraftTokens) const;

    void acceptTokens(std::vector<TokensVec> const& predictionTokens, DraftTokensVec const& lastDraftTokens,
        DraftPaths const& lastDraftPaths);

    std::vector<std::vector<std::vector<bool>>> createMasks(DraftPaths const& paths) const;

    void setSamplingParams(SamplingParams const& params)
    {
        mSamplingParams = params;
    }

    std::vector<TokensVec> getPrompts() const
    {
        return mPrompts;
    }

    std::vector<TokensVec> getOutputIds() const
    {
        return mOutputIds;
    }

    DraftTokensVec getNextDraftTokens() const
    {
        return mNextDraftTokens;
    }

    std::vector<SizeType32> getNextDraftLens() const
    {
        return mNextDraftLens;
    }

    DraftPaths getNextDraftPaths() const
    {
        return mNextDraftPaths;
    }

    DraftTokensVec getLastDraftTokens() const
    {
        return mLastDraftTokens;
    }

    std::vector<SizeType32> getLastDraftLens() const
    {
        return mLastDraftLens;
    }

    DraftPaths getLastDraftPaths() const
    {
        return mLastDraftPaths;
    }

    std::vector<TokensVec> getAcceptedTokens() const
    {
        return mAcceptedTokens;
    }

    std::vector<SizeType32> getAcceptedLens() const
    {
        return mAcceptedLens;
    }

    std::vector<SizeType32> getAcceptedPathIds() const
    {
        return mAcceptedPathIds;
    }

    std::vector<std::vector<std::vector<bool>>> getNextMasks() const
    {
        return mMasks;
    }

private:
    SamplingParams mSamplingParams;

    std::vector<TokensVec> mPrompts;
    std::vector<TokensVec> mOutputIds;

    DraftTokensVec mNextDraftTokens;
    std::vector<SizeType32> mNextDraftLens;
    DraftPaths mNextDraftPaths;

    DraftTokensVec mLastDraftTokens;
    std::vector<SizeType32> mLastDraftLens;
    DraftPaths mLastDraftPaths;

    std::vector<TokensVec> mAcceptedTokens;
    std::vector<SizeType32> mAcceptedLens;
    std::vector<SizeType32> mAcceptedPathIds;

    std::vector<std::vector<std::vector<bool>>> mMasks;
};

template <typename T>
class EagleDecodingLayerTest : public testing::Test
{
private:
    void SetUp() override;

private:
    SamplingParams mSamplingParams;

    // Outputs
    TensorPtr mOutputIds;
    TensorPtr mSeqLengths;
    TensorPtr mOutputNextDraftTokens;
    TensorPtr mOutputUnpackedNextDraftTokens;
    TensorPtr mAcceptedLengths;
    TensorPtr mNextPosIds;
    TensorPtr mPrevDraftLengths;
    TensorPtr mNextDraftLengths;
    TensorPtr mNextGenerationLengths;
    TensorPtr mNextGenerationLengthsHost;
    TensorPtr mAcceptedLengthCumSum;
    TensorPtr mPathsOffsets;
    TensorPtr mPackedMasks;
    TensorPtr mRandomDataSample;
    TensorPtr mRandomDataValidation;
    TensorPtr mOutputTemperatures;
    TensorPtr mOutputNextDraftPaths;
    TensorPtr mEagleNetCtxRequestTypesHost;
    TensorPtr mEagleNetCtxContextLengthsHost;
    TensorPtr mEagleNetCtxPastKeyValueLengthsHost;
    TensorPtr mEagleNetGenRequestTypesHost;
    TensorPtr mEagleNetGenContextLengthsHost;
    TensorPtr mEagleNetGenPastKeyValueLengthsHost;

    // inputs
    TensorPtr mBatchSlots;
    TensorPtr mEndIds;

    TensorPtr mInputNextDraftTokens;
    TensorPtr mInputNextDraftLens;
    TensorPtr mInputNextDraftPaths;
    TensorPtr mInputLastDraftTokens;
    TensorPtr mInputLastDraftLens;
    TensorPtr mInputLastDraftPaths;
    TensorPtr mInputAcceptedTokens;
    TensorPtr mInputAcceptedLens;
    TensorPtr mInputAcceptedPathIds;
    TensorPtr mChunkedContextNextTokens;

    // Setup params
    std::vector<uint64_t> mRandomSeeds;
    std::vector<float> mTemperatures;

    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::layers::EagleDecodingLayer<T>> mEagleLayer;
    std::shared_ptr<runtime::DecodingLayerWorkspace> mDecodingWorkspace;

    EagleDummyNetwork mNetwork;

private:
    void allocateBuffers();

    void setup();

    std::shared_ptr<tensorrt_llm::layers::EagleInputs> createInputTensors();

    std::shared_ptr<tensorrt_llm::layers::EagleOutputs> createOutputTensors();

    void checkLayerResult();

public:
    void runTest(std::vector<std::string> const& prompts, std::vector<DraftLettersVec> const& predictions,
        std::vector<DraftLettersVec> const& nextDraftLetters, std::vector<DraftLettersVec> const& lastDraftLetters,
        SamplingParams& params);
};

typedef testing::Types<float, half> FloatAndHalfTypes;

} // namespace tensorrt_llm::tests::layers
