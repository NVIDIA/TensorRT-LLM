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

#include "tensorrt_llm/layers/explicitDraftTokensLayer.h"
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

    inline void setMaxNumPaths(runtime::SizeType32 maxNumPaths)
    {
        mMaxNumPaths = maxNumPaths;
    }

    inline void setMaxDraftPathLen(runtime::SizeType32 maxDraftPathLen)
    {
        mMaxDraftPathLen = maxDraftPathLen;
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

    [[nodiscard]] inline runtime::SizeType32 getMaxDraftPathLen() const
    {
        return mMaxDraftPathLen;
    }

    [[nodiscard]] inline runtime::SizeType32 getMaxPathLen() const
    {
        return getMaxDraftPathLen() + 1;
    }

    [[nodiscard]] inline runtime::SizeType32 getMaxNumPaths() const
    {
        return mMaxNumPaths;
    }

    [[nodiscard]] inline runtime::SizeType32 getMaxDecodingDraftTokens() const
    {
        return getMaxDraftPathLen() * getMaxNumPaths();
    }

    [[nodiscard]] inline runtime::SizeType32 getMaxDecodingTokens() const
    {
        return getMaxDecodingDraftTokens() + 1;
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
    runtime::SizeType32 mMaxDraftPathLen{6};
    runtime::SizeType32 mMaxNumPaths{4};
    runtime::TokenIdType mPadId{-1};
    runtime::SizeType32 mVocabSize{256};
};

using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;
using BufferPtr = tensorrt_llm::runtime::IBuffer::SharedPtr;
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using TokenIdType = tensorrt_llm::runtime::TokenIdType;

using TokensVec = std::vector<TokenIdType>;
using DraftLettersVec = std::vector<std::vector<std::string>>;
using DraftTokensVec = std::vector<std::vector<TokensVec>>;
using DraftTokensIndices = std::vector<std::vector<std::vector<SizeType32>>>;

class ExplicitDraftTokensDummyNetwork
{
public:
    void forward(SamplingParams const& params, std::vector<std::string> const& prompts,
        std::vector<std::string> const& predictionLetters, DraftLettersVec const& nextDraftLetters,
        DraftLettersVec const& lastDraftLetters);

    TokensVec tokenize(std::string const& letters) const;

    std::string detokenize(TokensVec const& tokens) const;

    DraftTokensVec draftLettersToTokens(DraftLettersVec const& draftLetters) const;

    SizeType32 longestCommonPrefixLength(TokensVec const& a, TokensVec const& b) const;

    SizeType32 computeCompressedVectorAndIndices(TokensVec& compressedVector, std::vector<SizeType32>& packedPosIds,
        DraftTokensIndices& indices, std::vector<TokensVec> const& vectors, SizeType32 basePosId);

    void compressTokens(TokensVec& compressedVector, std::vector<SizeType32>& packedPosIds, DraftTokensIndices& indices,
        std::vector<SizeType32>& generationLengths, DraftTokensVec const& draftTokens,
        std::vector<SizeType32> const& basePosIds);

    void acceptTokens(std::vector<TokensVec> const& predictionTokens, DraftTokensVec const& lastDraftTokens,
        DraftTokensVec const& nextDraftTokens);

    void createNextMasks(DraftTokensIndices const& indices, DraftTokensVec const& draftTokens, SizeType32 maxGenLength);

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

    TokensVec getNextFlatTokens() const
    {
        return mNextCompressedVector;
    }

    DraftTokensVec getNextDraftTokens() const
    {
        return mNextDraftTokens;
    }

    DraftTokensIndices getNextDraftIndices() const
    {
        return mNextDraftTokenIndices;
    }

    DraftTokensIndices getLastDraftIndices() const
    {
        return mLastDraftTokenIndices;
    }

    DraftTokensVec getLastDraftTokens() const
    {
        return mLastDraftTokens;
    }

    std::vector<SizeType32> getBestPathLengths() const
    {
        return mBestPathLengths;
    }

    std::vector<SizeType32> getBestPathIndices() const
    {
        return mBestPathIndices;
    }

    std::vector<SizeType32> getNextPackedPosId() const
    {
        return mNextPackedPosIds;
    }

    std::vector<SizeType32> getNextGenerationLengths() const
    {
        return mNextGenerationLengths;
    }

    SizeType32 getMaxNextGenerationLength() const
    {
        return mMaxNextGenLength;
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
    DraftTokensVec mLastDraftTokens;

    TokensVec mNextCompressedVector;
    std::vector<SizeType32> mNextPackedPosIds;
    DraftTokensIndices mNextDraftTokenIndices;

    TokensVec mLastCompressedVector;
    std::vector<SizeType32> mLastPackedPosIds;
    DraftTokensIndices mLastDraftTokenIndices;

    std::vector<SizeType32> mBestPathLengths;
    std::vector<SizeType32> mBestPathIndices;

    std::vector<SizeType32> mNextGenerationLengths;
    std::vector<SizeType32> mLastGenerationLengths;
    SizeType32 mMaxNextGenLength;

    std::vector<std::vector<std::vector<bool>>> mMasks;
};

template <typename T>
class ExplicitDraftTokensLayerTest : public testing::Test
{
private:
    void SetUp() override;

private:
    SamplingParams mSamplingParams;

    // Outputs
    TensorPtr mSeqLengths;
    TensorPtr mAcceptedLengths;
    TensorPtr mOutputIds;
    TensorPtr mOutputNextDraftTokens;
    TensorPtr mOutputPositionIdsBase;
    TensorPtr mRandomDataSample;
    TensorPtr mRandomDataValidation;
    TensorPtr mAcceptedLengthCumSum;
    TensorPtr mPackedMasks;
    TensorPtr mPathsOffsets;
    TensorPtr mNextPosIds;
    TensorPtr mNextDraftLengths;
    TensorPtr mPrevDraftLengths;
    TensorPtr mOutputUnpackedNextDraftTokens;
    TensorPtr mOutputUnpackedNextDraftIndices;
    TensorPtr mOutputDraftProbs;
    TensorPtr mOutputTemperatures;
    TensorPtr mOutputGenerationLengths;
    TensorPtr mOutputGenerationLengthsHost;
    TensorPtr mMaxGenLengthHost;

    // inputs
    TensorPtr mBatchSlots;
    TensorPtr mMasks;
    TensorPtr mInputNextDraftTokens;
    TensorPtr mNextDraftIndices;
    TensorPtr mLastDraftTokens;
    TensorPtr mLastDraftIndices;
    TensorPtr mNextDraftProbs;
    TensorPtr mPackedPosIds;
    TensorPtr mBestPathLengths;
    TensorPtr mBestPathIndices;
    TensorPtr mSpecDecodingGenerationLengths;
    TensorPtr mTokensPerStep;
    TensorPtr mNextFlatTokens;
    TensorPtr mInputPositionIdsBase;
    TensorPtr mEndIds;
    TensorPtr mMaxGenLengthDevice;

    // Packed inputs
    TensorPtr mMaxGenerationLength;
    TensorPtr mCumSumGenerationLengths;

    // Packed outputs
    TensorPtr mPackedPositionIdsBase;
    TensorPtr mPackedGenerationLengths;
    TensorPtr mPackedRandomDataSample;
    TensorPtr mPackedRandomDataVerification;
    TensorPtr mPackedNextDraftTokens;
    TensorPtr mPackedNextDraftIndices;
    TensorPtr mPackedPackedMasks;
    TensorPtr mPackedPositionOffsets;
    TensorPtr mPackedPackedPosIds;
    TensorPtr mPackedDraftProbs;
    TensorPtr mPackedTemperatures;

    // Setup params
    std::vector<uint64_t> mRandomSeeds;
    std::vector<float> mTemperatures;

    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::layers::ExplicitDraftTokensLayer<typename T::LayerType>> mExplicitDraftTokensLayer;
    std::shared_ptr<runtime::DecodingLayerWorkspace> mDecodingWorkspace;

    ExplicitDraftTokensDummyNetwork mNetwork;

private:
    void allocateBuffers();

    void setup();

    std::shared_ptr<tensorrt_llm::layers::ExplicitDraftTokensInputs> createInputTensors();

    std::shared_ptr<tensorrt_llm::layers::ExplicitDraftTokensOutputs> createOutputTensors();

    void checkLayerResult();

    void packData();

    void checkPackResult();

public:
    void runTest(std::vector<std::string> const& prompts, std::vector<std::string> const& predictions,
        DraftLettersVec const& nextDraftLetters, DraftLettersVec const& lastDraftLetters, SamplingParams& params);
};

template <typename T, typename U>
struct TypePair
{
    using LayerType = T;
    using DataType = U;
};

#ifdef ENABLE_BF16
using TestTypes = testing::Types<TypePair<float, float>, TypePair<half, half>, TypePair<half, __nv_bfloat16>>;
#else
using TestTypes = testing::Types<TypePair<float, float>, TypePair<half, half>>;
#endif // ENABLE_BF16

} // namespace tensorrt_llm::tests::layers
