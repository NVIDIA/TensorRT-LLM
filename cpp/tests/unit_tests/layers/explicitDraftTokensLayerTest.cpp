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

#include "tests/unit_tests/layers/explicitDraftTokensLayerTest.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/speculativeDecoding/explicitDraftTokensKernels.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/speculativeDecodingModule.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include <NvInferRuntimeBase.h>

#include <algorithm>
#include <cstdint>
#include <random>

namespace tensorrt_llm::tests::layers
{
// TODO verify context + gen mix

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::layers;
using namespace tensorrt_llm::common;

namespace tk = tensorrt_llm::kernels;
namespace tksd = tensorrt_llm::kernels::speculative_decoding;
namespace trk = tensorrt_llm::runtime::kernels;

TokensVec ExplicitDraftTokensDummyNetwork::tokenize(std::string const& letters) const
{
    TokensVec tokens;
    for (char c : letters)
    {
        tokens.push_back(static_cast<TokenIdType>(c));
    }
    return tokens;
}

std::string ExplicitDraftTokensDummyNetwork::detokenize(TokensVec const& tokens) const
{
    std::string letters;
    for (int token : tokens)
    {
        letters += static_cast<char>(token);
    }
    return letters;
}

DraftTokensVec ExplicitDraftTokensDummyNetwork::draftLettersToTokens(DraftLettersVec const& draftLetters) const
{
    DraftTokensVec draftTokens(draftLetters.size());
    for (SizeType32 bi = 0; bi < draftLetters.size(); ++bi)
    {
        draftTokens[bi].resize(draftLetters[bi].size());
        for (SizeType32 pi = 0; pi < draftLetters[bi].size(); ++pi)
        {
            draftTokens[bi][pi] = tokenize(draftLetters[bi][pi]);
        }
    }
    return draftTokens;
}

SizeType32 ExplicitDraftTokensDummyNetwork::longestCommonPrefixLength(TokensVec const& a, TokensVec const& b) const
{
    SizeType32 minLength = std::min(a.size(), b.size());
    SizeType32 idx = 0;
    while (idx < minLength && a[idx] == b[idx])
    {
        ++idx;
    }
    return idx;
}

SizeType32 ExplicitDraftTokensDummyNetwork::computeCompressedVectorAndIndices(TokensVec& compressedVector,
    std::vector<SizeType32>& packedPosIds, DraftTokensIndices& indices, std::vector<TokensVec> const& vectors,
    SizeType32 basePosId)
{
    TokensVec localCompressedVector;
    std::vector<SizeType32> localPackedPosIds;
    std::vector<std::vector<SizeType32>> localIndices;

    // FIXME always take the 1st beam as the reference. Is that correct?
    // Add whole first vector to compressed vector
    localCompressedVector = vectors[0];
    // All indices of first vector.
    localIndices.push_back(std::vector<SizeType32>(localCompressedVector.size()));
    for (SizeType32 ti = 0; ti < localCompressedVector.size(); ++ti)
    {
        localIndices[0][ti] = ti;
        // Set local to batch packed pos ids.
        localPackedPosIds.push_back(basePosId + ti);
    }

    // Starting from the 1st path.
    for (SizeType32 pi = 1; pi < vectors.size(); ++pi)
    {
        // Match path to compressed vector (aka path 0).
        auto const prefixLength = longestCommonPrefixLength(localCompressedVector, vectors[pi]);
        localIndices.push_back(std::vector<SizeType32>(vectors[pi].size()));
        // Set indices of the matched prefix.
        for (SizeType32 ti = 0; ti < prefixLength; ++ti)
        {
            localIndices[pi][ti] = ti;
        }
        // For non-matched part.
        for (SizeType32 ti = prefixLength; ti < vectors[pi].size(); ++ti)
        {
            // Add new tokens to compressed vector.
            localCompressedVector.push_back(vectors[pi][ti]);
            // Set new pos ids.
            localPackedPosIds.push_back(basePosId + ti);
            // Set their indices.
            localIndices[pi][ti] = localCompressedVector.size() - 1;
        }
    }

    compressedVector.insert(compressedVector.end(), localCompressedVector.begin(), localCompressedVector.end());
    packedPosIds.insert(packedPosIds.end(), localPackedPosIds.begin(), localPackedPosIds.end());
    indices.push_back(localIndices);
    return static_cast<SizeType32>(localCompressedVector.size());
}

void ExplicitDraftTokensDummyNetwork::createNextMasks(
    DraftTokensIndices const& indices, DraftTokensVec const& draftTokens, SizeType32 maxGenLength)
{
    for (SizeType32 bi = 0; bi < indices.size(); ++bi)
    {
        std::vector<std::vector<bool>> localMask(maxGenLength, std::vector<bool>(maxGenLength));
        // Create fill diagonal.
        for (SizeType32 ti = 0; ti < maxGenLength; ++ti)
        {
            localMask[ti][ti] = true;
        }

        SizeType32 rowIdx = 0;
        for (SizeType32 pi = 0; pi < draftTokens[bi].size(); ++pi)
        {
            auto const prefixLength = pi == 0 ? 0 : longestCommonPrefixLength(draftTokens[bi][0], draftTokens[bi][pi]);
            for (SizeType32 ti = 0; ti < draftTokens[bi][pi].size(); ++ti)
            {
                auto const index = indices[bi][pi][ti];
                // If we are in the "prefix" part of the sequence skip it as it does not represent real mask row.
                if (ti < prefixLength)
                {
                    continue;
                }
                // Fill lower triangular part according to the prefix.
                for (SizeType32 tti = 0; tti < ti; ++tti)
                {
                    localMask[rowIdx][indices[bi][pi][tti]] = true;
                }
                rowIdx++;
            }
        }
        mMasks.push_back(localMask);
    }
}

void ExplicitDraftTokensDummyNetwork::compressTokens(TokensVec& compressedVector, std::vector<SizeType32>& packedPosIds,
    DraftTokensIndices& indices, std::vector<SizeType32>& generationLengths, DraftTokensVec const& draftTokens,
    std::vector<SizeType32> const& basePosIds)
{
    generationLengths.resize(draftTokens.size());
    for (SizeType32 bi = 0; bi < draftTokens.size(); ++bi)
    {
        auto numGeneratedTokens = computeCompressedVectorAndIndices(
            compressedVector, packedPosIds, indices, draftTokens[bi], basePosIds[bi]);
        generationLengths[bi] = numGeneratedTokens;
    }
    // Pad vectors to the maximum size
    auto const padSize
        = mSamplingParams.getMaxDecodingTokens() * mSamplingParams.getBatchSize() - compressedVector.size();
    compressedVector.insert(compressedVector.end(), padSize, mSamplingParams.getPadId());
    packedPosIds.insert(packedPosIds.end(), padSize, 0);
}

void ExplicitDraftTokensDummyNetwork::acceptTokens(std::vector<TokensVec> const& predictionTokens,
    DraftTokensVec const& lastDraftTokens, DraftTokensVec const& nextDraftTokens)
{
    TLLM_CHECK_WITH_INFO(predictionTokens.size() == lastDraftTokens.size(),
        "Batch size of predictions (%d) does not match the batch size of last draft tokens (%d)",
        static_cast<SizeType32>(predictionTokens.size()), static_cast<SizeType32>(lastDraftTokens.size()));
    TLLM_CHECK_WITH_INFO(predictionTokens.size() == nextDraftTokens.size(),
        "Batch size of predictions (%d) does not match the batch size of next draft tokens (%d)",
        static_cast<SizeType32>(predictionTokens.size()), static_cast<SizeType32>(nextDraftTokens.size()));
    mBestPathLengths.resize(predictionTokens.size());
    mBestPathIndices.resize(predictionTokens.size());
    // Needed for unit test of ExplicitDraftTokensDummyNetwork only.
    if (mOutputIds.size() == 0)
    {
        mOutputIds.resize(lastDraftTokens.size());
    }
    for (SizeType32 bi = 0; bi < predictionTokens.size(); ++bi)
    {
        SizeType32 maxMatchLen = -1;
        SizeType32 maxMatchIdx = -1;
        // Find path with largest prefix shared with the predicted tokens.
        for (SizeType32 pi = 0; pi < lastDraftTokens[bi].size(); ++pi)
        {
            TLLM_CHECK_WITH_INFO(predictionTokens[bi][0] == lastDraftTokens[bi][pi][0],
                "First token of prediction and draft token must match");
            auto const matchLen = longestCommonPrefixLength(lastDraftTokens[bi][pi], predictionTokens[bi]);
            if (matchLen > maxMatchLen)
            {
                maxMatchLen = matchLen;
                maxMatchIdx = pi;
            }
        }
        mBestPathLengths[bi] = maxMatchLen;
        mBestPathIndices[bi] = maxMatchIdx;
        // Update output ids. First draft token is already counted in outputs
        mOutputIds[bi].insert(mOutputIds[bi].end(), lastDraftTokens[bi][maxMatchIdx].begin() + 1,
            lastDraftTokens[bi][maxMatchIdx].begin() + maxMatchLen);
        mOutputIds[bi].push_back(nextDraftTokens[bi][0][0]);
    }
}

void ExplicitDraftTokensDummyNetwork::forward(SamplingParams const& params,
    std::vector<std::string> const& promptsLetters, std::vector<std::string> const& predictionLetters,
    DraftLettersVec const& nextDraftLetters, DraftLettersVec const& lastDraftLetters)
{
    mSamplingParams = params;

    TLLM_CHECK(params.getBatchSize() == promptsLetters.size());
    TLLM_CHECK(params.getBatchSize() == predictionLetters.size());
    TLLM_CHECK(params.getBatchSize() == nextDraftLetters.size());
    TLLM_CHECK(params.getBatchSize() == lastDraftLetters.size());

    // Tokenize
    mNextDraftTokens = draftLettersToTokens(nextDraftLetters);
    mLastDraftTokens = draftLettersToTokens(lastDraftLetters);
    std::vector<TokensVec> predictionTokens;
    for (SizeType32 bi = 0; bi < predictionLetters.size(); ++bi)
    {
        predictionTokens.push_back(tokenize(predictionLetters[bi]));
        mPrompts.push_back(tokenize(promptsLetters[bi]));
    }

    std::vector<SizeType32> basePosIds;
    for (auto const& prompt : mPrompts)
    {
        basePosIds.push_back(prompt.size());
    }

    mOutputIds = mPrompts;

    // Make compressed tensors and pos ids for the current and next tokens
    compressTokens(mNextCompressedVector, mNextPackedPosIds, mNextDraftTokenIndices, mNextGenerationLengths,
        mNextDraftTokens, basePosIds);
    compressTokens(mLastCompressedVector, mLastPackedPosIds, mLastDraftTokenIndices, mLastGenerationLengths,
        mLastDraftTokens, basePosIds);

    mMaxNextGenLength = *std::max_element(mNextGenerationLengths.begin(), mNextGenerationLengths.end());

    acceptTokens(predictionTokens, mLastDraftTokens, mNextDraftTokens);

    createNextMasks(mNextDraftTokenIndices, mNextDraftTokens, mMaxNextGenLength);
}

TEST(ExplicitDraftTokensDummyNetworkTest, tokenizeTest)
{
    ExplicitDraftTokensDummyNetwork network;

    {
        auto tokens = network.tokenize("hello world");
        EXPECT_EQ(tokens, std::vector<TokenIdType>({104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100}));
    }
    {
        DraftLettersVec lettersVec = {{"hello world", "hello"}, {"world"}};
        auto draftTokens = network.draftLettersToTokens(lettersVec);
        ASSERT_EQ(draftTokens.size(), 2);
        ASSERT_EQ(draftTokens[0].size(), 2);
        ASSERT_EQ(draftTokens[1].size(), 1);
        EXPECT_EQ(draftTokens[0][0], std::vector<TokenIdType>({104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100}));
        EXPECT_EQ(draftTokens[0][1], std::vector<TokenIdType>({104, 101, 108, 108, 111}));
        EXPECT_EQ(draftTokens[1][0], std::vector<TokenIdType>({119, 111, 114, 108, 100}));
    }
}

TEST(ExplicitDraftTokensDummyNetworkTest, detokenizeTest)
{
    ExplicitDraftTokensDummyNetwork network;

    {
        auto letters
            = network.detokenize(std::vector<TokenIdType>({104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100}));
        EXPECT_EQ(letters, "hello world");
    }
}

TEST(ExplicitDraftTokensDummyNetworkTest, longestCommonPrefixLengthTest)
{
    ExplicitDraftTokensDummyNetwork network;
    EXPECT_EQ(network.longestCommonPrefixLength({1, 2, 3}, {1, 2}), 2);
    EXPECT_EQ(network.longestCommonPrefixLength({1, 2, 3}, {1, 2, 3}), 3);
    EXPECT_EQ(network.longestCommonPrefixLength({1, 2, 3}, {1, 5, 6}), 1);
    EXPECT_EQ(network.longestCommonPrefixLength({1, 2, 3}, {2, 5, 6}), 0);
    EXPECT_EQ(network.longestCommonPrefixLength({1, 2, 3}, {}), 0);
}

TEST(ExplicitDraftTokensDummyNetworkTest, computeCompressedVectorAndIndicesTest)
{
    ExplicitDraftTokensDummyNetwork network;

    {
        std::vector<TokenIdType> compressedVector;
        std::vector<SizeType32> packedPosIds;
        DraftTokensIndices indices;

        SizeType32 basePosId{0};

        std::vector<std::vector<TokenIdType>> tokens = {{0, 1, 2, 3}};

        auto const totalGen
            = network.computeCompressedVectorAndIndices(compressedVector, packedPosIds, indices, tokens, basePosId);

        EXPECT_EQ(totalGen, 4);
        EXPECT_EQ(compressedVector, std::vector<TokenIdType>({0, 1, 2, 3}));
        EXPECT_EQ(packedPosIds, std::vector<SizeType32>({0, 1, 2, 3}));
        ASSERT_EQ(indices.size(), 1);
        ASSERT_EQ(indices[0].size(), 1);
        EXPECT_EQ(indices[0][0], std::vector<SizeType32>({0, 1, 2, 3}));
    }

    {
        std::vector<TokenIdType> compressedVector;
        std::vector<SizeType32> packedPosIds;
        DraftTokensIndices indices;

        SizeType32 basePosId{0};

        std::vector<std::vector<TokenIdType>> tokens = {{0, 1, 2, 3}, {0, 2, 3, 4}};

        auto const totalGen
            = network.computeCompressedVectorAndIndices(compressedVector, packedPosIds, indices, tokens, basePosId);

        EXPECT_EQ(totalGen, 7);
        EXPECT_EQ(compressedVector, std::vector<TokenIdType>({0, 1, 2, 3, 2, 3, 4}));
        EXPECT_EQ(packedPosIds, std::vector<SizeType32>({0, 1, 2, 3, 1, 2, 3}));
        ASSERT_EQ(indices.size(), 1);
        ASSERT_EQ(indices[0].size(), 2);
        EXPECT_EQ(indices[0][0], std::vector<SizeType32>({0, 1, 2, 3}));
        EXPECT_EQ(indices[0][1], std::vector<SizeType32>({0, 4, 5, 6}));
    }

    {
        std::vector<TokenIdType> compressedVector;
        std::vector<SizeType32> packedPosIds;
        DraftTokensIndices indices;

        SizeType32 basePosId{0};

        std::vector<std::vector<TokenIdType>> tokens = {{0, 1, 2, 3}, {0, 1, 6, 2}, {0, 5, 6, 2}};

        auto const totalGen
            = network.computeCompressedVectorAndIndices(compressedVector, packedPosIds, indices, tokens, basePosId);

        EXPECT_EQ(totalGen, 9);
        EXPECT_EQ(compressedVector, std::vector<TokenIdType>({0, 1, 2, 3, 6, 2, 5, 6, 2}));
        EXPECT_EQ(packedPosIds, std::vector<SizeType32>({0, 1, 2, 3, 2, 3, 1, 2, 3}));
        ASSERT_EQ(indices.size(), 1);
        ASSERT_EQ(indices[0].size(), 3);
        EXPECT_EQ(indices[0][0], std::vector<SizeType32>({0, 1, 2, 3}));
        EXPECT_EQ(indices[0][1], std::vector<SizeType32>({0, 1, 4, 5}));
        EXPECT_EQ(indices[0][2], std::vector<SizeType32>({0, 6, 7, 8}));
    }

    {
        std::vector<TokenIdType> compressedVector;
        std::vector<SizeType32> packedPosIds;
        DraftTokensIndices indices;

        SizeType32 basePosId{10};

        std::vector<std::vector<TokenIdType>> tokens = {{0, 1, 2, 3}, {0, 1, 6, 2}, {0, 5, 6, 2}};

        auto const totalGen
            = network.computeCompressedVectorAndIndices(compressedVector, packedPosIds, indices, tokens, basePosId);

        EXPECT_EQ(totalGen, 9);
        EXPECT_EQ(compressedVector, std::vector<TokenIdType>({0, 1, 2, 3, 6, 2, 5, 6, 2}));
        EXPECT_EQ(packedPosIds, std::vector<SizeType32>({10, 11, 12, 13, 12, 13, 11, 12, 13}));
        ASSERT_EQ(indices.size(), 1);
        ASSERT_EQ(indices[0].size(), 3);
        EXPECT_EQ(indices[0][0], std::vector<SizeType32>({0, 1, 2, 3}));
        EXPECT_EQ(indices[0][1], std::vector<SizeType32>({0, 1, 4, 5}));
        EXPECT_EQ(indices[0][2], std::vector<SizeType32>({0, 6, 7, 8}));
    }
}

TEST(ExplicitDraftTokensDummyNetworkTest, compressTokensTest)
{
    {
        ExplicitDraftTokensDummyNetwork network;
        std::vector<TokenIdType> compressedVector;
        std::vector<SizeType32> packedPosIds;
        DraftTokensIndices indices;
        std::vector<SizeType32> genLengths;

        SamplingParams params;
        params.setBatchSize(1);
        params.setMaxNumPaths(1);
        params.setMaxDraftPathLen(6);
        network.setSamplingParams(params);

        DraftTokensVec tokens = {{{0, 1, 2, 3}}};

        std::vector<SizeType32> basePosIds = {0};

        network.compressTokens(compressedVector, packedPosIds, indices, genLengths, tokens, basePosIds);

        EXPECT_EQ(compressedVector, std::vector<TokenIdType>({0, 1, 2, 3, -1, -1, -1}));
        EXPECT_EQ(packedPosIds, std::vector<SizeType32>({0, 1, 2, 3, 0, 0, 0}));
        ASSERT_EQ(indices.size(), 1);
        ASSERT_EQ(indices[0].size(), 1);
        EXPECT_EQ(indices[0][0], std::vector<SizeType32>({0, 1, 2, 3}));
        ASSERT_EQ(genLengths.size(), 1);
        EXPECT_EQ(genLengths[0], 4);

        network.createNextMasks(indices, tokens, 4);
        auto masks = network.getNextMasks();
        ASSERT_EQ(masks.size(), 1);
        ASSERT_EQ(masks[0].size(), 4);
        ASSERT_EQ(masks[0][0].size(), 4);

        EXPECT_EQ(masks[0][0], std::vector<bool>({true, false, false, false}));
        EXPECT_EQ(masks[0][1], std::vector<bool>({true, true, false, false}));
        EXPECT_EQ(masks[0][2], std::vector<bool>({true, true, true, false}));
        EXPECT_EQ(masks[0][3], std::vector<bool>({true, true, true, true}));
    }

    {
        ExplicitDraftTokensDummyNetwork network;
        std::vector<TokenIdType> compressedVector;
        std::vector<SizeType32> packedPosIds;
        DraftTokensIndices indices;
        std::vector<SizeType32> genLengths;

        SamplingParams params;
        params.setBatchSize(2);
        params.setMaxNumPaths(1);
        params.setMaxDraftPathLen(6);
        network.setSamplingParams(params);

        std::vector<SizeType32> basePosIds = {10, 10};

        DraftTokensVec tokens = {{{0, 1, 2, 3}}, {{0, 1, 2, 3}}};

        network.compressTokens(compressedVector, packedPosIds, indices, genLengths, tokens, basePosIds);

        EXPECT_EQ(compressedVector, std::vector<TokenIdType>({0, 1, 2, 3, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1}));
        EXPECT_EQ(packedPosIds, std::vector<SizeType32>({10, 11, 12, 13, 10, 11, 12, 13, 0, 0, 0, 0, 0, 0}));
        ASSERT_EQ(indices.size(), 2);
        ASSERT_EQ(indices[0].size(), 1);
        ASSERT_EQ(indices[1].size(), 1);
        EXPECT_EQ(indices[0][0], std::vector<SizeType32>({0, 1, 2, 3}));
        EXPECT_EQ(indices[1][0], std::vector<SizeType32>({0, 1, 2, 3}));
        ASSERT_EQ(genLengths.size(), 2);
        EXPECT_EQ(genLengths[0], 4);
        EXPECT_EQ(genLengths[1], 4);

        network.createNextMasks(indices, tokens, 4);
        auto masks = network.getNextMasks();
        ASSERT_EQ(masks.size(), 2);
        ASSERT_EQ(masks[0].size(), 4);
        ASSERT_EQ(masks[1].size(), 4);
        ASSERT_EQ(masks[0][0].size(), 4);
        ASSERT_EQ(masks[1][0].size(), 4);

        EXPECT_EQ(masks[0][0], std::vector<bool>({true, false, false, false}));
        EXPECT_EQ(masks[0][1], std::vector<bool>({true, true, false, false}));
        EXPECT_EQ(masks[0][2], std::vector<bool>({true, true, true, false}));
        EXPECT_EQ(masks[0][3], std::vector<bool>({true, true, true, true}));

        EXPECT_EQ(masks[1][0], std::vector<bool>({true, false, false, false}));
        EXPECT_EQ(masks[1][1], std::vector<bool>({true, true, false, false}));
        EXPECT_EQ(masks[1][2], std::vector<bool>({true, true, true, false}));
        EXPECT_EQ(masks[1][3], std::vector<bool>({true, true, true, true}));
    }
    {
        ExplicitDraftTokensDummyNetwork network;
        std::vector<TokenIdType> compressedVector;
        std::vector<SizeType32> packedPosIds;
        DraftTokensIndices indices;
        std::vector<SizeType32> genLengths;

        SamplingParams params;
        params.setBatchSize(2);
        params.setMaxNumPaths(3);
        params.setMaxDraftPathLen(4);
        network.setSamplingParams(params);

        std::vector<SizeType32> basePosIds = {10, 0};

        DraftTokensVec tokens = {{{0, 1, 2, 3}, {0, 1, 6, 2}, {0, 5, 6, 2}}, {{0, 1, 2, 3}, {0, 1, 2, 4}}};

        network.compressTokens(compressedVector, packedPosIds, indices, genLengths, tokens, basePosIds);

        EXPECT_EQ(compressedVector,
            std::vector<TokenIdType>(
                {0, 1, 2, 3, 6, 2, 5, 6, 2, 0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}));
        EXPECT_EQ(packedPosIds,
            std::vector<SizeType32>(
                {10, 11, 12, 13, 12, 13, 11, 12, 13, 0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        ASSERT_EQ(indices.size(), 2);
        ASSERT_EQ(indices[0].size(), 3);
        ASSERT_EQ(indices[1].size(), 2);
        EXPECT_EQ(indices[0][0], std::vector<SizeType32>({0, 1, 2, 3}));
        EXPECT_EQ(indices[0][1], std::vector<SizeType32>({0, 1, 4, 5}));
        EXPECT_EQ(indices[0][2], std::vector<SizeType32>({0, 6, 7, 8}));
        EXPECT_EQ(indices[1][0], std::vector<SizeType32>({0, 1, 2, 3}));
        EXPECT_EQ(indices[1][1], std::vector<SizeType32>({0, 1, 2, 4}));
        ASSERT_EQ(genLengths.size(), 2);
        EXPECT_EQ(genLengths[0], 9);
        EXPECT_EQ(genLengths[1], 5);

        network.createNextMasks(indices, tokens, 9);
        auto masks = network.getNextMasks();
        ASSERT_EQ(masks.size(), 2);
        ASSERT_EQ(masks[0].size(), 9);
        ASSERT_EQ(masks[1].size(), 9);
        ASSERT_EQ(masks[0][0].size(), 9);
        ASSERT_EQ(masks[1][0].size(), 9);

        EXPECT_EQ(masks[0][0], std::vector<bool>({true, false, false, false, false, false, false, false, false}));
        EXPECT_EQ(masks[0][1], std::vector<bool>({true, true, false, false, false, false, false, false, false}));
        EXPECT_EQ(masks[0][2], std::vector<bool>({true, true, true, false, false, false, false, false, false}));
        EXPECT_EQ(masks[0][3], std::vector<bool>({true, true, true, true, false, false, false, false, false}));
        EXPECT_EQ(masks[0][4], std::vector<bool>({true, true, false, false, true, false, false, false, false}));
        EXPECT_EQ(masks[0][5], std::vector<bool>({true, true, false, false, true, true, false, false, false}));
        EXPECT_EQ(masks[0][6], std::vector<bool>({true, false, false, false, false, false, true, false, false}));
        EXPECT_EQ(masks[0][7], std::vector<bool>({true, false, false, false, false, false, true, true, false}));
        EXPECT_EQ(masks[0][8], std::vector<bool>({true, false, false, false, false, false, true, true, true}));

        EXPECT_EQ(masks[1][0], std::vector<bool>({true, false, false, false, false, false, false, false, false}));
        EXPECT_EQ(masks[1][1], std::vector<bool>({true, true, false, false, false, false, false, false, false}));
        EXPECT_EQ(masks[1][2], std::vector<bool>({true, true, true, false, false, false, false, false, false}));
        EXPECT_EQ(masks[1][3], std::vector<bool>({true, true, true, true, false, false, false, false, false}));
        EXPECT_EQ(masks[1][4], std::vector<bool>({true, true, true, false, true, false, false, false, false}));
        EXPECT_EQ(masks[1][5], std::vector<bool>({false, false, false, false, false, true, false, false, false}));
        EXPECT_EQ(masks[1][6], std::vector<bool>({false, false, false, false, false, false, true, false, false}));
        EXPECT_EQ(masks[1][7], std::vector<bool>({false, false, false, false, false, false, false, true, false}));
        EXPECT_EQ(masks[1][8], std::vector<bool>({false, false, false, false, false, false, false, false, true}));
    }
}

TEST(ExplicitDraftTokensDummyNetworkTest, acceptTokensTest)
{
    {
        ExplicitDraftTokensDummyNetwork network;
        std::vector<TokensVec> predictionTokens = {network.tokenize("how things")};
        DraftLettersVec lastDraftLetters = {{"how do ", "how are", "however", "hello w"}};
        DraftLettersVec nextDraftLetters = {{"things ", "that is", "to crea", "touchab"}};
        auto lastDraftTokens = network.draftLettersToTokens(lastDraftLetters);
        auto nextDraftTokens = network.draftLettersToTokens(nextDraftLetters);

        network.acceptTokens(predictionTokens, lastDraftTokens, nextDraftTokens);

        auto bestPathLengths = network.getBestPathLengths();
        auto bestPathIndices = network.getBestPathIndices();
        auto outputIds = network.getOutputIds();

        ASSERT_EQ(bestPathLengths.size(), 1);
        ASSERT_EQ(bestPathIndices.size(), 1);
        ASSERT_EQ(outputIds.size(), 1);
        EXPECT_EQ(bestPathLengths[0], 4);
        EXPECT_EQ(bestPathIndices[0], 0);
        EXPECT_EQ(network.detokenize(outputIds[0]), "ow t");
    }

    {
        ExplicitDraftTokensDummyNetwork network;
        std::vector<TokensVec> predictionTokens = {network.tokenize("however you")};
        DraftLettersVec lastDraftLetters = {{"how do ", "how tho", "however", "hello w"}};
        DraftLettersVec nextDraftLetters = {{" increme", " introdu", " i = 0; ", " importa"}};
        auto lastDraftTokens = network.draftLettersToTokens(lastDraftLetters);
        auto nextDraftTokens = network.draftLettersToTokens(nextDraftLetters);

        network.acceptTokens(predictionTokens, lastDraftTokens, nextDraftTokens);

        auto bestPathLengths = network.getBestPathLengths();
        auto bestPathIndices = network.getBestPathIndices();
        auto outputIds = network.getOutputIds();

        ASSERT_EQ(bestPathLengths.size(), 1);
        ASSERT_EQ(bestPathIndices.size(), 1);
        ASSERT_EQ(outputIds.size(), 1);
        EXPECT_EQ(bestPathLengths[0], 7);
        EXPECT_EQ(bestPathIndices[0], 2);
        EXPECT_EQ(network.detokenize(outputIds[0]), "owever ");
    }

    {
        ExplicitDraftTokensDummyNetwork network;
        std::vector<TokensVec> predictionTokens = {network.tokenize("how things")};
        DraftLettersVec lastDraftLetters = {{"heruist", "habit i", "handove", "hammer "}};
        DraftLettersVec nextDraftLetters = {{"oatmeal", "ocean b", "occupat", "oblivio"}};
        auto lastDraftTokens = network.draftLettersToTokens(lastDraftLetters);
        auto nextDraftTokens = network.draftLettersToTokens(nextDraftLetters);

        network.acceptTokens(predictionTokens, lastDraftTokens, nextDraftTokens);

        auto bestPathLengths = network.getBestPathLengths();
        auto bestPathIndices = network.getBestPathIndices();
        auto outputIds = network.getOutputIds();

        ASSERT_EQ(bestPathLengths.size(), 1);
        ASSERT_EQ(bestPathIndices.size(), 1);
        ASSERT_EQ(outputIds.size(), 1);
        EXPECT_EQ(bestPathLengths[0], 1);
        EXPECT_EQ(bestPathIndices[0], 0);
        EXPECT_EQ(network.detokenize(outputIds[0]), "o");
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void ExplicitDraftTokensLayerTest<T>::SetUp()
{
    mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
    mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);
}

template <typename T>
void ExplicitDraftTokensLayerTest<T>::allocateBuffers()
{
    using DataType = typename T::DataType;
    auto const dataType = TRTDataType<DataType>::value;

    auto speculativeDecodingModule = std::make_shared<SpeculativeDecodingModule>(mSamplingParams.getMaxDraftPathLen(),
        mSamplingParams.getMaxDecodingDraftTokens(), mSamplingParams.getMaxNumPaths());
    auto const decodingDomain = tensorrt_llm::layers::DecoderDomain(mSamplingParams.getMaxBatchSize(), 1,
        mSamplingParams.getVocabSize(), mSamplingParams.getVocabSize(), speculativeDecodingModule);

    mExplicitDraftTokensLayer = std::make_shared<tensorrt_llm::layers::ExplicitDraftTokensLayer<typename T::LayerType>>(
        decodingDomain, mBufferManager);

    // outputs
    mOutputIds = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxSeqLen()}),
        nvinfer1::DataType::kINT32);

    mSeqLengths = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mAcceptedLengths = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mNextDraftLengths = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mPrevDraftLengths = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mAcceptedLengthCumSum = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize() + 1}), nvinfer1::DataType::kINT32);

    mOutputNextDraftTokens = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxDecodingDraftTokens()}),
        nvinfer1::DataType::kINT32);

    mOutputPositionIdsBase = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mRandomDataSample = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), dataType);

    mRandomDataValidation
        = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getMaxBatchSize(),
                                        mSamplingParams.getMaxNumPaths(), mSamplingParams.getMaxDraftPathLen()}),
            dataType);

    mPackedMasks = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxDecodingTokens(),
            static_cast<SizeType32>(divUp(mSamplingParams.getMaxDecodingTokens(), 32))}),
        nvinfer1::DataType::kINT32);

    mNextPosIds = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxDecodingTokens()}),
        nvinfer1::DataType::kINT32);

    mOutputUnpackedNextDraftTokens = BufferManager::pinnedPool(
        ITensor::makeShape(
            {mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxNumPaths(), mSamplingParams.getMaxPathLen()}),
        nvinfer1::DataType::kINT32);

    mOutputUnpackedNextDraftIndices = BufferManager::pinnedPool(
        ITensor::makeShape(
            {mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxNumPaths(), mSamplingParams.getMaxPathLen()}),
        nvinfer1::DataType::kINT32);

    mOutputDraftProbs = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxNumPaths(),
            mSamplingParams.getMaxDraftPathLen(), mSamplingParams.getVocabSize()}),
        dataType);

    mOutputTemperatures = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), dataType);

    mOutputGenerationLengths = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mOutputGenerationLengthsHost = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mMaxGenLengthHost = BufferManager::pinnedPool(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);

    // inputs
    mBatchSlots
        = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getBatchSize()}), nvinfer1::DataType::kINT32);

    mTokensPerStep = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mPathsOffsets = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize() * mSamplingParams.getMaxDraftPathLen()}),
        nvinfer1::DataType::kINT32);

    mMasks = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxDecodingTokens(),
            mSamplingParams.getMaxDecodingTokens()}),
        nvinfer1::DataType::kBOOL);

    mInputNextDraftTokens = BufferManager::pinnedPool(
        ITensor::makeShape(
            {mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxNumPaths(), mSamplingParams.getMaxPathLen()}),
        nvinfer1::DataType::kINT32);

    mLastDraftTokens = BufferManager::pinnedPool(
        ITensor::makeShape(
            {mSamplingParams.getBatchSize(), mSamplingParams.getMaxNumPaths(), mSamplingParams.getMaxPathLen()}),
        nvinfer1::DataType::kINT32);

    mPackedPosIds = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxDecodingTokens()}),
        nvinfer1::DataType::kINT32);

    mBestPathLengths = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mBestPathIndices = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mSpecDecodingGenerationLengths = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mNextFlatTokens = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize() * mSamplingParams.getMaxDecodingTokens()}),
        nvinfer1::DataType::kINT32);

    mInputPositionIdsBase = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mNextDraftIndices = BufferManager::pinnedPool(
        ITensor::makeShape(
            {mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxNumPaths(), mSamplingParams.getMaxPathLen()}),
        nvinfer1::DataType::kINT32);

    mLastDraftIndices = BufferManager::pinnedPool(
        ITensor::makeShape(
            {mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxNumPaths(), mSamplingParams.getMaxPathLen()}),
        nvinfer1::DataType::kINT32);

    mNextDraftProbs = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxNumPaths(),
            mSamplingParams.getMaxDraftPathLen(), mSamplingParams.getVocabSize()}),
        dataType);

    mEndIds = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mMaxGenLengthDevice = BufferManager::pinnedPool(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);

    // Packed inputs
    mMaxGenerationLength = BufferManager::pinnedPool(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    mCumSumGenerationLengths
        = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getBatchSize()}), nvinfer1::DataType::kINT32);

    // Packed outputs
    mPackedPositionIdsBase
        = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getBatchSize()}), nvinfer1::DataType::kINT32);
    mPackedGenerationLengths
        = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getBatchSize()}), nvinfer1::DataType::kINT32);
    mPackedRandomDataSample = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getBatchSize()}), dataType);
    mPackedRandomDataVerification = BufferManager::pinnedPool(
        ITensor::makeShape(
            {mSamplingParams.getBatchSize(), mSamplingParams.getMaxNumPaths(), mSamplingParams.getMaxDraftPathLen()}),
        dataType);
    mPackedNextDraftTokens = BufferManager::pinnedPool(
        ITensor::makeShape(
            {mSamplingParams.getBatchSize(), mSamplingParams.getMaxNumPaths(), mSamplingParams.getMaxPathLen()}),
        nvinfer1::DataType::kINT32);
    mPackedNextDraftIndices = BufferManager::pinnedPool(
        ITensor::makeShape(
            {mSamplingParams.getBatchSize(), mSamplingParams.getMaxNumPaths(), mSamplingParams.getMaxPathLen()}),
        nvinfer1::DataType::kINT32);
    mPackedPackedMasks = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getBatchSize(), mSamplingParams.getMaxDecodingTokens(),
            static_cast<SizeType32>(divUp(mSamplingParams.getMaxDecodingTokens(), 32))}),
        nvinfer1::DataType::kINT32);
    mPackedPositionOffsets = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getBatchSize(), mSamplingParams.getMaxDecodingTokens()}),
        nvinfer1::DataType::kINT32);
    mPackedPackedPosIds = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getBatchSize(), mSamplingParams.getMaxDecodingTokens()}),
        nvinfer1::DataType::kINT32);
    mPackedDraftProbs = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getBatchSize(), mSamplingParams.getMaxNumPaths(),
            mSamplingParams.getMaxDraftPathLen(), mSamplingParams.getVocabSize()}),
        dataType);
    mPackedTemperatures = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getBatchSize()}), dataType);
    mDecodingWorkspace = std::make_shared<tensorrt_llm::runtime::DecodingLayerWorkspace>(mBufferManager, decodingDomain,
        TRTDataType<typename T::DataType>::value, mExplicitDraftTokensLayer->getWorkspaceSize());
}

template <typename T>
void ExplicitDraftTokensLayerTest<T>::setup()
{
    using DataType = typename T::DataType;
    // outputs
    trk::invokeFill(*mOutputIds, TokenIdType{-1}, *mStream);
    trk::invokeFill(*mSeqLengths, SizeType32{0}, *mStream);
    trk::invokeFill(*mAcceptedLengths, SizeType32{0}, *mStream);
    trk::invokeFill(*mAcceptedLengthCumSum, SizeType32{-1}, *mStream);
    trk::invokeFill(*mOutputNextDraftTokens, TokenIdType{-1}, *mStream);
    trk::invokeFill(*mOutputPositionIdsBase, SizeType32{0}, *mStream);
    trk::invokeFill(*mRandomDataSample, DataType{0}, *mStream);
    trk::invokeFill(*mRandomDataValidation, DataType{0}, *mStream);
    trk::invokeFill(*mPackedMasks, SizeType32{0}, *mStream);
    trk::invokeFill(*mNextPosIds, SizeType32{0}, *mStream);
    trk::invokeFill(*mOutputUnpackedNextDraftTokens, TokenIdType{-1}, *mStream);
    trk::invokeFill(*mOutputUnpackedNextDraftIndices, SizeType32{0}, *mStream);
    trk::invokeFill(*mEndIds, TokenIdType{-1}, *mStream);

    auto inDraftProbs = BufferRange<DataType>(*mNextDraftProbs);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> distr(0.0, 1.0);
    std::generate(
        inDraftProbs.begin(), inDraftProbs.end(), [&gen, &distr]() { return static_cast<DataType>(distr(gen)); });

    auto batchSlotsPtr = bufferCast<SizeType32>(*mBatchSlots);
    for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
    {
        batchSlotsPtr[bi] = 2 * bi;
    }

    auto setupParams = std::make_shared<ExplicitDraftTokensSetupParams>();
    mRandomSeeds = std::vector<uint64_t>(mSamplingParams.getBatchSize());
    mTemperatures = std::vector<float>(mSamplingParams.getBatchSize());

    std::mt19937 generator(42);
    std::uniform_int_distribution<uint64_t> seedDistr(1, 1000);
    std::uniform_real_distribution<float> temperatureDistr(0.001f, 1.f);
    std::generate(
        mRandomSeeds.begin(), mRandomSeeds.end(), [&generator, &seedDistr]() { return seedDistr(generator); });
    std::generate(mTemperatures.begin(), mTemperatures.end(),
        [&generator, &temperatureDistr]() { return temperatureDistr(generator); });
    setupParams->randomSeed = mRandomSeeds;
    setupParams->temperature = mTemperatures;
    setupParams->randomDataSample = mRandomDataSample;
    setupParams->temperatures = mOutputTemperatures;
    setupParams->dtype = TRTDataType<DataType>::value;

    mDecodingWorkspace->setDeviceBatchSlots(mBatchSlots);
    mExplicitDraftTokensLayer->setup(mSamplingParams.getBatchSize(), 1, mBatchSlots, setupParams, mDecodingWorkspace);

    mStream->synchronize();

    mBestPathLengths = mBufferManager->copyFrom(mNetwork.getBestPathLengths(),
        ITensor::makeShape({mSamplingParams.getBatchSize()}), runtime::MemoryType::kPINNEDPOOL);
    mBestPathIndices = mBufferManager->copyFrom(mNetwork.getBestPathIndices(),
        ITensor::makeShape({mSamplingParams.getBatchSize()}), runtime::MemoryType::kPINNEDPOOL);
    mPackedPosIds = mBufferManager->copyFrom(mNetwork.getNextPackedPosId(),
        ITensor::makeShape({mSamplingParams.getMaxDecodingTokens() * mSamplingParams.getBatchSize()}),
        runtime::MemoryType::kPINNEDPOOL);

    auto const nextDraftTokens = mNetwork.getNextDraftTokens();
    auto const lastDraftTokens = mNetwork.getLastDraftTokens();
    auto const nextDraftIndices = mNetwork.getNextDraftIndices();
    auto const lastDraftIndices = mNetwork.getLastDraftIndices();
    auto sequenceLength = BufferRange<SizeType32>(*mSeqLengths);
    auto nextDraftTokensRange = BufferRange<TokenIdType>(*mInputNextDraftTokens);
    auto lastDraftTokensRange = BufferRange<TokenIdType>(*mLastDraftTokens);
    auto nextDraftIndicesRange = BufferRange<SizeType32>(*mNextDraftIndices);
    auto lastDraftIndicesRange = BufferRange<SizeType32>(*mLastDraftIndices);
    auto inputPositionIdsBase = BufferRange<SizeType32>(*mInputPositionIdsBase);

    auto outputIds = BufferRange<TokenIdType>(*mOutputIds);
    auto generationLengths = mNetwork.getNextGenerationLengths();
    auto prompts = mNetwork.getPrompts();
    for (SizeType32 bi = 0; bi < nextDraftTokens.size(); ++bi)
    {
        for (SizeType32 pi = 0; pi < nextDraftTokens[bi].size(); ++pi)
        {
            for (SizeType32 ti = 0; ti < nextDraftTokens[bi][pi].size(); ++ti)
            {
                auto idx = flat_index3(bi, pi, ti, mSamplingParams.getMaxNumPaths(), mSamplingParams.getMaxPathLen());
                nextDraftTokensRange[idx] = nextDraftTokens[bi][pi][ti];
                lastDraftTokensRange[idx] = lastDraftTokens[bi][pi][ti];
                nextDraftIndicesRange[idx] = nextDraftIndices[bi][pi][ti];
                lastDraftIndicesRange[idx] = lastDraftIndices[bi][pi][ti];
            }
        }
        bufferCast<SizeType32>(*mSpecDecodingGenerationLengths)[bi] = generationLengths[bi];

        sequenceLength[batchSlotsPtr[bi]] = prompts[bi].size();
        std::copy(prompts[bi].begin(), prompts[bi].end(),
            outputIds.begin() + batchSlotsPtr[bi] * mSamplingParams.getMaxSeqLen());

        inputPositionIdsBase[bi] = prompts[bi].size();
    }

    auto nextFlatTokens = mNetwork.getNextFlatTokens();
    TLLM_LOG_DEBUG("Next flat tokens are \"%s\"", mNetwork.detokenize(nextFlatTokens).c_str());
    auto nextFlatTokensRange = BufferRange<TokenIdType>(*mNextFlatTokens);
    std::copy(nextFlatTokens.begin(), nextFlatTokens.end(), nextFlatTokensRange.begin());

    auto const masks = mNetwork.getNextMasks();
    auto masksRange = BufferRange<bool>(*mMasks);
    auto const maxGenLength = mNetwork.getMaxNextGenerationLength();
    bufferCast<SizeType32>(*mMaxGenerationLength)[0] = maxGenLength;
    for (SizeType32 bi = 0; bi < masks.size(); ++bi)
    {
        TLLM_CHECK(maxGenLength == masks[bi].size());
        for (SizeType32 ri = 0; ri < masks[bi].size(); ++ri)
        {
            TLLM_CHECK(maxGenLength == masks[bi][ri].size());
            for (SizeType32 ci = 0; ci < masks[bi][ri].size(); ++ci)
            {
                masksRange[bi * maxGenLength * maxGenLength + ri * maxGenLength + ci] = masks[bi][ri][ci];
            }
        }
    }
}

template <typename T>
std::shared_ptr<ExplicitDraftTokensInputs> ExplicitDraftTokensLayerTest<T>::createInputTensors()
{
    auto forwardParams
        = std::make_shared<ExplicitDraftTokensInputs>(mEndIds, mBatchSlots, mSamplingParams.getBatchSize());

    forwardParams->seqSlots = mBatchSlots;

    forwardParams->masks = mMasks;

    forwardParams->nextDraftTokens = mInputNextDraftTokens;

    forwardParams->nextDraftIndices = mNextDraftIndices;

    forwardParams->lastDraftTokens = mLastDraftTokens;

    forwardParams->lastDraftIndices = mLastDraftIndices;

    forwardParams->packedPosIds = mPackedPosIds;

    forwardParams->bestPathLengths = mBestPathLengths;

    forwardParams->bestPathIndices = mBestPathIndices;

    forwardParams->generationLengths = mSpecDecodingGenerationLengths;

    forwardParams->nextFlatTokens = mNextFlatTokens;

    forwardParams->positionIdsBase = mInputPositionIdsBase;

    forwardParams->nextDraftProbs = mNextDraftProbs;

    forwardParams->maxGenLengthDevice = mMaxGenLengthDevice;

    return forwardParams;
}

template <typename T>
std::shared_ptr<ExplicitDraftTokensOutputs> ExplicitDraftTokensLayerTest<T>::createOutputTensors()
{
    auto outputParams = std::make_shared<ExplicitDraftTokensOutputs>(mOutputIds);

    outputParams->sequenceLength = mSeqLengths;

    outputParams->nextDraftTokens = mOutputNextDraftTokens;

    outputParams->numNewTokens = mAcceptedLengths;

    outputParams->nextDraftLengths = mNextDraftLengths;

    outputParams->prevDraftLengths = mPrevDraftLengths;

    outputParams->numNewTokensCumSum = mAcceptedLengthCumSum;

    outputParams->pathsOffsets = mPathsOffsets;

    outputParams->nextDraftPosIds = mNextPosIds;

    outputParams->positionIdsBase = mOutputPositionIdsBase;

    outputParams->randomDataSample = mRandomDataSample;

    outputParams->randomDataValidation = mRandomDataValidation;

    outputParams->packedMasks = mPackedMasks;

    outputParams->packedMasks = mPackedMasks;

    outputParams->unpackedNextDraftTokens = mOutputUnpackedNextDraftTokens;

    outputParams->unpackedNextDraftIndices = mOutputUnpackedNextDraftIndices;

    outputParams->nextDraftProbs = mOutputDraftProbs;

    outputParams->temperatures = mOutputTemperatures;

    outputParams->generationLengths = mOutputGenerationLengths;

    outputParams->generationLengthsHost = mOutputGenerationLengthsHost;

    outputParams->maxGenLengthHost = mMaxGenLengthHost;

    return outputParams;
}

std::vector<int32_t> boolArrayToBitmask(BufferRange<bool>::iterator boolIterator, size_t pathLen)
{
    std::vector<int32_t> bitmask(divUp(pathLen, 32));
    for (size_t bi = 0; bi < pathLen; ++bi)
    {
        auto slice = bi / 32;
        if (boolIterator[bi])
        {
            bitmask[slice] |= (1 << (bi % 32));
        }
    }
    return bitmask;
}

template <typename T>
void ExplicitDraftTokensLayerTest<T>::checkLayerResult()
{
    using DataType = typename T::DataType;
    auto const batchSlots = BufferRange<SizeType32>(*mBatchSlots);

    // Check generated random data
    {
        auto const randomDataSample = BufferRange<DataType>(*mRandomDataSample);
        auto const randomDataValidation = BufferRange<DataType>(*mRandomDataValidation);
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            auto const batchSlot = batchSlots[bi];
            // Check that all fields are filled with non zero data
            EXPECT_NE(randomDataSample[batchSlot], DataType{0}) << " bi: " << bi;
            auto const stride = mSamplingParams.getMaxNumPaths() * mSamplingParams.getMaxDraftPathLen();
            EXPECT_FALSE(std::any_of(randomDataValidation.begin() + batchSlot * stride,
                randomDataValidation.begin() + (batchSlot + 1) * stride,
                [](DataType val) { return val == DataType{0}; }))
                << " bi: " << bi;
        }
    }

    // Check masks
    {
        auto const packedMasks = BufferRange<int32_t>(*mPackedMasks);
        auto masks = BufferRange<bool>(*mMasks);
        auto generationLengths = mNetwork.getNextGenerationLengths();
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            for (SizeType32 ti = 0; ti < generationLengths[bi]; ++ti)
            {
                auto const batchSlot = batchSlots[bi];
                auto const maskIdx = flat_index3(
                    bi, ti, 0, mNetwork.getMaxNextGenerationLength(), mNetwork.getMaxNextGenerationLength());
                auto const bitmask = boolArrayToBitmask(masks.begin() + maskIdx, mNetwork.getMaxNextGenerationLength());
                for (SizeType32 mi = 0; mi < bitmask.size(); ++mi)
                {
                    auto const packedMaskIdx = flat_index3(batchSlot, ti, mi, mSamplingParams.getMaxDecodingTokens(),
                        static_cast<SizeType32>(divUp(mSamplingParams.getMaxDecodingTokens(), 32)));
                    EXPECT_EQ(bitmask[mi], packedMasks[packedMaskIdx]) << " bi: " << bi << " ti: " << ti;
                }
            }
        }
    }

    // Check accepted tokens
    auto const outputIds = BufferRange<TokenIdType>(*mOutputIds);
    auto const refOutputIds = mNetwork.getOutputIds();
    auto const promptIds = mNetwork.getPrompts();
    auto const seqLenghts = BufferRange<SizeType32>(*mSeqLengths);
    auto const lastDraftTokens = BufferRange<TokenIdType>(*mLastDraftTokens);
    auto const bestPathLengths = BufferRange<SizeType32>(*mBestPathLengths);
    auto const bestPathIndices = BufferRange<SizeType32>(*mBestPathIndices);
    for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
    {
        auto const batchSlot = batchSlots[bi];
        // Updated seq length is prompt length and newly accepted tokens.
        EXPECT_EQ(seqLenghts[batchSlot], promptIds[bi].size() + bestPathLengths[bi]) << " bi: " << bi;
        // Check that output ids contains accepted tokens.
        for (SizeType32 ti = 0; ti < promptIds[bi].size() + bestPathLengths[bi]; ++ti)
        {
            EXPECT_EQ(outputIds[batchSlot * mSamplingParams.getMaxSeqLen() + ti], refOutputIds[bi][ti])
                << " bi: " << bi << " ti: " << ti;
        }
        auto outputIter = outputIds.begin() + batchSlot * mSamplingParams.getMaxSeqLen();
        std::vector<TokenIdType> outputVec(outputIter, outputIter + seqLenghts[batchSlot]);
        TLLM_LOG_DEBUG("Output ids at %d request is \"%s\"", bi, mNetwork.detokenize(outputVec).c_str());
        TLLM_LOG_DEBUG("Ref output ids at %d request is \"%s\"", bi, mNetwork.detokenize(refOutputIds[bi]).c_str());
    }

    // Check new draft tokens
    {
        auto const outputNextDraftTokens = BufferRange<TokenIdType>(*mOutputNextDraftTokens);
        auto const generationLengths = BufferRange<SizeType32>(*mSpecDecodingGenerationLengths);
        auto const compressedDraftTokens = mNetwork.getNextFlatTokens();
        TLLM_LOG_DEBUG("Next compressed draft tokens are \"%s\"", mNetwork.detokenize(compressedDraftTokens).c_str());
        SizeType32 compressedIdx = 0;
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            auto const batchSlot = batchSlots[bi];
            auto const generatedLength = generationLengths[bi];
            // Check draft tokens for the next iteration.
            for (SizeType32 ti = 0; ti < generatedLength - 1; ++ti)
            {
                auto const idx = flat_index2(batchSlot, ti, mSamplingParams.getMaxDecodingDraftTokens());
                EXPECT_EQ(outputNextDraftTokens[idx], compressedDraftTokens[compressedIdx + ti + 1])
                    << " bi: " << bi << " ti: " << ti;
            }
            // Check length of the draft tokens.
            EXPECT_EQ(BufferRange<SizeType32>(*mNextDraftLengths)[batchSlot], generatedLength - 1) << " bi: " << bi;
            // Check accepted length.
            EXPECT_EQ(BufferRange<SizeType32>(*mAcceptedLengths)[batchSlot], bestPathLengths[bi]) << " bi: " << bi;
            compressedIdx += generatedLength;
        }
    }

    // Check position ids
    {
        auto const outputPositionIdsBase = BufferRange<SizeType32>(*mOutputPositionIdsBase);
        auto const nextPosIds = BufferRange<SizeType32>(*mNextPosIds);
        auto const generationLengths = BufferRange<SizeType32>(*mSpecDecodingGenerationLengths);
        auto const packedPosIds = mNetwork.getNextPackedPosId();
        SizeType32 compressedIdx = 0;
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            auto const batchSlot = batchSlots[bi];
            EXPECT_EQ(outputPositionIdsBase[batchSlot], seqLenghts[batchSlot]);

            auto const generatedLength = generationLengths[bi];
            // Check pos ids for the next iteration.
            for (SizeType32 ti = 0; ti < generatedLength; ++ti)
            {
                auto const idx = flat_index2(batchSlot, ti, mSamplingParams.getMaxDecodingTokens());
                // Minus -1 to account for context phase correction of pos ids
                EXPECT_EQ(nextPosIds[idx], packedPosIds[compressedIdx + ti] - 1) << " bi: " << bi << " ti: " << ti;
            }
            compressedIdx += generatedLength;
        }
    }

    // Check unpacked indices and tokens
    {
        auto const nextDraftTokens = mNetwork.getNextDraftTokens();
        auto const nextDraftIndices = mNetwork.getNextDraftIndices();
        auto const nextDraftTokensRange = BufferRange<TokenIdType>(*mOutputUnpackedNextDraftTokens);
        auto const nextDraftIndicesRange = BufferRange<SizeType32>(*mOutputUnpackedNextDraftIndices);
        for (SizeType32 bi = 0; bi < nextDraftTokens.size(); ++bi)
        {
            auto const batchSlot = batchSlots[bi];
            for (SizeType32 pi = 0; pi < nextDraftTokens[bi].size(); ++pi)
            {
                for (SizeType32 ti = 0; ti < nextDraftTokens[bi][pi].size(); ++ti)
                {
                    auto idx = flat_index3(
                        batchSlot, pi, ti, mSamplingParams.getMaxNumPaths(), mSamplingParams.getMaxPathLen());
                    EXPECT_EQ(nextDraftTokensRange[idx], nextDraftTokens[bi][pi][ti])
                        << "bi: " << bi << " pi: " << pi << " ti: " << ti;
                    EXPECT_EQ(nextDraftIndicesRange[idx], nextDraftIndices[bi][pi][ti])
                        << "bi: " << bi << " pi: " << pi << " ti: " << ti;
                }
            }
        }
    }

    // Check accumulated cum sum and paths offsets
    {
        auto const accumulatedCumSum = BufferRange<SizeType32>(*mAcceptedLengthCumSum);
        auto const pathsOffsets = BufferRange<SizeType32>(*mPathsOffsets);
        auto const acceptedLengths = BufferRange<SizeType32>(*mAcceptedLengths);
        auto const bestPathIndices = BufferRange<SizeType32>(*mBestPathIndices);
        auto const lastDraftIndices = mNetwork.getLastDraftIndices();
        SizeType32 sum = 0;
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            auto const batchSlot = batchSlots[bi];
            EXPECT_EQ(sum, accumulatedCumSum[bi]) << "bi: " << bi;
            auto const acceptedLength = acceptedLengths[batchSlot] - 1;
            for (SizeType32 ti = 0; ti < acceptedLength; ++ti)
            {
                EXPECT_EQ(pathsOffsets[sum + ti], lastDraftIndices[bi][bestPathIndices[bi]][ti + 1] - 1)
                    << "bi: " << bi << " ti: " << ti;
            }
            sum += acceptedLength;
        }
        EXPECT_EQ(sum, accumulatedCumSum[mSamplingParams.getBatchSize()]);
    }

    // Check draft probs
    {
        auto const outDraftProbs = BufferRange<DataType>(*mOutputDraftProbs);
        auto const inDraftProbs = BufferRange<DataType>(*mNextDraftProbs);
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            auto const batchSlot = batchSlots[bi];
            for (SizeType32 pi = 0; pi < mSamplingParams.getMaxNumPaths(); ++pi)
            {
                for (SizeType32 ti = 0; ti < mSamplingParams.getMaxDraftPathLen(); ++ti)
                {
                    for (SizeType32 vi = 0; vi < mSamplingParams.getVocabSize(); ++vi)
                    {
                        auto const outProbIdx = flat_index4(batchSlot, pi, ti, vi, mSamplingParams.getMaxNumPaths(),
                            mSamplingParams.getMaxDraftPathLen(), mSamplingParams.getVocabSize());
                        auto const inProbIdx = flat_index4(bi, pi, ti, vi, mSamplingParams.getMaxNumPaths(),
                            mSamplingParams.getMaxDraftPathLen(), mSamplingParams.getVocabSize());
                        EXPECT_EQ(outDraftProbs[outProbIdx], inDraftProbs[inProbIdx])
                            << "bi: " << bi << " pi: " << pi << " ti: " << ti << " vi: " << vi;
                    }
                }
            }
        }
    }

    // Check temperature
    {
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            auto const batchSlot = batchSlots[bi];
            EXPECT_EQ(
                BufferRange<DataType>(*mOutputTemperatures)[batchSlot], static_cast<DataType>(1.f / mTemperatures[bi]))
                << " bi: " << bi;
        }
    }
}

template <typename T>
void ExplicitDraftTokensLayerTest<T>::packData()
{
    using DataType = typename T::DataType;
    tksd::PackExplicitDraftTokensParams<DataType> params;
    params.batchSlots = bufferCast<SizeType32>(*mBatchSlots);
    params.cumSumGenerationLengths = bufferCast<SizeType32>(*mCumSumGenerationLengths);
    params.maxGenerationLength = bufferCast<SizeType32>(*mMaxGenerationLength);

    params.outputPositionIdsBase = bufferCast<SizeType32>(*mPackedPositionIdsBase);
    params.inputPositionIdsBase = bufferCast<SizeType32>(*mOutputPositionIdsBase);

    params.outputGenerationLengths = bufferCast<SizeType32>(*mPackedGenerationLengths);
    params.inputGenerationLengths = bufferCast<SizeType32>(*mSpecDecodingGenerationLengths);

    params.outputRandomDataSample = bufferCast<DataType>(*mPackedRandomDataSample);
    params.inputRandomDataSample = bufferCast<DataType>(*mRandomDataSample);

    params.outputRandomDataValidation = bufferCast<DataType>(*mPackedRandomDataVerification);
    params.inputRandomDataValidation = bufferCast<DataType>(*mRandomDataValidation);

    params.outputNextDraftTokens = bufferCast<TokenIdType>(*mPackedNextDraftTokens);
    params.inputNextDraftTokens = bufferCast<TokenIdType>(*mOutputUnpackedNextDraftTokens);

    params.outputNextDraftIndices = bufferCast<SizeType32>(*mPackedNextDraftIndices);
    params.inputNextDraftIndices = bufferCast<SizeType32>(*mOutputUnpackedNextDraftIndices);

    params.outputPackedMask = bufferCast<int32_t>(*mPackedPackedMasks);
    params.inputPackedMask = bufferCast<int32_t>(*mPackedMasks);

    params.inputPositionIds = bufferCast<SizeType32>(*mNextPosIds);
    params.outputPositionOffsets = bufferCast<SizeType32>(*mPackedPositionOffsets);
    params.outputPositionIds = bufferCast<SizeType32>(*mPackedPackedPosIds);

    params.outputDraftProbs = bufferCast<DataType>(*mPackedDraftProbs);
    params.inputDraftProbs = bufferCast<DataType>(*mOutputDraftProbs);

    params.outputTemperatures = bufferCast<DataType>(*mPackedTemperatures);
    params.inputTemperatures = bufferCast<DataType>(*mOutputTemperatures);

    params.batchSize = mSamplingParams.getBatchSize();
    params.numPaths = mSamplingParams.getMaxNumPaths();
    params.maxPathLength = mSamplingParams.getMaxPathLen();
    params.vocabSize = mSamplingParams.getVocabSize();
    params.numGenerationRequests = mSamplingParams.getBatchSize();
    params.numContextTokens = 0;

    params.checkParams();

    tksd::invokePackGenerationLengths(params, mStream->get());

    // Compute inclusive sum
    auto reduceTempStorageBytes = tksd::invokeScanGenerationLengths(
        nullptr, 0, nullptr, nullptr, mSamplingParams.getBatchSize(), mStream->get());
    auto reduceMaxTempStorage = mBufferManager->gpu(reduceTempStorageBytes);
    tksd::invokeScanGenerationLengths(bufferCast<uint8_t>(*reduceMaxTempStorage), reduceTempStorageBytes,
        bufferCast<SizeType32>(*mSpecDecodingGenerationLengths), bufferCast<SizeType32>(*mCumSumGenerationLengths),
        mSamplingParams.getBatchSize(), mStream->get());

    // Pack tensors from batch slot position to continuous array
    tksd::invokePackExplicitDraftTokens(params, mStream->get());

    // Copy draft probs
    tksd::invokeCopyProbs(params, mStream->get());
}

template <typename T>
void ExplicitDraftTokensLayerTest<T>::checkPackResult()
{
    using DataType = typename T::DataType;
    auto const batchSlots = BufferRange<SizeType32>(*mBatchSlots);
    auto const maxGenLength = mNetwork.getMaxNextGenerationLength();
    auto const numPackedMasks = static_cast<SizeType32>(divUp(mSamplingParams.getMaxDecodingTokens(), 32));
    for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
    {
        auto const batchSlot = batchSlots[bi];
        EXPECT_EQ(BufferRange<SizeType32>(*mPackedPositionIdsBase)[bi],
            BufferRange<SizeType32>(*mOutputPositionIdsBase)[batchSlot])
            << "bi: " << bi;
        EXPECT_EQ(BufferRange<SizeType32>(*mPackedGenerationLengths)[bi],
            BufferRange<SizeType32>(*mSpecDecodingGenerationLengths)[batchSlot])
            << "bi: " << bi;
        EXPECT_EQ(
            BufferRange<DataType>(*mPackedRandomDataSample)[bi], BufferRange<DataType>(*mRandomDataSample)[batchSlot])
            << "bi: " << bi;
        EXPECT_EQ(
            BufferRange<DataType>(*mPackedTemperatures)[bi], BufferRange<DataType>(*mOutputTemperatures)[batchSlot])
            << "bi: " << bi;

        for (SizeType32 pi = 0; pi < mSamplingParams.getMaxNumPaths(); ++pi)
        {
            for (SizeType32 ti = 0; ti < mSamplingParams.getMaxDraftPathLen(); ++ti)
            {
                EXPECT_EQ(bufferCast<DataType>(*ITensor::at(mPackedRandomDataVerification, {bi, pi, ti}))[0],
                    bufferCast<DataType>(*ITensor::at(mRandomDataValidation, {batchSlot, pi, ti}))[0])
                    << "bi: " << bi << " pi: " << pi << " ti: " << ti;
                for (SizeType32 vi = 0; vi < mSamplingParams.getVocabSize(); ++vi)
                {
                    EXPECT_EQ(bufferCast<DataType>(*ITensor::at(mPackedDraftProbs, {bi, pi, ti, vi}))[0],
                        bufferCast<DataType>(*ITensor::at(mOutputDraftProbs, {batchSlot, pi, ti, vi}))[0])
                        << "bi: " << bi << " pi: " << pi << " ti: " << ti << " vi: " << vi;
                }
            }
            for (SizeType32 ti = 0; ti < mSamplingParams.getMaxPathLen(); ++ti)
            {
                EXPECT_EQ(bufferCast<TokenIdType>(*ITensor::at(mPackedNextDraftTokens, {bi, pi, ti}))[0],
                    bufferCast<TokenIdType>(*ITensor::at(mOutputUnpackedNextDraftTokens, {batchSlot, pi, ti}))[0])
                    << "bi: " << bi << " pi: " << pi << " ti: " << ti;
                EXPECT_EQ(bufferCast<SizeType32>(*ITensor::at(mPackedNextDraftIndices, {bi, pi, ti}))[0],
                    bufferCast<SizeType32>(*ITensor::at(mOutputUnpackedNextDraftIndices, {batchSlot, pi, ti}))[0])
                    << "bi: " << bi << " pi: " << pi << " ti: " << ti;
            }
        }
        auto const basePosId = BufferRange<SizeType32>(*mPackedPositionIdsBase)[bi];
        for (SizeType32 ti = 0; ti < maxGenLength; ++ti)
        {
            auto const outPosOffsetIdx = flat_index2(bi, ti, maxGenLength);
            auto const inPosOffsetIdx = flat_index2(batchSlot, ti, mSamplingParams.getMaxDecodingTokens());
            EXPECT_EQ(BufferRange<SizeType32>(*mPackedPositionOffsets)[outPosOffsetIdx],
                BufferRange<SizeType32>(*mNextPosIds)[inPosOffsetIdx] - basePosId + 1)
                << "bi: " << bi << " ti: " << ti;
        }
        auto const outputMaskStartId = (bi == 0) ? 0 : BufferRange<SizeType32>(*mCumSumGenerationLengths)[bi - 1];
        auto const numTokens = (bi == 0) ? BufferRange<SizeType32>(*mCumSumGenerationLengths)[0]
                                         : BufferRange<SizeType32>(*mCumSumGenerationLengths)[bi]
                - BufferRange<SizeType32>(*mCumSumGenerationLengths)[bi - 1];
        for (SizeType32 mi = 0; mi < numTokens * numPackedMasks; ++mi)
        {
            auto const outMaskIdx = outputMaskStartId * numPackedMasks + mi;
            auto const inMaskIdx = flat_index2(batchSlot, mi, mSamplingParams.getMaxDecodingTokens() * numPackedMasks);
            EXPECT_EQ(
                BufferRange<int32_t>(*mPackedPackedMasks)[outMaskIdx], BufferRange<int32_t>(*mPackedMasks)[inMaskIdx])
                << "bi: " << bi << " mi: " << mi;
        }
    }
}

template <typename T>
void ExplicitDraftTokensLayerTest<T>::runTest(std::vector<std::string> const& prompts,
    std::vector<std::string> const& predictions, DraftLettersVec const& nextDraftLetters,
    DraftLettersVec const& lastDraftLetters, SamplingParams& params)
{
    mSamplingParams = params;

    mNetwork.forward(params, prompts, predictions, nextDraftLetters, lastDraftLetters);

    allocateBuffers();

    setup();

    auto inputTensors = createInputTensors();
    auto outputTensors = createOutputTensors();

    mDecodingWorkspace->setDeviceBatchSlots(mBatchSlots);
    mExplicitDraftTokensLayer->forwardAsync(outputTensors, inputTensors, mDecodingWorkspace);

    mStream->synchronize();

    checkLayerResult();

    packData();

    mStream->synchronize();

    checkPackResult();
}

template class ExplicitDraftTokensLayerTest<TypePair<float, float>>;
template class ExplicitDraftTokensLayerTest<TypePair<half, half>>;
#ifdef ENABLE_BF16
template class ExplicitDraftTokensLayerTest<TypePair<half, __nv_bfloat16>>;
#endif // ENABLE_BF16

TYPED_TEST_SUITE(ExplicitDraftTokensLayerTest, TestTypes);

TYPED_TEST(ExplicitDraftTokensLayerTest, SimpleTestBS1)
{
    SamplingParams params;

    std::vector<std::string> prompt = {"Hi mate, h"};
    std::vector<std::string> predictions = {"how things"};
    DraftLettersVec lastDraftLetters = {{"how do ", "how are", "however", "hello w"}};
    DraftLettersVec nextDraftLetters = {{"things ", "that is", "to crea", "touchab"}};

    params.setBatchSize(1);

    this->runTest(prompt, predictions, nextDraftLetters, lastDraftLetters, params);
}

TYPED_TEST(ExplicitDraftTokensLayerTest, SimpleTestBS1OnePaths)
{
    SamplingParams params;

    std::vector<std::string> prompt = {"Hi mate, h"};
    std::vector<std::string> predictions = {"how things"};
    DraftLettersVec lastDraftLetters = {{"how do "}};
    DraftLettersVec nextDraftLetters = {{"things "}};

    params.setBatchSize(1);
    params.setMaxNumPaths(1);

    this->runTest(prompt, predictions, nextDraftLetters, lastDraftLetters, params);
}

TYPED_TEST(ExplicitDraftTokensLayerTest, SimpleTestSecondPathAcceptedBS1)
{
    SamplingParams params;

    std::vector<std::string> prompt = {"Hi mate, h"};
    std::vector<std::string> predictions = {"how things"};
    DraftLettersVec lastDraftLetters = {{"howdy f", "how are", "however", "hello w"}};
    DraftLettersVec nextDraftLetters = {{"things ", "that is", "to crea", "touchab"}};

    params.setBatchSize(1);

    this->runTest(prompt, predictions, nextDraftLetters, lastDraftLetters, params);
}

TYPED_TEST(ExplicitDraftTokensLayerTest, SimpleTestNoDraftAcceptedBS1)
{
    SamplingParams params;

    std::vector<std::string> prompt = {"Hi mate, h"};
    std::vector<std::string> predictions = {"how things"};
    DraftLettersVec lastDraftLetters = {{"handove", "human f", "heavy l", "hello h"}};
    DraftLettersVec nextDraftLetters = {{"oatmeal", "ocean b", "occupat", "oblivio"}};

    params.setBatchSize(1);

    this->runTest(prompt, predictions, nextDraftLetters, lastDraftLetters, params);
}

TYPED_TEST(ExplicitDraftTokensLayerTest, SimpleTestBS2SameSequence)
{
    SamplingParams params;

    std::vector<std::string> prompt = {"Hi mate, h", "Hi mate, h"};
    std::vector<std::string> predictions = {"how things", "how things"};
    DraftLettersVec lastDraftLetters
        = {{"how do ", "how are", "however", "hello w"}, {"how do ", "how are", "however", "hello w"}};
    DraftLettersVec nextDraftLetters
        = {{"things ", "that is", "to crea", "touchab"}, {"things ", "that is", "to crea", "touchab"}};

    params.setBatchSize(2);

    this->runTest(prompt, predictions, nextDraftLetters, lastDraftLetters, params);
}

TYPED_TEST(ExplicitDraftTokensLayerTest, SimpleTestBS2Long)
{
    SamplingParams params;

    std::vector<std::string> prompt = {"Hi mate, h", "London is t"};
    std::vector<std::string> predictions = {"how things are going", "the capital of Great Britain"};
    DraftLettersVec lastDraftLetters = {{"how do you ", "how are you", "however you", "hello world"},
        {"the bar and", "the best ci", "the capital", "thoughest p"}};
    DraftLettersVec nextDraftLetters = {{"things are ", "that is sad", "to create a", "touchable y"},
        {" of Great B", " and the ma", " of country", " also known"}};

    params.setBatchSize(2);
    // ceil(4 * 10 / 32) = 2 masks per request
    params.setMaxNumPaths(4);
    params.setMaxDraftPathLen(10);

    this->runTest(prompt, predictions, nextDraftLetters, lastDraftLetters, params);
}

TYPED_TEST(ExplicitDraftTokensLayerTest, SimpleTestBS2DifferentSequences)
{
    SamplingParams params;

    std::vector<std::string> prompt = {"Hi mate, h", "London is t"};
    std::vector<std::string> predictions = {"how things", "the cap"};
    DraftLettersVec lastDraftLetters
        = {{"how do ", "how are", "however", "hello w"}, {"the bar", "the bes", "the cap", "thoughe"}};
    DraftLettersVec nextDraftLetters
        = {{"things ", "that is", "to crea", "touchab"}, {"itan of", "iteract", "ital of", "importa"}};

    params.setBatchSize(2);

    this->runTest(prompt, predictions, nextDraftLetters, lastDraftLetters, params);
}

TYPED_TEST(ExplicitDraftTokensLayerTest, SimpleTestB4DifferentSequences)
{
    SamplingParams params;

    std::vector<std::string> prompt = {"Hi mate, h", "London is t", "Short", "Very long prompt but should not m"};
    std::vector<std::string> predictions = {"how things", "the cap", "twave o", "matter "};
    DraftLettersVec lastDraftLetters
        = {{"how do ", "how are", "however", "hello w"}, {"the bar", "the bes", "the cap", "thoughe"},
            {"t promp", "ts on Y", "ter out", "twave o"}, {"matter ", "mean an", "make th", "modify "}};
    DraftLettersVec nextDraftLetters
        = {{"things ", "that is", "to crea", "touchab"}, {"itan of", "iteract", "ital of", "importa"},
            {" chips ", " oil an", " semico", " exampl"}, {"at all ", "anythin", "above a", "albeit "}};

    params.setBatchSize(4);

    this->runTest(prompt, predictions, nextDraftLetters, lastDraftLetters, params);
}

template <typename T>
class FillRandDataTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    static auto constexpr mDataType{TRTDataType<T>::value};

    FillRandDataTest() {}

    void SetUp() override
    {
        mLogger = std::make_shared<TllmLogger>();
        mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);
    }

    void TearDown() override {}

    void runTest(SizeType32 batchSize, SizeType32 numPaths, SizeType32 draftLength, bool skipVerification,
        uint64_t randomSeed, bool batchInit)
    {
        SizeType32* batchSlotsPtr{nullptr};

        auto curandState = mBufferManager->gpu(ITensor::makeShape({batchSize, 48}), nvinfer1::DataType::kUINT8);
        auto* curandStatePtr = reinterpret_cast<curandState_t*>(bufferCast<uint8_t>(*curandState));

        if (batchInit)
        {
            auto randomSeeds = mBufferManager->gpu(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT64);
            trk::invokeFill(*randomSeeds, static_cast<int64_t>(randomSeed), *mStream);
            auto* randomSeedsPtr = bufferCast<uint64_t>(*randomSeeds);
            tk::invokeCurandBatchInitialize(curandStatePtr, batchSlotsPtr, batchSize, randomSeedsPtr, mStream->get());
        }
        else
        {
            tk::invokeCurandInitialize(curandStatePtr, batchSlotsPtr, batchSize, randomSeed, mStream->get());
        }
        mStream->synchronize();

        tksd::FillRandDataExplicitDraftTokensParams<T> params;
        params.batchSize = batchSize;
        params.numPaths = numPaths;
        params.draftLength = draftLength;
        params.skipVerification = skipVerification;

        auto randDataSample = mBufferManager->gpu(ITensor::makeShape({batchSize}), mDataType);
        auto randDataValidation
            = mBufferManager->gpu(ITensor::makeShape({batchSize, numPaths, draftLength}), mDataType);

        params.randDataSample = bufferCast<T>(*randDataSample);
        params.randDataVerification = bufferCast<T>(*randDataValidation);
        params.curandState = curandStatePtr;
        params.batchSlots = batchSlotsPtr;

        tksd::invokeFillRandData(params, mStream->get());
        mStream->synchronize();

        auto randDataSampleHost = mBufferManager->copyFrom(*randDataSample, MemoryType::kCPU);
        auto randDataSampleHostPtr = bufferCast<T>(*randDataSampleHost);
        EXPECT_GE(randDataSampleHostPtr[0], T(0));
        EXPECT_LE(randDataSampleHostPtr[0], T(1));

        auto randDataValidationHost = mBufferManager->copyFrom(*randDataValidation, MemoryType::kCPU);
        auto randDataValidationHostRange = BufferRange<T>(*randDataValidationHost);
        for (auto i = 0; i < randDataValidationHostRange.size(); ++i)
        {
            EXPECT_GE(randDataValidationHostRange[i], T(0)) << "index " << i;
            EXPECT_LE(randDataValidationHostRange[i], T(1)) << "index " << i;
        }
    }

private:
    std::shared_ptr<nvinfer1::ILogger> mLogger;
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
};

#ifdef ENABLE_BF16
using FloatHalfBfloatTypes = testing::Types<float, half, __nv_bfloat16>;
TYPED_TEST_SUITE(FillRandDataTest, FloatHalfBfloatTypes);
#else
TYPED_TEST_SUITE(FillRandDataTest, FloatAndHalfTypes);
#endif

TYPED_TEST(FillRandDataTest, SimpleTest)
{
    SizeType32 constexpr batchSize{2};
    SizeType32 constexpr numPaths{3};
    SizeType32 constexpr draftLength{4};
    bool constexpr skipVerification{false};

    uint64_t randomSeed{0};

    this->runTest(batchSize, numPaths, draftLength, skipVerification, randomSeed, false);
}

TYPED_TEST(FillRandDataTest, BatchInit)
{
    SizeType32 constexpr batchSize{3};
    SizeType32 constexpr numPaths{2};
    SizeType32 constexpr draftLength{5};
    bool constexpr skipVerification{false};

    uint64_t randomSeed{42};

    this->runTest(batchSize, numPaths, draftLength, skipVerification, randomSeed, true);
}

} // namespace tensorrt_llm::tests::layers
