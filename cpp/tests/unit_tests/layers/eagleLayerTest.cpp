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

#include "eagleLayerTest.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/speculativeDecoding/eagleDecodingKernels.h"
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

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::layers;
using namespace tensorrt_llm::common;

namespace tk = tensorrt_llm::kernels;
namespace tksd = tensorrt_llm::kernels::speculative_decoding;
namespace trk = tensorrt_llm::runtime::kernels;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TokensVec EagleDummyNetwork::tokenize(std::string const& letters) const
{
    TokensVec tokens;
    for (char c : letters)
    {
        tokens.push_back(static_cast<TokenIdType>(c));
    }
    return tokens;
}

std::string EagleDummyNetwork::detokenize(TokensVec const& tokens) const
{
    std::string letters;
    for (int token : tokens)
    {
        letters += static_cast<char>(token);
    }
    return letters;
}

DraftTokensVec EagleDummyNetwork::draftLettersToTokens(DraftLettersVec const& draftLetters) const
{
    DraftTokensVec draftTokens(draftLetters.size());
    for (SizeType32 bi = 0; bi < draftLetters.size(); ++bi)
    {
        draftTokens[bi] = tokenize(draftLetters[bi]);
    }
    return draftTokens;
}

SizeType32 EagleDummyNetwork::longestCommonPrefixLength(TokensVec const& a, TokensVec const& b) const
{
    SizeType32 minLength = std::min(a.size(), b.size());
    SizeType32 idx = 0;
    while (idx < minLength && a[idx] == b[idx])
    {
        ++idx;
    }
    return idx;
}

DraftPath EagleDummyNetwork::pathFromDraftTokens(
    DraftTokensVec const& tokens, SizeType32 maxDecodingTokens, SizeType32 maxPathLen) const
{
    DraftPath path(maxDecodingTokens);
    for (SizeType32 pi = 0; pi < maxDecodingTokens; ++pi)
    {
        path[pi].resize(maxPathLen);
        for (SizeType32 ti = 0; ti < maxPathLen; ++ti)
        {
            path[pi][ti] = -1;
        }
    }
    SizeType32 draftPosCounter = 1;
    for (SizeType32 ti = 1; ti < maxPathLen; ++ti)
    {
        std::unordered_map<std::string, SizeType32> tokenPosMap;
        for (SizeType32 pi = 0; pi < tokens.size(); ++pi)
        {
            if (tokens[pi].size() > ti - 1)
            {
                path[pi][0] = 0;
                auto const token = tokens[pi][ti - 1];
                auto const draftPrefix = detokenize(tokens[pi]).substr(0, ti);
                if (tokenPosMap.count(draftPrefix) == 0)
                {
                    tokenPosMap[draftPrefix] = draftPosCounter++;
                }
                path[pi][ti] = tokenPosMap[draftPrefix];
            }
        }
    }
    return path;
}

TokensVec EagleDummyNetwork::flattenTokens(
    DraftTokensVec const& tokens, DraftPath const& path, bool isDraftTokens) const
{
    SizeType32 maxPathIdx{-1};
    for (SizeType32 pi = 0; pi < path.size(); ++pi)
    {
        for (SizeType32 ti = 0; ti < path[pi].size(); ++ti)
        {
            auto const pathIdx = path[pi][ti];
            maxPathIdx = std::max(pathIdx, maxPathIdx);
        }
    }
    if (!isDraftTokens)
    {
        maxPathIdx++;
    }
    TokensVec flattenedTokens(maxPathIdx);
    for (SizeType32 pi = 0; pi < path.size(); ++pi)
    {
        for (SizeType32 ti = 0; ti < path[pi].size(); ++ti)
        {
            if (isDraftTokens && ti == 0)
            {
                continue;
            }

            auto const pathIdx = path[pi][ti];
            if (pathIdx != -1)
            {
                if (isDraftTokens)
                {
                    flattenedTokens[pathIdx - 1] = tokens[pi][ti - 1];
                }
                else
                {
                    flattenedTokens[pathIdx] = tokens[pi][ti];
                }
            }
        }
    }
    return flattenedTokens;
}

std::vector<std::vector<std::vector<bool>>> EagleDummyNetwork::createMasks(DraftPaths const& paths) const
{
    std::vector<std::vector<std::vector<bool>>> masks;
    for (SizeType32 bi = 0; bi < paths.size(); ++bi)
    {
        std::vector<std::vector<bool>> localMask(paths[bi].size());
        for (SizeType32 ti = 0; ti < paths[bi].size(); ++ti)
        {
            localMask[ti].resize(paths[bi].size());
        }
        localMask[0][0] = true;

        for (SizeType32 pi = 0; pi < paths[bi].size(); ++pi)
        {
            for (SizeType32 ti = 1; ti < paths[bi][pi].size(); ++ti)
            {
                auto const to = paths[bi][pi][ti];
                if (to == -1)
                {
                    break;
                }
                localMask[to][to] = true;
                for (SizeType32 fi = 0; fi < ti; ++fi)
                {
                    auto const from = paths[bi][pi][fi];
                    localMask[to][from] = true;
                }
            }
        }
        masks.push_back(localMask);
    }
    return masks;
}

void EagleDummyNetwork::acceptTokens(std::vector<TokensVec> const& predictionTokens,
    DraftTokensVec const& lastDraftTokens, DraftPaths const& lastDraftPaths)
{
    TLLM_CHECK_WITH_INFO(predictionTokens.size() == lastDraftTokens.size(),
        "Batch size of predictions (%d) does not match the batch size of last draft tokens (%d)",
        static_cast<SizeType32>(predictionTokens.size()), static_cast<SizeType32>(lastDraftTokens.size()));
    TLLM_CHECK_WITH_INFO(predictionTokens.size() == lastDraftPaths.size(),
        "Batch size of predictions (%d) does not match the batch size of last draft paths (%d)",
        static_cast<SizeType32>(predictionTokens.size()), static_cast<SizeType32>(lastDraftPaths.size()));

    mAcceptedTokens.resize(predictionTokens.size());
    mAcceptedLens.resize(predictionTokens.size());
    mAcceptedPathIds.resize(predictionTokens.size());
    // Needed for unit test of EagleDummyNetwork only.
    if (mOutputIds.size() == 0)
    {
        mOutputIds.resize(lastDraftTokens.size());
    }
    for (SizeType32 bi = 0; bi < lastDraftPaths.size(); ++bi)
    {
        SizeType32 maxMatchLen = -1;
        SizeType32 maxMatchIdx = -1;
        std::vector<TokenIdType> bestDraftPath;
        // Find path with largest prefix shared with the predicted tokens.
        for (SizeType32 pi = 0; pi < lastDraftPaths[bi].size(); ++pi)
        {
            TokensVec predictedPath(lastDraftPaths[bi][pi].size());
            TokensVec draftPath(lastDraftPaths[bi][pi].size());
            for (SizeType32 ti = 0; ti < lastDraftPaths[bi][pi].size(); ++ti)
            {
                predictedPath[ti] = predictionTokens[bi][lastDraftPaths[bi][pi][ti]];
                if (ti > 0)
                {
                    draftPath[ti - 1] = lastDraftTokens[bi][lastDraftPaths[bi][pi][ti] - 1];
                }
            }
            auto const matchLen = longestCommonPrefixLength(draftPath, predictedPath);
            if (matchLen > maxMatchLen)
            {
                maxMatchLen = matchLen;
                maxMatchIdx = pi;
                bestDraftPath = predictedPath;
            }
        }

        mAcceptedTokens[bi] = bestDraftPath;
        mAcceptedLens[bi] = maxMatchLen + 1;
        mAcceptedPathIds[bi] = maxMatchIdx;
        // Update output ids. First draft token is already counted in outputs
        mOutputIds[bi].insert(mOutputIds[bi].end(), bestDraftPath.begin(), bestDraftPath.begin() + maxMatchLen + 1);
    }
}

void EagleDummyNetwork::forward(SamplingParams const& params, std::vector<std::string> const& prompts,
    std::vector<std::vector<std::string>> const& predictionLetters,
    std::vector<DraftLettersVec> const& nextDraftLetters, std::vector<DraftLettersVec> const& lastDraftLetters)
{
    mSamplingParams = params;

    TLLM_CHECK(params.getBatchSize() == nextDraftLetters.size());
    TLLM_CHECK(params.getBatchSize() == lastDraftLetters.size());

    DraftPaths lastDraftPaths;
    DraftPaths nextDraftPaths;
    DraftTokensVec lastDraftTokensFlattened;
    DraftTokensVec nextDraftTokensFlattened;
    std::vector<TokensVec> predictionTokensFlattened;
    for (SizeType32 bi = 0; bi < params.getBatchSize(); ++bi)
    {
        auto const lastDraftTokens = draftLettersToTokens(lastDraftLetters[bi]);
        auto const nextDraftTokens = draftLettersToTokens(nextDraftLetters[bi]);
        auto const lastDraftPath
            = pathFromDraftTokens(lastDraftTokens, params.getMaxDecodingTokens(), params.getMaxPathLen());
        auto const nextDraftPath
            = pathFromDraftTokens(nextDraftTokens, params.getMaxDecodingTokens(), params.getMaxPathLen());
        auto const predictionTokens = draftLettersToTokens(predictionLetters[bi]);

        lastDraftPaths.push_back(lastDraftPath);
        nextDraftPaths.push_back(nextDraftPath);
        lastDraftTokensFlattened.push_back(flattenTokens(lastDraftTokens, lastDraftPath, /* isDraftTokens */ true));
        nextDraftTokensFlattened.push_back(flattenTokens(nextDraftTokens, nextDraftPath, /* isDraftTokens */ true));
        predictionTokensFlattened.push_back(flattenTokens(predictionTokens, lastDraftPath, /* isDraftTokens */ false));
    }

    mNextDraftTokens = nextDraftTokensFlattened;
    mLastDraftTokens = lastDraftTokensFlattened;

    mNextDraftPaths = nextDraftPaths;
    mLastDraftPaths = lastDraftPaths;

    mNextDraftLens.resize(params.getBatchSize());
    mLastDraftLens.resize(params.getBatchSize());
    for (SizeType32 bi = 0; bi < params.getBatchSize(); ++bi)
    {
        mNextDraftLens[bi] = mNextDraftTokens[bi].size();
        mLastDraftLens[bi] = mLastDraftTokens[bi].size();
    }

    std::vector<TokensVec> predictionTokens;
    for (SizeType32 bi = 0; bi < predictionLetters.size(); ++bi)
    {
        mPrompts.push_back(tokenize(prompts[bi]));
    }

    mOutputIds = mPrompts;

    acceptTokens(predictionTokensFlattened, mLastDraftTokens, lastDraftPaths);

    mMasks = createMasks(mNextDraftPaths);
}

TEST(EagleDummyNetworkTest, tokenizeTest)
{
    EagleDummyNetwork network;

    {
        auto tokens = network.tokenize("hello world");
        EXPECT_EQ(tokens, std::vector<TokenIdType>({104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100}));
    }
    {
        DraftLettersVec lettersVec = {{"hello world"}, {"world"}};
        auto draftTokens = network.draftLettersToTokens(lettersVec);
        ASSERT_EQ(draftTokens.size(), 2);
        ASSERT_EQ(draftTokens[0].size(), 11);
        ASSERT_EQ(draftTokens[1].size(), 5);
        EXPECT_EQ(draftTokens[0], std::vector<TokenIdType>({104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100}));
        EXPECT_EQ(draftTokens[1], std::vector<TokenIdType>({119, 111, 114, 108, 100}));
    }
}

TEST(EagleDummyNetworkTest, detokenizeTest)
{
    EagleDummyNetwork network;

    {
        auto letters
            = network.detokenize(std::vector<TokenIdType>({104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100}));
        EXPECT_EQ(letters, "hello world");
    }
}

TEST(EagleDummyNetworkTest, longestCommonPrefixLengthTest)
{
    EagleDummyNetwork network;
    EXPECT_EQ(network.longestCommonPrefixLength({1, 2, 3}, {1, 2}), 2);
    EXPECT_EQ(network.longestCommonPrefixLength({1, 2, 3}, {1, 2, 3}), 3);
    EXPECT_EQ(network.longestCommonPrefixLength({1, 2, 3}, {1, 5, 6}), 1);
    EXPECT_EQ(network.longestCommonPrefixLength({1, 2, 3}, {2, 5, 6}), 0);
    EXPECT_EQ(network.longestCommonPrefixLength({1, 2, 3}, {}), 0);
}

TEST(EagleDummyNetworkTest, pathFromDraftTokensTest)
{
    EagleDummyNetwork network;
    {
        SizeType32 const maxDecodingTokens = 5;
        SizeType32 const maxPathLen = 4;
        DraftTokensVec draftTokens = {{1, 4, 8}, {1, 5}, {2, 6, 9}, {2, 7}, {3}};
        auto const paths = network.pathFromDraftTokens(draftTokens, maxDecodingTokens, maxPathLen);
        ASSERT_EQ(paths.size(), maxDecodingTokens);
        for (SizeType32 pi = 0; pi < maxDecodingTokens; ++pi)
        {
            ASSERT_EQ(paths[pi].size(), maxPathLen);
            if (pi < draftTokens.size())
            {
                for (SizeType32 ti = 0; ti < maxPathLen; ++ti)
                {
                    if (ti == 0)
                    {
                        EXPECT_EQ(paths[pi][ti], 0);
                    }
                    else if (ti - 1 < draftTokens[pi].size())
                    {
                        EXPECT_EQ(paths[pi][ti], draftTokens[pi][ti - 1]);
                    }
                    else
                    {
                        EXPECT_EQ(paths[pi][ti], -1);
                    }
                }
            }
            else
            {
                for (SizeType32 ti = 0; ti < maxPathLen; ++ti)
                {
                    EXPECT_EQ(paths[pi][ti], -1);
                }
            }
        }
    }
}

TEST(EagleDummyNetworkTest, flattenedTokensTest)
{
    {
        EagleDummyNetwork network;
        DraftTokensVec draftTokens = {{1, 4, 8}, {1, 5}, {2, 6, 9}, {2, 7}, {3}};
        DraftPath path = {{0, 1, 4, 8}, {0, 1, 5, -1}, {0, 2, 6, 9}, {0, 2, 7, -1}, {0, 3, -1, -1}, {-1, -1, -1, -1},
            {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}};

        auto const flattenTokens = network.flattenTokens(draftTokens, path, /* isDraftTokens*/ true);
        EXPECT_EQ(flattenTokens, TokensVec({1, 2, 3, 4, 5, 6, 7, 8, 9}));
    }
    {
        EagleDummyNetwork network;
        DraftTokensVec predictionTokens = {{0, 1, 4, 8}, {0, 1, 5}, {0, 2, 6, 9}, {0, 2, 7}, {0, 3}};
        DraftPath path = {{0, 1, 4, 8}, {0, 1, 5, -1}, {0, 2, 6, 9}, {0, 2, 7, -1}, {0, 3, -1, -1}, {-1, -1, -1, -1},
            {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}};

        auto const flattenTokens = network.flattenTokens(predictionTokens, path, /* isDraftTokens*/ false);
        EXPECT_EQ(flattenTokens, TokensVec({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
    }
}

TEST(EagleDummyNetworkTest, createMasksTest)
{
    {
        EagleDummyNetwork network;
        DraftPaths paths = {{{0, 1, -1, -1}, {-1, -1, -1, -1}}};

        auto const mask = network.createMasks(paths);
        std::vector<std::vector<std::vector<bool>>> refMask = {{{true, false}, {true, true}}};
        EXPECT_EQ(mask, refMask);
    }
    {
        EagleDummyNetwork network;
        DraftPaths paths = {{{0, 1, 4, 8}, {0, 1, 5, -1}, {0, 2, 6, 9}, {0, 2, 7, -1}, {0, 3, -1, -1}, {-1, -1, -1, -1},
            {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}}};

        auto const mask = network.createMasks(paths);
        std::vector<std::vector<std::vector<bool>>> refMask
            = {{{true, false, false, false, false, false, false, false, false, false},
                {true, true, false, false, false, false, false, false, false, false},
                {true, false, true, false, false, false, false, false, false, false},
                {true, false, false, true, false, false, false, false, false, false},
                {true, true, false, false, true, false, false, false, false, false},
                {true, true, false, false, false, true, false, false, false, false},
                {true, false, true, false, false, false, true, false, false, false},
                {true, false, true, false, false, false, false, true, false, false},
                {true, true, false, false, true, false, false, false, true, false},
                {true, false, true, false, false, false, true, false, false, true}}};
        EXPECT_EQ(mask, refMask);
    }
    {
        EagleDummyNetwork network;
        DraftPaths paths = {{{0, 1, 3}, {0, 2, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}},
            {{0, 1, 3}, {0, 2, 4}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}}};

        auto const mask = network.createMasks(paths);
        std::vector<std::vector<std::vector<bool>>> refMask = {
            {{true, false, false, false, false}, {true, true, false, false, false}, {true, false, true, false, false},
                {true, true, false, true, false}, {false, false, false, false, false}},
            {{true, false, false, false, false}, {true, true, false, false, false}, {true, false, true, false, false},
                {true, true, false, true, false}, {true, false, true, false, true}}};
        EXPECT_EQ(mask, refMask);
    }
}

TEST(EagleDummyNetworkTest, acceptTokensTest)
{
    {
        EagleDummyNetwork network;
        SizeType32 const batchSize{1};
        SizeType32 const maxDecodingTokens{10};
        SizeType32 const maxPathLen{4};
        std::vector<DraftLettersVec> predictionLetters = {{"howe", "hoc", "hecl", "hea", "hu"}};
        std::vector<DraftLettersVec> lastDraftLetters = {{"how", "he", "wow", "we", "a"}};
        DraftPaths lastDraftPaths;
        DraftTokensVec lastDraftTokensFlattened;
        std::vector<TokensVec> predictionTokensFlattened;
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto const lastDraftTokens = network.draftLettersToTokens(lastDraftLetters[bi]);
            auto const lastDraftPath = network.pathFromDraftTokens(lastDraftTokens, maxDecodingTokens, maxPathLen);
            auto const predictionTokens = network.draftLettersToTokens(predictionLetters[bi]);
            lastDraftPaths.push_back(lastDraftPath);
            lastDraftTokensFlattened.push_back(
                network.flattenTokens(lastDraftTokens, lastDraftPath, /* isDraftTokens */ true));
            predictionTokensFlattened.push_back(
                network.flattenTokens(predictionTokens, lastDraftPath, /* isDraftTokens */ false));
        }

        network.acceptTokens(predictionTokensFlattened, lastDraftTokensFlattened, lastDraftPaths);

        auto acceptedLens = network.getAcceptedLens();
        auto acceptedPathIds = network.getAcceptedPathIds();
        auto outputIds = network.getOutputIds();

        ASSERT_EQ(acceptedLens.size(), 1);
        ASSERT_EQ(acceptedPathIds.size(), 1);
        ASSERT_EQ(outputIds.size(), 1);
        EXPECT_EQ(acceptedLens[0], 4);
        EXPECT_EQ(acceptedPathIds[0], 0);
        EXPECT_EQ(network.detokenize(outputIds[0]), "howe");
    }

    {
        EagleDummyNetwork network;
        SizeType32 const batchSize{2};
        SizeType32 const maxDecodingTokens{10};
        SizeType32 const maxPathLen{4};
        std::vector<DraftLettersVec> predictionLetters
            = {{"howe", "hoc", "hecl", "hea", "hu"}, {"bcde", "bcdc", "bca", "bcc", "bo"}};
        std::vector<DraftLettersVec> lastDraftLetters
            = {{"how", "he", "wow", "we", "a"}, {"inc", "inf", "ir", "im", "b"}};
        DraftPaths lastDraftPaths;
        DraftTokensVec lastDraftTokensFlattened;
        std::vector<TokensVec> predictionTokensFlattened;
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto const lastDraftTokens = network.draftLettersToTokens(lastDraftLetters[bi]);
            auto const lastDraftPath = network.pathFromDraftTokens(lastDraftTokens, maxDecodingTokens, maxPathLen);
            auto const predictionTokens = network.draftLettersToTokens(predictionLetters[bi]);
            lastDraftPaths.push_back(lastDraftPath);
            lastDraftTokensFlattened.push_back(
                network.flattenTokens(lastDraftTokens, lastDraftPath, /* isDraftTokens */ true));
            predictionTokensFlattened.push_back(
                network.flattenTokens(predictionTokens, lastDraftPath, /* isDraftTokens */ false));
        }

        network.acceptTokens(predictionTokensFlattened, lastDraftTokensFlattened, lastDraftPaths);

        auto acceptedLens = network.getAcceptedLens();
        auto acceptedPathIds = network.getAcceptedPathIds();
        auto outputIds = network.getOutputIds();

        ASSERT_EQ(acceptedLens.size(), 2);
        ASSERT_EQ(acceptedPathIds.size(), 2);
        ASSERT_EQ(outputIds.size(), 2);
        EXPECT_EQ(acceptedLens[0], 4);
        EXPECT_EQ(acceptedLens[1], 2);
        EXPECT_EQ(acceptedPathIds[0], 0);
        EXPECT_EQ(acceptedPathIds[1], 4);
        EXPECT_EQ(network.detokenize(outputIds[0]), "howe");
        EXPECT_EQ(network.detokenize(outputIds[1]), "bo");
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void EagleDecodingLayerTest<T>::SetUp()
{
    mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
    mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);
}

template <typename T>
void EagleDecodingLayerTest<T>::allocateBuffers()
{
    auto speculativeDecodingModule = std::make_shared<SpeculativeDecodingModule>(mSamplingParams.getMaxDraftPathLen(),
        mSamplingParams.getMaxDecodingDraftTokens(), mSamplingParams.getMaxDecodingTokens());
    auto const decodingDomain = tensorrt_llm::layers::DecoderDomain(mSamplingParams.getMaxBatchSize(), 1,
        mSamplingParams.getVocabSize(), mSamplingParams.getVocabSize(), speculativeDecodingModule);

    mEagleLayer = std::make_shared<tensorrt_llm::layers::EagleDecodingLayer<T>>(decodingDomain, mBufferManager);

    // outputs
    mOutputIds = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxSeqLen()}),
        nvinfer1::DataType::kINT32);

    mSeqLengths = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mOutputNextDraftTokens = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxDecodingDraftTokens()}),
        nvinfer1::DataType::kINT32);

    mOutputUnpackedNextDraftTokens = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxDecodingDraftTokens()}),
        nvinfer1::DataType::kINT32);

    mAcceptedLengths = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mNextPosIds = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxDecodingTokens()}),
        nvinfer1::DataType::kINT32);

    mPrevDraftLengths = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mNextDraftLengths = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mNextGenerationLengths
        = mBufferManager->gpu(ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mNextGenerationLengthsHost = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mAcceptedLengthCumSum = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize() + 1}), nvinfer1::DataType::kINT32);

    mPathsOffsets = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize() * mSamplingParams.getMaxDraftPathLen()}),
        nvinfer1::DataType::kINT32);

    mPackedMasks = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxDecodingTokens(),
            static_cast<SizeType32>(divUp(mSamplingParams.getMaxDecodingTokens(), 32))}),
        nvinfer1::DataType::kINT32);

    mRandomDataSample = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kFLOAT);

    mRandomDataValidation = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize(), mSamplingParams.getMaxDecodingTokens()}),
        nvinfer1::DataType::kFLOAT);

    mOutputTemperatures = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kFLOAT);

    mOutputNextDraftPaths
        = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getMaxBatchSize(),
                                        mSamplingParams.getMaxDecodingTokens(), mSamplingParams.getMaxPathLen()}),
            nvinfer1::DataType::kINT32);

    mEagleNetCtxRequestTypesHost = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mEagleNetCtxContextLengthsHost = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mEagleNetCtxPastKeyValueLengthsHost = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mEagleNetGenRequestTypesHost = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mEagleNetGenContextLengthsHost = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    mEagleNetGenPastKeyValueLengthsHost = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getMaxBatchSize()}), nvinfer1::DataType::kINT32);

    // inputs
    mBatchSlots
        = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getBatchSize()}), nvinfer1::DataType::kINT32);

    mEndIds
        = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getBatchSize()}), nvinfer1::DataType::kINT32);

    mInputNextDraftTokens = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getBatchSize(), mSamplingParams.getMaxDecodingDraftTokens()}),
        nvinfer1::DataType::kINT32);

    mInputNextDraftLens
        = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getBatchSize()}), nvinfer1::DataType::kINT32);

    mInputNextDraftPaths = BufferManager::pinnedPool(
        ITensor::makeShape(
            {mSamplingParams.getBatchSize(), mSamplingParams.getMaxDecodingTokens(), mSamplingParams.getMaxPathLen()}),
        nvinfer1::DataType::kINT32);

    mInputLastDraftTokens = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getBatchSize(), mSamplingParams.getMaxDecodingDraftTokens()}),
        nvinfer1::DataType::kINT32);

    mInputLastDraftLens
        = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getBatchSize()}), nvinfer1::DataType::kINT32);

    mInputLastDraftPaths = BufferManager::pinnedPool(
        ITensor::makeShape(
            {mSamplingParams.getBatchSize(), mSamplingParams.getMaxDecodingTokens(), mSamplingParams.getMaxPathLen()}),
        nvinfer1::DataType::kINT32);

    mInputAcceptedTokens = BufferManager::pinnedPool(
        ITensor::makeShape({mSamplingParams.getBatchSize(), mSamplingParams.getMaxPathLen()}),
        nvinfer1::DataType::kINT32);

    mInputAcceptedLens
        = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getBatchSize()}), nvinfer1::DataType::kINT32);

    mInputAcceptedPathIds
        = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getBatchSize()}), nvinfer1::DataType::kINT32);

    mChunkedContextNextTokens
        = BufferManager::pinnedPool(ITensor::makeShape({mSamplingParams.getBatchSize()}), nvinfer1::DataType::kINT32);

    mDecodingWorkspace = std::make_shared<tensorrt_llm::runtime::DecodingLayerWorkspace>(mBufferManager, decodingDomain,
        TRTDataType<float>::value, mSamplingParams.getMaxBatchSize() * sizeof(curandState_t));
}

template <typename T>
void EagleDecodingLayerTest<T>::setup()
{
    // outputs
    trk::invokeFill(*mOutputIds, TokenIdType{-1}, *mStream);
    trk::invokeFill(*mSeqLengths, SizeType32{0}, *mStream);
    trk::invokeFill(*mOutputNextDraftTokens, TokenIdType{-1}, *mStream);
    trk::invokeFill(*mOutputUnpackedNextDraftTokens, TokenIdType{-1}, *mStream);
    trk::invokeFill(*mAcceptedLengths, SizeType32{0}, *mStream);
    trk::invokeFill(*mNextPosIds, SizeType32{0}, *mStream);
    trk::invokeFill(*mPrevDraftLengths, SizeType32{0}, *mStream);
    trk::invokeFill(*mNextDraftLengths, SizeType32{0}, *mStream);
    trk::invokeFill(*mNextGenerationLengths, SizeType32{0}, *mStream);
    trk::invokeFill(*mNextGenerationLengthsHost, SizeType32{0}, *mStream);
    trk::invokeFill(*mAcceptedLengthCumSum, SizeType32{-1}, *mStream);
    trk::invokeFill(*mPathsOffsets, SizeType32{0}, *mStream);
    trk::invokeFill(*mPackedMasks, SizeType32{0}, *mStream);
    trk::invokeFill(*mEndIds, TokenIdType{-1}, *mStream);
    trk::invokeFill(*mRandomDataSample, float{0}, *mStream);
    trk::invokeFill(*mRandomDataValidation, float{0}, *mStream);
    trk::invokeFill(*mOutputTemperatures, float{0}, *mStream);
    trk::invokeFill(*mOutputNextDraftPaths, SizeType32{0}, *mStream);
    trk::invokeFill(*mChunkedContextNextTokens, SizeType32{-1}, *mStream);

    std::mt19937 gen(42);

    auto batchSlotsPtr = bufferCast<SizeType32>(*mBatchSlots);
    for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
    {
        batchSlotsPtr[bi] = 2 * bi;
    }

    auto setupParams = std::make_shared<EagleSetupParams>();
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

    mDecodingWorkspace->setDeviceBatchSlots(mBatchSlots);
    mEagleLayer->setup(mSamplingParams.getBatchSize(), 1, mBatchSlots, setupParams, mDecodingWorkspace);

    mStream->synchronize();

    mInputAcceptedLens = mBufferManager->copyFrom(mNetwork.getAcceptedLens(),
        ITensor::makeShape({mSamplingParams.getBatchSize()}), runtime::MemoryType::kPINNEDPOOL);
    mInputAcceptedPathIds = mBufferManager->copyFrom(mNetwork.getAcceptedPathIds(),
        ITensor::makeShape({mSamplingParams.getBatchSize()}), runtime::MemoryType::kPINNEDPOOL);

    auto const nextDraftTokens = mNetwork.getNextDraftTokens();
    auto const lastDraftTokens = mNetwork.getLastDraftTokens();
    auto const nextDraftPaths = mNetwork.getNextDraftPaths();
    auto const lastDraftPaths = mNetwork.getLastDraftPaths();
    auto const nextDraftLens = mNetwork.getNextDraftLens();
    auto const lastDraftLens = mNetwork.getLastDraftLens();
    auto const acceptedTokens = mNetwork.getAcceptedTokens();
    auto sequenceLength = BufferRange<SizeType32>(*mSeqLengths);
    auto inputNextDraftTokensRange = BufferRange<TokenIdType>(*mInputNextDraftTokens);
    auto inputLastDraftTokensRange = BufferRange<TokenIdType>(*mInputLastDraftTokens);
    auto inputNextDraftPathsRange = BufferRange<SizeType32>(*mInputNextDraftPaths);
    auto inputLastDraftPathsRange = BufferRange<SizeType32>(*mInputLastDraftPaths);
    auto inputNextDraftLensRange = BufferRange<SizeType32>(*mInputNextDraftLens);
    auto inputLastDraftLensRange = BufferRange<SizeType32>(*mInputLastDraftLens);
    auto inputAcceptedTokensRange = BufferRange<SizeType32>(*mInputAcceptedTokens);

    auto outputIds = BufferRange<TokenIdType>(*mOutputIds);
    auto prompts = mNetwork.getPrompts();
    for (SizeType32 bi = 0; bi < nextDraftTokens.size(); ++bi)
    {
        for (SizeType32 ti = 0; ti < nextDraftTokens[bi].size(); ++ti)
        {
            auto idx = flat_index2(bi, ti, mSamplingParams.getMaxDecodingDraftTokens());
            inputNextDraftTokensRange[idx] = nextDraftTokens[bi][ti];
        }
        for (SizeType32 ti = 0; ti < lastDraftTokens[bi].size(); ++ti)
        {
            auto idx = flat_index2(bi, ti, mSamplingParams.getMaxDecodingDraftTokens());
            inputLastDraftTokensRange[idx] = lastDraftTokens[bi][ti];
        }
        for (SizeType32 pi = 0; pi < nextDraftPaths[bi].size(); ++pi)
        {
            for (SizeType32 ti = 0; ti < nextDraftPaths[bi][pi].size(); ++ti)
            {
                auto idx
                    = flat_index3(bi, pi, ti, mSamplingParams.getMaxDecodingTokens(), mSamplingParams.getMaxPathLen());
                inputNextDraftPathsRange[idx] = nextDraftPaths[bi][pi][ti];
            }
        }
        for (SizeType32 pi = 0; pi < lastDraftPaths[bi].size(); ++pi)
        {
            for (SizeType32 ti = 0; ti < lastDraftPaths[bi][pi].size(); ++ti)
            {
                auto idx
                    = flat_index3(bi, pi, ti, mSamplingParams.getMaxDecodingTokens(), mSamplingParams.getMaxPathLen());
                inputLastDraftPathsRange[idx] = lastDraftPaths[bi][pi][ti];
            }
        }
        inputNextDraftLensRange[bi] = nextDraftLens[bi];
        inputLastDraftLensRange[bi] = lastDraftLens[bi];
        sequenceLength[batchSlotsPtr[bi]] = prompts[bi].size();
    }
    for (SizeType32 bi = 0; bi < acceptedTokens.size(); ++bi)
    {
        for (SizeType32 ti = 0; ti < acceptedTokens[bi].size(); ++ti)
        {
            auto idx = flat_index2(bi, ti, mSamplingParams.getMaxPathLen());
            inputAcceptedTokensRange[idx] = acceptedTokens[bi][ti];
        }
    }
}

template <typename T>
std::shared_ptr<EagleInputs> EagleDecodingLayerTest<T>::createInputTensors()
{
    auto forwardParams
        = std::make_shared<EagleInputs>(mEndIds, mBatchSlots, mSamplingParams.getBatchSize(), mInputNextDraftTokens,
            mInputNextDraftLens, mInputNextDraftPaths, mInputLastDraftTokens, mInputLastDraftLens, mInputLastDraftPaths,
            mInputAcceptedTokens, mInputAcceptedLens, mInputAcceptedPathIds, mChunkedContextNextTokens, mBatchSlots);

    return forwardParams;
}

template <typename T>
std::shared_ptr<EagleOutputs> EagleDecodingLayerTest<T>::createOutputTensors()
{
    auto outputParams = std::make_shared<EagleOutputs>(mOutputIds);

    outputParams->sequenceLength = mSeqLengths;

    outputParams->unpackedNextDraftTokens = mOutputUnpackedNextDraftTokens;

    outputParams->nextDraftTokens = mOutputNextDraftTokens;

    outputParams->numNewTokens = mAcceptedLengths;

    outputParams->nextDraftPosIds = mNextPosIds;

    outputParams->prevDraftLengths = mPrevDraftLengths;

    outputParams->nextDraftLengths = mNextDraftLengths;

    outputParams->generationLengths = mNextGenerationLengths;

    outputParams->generationLengthsHost = mNextGenerationLengthsHost;

    outputParams->numNewTokensCumSum = mAcceptedLengthCumSum;

    outputParams->pathsOffsets = mPathsOffsets;

    outputParams->packedMasks = mPackedMasks;

    outputParams->randomDataSample = mRandomDataSample;

    outputParams->randomDataValidation = mRandomDataValidation;

    outputParams->temperatures = mOutputTemperatures;

    outputParams->nextDraftPaths = mOutputNextDraftPaths;

    outputParams->eagleNetCtxRequestTypesHost = mEagleNetCtxRequestTypesHost;

    outputParams->eagleNetCtxContextLengthsHost = mEagleNetCtxContextLengthsHost;

    outputParams->eagleNetCtxPastKeyValueLengthsHost = mEagleNetCtxPastKeyValueLengthsHost;

    outputParams->eagleNetGenRequestTypesHost = mEagleNetGenRequestTypesHost;

    outputParams->eagleNetGenContextLengthsHost = mEagleNetGenContextLengthsHost;

    outputParams->eagleNetGenPastKeyValueLengthsHost = mEagleNetGenPastKeyValueLengthsHost;

    return outputParams;
}

std::vector<int32_t> boolArrayToBitmask(std::vector<bool>::iterator boolIterator, size_t pathLen)
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
void EagleDecodingLayerTest<T>::checkLayerResult()
{
    auto const batchSlots = BufferRange<SizeType32>(*mBatchSlots);

    // Check generated random data
    {
        auto const randomDataSample = BufferRange<float>(*mRandomDataSample);
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            auto const batchSlot = batchSlots[bi];
            // Check that all fields are filled with non zero data
            EXPECT_NE(randomDataSample[batchSlot], float{0}) << " bi: " << bi;
        }
    }

    // Check masks
    {
        auto const randomDataValidation = BufferRange<float>(*mRandomDataValidation);
        auto const packedMasks = BufferRange<int32_t>(*mPackedMasks);
        auto masks = mNetwork.getNextMasks();
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            for (SizeType32 ti = 0; ti < mSamplingParams.getMaxDecodingTokens(); ++ti)
            {
                auto const batchSlot = batchSlots[bi];
                auto const bitmask = boolArrayToBitmask(masks[bi][ti].begin(), mSamplingParams.getMaxDecodingTokens());

                EXPECT_NE(randomDataValidation[batchSlot * mSamplingParams.getMaxDecodingTokens() + ti], float{0})
                    << " bi: " << bi;

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
    auto const acceptedLengths = BufferRange<SizeType32>(*mAcceptedLengths);
    auto const inputAcceptedLens = BufferRange<SizeType32>(*mInputAcceptedLens);
    for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
    {
        auto const batchSlot = batchSlots[bi];
        // Check accepted length.
        EXPECT_EQ(inputAcceptedLens[bi], acceptedLengths[batchSlot]) << " bi:" << bi;
        // Updated seq length is prompt length and newly accepted tokens.
        EXPECT_EQ(seqLenghts[batchSlot], promptIds[bi].size() + acceptedLengths[batchSlot]) << " bi: " << bi;
        // Check that output ids contains accepted tokens.
        for (SizeType32 ti = promptIds[bi].size(); ti < acceptedLengths[batchSlot]; ++ti)
        {
            EXPECT_EQ(outputIds[batchSlot * mSamplingParams.getMaxSeqLen() + ti], refOutputIds[bi][ti])
                << " bi: " << bi << " ti: " << ti;
        }
    }

    // Check new draft tokens
    {
        auto const outputNextDraftTokens = BufferRange<TokenIdType>(*mOutputNextDraftTokens);
        auto const outputUnpackedNextDraftTokens = BufferRange<TokenIdType>(*mOutputUnpackedNextDraftTokens);
        auto const nextDraftLens = mNetwork.getNextDraftLens();
        auto const prevDraftLens = mNetwork.getLastDraftLens();
        auto const nextDraftTokens = mNetwork.getNextDraftTokens();
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            auto const batchSlot = batchSlots[bi];
            auto const nextDraftLen = nextDraftLens[bi];
            auto const prevDraftLen = prevDraftLens[bi];
            // Check draft tokens for the next iteration.
            for (SizeType32 ti = 0; ti < nextDraftLen; ++ti)
            {
                auto const idx = flat_index2(batchSlot, ti, mSamplingParams.getMaxDecodingDraftTokens());
                EXPECT_EQ(outputNextDraftTokens[idx], nextDraftTokens[bi][ti]) << " bi: " << bi << " ti: " << ti;
                EXPECT_EQ(outputUnpackedNextDraftTokens[idx], nextDraftTokens[bi][ti])
                    << " bi: " << bi << " ti: " << ti;
            }
            // Check length of the draft tokens.
            EXPECT_EQ(BufferRange<SizeType32>(*mNextGenerationLengthsHost)[batchSlot], nextDraftLen + 1)
                << " bi: " << bi;
            EXPECT_EQ(BufferRange<SizeType32>(*mNextDraftLengths)[batchSlot], nextDraftLen) << " bi: " << bi;
            EXPECT_EQ(BufferRange<SizeType32>(*mPrevDraftLengths)[batchSlot], prevDraftLen) << " bi: " << bi;

            for (SizeType32 pi = 0; pi < mSamplingParams.getMaxDecodingTokens(); ++pi)
            {
                for (SizeType32 ti = 0; ti < mSamplingParams.getMaxPathLen(); ++ti)
                {
                    auto const idx = flat_index3(
                        bi, pi, ti, mSamplingParams.getMaxDecodingTokens(), mSamplingParams.getMaxPathLen());
                    auto const idxSlot = flat_index3(
                        batchSlot, pi, ti, mSamplingParams.getMaxDecodingTokens(), mSamplingParams.getMaxPathLen());
                    EXPECT_EQ(BufferRange<SizeType32>(*mOutputNextDraftPaths)[idxSlot],
                        BufferRange<SizeType32>(*mInputNextDraftPaths)[idx])
                        << " bi: " << bi << " pi:" << pi << " ti: " << ti;
                }
            }
        }
    }

    // Check position ids
    {
        auto const nextPosIds = BufferRange<SizeType32>(*mNextPosIds);
        auto const nextDraftPaths = mNetwork.getNextDraftPaths();
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            auto const batchSlot = batchSlots[bi];
            // Check pos ids for the next iteration.
            for (SizeType32 pi = 0; pi < mSamplingParams.getMaxDecodingTokens(); ++pi)
            {
                for (SizeType32 li = 0; li < mSamplingParams.getMaxPathLen(); ++li)
                {
                    auto const pathIdx = nextDraftPaths[bi][pi][li];
                    auto const idx = flat_index2(batchSlot, pathIdx, mSamplingParams.getMaxDecodingTokens());
                    if (pathIdx != -1)
                    {
                        EXPECT_EQ(nextPosIds[idx], li) << " bi: " << bi << " pi: " << pi << " li: " << li;
                    }
                }
            }
        }
    }

    // Check accumulated cum sum and paths offsets
    {
        auto const accumulatedCumSum = BufferRange<SizeType32>(*mAcceptedLengthCumSum);
        auto const pathsOffsets = BufferRange<SizeType32>(*mPathsOffsets);
        auto const acceptedLengths = BufferRange<SizeType32>(*mAcceptedLengths);
        auto const inputAcceptedPathIds = BufferRange<SizeType32>(*mInputAcceptedPathIds);
        auto const lastDraftPaths = mNetwork.getLastDraftPaths();
        SizeType32 sum = 0;
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            auto const batchSlot = batchSlots[bi];
            EXPECT_EQ(sum, accumulatedCumSum[bi]) << "bi: " << bi;
            auto const acceptedLength = acceptedLengths[batchSlot] - 1;
            for (SizeType32 ti = 0; ti < acceptedLength; ++ti)
            {
                EXPECT_EQ(pathsOffsets[sum + ti], lastDraftPaths[bi][inputAcceptedPathIds[bi]][ti + 1] - 1)
                    << "bi: " << bi << " ti: " << ti;
            }
            sum += acceptedLength;
        }
        EXPECT_EQ(sum, accumulatedCumSum[mSamplingParams.getBatchSize()]);
    }

    // Check temperature
    {
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            auto const batchSlot = batchSlots[bi];
            EXPECT_EQ(BufferRange<float>(*mOutputTemperatures)[batchSlot], static_cast<float>(mTemperatures[bi]))
                << " bi: " << bi;
        }
    }

    // Check EagleNet host buffers
    {
        for (SizeType32 bi = 0; bi < mSamplingParams.getBatchSize(); ++bi)
        {
            auto const batchSlot = batchSlots[bi];
            EXPECT_EQ(BufferRange<SizeType32>(*mEagleNetCtxRequestTypesHost)[batchSlot], 0) << " bi: " << bi;
            EXPECT_EQ(BufferRange<SizeType32>(*mEagleNetGenRequestTypesHost)[batchSlot], 1) << " bi: " << bi;

            EXPECT_EQ(
                BufferRange<SizeType32>(*mEagleNetCtxContextLengthsHost)[batchSlot], mSamplingParams.getMaxPathLen())
                << " bi: " << bi;
            EXPECT_EQ(BufferRange<SizeType32>(*mEagleNetGenContextLengthsHost)[batchSlot],
                seqLenghts[batchSlot] + mSamplingParams.getMaxPathLen())
                << " bi: " << bi;

            EXPECT_EQ(BufferRange<SizeType32>(*mEagleNetCtxPastKeyValueLengthsHost)[batchSlot],
                seqLenghts[batchSlot] + mSamplingParams.getMaxPathLen())
                << " bi: " << bi;
            EXPECT_EQ(BufferRange<SizeType32>(*mEagleNetGenPastKeyValueLengthsHost)[batchSlot],
                seqLenghts[batchSlot] + mSamplingParams.getMaxPathLen() - 1)
                << " bi: " << bi;
        }
    }
}

template <typename T>
void EagleDecodingLayerTest<T>::runTest(std::vector<std::string> const& prompts,
    std::vector<DraftLettersVec> const& predictions, std::vector<DraftLettersVec> const& nextDraftLetters,
    std::vector<DraftLettersVec> const& lastDraftLetters, SamplingParams& params)
{
    mSamplingParams = params;

    mNetwork.forward(params, prompts, predictions, nextDraftLetters, lastDraftLetters);

    allocateBuffers();

    setup();

    auto inputTensors = createInputTensors();
    auto outputTensors = createOutputTensors();

    mDecodingWorkspace->setDeviceBatchSlots(mBatchSlots);
    mEagleLayer->forwardAsync(outputTensors, inputTensors, mDecodingWorkspace);

    mStream->synchronize();

    checkLayerResult();
}

TYPED_TEST_SUITE(EagleDecodingLayerTest, FloatAndHalfTypes);

TYPED_TEST(EagleDecodingLayerTest, IOSamePathsBs1)
{
    SamplingParams params;

    params.setBatchSize(1);
    params.setMaxPathLen(4);
    params.setMaxDecodingTokens(10);

    std::vector<std::string> prompts = {"Hi mate, "};
    std::vector<DraftLettersVec> predictionLetters = {{"how ", "hoc", "hecl", "hea", "hu"}};
    std::vector<DraftLettersVec> lastDraftLetters = {{"how", "he", "wow", "we", "a"}};
    std::vector<DraftLettersVec> nextDraftLetters = {{"are", "ap", "cre", "co", "i"}};

    this->runTest(prompts, predictionLetters, nextDraftLetters, lastDraftLetters, params);
}

TYPED_TEST(EagleDecodingLayerTest, IODifferentPathsBs1)
{
    SamplingParams params;

    params.setBatchSize(1);
    params.setMaxPathLen(4);
    params.setMaxDecodingTokens(10);

    std::vector<std::string> prompts = {"Hi mate, "};
    std::vector<DraftLettersVec> predictionLetters = {{"how ", "hoc", "hecl", "hea", "hu"}};
    std::vector<DraftLettersVec> lastDraftLetters = {{"how", "he", "wow", "we", "a"}};
    std::vector<DraftLettersVec> nextDraftLetters = {{"are", "is", "imp", "do"}};

    this->runTest(prompts, predictionLetters, nextDraftLetters, lastDraftLetters, params);
}

TYPED_TEST(EagleDecodingLayerTest, IODifferentPathsNoDraftAcceptedBs1)
{
    SamplingParams params;

    params.setBatchSize(1);
    params.setMaxPathLen(4);
    params.setMaxDecodingTokens(10);

    std::vector<std::string> prompts = {"Hi mate, "};
    std::vector<DraftLettersVec> predictionLetters = {{"how ", "hoc", "hecl", "hea", "hu"}};
    std::vector<DraftLettersVec> lastDraftLetters = {{"my", "I'd", "wow", "we", "a"}};
    std::vector<DraftLettersVec> nextDraftLetters = {{"are", "ap", "cre", "co", "i"}};

    this->runTest(prompts, predictionLetters, nextDraftLetters, lastDraftLetters, params);
}

TYPED_TEST(EagleDecodingLayerTest, IODifferentPathsBs2)
{
    SamplingParams params;

    params.setBatchSize(2);
    params.setMaxPathLen(4);
    params.setMaxDecodingTokens(10);

    std::vector<std::string> prompts = {"Hi mate, ", "Let's go "};
    std::vector<DraftLettersVec> predictionLetters
        = {{"how ", "hoc", "hecl", "hea", "hu"}, {"bcde", "bcdc", "bca", "bcc", "bo"}};
    std::vector<DraftLettersVec> lastDraftLetters = {{"how", "he", "wow", "we", "a"}, {"inc", "inf", "ir", "im", "b"}};
    std::vector<DraftLettersVec> nextDraftLetters = {{"are", "is", "imp", "do"}, {"wli", "mbi", "ard"}};

    this->runTest(prompts, predictionLetters, nextDraftLetters, lastDraftLetters, params);
}

} // namespace tensorrt_llm::tests::layers
