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

#include "tensorrt_llm/layers/lookaheadAlgorithm.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <cstddef>
#include <memory>
#include <tuple>

namespace tensorrt_llm::layers
{

using namespace tensorrt_llm::runtime;

LookaheadAlgorithm::LookaheadAlgorithm(
    runtime::SizeType32 maxW, runtime::SizeType32 maxN, runtime::SizeType32 maxG, runtime::SizeType32 id)
    : mPoolManager(maxG)
    , mPrefillsMax(runtime::BufferManager::cpu(
          runtime::ITensor::makeShape({(maxN <= 1 ? 0 : maxN - 2)}), nvinfer1::DataType::kINT32))
    , mPastTokensMax(
          runtime::BufferManager::cpu(runtime::ITensor::makeShape({maxW * (maxN - 1)}), nvinfer1::DataType::kINT32))
    , mKeyTokensMax(runtime::BufferManager::cpu(runtime::ITensor::makeShape({maxW}), nvinfer1::DataType::kINT32))
    , mGoldenTokensMax(
          runtime::BufferManager::cpu(runtime::ITensor::makeShape({maxN * 2 - 1}), nvinfer1::DataType::kINT32))
    , mGuessTokensMax(
          runtime::BufferManager::cpu(runtime::ITensor::makeShape({maxG * (maxN - 1)}), nvinfer1::DataType::kINT32))
    , mMaxW(maxW)
    , mMaxN(maxN)
    , mMaxG(maxG)
    , mFilling(0)
{
    runtime::SizeType32 maxGeneratedLen, maxDraftLen;
    std::tie(maxGeneratedLen, std::ignore, maxDraftLen, std::ignore)
        = executor::LookaheadDecodingConfig(maxW, maxN, maxG).calculateSpeculativeResource();
    mAttentionMask = runtime::BufferManager::cpu(
        runtime::ITensor::makeShape({maxDraftLen, maxDraftLen}), nvinfer1::DataType::kBOOL);
    mDraftTokensMax
        = runtime::BufferManager::cpu(runtime::ITensor::makeShape({maxDraftLen}), nvinfer1::DataType::kINT32);
    mSampledTokensMax
        = runtime::BufferManager::cpu(runtime::ITensor::makeShape({maxGeneratedLen}), nvinfer1::DataType::kINT32);
    mEncodeMapMax = runtime::BufferManager::cpu(runtime::ITensor::makeShape({maxDraftLen}), nvinfer1::DataType::kINT32);
}

void LookaheadAlgorithm::setup(TensorConstPtr const& prompt, SizeType32 w, SizeType32 n, SizeType32 g, uint64_t seed)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK_WITH_INFO(w <= mMaxW, "lookahead requires setup w (%d) <= max_w (%d)", w, mMaxW);
    TLLM_CHECK_WITH_INFO(n <= mMaxN, "lookahead requires setup n (%d) <= max_n (%d)", n, mMaxN);
    TLLM_CHECK_WITH_INFO(g <= mMaxG, "lookahead requires setup g (%d) <= max_g (%d)", g, mMaxG);
    mW = w;
    mN = n;
    mG = g;
    std::tie(std::ignore, std::ignore, mRuntimeMaxDraftLen, mRuntimeMaxDraftPathLen)
        = executor::LookaheadDecodingConfig(mW, mN, mG).calculateSpeculativeResource();

    mPoolManager.setup(mG);
    mPoolManager.accept(prompt, mN);
    mGoldenTokens = ITensor::slice(mGoldenTokensMax, 0, mN * 2 - 1);
    mPrefills = ITensor::slice(mPrefillsMax, 0, mN <= 1 ? 0 : mN - 2);
    mKeyTokens = ITensor::slice(mKeyTokensMax, 0, mW);
    mPastTokens = ITensor::slice(mPastTokensMax, 0, mW * (mN - 1));
    mPastTokens->reshape(ITensor::makeShape({mW, mN - 1}));

    BufferRange<TokenIdType const> promptRange(*prompt);
    BufferRange<TokenIdType> prefillRange(*mPrefills);
    BufferRange<TokenIdType> pastRange(*mPastTokens);
    BufferRange<TokenIdType> goldRange(*mGoldenTokens);

    srand(seed);

    auto randToken = [&promptRange](auto& item) { item = promptRange[rand() % promptRange.size()]; };
    std::for_each(prefillRange.begin(), prefillRange.end(), randToken);
    std::for_each(pastRange.begin(), pastRange.end(), [](auto& a) { a = -1; });
    for (SizeType32 i = 0; i < mW; i++)
    {
        if (mN - 1 > 0)
        {
            randToken(pastRange[i * (mN - 1)]);
        }
    }
    std::copy(std::prev(promptRange.end(), mN - 1), promptRange.end(), goldRange.begin());
    mGuessTokens = ITensor::slice(mGuessTokensMax, 0, 0);
    mFilling = (mN - 1) > 0 ? 1 : 0;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadAlgorithm::accept(TensorConstPtr const& generatedTokens)
{
    TLLM_CHECK(ITensor::volume(generatedTokens->getShape()) <= mN);
    BufferRange<TokenIdType const> generatedRange(*generatedTokens);
    BufferRange<TokenIdType> goldRange(*mGoldenTokens);
    auto genLen = generatedTokens->getShape().d[0];
    TLLM_CHECK(genLen <= mN);
    std::copy(generatedRange.begin(), generatedRange.end(), goldRange.begin() + mN - 1);
    TensorPtr newGold = ITensor::slice(mGoldenTokens, 0, mN - 1 + genLen);
    mPoolManager.accept(newGold, mN);
    std::copy(goldRange.begin() + genLen, goldRange.begin() + genLen + mN - 1, goldRange.begin());
}

//! lookahead has two phase, prefill the past tokens matrix and maintain past tokens matrix.
runtime::SizeType32 LookaheadAlgorithm::lookahead(
    TensorPtr const& draftTokens, TensorPtr const& positionIds, runtime::SizeType32 startPosId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    SizeType32 prefill = mN - 2 - mFilling;
    SizeType32 len = prefill + mFilling * mW;
    TLLM_CHECK(len <= ITensor::volume(draftTokens->getShape()));
    TLLM_CHECK(len <= ITensor::volume(positionIds->getShape()));
    BufferRange<TokenIdType> prefillRange(*mPrefills);
    BufferRange<TokenIdType> pastRange(*mPastTokens);
    BufferRange<TokenIdType> draftRange(*draftTokens);
    PRINT_TOKENS(mPrefills);

    if (mFilling < mN - 1)
    { // prefilling
        std::copy(prefillRange.begin() + mFilling, prefillRange.end(), draftRange.begin());
        for (SizeType32 i = 0; i < mW; i++)
        {
            auto start = pastRange.begin() + i * (mN - 1);
            auto end = pastRange.begin() + i * (mN - 1) + mFilling;
            std::copy(start, end, draftRange.begin() + prefill + i * mFilling);
        }
    }
    else
    { // shift up
        std::copy(pastRange.begin() + 1, pastRange.begin() + mFilling * mW, draftRange.begin());
    }

    BufferRange<TokenIdType> positionIdsRange(*positionIds);
    SizeType32 idx = 0, wj = 0;
    auto fillPosition = [&positionIdsRange, &idx](SizeType32 start, SizeType32 len)
    {
        for (SizeType32 i = start; i < start + len; i++)
        {
            positionIdsRange[idx++] = i;
        }
    };
    if (prefill >= 0)
    {
        fillPosition(startPosId, prefill);
        for (wj = 0; wj < mW; wj++)
        {
            fillPosition(startPosId + prefill + wj, mFilling);
        }
    }
    else
    {
        fillPosition(startPosId, mFilling - 1);
        for (wj = 1; wj < mW; wj++)
        {
            fillPosition(startPosId - 1 + wj, mFilling);
        }
    }
    PRINT_VALUES(positionIds);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return len;
}

runtime::SizeType32 LookaheadAlgorithm::guess(TensorPtr const& guessTokens, TensorPtr const& guessIds,
    runtime::SizeType32 startPosId, runtime::TokenIdType lastToken)
{
    auto guesses = mPoolManager.guess(lastToken, mW);

    SizeType32 len = 0;
    std::for_each(guesses.begin(), guesses.end(), [&len](auto& a) { len += ITensor::volume(a->getShape()); });
    TLLM_CHECK(len <= ITensor::volume(guessTokens->getShape()));
    TLLM_CHECK(len <= ITensor::volume(guessIds->getShape()));
    BufferRange<TokenIdType> guessTokensRange(*guessTokens);
    BufferRange<SizeType32> guessIdsRange(*guessIds);

    SizeType32 cur = 0;
    for (auto guess : guesses)
    {
        BufferRange<TokenIdType const> guessRange(*guess);
        std::copy(guessRange.begin(), guessRange.end(), guessTokensRange.begin() + cur);
        SizeType32 tmp = startPosId;
        std::for_each(
            guessIdsRange.begin() + cur, guessIdsRange.begin() + cur + mN - 1, [&tmp](auto& v) { v = tmp++; });
        cur += ITensor::volume(guess->getShape());
    }

    return len;
}

void LookaheadAlgorithm::posIdsToMask(TensorPtr const& mask, TensorConstPtr const& posIds)
{
    auto len = ITensor::volume(posIds->getShape());
    TLLM_CHECK(mask->getDimension<0>() >= len);
    TLLM_CHECK(mask->getDimension<1>() >= len);
    auto posIdsRange = BufferRange<SizeType32 const>(*posIds);
    auto maskLocation = BufferLocation<bool>(*mask);

    for (auto& item : maskLocation)
    {
        item = false;
    }

    if (len > 0)
    {
        std::vector<std::pair<SizeType32, SizeType32>> stack;
        for (auto i = 0; i < len; i++)
        {
            auto cur = posIdsRange[i];
            while (stack.size() > 0 && cur <= stack.back().second)
            {
                stack.pop_back();
            }
            TLLM_CHECK(stack.size() > 0 ? cur == stack.back().second + 1 : true);
            stack.push_back(std::make_pair(i, cur));
            for (auto prev : stack)
            {
                maskLocation.at(i, prev.first) = true;
            }
        }
    }
}

struct TreeValue;
using TreeMap = std::unordered_map<TokenIdType, TreeValue>;

struct TreeValue
{
    TreeValue()
        : nexts(std::make_shared<TreeMap>())
    {
    }

    using Nexts = std::shared_ptr<TreeMap>;
    Nexts nexts{nullptr};
    std::list<SizeType32> sources;
};

using TreeNode = TreeMap::value_type;

template <typename BF, typename AF>
void treeDFS(TreeNode& node, BF const& visitBefore, AF const& visitAfter)
{
    visitBefore(node);
    for (auto& next : *(node.second.nexts))
    {
        treeDFS(next, visitBefore, visitAfter);
    }
    visitAfter(node);
}

SizeType32 LookaheadAlgorithm::treeEncode(
    TensorPtr const& tokens, TensorPtr const& posIds, TensorPtr const& mask, TensorPtr const& encodeMap)
{
    TLLM_CHECK(ITensor::volume(tokens->getShape()) == ITensor::volume(posIds->getShape()));
    auto len = ITensor::volume(tokens->getShape());

    BufferRange<TokenIdType> tokensRange(*tokens);
    BufferRange<SizeType32> posIdsRange(*posIds);
    BufferLocation<bool> maskLocation(*mask);
    BufferRange<SizeType32> mapRange(*encodeMap);

    auto branches = std::make_shared<TreeMap>();

    for (auto i = 0; i < len; i++)
    {
        auto nexts = branches;
        for (auto j = 0; j <= i; j++)
        {
            if (maskLocation.at(i, j))
            {
                auto tok = tokensRange[j];
                auto found = nexts->find(tok);
                if (found != nexts->end())
                {
                    found->second.sources.push_back(j);
                    nexts = found->second.nexts;
                }
                else
                {
                    auto [inserted, ok] = nexts->insert({tok, TreeValue()});
                    inserted->second.sources.push_back(j);
                    nexts = inserted->second.nexts;
                }
            }
        }
    }

    for (auto& item : maskLocation)
    {
        item = 0;
    }
    std::vector<std::pair<SizeType32, TokenIdType>> stack;
    SizeType32 offset = 0;
    SizeType32 posId = posIdsRange.size() ? posIdsRange[0] : 0;

    auto visitBefore
        = [&stack, &maskLocation, &tokensRange, &posIdsRange, &posId, &offset, &mapRange](TreeNode const& node)
    {
        stack.push_back(std::make_pair(offset, node.first));
        for (auto const& source : node.second.sources)
        {
            mapRange[source] = offset;
        }
        for (auto const& prev : stack)
        {
            maskLocation.at(offset, prev.first) = true;
        }
        tokensRange[offset] = node.first;
        posIdsRange[offset] = posId;
        offset++;
        posId++;
    };
    auto visitAfter = [&stack, &posId](TreeNode const& node)
    {
        stack.pop_back();
        posId--;
    };

    for (auto& next : *branches)
    {
        treeDFS(next, visitBefore, visitAfter);
    }

    for (SizeType32 i = offset; i < len; i++)
    {
        tokensRange[i] = 0;
        posIdsRange[i] = 0;
    }
    for (SizeType32 i = 0; i < len; i++)
    {
        for (SizeType32 j = i < offset ? offset : 0; j < len; j++)
        {
            maskLocation.at(i, j) = false;
        }
    }

    return offset;
}

void LookaheadAlgorithm::prepare(TensorPtr const& draftTokens, TensorPtr const& positionIds,
    TensorPtr const& draftLengthPtr, TensorPtr const& attentionMask, SizeType32 attentionMaskOffset,
    TensorConstPtr const& lastPositionIdPtr, TensorConstPtr const& lastTokenPtr)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mRuntimeMaxDraftLen == 0)
    {
        mDraftTokens = ITensor::slice(mDraftTokensMax, 0, 0);
        mEncodeMap = ITensor::slice(mEncodeMapMax, 0, 0);
        (BufferRange<SizeType32>(*draftLengthPtr))[0] = 0;
        return;
    }

    auto lastToken = BufferRange<TokenIdType const>(*lastTokenPtr)[0];
    auto offset = BufferRange<SizeType32 const>(*lastPositionIdPtr)[0];

    SizeType32 inputLen = ITensor::volume(draftTokens->getShape());
    TLLM_CHECK(inputLen >= mRuntimeMaxDraftLen);

    BufferRange<TokenIdType> draftRange(*draftTokens);
    BufferRange<TokenIdType> positionRange(*positionIds);

    SizeType32 filledLen = 0;

    filledLen += lookahead(ITensor::slice(draftTokens, filledLen, mRuntimeMaxDraftLen - filledLen),
        ITensor::slice(positionIds, filledLen, mRuntimeMaxDraftLen - filledLen), offset);

    auto guessStart = filledLen;
    filledLen += guess(ITensor::slice(draftTokens, filledLen, mRuntimeMaxDraftLen - filledLen),
        ITensor::slice(positionIds, filledLen, mRuntimeMaxDraftLen - filledLen), offset, lastToken);
    auto guessEnd = filledLen;

    std::copy(draftRange.begin() + guessStart, draftRange.begin() + guessEnd,
        BufferRange<TokenIdType>(*mGuessTokensMax).begin());
    mGuessTokens = ITensor::slice(mGuessTokensMax, 0, guessEnd - guessStart);

    posIdsToMask(mAttentionMask, ITensor::slice(positionIds, 0, filledLen));

    auto draftLen = treeEncode(ITensor::slice(draftTokens, 0, filledLen), ITensor::slice(positionIds, 0, filledLen),
        mAttentionMask, mEncodeMapMax);

    for (SizeType32 i = 0; i < draftLen; i++)
    {
        BufferRange<bool> srcRange(*ITensor::at(mAttentionMask, {i}));
        BufferRange<bool> dstRange(*ITensor::slice(attentionMask, {i + attentionMaskOffset, attentionMaskOffset}));
        std::copy(srcRange.begin(), srcRange.end(), dstRange.begin());
    }

    std::copy(draftRange.begin(), draftRange.begin() + draftLen, BufferRange<TokenIdType>(*mDraftTokensMax).begin());
    mDraftTokens = ITensor::slice(mDraftTokensMax, 0, draftLen);
    (BufferRange<SizeType32>(*draftLengthPtr))[0] = draftLen;
    mEncodeMap = ITensor::slice(mEncodeMapMax, 0, filledLen);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadAlgorithm::verify(TensorPtr const& accepted, TensorPtr const& acceptedOffsets,
    TensorPtr const& acceptedLength, TokenIdType newLastToken, TensorConstPtr const& goldenTokens,
    TensorConstPtr const& endToken)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(ITensor::volume(goldenTokens->getShape()) == ITensor::volume(mDraftTokens->getShape()));
    BufferRange<TokenIdType const> goldRange(*goldenTokens);
    BufferRange<TokenIdType> draftRange(*mDraftTokens);
    BufferLocation<bool const> maskLocation(*mAttentionMask);
    auto draftSize = ITensor::volume(mDraftTokens->getShape());
    auto end = *BufferRange<TokenIdType const>(*endToken).begin();

    SizeType32 maxHit = 0, hitIdx = 0;
    for (SizeType32 i = 0; i < draftSize; i++)
    {
        SizeType32 hit = 0;
        TokenIdType cur = newLastToken;
        for (SizeType32 j = 0; j < draftSize; j++)
        {
            if (maskLocation.at(i, j))
            {
                if (draftRange[j] == cur && draftRange[j] != end)
                {
                    hit++;
                    cur = goldRange[j];
                }
                else
                {
                    break;
                }
            }
        }
        if (hit > maxHit)
        {
            maxHit = hit;
            hitIdx = i;
        }
    }

    maxHit = maxHit > mRuntimeMaxDraftPathLen ? mRuntimeMaxDraftPathLen : maxHit;

    SizeType32 acceptedIdx = 0;
    BufferRange<TokenIdType> acceptedRange(*accepted);
    BufferRange<SizeType32> acceptedOffsetsRange(*acceptedOffsets);
    acceptedRange[acceptedIdx] = newLastToken;
    for (SizeType32 j = 0; j < draftSize; j++)
    {
        if (maskLocation.at(hitIdx, j) && acceptedIdx < maxHit)
        {
            acceptedOffsetsRange[acceptedIdx++] = j;
            acceptedRange[acceptedIdx] = goldRange[j];
        }
    }

    *BufferRange<SizeType32>(*acceptedLength).begin() = maxHit + 1;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

//! lookahead Jacobi matrix has prefilling phase and maintenance phase.
//! W=5, N=5.
//! *prefilling phase*
//! mFilling = 1->2, Tokens initialized from prompt. To fill the second line.
//! 0>1 2 3 *
//!       4 *
//!       5 *
//!       6 *
//!       7 *
//! mFilling = 2->3.
//! 0 1>2 3 4 *
//!       4 5 *
//!       5 6 *
//!       6 7 *
//!       7 8 *
//! mFilling = 3->4.
//! 0 1 2>3 4 5 *
//!       4 5 6 *
//!       5 6 7 *
//!       6 7 9 *
//!       7 8 a *
//! *maintenance phase*
//! mFilling = 4->4. shift up and generate five n-grams.
//! 0 1 2 3>4 5 6 *
//!       4 5 6 7 *
//!       5 6 7 8 *
//!       6 7 8 9 *
//!       7 8 9 a *
//! mFilling = 4.
//! 0 1 2 3 4>5 6 7 *
//!         5 6 7 8 *
//!         6 7 8 9 *
//!         7 8 9 a *
//!         8 9 a b *
//! mFilling = 4.
//! 0 1 2 3 4 5>6 7 8 *
//!           6 7 8 9 *
//!           7 8 9 a *
//!           8 9 a b *
//!           9 a b c *
void LookaheadAlgorithm::update(TensorPtr const& acceptedTokens, TensorPtr const& acceptedOffsets,
    TensorPtr const& acceptedLength, TensorConstPtr const& sampledTokens, TensorConstPtr const& endToken)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(ITensor::volume(acceptedTokens->getShape()) >= mN);
    BufferRange<TokenIdType const> zippedTokensRange(*sampledTokens);
    BufferRange<TokenIdType const> sampledRange(*mSampledTokensMax);

    BufferRange<SizeType32 const> mapRange(*mEncodeMap);
    BufferRange<TokenIdType> unzipRange(*mSampledTokensMax);
    mSampledTokens = ITensor::slice(mSampledTokensMax, 0, mEncodeMap->getShape().d[0] + 1);

    unzipRange[0] = zippedTokensRange[0];
    for (size_t i = 0; i < mapRange.size(); i++)
    {
        unzipRange[i + 1] = zippedTokensRange[mapRange[i] + 1];
    }

    BufferRange<TokenIdType> keyRange(*mKeyTokens);
    BufferRange<TokenIdType> pastRange(*mPastTokens);

    auto newLastToken = sampledRange[0];
    SizeType32 prefill = mN - 2 - mFilling;
    for (SizeType32 i = 0; i < mW; i++)
    {
        keyRange[i] = sampledRange[prefill + i * mFilling + mFilling];
    }

    if (mFilling < mN - 1)
    {
        for (SizeType32 i = 0; i < mW; i++)
        {
            pastRange[i * (mN - 1) + mFilling] = keyRange[i];
        }
    }
    else if (mN > 1)
    {
        for (SizeType32 i = 0; i < mW; i++)
        {
            auto begin = pastRange.begin() + i * (mN - 1);
            auto end = pastRange.begin() + i * (mN - 1) + mN - 1;
            auto key = *begin;
            std::copy(begin + 1, end, begin);
            *(std::prev(end, 1)) = keyRange[i];
            keyRange[i] = key;
        }
        keyRange[0] = newLastToken;
        mPoolManager.update(mKeyTokens, mPastTokens);
    }

    auto guessSize = ITensor::volume(mGuessTokens->getShape());
    auto outputSize = ITensor::volume(mSampledTokens->getShape());
    auto lookSize = 1 + (mN > 1 ? mN - 2 : 0) - mFilling + mFilling * mW;
    TLLM_CHECK(guessSize + lookSize == outputSize);

    TensorConstPtr goldenTokens = ITensor::slice(mSampledTokens, lookSize, guessSize);

    verify(acceptedTokens, acceptedOffsets, acceptedLength, newLastToken, ITensor::slice(sampledTokens, 1), endToken);

    accept(ITensor::slice(acceptedTokens, 0, *BufferRange<SizeType32>(*acceptedLength).begin()));

    if (mFilling < mN - 1)
    {
        mFilling++;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::layers
