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

    mPrefills = ITensor::slice(mPrefillsMax, 0, mN > 2 ? mN - 2 : 0);
    mPastTokens = ITensor::slice(mPastTokensMax, 0, mW * (mN - 1));
    mPastTokens->reshape(ITensor::makeShape({mW, mN - 1}));
    mKeyTokens = ITensor::slice(mKeyTokensMax, 0, mW);
    mGoldenTokens = ITensor::slice(mGoldenTokensMax, 0, mN * 2 - 1);
    mGuessTokens = ITensor::slice(mGuessTokensMax, 0, 0);

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
        if (mN > 1)
        {
            randToken(pastRange[i * (mN - 1)]);
        }
    }
    std::copy(std::prev(promptRange.end(), mN - 1), promptRange.end(), goldRange.begin());

    mFilling = mN > 1 ? 1 : 0;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadAlgorithm::accept(TensorConstPtr const& generatedTokens)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    PRINT_TOKEN(generatedTokens);
    PRINT_TOKEN(mGoldenTokens);

    TLLM_CHECK(ITensor::volume(generatedTokens->getShape()) <= mN);
    BufferRange<TokenIdType const> generatedRange(*generatedTokens);
    BufferRange<TokenIdType> goldRange(*mGoldenTokens);
    auto genLen = generatedTokens->getShape().d[0];
    TLLM_CHECK(genLen <= mN);
    std::copy(generatedRange.begin(), generatedRange.end(), goldRange.begin() + mN - 1);
    TensorPtr newGold = ITensor::slice(mGoldenTokens, 0, mN - 1 + genLen);
    TLLM_LOG_TRACE("genLen = %d, mN - 1 + genLen = %d", genLen, mN - 1 + genLen);
    PRINT_TOKEN(newGold);

    mPoolManager.accept(newGold, mN);
    // Remove the first `genLen` tokens in mGoldenTokens
    std::copy(goldRange.begin() + genLen, goldRange.begin() + genLen + mN - 1, goldRange.begin());

    TLLM_LOG_TRACE("%s end", __PRETTY_FUNCTION__);
}

//! lookahead has two phase, prefill the past tokens matrix and maintain past tokens matrix.
runtime::SizeType32 LookaheadAlgorithm::lookahead(
    TensorPtr const& draftTokens, TensorPtr const& positionIds, runtime::SizeType32 startPosId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    SizeType32 const prefill = mN - 2 - mFilling;
    SizeType32 const len = prefill + mFilling * mW;
    TLLM_CHECK(len <= ITensor::volume(draftTokens->getShape()));
    TLLM_CHECK(len <= ITensor::volume(positionIds->getShape()));
    BufferRange<TokenIdType> prefillRange(*mPrefills);
    BufferRange<TokenIdType> pastRange(*mPastTokens);
    BufferRange<TokenIdType> draftRange(*draftTokens);
    TLLM_LOG_TRACE("[wili]mFilling=%d", mFilling);
    TLLM_LOG_TRACE("[wili]prefill=%d", prefill);
    TLLM_LOG_TRACE("[wili]len=%d", len);
    PRINT_TOKEN(mPrefills);
    PRINT_TOKEN(mPastTokens);
    PRINT_TOKEN(draftTokens);

    if (mFilling < mN - 1) // prefilling
    {
        std::copy(prefillRange.begin() + mFilling, prefillRange.end(), draftRange.begin());
        for (auto i = 0; i < mW; i++)
        {
            auto const start = pastRange.begin() + i * (mN - 1);
            auto const end = pastRange.begin() + i * (mN - 1) + mFilling;
            std::copy(start, end, draftRange.begin() + prefill + i * mFilling);
        }
    }
    else // shift up
    {
        std::copy(pastRange.begin() + 1, pastRange.begin() + mFilling * mW, draftRange.begin());
    }
    PRINT_TOKEN(draftTokens);

    PRINT_TOKEN(positionIds);
    BufferRange<TokenIdType> positionIdsRange(*positionIds);
    SizeType32 idx = 0;
    auto fillPosition = [&positionIdsRange, &idx](SizeType32 start, SizeType32 length)
    {
        for (auto i = start; i < start + length; i++)
        {
            positionIdsRange[idx++] = i;
        }
    };

    if (prefill >= 0)
    {
        fillPosition(startPosId, prefill);
        for (auto wj = 0; wj < mW; wj++)
        {
            fillPosition(startPosId + prefill + wj, mFilling);
        }
    }
    else
    {
        fillPosition(startPosId, mFilling - 1);
        for (auto wj = 1; wj < mW; wj++)
        {
            fillPosition(startPosId - 1 + wj, mFilling);
        }
    }
    PRINT_VALUE(positionIds);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return len;
}

runtime::SizeType32 LookaheadAlgorithm::guess(TensorPtr const& draftTokens, TensorPtr const& positionIds,
    runtime::SizeType32 startPosId, runtime::TokenIdType lastToken)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    PRINT_TOKEN(draftTokens);
    PRINT_TOKEN(positionIds);
    TLLM_LOG_TRACE("[wili]startPosId=%d", startPosId);
    TLLM_LOG_TRACE("[wili]lastToken=%d", lastToken);

    auto guesses = mPoolManager.guess(lastToken, mG);

    SizeType32 len = 0;
    std::for_each(guesses.begin(), guesses.end(), [&len](auto& a) { len += ITensor::volume(a->getShape()); });
    TLLM_CHECK(len <= ITensor::volume(draftTokens->getShape()));
    TLLM_CHECK(len <= ITensor::volume(positionIds->getShape()));
    BufferRange<TokenIdType> draftTokensRange(*draftTokens);
    BufferRange<SizeType32> positionIdsRange(*positionIds);

    SizeType32 cur = 0;
    for (auto guess : guesses)
    {
        BufferRange<TokenIdType const> guessRange(*guess);
        std::copy(guessRange.begin(), guessRange.end(), draftTokensRange.begin() + cur);
        SizeType32 positionId = startPosId;
        std::for_each(positionIdsRange.begin() + cur, positionIdsRange.begin() + cur + mN - 1,
            [&positionId](auto& v) { v = positionId++; });
        cur += ITensor::volume(guess->getShape());

        PRINT_TOKEN(draftTokens);
        PRINT_TOKEN(positionIds);
        TLLM_LOG_TRACE("[wili]cur=%d", cur);
        TLLM_LOG_TRACE("[wili]positionId=%d", positionId);
    }

    TLLM_LOG_TRACE("%s end", __PRETTY_FUNCTION__);

    return len;
}

void LookaheadAlgorithm::posIdsToMask(TensorPtr const& mask, TensorConstPtr const& posIds)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto len = ITensor::volume(posIds->getShape());
    TLLM_CHECK(mask->getDimension<0>() >= len);
    TLLM_CHECK(mask->getDimension<1>() >= len);
    auto maskLocation = BufferLocation<bool>(*mask);
    auto posIdsRange = BufferRange<SizeType32 const>(*posIds);
    PRINT_TOKEN(mask);
    PRINT_TOKEN(posIds);
    for (auto& item : maskLocation)
    {
        item = false;
    }
    if (len == 0)
    {
        return;
    }

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
    PRINT_TOKEN(mask);

    TLLM_LOG_TRACE("%s end", __PRETTY_FUNCTION__);
}

struct TreeValue;
using TreeMap = std::unordered_map<TokenIdType, TreeValue>;
using TreeNode = TreeMap::value_type; // std::pair<TokenIdType, TreeValue>;

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

void printTreeValue(TreeValue const& value, int const depth = 0)
{
    auto const space = std::string(depth * 2, ' ');
    printf("%svalue(%p): nexts(%p)=[", space.c_str(), &value, value.nexts.get());
    for (auto const& node : *value.nexts)
    {
        printf("%d, ", node.first);
    }
    printf("], sources=[");
    for (auto const& source : value.sources)
    {
        printf("%d, ", source);
    }
    printf("]\n");
}

void printTreeNode(TreeNode const& node, int const depth = 0)
{
    auto const space = std::string(depth * 2, ' ');
    printf("%snode(%p): id=%d, value=%p\n", space.c_str(), &node, node.first, &node.second);
    printTreeValue(node.second, depth); // Only one value for one node
    printf("\n");
}

void printTreeMap(TreeMap const& map, int const depth = 0)
{
    if (depth == 0)
    {
        printf("==== Tree start ====\n");
    }
    if (map.size() > 0)
    {

        auto const space = std::string(depth * 2, ' ');
        printf("%smap(%p): ids=[", space.c_str(), &map);
        for (auto const& node : map)
        {
            printf("%d, ", node.first);
        }
        printf("]\n");
        for (auto const& node : map)
        {
            printTreeNode(node, depth + 1);
            printTreeMap(*(node.second.nexts), depth + 1);
        }
    }
    if (depth == 0)
    {
        printf("==== Tree stop ====\n");
    }
}

template <typename BF, typename AF>
void treeDFS(TreeNode& node, BF const& visitBefore, AF const& visitAfter, int const depth = 0)
{
    printTreeNode(node, depth);
    visitBefore(node);
    for (auto& next : *(node.second.nexts))
    {
        treeDFS(next, visitBefore, visitAfter, depth + 1);
    }
    visitAfter(node);
}

SizeType32 LookaheadAlgorithm::treeEncode(
    TensorPtr const& tokens, TensorPtr const& posIds, TensorPtr const& mask, TensorPtr const& encodeMap)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(ITensor::volume(tokens->getShape()) == ITensor::volume(posIds->getShape()));
    auto len = ITensor::volume(tokens->getShape());

    BufferRange<TokenIdType> tokensRange(*tokens);
    BufferRange<SizeType32> posIdsRange(*posIds);
    BufferLocation<bool> maskLocation(*mask);
    BufferRange<SizeType32> mapRange(*encodeMap);

    TLLM_LOG_TRACE("Before DFS");
    PRINT_TOKEN(tokens);
    PRINT_TOKEN(posIds);
    PRINT_TOKEN(mask);
    PRINT_TOKEN(encodeMap);

    auto branches = std::make_shared<TreeMap>();
    int depth = 0;
    // printTreeMap(*branches, 0);
    for (auto i = 0; i < len; i++)
    {
        auto nexts = branches;
        depth = 0;
        for (auto j = 0; j <= i; j++)
        {
            if (maskLocation.at(i, j))
            {
                printf("[%2d,%2d] head, nexts=%p\n", i, j, nexts.get());
                auto tok = tokensRange[j];
                auto found = nexts->find(tok);
                printf("> tok=%d,found=%d\n", tok, (found != nexts->end()));
                if (found != nexts->end())
                {
                    printf("  id=%d, value=%p\n", found->first, &found->second);
                    found->second.sources.push_back(j);
                    nexts = found->second.nexts;
                    depth += 1;
                }
                else
                {
                    printf("  insert %d to list of %p\n", tok, nexts.get());
                    auto [inserted, ok] = nexts->insert({tok, TreeValue()});
                    inserted->second.sources.push_back(j);
                    nexts = inserted->second.nexts;
                    depth += 1;
                }
                printf("[%2d,%2d]tail: nexts=%p\n", i, j, nexts.get());
                // printTreeMap(*branches, 0);
            }
        }
    }

    TLLM_LOG_TRACE("Before DFS  -- after building tree");
    printTreeMap(*branches, 0);

    for (auto& item : maskLocation)
    {
        item = 0;
    }
    std::vector<std::pair<SizeType32, TokenIdType>> stack;
    SizeType32 posId = posIdsRange.size() ? posIdsRange[0] : 0;
    SizeType32 offset = 0;

    auto visitBefore
        = [&stack, &maskLocation, &tokensRange, &posIdsRange, &posId, &offset, &mapRange](TreeNode const& node)
    {
        printf("(visitBefore)\n");
        printf("stack=");
        for (auto const& pair : stack)
        {
            printf("[%d,%d],", pair.first, pair.second);
        }
        printf("\n");
        printf("newPair=[%d,%d]\n", offset, node.first);

        stack.push_back(std::make_pair(offset, node.first));
        for (auto const& source : node.second.sources)
        {
            printf("source=%d, offset=%d\n", source, offset);
            mapRange[source] = offset;
        }
        for (auto const& prev : stack)
        {
            printf("offset=%d, prev.first=%d\n", offset, prev.first);
            printf("~~~~~~~~~~~~~~~~ mask[%d,%d]=1\n", offset, prev.first);
            maskLocation.at(offset, prev.first) = true;
        }
        printf("offset=%d\n", offset);
        printf("node.first=%d\n", node.first);
        printf("posId=%d\n", posId);
        tokensRange[offset] = node.first;
        posIdsRange[offset] = posId;
        offset++; // Always increases during DFS
        posId++;
        printf("offset=%d\n", offset);
        printf("posId=%d\n", posId);
    };
    auto visitAfter = [&stack, &posId](TreeNode const& node)
    {
        printf("(visitAfter)\n");
        printf("stack=");
        for (auto const& pair : stack)
        {
            printf("[%d,%d],", pair.first, pair.second);
        }
        printf("\n");
        printf("posId=%d\n", posId);
        stack.pop_back();
        posId--;
        printf("posId=%d\n", posId);
    };

    for (auto& next : *branches)
    {
        treeDFS(next, visitBefore, visitAfter);
    }

    TLLM_LOG_TRACE("After DFS");
    PRINT_TOKEN(tokens);
    PRINT_TOKEN(posIds);
    PRINT_TOKEN(mask);
    PRINT_TOKEN(encodeMap);

    for (SizeType32 i = offset; i < len; i++)
    {
        tokensRange[i] = 0;
        posIdsRange[i] = 0;
    }
    for (auto i = 0; i < len; i++)
    {
        for (auto j = i < offset ? offset : 0; j < len; j++)
        {
            maskLocation.at(i, j) = false;
        }
    }

    TLLM_LOG_TRACE("After all");
    PRINT_TOKEN(tokens);
    PRINT_TOKEN(posIds);
    PRINT_TOKEN(mask);
    PRINT_TOKEN(encodeMap);

    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    return offset;
}

void LookaheadAlgorithm::prepare(TensorPtr const& draftTokens, TensorPtr const& positionIds,
    TensorPtr const& draftLengthPtr, TensorPtr const& attentionMask, SizeType32 const offset,
    TokenIdType const lastToken)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    constexpr SizeType32 attentionMaskOffset = 1;

    PRINT_TOKEN(draftTokens);
    PRINT_TOKEN(positionIds);
    PRINT_TOKEN(draftLengthPtr);
    PRINT_VALUE(attentionMask);
    TLLM_LOG_TRACE("L%3d, offset=%d", __LINE__, offset);
    TLLM_LOG_TRACE("L%3d, lastToken=%d", __LINE__, lastToken);

    if (mRuntimeMaxDraftLen == 0)
    {
        mDraftTokens = ITensor::slice(mDraftTokensMax, 0, 0);
        mEncodeMap = ITensor::slice(mEncodeMapMax, 0, 0);
        (BufferRange<SizeType32>(*draftLengthPtr))[0] = 0;
        return;
    }

    SizeType32 inputLen = ITensor::volume(draftTokens->getShape());
    TLLM_CHECK(inputLen >= mRuntimeMaxDraftLen);

    TLLM_LOG_TRACE("L%3d, inputLen=%d", __LINE__, inputLen);

    BufferRange<TokenIdType> draftRange(*draftTokens);
    BufferRange<TokenIdType> positionRange(*positionIds);

    SizeType32 filledLen = 0;

    filledLen += lookahead(ITensor::slice(draftTokens, filledLen, mRuntimeMaxDraftLen - filledLen),
        ITensor::slice(positionIds, filledLen, mRuntimeMaxDraftLen - filledLen), offset);

    TLLM_LOG_TRACE("L%3d, filledLen=%d", __LINE__, filledLen);

    auto guessStart = filledLen;
    filledLen += guess(ITensor::slice(draftTokens, filledLen, mRuntimeMaxDraftLen - filledLen),
        ITensor::slice(positionIds, filledLen, mRuntimeMaxDraftLen - filledLen), offset, lastToken);
    auto guessEnd = filledLen;

    TLLM_LOG_TRACE("L%3d, filledLen=%d", __LINE__, filledLen);

    std::copy(draftRange.begin() + guessStart, draftRange.begin() + guessEnd,
        BufferRange<TokenIdType>(*mGuessTokensMax).begin());
    mGuessTokens = ITensor::slice(mGuessTokensMax, 0, guessEnd - guessStart);

    posIdsToMask(mAttentionMask, ITensor::slice(positionIds, 0, filledLen));

    auto draftLen = treeEncode(ITensor::slice(draftTokens, 0, filledLen), ITensor::slice(positionIds, 0, filledLen),
        mAttentionMask, mEncodeMapMax);

    TLLM_LOG_TRACE("L%3d, draftLen=%d", __LINE__, draftLen);

    for (auto i = 0; i < draftLen; i++)
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
    BufferRange<TokenIdType> acceptedRange(*accepted);
    BufferRange<SizeType32> acceptedOffsetsRange(*acceptedOffsets);
    BufferRange<TokenIdType const> goldRange(*goldenTokens);
    BufferRange<TokenIdType> draftRange(*mDraftTokens);
    BufferLocation<bool const> maskLocation(*mAttentionMask);
    auto draftSize = ITensor::volume(mDraftTokens->getShape());
    auto end = *BufferRange<TokenIdType const>(*endToken).begin();

    TLLM_LOG_TRACE("newLastToken=%d", newLastToken);
    PRINT_TOKEN(accepted);
    PRINT_TOKEN(acceptedOffsets);
    PRINT_TOKEN(acceptedLength);
    PRINT_TOKEN(goldenTokens);
    PRINT_TOKEN(endToken);
    PRINT_TOKEN(mDraftTokens);
    PRINT_TOKEN(mAttentionMask);

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

    PRINT_TOKEN(accepted);
    PRINT_TOKEN(acceptedOffsets);
    PRINT_TOKEN(acceptedLength);

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
    BufferRange<TokenIdType const> sampledRange(*sampledTokens);
    BufferRange<TokenIdType> sampledMaxRange(*mSampledTokensMax);
    BufferRange<SizeType32 const> mapRange(*mEncodeMap);
    mSampledTokens = ITensor::slice(mSampledTokensMax, 0, mEncodeMap->getShape().d[0] + 1);

    sampledMaxRange[0] = sampledRange[0];
    for (size_t i = 0; i < mapRange.size(); i++)
    {
        sampledMaxRange[i + 1] = sampledRange[mapRange[i] + 1];
    }
    PRINT_TOKEN(mEncodeMap);
    PRINT_TOKEN(mSampledTokensMax);

    BufferRange<TokenIdType> keyRange(*mKeyTokens);
    BufferRange<TokenIdType> pastRange(*mPastTokens);

    PRINT_TOKEN(mKeyTokens);
    PRINT_TOKEN(mPastTokens);

    auto newLastToken = sampledMaxRange[0];
    SizeType32 prefill = mN - 2 - mFilling;
    for (SizeType32 i = 0; i < mW; i++)
    {
        keyRange[i] = sampledMaxRange[prefill + i * mFilling + mFilling];
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

    PRINT_TOKEN(mKeyTokens);
    PRINT_TOKEN(mPastTokens);

    auto guessSize = ITensor::volume(mGuessTokens->getShape());
    auto lookSize = 1 + (mN > 1 ? mN - 2 : 0) - mFilling + mFilling * mW;
    auto outputSize = ITensor::volume(mSampledTokens->getShape());
    TLLM_CHECK(guessSize + lookSize == outputSize);
    TLLM_LOG_TRACE("guessSize=%d", guessSize);
    TLLM_LOG_TRACE("lookSize=%d", lookSize);
    TLLM_LOG_TRACE("outputSize=%d", outputSize);

    TensorConstPtr goldenTokens = ITensor::slice(mSampledTokens, lookSize, guessSize);
    PRINT_TOKEN(goldenTokens);

    verify(acceptedTokens, acceptedOffsets, acceptedLength, newLastToken, ITensor::slice(sampledTokens, 1), endToken);

    accept(ITensor::slice(acceptedTokens, 0, *BufferRange<SizeType32>(*acceptedLength).begin()));

    if (mFilling < mN - 1)
    {
        mFilling++;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadAlgorithm::print(char const* functionName, int const lineNumber) const noexcept
{
    TLLM_LOG_TRACE("======== printAlgorithm @%s @L%d start", functionName, lineNumber);

    TLLM_LOG_TRACE("mMaxW=%d, ", mMaxW);
    TLLM_LOG_TRACE("mMaxN=%d, ", mMaxN);
    TLLM_LOG_TRACE("mMaxG=%d, ", mMaxG);
    TLLM_LOG_TRACE("mW=%d, ", mW);
    TLLM_LOG_TRACE("mN=%d, ", mN);
    TLLM_LOG_TRACE("mG=%d, ", mG);
    TLLM_LOG_TRACE("mRuntimeMaxDraftLen=%d, ", mRuntimeMaxDraftLen);
    TLLM_LOG_TRACE("mRuntimeMaxDraftPathLen=%d, ", mRuntimeMaxDraftPathLen);
    TLLM_LOG_TRACE("mFilling=%d, ", mFilling);

    PRINT_TOKEN(mPrefillsMax);
    PRINT_TOKEN(mPrefills);
    PRINT_TOKEN(mPastTokensMax);
    PRINT_TOKEN(mPastTokens);
    PRINT_TOKEN(mKeyTokensMax);
    PRINT_TOKEN(mKeyTokens);
    PRINT_TOKEN(mGoldenTokensMax);
    PRINT_TOKEN(mGoldenTokens);
    PRINT_TOKEN(mGuessTokensMax);
    PRINT_TOKEN(mGuessTokens);
    PRINT_TOKEN(mDraftTokensMax);
    PRINT_TOKEN(mDraftTokens);
    PRINT_VALUE(mAttentionMask);
    PRINT_TOKEN(mEncodeMapMax);
    PRINT_TOKEN(mEncodeMap);
    PRINT_TOKEN(mSampledTokensMax);
    PRINT_TOKEN(mSampledTokens);

    mPoolManager.print(functionName, lineNumber);

    TLLM_LOG_TRACE("======== printAlgorithm @%s @L%d stop", functionName, lineNumber);
}

} // namespace tensorrt_llm::layers
