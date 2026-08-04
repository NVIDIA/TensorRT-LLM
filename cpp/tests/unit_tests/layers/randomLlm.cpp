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
#include "tests/unit_tests/layers/randomLlm.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::tests::layers
{

using namespace tensorrt_llm::layers;

TensorPtr initTensor(std::string str, std::optional<ITensor::Shape> shape)
{
    auto shape1d = ITensor::makeShape({static_cast<SizeType32>(str.size())});
    if (shape)
    {
        TLLM_CHECK(ITensor::volume(shape1d) == ITensor::volume(shape.value()));
    }
    TensorPtr tensor = BufferManager::cpu(shape.value_or(shape1d), nvinfer1::DataType::kINT32);
    auto tensorRange = BufferRange<TokenIdType>(*tensor);
    std::copy(str.begin(), str.end(), tensorRange.begin());
    return tensor;
}

TensorConstPtr RandomTokenLogits::tokenToLogits(TokenIdType token) const
{
    TensorPtr logits = BufferManager::cpu(mVocabulary->getShape(), nvinfer1::DataType::kFLOAT);
    tokenToLogits(logits, token);
    return logits;
}

void RandomTokenLogits::tokenToLogits(TensorPtr const& logits, TokenIdType token) const
{
    TLLM_CHECK_WITH_INFO(logits->shapeEquals({getVocabSize()}), "%s != {%d}",
        ITensor::toString(logits->getShape()).c_str(), getVocabSize());

    auto logitsRange = BufferRange<float>(*logits);
    auto vocabRange = BufferRange<TokenIdType const>(*mVocabulary);
    auto itl = logitsRange.begin();
    auto itv = vocabRange.begin();
    for (; itl != logitsRange.end() && itv != vocabRange.end(); itl++, itv++)
    {
        bool match = (*itv == token);
        *itl = (match ? 1.0 : 0.0) + (static_cast<float>(rand() % 256) / 1000.0);
    }
}

TokenIdType RandomTokenLogits::logitsToToken(TensorConstPtr const& logits) const
{
    TLLM_CHECK(logits->shapeEquals({getVocabSize()}));
    auto logitsRange = BufferRange<float const>(*logits);
    auto vocabRange = BufferRange<TokenIdType const>(*mVocabulary);
    float max = -FLT_MAX;
    TokenIdType result;
    auto itl = logitsRange.begin();
    auto itv = vocabRange.begin();
    for (; itl != logitsRange.end() && itv != vocabRange.end(); itl++, itv++)
    {
        float cur = exp(*itl);
        if (cur > max)
        {
            max = cur;
            result = *itv;
        }
    }
    return result;
}

std::list<TensorConstPtr> RandomTokenLogits::stringToLogits(std::string tokens) const
{
    std::list<TensorConstPtr> result;
    for (auto& token : tokens)
    {
        result.push_back(tokenToLogits(static_cast<TokenIdType>(token)));
    }
    return result;
}

void RandomTokenLogits::stringToLogits(TensorPtr const& logits, std::string tokens) const
{
    TLLM_CHECK(logits->shapeEquals({static_cast<SizeType32>(tokens.size()), getVocabSize()}));

    auto i = 0;
    for (auto& token : tokens)
    {
        tokenToLogits(ITensor::at(logits, {i++}), static_cast<TokenIdType>(token));
    }
}

void RandomTokenLogits::tensorToLogits(TensorPtr const& logits, TensorConstPtr const& tokens) const
{
    TLLM_CHECK(ITensor::volume(logits->getShape()) == ITensor::volume(tokens->getShape()) * getVocabSize());
    // TLLM_CHECK(logits->shapeEquals({static_cast<SizeType32>(tokens.size()), getVocabSize()}));
    auto tokensRange = BufferRange<TokenIdType const>(*tokens);
    auto i = 0;
    for (auto it = tokensRange.begin(); it != tokensRange.end(); it++)
    {
        tokenToLogits(ITensor::at(logits, {i++}), *it);
    }
}

std::string RandomTokenLogits::logitsToString(std::list<TensorConstPtr> logits) const
{
    std::string result;
    for (auto& token : logits)
    {
        result.push_back(logitsToToken(token));
    }
    return result;
}

std::string RandomTokenLogits::logitsToString(TensorConstPtr const& logits) const
{
    auto len = logits->getShape().d[0];
    std::string result;
    for (auto i = 0; i < len; i++)
    {
        result.push_back(logitsToToken(ITensor::at(logits, {i})));
    }
    return result;
}

void RandomTokenLogits::logitsToTensor(TensorPtr const& tokens, TensorConstPtr const& logits) const
{
    auto len = logits->getShape().d[0];
    TLLM_CHECK(tokens->getShape().d[0] >= len);
    auto tokensRange = BufferRange<TokenIdType>(*tokens);
    for (auto i = 0; i < len; i++)
    {
        tokensRange[i] = logitsToToken(ITensor::at(logits, {i}));
    }
}

TensorConstPtr RandomTokenLogits::logitsToTensor(TensorConstPtr const& logits) const
{
    auto len = logits->getShape().d[0];
    TensorPtr result = BufferManager::cpu(ITensor::makeShape({len}), nvinfer1::DataType::kINT32);
    logitsToTensor(result, logits);
    return result;
}

SizeType32 RandomTokenLogits::getVocabSize() const
{
    return ITensor::volume(mVocabulary->getShape());
}

TokenIdType const RandomTokenLogits::getInvalidToken() const
{
    return *(BufferRange<TokenIdType const>(*mVocabulary).end() - 1);
}

TokenIdType const RandomTokenLogits::getEndToken() const
{
    return *(BufferRange<TokenIdType const>(*mVocabulary).end() - 2);
}

void RandomLlm::sampleByMask(TensorPtr const& inout, TensorConstPtr const& mask) const
{
    auto len = ITensor::volume(mask->getShape());
    TLLM_CHECK(len == ITensor::volume(mask->getShape()));
    auto inoutRange = BufferRange<TokenIdType>(*inout);
    auto maskRange = BufferRange<bool const>(*mask);
    auto invalid = mTable->getInvalidToken();

    for (SizeType32 i = 0; i < len; i++)
    {
        if (!maskRange[i])
        {
            inoutRange[i] = invalid;
        }
    }
}

bool RandomLlm::verify(SizeType32 const offset, TensorConstPtr const& script) const
{
    auto oracleRange = BufferRange<TokenIdType const>(*mOracle);
    auto scriptRange = BufferRange<TokenIdType const>(*script);
    auto len = ITensor::volume(script->getShape());
    auto result = std::equal(oracleRange.begin() + offset, oracleRange.begin() + offset + len, scriptRange.begin());
    if (!result)
    {
        std::string gold(len, '#');
        std::string wrong(len, '#');
        std::copy(oracleRange.begin() + offset, oracleRange.begin() + offset + len, gold.begin());
        std::copy(scriptRange.begin(), scriptRange.end(), wrong.begin());
        TLLM_CHECK_WITH_INFO(result, "len=%ld, gold='%s', script='%s'", len, gold.c_str(), wrong.c_str());
    }
    return result;
}

void RandomLlm::forward(TensorPtr const& output, runtime::SizeType32 startId, TensorConstPtr const& input,
    TensorConstPtr const& offsets, TensorConstPtr const mask) const
{
    TensorPtr posIds = BufferManager::cpu(input->getShape(), nvinfer1::DataType::kINT32);
    BufferRange<SizeType32> idRange(*posIds);
    BufferRange<SizeType32 const> offsetRange(*offsets);
    for (auto i = 0; i < idRange.size(); i++)
    {
        idRange[i] = startId + offsetRange[i];
    }
    forward(output, input, posIds, mask);
}

void RandomLlm::forward(TensorPtr const& output, TensorConstPtr const& input, TensorConstPtr const& position,
    TensorConstPtr const mask) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(ITensor::volume(input->getShape()) == ITensor::volume(position->getShape()));
    TLLM_CHECK(ITensor::volume(output->getShape()) == ITensor::volume(input->getShape()) * mTable->getVocabSize());

    TensorPtr tokens = BufferManager::cpu(input->getShape(), nvinfer1::DataType::kINT32);
    foretell(tokens, input, position, mask);
    // foretellOld(tokens, input, position);
    mTable->tensorToLogits(output, tokens);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadRandomLlm::foretell(TensorPtr const& output, TensorConstPtr const& input, TensorConstPtr const& position,
    TensorConstPtr const mask) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto len = ITensor::volume(input->getShape());
    TLLM_CHECK(ITensor::volume(position->getShape()) == len);
    TLLM_CHECK(ITensor::volume(output->getShape()) >= len);
    if (mask)
    {
        TLLM_CHECK(ITensor::volume(mask->getShape()) >= len * len);
        TLLM_CHECK(mask->getShape().d[0] >= len);
        TLLM_CHECK(mask->getShape().d[1] >= len);
    }

    TensorPtr maskRebuilt = BufferManager::cpu(ITensor::makeShape({len, len}), nvinfer1::DataType::kBOOL);
    posIdsToMask(maskRebuilt, position);

    auto outputRange = BufferRange<TokenIdType>(*output);
    auto inputRange = BufferRange<TokenIdType const>(*input);
    auto positionRange = BufferRange<SizeType32 const>(*position);
    auto maskLocation = mask ? BufferLocation<bool const>(*mask) : BufferLocation<bool const>(*maskRebuilt);
    auto oracleRange = BufferRange<SizeType32 const>(*mOracle);
    auto olen = ITensor::volume(mOracle->getShape());

    auto verifyStart = 2;
    for (; verifyStart < len - 1; verifyStart++)
    {
        if (positionRange[verifyStart] == positionRange[0] + 1)
        {
            break;
        }
    }

    auto invalid = mTable->getInvalidToken();
    TLLM_CHECK(positionRange[0] + 1 < olen);
    for (auto i = 0; i < len; i++)
    {
        bool legal = positionRange[i] + 1 < olen;
        bool right = true;
        for (auto j = 0; j < i; j++)
        {
            right &= maskLocation.at(i, j) ? oracleRange[positionRange[j]] == inputRange[j] : true;
        }
        if (i < verifyStart && false)
        { // lookahead might be right. Since we verify lookahead branch, then must be right.
            outputRange[i] = ((right || rand() % 5) && legal) ? oracleRange[positionRange[i] + 1] : invalid;
        }
        else
        { // verify should be wrong.
            outputRange[i] = (right && legal) ? oracleRange[positionRange[i] + 1] : invalid;
        }
    }
}

void LookaheadRandomLlm::posIdsToMask(TensorPtr const& mask, TensorConstPtr const& posIds) const
{
    auto len = ITensor::volume(posIds->getShape());
    TLLM_CHECK(ITensor::volume(mask->getShape()) >= len * len);
    auto posIdsRange = BufferRange<SizeType32 const>(*posIds);
    auto maskRange = BufferRange<bool>(*mask);

    for (auto i = 0; i < maskRange.size(); i++)
    {
        maskRange[i] = false;
    }

    std::vector<std::pair<SizeType32, SizeType32>> stack;
    stack.push_back(std::make_pair(0, posIdsRange[0]));
    maskRange[0 * len + 0] = true;
    for (auto i = 1; i < len; i++)
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
            maskRange[i * len + prev.first] = true;
        }
    }
}

void LookaheadRandomLlm::maskToPosIds(TensorPtr const& posIds, TensorConstPtr const& mask, SizeType32 start) const
{
    auto len = ITensor::volume(posIds->getShape());
    TLLM_CHECK(ITensor::volume(mask->getShape()) >= len * len);
    auto posIdsRange = BufferRange<SizeType32>(*posIds);
    auto maskLocation = BufferLocation<bool const>(*mask);
    for (auto i = 0; i < len; i++)
    {
        posIdsRange[i] = start;
        for (auto j = 0; j < i; j++)
        {
            posIdsRange[i] += maskLocation.at(i, j);
        }
    }
}

} // namespace tensorrt_llm::tests::layers
