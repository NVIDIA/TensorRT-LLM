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
#include "tests/layers/randomLlm.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"

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

TensorPtr RandomTokenLogits::tokenToLogits(TokenIdType token) const
{
    TensorPtr logits = BufferManager::cpu(mVocabulary->getShape(), nvinfer1::DataType::kFLOAT);
    tokenToLogits(logits, token);
    return logits;
}

void RandomTokenLogits::tokenToLogits(TensorPtr logits, TokenIdType token) const
{
    TLLM_CHECK_WITH_INFO(logits->shapeEquals({getVocabSize()}), "%s != {%d}",
        ITensor::toString(logits->getShape()).c_str(), getVocabSize());

    auto logitsRange = BufferRange<float>(*logits);
    auto vocabRange = BufferRange<TokenIdType>(*mVocabulary);
    auto itl = logitsRange.begin();
    auto itv = vocabRange.begin();
    for (; itl != logitsRange.end() && itv != vocabRange.end(); itl++, itv++)
    {
        bool match = (*itv == token);
        *itl = (match ? 1.0 : 0.0) + (static_cast<float>(rand() % 256) / 1000.0);
    }
}

TokenIdType RandomTokenLogits::logitsToToken(TensorPtr logits) const
{
    TLLM_CHECK(logits->shapeEquals({getVocabSize()}));
    auto logitsRange = BufferRange<float>(*logits);
    auto vocabRange = BufferRange<TokenIdType>(*mVocabulary);
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

std::list<TensorPtr> RandomTokenLogits::stringToLogits(std::string tokens) const
{
    std::list<TensorPtr> result;
    for (auto& token : tokens)
    {
        result.push_back(tokenToLogits(static_cast<TokenIdType>(token)));
    }
    return result;
}

void RandomTokenLogits::stringToLogits(TensorPtr logits, std::string tokens) const
{
    TLLM_CHECK(logits->shapeEquals({static_cast<SizeType32>(tokens.size()), getVocabSize()}));

    auto i = 0;
    for (auto& token : tokens)
    {
        tokenToLogits(ITensor::at(logits, {i++}), static_cast<TokenIdType>(token));
    }
}

void RandomTokenLogits::tensorToLogits(TensorPtr logits, TensorPtr tokens) const
{
    TLLM_CHECK(ITensor::volume(logits->getShape()) == ITensor::volume(tokens->getShape()) * getVocabSize());
    // TLLM_CHECK(logits->shapeEquals({static_cast<SizeType32>(tokens.size()), getVocabSize()}));
    auto tokensRange = BufferRange<TokenIdType>(*tokens);
    auto i = 0;
    for (auto it = tokensRange.begin(); it != tokensRange.end(); it++)
    {
        tokenToLogits(ITensor::at(logits, {i++}), *it);
    }
}

std::string RandomTokenLogits::logitsToString(std::list<TensorPtr> logits) const
{
    std::string result;
    for (auto& token : logits)
    {
        result.push_back(logitsToToken(token));
    }
    return result;
}

std::string RandomTokenLogits::logitsToString(TensorPtr logits) const
{
    auto len = logits->getShape().d[0];
    std::string result;
    for (auto i = 0; i < len; i++)
    {
        result.push_back(logitsToToken(ITensor::at(logits, {i})));
    }
    return result;
}

TensorPtr RandomTokenLogits::logitsToTensor(TensorPtr logits) const
{
    auto len = logits->getShape().d[0];
    TensorPtr result = BufferManager::cpu(ITensor::makeShape({len}), nvinfer1::DataType::kINT32);
    auto resultRange = BufferRange<TokenIdType>(*result);
    for (auto i = 0; i < len; i++)
    {
        resultRange[i] = logitsToToken(ITensor::at(logits, {i}));
    }
    return result;
}

SizeType32 RandomTokenLogits::getVocabSize() const
{
    return ITensor::volume(mVocabulary->getShape());
}

TokenIdType RandomTokenLogits::getInvalidToken() const
{
    return *(BufferRange<TokenIdType>(*mVocabulary).end() - 1);
}

TokenIdType RandomTokenLogits::getEndToken() const
{
    return *(BufferRange<TokenIdType>(*mVocabulary).end() - 2);
}

void RandomLlm::sampleByMask(TensorPtr inout, TensorPtr mask) const
{
    auto len = ITensor::volume(mask->getShape());
    TLLM_CHECK(len == ITensor::volume(mask->getShape()));
    auto inoutRange = BufferRange<TokenIdType>(*inout);
    auto maskRange = BufferRange<bool>(*mask);
    auto invalid = mTable->getInvalidToken();

    for (SizeType32 i = 0; i < len; i++)
    {
        if (!maskRange[i])
        {
            inoutRange[i] = invalid;
        }
    }
}

bool RandomLlm::verify(SizeType32 const offset, TensorPtr const script) const
{
    auto oracleRange = BufferRange<TokenIdType>(*mOracle);
    auto scriptRange = BufferRange<TokenIdType>(*script);
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

void RandomLlm::forward(TensorPtr output, TensorPtr const input, TensorPtr const position) const
{
    TLLM_CHECK(ITensor::volume(input->getShape()) == ITensor::volume(position->getShape()));
    TLLM_CHECK(ITensor::volume(output->getShape()) == ITensor::volume(input->getShape()) * mTable->getVocabSize());

    TensorPtr tokens = BufferManager::cpu(input->getShape(), nvinfer1::DataType::kINT32);
    foretell(tokens, input, position);
    if (mId == 4)
    {
        TLLM_LOG_DEBUG("batch[%d] DEBUG", mId);
        PRINT_TOKENS(tokens);
        PRINT_TOKENS(input);
        PRINT_TOKENS(position);
    }
    mTable->tensorToLogits(output, tokens);
}

void LookaheadRandomLlm::foretell(TensorPtr output, TensorPtr const input, TensorPtr const position) const
{
    TLLM_CHECK(ITensor::volume(input->getShape()) == ITensor::volume(position->getShape()));
    TLLM_CHECK(ITensor::volume(output->getShape()) >= ITensor::volume(input->getShape()));

    auto outputRange = BufferRange<TokenIdType>(*output);
    auto inputRange = BufferRange<TokenIdType>(*input);
    auto positionRange = BufferRange<TokenIdType>(*position);
    auto oracleRange = BufferRange<TokenIdType>(*mOracle);
    auto len = ITensor::volume(input->getShape());
    auto olen = ITensor::volume(mOracle->getShape());

    std::vector<std::vector<bool>> mask(len, std::vector<bool>(len, false));
    std::vector<std::pair<SizeType32, SizeType32>> stack;
    stack.push_back(std::make_pair(0, positionRange[0]));
    mask[0][0] = true;
    for (auto i = 1; i < len; i++)
    {
        auto cur = positionRange[i];
        while (stack.size() > 0 && cur <= stack.back().second)
        {
            stack.pop_back();
        }
        TLLM_CHECK(stack.size() > 0 ? cur == stack.back().second + 1 : true);
        stack.push_back(std::make_pair(i, cur));
        for (auto prev : stack)
        {
            mask[i][prev.first] = true;
        }
    }
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
            right &= mask[i][j] ? oracleRange[positionRange[j]] == inputRange[j] : true;
        }
        if (i < verifyStart)
        { // lookahead might be right
            outputRange[i] = ((right || rand() % 5) && legal) ? oracleRange[positionRange[i] + 1] : invalid;
        }
        else
        { // verify should be wrong.
            outputRange[i] = (right && legal) ? oracleRange[positionRange[i] + 1] : invalid;
        }
    }
}

} // namespace tensorrt_llm::tests::layers
