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

#include <gtest/gtest.h>
#include <list>

#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

namespace tensorrt_llm::tests::layers
{
using namespace tensorrt_llm::runtime;
using TensorPtr = runtime::ITensor::SharedPtr;

//! Initialize a tensor with data from string @param str. Shape {str.size} by default.
TensorPtr initTensor(std::string str, std::optional<ITensor::Shape> shape = std::nullopt);

//! Convert tokens to logits and vice versa according to a vocabulary.
class RandomTokenLogits
{
public:
    RandomTokenLogits(TensorPtr vocab)
        : mVocabulary(vocab)
    {
    }

    RandomTokenLogits(std::string vocab)
        : mVocabulary(initTensor(vocab))
    {
    }

    TensorPtr tokenToLogits(TokenIdType token) const;
    void tokenToLogits(TensorPtr logits, TokenIdType token) const;

    TokenIdType logitsToToken(TensorPtr logits) const;

    std::list<TensorPtr> stringToLogits(std::string tokens) const;
    void stringToLogits(TensorPtr logits, std::string tokens) const;
    void tensorToLogits(TensorPtr logits, TensorPtr tokens) const;

    std::string logitsToString(std::list<TensorPtr> logits) const;
    std::string logitsToString(TensorPtr logits) const;
    TensorPtr logitsToTensor(TensorPtr logits) const;

    SizeType32 getVocabSize() const;
    //! @return the last token in mVocabulary as invalid token;
    virtual TokenIdType getInvalidToken() const;
    //! @return the second-to-last token in mVocabulary as end token;
    virtual TokenIdType getEndToken() const;

private:
    TensorPtr const mVocabulary;
};

//! vocabulary is ascii table from 0 to 127. tokenId == token.
class AsciiRandomTokenLogits : public RandomTokenLogits
{
public:
    AsciiRandomTokenLogits()
        : RandomTokenLogits(
            []()
            {
                auto vocab = BufferManager::cpu(ITensor::makeShape({128}), nvinfer1::DataType::kINT32);
                auto vocabRange = BufferRange<TokenIdType>(*vocab);
                TokenIdType token{0};
                std::for_each(vocabRange.begin(), vocabRange.end(), [&token](auto& v) { v = token++; });
                return vocab;
            }())
    {
    }

    virtual TokenIdType getInvalidToken() const
    {
        return static_cast<TokenIdType>('#');
    }

    virtual TokenIdType getEndToken() const
    {
        return static_cast<TokenIdType>('&');
    }
};

//! random LLM to simulate functions of a real LLM.
class RandomLlm
{
public:
    RandomLlm(std::shared_ptr<RandomTokenLogits> table, std::string oracle, runtime::SizeType32 id = 0)
        : mTable(table)
        , mOracle(initTensor(oracle))
        , mId(id)
    {
    }

    // simulate forward in a LLM.
    void forward(TensorPtr output, TensorPtr const input, TensorPtr const position) const;
    //! set inout[i] invalid if mask[i]==false;
    void sampleByMask(TensorPtr inout, TensorPtr const mask) const;
    //! @return true when @param script is a sub-string started from @param offset.
    bool verify(SizeType32 const offset, TensorPtr const script) const;

    //! foretell @param output tokens from @param input tokens and @param position ids.
    //! It depends on different algorithms implementations.
    virtual void foretell(TensorPtr output, TensorPtr const input, TensorPtr const position) const = 0;

protected:
    std::shared_ptr<RandomTokenLogits> mTable;
    TensorPtr mOracle;
    runtime::SizeType32 mId;
};

//! a lookahead implementation for RandomLlm.
class LookaheadRandomLlm : public RandomLlm
{
public:
    LookaheadRandomLlm(std::shared_ptr<RandomTokenLogits> table, std::string oracle, runtime::SizeType32 id = 0)
        : RandomLlm(table, oracle, id)
    {
    }

    void foretell(TensorPtr output, TensorPtr const input, TensorPtr const position) const override;
};

} // namespace tensorrt_llm::tests::layers
