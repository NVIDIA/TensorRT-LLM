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

template <typename T>
class BufferLocation : BufferRange<T>
{
public:
    using BufferRange<T>::begin;

    BufferLocation(ITensor& t)
        : BufferRange<T>(t)
        , volumes(t.getShape().nbDims)
    {
        auto shape = t.getShape();
        for (SizeType32 i = 0; i < shape.nbDims; i++)
        {
            SizeType32 volume = 1;
            for (SizeType32 j = i + 1; j < shape.nbDims; j++)
            {
                volume *= shape.d[j];
            }
            volumes[i] = volume;
        }
    }

    T& operator()(std::initializer_list<SizeType32> const& dims)
    {
        TLLM_CHECK(volumes.size() == dims.size());
        SizeType32 offset = 0;
        auto itd = dims.begin();
        auto itv = volumes.begin();
        for (; itd != dims.end() && itv != volumes.end(); itd++, itv++)
        {
            offset += (*itd) * (*itv);
        }
        return *(begin() + offset);
    }

private:
    std::vector<SizeType32> volumes;
};

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
    RandomLlm(std::shared_ptr<RandomTokenLogits> table, std::string oracle)
        : mTable(table)
        , mOracle(initTensor(oracle))
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
};

//! a lookahead implementation for RandomLlm.
class LookaheadRandomLlm : public RandomLlm
{
public:
    LookaheadRandomLlm(std::shared_ptr<RandomTokenLogits> table, std::string oracle)
        : RandomLlm(table, oracle)
    {
    }

    void foretell(TensorPtr output, TensorPtr const input, TensorPtr const position) const override;
};

} // namespace tensorrt_llm::tests::layers
