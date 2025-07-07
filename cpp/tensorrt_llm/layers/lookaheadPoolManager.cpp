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

#include "tensorrt_llm/layers/lookaheadPoolManager.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"
#include <cstddef>

namespace tensorrt_llm::layers
{

using namespace tensorrt_llm::runtime;

void LookaheadPoolManager::setup(SizeType32 guessSetSize)
{
    TLLM_CHECK(guessSetSize >= 0 && guessSetSize <= mGuessSetSizeMax);
    mGuessSetSize = guessSetSize;
    mTokenMap.clear();
}

void LookaheadPoolManager::insertOne(Key key, TensorConstPtr const& ngram)
{
    if (TLLM_UNLIKELY(ITensor::volume(ngram->getShape()) == 0 || mGuessSetSize == 0))
    {
        return;
    }

    auto search = mTokenMap.find(key);
    if (search != mTokenMap.end())
    {
        search->second.remove_if(
            [&ngram](TensorConstPtr const& item)
            {
                BufferRange<TokenIdType const> ngramRange(*ngram);
                BufferRange<TokenIdType const> itemRange(*item);
                return std::equal(ngramRange.begin(), ngramRange.end(), itemRange.begin());
            });
        if (mGuessSetSize > 0 && search->second.size() >= static_cast<size_t>(mGuessSetSize))
        {
            search->second.pop_front();
        }
        search->second.push_back(ngram);
    }
    else
    {
        mTokenMap.insert({key, std::list<TensorConstPtr>({ngram})});
    }
}

void LookaheadPoolManager::accept(TensorConstPtr const& prompt, SizeType32 level)
{
    SizeType32 length = prompt->getShape().d[0];
    BufferRange<Key const> promptRange(*prompt);
    for (SizeType32 ti = 0; ti + level - 1 < length; ti++)
    {
        auto key = promptRange[ti];
        TensorPtr ngram = BufferManager::cpu(ITensor::makeShape({level - 1}), nvinfer1::DataType::kINT32);
        BufferRange<TokenIdType const> sourceRange(*ITensor::slice(prompt, ti + 1, level - 1));
        BufferRange<TokenIdType> ngramRange(*ngram);
        std::copy(sourceRange.begin(), sourceRange.end(), ngramRange.begin());

        insertOne(key, ngram);
    }
}

std::list<LookaheadPoolManager::TensorConstPtr> LookaheadPoolManager::guess(Key lastToken, SizeType32 guessSize) const
{
    auto search = mTokenMap.find(lastToken);
    if (search != mTokenMap.end())
    {
        auto ngrams = search->second;
        if (ngrams.size() > static_cast<size_t>(guessSize)) // Always return the latest `guessSize` ngrams (MRU)
        {
            auto it = std::prev(ngrams.end(), guessSize);
            return std::list<TensorConstPtr>(it, ngrams.end());
        }
        else
        {
            return ngrams;
        }
    }
    else
    {
        return std::list<TensorConstPtr>();
    }
}

void LookaheadPoolManager::update(TensorConstPtr const& keyTokens, TensorConstPtr const& ngramTokens)
{
    TLLM_CHECK(keyTokens->getShape().d[0] == ngramTokens->getShape().d[0]);
    BufferRange<Key const> keyRange(*keyTokens);
    auto window = ngramTokens->getShape().d[0];

    for (SizeType32 wi = 0; wi < window; wi++)
    {
        TensorConstPtr source = ITensor::at(ngramTokens, {wi});
        TensorPtr ngram = BufferManager::cpu(source->getShape(), nvinfer1::DataType::kINT32);
        BufferRange<TokenIdType const> sourceRange(*source);
        BufferRange<TokenIdType> ngramRange(*ngram);
        std::copy(sourceRange.begin(), sourceRange.end(), ngramRange.begin());
        insertOne(keyRange[wi], ngram);
    }
}

void LookaheadPoolManager::print(char const* functionName, int const lineNumber) const noexcept
{
    TLLM_LOG_TRACE("==== printPoolManager start @%s @L%d start", functionName, lineNumber);

    TLLM_LOG_TRACE("Pool size=%zu", mTokenMap.size());
    for (auto const& [key, list] : mTokenMap)
    {
        TLLM_LOG_TRACE("Key=%zu (size=%zu)", key, list.size());
        for (auto const& value : list)
        {
            std::string info{"    ->("};
            auto const& buffer = tensorrt_llm::runtime::bufferCast<runtime::TokenIdType>(*value);

            for (auto j = 0; j < tensorrt_llm::runtime::ITensor::volume((*value).getShape()); ++j)
            {
                info += std::to_string(buffer[j]) + ",";
            }
            info += ")";
            TLLM_LOG_TRACE(info.c_str());
        }
    }

    TLLM_LOG_TRACE("==== printPoolManager stop @%s @L%d start", functionName, lineNumber);
}

} // namespace tensorrt_llm::layers
