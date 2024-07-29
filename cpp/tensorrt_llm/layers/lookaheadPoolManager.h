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

#include <list>
#include <unordered_map>

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::layers
{

//! @brief A helper class for managing key-ngram pool.
class LookaheadPoolManager
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorConstPtr = runtime::ITensor::SharedConstPtr;
    using Key = runtime::TokenIdType;

    LookaheadPoolManager(runtime::SizeType32 maxG)
        : mGuessSetSizeMax(maxG)
    {
    }

    //! @brief setup runtime resource
    //! @param guessSetSize the runtime guessSetSize.
    void setup(runtime::SizeType32 guessSetSize);

    //! @brief fill token map from accepted tokens, including prompt.
    //! @param prompt the user input prompt, [length] on cpu
    //! @param level the n-gram length
    void accept(TensorConstPtr const& prompt, runtime::SizeType32 level);

    //! @brief  get a list of guess tokens
    //! @param lastToken the newest golden token
    //! @param guessSize at most guessSize candidates returned
    //! @return the list guess tokens, with list size <= guessSize
    std::list<TensorConstPtr> guess(Key lastToken, runtime::SizeType32 guessSize) const;

    //! @brief update token map with new generated tokens
    //! @param keyTokens the new shifted out tokens from each window, as the key, [window] on cpu
    //! @param ngramTokens the new shifted lookahead window, as the ngrams, [window, ngramLen] on cpu
    void update(TensorConstPtr const& keyTokens, TensorConstPtr const& ngramTokens);

    std::unordered_map<Key, std::list<TensorConstPtr>> const& getMap() const
    {
        return mTokenMap;
    }

private:
    void insertOne(Key key, TensorConstPtr const& ngram);

private:
    //! @brief the token map with token as key and list of n-gram as value
    std::unordered_map<Key, std::list<TensorConstPtr>> mTokenMap;
    //! @brief guess set size, -1 for infinite size
    runtime::SizeType32 const mGuessSetSizeMax;
    runtime::SizeType32 mGuessSetSize;
};

} // namespace tensorrt_llm::layers
