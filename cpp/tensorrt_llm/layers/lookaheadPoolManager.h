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
    using Key = runtime::TokenIdType;

    LookaheadPoolManager(runtime::SizeType32 g, std::shared_ptr<runtime::BufferManager> bufferManager)
        : mGuessSetSize(g)
        , mBufferManager(bufferManager)
    {
    }

    //! @brief fill token map from prompt
    //! @param prompt the user input prompt, [length] on cpu
    //! @param level the n-gram length
    void fillWithPrompt(TensorPtr prompt, runtime::SizeType32 level);

    //! @brief  get a list of guess tokens
    //! @param lastToken the newest golden token
    //! @param guessSize at most guessSize candidates returned
    //! @return the list guess tokens, with list size <= guessSize
    std::list<TensorPtr> guess(Key lastToken, runtime::SizeType32 guessSize) const;

    //! @brief update token map with new generated tokens
    //! @param keyTokens the new shifted out tokens from each window, as the key, [window] on cpu
    //! @param ngramTokens the new shifted lookahead window, as the ngrams, [window, ngramLen] on cpu
    void update(TensorPtr keyTokens, TensorPtr ngramTokens);

    void clear(void);

    std::unordered_map<Key, std::list<TensorPtr>> const& getMap() const
    {
        return mTokenMap;
    }

private:
    void insertOne(Key key, TensorPtr ngram);

private:
    std::shared_ptr<runtime::BufferManager> mBufferManager;
    //! @brief the token map with token as key and list of n-gram as value
    std::unordered_map<Key, std::list<TensorPtr>> mTokenMap;
    //! @brief guess set size, -1 for infinite size
    runtime::SizeType32 mGuessSetSize;
};

} // namespace tensorrt_llm::layers
