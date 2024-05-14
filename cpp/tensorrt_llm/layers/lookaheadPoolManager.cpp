

#include "tensorrt_llm/layers/lookaheadPoolManager.h"

namespace tensorrt_llm::layers
{

using namespace tensorrt_llm::runtime;

void LookaheadPoolManager::insertOne(Key key, TensorPtr ngram)
{
    auto search = mTokenMap.find(key);
    if (search != mTokenMap.end())
    {
        search->second.remove_if(
            [&ngram](TensorPtr const& item)
            {
                auto ngramRange = BufferRange<TokenIdType>(*ngram);
                auto itemRange = BufferRange<TokenIdType>(*item);
                return std::equal(ngramRange.begin(), ngramRange.end(), itemRange.begin());
            });
        if (mGuessSetSize >= 0 && search->second.size() >= mGuessSetSize)
        {
            search->second.pop_front();
        }
        search->second.push_back(ngram);
    }
    else
    {
        mTokenMap.insert({key, std::list<TensorPtr>({ngram})});
    }
}

void LookaheadPoolManager::fillWithPrompt(TensorPtr prompt, SizeType32 level)
{
    SizeType32 length = prompt->getShape().d[0];
    auto promptRange = BufferRange<Key>(*prompt);
    for (SizeType32 ti = 0; ti + level - 1 < length; ti++)
    {
        auto key = promptRange[ti];
        TensorPtr ngram
            = mBufferManager->copyFrom(*ITensor::slice(prompt, ti + 1, level - 1), runtime::MemoryType::kCPU);
        insertOne(key, ngram);
    }
}

std::list<LookaheadPoolManager::TensorPtr> LookaheadPoolManager::guess(Key lastToken, SizeType32 guessSize) const
{
    auto search = mTokenMap.find(lastToken);
    if (search != mTokenMap.end())
    {
        auto ngrams = search->second;
        if (ngrams.size() > guessSize)
        {
            auto it = std::prev(ngrams.end(), guessSize);
            return std::list<TensorPtr>(it, ngrams.end());
        }
        else
        {
            return ngrams;
        }
    }
    else
    {
        return std::list<TensorPtr>();
    }
}

void LookaheadPoolManager::update(TensorPtr keyTokens, TensorPtr ngramTokens)
{
    TLLM_CHECK(keyTokens->getShape().d[0] == ngramTokens->getShape().d[0]);
    auto keyRange = BufferRange<Key>(*keyTokens);
    auto window = ngramTokens->getShape().d[0];

    for (SizeType32 wi = 0; wi < window; wi++)
    {
        TensorPtr ngram = mBufferManager->copyFrom(*ITensor::slice(ngramTokens, wi, 1), runtime::MemoryType::kCPU);
        ngram->squeeze(0);
        insertOne(keyRange[wi], ngram);
    }
}

} // namespace tensorrt_llm::layers
