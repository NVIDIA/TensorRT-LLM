WindowBlockManager::ReuseMatchResult WindowBlockManager::findReusableBlockMatches(
    std::vector<BlockKey> const& blockKeys, bool enablePartialReuse, bool copyOnPartialReuse,
    SizeType32 maxMatchedTokens) const
{
    ReuseMatchResult result;
    std::vector<ReuseMatch> candidateMatches;
    candidateMatches.reserve(blockKeys.size());
    auto searchNode = mCachedBlocksRoot ? mCachedBlocksRoot->getLookupNode() : nullptr;
    SizeType32 candidateMatchedTokens{0};
    SizeType32 latestMissingAnchorEndToken{0};

    auto updateSafePrefix = [&]()
    {
        if (!mIsSWA || latestMissingAnchorEndToken == 0
            || candidateMatchedTokens >= latestMissingAnchorEndToken + mWindowSize)
        {
            result.matches = candidateMatches;
            result.totalMatchedTokens = candidateMatchedTokens;
        }
    };

    for (auto const& blockKey : blockKeys)
    {
        if (!searchNode || blockKey.uniqueTokens.empty())
        {
            break;
        }

        auto exactMatch = searchNode->findMatchingNode(blockKey);
        if (exactMatch.has_value())
        {
            auto const numMatchedTokens = static_cast<SizeType32>(blockKey.uniqueTokens.size());
            if (candidateMatchedTokens + numMatchedTokens > maxMatchedTokens)
            {
                break;
            }

            auto const node = exactMatch->node;
            auto existing = node->getValue(mWindowSize);
            candidateMatchedTokens += numMatchedTokens;

            if (existing.has_value() && *existing)
            {
                auto block = *existing;
                candidateMatches.push_back(ReuseMatch{block, numMatchedTokens, !block->isFull(), false});
            }
            else if (mIsSWA)
            {
                candidateMatches.push_back(ReuseMatch{nullptr, numMatchedTokens, false, true});
                latestMissingAnchorEndToken = std::max(latestMissingAnchorEndToken, candidateMatchedTokens);
            }
            else
            {
                break;
            }

            searchNode = node;
            updateSafePrefix();
            continue;
        }

        if (enablePartialReuse)
        {
            auto partialMatches = searchNode->findPartiallyMatchingNodes(blockKey);
            for (auto const& match : partialMatches)
            {
                auto existing = match.node->getValue(mWindowSize);
                if (!existing.has_value() || !(*existing))
                {
                    continue;
                }

                auto block = *existing;
                if (copyOnPartialReuse || (!block->hasRefs() && block->isLeaf()))
                {
                    auto const numMatchedTokens = static_cast<SizeType32>(match.key.uniqueTokens.size());
                    if (candidateMatchedTokens + numMatchedTokens <= maxMatchedTokens)
                    {
                        candidateMatchedTokens += numMatchedTokens;
                        candidateMatches.push_back(ReuseMatch{block, numMatchedTokens, true, false});
                        updateSafePrefix();
                    }
                    break;
                }
            }
        }
        break;
    }

    if (result.matches.size() < blockKeys.size())
    {
        result.firstNewBlock = blockKeys[result.matches.size()];
    }
    return result;
}

