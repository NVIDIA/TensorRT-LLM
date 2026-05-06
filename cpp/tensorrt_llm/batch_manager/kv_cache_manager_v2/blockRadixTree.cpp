/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kv_cache_manager_v2/blockRadixTree.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/page.h"
#include "kv_cache_manager_v2/storageManager.h"

#include "blake3.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <variant>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// Hasher
// ---------------------------------------------------------------------------

static void hashInt64(blake3_hasher* h, int64_t v)
{
    // Little-endian 8-byte encoding — mirrors Python's int.to_bytes(8, 'little').
    uint8_t buf[8];
    for (int i = 0; i < 8; ++i)
        buf[i] = static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * i)) & 0xFF);
    blake3_hasher_update(h, buf, sizeof(buf));
}

Hasher::Hasher()
{
    blake3_hasher_init(&mState);
}

Hasher::Hasher(std::optional<int64_t> seed)
{
    blake3_hasher_init(&mState);
    if (seed.has_value())
        hashInt64(&mState, *seed);
}

Hasher& Hasher::update(TokenId token)
{
    hashInt64(&mState, static_cast<int64_t>(token));
    return *this;
}

Hasher& Hasher::update(BlockKey const& key)
{
    blake3_hasher_update(&mState, reinterpret_cast<uint8_t const*>(key.data()), key.size());
    return *this;
}

Hasher& Hasher::update(TokenIdExt const& tokenExt)
{
    std::visit(
        [this](auto const& v)
        {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, TokenId>)
                hashInt64(&mState, static_cast<int64_t>(v));
            else
                blake3_hasher_update(&mState, reinterpret_cast<uint8_t const*>(v.data()), v.size());
        },
        tokenExt);
    return *this;
}

Hasher& Hasher::update(TokenIdExt const* tokens, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        update(tokens[i]);
    return *this;
}

BlockKey Hasher::digest() const
{
    // BLAKE3 supports streaming output; finalize into a 32-byte key.
    BlockKey out;
    // We need a copy of the hasher state to finalize without mutating.
    blake3_hasher copy = mState;
    blake3_hasher_finalize(&copy, reinterpret_cast<uint8_t*>(out.data()), out.size());
    return out;
}

// ---------------------------------------------------------------------------
// genMultiModalTokens
// ---------------------------------------------------------------------------

std::vector<TokenIdExt> genMultiModalTokens(
    int idOffset, std::vector<uint8_t> const& multiModalDataDigest, int numTokens)
{
    assert(numTokens > 0);
    assert(multiModalDataDigest.size() == kDIGEST_LEN);
    std::vector<TokenIdExt> result;
    result.reserve(static_cast<size_t>(numTokens));
    for (int i = 0; i < numTokens; ++i)
    {
        if (i == 0)
        {
            Digest d;
            std::memcpy(d.data(), multiModalDataDigest.data(), kDIGEST_LEN);
            result.emplace_back(DigestToken(d));
        }
        else
            result.emplace_back(TokenId(idOffset + i));
    }
    return result;
}

// ---------------------------------------------------------------------------
// makeBlockchainKeyGenerator — lazy key generator.
// Returns a callable that yields one BlockKey per call (nullopt when done).
// First call yields root entry (empty token block). Mirrors Python's generator.
// ---------------------------------------------------------------------------

static auto makeBlockchainKeyGenerator(
    int tokensPerBlock, std::optional<int64_t> loraTaskId, TokenIdExt const* tokens, size_t numTokens)
{
    // digest carries the running hash from the previous block.
    BlockKey digest = Hasher(loraTaskId).digest();
    // ordinal = -1: next call yields root (loraTaskId digest).
    // ordinal >= 0: next call yields key for tokens[ordinal*tpb .. (ordinal+1)*tpb).
    int ordinal = -1;

    return [=]() mutable -> std::optional<BlockKey>
    {
        if (ordinal == -1)
        {
            ordinal++;
            return digest; // root key
        }

        size_t beg = static_cast<size_t>(ordinal) * static_cast<size_t>(tokensPerBlock);
        if (beg >= numTokens)
            return std::nullopt;

        size_t end = std::min(beg + static_cast<size_t>(tokensPerBlock), numTokens);

        Hasher h;
        h.update(digest);
        h.update(tokens + beg, end - beg);
        digest = h.digest();

        ordinal++;
        return digest;
    };
}

// Eager wrapper for callers that need all keys at once.
std::vector<BlockKey> sequenceToBlockchainKeys(
    int tokensPerBlock, std::optional<int64_t> loraTaskId, std::vector<TokenIdExt> const& tokens)
{
    std::vector<BlockKey> result;
    auto gen = makeBlockchainKeyGenerator(tokensPerBlock, loraTaskId, tokens.data(), tokens.size());
    while (auto key = gen())
        result.push_back(*key);
    return result;
}

// ---------------------------------------------------------------------------
// RootBlock
// ---------------------------------------------------------------------------

BlockKey RootBlock::makeKey(std::optional<int64_t> loraTaskId)
{
    return Hasher(loraTaskId).digest();
}

RootBlock::RootBlock(std::optional<int64_t> loraTaskId, std::shared_ptr<BlockRadixTree> const& treePtr)
    : key(makeKey(loraTaskId))
    , loraTaskId(loraTaskId)
    , tree(treePtr)
{
}

int RootBlock::numLifeCycles() const
{
    auto t = tree.lock();
    assert(t && "BlockRadixTree destroyed before RootBlock");
    return t->numLifeCycles();
}

int RootBlock::tokensPerBlock() const
{
    auto t = tree.lock();
    assert(t && "BlockRadixTree destroyed before RootBlock");
    return t->tokensPerBlock();
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

namespace
{

static void tryExcludeFromEviction(std::weak_ptr<CommittedPage> const& weakPage)
{
    auto page = weakPage.lock();
    if (page && page->status() == PageStatus::DROPPABLE && page->nodeRef.has_value())
    {
        if (auto mgr = page->manager.lock())
            mgr->excludeFromEviction(*page);
    }
}

static bool isPrefix(std::vector<TokenIdExt> const& prefix, std::vector<TokenIdExt> const& full)
{
    if (prefix.size() > full.size())
        return false;
    for (size_t i = 0; i < prefix.size(); ++i)
    {
        if (prefix[i] != full[i])
            return false;
    }
    return true;
}

} // anonymous namespace

BlockKey Block::makeKey(BlockKey const& prevKey, TokenIdExt const* tokens, size_t count)
{
    Hasher h;
    h.update(prevKey);
    h.update(tokens, count);
    return h.digest();
}

Block::~Block()
{
    // If we have cached pages that are droppable+scheduled, exclude them from eviction.
    // This mirrors Python Block.__del__.
    for (auto const& weakPage : storage)
    {
        tryExcludeFromEviction(weakPage);
    }
    // If we're the last child of a root block, remove the root block from the tree.
    // Mirrors Python: `self.prev.prev.next.pop(self.prev.key)`
    if (parentRoot && parentRoot->next.empty())
    {
        if (auto t = parentRoot->tree.lock())
        {
            auto rootKey = parentRoot->key;
            t->eraseRoot(rootKey);
        }
    }
}

int Block::tokensPerBlock() const noexcept
{
    if (parentBlock)
        return static_cast<int>(parentBlock->tokens.size());
    if (parentRoot)
        return parentRoot->tokensPerBlock();
    return 0;
}

bool Block::isOrphan() const noexcept
{
    auto* map = const_cast<Block*>(this)->parentNextMap();
    if (!map)
        return true;
    auto it = map->find(key);
    return it == map->end() || it->second.get() != this;
}

int Block::partialMatchThisNode(TokenIdExt const* otherTokens, size_t otherCount) const
{
    int count = 0;
    for (size_t i = 0; i < std::min(tokens.size(), otherCount); ++i)
    {
        if (tokens[i] != otherTokens[i])
            break;
        ++count;
    }
    return count;
}

std::unordered_map<BlockKey, std::shared_ptr<Block>>* Block::parentNextMap()
{
    if (parentRoot)
        return &parentRoot->next;
    if (parentBlock)
        return &parentBlock->next;
    return nullptr;
}

BlockKey const& Block::parentKey() const
{
    if (parentRoot)
        return parentRoot->key;
    if (parentBlock)
        return parentBlock->key;
    static BlockKey empty{};
    return empty;
}

void Block::unsetPage(LifeCycleId lcIdx, LifeCycle const& lc)
{
    if (storage.at(static_cast<size_t>(lcIdx)).expired())
        return;
    storage[static_cast<size_t>(lcIdx)].reset();

    // Reuse cleanup only applies to attention lifecycles.
    // SSM lifecycles are allowed in the tree but don't trigger subtree eviction.
    if (auto const* alc = std::get_if<AttnLifeCycle>(&lc))
    {
        // If this is a non-sink block with no window (or sink block): evict descendants.
        // Mirrors Python's remove_subtree(self), which clears ALL storage on self and descendants.
        if (!alc->windowSize.has_value() || ordinal < alc->numSinkBlocks)
        {
            std::vector<BlockKey> childKeys;
            childKeys.reserve(next.size());
            for (auto const& [k, child] : next)
                childKeys.push_back(k);

            for (auto const& k : childKeys)
            {
                auto pages = removeSubtree(next, k);
                for (auto const& weakPage : pages)
                    tryExcludeFromEviction(weakPage);
            }

            // Also clear self's storage for OTHER lifecycles (Python's remove_subtree(self)
            // clears all entries in self.storage, not just lcIdx).
            for (size_t i = 0; i < storage.size(); ++i)
            {
                if (i != static_cast<size_t>(lcIdx))
                {
                    tryExcludeFromEviction(storage[i]);
                    storage[i].reset();
                }
            }
        }
    }

    // Prune empty tail nodes up the chain.
    Block* curr = this;
    while (curr && curr->storage.at(static_cast<size_t>(lcIdx)).expired() && curr->next.empty())
    {
        auto* map = curr->parentNextMap();
        if (map)
            map->erase(curr->key);
        curr = curr->parentBlock;
    }
}

// ---------------------------------------------------------------------------
// addOrGetExistingBlock
// ---------------------------------------------------------------------------

std::shared_ptr<Block> addOrGetExistingBlock(std::unordered_map<BlockKey, std::shared_ptr<Block>>& parentNext,
    BlockKey const& parentKey, int numLifeCycles, int tokensPerBlock, std::vector<TokenIdExt> tokens,
    RootBlock* parentRoot, Block* parentBlock, bool* isNew)
{
    // Parent must be a full block (mirrors Python: "prev must be a full block").
    if (parentBlock)
    {
        assert(parentBlock->isFull() && "prev must be a full block");
    }

    BlockKey newKey = Block::makeKey(parentKey, tokens.data(), tokens.size());

    // Exact match: return existing block (not new — mirrors Python's UselessBlockError path).
    auto it = parentNext.find(newKey);
    if (it != parentNext.end())
    {
        if (isNew)
            *isNew = false;
        return it->second;
    }

    // Useless check: is this block's token prefix covered by a sibling?
    // Mirrors Python's UselessBlockError — throw with the sibling block.
    if (static_cast<int>(tokens.size()) < tokensPerBlock)
    {
        for (auto const& [k, sibling] : parentNext)
        {
            if (sibling->tokens.size() >= tokens.size() && isPrefix(tokens, sibling->tokens))
                throw UselessBlockError(sibling);
        }
    }

    // Remove siblings whose tokens are a strict prefix of ours.
    std::vector<BlockKey> toRemove;
    for (auto const& [k, sibling] : parentNext)
    {
        if (sibling->tokens.size() < tokens.size() && isPrefix(sibling->tokens, tokens))
        {
            assert(!sibling->isFull() && sibling->key == k && sibling->next.empty());
            toRemove.push_back(k);
        }
    }
    for (auto const& k : toRemove)
    {
        auto erased = parentNext.find(k);
        assert(erased != parentNext.end());
        auto erasedBlock = erased->second;
        parentNext.erase(erased);
        assert(erasedBlock->isOrphan() && "erased sibling must be orphan after removal");
    }

    // Create the new block.
    auto block = std::make_shared<Block>();
    block->key = newKey;
    block->tokens = std::move(tokens);
    block->ordinal = parentRoot ? BlockOrdinal(0) : BlockOrdinal(parentBlock->ordinal + 1);
    block->parentRoot = parentRoot;
    block->parentBlock = parentBlock;
    block->storage.assign(static_cast<size_t>(numLifeCycles), {});

    parentNext[newKey] = block;
    if (isNew)
        *isNew = true;
    return block;
}

// ---------------------------------------------------------------------------
// removeSubtree
// ---------------------------------------------------------------------------

std::vector<std::weak_ptr<CommittedPage>> removeSubtree(
    std::unordered_map<BlockKey, std::shared_ptr<Block>>& parentNext, BlockKey const& rootKey)
{
    std::vector<std::weak_ptr<CommittedPage>> ret;
    auto it = parentNext.find(rootKey);
    if (it == parentNext.end())
        return ret;

    std::vector<std::shared_ptr<Block>> stack = {it->second};
    parentNext.erase(it);

    while (!stack.empty())
    {
        auto block = stack.back();
        stack.pop_back();

        // Collect pages.
        for (auto const& weakPage : block->storage)
            if (!weakPage.expired())
                ret.push_back(weakPage);

        // Clear storage.
        block->storage.assign(block->storage.size(), {});
        assert(gNdebug
            || (std::all_of(block->storage.begin(), block->storage.end(), [](auto const& wp) { return wp.expired(); })
                && "storage must be cleared after assign"));

        // Push children (they will be destroyed when removed from parent's next).
        for (auto& [k, child] : block->next)
            stack.push_back(child);
        block->next.clear();
    }

    return ret;
}

// ---------------------------------------------------------------------------
// BlockRadixTree
// ---------------------------------------------------------------------------

BlockRadixTree::BlockRadixTree(LifeCycleRegistry const& lifeCycles, int tokensPerBlock)
    : mLifeCycles(lifeCycles)
    , mTokensPerBlock(tokensPerBlock)
{
}

BlockRadixTree::~BlockRadixTree()
{
    // Clear all roots (which will drop all blocks).
    mRoots.clear();
}

int BlockRadixTree::numLifeCycles() const noexcept
{
    return static_cast<int>(mLifeCycles.size());
}

RootBlock& BlockRadixTree::addOrGetExisting(std::optional<int64_t> loraTaskId)
{
    BlockKey key = RootBlock::makeKey(loraTaskId);
    auto it = mRoots.find(key);
    if (it != mRoots.end())
        return it->second;

    auto [newIt, inserted] = mRoots.emplace(key, RootBlock(loraTaskId, shared_from_this()));
    return newIt->second;
}

// Among all child nodes, find the one whose tokens have the longest leading match.
// Returns (block, numMatchedTokens) or (nullptr, 0) if no match.
// Mirrors Python's find_best_partial_match_in_next_nodes().
std::pair<Block*, int> findBestPartialMatchInNextNodes(
    std::unordered_map<BlockKey, std::shared_ptr<Block>> const& nextMap, TokenIdExt const* tokens, size_t tokenCount)
{
    // Skip heuristic: too many children would be slow to iterate.
    if (nextMap.size() >= 32)
        return {nullptr, 0};
    Block* best = nullptr;
    int bestMatch = 0;
    for (auto const& [k, child] : nextMap)
    {
        int m = child->partialMatchThisNode(tokens, tokenCount);
        if (m > bestMatch)
        {
            bestMatch = m;
            best = child.get();
        }
    }
    return {best, bestMatch};
}

std::vector<BlockRadixTree::MatchResult> BlockRadixTree::match(
    std::optional<int64_t> loraTaskId, std::vector<TokenIdExt> const& tokens, bool enablePartialMatch) const
{
    std::vector<MatchResult> results;

    // Lazily compute one key per iteration — no wasted hashing on early miss.
    auto gen = makeBlockchainKeyGenerator(mTokensPerBlock, loraTaskId, tokens.data(), tokens.size());

    // First key is the root key.
    auto rootKey = gen();
    if (!rootKey)
        return results;
    auto rootIt = mRoots.find(*rootKey);
    if (rootIt == mRoots.end())
        return results;

    RootBlock const& root = rootIt->second;
    std::unordered_map<BlockKey, std::shared_ptr<Block>> const* currentNext = &root.next;
    // ordinal tracks which block we're on (0-based, after root).
    int ordinal = 0;
    bool missed = false;

    while (auto key = gen())
    {
        auto blockIt = currentNext->find(*key);
        if (blockIt == currentNext->end())
        {
            missed = true;
            break;
        }
        size_t beg = static_cast<size_t>(ordinal) * static_cast<size_t>(mTokensPerBlock);
        int numTokens = static_cast<int>(std::min(static_cast<size_t>(mTokensPerBlock), tokens.size() - beg));
        Block* block = blockIt->second.get();
        results.push_back({block, numTokens});
        currentNext = &block->next;
        ordinal++;
    }

    // Partial match in children of current node.
    if (missed && enablePartialMatch)
    {
        size_t beg = static_cast<size_t>(ordinal) * static_cast<size_t>(mTokensPerBlock);
        size_t missedCount = std::min(static_cast<size_t>(mTokensPerBlock), tokens.size() - beg);
        auto [best, bestMatch] = findBestPartialMatchInNextNodes(*currentNext, tokens.data() + beg, missedCount);
        if (best)
            results.push_back({best, bestMatch});
    }

    return results;
}

std::vector<std::weak_ptr<CommittedPage>> BlockRadixTree::clear()
{
    std::vector<std::weak_ptr<CommittedPage>> ret;

    // Swap mRoots into a local so that if ~Block() calls eraseRoot() during
    // removeSubtree() destruction, it operates on the (now-empty) mRoots — safe no-op.
    std::unordered_map<BlockKey, RootBlock> roots;
    roots.swap(mRoots);

    for (auto& [rootKey, root] : roots)
    {
        // Collect child keys, then delegate to removeSubtree() for each — mirrors Python's clear().
        std::vector<BlockKey> childKeys;
        childKeys.reserve(root.next.size());
        for (auto const& [k, child] : root.next)
            childKeys.push_back(k);

        for (auto const& k : childKeys)
        {
            auto pages = removeSubtree(root.next, k);
            for (auto const& weakPage : pages)
                ret.push_back(weakPage);
        }
    }

    assert(mRoots.empty() && "mRoots must be empty after clear");
    return ret;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
