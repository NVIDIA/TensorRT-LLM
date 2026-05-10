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

RootBlock::RootBlock(std::optional<int64_t> loraTaskId, BlockRadixTree* treePtr)
    : NodeBase(makeKey(loraTaskId))
    , loraTaskId(loraTaskId)
    , tree(treePtr)
{
}

int RootBlock::tokensPerBlock() const noexcept
{
    return tree->tokensPerBlock();
}

// ---------------------------------------------------------------------------
// NodeBase
// ---------------------------------------------------------------------------

NodeBase::~NodeBase()
{
    // Detach children before next is destroyed (implicit member destruction).
    // This ensures that when a child's ~Block() runs, it sees prev == nullptr
    // and skips parent cleanup — avoiding virtual calls on a mid-destruction parent.
    for (auto& [k, child] : next)
    {
        child->prev = nullptr;
    }
}

std::shared_ptr<Block> NodeBase::detachNext(BlockKey const& blockKey)
{
    auto it = next.find(blockKey);
    if (it == next.end())
    {
        return nullptr;
    }

    auto block = it->second;
    block->prev = nullptr;
    next.erase(it);
    if (type() == Type::kROOT_BLOCK && next.empty())
    {
        auto* root = static_cast<RootBlock*>(this);
        root->tree->proposeToEraseEmptyRoot(root->key);
    }
    return block;
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

namespace
{

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

Block::Block(BlockKey k, std::vector<TokenIdExt> toks, NodeBase* prevNode, int nlc)
    : NodeBase(k)
    , tokens(std::move(toks))
    , prev(prevNode)
    , storage(static_cast<size_t>(nlc), nullptr)
    , mOrdinal(prevNode->ordinal() + 1)
{
}

int Block::tokensPerBlock() const noexcept
{
    assert(prev && "Block must have a prev");
    // Mirrors Python: prev.tokens_per_block if isinstance(prev, RootBlock) else len(prev.tokens)
    if (prev->type() == Type::kROOT_BLOCK)
        return prev->tokensPerBlock();
    return static_cast<int>(static_cast<Block const*>(prev)->tokens.size());
}

Block::~Block()
{
    // Mirrors Python Block.__del__: for each stored page, if alive and
    // DROPPABLE and scheduled for eviction, exclude from eviction.
    // Also null out the page's back-pointer so that CommittedPage::~CommittedPage()
    // doesn't attempt cleanup through this dead Block.
    for (size_t i = 0; i < storage.size(); ++i)
    {
        auto const page = storage[i];
        if (page != nullptr)
        {
            unlinkPage(static_cast<LifeCycleId>(i));
            if (page->status() == PageStatus::DROPPABLE && page->scheduledForEviction())
            {
                page->manager->excludeFromEviction(*page);
            }
        }
    }
}

bool Block::isOrphan() const noexcept
{
    assert(prev == nullptr || (prev->next.count(key) == 1 && prev->next.at(key).get() == this));
    return prev == nullptr;
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

void Block::unlinkPage(LifeCycleId lcIdx)
{
    auto& page = storage.at(static_cast<size_t>(lcIdx));
    if (page != nullptr)
    {
        page->block = nullptr;
        page = nullptr;
    }
}

std::vector<std::shared_ptr<Block>> Block::clearStaleBlocksAfterPageUnlink(
    Block& block, LifeCycleId lcIdx, LifeCycle const& lc)
{
    std::vector<std::shared_ptr<Block>> detachedBlocks;
    assert(block.storage.at(static_cast<size_t>(lcIdx)) == nullptr);
    if (block.isOrphan())
    {
        return detachedBlocks;
    }

    // Reuse cleanup only applies to attention lifecycles.
    // SSM lifecycles are allowed in the tree but don't trigger subtree eviction.
    auto const* const alc = std::get_if<AttnLifeCycle>(&lc);
    NodeBase* pruneStart = &block;

    // If this is a full-attention block or a sink block: evict subtree.
    // Mirrors Python: pages = remove_subtree(self)
    if (alc && (!alc->windowSize.has_value() || block.ordinal() < alc->numSinkBlocks))
    {
        pruneStart = block.prev;
        detachedBlocks.push_back(removeSubtree(block));
    }

    // Prune empty tail nodes up the chain.
    // Save prev, key, and type before erasing, because the erase may destroy
    // curr when its last shared_ptr is dropped.
    Block* curr
        = pruneStart && pruneStart->type() == NodeBase::Type::kBLOCK ? static_cast<Block*>(pruneStart) : nullptr;
    while (curr && curr->next.empty() && curr->storage.at(static_cast<size_t>(lcIdx)) == nullptr)
    {
        NodeBase* prevNode = curr->prev;
        BlockKey const currKey = curr->key;
        bool const prevIsBlock = prevNode && prevNode->type() == NodeBase::Type::kBLOCK;
        if (prevNode)
        {
            auto detached = prevNode->detachNext(currKey); // may destroy curr
            assert(detached && detached.get() == curr);
            detachedBlocks.push_back(std::move(detached));
        }
        // Walk up only through Block nodes; stop at RootBlock.
        curr = prevIsBlock ? static_cast<Block*>(prevNode) : nullptr;
    }
    return detachedBlocks;
}

// ---------------------------------------------------------------------------
// addOrGetExistingBlock
// ---------------------------------------------------------------------------

std::shared_ptr<Block> addOrGetExistingBlock(
    NodeBase* prev, int numLifeCycles, std::vector<TokenIdExt> tokens, bool* isNew)
{
    assert(prev && "prev must not be null");

    // Prev must be a full block if it is a Block (mirrors Python: "prev must be a full block").
    if (prev->type() == NodeBase::Type::kBLOCK)
    {
        assert(static_cast<Block*>(prev)->isFull() && "prev must be a full block");
    }

    auto& prevNext = prev->next;
    int const tpb = prev->tokensPerBlock();
    BlockKey newKey = Block::makeKey(prev->key, tokens.data(), tokens.size());

    // Exact match: return existing block (not new — mirrors Python's UselessBlockError path).
    auto it = prevNext.find(newKey);
    if (it != prevNext.end())
    {
        if (isNew)
            *isNew = false;
        return it->second;
    }

    // Useless check: is this block's token prefix covered by a sibling?
    // Mirrors Python's UselessBlockError — throw with the sibling block.
    if (static_cast<int>(tokens.size()) < tpb)
    {
        for (auto const& [k, sibling] : prevNext)
        {
            if (sibling->tokens.size() >= tokens.size() && isPrefix(tokens, sibling->tokens))
                throw UselessBlockError(sibling);
        }
    }

    // Remove siblings whose tokens are a strict prefix of ours.
    std::vector<BlockKey> toRemove;
    for (auto const& [k, sibling] : prevNext)
    {
        if (sibling->tokens.size() < tokens.size() && isPrefix(sibling->tokens, tokens))
        {
            assert(!sibling->isFull() && sibling->key == k && sibling->next.empty());
            toRemove.push_back(k);
        }
    }
    for (auto const& k : toRemove)
    {
        auto erasedBlock = prev->detachNext(k);
        assert(erasedBlock);
        assert(erasedBlock->isOrphan() && "erased sibling must be orphan after removal");
        (void) erasedBlock;
    }

    // Create the new block. ordinal and tokensPerBlock are derived from prev inside the Block ctor.
    auto block = std::make_shared<Block>(newKey, std::move(tokens), prev, numLifeCycles);

    prevNext[newKey] = block;
    if (isNew)
        *isNew = true;
    return block;
}

// ---------------------------------------------------------------------------
// removeSubtree
// ---------------------------------------------------------------------------

std::shared_ptr<Block> removeSubtree(Block& root)
{
    Block* current = &root;
    std::shared_ptr<Block> detachedRoot;

    // Post-order traversal using prev/next links — O(1) extra space.
    // Descend to leaves first, remove on the way back up.
    // ~Block() handles page cleanup (nulling back-pointers, excludeFromEviction).
    // Mirrors Python's remove_subtree().
    while (true)
    {
        // Descend: if the current block has children, go to the first child.
        if (!current->next.empty())
        {
            current = current->next.begin()->second.get();
        }
        else
        {
            // Remove this block from its parent's next map.
            // Null prev to detach — the block may outlive the tree if held
            // externally (e.g., by nanobind/Python shared_ptr).
            NodeBase* parent = current->prev;
            BlockKey const currentKey = current->key;
            auto detached = parent->detachNext(currentKey);
            assert(detached && detached.get() == current);
            (void) detached;

            if (current == &root)
            {
                detachedRoot = std::move(detached);
                break;
            }

            assert(parent->type() == NodeBase::Type::kBLOCK);
            current = static_cast<Block*>(parent);
        }
    }
    assert(detachedRoot);
    return detachedRoot;
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

void BlockRadixTree::drainPendingRootErases() const
{
    if (mPendingRootErases.empty())
    {
        return;
    }
    // Move to local to allow re-entrancy (proposeToEraseEmptyRoot during erase).
    std::vector<BlockKey> pending;
    pending.swap(mPendingRootErases);
    auto& roots = const_cast<std::unordered_map<BlockKey, std::shared_ptr<RootBlock>>&>(mRoots);
    for (auto const& key : pending)
    {
        auto it = roots.find(key);
        // Only erase if the root exists and is still childless.
        if (it != roots.end() && it->second->next.empty())
        {
            roots.erase(it);
        }
    }
}

RootBlock& BlockRadixTree::addOrGetExisting(std::optional<int64_t> loraTaskId)
{
    drainPendingRootErases();

    BlockKey key = RootBlock::makeKey(loraTaskId);
    auto it = mRoots.find(key);
    if (it != mRoots.end())
    {
        return *it->second;
    }

    auto rb = std::make_shared<RootBlock>(loraTaskId, this);
    auto [newIt, inserted] = mRoots.emplace(key, std::move(rb));
    return *newIt->second;
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
    drainPendingRootErases();

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

    RootBlock const& root = *rootIt->second;
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

void BlockRadixTree::clear()
{
    // detachNext() may call proposeToEraseEmptyRoot, but won't modify mRoots directly.
    for (auto& [rootKey, root] : mRoots)
    {
        while (!root->next.empty())
        {
            removeSubtree(*root->next.begin()->second);
        }
    }
    assert(mRoots.size() == mPendingRootErases.size());
    mRoots.clear();
    mPendingRootErases.clear();
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
