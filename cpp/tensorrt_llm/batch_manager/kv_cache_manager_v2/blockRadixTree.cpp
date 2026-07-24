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
#include "kv_cache_manager_v2/utils/math.h"

#include "sha256.h"

#include "tensorrt_llm/common/assert.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <variant>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// ReuseScope
// ---------------------------------------------------------------------------

// Serialized layout of a ReuseScope, consumed by Hasher::update(ReuseScope).
// Must stay byte-identical to the Python ReuseScope.to_bytes(): a mask byte
// followed by one little-endian uint64 per present field (signed=False).
template <typename Emit>
static void emitReuseScopeBytes(ReuseScope const& scope, Emit&& emit)
{
    uint8_t mask = 0;
    if (scope.loraId.has_value())
    {
        mask |= 1U << 0;
    }
    if (scope.salt.has_value())
    {
        mask |= 1U << 1;
    }
    emit(&mask, sizeof(mask));
    if (scope.loraId.has_value())
    {
        std::uint64_t const value = *scope.loraId;
        emit(reinterpret_cast<uint8_t const*>(&value), sizeof(value));
    }
    if (scope.salt.has_value())
    {
        std::uint64_t const value = *scope.salt;
        emit(reinterpret_cast<uint8_t const*>(&value), sizeof(value));
    }
}

// ---------------------------------------------------------------------------
// Hasher
// ---------------------------------------------------------------------------

namespace
{
// Select the best available SHA-256 back-end (x86 SHA-NI, ARMv8 crypto, SSE4,
// AVX2) once, falling back to a portable scalar transform. CSHA256 dispatches
// through function pointers that SHA256AutoDetect() installs; the magic-static
// guarantees this runs exactly once and is thread-safe.
void ensureSha256Detected()
{
    static std::string const impl = SHA256AutoDetect();
    (void) impl;
}
} // namespace

static void hashInt64(CSHA256& h, int64_t v)
{
    unsigned char buf[8];
    auto const unsignedValue = static_cast<uint64_t>(v);
    for (int i = 0; i < 8; ++i)
    {
        buf[i] = static_cast<unsigned char>((unsignedValue >> (8 * i)) & 0xFFU);
    }
    h.Write(buf, sizeof(buf));
}

Hasher::Hasher()
{
    ensureSha256Detected();
}

Hasher::Hasher(ReuseScope const& seed)
{
    ensureSha256Detected();
    update(seed);
}

Hasher& Hasher::update(ReuseScope const& scope)
{
    // Feed the serialized ReuseScope straight into the hash state without any
    // intermediate heap buffer.
    emitReuseScopeBytes(scope, [this](uint8_t const* data, size_t count) { mState.Write(data, count); });
    return *this;
}

Hasher& Hasher::update(TokenId token)
{
    hashInt64(mState, static_cast<int64_t>(token));
    return *this;
}

Hasher& Hasher::update(BlockKey const& key)
{
    mState.Write(reinterpret_cast<unsigned char const*>(key.data()), key.size());
    return *this;
}

Hasher& Hasher::update(std::vector<uint8_t> const& bytes)
{
    mState.Write(bytes.data(), bytes.size());
    return *this;
}

Hasher& Hasher::update(TokenIdExt const& tokenExt)
{
    std::visit(
        [this](auto const& v)
        {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, TokenId>)
                hashInt64(mState, static_cast<int64_t>(v));
            else
                mState.Write(reinterpret_cast<unsigned char const*>(v.data()), v.size());
        },
        tokenExt);
    return *this;
}

Hasher& Hasher::update(TokenIdExt const* tokens, size_t count)
{
    // Python uses array("Q", data).tobytes() to reduce per-token interpreter
    // overhead.  In C++ the compiler inlines each update() call, so the loop
    // is already optimal; batching would only add a heap allocation.
    for (size_t i = 0; i < count; ++i)
        update(tokens[i]);
    return *this;
}

BlockKey Hasher::digest() const
{
    // Finalize into a 32-byte key. CSHA256::Finalize consumes the state, so we
    // finalize a copy to keep this method const and allow further updates.
    BlockKey out;
    CSHA256 copy = mState;
    copy.Finalize(reinterpret_cast<unsigned char*>(out.data()));
    return out;
}

// ---------------------------------------------------------------------------
// genMultiModalTokens
// ---------------------------------------------------------------------------

std::vector<TokenIdExt> genMultiModalTokens(
    int idOffset, std::vector<uint8_t> const& multiModalDataDigest, int numTokens, int tokenOffset)
{
    TLLM_CHECK_DEBUG(numTokens > 0);
    TLLM_CHECK_DEBUG(tokenOffset >= 0);
    TLLM_CHECK_DEBUG(multiModalDataDigest.size() == kDIGEST_LEN);
    std::vector<TokenIdExt> result;
    result.reserve(static_cast<size_t>(numTokens));
    for (int i = 0; i < numTokens; ++i)
    {
        if (tokenOffset + i == 0)
        {
            Digest d;
            std::memcpy(d.data(), multiModalDataDigest.data(), kDIGEST_LEN);
            result.emplace_back(DigestToken(d));
        }
        else
        {
            result.emplace_back(TokenId(idOffset + tokenOffset + i));
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// makeBlockchainKeyGenerator — lazy key generator.
// Returns a callable that yields one BlockKey per call (nullopt when done).
// First call yields root entry (empty token block). Mirrors Python's generator.
// ---------------------------------------------------------------------------

static auto makeBlockchainKeyGenerator(
    int tokensPerBlock, ReuseScope reuseScope, TokenIdExt const* tokens, size_t numTokens)
{
    // digest carries the running hash from the previous block.
    BlockKey digest = Hasher(reuseScope).digest();
    // ordinal = -1: next call yields root (reuseScope digest).
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
    int tokensPerBlock, ReuseScope const& reuseScope, std::vector<TokenIdExt> const& tokens)
{
    std::vector<BlockKey> result;
    auto gen = makeBlockchainKeyGenerator(tokensPerBlock, reuseScope, tokens.data(), tokens.size());
    while (auto key = gen())
        result.push_back(*key);
    return result;
}

// ---------------------------------------------------------------------------
// RootBlock
// ---------------------------------------------------------------------------

BlockKey RootBlock::makeKey(ReuseScope const& reuseScope)
{
    return Hasher(reuseScope).digest();
}

RootBlock::RootBlock(ReuseScope reuseScope_, BlockRadixTree* treePtr)
    : NodeBase(makeKey(reuseScope_), treePtr->eventSink().get())
    , reuseScope(std::move(reuseScope_))
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

SharedPtr<Block> NodeBase::detachNext(BlockKey const& blockKey)
{
    auto it = next.find(blockKey);
    if (it == next.end())
    {
        return nullptr;
    }

    auto block = it->second;
    block->prev = nullptr;
    next.erase(it);
    if (eventSink)
    {
        eventSink->addRemovedBlock(block->key);
    }
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

Block::Block(BlockKey k, std::vector<TokenIdExt> toks, NodeBase* prevNode, LifeCycleId numLifeCycles)
    : NodeBase(k, prevNode->eventSink)
    , tokens(std::move(toks))
    , prev(prevNode)
    , storage(numLifeCycles, nullptr)
    , mOrdinal(prevNode->ordinal() + 1)
{
}

int Block::tokensPerBlock() const noexcept
{
    TLLM_CHECK_DEBUG_WITH_INFO(prev, "Block must have a prev");
    // Mirrors Python: prev.tokens_per_block if isinstance(prev, RootBlock) else len(prev.tokens)
    if (prev->type() == Type::kROOT_BLOCK)
        return prev->tokensPerBlock();
    return static_cast<int>(static_cast<Block const*>(prev)->tokens.size());
}

void Block::releasePages()
{
    // Mirrors Python Block._release_pages(): for each stored page, if alive and
    // DROPPABLE and scheduled for eviction, exclude from eviction. Also null out
    // the page's back-pointer so that CommittedPage::~CommittedPage() doesn't
    // attempt cleanup through this Block. Idempotent — storage is empty afterwards.
    for (LifeCycleId lcIdx{0}; lcIdx < storage.size(); ++lcIdx)
    {
        auto const page = storage[lcIdx];
        if (page != nullptr)
        {
            TLLM_CHECK_DEBUG(page->block == this);
            unlinkPage(lcIdx);
            if (page->status() == PageStatus::DROPPABLE && page->scheduledForEviction())
            {
                page->manager->excludeFromEviction(*page);
            }
        }
    }
}

Block::~Block()
{
    releasePages();
}

bool Block::isOrphan() const noexcept
{
    TLLM_CHECK_DEBUG(prev == nullptr || (prev->next.count(key) == 1 && prev->next.at(key).get() == this));
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

CommittedPage* Block::unlinkPage(LifeCycleId lcIdx, CommittedPage* expectedPage)
{
    auto& slot = storage.at(lcIdx);
    CommittedPage* page = slot;
    if (page == nullptr)
        return nullptr;
    if (expectedPage != nullptr && page != expectedPage)
        return nullptr;
    page->block = nullptr;
    slot = nullptr;
    return page;
}

std::vector<SharedPtr<Block>> Block::clearStaleBlocksAfterPageUnlink(
    Block& block, LifeCycleId lcIdx, LifeCycle const& lc)
{
    std::vector<SharedPtr<Block>> detachedBlocks;
    TLLM_CHECK_DEBUG(block.storage.at(lcIdx) == nullptr);
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
    if (alc && (!alc->windowSize.has_value() || block.ordinal() < BlockOrdinal{alc->numSinkBlocks}))
    {
        pruneStart = block.prev;
        detachedBlocks.push_back(removeSubtree(block));
    }
    else if (block.eventSink)
    {
        block.eventSink->addRemovedLifeCycle(block.key, lcIdx);
    }

    // Prune empty tail nodes up the chain.
    // Save prev, key, and type before erasing, because the erase may destroy
    // curr when its last shared_ptr is dropped.
    Block* curr
        = pruneStart && pruneStart->type() == NodeBase::Type::kBLOCK ? static_cast<Block*>(pruneStart) : nullptr;
    while (curr && curr->next.empty() && curr->storage.at(lcIdx) == nullptr)
    {
        NodeBase* prevNode = curr->prev;
        BlockKey const currKey = curr->key;
        bool const prevIsBlock = prevNode && prevNode->type() == NodeBase::Type::kBLOCK;
        if (prevNode)
        {
            auto detached = prevNode->detachNext(currKey); // may destroy curr
            TLLM_CHECK_DEBUG(detached && detached.get() == curr);
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

SharedPtr<Block> addOrGetExistingBlock(
    NodeBase* prev, LifeCycleId numLifeCycles, std::vector<TokenIdExt> tokens, bool* isNew)
{
    TLLM_CHECK_DEBUG_WITH_INFO(prev, "prev must not be null");

    // Prev must be a full block if it is a Block (mirrors Python: "prev must be a full block").
    if (prev->type() == NodeBase::Type::kBLOCK)
    {
        TLLM_CHECK_DEBUG_WITH_INFO(static_cast<Block*>(prev)->isFull(), "prev must be a full block");
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
            TLLM_CHECK_DEBUG(!sibling->isFull() && sibling->key == k && sibling->next.empty());
            toRemove.push_back(k);
        }
    }
    for (auto const& k : toRemove)
    {
        auto erasedBlock = prev->detachNext(k);
        TLLM_CHECK_DEBUG(erasedBlock);
        TLLM_CHECK_DEBUG_WITH_INFO(erasedBlock->isOrphan(), "erased sibling must be orphan after removal");
        (void) erasedBlock;
    }

    // Create the new block. ordinal and tokensPerBlock are derived from prev inside the Block ctor.
    auto block = makeShared<Block>(newKey, std::move(tokens), prev, numLifeCycles);

    prevNext[newKey] = block;
    if (isNew)
        *isNew = true;
    return block;
}

// ---------------------------------------------------------------------------
// removeSubtree
// ---------------------------------------------------------------------------

SharedPtr<Block> removeSubtree(Block& root)
{
    Block* current = &root;
    SharedPtr<Block> detachedRoot;

    // Post-order traversal using prev/next links — O(1) extra space.
    // Descend to leaves first, remove on the way back up.
    // Each block's pages are reclaimed eagerly via releasePages() while the
    // StorageManager is still alive, rather than deferring to ~Block(): an external
    // reference can keep a Block alive past StorageManager teardown, after which
    // page->manager would be dangling. Mirrors Python's remove_subtree().
    while (true)
    {
        // Descend: if the current block has children, go to the first child.
        if (!current->next.empty())
        {
            current = current->next.begin()->second.get();
        }
        else
        {
            current->releasePages();
            // Remove this block from its parent's next map.
            // Null prev to detach — the block may outlive the tree if held
            // externally (e.g., by nanobind/Python shared_ptr).
            NodeBase* parent = current->prev;
            BlockKey const currentKey = current->key;
            auto detached = parent->detachNext(currentKey);
            TLLM_CHECK_DEBUG(detached && detached.get() == current);
            (void) detached;

            if (current == &root)
            {
                detachedRoot = std::move(detached);
                break;
            }

            TLLM_CHECK_DEBUG(parent->type() == NodeBase::Type::kBLOCK);
            current = static_cast<Block*>(parent);
        }
    }
    TLLM_CHECK_DEBUG(detachedRoot);
    return detachedRoot;
}

// ---------------------------------------------------------------------------
// BlockRadixTree
// ---------------------------------------------------------------------------

BlockRadixTree::BlockRadixTree(
    LifeCycleRegistry const& lifeCycles, int tokensPerBlock, std::shared_ptr<EventSink> eventSink)
    : mLifeCycles(lifeCycles)
    , mTokensPerBlock(tokensPerBlock)
    , mEventSink(std::move(eventSink))
{
}

BlockRadixTree::~BlockRadixTree()
{
    // Clear all roots (which will drop all blocks).
    mRoots.clear();
}

LifeCycleId BlockRadixTree::numLifeCycles() const noexcept
{
    return mLifeCycles.size();
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
    auto& roots = const_cast<std::unordered_map<BlockKey, SharedPtr<RootBlock>>&>(mRoots);
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

RootBlock& BlockRadixTree::addOrGetExisting(ReuseScope const& reuseScope)
{
    drainPendingRootErases();

    BlockKey key = RootBlock::makeKey(reuseScope);
    auto it = mRoots.find(key);
    if (it != mRoots.end())
    {
        return *it->second;
    }

    auto rb = makeShared<RootBlock>(reuseScope, this);
    auto [newIt, inserted] = mRoots.emplace(key, std::move(rb));
    return *newIt->second;
}

// Among all child nodes, find the one whose tokens have the longest leading match.
// Returns (block, numMatchedTokens) or (nullptr, 0) if no match.
// Mirrors Python's find_best_partial_match_in_next_nodes().
std::pair<Block*, int> findBestPartialMatchInNextNodes(
    std::unordered_map<BlockKey, SharedPtr<Block>> const& nextMap, TokenIdExt const* tokens, size_t tokenCount)
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

namespace
{

int numMatchedTokens(std::vector<BlockRadixTree::MatchResult> const& matched, int tokensPerBlock)
{
    if (matched.empty())
    {
        return 0;
    }
    return tokensPerBlock * (static_cast<int>(matched.size()) - 1) + matched.back().numMatchedTokens;
}

bool hasPage(Block const& block, LifeCycleId lcId)
{
    return block.storage.at(lcId) != nullptr;
}

} // anonymous namespace

std::vector<BlockRadixTree::MatchResult> BlockRadixTree::matchTokenPath(
    ReuseScope const& reuseScope, std::vector<TokenIdExt> const& tokens, bool enablePartialMatch) const
{
    drainPendingRootErases();

    std::vector<MatchResult> results;

    // Lazily compute one key per iteration — no wasted hashing on early miss.
    auto gen = makeBlockchainKeyGenerator(mTokensPerBlock, reuseScope, tokens.data(), tokens.size());

    // First key is the root key.
    auto rootKey = gen();
    if (!rootKey)
        return results;
    auto rootIt = mRoots.find(*rootKey);
    if (rootIt == mRoots.end())
        return results;

    RootBlock const& root = *rootIt->second;
    std::unordered_map<BlockKey, SharedPtr<Block>> const* currentNext = &root.next;
    // ordinal tracks which block we're on (0-based, after root).
    BlockOrdinal ordinal{0};
    bool missed = false;

    while (auto key = gen())
    {
        auto blockIt = currentNext->find(*key);
        if (blockIt == currentNext->end())
        {
            missed = true;
            break;
        }
        size_t beg = toSizeT(ordinal) * static_cast<size_t>(mTokensPerBlock);
        int numTokens = static_cast<int>(std::min(static_cast<size_t>(mTokensPerBlock), tokens.size() - beg));
        Block* block = blockIt->second.get();
        results.push_back({block, numTokens});
        currentNext = &block->next;
        ordinal++;
    }

    // Partial match in children of current node.
    if (missed && enablePartialMatch)
    {
        size_t beg = toSizeT(ordinal) * static_cast<size_t>(mTokensPerBlock);
        size_t missedCount = std::min(static_cast<size_t>(mTokensPerBlock), tokens.size() - beg);
        auto [best, bestMatch] = findBestPartialMatchInNextNodes(*currentNext, tokens.data() + beg, missedCount);
        if (best)
            results.push_back({best, bestMatch});
    }

    return results;
}

BlockRadixTree::ReuseMatch BlockRadixTree::pruneMatch(std::vector<MatchResult> matched) const
{
    // All blocks except the last must be fully matched (mirrors Python: matched[:-1]).
    TLLM_CHECK_DEBUG(matched.size() <= 1
        || std::all_of(matched.begin(), matched.end() - 1,
            [this](auto const& m) { return m.numMatchedTokens == mTokensPerBlock; }));

    auto attnLcs = mLifeCycles.attentionLifeCycles();

    // Full-attention layers require pages on every matched block.
    std::vector<LifeCycleId> fullAttnLcList;
    for (auto [lcId, attn] : attnLcs)
    {
        if (!attn->windowSize.has_value())
        {
            fullAttnLcList.push_back(lcId);
        }
    }
    if (!fullAttnLcList.empty())
    {
        int n = findIndex(matched.begin(), matched.end(),
            [&](auto const& match)
            {
                return std::any_of(fullAttnLcList.begin(), fullAttnLcList.end(),
                    [&](LifeCycleId lcId) { return !hasPage(*match.block, lcId); });
            });
        matched.resize(static_cast<size_t>(n));
    }

    std::vector<std::pair<LifeCycleId, AttnLifeCycle const*>> swaLcs;
    for (auto [lcId, attn] : attnLcs)
    {
        if (attn->windowSize.has_value())
        {
            swaLcs.push_back({lcId, attn});
        }
    }

    // SWA sink blocks must all be available.
    for (auto [lcId, attn] : swaLcs)
    {
        int const sinkBlocks = attn->numSinkBlocks;
        int const limit = std::min(sinkBlocks, static_cast<int>(matched.size()));
        int n = findIndex(matched.begin(), matched.begin() + limit,
            [&, lcId = lcId](auto const& match) { return !hasPage(*match.block, lcId); });
        if (n < sinkBlocks)
        {
            matched.resize(static_cast<size_t>(n));
        }
    }

    auto ssmLcId = mLifeCycles.ssmLifeCycleId();
    while (!matched.empty())
    {
        if (ssmLcId.has_value())
        {
            // Truncate to the last block whose SSM snapshot is reusable at that
            // block's matched-token count, then clamp the tail entry's matched
            // token count to the snapshot length (mirrors _block_radix_tree.py).
            int ssmTrunc = 0;
            int ssmMatchLen = 0;
            for (int i = static_cast<int>(matched.size()) - 1; i >= 0; --i)
            {
                CommittedPage* page = matched[static_cast<size_t>(i)].block->storage.at(*ssmLcId);
                if (page == nullptr)
                {
                    continue;
                }
                auto* ssmPage = dynamic_cast<SsmCommittedPage*>(page);
                TLLM_CHECK_DEBUG(ssmPage != nullptr);
                int const snapshotLen = ssmPage->numTokensInBlock;
                if (matched[static_cast<size_t>(i)].numMatchedTokens >= snapshotLen)
                {
                    ssmTrunc = i + 1;
                    ssmMatchLen = snapshotLen;
                    break;
                }
            }
            matched.resize(static_cast<size_t>(ssmTrunc));
            if (matched.empty())
            {
                break;
            }
            matched.back().numMatchedTokens = ssmMatchLen;
        }

        int const numTok = numMatchedTokens(matched, mTokensPerBlock);
        bool trimmed = false;
        for (auto [lcId, attn] : swaLcs)
        {
            int n = findIndex(matched.rbegin(), matched.rend(),
                [&, lcId = lcId](auto const& match) { return hasPage(*match.block, lcId); });
            if (n != 0)
            {
                matched.resize(matched.size() - static_cast<size_t>(n));
                trimmed = true;
                break;
            }

            auto staleRange = attn->getStaleRange(numTok, mTokensPerBlock);
            BlockOrdinal const staleEnd = staleRange.end;
            if (staleEnd < BlockOrdinal{static_cast<int>(matched.size())})
            {
                auto tailBegin = matched.begin() + static_cast<ptrdiff_t>(toSizeT(staleEnd));
                int nMissing = findIndex(matched.rbegin(), std::make_reverse_iterator(tailBegin),
                    [&, lcId = lcId](auto const& match) { return !hasPage(*match.block, lcId); });
                if (BlockOrdinal{static_cast<int>(matched.size()) - nMissing} > staleEnd)
                {
                    matched.resize(matched.size() - static_cast<size_t>(nMissing) - 1);
                    trimmed = true;
                    break;
                }
            }
        }
        if (!trimmed)
        {
            break;
        }
    }

    ReuseMatch result;
    result.numTokens = numMatchedTokens(matched, mTokensPerBlock);
    result.blocks.reserve(matched.size());
    for (auto const& match : matched)
    {
        result.blocks.push_back(match.block);
    }
    return result;
}

BlockRadixTree::ReuseMatch BlockRadixTree::match(
    ReuseScope const& reuseScope, std::vector<TokenIdExt> const& tokens, bool enablePartialMatch) const
{
    return pruneMatch(matchTokenPath(reuseScope, tokens, enablePartialMatch));
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
    TLLM_CHECK_DEBUG(mRoots.size() == mPendingRootErases.size());
    mRoots.clear();
    mPendingRootErases.clear();
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
