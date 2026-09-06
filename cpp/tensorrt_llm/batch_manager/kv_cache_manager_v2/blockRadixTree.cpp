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
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <variant>

// Token hashing reinterprets TokenIdExt bytes as a raw little-endian stream (both
// the per-element and the bulk paths), so a normal token's 4 bytes equal its
// integer id. Guard the assumption at compile time (std::endian is C++20; C++17 here).
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "kv_cache_manager_v2 block hashing requires a little-endian target"
#endif

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
    static_assert(sizeof(TokenId) == sizeof(TokenIdExt));
    assert(TokenIdExt(token).tokenId() == token);
    mState.Write(reinterpret_cast<unsigned char const*>(&token), sizeof(token));
    return *this;
}

Hasher& Hasher::update(Digest const& digest)
{
    mState.Write(reinterpret_cast<unsigned char const*>(digest.data()), digest.size());
    return *this;
}

Hasher& Hasher::update(std::vector<uint8_t> const& bytes)
{
    mState.Write(bytes.data(), bytes.size());
    return *this;
}

Hasher& Hasher::update(TokenIdExt const& tokenExt)
{
    if (tokenExt.isDigest())
    {
        return update(tokenExt.digest());
    }
    return update(tokenExt.tokenId());
}

Hasher& Hasher::update(TokenIdExt const* tokens, size_t count, bool knownNoDigest)
{
    TokenIdExt const* const end = tokens + count;

    // Bulk-write a contiguous run of normal tokens [begin, stop) as one raw
    // little-endian uint32 block — the whole point of the 4-byte layout. A single
    // Write of k tokens is byte-identical to k per-token writes.
    auto const writeNormalRun = [this](TokenIdExt const* begin, TokenIdExt const* stop)
    {
        if (begin != stop)
        {
            mState.Write(
                reinterpret_cast<unsigned char const*>(begin), static_cast<size_t>(stop - begin) * sizeof(TokenIdExt));
        }
    };

    if (knownNoDigest)
    {
        TLLM_CHECK_DEBUG(std::none_of(tokens, end, [](TokenIdExt const& t) { return t.isDigest(); }));
        writeNormalRun(tokens, end);
        return *this;
    }

    // Unknown/digest-bearing: bulk-write each maximal run of normal tokens and
    // hash each (rare, multi-modal) digest on its own. All-normal collapses to a
    // single Write.
    for (TokenIdExt const* pos = tokens; pos != end;)
    {
        TokenIdExt const* const digestIt = std::find_if(pos, end, [](TokenIdExt const& t) { return t.isDigest(); });
        writeNormalRun(pos, digestIt);
        if (digestIt == end)
        {
            break;
        }
        update(digestIt->digest()); // 32 digest bytes
        pos = digestIt + 1;
    }
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
// genMultimodalCacheKeyTokens
// ---------------------------------------------------------------------------

std::vector<TokenIdExt> genMultimodalCacheKeyTokens(
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
            Digest digest;
            std::memcpy(digest.data(), multiModalDataDigest.data(), kDIGEST_LEN);
            result.emplace_back(digest);
        }
        else
        {
            result.emplace_back(TokenId{idOffset + tokenOffset + i});
        }
    }
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

// Takes raw (ptr, size) so it works uniformly over any TokenIdExt buffer
// (the query vector and the std::vector<TokenIdExt> that backs Block::tokens).
bool isPrefix(TokenIdExt const* prefix, size_t prefixLen, TokenIdExt const* full, size_t fullLen)
{
    if (prefixLen > fullLen)
    {
        return false;
    }
    for (size_t i = 0; i < prefixLen; ++i)
    {
        if (prefix[i] != full[i])
        {
            return false;
        }
    }
    return true;
}

} // anonymous namespace

BlockKey Block::makeKey(BlockKey const& prevKey, TokenIdExt const* tokens, size_t count, bool knownNoDigest)
{
    Hasher h;
    h.update(prevKey);
    h.update(tokens, count, knownNoDigest);
    return h.digest();
}

Block::Block(BlockKey k, std::vector<TokenIdExt> toks, NodeBase* prevNode)
    : NodeBase(k, prevNode->eventSink)
    , tokens(std::move(toks))
    , prev(prevNode)
    , storage(prevNode->numLifeCycles(), nullptr) // tree-wide count, derived from prev
    , mOrdinal(prevNode->ordinal() + 1)
{
    // key is a caller-supplied second source of truth (precomputed for the
    // pre-construction dedup lookup). Verify it matches what we'd derive from
    // prev + tokens so the two can never silently drift. Debug-only and fully
    // compiled out in release: this re-hashes (exactly the recomputation the param
    // exists to avoid). knownNoDigest=false lets makeKey's update() scan for digests
    // itself — correct regardless of content, so no separate scan is needed here.
    TLLM_CHECK_DEBUG(k == Block::makeKey(prevNode->key, tokens.data(), tokens.size(), /*knownNoDigest=*/false));
}

// Delegates to the tree, mirroring Python's RootBlock.num_life_cycles. Defined
// out-of-line so BlockRadixTree is complete at the point of use.
LifeCycleId RootBlock::numLifeCycles() const noexcept
{
    return tree->numLifeCycles();
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

int Block::pageCoverage(LifeCycleId lcIdx) const
{
    auto const* page = getPage(lcIdx);
    return page != nullptr ? page->numTokensInBlock : 0;
}

bool Block::holdsPage(CommittedPage const& page) const
{
    return getPage(page.lifeCycle) == &page;
}

bool Block::canReplacePage(LifeCycleId lcIdx, int numTokensInBlock) const
{
    auto const* existing = getPage(lcIdx);
    return existing == nullptr || existing->numTokensInBlock < numTokensInBlock;
}

void Block::replacePage(LifeCycleId lcIdx, CommittedPage* page)
{
    TLLM_CHECK_DEBUG(canReplacePage(lcIdx, page->numTokensInBlock));
    // Unlink first: excludeFromEviction() below may drop the eviction list's last
    // reference and destroy the page, so the slot must already be empty.
    auto* existing = unlinkPage(lcIdx);
    if (existing != nullptr && existing->scheduledForEviction())
    {
        existing->manager->excludeFromEviction(*existing);
        existing = nullptr; // May be dangling now, set to nullptr
    }
    page->block = this;
    storage.at(lcIdx) = page;
}

void Block::adoptPagesFrom(Block& other)
{
    TLLM_CHECK_DEBUG(other.ordinal() == ordinal());
    for (LifeCycleId lcIdx{0}; lcIdx < storage.size(); ++lcIdx)
    {
        auto* page = other.getPage(lcIdx);
        if (page == nullptr || !canReplacePage(lcIdx, page->numTokensInBlock))
        {
            continue;
        }
        // Clear the source slot directly rather than via unlinkPage(), which would
        // null the back-pointer replacePage() is about to overwrite.
        other.storage.at(lcIdx) = nullptr;
        replacePage(lcIdx, page);
    }
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
    while (curr && curr->next.empty()
        && std::all_of(curr->storage.begin(), curr->storage.end(), [](auto p) { return p == nullptr; }))
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

SharedPtr<Block> getExistingBlock(NodeBase* prev, BlockKey const& key, TokenIdExt const* tokens, size_t numTokens)
{
    TLLM_CHECK_DEBUG_WITH_INFO(prev, "prev must not be null");

    // Only a full block may be a parent; that is also why returning a covering sibling
    // below is safe, since only partial blocks can be covered and never become parents.
    TLLM_CHECK_DEBUG_WITH_INFO(
        prev->type() != NodeBase::Type::kBLOCK || static_cast<Block*>(prev)->isFull(), "prev must be a full block");

    auto& prevNext = prev->next;

    // Exact match. On the re-attach path this is another request having re-committed the
    // same prefix while we were detached; its key is identical, so the chain stays valid.
    auto it = prevNext.find(key);
    if (it != prevNext.end())
    {
        return it->second;
    }

    // Covered by a longer sibling: reuse it rather than insert a redundant shorter node.
    // A short page on a longer block is well defined -- CommittedPage::numTokensInBlock
    // records the span and canReplacePage() will not supersede a wider page.
    if (static_cast<int>(numTokens) < prev->tokensPerBlock())
    {
        for (auto const& [k, sibling] : prevNext)
        {
            if (sibling->tokens.size() >= numTokens
                && isPrefix(tokens, numTokens, sibling->tokens.data(), sibling->tokens.size()))
            {
                return sibling;
            }
        }
    }

    return nullptr;
}

void attachBlock(NodeBase* prev, SharedPtr<Block> const& block)
{
    TLLM_CHECK_DEBUG_WITH_INFO(prev, "prev must not be null");
    TLLM_CHECK_DEBUG(block);
    // Precondition: nothing in the tree supersedes this block, so we cannot shadow a sibling.
    TLLM_CHECK_DEBUG(getExistingBlock(prev, block->key, block->tokens.data(), block->tokens.size()) == nullptr);

    auto& prevNext = prev->next;
    auto const& tokens = block->tokens;

    // A later turn may extend a partial endpoint to this longer block, replacing the
    // partial sibling. That turn may not have a committable SWA page for this block:
    // commitMinSnapshot releases out-of-window pages, while SWA scratch reuse uses
    // temporary shared storage that is not preserved. Adopt the partial sibling's pages
    // to keep the shorter endpoint reusable, retaining each page's recorded token count
    // (see CommittedPage::numTokensInBlock).
    std::vector<BlockKey> toRemove;
    for (auto const& [k, sibling] : prevNext)
    {
        if (sibling->tokens.size() < tokens.size()
            && isPrefix(sibling->tokens.data(), sibling->tokens.size(), tokens.data(), tokens.size()))
        {
            TLLM_CHECK_DEBUG(!sibling->isFull() && sibling->key == k && sibling->next.empty());
            toRemove.push_back(k);
        }
    }
    // Two covered siblings would be prefixes of each other; the insertion logic
    // would already have replaced the shorter one.
    TLLM_CHECK_DEBUG(toRemove.size() <= 1);

    // Redundant for a freshly constructed block; the re-attach path needs it to restore
    // the link the prune walk cleared.
    block->prev = prev;
    // Keep the parent attached while covered children are replaced. Adding the replacement
    // first prevents detachNext() from pruning an emptied RootBlock out of the tree.
    prevNext[block->key] = block;

    for (auto const& k : toRemove)
    {
        auto erasedBlock = prev->detachNext(k);
        TLLM_CHECK_DEBUG(erasedBlock);
        block->adoptPagesFrom(*erasedBlock);
        TLLM_CHECK_DEBUG_WITH_INFO(erasedBlock->isOrphan(), "erased sibling must be orphan after removal");
    }
}

SharedPtr<Block> addOrGetExistingBlock(NodeBase* prev, std::vector<TokenIdExt> tokens, bool knownNoDigest, bool* isNew)
{
    TLLM_CHECK_DEBUG_WITH_INFO(prev, "prev must not be null");

    BlockKey const newKey = Block::makeKey(prev->key, tokens.data(), tokens.size(), knownNoDigest);

    // Query first so a block we would discard is never built. Must precede the move below.
    if (auto existing = getExistingBlock(prev, newKey, tokens.data(), tokens.size()))
    {
        if (isNew)
            *isNew = false;
        return existing;
    }

    // ordinal, tokensPerBlock, and numLifeCycles are all derived from prev.
    // Block stores the tokens as a plain vector (moved in).
    auto block = makeShared<Block>(newKey, std::move(tokens), prev);
    attachBlock(prev, block);
    if (isNew)
        *isNew = true;
    return block;
}

SharedPtr<Block> attachOrGetExistingBlock(NodeBase* prev, SharedPtr<Block> block, bool* attached)
{
    TLLM_CHECK_DEBUG(block);

    if (auto existing = getExistingBlock(prev, block->key, block->tokens.data(), block->tokens.size()))
    {
        // Someone else installed an equivalent block while we held ours (on the re-attach
        // path, another request re-committed this prefix during our orphan window). Hand
        // over any pages the winner lacks rather than dropping them: ours are still valid
        // for the same tokens, and adoptPagesFrom() keeps whichever page covers more.
        if (existing != block && existing->ordinal() == block->ordinal())
        {
            existing->adoptPagesFrom(*block);
        }
        if (attached)
            *attached = false;
        return existing;
    }

    attachBlock(prev, block);
    if (attached)
        *attached = true;
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
    while (true)
    {
        // Descend: if the current block has children, go to the first child.
        if (!current->next.empty())
        {
            current = current->next.begin()->second.get();
        }
        else
        {
            // Remove this block from its parent's next map and null prev to detach it.
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
    // Clear all roots (which will drop all blocks without external owners).
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

} // anonymous namespace

std::vector<BlockRadixTree::MatchResult> BlockRadixTree::matchTokenPath(
    ReuseScope const& reuseScope, TokenSpan tokens, bool knownNoDigest, bool enablePartialMatch) const
{
    drainPendingRootErases();

    std::vector<MatchResult> results;

    // Lazily compute one key per iteration — no wasted hashing on early miss.
    auto gen = sequenceToBlockchainKeys(mTokensPerBlock, reuseScope, tokens.begin(), tokens.size(), knownNoDigest);

    // First step is the root key (empty token range).
    auto rootStep = gen();
    if (!rootStep)
        return results;
    auto rootIt = mRoots.find(rootStep->key);
    if (rootIt == mRoots.end())
        return results;

    RootBlock const& root = *rootIt->second;
    std::unordered_map<BlockKey, SharedPtr<Block>> const* currentNext = &root.next;
    // Token range of the first unmatched block, captured on miss for the partial pass.
    HalfOpenRange<size_t> missedRange;
    bool missed = false;

    // Each step carries the block's key and its token range — no need to re-derive
    // block boundaries here.
    while (auto step = gen())
    {
        auto blockIt = currentNext->find(step->key);
        if (blockIt == currentNext->end())
        {
            missedRange = step->tokens;
            missed = true;
            break;
        }
        Block* block = blockIt->second.get();
        results.push_back({block, static_cast<int>(step->tokens.length())});
        currentNext = &block->next;
    }

    // Partial match in children of current node.
    if (missed && enablePartialMatch)
    {
        auto [best, bestMatch]
            = findBestPartialMatchInNextNodes(*currentNext, tokens.begin() + missedRange.beg, missedRange.length());
        if (best)
            results.push_back({best, bestMatch});
    }

    return results;
}

std::vector<BlockRadixTree::MatchResult> BlockRadixTree::pruneMatch(
    std::vector<MatchResult> matched, std::optional<LifeCycleId> ssmLcId) const
{
    // All blocks except the last must be fully matched (mirrors Python: matched[:-1]).
    TLLM_CHECK_DEBUG(matched.size() <= 1
        || std::all_of(matched.begin(), matched.end() - 1,
            [this](auto const& m) { return m.numMatchedTokens == mTokensPerBlock; }));

    auto attnLcs = mLifeCycles.attentionLifeCycles();

    // Fixed-point loop: SSM may select an earlier exact snapshot, while attention may
    // shorten the match to the coverage of a required page. Every retry strictly
    // shortens the match, so the loop terminates.
    while (!matched.empty())
    {
        // Check SSM snapshot availability first: truncating to the last reusable SSM
        // snapshot changes the matched length that all the attention checks use.
        if (ssmLcId.has_value())
        {
            int ssmTrunc = 0;
            int ssmMatchLen = 0;
            for (int i = static_cast<int>(matched.size()) - 1; i >= 0; --i)
            {
                auto const& entry = matched[static_cast<size_t>(i)];
                // An SSM page holds the recurrent state after exactly this many tokens,
                // so reuse must stop there instead of anywhere inside the block.
                int const snapshotLen = entry.block->pageCoverage(*ssmLcId);
                if (snapshotLen > 0 && entry.numMatchedTokens >= snapshotLen)
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

        // Only pages that are active at this candidate endpoint constrain attention
        // reuse. Full attention requires every block. SWA requires sink blocks and the
        // trailing window, but not the stale blocks between them. In particular, at an
        // exact block boundary with windowSize=1, every historical block is stale.
        int const numTok = numMatchedTokens(matched, mTokensPerBlock);
        bool shortened = false;
        for (auto [lcId, attn] : attnLcs)
        {
            auto const staleRange = attn->getStaleRange(numTok, mTokensPerBlock);
            int const staleBeg = staleRange.beg.value();
            int const staleEnd = staleRange.end.value();
            int const numMatchedBlocks = static_cast<int>(matched.size());
            for (int i = 0; i < numMatchedBlocks; ++i)
            {
                // Mirrors Python's chain(range(stale.beg), range(stale.end, len(matched))).
                if (staleBeg <= i && i < staleEnd)
                {
                    continue;
                }
                int const numMatched = matched[static_cast<size_t>(i)].numMatchedTokens;
                int const coverage = matched[static_cast<size_t>(i)].block->pageCoverage(lcId);
                if (coverage >= numMatched)
                {
                    continue;
                }
                if (coverage > 0)
                {
                    matched.resize(static_cast<size_t>(i) + 1);
                    matched.back().numMatchedTokens = coverage;
                }
                else
                {
                    matched.resize(static_cast<size_t>(i));
                }
                shortened = true;
                break;
            }
            if (shortened)
            {
                break;
            }
        }
        if (!shortened)
        {
            break;
        }
    }

    return matched;
}

BlockRadixTree::ReuseMatch BlockRadixTree::match(
    ReuseScope const& reuseScope, TokenSpan tokens, bool knownNoDigest, bool enablePartialMatch) const
{
    auto rawMatched = matchTokenPath(reuseScope, tokens, knownNoDigest, enablePartialMatch);
    // Content-divergence depth is measured before page or recurrent-snapshot pruning.
    int const numReusableTokensBeforePruning = numMatchedTokens(rawMatched, mTokensPerBlock);
    auto const ssmLcId = mLifeCycles.ssmLifeCycleId();
    // Diagnostic only: re-prune ignoring recurrent-snapshot availability to get
    // the prefix the attention pages alone support. Only hybrid models pay for
    // the second pass; without an SSM life cycle the two results are identical.
    std::optional<int> numReusableTokensBeforeHybridPruning;
    if (ssmLcId.has_value())
    {
        numReusableTokensBeforeHybridPruning = numMatchedTokens(pruneMatch(rawMatched, std::nullopt), mTokensPerBlock);
    }
    auto const matched = pruneMatch(std::move(rawMatched), ssmLcId);
    ReuseMatch result{};
    result.numTokens = numMatchedTokens(matched, mTokensPerBlock);
    result.numLookupTokens = static_cast<int>(tokens.size());
    result.numReusableTokensBeforeHybridPruning = numReusableTokensBeforeHybridPruning.value_or(result.numTokens);
    result.numReusableTokensBeforePruning = numReusableTokensBeforePruning;
    result.blocks.reserve(BlockOrdinal{static_cast<int>(matched.size())});
    for (auto const& match : matched)
    {
        result.blocks.push_back(match.block);
    }
    return result;
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
