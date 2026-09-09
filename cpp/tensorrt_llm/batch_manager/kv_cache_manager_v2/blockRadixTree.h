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

#pragma once

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/eventSink.h"
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/tokenIdExt.h"
#include "kv_cache_manager_v2/utils/math.h" // HalfOpenRange
#include "kv_cache_manager_v2/utils/sharedPtr.h"

#include "sha256.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// Forward declarations
class CommittedPage;
class BlockRadixTree;
struct NodeBase;
struct RootBlock;
struct Block;

// ---------------------------------------------------------------------------
// ReuseScope — per-request namespace for prefix reuse.
// Mirrors Python's ReuseScope(lora_id, salt).
// ---------------------------------------------------------------------------
struct ReuseScope
{
    std::optional<LoraTaskIdType> loraId;
    std::optional<std::uint64_t> salt;

    bool operator==(ReuseScope const& other) const noexcept
    {
        return loraId == other.loraId && salt == other.salt;
    }
};

// ---------------------------------------------------------------------------
// BlockKey — SHA-256 digest (32 bytes), used as radix-tree node identifier.
// Matches Python's hashlib.sha256 32-byte digest.
//
// SECURITY INVARIANT: the block hash MUST remain cryptographically
// collision-resistant and >= 256-bit. The radix tree is a globally shared,
// cross-request/cross-tenant cache index, prefix matching is decided purely by
// digest equality with NO re-verification of the underlying tokens, and the
// hashed input (tokens, the user-supplied cache_salt in ReuseScope, multimodal
// content bytes) is attacker-influenceable. A hash collision therefore silently
// reuses another request's KV blocks (cross-request corruption / data leak),
// and tenant isolation via cache_salt relies entirely on this hash's collision
// resistance. Do NOT substitute a non-cryptographic hash (xxHash, HighwayHash,
// City, ...) or truncate below 256 bits without first adding a token-content
// equality check on match.
// ---------------------------------------------------------------------------
using BlockKey = Digest;
static_assert(kDIGEST_LEN == CSHA256::OUTPUT_SIZE); // 32 bytes

// ---------------------------------------------------------------------------
// Hasher — thin wrapper around SHA-256 (CSHA256) for incremental digests.
// Mirrors Python's Hasher class (hashlib.sha256). See the SECURITY INVARIANT on
// BlockKey above before changing the hash algorithm or digest width.
// ---------------------------------------------------------------------------
class Hasher
{
public:
    Hasher();
    explicit Hasher(ReuseScope const& seed);

    Hasher& update(TokenId token);
    Hasher& update(Digest const& digest); // 32 raw bytes (BlockKey is a Digest alias)
    Hasher& update(ReuseScope const& scope);
    Hasher& update(std::vector<uint8_t> const& bytes);
    Hasher& update(TokenIdExt const& tokenExt);
    // knownNoDigest: caller guarantees the range holds no digest, enabling the bulk
    // fast path. Only pass true from external knowledge (request/model text_only) — never
    // from scanning the tokens, since false already makes update() scan internally.
    Hasher& update(TokenIdExt const* tokens, size_t count, bool knownNoDigest = false);

    BlockKey digest() const;

private:
    CSHA256 mState;
};

// One step of the blockchain-key generator: a block's key plus the half-open token
// index range [beg, end) it covers ([0, 0) for the root). Carrying the range lets
// callers slice the tokens without re-deriving block boundaries, mirroring Python's
// (token_block, key) pairs.
struct BlockchainKeyStep
{
    BlockKey key;
    HalfOpenRange<size_t> tokens;
};

// ---------------------------------------------------------------------------
// sequenceToBlockchainKeys — lazy per-block key generator.
// Returns a callable yielding one BlockchainKeyStep per call (nullopt when done):
// the first call yields the root (reuseScope digest, empty [0,0) range), then one
// step per tokensPerBlock chunk chained on the previous digest. Mirrors Python's
// sequence_to_blockchain_keys(). Lazy so a caller that stops early (e.g. on the
// first mismatch) skips the remaining hashing. Inline so both the tree
// (matchTokenPath) and the nanobind layer drive the same generator.
// knownNoDigest: from external text_only knowledge, never a scan (see Hasher::update).
// ---------------------------------------------------------------------------
inline auto sequenceToBlockchainKeys(
    int tokensPerBlock, ReuseScope reuseScope, TokenIdExt const* tokens, size_t numTokens, bool knownNoDigest = false)
{
    // digest carries the running hash from the previous block.
    BlockKey digest = Hasher(reuseScope).digest();
    // ordinal = -1: next call yields root (reuseScope digest).
    // ordinal >= 0: next call yields key for tokens[ordinal*tpb .. (ordinal+1)*tpb).
    int ordinal = -1;

    return [=]() mutable -> std::optional<BlockchainKeyStep>
    {
        if (ordinal == -1)
        {
            ordinal++;
            return BlockchainKeyStep{digest, {}}; // root key, empty [0,0) token range
        }

        size_t beg = static_cast<size_t>(ordinal) * static_cast<size_t>(tokensPerBlock);
        if (beg >= numTokens)
            return std::nullopt;

        size_t end = std::min(beg + static_cast<size_t>(tokensPerBlock), numTokens);

        Hasher h;
        h.update(digest);
        h.update(tokens + beg, end - beg, knownNoDigest);
        digest = h.digest();

        ordinal++;
        return BlockchainKeyStep{digest, {beg, end}};
    };
}

// Generate multi-modal token IDs (mirrors gen_multimodal_cache_key_tokens in Python).
std::vector<TokenIdExt> genMultimodalCacheKeyTokens(
    int idOffset, std::vector<uint8_t> const& multiModalDataDigest, int numTokens, int tokenOffset = 0);

// ---------------------------------------------------------------------------
// NodeBase — common base for RootBlock and Block (nodes in the radix tree).
// Holds shared fields: key, next map, ordinal, and tokens-per-block.
// Mirrors Python's common interface between RootBlock and Block.
// ---------------------------------------------------------------------------
struct NodeBase
{
    enum class Type : uint8_t
    {
        kROOT_BLOCK,
        kBLOCK
    };

    BlockKey key;
    std::unordered_map<BlockKey, SharedPtr<Block>> next;
    EventSink* eventSink;

    virtual ~NodeBase();

    virtual Type type() const noexcept = 0;
    virtual BlockOrdinal ordinal() const noexcept = 0;

    SharedPtr<Block> detachNext(BlockKey const& key);

    /// RootBlock: delegates to tree. Block: len(prev->tokens) or prev->tokensPerBlock().
    virtual int tokensPerBlock() const noexcept = 0;

    /// Tree-wide life-cycle count. RootBlock: delegates to tree. Block: storage.size().
    /// Mirrors Python's num_life_cycles property.
    virtual LifeCycleId numLifeCycles() const noexcept = 0;

protected:
    NodeBase(BlockKey k, EventSink* sink)
        : key(k)
        , eventSink(sink)
    {
    }
};

// ---------------------------------------------------------------------------
// RootBlock — one root per ReuseScope in a BlockRadixTree.
// Holds a map of child Blocks keyed by BlockKey.
// Mirrors Python's RootBlock.
// ---------------------------------------------------------------------------
struct RootBlock : NodeBase
{
    ReuseScope reuseScope;
    BlockRadixTree* tree; // back-reference (non-owning)

    RootBlock(ReuseScope reuseScope, BlockRadixTree* tree);

    static BlockKey makeKey(ReuseScope const& reuseScope);

    Type type() const noexcept override
    {
        return Type::kROOT_BLOCK;
    }

    BlockOrdinal ordinal() const noexcept override
    {
        return kBadBlockOrdinal;
    }

    int tokensPerBlock() const noexcept override;
    LifeCycleId numLifeCycles() const noexcept override; // delegates to tree
};

// ---------------------------------------------------------------------------
// Block — one full (or partial) token block in the radix tree.
// storage[lifeCycleId] = raw observer pointer to CommittedPage (null if not cached).
// Mirrors Python's Block.
// ---------------------------------------------------------------------------
struct Block : NodeBase, EnableSharedFromThis<Block>
{
    // A block's tokens are written once and never re-hashed because its key is
    // computed before construction, so store them in a plain vector.
    std::vector<TokenIdExt> tokens;

    // Previous node in the chain (RootBlock or Block). Null after detaching from the tree.
    // Raw non-owning pointer: while attached, the prev node's `next` map owns us via shared_ptr.
    NodeBase* prev{nullptr};

    TypedVec<LifeCycleId, CommittedPage*> storage;

    // key is precomputed by the caller (for the pre-construction dedup lookup);
    // numLifeCycles is derived from prev. tokens is moved in as a plain vector.
    Block(BlockKey key, std::vector<TokenIdExt> tokens, NodeBase* prev);
    ~Block() override;

    // knownNoDigest: from external text_only knowledge, never a scan (see Hasher::update).
    static BlockKey makeKey(
        BlockKey const& prevKey, TokenIdExt const* tokens, size_t count, bool knownNoDigest = false);

    Type type() const noexcept override
    {
        return Type::kBLOCK;
    }

    BlockOrdinal ordinal() const noexcept override
    {
        return mOrdinal;
    }

    int tokensPerBlock() const noexcept override;

    LifeCycleId numLifeCycles() const noexcept override
    {
        return storage.size();
    }

    bool isFull() const noexcept
    {
        return static_cast<int>(tokens.size()) == tokensPerBlock();
    }

    bool isOrphan() const noexcept;

    // Returns how many leading tokens match `otherTokens`.
    int partialMatchThisNode(TokenIdExt const* otherTokens, size_t count) const;

    // Break the bidirectional link to the cached page for a lifecycle.
    // Returns the previously-stored CommittedPage* (nullptr if already unlinked).
    // If `expectedPage` is non-null and the stored page differs from it, the link
    // is left untouched and nullptr is returned (mirrors Python's unset_page
    // `expected_page` guard: a newer page may already occupy the slot).
    CommittedPage* unlinkPage(LifeCycleId lcIdx, CommittedPage* expectedPage = nullptr);

    // Return the cached page for a lifecycle (nullptr if none). Mirrors Python's Block.get_page().
    CommittedPage* getPage(LifeCycleId lcIdx) const
    {
        return storage[lcIdx];
    }

    // Return the page's recorded token count, or zero if the slot is empty. For attention
    // this is prefix coverage; for SSM it is an exact checkpoint position.
    // Mirrors Python's Block.page_coverage().
    int pageCoverage(LifeCycleId lcIdx) const;

    // True when `page` currently occupies its lifecycle's slot in this block.
    // Mirrors Python's Block.holds_page().
    bool holdsPage(CommittedPage const& page) const;

    // Whether a page recording `numTokensInBlock` may take over slot `lcIdx`. A slot
    // keeps only the page with the largest recorded token count; for SSM that means only
    // the latest checkpoint, and a rare second endpoint in one block is a reuse miss.
    // Pure; use replacePage() to install. Mirrors Python's Block.can_replace_page().
    bool canReplacePage(LifeCycleId lcIdx, int numTokensInBlock) const;

    // Install `page` in slot `lcIdx`, detaching whatever it supersedes. The superseded
    // page may outlive this call while a request still holds it, so unlinkPage() must
    // clear its back-pointer — releasePages() walks `storage` and would never see it.
    // Mirrors Python's Block.replace_page().
    void replacePage(LifeCycleId lcIdx, CommittedPage* page);

    // Move `other`'s pages into this block without changing their recorded token counts.
    // Mirrors Python's Block._adopt_pages_from().
    void adoptPagesFrom(Block& other);

    // Clear stale tree nodes after a lifecycle page has been unlinked.
    // Returns detached blocks that must stay alive until cleanup completes.
    static std::vector<SharedPtr<Block>> clearStaleBlocksAfterPageUnlink(
        Block& block, LifeCycleId lcIdx, LifeCycle const& lc);

    // Reclaim every page held by this block: null each page's back-pointer and, for
    // DROPPABLE pages still scheduled for eviction, remove them from the eviction
    // controller (releasing their storage slots). Idempotent. Cleanup is normally
    // deferred to ~Block(): an orphan block may remain referenced by a live KvCache,
    // and every KvCache must close before StorageManager teardown. Mirrors Python's
    // Block._release_pages().
    void releasePages();

private:
    BlockOrdinal mOrdinal;
};

// ---------------------------------------------------------------------------
// BlockRadixTree — the global cache index.
// next: reuseScope digest → RootBlock.
// Mirrors Python's BlockRadixTree.
// ---------------------------------------------------------------------------
class BlockRadixTree
{
public:
    BlockRadixTree(
        LifeCycleRegistry const& lifeCycles, int tokensPerBlock, std::shared_ptr<EventSink> eventSink = nullptr);
    ~BlockRadixTree();

    // Get (or create) the RootBlock for the given reuse scope.
    RootBlock& addOrGetExisting(ReuseScope const& reuseScope);

    // Match tokens against the tree, yielding (block, numMatchedTokens) pairs.
    // Partial matching: if enablePartialMatch, also yields blocks with a partial
    // leading-token match.
    struct MatchResult
    {
        Block* block;
        int numMatchedTokens;
    };

    struct ReuseMatch
    {
        TypedVec<BlockOrdinal, Block*> blocks;
        int numTokens;
        // Total query length passed to match() (== len(tokens)).
        int numLookupTokens;
        // Internal diagnostic: the prefix the attention pages alone would
        // support, i.e. before recurrent-state (SSM) snapshot availability
        // shortens it. Equal to numTokens when the model has no SSM life
        // cycle. Separates "attention prefix matched N tokens" from
        // "recurrent-snapshot pruning cut it to M".
        int numReusableTokensBeforeHybridPruning;
        // Raw token-path walk depth, before any pruning. Locates where this
        // request's content diverges from the tree, independent of page
        // residency. Equal to numLookupTokens when there is no fork.
        int numReusableTokensBeforePruning;
    };

    // knownNoDigest: from external text_only knowledge, never a scan (see Hasher::update).
    // Takes a non-owning TokenSpan so a zero-copy int32 token buffer can be matched without
    // allocating/copying (the hot path). Callers holding a std::vector pass toSpan(vec).
    // backoff: tokens trimmed off the tail of the match (see KVCacheManagerConfig::reuseMatchBackoff).
    ReuseMatch match(ReuseScope const& reuseScope, TokenSpan tokens, bool knownNoDigest = false,
        bool enablePartialMatch = false, int backoff = 0) const;

    // Detach all cached blocks. ~Block() releases pages when the last owner drops a block.
    void clear();

    int tokensPerBlock() const noexcept
    {
        return mTokensPerBlock;
    }

    LifeCycleId numLifeCycles() const noexcept;

    LifeCycleRegistry const& lifeCycles() const noexcept
    {
        return mLifeCycles;
    }

    std::shared_ptr<EventSink> const& eventSink() const noexcept
    {
        return mEventSink;
    }

    // Read-only access to the root map (used by nanobind introspection).
    std::unordered_map<BlockKey, SharedPtr<RootBlock>> const& roots() const noexcept
    {
        return mRoots;
    }

    // Propose removal of an empty root block. Deferred to avoid destroying
    // objects during destructor chains. Drained at safe points (addOrGetExisting, match).
    void proposeToEraseEmptyRoot(BlockKey const& key)
    {
        mPendingRootErases.push_back(key);
    }

private:
    // knownNoDigest: from external text_only knowledge, never a scan (see Hasher::update).
    std::vector<MatchResult> matchTokenPath(
        ReuseScope const& reuseScope, TokenSpan tokens, bool knownNoDigest, bool enablePartialMatch) const;
    // Shorten `matched` to the prefix that is actually reusable. Passing
    // std::nullopt for `ssmLcId` skips the recurrent-snapshot constraint and
    // yields the attention-only prefix (used for numReusableTokensBeforeHybridPruning).
    std::vector<MatchResult> pruneMatch(std::vector<MatchResult> matched, std::optional<LifeCycleId> ssmLcId) const;

    // Erase any pending empty root blocks from mRoots.
    // Const-qualified: deferred cleanup is not a logical mutation.
    void drainPendingRootErases() const;

    LifeCycleRegistry const& mLifeCycles;
    int mTokensPerBlock;
    std::shared_ptr<EventSink> mEventSink;

    std::unordered_map<BlockKey, SharedPtr<RootBlock>> mRoots;
    mutable std::vector<BlockKey> mPendingRootErases;
};

// ---------------------------------------------------------------------------
// Helpers used by Block and the tree traversal.
// ---------------------------------------------------------------------------

// Add a block to prev's `next` map, or return the existing one on collision.
// A partial block whose tokens are a prefix of an existing sibling returns that longer
// sibling instead of inserting a redundant node.
// If isNew is non-null, *isNew is set to true if a new block was created, false
// if an existing block was returned.
// knownNoDigest: from external text_only knowledge, never a scan (see Hasher::update).
SharedPtr<Block> addOrGetExistingBlock(
    NodeBase* prev, std::vector<TokenIdExt> tokens, bool knownNoDigest, bool* isNew = nullptr);

// Query: the block already in the tree that supersedes inserting `key`/`tokens` under
// `prev` -- an exact-key match, or a longer sibling covering these tokens -- else nullptr.
// Pure; lets callers avoid building a block they would discard.
SharedPtr<Block> getExistingBlock(NodeBase* prev, BlockKey const& key, TokenIdExt const* tokens, size_t numTokens);

// Mutation: link `block` under `prev` and absorb any covered shorter sibling.
// Precondition (debug-asserted): getExistingBlock() returns nullptr for it.
void attachBlock(NodeBase* prev, SharedPtr<Block> const& block);

// The above two composed, for a caller that already holds a Block. Returns the block now
// in the tree, which may be a pre-existing one; `attached` reports whether `block` itself
// went in. Used by KvCache::_reattachOrphanTreeBlocks() to re-insert a block the tail-prune
// walk detached while the request still held it -- see https://nvbugs/6625710.
SharedPtr<Block> attachOrGetExistingBlock(NodeBase* prev, SharedPtr<Block> block, bool* attached = nullptr);

// Post-order traversal: remove a subtree rooted at `root` from its parent's
// next map. ~Block() handles page cleanup. Mirrors Python's remove_subtree().
SharedPtr<Block> removeSubtree(Block& root);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
