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
#include "kv_cache_manager_v2/utils/sharedPtr.h"

#include "sha256.h"

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
    Hasher& update(BlockKey const& key);
    Hasher& update(ReuseScope const& scope);
    Hasher& update(std::vector<uint8_t> const& bytes);
    Hasher& update(TokenIdExt const& tokenExt);
    Hasher& update(TokenIdExt const* tokens, size_t count);

    BlockKey digest() const;

private:
    CSHA256 mState;
};

// ---------------------------------------------------------------------------
// Utility: convert a token sequence → list of BlockKeys.
// First key is the root (reuseScope digest), then one per token block.
// Mirrors Python's sequence_to_blockchain_keys().
// ---------------------------------------------------------------------------
std::vector<BlockKey> sequenceToBlockchainKeys(
    int tokensPerBlock, ReuseScope const& reuseScope, std::vector<TokenIdExt> const& tokens);

// Generate multi-modal token IDs (mirrors gen_multi_modal_tokens in Python).
std::vector<TokenIdExt> genMultiModalTokens(
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
};

// ---------------------------------------------------------------------------
// Block — one full (or partial) token block in the radix tree.
// storage[lifeCycleId] = raw observer pointer to CommittedPage (null if not cached).
// Mirrors Python's Block.
// ---------------------------------------------------------------------------
struct Block : NodeBase, EnableSharedFromThis<Block>
{
    std::vector<TokenIdExt> tokens;

    // Previous node in the chain (RootBlock or Block). Null after detaching from the tree.
    // Raw non-owning pointer: while attached, the prev node's `next` map owns us via shared_ptr.
    NodeBase* prev{nullptr};

    TypedVec<LifeCycleId, CommittedPage*> storage;

    Block(BlockKey key, std::vector<TokenIdExt> tokens, NodeBase* prev, LifeCycleId numLifeCycles);
    ~Block() override;

    static BlockKey makeKey(BlockKey const& prevKey, TokenIdExt const* tokens, size_t count);

    Type type() const noexcept override
    {
        return Type::kBLOCK;
    }

    BlockOrdinal ordinal() const noexcept override
    {
        return mOrdinal;
    }

    int tokensPerBlock() const noexcept override;

    LifeCycleId numLifeCycles() const noexcept
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

    // Clear stale tree nodes after a lifecycle page has been unlinked.
    // Returns detached blocks that must stay alive until cleanup completes.
    static std::vector<SharedPtr<Block>> clearStaleBlocksAfterPageUnlink(
        Block& block, LifeCycleId lcIdx, LifeCycle const& lc);

    // Reclaim every page held by this block: null each page's back-pointer and, for
    // DROPPABLE pages still scheduled for eviction, remove them from the eviction
    // controller (releasing their storage slots). Idempotent. Must run during tree
    // teardown (removeSubtree) rather than being deferred to ~Block(), so page
    // reclamation does not depend on this Block's destruction timing — an external
    // reference can keep a Block alive past StorageManager teardown, after which
    // page->manager would be dangling. Mirrors Python's Block._release_pages().
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
        std::vector<Block*> blocks;
        int numTokens = 0;
    };

    ReuseMatch match(
        ReuseScope const& reuseScope, std::vector<TokenIdExt> const& tokens, bool enablePartialMatch = false) const;

    // Clear all cached pages. ~Block() handles excludeFromEviction for DROPPABLE pages.
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
    std::vector<MatchResult> matchTokenPath(
        ReuseScope const& reuseScope, std::vector<TokenIdExt> const& tokens, bool enablePartialMatch) const;
    ReuseMatch pruneMatch(std::vector<MatchResult> matched) const;

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
// Throws UselessBlockError (with the sibling block) if the block's tokens are a
// prefix of an existing sibling — mirrors Python's UselessBlockError.
// If isNew is non-null, *isNew is set to true if a new block was created, false
// if an existing block was returned.
SharedPtr<Block> addOrGetExistingBlock(
    NodeBase* prev, LifeCycleId numLifeCycles, std::vector<TokenIdExt> tokens, bool* isNew = nullptr);

// Post-order traversal: remove a subtree rooted at `root` from its parent's
// next map. ~Block() handles page cleanup. Mirrors Python's remove_subtree().
SharedPtr<Block> removeSubtree(Block& root);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
