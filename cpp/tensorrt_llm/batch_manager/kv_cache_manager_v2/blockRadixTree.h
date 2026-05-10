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
#include "kv_cache_manager_v2/lifeCycleRegistry.h"

#include "blake3.h"

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
// BlockKey — BLAKE3 digest (32 bytes), used as radix-tree node identifier.
// (Replaces Python SHA-256 32-byte digest.)
// ---------------------------------------------------------------------------
using BlockKey = Digest;
static_assert(kDIGEST_LEN == BLAKE3_OUT_LEN); // 32 bytes

// ---------------------------------------------------------------------------
// Hasher — thin wrapper around BLAKE3 for incremental digests.
// Mirrors Python's Hasher class.
// ---------------------------------------------------------------------------
class Hasher
{
public:
    Hasher();
    explicit Hasher(std::optional<int64_t> seed); // for lora_task_id

    Hasher& update(TokenId token);
    Hasher& update(BlockKey const& key);
    Hasher& update(TokenIdExt const& tokenExt);
    Hasher& update(TokenIdExt const* tokens, size_t count);

    BlockKey digest() const;

private:
    blake3_hasher mState;
};

// ---------------------------------------------------------------------------
// Utility: convert a token sequence → list of BlockKeys.
// First key is the root (loraTaskId digest), then one per token block.
// Mirrors Python's sequence_to_blockchain_keys().
// ---------------------------------------------------------------------------
std::vector<BlockKey> sequenceToBlockchainKeys(
    int tokensPerBlock, std::optional<int64_t> loraTaskId, std::vector<TokenIdExt> const& tokens);

// Generate multi-modal token IDs (mirrors gen_multi_modal_tokens in Python).
std::vector<TokenIdExt> genMultiModalTokens(
    int idOffset, std::vector<uint8_t> const& multiModalDataDigest, int numTokens);

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
    std::unordered_map<BlockKey, std::shared_ptr<Block>> next;

    virtual ~NodeBase();

    virtual Type type() const noexcept = 0;
    virtual BlockOrdinal ordinal() const noexcept = 0;

    std::shared_ptr<Block> detachNext(BlockKey const& key);

    /// RootBlock: delegates to tree. Block: len(prev->tokens) or prev->tokensPerBlock().
    virtual int tokensPerBlock() const noexcept = 0;

protected:
    NodeBase(BlockKey k)
        : key(k)
    {
    }
};

// ---------------------------------------------------------------------------
// RootBlock — one root per (lora_task_id) in a BlockRadixTree.
// Holds a map of child Blocks keyed by BlockKey.
// Mirrors Python's RootBlock.
// ---------------------------------------------------------------------------
struct RootBlock : NodeBase
{
    std::optional<int64_t> loraTaskId;
    BlockRadixTree* tree; // back-reference (non-owning)

    RootBlock(std::optional<int64_t> loraTaskId, BlockRadixTree* tree);

    static BlockKey makeKey(std::optional<int64_t> loraTaskId);

    Type type() const noexcept override
    {
        return Type::kROOT_BLOCK;
    }

    BlockOrdinal ordinal() const noexcept override
    {
        return -1;
    }

    int tokensPerBlock() const noexcept override;
};

// ---------------------------------------------------------------------------
// Block — one full (or partial) token block in the radix tree.
// storage[lifeCycleId] = raw observer pointer to CommittedPage (null if not cached).
// Mirrors Python's Block.
// ---------------------------------------------------------------------------
struct Block : NodeBase, std::enable_shared_from_this<Block>
{
    std::vector<TokenIdExt> tokens;

    // Previous node in the chain (RootBlock or Block). Null after detaching from the tree.
    // Raw non-owning pointer: while attached, the prev node's `next` map owns us via shared_ptr.
    NodeBase* prev{nullptr};

    // indexed by LifeCycleId; nullptr = no cached page for that lifecycle
    std::vector<CommittedPage*> storage;

    Block(BlockKey key, std::vector<TokenIdExt> tokens, NodeBase* prev, int numLifeCycles);
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

    int numLifeCycles() const noexcept
    {
        return static_cast<int>(storage.size());
    }

    bool isFull() const noexcept
    {
        return static_cast<int>(tokens.size()) == tokensPerBlock();
    }

    bool isOrphan() const noexcept;

    // Returns how many leading tokens match `otherTokens`.
    int partialMatchThisNode(TokenIdExt const* otherTokens, size_t count) const;

    // Break the bidirectional link to the cached page for a lifecycle.
    void unlinkPage(LifeCycleId lcIdx);

    // Clear stale tree nodes after a lifecycle page has been unlinked.
    // Returns detached blocks that must stay alive until cleanup completes.
    static std::vector<std::shared_ptr<Block>> clearStaleBlocksAfterPageUnlink(
        Block& block, LifeCycleId lcIdx, LifeCycle const& lc);

private:
    BlockOrdinal mOrdinal;
};

// ---------------------------------------------------------------------------
// BlockRadixTree — the global cache index.
// next: loraTaskId → RootBlock.
// Mirrors Python's BlockRadixTree.
// ---------------------------------------------------------------------------
class BlockRadixTree
{
public:
    BlockRadixTree(LifeCycleRegistry const& lifeCycles, int tokensPerBlock);
    ~BlockRadixTree();

    // Get (or create) the RootBlock for the given LoRA task ID.
    RootBlock& addOrGetExisting(std::optional<int64_t> loraTaskId);

    // Match tokens against the tree, yielding (block, numMatchedTokens) pairs.
    // Partial matching: if enablePartialMatch, also yields blocks with a partial
    // leading-token match.
    struct MatchResult
    {
        Block* block;
        int numMatchedTokens;
    };

    std::vector<MatchResult> match(std::optional<int64_t> loraTaskId, std::vector<TokenIdExt> const& tokens,
        bool enablePartialMatch = false) const;

    // Clear all cached pages. ~Block() handles excludeFromEviction for DROPPABLE pages.
    void clear();

    int tokensPerBlock() const noexcept
    {
        return mTokensPerBlock;
    }

    int numLifeCycles() const noexcept;

    LifeCycleRegistry const& lifeCycles() const noexcept
    {
        return mLifeCycles;
    }

    // Read-only access to the root map (used by nanobind introspection).
    std::unordered_map<BlockKey, std::shared_ptr<RootBlock>> const& roots() const noexcept
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
    // Erase any pending empty root blocks from mRoots.
    // Const-qualified: deferred cleanup is not a logical mutation.
    void drainPendingRootErases() const;

    LifeCycleRegistry const& mLifeCycles;
    int mTokensPerBlock;

    std::unordered_map<BlockKey, std::shared_ptr<RootBlock>> mRoots;
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
std::shared_ptr<Block> addOrGetExistingBlock(
    NodeBase* prev, int numLifeCycles, std::vector<TokenIdExt> tokens, bool* isNew = nullptr);

// Post-order traversal: remove a subtree rooted at `root` from its parent's
// next map. ~Block() handles page cleanup. Mirrors Python's remove_subtree().
std::shared_ptr<Block> removeSubtree(Block& root);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
