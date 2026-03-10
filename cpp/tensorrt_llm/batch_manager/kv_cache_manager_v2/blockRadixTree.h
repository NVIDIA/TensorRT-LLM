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
// RootBlock — one root per (lora_task_id) in a BlockRadixTree.
// Holds a map of child Blocks keyed by BlockKey.
// Mirrors Python's RootBlock.
// ---------------------------------------------------------------------------
struct RootBlock
{
    BlockKey key;
    std::optional<int64_t> loraTaskId;
    std::unordered_map<BlockKey, std::shared_ptr<Block>> next;
    std::weak_ptr<BlockRadixTree> tree; // back-reference (weak)

    RootBlock(std::optional<int64_t> loraTaskId, std::shared_ptr<BlockRadixTree> const& tree);

    static BlockKey makeKey(std::optional<int64_t> loraTaskId);

    int ordinal() const noexcept
    {
        return -1;
    }

    int numLifeCycles() const;
    int tokensPerBlock() const;
};

// ---------------------------------------------------------------------------
// Block — one full (or partial) token block in the radix tree.
// storage[lifeCycleId] = weak pointer to CommittedPage (null if not cached).
// Mirrors Python's Block.
// ---------------------------------------------------------------------------
struct Block : std::enable_shared_from_this<Block>
{
    BlockKey key;
    std::vector<TokenIdExt> tokens;
    BlockOrdinal ordinal;

    // Parent: either a RootBlock (ordinal=0) or another Block.
    // We keep a raw pointer to the parent because the parent owns us via
    // shared_ptr in its `next` map; a shared_ptr back would be a cycle.
    // The parent is always valid while we exist (the parent's next map holds us).
    RootBlock* parentRoot{nullptr}; // non-null when ordinal==0
    Block* parentBlock{nullptr};    // non-null when ordinal>0

    std::unordered_map<BlockKey, std::shared_ptr<Block>> next;

    // indexed by LifeCycleId; nullptr = no cached page for that lifecycle
    std::vector<std::weak_ptr<CommittedPage>> storage;

    ~Block();

    static BlockKey makeKey(BlockKey const& prevKey, TokenIdExt const* tokens, size_t count);

    // Number of lifecycle IDs (== storage.size())
    int numLifeCycles() const noexcept
    {
        return static_cast<int>(storage.size());
    }

    int tokensPerBlock() const noexcept;

    bool isFull() const noexcept
    {
        return static_cast<int>(tokens.size()) == tokensPerBlock();
    }

    bool isOrphan() const noexcept;

    // Returns how many leading tokens match `otherTokens`.
    int partialMatchThisNode(TokenIdExt const* otherTokens, size_t count) const;

    // Unset the cached page for a lifecycle (used when a CommittedPage is evicted/dropped).
    void unsetPage(LifeCycleId lcIdx, LifeCycle const& lc);

    // Access parent as a type-erased pointer for key lookup helpers.
    std::unordered_map<BlockKey, std::shared_ptr<Block>>* parentNextMap();
    BlockKey const& parentKey() const;
};

// ---------------------------------------------------------------------------
// BlockRadixTree — the global cache index.
// next: loraTaskId → RootBlock.
// Mirrors Python's BlockRadixTree.
// ---------------------------------------------------------------------------
class BlockRadixTree : public std::enable_shared_from_this<BlockRadixTree>
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

    // Clear all cached pages (returns weak pointers to evicted CommittedPages).
    std::vector<std::weak_ptr<CommittedPage>> clear();

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
    std::unordered_map<BlockKey, RootBlock> const& roots() const noexcept
    {
        return mRoots;
    }

private:
    LifeCycleRegistry const& mLifeCycles;
    int mTokensPerBlock;

    std::unordered_map<BlockKey, RootBlock> mRoots; // keyed by root BlockKey
};

// ---------------------------------------------------------------------------
// Helpers used by Block and the tree traversal.
// ---------------------------------------------------------------------------

// Add a block to a parent's `next` map, or return the existing one on collision.
// Returns nullptr if the block would be "useless" (its tokens are a prefix of an
// existing sibling).
// If isNew is non-null, *isNew is set to true if a new block was created, false
// if an existing block was returned (matches Python's UselessBlockError path).
std::shared_ptr<Block> addOrGetExistingBlock(std::unordered_map<BlockKey, std::shared_ptr<Block>>& parentNext,
    BlockKey const& parentKey, int numLifeCycles, int tokensPerBlock, std::vector<TokenIdExt> tokens,
    RootBlock* parentRoot, Block* parentBlock, bool* isNew = nullptr);

// Post-order traversal helper: remove a subtree and collect orphaned page refs.
std::vector<std::weak_ptr<CommittedPage>> removeSubtree(
    std::unordered_map<BlockKey, std::shared_ptr<Block>>& parentNext, BlockKey const& rootKey);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
