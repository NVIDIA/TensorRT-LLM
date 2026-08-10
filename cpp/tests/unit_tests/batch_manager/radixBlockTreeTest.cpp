/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/batch_manager/radixBlockTree.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"

#include <gtest/gtest.h>

#include <limits>

using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::batch_manager::radix_block_tree;
using namespace tensorrt_llm::kernels;

namespace
{

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

BlockPtr makeBlock(KVCacheBlock::IdType id)
{
    return std::make_shared<KVCacheBlock>(id, KVCacheIndex{id, false});
}

BlockKey makeKey(std::vector<int> const& tokens)
{
    return BlockKey{VecTokens(tokens.begin(), tokens.end())};
}

// Build a root block wired to a fresh UnifiedBlockTree at the given window size.
// Returns {rootBlock, tree}.
std::pair<BlockPtr, std::shared_ptr<UnifiedBlockTree>> makeRootedTree(int windowSize)
{
    auto tree = std::make_shared<UnifiedBlockTree>();
    auto root = makeBlock(KVCacheBlock::kCachedBlocksRootId);
    root->setAsRoot(tree->getRoot(), windowSize);
    return {root, tree};
}

} // namespace

// ---------------------------------------------------------------------------
// 1. attachToLookupNode / detachFromLookupNode lifecycle
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, AttachDetachLifecycle)
{
    UnifiedBlockTree tree;
    constexpr int kWindowSize = 64;

    auto root = makeBlock(KVCacheBlock::kCachedBlocksRootId);
    root->setAsRoot(tree.getRoot(), kWindowSize);

    auto block = makeBlock(0);
    EXPECT_FALSE(block->isShared()); // not in tree yet

    // Attach
    auto key = makeKey({1, 2, 3});
    auto childNode = tree.getRoot()->findOrInsertChild(key, tree.getRoot());
    block->attachToLookupNode(childNode, kWindowSize);

    EXPECT_TRUE(block->isShared());

    // Detach
    block->detachFromLookupNode();
    EXPECT_FALSE(block->isShared());

    // Tree should now be empty (child node was pruned)
    EXPECT_EQ(tree.countNumberOfNodes(), 0);
}

// ---------------------------------------------------------------------------
// 2. getPrevBlock() traversal via lookup node
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, GetPrevBlockViaLookupNode)
{
    constexpr int kWindowSize = 64;
    auto [root, tree] = makeRootedTree(kWindowSize);

    auto blockA = makeBlock(0);
    auto blockB = makeBlock(1);

    BlockKey keyA = makeKey({1, 2, 3});
    BlockKey keyB = makeKey({4, 5, 6});

    // Insert root -> blockA
    root->addNextBlock(keyA, blockA);
    // Insert blockA -> blockB
    blockA->addNextBlock(keyB, blockB);

    EXPECT_EQ(blockA->getPrevBlock(), root);
    EXPECT_EQ(blockB->getPrevBlock(), blockA);

    // root is the tree root and stores itself as value, so its own getPrevBlock()
    // goes one level up to the Trie root node which has no parent -> nullptr.
    EXPECT_EQ(root->getPrevBlock(), nullptr);
}

// ---------------------------------------------------------------------------
// 3. Auto-prune: detaching a leaf removes it from parent's children
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, AutoPruneLeafOnDetach)
{
    constexpr int kWindowSize = 64;
    auto [root, tree] = makeRootedTree(kWindowSize);

    auto blockA = makeBlock(0);
    auto blockB = makeBlock(1);

    root->addNextBlock(makeKey({1, 2, 3}), blockA);
    blockA->addNextBlock(makeKey({4, 5, 6}), blockB);

    // 2 nodes in tree: blockA's node and blockB's node
    EXPECT_EQ(tree->countNumberOfNodes(), 2);

    // Detach leaf B
    blockB->detachFromLookupNode();

    // blockB's node is pruned; blockA's node still has blockA's value so it survives
    EXPECT_EQ(tree->countNumberOfNodes(), 1);
    EXPECT_TRUE(blockA->isShared()); // blockA is still in tree
}

// ---------------------------------------------------------------------------
// 4. Auto-prune cascade: detaching a leaf prunes all empty ancestors
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, AutoPruneCascade)
{
    constexpr int kWindowSize = 64;
    auto [root, tree] = makeRootedTree(kWindowSize);

    auto blockA = makeBlock(0);
    auto blockB = makeBlock(1);

    root->addNextBlock(makeKey({1, 2, 3}), blockA);
    blockA->addNextBlock(makeKey({4, 5, 6}), blockB);

    EXPECT_EQ(tree->countNumberOfNodes(), 2);

    // Detach A first — but B is still in tree, so A's node is NOT pruned
    blockA->detachFromLookupNode();
    // A's node has no value but still has B as child → not pruned
    EXPECT_EQ(tree->countNumberOfNodes(), 2);

    // Now detach B — B's node becomes empty and is pruned, which makes A's node empty
    // (no value, no children) and also causes A's node to be pruned
    blockB->detachFromLookupNode();
    EXPECT_EQ(tree->countNumberOfNodes(), 0);
}

// ---------------------------------------------------------------------------
// 5. Multi-window-size sharing of one lookup node
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, MultiWindowSizeSameNode)
{
    UnifiedBlockTree tree;
    constexpr int kWin128 = 128;
    constexpr int kWin512 = 512;

    auto root128 = makeBlock(KVCacheBlock::kCachedBlocksRootId);
    root128->setAsRoot(tree.getRoot(), kWin128);

    auto root512 = makeBlock(KVCacheBlock::kCachedBlocksRootId);
    root512->setAsRoot(tree.getRoot(), kWin512);

    auto block128 = makeBlock(0);
    auto block512 = makeBlock(1);

    BlockKey key = makeKey({1, 2, 3});

    // Both blocks go to the same tree node (same key prefix) but different value slots
    auto childNode = tree.getRoot()->findOrInsertChild(key, tree.getRoot());
    block128->attachToLookupNode(childNode, kWin128);
    block512->attachToLookupNode(childNode, kWin512);

    EXPECT_EQ(tree.countNumberOfNodes(), 1);

    // Detach window-128 block; node still has window-512 block → NOT pruned
    block128->detachFromLookupNode();
    EXPECT_EQ(tree.countNumberOfNodes(), 1);
    EXPECT_TRUE(block512->isShared()); // still in tree

    // Detach window-512 block; node is now empty → pruned
    block512->detachFromLookupNode();
    EXPECT_EQ(tree.countNumberOfNodes(), 0);
}

// ---------------------------------------------------------------------------
// 6. UnifiedBlockTree::insertBlock / lookupBlock convenience wrappers
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, InsertBlockConvenienceWrapper)
{
    UnifiedBlockTree tree;
    constexpr int kWindowSize = 64;

    auto block = makeBlock(0);

    BlockKey k1 = makeKey({1, 2, 3});
    BlockKey k2 = makeKey({4, 5, 6});
    UnifiedBlockTree::PrefixKey prefix = {k1, k2};

    tree.insertBlock(prefix, kWindowSize, block);

    auto found = tree.lookupBlock(prefix, kWindowSize, /*allowPartialMatch=*/false);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(*found, block);

    // Wrong window size → not found
    auto notFound = tree.lookupBlock(prefix, kWindowSize + 1, false);
    EXPECT_FALSE(notFound.has_value());
}

// ---------------------------------------------------------------------------
// 7. Re-attaching a block to a different node clears the old attachment
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, ReAttachToDifferentNode)
{
    UnifiedBlockTree tree;
    constexpr int kWindowSize = 64;

    BlockKey keyA = makeKey({1, 2, 3});
    BlockKey keyB = makeKey({7, 8, 9});

    auto nodeA = tree.getRoot()->findOrInsertChild(keyA, tree.getRoot());
    auto nodeB = tree.getRoot()->findOrInsertChild(keyB, tree.getRoot());

    auto block = makeBlock(0);

    block->attachToLookupNode(nodeA, kWindowSize);
    EXPECT_TRUE(nodeA->getValue(kWindowSize).has_value());
    EXPECT_FALSE(nodeB->getValue(kWindowSize).has_value());

    // Re-attach to nodeB: old slot in nodeA must be cleared
    block->attachToLookupNode(nodeB, kWindowSize);
    EXPECT_FALSE(nodeA->getValue(kWindowSize).has_value());
    EXPECT_TRUE(nodeB->getValue(kWindowSize).has_value());

    // nodeA is now empty and should have been pruned
    EXPECT_EQ(tree.countNumberOfNodes(), 1); // only nodeB remains
}

// ---------------------------------------------------------------------------
// 8. addNextBlock / findMatchingBlock round-trip
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, AddNextBlockRoundTrip)
{
    constexpr int kWindowSize = 64;
    auto [root, tree] = makeRootedTree(kWindowSize);

    auto block = makeBlock(0);
    block->incRefCount(); // simulate claimed block

    BlockKey key = makeKey({1, 2, 3});
    root->addNextBlock(key, block);

    // findMatchingBlock returns {!block->isFull(), numMatched, block} for exact key match.
    // The block has not been marked full yet, so partial=true (the block content is partial).
    // This matches the original mNextBlocks-based implementation semantics.
    auto [partial, numMatched, found] = root->findMatchingBlock(key, /*enablePartialReuse=*/false, false);
    EXPECT_TRUE(partial); // block is not full -> partial content flag is true
    EXPECT_EQ(static_cast<std::size_t>(numMatched), key.uniqueTokens.size());
    EXPECT_EQ(found, block);

    // After marking full, partial flag becomes false
    block->setBlockKey(key, /*isFull=*/true);
    auto [partial2, numMatched2, found2] = root->findMatchingBlock(key, false, false);
    EXPECT_FALSE(partial2);
    EXPECT_EQ(static_cast<std::size_t>(numMatched2), key.uniqueTokens.size());
    EXPECT_EQ(found2, block);
}

// ---------------------------------------------------------------------------
// 9. freeLeafBlock removes block from parent's children
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, FreeLeafBlockRemovesFromTree)
{
    constexpr int kWindowSize = 64;
    auto [root, tree] = makeRootedTree(kWindowSize);

    auto block = makeBlock(0);
    BlockKey key = makeKey({1, 2, 3});
    root->addNextBlock(key, block);

    EXPECT_EQ(tree->countNumberOfNodes(), 1);
    EXPECT_TRUE(block->isLeaf());

    block->freeLeafBlock();

    // Block detached; its node pruned
    EXPECT_EQ(tree->countNumberOfNodes(), 0);
    EXPECT_FALSE(block->isShared());
}

// ---------------------------------------------------------------------------
// 10. detachDescendantsFromLookupTree clears entire subtree
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, FreeDescendantsRecursively)
{
    constexpr int kWindowSize = 64;
    auto [root, tree] = makeRootedTree(kWindowSize);

    auto blockA = makeBlock(0);
    auto blockB = makeBlock(1);
    auto blockC = makeBlock(2);

    root->addNextBlock(makeKey({1, 2, 3}), blockA);
    blockA->addNextBlock(makeKey({4, 5, 6}), blockB);
    blockA->addNextBlock(makeKey({7, 8, 9}), blockC);

    EXPECT_EQ(tree->countNumberOfNodes(), 3); // A, B, C

    // Free blockA and all descendants
    blockA->freeBlockAndAllDescendants();

    EXPECT_EQ(tree->countNumberOfNodes(), 0);
    EXPECT_FALSE(blockA->isShared());
    EXPECT_FALSE(blockB->isShared());
    EXPECT_FALSE(blockC->isShared());
}

// ---------------------------------------------------------------------------
// 11. isLeaf() reflects child presence in lookup tree
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, IsLeafReflectsLookupTree)
{
    constexpr int kWindowSize = 64;
    auto [root, tree] = makeRootedTree(kWindowSize);

    auto blockA = makeBlock(0);
    auto blockB = makeBlock(1);

    root->addNextBlock(makeKey({1, 2, 3}), blockA);

    EXPECT_TRUE(blockA->isLeaf()); // no children yet

    blockA->addNextBlock(makeKey({4, 5, 6}), blockB);
    EXPECT_FALSE(blockA->isLeaf()); // now has child

    blockB->freeLeafBlock();
    EXPECT_TRUE(blockA->isLeaf()); // child removed
}

// ---------------------------------------------------------------------------
// 12. Partial match returns best (longest) matching child
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, PartialMatchReturnsBestChild)
{
    constexpr int kWindowSize = 64;
    auto [root, tree] = makeRootedTree(kWindowSize);

    // Insert a block with key [1,2,3,4]
    BlockKey storedKey = makeKey({1, 2, 3, 4});
    auto block = makeBlock(0);
    root->addNextBlock(storedKey, block);

    // Query with [1,2,3,9] — should partially match 3 tokens
    BlockKey queryKey = makeKey({1, 2, 3, 9});
    auto [partial, numMatched, found] = root->findMatchingBlock(queryKey, /*enablePartialReuse=*/true,
        /*copyOnPartialReuse=*/true);

    EXPECT_TRUE(partial);
    EXPECT_EQ(numMatched, 3);
    EXPECT_EQ(found, block);
}

// ---------------------------------------------------------------------------
// 13. getNextBlocks() reflects children in the lookup tree
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, GetNextBlocksReflectsChildren)
{
    constexpr int kWindowSize = 64;
    auto [root, tree] = makeRootedTree(kWindowSize);

    auto blockA = makeBlock(0);
    auto blockB = makeBlock(1);

    BlockKey keyA = makeKey({1, 2, 3});
    BlockKey keyB = makeKey({4, 5, 6});

    root->addNextBlock(keyA, blockA);
    root->addNextBlock(keyB, blockB);

    auto nextBlocks = root->getNextBlocks();
    ASSERT_EQ(nextBlocks.size(), 2u);
    EXPECT_EQ(nextBlocks.at(keyA), blockA);
    EXPECT_EQ(nextBlocks.at(keyB), blockB);

    // After detaching blockA its entry disappears
    blockA->detachFromLookupNode();
    nextBlocks = root->getNextBlocks();
    ASSERT_EQ(nextBlocks.size(), 1u);
    EXPECT_EQ(nextBlocks.count(keyA), 0u);
    EXPECT_EQ(nextBlocks.at(keyB), blockB);
}

// ---------------------------------------------------------------------------
// 14. removeNextBlock() removes child from parent's lookup tree
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, RemoveNextBlockUpdatesTree)
{
    constexpr int kWindowSize = 64;
    auto [root, tree] = makeRootedTree(kWindowSize);

    auto blockA = makeBlock(0);
    auto blockB = makeBlock(1);

    BlockKey keyA = makeKey({1, 2, 3});
    BlockKey keyB = makeKey({4, 5, 6});

    root->addNextBlock(keyA, blockA);
    root->addNextBlock(keyB, blockB);

    EXPECT_EQ(tree->countNumberOfNodes(), 2);

    // removeNextBlock(keyA) via root should remove blockA's node
    root->removeNextBlock(keyA);

    EXPECT_EQ(tree->countNumberOfNodes(), 1);
    auto nextBlocks = root->getNextBlocks();
    EXPECT_EQ(nextBlocks.count(keyA), 0u);
    EXPECT_EQ(nextBlocks.at(keyB), blockB);
}

// ---------------------------------------------------------------------------
// 15. addNextBlock is idempotent: a second call with the same key is a no-op
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, AddNextBlockIdempotent)
{
    constexpr int kWindowSize = 64;
    auto [root, tree] = makeRootedTree(kWindowSize);

    auto block1 = makeBlock(0);
    auto block2 = makeBlock(1); // different object, same key

    BlockKey key = makeKey({1, 2, 3});

    root->addNextBlock(key, block1);
    root->addNextBlock(key, block2); // should not overwrite

    EXPECT_EQ(tree->countNumberOfNodes(), 1);
    auto [partial, numMatched, found] = root->findMatchingBlock(key, false, false);
    EXPECT_EQ(found, block1); // block1 still present, block2 was not inserted
}

// ---------------------------------------------------------------------------
// 16. findMatchingBlock returns nothing when block is not in the tree
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, FindMatchingBlockNullLookupNode)
{
    // A block that was never inserted into any tree has mLookupNode == nullptr.
    // findMatchingBlock on it should return {false, 0, nullptr}.
    auto orphan = makeBlock(0);

    BlockKey key = makeKey({1, 2, 3});
    auto [partial, numMatched, found] = orphan->findMatchingBlock(key, false, false);
    EXPECT_FALSE(partial);
    EXPECT_EQ(numMatched, 0);
    EXPECT_EQ(found, nullptr);
}

// ---------------------------------------------------------------------------
// 17. Partial match skips a child block that has active refs
//     (when copyOnPartialReuse=false)
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, PartialMatchSkipsRefedBlockWhenNoCopy)
{
    constexpr int kWindowSize = 64;
    auto [root, tree] = makeRootedTree(kWindowSize);

    // Insert a block with key [1,2,3,4]; simulate it being in-use (has refs)
    BlockKey storedKey = makeKey({1, 2, 3, 4});
    auto block = makeBlock(0);
    block->incRefCount(); // block->hasRefs() == true
    root->addNextBlock(storedKey, block);

    // copyOnPartialReuse=false: refed block must be skipped
    BlockKey queryKey = makeKey({1, 2, 3, 9});
    auto [partial, numMatched, found] = root->findMatchingBlock(queryKey, /*enablePartialReuse=*/true,
        /*copyOnPartialReuse=*/false);

    EXPECT_FALSE(partial);
    EXPECT_EQ(numMatched, 0);
    EXPECT_EQ(found, nullptr);

    // With copyOnPartialReuse=true the same block is accepted
    auto [partial2, numMatched2, found2] = root->findMatchingBlock(queryKey, /*enablePartialReuse=*/true,
        /*copyOnPartialReuse=*/true);
    EXPECT_TRUE(partial2);
    EXPECT_EQ(numMatched2, 3);
    EXPECT_EQ(found2, block);
}

// ---------------------------------------------------------------------------
// 18. kRecurrentStates sentinel is negative (distinguishes from all valid window sizes)
// ---------------------------------------------------------------------------

TEST(MambaTest, kRecurrentStatesSentinelIsNegative)
{
    EXPECT_LT(kRecurrentStates, 0);
}

// ---------------------------------------------------------------------------
// 19. createPlaceholder / isPlaceholder round-trip
// ---------------------------------------------------------------------------

TEST(MambaTest, CreatePlaceholderIsPlaceholder)
{
    auto ph = KVCacheBlock::createPlaceholder(42, 100);
    ASSERT_NE(ph, nullptr);
    EXPECT_TRUE(ph->isPlaceholder());
    EXPECT_EQ(ph->getBlockId(), 42);
}

TEST(MambaTest, CreatePlaceholderNoArgUsesSentinelId)
{
    auto ph = KVCacheBlock::createPlaceholder();
    ASSERT_NE(ph, nullptr);
    EXPECT_TRUE(ph->isPlaceholder());
    EXPECT_EQ(ph->getBlockId(), KVCacheBlock::kPlaceholderBlockId);
    EXPECT_NE(ph->getBlockId(), std::numeric_limits<KVCacheBlock::IdType>::min());
    EXPECT_GT(-ph->getBlockId(), 0);
}

TEST(MambaTest, RegularBlockIsNotPlaceholder)
{
    auto block = makeBlock(7);
    EXPECT_FALSE(block->isPlaceholder());
}

// ---------------------------------------------------------------------------
// 20. insertBlocks / lookupBlock with kRecurrentStates
// ---------------------------------------------------------------------------

TEST(MambaTest, InsertBlocksLookupBlockExactPosition)
{
    UnifiedBlockTree tree;

    BlockKey k0 = makeKey({1, 2, 3});
    BlockKey k1 = makeKey({4, 5, 6});
    BlockKey k2 = makeKey({7, 8, 9});
    UnifiedBlockTree::PrefixKey prefix = {k0, k1, k2};

    auto b0 = makeBlock(10);
    auto b2 = makeBlock(12);
    // Position 1 is nullptr (placeholder)
    tree.insertBlocks(prefix, kRecurrentStates, {b0, nullptr, b2});

    // lookupBlock returns the block at the exact last position = b2
    auto result = tree.lookupBlock(prefix, kRecurrentStates, /*allowPartialMatch=*/false);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, b2);
}

// ---------------------------------------------------------------------------
// 21. lookupBlocksAtAllPositions gives per-position view with nullopt placeholders
// ---------------------------------------------------------------------------

TEST(MambaTest, LookupBlocksAtAllPositionsPerPositionView)
{
    UnifiedBlockTree tree;

    BlockKey k0 = makeKey({1, 2, 3});
    BlockKey k1 = makeKey({4, 5, 6});
    BlockKey k2 = makeKey({7, 8, 9});
    UnifiedBlockTree::PrefixKey prefix = {k0, k1, k2};

    auto b0 = makeBlock(10);
    auto b2 = makeBlock(12);
    tree.insertBlocks(prefix, kRecurrentStates, {b0, nullptr, b2});

    auto all = tree.lookupBlocksAtAllPositions(prefix, kRecurrentStates);
    ASSERT_EQ(all.size(), 3u);
    ASSERT_TRUE(all[0].has_value());
    EXPECT_EQ(*all[0], b0);
    EXPECT_FALSE(all[1].has_value()); // placeholder → nullopt
    ASSERT_TRUE(all[2].has_value());
    EXPECT_EQ(*all[2], b2);
}

// ---------------------------------------------------------------------------
// 22. lookupBlocksAtAllPositions pads with nullopt for missing trie nodes
// ---------------------------------------------------------------------------

TEST(MambaTest, LookupBlocksAtAllPositionsPaddingForMissingNodes)
{
    UnifiedBlockTree tree;

    BlockKey k0 = makeKey({1, 2, 3});
    UnifiedBlockTree::PrefixKey prefix1 = {k0};

    auto b0 = makeBlock(10);
    tree.insertBlocks(prefix1, kRecurrentStates, {b0});

    BlockKey k1 = makeKey({4, 5, 6});
    BlockKey k2 = makeKey({7, 8, 9});
    UnifiedBlockTree::PrefixKey prefix3 = {k0, k1, k2};

    // Lookup a longer prefix — last two positions have no nodes → padded with nullopt
    auto all = tree.lookupBlocksAtAllPositions(prefix3, kRecurrentStates);
    ASSERT_EQ(all.size(), 3u);
    EXPECT_TRUE(all[0].has_value());
    EXPECT_FALSE(all[1].has_value());
    EXPECT_FALSE(all[2].has_value());
}

// ---------------------------------------------------------------------------
// 23. insertBlock does not overwrite an existing block for the same prefix+window
// ---------------------------------------------------------------------------

TEST(UnifiedBlockTreeTest, InsertBlockDoesNotOverwrite)
{
    UnifiedBlockTree tree;
    constexpr int kWindowSize = 64;

    BlockKey k1 = makeKey({1, 2, 3});
    UnifiedBlockTree::PrefixKey prefix = {k1};

    auto block1 = makeBlock(1);
    auto block2 = makeBlock(2);
    tree.insertBlock(prefix, kWindowSize, block1);
    tree.insertBlock(prefix, kWindowSize, block2); // should be a no-op

    auto result = tree.lookupBlock(prefix, kWindowSize, /*allowPartialMatch=*/false);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, block1); // first block retained
}

// ---------------------------------------------------------------------------
// 24. lookupBlock returns nullopt when prefix chain is broken (missing intermediate)
// ---------------------------------------------------------------------------

TEST(UnifiedBlockTreeTest, LookupBlockBrokenChainReturnsNullopt)
{
    UnifiedBlockTree tree;
    constexpr int kWindowSize = 64;

    // Only insert a block at depth 1 (one key)
    BlockKey k0 = makeKey({1, 2, 3});
    UnifiedBlockTree::PrefixKey prefix1 = {k0};
    auto b0 = makeBlock(10);
    tree.insertBlock(prefix1, kWindowSize, b0);

    // Lookup with a 2-step prefix; depth-2 node doesn't exist → chain broken → nullopt
    BlockKey k1 = makeKey({4, 5, 6});
    UnifiedBlockTree::PrefixKey prefix2 = {k0, k1};
    auto result = tree.lookupBlock(prefix2, kWindowSize, /*allowPartialMatch=*/false);
    EXPECT_FALSE(result.has_value());
}

// ---------------------------------------------------------------------------
// 25. lookupBlock returns exact match when multiple positions have valid blocks
// ---------------------------------------------------------------------------

TEST(UnifiedBlockTreeTest, LookupBlockReturnsExactMatch)
{
    UnifiedBlockTree tree;
    constexpr int kWindowSize = 64;

    BlockKey k0 = makeKey({1, 2, 3});
    BlockKey k1 = makeKey({4, 5, 6});
    UnifiedBlockTree::PrefixKey prefix1 = {k0};
    UnifiedBlockTree::PrefixKey prefix2 = {k0, k1};

    auto blockShallow = makeBlock(1);
    auto blockDeep = makeBlock(2);
    tree.insertBlock(prefix1, kWindowSize, blockShallow);
    tree.insertBlock(prefix2, kWindowSize, blockDeep);

    // Lookup the full 2-step prefix — should return the exact match at the last position
    auto result = tree.lookupBlock(prefix2, kWindowSize, /*allowPartialMatch=*/false);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, blockDeep);
}

// ---------------------------------------------------------------------------
// 25b. lookupBlock returns nullopt when target node has no value, even if an
//      ancestor does (exact-match semantics, not deepest-ancestor fallback)
// ---------------------------------------------------------------------------

TEST(UnifiedBlockTreeTest, LookupBlockExactMatchNoAncestorFallback)
{
    UnifiedBlockTree tree;
    constexpr int kWindowSize = 64;

    BlockKey k0 = makeKey({1, 2, 3});
    BlockKey k1 = makeKey({4, 5, 6});
    UnifiedBlockTree::PrefixKey prefix1 = {k0};
    UnifiedBlockTree::PrefixKey prefix2 = {k0, k1};

    // Insert a block only at depth-1, not at depth-2
    auto blockShallow = makeBlock(1);
    tree.insertBlock(prefix1, kWindowSize, blockShallow);

    // Lookup depth-2: the chain is broken (depth-2 node doesn't exist) → nullopt
    auto result = tree.lookupBlock(prefix2, kWindowSize, /*allowPartialMatch=*/false);
    EXPECT_FALSE(result.has_value());

    // Now insert a depth-2 node that exists but has NO value for kWindowSize
    // (simulate by inserting for a different window size)
    auto blockOther = makeBlock(2);
    tree.insertBlock(prefix2, kWindowSize + 1, blockOther);

    // Depth-2 node now exists (chain complete) but has no value for kWindowSize → nullopt
    auto result2 = tree.lookupBlock(prefix2, kWindowSize, /*allowPartialMatch=*/false);
    EXPECT_FALSE(result2.has_value());
}

// ---------------------------------------------------------------------------
// 26. getEdges() returns ALL nodes with values, including nodes that are both
//     terminal (have a value) and internal (have children).
//     Regression test for the _getEdges `else` bug.
// ---------------------------------------------------------------------------

TEST(UnifiedBlockTreeTest, GetEdgesTerminalAndInternalNode)
{
    UnifiedBlockTree tree;
    constexpr int kWindowSize = 64;

    BlockKey k0 = makeKey({1, 2, 3});
    BlockKey k1 = makeKey({4, 5, 6});
    UnifiedBlockTree::PrefixKey prefix1 = {k0};
    UnifiedBlockTree::PrefixKey prefix2 = {k0, k1};

    auto blockShallow = makeBlock(1);
    auto blockDeep = makeBlock(2);
    // k0 node is both terminal (has blockShallow) and internal (has k1 child).
    tree.insertBlock(prefix1, kWindowSize, blockShallow);
    tree.insertBlock(prefix2, kWindowSize, blockDeep);

    auto edges = tree.getEdges();
    ASSERT_EQ(edges.size(), 2u);

    // Both prefix paths should be present (order not guaranteed)
    bool foundShallow = false;
    bool foundDeep = false;
    for (auto const& edge : edges)
    {
        if (edge.size() == 1 && edge[0] == k0)
        {
            foundShallow = true;
        }
        if (edge.size() == 2 && edge[0] == k0 && edge[1] == k1)
        {
            foundDeep = true;
        }
    }
    EXPECT_TRUE(foundShallow) << "Expected edge [k0] in getEdges() output";
    EXPECT_TRUE(foundDeep) << "Expected edge [k0, k1] in getEdges() output";
}

// ---------------------------------------------------------------------------
// 27. BlockKey::numMatchingTokens returns 0 when usesExtraIds differs.
//     Regression test for bug: the check previously omitted usesExtraIds.
// ---------------------------------------------------------------------------

TEST(BlockKeyTest, NumMatchingTokensUsesExtraIdsMismatch)
{
    VecUniqueTokens tokens = {UniqueToken{1, 0}, UniqueToken{2, 0}, UniqueToken{3, 0}};

    // Two keys with identical token content but different usesExtraIds.
    BlockKey keyWithExtra{/*usesExtraIds=*/true, /*loraTaskId=*/std::nullopt, tokens};
    BlockKey keyWithoutExtra{/*usesExtraIds=*/false, /*loraTaskId=*/std::nullopt, tokens};

    // Should return 0 because usesExtraIds differs.
    EXPECT_EQ(keyWithExtra.numMatchingTokens(keyWithoutExtra), 0);
    EXPECT_EQ(keyWithoutExtra.numMatchingTokens(keyWithExtra), 0);

    // Identical keys should return full match.
    EXPECT_EQ(keyWithExtra.numMatchingTokens(keyWithExtra), static_cast<int>(tokens.size()));
}

// ---------------------------------------------------------------------------
// 28. detachFromLookupNode on an unattached block is a no-op (no crash).
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, DetachUnattachedBlockIsNoOp)
{
    auto block = makeBlock(0);
    EXPECT_NO_THROW(block->detachFromLookupNode()); // must not crash or assert
    EXPECT_FALSE(block->isShared());
}

// ---------------------------------------------------------------------------
// 29. getNextBlocks() on an unattached block returns empty map.
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, GetNextBlocksUnattachedReturnsEmpty)
{
    auto block = makeBlock(0);
    auto nextBlocks = block->getNextBlocks();
    EXPECT_TRUE(nextBlocks.empty());
}

// ===========================================================================
// Group I: NVBugs 6104831 — cascade-prune assertion regression
//
// Background:
//   Production failure (Dynamo 1P1D disagg deployment, TRT-LLM 1.3.0rc11)
//   wedges with `templatedTrie.h:249  cascade prune: parent did not find
//   this node as a child` on the prefill-worker call path
//     KVCacheManager::addSequence
//       -> WindowBlockManager::addSequence
//         -> loadOrAllocateBlocks
//           -> getFreeBlock
//             -> KVCacheBlock::freeBlockAndAllDescendants
//               -> detachDescendantsFromLookupTree.
//
// Trie-invariant fix:
//   `KVCacheBlock::removeNextBlock(blockKey)` (kvCacheManager.cpp ~L437) maps
//   to `mLookupNode->clearNode(blockKey)`, which erases the child entry from
//   the parent's `mNextNodes` map. The erased child node's `mPrevNode` weak
//   back-pointer must also be detached; otherwise, if the child block (or any
//   of its descendants) is later detached from its lookup node and becomes
//   empty, cascade-prune calls `parent->clearNode(child.mKey)` on the parent
//   that no longer holds the entry — firing the assertion.
//
//   The only production caller of `removeNextBlock` is
//   `WindowBlockManager::storeBlocks` (kvCacheManager.cpp ~L2290), which uses
//   the pattern
//     block->getPrevBlock()->removeNextBlock(block->getBlockKey());  // (1)
//     ... // setBlockKey / setPrevBlockInSeq
//     searchRoot->addNextBlock(blockKey, block);                     // (2)
//   Step (2) calls `attachToLookupNode` on the new child node, which first
//   does `oldLookupNode->clearValue(W)` on the orphaned old node. If the old
//   node has no remaining values or children, the cascade fires the
//   assertion.
//
// The four tests below isolate the regression at three layers:
//   30. Minimal direct repro (one block, no descendants).
//   31. Same root cause through `freeBlockAndAllDescendants`, matching the
//       bug-report stack trace literally.
//   32. Production storeBlocks-style re-keying via removeNextBlock +
//       addNextBlock.
//   33. Stress loop on prefix-overlapping sequences mirroring the disagg
//       insert/evict workload.
//
// These tests are regression nets for the fixed behavior: orphaned child
// subtrees become cascade-prune roots, so detach/re-key/eviction paths must not
// throw the cascade-prune assertion.
// ===========================================================================

// ---------------------------------------------------------------------------
// 30. Minimal direct regression for the cascade-prune assertion.
//
// Setup:
//   root.lookupNode -> nodeA (keyed by keyA, value at W = blockA)
//
// Trigger:
//   root->removeNextBlock(keyA)  // (1) erases keyA from root.mNextNodes
//                                //     and detaches nodeA.mPrevNode.
//   blockA->detachFromLookupNode()
//      -> nodeA.clearValue(W)   // (2) nodeA becomes empty (no value, no
//                               //     children) but now behaves as a root,
//                               //     so cascade-prune stops there.
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, Regression_NVBugs6104831_DetachOrphanedBlockDoesNotFireAssertion)
{
    constexpr int kWindowSize = 64;
    auto [root, tree] = makeRootedTree(kWindowSize);

    auto blockA = makeBlock(0);
    BlockKey keyA = makeKey({1, 2, 3});
    root->addNextBlock(keyA, blockA);

    ASSERT_EQ(tree->countNumberOfNodes(), 1);
    ASSERT_TRUE(blockA->isShared());

    // (1) Orphan blockA's lookup node from the trie root.
    root->removeNextBlock(keyA);

    // The trie root no longer references blockA's node, but blockA still
    // owns the detached lookup node.
    EXPECT_EQ(tree->countNumberOfNodes(), 0);
    EXPECT_TRUE(blockA->isShared()); // blockA still owns its (now-orphaned) lookup node
    EXPECT_EQ(blockA->getPrevBlock(), nullptr);

    // (2) Detaching blockA must stop cascade-prune at the detached node instead
    //     of walking back to the old root.
    EXPECT_NO_THROW(blockA->detachFromLookupNode());
    EXPECT_FALSE(blockA->isShared());
    EXPECT_EQ(tree->countNumberOfNodes(), 0);
}

// ---------------------------------------------------------------------------
// 31. Same root cause via the bug-report call path
//     `freeBlockAndAllDescendants -> detachDescendantsFromLookupTree`.
//
// Setup:
//   root.lookupNode -> nodeA (value blockA) -> nodeB (value blockB)
//
// Trigger:
//   root->removeNextBlock(keyA)  // orphan A; B is still A's child
//   blockA->freeBlockAndAllDescendants()
//      -> detachDescendantsFromLookupTree() // detaches B; cascade stops at A
//                                           // (A still owns blockA's value)
//      -> detachFromLookupNode()            // removes A's value -> A is now
//                                           // empty, but A behaves as a root
//                                           // and cascade-prune stops there.
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, Regression_NVBugs6104831_FreeBlockAndAllDescendantsOnOrphanedSubtreeDoesNotFireAssertion)
{
    constexpr int kWindowSize = 64;
    auto [root, tree] = makeRootedTree(kWindowSize);

    auto blockA = makeBlock(0);
    auto blockB = makeBlock(1);
    BlockKey keyA = makeKey({1, 2, 3});
    BlockKey keyB = makeKey({4, 5, 6});

    root->addNextBlock(keyA, blockA);
    blockA->addNextBlock(keyB, blockB);
    ASSERT_EQ(tree->countNumberOfNodes(), 2);

    // Orphan blockA's subtree from root (this is what storeBlocks line ~2290
    // does when re-keying a block whose previous-block is the eviction
    // policy's claimed block).
    root->removeNextBlock(keyA);

    // The trie root no longer reaches A or B, but blockA/blockB still own
    // their lookup nodes via mLookupNode shared_ptrs. The full chain
    // root.lookupNode <- nodeA <- nodeB is alive but disconnected from root.
    EXPECT_EQ(tree->countNumberOfNodes(), 0);
    EXPECT_TRUE(blockA->isShared());
    EXPECT_TRUE(blockB->isShared());
    EXPECT_EQ(blockA->getPrevBlock(), nullptr);

    // Mirror the production stack trace verbatim.
    EXPECT_NO_THROW(blockA->freeBlockAndAllDescendants());
    EXPECT_FALSE(blockA->isShared());
    EXPECT_FALSE(blockB->isShared());
    EXPECT_EQ(tree->countNumberOfNodes(), 0);
}

// ---------------------------------------------------------------------------
// 32. Reproduce the exact `WindowBlockManager::storeBlocks` re-keying pattern
//     (kvCacheManager.cpp ~L2290-L2296):
//
//       if (block->getPrevBlock() != nullptr) {
//           block->getPrevBlock()->removeNextBlock(block->getBlockKey()); // (a)
//       }
//       block->setBlockKey(newKey, ...);                                  // (b)
//       block->setPrevBlockInSeq(searchRoot);
//       searchRoot->addNextBlock(newKey, block);                          // (c)
//
// Step (a) orphans blockA's lookup node from its previous block (P) and
// detaches the old lookup node's parent pointer. Step (c) re-attaches blockA
// under a different parent. attachToLookupNode first does
// `oldLookupNode->clearValue(W)` on the orphaned node; cascade-prune must stop
// there instead of walking back to P.
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, Regression_NVBugs6104831_StoreBlocksRekeyDoesNotTriggerAssertion)
{
    constexpr int kWindowSize = 64;
    auto [root, tree] = makeRootedTree(kWindowSize);

    // Build a chain: root -> P -> A. P plays the role of `block->getPrevBlock()`,
    // A plays the role of `block` in the storeBlocks loop.
    auto blockP = makeBlock(0);
    auto blockA = makeBlock(1);
    BlockKey keyP = makeKey({1, 2, 3});
    BlockKey keyA_old = makeKey({4, 5, 6});
    BlockKey keyA_new = makeKey({7, 8, 9});

    root->addNextBlock(keyP, blockP);
    blockP->addNextBlock(keyA_old, blockA);
    blockA->setBlockKey(keyA_old, /*isFull=*/true);
    ASSERT_EQ(tree->countNumberOfNodes(), 2);
    ASSERT_EQ(blockA->getPrevBlock(), blockP);

    // Sanity: the storeBlocks pattern looks up `block->getPrevBlock()` and
    // calls `removeNextBlock(block->getBlockKey())`.
    auto prev = blockA->getPrevBlock();
    ASSERT_NE(prev, nullptr);

    // (a) orphan blockA's lookup node from blockP.
    prev->removeNextBlock(blockA->getBlockKey());
    ASSERT_EQ(blockA->getPrevBlock(), nullptr);
    EXPECT_EQ(tree->countNumberOfNodes(), 1);

    // (b) re-key blockA. setBlockKey only updates the in-block field; it
    //     does not move blockA in the lookup tree.
    blockA->setBlockKey(keyA_new, /*isFull=*/true);

    // (c) re-attach blockA under root with the new key.
    EXPECT_NO_THROW(root->addNextBlock(keyA_new, blockA));
    EXPECT_EQ(blockA->getPrevBlock(), root);
    EXPECT_EQ(tree->countNumberOfNodes(), 2);
}

// ---------------------------------------------------------------------------
// 33. Stress mirror of the disagg workload.
//
// Pattern (per the bug report's loadgen):
//   - One shared "system prompt" prefix.
//   - Many short user prefixes branching off the shared prefix (concurrency
//     16/48/128 in the repro).
//   - Each request is finalized with a storeBlocks-style re-key pass.
//   - Blocks are then evicted/reused for the next burst.
//
// The loop below builds prefix-overlapping sequences, runs the storeBlocks
// re-key step on each, and on a subset also calls
// `freeBlockAndAllDescendants` to mimic the eviction path. The sequences
// share a 2-block "system" prefix and have a 1-block per-request suffix.
//
// Every run-through of either re-key or eviction must complete without
// throwing the cascade-prune assertion.
// ---------------------------------------------------------------------------

TEST(RadixBlockTreeTest, Regression_NVBugs6104831_StressPrefixOverlappingInsertEvictReuseLoop)
{
    constexpr int kWindowSize = 64;
    constexpr int kNumIterations = 16;

    auto [root, tree] = makeRootedTree(kWindowSize);

    // Shared system-prompt blocks (kept alive across iterations to mimic the
    // pinned reusable prefix).
    auto sysBlock0 = makeBlock(1000);
    auto sysBlock1 = makeBlock(1001);
    BlockKey sysKey0 = makeKey({1, 2, 3});
    BlockKey sysKey1 = makeKey({4, 5, 6});
    root->addNextBlock(sysKey0, sysBlock0);
    sysBlock0->addNextBlock(sysKey1, sysBlock1);

    int blockIdCounter = 0;

    // Each iteration:
    //   1. Allocate a leaf "user-suffix" block under sysBlock1, record its
    //      key in suffix->mBlockKey (mirrors how the production allocation
    //      path leaves blocks with mBlockKey set).
    //   2. Re-key the suffix via the storeBlocks pattern (removeNextBlock +
    //      addNextBlock with a different key under sysBlock0). This is the
    //      same shape as production: take the block out of one parent's
    //      mNextNodes, then re-attach it elsewhere.
    //   3. Every fourth iteration, call freeBlockAndAllDescendants on a
    //      previously-retained block to mirror the getFreeBlock eviction
    //      path.
    std::vector<BlockPtr> retained;
    for (int it = 0; it < kNumIterations; ++it)
    {
        auto suffix = makeBlock(blockIdCounter++);
        BlockKey suffixOldKey = makeKey({100 + it, 200 + it, 300 + it});
        BlockKey suffixNewKey = makeKey({400 + it, 500 + it, 600 + it});

        sysBlock1->addNextBlock(suffixOldKey, suffix);
        // Production allocation paths leave block.mBlockKey set to the
        // current trie key. Mirror that here: addNextBlock places the
        // block under suffixOldKey but does not touch suffix->mBlockKey,
        // so we set it explicitly before the storeBlocks re-key step
        // reads block->getBlockKey().
        suffix->setBlockKey(suffixOldKey, /*isFull=*/true);
        ASSERT_EQ(suffix->getPrevBlock(), sysBlock1);

        // storeBlocks-style re-key: remove from old parent then add to a
        // different parent under a new key. This mirrors
        // WindowBlockManager::storeBlocks ~L2290-L2296 verbatim.
        EXPECT_NO_THROW(suffix->getPrevBlock()->removeNextBlock(suffix->getBlockKey()));
        EXPECT_EQ(suffix->getPrevBlock(), nullptr);
        suffix->setBlockKey(suffixNewKey, /*isFull=*/true);
        EXPECT_NO_THROW(sysBlock0->addNextBlock(suffixNewKey, suffix));

        retained.push_back(suffix);

        if ((it % 4) == 3 && !retained.empty())
        {
            // Evict the oldest retained block via the same path the
            // production code uses (getFreeBlock -> freeChildren ->
            // freeBlockAndAllDescendants).
            auto victim = retained.front();
            retained.erase(retained.begin());
            EXPECT_NO_THROW(victim->freeBlockAndAllDescendants());
        }
    }
}
