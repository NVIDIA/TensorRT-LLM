/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/batch_manager/radixTree.h"
#include "tensorrt_llm/batch_manager/radixTreeUtils.h"

#include <gtest/gtest.h>

using namespace tensorrt_llm::batch_manager::radix_tree;

// ---------------------------------------------------------------------------
// TokensKey: a NodeKey that supports partial matching.
//
// Wraps a vector of ints and implements the three methods required by
// findPartiallyMatchingNodes:
//   numMatchingTokens(other) -- shared prefix length
//   getNumTokens()           -- total length of this key
//   shorten(n)               -- new key containing the first n tokens
// ---------------------------------------------------------------------------
struct TokensKey
{
    std::vector<int> tokens;

    bool operator==(TokensKey const& o) const
    {
        return tokens == o.tokens;
    }

    int numMatchingTokens(TokensKey const& o) const
    {
        int n = 0;
        for (size_t i = 0; i < std::min(tokens.size(), o.tokens.size()); ++i)
        {
            if (tokens[i] != o.tokens[i])
            {
                break;
            }
            ++n;
        }
        return n;
    }

    int getNumTokens() const
    {
        return static_cast<int>(tokens.size());
    }

    TokensKey shorten(int n) const
    {
        return TokensKey{{tokens.begin(), tokens.begin() + n}};
    }
};

struct TokensKeyHasher
{
    size_t operator()(TokensKey const& k) const
    {
        size_t seed = 0;
        for (auto t : k.tokens)
        {
            seed ^= std::hash<int>{}(t) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// ---------------------------------------------------------------------------
// Type aliases used throughout the tests.
// ---------------------------------------------------------------------------
using IntTree = RadixTree<int, std::hash<int>, int, std::hash<int>, int, /*supportsPartialMatching*/ false>;
using PartialTree = RadixTree<TokensKey, TokensKeyHasher, int, std::hash<int>, int, /*supportsPartialMatching*/ true>;

// ---------------------------------------------------------------------------
// Test fixture with shared helpers.
//
// Test groups:
//   StringSet Tests -- RadixTreeStringSet end-to-end: insert, lookup, erase
//   Group A         -- insertNodes: prefix insertion, wasCreated / exactMatch flags
//   Group B         -- lookupNodes (exact): key traversal, short queries, cache misses
//   Group C         -- lookupNodes (partial): supportsPartialMatching=true, result ordering,
//                      allowPartialMatch flag
//   Group D         -- setValue / getValue / clearValue: value storage, overwrite policy,
//                      auto-cleanup cascade
//   Group E         -- multi-ValueKey / Variable Sliding Window Attention (VSWA): single node
//                      holding one cached block per attention window size
//   Group F         -- getEdges / countNumberOfNodes: introspection and debug helpers
// ---------------------------------------------------------------------------
class RadixTreeTest : public ::testing::Test
{
protected:
    // Window-size constants used in VSWA tests (Group E).
    static constexpr int kWindowSWA = 128;
    static constexpr int kWindowFull = 4096;

    // Call setValue on a slot that must not already hold a value.
    // setValue returns false when no prior value existed.
    template <class NodePtr>
    static void setFresh(NodePtr node, int vkey, int val)
    {
        EXPECT_FALSE(node->setValue(vkey, val, /*overwrite=*/false));
    }

    // Call clearValue and assert the value was found and removed.
    template <class NodePtr>
    static void clearFound(NodePtr node, int vkey)
    {
        EXPECT_TRUE(node->clearValue(vkey));
    }

    // Insert a single-element prefix and return the resulting node.
    static IntTree::NodePtr singleNode(IntTree& tree, int key)
    {
        return tree.insertNodes({key}).exactMatches[0].node;
    }

    // Insert a multi-element prefix and return the vector of resulting nodes.
    static std::vector<IntTree::NodePtr> buildChain(IntTree& tree, std::vector<int> const& keys)
    {
        auto const& result = tree.insertNodes(keys);
        std::vector<IntTree::NodePtr> nodes;
        nodes.reserve(result.exactMatches.size());
        for (auto const& m : result.exactMatches)
        {
            nodes.push_back(m.node);
        }
        return nodes;
    }

    // Build a PartialTree with two nodes:
    //   root -> [1,2]        (2-token key)
    //   root -> [1,2,3,4]    (4-token key)
    static PartialTree makeTwoNodePartialTree()
    {
        PartialTree tree;
        tree.insertNodes({TokensKey{{1, 2}}});
        tree.insertNodes({TokensKey{{1, 2, 3, 4}}});
        return tree;
    }
};

// ===========================================================================
// StringSet Tests
// ===========================================================================

TEST_F(RadixTreeTest, StringSetInsertAndContains)
{
    RadixTreeStringSet stringSet;

    std::vector<std::string> strings = {"This is string 1, yeah", "This is string 2, yeah", "This is a loaf of bread"};
    for (auto const& str : strings)
    {
        stringSet.insert(str);
    }
    EXPECT_EQ(stringSet.countNumberOfNodes(), 44);

    auto edges = stringSet.getEdges();
    for (auto const& edge : edges)
    {
        std::string str(edge.begin(), edge.end());
        printf("edge :: %s\n", str.c_str());
    }
    printf("stringSet has %d nodes\n", stringSet.countNumberOfNodes());

    EXPECT_TRUE(stringSet.contains("This is string 1, yeah"));
    EXPECT_TRUE(stringSet.contains("This is string 2, yeah"));
    EXPECT_TRUE(stringSet.contains("This is a loaf of bread"));
    EXPECT_FALSE(stringSet.contains("This is a loaf of bread with walnuts"));
    EXPECT_FALSE(stringSet.contains("Cats"));
}

TEST_F(RadixTreeTest, StringSetErase)
{
    RadixTreeStringSet stringSet;

    std::vector<std::string> strings = {"This is string 1, yeah", "This is string 2, yeah", "This is a loaf of bread"};
    for (auto const& str : strings)
    {
        stringSet.insert(str);
    }

    stringSet.erase("This is string 2, yeah");
    EXPECT_TRUE(stringSet.contains("This is string 1, yeah"));
    EXPECT_FALSE(stringSet.contains("This is string 2, yeah"));
    EXPECT_TRUE(stringSet.contains("This is a loaf of bread"));
    EXPECT_EQ(stringSet.countNumberOfNodes(), 37);
}

// ===========================================================================
// Group A: insertNodes
// ===========================================================================

TEST_F(RadixTreeTest, InsertSingleKey)
{
    IntTree tree;
    auto result = tree.insertNodes({42});
    ASSERT_EQ(result.exactMatches.size(), 1u);
    EXPECT_EQ(result.exactMatches[0].key, 42);
    EXPECT_TRUE(result.exactMatches[0].exactMatch);
    EXPECT_TRUE(result.exactMatches[0].wasCreated);
}

TEST_F(RadixTreeTest, InsertSamePrefixTwice)
{
    IntTree tree;
    tree.insertNodes({1, 2, 3});
    auto result = tree.insertNodes({1, 2, 3});
    ASSERT_EQ(result.exactMatches.size(), 3u);
    for (auto const& m : result.exactMatches)
    {
        EXPECT_FALSE(m.wasCreated);
    }
}

TEST_F(RadixTreeTest, InsertExtendingPrefix)
{
    IntTree tree;
    tree.insertNodes({1, 2});
    auto result = tree.insertNodes({1, 2, 3});
    ASSERT_EQ(result.exactMatches.size(), 3u);
    EXPECT_FALSE(result.exactMatches[0].wasCreated); // key 1 already existed
    EXPECT_FALSE(result.exactMatches[1].wasCreated); // key 2 already existed
    EXPECT_TRUE(result.exactMatches[2].wasCreated);  // key 3 is new
}

TEST_F(RadixTreeTest, InsertBranchingPrefixes)
{
    IntTree tree;
    tree.insertNodes({1, 2, 3});
    tree.insertNodes({1, 2, 4});
    // Shared prefix [1,2] + two branch leaves 3 and 4 = 4 nodes total
    EXPECT_EQ(tree.countNumberOfNodes(), 4);

    auto r3 = tree.lookupNodes({1, 2, 3}, /*allowPartialMatch=*/false);
    auto r4 = tree.lookupNodes({1, 2, 4}, /*allowPartialMatch=*/false);
    EXPECT_EQ(r3.exactMatches.size(), 3u);
    EXPECT_EQ(r4.exactMatches.size(), 3u);
}

TEST_F(RadixTreeTest, InsertEmptyPrefix)
{
    IntTree tree;
    auto result = tree.insertNodes({});
    EXPECT_TRUE(result.exactMatches.empty());
    EXPECT_EQ(tree.countNumberOfNodes(), 0);
}

// ===========================================================================
// Group B: lookupNodes (exact match)
// ===========================================================================

TEST_F(RadixTreeTest, LookupExactHit)
{
    IntTree tree;
    tree.insertNodes({10, 20, 30});
    auto result = tree.lookupNodes({10, 20, 30}, /*allowPartialMatch=*/false);
    EXPECT_EQ(result.exactMatches.size(), 3u);
    EXPECT_TRUE(result.partialMatches.empty());
}

TEST_F(RadixTreeTest, LookupPartialPrefix)
{
    IntTree tree;
    tree.insertNodes({10, 20, 30});
    // Query is shorter than what was inserted
    auto result = tree.lookupNodes({10, 20}, /*allowPartialMatch=*/false);
    EXPECT_EQ(result.exactMatches.size(), 2u);
    EXPECT_TRUE(result.partialMatches.empty());
}

TEST_F(RadixTreeTest, LookupMiss)
{
    IntTree tree;
    tree.insertNodes({1, 2, 3});
    auto result = tree.lookupNodes({7, 8, 9}, /*allowPartialMatch=*/false);
    EXPECT_TRUE(result.exactMatches.empty());
    EXPECT_TRUE(result.partialMatches.empty());
}

TEST_F(RadixTreeTest, LookupAfterErase)
{
    // Insert a 3-node chain and store a value at the leaf.
    // Clearing the value triggers auto-cleanup; subsequent lookup finds nothing.
    IntTree tree;
    auto nodes = buildChain(tree, {1, 2, 3});
    setFresh(nodes[2], /*vkey=*/0, /*val=*/99);

    clearFound(nodes[2], /*vkey=*/0);

    // All three nodes were automatically pruned because leaf had no remaining
    // value and no children, which cascades up through the empty chain.
    auto result = tree.lookupNodes({1, 2, 3}, /*allowPartialMatch=*/false);
    EXPECT_TRUE(result.exactMatches.empty());
}

// ===========================================================================
// Group C: lookupNodes (partial matching)
// ===========================================================================

TEST_F(RadixTreeTest, PartialMatchOneToken)
{
    // Tree has a node keyed by [1,2,3]. Query [1,2,9] shares the 2-token prefix
    // [1,2] but differs at position 2. With allowPartialMatch=true the function
    // should return one partial match with 2 matched tokens.
    PartialTree tree;
    tree.insertNodes({TokensKey{{1, 2, 3}}});

    auto result = tree.lookupNodes({TokensKey{{1, 2, 9}}}, /*allowPartialMatch=*/true);
    EXPECT_TRUE(result.exactMatches.empty());
    ASSERT_EQ(result.partialMatches.size(), 1u);
    EXPECT_FALSE(result.partialMatches[0].exactMatch);
    EXPECT_EQ(result.partialMatches[0].key.getNumTokens(), 2); // 2 tokens matched
}

TEST_F(RadixTreeTest, PartialMatchSorted)
{
    // Tree has [1,2,3] and [1,2,3,4]. Query [1,2,3,5] partially matches both.
    // Result must be sorted longest match first: [1,2,3,4] (3 tokens) before [1,2,3] (2 tokens).
    auto tree = makeTwoNodePartialTree();

    auto result = tree.lookupNodes({TokensKey{{1, 2, 3, 5}}}, /*allowPartialMatch=*/true);
    EXPECT_TRUE(result.exactMatches.empty());
    ASSERT_GE(result.partialMatches.size(), 2u);
    EXPECT_GE(result.partialMatches[0].key.getNumTokens(), result.partialMatches[1].key.getNumTokens());
}

TEST_F(RadixTreeTest, PartialMatchDisabled)
{
    // Same tree as PartialMatchSorted, but allowPartialMatch=false: no partial results.
    auto tree = makeTwoNodePartialTree();

    auto result = tree.lookupNodes({TokensKey{{1, 2, 3, 5}}}, /*allowPartialMatch=*/false);
    EXPECT_TRUE(result.exactMatches.empty());
    EXPECT_TRUE(result.partialMatches.empty());
}

TEST_F(RadixTreeTest, PartialMatchFull)
{
    // Query [1,2,3] exactly covers the 3-token key stored in the tree.
    // Even though we call findPartiallyMatchingNodes, the result should have
    // exactMatch=true because all query tokens were matched.
    PartialTree tree;
    tree.insertNodes({TokensKey{{1, 2, 3}}});

    // Use a query that has more tokens than the stored key so the exact path
    // doesn't fire; the stored [1,2,3] is a full match of the 3-token sub-key.
    auto result = tree.lookupNodes({TokensKey{{1, 2, 9}}}, /*allowPartialMatch=*/true);
    // [1,2,3] shares 2 tokens with [1,2,9]; numMatched(2) != getNumTokens(3) so not a full match.
    // Use a query where the stored key is a full match of the query key.
    PartialTree tree2;
    tree2.insertNodes({TokensKey{{1, 2, 3}}});
    auto result2 = tree2.lookupNodes({TokensKey{{1, 2, 3}}}, /*allowPartialMatch=*/true);
    // Exact path fires first (findMatchingNode succeeds) so it lands in exactMatches.
    ASSERT_EQ(result2.exactMatches.size(), 1u);
    EXPECT_TRUE(result2.exactMatches[0].exactMatch);
}

// ===========================================================================
// Group D: setValue / getValue / clearValue
// ===========================================================================

TEST_F(RadixTreeTest, SetAndGetValue)
{
    IntTree tree;
    auto node = singleNode(tree, 5);
    setFresh(node, /*vkey=*/0, /*val=*/42);
    auto val = node->getValue(0);
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 42);
}

TEST_F(RadixTreeTest, OverwriteValue)
{
    IntTree tree;
    auto node = singleNode(tree, 5);
    setFresh(node, /*vkey=*/0, /*val=*/42);
    // Overwrite=true returns true when a prior value existed.
    EXPECT_TRUE(node->setValue(0, 99, /*overwrite=*/true));
    auto val = node->getValue(0);
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 99);
}

TEST_F(RadixTreeTest, NoOverwrite)
{
    IntTree tree;
    auto node = singleNode(tree, 5);
    setFresh(node, /*vkey=*/0, /*val=*/42);
    // Overwrite=false must not replace an existing value.
    EXPECT_FALSE(node->setValue(0, 999, /*overwrite=*/false));
    auto val = node->getValue(0);
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 42); // original value preserved
}

TEST_F(RadixTreeTest, ClearValueLeafCleanup)
{
    // A leaf node with a value and no children auto-deletes after clearValue.
    IntTree tree;
    auto nodes = buildChain(tree, {1, 2, 3});
    setFresh(nodes[2], /*vkey=*/0, /*val=*/7);
    EXPECT_EQ(tree.countNumberOfNodes(), 3);

    clearFound(nodes[2], /*vkey=*/0);
    // Leaf was empty after clear; cascade prunes the whole chain.
    EXPECT_EQ(tree.countNumberOfNodes(), 0);
}

TEST_F(RadixTreeTest, ClearValueNonLeaf)
{
    // Clearing a value from a node that still has children must not delete it.
    IntTree tree;
    auto nodes = buildChain(tree, {1, 2, 3});
    // Set values on the intermediate node and the leaf.
    setFresh(nodes[1], /*vkey=*/0, /*val=*/10); // node for key=2
    setFresh(nodes[2], /*vkey=*/0, /*val=*/20); // node for key=3 (child of above)
    EXPECT_EQ(tree.countNumberOfNodes(), 3);

    // Clear the intermediate node's value; it still has node[2] as a child.
    clearFound(nodes[1], /*vkey=*/0);
    EXPECT_EQ(tree.countNumberOfNodes(), 3); // still 3 nodes
}

// ===========================================================================
// Group E: multi-ValueKey / Variable Sliding Window Attention (VSWA)
//
// In a VSWA model, layers use different attention window sizes simultaneously.
// Each RadixTreeNode can store one KV cache block pointer per window size via
// the ValueKey slot.  A single tree traversal retrieves cached blocks for all
// window sizes in one pass, and evicting an SWA block never disturbs the
// full-attention block at the same prefix position.
// ===========================================================================

TEST_F(RadixTreeTest, MultiWindowSize)
{
    IntTree tree;
    auto node = singleNode(tree, 1);
    setFresh(node, kWindowSWA, /*val=*/10);
    setFresh(node, kWindowFull, /*val=*/20);

    EXPECT_EQ(node->getValue(kWindowSWA).value(), 10);
    EXPECT_EQ(node->getValue(kWindowFull).value(), 20);
}

TEST_F(RadixTreeTest, ClearOneWindowSize)
{
    // Clearing the SWA slot must leave the full-attention slot intact and
    // must not delete the node (it still holds a value).
    IntTree tree;
    auto node = singleNode(tree, 1);
    setFresh(node, kWindowSWA, /*val=*/10);
    setFresh(node, kWindowFull, /*val=*/20);

    clearFound(node, kWindowSWA);
    EXPECT_FALSE(node->getValue(kWindowSWA).has_value());
    EXPECT_EQ(node->getValue(kWindowFull).value(), 20);
    EXPECT_EQ(tree.countNumberOfNodes(), 1); // node survives
}

TEST_F(RadixTreeTest, LookupAllWindowSizes)
{
    // A single lookupNodes call traverses the shared prefix tree once.
    // The caller can then call getValue(windowSize) for each window size
    // independently on the returned nodes.
    IntTree tree;
    auto nodes = buildChain(tree, {1, 2});
    setFresh(nodes[0], kWindowSWA, 10);
    setFresh(nodes[0], kWindowFull, 11);
    setFresh(nodes[1], kWindowSWA, 20);
    setFresh(nodes[1], kWindowFull, 21);

    auto result = tree.lookupNodes({1, 2}, /*allowPartialMatch=*/false);
    ASSERT_EQ(result.exactMatches.size(), 2u);

    auto n0 = result.exactMatches[0].node;
    auto n1 = result.exactMatches[1].node;
    EXPECT_EQ(n0->getValue(kWindowSWA).value(), 10);
    EXPECT_EQ(n0->getValue(kWindowFull).value(), 11);
    EXPECT_EQ(n1->getValue(kWindowSWA).value(), 20);
    EXPECT_EQ(n1->getValue(kWindowFull).value(), 21);
}

// ===========================================================================
// Group F: getEdges / countNumberOfNodes
// ===========================================================================

TEST_F(RadixTreeTest, GetEdgesEmpty)
{
    IntTree tree;
    EXPECT_TRUE(tree.getEdges().empty());
}

TEST_F(RadixTreeTest, GetEdgesAfterInsert)
{
    // Insert two diverging prefixes, each with a value at its leaf.
    // getEdges should return exactly two edges.
    IntTree tree;
    auto nodes_abc = buildChain(tree, {1, 2, 3});
    auto nodes_abd = buildChain(tree, {1, 2, 4});
    setFresh(nodes_abc[2], /*vkey=*/0, /*val=*/1);
    setFresh(nodes_abd[2], /*vkey=*/0, /*val=*/2);

    auto edges = tree.getEdges();
    EXPECT_EQ(edges.size(), 2u);
}

TEST_F(RadixTreeTest, CountNodesConsistency)
{
    // Insert two branching prefixes, verify node count, erase one branch,
    // verify count again.
    IntTree tree;
    auto nodes_abc = buildChain(tree, {1, 2, 3});
    auto nodes_abd = buildChain(tree, {1, 2, 4});
    // 4 nodes: key=1, key=2, key=3, key=4 (two leaves branch off from key=2)
    EXPECT_EQ(tree.countNumberOfNodes(), 4);

    setFresh(nodes_abd[2], /*vkey=*/0, /*val=*/99);
    clearFound(nodes_abd[2], /*vkey=*/0); // removes the key=4 leaf
    // key=4 auto-deleted (leaf, now empty); key=2 still has key=3 as child so
    // the cascade stops there.  key=3 has no value but was not the node being
    // cleared so it is not pruned automatically.
    EXPECT_EQ(tree.countNumberOfNodes(), 3); // key=1, key=2, key=3 remain
}
