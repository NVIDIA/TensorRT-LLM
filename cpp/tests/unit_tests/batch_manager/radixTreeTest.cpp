#include "tensorrt_llm/batch_manager/radixTree.h"
#include "tensorrt_llm/batch_manager/radixTreeUtils.h"

#include <gtest/gtest.h>

using namespace tensorrt_llm::batch_manager::radix_tree;

class RadixTreeTest : public ::testing::Test
{
};

TEST_F(RadixTreeTest, SimpleTree)
{
    RadixTree<int,std::hash<int>,int,std::hash<int>,int,false> radixTree;

    std::vector<int> prefix = {1,2,3};
    auto nodes1 = radixTree.insertNodes(prefix);
    auto itr1 = nodes1.exactMatches.begin();
    EXPECT_EQ(itr1->key, 1);
    ++itr1;
    EXPECT_EQ(itr1->key, 2);
    ++itr1;
    EXPECT_EQ(itr1->key, 3);
}

TEST_F(RadixTreeTest, Strings)
{
    RadixTreeStringSet stringSet;

    // Insert some strings
    std::vector<std::string> strings = {
        "This is string 1, yeah",
        "This is string 2, yeah",
        "This is a loaf of bread"
    };
    for (auto const& str : strings)
    {
        stringSet.insert(str);
    }
    EXPECT_EQ(stringSet.countNumberOfNodes(), 44);

    // Dump all edges
    auto edges = stringSet.getEdges();
    for (auto const& edge : edges)
    {
        std::string str(edge.begin(), edge.end());
        printf("edge :: %s\n",str.c_str());
    }
    printf("stringSet has %d nodes\n",stringSet.countNumberOfNodes());

    // Verify state
    EXPECT_TRUE(stringSet.contains("This is string 1, yeah"));
    EXPECT_TRUE(stringSet.contains("This is string 2, yeah"));
    EXPECT_TRUE(stringSet.contains("This is a loaf of bread"));
    EXPECT_FALSE(stringSet.contains("This is a loaf of bread with walnuts"));
    EXPECT_FALSE(stringSet.contains("Cats"));

    // Remove one string
    stringSet.erase("This is string 2, yeah");
    EXPECT_TRUE(stringSet.contains("This is string 1, yeah"));
    EXPECT_FALSE(stringSet.contains("This is string 2, yeah"));
    EXPECT_TRUE(stringSet.contains("This is a loaf of bread"));
    EXPECT_EQ(stringSet.countNumberOfNodes(), 37);
}
