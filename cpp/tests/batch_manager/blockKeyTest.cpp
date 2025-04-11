#include "tensorrt_llm/batch_manager/kvCacheManager.h"

#include <gtest/gtest.h>

using namespace tensorrt_llm::batch_manager::kv_cache_manager;

class BlockKeyTest : public ::testing::Test
{
};

TEST_F(BlockKeyTest, PartialMatch)
{
    VecUniqueTokens tokens0 = {{0, 0}, {0, 0}};
    VecUniqueTokens tokens1 = {{0, 0}};
    BlockKey bk0(false, 0, tokens0);
    BlockKey bk1(false, 0, tokens1);

    bk1.uniqueTokens.reserve(2);
    auto ptr = reinterpret_cast<char*>(bk1.uniqueTokens.data());
    std::fill(ptr, ptr + bk1.uniqueTokens.capacity() * sizeof(UniqueToken), 0);

    EXPECT_EQ(bk0.partialMatch(bk1), 1);
}
