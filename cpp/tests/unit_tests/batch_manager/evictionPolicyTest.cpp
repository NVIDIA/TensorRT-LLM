/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "tensorrt_llm/batch_manager/evictionPolicy.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace tk = tensorrt_llm::kernels;

using namespace tensorrt_llm::batch_manager::eviction_policy;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::executor;
using namespace ::testing;

using ::testing::Return;

#define NUM_PRIMARY_BLOCKS 8
#define NUM_SECONDARY_BLOCKS 4

class MockLRUPolicy : public LRUEvictionPolicy
{
public:
    MOCK_METHOD(std::chrono::steady_clock::time_point::duration, getTime, (), (const, override));
};

class LRUPolicyTest : public ::testing::Test
{
public:
    void SetUp() override
    {
        policy = std::make_shared<MockLRUPolicy>();
        std::vector<BlockPtr> allBlocksById;

        for (KVCacheBlock::IdType blockId = 0; blockId < NUM_PRIMARY_BLOCKS; ++blockId)
        {
            allBlocksById.push_back(std::make_shared<KVCacheBlock>(blockId, tk::KVCacheIndex{blockId, false}));
        }

        for (KVCacheBlock::IdType blockId = 0; blockId < NUM_SECONDARY_BLOCKS; ++blockId)
        {
            allBlocksById.push_back(
                std::make_shared<KVCacheBlock>(NUM_PRIMARY_BLOCKS + blockId, tk::KVCacheIndex{blockId, true}));
        }
        policy->initialize(allBlocksById, {NUM_PRIMARY_BLOCKS, NUM_SECONDARY_BLOCKS}, std::nullopt);
    }

    void TearDown() override {}

    std::shared_ptr<MockLRUPolicy> policy;
};

class KvCacheRetentionConfigTest : public ::testing::Test
{
public:
    void SetUp() {}

    void TearDown() {}
};

TEST_F(LRUPolicyTest, NumFreeBlocksTest)
{
    EXPECT_EQ(NUM_PRIMARY_BLOCKS, policy->getNumFreeBlocks(0));
    EXPECT_EQ(NUM_SECONDARY_BLOCKS, policy->getNumFreeBlocks(1));

    auto primaryBlock = std::get<0>(policy->getFreeBlock(0));
    policy->claimBlock(primaryBlock);
    EXPECT_EQ(NUM_PRIMARY_BLOCKS - 1, policy->getNumFreeBlocks(0));
    EXPECT_EQ(NUM_SECONDARY_BLOCKS, policy->getNumFreeBlocks(1));

    auto secondaryBlock = std::get<0>(policy->getFreeBlock(1));
    policy->claimBlock(secondaryBlock);
    EXPECT_EQ(NUM_PRIMARY_BLOCKS - 1, policy->getNumFreeBlocks(0));
    EXPECT_EQ(NUM_SECONDARY_BLOCKS - 1, policy->getNumFreeBlocks(1));
}

TEST_F(LRUPolicyTest, GetFreeBlockTest)
{
    auto primaryBlock = std::get<0>(policy->getFreeBlock(0));
    EXPECT_FALSE(primaryBlock->hasRefs());
    EXPECT_TRUE(primaryBlock->isPrimary());

    auto secondaryBlock = std::get<0>(policy->getFreeBlock(1));
    EXPECT_FALSE(secondaryBlock->hasRefs());
    EXPECT_FALSE(secondaryBlock->isPrimary());
}

TEST_F(LRUPolicyTest, ReleaseBlockTest)
{
    auto origPrimaryBlock = std::get<0>(policy->getFreeBlock(0));
    policy->claimBlock(origPrimaryBlock);

    EXPECT_NE(origPrimaryBlock->getBlockId(), std::get<0>(policy->getFreeBlock(0))->getBlockId());

    policy->releaseBlock(origPrimaryBlock, true);
    EXPECT_EQ(origPrimaryBlock->getBlockId(), std::get<0>(policy->getFreeBlock(0))->getBlockId());

    policy->claimBlock(origPrimaryBlock);
    policy->releaseBlock(origPrimaryBlock);

    EXPECT_NE(origPrimaryBlock->getBlockId(), std::get<0>(policy->getFreeBlock(0))->getBlockId());
}

TEST_F(LRUPolicyTest, LRUTest)
{
    auto block1 = std::get<0>(policy->getFreeBlock(0));
    policy->claimBlock(block1);

    auto block2 = std::get<0>(policy->getFreeBlock(0));
    policy->claimBlock(block2);

    policy->releaseBlock(block2);
    policy->releaseBlock(block1);

    for (int i = 0; i < NUM_PRIMARY_BLOCKS - 2; i++)
    {
        auto block = std::get<0>(policy->getFreeBlock(0));
        policy->claimBlock(block);
    }
    ASSERT_EQ(std::get<0>(policy->getFreeBlock(0))->getBlockId(), block2->getBlockId());

    policy->claimBlock(std::get<0>(policy->getFreeBlock(0)));

    ASSERT_EQ(std::get<0>(policy->getFreeBlock(0))->getBlockId(), block1->getBlockId());
}

TEST_F(LRUPolicyTest, PriorityTest)
{
    // Test that min priority blocks don't get offloaded
    auto [block, shouldOffload] = policy->getFreeBlock(0);
    EXPECT_TRUE(shouldOffload);
    policy->claimBlock(block, 5, std::nullopt);
    policy->releaseBlock(block);

    std::tie(block, shouldOffload) = policy->getFreeBlock(0);
    EXPECT_FALSE(shouldOffload);
    policy->claimBlock(block);
    policy->releaseBlock(block);

    auto [block1, shouldOffload1] = policy->getFreeBlock(0);
    policy->claimBlock(block, 80, std::nullopt);

    auto [block2, shouldOffload2] = policy->getFreeBlock(0);
    policy->claimBlock(block2, 5, std::nullopt);

    for (int i = 0; i < NUM_PRIMARY_BLOCKS - 2; i++)
    {
        policy->claimBlock(std::get<0>(policy->getFreeBlock(0)));
    }

    policy->releaseBlock(block1);
    policy->releaseBlock(block2);

    EXPECT_EQ(std::get<0>(policy->getFreeBlock(0))->getBlockId(), block2->getBlockId());
}

TEST_F(LRUPolicyTest, TimedBlockTest)
{

    using namespace std::chrono_literals;

    EXPECT_CALL(*policy, getTime)
        .WillOnce(Return(0ms))
        .WillOnce(Return(0ms))
        .WillOnce(Return(2ms))

        .WillOnce(Return(5ms))
        .WillOnce(Return(5ms))
        .WillOnce(Return(5ms))

        .WillOnce(Return(9ms))
        .WillOnce(Return(9ms));

    // Test the priority expiration
    auto [block0, canOffload0] = policy->getFreeBlock(0);
    policy->claimBlock(block0, 80, 1ms);
    // Time: 0
    policy->releaseBlock(block0);
    // Time: 0. Refresh the blocks
    policy->refresh();
    EXPECT_EQ(block0->getPriority(), 80);
    // Time: 2. Refresh the blocks again. This time, `block` should be moved down to the default priority.
    policy->refresh();
    EXPECT_EQ(block0->getPriority(), KvCacheRetentionConfig::kDefaultRetentionPriority);

    auto [block1, canOffload1] = policy->getFreeBlock(0);
    policy->claimBlock(block1, 80, 3ms);

    auto [block2, canOffload2] = policy->getFreeBlock(0);
    policy->claimBlock(block2, 50, 8ms);

    for (int i = 0; i < NUM_PRIMARY_BLOCKS - 2; i++)
    {
        policy->claimBlock(std::get<0>(policy->getFreeBlock(0)));
    }

    // Time: 5
    policy->releaseBlock(block1);
    // Time: 5
    policy->releaseBlock(block2);
    // Time: 5 (Called once)
    policy->refresh();
    EXPECT_EQ(std::get<0>(policy->getFreeBlock(0)), block2);
    // Time: 9 (Called twice)
    policy->refresh();
    EXPECT_EQ(std::get<0>(policy->getFreeBlock(0)), block1);
}

TEST_F(KvCacheRetentionConfigTest, InitializeTest)
{
    // Invalid EOS
    EXPECT_THROW(KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 80),

                                            KvCacheRetentionConfig::TokenRangeRetentionConfig(64, 128, 80)},

                     30),
        std::invalid_argument);
    // Range must not have negative length
    EXPECT_THROW(KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 64, 80),

                                            KvCacheRetentionConfig::TokenRangeRetentionConfig(64, 32, 80)},

                     30),
        std::invalid_argument);
    // Ranges can't overlap
    EXPECT_THROW(KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 64, 80),

                                            KvCacheRetentionConfig::TokenRangeRetentionConfig(30, 128, 80)},

                     30),
        std::invalid_argument);
}

std::vector<std::optional<RetentionPriority>> getPriorities(
    std::vector<RetentionPriorityAndDuration> const& perBlockRetentions)
{
    std::vector<std::optional<RetentionPriority>> retentionPriorities;
    for (auto const& blockRetention : perBlockRetentions)
    {
        retentionPriorities.emplace_back(blockRetention.retentionPriority);
    }
    return retentionPriorities;
}

TEST_F(KvCacheRetentionConfigTest, BlockConfigTest)
{
    auto perBlockConfig = KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 64, 80),

                                                     KvCacheRetentionConfig::TokenRangeRetentionConfig(64, 128, 50)},

        30)
                              .getPerBlockRetentionPriorityDuration(64, 256);
    ASSERT_THAT(getPriorities(perBlockConfig), ElementsAre(80, 50, std::nullopt, std::nullopt));

    perBlockConfig = KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 63, 80),

                                                KvCacheRetentionConfig::TokenRangeRetentionConfig(63, 127, 30)},

        30)
                         .getPerBlockRetentionPriorityDuration(32, 127);
    ASSERT_THAT(getPriorities(perBlockConfig), ElementsAre(80, 80, 30, 30));

    perBlockConfig = KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 1, 80),

                                                KvCacheRetentionConfig::TokenRangeRetentionConfig(1, 2, 5)},

        30)
                         .getPerBlockRetentionPriorityDuration(32, 128);
    ASSERT_THAT(getPriorities(perBlockConfig), ElementsAre(80, std::nullopt, std::nullopt, std::nullopt));

    perBlockConfig = KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 128, 80)}, 30)

                         .getPerBlockRetentionPriorityDuration(256, 192);
    ASSERT_THAT(getPriorities(perBlockConfig), ElementsAre(80));

    perBlockConfig = KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 65, 80),

                                                KvCacheRetentionConfig::TokenRangeRetentionConfig(65, 129, 30)},

        30)
                         .getPerBlockRetentionPriorityDuration(64, 192);
    ASSERT_THAT(getPriorities(perBlockConfig), ElementsAre(80, 80, 30));

    perBlockConfig
        = KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 64, 80),

                                     KvCacheRetentionConfig::TokenRangeRetentionConfig(129, std::nullopt, 50)},

            30)
              .getPerBlockRetentionPriorityDuration(64, 256);
    ASSERT_THAT(getPriorities(perBlockConfig), ElementsAre(80, std::nullopt, std::nullopt, 50));

    perBlockConfig = KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 64, 80),

                                                KvCacheRetentionConfig::TokenRangeRetentionConfig(66, 67, 50)},

        30)
                         .getPerBlockRetentionPriorityDuration(64, 256);
    ASSERT_THAT(getPriorities(perBlockConfig), ElementsAre(80, std::nullopt, std::nullopt, std::nullopt));

    perBlockConfig = KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(127, 193, 80)}, 30)

                         .getPerBlockRetentionPriorityDuration(64, 256);
    ASSERT_THAT(getPriorities(perBlockConfig), ElementsAre(std::nullopt, std::nullopt, 80, 80));

    perBlockConfig = KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(127, 193, 80)}, 30)

                         .getPerBlockRetentionPriorityDuration(64, 256);
    ASSERT_THAT(getPriorities(perBlockConfig), ElementsAre(std::nullopt, std::nullopt, 80, 80));

    perBlockConfig = KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(127, 193, 80)}, 30)

                         .getPerBlockRetentionPriorityDuration(64, 32);
    ASSERT_THAT(getPriorities(perBlockConfig), ElementsAre(std::nullopt));

    perBlockConfig
        = KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(1, std::nullopt, 80)}, 30)

              .getPerBlockRetentionPriorityDuration(64, 128);
    ASSERT_THAT(getPriorities(perBlockConfig), ElementsAre(std::nullopt, 80));
}
