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

// Host-drop hand-off of the last host-memory cache level: the synchronous hook and the pinned ring.

#include "kvCacheManagerV2TestUtils.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/page.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeConfig;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeTieredConfig;

constexpr CacheLevel kHostLevel{1};
constexpr LifeCycleId kLifeCycle{0};

std::string keyOf(Block const& block)
{
    return std::string(reinterpret_cast<char const*>(block.key.data()), block.key.size());
}

bool allBytesEqual(void const* data, size_t size, uint8_t expected)
{
    auto const* bytes = static_cast<uint8_t const*>(data);
    return std::all_of(bytes, bytes + size, [expected](uint8_t byte) { return byte == expected; });
}

//! makeTieredConfig(): two GPU slots and two host slots of 2 MiB for one life cycle. Every round commits a two-block
//! chain on the GPU; the next round's allocation offloads them to the host level, which drops the round before them.
class HostDropTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        mManager = std::make_shared<KvCacheManager>(makeTieredConfig());
        mRoot = &mManager->radixTree().addOrGetExisting({});
        auto& storage = mManager->storage();
        ASSERT_EQ(storage.getStatistics(kHotLevel).total, 2);
        ASSERT_EQ(storage.getStatistics(kHostLevel).total, 2);
        mHotBytes = storage.slotSize(storage.getPoolGroupIndex(kHotLevel, kLifeCycle)).at(PoolIndex{0});
    }

    void TearDown() override
    {
        mManager.reset();
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    //! Allocates two GPU slots (pushing the previous round down a level) and commits two pages into them, filled
    //! with one byte pattern each. Returns the pages' radix keys, in commit order.
    std::vector<std::string> commitRound()
    {
        auto& storage = mManager->storage();
        TypedVec<LifeCycleId, SlotCount> twoSlots(LifeCycleId{1}, 2);
        auto slots = storage.newGpuSlots(twoSlots);
        // The slots may come straight from pages this allocation offloaded: a writer waits for those copies (their
        // ready event) before it overwrites the slot, as a model step would in-stream.
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        PoolGroupIndex const hotPoolGroup = storage.getPoolGroupIndex(kHotLevel, kLifeCycle);
        std::vector<std::string> keys;
        // Each round is its own chain under the root: a dropped round takes only its own blocks out of the tree.
        NodeBase* previous = mRoot;
        cudaStream_t fillStream{};
        EXPECT_EQ(cudaStreamCreate(&fillStream), cudaSuccess);
        for (auto& slot : slots[kLifeCycle])
        {
            std::vector<TokenIdExt> tokens;
            for (int i = 0; i < mManager->tokensPerBlock(); ++i)
            {
                tokens.emplace_back(TokenId{mNextToken++});
            }
            auto block = addOrGetExistingBlock(previous, std::move(tokens), /*knownNoDigest=*/true);
            auto page = makeShared<CommittedPage>(
                &storage, block, kLifeCycle, kHotLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
            page->setSlot(slot);
            block->storage[kLifeCycle] = page.get();
            uint8_t const pattern = mNextPattern++;
            MemAddress const address
                = std::get<MemAddress>(storage.slotAddress(kHotLevel, hotPoolGroup, page->slotId(), PoolIndex{0}));
            EXPECT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(address), pattern, mHotBytes, fillStream), cudaSuccess);
            mPatterns[keyOf(*block)] = pattern;
            keys.push_back(keyOf(*block));
            previous = block.get();
            // The eviction queue owns a droppable page; the block lives as long as its page.
            storage.scheduleForEviction(*page);
        }
        // The offload stream does not wait on these fills, so they must be complete before the next allocation.
        EXPECT_EQ(cudaStreamSynchronize(fillStream), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(fillStream), cudaSuccess);
        return keys;
    }

    //! The pages the async ring has finished copying, waiting (bounded) for at least `expected` records.
    std::vector<HostDropRingRecord> drainAtLeast(size_t expected)
    {
        auto& storage = mManager->storage();
        std::vector<HostDropRingRecord> out;
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (out.size() < expected && std::chrono::steady_clock::now() < deadline)
        {
            auto batch = storage.drainHostDrops(4096);
            if (batch.empty())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            for (auto& record : batch)
            {
                out.push_back(std::move(record));
            }
        }
        return out;
    }

    std::shared_ptr<KvCacheManager> mManager;
    RootBlock* mRoot = nullptr;
    size_t mHotBytes = 0;
    int mNextToken = 0;
    uint8_t mNextPattern = 0x11;
    std::map<std::string, uint8_t> mPatterns; // radix key -> fill byte
};

TEST_F(HostDropTest, InertWithoutHookOrRing)
{
    auto& storage = mManager->storage();
    EXPECT_FALSE(storage.hasHostDropHook());
    EXPECT_FALSE(storage.hostDropAsync());
    for (int round = 0; round < 4; ++round)
    {
        commitRound();
    }
    EXPECT_EQ(storage.hostDropStats().handed.load(), 0);
    EXPECT_EQ(storage.hostDropStats().ringFull.load(), 0);
    EXPECT_EQ(storage.hostDropStats().errors.load(), 0);
}

TEST_F(HostDropTest, HookReceivesEachDroppedPageWithItsHostBytes)
{
    auto& storage = mManager->storage();
    std::vector<HostDropRecord> seen;
    std::vector<bool> contentMatches;
    size_t hostBytes = 0;
    storage.setHostDropHook(
        [&](std::vector<HostDropRecord> const& records)
        {
            for (auto const& record : records)
            {
                seen.push_back(record);
                // The address is valid only during the call: check the host copy's bytes here.
                auto const it = mPatterns.find(record.key);
                contentMatches.push_back(it != mPatterns.end()
                    && allBytesEqual(reinterpret_cast<void const*>(record.address), mHotBytes, it->second));
            }
        });
    ASSERT_TRUE(storage.hasHostDropHook());
    hostBytes = storage.slotSize(kHostLevel, storage.getPoolGroupIndex(kHostLevel, kLifeCycle)).at(PoolIndex{0});

    auto const first = commitRound();
    auto const second = commitRound();
    EXPECT_TRUE(seen.empty()) << "nothing drops while the host level has room";
    auto const third = commitRound(); // second -> host, first dropped
    ASSERT_EQ(seen.size(), 2U);
    std::vector<std::string> keys;
    for (size_t i = 0; i < seen.size(); ++i)
    {
        keys.push_back(seen[i].key);
        EXPECT_EQ(seen[i].lifeCycle, kLifeCycle.value());
        EXPECT_EQ(seen[i].numTokens, mManager->tokensPerBlock());
        EXPECT_EQ(seen[i].bytes, hostBytes);
        EXPECT_NE(seen[i].address, 0U);
        EXPECT_TRUE(contentMatches[i]) << "record " << i << " does not point at its page's host copy";
    }
    std::sort(keys.begin(), keys.end());
    auto expected = first;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(keys, expected);
    // A page handed off on eviction is not handed off again when it is destroyed.
    EXPECT_EQ(storage.hostDropStats().handed.load(), 2);

    commitRound(); // third -> host, second dropped
    ASSERT_EQ(seen.size(), 4U);
    EXPECT_EQ(storage.hostDropStats().handed.load(), 4);
    EXPECT_EQ(storage.hostDropStats().errors.load(), 0);

    // Removing the hook stops the hand-off.
    storage.setHostDropHook({});
    EXPECT_FALSE(storage.hasHostDropHook());
    commitRound();
    EXPECT_EQ(seen.size(), 4U);
    EXPECT_EQ(storage.hostDropStats().handed.load(), 4);
    (void) second;
    (void) third;
}

TEST_F(HostDropTest, HookExceptionIsCountedAndEvictionContinues)
{
    auto& storage = mManager->storage();
    int calls = 0;
    storage.setHostDropHook(
        [&](std::vector<HostDropRecord> const&)
        {
            ++calls;
            throw std::runtime_error("consumer failure");
        });
    for (int round = 0; round < 4; ++round)
    {
        commitRound();
    }
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(storage.hostDropStats().errors.load(), 2);
    EXPECT_EQ(storage.hostDropStats().handed.load(), 0);
    // The level still has its two slots in use and frees the dropped ones.
    EXPECT_EQ(storage.getStatistics(kHostLevel).total, 2);
}

TEST_F(HostDropTest, NoHostLevelMeansNoRing)
{
    auto gpuOnly = std::make_shared<KvCacheManager>(makeConfig());
    EXPECT_THROW(gpuOnly->storage().setUpHostDropRing(1 << 20, true), std::exception);
}

class HostDropRingTest : public HostDropTest, public ::testing::WithParamInterface<bool>
{
};

TEST_P(HostDropRingTest, RingCopiesDroppedPagesAndRecyclesSlots)
{
    bool const preferKernel = GetParam();
    auto& storage = mManager->storage();
    auto const info = storage.setUpHostDropRing(/*ringBytes=*/0, preferKernel);
    EXPECT_TRUE(storage.hostDropAsync());
    EXPECT_NE(info.address, 0U);
    EXPECT_EQ(info.numSlots, 16) << "the ring has at least 16 slots";
    EXPECT_GE(info.slotBytes, mHotBytes);
    EXPECT_EQ(info.slotBytes % 4096, 0U);
    if (!preferKernel)
    {
        EXPECT_FALSE(info.kernelCopy);
    }
    // A second call returns the same ring.
    auto const again = storage.setUpHostDropRing(1 << 30, preferKernel);
    EXPECT_EQ(again.address, info.address);
    EXPECT_EQ(again.numSlots, info.numSlots);

    // The synchronous hook is not called once the ring is up.
    int hookCalls = 0;
    storage.setHostDropHook([&](std::vector<HostDropRecord> const&) { ++hookCalls; });

    auto const first = commitRound();
    commitRound();
    EXPECT_TRUE(storage.drainHostDrops(4096).empty());
    commitRound(); // first dropped into the ring

    auto records = drainAtLeast(2);
    ASSERT_EQ(records.size(), 2U);
    EXPECT_EQ(hookCalls, 0);
    std::vector<int32_t> slots;
    for (auto const& record : records)
    {
        EXPECT_NE(std::find(first.begin(), first.end(), record.key), first.end());
        EXPECT_EQ(record.lifeCycle, kLifeCycle.value());
        EXPECT_EQ(record.numTokens, mManager->tokensPerBlock());
        ASSERT_GE(record.slot, 0);
        ASSERT_LT(record.slot, info.numSlots);
        EXPECT_GE(info.slotBytes, record.bytes);
        // The record is released only after its copy completed: the ring slot holds the page's bytes.
        void const* data
            = reinterpret_cast<void const*>(info.address + static_cast<uint64_t>(record.slot) * info.slotBytes);
        EXPECT_TRUE(allBytesEqual(data, mHotBytes, mPatterns.at(record.key)));
        slots.push_back(record.slot);
    }
    EXPECT_NE(slots[0], slots[1]);
    EXPECT_EQ(storage.hostDropStats().handed.load(), 2);
    storage.releaseHostDropSlots(slots);

    // Without releases the ring runs out after 16 pages; the rest are counted as ring-full drops.
    int64_t drained = 2;
    for (int round = 0; round < 10; ++round)
    {
        commitRound(); // two more drops per round
    }
    auto const more = drainAtLeast(16);
    drained += static_cast<int64_t>(more.size());
    EXPECT_EQ(more.size(), 16U);
    for (auto const& record : more)
    {
        void const* data
            = reinterpret_cast<void const*>(info.address + static_cast<uint64_t>(record.slot) * info.slotBytes);
        EXPECT_TRUE(allBytesEqual(data, mHotBytes, mPatterns.at(record.key)));
    }
    EXPECT_EQ(storage.hostDropStats().handed.load(), drained);
    EXPECT_EQ(storage.hostDropStats().ringFull.load(), 20 + 2 - drained);
    EXPECT_EQ(storage.hostDropStats().errors.load(), 0);

    // Released slots are reused.
    std::vector<int32_t> moreSlots;
    for (auto const& record : more)
    {
        moreSlots.push_back(record.slot);
    }
    storage.releaseHostDropSlots(moreSlots);
    commitRound();
    EXPECT_EQ(drainAtLeast(2).size(), 2U);
    EXPECT_EQ(storage.hostDropStats().handed.load(), drained + 2);
}

TEST_P(HostDropRingTest, LosslessRingWaitsForReleasedSlotsInsteadOfDropping)
{
    bool const preferKernel = GetParam();
    auto& storage = mManager->storage();
    auto const info = storage.setUpHostDropRing(
        /*ringBytes=*/0, preferKernel, HostDropLosslessConfig{/*enabled=*/true, /*maxWaitS=*/10.0, /*minChunk=*/2});
    ASSERT_EQ(info.numSlots, 16);

    // A consumer that gives slots back only after a while, like the pool's put threads in a burst. It records each
    // slot's fill byte before releasing the slot (-1: not one pattern); the patterns are checked after it stops.
    std::atomic<bool> stop{false};
    std::mutex recordsMutex;
    std::vector<std::pair<std::string, int>> seen;
    std::thread consumer(
        [&]
        {
            while (!stop.load())
            {
                auto batch = storage.drainHostDrops(4096);
                if (batch.empty())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                std::vector<int32_t> slots;
                {
                    std::lock_guard<std::mutex> lock(recordsMutex);
                    for (auto& record : batch)
                    {
                        auto const* data = reinterpret_cast<uint8_t const*>(
                            info.address + static_cast<uint64_t>(record.slot) * info.slotBytes);
                        seen.emplace_back(record.key, allBytesEqual(data, mHotBytes, data[0]) ? int{data[0]} : -1);
                        slots.push_back(record.slot);
                    }
                }
                storage.releaseHostDropSlots(slots);
            }
        });
    constexpr int kRounds = 14; // 2 drops per round after the first two: 24 drops through a 16-slot ring
    for (int round = 0; round < kRounds; ++round)
    {
        commitRound();
    }
    int64_t const drops = 2 * (kRounds - 2);
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline)
    {
        {
            std::lock_guard<std::mutex> lock(recordsMutex);
            if (static_cast<int64_t>(seen.size()) >= drops)
            {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    stop.store(true);
    consumer.join();
    for (auto const& [key, fill] : seen)
    {
        EXPECT_EQ(fill, int{mPatterns.at(key)}) << "the ring slot held the dropped page's bytes";
    }
    auto const& stats = storage.hostDropStats();
    EXPECT_EQ(stats.ringFull.load(), 0) << "no page is dropped without a hand-off";
    EXPECT_EQ(stats.handed.load(), drops);
    EXPECT_EQ(static_cast<int64_t>(seen.size()), drops);
    EXPECT_GT(stats.waits.load(), 0) << "the ring filled and eviction waited for released slots";
    EXPECT_EQ(stats.timeouts.load(), 0);
    EXPECT_EQ(stats.gilSkips.load(), 0);
    EXPECT_EQ(stats.errors.load(), 0);
}

TEST_P(HostDropRingTest, LosslessWaitIsBoundedWhenSlotsNeverComeBack)
{
    bool const preferKernel = GetParam();
    auto& storage = mManager->storage();
    storage.setUpHostDropRing(
        /*ringBytes=*/0, preferKernel, HostDropLosslessConfig{/*enabled=*/true, /*maxWaitS=*/0.2, /*minChunk=*/2});
    auto const t0 = std::chrono::steady_clock::now();
    for (int round = 0; round < 14; ++round)
    {
        commitRound(); // nobody releases a slot: the ring fills after 16 drops
    }
    auto const elapsed = std::chrono::steady_clock::now() - t0;
    auto const& stats = storage.hostDropStats();
    EXPECT_EQ(stats.handed.load(), 16);
    EXPECT_EQ(stats.ringFull.load(), 2 * (14 - 2) - 16);
    EXPECT_GE(stats.timeouts.load(), 1);
    // One timed-out wait pauses waiting; with the pause there is at most one wait per 0.2 s, never a hang.
    EXPECT_LT(elapsed, std::chrono::seconds(5));
}

INSTANTIATE_TEST_SUITE_P(CopyPath, HostDropRingTest, ::testing::Values(true, false),
    [](::testing::TestParamInfo<bool> const& info) { return info.param ? "PreferKernel" : "HostCallback"; });

} // namespace
