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
#include "tensorrt_llm/common/tllmException.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
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
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeHybridTieredConfig;
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

//! A worker that polls `stop`: on every exit path, a failed assertion or an exception included, it is told to stop
//! before it is joined, so the test reports its failure instead of hanging on the join.
class StoppingThread
{
public:
    template <typename F>
    StoppingThread(std::atomic<bool>& stop, F&& body)
        : mStop(stop)
        , mThread(std::forward<F>(body))
    {
    }

    ~StoppingThread()
    {
        stopAndJoin();
    }

    StoppingThread(StoppingThread const&) = delete;
    StoppingThread& operator=(StoppingThread const&) = delete;

    void stopAndJoin()
    {
        mStop.store(true);
        if (mThread.joinable())
        {
            mThread.join();
        }
    }

private:
    std::atomic<bool>& mStop;
    std::thread mThread;
};

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
        return commitChain(2);
    }

    //! commitRound() for a chain of `numPages` pages.
    std::vector<std::string> commitChain(int numPages)
    {
        auto& storage = mManager->storage();
        TypedVec<LifeCycleId, SlotCount> numSlots(LifeCycleId{1}, numPages);
        auto slots = storage.newGpuSlots(numSlots);
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
    (void) storage.setHostDropHook(
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
    (void) storage.setHostDropHook({});
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
    (void) storage.setHostDropHook(
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
    // Eviction went on: the host level holds the last round, and the dropped pages' slots were reused for it.
    auto const hostStats = storage.getStatistics(kHostLevel);
    EXPECT_EQ(hostStats.evictable, 2);
    EXPECT_EQ(hostStats.free, 0);
}

TEST_F(HostDropTest, NoHostLevelMeansNoRing)
{
    auto gpuOnly = std::make_shared<KvCacheManager>(makeConfig());
    EXPECT_THROW(gpuOnly->storage().setUpHostDropRing(1 << 20, true), tensorrt_llm::common::TllmException);
}

TEST_F(HostDropTest, ShutdownReleasesTheHook)
{
    // A hook that refers back to its owner forms a cycle through the manager; shutdown() must break it.
    auto owner = std::make_shared<int>(0);
    std::weak_ptr<int> const watch = owner;
    (void) mManager->storage().setHostDropHook([owner](std::vector<HostDropRecord> const&) {});
    owner.reset();
    ASSERT_FALSE(watch.expired());
    mManager->shutdown();
    EXPECT_TRUE(watch.expired());
    EXPECT_FALSE(mManager->storage().hasHostDropHook());
}

TEST_F(HostDropTest, RingSetUpValidatesItsArguments)
{
    auto& storage = mManager->storage();
    for (double const maxWaitS :
        {std::nan(""), std::numeric_limits<double>::infinity(), 0.0, -1.0, kHostDropMaxWaitS * 2})
    {
        EXPECT_THROW(storage.setUpHostDropRing(0, false, HostDropLosslessConfig{.enabled = true, .maxWaitS = maxWaitS}),
            tensorrt_llm::common::TllmException)
            << maxWaitS;
    }
    EXPECT_FALSE(storage.hostDropAsync());
    auto const info = storage.setUpHostDropRing(0, false);
    // A repeated call returns the same ring, and only for the same arguments.
    EXPECT_EQ(storage.setUpHostDropRing(0, false).address, info.address);
    EXPECT_THROW(storage.setUpHostDropRing(1 << 30, false), tensorrt_llm::common::TllmException);
    EXPECT_THROW(storage.setUpHostDropRing(0, false, HostDropLosslessConfig{.enabled = true}),
        tensorrt_llm::common::TllmException);
}

TEST_F(HostDropTest, FailedRingSetUpPublishesNothingAndCanBeRetried)
{
    auto& storage = mManager->storage();
    // No host can pin this much: each call fails at the ring, its last allocation, on either copy path.
    for (bool const preferKernel : {true, false})
    {
        EXPECT_ANY_THROW(storage.setUpHostDropRing(size_t{1} << 52, preferKernel));
        EXPECT_FALSE(storage.hostDropAsync());
    }
    // Nothing was published: a retry with other arguments is not taken for a repeat of a ring set up already, and
    // sets the ring up afresh.
    auto const info = storage.setUpHostDropRing(0, false);
    EXPECT_NE(info.address, 0U);
    EXPECT_EQ(info.numSlots, 16);
    EXPECT_TRUE(storage.hostDropAsync());
}

TEST_F(HostDropTest, PagesPrunedWithADroppedParentAreHandedOff)
{
    auto& storage = mManager->storage();
    std::vector<std::string> handedKeys;
    (void) storage.setHostDropHook(
        [&](std::vector<HostDropRecord> const& records)
        {
            for (auto const& record : records)
            {
                handedKeys.push_back(record.key);
            }
        });
    auto first = commitRound();
    commitRound(); // moves `first` down to the host level
    // A one-page round moves one page down, so the host level drops one page. Dropping the parent of `first` removes
    // its block's subtree, which frees the child too: it must be handed off with the parent, not lost silently.
    commitChain(1);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::sort(handedKeys.begin(), handedKeys.end());
    std::sort(first.begin(), first.end());
    EXPECT_EQ(handedKeys, first);
    EXPECT_EQ(storage.hostDropStats().handed.load(), 2);
}

TEST_F(HostDropTest, TeardownClosesTheRing)
{
    auto& storage = mManager->storage();
    (void) storage.setUpHostDropRing(/*ringBytes=*/0, /*preferKernel=*/false);
    commitRound();
    commitRound();
    commitRound(); // the first round dropped into the ring
    auto const drained = drainAtLeast(2);
    ASSERT_EQ(drained.size(), 2U);
    commitRound(); // two more records, which the consumer never drains
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(storage.hostDropStats().handed.load(), 4);

    // The consumer still holds the drained slots when shutdown starts, and returns them a little later. Shutdown
    // waits for them instead of leaking the ring, and drops the undrained records.
    std::vector<int32_t> slots;
    for (auto const& record : drained)
    {
        slots.push_back(record.slot);
    }
    std::atomic<bool> stop{false};
    StoppingThread consumer(stop,
        [&]
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            storage.releaseHostDropSlots(slots);
        });
    auto const t0 = std::chrono::steady_clock::now();
    mManager->shutdown();
    auto const elapsed = std::chrono::steady_clock::now() - t0;
    consumer.stopAndJoin();

    EXPECT_LT(elapsed, std::chrono::duration<double>(kHostDropTeardownMaxWaitS / 2)) << "the ring was not leaked";
    EXPECT_TRUE(storage.drainHostDrops(4096).empty());
    auto const& stats = storage.hostDropStats();
    EXPECT_EQ(stats.handed.load(), 2);
    EXPECT_EQ(stats.ringFull.load(), 2) << "the undrained records are dropped";
    // After teardown a release is lenient: a repeated one is ignored instead of raised on the consumer's thread.
    EXPECT_NO_THROW(storage.releaseHostDropSlots(slots));
}

TEST_F(HostDropTest, ReleaseRejectsSlotsNotDrainedOrReleasedTwice)
{
    auto& storage = mManager->storage();
    auto const info = storage.setUpHostDropRing(/*ringBytes=*/0, /*preferKernel=*/false);
    commitRound();
    commitRound();
    commitRound(); // the first round dropped into the ring
    auto const records = drainAtLeast(2);
    ASSERT_EQ(records.size(), 2U);
    int32_t const a = records[0].slot;
    int32_t const b = records[1].slot;
    int32_t notDrained = 0;
    while (notDrained == a || notDrained == b)
    {
        ++notDrained;
    }
    using tensorrt_llm::common::TllmException;
    EXPECT_THROW(storage.releaseHostDropSlots({-1}), TllmException);
    EXPECT_THROW(storage.releaseHostDropSlots({static_cast<int32_t>(info.numSlots)}), TllmException);
    EXPECT_THROW(storage.releaseHostDropSlots({notDrained}), TllmException);
    // A rejected call releases nothing, not even its valid slots.
    EXPECT_THROW(storage.releaseHostDropSlots({a, a}), TllmException);
    EXPECT_THROW(storage.releaseHostDropSlots({a, notDrained}), TllmException);
    storage.releaseHostDropSlots({a, b});
    EXPECT_THROW(storage.releaseHostDropSlots({a}), TllmException);
}

TEST_F(HostDropTest, DrainWorksWithoutAContextAndLeavesTheCallersCurrent)
{
    auto& storage = mManager->storage();
    (void) storage.setUpHostDropRing(/*ringBytes=*/0, /*preferKernel=*/false);
    commitRound();
    commitRound();
    commitRound(); // the first round dropped into the ring

    // A consumer thread starts with no current context: the drain makes the ring's current for its event queries,
    // then gives the thread back the one it had.
    std::vector<HostDropRingRecord> records;
    CUcontext afterDrain = reinterpret_cast<CUcontext>(1);
    CUresult status = CUDA_ERROR_UNKNOWN;
    std::thread consumer(
        [&]
        {
            records = drainAtLeast(2);
            status = cuCtxGetCurrent(&afterDrain);
        });
    consumer.join();
    ASSERT_EQ(status, CUDA_SUCCESS);
    EXPECT_EQ(afterDrain, nullptr);
    ASSERT_EQ(records.size(), 2U);
    std::vector<int32_t> slots;
    for (auto const& record : records)
    {
        slots.push_back(record.slot);
    }
    storage.releaseHostDropSlots(slots);
}

//! makeHybridTieredConfig(): full attention (life cycle 0) and SSM (life cycle 1) in separate pool groups, with 4 and
//! 1 GPU slots and 2 and 1 host slots.
class HybridHostDropTest : public ::testing::Test
{
protected:
    static constexpr LifeCycleId kAttention{0};
    static constexpr LifeCycleId kSsm{1};

    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        mManager = std::make_shared<KvCacheManager>(makeHybridTieredConfig());
        mRoot = &mManager->radixTree().addOrGetExisting({});
        auto& storage = mManager->storage();
        ASSERT_NE(storage.getPoolGroupIndex(kHostLevel, kAttention), storage.getPoolGroupIndex(kHostLevel, kSsm));
    }

    void TearDown() override
    {
        mManager.reset();
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    //! Commits a chain of `numBlocks` blocks on the GPU, each with an attention page, and an SSM page on the last.
    //! Returns the blocks' radix keys, in chain order.
    std::vector<std::string> commitChain(int numBlocks)
    {
        auto& storage = mManager->storage();
        TypedVec<LifeCycleId, SlotCount> numSlots(LifeCycleId{2}, 0);
        numSlots[kAttention] = numBlocks;
        numSlots[kSsm] = 1;
        auto slots = storage.newGpuSlots(numSlots);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<std::string> keys;
        NodeBase* previous = mRoot;
        for (int i = 0; i < numBlocks; ++i)
        {
            std::vector<TokenIdExt> tokens;
            for (int j = 0; j < mManager->tokensPerBlock(); ++j)
            {
                tokens.emplace_back(TokenId{mNextToken++});
            }
            auto block = addOrGetExistingBlock(previous, std::move(tokens), /*knownNoDigest=*/true);
            commitPage(block, kAttention, slots[kAttention][static_cast<size_t>(i)]);
            if (i + 1 == numBlocks)
            {
                commitPage(block, kSsm, slots[kSsm][0]);
            }
            keys.push_back(keyOf(*block));
            previous = block.get();
        }
        return keys;
    }

    std::shared_ptr<KvCacheManager> mManager;
    RootBlock* mRoot = nullptr;
    int mNextToken = 0;

private:
    void commitPage(SharedPtr<Block> const& block, LifeCycleId lifeCycle, Slot& slot)
    {
        auto& storage = mManager->storage();
        auto page = makeShared<CommittedPage>(
            &storage, block, lifeCycle, kHotLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
        page->setSlot(slot);
        block->storage[lifeCycle] = page.get();
        // The eviction queue owns a droppable page; the block lives as long as its pages.
        storage.scheduleForEviction(*page);
    }
};

TEST_F(HybridHostDropTest, PagesOfEveryPoolGroupAreHandedOffBeforeAnyIsCleared)
{
    auto& storage = mManager->storage();
    std::vector<std::pair<std::string, int32_t>> handed;
    (void) storage.setHostDropHook(
        [&](std::vector<HostDropRecord> const& records)
        {
            for (auto const& record : records)
            {
                handed.emplace_back(record.key, record.lifeCycle);
            }
        });
    // Chain A (two attention pages, an SSM page on its child) goes down to the host level, filling both of its pools.
    auto const chainA = commitChain(2);
    commitChain(4);
    ASSERT_TRUE(handed.empty());
    ASSERT_EQ(storage.getStatistics(kHostLevel, storage.getPoolGroupIndex(kHostLevel, kSsm)).free, 0);
    // One more page of each life cycle comes down, so one eviction drops an attention page of chain A and its SSM
    // page together, from two pool groups. Clearing the attention group first prunes chain A's subtree, which would
    // unlink the SSM page already popped for the other group: every page of chain A must still be handed off.
    commitChain(1);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::sort(handed.begin(), handed.end());
    std::vector<std::pair<std::string, int32_t>> const expected{
        {chainA[0], kAttention.value()}, {chainA[1], kAttention.value()}, {chainA[1], kSsm.value()}};
    auto sortedExpected = expected;
    std::sort(sortedExpected.begin(), sortedExpected.end());
    EXPECT_EQ(handed, sortedExpected);
    EXPECT_EQ(storage.hostDropStats().handed.load(), 3);
}

class HostDropRingTest : public HostDropTest, public ::testing::WithParamInterface<bool>
{
};

TEST_P(HostDropRingTest, RingCopiesDroppedPagesAndRecyclesSlots)
{
    bool const preferKernel = GetParam();
    auto& storage = mManager->storage();
    auto const info = storage.setUpHostDropRing(/*ringBytes=*/0, preferKernel);
    if (preferKernel && !info.kernelCopy)
    {
        GTEST_SKIP() << "the batched page-copy kernel is not available on this device";
    }
    EXPECT_TRUE(storage.hostDropAsync());
    EXPECT_NE(info.address, 0U);
    EXPECT_EQ(info.numSlots, 16) << "the ring has at least 16 slots";
    EXPECT_GE(info.slotBytes, mHotBytes);
    EXPECT_EQ(info.slotBytes % 4096, 0U);
    if (!preferKernel)
    {
        EXPECT_FALSE(info.kernelCopy);
    }
    // A second call with the same arguments returns the same ring.
    auto const again = storage.setUpHostDropRing(/*ringBytes=*/0, preferKernel);
    EXPECT_EQ(again.address, info.address);
    EXPECT_EQ(again.numSlots, info.numSlots);

    // The synchronous hook is not called once the ring is up.
    int hookCalls = 0;
    (void) storage.setHostDropHook([&](std::vector<HostDropRecord> const&) { ++hookCalls; });

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
    auto const last = drainAtLeast(2);
    EXPECT_EQ(last.size(), 2U);
    EXPECT_EQ(storage.hostDropStats().handed.load(), drained + 2);
    // Every drained slot is back, so teardown does not wait for the consumer.
    std::vector<int32_t> lastSlots;
    for (auto const& record : last)
    {
        lastSlots.push_back(record.slot);
    }
    storage.releaseHostDropSlots(lastSlots);
}

TEST_P(HostDropRingTest, LosslessRingWaitsForReleasedSlotsInsteadOfDropping)
{
    bool const preferKernel = GetParam();
    auto& storage = mManager->storage();
    auto const info = storage.setUpHostDropRing(
        /*ringBytes=*/0, preferKernel, HostDropLosslessConfig{/*enabled=*/true, /*maxWaitS=*/10.0, /*minChunk=*/2});
    if (preferKernel && !info.kernelCopy)
    {
        GTEST_SKIP() << "the batched page-copy kernel is not available on this device";
    }
    ASSERT_EQ(info.numSlots, 16);

    // A consumer that gives slots back only once eviction is waiting for them, like the pool's put threads in a burst,
    // so the wait is certain. It records each slot's fill byte before releasing the slot (-1: not one pattern); the
    // patterns are checked after it stops.
    std::atomic<bool> stop{false};
    std::mutex recordsMutex;
    std::vector<std::pair<std::string, int>> seen;
    StoppingThread consumer(stop,
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
                while (!stop.load() && storage.hostDropStats().waits.load() == 0)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
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
    consumer.stopAndJoin();
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
    EXPECT_EQ(stats.errors.load(), 0);
}

TEST_P(HostDropRingTest, LosslessWaitIsBoundedWhenSlotsNeverComeBack)
{
    bool const preferKernel = GetParam();
    auto& storage = mManager->storage();
    auto const info = storage.setUpHostDropRing(
        /*ringBytes=*/0, preferKernel, HostDropLosslessConfig{.enabled = true, .maxWaitS = 1.0, .minChunk = 2});
    if (preferKernel && !info.kernelCopy)
    {
        GTEST_SKIP() << "the batched page-copy kernel is not available on this device";
    }
    auto const t0 = std::chrono::steady_clock::now();
    for (int round = 0; round < 14; ++round)
    {
        commitRound(); // nobody releases a slot: the ring fills after 16 drops
    }
    auto const elapsed = std::chrono::steady_clock::now() - t0;
    auto const& stats = storage.hostDropStats();
    EXPECT_EQ(stats.handed.load(), 16);
    EXPECT_EQ(stats.ringFull.load(), 2 * (14 - 2) - 16);
    // One wait spends the budget and pauses waiting; the remaining rounds, well within the pause, drop without one.
    EXPECT_EQ(stats.waits.load(), 1);
    EXPECT_EQ(stats.timeouts.load(), 1);
    EXPECT_LT(elapsed, std::chrono::seconds(5));
}

TEST_P(HostDropRingTest, LosslessWaitsShareOneBudget)
{
    bool const preferKernel = GetParam();
    auto& storage = mManager->storage();
    constexpr double kMaxWaitS = 0.5;
    auto const info = storage.setUpHostDropRing(
        /*ringBytes=*/0, preferKernel, HostDropLosslessConfig{.enabled = true, .maxWaitS = kMaxWaitS, .minChunk = 2});
    if (preferKernel && !info.kernelCopy)
    {
        GTEST_SKIP() << "the batched page-copy kernel is not available on this device";
    }
    // A slow but live consumer: it frees two slots every 0.9 x maxWaitS, so each wait alone would end in time. The
    // waits share one budget per 2 x maxWaitS, so they still give up and pause instead of stalling every eviction.
    std::atomic<bool> stop{false};
    StoppingThread consumer(stop,
        [&]
        {
            std::vector<int32_t> held;
            auto nextRelease = std::chrono::steady_clock::now();
            while (!stop.load())
            {
                for (auto const& record : storage.drainHostDrops(4096))
                {
                    held.push_back(record.slot);
                }
                if (held.size() >= 2 && std::chrono::steady_clock::now() >= nextRelease)
                {
                    storage.releaseHostDropSlots({held.end() - 2, held.end()});
                    held.resize(held.size() - 2);
                    nextRelease = std::chrono::steady_clock::now()
                        + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(0.9 * kMaxWaitS));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            // Drain and release the rest, so teardown does not wait for the consumer.
            for (auto const& record : storage.drainHostDrops(4096))
            {
                held.push_back(record.slot);
            }
            if (!held.empty())
            {
                storage.releaseHostDropSlots(held);
            }
        });
    auto const t0 = std::chrono::steady_clock::now();
    for (int round = 0; round < 2 + 8 + 6; ++round)
    {
        commitRound(); // 16 drops fill the ring; the last 6 rounds each need released slots
    }
    double const elapsedS = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // Let the consumer catch up on the drops still in flight before it stops.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    consumer.stopAndJoin();
    auto const& stats = storage.hostDropStats();
    EXPECT_GE(stats.timeouts.load(), 1) << "a wait spent the shared budget";
    EXPECT_GT(stats.ringFull.load(), 0) << "drops during the pause go without a hand-off";
    double const waitedS = static_cast<double>(stats.waitNs.load()) * 1e-9;
    double const windows = std::floor(elapsedS / (2 * kMaxWaitS)) + 1;
    EXPECT_LE(waitedS, windows * kMaxWaitS + 0.1) << "at most maxWaitS of waiting per 2 x maxWaitS";
    EXPECT_EQ(stats.errors.load(), 0);
}

TEST_P(HostDropRingTest, LosslessBudgetHoldsOverAnySlidingInterval)
{
    bool const preferKernel = GetParam();
    auto& storage = mManager->storage();
    constexpr double kMaxWaitS = 0.5;
    auto const info = storage.setUpHostDropRing(
        /*ringBytes=*/0, preferKernel, HostDropLosslessConfig{.enabled = true, .maxWaitS = kMaxWaitS, .minChunk = 2});
    if (preferKernel && !info.kernelCopy)
    {
        GTEST_SKIP() << "the batched page-copy kernel is not available on this device";
    }
    for (int round = 0; round < 2 + 8; ++round)
    {
        commitRound(); // 16 drops fill the ring; nobody releases a slot
    }
    ASSERT_EQ(storage.hostDropStats().waits.load(), 0);
    // A wait that spends the whole budget ...
    auto const t0 = std::chrono::steady_clock::now();
    commitRound();
    auto const& stats = storage.hostDropStats();
    ASSERT_EQ(stats.timeouts.load(), 1);
    int64_t const firstWaitNs = stats.waitNs.load();
    // ... then one starting 2.5 x maxWaitS after it began: half of the first lies in the interval of 2 x maxWaitS
    // ending there, so only the other half of the budget is left. A budget reset per fixed window would grant it all.
    std::this_thread::sleep_until(t0
        + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(2.5 * kMaxWaitS)));
    commitRound();
    double const secondWaitS = static_cast<double>(stats.waitNs.load() - firstWaitNs) * 1e-9;
    EXPECT_EQ(stats.waits.load(), 2);
    EXPECT_GT(secondWaitS, 0.3 * kMaxWaitS);
    EXPECT_LT(secondWaitS, 0.8 * kMaxWaitS);
}

INSTANTIATE_TEST_SUITE_P(CopyPath, HostDropRingTest, ::testing::Values(true, false),
    [](::testing::TestParamInfo<bool> const& info) { return info.param ? "PreferKernel" : "HostCallback"; });

} // namespace
