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

// Regression tests for the KVCM2 concurrency model (see kv_cache_manager_v2/AGENTS.md).
//
// The rest of the KVCM2 suite is single-threaded and would pass just as happily with the bugs these
// cover still present, so the invariants are asserted directly rather than inferred from a stress
// run wherever that is possible.

#include "kvCacheManagerV2TestUtils.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeConfig;

// Creates `count` roots and proposes all of them for erasure, leaving that many pending entries and
// an unchanged root map. Returns the number of roots now present.
size_t seedPendingRootErases(BlockRadixTree& tree, size_t count)
{
    // Create every root first: addOrGetExisting() drains, so proposing as we go would erase each
    // root on the following iteration and leave only the last one standing.
    for (size_t index = 0; index < count; ++index)
    {
        ReuseScope scope;
        scope.salt = static_cast<std::uint64_t>(index) + 1;
        tree.addOrGetExisting(scope);
    }
    for (size_t index = 0; index < count; ++index)
    {
        ReuseScope scope;
        scope.salt = static_cast<std::uint64_t>(index) + 1;
        // Freshly created roots are childless, which is exactly the state the drain acts on.
        tree.proposeToEraseEmptyRoot(RootBlock::makeKey(scope));
    }
    return tree.roots().size();
}

// The property the shared lock on probeReuse() depends on: matching must not mutate the tree.
//
// Draining the pending root erases here would erase from mRoots and destroy a SharedPtr<RootBlock>
// whose refcount is non-atomic, so two concurrent probes would double-erase and double-free.
TEST(KvCacheManagerV2ConcurrencyTest, ProbeReuseDoesNotMutateTheRadixTree)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeConfig());
    auto& tree = manager->radixTree();

    constexpr size_t kNumRoots = 8;
    size_t const rootsAfterSeeding = seedPendingRootErases(tree, kNumRoots);
    ASSERT_EQ(rootsAfterSeeding, kNumRoots);

    std::vector<TokenIdExt> tokens;
    for (int token = 0; token < manager->tokensPerBlock(); ++token)
    {
        tokens.emplace_back(TokenId{token});
    }

    // Probing must leave every pending root in place, however many times it runs.
    for (int attempt = 0; attempt < 4; ++attempt)
    {
        ReuseScope scope;
        scope.salt = 1;
        EXPECT_EQ(manager->probeReuse(scope, toSpan(tokens), /*knownNoDigest=*/true), 0);
        EXPECT_EQ(tree.roots().size(), kNumRoots) << "probeReuse() drained pending root erases on attempt " << attempt;
    }

    // The drain still happens, just at an exclusive-locked entry point.
    ReuseScope fresh;
    fresh.salt = 9999;
    tree.addOrGetExisting(fresh);
    EXPECT_EQ(tree.roots().size(), 1U) << "addOrGetExisting() must drain the pending root erases";

    // Committing a block is what creates a root in production, so every root has a child and is
    // proposed for erase when that child goes. This test drives the bookkeeping directly and
    // leaves a childless root that nothing proposed, which tree teardown asserts against.
    tree.proposeToEraseEmptyRoot(RootBlock::makeKey(fresh));
}

// Stress form of the above: with pending erases present, concurrent probes must neither corrupt the
// tree nor erase anything. Run under TSan this also flags the data race directly.
TEST(KvCacheManagerV2ConcurrencyTest, ConcurrentProbeReuseIsSafe)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeConfig());
    auto& tree = manager->radixTree();

    constexpr size_t kNumRoots = 64;
    ASSERT_EQ(seedPendingRootErases(tree, kNumRoots), kNumRoots);

    std::vector<TokenIdExt> tokens;
    for (int token = 0; token < manager->tokensPerBlock(); ++token)
    {
        tokens.emplace_back(TokenId{token});
    }

    constexpr int kNumThreads = 8;
    constexpr int kIterations = 2000;
    std::atomic<int> nonZeroMatches{0};
    // Start barrier: without it the scheduler may run the workers one after another, and the test
    // would pass having never had two probes in flight at once -- which is the whole point.
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (int threadIndex = 0; threadIndex < kNumThreads; ++threadIndex)
    {
        threads.emplace_back(
            [&manager, &tokens, &nonZeroMatches, &ready, &go, threadIndex]
            {
                ready.fetch_add(1, std::memory_order_relaxed);
                // Bounded, so a worker cannot hang if the release never comes.
                auto const goDeadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
                while (!go.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < goDeadline)
                {
                    std::this_thread::yield();
                }
                for (int iteration = 0; iteration < kIterations; ++iteration)
                {
                    ReuseScope scope;
                    scope.salt = static_cast<std::uint64_t>((threadIndex + iteration) % kNumRoots) + 1;
                    if (manager->probeReuse(scope, toSpan(tokens), /*knownNoDigest=*/true) != 0)
                    {
                        nonZeroMatches.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            });
    }

    auto const readyDeadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (ready.load(std::memory_order_relaxed) < kNumThreads && std::chrono::steady_clock::now() < readyDeadline)
    {
        std::this_thread::yield();
    }
    // EXPECT, not ASSERT: the workers are joinable, and returning here would terminate. Release
    // unconditionally either way, so nobody waits on a barrier that never opens.
    EXPECT_EQ(ready.load(), kNumThreads) << "not every prober reached the start barrier";
    go.store(true, std::memory_order_release);

    for (auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(nonZeroMatches.load(), 0);
    EXPECT_EQ(tree.roots().size(), kNumRoots) << "concurrent probes must not drain pending root erases";
}

// probeReuse() must remain callable while another thread holds the exclusive lock across a
// manager-mutating call; this pins down that the two do not deadlock and that probes resume.
TEST(KvCacheManagerV2ConcurrencyTest, ProbeReuseInterleavesWithExclusiveWork)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeConfig());

    std::vector<TokenIdExt> tokens;
    for (int token = 0; token < manager->tokensPerBlock(); ++token)
    {
        tokens.emplace_back(TokenId{token});
    }

    std::atomic<bool> stop{false};
    std::atomic<long long> probeCount{0};
    // The prober also stops on its own deadline. glibc's rwlock prefers readers, so a continuous
    // stream of shared acquisitions could in principle starve the writer; without the deadline the
    // writer's next exclusive call would block forever and hang CI instead of failing.
    auto const proberDeadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    std::thread prober(
        [&]
        {
            while (!stop.load(std::memory_order_relaxed) && std::chrono::steady_clock::now() < proberDeadline)
            {
                manager->probeReuse({}, toSpan(tokens), /*knownNoDigest=*/true);
                probeCount.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::yield(); // leave the writer a window to acquire
            }
        });

    // Do not start the writer until the prober is demonstrably running, so "probes progressed
    // alongside exclusive work" cannot be satisfied by the writer finishing first.
    auto const startDeadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (probeCount.load(std::memory_order_relaxed) == 0 && std::chrono::steady_clock::now() < startDeadline)
    {
        std::this_thread::yield();
    }
    // EXPECT, not ASSERT: `prober` is joinable, and returning here would terminate on its destructor.
    bool const proberRan = probeCount.load(std::memory_order_relaxed) > 0;
    EXPECT_TRUE(proberRan) << "prober never ran";

    long long const probesBeforeWriter = probeCount.load(std::memory_order_relaxed);
    constexpr long long kMinProbes = 1000;
    int writerIterations = 0;
    if (proberRan)
    {
        // do/while so at least one exclusive mutation always happens: if the prober had already
        // cleared the threshold by the time the loop was entered, a while-loop would run zero
        // iterations and the test would assert nothing about interleaving.
        do
        {
            // Exclusive-locked, and re-entrant into other locked methods.
            manager->getAndResetIterationStats();
            manager->markStatsDirty(std::nullopt);
            manager->clearStatsDirty(std::nullopt);
            ++writerIterations;
        } while (
            writerIterations < 100000 && probeCount.load(std::memory_order_relaxed) - probesBeforeWriter < kMinProbes);
    }

    stop.store(true, std::memory_order_relaxed);
    prober.join();

    if (proberRan)
    {
        EXPECT_GE(probeCount.load() - probesBeforeWriter, kMinProbes)
            << "probes made no progress against " << writerIterations << " exclusive mutations";
    }
}

} // namespace
