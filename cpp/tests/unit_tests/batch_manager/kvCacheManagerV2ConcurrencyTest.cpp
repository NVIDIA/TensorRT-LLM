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
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (int threadIndex = 0; threadIndex < kNumThreads; ++threadIndex)
    {
        threads.emplace_back(
            [&manager, &tokens, &nonZeroMatches, threadIndex]
            {
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
    std::thread prober(
        [&]
        {
            while (!stop.load(std::memory_order_relaxed))
            {
                manager->probeReuse({}, toSpan(tokens), /*knownNoDigest=*/true);
                probeCount.fetch_add(1, std::memory_order_relaxed);
            }
        });

    // Keep the writer going until the reader has demonstrably made progress alongside it, so the
    // test cannot pass by finishing before the prober thread is scheduled.
    constexpr long long kMinProbes = 1000;
    for (int iteration = 0; iteration < 100000 && probeCount.load(std::memory_order_relaxed) < kMinProbes; ++iteration)
    {
        // Exclusive-locked, and re-entrant into other locked methods.
        manager->getAndResetIterationStats();
        manager->markStatsDirty(std::nullopt);
        manager->clearStatsDirty(std::nullopt);
    }

    stop.store(true, std::memory_order_relaxed);
    prober.join();
    EXPECT_GE(probeCount.load(), kMinProbes) << "probes made no progress against concurrent exclusive work";
}

} // namespace
