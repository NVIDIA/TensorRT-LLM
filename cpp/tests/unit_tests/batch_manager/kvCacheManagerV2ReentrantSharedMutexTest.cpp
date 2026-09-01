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

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/reentrantSharedMutex.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace kv = tensorrt_llm::batch_manager::kv_cache_manager_v2;

namespace
{

// Nested exclusive acquisition on the owning thread must not deadlock, and only the outermost
// guard may own the lock. This is the property that lets public APIs call one another.
TEST(ReentrantSharedMutexTest, NestedExclusiveIsNoOpAndDoesNotDeadlock)
{
    kv::ReentrantSharedMutex mutex;
    auto const outer = mutex.lockExclusive();
    EXPECT_TRUE(outer.owns());
    EXPECT_TRUE(mutex.heldExclusiveByThisThread());
    {
        auto const inner = mutex.lockExclusive();
        EXPECT_FALSE(inner.owns()); // nested: the outer guard still owns the release
        {
            auto const innermost = mutex.lockExclusive();
            EXPECT_FALSE(innermost.owns());
        }
    }
    // Still held after the nested guards are destroyed.
    EXPECT_TRUE(mutex.heldExclusiveByThisThread());
}

TEST(ReentrantSharedMutexTest, ExclusiveIsReleasedWhenOutermostGuardDies)
{
    kv::ReentrantSharedMutex mutex;
    {
        auto const guard = mutex.lockExclusive();
        EXPECT_TRUE(mutex.heldExclusiveByThisThread());
    }
    EXPECT_FALSE(mutex.heldExclusiveByThisThread());
    // A second thread must be able to take it now.
    std::atomic<bool> acquired{false};
    std::thread other(
        [&]
        {
            auto const guard = mutex.lockExclusive();
            acquired.store(true);
        });
    other.join();
    EXPECT_TRUE(acquired.load());
}

// Nesting is per-thread: another thread must still be excluded while one holds it.
TEST(ReentrantSharedMutexTest, NestingDoesNotLeakToOtherThreads)
{
    kv::ReentrantSharedMutex mutex;
    std::atomic<bool> otherEntered{false};
    std::thread other;
    {
        auto const outer = mutex.lockExclusive();
        auto const nested = mutex.lockExclusive();
        EXPECT_FALSE(nested.owns());

        other = std::thread(
            [&]
            {
                EXPECT_FALSE(mutex.heldExclusiveByThisThread()); // not this thread's lock
                auto const guard = mutex.lockExclusive();
                otherEntered.store(true);
            });
        // Give the other thread a chance to (incorrectly) proceed.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        EXPECT_FALSE(otherEntered.load()) << "another thread entered while the lock was held";
    } // both guards released here
    other.join();
    EXPECT_TRUE(otherEntered.load());
}

// Multiple readers must run concurrently -- the point of using a shared_mutex at all.
TEST(ReentrantSharedMutexTest, SharedLocksAreConcurrent)
{
    kv::ReentrantSharedMutex mutex;
    constexpr int kReaders = 4;
    std::atomic<int> inside{0};
    std::atomic<int> maxInside{0};
    std::vector<std::thread> readers;
    for (int i = 0; i < kReaders; ++i)
    {
        readers.emplace_back(
            [&]
            {
                auto const guard = mutex.lockShared();
                int const now = ++inside;
                int prev = maxInside.load();
                while (now > prev && !maxInside.compare_exchange_weak(prev, now))
                {
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                --inside;
            });
    }
    for (auto& t : readers)
    {
        t.join();
    }
    EXPECT_GT(maxInside.load(), 1) << "readers serialised; shared lock is behaving exclusively";
}

// A writer must be excluded while readers hold the lock.
TEST(ReentrantSharedMutexTest, WriterWaitsForReaders)
{
    kv::ReentrantSharedMutex mutex;
    std::atomic<bool> writerIn{false};
    std::thread writer;
    {
        auto const reader = mutex.lockShared();
        writer = std::thread(
            [&]
            {
                auto const guard = mutex.lockExclusive();
                writerIn.store(true);
            });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        EXPECT_FALSE(writerIn.load()) << "writer entered while a reader held the lock";
    } // reader released here
    writer.join();
    EXPECT_TRUE(writerIn.load());
}

} // namespace
