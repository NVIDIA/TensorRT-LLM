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

//! Spins until `pred` holds, or the deadline passes. Returns whether it held.
//!
//! Bounded so a regression fails the test instead of hanging CI, and used instead of a fixed sleep
//! so the test does not depend on how promptly the scheduler runs a thread.
template <typename Pred>
bool waitFor(Pred pred, std::chrono::milliseconds timeout = std::chrono::seconds{5})
{
    auto const deadline = std::chrono::steady_clock::now() + timeout;
    while (!pred())
    {
        if (std::chrono::steady_clock::now() >= deadline)
        {
            return false;
        }
        std::this_thread::yield();
    }
    return true;
}

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

        std::atomic<bool> otherAttempting{false};
        other = std::thread(
            [&]
            {
                EXPECT_FALSE(mutex.heldExclusiveByThisThread()); // not this thread's lock
                otherAttempting.store(true);
                auto const guard = mutex.lockExclusive();
                otherEntered.store(true);
            });
        // Assert the thread is blocked, not merely unscheduled: wait until it reports that it is
        // about to acquire. The remaining window between that store and the lock() call itself is
        // inherent, so still allow a brief grace period -- but the test now fails if the thread
        // never runs at all, which a fixed sleep alone would have passed.
        ASSERT_TRUE(waitFor([&] { return otherAttempting.load(); })) << "contending thread never ran";
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
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
    std::atomic<bool> go{false};
    std::atomic<bool> allInside{false};
    std::vector<std::thread> readers;
    for (int i = 0; i < kReaders; ++i)
    {
        readers.emplace_back(
            [&]
            {
                // Released together, so the test does not depend on the first reader still holding
                // the lock by the time the last one is scheduled.
                waitFor([&] { return go.load(); });
                auto const guard = mutex.lockShared();
                ++inside;
                // Hold until every reader is inside. If the lock serialised them this never
                // happens and the bounded wait expires, failing rather than hanging.
                waitFor(
                    [&]
                    {
                        if (inside.load() == kReaders)
                        {
                            allInside.store(true);
                        }
                        return allInside.load();
                    });
                --inside;
            });
    }
    go.store(true);
    for (auto& t : readers)
    {
        t.join();
    }
    EXPECT_TRUE(allInside.load()) << "readers serialised; shared lock is behaving exclusively";
}

// A writer must be excluded while readers hold the lock.
TEST(ReentrantSharedMutexTest, WriterWaitsForReaders)
{
    kv::ReentrantSharedMutex mutex;
    std::atomic<bool> writerIn{false};
    std::thread writer;
    {
        auto const reader = mutex.lockShared();
        std::atomic<bool> writerAttempting{false};
        writer = std::thread(
            [&]
            {
                writerAttempting.store(true);
                auto const guard = mutex.lockExclusive();
                writerIn.store(true);
            });
        // As above: confirm the writer actually reached its acquisition point before asserting
        // that it is still outside.
        ASSERT_TRUE(waitFor([&] { return writerAttempting.load(); })) << "writer thread never ran";
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        EXPECT_FALSE(writerIn.load()) << "writer entered while a reader held the lock";
    } // reader released here
    writer.join();
    EXPECT_TRUE(writerIn.load());
}

} // namespace
