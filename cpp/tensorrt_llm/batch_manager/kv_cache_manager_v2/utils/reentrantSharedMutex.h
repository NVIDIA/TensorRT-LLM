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

#pragma once

#include <atomic>
#include <shared_mutex>
#include <thread>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! A shared_mutex whose *exclusive* side is re-entrant on the owning thread.
//!
//! Motivation: guarding an API surface whose public methods call one another. std::shared_mutex
//! is not recursive, so the usual options are to split every method into a locking wrapper plus
//! an unlocked `_impl`, or to thread a "already locked" flag through every call site. Both are
//! easy to get wrong, and getting them wrong produces a deadlock rather than a diagnosable
//! failure. Here a nested exclusive acquisition on the thread that already owns the lock is a
//! no-op, and the outermost guard performs the release, so internal call sites need no
//! annotation at all.
//!
//! Deliberate non-features. None of these are checked -- a violation is a hang, so the rules
//! below have to be respected by construction:
//!   * The shared side is NOT re-entrant. Acquiring it twice on one thread is undefined behaviour:
//!     [thread.sharedmutex.requirements.general] forbids a thread that already owns the mutex in
//!     any mode from acquiring shared ownership. It happens not to hang on libstdc++ today, whose
//!     default rwlock kind prefers readers, but on a writer-preferring implementation the inner
//!     acquisition queues behind a waiting writer that is itself waiting on the outer one. No
//!     shared holder currently calls another, and none may be added.
//!   * Upgrading shared -> exclusive on one thread is not supported and cannot be: it deadlocks
//!     on std::shared_mutex. Restructure so the write happens outside the read.
//!   * Holding one instance while locking a *different* one is not supported: nothing here defines
//!     a lock order, so two threads nesting a pair of instances in opposite orders would deadlock.
//!     There is exactly one instance per KvCacheManager and no manager calls into another, so the
//!     situation does not arise today; should managers ever need to interact, this class needs a
//!     documented lock order first.
//!   * There is no recursion counter for the exclusive side; correctness comes from the
//!     outermost guard being the only owner, which RAII scoping guarantees.
class ReentrantSharedMutex
{
public:
    ReentrantSharedMutex() = default;
    ReentrantSharedMutex(ReentrantSharedMutex const&) = delete;
    ReentrantSharedMutex& operator=(ReentrantSharedMutex const&) = delete;

    //! RAII handle. Released in reverse order of acquisition; a nested exclusive guard releases
    //! nothing, leaving the outermost one to unlock.
    class Guard
    {
    public:
        Guard(ReentrantSharedMutex const& owner, bool exclusive)
            : mOwner(owner)
            , mExclusive(exclusive)
        {
            if (mOwner.mOwnerThread.load(std::memory_order_relaxed) == std::this_thread::get_id())
            {
                // This thread already holds it exclusively: nested call, nothing to do.
                mOwns = false;
                return;
            }
            if (mExclusive)
            {
                mOwner.mMutex.lock();
                mOwner.mOwnerThread.store(std::this_thread::get_id(), std::memory_order_relaxed);
            }
            else
            {
                mOwner.mMutex.lock_shared();
            }
        }

        ~Guard()
        {
            if (!mOwns)
            {
                return;
            }
            if (mExclusive)
            {
                mOwner.mOwnerThread.store(std::thread::id{}, std::memory_order_relaxed);
                mOwner.mMutex.unlock();
            }
            else
            {
                mOwner.mMutex.unlock_shared();
            }
        }

        Guard(Guard const&) = delete;
        Guard& operator=(Guard const&) = delete;

        // Deliberately not movable. A guard is a scope-local object and cannot cross threads:
        // std::shared_mutex requires the unlocking thread to be the locking thread, and
        // mOwnerThread names the acquiring thread -- both break silently on a cross-thread move.
        // Immobility makes that unrepresentable. Conditional locking, if ever needed, should use
        // a distinct type.
        Guard(Guard&&) = delete;
        Guard& operator=(Guard&&) = delete;

        //! False when this guard nested inside one already held by this thread.
        [[nodiscard]] bool owns() const noexcept
        {
            return mOwns;
        }

    private:
        ReentrantSharedMutex const& mOwner;
        bool mExclusive;
        bool mOwns{true};
    };

    [[nodiscard]] Guard lockExclusive() const
    {
        return {*this, true};
    }

    [[nodiscard]] Guard lockShared() const
    {
        return {*this, false};
    }

    //! True when the calling thread holds the exclusive lock. For assertions in code that
    //! requires the caller to already hold it.
    [[nodiscard]] bool heldExclusiveByThisThread() const noexcept
    {
        return mOwnerThread.load(std::memory_order_relaxed) == std::this_thread::get_id();
    }

private:
    mutable std::shared_mutex mMutex;
    //! Thread holding mMutex exclusively, or a default-constructed id when free. Load-bearing in
    //! all builds. Relaxed ordering suffices because the value is only ever compared against the
    //! *calling* thread's own id: a thread always observes its own stores in program order, and
    //! two live threads cannot share an id, so a stale read can never produce a false
    //! "I already own this".
    mutable std::atomic<std::thread::id> mOwnerThread{std::thread::id{}};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
