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

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/utils/funcGuard.h"

#include <algorithm>
#include <atomic>
#include <cuda.h>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// SimplePool<T, Derived> — generic resource pool for opaque handle types.
// Mirrors _utils.py::SimplePool.
//
// T is the pointed-to type (e.g. CUevent_st, CUstream_st).
// CreateFn returns T*, DestroyFn takes T*.
// get() returns a PoolItem (unique_ptr<T, Deleter>) with zero extra allocation —
// the unique_ptr directly wraps the handle pointer.
//
// Derived (CRTP, default void):
//   - void: instance pool — Deleter stores a SimplePool* pointer (8 bytes).
//   - non-void: singleton pool — Deleter is stateless (0 bytes), calls
//     Derived::instance() to find the pool. PoolItem is pointer-sized.
//
// Thread safety: all public methods are mutex-guarded. The singleton pools are
// process-wide, so they are shared by every KvCacheManager and cannot be
// covered by any manager's API lock; and for events, put() runs from whichever
// thread retires or last drops a CachedCudaEvent, which is by design not tied
// to a lock at all.
//
// The mutex costs ~6ns against the ~160ns of make_shared plus cuEventRecord
// that every get() already pays, and the pools are touched per resize,
// migration and suspend/resume rather than per page. Should that ever show up
// in a profile, the fix is a thread_local pool per thread, with a depleted
// pool stealing a batch from another thread's pool; the mutex would then be
// needed only on the steal path, off the common path entirely.
// ---------------------------------------------------------------------------

// Forward declare so Deleters can reference it.
template <typename T, typename Derived = void>
class SimplePool;

// Deleter for instance pools (Derived == void): stores a pool pointer.
template <typename T>
struct InstancePoolDeleter
{
    SimplePool<T, void>* pool = nullptr;

    void operator()(T* ptr) const noexcept;
};

// Deleter for singleton pools (Derived != void): stateless, zero-size.
template <typename T, typename Derived>
struct SingletonPoolDeleter
{
    void operator()(T* ptr) const noexcept;
};

template <typename T, typename Derived>
class SimplePool
{
public:
    using CreateFn = std::function<T*()>;
    using DestroyFn = std::function<void(T*)>;

    using Deleter
        = std::conditional_t<std::is_void_v<Derived>, InstancePoolDeleter<T>, SingletonPoolDeleter<T, Derived>>;
    using PoolItem = std::unique_ptr<T, Deleter>;

    SimplePool(CreateFn createFn, DestroyFn destroyFn, int initSize = 0, std::optional<int> maxSize = std::nullopt)
        : mCreateFn(std::move(createFn))
        , mDestroyFn(std::move(destroyFn))
        , mMaxSize(maxSize)
        , mOutstandingCount(0)
    {
        for (int i = 0; i < initSize; ++i)
        {
            mItems.push_back(mCreateFn());
        }
    }

    ~SimplePool()
    {
        clear();
    }

    SimplePool(SimplePool const&) = delete;
    SimplePool& operator=(SimplePool const&) = delete;

    // Get a resource wrapped in a PoolItem that auto-returns to pool on destruction.
    [[nodiscard]] PoolItem get()
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        // Increment only after the item is successfully obtained, so a throwing
        // mCreateFn() leaves mOutstandingCount unchanged (no leak in stats).
        T* item = mItems.empty() ? mCreateFn() : popFront();
        ++mOutstandingCount;
        if constexpr (std::is_void_v<Derived>)
        {
            return PoolItem(item, Deleter{this});
        }
        else
        {
            return PoolItem(item, Deleter{});
        }
    }

    void clear()
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        while (!mItems.empty())
        {
            mDestroyFn(popFront());
        }
    }

    [[nodiscard]] int outstandingCount() const noexcept
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        return mOutstandingCount;
    }

    [[nodiscard]] int cachedCount() const noexcept
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        return static_cast<int>(mItems.size());
    }

private:
    friend struct InstancePoolDeleter<T>;
    friend struct SingletonPoolDeleter<T, Derived>;

    T* popFront()
    {
        T* item = mItems.front();
        mItems.pop_front();
        return item;
    }

    void put(T* item)
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        --mOutstandingCount;
        if (mMaxSize.has_value() && static_cast<int>(mItems.size()) >= *mMaxSize)
        {
            mDestroyFn(item);
        }
        else
        {
            mItems.push_back(item);
        }
    }

    CreateFn mCreateFn;
    DestroyFn mDestroyFn;
    std::optional<int> mMaxSize;
    std::deque<T*> mItems;
    int mOutstandingCount;
    //! Guards every member above. Mutable so the const count accessors can lock.
    mutable std::mutex mMutex;
};

// Deleter implementations (after SimplePool is fully defined).
template <typename T>
void InstancePoolDeleter<T>::operator()(T* ptr) const noexcept
{
    if (pool)
    {
        pool->put(ptr);
    }
}

template <typename T, typename Derived>
void SingletonPoolDeleter<T, Derived>::operator()(T* ptr) const noexcept
{
    Derived::instance().put(ptr);
}

// ---------------------------------------------------------------------------
// CudaEventPool — singleton CRTP pool for CUevent handles.
//
// Two properties of this pool are load-bearing for CachedCudaEvent, which
// retires events without holding any manager's API lock:
//
//   Unbounded. Constructed with no maxSize, so put() always returns the handle
//   to mItems and never calls cuEventDestroy. A bounded event pool would let
//   put() destroy a CUevent that a caller has already copied out of a
//   CachedCudaEvent (see streamWaitEvents) and is about to pass to the driver.
//
//   FIFO. get() pops the front and put() pushes the back, so a retired handle
//   is reissued only after everything else cached ahead of it. That reuse
//   distance is what bounds the window in which a copied-out handle can be
//   re-recorded by another owner. While a CachedCudaEvent copy is alive the
//   worst case there is a spurious dependency, since every retirement it can
//   observe follows completion; a handle outliving the last copy is not covered,
//   because the destructor retires with work possibly still in flight.
// ---------------------------------------------------------------------------
class CudaEventPool : public SimplePool<CUevent_st, CudaEventPool>
{
public:
    static CudaEventPool& instance();

private:
    CudaEventPool();
};

// ---------------------------------------------------------------------------
// PooledEvent — the payload CachedCudaEvent copies share.
//
// Holds the CUevent in an atomic so that retirement is a single exchange:
// concurrent retirers race, exactly one observes the non-null handle, and only
// that one returns it to the pool. Relaxed ordering suffices — the atomicity of
// the exchange is what picks the winner, and the pool's own mutex orders the
// handle's reuse against the next get().
//
// Retirement is deliberately not tied to any lock: it runs from the
// exclusive-lock slot sweep and from whichever thread drops the last reference.
// ---------------------------------------------------------------------------
class PooledEvent
{
public:
    // Takes the handle out of the PoolItem: from here on it is this object's
    // destructor, not the item's, that returns it.
    explicit PooledEvent(CudaEventPool::PoolItem item) noexcept
        : mHandle(item.release())
    {
    }

    PooledEvent(PooledEvent const&) = delete;
    PooledEvent& operator=(PooledEvent const&) = delete;

    ~PooledEvent()
    {
        retire();
    }

    // The handle, or nullptr once retired. Callers must load once and use the
    // loaded value: a second load may observe another thread's retirement.
    [[nodiscard]] CUevent load() const noexcept
    {
        return mHandle.load(std::memory_order_relaxed);
    }

    // Return the handle to the pool. Safe to call concurrently and repeatedly;
    // only the caller that wins the exchange puts.
    void retire() const noexcept
    {
        if (CUevent event = mHandle.exchange(nullptr, std::memory_order_relaxed); event != nullptr)
        {
            CudaEventPool::Deleter{}(event);
        }
    }

private:
    static_assert(std::atomic<CUevent>::is_always_lock_free);

    mutable std::atomic<CUevent> mHandle;
};

// ---------------------------------------------------------------------------
// CudaStreamPool — singleton CRTP pool for CUstream handles.
// ---------------------------------------------------------------------------
class CudaStreamPool : public SimplePool<CUstream_st, CudaStreamPool>
{
public:
    static CudaStreamPool& instance();

private:
    CudaStreamPool();
};

// ---------------------------------------------------------------------------
// CachedCudaEvent — pooled CUevent (no timing).
// Mirrors _utils.py::CachedCudaEvent.
//
// On construction: gets an event from the global pool and records it to stream.
// Copyable: copies share the same underlying CUevent via a PooledEvent.
//           Closing any copy returns the event to the pool, for all of them.
// NULL sentinel: always considered complete, no event in flight.
//
// Internal only -- do not bind to Python. Handing one out means copying it out of a live
// KvCache, and that read races the exclusive-lock paths that assign the member it comes from;
// such a copy must be taken under the API lock. The copy itself is then safe to use unlocked.
// Tests reach it through the _introspection submodule.
//
// Thread safety. The query-and-close methods may run concurrently on copies of
// the same event. What makes that sound is PooledEvent's exchange, which lets
// exactly one caller return the handle. Two rules follow for anything added here:
//
//   Load the handle once per call and operate on the loaded value. A method
//   that tests the handle and then re-reads it can see a concurrent close
//   between the two and pass nullptr to the driver.
//
//   Treat a handle that has left this object as valid only while the caller
//   holds the exclusive API lock. Concurrent closers cannot run under it, so
//   the handle cannot be reissued mid-call; without it the handle may name
//   another owner's work by the time it is used.
// ---------------------------------------------------------------------------
class CachedCudaEvent
{
public:
    // NULL sentinel: always considered complete, no event in flight.
    static CachedCudaEvent makeNull() noexcept;

    // Gets an event and records it on stream. Failure terminates because KVCM2 cannot safely operate without events.
    explicit CachedCudaEvent(CudaStream stream) noexcept;

    // Copyable and movable (shared ownership of the underlying CUevent).
    CachedCudaEvent(CachedCudaEvent const&) = default;
    CachedCudaEvent& operator=(CachedCudaEvent const&) = default;
    CachedCudaEvent(CachedCudaEvent&&) noexcept = default;
    CachedCudaEvent& operator=(CachedCudaEvent&&) noexcept = default;
    ~CachedCudaEvent() = default;

    // Query if the recorded work is done, closing the event if it is.
    [[nodiscard]] bool queryComplete() const;

    // Block until complete.
    void synchronize() const;

    // Insert a stream dependency on this event.
    void waitInStream(CudaStream stream) const;

    // True if no CUevent is held (NULL or already closed by any copy).
    [[nodiscard]] bool isClosed() const noexcept
    {
        return handle() == nullptr;
    }

    // Release the event back to pool. Visible to ALL copies sharing this event.
    void close() const noexcept;

    // Raw CUevent handle. Returns nullptr for NULL/closed events.
    // Also serves as identity key for deduplication.
    [[nodiscard]] CUevent handle() const noexcept
    {
        return mEvent ? mEvent->load() : nullptr;
    }

private:
    explicit CachedCudaEvent() noexcept = default; // used by makeNull()

    // Shared ownership of the payload. close() retires the handle inside it,
    // visible to all copies. Last shared_ptr drop retires whatever remains.
    std::shared_ptr<PooledEvent> mEvent;
};

// ---------------------------------------------------------------------------
// Stream-level helpers.
// ---------------------------------------------------------------------------

// Wait for all events on the given stream, skipping nulls and issuing one wait per distinct
// event. Mirrors Python's stream_wait_events(), which converts to set() before iterating.
// Waiting does not consume an event, so raw CUevent values suffice and a caller-owned
// CachedCudaEvent need not outlive the call.
inline void streamWaitEvents(CudaStream stream, std::vector<CUevent> events)
{
    events.erase(std::remove(events.begin(), events.end(), nullptr), events.end());
    std::sort(events.begin(), events.end());
    events.erase(std::unique(events.begin(), events.end()), events.end());
    for (CUevent h : events)
        cuCheck(cuStreamWaitEvent(reinterpret_cast<CUstream>(stream), h, 0));
}

// ---------------------------------------------------------------------------
// CachedCudaStream — pooled non-blocking CUstream.
// Mirrors _utils.py::CachedCudaStream.
// ---------------------------------------------------------------------------
class CachedCudaStream
{
public:
    CachedCudaStream();

    [[nodiscard]] CUstream handle() const noexcept
    {
        return mPoolItem.get(); // CUstream = CUstream_st*
    }

    // Wait for a single event on this stream.
    void waitEvent(CachedCudaEvent const& event) const
    {
        event.waitInStream(reinterpret_cast<CudaStream>(handle()));
    }

    // Wait for all events on this stream. Deduplicates internally.
    void waitEvents(std::vector<CUevent> events)
    {
        streamWaitEvents(reinterpret_cast<CudaStream>(handle()), std::move(events));
    }

    CachedCudaEvent recordEvent() noexcept;
    void synchronize();

private:
    CudaStreamPool::PoolItem mPoolItem; // returns to pool on destruction
};

// ---------------------------------------------------------------------------
// TemporaryCudaStream — pooled stream with finish-event tracking.
// Unlike the Python context manager, it records the finish event during exception unwinding so that work submitted
// before the exception can still be fenced.
//
// Usage (matches Python's `with TemporaryCudaStream(events) as stream:`):
//
//   TemporaryCudaStream tempStream(priorEvents);
//   {
//       auto scope = tempStream.enter();   // __enter__
//       launchKernel(tempStream.get());
//   }                                      // scope exit records the event; failure terminates
//   auto ev = tempStream.takeFinishEvent(); // after with block
//
// ---------------------------------------------------------------------------
class TemporaryCudaStream
{
public:
    // Acquire a stream from pool and issue cuStreamWaitEvent for each prior event.
    explicit TemporaryCudaStream(std::vector<CUevent> priorEvents);

    // Begin a scoped block. Destructor records the finish event, including during stack unwinding.
    [[nodiscard]] auto enter()
    {
        return FuncGuard([this]() noexcept { mFinishEvent = mStream.recordEvent(); });
    }

    [[nodiscard]] CUstream get() const noexcept
    {
        return mStream.handle();
    }

    // Consume the finish event recorded by Scope destructor.
    [[nodiscard]] CachedCudaEvent takeFinishEvent()
    {
        auto result = std::move(mFinishEvent);
        mFinishEvent = CachedCudaEvent::makeNull();
        return result;
    }

private:
    CachedCudaStream mStream;
    CachedCudaEvent mFinishEvent = CachedCudaEvent::makeNull();
};

// Merge multiple CUDA events into one.
// Returns makeNull() for 0 live events, the single live event for 1,
// or a TemporaryCudaStream-merged event for many.
// Mirrors Python's merge_events() utility.
CachedCudaEvent mergeEvents(std::vector<CachedCudaEvent>& events);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
