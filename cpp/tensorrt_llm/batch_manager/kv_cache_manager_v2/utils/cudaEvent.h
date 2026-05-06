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

#include <algorithm>
#include <cuda.h>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// FuncGuard<F> — generic RAII scope guard that calls a void() callable on destruction.
// Movable (moved-from instance is disarmed). Not copyable.
// ---------------------------------------------------------------------------
template <typename F>
class FuncGuard
{
public:
    explicit FuncGuard(F&& func)
        : mFunc(std::forward<F>(func))
        , mActive(true)
    {
    }

    ~FuncGuard()
    {
        if (mActive)
        {
            mFunc();
        }
    }

    FuncGuard(FuncGuard&& other) noexcept
        : mFunc(std::move(other.mFunc))
        , mActive(other.mActive)
    {
        other.mActive = false;
    }

    FuncGuard(FuncGuard const&) = delete;
    FuncGuard& operator=(FuncGuard const&) = delete;
    FuncGuard& operator=(FuncGuard&&) = delete;

private:
    F mFunc;
    bool mActive;
};

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
        ++mOutstandingCount;
        T* item = mItems.empty() ? mCreateFn() : popFront();
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
        while (!mItems.empty())
        {
            mDestroyFn(popFront());
        }
    }

    [[nodiscard]] int outstandingCount() const noexcept
    {
        return mOutstandingCount;
    }

    [[nodiscard]] int cachedCount() const noexcept
    {
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
// ---------------------------------------------------------------------------
class CudaEventPool : public SimplePool<CUevent_st, CudaEventPool>
{
public:
    static CudaEventPool& instance();

private:
    CudaEventPool();
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
// Copyable: copies share the same underlying CUevent via shared_ptr.
//           Last copy returns the event to the pool.
// NULL sentinel: always considered complete, no event in flight.
// ---------------------------------------------------------------------------
class CachedCudaEvent
{
public:
    // NULL sentinel: always considered complete, no event in flight.
    static CachedCudaEvent makeNull() noexcept;

    // Normal constructor: gets an event and records it on stream.
    explicit CachedCudaEvent(CudaStream stream);

    // Copyable and movable (shared ownership of the underlying CUevent).
    CachedCudaEvent(CachedCudaEvent const&) = default;
    CachedCudaEvent& operator=(CachedCudaEvent const&) = default;
    CachedCudaEvent(CachedCudaEvent&&) noexcept = default;
    CachedCudaEvent& operator=(CachedCudaEvent&&) noexcept = default;
    ~CachedCudaEvent() = default;

    // Query if the recorded work is done.
    bool queryComplete();

    // Block until complete.
    void synchronize();

    // Insert a stream dependency on this event.
    void waitInStream(CudaStream stream) const;

    // True if no CUevent is held (NULL or already closed by any copy).
    [[nodiscard]] bool isClosed() const noexcept
    {
        return !mEvent || !*mEvent;
    }

    // Release the event back to pool. Visible to ALL copies sharing this event.
    void close();

    // Raw CUevent handle. Returns nullptr for NULL/closed events.
    // Also serves as identity key for deduplication.
    [[nodiscard]] CUevent handle() const noexcept
    {
        return isClosed() ? nullptr : mEvent->get();
    }

private:
    explicit CachedCudaEvent() noexcept = default; // used by makeNull()

    // Shared ownership of the PoolItem. close() resets the inner unique_ptr,
    // visible to all copies. Last shared_ptr drop is a no-op (inner already empty).
    std::shared_ptr<CudaEventPool::PoolItem> mEvent;
};

// ---------------------------------------------------------------------------
// Stream-level helpers.
// ---------------------------------------------------------------------------

// Wait for all events on the given stream. Deduplicates internally.
// Mirrors Python's stream_wait_events() which converts to set() before iterating.
inline void streamWaitEvents(CudaStream stream, std::vector<CachedCudaEvent const*> const& events)
{
    thread_local std::vector<CUevent> handles;
    handles.clear();
    handles.reserve(events.size());
    for (auto const* ev : events)
    {
        if (ev && !ev->isClosed())
            handles.push_back(ev->handle());
    }
    std::sort(handles.begin(), handles.end());
    handles.erase(std::unique(handles.begin(), handles.end()), handles.end());
    for (CUevent h : handles)
        cuCheck(cuStreamWaitEvent(reinterpret_cast<CUstream>(stream), h, 0));
}

// Synchronize and close all events. Deduplicates internally.
// Mirrors Python's set()-based synchronization pattern.
inline void synchronizeAll(std::vector<CachedCudaEvent*> const& events)
{
    thread_local std::vector<CUevent> handles;
    handles.clear();
    handles.reserve(events.size());
    for (auto* ev : events)
    {
        if (!ev->isClosed())
            handles.push_back(ev->handle());
    }
    std::sort(handles.begin(), handles.end());
    handles.erase(std::unique(handles.begin(), handles.end()), handles.end());
    for (CUevent h : handles)
        cuCheck(cuEventSynchronize(h));
    for (auto* ev : events)
        ev->close();
}

// ---------------------------------------------------------------------------
// CachedCudaStream — pooled non-blocking CUstream.
// Mirrors _utils.py::CachedCudaStream.
// ---------------------------------------------------------------------------
class CachedCudaStream
{
public:
    CachedCudaStream();

    CachedCudaStream(CachedCudaStream&&) noexcept = default;
    CachedCudaStream& operator=(CachedCudaStream&&) noexcept = default;
    CachedCudaStream(CachedCudaStream const&) = delete;
    CachedCudaStream& operator=(CachedCudaStream const&) = delete;

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
    void waitEvents(std::vector<CachedCudaEvent const*> const& events)
    {
        streamWaitEvents(reinterpret_cast<CudaStream>(handle()), events);
    }

    CachedCudaEvent recordEvent();
    void synchronize();

private:
    CudaStreamPool::PoolItem mPoolItem; // returns to pool on destruction
};

// ---------------------------------------------------------------------------
// TemporaryCudaStream — pooled stream with finish-event tracking.
// Mirrors Python's TemporaryCudaStream context manager.
//
// Usage (matches Python's `with TemporaryCudaStream(events) as stream:`):
//
//   TemporaryCudaStream tempStream(priorEvents);
//   {
//       auto scope = tempStream.enter();   // __enter__
//       launchKernel(tempStream.get());
//   }                                      // ~Scope → __exit__ records finish event
//   auto ev = tempStream.takeFinishEvent(); // after with block
//
// ---------------------------------------------------------------------------
class TemporaryCudaStream
{
public:
    // Acquire a stream from pool and issue cuStreamWaitEvent for each prior event.
    explicit TemporaryCudaStream(std::vector<CachedCudaEvent const*> const& priorEvents);

    // Begin a scoped block. Destructor records the finish event (= Python __exit__).
    // Skips recording during stack unwinding to match Python's `if not exc_type:` guard.
    [[nodiscard]] auto enter()
    {
        int const exCount = std::uncaught_exceptions();
        return FuncGuard(
            [this, exCount]()
            {
                if (std::uncaught_exceptions() == exCount)
                    mFinishEvent = mStream.recordEvent();
            });
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

    TemporaryCudaStream(TemporaryCudaStream const&) = delete;
    TemporaryCudaStream& operator=(TemporaryCudaStream const&) = delete;

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
