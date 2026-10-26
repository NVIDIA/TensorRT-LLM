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

// Non-atomic SharedPtr / WeakPtr — drop-in replacements for std::shared_ptr
// and std::weak_ptr that use plain int refcounts instead of atomics.
//
// Motivation: std::shared_ptr uses atomic increments/decrements for thread
// safety.  On ARM (Grace), atomics are significantly more expensive than on
// x86, and the KV cache manager does not need them.
//
// THREADING REQUIREMENT: these refcounts are plain ints, so concurrent
// manipulation of the same control block is a data race -- including between
// two readers.  Safety comes from KvCacheManager's API lock serialising every
// path that touches a SharedPtr, plus the fact that a KvCache is owned by one
// thread at a time (see AGENTS.md, "Concurrency model").  Adding an unlocked
// code path that copies or destroys one of these breaks that invariant
// silently: there is no atomic to save you.
//
// Usage:
//   SharedPtr<T>              replaces  std::shared_ptr<T>
//   WeakPtr<T>                replaces  std::weak_ptr<T>
//   EnableSharedFromThis<T>   replaces  std::enable_shared_from_this<T>
//   makeShared<T>(args...)    replaces  std::make_shared<T>(args...)
//   dynamicPointerCast<T>(p)  replaces  std::dynamic_pointer_cast<T>(p)
//   toStd(SharedPtr<T>)       bridges   SharedPtr<T> → std::shared_ptr<T>

#pragma once

#include "tensorrt_llm/common/assert.h"
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// Forward declarations.
template <typename T>
class SharedPtr;
template <typename T>
class WeakPtr;
template <typename T>
class EnableSharedFromThis;
template <typename... Ts>
class SlabArena;

// =========================================================================
// Control block
// =========================================================================

namespace detail
{

struct ControlBlockBase
{
    int strongCount = 1;
    int weakCount = 1; // +1 bias while strongCount > 0

    virtual void destroyObject() noexcept = 0;
    virtual void deallocate() noexcept = 0;

    void addStrongRef() noexcept
    {
        ++strongCount;
    }

    void releaseStrongRef() noexcept
    {
        if (--strongCount == 0)
        {
            destroyObject();
            releaseWeakRef(); // release the bias
        }
    }

    void addWeakRef() noexcept
    {
        ++weakCount;
    }

    void releaseWeakRef() noexcept
    {
        if (--weakCount == 0)
        {
            deallocate();
        }
    }

protected:
    ~ControlBlockBase() = default; // prevent polymorphic delete via base
};

template <typename T>
struct InplaceControlBlock final : ControlBlockBase
{
    // Aligned uninitialized storage for T.
    alignas(T) unsigned char storage[sizeof(T)];

    T* ptr() noexcept
    {
        return std::launder(reinterpret_cast<T*>(&storage));
    }

    void destroyObject() noexcept override
    {
        ptr()->~T();
    }

    void deallocate() noexcept override
    {
        delete this;
    }
};

// One allocation backing many control blocks, freed after the last of them and its arena.
struct SlabHeader
{
    int liveBlocks = 1;
};

inline void releaseSlab(SlabHeader* slab) noexcept
{
    if (--slab->liveBlocks == 0)
    {
        slab->~SlabHeader();
        ::operator delete(static_cast<void*>(slab));
    }
}

template <typename T>
struct SlabControlBlock final : ControlBlockBase
{
    explicit SlabControlBlock(SlabHeader* owner) noexcept
        : slab(owner)
    {
    }

    alignas(T) unsigned char storage[sizeof(T)];
    SlabHeader* slab;

    T* ptr() noexcept
    {
        return std::launder(reinterpret_cast<T*>(&storage));
    }

    void destroyObject() noexcept override
    {
        ptr()->~T();
    }

    void deallocate() noexcept override
    {
        SlabHeader* const owner = slab;
        this->~SlabControlBlock();
        releaseSlab(owner);
    }
};

// ---------------------------------------------------------------------------
// EnableSharedFromThis detection — handles polymorphic inheritance.
// E.g. makeShared<CommittedPage>() detects EnableSharedFromThis<Page>
// inherited through Page.
// ---------------------------------------------------------------------------
template <typename T>
auto detectEnableImpl(EnableSharedFromThis<T>*) -> EnableSharedFromThis<T>;

auto detectEnableImpl(...) -> void;

template <typename T>
using EnableBase = decltype(detectEnableImpl(std::declval<T*>()));

template <typename T>
inline constexpr bool hasEnable = !std::is_void_v<EnableBase<T>>;

} // namespace detail

// =========================================================================
// SharedPtr<T>
// =========================================================================

template <typename T>
class SharedPtr
{
public:
    // -- Constructors -------------------------------------------------------

    SharedPtr() noexcept
        : mPtr(nullptr)
        , mCb(nullptr)
    {
    }

    SharedPtr(std::nullptr_t) noexcept // NOLINT(google-explicit-constructor)
        : mPtr(nullptr)
        , mCb(nullptr)
    {
    }

    SharedPtr(SharedPtr const& other) noexcept
        : mPtr(other.mPtr)
        , mCb(other.mCb)
    {
        if (mCb)
            mCb->addStrongRef();
    }

    SharedPtr(SharedPtr&& other) noexcept
        : mPtr(other.mPtr)
        , mCb(other.mCb)
    {
        other.mPtr = nullptr;
        other.mCb = nullptr;
    }

    // Converting copy (U* implicitly convertible to T*).
    template <typename U, std::enable_if_t<std::is_convertible_v<U*, T*>, int> = 0>
    SharedPtr(SharedPtr<U> const& other) noexcept // NOLINT(google-explicit-constructor)
        : mPtr(other.mPtr)
        , mCb(other.mCb)
    {
        if (mCb)
            mCb->addStrongRef();
    }

    // Converting move.
    template <typename U, std::enable_if_t<std::is_convertible_v<U*, T*>, int> = 0>
    SharedPtr(SharedPtr<U>&& other) noexcept // NOLINT(google-explicit-constructor)
        : mPtr(other.mPtr)
        , mCb(other.mCb)
    {
        other.mPtr = nullptr;
        other.mCb = nullptr;
    }

    ~SharedPtr()
    {
        if (mCb)
            mCb->releaseStrongRef();
    }

    // -- Assignment ---------------------------------------------------------

    SharedPtr& operator=(SharedPtr const& other) noexcept
    {
        if (this != &other)
        {
            SharedPtr tmp(other);
            swap(tmp);
        }
        return *this;
    }

    SharedPtr& operator=(SharedPtr&& other) noexcept
    {
        if (this != &other)
        {
            SharedPtr tmp(std::move(other));
            swap(tmp);
        }
        return *this;
    }

    template <typename U, std::enable_if_t<std::is_convertible_v<U*, T*>, int> = 0>
    SharedPtr& operator=(SharedPtr<U> const& other) noexcept
    {
        SharedPtr tmp(other);
        swap(tmp);
        return *this;
    }

    template <typename U, std::enable_if_t<std::is_convertible_v<U*, T*>, int> = 0>
    SharedPtr& operator=(SharedPtr<U>&& other) noexcept
    {
        SharedPtr tmp(std::move(other));
        swap(tmp);
        return *this;
    }

    SharedPtr& operator=(std::nullptr_t) noexcept
    {
        reset();
        return *this;
    }

    // -- Observers ----------------------------------------------------------

    T* get() const noexcept
    {
        return mPtr;
    }

    T& operator*() const noexcept
    {
        return *mPtr;
    }

    T* operator->() const noexcept
    {
        return mPtr;
    }

    explicit operator bool() const noexcept
    {
        return mPtr != nullptr;
    }

    int useCount() const noexcept
    {
        return mCb ? mCb->strongCount : 0;
    }

    // -- Modifiers ----------------------------------------------------------

    void reset() noexcept
    {
        SharedPtr().swap(*this);
    }

    void swap(SharedPtr& other) noexcept
    {
        std::swap(mPtr, other.mPtr);
        std::swap(mCb, other.mCb);
    }

    // -- Comparisons --------------------------------------------------------

    bool operator==(SharedPtr const& other) const noexcept
    {
        return mPtr == other.mPtr;
    }

    bool operator!=(SharedPtr const& other) const noexcept
    {
        return mPtr != other.mPtr;
    }

    bool operator==(std::nullptr_t) const noexcept
    {
        return mPtr == nullptr;
    }

    bool operator!=(std::nullptr_t) const noexcept
    {
        return mPtr != nullptr;
    }

private:
    // Aliasing constructor — shares control block from `owner`, stores `ptr`.
    // Used by dynamicPointerCast and EnableSharedFromThis.
    SharedPtr(detail::ControlBlockBase* cb, T* ptr) noexcept
        : mPtr(ptr)
        , mCb(cb)
    {
        if (mCb)
            mCb->addStrongRef();
    }

    template <typename U>
    friend class SharedPtr;
    template <typename U>
    friend class WeakPtr;
    template <typename U>
    friend class EnableSharedFromThis;
    template <typename U, typename... Args>
    friend SharedPtr<U> makeShared(Args&&... args);
    template <typename To, typename From>
    friend SharedPtr<To> dynamicPointerCast(SharedPtr<From> const&);
    template <typename U>
    friend std::shared_ptr<U> toStd(SharedPtr<U> const&);
    template <typename... Us>
    friend class SlabArena;

    T* mPtr;
    detail::ControlBlockBase* mCb;
};

// Free-standing comparison with nullptr (reversed operand order).
template <typename T>
bool operator==(std::nullptr_t, SharedPtr<T> const& sp) noexcept
{
    return sp == nullptr;
}

template <typename T>
bool operator!=(std::nullptr_t, SharedPtr<T> const& sp) noexcept
{
    return sp != nullptr;
}

// =========================================================================
// WeakPtr<T>
// =========================================================================

template <typename T>
class WeakPtr
{
public:
    WeakPtr() noexcept
        : mPtr(nullptr)
        , mCb(nullptr)
    {
    }

    WeakPtr(SharedPtr<T> const& sp) noexcept // NOLINT(google-explicit-constructor)
        : mPtr(sp.mPtr)
        , mCb(sp.mCb)
    {
        if (mCb)
            mCb->addWeakRef();
    }

    // Converting constructor from SharedPtr<U>.
    template <typename U, std::enable_if_t<std::is_convertible_v<U*, T*>, int> = 0>
    WeakPtr(SharedPtr<U> const& sp) noexcept // NOLINT(google-explicit-constructor)
        : mPtr(sp.mPtr)
        , mCb(sp.mCb)
    {
        if (mCb)
            mCb->addWeakRef();
    }

    WeakPtr(WeakPtr const& other) noexcept
        : mPtr(other.mPtr)
        , mCb(other.mCb)
    {
        if (mCb)
            mCb->addWeakRef();
    }

    WeakPtr(WeakPtr&& other) noexcept
        : mPtr(other.mPtr)
        , mCb(other.mCb)
    {
        other.mPtr = nullptr;
        other.mCb = nullptr;
    }

    ~WeakPtr()
    {
        if (mCb)
            mCb->releaseWeakRef();
    }

    // -- Assignment ---------------------------------------------------------

    WeakPtr& operator=(SharedPtr<T> const& sp) noexcept
    {
        WeakPtr tmp(sp);
        swap(tmp);
        return *this;
    }

    template <typename U, std::enable_if_t<std::is_convertible_v<U*, T*>, int> = 0>
    WeakPtr& operator=(SharedPtr<U> const& sp) noexcept
    {
        WeakPtr tmp(sp);
        swap(tmp);
        return *this;
    }

    WeakPtr& operator=(WeakPtr const& other) noexcept
    {
        if (this != &other)
        {
            WeakPtr tmp(other);
            swap(tmp);
        }
        return *this;
    }

    WeakPtr& operator=(WeakPtr&& other) noexcept
    {
        if (this != &other)
        {
            WeakPtr tmp(std::move(other));
            swap(tmp);
        }
        return *this;
    }

    // -- Observers ----------------------------------------------------------

    bool expired() const noexcept
    {
        return !mCb || mCb->strongCount == 0;
    }

    SharedPtr<T> lock() const noexcept
    {
        if (expired())
            return SharedPtr<T>();
        // Object still alive — construct a SharedPtr sharing the control block.
        SharedPtr<T> result;
        result.mPtr = mPtr;
        result.mCb = mCb;
        mCb->addStrongRef();
        return result;
    }

    // -- Modifiers ----------------------------------------------------------

    void reset() noexcept
    {
        WeakPtr().swap(*this);
    }

    void swap(WeakPtr& other) noexcept
    {
        std::swap(mPtr, other.mPtr);
        std::swap(mCb, other.mCb);
    }

private:
    template <typename U>
    friend class SharedPtr;
    template <typename U>
    friend class WeakPtr;
    template <typename U>
    friend class EnableSharedFromThis;

    T* mPtr;
    detail::ControlBlockBase* mCb;
};

// =========================================================================
// EnableSharedFromThis<T>
// =========================================================================

template <typename T>
class EnableSharedFromThis
{
public:
    SharedPtr<T> sharedFromThis()
    {
        auto sp = mWeakThis.lock();
        TLLM_CHECK_DEBUG_WITH_INFO(sp, "sharedFromThis() called on object not owned by SharedPtr");
        return sp;
    }

    SharedPtr<T const> sharedFromThis() const
    {
        auto sp = mWeakThis.lock();
        TLLM_CHECK_DEBUG_WITH_INFO(sp, "sharedFromThis() called on object not owned by SharedPtr");
        // Convert SharedPtr<T> to SharedPtr<T const> via the converting constructor.
        return sp;
    }

protected:
    EnableSharedFromThis() noexcept = default;

    // Copy/move must NOT copy mWeakThis — the new object is a distinct entity.
    EnableSharedFromThis(EnableSharedFromThis const&) noexcept {}

    EnableSharedFromThis& operator=(EnableSharedFromThis const&) noexcept
    {
        return *this;
    }

    ~EnableSharedFromThis() = default;

private:
    template <typename U, typename... Args>
    friend SharedPtr<U> makeShared(Args&&... args);
    template <typename... Us>
    friend class SlabArena;

    mutable WeakPtr<T> mWeakThis;
};

// =========================================================================
// makeShared<T>(args...)
// =========================================================================

template <typename T, typename... Args>
SharedPtr<T> makeShared(Args&&... args)
{
    auto* cb = new detail::InplaceControlBlock<T>();
    try
    {
        new (cb->storage) T(std::forward<Args>(args)...);
    }
    catch (...)
    {
        // T's constructor threw — control block was never fully initialized.
        // Release directly; destroyObject() must not run.
        delete cb;
        throw;
    }

    SharedPtr<T> result;
    result.mPtr = cb->ptr();
    result.mCb = cb;

    // Wire up EnableSharedFromThis if T (or a base) derives from it.
    if constexpr (detail::hasEnable<T>)
    {
        result.mPtr->detail::template EnableBase<T>::mWeakThis = result;
    }

    return result;
}

// =========================================================================
// SlabArena<Ts...> — control blocks for up to `capacity` objects of each listed type in one
// allocation, so objects created together sit together in memory. Each slot is used at most
// once; the allocation is freed after the arena and every object made from it are gone.
// Same threading requirement as SharedPtr: the counts are plain ints.
// =========================================================================

template <typename... Ts>
class SlabArena
{
public:
    explicit SlabArena(size_t capacity)
        : mCapacity(capacity)
    {
        if (capacity == 0)
        {
            return;
        }
        size_t bytes = sizeof(detail::SlabHeader);
        ((bytes = placeArray<Ts>(bytes)), ...);
        mBase = static_cast<unsigned char*>(::operator new(bytes));
        mHeader = new (mBase) detail::SlabHeader();
    }

    ~SlabArena()
    {
        if (mHeader != nullptr)
        {
            detail::releaseSlab(mHeader);
        }
    }

    SlabArena(SlabArena const&) = delete;
    SlabArena& operator=(SlabArena const&) = delete;

    //! Like makeShared<T>, in the next unused T slot; falls back to the heap when none is left.
    template <typename T, typename... Args>
    SharedPtr<T> makeShared(Args&&... args)
    {
        constexpr size_t kIndex = indexOf<T>();
        if (mHeader == nullptr || mUsed[kIndex] == mCapacity)
        {
            return kv_cache_manager_v2::makeShared<T>(std::forward<Args>(args)...);
        }
        void* const address = mBase + mOffsets[kIndex] + mUsed[kIndex] * sizeof(detail::SlabControlBlock<T>);
        auto* cb = new (address) detail::SlabControlBlock<T>(mHeader);
        try
        {
            new (cb->storage) T(std::forward<Args>(args)...);
        }
        catch (...)
        {
            cb->~SlabControlBlock();
            throw;
        }
        ++mUsed[kIndex];
        ++mHeader->liveBlocks;

        SharedPtr<T> result;
        result.mPtr = cb->ptr();
        result.mCb = cb;
        if constexpr (detail::hasEnable<T>)
        {
            result.mPtr->detail::template EnableBase<T>::mWeakThis = result;
        }
        return result;
    }

private:
    template <typename T>
    static constexpr size_t indexOf()
    {
        static_assert((std::is_same_v<T, Ts> || ...), "type is not part of this arena");
        size_t index = 0;
        bool found = false;
        ((found = found || std::is_same_v<T, Ts>, index += found ? 0 : 1), ...);
        return index;
    }

    template <typename T>
    size_t placeArray(size_t offset)
    {
        constexpr size_t kAlign = alignof(detail::SlabControlBlock<T>);
        static_assert(kAlign <= __STDCPP_DEFAULT_NEW_ALIGNMENT__);
        offset = (offset + kAlign - 1) / kAlign * kAlign;
        mOffsets[indexOf<T>()] = offset;
        return offset + mCapacity * sizeof(detail::SlabControlBlock<T>);
    }

    size_t mCapacity;
    unsigned char* mBase = nullptr;
    detail::SlabHeader* mHeader = nullptr;
    size_t mOffsets[sizeof...(Ts)] = {};
    size_t mUsed[sizeof...(Ts)] = {};
};

// =========================================================================
// dynamicPointerCast<To>(SharedPtr<From>)
// =========================================================================

template <typename To, typename From>
SharedPtr<To> dynamicPointerCast(SharedPtr<From> const& src)
{
    To* raw = dynamic_cast<To*>(src.get());
    if (!raw)
        return SharedPtr<To>();
    // Aliasing: share control block, different pointer.
    return SharedPtr<To>(src.mCb, raw);
}

// =========================================================================
// toStd — one-way bridge to std::shared_ptr for nanobind boundary
// =========================================================================

template <typename T>
std::shared_ptr<T> toStd(SharedPtr<T> const& sp)
{
    if (!sp)
        return nullptr;
    // The custom deleter captures a copy of the SharedPtr, keeping
    // the non-atomic refcount alive.  When the last std::shared_ptr
    // copy dies, the captured SharedPtr is destroyed and releases its
    // non-atomic reference.
    SharedPtr<T> copy(sp);
    return std::shared_ptr<T>(sp.get(), [captured = std::move(copy)](T*) mutable { captured.reset(); });
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
