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

#include "tensorrt_llm/common/assert.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! A move-only vector holding up to N elements inline and moving to the heap beyond that.
template <typename T, uint32_t N>
class SmallVector
{
    static_assert(N > 0);
    static_assert(std::is_nothrow_move_constructible_v<T> && std::is_nothrow_move_assignable_v<T>);

public:
    using value_type = T;
    using iterator = T*;
    using const_iterator = T const*;

    SmallVector() noexcept = default;

    explicit SmallVector(size_t count)
    {
        resize(count);
    }

    SmallVector(SmallVector&& other) noexcept
    {
        take(other);
    }

    SmallVector& operator=(SmallVector&& other) noexcept
    {
        if (this != &other)
        {
            release();
            take(other);
        }
        return *this;
    }

    ~SmallVector()
    {
        release();
    }

    SmallVector(SmallVector const&) = delete;
    SmallVector& operator=(SmallVector const&) = delete;

    [[nodiscard]] size_t size() const noexcept
    {
        return mSize;
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return mSize == 0;
    }

    [[nodiscard]] T* data() noexcept
    {
        return mData;
    }

    [[nodiscard]] T const* data() const noexcept
    {
        return mData;
    }

    [[nodiscard]] T* begin() noexcept
    {
        return mData;
    }

    [[nodiscard]] T* end() noexcept
    {
        return mData + mSize;
    }

    [[nodiscard]] T const* begin() const noexcept
    {
        return mData;
    }

    [[nodiscard]] T const* end() const noexcept
    {
        return mData + mSize;
    }

    [[nodiscard]] T const* cbegin() const noexcept
    {
        return mData;
    }

    [[nodiscard]] T const* cend() const noexcept
    {
        return mData + mSize;
    }

    [[nodiscard]] T& operator[](size_t index) noexcept
    {
        TLLM_CHECK_DEBUG(index < mSize);
        return mData[index];
    }

    [[nodiscard]] T const& operator[](size_t index) const noexcept
    {
        TLLM_CHECK_DEBUG(index < mSize);
        return mData[index];
    }

    [[nodiscard]] T& at(size_t index)
    {
        if (index >= mSize)
        {
            throw std::out_of_range("SmallVector index out of range");
        }
        return mData[index];
    }

    [[nodiscard]] T const& at(size_t index) const
    {
        if (index >= mSize)
        {
            throw std::out_of_range("SmallVector index out of range");
        }
        return mData[index];
    }

    [[nodiscard]] T& front() noexcept
    {
        TLLM_CHECK_DEBUG(mSize > 0);
        return mData[0];
    }

    [[nodiscard]] T const& front() const noexcept
    {
        TLLM_CHECK_DEBUG(mSize > 0);
        return mData[0];
    }

    [[nodiscard]] T& back() noexcept
    {
        TLLM_CHECK_DEBUG(mSize > 0);
        return mData[mSize - 1];
    }

    [[nodiscard]] T const& back() const noexcept
    {
        TLLM_CHECK_DEBUG(mSize > 0);
        return mData[mSize - 1];
    }

    void reserve(size_t capacity)
    {
        if (capacity > mCapacity)
        {
            TLLM_CHECK(capacity <= std::numeric_limits<uint32_t>::max());
            moveTo(static_cast<T*>(::operator new(sizeof(T) * capacity)));
            mCapacity = static_cast<uint32_t>(capacity);
        }
    }

    //! Default-construct new elements or destroy the trailing ones.
    void resize(size_t count)
    {
        reserve(count);
        while (mSize < count)
        {
            new (mData + mSize) T();
            ++mSize;
        }
        while (mSize > count)
        {
            pop_back();
        }
    }

    void pop_back() noexcept
    {
        TLLM_CHECK_DEBUG(mSize > 0);
        mData[--mSize].~T();
    }

    template <typename... Args>
    T& emplace_back(Args&&... args)
    {
        if (mSize < mCapacity)
        {
            T* const element = new (mData + mSize) T(std::forward<Args>(args)...);
            ++mSize;
            return *element;
        }
        // Full here (mSize == mCapacity >= N); max() lets GCC's -Warray-bounds see a nonzero size.
        uint32_t const capacity = std::max(mSize, N) * 2;
        T* const buffer = static_cast<T*>(::operator new(sizeof(T) * capacity));
        T* element = nullptr;
        try
        {
            // Built before the old elements move, since args may name one of them.
            element = new (buffer + mSize) T(std::forward<Args>(args)...);
        }
        catch (...)
        {
            ::operator delete(static_cast<void*>(buffer));
            throw;
        }
        moveTo(buffer);
        mCapacity = capacity;
        ++mSize;
        return *element;
    }

    void push_back(T const& value)
    {
        emplace_back(value);
    }

    void push_back(T&& value)
    {
        emplace_back(std::move(value));
    }

    T* erase(T* position) noexcept
    {
        TLLM_CHECK_DEBUG(position >= begin() && position < end());
        std::move(position + 1, end(), position);
        mData[--mSize].~T();
        return position;
    }

    //! Destroy the elements and keep the storage.
    void clear() noexcept
    {
        for (uint32_t i = 0; i < mSize; ++i)
        {
            mData[i].~T();
        }
        mSize = 0;
    }

private:
    [[nodiscard]] bool isInline() const noexcept
    {
        return mData == reinterpret_cast<T const*>(mInline);
    }

    //! Destroy the elements and free a heap buffer, leaving the object empty and inline.
    void release() noexcept
    {
        clear();
        if (!isInline())
        {
            ::operator delete(static_cast<void*>(mData));
            mData = reinterpret_cast<T*>(mInline);
            mCapacity = N;
        }
    }

    //! Take the elements of `other`, which must be empty and inline here; `other` ends empty.
    void take(SmallVector& other) noexcept
    {
        if (other.isInline())
        {
            for (uint32_t i = 0; i < other.mSize; ++i)
            {
                new (mData + i) T(std::move(other.mData[i]));
                other.mData[i].~T();
            }
        }
        else
        {
            mData = std::exchange(other.mData, reinterpret_cast<T*>(other.mInline));
            mCapacity = std::exchange(other.mCapacity, N);
        }
        mSize = std::exchange(other.mSize, 0);
    }

    void moveTo(T* buffer) noexcept
    {
        for (uint32_t i = 0; i < mSize; ++i)
        {
            new (buffer + i) T(std::move(mData[i]));
            mData[i].~T();
        }
        if (!isInline())
        {
            ::operator delete(static_cast<void*>(mData));
        }
        mData = buffer;
    }

    alignas(T) unsigned char mInline[sizeof(T) * N];
    T* mData = reinterpret_cast<T*>(mInline);
    uint32_t mSize = 0;
    uint32_t mCapacity = N;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
