/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * Adapted from Baseten's sa_spec library (Apache-2.0)
 * https://github.com/basetenlabs/sa_spec
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "saCudaCallable.h"

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding::suffix_automaton
{

/**
 * @brief A fixed-capacity buffer that uses external memory (pointer-based).
 *
 * This is a view into externally-managed memory. The buffer does not own
 * the memory and does not perform any allocation/deallocation.
 *
 * This design enables:
 * - Runtime-configurable capacity (no compile-time template parameter)
 * - Trivially copyable (can be memcpy'd between host and GPU)
 * - CUDA graph compatible (fixed memory addresses after initialization)
 *
 * @tparam T Element type (must be trivially copyable)
 * @tparam IndexT Index type (default size_t)
 */
template <typename T, typename IndexT = size_t>
struct SABuffer
{
    T* mData{nullptr};
    size_t mCapacity{0};

    T const& at(IndexT, IndexT) const = delete;
    T& at(IndexT, IndexT) = delete;

    SABuffer() = default;

    SA_CUDA_CALLABLE void init(T* data, size_t capacity)
    {
        mData = data;
        mCapacity = capacity;
    }

    SA_CUDA_CALLABLE T const& at(IndexT row) const
    {
        assert(static_cast<size_t>(+row) < mCapacity);
        return mData[+row];
    }

    SA_CUDA_CALLABLE T& at(IndexT row)
    {
        assert(static_cast<size_t>(+row) < mCapacity);
        return mData[+row];
    }

    struct Iterator
    {
        SABuffer const& buffer;
        IndexT index;

        SA_CUDA_CALLABLE Iterator(SABuffer const& buffer, IndexT index)
            : buffer(buffer)
            , index(index)
        {
        }

        SA_CUDA_CALLABLE T const& operator*() const
        {
            return buffer.at(index);
        }

        SA_CUDA_CALLABLE Iterator& operator++()
        {
            index = IndexT(+index + 1);
            return *this;
        }

        SA_CUDA_CALLABLE bool operator==(Iterator const& other) const
        {
            return index == other.index;
        }

        SA_CUDA_CALLABLE bool operator!=(Iterator const& other) const
        {
            return index != other.index;
        }
    };

    SA_CUDA_CALLABLE Iterator begin() const
    {
        return Iterator(*this, IndexT(0));
    }

    SA_CUDA_CALLABLE Iterator end() const
    {
        return Iterator(*this, IndexT(mCapacity));
    }

    SA_CUDA_CALLABLE size_t size() const
    {
        return mCapacity;
    }

    SA_CUDA_CALLABLE size_t capacity() const
    {
        return mCapacity;
    }

    void clear()
    {
        if (mData && mCapacity > 0)
        {
            memset(static_cast<void*>(mData), 0, mCapacity * sizeof(T));
        }
    }

    SA_CUDA_CALLABLE T* data()
    {
        return mData;
    }

    SA_CUDA_CALLABLE T const* data() const
    {
        return mData;
    }

    /**
     * @brief Relocate the data pointer by a given delta.
     *
     * Used when copying between host and GPU memory to adjust pointers.
     */
    void relocate(ptrdiff_t delta)
    {
        if (mData)
        {
            mData = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(mData) + delta);
        }
    }

    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
};

/**
 * @brief A dynamic buffer with runtime-configurable capacity using external memory.
 *
 * Like SABuffer, but tracks current length separately from capacity.
 * Supports push/pop operations up to the capacity limit.
 *
 * @tparam T Element type (must be trivially copyable)
 * @tparam IndexT Index type (default size_t)
 */
template <typename T, typename IndexT = size_t>
struct SADynamicBuffer
{
    T* mData{nullptr};
    size_t mCapacity{0};
    IndexT mLength{0};

    SADynamicBuffer() = default;

    SA_CUDA_CALLABLE void init(T* data, size_t capacity)
    {
        mData = data;
        mCapacity = capacity;
        mLength = IndexT(0);
    }

    SA_CUDA_CALLABLE void clear()
    {
        mLength = IndexT(0);
    }

    SA_CUDA_CALLABLE IndexT size() const
    {
        return mLength;
    }

    SA_CUDA_CALLABLE size_t capacity() const
    {
        return mCapacity;
    }

    SA_CUDA_CALLABLE bool empty() const
    {
        return +size() == 0;
    }

    SA_CUDA_CALLABLE void extend(size_t n)
    {
        mLength = IndexT(+mLength + n);
        assert(static_cast<size_t>(+mLength) <= mCapacity);
    }

    SA_CUDA_CALLABLE T& pushBack(T const& value)
    {
        assert(static_cast<size_t>(+mLength) < mCapacity);

        T& result = mData[+mLength];
        result = value;
        mLength = IndexT(+mLength + 1);
        return result;
    }

    SA_CUDA_CALLABLE T& pushBack(T&& value)
    {
        assert(static_cast<size_t>(+mLength) < mCapacity);
        T& result = mData[+mLength];
        result = std::move(value);
        mLength = IndexT(+mLength + 1);
        return result;
    }

    SA_CUDA_CALLABLE T& popBack()
    {
        assert(!empty());
        T& result = mData[+mLength - 1];
        mLength = IndexT(+mLength - 1);
        return result;
    }

    SA_CUDA_CALLABLE T const& at(IndexT row) const
    {
        assert(row < mLength);
        return mData[+row];
    }

    SA_CUDA_CALLABLE T& at(IndexT row)
    {
        assert(row < mLength);
        return mData[+row];
    }

    SA_CUDA_CALLABLE T* data()
    {
        return mData;
    }

    SA_CUDA_CALLABLE T const* data() const
    {
        return mData;
    }

    SA_CUDA_CALLABLE bool hasCapacity() const
    {
        return static_cast<size_t>(+mLength) < mCapacity;
    }

    /**
     * @brief Relocate the data pointer by a given delta.
     *
     * Used when copying between host and GPU memory to adjust pointers.
     */
    void relocate(ptrdiff_t delta)
    {
        if (mData)
        {
            mData = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(mData) + delta);
        }
    }

    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
};

// Verify that our buffer types are trivially copyable (required for GPU memcpy)
static_assert(std::is_trivially_copyable<SABuffer<int>>::value, "SABuffer must be trivially copyable");
static_assert(std::is_trivially_copyable<SADynamicBuffer<int>>::value, "SADynamicBuffer must be trivially copyable");

} // namespace kernels::speculative_decoding::suffix_automaton

TRTLLM_NAMESPACE_END
