/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @brief A dynamic buffer with runtime-configurable capacity using external memory.
 *
 * A view into externally-managed memory that tracks current length separately
 * from capacity, supporting push operations up to the capacity limit.
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
static_assert(std::is_trivially_copyable<SADynamicBuffer<int>>::value, "SADynamicBuffer must be trivially copyable");

} // namespace kernels::speculative_decoding::suffix_automaton

TRTLLM_NAMESPACE_END
