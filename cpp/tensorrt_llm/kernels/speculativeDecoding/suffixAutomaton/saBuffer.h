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

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <vector>

#include "saCudaCallable.h"
#include "tensorrt_llm/common/assert.h"

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding::suffix_automaton
{

template <typename T, size_t Size, typename IndexT = size_t>
struct SABuffer
{
    T const& at(IndexT, IndexT) const = delete;
    T& at(IndexT, IndexT) = delete;

    SABuffer() = default;

    SA_CUDA_CALLABLE T const& at(IndexT row) const
    {
        TLLM_ASSERT(static_cast<size_t>(+row) < Size);
        return mData[+row];
    }

    SA_CUDA_CALLABLE T& at(IndexT row)
    {
        TLLM_ASSERT(static_cast<size_t>(+row) < Size);
        return mData[+row];
    }

    struct Iterator
    {
        SABuffer const& vector;
        IndexT index;

        SA_CUDA_CALLABLE Iterator(SABuffer const& vector, IndexT index)
            : vector(vector)
            , index(index)
        {
        }

        SA_CUDA_CALLABLE T const& operator*() const
        {
            return vector.at(index);
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
        return Iterator(*this, IndexT(Size));
    }

    SA_CUDA_CALLABLE size_t size() const
    {
        return Size;
    }

    bool operator==(SABuffer<T, Size, IndexT> const& other) const
    {
        static_assert(sizeof(decltype(*this)) == sizeof(mData));
        return std::memcmp(this, &other, sizeof(mData)) == 0;
    }

    template <typename Func>
    void visitChunks(Func&& func) const
    {
        func(static_cast<void const*>(this), sizeof(*this));
    }

    void clear()
    {
        memset(static_cast<void*>(&mData[0]), 0, sizeof(mData));
    }

    T* data()
    {
        return &mData[0];
    }

    T const* data() const
    {
        return &mData[0];
    }

    std::array<T, Size> mData;

    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
};

template <typename T, size_t Size, typename IndexT = size_t>
struct SADynamicBuffer
{

    SADynamicBuffer() = default;

    SA_CUDA_CALLABLE void clear()
    {
        mLength = IndexT(0);
    }

    SA_CUDA_CALLABLE IndexT size() const
    {
        return mLength;
    }

    SA_CUDA_CALLABLE bool empty() const
    {
        return +size() == 0;
    }

    SA_CUDA_CALLABLE void extend(size_t n)
    {
        mLength = IndexT(+mLength + n);
        TLLM_ASSERT(static_cast<size_t>(+mLength) <= Size);
    }

    SA_CUDA_CALLABLE T& pushBack(T const& value)
    {
        TLLM_ASSERT(static_cast<size_t>(+mLength) < Size);

        T& result = mData.at(mLength);
        result = value;
        mLength = IndexT(+mLength + 1);
        return result;
    }

    SA_CUDA_CALLABLE T& pushBack(T&& value)
    {
        TLLM_ASSERT(static_cast<size_t>(+mLength) < Size);
        T& result = mData.at(mLength);
        result = std::move(value);
        mLength = IndexT(+mLength + 1);
        return result;
    }

    SA_CUDA_CALLABLE T& popBack()
    {
        TLLM_ASSERT(!empty());
        T& result = mData.at(IndexT(+mLength - 1));
        mLength = IndexT(+mLength - 1);
        return result;
    }

    SA_CUDA_CALLABLE T const& at(IndexT row) const
    {
        TLLM_ASSERT(row < mLength);
        return mData.at(row);
    }

    SA_CUDA_CALLABLE T& at(IndexT row)
    {
        TLLM_ASSERT(row < mLength);
        return mData.at(row);
    }

    template <typename Func>
    void visitChunks(Func&& func) const
    {
        func(static_cast<void const*>(this),
            reinterpret_cast<std::byte const*>(&mData.mData[+mLength]) - reinterpret_cast<std::byte const*>(this));
    }

    T* data()
    {
        return mData.data();
    }

    T const* data() const
    {
        return mData.data();
    }

    SA_CUDA_CALLABLE bool hasCapacity() const
    {
        return +mLength < Size;
    }

    // IMPORTANT: pay attention to visitChunks when modifying below
    IndexT mLength{0};
    SABuffer<T, Size, IndexT> mData;
};

} // namespace kernels::speculative_decoding::suffix_automaton

TRTLLM_NAMESPACE_END
