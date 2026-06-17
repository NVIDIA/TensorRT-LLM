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
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

template <typename T, typename Tag, T DefaultValue = T{}>
class StrongIndex
{
    static_assert(std::is_integral<T>::value, "StrongIndex requires an integral underlying type");

public:
    using ValueType = T;

    constexpr StrongIndex() noexcept = default;

    explicit constexpr StrongIndex(T value) noexcept
        : mValue(value)
    {
    }

    [[nodiscard]] constexpr T value() const noexcept
    {
        return mValue;
    }

    template <typename U, typename = std::enable_if_t<std::is_integral<U>::value>>
    constexpr StrongIndex& operator+=(U rhs) noexcept
    {
        mValue = static_cast<T>(mValue + static_cast<T>(rhs));
        return *this;
    }

    template <typename U, typename = std::enable_if_t<std::is_integral<U>::value>>
    constexpr StrongIndex& operator-=(U rhs) noexcept
    {
        mValue = static_cast<T>(mValue - static_cast<T>(rhs));
        return *this;
    }

    constexpr StrongIndex& operator++() noexcept
    {
        ++mValue;
        return *this;
    }

    constexpr StrongIndex operator++(int) noexcept
    {
        StrongIndex old{*this};
        ++(*this);
        return old;
    }

    constexpr StrongIndex& operator--() noexcept
    {
        --mValue;
        return *this;
    }

    constexpr StrongIndex operator--(int) noexcept
    {
        StrongIndex old{*this};
        --(*this);
        return old;
    }

private:
    T mValue{DefaultValue};
};

template <typename T, typename Tag, T DefaultValue>
constexpr bool operator==(StrongIndex<T, Tag, DefaultValue> lhs, StrongIndex<T, Tag, DefaultValue> rhs) noexcept
{
    return lhs.value() == rhs.value();
}

template <typename T, typename Tag, T DefaultValue>
constexpr bool operator!=(StrongIndex<T, Tag, DefaultValue> lhs, StrongIndex<T, Tag, DefaultValue> rhs) noexcept
{
    return !(lhs == rhs);
}

template <typename T, typename Tag, T DefaultValue>
constexpr bool operator<(StrongIndex<T, Tag, DefaultValue> lhs, StrongIndex<T, Tag, DefaultValue> rhs) noexcept
{
    return lhs.value() < rhs.value();
}

template <typename T, typename Tag, T DefaultValue>
constexpr bool operator<(StrongIndex<T, Tag, DefaultValue> lhs, T rhs) noexcept
{
    return lhs.value() < rhs;
}

template <typename T, typename Tag, T DefaultValue>
constexpr bool operator>(StrongIndex<T, Tag, DefaultValue> lhs, StrongIndex<T, Tag, DefaultValue> rhs) noexcept
{
    return rhs < lhs;
}

template <typename T, typename Tag, T DefaultValue>
constexpr bool operator<=(StrongIndex<T, Tag, DefaultValue> lhs, StrongIndex<T, Tag, DefaultValue> rhs) noexcept
{
    return !(rhs < lhs);
}

template <typename T, typename Tag, T DefaultValue>
constexpr bool operator>=(StrongIndex<T, Tag, DefaultValue> lhs, StrongIndex<T, Tag, DefaultValue> rhs) noexcept
{
    return !(lhs < rhs);
}

template <typename T, typename Tag, T DefaultValue>
constexpr bool operator>=(StrongIndex<T, Tag, DefaultValue> lhs, T rhs) noexcept
{
    return lhs.value() >= rhs;
}

template <typename T, typename Tag, T DefaultValue, typename U, typename = std::enable_if_t<std::is_integral<U>::value>>
constexpr StrongIndex<T, Tag, DefaultValue> operator+(StrongIndex<T, Tag, DefaultValue> lhs, U rhs) noexcept
{
    lhs += rhs;
    return lhs;
}

template <typename U, typename T, typename Tag, T DefaultValue, typename = std::enable_if_t<std::is_integral<U>::value>>
constexpr StrongIndex<T, Tag, DefaultValue> operator+(U lhs, StrongIndex<T, Tag, DefaultValue> rhs) noexcept
{
    rhs += lhs;
    return rhs;
}

template <typename T, typename Tag, T DefaultValue, typename U, typename = std::enable_if_t<std::is_integral<U>::value>>
constexpr StrongIndex<T, Tag, DefaultValue> operator-(StrongIndex<T, Tag, DefaultValue> lhs, U rhs) noexcept
{
    lhs -= rhs;
    return lhs;
}

template <typename T, typename Tag, T LhsDefaultValue, T RhsDefaultValue>
constexpr T operator-(StrongIndex<T, Tag, LhsDefaultValue> lhs, StrongIndex<T, Tag, RhsDefaultValue> rhs) noexcept
{
    return static_cast<T>(lhs.value() - rhs.value());
}

template <typename T, typename Tag, T DefaultValue>
[[nodiscard]] std::size_t toSizeT(StrongIndex<T, Tag, DefaultValue> index) noexcept
{
    if constexpr (std::is_signed<T>::value)
    {
        TLLM_CHECK_DEBUG_WITH_INFO(index.value() >= 0, "StrongIndex value must be non-negative for size_t conversion");
    }
    return static_cast<std::size_t>(index.value());
}

template <typename Index, typename T>
class TypedVec
{
public:
    using IndexType = Index;
    using ValueType = T;
    using ContainerType = std::vector<T>;
    using iterator = typename ContainerType::iterator;
    using const_iterator = typename ContainerType::const_iterator;

    TypedVec() = default;

    explicit TypedVec(Index count)
        : mData(toSizeT(count))
    {
    }

    TypedVec(Index count, T const& value)
        : mData(toSizeT(count), value)
    {
    }

    TypedVec(std::initializer_list<T> values)
        : mData(values)
    {
    }

    explicit TypedVec(ContainerType data)
        : mData(std::move(data))
    {
    }

    [[nodiscard]] T& operator[](Index index) noexcept
    {
        return mData[toSizeT(index)];
    }

    [[nodiscard]] T const& operator[](Index index) const noexcept
    {
        return mData[toSizeT(index)];
    }

    [[nodiscard]] T& at(Index index)
    {
        return mData.at(toSizeT(index));
    }

    [[nodiscard]] T const& at(Index index) const
    {
        return mData.at(toSizeT(index));
    }

    [[nodiscard]] Index size() const noexcept
    {
        return Index{static_cast<typename Index::ValueType>(mData.size())};
    }

    [[nodiscard]] std::size_t stdSize() const noexcept
    {
        return mData.size();
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return mData.empty();
    }

    void clear() noexcept
    {
        mData.clear();
    }

    void reserve(Index count)
    {
        mData.reserve(toSizeT(count));
    }

    void resize(Index count)
    {
        mData.resize(toSizeT(count));
    }

    void resize(Index count, T const& value)
    {
        mData.resize(toSizeT(count), value);
    }

    void push_back(T const& value)
    {
        mData.push_back(value);
    }

    void push_back(T&& value)
    {
        mData.push_back(std::move(value));
    }

    template <typename... Args>
    T& emplace_back(Args&&... args)
    {
        return mData.emplace_back(std::forward<Args>(args)...);
    }

    void pop_back()
    {
        mData.pop_back();
    }

    [[nodiscard]] T& front()
    {
        return mData.front();
    }

    [[nodiscard]] T const& front() const
    {
        return mData.front();
    }

    [[nodiscard]] T& back()
    {
        return mData.back();
    }

    [[nodiscard]] T const& back() const
    {
        return mData.back();
    }

    [[nodiscard]] iterator begin() noexcept
    {
        return mData.begin();
    }

    [[nodiscard]] const_iterator begin() const noexcept
    {
        return mData.begin();
    }

    [[nodiscard]] const_iterator cbegin() const noexcept
    {
        return mData.cbegin();
    }

    [[nodiscard]] iterator end() noexcept
    {
        return mData.end();
    }

    [[nodiscard]] const_iterator end() const noexcept
    {
        return mData.end();
    }

    [[nodiscard]] const_iterator cend() const noexcept
    {
        return mData.cend();
    }

    [[nodiscard]] ContainerType& raw() noexcept
    {
        return mData;
    }

    [[nodiscard]] ContainerType const& raw() const noexcept
    {
        return mData;
    }

    friend bool operator==(TypedVec const& lhs, TypedVec const& rhs)
    {
        return lhs.mData == rhs.mData;
    }

    friend bool operator!=(TypedVec const& lhs, TypedVec const& rhs)
    {
        return !(lhs == rhs);
    }

private:
    ContainerType mData;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2

namespace std
{

template <typename T, typename Tag, T DefaultValue>
struct hash<tensorrt_llm::batch_manager::kv_cache_manager_v2::StrongIndex<T, Tag, DefaultValue>>
{
    size_t operator()(
        tensorrt_llm::batch_manager::kv_cache_manager_v2::StrongIndex<T, Tag, DefaultValue> index) const noexcept
    {
        return std::hash<T>{}(index.value());
    }
};

} // namespace std
