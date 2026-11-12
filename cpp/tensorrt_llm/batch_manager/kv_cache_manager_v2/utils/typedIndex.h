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
#include <array>
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

namespace detail
{
//! A StrongIndex is its value and nothing else: one member, no padding, no vtable. Code that
//! stages indices in a buffer a kernel reads, or hands them to a copy that moves bytes, rests
//! on this; checked here so none of it has to say so again.
//!
//! Spelled in the native types rather than the fixed-width aliases: int64_t names one of
//! long and long long, so those would leave the other unchecked. Every type is_integral
//! admits is here, which is wider than the arithmetic ones -- char is a third type beside
//! its signed and unsigned spellings, and bool and the character types are integral too.
template <typename T>
using IndexLayoutProbe = StrongIndex<T, struct IndexLayoutProbeTag>;

#define KVCM2_CHECK_INDEX_LAYOUT(T)                                                                                    \
    static_assert(sizeof(IndexLayoutProbe<T>) == sizeof(T));                                                           \
    static_assert(alignof(IndexLayoutProbe<T>) == alignof(T));                                                         \
    static_assert(std::is_trivially_copyable_v<IndexLayoutProbe<T>>);                                                  \
    static_assert(std::is_standard_layout_v<IndexLayoutProbe<T>>)

KVCM2_CHECK_INDEX_LAYOUT(bool);
KVCM2_CHECK_INDEX_LAYOUT(char);
KVCM2_CHECK_INDEX_LAYOUT(signed char);
KVCM2_CHECK_INDEX_LAYOUT(short);
KVCM2_CHECK_INDEX_LAYOUT(int);
KVCM2_CHECK_INDEX_LAYOUT(long);
KVCM2_CHECK_INDEX_LAYOUT(long long);
KVCM2_CHECK_INDEX_LAYOUT(unsigned char);
KVCM2_CHECK_INDEX_LAYOUT(unsigned short);
KVCM2_CHECK_INDEX_LAYOUT(unsigned int);
KVCM2_CHECK_INDEX_LAYOUT(unsigned long);
KVCM2_CHECK_INDEX_LAYOUT(unsigned long long);
KVCM2_CHECK_INDEX_LAYOUT(wchar_t);
KVCM2_CHECK_INDEX_LAYOUT(char8_t);
KVCM2_CHECK_INDEX_LAYOUT(char16_t);
KVCM2_CHECK_INDEX_LAYOUT(char32_t);

#undef KVCM2_CHECK_INDEX_LAYOUT
} // namespace detail

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
    //! The container's own reference types rather than `T&`/`T const&`, so that
    //! TypedVec<Index, bool> works: std::vector<bool> hands out a proxy, not a `bool&`.
    //! For every other T these are exactly `T&` and `T const&`.
    using reference = typename ContainerType::reference;
    using const_reference = typename ContainerType::const_reference;

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

    [[nodiscard]] reference operator[](Index index) noexcept
    {
        return mData[toSizeT(index)];
    }

    [[nodiscard]] const_reference operator[](Index index) const noexcept
    {
        return mData[toSizeT(index)];
    }

    [[nodiscard]] reference at(Index index)
    {
        return mData.at(toSizeT(index));
    }

    [[nodiscard]] const_reference at(Index index) const
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
    reference emplace_back(Args&&... args)
    {
        return mData.emplace_back(std::forward<Args>(args)...);
    }

    void pop_back()
    {
        mData.pop_back();
    }

    [[nodiscard]] reference front()
    {
        return mData.front();
    }

    [[nodiscard]] const_reference front() const
    {
        return mData.front();
    }

    [[nodiscard]] reference back()
    {
        return mData.back();
    }

    [[nodiscard]] const_reference back() const
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

//! Index of an EnumMap: a StrongIndex as elsewhere, or a plain enum whose enumerators run
//! from zero. Both name a position; neither is an integer a caller can pass by accident.
template <typename Index>
[[nodiscard]] constexpr std::size_t toMapSlot(Index index) noexcept
{
    if constexpr (std::is_enum_v<Index>)
    {
        return static_cast<std::size_t>(index);
    }
    else
    {
        return toSizeT(index);
    }
}

// ---------------------------------------------------------------------------
// EnumMap -- one value per enumerator, in an array rather than an allocation.
//
// For a set whose size is known when the type is written: the slots are fixed, so a lookup
// is a load at a constant displacement where the value it replaces was arithmetic over
// members the compiler cannot fold. TypedVec is the one to reach for when the count is a
// runtime answer.
// ---------------------------------------------------------------------------
template <typename Index, typename T, std::size_t Count>
class EnumMap
{
public:
    using IndexType = Index;
    using ValueType = T;

    constexpr EnumMap() = default;

    constexpr explicit EnumMap(std::array<T, Count> const& values)
        : mData(values)
    {
    }

    [[nodiscard]] constexpr T& operator[](Index index) noexcept
    {
        return mData[toMapSlot(index)];
    }

    [[nodiscard]] constexpr T const& operator[](Index index) const noexcept
    {
        return mData[toMapSlot(index)];
    }

    [[nodiscard]] constexpr T& at(Index index)
    {
        return mData.at(toMapSlot(index));
    }

    [[nodiscard]] constexpr T const& at(Index index) const
    {
        return mData.at(toMapSlot(index));
    }

    [[nodiscard]] static constexpr std::size_t size() noexcept
    {
        return Count;
    }

    [[nodiscard]] constexpr typename std::array<T, Count>::iterator begin() noexcept
    {
        return mData.begin();
    }

    [[nodiscard]] constexpr typename std::array<T, Count>::iterator end() noexcept
    {
        return mData.end();
    }

    [[nodiscard]] constexpr typename std::array<T, Count>::const_iterator begin() const noexcept
    {
        return mData.begin();
    }

    [[nodiscard]] constexpr typename std::array<T, Count>::const_iterator end() const noexcept
    {
        return mData.end();
    }

private:
    std::array<T, Count> mData{};
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
