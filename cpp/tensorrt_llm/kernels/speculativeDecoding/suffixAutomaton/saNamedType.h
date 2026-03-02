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

#include <functional>
#include <type_traits>

#include "saCudaCallable.h"

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding::suffix_automaton
{

template <typename T, typename Tag, T DefaultValue = T()>
class NamedType
{
public:
    using ValueType = T;
    static constexpr auto Default = DefaultValue;

    constexpr NamedType()
    {
        static_assert(sizeof(decltype(*this)) == sizeof(T));
    }

    SA_CUDA_CALLABLE explicit constexpr NamedType(T const& value)
        : value_(value)
    {
    }

    SA_CUDA_CALLABLE explicit constexpr NamedType(T&& value)
        : value_(std::move(value))
    {
    }

    SA_CUDA_CALLABLE constexpr T const& get() const
    {
        return value_;
    }

    SA_CUDA_CALLABLE constexpr T& get()
    {
        return value_;
    }

    SA_CUDA_CALLABLE friend constexpr bool operator==(NamedType const& lhs, NamedType const& rhs)
    {
        return lhs.value_ == rhs.value_;
    }

    SA_CUDA_CALLABLE friend constexpr bool operator!=(NamedType const& lhs, NamedType const& rhs)
    {
        return lhs.value_ != rhs.value_;
    }

    SA_CUDA_CALLABLE friend constexpr bool operator<(NamedType const& lhs, NamedType const& rhs)
    {
        return lhs.value_ < rhs.value_;
    }

    SA_CUDA_CALLABLE friend constexpr T operator+(NamedType const& lhs)
    {
        return lhs.value_;
    }

    T value_{DefaultValue};

    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
};

} // namespace kernels::speculative_decoding::suffix_automaton

TRTLLM_NAMESPACE_END

// Hash function for std::unordered_map
namespace std
{
template <typename T, typename Tag>
struct hash<tensorrt_llm::kernels::speculative_decoding::suffix_automaton::NamedType<T, Tag>>
{
    std::size_t operator()(
        tensorrt_llm::kernels::speculative_decoding::suffix_automaton::NamedType<T, Tag> const& namedType) const
    {
        return std::hash<T>{}(namedType.get());
    }
};
} // namespace std
