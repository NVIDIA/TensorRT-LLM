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

#include <optional>

#include "saCudaCallable.h"

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding::suffix_automaton
{

// ABI-consistent alternative to std::optional for CUDA compatibility
template <typename T>
struct SAOptional
{
    T mData;
    bool mHasValue = false;

    SA_CUDA_CALLABLE bool hasValue() const
    {
        return mHasValue;
    }

    SA_CUDA_CALLABLE T& value()
    {
        return mData;
    }

    SA_CUDA_CALLABLE T const& value() const
    {
        return mData;
    }

    SA_CUDA_CALLABLE T& operator*()
    {
        return mData;
    }

    SA_CUDA_CALLABLE T const& operator*() const
    {
        return mData;
    }

    SA_CUDA_CALLABLE T* operator->()
    {
        return &mData;
    }

    SA_CUDA_CALLABLE T const* operator->() const
    {
        return &mData;
    }

    SA_CUDA_CALLABLE SAOptional& operator=(T const& value)
    {
        mData = value;
        mHasValue = true;
        return *this;
    }

    SA_CUDA_CALLABLE SAOptional& operator=(T&& value)
    {
        mData = std::move(value);
        mHasValue = true;
        return *this;
    }

    SA_CUDA_CALLABLE SAOptional& operator=(std::nullopt_t)
    {
        mHasValue = false;
        return *this;
    }

    SA_CUDA_CALLABLE SAOptional()
        : mHasValue(false)
    {
    }

    SA_CUDA_CALLABLE SAOptional(T const& value)
        : mData(value)
        , mHasValue(true)
    {
    }

    SA_CUDA_CALLABLE SAOptional(T&& value)
        : mData(std::move(value))
        , mHasValue(true)
    {
    }

    SA_CUDA_CALLABLE SAOptional(std::nullopt_t)
        : mHasValue(false)
    {
    }

    SA_CUDA_CALLABLE void reset()
    {
        mHasValue = false;
    }
};

} // namespace kernels::speculative_decoding::suffix_automaton

TRTLLM_NAMESPACE_END
