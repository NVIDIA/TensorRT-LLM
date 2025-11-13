/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <limits>

#include "cutlass/epilogue/thread/activation.h"

namespace tensorrt_llm
{
namespace kernels
{

// TODO Could linear search be better for small # experts
template <class T>
__device__ inline int64_t findTotalEltsLessThanTarget(T const* sorted_indices, int64_t const arr_length, T const target)
{
    int64_t low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high)
    {
        int64_t mid = (low + high) / 2;

        if (sorted_indices[mid] >= target)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

namespace cutlass_kernels
{
// ============================== Activation Adaptors =================================

template <template <class> class ActFn>
struct IdentityAdaptor
{
    constexpr static bool IS_GLU = false;
    float alpha = 1.0f;
    float beta = 0.0f;
    float limit = std::numeric_limits<float>::infinity();

    template <class T>
    __device__ T operator()(T const& x) const
    {
        ActFn<T> fn{};
        return fn(x);
    }
};

template <template <class> class ActFn>
struct GLUAdaptor
{
    constexpr static bool IS_GLU = true;
    float alpha = 1.0f;
    float beta = 0.0f;
    float limit = std::numeric_limits<float>::infinity();

    template <class T>
    __device__ T operator()(T const& gate, T const& linear) const
    {
        ActFn<T> fn{};
        return fn(gate) * linear;
    }
};

struct SwigluBiasAdaptor
{
    constexpr static bool IS_GLU = true;
    float alpha = 1.0f;
    float beta = 0.0f;
    float limit = std::numeric_limits<float>::infinity();

    template <class T>
    __device__ T operator()(T const& gate, T const& linear) const
    {
        cutlass::epilogue::thread::Sigmoid<T> fn{};
        T linear_clamped = cutlass::maximum<T>{}(cutlass::minimum<T>{}(linear, limit), -limit);
        T gate_clamped = cutlass::minimum<T>{}(gate, limit);
        return gate_clamped * fn(gate_clamped * alpha) * (linear_clamped + beta);
    }
};

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
