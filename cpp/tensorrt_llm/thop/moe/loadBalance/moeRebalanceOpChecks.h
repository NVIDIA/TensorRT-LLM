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

// Argument checks shared by the per-layer dynamic EPLB ops (HALO-Q, its in-switch copy, and HALO-R).

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace moe_rebalance_op
{

//! Checks that `tensor` is a contiguous CUDA tensor of `dtype` on `device` with at least `minElements` elements.
//! `deviceName` names `device` in the error, such as "the routes CUDA device".
inline void checkCudaTensor(torch::Tensor const& tensor, c10::Device const& device, c10::ScalarType dtype,
    std::int64_t minElements, char const* name, char const* deviceName)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == device, name, " must be on ", deviceName);
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.numel() >= minElements, name, " is too small: expected at least ", minElements,
        " elements, got ", tensor.numel());
}

inline void checkCudaResult(cudaError_t error, char const* operation)
{
    TORCH_CHECK(error == cudaSuccess, operation, " failed: ", cudaGetErrorString(error));
}

inline int checkedInt(std::int64_t value, char const* name)
{
    TORCH_CHECK(value >= INT_MIN && value <= INT_MAX, name, " does not fit in int32");
    return static_cast<int>(value);
}

} // namespace moe_rebalance_op
} // namespace torch_ext

TRTLLM_NAMESPACE_END
