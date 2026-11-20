/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <initializer_list>
#include <limits>
#include <torch/extension.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace prob1Checks
{
inline void devices(torch::Tensor const& base, std::initializer_list<torch::Tensor const*> tensors)
{
    TORCH_CHECK(base.is_cuda(), "draft_prob1: base must be CUDA");
    for (auto const* tensor : tensors)
        TORCH_CHECK(tensor->is_cuda() && tensor->device() == base.device(),
            "draft_prob1: every tensor must be CUDA on base.device()");
}

inline void narrow(int64_t value, char const* name)
{
    TORCH_CHECK(value >= std::numeric_limits<int>::min() && value <= std::numeric_limits<int>::max(),
        "draft_prob1: ", name, " does not fit int32");
}
} // namespace prob1Checks

TRTLLM_MV2_NAMESPACE_END
