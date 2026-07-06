/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <optional>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

void runCompactPseudoKvAttention(torch::Tensor const& q, torch::Tensor& output,
    std::optional<torch::Tensor> const& compactPseudokvKey, std::optional<torch::Tensor> const& compactPseudokvValue,
    std::optional<torch::Tensor> const& compactPseudokvPositions,
    std::optional<torch::Tensor> const& compactPseudokvCausalMask,
    std::optional<int64_t> compactPseudokvSourceSeqLen);

torch::Tensor compactPseudoKvAttention(torch::Tensor const& q, torch::Tensor output,
    torch::Tensor const& compactPseudokvKey, torch::Tensor const& compactPseudokvValue,
    torch::Tensor const& compactPseudokvPositions, torch::Tensor const& compactPseudokvCausalMask,
    int64_t compactPseudokvSourceSeqLen);

} // namespace torch_ext

TRTLLM_NAMESPACE_END
