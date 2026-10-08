/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <ATen/ATen.h>
#include <optional>
#include <tuple>

namespace tensorrt_llm::rdna4
{
at::Tensor rmsNorm(at::Tensor const& input, at::Tensor const& weight, double epsilon);
std::tuple<at::Tensor, at::Tensor> addRmsNorm(
    at::Tensor const& input, at::Tensor const& residual, at::Tensor const& weight, double epsilon);
at::Tensor layerNorm(
    at::Tensor const& input, at::Tensor const& weight, std::optional<at::Tensor> const& bias, double epsilon);
at::Tensor gatedActivation(at::Tensor const& gate, at::Tensor const& up, int64_t activation);
at::Tensor rotary(at::Tensor const& input, at::Tensor const& cos, at::Tensor const& sin, bool interleaved);
at::Tensor attention(at::Tensor const& query, at::Tensor const& key, at::Tensor const& value,
    std::optional<at::Tensor> const& bias, bool causal, int64_t queryStart, double scale);
} // namespace tensorrt_llm::rdna4
