/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <torch/extension.h>
#include <vector>

namespace tensorrt_llm
{
inline namespace _v1
{
namespace torch_ext
{

torch::Tensor create_nccl_window_tensor(
    torch::List<int64_t> const& group, at::IntArrayRef shape, torch::ScalarType dtype);

torch::Tensor copy_to_nccl_window(torch::Tensor const& input, torch::List<int64_t> const& group);

torch::Tensor matmul_to_nccl_window(torch::Tensor const& a, torch::Tensor const& b, torch::List<int64_t> const& group);

torch::Tensor add_to_nccl_window(torch::Tensor const& a, torch::Tensor const& b, torch::List<int64_t> const& group);

} // namespace torch_ext
} // namespace _v1
} // namespace tensorrt_llm
