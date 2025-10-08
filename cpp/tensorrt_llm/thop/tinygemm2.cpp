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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

// CUDA forward declarations
torch::Tensor tinygemm2_cuda_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);

// C++ interface
namespace torch_ext
{
torch::Tensor tinygemm2_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{
    auto const smVersion = tensorrt_llm::common::getSMVersion();
    TORCH_CHECK(
        smVersion == 90 || smVersion == 100 || smVersion == 103, "tinygemm2 only supports SM90, SM100, and SM103.");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(input.sizes()[1] == weight.sizes()[1], "input.size(1) must match weight.size(1)");
    TORCH_CHECK(weight.sizes()[0] == bias.sizes()[0], "weight.size(0) must match bias.size(0)");
    CHECK_INPUT(input, torch::kBFloat16);
    CHECK_INPUT(weight, torch::kBFloat16);
    CHECK_INPUT(bias, torch::kBFloat16);
    return tinygemm2_cuda_forward(input, weight, bias);
}
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("tinygemm2(Tensor input, Tensor weight, Tensor bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("tinygemm2", &torch_ext::tinygemm2_forward);
}
