
/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/runtime/mcastDeviceMemory.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <torch/extension.h>

namespace tensorrt_llm::kernels
{
torch::Tensor twoShotAllReduceDispatch(tensorrt_llm::runtime::McastDeviceMemory* mcast_mem, at::Tensor& output,
    at::Tensor& input, at::Tensor& comm_buffer, at::Tensor& buffer_flags, bool wait_for_results);
void twoShotRMSNorm(torch::Tensor& prenorm_output, torch::Tensor& normed_output, torch::Tensor& input,
    torch::Tensor& gamma, double epsilon, torch::Tensor& residual, torch::Tensor& buffer_flags);
} // namespace tensorrt_llm::kernels
