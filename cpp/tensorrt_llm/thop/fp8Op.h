/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

#include <cuda_fp16.h>

#include <cstdint>

namespace torch_ext
{
std::tuple<torch::Tensor, torch::Tensor> symmetric_quantize_weight(torch::Tensor weight);
std::tuple<torch::Tensor, torch::Tensor> symmetric_quantize_activation(torch::Tensor activation);
std::tuple<torch::Tensor, torch::Tensor> symmetric_quantize_per_tensor(torch::Tensor input);
std::tuple<torch::Tensor, torch::Tensor> symmetric_static_quantize_weight(torch::Tensor weight, torch::Tensor scales);
std::tuple<torch::Tensor, torch::Tensor> symmetric_static_quantize_activation(
    torch::Tensor activation, torch::Tensor scales);
std::tuple<torch::Tensor, torch::Tensor> symmetric_static_quantize_per_tensor(
    torch::Tensor input, torch::Tensor scales);

torch::Tensor symmetric_dequantize_weight(torch::Tensor weight, torch::Tensor scales);
torch::Tensor symmetric_dequantize_activation(torch::Tensor activation, torch::Tensor scales);
torch::Tensor symmetric_dequantize_per_tensor(torch::Tensor input, torch::Tensor scales);

} // namespace torch_ext
