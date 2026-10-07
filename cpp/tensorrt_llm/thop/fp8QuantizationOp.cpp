/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "fp8Op.h"

TORCH_LIBRARY_FRAGMENT(tensorrt_llm, m)
{
    m.def("quantize_e4m3_weight(Tensor weight) -> (Tensor, Tensor)");
    m.def("quantize_e4m3_activation(Tensor activation) -> (Tensor, Tensor)");
    m.def("quantize_e4m3_per_tensor(Tensor input) -> (Tensor, Tensor)");
    m.def("static_quantize_e4m3_weight(Tensor weight, Tensor scales) -> (Tensor, Tensor)");
    m.def("static_quantize_e4m3_activation(Tensor activation, Tensor scales) -> (Tensor, Tensor)");
    m.def("static_quantize_e4m3_per_tensor(Tensor input, Tensor scales) -> (Tensor, Tensor)");
    m.def("dequantize_e4m3_weight(Tensor weight, Tensor scales) -> Tensor");
    m.def("dequantize_e4m3_activation(Tensor activation, Tensor scales) -> Tensor");
    m.def("dequantize_e4m3_per_tensor(Tensor input, Tensor scales) -> Tensor");
    m.def("vectorized_per_token_fp8_quant(Tensor input) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(tensorrt_llm, CUDA, m)
{
    m.impl("quantize_e4m3_weight", &tensorrt_llm::torch_ext::symmetric_quantize_weight);
    m.impl("quantize_e4m3_activation", &tensorrt_llm::torch_ext::symmetric_quantize_activation);
    m.impl("quantize_e4m3_per_tensor", &tensorrt_llm::torch_ext::symmetric_quantize_per_tensor);
    m.impl("static_quantize_e4m3_weight", &tensorrt_llm::torch_ext::symmetric_static_quantize_weight);
    m.impl("static_quantize_e4m3_activation", &tensorrt_llm::torch_ext::symmetric_static_quantize_activation);
    m.impl("static_quantize_e4m3_per_tensor", &tensorrt_llm::torch_ext::symmetric_static_quantize_per_tensor);
    m.impl("dequantize_e4m3_weight", &tensorrt_llm::torch_ext::symmetric_dequantize_weight);
    m.impl("dequantize_e4m3_activation", &tensorrt_llm::torch_ext::symmetric_dequantize_activation);
    m.impl("dequantize_e4m3_per_tensor", &tensorrt_llm::torch_ext::symmetric_dequantize_per_tensor);
    m.impl("vectorized_per_token_fp8_quant", &tensorrt_llm::torch_ext::vectorized_per_token_fp8_quant);
}

static auto dequantize_mxe4m3_host = torch::RegisterOperators(
    "tensorrt_llm::dequantize_mxe4m3_host", &tensorrt_llm::torch_ext::dequantize_mxe4m3_host);

static auto quantize_mxe4m3_host
    = torch::RegisterOperators("tensorrt_llm::quantize_mxe4m3_host", &tensorrt_llm::torch_ext::quantize_mxe4m3_host);
