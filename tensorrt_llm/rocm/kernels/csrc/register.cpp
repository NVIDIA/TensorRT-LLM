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
#include "ops.h"
#include <torch/library.h>

TORCH_LIBRARY(trtllm_rdna4, module)
{
    module.def("rms_norm(Tensor input, Tensor weight, float epsilon) -> Tensor");
    module.def("add_rms_norm(Tensor input, Tensor residual, Tensor weight, float epsilon) -> (Tensor, Tensor)");
    module.def("layer_norm(Tensor input, Tensor weight, Tensor? bias, float epsilon) -> Tensor");
    module.def("gated_activation(Tensor gate, Tensor up, int activation) -> Tensor");
    module.def("rotary(Tensor input, Tensor cos, Tensor sin, bool interleaved) -> Tensor");
    module.def(
        "attention(Tensor query, Tensor key, Tensor value, Tensor? bias, bool causal, int query_start, float scale) -> "
        "Tensor");
}

// HIP PyTorch uses the CUDA dispatch key for ROCm tensors, by design.
TORCH_LIBRARY_IMPL(trtllm_rdna4, CUDA, module)
{
    module.impl("rms_norm", TORCH_FN(tensorrt_llm::rdna4::rmsNorm));
    module.impl("add_rms_norm", TORCH_FN(tensorrt_llm::rdna4::addRmsNorm));
    module.impl("layer_norm", TORCH_FN(tensorrt_llm::rdna4::layerNorm));
    module.impl("gated_activation", TORCH_FN(tensorrt_llm::rdna4::gatedActivation));
    module.impl("rotary", TORCH_FN(tensorrt_llm::rdna4::rotary));
    module.impl("attention", TORCH_FN(tensorrt_llm::rdna4::attention));
}
