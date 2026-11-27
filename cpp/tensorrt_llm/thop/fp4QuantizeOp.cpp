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

#include "fp4Quantize.h"

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fp4_quantize(Tensor input, Tensor? globalScale, int sfVecSize, bool sfUseUE8M0=False, bool "
        "isSfSwizzledLayout=True) -> (Tensor, Tensor)");
    m.def(
        "fp4_quantize.sm_budget(Tensor input, Tensor? globalScale, int sfVecSize, bool sfUseUE8M0=False, bool "
        "isSfSwizzledLayout=True, int reserved_sms=0) -> (Tensor, Tensor)");
    m.def("calculate_nvfp4_global_scale(Tensor input, Tensor? tokensPerBatch) -> Tensor");
    m.def(
        "fp4_quantize_with_reorder_residual(Tensor X, Tensor input_scale, Tensor reorder_index, int KE, bool is_act) "
        "-> (Tensor, Tensor)");
    m.def("fp4_quantize_with_residual(Tensor X, Tensor input_scale, int KE, bool is_act) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp4_quantize", TORCH_FN(tensorrt_llm::torch_ext::fp4_quantize));
    m.impl("fp4_quantize.sm_budget", TORCH_FN(tensorrt_llm::torch_ext::fp4_quantize_sm_budget));
    m.impl("calculate_nvfp4_global_scale", TORCH_FN(tensorrt_llm::torch_ext::calculate_nvfp4_global_scale));
    m.impl("fp4_quantize_with_reorder_residual", TORCH_FN(tensorrt_llm::torch_ext::fp4_quantize_with_reorder_residual));
    m.impl("fp4_quantize_with_residual", TORCH_FN(tensorrt_llm::torch_ext::fp4_quantize_with_residual));
}
