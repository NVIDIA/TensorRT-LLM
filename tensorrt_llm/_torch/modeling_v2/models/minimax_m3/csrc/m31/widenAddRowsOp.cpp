/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "minimax_m3/csrc/m31/widenAddRowsSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

//! DEP_MARKOV_GLUE's widening add: base + bias.to(float32) bit for bit, as a contiguous float32 [rows, cols]. base:
//! float32 2-D CUDA rows with a unit inner stride (any row stride, e.g. gen_logits[:, step]); bias: contiguous bf16 /
//! float32 [rows, cols] on its device.
torch::Tensor minimaxM31WidenAddRows(torch::Tensor const& base, torch::Tensor const& bias, bool pdl)
{
    TORCH_CHECK(base.is_cuda() && base.dim() == 2 && base.stride(1) == 1 && base.scalar_type() == at::kFloat,
        "minimax_m31_widen_add_rows: base must be float32 2-D CUDA rows with a unit inner stride");
    TORCH_CHECK(bias.scalar_type() == at::kBFloat16 || bias.scalar_type() == at::kFloat,
        "minimax_m31_widen_add_rows: bias must be bf16 / float32");
    TORCH_CHECK(bias.is_contiguous() && bias.dim() == 2 && bias.size(0) == base.size(0) && bias.size(1) == base.size(1)
            && bias.device() == base.device(),
        "minimax_m31_widen_add_rows: bias must be a contiguous [rows, cols] on the base's device");
    at::cuda::CUDAGuard const guard(base.device());
    torch::Tensor out = torch::empty({base.size(0), base.size(1)}, base.options());
    cudaError_t const e = TRTLLM_MV2_NS::kernels::minimaxM31::invokeWidenAddRows(base.data_ptr<float>(), base.stride(0),
        bias.data_ptr(), bias.scalar_type() == at::kBFloat16, out.data_ptr<float>(), base.size(0), base.size(1), pdl,
        at::cuda::getCurrentCUDAStream(base.get_device()));
    TORCH_CHECK(e == cudaSuccess, "minimax_m31_widen_add_rows launch: ", cudaGetErrorString(e));
    return out;
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def("minimax_m31_widen_add_rows(Tensor base, Tensor bias, bool pdl=True) -> Tensor");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_widen_add_rows", &TRTLLM_MV2_NS::torch_ext::minimaxM31WidenAddRows);
}
