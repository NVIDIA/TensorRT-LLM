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

#include "minimax_m3/csrc/m31/widenRowsSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

//! DEP_LOGITS_WIDEN: x.float() bit for bit for a 2-D bf16 / fp16 CUDA tensor with a unit inner stride (any row stride,
//! e.g. the padded lm_head's vocabulary slice), as a contiguous float32 [rows, cols].
torch::Tensor minimaxM31WidenRows(torch::Tensor const& x, bool pdl)
{
    TORCH_CHECK(
        x.is_cuda() && x.dim() == 2 && x.stride(1) == 1, "minimax_m31_widen_rows: 2-D CUDA rows, unit inner stride");
    TORCH_CHECK(
        x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf, "minimax_m31_widen_rows: bf16 / fp16");
    at::cuda::CUDAGuard const guard(x.device());
    torch::Tensor out = torch::empty({x.size(0), x.size(1)}, x.options().dtype(at::kFloat));
    cudaError_t const e = TRTLLM_MV2_NS::kernels::minimaxM31::invokeWidenRows(
        reinterpret_cast<uint16_t const*>(x.data_ptr()), x.stride(0), out.data_ptr<float>(), x.size(0), x.size(1),
        x.scalar_type() == at::kBFloat16, pdl, at::cuda::getCurrentCUDAStream(x.get_device()));
    TORCH_CHECK(e == cudaSuccess, "minimax_m31_widen_rows launch: ", cudaGetErrorString(e));
    return out;
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def("minimax_m31_widen_rows(Tensor x, bool pdl=True) -> Tensor");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_widen_rows", &TRTLLM_MV2_NS::torch_ext::minimaxM31WidenRows);
}
