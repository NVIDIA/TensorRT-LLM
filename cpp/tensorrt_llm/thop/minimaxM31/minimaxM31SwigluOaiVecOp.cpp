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

#include "tensorrt_llm/kernels/minimaxM31/swigluOaiVecSm107.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! NATIVE_SWIGLU: x = [gate | up] contiguous bf16 [rows, 2d] -> bf16 [rows, d], bit for bit the
//! Triton silu_and_mul at limit 7 / alpha 1.702 / beta 1. nv: 16-byte output vectors per thread (1 / 2 / 4).
torch::Tensor minimaxM31SwigluOaiVec(torch::Tensor x, bool pdl, int64_t nv)
{
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.size(1) % 2 == 0,
        "x: contiguous bf16 [rows, 2d]");
    int64_t const rows = x.size(0), d = x.size(1) / 2;
    TORCH_CHECK(d % 2 == 0, "d must be even (packed pairs)");
    TORCH_CHECK(nv == 1 || nv == 2 || nv == 4, "nv must be 1, 2 or 4");
    auto out = torch::empty({rows, d}, x.options());
    if (rows == 0)
    {
        return out;
    }
    at::cuda::CUDAGuard const guard(x.device());
    cudaError_t const e = tensorrt_llm::kernels::minimaxM31::invokeSwigluOaiVec(x.data_ptr(), out.data_ptr(), rows, d,
        static_cast<int>(nv), pdl, at::cuda::getCurrentCUDAStream(x.get_device()));
    TORCH_CHECK(e == cudaSuccess, "minimax_m31_swiglu_oai_vec launch: ", cudaGetErrorString(e));
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("minimax_m31_swiglu_oai_vec(Tensor x, bool pdl=True, int nv=1) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_swiglu_oai_vec", &tensorrt_llm::torch_ext::minimaxM31SwigluOaiVec);
}
