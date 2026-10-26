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

#include "tensorrt_llm/kernels/minimaxM31/castBf16Fp32.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! x.to(torch.float32) for a contiguous bf16 CUDA tensor, bit for bit, through the vectorized widening kernel (the
//! MiniMax-M3.1 router gate input, TP2CTX_GATE_CAST). Same shape, float32.
torch::Tensor minimaxM31Bf16ToFp32(torch::Tensor const& x)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16,
        "minimax_m31_bf16_to_fp32: a contiguous bf16 CUDA tensor");
    at::cuda::CUDAGuard const guard(x.device());
    torch::Tensor out = torch::empty(x.sizes(), x.options().dtype(at::kFloat));
    tensorrt_llm::kernels::minimaxM31::launchCastBf16ToFp32(reinterpret_cast<uint16_t const*>(x.data_ptr()),
        out.data_ptr<float>(), x.numel(), at::cuda::getCurrentCUDAStream(x.get_device()));
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("minimax_m31_bf16_to_fp32(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_bf16_to_fp32", &tensorrt_llm::torch_ext::minimaxM31Bf16ToFp32);
}
