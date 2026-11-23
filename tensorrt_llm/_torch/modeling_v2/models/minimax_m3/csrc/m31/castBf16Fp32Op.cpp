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

#include "minimax_m3/csrc/m31/castBf16Fp32.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_MV2_NAMESPACE_BEGIN

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
    TRTLLM_MV2_NS::kernels::minimaxM31::launchCastBf16ToFp32(reinterpret_cast<uint16_t const*>(x.data_ptr()),
        out.data_ptr<float>(), x.numel(), at::cuda::getCurrentCUDAStream(x.get_device()));
    return out;
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def("minimax_m31_bf16_to_fp32(Tensor x) -> Tensor");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_bf16_to_fp32", &TRTLLM_MV2_NS::torch_ext::minimaxM31Bf16ToFp32);
}
