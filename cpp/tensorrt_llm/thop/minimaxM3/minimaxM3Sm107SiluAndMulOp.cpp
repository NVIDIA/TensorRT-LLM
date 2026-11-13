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

#include "tensorrt_llm/kernels/minimaxM3Sm107/siluAndMul.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! The drafter MLP's plain SwiGLU (see kernels/minimaxM3Sm107/siluAndMul.h): x bf16 [rows, 2d] -> out bf16 [rows,
//! d].
template <int kTp>
void minimaxM3Sm107SiluAndMul(torch::Tensor const& x, torch::Tensor& out)
{
    constexpr char const* kOpName = kTp == 2 ? "minimax_m3_sm107_tp2_silu_and_mul"
        : kTp == 4                           ? "minimax_m3_sm107_tp4_silu_and_mul"
                                             : "minimax_m3_sm107_tp8_silu_and_mul";
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(), kOpName,
        ": x must be a contiguous 2-D bfloat16 CUDA tensor");
    int64_t const rows = x.size(0);
    int64_t const d = x.size(1) / 2;
    TORCH_CHECK(x.size(1) % 16 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, kOpName,
        ": x rows must be 16-byte aligned with a half width a multiple of 8");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.dim() == 2 && out.is_contiguous()
            && out.size(0) == rows && out.size(1) == d && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
        kOpName, ": out must be a contiguous, 16-byte aligned bfloat16 [rows, d]");
    TORCH_CHECK(rows <= 65535, kOpName, ": at most 65535 rows");
    if (rows == 0)
    {
        return;
    }
    c10::cuda::CUDAGuard const deviceGuard{x.device()};
    auto const err
        = tensorrt_llm::kernels::minimaxM3Sm107::invokeSiluAndMul(reinterpret_cast<uint16_t const*>(x.data_ptr()),
            reinterpret_cast<uint16_t*>(out.data_ptr()), rows, d, at::cuda::getCurrentCUDAStream(x.get_device()));
    TORCH_CHECK(err == cudaSuccess, kOpName, " launch failed: ", cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("minimax_m3_sm107_tp2_silu_and_mul(Tensor x, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_sm107_tp2_silu_and_mul", &tensorrt_llm::torch_ext::minimaxM3Sm107SiluAndMul<2>);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("minimax_m3_sm107_tp4_silu_and_mul(Tensor x, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_sm107_tp4_silu_and_mul", &tensorrt_llm::torch_ext::minimaxM3Sm107SiluAndMul<4>);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("minimax_m3_sm107_tp8_silu_and_mul(Tensor x, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_sm107_tp8_silu_and_mul", &tensorrt_llm::torch_ext::minimaxM3Sm107SiluAndMul<8>);
}
