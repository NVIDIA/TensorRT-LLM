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

#include "tensorrt_llm/kernels/minimaxM31/moeFinalizeSm107.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! NATIVE_FINALIZE: the routed experts' finalize from the trtllm-gen runner's unfinalized parts (fc2 [P, H] bf16 rows,
//! weights [T, K] bf16 / fp32, idx = expanded -> permuted int32 [T, K], -1 = dropped), + optionally the shared expert.
//! Bit for bit the Triton kernels/m3_moe_finalize. cfg -1 = table entry 0.
torch::Tensor minimaxM31MoeFinalize(
    torch::Tensor fc2, torch::Tensor weights, torch::Tensor idx, std::optional<torch::Tensor> shared, int64_t cfg)
{
    TORCH_CHECK(fc2.dim() == 2 && fc2.scalar_type() == at::kBFloat16 && fc2.stride(1) == 1, "fc2: bf16 [P, H] rows");
    TORCH_CHECK(weights.dim() == 2 && weights.is_contiguous() && idx.sizes() == weights.sizes()
            && idx.scalar_type() == at::kInt && idx.is_contiguous(),
        "weights / idx: contiguous [T, K], idx int32");
    TORCH_CHECK(weights.scalar_type() == at::kBFloat16 || weights.scalar_type() == at::kFloat, "weights bf16 / fp32");
    int64_t const tokens = weights.size(0), topk = weights.size(1), hidden = fc2.size(1);
    TORCH_CHECK(topk >= 1 && topk <= 10, "top_k must be in 1..10");
    TORCH_CHECK(cfg >= -1 && cfg <= 7, "cfg must be -1 or 0..7");
    TORCH_CHECK(hidden % 8 == 0 && fc2.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(fc2.data_ptr()) % 16 == 0,
        "fc2: 16-byte aligned rows");
    if (shared.has_value())
    {
        TORCH_CHECK(shared->scalar_type() == at::kBFloat16 && shared->is_contiguous() && shared->size(0) == tokens
                && shared->size(1) == hidden && reinterpret_cast<uintptr_t>(shared->data_ptr()) % 16 == 0,
            "shared: contiguous bf16 [T, H]");
    }
    auto out = torch::empty({tokens, hidden}, fc2.options());
    if (tokens == 0)
    {
        return out;
    }
    at::cuda::CUDAGuard const guard(fc2.device());
    cudaError_t const e
        = tensorrt_llm::kernels::minimaxM31::invokeMoeFinalize(reinterpret_cast<__nv_bfloat16 const*>(fc2.data_ptr()),
            fc2.stride(0), weights.data_ptr(), weights.scalar_type() == at::kFloat, idx.data_ptr<int32_t>(),
            shared.has_value() ? reinterpret_cast<__nv_bfloat16 const*>(shared->data_ptr()) : nullptr,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), tokens, static_cast<int>(topk), static_cast<int>(hidden),
            static_cast<int>(cfg), at::cuda::getCurrentCUDAStream(fc2.get_device()));
    TORCH_CHECK(e == cudaSuccess, "minimax_m31_moe_finalize launch: ", cudaGetErrorString(e));
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m31_moe_finalize(Tensor fc2, Tensor weights, Tensor idx, Tensor? shared=None, int cfg=-1) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_moe_finalize", &tensorrt_llm::torch_ext::minimaxM31MoeFinalize);
}
