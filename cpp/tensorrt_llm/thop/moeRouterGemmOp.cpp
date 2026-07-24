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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/moeRouterGemm.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <cuda_bf16.h>

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// MoE router GEMM: logits[M, N] = act[M, K] @ weight[N, K]^T.
//
// mat_a is the bf16 or fp16 activation [num_tokens, hidden]. mat_b is the raw
// fp32 router weight [num_experts, hidden], passed without transpose. See
// moeRouterGemm.h for the precision contract.
th::Tensor moe_router_gemm_op(
    th::Tensor const& mat_a, th::Tensor const& mat_b, std::optional<c10::ScalarType> const& out_dtype)
{
    TORCH_CHECK(mat_a.dim() == 2, "moe_router_gemm: mat_a must be 2D [num_tokens, hidden]");
    TORCH_CHECK(mat_b.dim() == 2, "moe_router_gemm: mat_b must be 2D [num_experts, hidden]");
    TORCH_CHECK(mat_a.sizes()[1] == mat_b.sizes()[1], "moe_router_gemm: hidden dim mismatch between mat_a and mat_b");
    TORCH_CHECK(mat_b.scalar_type() == torch::kFloat32, "moe_router_gemm: mat_b (weight) must be float32");

    auto const out_dtype_ = out_dtype.value_or(torch::kFloat32);
    TORCH_CHECK(out_dtype_ == torch::kFloat32, "moe_router_gemm: only float32 output is supported");

    int const num_tokens = mat_a.sizes()[0];
    int const hidden_dim = mat_a.sizes()[1];
    int const num_experts = mat_b.sizes()[0];

    // Row-major, contiguous last dim.
    TORCH_CHECK(mat_a.stride(1) == 1 && mat_b.stride(1) == 1, "moe_router_gemm: inputs must be row-major");

    th::Tensor out = th::empty({num_tokens, num_experts}, mat_a.options().dtype(out_dtype_));
    if (num_tokens == 0)
    {
        return out;
    }

    auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());
    auto const data_type = mat_a.scalar_type();
    switch (data_type)
    {
    case torch::kBFloat16:
        tk::invokeMoeRouterGemm<__nv_bfloat16>(reinterpret_cast<float*>(out.mutable_data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()), reinterpret_cast<float const*>(mat_b.data_ptr()),
            num_tokens, num_experts, hidden_dim, stream);
        break;
    case torch::kHalf:
        tk::invokeMoeRouterGemm<half>(reinterpret_cast<float*>(out.mutable_data_ptr()),
            reinterpret_cast<half const*>(mat_a.data_ptr()), reinterpret_cast<float const*>(mat_b.data_ptr()),
            num_tokens, num_experts, hidden_dim, stream);
        break;
    default: TORCH_CHECK(false, "moe_router_gemm: mat_a must be bfloat16 or float16");
    }

    return out;
}

} // end namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("moe_router_gemm_op(Tensor mat_a, Tensor mat_b, ScalarType? out_dtype) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_router_gemm_op", &tensorrt_llm::torch_ext::moe_router_gemm_op);
}
