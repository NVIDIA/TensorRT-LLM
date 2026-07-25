/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/cublasScaledMM.h"

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{
// Supported (hd_in, hd_out) shapes must have explicit invokeFusedAGemm instantiations
// in dsv3FusedAGemm.cu.
template <int kHdIn, int kHdOut>
void runFusedAGemm(th::Tensor& out, th::Tensor const& mat_a, th::Tensor const& mat_b, int num_tokens)
{
    auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());
    if (num_tokens <= 8)
    {
        tk::dsv3MinLatencyKernels::invokeFusedAGemm<__nv_bfloat16, kHdIn, kHdOut, 8>(
            reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), num_tokens, stream);
    }
    else
    {
        tk::dsv3MinLatencyKernels::invokeFusedAGemm<__nv_bfloat16, kHdIn, kHdOut, 16>(
            reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), num_tokens, stream);
    }
}
} // namespace

th::Tensor dsv3_fused_a_gemm_op(th::Tensor const& mat_a, th::Tensor const& mat_b, std::optional<at::Tensor> const& bias,
    std::optional<c10::ScalarType> const& out_dtype)
{
    int const num_tokens = mat_a.sizes()[0];
    int const hd_in = mat_a.sizes()[1];
    int const hd_out = mat_b.sizes()[1];
    auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
    auto const data_type = mat_a.scalar_type();
    std::vector<int64_t> output_size = {num_tokens, hd_out};
    th::Tensor out = th::empty(output_size, mat_a.options().dtype(out_dtype_));

    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    TORCH_CHECK(mat_a.strides()[1] == 1 && out.strides()[1] == 1); // Row-major
    TORCH_CHECK(mat_b.strides()[0] == 1);                          // Column-major
    TORCH_CHECK(!bias.has_value(), "bias is not support yet");
    auto const sm = tensorrt_llm::common::getSMVersion();
    bool const dtype_ok = num_tokens >= 1 && num_tokens <= 16 && data_type == torch::kBFloat16
        && out_dtype_ == torch::kBFloat16 && sm >= 90;

    if (dtype_ok && hd_in == 7168 && hd_out == 2112) // DeepSeek-V3 fused q_a/kv_a proj
    {
        runFusedAGemm<7168, 2112>(out, mat_a, mat_b, num_tokens);
    }
    else if (dtype_ok && hd_in == 7168 && hd_out == 3584) // Kimi K3 latent MoE down proj
    {
        runFusedAGemm<7168, 3584>(out, mat_a, mat_b, num_tokens);
    }
    else if (dtype_ok && hd_in == 3584 && hd_out == 7168) // Kimi K3 latent MoE up proj
    {
        runFusedAGemm<3584, 7168>(out, mat_a, mat_b, num_tokens);
    }
    else // fallback to cublas, can be slow
    {
        cublas_mm_out(mat_a, mat_b, bias, out);
    }
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("dsv3_fused_a_gemm_op(Tensor mat_a, Tensor mat_b, Tensor? bias, ScalarType? out_dtype) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("dsv3_fused_a_gemm_op", &tensorrt_llm::torch_ext::dsv3_fused_a_gemm_op);
}
