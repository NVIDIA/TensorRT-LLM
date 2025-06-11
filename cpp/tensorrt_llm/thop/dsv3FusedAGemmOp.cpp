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

namespace torch_ext
{
th::Tensor dsv3_fused_a_gemm_op(th::Tensor const& mat_a, th::Tensor const& mat_b, std::optional<at::Tensor> const& bias,
    std::optional<c10::ScalarType> const& out_dtype)
{
    int const num_tokens = mat_a.sizes()[0];
    int const hd_in = mat_a.sizes()[1];
    int const hd_out = mat_b.sizes()[1];
    // auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
    auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
    auto const data_type = mat_a.scalar_type();
    constexpr int kHdIn = 7168;
    constexpr int kHdOut = 2112;
    std::vector<int64_t> output_size = {num_tokens, hd_out};
    th::Tensor out = th::empty(output_size, mat_a.options().dtype(out_dtype_));

    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    TORCH_CHECK(mat_a.strides()[1] == 1 && out.strides()[1] == 1); // Row-major
    TORCH_CHECK(mat_b.strides()[0] == 1);                          // Column-major
    TORCH_CHECK(!bias.has_value(), "bias is not support yet");
    auto const sm = tensorrt_llm::common::getSMVersion();
    if (sm >= 90)
    {
        bool use_custom_kernel = false;
        if (num_tokens >= 1 && num_tokens <= 16 && hd_in == kHdIn && hd_out == kHdOut && data_type == torch::kBFloat16
            && out_dtype_ == torch::kBFloat16)
        {
            use_custom_kernel = true;
        }
        if (use_custom_kernel)
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
        else
        {
            cublas_mm_out(mat_a, mat_b, bias, out);
        }
    }
    else
    {
        cublas_mm_out(mat_a, mat_b, bias, out);
    }
    return out;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("dsv3_fused_a_gemm_op(Tensor mat_a, Tensor mat_b, Tensor? bias, ScalarType? out_dtype) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("dsv3_fused_a_gemm_op", &torch_ext::dsv3_fused_a_gemm_op);
}
