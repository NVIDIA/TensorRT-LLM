/*
 * SPDX-FileCopyrightText: Copyright (out) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/userbuffers/ub_interface.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/cudaCoreGemm.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "userbuffersTensor.h"
#include <torch/extension.h>

using torch::Tensor;

namespace torch_ext
{

namespace
{

using tensorrt_llm::common::check;

void cuda_gemm_caller(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
    std::optional<at::Tensor> const& scale_a, std::optional<at::Tensor> const& scale_b, bool fast_acc = false)
{
    bool use_scale = false;
    if (scale_a.has_value() && scale_b.has_value())
    {
        use_scale = true;
    }

    int32_t m = a.sizes()[0];
    int32_t n = b.sizes()[1];
    int32_t k = a.sizes()[1];

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    auto* a_ptr = static_cast<void*>(a.data_ptr());
    auto* b_ptr = static_cast<void*>(b.data_ptr());
    auto* out_ptr = static_cast<void*>(out.data_ptr());
    void* a_scale = nullptr;
    void* b_scale = nullptr;
    if (use_scale)
    {
        a_scale = static_cast<void*>(scale_a.value().data_ptr());
        b_scale = static_cast<void*>(scale_b.value().data_ptr());
    }

    cudaDataType_t aType = convert_torch_dtype(a.scalar_type());
    cudaDataType_t bType = convert_torch_dtype(b.scalar_type());
    cudaDataType_t outType = convert_torch_dtype(out.scalar_type());
    TORCH_CHECK(aType == bType);

    tensorrt_llm::kernels::cuda_core_gemm::Params params(a_ptr, b_ptr, out_ptr, m, n, k,
        reinterpret_cast<float const*>(a_scale), reinterpret_cast<float const*>(b_scale), aType, outType);
    tensorrt_llm::kernels::cuda_core_gemm::cudaCoreGemmDispatcher(params, stream);
}

} // namespace

Tensor& cuda_scaled_mm_out(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    std::optional<at::Tensor> const& bias, Tensor& out)
{
    // Check device
    CHECK_TH_CUDA(mat_a);
    CHECK_TH_CUDA(mat_b);
    CHECK_TH_CUDA(scale_a);
    CHECK_TH_CUDA(scale_b);
    CHECK_TH_CUDA(out);

    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2 && out.dim() == 2);
    TORCH_CHECK(out.sizes()[0] == mat_a.sizes()[0] && mat_a.sizes()[1] == mat_b.sizes()[0]
        && mat_b.sizes()[1] == out.sizes()[1]);

    // Check for strides and alignment
    TORCH_CHECK(mat_a.strides()[1] == 1 && out.strides()[1] == 1); // Row-major
    TORCH_CHECK(mat_b.strides()[0] == 1);                          // Column-major

    TORCH_CHECK(!bias.has_value(), "bias is not support yet");

    TORCH_CHECK(mat_a.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(mat_b.dtype() == torch::kFloat8_e4m3fn);

    cuda_gemm_caller(out, mat_a, mat_b, scale_a, scale_b, true);
    return out;
}

Tensor cuda_scaled_mm(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    std::optional<at::Tensor> const& bias, std::optional<c10::ScalarType> out_dtype, bool to_userbuffers = false)
{
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());

    // std::vector<int64_t> output_size = {mat_a.sizes()[0], mat_b.sizes()[0]};
    std::vector<int64_t> output_size = {mat_a.sizes()[0], mat_b.sizes()[1]};

    Tensor out;
    if (to_userbuffers)
    {
        out = torch_ext::create_userbuffers_tensor(output_size, out_dtype_).first;
    }
    else
    {
        out = at::empty(output_size, mat_a.options().dtype(out_dtype_));
    }

    return cuda_scaled_mm_out(mat_a, mat_b, scale_a, scale_b, bias, out);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "cuda_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scale_a, Tensor scale_b, Tensor? bias,"
        " ScalarType? out_dtype, bool to_userbuffers=False) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("cuda_scaled_mm", &torch_ext::cuda_scaled_mm);
}
