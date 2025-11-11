/*
 * SPDX-FileCopyrightText: Copyright (out) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/cudaCoreGemmNVFP4.h"
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

void cuda_core_nvfp4_gemm_caller(Tensor& out, Tensor const& a, Tensor const& b, Tensor const& scale_a,
    Tensor const& scale_b, Tensor const& alpha, bool fast_acc = false)
{
    int32_t m = a.sizes()[0];
    int32_t n = b.sizes()[0];
    int32_t k = a.sizes()[1] * 2;
    TORCH_CHECK(a.sizes()[1] == b.sizes()[1]);

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    auto* a_ptr = static_cast<void*>(a.data_ptr());
    auto* b_ptr = static_cast<void*>(b.data_ptr());
    auto* out_ptr = static_cast<void*>(out.data_ptr());
    void* a_scale = static_cast<void*>(scale_a.data_ptr());
    void* b_scale = static_cast<void*>(scale_b.data_ptr());
    void* alpha_ptr = static_cast<void*>(alpha.data_ptr());

    TORCH_CHECK(a_scale != nullptr);
    TORCH_CHECK(b_scale != nullptr);
    TORCH_CHECK(alpha_ptr != nullptr);

    cudaDataType_t aType = convert_torch_dtype(a.scalar_type());
    cudaDataType_t bType = convert_torch_dtype(b.scalar_type());
    cudaDataType_t outType = convert_torch_dtype(out.scalar_type());
    TORCH_CHECK(aType == bType);
    cudaDataType_t scaleAType = convert_torch_dtype(scale_a.scalar_type());
    cudaDataType_t scaleBType = convert_torch_dtype(scale_b.scalar_type());
    TORCH_CHECK(scaleAType == CUDA_R_8U);
    TORCH_CHECK(scaleBType == CUDA_R_8U);
    cudaDataType_t alphaType = convert_torch_dtype(alpha.scalar_type());
    TORCH_CHECK(alphaType == CUDA_R_32F);

    tensorrt_llm::kernels::cuda_core_gemm_nvfp4::Params params(a_ptr, b_ptr, out_ptr, m, n, k,
        reinterpret_cast<__nv_fp8_e4m3 const*>(a_scale), reinterpret_cast<__nv_fp8_e4m3 const*>(b_scale), aType,
        outType, reinterpret_cast<float const*>(alpha_ptr));
    bool dispatched = tensorrt_llm::kernels::cuda_core_gemm_nvfp4::cudaCoreGemmDispatcher(params, stream);
    TORCH_CHECK(dispatched, "Failed to dispatch cudaCoreGemmLauncher");
}

} // namespace

Tensor& cuda_core_nvfp4_gemm_out(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    Tensor const& alpha, std::optional<at::Tensor> const& bias, Tensor& out)
{
    CHECK_TH_CUDA(mat_a);
    CHECK_TH_CUDA(mat_b);
    CHECK_TH_CUDA(scale_a);
    CHECK_TH_CUDA(scale_b);
    CHECK_TH_CUDA(alpha);
    CHECK_TH_CUDA(out);

    CHECK_INPUT(mat_a, FLOAT4_E2M1X2);
    CHECK_INPUT(mat_b, FLOAT4_E2M1X2);

    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2 && out.dim() == 2);
    TORCH_CHECK(mat_a.sizes()[0] == out.sizes()[0]);
    TORCH_CHECK(mat_a.sizes()[1] == mat_b.sizes()[1]);
    TORCH_CHECK(mat_b.sizes()[0] == out.sizes()[1]);

    TORCH_CHECK(!bias.has_value(), "bias is not support yet");

    TORCH_CHECK(scale_a.dtype() == SF_DTYPE);
    TORCH_CHECK(scale_b.dtype() == SF_DTYPE);

    cuda_core_nvfp4_gemm_caller(out, mat_a, mat_b, scale_a, scale_b, alpha, true);
    return out;
}

// mat_a: [M, K / 2], FLOAT4_E2M1X2
// mat_b: [N, K / 2], FLOAT4_E2M1X2
// out: [M, N], fp16/bf16/fp32
// scale_a: ceil(M / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
// scale_b: ceil(N / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
// bias: fp16/bf16/fp32
// out_dtype: fp16/bf16/fp32
Tensor cuda_core_nvfp4_gemm(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    Tensor const& alpha, std::optional<at::Tensor> const& bias, std::optional<c10::ScalarType> out_dtype,
    bool to_userbuffers = false)
{
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());

    std::vector<int64_t> output_size = {mat_a.sizes()[0], mat_b.sizes()[0]};

    Tensor out;
    if (to_userbuffers)
    {
        out = torch_ext::create_userbuffers_tensor(output_size, out_dtype_).first;
    }
    else
    {
        out = at::empty(output_size, mat_a.options().dtype(out_dtype_));
    }

    return cuda_core_nvfp4_gemm_out(mat_a, mat_b, scale_a, scale_b, alpha, bias, out);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "cuda_core_nvfp4_gemm(Tensor mat_a, Tensor mat_b, Tensor scale_a, Tensor scale_b, Tensor alpha, Tensor? bias,"
        " ScalarType? out_dtype, bool to_userbuffers=False) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("cuda_core_nvfp4_gemm", &torch_ext::cuda_core_nvfp4_gemm);
}
