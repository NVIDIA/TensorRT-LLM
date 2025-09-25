/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cublasFp4ScaledMM.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <unordered_map>

using torch::Tensor;

namespace torch_ext
{

namespace
{

using tensorrt_llm::common::check;
using tensorrt_llm::common::CublasMMWrapper;

void cublas_fp4_gemm_caller(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& scale_a, torch::Tensor const& scale_b, torch::Tensor const& alpha, torch::Tensor const& beta)
{
    int32_t m = a.sizes()[0];
    int32_t n = b.sizes()[0];
    int32_t k_compressed = a.sizes()[1];
    int32_t k = k_compressed * 2;

    // Use device-aware thread-local CublasMMWrapper for FP4 GEMM
    at::cuda::CUDAGuard deviceGuard(a.device());

    thread_local std::unordered_map<int, std::shared_ptr<CublasMMWrapper>> cublasWrappers;
    auto& cublasWrapper = cublasWrappers[a.get_device()];
    if (!cublasWrapper)
    {
        auto cublasHandle = getCublasHandle();
        auto cublasLtHandle = getCublasLtHandle();
        cublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);
    }

    // Set FP4 configuration
    cublasWrapper->setFP4GemmConfig(CUDA_R_16BF); // Output as BF16

    // Get workspace
    auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
    auto workspace = torch::empty(CUBLAS_WORKSPACE_SIZE, workspace_options);

    // Get stream
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    // Get data pointers
    auto* a_ptr = static_cast<void*>(a.data_ptr());
    auto* b_ptr = static_cast<void*>(b.data_ptr());
    auto* out_ptr = static_cast<void*>(out.data_ptr());
    auto* ws_ptr = static_cast<void*>(workspace.data_ptr());

    // Convert scaling factors to __nv_fp8_e4m3 format for cuBLASLt
    void const* a_sf_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(scale_a.data_ptr());
    void const* b_sf_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(scale_b.data_ptr());

    // Validate pointers
    TLLM_CHECK_WITH_INFO(a_sf_ptr != nullptr, "a_sf_ptr is null");
    TLLM_CHECK_WITH_INFO(b_sf_ptr != nullptr, "b_sf_ptr is null");

    // Validate alpha and beta tensors before accessing data
    TLLM_CHECK_WITH_INFO(alpha.numel() > 0, "Alpha tensor is empty");
    TLLM_CHECK_WITH_INFO(beta.numel() > 0, "Beta tensor is empty");
    TLLM_CHECK_WITH_INFO(alpha.dtype() == torch::kFloat32, "Alpha tensor must be float32");
    TLLM_CHECK_WITH_INFO(beta.dtype() == torch::kFloat32, "Beta tensor must be float32");

    auto* alpha_ptr = alpha.data_ptr<float>();
    auto* beta_ptr = beta.data_ptr<float>();

    TLLM_CHECK_WITH_INFO(alpha_ptr != nullptr, "alpha_ptr is null");
    TLLM_CHECK_WITH_INFO(beta_ptr != nullptr, "beta_ptr is null");

    // Set workspace and stream
    cublasWrapper->setStream(stream);
    cublasWrapper->setWorkspace(ws_ptr);

    // Perform FP4 GEMM using CublasMMWrapper
    // Note: A is column major, B is row major, so we swap A and B
    cublasWrapper->Fp4Gemm(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, b_ptr, k, // B matrix (swapped)
        a_ptr, k,                                                       // A matrix (swapped)
        out_ptr, n,                                                     // Output matrix
        b_sf_ptr, a_sf_ptr, alpha_ptr, beta_ptr);
}

} // namespace

Tensor& cublas_fp4_scaled_mm_out(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    Tensor const& alpha, Tensor const& beta, Tensor& out)
{
    // Check device
    CHECK_TH_CUDA(mat_a);
    CHECK_TH_CUDA(mat_b);
    CHECK_TH_CUDA(scale_a);
    CHECK_TH_CUDA(scale_b);
    CHECK_TH_CUDA(alpha);
    CHECK_TH_CUDA(beta);
    CHECK_TH_CUDA(out);

    // Ensure all tensors are on the same device
    auto const deviceIndex = mat_a.get_device();
    TORCH_CHECK(mat_b.get_device() == deviceIndex, "mat_b must be colocated with mat_a");
    TORCH_CHECK(scale_a.get_device() == deviceIndex, "scale_a must be colocated with mat_a");
    TORCH_CHECK(scale_b.get_device() == deviceIndex, "scale_b must be colocated with mat_a");
    TORCH_CHECK(alpha.get_device() == deviceIndex, "alpha must be colocated with mat_a");
    TORCH_CHECK(beta.get_device() == deviceIndex, "beta must be colocated with mat_a");
    TORCH_CHECK(out.get_device() == deviceIndex, "out must be colocated with mat_a");

    // Check dimensions
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2 && out.dim() == 2);
    TORCH_CHECK(out.sizes()[0] == mat_a.sizes()[0] && // m
        mat_a.sizes()[1] == mat_b.sizes()[1] &&       // k
        mat_b.sizes()[0] == out.sizes()[1]);          // n

    // Check scaling factors
    TORCH_CHECK(alpha.numel() == 1);
    TORCH_CHECK(beta.numel() == 1);

    // Check data types - FP4 is typically represented as uint8 in PyTorch
    TORCH_CHECK(mat_a.dtype() == torch::kUInt8);
    TORCH_CHECK(mat_b.dtype() == torch::kUInt8);
    TORCH_CHECK(scale_a.dtype() == torch::kUInt8);
    TORCH_CHECK(scale_b.dtype() == torch::kUInt8);
    TORCH_CHECK(alpha.dtype() == torch::kFloat32);
    TORCH_CHECK(beta.dtype() == torch::kFloat32);

    cublas_fp4_gemm_caller(out, mat_a, mat_b, scale_a, scale_b, alpha, beta);
    return out;
}

Tensor cublas_fp4_scaled_mm(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    Tensor const& alpha, Tensor const& beta, std::optional<c10::ScalarType> out_dtype)
{
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    TORCH_CHECK(mat_a.sizes()[1] == mat_b.sizes()[1]);            // mat_a is [m, k], mat_b is [n, k]

    auto const out_dtype_ = out_dtype.value_or(torch::kBFloat16); // Default to BF16
    std::vector<int64_t> output_size = {mat_a.sizes()[0], mat_b.sizes()[0]};

    Tensor out = at::empty(output_size, mat_a.options().dtype(out_dtype_));

    return cublas_fp4_scaled_mm_out(mat_a, mat_b, scale_a, scale_b, alpha, beta, out);
}

torch::Tensor cublas_fp4_scaled_mm_meta(torch::Tensor const& mat_a, torch::Tensor const& mat_b,
    torch::Tensor const& scale_a, torch::Tensor const& scale_b, torch::Tensor const& alpha, torch::Tensor const& beta,
    c10::optional<torch::ScalarType> out_dtype)
{
    auto const out_dtype_ = out_dtype.value_or(torch::kBFloat16);

    // Simplified and more stable shape inference
    // Avoid complex checks that might trigger recompilation
    auto m = mat_a.size(0);
    auto n = mat_b.size(0);

    // Output shape: [M, N]
    std::vector<int64_t> output_size = {m, n};

    // Use the most stable tensor creation method
    // Copy all properties from input tensor to ensure consistency
    return torch::empty(output_size, mat_a.options().dtype(out_dtype_));
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "cublas_fp4_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scale_a, Tensor scale_b,"
        " Tensor alpha, Tensor beta, ScalarType? out_dtype=None) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("cublas_fp4_scaled_mm", &torch_ext::cublas_fp4_scaled_mm);
}
