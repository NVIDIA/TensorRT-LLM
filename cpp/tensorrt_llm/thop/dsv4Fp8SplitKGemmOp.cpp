/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_llm/kernels/mhcKernels/mhcKernels.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace tk = tensorrt_llm::kernels::mhc;

namespace
{

void dsv4Fp8SplitKGemmOp(
    torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, torch::Tensor partials, int64_t numSplits)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && sfa.is_cuda() && sfb.is_cuda() && partials.is_cuda(),
        "dsv4_fp8_splitk_gemm: all tensors must be CUDA tensors");
    TORCH_CHECK(a.device() == b.device() && a.device() == sfa.device() && a.device() == sfb.device()
            && a.device() == partials.device(),
        "dsv4_fp8_splitk_gemm: all tensors must be on the same device");
    TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn && b.scalar_type() == torch::kFloat8_e4m3fn,
        "dsv4_fp8_splitk_gemm: A and B must be Float8_e4m3fn");
    TORCH_CHECK(sfa.scalar_type() == torch::kInt32 && sfb.scalar_type() == torch::kInt32,
        "dsv4_fp8_splitk_gemm: SFA and SFB must be packed int32 UE8M0 scales");
    TORCH_CHECK(partials.scalar_type() == torch::kBFloat16, "dsv4_fp8_splitk_gemm: partials must be bfloat16");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "dsv4_fp8_splitk_gemm: A and B must be rank-2");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "dsv4_fp8_splitk_gemm: A and B must be contiguous");

    int64_t const M = a.size(0);
    int64_t const K = a.size(1);
    int64_t const N = b.size(0);
    TORCH_CHECK(b.size(1) == K, "dsv4_fp8_splitk_gemm: A and B K dimensions must match");
    TORCH_CHECK(numSplits == 2 || numSplits == 4, "dsv4_fp8_splitk_gemm: num_splits must be 2 or 4");
    TORCH_CHECK(partials.dim() == 3 && partials.size(0) == numSplits && partials.size(1) == M && partials.size(2) == N
            && partials.is_contiguous(),
        "dsv4_fp8_splitk_gemm: partials must be contiguous [num_splits, M, N]");

    TORCH_CHECK(K % 512 == 0, "dsv4_fp8_splitk_gemm: K must be divisible by 512");
    int64_t const packedK = K / 512;
    TORCH_CHECK(sfa.dim() == 2 && sfa.size(0) == M && sfa.size(1) == packedK && sfa.stride(0) == 1 && sfa.stride(1) >= M
            && sfa.stride(1) % 4 == 0,
        "dsv4_fp8_splitk_gemm: SFA must be packed MN-major [M, K/512] with 4-aligned K stride");
    TORCH_CHECK(
        sfb.dim() == 2 && sfb.size(0) == N && sfb.size(1) == packedK && sfb.stride(0) == 1 && sfb.stride(1) == N,
        "dsv4_fp8_splitk_gemm: SFB must be packed MN-major [N, K/512]");

    at::cuda::CUDAGuard deviceGuard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    tk::dsv4Fp8SplitKGemmLaunch(a.data_ptr(), sfa.data_ptr<int32_t>(), b.data_ptr(), sfb.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(partials.data_ptr<at::BFloat16>()), static_cast<int>(M), static_cast<int>(N),
        static_cast<int>(K), static_cast<int>(sfa.stride(1)), static_cast<int>(sfb.stride(1)),
        static_cast<int>(numSplits), stream);
}

} // namespace

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "dsv4_fp8_splitk_gemm("
        "Tensor a, Tensor sfa, Tensor b, Tensor sfb, Tensor(a!) partials, int num_splits) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("dsv4_fp8_splitk_gemm", &dsv4Fp8SplitKGemmOp);
}
