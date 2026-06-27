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

// Torch ops for Qwen-Image NVFP4 SVDQuant (W ~= R + L1.L2, per-input-channel pre_quant_scale):
//   nvfp4_quantize_smooth : (xq, sf) = NVFP4-quantize(x * pre_quant_scale) in one pass (the
//                           smoothing is fused into the quantize -- byte-identical to a separate
//                           x_hat = x*s elementwise pass followed by fp4_quantize).
//   nvfp4_svdquant_gemm   : out = alpha * (A @ Bᵀ) + (D @ L1ᵀ), the residual NVFP4 GEMM fused with
//                           the rank-r LoRA-up via a 2nd tcgen05 MMA in the same accumulator (SM100).

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/include/nvfp4_svdquant_gemm.h"
#include "tensorrt_llm/kernels/nvfp4SmoothQuantize.h"
#include "tensorrt_llm/kernels/quantization.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <optional>
#include <tuple>

namespace torch_ext
{

// (xq, sf) = NVFP4-quantize(x * pqs). x [m,n] bf16, pqs [n] bf16 (per-input-channel smoothing),
// global_scale f32[1]. xq [m,n/2] uint8 (packed e2m1), sf swizzled UE4M3 block scales.
std::tuple<at::Tensor, at::Tensor> nvfp4_quantize_smooth(
    at::Tensor const& x, at::Tensor const& pqs, at::Tensor const& global_scale)
{
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && pqs.scalar_type() == at::kBFloat16, "x, pqs must be bf16");
    TORCH_CHECK(global_scale.scalar_type() == at::kFloat, "global_scale must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be [m, n]");
    int const m = static_cast<int>(x.size(0));
    int const n = static_cast<int>(x.size(1));
    TORCH_CHECK(n % 16 == 0, "n must be divisible by 16 (NVFP4 SF vector size)");
    TORCH_CHECK(static_cast<int>(pqs.numel()) == n, "pqs must have n elements");
    auto x_ = x.contiguous();
    auto pqs_ = pqs.contiguous();
    auto gs_ = global_scale.contiguous();
    at::Tensor xq = at::empty({m, n / 2}, x.options().dtype(at::kByte));
    int64_t const sfSize = tensorrt_llm::computeSwizzledLayoutSFSize(m, n / 16);
    at::Tensor sf = at::empty({sfSize}, x.options().dtype(at::kByte));
    int const smCount = tensorrt_llm::common::getMultiProcessorCount();
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
    tensorrt_llm::kernels::nvfp4_smooth_quantize(xq.data_ptr(), sf.data_ptr(), x_.const_data_ptr(),
        pqs_.const_data_ptr(), gs_.const_data_ptr<float>(), m, n, smCount, stream);
    return {xq, sf};
}

// out = alpha * (A @ Bᵀ) + (D @ L1ᵀ). A=quant(x_hat) [m,k/2] uint8, B=R [n,k/2] uint8 (packed e2m1),
// a_sf/w_sf swizzled UE4M3 block scales, alpha f32[1] (residual dequant scale). D [m,r]=x_hat@L2ᵀ
// and L1 [n,r]=svdquant_lora_b/alpha (1/alpha folded so the epilogue out=alpha*acc yields the LoRA).
at::Tensor nvfp4_svdquant_gemm(at::Tensor const& a, at::Tensor const& wq, at::Tensor const& a_sf,
    at::Tensor const& w_sf, at::Tensor const& alpha, at::Tensor const& D, at::Tensor const& L1,
    std::optional<c10::ScalarType> out_dtype)
{
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == at::kByte && wq.scalar_type() == at::kByte, "a, wq must be uint8 (packed FP4)");
    TORCH_CHECK(a_sf.scalar_type() == at::kByte && w_sf.scalar_type() == at::kByte, "a_sf, w_sf must be uint8");
    TORCH_CHECK(alpha.scalar_type() == at::kFloat, "alpha must be float32");
    TORCH_CHECK(D.scalar_type() == at::kBFloat16 && L1.scalar_type() == at::kBFloat16, "D, L1 must be bf16");
    TORCH_CHECK(a.dim() == 2 && wq.dim() == 2, "a, wq must be 2-D");
    int const m = static_cast<int>(a.size(0));
    int const kPacked = static_cast<int>(a.size(1));
    int const k = kPacked * 2;
    int const n = static_cast<int>(wq.size(0));
    TORCH_CHECK(static_cast<int>(wq.size(1)) == kPacked, "a, wq inner dim mismatch");
    TORCH_CHECK(n % 32 == 0 && k % 32 == 0, "n and k must be divisible by 32");
    auto const outType = out_dtype.value_or(at::kBFloat16);
    TORCH_CHECK(outType == at::kBFloat16, "nvfp4_svdquant_gemm currently supports bf16 output only");
    auto a_ = a.contiguous();
    auto wq_ = wq.contiguous();
    auto aSf_ = a_sf.contiguous();
    auto wSf_ = w_sf.contiguous();
    auto alpha_ = alpha.contiguous();
    auto D_ = D.contiguous();
    auto L1_ = L1.contiguous();
    at::Tensor out = at::empty({m, n}, a.options().dtype(outType));
    size_t const wsBytes = tensorrt_llm::kernels::cutlass_kernels::nvfp4_svdquant_gemm_workspace_size(m, n, k);
    at::Tensor workspace = at::empty({static_cast<int64_t>(wsBytes > 0 ? wsBytes : 1)}, a.options().dtype(at::kByte));
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
    tensorrt_llm::kernels::cutlass_kernels::nvfp4_svdquant_gemm_run(out.data_ptr(), a_.const_data_ptr(),
        wq_.const_data_ptr(), aSf_.const_data_ptr(), wSf_.const_data_ptr(), alpha_.const_data_ptr<float>(),
        D_.const_data_ptr(), L1_.const_data_ptr(), m, n, k, reinterpret_cast<char*>(workspace.data_ptr()), wsBytes,
        stream);
    return out;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("nvfp4_quantize_smooth(Tensor x, Tensor pqs, Tensor global_scale) -> (Tensor, Tensor)");
    m.def(
        "nvfp4_svdquant_gemm(Tensor a, Tensor wq, Tensor a_sf, Tensor w_sf, Tensor alpha, "
        "Tensor D, Tensor L1, ScalarType? out_dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("nvfp4_quantize_smooth", &torch_ext::nvfp4_quantize_smooth);
    m.impl("nvfp4_svdquant_gemm", &torch_ext::nvfp4_svdquant_gemm);
}
