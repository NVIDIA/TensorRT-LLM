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

#include <cstdint>
#include <optional>
#include <tuple>

namespace torch_ext
{

// (xq, sf) = NVFP4-quantize(x * pqs). x [m,n] bf16, pqs [n] bf16 (per-input-channel smoothing),
// global_scale f32[1]. xq [m,n/2] uint8 (packed e2m1), sf swizzled UE4M3 block scales.
std::tuple<at::Tensor, at::Tensor> nvfp4_quantize_smooth(
    at::Tensor const& x, at::Tensor const& pqs, at::Tensor const& global_scale)
{
    TORCH_CHECK(x.is_cuda() && pqs.is_cuda() && global_scale.is_cuda(), "x, pqs, global_scale must be CUDA tensors");
    TORCH_CHECK(pqs.device() == x.device() && global_scale.device() == x.device(),
        "x, pqs, global_scale must reside on the same CUDA device");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && pqs.scalar_type() == at::kBFloat16, "x, pqs must be bf16");
    TORCH_CHECK(global_scale.scalar_type() == at::kFloat, "global_scale must be float32");
    TORCH_CHECK(global_scale.numel() >= 1, "global_scale must contain at least one element");
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
    std::optional<c10::ScalarType> out_dtype, std::optional<at::Tensor> const& bias = std::nullopt, int64_t tactic = -1)
{
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(wq.is_cuda() && a_sf.is_cuda() && w_sf.is_cuda() && alpha.is_cuda(),
        "wq, a_sf, w_sf, alpha must be CUDA tensors");
    TORCH_CHECK(wq.device() == a.device() && a_sf.device() == a.device() && w_sf.device() == a.device()
            && alpha.device() == a.device(),
        "a, wq, a_sf, w_sf, alpha must reside on the same CUDA device");
    TORCH_CHECK(a.scalar_type() == at::kByte && wq.scalar_type() == at::kByte, "a, wq must be uint8 (packed FP4)");
    TORCH_CHECK(a_sf.scalar_type() == at::kByte && w_sf.scalar_type() == at::kByte, "a_sf, w_sf must be uint8");
    TORCH_CHECK(alpha.scalar_type() == at::kFloat, "alpha must be float32");
    TORCH_CHECK(alpha.numel() >= 1, "alpha must contain at least one element");
    TORCH_CHECK(D.is_cuda(), "D must be a CUDA tensor");
    TORCH_CHECK(D.scalar_type() == at::kBFloat16, "D must be bf16");
    TORCH_CHECK(D.dim() == 2, "D must be 2-D");
    TORCH_CHECK(L1.is_cuda(), "L1 must be a CUDA tensor");
    TORCH_CHECK(L1.scalar_type() == at::kBFloat16, "L1 must be bf16");
    TORCH_CHECK(L1.dim() == 2, "L1 must be 2-D");
    TORCH_CHECK(a.dim() == 2 && wq.dim() == 2, "a, wq must be 2-D");
    int const m = static_cast<int>(a.size(0));
    int const kPacked = static_cast<int>(a.size(1));
    int const k = kPacked * 2;
    int const n = static_cast<int>(wq.size(0));
    TORCH_CHECK(static_cast<int>(wq.size(1)) == kPacked, "a, wq inner dim mismatch");
    TORCH_CHECK(n % 32 == 0 && k % 32 == 0, "n and k must be divisible by 32");
    int64_t const expectedActSfSize = tensorrt_llm::computeSwizzledLayoutSFSize(m, k / 16);
    int64_t const expectedWeightSfSize = tensorrt_llm::computeSwizzledLayoutSFSize(n, k / 16);
    TORCH_CHECK(a_sf.numel() >= expectedActSfSize, "a_sf is smaller than the required swizzled scale layout");
    TORCH_CHECK(w_sf.numel() >= expectedWeightSfSize, "w_sf is smaller than the required swizzled scale layout");
    TORCH_CHECK(D.device() == a.device(), "D must reside on the same CUDA device as a");
    TORCH_CHECK(D.size(0) == m && D.size(1) == 32, "D must have shape [m, 32]");
    TORCH_CHECK(D.is_contiguous(), "D must be contiguous");
    auto const downAddress = reinterpret_cast<std::uintptr_t>(D.const_data_ptr<at::BFloat16>());
    TORCH_CHECK(downAddress % 16 == 0, "D must be 16-byte aligned for TMA");
    TORCH_CHECK(L1.device() == a.device(), "L1 must reside on the same CUDA device as a");
    TORCH_CHECK(L1.size(0) == n && L1.size(1) == 32, "L1 must have shape [n, 32]");
    auto const outType = out_dtype.value_or(at::kBFloat16);
    TORCH_CHECK(outType == at::kBFloat16, "nvfp4_svdquant_gemm currently supports bf16 output only");
    auto a_ = a.contiguous();
    auto wq_ = wq.contiguous();
    auto aSf_ = a_sf.contiguous();
    auto wSf_ = w_sf.contiguous();
    auto alpha_ = alpha.contiguous();
    auto L1_ = L1.contiguous();
    at::Tensor out = at::empty({m, n}, a.options().dtype(outType));
    void const* biasPtr = nullptr;
    if (bias.has_value() && bias->defined())
    {
        auto const& biasTensor = *bias;
        TORCH_CHECK(biasTensor.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(biasTensor.device() == out.device(), "bias must reside on the same CUDA device as the output");
        TORCH_CHECK(biasTensor.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(biasTensor.dim() == 1 && biasTensor.size(0) == n, "bias must have shape [n]");
        TORCH_CHECK(biasTensor.scalar_type() == outType, "bias dtype must match the output dtype");
        biasPtr = biasTensor.const_data_ptr();
    }
    TORCH_CHECK(tactic >= -1 && tactic < tensorrt_llm::kernels::cutlass_kernels::kNvfp4SvdquantGemmNumTactics,
        "invalid NVFP4 SVDQuant tactic: ", tactic);
    int const tacticId = tactic < 0 ? 0 : static_cast<int>(tactic);
    size_t const wsBytes
        = tensorrt_llm::kernels::cutlass_kernels::nvfp4_svdquant_gemm_workspace_size(m, n, k, tacticId);
    at::Tensor workspace = at::empty({static_cast<int64_t>(wsBytes > 0 ? wsBytes : 1)}, a.options().dtype(at::kByte));
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
    tensorrt_llm::kernels::cutlass_kernels::nvfp4_svdquant_gemm_run(out.data_ptr(), a_.const_data_ptr(),
        wq_.const_data_ptr(), aSf_.const_data_ptr(), wSf_.const_data_ptr(), alpha_.const_data_ptr<float>(),
        D.const_data_ptr(), L1_.const_data_ptr(), biasPtr, m, n, k, reinterpret_cast<char*>(workspace.data_ptr()),
        wsBytes, stream, tacticId);
    return out;
}

class NVFP4SVDQuantGemmRunner : public torch::CustomClassHolder
{
public:
    at::Tensor runGemm(at::Tensor const& a, at::Tensor const& wq, at::Tensor const& a_sf, at::Tensor const& w_sf,
        at::Tensor const& alpha, at::Tensor const& D, at::Tensor const& L1, std::optional<c10::ScalarType> out_dtype,
        std::optional<at::Tensor> const& bias, int64_t tactic) const
    {
        return nvfp4_svdquant_gemm(a, wq, a_sf, w_sf, alpha, D, L1, out_dtype, bias, tactic);
    }

    int64_t getNumConfigs() const
    {
        return tensorrt_llm::kernels::cutlass_kernels::kNvfp4SvdquantGemmNumTactics;
    }
};

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<torch_ext::NVFP4SVDQuantGemmRunner>("NVFP4SVDQuantGemmRunner")
        .def(torch::init<>())
        .def("run_gemm", &torch_ext::NVFP4SVDQuantGemmRunner::runGemm)
        .def("get_num_configs", &torch_ext::NVFP4SVDQuantGemmRunner::getNumConfigs);

    m.def("nvfp4_quantize_smooth(Tensor x, Tensor pqs, Tensor global_scale) -> (Tensor, Tensor)");
    m.def(
        "nvfp4_svdquant_gemm(Tensor a, Tensor wq, Tensor a_sf, Tensor w_sf, Tensor alpha, "
        "Tensor D, Tensor L1, ScalarType? out_dtype, Tensor? bias=None, int tactic=-1) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("nvfp4_quantize_smooth", &torch_ext::nvfp4_quantize_smooth);
    m.impl("nvfp4_svdquant_gemm", &torch_ext::nvfp4_svdquant_gemm);
}
