/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "cutlass_extensions/gemm_configs.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/include/fp4_gemm.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

#include <cstdint>
#include <cuda_fp16.h>
#include <vector>

namespace tkc = tensorrt_llm::cutlass_extensions;
using tensorrt_llm::kernels::cutlass_kernels::CutlassFp4GemmRunner;
using tensorrt_llm::kernels::cutlass_kernels::FP4GemmType;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

tkc::CutlassGemmConfig getDefaultMxfp8GemmConfig()
{
    // Reuse the same default tile/cluster as MXFP8xMXFP4 -- the B operand is
    // 2x wider in MXFP8xMXFP8, but the same 4x4 cluster/256x256 tile shape is
    // a reasonable starting point on B200.
    return tkc::CutlassGemmConfig(tkc::CutlassTileConfigSM100::CtaShape128x256x256B, tkc::MainloopScheduleType::AUTO,
        tkc::EpilogueScheduleType::AUTO, tkc::ClusterShape::ClusterShape_4x4x1);
}

template <typename T>
void runMxfp8Gemm(at::Tensor& out, at::Tensor const& act, at::Tensor const& weight, at::Tensor const& actScale,
    at::Tensor const& weightScale, at::Tensor const& globalScale, int64_t m, int64_t n, int64_t k,
    tkc::CutlassGemmConfig const& gemmConfig)
{
    CutlassFp4GemmRunner<T, FP4GemmType::W8A8_MXFP8_MXFP8> gemmRunner;
    int64_t const wsBytes = gemmRunner.getWorkspaceSize(m, n, k, /*batch_count=*/1);

    at::Tensor workspace = at::detail::empty_cuda({wsBytes}, at::ScalarType::Char, act.device(), std::nullopt);

    gemmRunner.gemm(out.data_ptr(), act.const_data_ptr(), weight.const_data_ptr(), actScale.const_data_ptr(),
        weightScale.const_data_ptr(), globalScale.data_ptr<float>(), m, n, k, /*batch_count=*/1, gemmConfig,
        reinterpret_cast<char*>(workspace.data_ptr()), wsBytes, at::cuda::getCurrentCUDAStream(act.get_device()));
}

} // namespace

// MXFP8 (e4m3 + UE8M0 1x32 block scales) x MXFP8 (e4m3 + UE8M0 1x32 block
// scales) GEMM on Blackwell sm_100/103.
//
// Operands (matching the CUTLASS block-scaled tensor-op convention):
//   act:          [M, K] Float8_e4m3fn, row-major.
//   actScale:     1D uint8 (UE8M0), swizzled layout produced by
//                 torch.ops.trtllm.mxfp8_quantize(input, swizzedLayout=True).
//   weight:       [N, K] Float8_e4m3fn, expected to be column-major in memory.
//                 The caller is responsible for ensuring the weight tensor is
//                 contiguous in the column-major sense that CUTLASS expects.
//   weightScale:  1D uint8 (UE8M0), swizzled layout produced by
//                 torch.ops.trtllm.block_scale_interleave(scale).
//   globalScale:  [1] float -- alpha multiplier baked into the epilogue.
//                 For pure MXFP8xMXFP8 this is usually [1.0].
//   out_dtype:    fp16 / bf16 / fp32 output element type.
at::Tensor mxfp8_mxfp8_gemm(at::Tensor const& act, at::Tensor const& actScale, at::Tensor const& weight,
    at::Tensor const& weightScale, at::Tensor const& globalScale, std::optional<c10::ScalarType> out_dtype)
{
    CHECK_INPUT(act, torch::kFloat8_e4m3fn);
    CHECK_INPUT(weight, torch::kFloat8_e4m3fn);
    CHECK_INPUT(actScale, SF_DTYPE);
    CHECK_INPUT(weightScale, SF_DTYPE);
    CHECK_INPUT(globalScale, at::ScalarType::Float);

    TORCH_CHECK(act.dim() == 2, "act must be a 2D tensor [M, K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor [N, K]");

    int64_t const m = act.sizes()[0];
    int64_t const k = act.sizes()[1];
    int64_t const n = weight.sizes()[0];
    TORCH_CHECK(
        weight.sizes()[1] == k, "act and weight K dims must match: act K=", k, ", weight K=", weight.sizes()[1]);

    // K must be divisible by the UE8M0 block size (32) for both A and B.
    constexpr int kBlock = 32;
    TORCH_CHECK(k % kBlock == 0, "K (", k, ") must be divisible by MXFP8 block size ", kBlock);
    // N must also be aligned to the kernel's tile-N alignment requirement.
    constexpr int kAlignmentN = 32;
    TORCH_CHECK(n % kAlignmentN == 0, "N (", n, ") must be divisible by ", kAlignmentN);

    auto chosen_dtype = out_dtype.value_or(torch::kBFloat16);
    TORCH_CHECK(chosen_dtype == torch::kFloat || chosen_dtype == torch::kHalf || chosen_dtype == torch::kBFloat16,
        "out_dtype must be one of fp16/bf16/fp32 (default bf16).");

    at::Tensor out = at::detail::empty_cuda({m, n}, chosen_dtype, act.device(), std::nullopt);

    auto const config = getDefaultMxfp8GemmConfig();
    switch (chosen_dtype)
    {
    case at::ScalarType::Half:
        runMxfp8Gemm<half>(out, act, weight, actScale, weightScale, globalScale, m, n, k, config);
        break;
    case at::ScalarType::BFloat16:
#ifdef ENABLE_BF16
        runMxfp8Gemm<__nv_bfloat16>(out, act, weight, actScale, weightScale, globalScale, m, n, k, config);
#else
        C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled to run MXFP8xMXFP8 GEMM with bf16 output.");
#endif
        break;
    case at::ScalarType::Float:
        runMxfp8Gemm<float>(out, act, weight, actScale, weightScale, globalScale, m, n, k, config);
        break;
    default: C10_THROW_ERROR(NotImplementedError, "out_dtype must be one of fp16/bf16/fp32.");
    }
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mxfp8_mxfp8_gemm(Tensor act, Tensor actScale, Tensor weight, Tensor weightScale, "
        "Tensor globalScale, ScalarType? out_dtype=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mxfp8_mxfp8_gemm", &tensorrt_llm::torch_ext::mxfp8_mxfp8_gemm);
}
