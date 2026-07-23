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

enum class MiniMaxM3Mxfp8GemmRole
{
    kAttentionQkv,
    kAttentionOutput,
    kSharedGateUp,
    kSharedDown,
    kDenseGateUp,
    kDenseDown,
    kOther,
};

enum class MiniMaxM3Mxfp8MBand
{
    k8k,
    k16k,
    k32k,
    kOther,
};

MiniMaxM3Mxfp8GemmRole getMiniMaxM3Mxfp8GemmRole(int64_t const n, int64_t const k)
{
    constexpr int64_t kHiddenSize = 6144;
    if (n == 9216 && k == kHiddenSize)
    {
        return MiniMaxM3Mxfp8GemmRole::kAttentionQkv;
    }
    if (n == kHiddenSize && k == 8192)
    {
        return MiniMaxM3Mxfp8GemmRole::kAttentionOutput;
    }
    if (n == kHiddenSize && k == kHiddenSize)
    {
        return MiniMaxM3Mxfp8GemmRole::kSharedGateUp;
    }
    if (n == kHiddenSize && k == 3072)
    {
        return MiniMaxM3Mxfp8GemmRole::kSharedDown;
    }
    if (n == 24576 && k == kHiddenSize)
    {
        return MiniMaxM3Mxfp8GemmRole::kDenseGateUp;
    }
    if (n == kHiddenSize && k == 12288)
    {
        return MiniMaxM3Mxfp8GemmRole::kDenseDown;
    }
    return MiniMaxM3Mxfp8GemmRole::kOther;
}

MiniMaxM3Mxfp8MBand getMiniMaxM3Mxfp8MBand(int64_t const m)
{
    // The 8K-input workload produces one-, two-, and three/four-request CTX
    // batches in these disjoint bands. Both endpoints of every band are
    // validated independently on SM100 and SM103.
    if (m >= 6553 && m <= 8192)
    {
        return MiniMaxM3Mxfp8MBand::k8k;
    }
    if (m >= 13106 && m <= 16384)
    {
        return MiniMaxM3Mxfp8MBand::k16k;
    }
    if (m >= 19659 && m <= 32768)
    {
        return MiniMaxM3Mxfp8MBand::k32k;
    }
    return MiniMaxM3Mxfp8MBand::kOther;
}

tkc::CutlassGemmConfig getMxfp8GemmConfig(int64_t const m, int64_t const n, int64_t const k)
{
    auto const defaultConfig = getDefaultMxfp8GemmConfig();
    auto const role = getMiniMaxM3Mxfp8GemmRole(n, k);
    auto const mBand = getMiniMaxM3Mxfp8MBand(m);
    if (role == MiniMaxM3Mxfp8GemmRole::kOther || mBand == MiniMaxM3Mxfp8MBand::kOther)
    {
        return defaultConfig;
    }

    constexpr int kSm100 = 100;
    constexpr int kSm103 = 103;
    // PyExecutor binds one GPU architecture per rank, so cache the query after the first eligible call.
    static int const smVersion = tensorrt_llm::common::getSMVersion();
    if (smVersion != kSm100 && smVersion != kSm103)
    {
        return defaultConfig;
    }

    auto const makeConfig = [smVersion](tkc::CutlassTileConfigSM100 const tile, tkc::ClusterShape const cluster)
    {
        return tkc::CutlassGemmConfig(tile, tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO, cluster,
            tkc::ClusterShape::Undefined, tkc::ClusterShape::Undefined, smVersion);
    };

    if (smVersion == kSm100)
    {
        if (mBand == MiniMaxM3Mxfp8MBand::k8k)
        {
            bool const useK128Tile = role == MiniMaxM3Mxfp8GemmRole::kSharedDown
                || role == MiniMaxM3Mxfp8GemmRole::kDenseGateUp || role == MiniMaxM3Mxfp8GemmRole::kDenseDown;
            return makeConfig(useK128Tile ? tkc::CutlassTileConfigSM100::CtaShape128x256x128B
                                          : tkc::CutlassTileConfigSM100::CtaShape128x128x256B,
                useK128Tile ? tkc::ClusterShape::ClusterShape_2x1x1 : tkc::ClusterShape::ClusterShape_2x2x1);
        }
        if (mBand == MiniMaxM3Mxfp8MBand::k16k)
        {
            if (role == MiniMaxM3Mxfp8GemmRole::kAttentionQkv)
            {
                return makeConfig(
                    tkc::CutlassTileConfigSM100::CtaShape128x128x256B, tkc::ClusterShape::ClusterShape_2x2x1);
            }
            if (role == MiniMaxM3Mxfp8GemmRole::kAttentionOutput)
            {
                return makeConfig(
                    tkc::CutlassTileConfigSM100::CtaShape128x128x256B, tkc::ClusterShape::ClusterShape_2x1x1);
            }
            return makeConfig(tkc::CutlassTileConfigSM100::CtaShape128x256x128B, tkc::ClusterShape::ClusterShape_2x1x1);
        }

        auto const cluster = role == MiniMaxM3Mxfp8GemmRole::kDenseGateUp ? tkc::ClusterShape::ClusterShape_4x2x1
                                                                          : tkc::ClusterShape::ClusterShape_2x1x1;
        return makeConfig(tkc::CutlassTileConfigSM100::CtaShape128x256x128B, cluster);
    }

    if (mBand == MiniMaxM3Mxfp8MBand::k8k)
    {
        auto const cluster = role == MiniMaxM3Mxfp8GemmRole::kDenseGateUp ? tkc::ClusterShape::ClusterShape_2x1x1
                                                                          : tkc::ClusterShape::ClusterShape_2x2x1;
        return makeConfig(tkc::CutlassTileConfigSM100::CtaShape128x128x256B, cluster);
    }
    if (mBand == MiniMaxM3Mxfp8MBand::k16k)
    {
        auto const use2x4Cluster = role == MiniMaxM3Mxfp8GemmRole::kAttentionQkv
            || role == MiniMaxM3Mxfp8GemmRole::kSharedGateUp || role == MiniMaxM3Mxfp8GemmRole::kSharedDown;
        auto const use2x1Cluster = role == MiniMaxM3Mxfp8GemmRole::kAttentionOutput;
        auto const cluster = use2x4Cluster
            ? tkc::ClusterShape::ClusterShape_2x4x1
            : (use2x1Cluster ? tkc::ClusterShape::ClusterShape_2x1x1 : tkc::ClusterShape::ClusterShape_2x2x1);
        return makeConfig(tkc::CutlassTileConfigSM100::CtaShape128x128x256B, cluster);
    }

    auto const use4x4Cluster
        = role == MiniMaxM3Mxfp8GemmRole::kDenseGateUp || role == MiniMaxM3Mxfp8GemmRole::kDenseDown;
    auto const use2x2Cluster = role == MiniMaxM3Mxfp8GemmRole::kSharedGateUp;
    auto const cluster = use4x4Cluster
        ? tkc::ClusterShape::ClusterShape_4x4x1
        : (use2x2Cluster ? tkc::ClusterShape::ClusterShape_2x2x1 : tkc::ClusterShape::ClusterShape_2x1x1);
    return makeConfig(tkc::CutlassTileConfigSM100::CtaShape128x128x256B, cluster);
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

    auto const config = getMxfp8GemmConfig(m, n, k);
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
