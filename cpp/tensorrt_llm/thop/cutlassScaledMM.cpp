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
#if defined(USING_OSS_CUTLASS_LOW_LATENCY_GEMM)
#include "tensorrt_llm/kernels/cutlass_kernels/include/low_latency_gemm.h"
#else
#include "low_latency_gemm.h"
#endif
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <torch/extension.h>

using torch::Tensor;
#if defined(USING_OSS_CUTLASS_LOW_LATENCY_GEMM)
using tensorrt_llm::kernels::cutlass_kernels::CutlassLowLatencyFp8GemmRunner;
using tensorrt_llm::kernels::cutlass_kernels::CutlassLowLatencyFp8GemmRunnerInterface;
using tensorrt_llm::kernels::cutlass_kernels::LowLatencyCutlassGemmConfig;
using tensorrt_llm::kernels::cutlass_kernels::KernelScheduleType;
#else
using tensorrt_llm::kernels::internal_cutlass_kernels::CutlassLowLatencyFp8GemmRunner;
using tensorrt_llm::kernels::internal_cutlass_kernels::CutlassLowLatencyFp8GemmRunnerInterface;
using tensorrt_llm::kernels::internal_cutlass_kernels::LowLatencyCutlassGemmConfig;
using tensorrt_llm::kernels::internal_cutlass_kernels::KernelScheduleType;
#endif
namespace torch_ext
{

namespace
{

namespace tkc = tensorrt_llm::cutlass_extensions;
using LowLatencyGemmRunnerPtr = std::shared_ptr<CutlassLowLatencyFp8GemmRunnerInterface>;
using FP8Type = __nv_fp8_e4m3;

void cutlass_gemm_caller(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& scale_a, torch::Tensor const& scale_b)
{
    int32_t m = a.sizes()[0];
    int32_t n = b.sizes()[1];
    int32_t k = a.sizes()[1];

    LowLatencyGemmRunnerPtr lowLatencyGemmRunner;
    auto const dtype = out.dtype();
    if (dtype == torch::kFloat32)
    {
        lowLatencyGemmRunner = std::make_shared<CutlassLowLatencyFp8GemmRunner<float>>();
    }
    else if (dtype == torch::kHalf)
    {
        lowLatencyGemmRunner = std::make_shared<CutlassLowLatencyFp8GemmRunner<half>>();
    }
#ifdef ENABLE_BF16
    else if (dtype == torch::kBFloat16)
    {
        lowLatencyGemmRunner = std::make_shared<CutlassLowLatencyFp8GemmRunner<__nv_bfloat16>>();
    }
#endif
    else
    {
        TLLM_THROW("Unsupported data type");
    }

    size_t workspace_size = lowLatencyGemmRunner->getWorkspaceSize(m, n, k);
    auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
    auto workspace = torch::empty(workspace_size, workspace_options);

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    auto env_pdl_overlap_ratio = getFloatEnv("TRTLLM_PDL_OVERLAP_RATIO");
    auto env_prefetch_ratio = getFloatEnv("TRTLLM_PREFETCH_RATIO");
    auto valid_ratio = [](std::optional<float>& env_val, float default_val)
    {
        if (env_val.has_value())
        {
            TLLM_CHECK_WITH_INFO(env_val.value() <= 1.0F, "Valid ratio should be less than or equal to 1.0");
            return env_val.value();
        }
        return default_val;
    };
    float pdl_overlap_ratio = valid_ratio(env_pdl_overlap_ratio, /*default_val=*/0.5);
    float prefetch_ratio = valid_ratio(env_prefetch_ratio, /*default_val=*/-1.0);

    auto* a_ptr = static_cast<FP8Type*>(a.data_ptr());
    auto* b_ptr = static_cast<FP8Type*>(b.data_ptr());
    auto* c_ptr = static_cast<FP8Type*>(out.data_ptr());
    auto* ws_ptr = static_cast<char*>(workspace.data_ptr());
    // TODO(zhenhuan): this will invoke D2H, will fix this when CutlassLowLatencyFp8GemmRunner support scale_a/b
    auto a_scale = scale_a.item().toFloat();
    auto b_scale = scale_b.item().toFloat();

    tkc::CutlassGemmConfig config;
    int32_t const mp2 = nextPowerOfTwo(m);

    if (mp2 <= 64)
    {
        // TODO(zhenhuan): ClusterShape_1x8x1 from vLLM is not support for Prefetch KernelScheduleType
        config = tkc::CutlassGemmConfig(tkc::CutlassTileConfigSM90::CtaShape64x64x128B, tkc::MainloopScheduleType::AUTO,
            tkc::EpilogueScheduleType::AUTO, tkc::ClusterShape::ClusterShape_8x1x1);
    }
    else if (mp2 <= 128)
    {
        config = tkc::CutlassGemmConfig(tkc::CutlassTileConfigSM90::CtaShape64x128x128B,
            tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO, tkc::ClusterShape::ClusterShape_2x1x1);
    }
    else
    {
        config = tkc::CutlassGemmConfig(tkc::CutlassTileConfigSM90::CtaShape128x128x128B,
            tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO, tkc::ClusterShape::ClusterShape_2x1x1);
    }

    lowLatencyGemmRunner->gemm(a_ptr, b_ptr, a_scale * b_scale, 0.F, nullptr, c_ptr, m, n, k, pdl_overlap_ratio,
        prefetch_ratio, LowLatencyCutlassGemmConfig{config, KernelScheduleType::WS_PREFETECH}, ws_ptr, workspace_size,
        stream);
}

} // namespace

Tensor& cutlass_scaled_mm_out(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    std::optional<at::Tensor> const& bias, std::optional<c10::ScalarType> out_dtype, Tensor& out)
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
    TORCH_CHECK(scale_a.numel() == 1 || scale_a.numel() == mat_a.sizes()[0]);
    TORCH_CHECK(scale_b.numel() == 1 || scale_b.numel() == mat_b.sizes()[1]);

    // Check for strides and alignment
    TORCH_CHECK(mat_a.strides()[1] == 1 && out.strides()[1] == 1);           // Row-major
    TORCH_CHECK(mat_b.strides()[0] == 1);                                    // Column-major
    TORCH_CHECK(out.strides()[0] % 16 == 0 && mat_b.strides()[1] % 16 == 0); // 16 Byte Alignment
    TORCH_CHECK(scale_a.is_contiguous() && scale_b.is_contiguous());

    TORCH_CHECK(!bias.has_value(), "bias is not support yet");

    TORCH_CHECK(mat_a.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(mat_b.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");

    cutlass_gemm_caller(out, mat_a, mat_b, scale_a, scale_b);
    return out;
}

Tensor cutlass_scaled_mm(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    std::optional<at::Tensor> const& bias, std::optional<c10::ScalarType> out_dtype)
{
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
    Tensor out = at::empty({mat_a.sizes()[0], mat_b.sizes()[1]}, mat_a.options().dtype(out_dtype_));
    return cutlass_scaled_mm_out(mat_a, mat_b, scale_a, scale_b, bias, out_dtype, out);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "cutlass_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scale_a, Tensor scale_b, Tensor? bias,"
        " ScalarType? out_dtype) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("cutlass_scaled_mm", &torch_ext::cutlass_scaled_mm);
}
