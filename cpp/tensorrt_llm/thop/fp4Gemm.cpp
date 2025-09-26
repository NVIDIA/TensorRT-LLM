/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "tensorrt_llm/thop/userbuffersTensor.h"
#if defined(USING_OSS_CUTLASS_FP4_GEMM)
#include "tensorrt_llm/kernels/cutlass_kernels/include/fp4_gemm.h"
#else
#include "fp4_gemm.h"
#endif

#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/cuda/Resize.h>

#include <cstddef>
#include <cuda_fp16.h>

#include <cstdint>
#include <functional>
#include <type_traits>
#include <vector>

namespace tkc = tensorrt_llm::cutlass_extensions;
#if defined(USING_OSS_CUTLASS_FP4_GEMM)
using tensorrt_llm::kernels::cutlass_kernels::FP4GemmType;
using tensorrt_llm::kernels::cutlass_kernels::CutlassFp4GemmRunner;
using tensorrt_llm::kernels::cutlass_kernels::CutlassFp4GemmRunnerInterface;
#else
using tensorrt_llm::kernels::internal_cutlass_kernels::FP4GemmType;
using tensorrt_llm::kernels::internal_cutlass_kernels::CutlassFp4GemmRunner;
using tensorrt_llm::kernels::internal_cutlass_kernels::CutlassFp4GemmRunnerInterface;
#endif

namespace torch_ext
{

namespace
{

tkc::CutlassGemmConfig getDefaultGemmConfig(int64_t m, int64_t n, int64_t k, FP4GemmType fp4GemmType)
{
    auto const sm = tensorrt_llm::common::getSMVersion();
    if (fp4GemmType == FP4GemmType::W4A4_NVFP4_NVFP4)
    {
        if (sm >= 120)
        {
            return tkc::CutlassGemmConfig(tkc::CutlassTileConfigSM120::CtaShape128x128x256B,
                tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO,
                tkc::ClusterShape::ClusterShape_1x1x1);
        }
        else if (sm == 103)
        {
            return tkc::CutlassGemmConfig(tkc::CutlassTileConfigSM100::CtaShape128x256x256B,
                tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO,
                tkc::ClusterShape::ClusterShape_1x1x1);
        }
        else
        {
            return tkc::CutlassGemmConfig(tkc::CutlassTileConfigSM100::CtaShape128x256x128B,
                tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO,
                tkc::ClusterShape::ClusterShape_1x1x1);
        }
    }
    else if (fp4GemmType == FP4GemmType::W4A8_MXFP4_MXFP8)
    {
        return tkc::CutlassGemmConfig(tkc::CutlassTileConfigSM100::CtaShape128x256x256B,
            tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO, tkc::ClusterShape::ClusterShape_4x4x1);
    }
    else
    {
        return tkc::CutlassGemmConfig(tkc::CutlassTileConfigSM100::CtaShape128x256x128B,
            tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO, tkc::ClusterShape::ClusterShape_1x1x1);
    }
}

template <typename T>
void runGemm(at::Tensor& out, at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
    at::Tensor const& mat2Scale, at::Tensor const& globalScale, int64_t m, int64_t n, int64_t k, int64_t batch_count,
    tkc::CutlassGemmConfig const& gemmConfig, FP4GemmType fp4GemmType)
{
    if (fp4GemmType == FP4GemmType::W4A8_MXFP4_MXFP8)
    {
        CutlassFp4GemmRunner<T, FP4GemmType::W4A8_MXFP4_MXFP8> gemmRunner;
        int64_t const wsBytes = gemmRunner.getWorkspaceSize(m, n, k, batch_count);

        at::Tensor workspace = at::detail::empty_cuda({wsBytes}, at::ScalarType::Char, mat1.device(), std::nullopt);

        gemmRunner.gemm(out.data_ptr(), mat1.const_data_ptr(), mat2.const_data_ptr(), mat1Scale.const_data_ptr(),
            mat2Scale.const_data_ptr(), globalScale.data_ptr<float>(), m, n, k, batch_count, gemmConfig,
            reinterpret_cast<char*>(workspace.data_ptr()), wsBytes, at::cuda::getCurrentCUDAStream(mat1.get_device()));
    }
    else if (fp4GemmType == FP4GemmType::W4A4_NVFP4_NVFP4)
    {
        CutlassFp4GemmRunner<T, FP4GemmType::W4A4_NVFP4_NVFP4> gemmRunner;
        int64_t const wsBytes = gemmRunner.getWorkspaceSize(m, n, k, batch_count);

        at::Tensor workspace = at::detail::empty_cuda({wsBytes}, at::ScalarType::Char, mat1.device(), std::nullopt);

        gemmRunner.gemm(out.data_ptr(), mat1.const_data_ptr(), mat2.const_data_ptr(), mat1Scale.const_data_ptr(),
            mat2Scale.const_data_ptr(), globalScale.data_ptr<float>(), m, n, k, batch_count, gemmConfig,
            reinterpret_cast<char*>(workspace.data_ptr()), wsBytes, at::cuda::getCurrentCUDAStream(mat1.get_device()));
    }
}

// mat1: [B, M, K / 2], FLOAT4_E2M1X2 or [B, M, K], FLOAT8_E4M3FN
// mat2: [B, N, K / 2], FLOAT4_E2M1X2
// out: [B, M, N], fp16/bf16/fp32
// mat1Scale: ceil(M / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
// mat2Scale: ceil(N / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
// globalScale: [1], 1 / (((448 * 6) / mat1.abs().max()) * ((448 * 6) / mat2.abs().max()))
// B = 1 for GEMM op as a special case
// Only W4A4_NVFP4 and W4A8_MXFP4_FP8 are currently supported
at::Tensor fp4_bmm_impl(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
    at::Tensor const& mat2Scale, at::Tensor const& globalScale, FP4GemmType fp4GemmType,
    std::optional<c10::ScalarType> out_dtype, bool to_userbuffers = false,
    tkc::CutlassGemmConfig const* maybe_config = nullptr)
{
    if (fp4GemmType == FP4GemmType::W4A8_MXFP4_MXFP8)
    {
        CHECK_INPUT(mat1, torch::kFloat8_e4m3fn);
        CHECK_INPUT(mat2, FLOAT4_E2M1X2);
    }
    else if (fp4GemmType == FP4GemmType::W4A4_NVFP4_NVFP4)
    {
        CHECK_INPUT(mat1, FLOAT4_E2M1X2);
        CHECK_INPUT(mat2, FLOAT4_E2M1X2);
    }
    int mat2_k_scale = fp4GemmType == FP4GemmType::W4A8_MXFP4_MXFP8 ? 2 : 1;

    CHECK_INPUT(mat1Scale, SF_DTYPE);
    CHECK_INPUT(mat2Scale, SF_DTYPE);

    CHECK_INPUT(globalScale, at::ScalarType::Float);

    int64_t m, n, k, b;
    if (mat1.dim() == 2)
    {
        TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
        TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[1] * mat2_k_scale, "mat1 and mat2 shapes cannot be multiplied (",
            mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");
        m = mat1.sizes()[0];
        n = mat2.sizes()[0];
        k = mat2.sizes()[1] * 2;
        b = 1;
    }
    else if (mat1.dim() == 3)
    {
        TORCH_CHECK(mat2.dim() == 3, "mat2 must be a batch of matrices");
        TORCH_CHECK(mat1.sizes()[0] == mat2.sizes()[0], "mat1 and mat2 must have the same batch size (",
            mat1.sizes()[0], " and ", mat2.sizes()[0], ")");
        TORCH_CHECK(mat1.sizes()[2] == mat2.sizes()[2] * mat2_k_scale, "mat1 and mat2 shapes cannot be multiplied (",
            mat1.sizes()[1], "x", mat1.sizes()[2], " and ", mat2.sizes()[1], "x", mat2.sizes()[2], ")");
        m = mat1.sizes()[1];
        n = mat2.sizes()[1];
        k = mat2.sizes()[2] * 2;
        b = mat1.sizes()[0];
    }
    else
    {
        C10_THROW_ERROR(NotImplementedError, "mat1 must be a matrix or a batch of matrices");
    }

    auto config = maybe_config ? *maybe_config : getDefaultGemmConfig(m, n, k, fp4GemmType);

    constexpr int alignment = 32;
    TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment, ", but got mat1 shape: (",
        mat1.sizes()[0], "x", mat1.sizes()[1], "), k: ", k, ".");
    TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment, ", but got mat2 shape: (",
        mat2.sizes()[0], "x", mat2.sizes()[1], ").");

    if (!out_dtype)
    {
        out_dtype = torch::kHalf;
    }
    TORCH_CHECK(out_dtype == torch::kFloat || out_dtype == torch::kHalf || out_dtype == torch::kBFloat16,
        "out_dtype must be one of fp16/bf16/fp32. It defaults to fp16.");

    std::vector<int64_t> out_shape = mat1.dim() == 2 ? std::vector<int64_t>{m, n} : std::vector<int64_t>{b, m, n};
    at::Tensor out;
    if (to_userbuffers)
    {
        out = torch_ext::create_userbuffers_tensor(out_shape, out_dtype.value()).first;
    }
    else
    {
        out = at::detail::empty_cuda(out_shape, out_dtype.value(), mat1.device(), std::nullopt);
    }
    switch (out_dtype.value())
    {
    case at::ScalarType::Half:
        runGemm<half>(out, mat1, mat2, mat1Scale, mat2Scale, globalScale, m, n, k, b, config, fp4GemmType);
        break;
    case at::ScalarType::BFloat16:
        runGemm<__nv_bfloat16>(out, mat1, mat2, mat1Scale, mat2Scale, globalScale, m, n, k, b, config, fp4GemmType);
        break;
    case at::ScalarType::Float:
        runGemm<float>(out, mat1, mat2, mat1Scale, mat2Scale, globalScale, m, n, k, b, config, fp4GemmType);
        break;
    default: C10_THROW_ERROR(NotImplementedError, "out_dtype must be one of fp16/bf16/fp32.");
    }
    return out;
}

} // namespace

at::Tensor fp4_bmm(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
    at::Tensor const& mat2Scale, at::Tensor const& globalScale, int64_t fp4GemmType,
    std::optional<c10::ScalarType> out_dtype)
{
    // The functional version of this op does not do any profiling; use the profiler class below instead for
    // better performance.
    // Note that we can still add a heuristic here.
    return fp4_bmm_impl(
        mat1, mat2, mat1Scale, mat2Scale, globalScale, static_cast<FP4GemmType>(fp4GemmType), out_dtype);
}

class FP4GemmRunner : public torch::CustomClassHolder
{
public:
    explicit FP4GemmRunner(at::ScalarType outputDtype, int64_t fp4GemmType)
        : mOutputDtype(outputDtype)
        , mfp4GemmType(static_cast<FP4GemmType>(fp4GemmType))
    {
        if (outputDtype == at::ScalarType::Half)
        {
            if (mfp4GemmType == FP4GemmType::W4A8_MXFP4_MXFP8)
            {
                mGemmRunner = std::make_unique<CutlassFp4GemmRunner<half, FP4GemmType::W4A8_MXFP4_MXFP8>>();
            }
            else if (mfp4GemmType == FP4GemmType::W4A4_NVFP4_NVFP4)
            {
                mGemmRunner = std::make_unique<CutlassFp4GemmRunner<half, FP4GemmType::W4A4_NVFP4_NVFP4>>();
            }
        }
        else if (outputDtype == at::ScalarType::Float)
        {
            if (mfp4GemmType == FP4GemmType::W4A8_MXFP4_MXFP8)
            {
                mGemmRunner = std::make_unique<CutlassFp4GemmRunner<float, FP4GemmType::W4A8_MXFP4_MXFP8>>();
            }
            else if (mfp4GemmType == FP4GemmType::W4A4_NVFP4_NVFP4)
            {
                mGemmRunner = std::make_unique<CutlassFp4GemmRunner<float, FP4GemmType::W4A4_NVFP4_NVFP4>>();
            }
        }
#ifdef ENABLE_BF16
        else if (outputDtype == at::ScalarType::BFloat16)
        {
            if (mfp4GemmType == FP4GemmType::W4A8_MXFP4_MXFP8)
            {
                mGemmRunner = std::make_unique<CutlassFp4GemmRunner<__nv_bfloat16, FP4GemmType::W4A8_MXFP4_MXFP8>>();
            }
            else if (mfp4GemmType == FP4GemmType::W4A4_NVFP4_NVFP4)
            {
                mGemmRunner = std::make_unique<CutlassFp4GemmRunner<__nv_bfloat16, FP4GemmType::W4A4_NVFP4_NVFP4>>();
            }
        }
#endif
        else
        {
            C10_THROW_ERROR(NotImplementedError, "out_dtype must be one of fp16/bf16/fp32.");
        }
        mConfigs = mGemmRunner->getConfigs();
    }

    at::Tensor runGemm(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
        at::Tensor const& mat2Scale, at::Tensor const& globalScale, bool to_userbuffers, int64_t configIdx) const
    {
        tkc::CutlassGemmConfig const* config = nullptr;
        if (configIdx != -1)
        {
            TORCH_CHECK(configIdx >= 0 && configIdx < getNumConfigs());
            config = &mConfigs.at(configIdx);
        }
        return fp4_bmm_impl(
            mat1, mat2, mat1Scale, mat2Scale, globalScale, mfp4GemmType, mOutputDtype, to_userbuffers, config);
    }

    at::ScalarType getOutputDtype() const
    {
        return mOutputDtype;
    }

    int64_t getNumConfigs() const
    {
        return static_cast<int64_t>(mConfigs.size());
    }

private:
    std::shared_ptr<CutlassFp4GemmRunnerInterface> mGemmRunner{nullptr};
    std::vector<tkc::CutlassGemmConfig> mConfigs;
    at::ScalarType mOutputDtype;
    FP4GemmType mfp4GemmType;
};
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<torch_ext::FP4GemmRunner>("FP4GemmRunner")
        .def(torch::init<at::ScalarType, int64_t>())
        .def("run_gemm", &torch_ext::FP4GemmRunner::runGemm)
        .def("get_num_configs", &torch_ext::FP4GemmRunner::getNumConfigs);

    m.def(
        "fp4_bmm(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale, Tensor globalScale, int fp4GemmType, "
        "ScalarType? out_dtype=None) -> Tensor");
    m.def(
        "fp4_gemm(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale, Tensor globalScale, int fp4GemmType, "
        "ScalarType? out_dtype=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp4_bmm", &torch_ext::fp4_bmm);
    m.impl("fp4_gemm", &torch_ext::fp4_bmm);
}
