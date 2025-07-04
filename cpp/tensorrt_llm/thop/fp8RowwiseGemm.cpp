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
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_rowwise_gemm/fp8_rowwise_gemm.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "tensorrt_llm/thop/userbuffersTensor.h"

#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/cuda/Resize.h>

#include <cstddef>
#include <cuda_fp16.h>

#include <cstdint>
#include <functional>
#include <type_traits>
#include <vector>

using tensorrt_llm::kernels::cutlass_kernels::CutlassFp8RowwiseGemmRunner;
using tensorrt_llm::kernels::cutlass_kernels::CutlassFp8RowwiseGemmRunnerInterface;

namespace torch_ext
{

namespace
{
void check_input_dtypes(torch::Tensor const& mat, torch::Tensor const& matScale)
{
    TORCH_CHECK(mat.scalar_type() == at::ScalarType::Float8_e4m3fn,
        "Matrix dtype must be FP8 (the matrix will be dequantized on the fly).");

    CHECK_INPUT(matScale, FP8_ROWWISE_SF_DTYPE);
}
} // namespace

template <typename OutputType>
torch::Tensor fp8_rowwise_gemm_launch(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale, bool to_userbuffers = false,
    tkc::CutlassGemmConfig const* maybe_config = nullptr)
{
    check_input_dtypes(mat1, mat1Scale);
    check_input_dtypes(mat2, mat2Scale);

    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
    TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[1], "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes()[0], "x",
        mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");
    TORCH_CHECK(mat1.sizes()[0] == mat1Scale.sizes()[0],
        "mat1Scale should be per-token scale, but got m=", mat1.sizes()[0], ", scale_dim=", mat1Scale.sizes()[0], ".");
    TORCH_CHECK(mat2.sizes()[0] == mat2Scale.sizes()[0],
        "mat2Scale should be per-channel scale, but got n=", mat2.sizes()[0], ", scale_dim=", mat2Scale.sizes()[0],
        ".");

    auto const m = mat1.sizes()[0];
    auto const n = mat2.sizes()[0];
    auto const k = mat1.sizes()[1];

    static_assert(std::is_same<OutputType, half>::value || std::is_same<OutputType, __nv_bfloat16>::value,
        "Output type must be half or bfloat16");
    static constexpr auto outType
        = std::is_same<OutputType, half>::value ? at::ScalarType::Half : at::ScalarType::BFloat16;
    at::Tensor out;
    if (to_userbuffers)
    {
        out = torch_ext::create_userbuffers_tensor({m, n}, outType).first;
    }
    else
    {
        out = at::detail::empty_cuda({m, n}, outType, mat1.device(), std::nullopt);
    }

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    auto mGemmRunner = std::make_shared<CutlassFp8RowwiseGemmRunner<OutputType>>();
    int64_t const wsSize = mGemmRunner->getWorkspaceSize(m, n, k);
    auto gemmConfig = maybe_config ? *maybe_config : mGemmRunner->getConfigs()[0];
    at::Tensor workspace = at::detail::empty_cuda({wsSize}, at::ScalarType::Char, torch::kCUDA, std::nullopt);

    OutputType* outPtr = reinterpret_cast<OutputType*>(out.data_ptr());
    __nv_fp8_e4m3 const* mat1Ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(mat1.data_ptr());
    __nv_fp8_e4m3 const* mat2Ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(mat2.data_ptr());
    float const* mat1ScalePtr = reinterpret_cast<float const*>(mat1Scale.data_ptr());
    float const* mat2ScalePtr = reinterpret_cast<float const*>(mat2Scale.data_ptr());
    char* workspacePtr = reinterpret_cast<char*>(workspace.data_ptr());

    tensorrt_llm::common::QuantMode quantMode = tensorrt_llm::common::QuantMode::fp8RowWise();
    mGemmRunner->gemm(outPtr, mat1Ptr, mat2Ptr, nullptr, quantMode, m, n, k, mat1ScalePtr, mat2ScalePtr, gemmConfig,
        workspacePtr, wsSize, stream);

    return out;
}

template torch::Tensor fp8_rowwise_gemm_launch<half>(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale, bool to_userbuffers = false,
    tkc::CutlassGemmConfig const* maybe_config = nullptr);
template torch::Tensor fp8_rowwise_gemm_launch<__nv_bfloat16>(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale, bool to_userbuffers = false,
    tkc::CutlassGemmConfig const* maybe_config = nullptr);

torch::Tensor fp8_rowwise_gemm_dispatch(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& mat1Scale, torch::Tensor const& mat2Scale, at::ScalarType outDataType,
    bool to_userbuffers = false, tkc::CutlassGemmConfig const* maybe_config = nullptr)
{
    // The functional version of this op does not do any profiling; use the profiler class below instead for
    // better performance.
    // Note that we can still add a heuristic here.
    switch (outDataType)
    {
    case at::ScalarType::Half:
        return fp8_rowwise_gemm_launch<half>(mat1, mat2, mat1Scale, mat2Scale, to_userbuffers, maybe_config);
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
        return fp8_rowwise_gemm_launch<__nv_bfloat16>(mat1, mat2, mat1Scale, mat2Scale, to_userbuffers, maybe_config);
#endif
    default: TORCH_CHECK(false, "Unsupported output dtype for FP8 block scaling GEMM");
    }
}

class FP8RowwiseGemmRunner : public torch::CustomClassHolder
{
public:
    explicit FP8RowwiseGemmRunner(at::ScalarType outputDtype)
        : mOutputDtype(outputDtype)
    {
        if (outputDtype == at::ScalarType::Half)
        {
            mGemmRunner = std::make_unique<CutlassFp8RowwiseGemmRunner<half>>();
        }
#ifdef ENABLE_BF16
        else if (outputDtype == at::ScalarType::BFloat16)
        {
            mGemmRunner = std::make_unique<CutlassFp8RowwiseGemmRunner<__nv_bfloat16>>();
        }
#endif
        else
        {
            C10_THROW_ERROR(NotImplementedError, "out_dtype must be one of fp16/bf16.");
        }
        mConfigs = mGemmRunner->getConfigs();
    }

    at::Tensor runGemm(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
        at::Tensor const& mat2Scale, bool to_userbuffers, int64_t configIdx) const
    {
        tkc::CutlassGemmConfig const* config = nullptr;
        if (configIdx != -1)
        {
            TORCH_CHECK(configIdx >= 0 && configIdx < getNumConfigs());
            config = &mConfigs.at(configIdx);
        }
        return fp8_rowwise_gemm_dispatch(mat1, mat2, mat1Scale, mat2Scale, mOutputDtype, to_userbuffers, config);
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
    std::shared_ptr<CutlassFp8RowwiseGemmRunnerInterface> mGemmRunner{nullptr};
    std::vector<tkc::CutlassGemmConfig> mConfigs;
    at::ScalarType mOutputDtype;
};
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<torch_ext::FP8RowwiseGemmRunner>("FP8RowwiseGemmRunner")
        .def(torch::init<at::ScalarType>())
        .def("run_gemm", &torch_ext::FP8RowwiseGemmRunner::runGemm)
        .def("get_num_configs", &torch_ext::FP8RowwiseGemmRunner::getNumConfigs);
}
