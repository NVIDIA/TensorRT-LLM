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

#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/gemm/KernelRunner.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/cuda/Resize.h>

#include <cuda_fp16.h>

#include <cstdint>

namespace torch_ext
{

namespace
{

namespace tg = gemm::trtllm::gen;

template <tg::Dtype outDtype>
void runGemm(at::Tensor& out, at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
    at::Tensor const& mat2Scale, at::Tensor const& globalScale, int64_t m, int64_t n, int64_t k)
{
    auto eltType = tg::Dtype::E2m1;

    tensorrt_llm::kernels::TrtllmGenGemmRunnerOptions options
        = {.eltTypeA = eltType, .outputType = outDtype, .deepSeekFp8 = false};

    tensorrt_llm::kernels::TrtllmGenGemmRunner runner(options);

    int64_t const numBytesWorkspace = runner.getWorkspaceSizeInBytes(m, n, k);
    at::Tensor workspace
        = at::detail::empty_cuda({numBytesWorkspace}, at::ScalarType::Char, torch::kCUDA, std::nullopt);

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    float* mat1ScalePtr = static_cast<float*>(mat1Scale.data_ptr());
    float* mat2ScalePtr = static_cast<float*>(mat2Scale.data_ptr());
    float* outScalePtr = globalScale.data_ptr<float>();

    runner.run(m, n, k, mat1.const_data_ptr(), mat1ScalePtr, mat2.const_data_ptr(), mat2ScalePtr, out.data_ptr(),
        outScalePtr, /* cScalePtr */ nullptr, workspace.data_ptr(), stream.stream(), mat1.get_device());
}

// mat1: [M, K / 2], FLOAT4_E2M1X2
// mat2: [N, K / 2], FLOAT4_E2M1X2
// out: [M, N], fp16/bf16/fp32
// mat1Scale: ceil(M / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
// mat2Scale: ceil(N / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
// globalScale: [1], 1 / (((448 * 6) / mat1.abs().max()) * ((448 * 6) / mat2.abs().max()))
// Only NVFP4 is currently supported
at::Tensor fp4_gemm_impl(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
    at::Tensor const& mat2Scale, at::Tensor const& globalScale, bool sfUseUE8M0,
    std::optional<c10::ScalarType> out_dtype)
{
    using tensorrt_llm::kernels::Data_type;

    CHECK_INPUT(mat1, FLOAT4_E2M1X2);
    CHECK_INPUT(mat2, FLOAT4_E2M1X2);

    CHECK_INPUT(mat1Scale, SF_DTYPE);
    CHECK_INPUT(mat2Scale, SF_DTYPE);

    CHECK_INPUT(globalScale, at::ScalarType::Float);

    TORCH_CHECK(!sfUseUE8M0, "use UE8M0 for FP4 Block Scale Factors is not supported yet");

    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
    TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[1], "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes()[0], "x",
        mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

    auto const m = mat1.sizes()[0];
    auto const n = mat2.sizes()[0];
    auto const k = mat1.sizes()[1] * 2;

    if (!out_dtype)
    {
        out_dtype = torch::kHalf;
    }
    TORCH_CHECK(out_dtype == torch::kFloat || out_dtype == torch::kHalf || out_dtype == torch::kBFloat16,
        "out_dtype must be one of fp16/bf16/fp32. It defaults to fp16.");

    at::Tensor out = at::detail::empty_cuda({m, n}, out_dtype.value(), mat1.device(), std::nullopt);

    switch (out_dtype.value())
    {
    case at::ScalarType::Half:
        runGemm<tg::Dtype::Fp16>(out, mat1, mat2, mat1Scale, mat2Scale, globalScale, m, n, k);
        break;
    case at::ScalarType::BFloat16:
        runGemm<tg::Dtype::Bfloat16>(out, mat1, mat2, mat1Scale, mat2Scale, globalScale, m, n, k);
        break;
    case at::ScalarType::Float:
        runGemm<tg::Dtype::Fp32>(out, mat1, mat2, mat1Scale, mat2Scale, globalScale, m, n, k);
        break;
    default: C10_THROW_ERROR(NotImplementedError, "out_dtype must be one of fp16/bf16/fp32.");
    }
    return out;
}

} // namespace

at::Tensor fp4_gemm_trtllmgen(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
    at::Tensor const& mat2Scale, at::Tensor const& globalScale, bool sfUseUE8M0,
    std::optional<c10::ScalarType> out_dtype)
{
    return fp4_gemm_impl(mat1, mat2, mat1Scale, mat2Scale, globalScale, sfUseUE8M0, out_dtype);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fp4_gemm_trtllmgen(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale, Tensor globalScale, bool "
        "sfUseUE8M0, "
        "ScalarType? out_dtype=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp4_gemm_trtllmgen", &torch_ext::fp4_gemm_trtllmgen);
}
