/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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
void runGemm(at::Tensor& out, at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat2Scale,
    at::Tensor const& globalScale, int64_t m, int64_t n, int64_t k)
{
    tensorrt_llm::kernels::TrtllmGenGemmRunnerOptions options = {.eltTypeA = tg::Dtype::E2m1,
        .eltTypeB = tg::Dtype::E4m3,
        .outputType = outDtype,
        .deepSeekFp8 = false,
        .transposeMmaOutput = true};

    tensorrt_llm::kernels::TrtllmGenGemmRunner runner(options);

    int64_t const numBytesWorkspace = runner.getWorkspaceSizeInBytes(m, n, k);
    at::Tensor workspace
        = at::detail::empty_cuda({numBytesWorkspace}, at::ScalarType::Char, torch::kCUDA, std::nullopt);

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    float* mat2ScalePtr = static_cast<float*>(mat2Scale.data_ptr());
    float* outScalePtr = globalScale.data_ptr<float>();

    runner.run(m, n, k, mat1.const_data_ptr(), nullptr, mat2.const_data_ptr(), /* bScale */ mat2ScalePtr,
        out.data_ptr(), outScalePtr, /* cScalePtr */ nullptr, workspace.data_ptr(), stream.stream(), mat1.get_device());
}

at::Tensor fp4_fp8_gemm_impl(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat2Scale,
    at::Tensor const& globalScale, std::optional<c10::ScalarType> out_dtype)
{
    using tensorrt_llm::kernels::Data_type;

    CHECK_INPUT(mat1, c10::ScalarType::Float8_e4m3fn);
    CHECK_INPUT(mat2, FLOAT4_E2M1X2);

    CHECK_INPUT(mat2Scale, c10::ScalarType::Float8_e4m3fn);

    CHECK_INPUT(globalScale, c10::ScalarType::Float);

    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");

    TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[1] * 2, "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes()[0],
        "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

    auto const m = mat1.sizes()[0];
    auto const n = mat2.sizes()[0];
    auto const k = mat1.sizes()[1];

    if (!out_dtype)
    {
        out_dtype = torch::kHalf;
    }
    TORCH_CHECK(out_dtype == torch::kFloat8_e4m3fn || out_dtype == torch::kHalf || out_dtype == torch::kBFloat16,
        "out_dtype must be one of fp8/fp16/bf16. It defaults to fp16.");

    at::Tensor out = at::detail::empty_cuda({m, n}, out_dtype.value(), mat1.device(), std::nullopt);

    switch (out_dtype.value())
    {
    case at::ScalarType::Float8_e4m3fn:
        runGemm<tg::Dtype::E4m3>(out, mat1, mat2, mat2Scale, globalScale, m, n, k);
        break;
    case at::ScalarType::Half: runGemm<tg::Dtype::Fp16>(out, mat1, mat2, mat2Scale, globalScale, m, n, k); break;
    case at::ScalarType::BFloat16:
        runGemm<tg::Dtype::Bfloat16>(out, mat1, mat2, mat2Scale, globalScale, m, n, k);
        break;
    default: C10_THROW_ERROR(NotImplementedError, "out_dtype must be one of fp8/fp16/bf16.");
    }
    return out;
}

} // namespace

at::Tensor fp4_fp8_gemm_trtllmgen(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat2Scale,
    at::Tensor const& globalScale, std::optional<c10::ScalarType> out_dtype)
{
    return fp4_fp8_gemm_impl(mat1, mat2, mat2Scale, globalScale, out_dtype);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fp4_fp8_gemm_trtllmgen(Tensor mat1, Tensor mat2, Tensor mat2Scale, Tensor globalScale, "
        "ScalarType? out_dtype=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp4_fp8_gemm_trtllmgen", &torch_ext::fp4_fp8_gemm_trtllmgen);
}
