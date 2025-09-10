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
#include "tensorrt_llm/kernels/trtllmGenKernels/gemmGatedAct/KernelRunner.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/cuda/Resize.h>

#include <cuda_fp16.h>

#include <cstdint>

namespace torch_ext
{

namespace
{
template <gemm::trtllm::gen::Dtype outDtype>
void runGemm(at::Tensor& out, at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& globalScale, int64_t m,
    int64_t n, int64_t k, bool lowLatencyKernel)
{
    auto eltType = gemm::trtllm::gen::Dtype::E4m3;

    tensorrt_llm::kernels::TrtllmGenGemmRunnerOptions options
        = {.eltTypeA = eltType, .outputType = outDtype, .deepSeekFp8 = false, .transposeMmaOutput = lowLatencyKernel};

    tensorrt_llm::kernels::TrtllmGenGemmRunner runner(options);

    int64_t const numBytesWorkspace = runner.getWorkspaceSizeInBytes(m, n, k);
    at::Tensor workspace
        = at::detail::empty_cuda({numBytesWorkspace}, at::ScalarType::Char, torch::kCUDA, std::nullopt);

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    float* outScalePtr = globalScale.data_ptr<float>();

    runner.run(m, n, k, mat1.const_data_ptr(), mat2.const_data_ptr(), out.data_ptr(), outScalePtr, workspace.data_ptr(),
        stream.stream(), mat1.get_device());
}

template <gemmGatedAct::trtllm::gen::Dtype outDtype>
void runGemmGatedAct(at::Tensor& out, at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& globalScale,
    at::Tensor const& globalScaleGate, int64_t m, int64_t n, int64_t k, bool lowLatencyKernel)
{
    auto eltType = gemmGatedAct::trtllm::gen::Dtype::E4m3;

    tensorrt_llm::kernels::TrtllmGenGemmGatedActRunnerOptions options
        = {.eltType = eltType, .outputType = outDtype, .deepSeekFp8 = false, .transposeMmaOutput = lowLatencyKernel};

    tensorrt_llm::kernels::TrtllmGenGemmGatedActRunner runner(options);

    int64_t const numBytesWorkspace = runner.getWorkspaceSizeInBytes(m, n, k);
    at::Tensor workspace
        = at::detail::empty_cuda({numBytesWorkspace}, at::ScalarType::Char, torch::kCUDA, std::nullopt);

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    float* outScalePtr = globalScale.data_ptr<float>();
    float* outScaleGatePtr = globalScaleGate.data_ptr<float>();

    runner.run(m, n, k, mat1.const_data_ptr(), mat2.const_data_ptr(), out.data_ptr(), outScalePtr, outScaleGatePtr,
        workspace.data_ptr(), stream.stream(), mat1.get_device());
}

torch::Tensor fp8_per_tensor_scaling_tllmg_gemm_impl(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& globalScale, std::optional<c10::ScalarType> outDtype,
    std::optional<torch::Tensor> globalScaleGate, bool lowLatencyKernel, bool gatedSilu)
{
    TORCH_CHECK(mat1.scalar_type() == at::ScalarType::Float8_e4m3fn,
        "Matrix1 dtype must be FP8 (the matrix will be dequantized on the fly).");
    TORCH_CHECK(mat2.scalar_type() == at::ScalarType::Float8_e4m3fn,
        "Matrix2 dtype must be FP8 (the matrix will be dequantized on the fly).");
    TORCH_CHECK(globalScale.scalar_type() == at::ScalarType::Float, "globalScale must be float.");

    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
    TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[1], "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes()[0], "x",
        mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

    auto const m = mat1.sizes()[0];
    auto const n = mat2.sizes()[0];
    auto const k = mat1.sizes()[1];

    if (!outDtype)
    {
        outDtype = torch::kHalf;
    }

    TORCH_CHECK(outDtype == at::ScalarType::Float8_e4m3fn || outDtype == torch::kHalf || outDtype == torch::kBFloat16,
        "outDtype must be one of fp16/bf16/e4m3. It defaults to fp16.");

    int32_t outHiddenDim = n;
    if (gatedSilu)
    {
        TORCH_CHECK(globalScaleGate.has_value(), "globalScaleGate must be provided for gatedSilu.");
        outHiddenDim = n / 2;
    }
    at::Tensor out = at::detail::empty_cuda({m, outHiddenDim}, outDtype.value(), mat1.device(), std::nullopt);

    if (gatedSilu)
    {
        switch (outDtype.value())
        {
        case at::ScalarType::Half:
            runGemmGatedAct<gemmGatedAct::trtllm::gen::Dtype::Fp16>(
                out, mat1, mat2, globalScale, globalScaleGate.value(), m, n, k, lowLatencyKernel);
            break;
        case at::ScalarType::BFloat16:
            runGemmGatedAct<gemmGatedAct::trtllm::gen::Dtype::Bfloat16>(
                out, mat1, mat2, globalScale, globalScaleGate.value(), m, n, k, lowLatencyKernel);
            break;
        case at::ScalarType::Float8_e4m3fn:
            runGemmGatedAct<gemmGatedAct::trtllm::gen::Dtype::E4m3>(
                out, mat1, mat2, globalScale, globalScaleGate.value(), m, n, k, lowLatencyKernel);
            break;
        default: C10_THROW_ERROR(NotImplementedError, "outDtype must be one of fp16/bf16/e4m3.");
        }
    }
    else
    {
        switch (outDtype.value())
        {
        case at::ScalarType::Half:
            runGemm<gemm::trtllm::gen::Dtype::Fp16>(out, mat1, mat2, globalScale, m, n, k, lowLatencyKernel);
            break;
        case at::ScalarType::BFloat16:
            runGemm<gemm::trtllm::gen::Dtype::Bfloat16>(out, mat1, mat2, globalScale, m, n, k, lowLatencyKernel);
            break;
        case at::ScalarType::Float8_e4m3fn:
            runGemm<gemm::trtllm::gen::Dtype::E4m3>(out, mat1, mat2, globalScale, m, n, k, lowLatencyKernel);
            break;
        default: C10_THROW_ERROR(NotImplementedError, "outDtype must be one of fp16/bf16/e4m3.");
        }
    }
    return out;
}
} // namespace

torch::Tensor fp8_per_tensor_scaling_tllmg_gemm(torch::Tensor const& mat1, torch::Tensor const& mat2,
    torch::Tensor const& globalScale, std::optional<c10::ScalarType> outDtype,
    std::optional<torch::Tensor> globalScaleGate, bool lowLatencyKernel, bool gatedSilu)
{
    return fp8_per_tensor_scaling_tllmg_gemm_impl(
        mat1, mat2, globalScale, outDtype, globalScaleGate, lowLatencyKernel, gatedSilu);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fp8_per_tensor_scaling_tllmg_gemm(Tensor mat1, Tensor mat2, Tensor global_scale, ScalarType? out_dtype=None, "
        "Tensor? global_scale_gate=None, bool low_latency_kernel=False, bool gated_silu=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp8_per_tensor_scaling_tllmg_gemm", &torch_ext::fp8_per_tensor_scaling_tllmg_gemm);
}
