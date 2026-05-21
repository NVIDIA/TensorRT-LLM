/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/cudaCoreGemmW4A16NVFP4.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/cutlassGemmW4A16NVFP4.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/nvfp4ScaleLayout.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>

using torch::Tensor;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{

void checkActDtype(Tensor const& act)
{
    TORCH_CHECK(act.scalar_type() == torch::kFloat16 || act.scalar_type() == torch::kBFloat16,
        "w4a16_nvfp4_gemm only supports FP16/BF16 activations, got ", act.scalar_type());
}

int64_t padUp(int64_t value, int64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

size_t getCudaDataTypeSize(cudaDataType_t dataType)
{
    switch (dataType)
    {
    case CUDA_R_16F:
    case CUDA_R_16BF: return 2;
    case CUDA_R_32F: return 4;
    default: return 0;
    }
}

void checkWeightScaleSize(Tensor const& weightScale, int64_t n, int64_t k)
{
    using namespace tensorrt_llm::kernels::w4a16_nvfp4;
    TORCH_CHECK(
        k % kScaleGranularity == 0, "K must be divisible by ", kScaleGranularity, " for W4A16 NVFP4 GEMM, got K=", k);
    int64_t const expectedNumel = padUp(n, kScaleRowsPerTile) * padUp(k / kScaleGranularity, kPackedScaleColsPerTile);
    TORCH_CHECK(weightScale.numel() >= expectedNumel, "weight_scale has too few elements for W4A16 NVFP4 GEMM: got ",
        weightScale.numel(), ", expected at least ", expectedNumel, " for N=", n, " K=", k);
}

void w4a16Nvfp4GemmCaller(
    Tensor& out, Tensor const& act, Tensor const& weight, Tensor const& weightScale, Tensor const& weightScale2)
{
    constexpr int32_t kCudaCoreMaxM = 16;
    auto const m = static_cast<int32_t>(act.sizes()[0]);
    auto const k = static_cast<int32_t>(act.sizes()[1]);
    auto const n = static_cast<int32_t>(weight.sizes()[0]);
    TORCH_CHECK(weight.sizes()[1] * 2 == k, "weight shape [N, K/2] must match activation shape [M, K]");

    auto stream = at::cuda::getCurrentCUDAStream(act.get_device());

    auto* actPtr = static_cast<void const*>(act.data_ptr());
    auto* weightPtr = static_cast<void const*>(weight.data_ptr());
    auto* weightScalePtr = static_cast<void const*>(weightScale.data_ptr());
    auto* weightScale2Ptr = static_cast<float const*>(weightScale2.data_ptr());
    auto* outPtr = static_cast<void*>(out.data_ptr());

    auto const inputType = convert_torch_dtype(act.scalar_type());
    auto const outType = convert_torch_dtype(out.scalar_type());

    tensorrt_llm::kernels::cuda_core_gemm_w4a16_nvfp4::Params params(
        actPtr, weightPtr, weightScalePtr, weightScale2Ptr, outPtr, m, n, k, inputType, outType);
    bool dispatched = false;
    if (m <= kCudaCoreMaxM)
    {
        dispatched = tensorrt_llm::kernels::cuda_core_gemm_w4a16_nvfp4::cudaCoreGemmDispatcher(params, stream);
    }
    else
    {
        size_t const inputElementSize = getCudaDataTypeSize(inputType);
        size_t const outputElementSize = getCudaDataTypeSize(outType);
        dispatched = inputElementSize != 0 && outputElementSize != 0;
        for (int32_t start = 0; dispatched && start < m; start += kCudaCoreMaxM)
        {
            int32_t const chunkM = std::min(kCudaCoreMaxM, m - start);
            auto const* chunkActPtr
                = static_cast<char const*>(actPtr) + static_cast<size_t>(start) * k * inputElementSize;
            auto* chunkOutPtr = static_cast<char*>(outPtr) + static_cast<size_t>(start) * n * outputElementSize;
            tensorrt_llm::kernels::cuda_core_gemm_w4a16_nvfp4::Params chunkParams(
                chunkActPtr, weightPtr, weightScalePtr, weightScale2Ptr, chunkOutPtr, chunkM, n, k, inputType, outType);
            dispatched = tensorrt_llm::kernels::cuda_core_gemm_w4a16_nvfp4::cudaCoreGemmDispatcher(chunkParams, stream);
        }
    }
    TORCH_CHECK(dispatched, "Failed to dispatch w4a16_nvfp4_gemm kernel");
}

void w4a16Nvfp4CutlassGemmCaller(
    Tensor& out, Tensor const& act, Tensor const& weight, Tensor const& weightScale, Tensor const& weightScale2)
{
    auto const m = static_cast<int32_t>(act.sizes()[0]);
    auto const k = static_cast<int32_t>(act.sizes()[1]);
    auto const n = static_cast<int32_t>(weight.sizes()[0]);
    TORCH_CHECK(weight.sizes()[1] * 2 == k, "weight shape [N, K/2] must match activation shape [M, K]");

    auto stream = at::cuda::getCurrentCUDAStream(act.get_device());

    auto* actPtr = static_cast<void const*>(act.data_ptr());
    auto* weightPtr = static_cast<void const*>(weight.data_ptr());
    auto* weightScalePtr = static_cast<void const*>(weightScale.data_ptr());
    auto* weightScale2Ptr = static_cast<float const*>(weightScale2.data_ptr());
    auto* outPtr = static_cast<void*>(out.data_ptr());

    auto const inputType = convert_torch_dtype(act.scalar_type());
    auto const outType = convert_torch_dtype(out.scalar_type());

    tensorrt_llm::kernels::cutlass_gemm_w4a16_nvfp4::Params params(
        actPtr, weightPtr, weightScalePtr, weightScale2Ptr, outPtr, m, n, k, inputType, outType);
    bool const dispatched = tensorrt_llm::kernels::cutlass_gemm_w4a16_nvfp4::cutlassGemmDispatcher(params, stream);
    TORCH_CHECK(dispatched, "Failed to dispatch w4a16_nvfp4_cutlass_gemm kernel");
}

} // namespace

Tensor& w4a16_nvfp4_gemm_out(Tensor const& act, Tensor const& weight, Tensor const& weightScale,
    Tensor const& weightScale2, std::optional<c10::ScalarType> outDtype, std::optional<Tensor> const& bias, Tensor& out)
{
    CHECK_TH_CUDA(act);
    CHECK_CONTIGUOUS(act);
    checkActDtype(act);
    CHECK_INPUT(weight, FLOAT4_E2M1X2);
    CHECK_INPUT(weightScale, SF_DTYPE);
    CHECK_INPUT(weightScale2, torch::kFloat32);
    CHECK_TH_CUDA(out);
    CHECK_CONTIGUOUS(out);

    TORCH_CHECK(act.dim() == 2 && weight.dim() == 2 && out.dim() == 2);
    TORCH_CHECK(act.sizes()[0] == out.sizes()[0]);
    TORCH_CHECK(weight.sizes()[0] == out.sizes()[1]);
    TORCH_CHECK(weight.sizes()[1] * 2 == act.sizes()[1]);
    checkWeightScaleSize(weightScale, weight.sizes()[0], act.sizes()[1]);
    TORCH_CHECK(weightScale2.numel() == 1, "weight_scale_2 must be a scalar tensor");
    TORCH_CHECK(!bias.has_value(), "w4a16_nvfp4_gemm does not support bias");
    TORCH_CHECK(!outDtype.has_value() || out.scalar_type() == outDtype.value());

    w4a16Nvfp4GemmCaller(out, act, weight, weightScale, weightScale2);
    return out;
}

Tensor w4a16_nvfp4_gemm(Tensor const& act, Tensor const& weight, Tensor const& weightScale, Tensor const& weightScale2,
    std::optional<c10::ScalarType> outDtype, std::optional<Tensor> const& bias)
{
    TORCH_CHECK(act.dim() == 2 && weight.dim() == 2);
    auto const outDtypeValue = outDtype.value_or(act.scalar_type());
    std::vector<int64_t> outputSize = {act.sizes()[0], weight.sizes()[0]};
    Tensor out = at::empty(outputSize, act.options().dtype(outDtypeValue));
    return w4a16_nvfp4_gemm_out(act, weight, weightScale, weightScale2, outDtype, bias, out);
}

Tensor w4a16_nvfp4_cutlass_gemm(Tensor const& act, Tensor const& weight, Tensor const& weightScale,
    Tensor const& weightScale2, std::optional<c10::ScalarType> outDtype, std::optional<Tensor> const& bias)
{
    CHECK_TH_CUDA(act);
    CHECK_CONTIGUOUS(act);
    checkActDtype(act);
    CHECK_INPUT(weight, FLOAT4_E2M1X2);
    CHECK_INPUT(weightScale, SF_DTYPE);
    CHECK_INPUT(weightScale2, torch::kFloat32);

    TORCH_CHECK(act.dim() == 2 && weight.dim() == 2);
    TORCH_CHECK(weight.sizes()[1] * 2 == act.sizes()[1]);
    checkWeightScaleSize(weightScale, weight.sizes()[0], act.sizes()[1]);
    TORCH_CHECK(weightScale2.numel() == 1, "weight_scale_2 must be a scalar tensor");
    TORCH_CHECK(!bias.has_value(), "w4a16_nvfp4_cutlass_gemm does not support bias");

    auto const outDtypeValue = outDtype.value_or(act.scalar_type());
    std::vector<int64_t> outputSize = {act.sizes()[0], weight.sizes()[0]};
    Tensor out = at::empty(outputSize, act.options().dtype(outDtypeValue));
    w4a16Nvfp4CutlassGemmCaller(out, act, weight, weightScale, weightScale2);
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "w4a16_nvfp4_gemm(Tensor act, Tensor weight, Tensor weight_scale, Tensor weight_scale_2, ScalarType? "
        "out_dtype, Tensor? bias=None) -> Tensor");
    m.def(
        "w4a16_nvfp4_cutlass_gemm(Tensor act, Tensor weight, Tensor weight_scale, Tensor weight_scale_2, ScalarType? "
        "out_dtype, Tensor? bias=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("w4a16_nvfp4_gemm", &tensorrt_llm::torch_ext::w4a16_nvfp4_gemm);
    m.impl("w4a16_nvfp4_cutlass_gemm", &tensorrt_llm::torch_ext::w4a16_nvfp4_cutlass_gemm);
}
