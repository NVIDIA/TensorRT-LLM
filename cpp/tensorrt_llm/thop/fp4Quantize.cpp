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

#include "tensorrt_llm/thop/fp4Quantize.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

#include <cuda_fp16.h>

#include <cstdint>
#include <optional>

namespace torch_ext
{
// self: [M, K], fp16/bf16/fp8_quantized
// globalScale: [1] float, = (448 * 6) / self.abs().max(). Not used when sfUseUE8M0 is true.
// nvfp4: sfVecSize = 16, sfUseUE8M0 = false
// mxfp4: sfVecSize = 32, sfUseUE8M0 = true
// alignment: sfVecSize
// isSfSwizzledLayout: bool, if true, the scale factors are stored in swizzled layout, otherwise in linear layout.
// See QuantizationSFLayout enum for more details about the two layouts.
// returns self_fp4, self_block_scale_factors
// self_fp4: [M, K / 2], FLOAT4_E2M1X2
// self_block_scale_factors: ceil(M / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
std::tuple<at::Tensor, at::Tensor> fp4_quantize(at::Tensor const& self, std::optional<at::Tensor> const& globalScale,
    int64_t sfVecSize, bool sfUseUE8M0, bool isSfSwizzledLayout)
{
    CHECK_TH_CUDA(self);
    CHECK_CONTIGUOUS(self);
    if (sfUseUE8M0)
    {
        TORCH_CHECK(sfVecSize == 32, "sfVecSize can only be 32, when sfUseUE8M0 is true");
    }
    else
    {
        TORCH_CHECK(globalScale.has_value(), "globalScale is required when sfUseUE8M0 is false");
        CHECK_INPUT(globalScale.value(), torch::kFloat32);
        TORCH_CHECK(sfVecSize == 16, "sfVecSize can only be 16, when sfUseUE8M0 is false");
    }

    float* globalScalePtr{nullptr};
    if (globalScale.has_value())
    {
        globalScalePtr = globalScale->data_ptr<float>();
    }

    auto const& inputShape = self.sizes();
    auto const& rank = inputShape.size();

    TORCH_CHECK(rank >= 2, "Input should be >=2D tensor.");
    int64_t m = 1;
    for (size_t i = 0; i < rank - 1; i++)
    {
        m *= inputShape[i];
    }
    auto const k = inputShape[rank - 1];
    TORCH_CHECK(k % sfVecSize == 0);

    std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
    outputShape[rank - 1] = k / 2;

    at::Tensor valueE2M1 = at::detail::empty_cuda(outputShape, FLOAT4_E2M1X2, self.device(), /* stride */ std::nullopt);

    int64_t SFSize = isSfSwizzledLayout ? tensorrt_llm::computeSwizzledLayoutSFSize(m, k / sfVecSize)
                                        : tensorrt_llm::computeLinearLayoutSFSize(m, k / sfVecSize);

    at::Tensor scaleFP8SF
        = at::detail::empty_cuda({SFSize}, SF_DTYPE, self.device(), /* stride */ std::nullopt); // 1D tensor

    const thread_local int mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

    auto const layout = isSfSwizzledLayout ? tensorrt_llm::QuantizationSFLayout::SWIZZLED
                                           : tensorrt_llm::QuantizationSFLayout::LINEAR;

#define LAUNCH_FP4_QUANTIZE_KERNEL(T, SF_VEC_SIZE)                                                                     \
    tensorrt_llm::kernels::invokeFP4Quantization<T, SF_VEC_SIZE>(1, m, k, reinterpret_cast<T*>(self.data_ptr()),       \
        globalScalePtr, reinterpret_cast<int64_t*>(valueE2M1.data_ptr()),                                              \
        reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()), sfUseUE8M0, layout, mMultiProcessorCount,                   \
        at::cuda::getCurrentCUDAStream(self.get_device()));

    if (sfUseUE8M0)
    {
        if (self.scalar_type() == at::ScalarType::Half)
        {
            LAUNCH_FP4_QUANTIZE_KERNEL(half, 32)
        }
        else if (self.scalar_type() == at::ScalarType::BFloat16)
        {
#ifdef ENABLE_BF16
            LAUNCH_FP4_QUANTIZE_KERNEL(__nv_bfloat16, 32)
#else
            C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled to quantize an bf16 tensor to fp4.");
#endif
        }
        else if (self.scalar_type() == at::ScalarType::Float8_e4m3fn)
        {
#ifdef ENABLE_FP8
            LAUNCH_FP4_QUANTIZE_KERNEL(__nv_fp8_e4m3, 32)
#else
            C10_THROW_ERROR(NotImplementedError, "FP8 must be enabled to quantize an fp8 tensor to fp4.");
#endif
        }
        else
        {
            C10_THROW_ERROR(NotImplementedError, "fp4_quantize only supports input tensor with dtypes fp16/bf16/e4m3.");
        }
    }
    else
    {
        if (self.scalar_type() == at::ScalarType::Half)
        {
            LAUNCH_FP4_QUANTIZE_KERNEL(half, 16)
        }
        else if (self.scalar_type() == at::ScalarType::BFloat16)
        {
#ifdef ENABLE_BF16
            LAUNCH_FP4_QUANTIZE_KERNEL(__nv_bfloat16, 16)
#else
            C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled to quantize an bf16 tensor to fp4.");
#endif
        }
        else if (self.scalar_type() == at::ScalarType::Float8_e4m3fn)
        {
#ifdef ENABLE_FP8
            LAUNCH_FP4_QUANTIZE_KERNEL(__nv_fp8_e4m3, 16)
#else
            C10_THROW_ERROR(NotImplementedError, "FP8 must be enabled to quantize an fp8 tensor to fp4.");
#endif
        }
        else
        {
            C10_THROW_ERROR(NotImplementedError, "fp4_quantize only supports input tensor with dtypes fp16/bf16/e4m3.");
        }
    }

#undef LAUNCH_FP4_QUANTIZE_KERNEL

    return {valueE2M1, scaleFP8SF};
}

at::Tensor calculate_nvfp4_global_scale(at::Tensor const& input, std::optional<at::Tensor> const& tokensPerBatch)
{
    CHECK_TH_CUDA(input);
    CHECK_CONTIGUOUS(input);

    auto const& inputShape = input.sizes();
    auto const& rank = inputShape.size();

    TORCH_CHECK(rank >= 2 && rank <= 3);

    // Calculate batch and token numbers
    int64_t batch_size = 1;
    int64_t token_num = 1;
    int64_t hidden_size = inputShape[rank - 1];

    if (rank == 2)
    {
        // [token_num, hidden_size]
        token_num = inputShape[0];
        batch_size = 1;
    }
    else if (rank == 3)
    {
        // [batch, token_num, hidden_size]
        batch_size = inputShape[0];
        token_num = inputShape[1];
    }

    // Create output tensor with same dimensions as input, but last dimension size is 1
    std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
    outputShape[rank - 1] = 1;

    at::Tensor globalScale = at::detail::empty_cuda(outputShape, torch::kFloat32, input.device(), std::nullopt);

    // Get multi-processor count
    static int multiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

    // Prepare tokensPerBatch pointer - should have shape (batch_size)
    int const* tokensPerBatchPtr = nullptr;
    if (tokensPerBatch.has_value())
    {
        CHECK_TH_CUDA(tokensPerBatch.value());
        CHECK_CONTIGUOUS(tokensPerBatch.value());

        auto const& tokensShape = tokensPerBatch.value().sizes();
        TORCH_CHECK(tokensShape.size() == 1, "tokensPerBatch should have exactly 1 dimension");
        TORCH_CHECK(tokensShape[0] == batch_size, "tokensPerBatch first dimension must match input batch size");

        tokensPerBatchPtr = tokensPerBatch.value().data_ptr<int>();
    }

    // Call corresponding kernel based on input data type
    if (input.scalar_type() == at::ScalarType::Half)
    {
        tensorrt_llm::kernels::computePerTokenGlobalScaleForFP4Quantization<half>(batch_size, token_num, hidden_size,
            reinterpret_cast<half const*>(input.data_ptr()), tokensPerBatchPtr, globalScale.data_ptr<float>(),
            multiProcessorCount, at::cuda::getCurrentCUDAStream(input.get_device()));
    }
    else if (input.scalar_type() == at::ScalarType::BFloat16)
    {
#ifdef ENABLE_BF16
        tensorrt_llm::kernels::computePerTokenGlobalScaleForFP4Quantization<__nv_bfloat16>(batch_size, token_num,
            hidden_size, reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr()), tokensPerBatchPtr,
            globalScale.data_ptr<float>(), multiProcessorCount, at::cuda::getCurrentCUDAStream(input.get_device()));
#else
        C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled to compute global scale for bf16 tensor.");
#endif
    }
    else
    {
        C10_THROW_ERROR(
            NotImplementedError, "calculate_nvfp4_global_scale only supports input tensor with dtypes fp16/bf16.");
    }

    return globalScale;
}
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fp4_quantize(Tensor input, Tensor? globalScale, int sfVecSize, bool sfUseUE8M0=False, bool "
        "isSfSwizzledLayout=True) "
        "-> (Tensor, Tensor)");
    m.def("calculate_nvfp4_global_scale(Tensor input, Tensor? tokensPerBatch) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp4_quantize", TORCH_FN(torch_ext::fp4_quantize));
    m.impl("calculate_nvfp4_global_scale", TORCH_FN(torch_ext::calculate_nvfp4_global_scale));
}
