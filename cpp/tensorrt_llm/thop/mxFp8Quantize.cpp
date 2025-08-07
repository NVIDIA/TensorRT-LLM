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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

#include <cuda_fp16.h>

#include <cstdint>

namespace torch_ext
{
// self: [M, K], fp16/bf16/fp8_quantized
// mxfp8: sfVecSize = 32
// alignment: sfVecSize
// isSfSwizzledLayout: bool, if true, the scale factors are stored in swizzled layout, otherwise in linear layout.
// See QuantizationSFLayout enum for more details about the two layouts.
// returns self_mxfp8, self_block_scale_factors
// self_mxfp8: [M, K], Float8_e4m3fn
// self_block_scale_factors: ceil(M / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE
std::tuple<at::Tensor, at::Tensor> mxfp8_quantize(
    at::Tensor const& self, bool isSfSwizzledLayout, int64_t alignment = 32)
{
    CHECK_TH_CUDA(self);
    CHECK_CONTIGUOUS(self);

    // Fixed SF_VEC_SIZE as 32
    static constexpr int SF_VEC_SIZE = 32;
    TORCH_CHECK(alignment % SF_VEC_SIZE == 0, "alignment must be divisible by SF_VEC_SIZE = 32");

    auto const& inputShape = self.sizes();
    auto const& rank = inputShape.size();

    TORCH_CHECK(rank >= 2, "Input should be >=2D tensor.");
    int64_t m = 1;
    for (size_t i = 0; i < rank - 1; i++)
    {
        m *= inputShape[i];
    }
    auto const k = inputShape[rank - 1];
    TORCH_CHECK(k % SF_VEC_SIZE == 0, "k must be divisible by SF_VEC_SIZE = 32");
    auto const padded_k = ((k + alignment - 1) / alignment) * alignment;

    std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
    outputShape[rank - 1] = padded_k;

    at::Tensor valMxFP8
        = at::detail::empty_cuda(outputShape, at::ScalarType::Float8_e4m3fn, self.device(), /* stride */ std::nullopt);

    int64_t SFSize = isSfSwizzledLayout ? tensorrt_llm::computeSwizzledLayoutSFSize(m, padded_k / SF_VEC_SIZE)
                                        : tensorrt_llm::computeLinearLayoutSFSize(m, padded_k / SF_VEC_SIZE);

    at::Tensor scaleFP8SF
        = at::detail::empty_cuda({SFSize}, SF_DTYPE, self.device(), /* stride */ std::nullopt); // 1D tensor

    const thread_local int mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

    auto const layout = isSfSwizzledLayout ? tensorrt_llm::QuantizationSFLayout::SWIZZLED
                                           : tensorrt_llm::QuantizationSFLayout::LINEAR;

#define LAUNCH_MXFP8_QUANTIZE_KERNEL(T)                                                                                \
    tensorrt_llm::kernels::invokeMxFP8Quantization(1, m, k, padded_k, reinterpret_cast<T*>(self.data_ptr()),           \
        reinterpret_cast<int64_t*>(valMxFP8.data_ptr()), reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()), layout,    \
        mMultiProcessorCount, at::cuda::getCurrentCUDAStream(self.get_device()));

    if (self.scalar_type() == at::ScalarType::Half)
    {
        LAUNCH_MXFP8_QUANTIZE_KERNEL(half)
    }
    else if (self.scalar_type() == at::ScalarType::BFloat16)
    {
#ifdef ENABLE_BF16
        LAUNCH_MXFP8_QUANTIZE_KERNEL(__nv_bfloat16)
#else
        C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled to quantize an bf16 tensor to mxfp8.");
#endif
    }
    else
    {
        C10_THROW_ERROR(NotImplementedError, "mxfp8_quantize only supports input tensor with dtypes fp16/bf16.");
    }

#undef LAUNCH_MXFP8_QUANTIZE_KERNEL

    return {valMxFP8, scaleFP8SF};
}
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mxfp8_quantize(Tensor input, bool swizzedLayout=True, int alignment=32) "
        "-> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mxfp8_quantize", &torch_ext::mxfp8_quantize);
}
