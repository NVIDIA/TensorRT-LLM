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
// self: [B, M, K], fp16/bf16/fp8_quantized
// globalScale: [1] float, = (448 * 6) / self.abs().max()
// nvfp4: sfVecSize = 16, sfUseUE8M0 = false
// mxfp4: sfVecSize = 32 (not supported yet), sfUseUE8M0 = true
// alignment: sfVecSize
// returns self_fp4, self_block_scale_factors
// self_fp4: [B, M, K / 2], FLOAT4_E2M1X2
// self_block_scale_factors: [B, ceil(M / 128) * 128 * ceil(K / sfVecSize / 4) * 4], SF_DTYPE (UE4M3 or UE8M0)
std::tuple<at::Tensor, at::Tensor> fp4_batched_quantize(
    at::Tensor const& self, at::Tensor const& globalScale, int64_t sfVecSize, bool sfUseUE8M0)
{
    CHECK_TH_CUDA(self);
    CHECK_CONTIGUOUS(self);
    CHECK_INPUT(globalScale, torch::kFloat32);
    TORCH_CHECK(sfVecSize == 16, "sfVecSize can only be 16");

    auto const& inputShape = self.sizes();
    auto const& rank = inputShape.size();

    TORCH_CHECK(rank == 3, "Input should be 3D tensor.");

    int64_t b = inputShape[0];
    int64_t m = inputShape[1];
    int64_t k = inputShape[2];

    TORCH_CHECK(k % sfVecSize == 0);

    std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
    outputShape[rank - 1] = k / 2;

    at::Tensor valueE2M1 = at::detail::empty_cuda(outputShape, FLOAT4_E2M1X2, self.device(), /* stride */ std::nullopt);
    at::Tensor scaleFP8SF = at::detail::empty_cuda({b, tensorrt_llm::computeSwizzledLayoutSFSize(m, k / sfVecSize)},
        SF_DTYPE, self.device(), /* stride */ std::nullopt); // 2D tensor

    const thread_local int mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

#define LAUNCH_FP4_QUANTIZE_KERNEL(T)                                                                                  \
    tensorrt_llm::kernels::invokeFP4Quantization(b, m, k, reinterpret_cast<T*>(self.data_ptr()),                       \
        globalScale.data_ptr<float>(), reinterpret_cast<int64_t*>(valueE2M1.data_ptr()),                               \
        reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()), sfUseUE8M0, tensorrt_llm::QuantizationSFLayout::SWIZZLED,   \
        mMultiProcessorCount, at::cuda::getCurrentCUDAStream(self.get_device()));

    if (self.scalar_type() == at::ScalarType::Half)
    {
        LAUNCH_FP4_QUANTIZE_KERNEL(half)
    }
    else if (self.scalar_type() == at::ScalarType::BFloat16)
    {
#ifdef ENABLE_BF16
        LAUNCH_FP4_QUANTIZE_KERNEL(__nv_bfloat16)
#else
        C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled to quantize an bf16 tensor to fp4.");
#endif
    }
    else if (self.scalar_type() == at::ScalarType::Float8_e4m3fn)
    {
#ifdef ENABLE_FP8
        LAUNCH_FP4_QUANTIZE_KERNEL(__nv_fp8_e4m3)
#else
        C10_THROW_ERROR(NotImplementedError, "FP8 must be enabled to quantize an fp8 tensor to fp4.");
#endif
    }
    else
    {
        C10_THROW_ERROR(NotImplementedError, "fp4_quantize only supports input tensor with dtypes fp16/bf16/e4m3.");
    }

#undef LAUNCH_FP4_QUANTIZE_KERNEL

    return {valueE2M1, scaleFP8SF};
}
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fp4_batched_quantize(Tensor input, Tensor globalScale, int sfVecSize, bool sfUseUE8M0=False) -> (Tensor, "
        "Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp4_batched_quantize", &torch_ext::fp4_batched_quantize);
}
