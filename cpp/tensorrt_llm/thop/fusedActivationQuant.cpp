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

#include "tensorrt_llm/kernels/fusedActivationQuant.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

std::tuple<at::Tensor, at::Tensor> fused_relu2_quantize(
    at::Tensor const& input, at::Tensor const& sf_scale, int64_t sf_vec_size)
{
    CHECK_TH_CUDA(input);
    CHECK_CONTIGUOUS(input);
    CHECK_INPUT(sf_scale, torch::kFloat32);

    auto const& inputShape = input.sizes();
    TORCH_CHECK(inputShape.size() == 2, "input should be 2D tensor [M, N].");

    int64_t const m = inputShape[0];
    int64_t const n = inputShape[1];

    TORCH_CHECK(sf_vec_size == 16, "sf_vec_size must be 16 for NVFP4.");
    TORCH_CHECK(n % sf_vec_size == 0, "N must be divisible by sf_vec_size.");

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

    at::Tensor output_fp4 = at::detail::empty_cuda({m, n / 2}, torch::kUInt8, input.device(), std::nullopt);
    int64_t const sfSize = tensorrt_llm::computeSwizzledLayoutSFSize(m, n / sf_vec_size);
    at::Tensor output_sf = at::detail::empty_cuda({sfSize}, SF_DTYPE, input.device(), std::nullopt);

    float const* sfScalePtr = sf_scale.data_ptr<float>();

    if (input.scalar_type() == at::ScalarType::Half)
    {
        kernels::invokeFusedRelu2Quantize<half>(reinterpret_cast<half const*>(input.data_ptr()), sfScalePtr,
            output_fp4.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(), static_cast<int>(m), static_cast<int>(n),
            static_cast<int>(sf_vec_size), stream);
    }
    else if (input.scalar_type() == at::ScalarType::BFloat16)
    {
#ifdef ENABLE_BF16
        kernels::invokeFusedRelu2Quantize<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr()),
            sfScalePtr, output_fp4.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(), static_cast<int>(m),
            static_cast<int>(n), static_cast<int>(sf_vec_size), stream);
#else
        C10_THROW_ERROR(NotImplementedError, "BFloat16 not enabled.");
#endif
    }
    else
    {
        C10_THROW_ERROR(NotImplementedError, "fused_relu2_quantize only supports fp16/bf16.");
    }

    return std::make_tuple(output_fp4, output_sf);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fused_relu2_quantize(Tensor input, Tensor sf_scale, int sf_vec_size=16) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_relu2_quantize", &tensorrt_llm::torch_ext::fused_relu2_quantize);
}
