/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/nvfp4QuantKf.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// The Kf NVFP4 quantization kernel that fp4_quantize(input, globalScale, 16, false, false) runs for the
// shapes TRTLLM_NVFP4_QUANT_KF=1 routes to it, here for any supported shape (tests and A/B runs). The outputs are
// those of that call: the E2M1 codes [..., K / 2] and the linear E4M3 scales [M * K / 16], both uint8.
std::tuple<at::Tensor, at::Tensor> fp4_quantize_kf(at::Tensor const& input, at::Tensor const& globalScale)
{
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "fp4_quantize_kf: input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16, "fp4_quantize_kf: input must be BF16");
    TORCH_CHECK(input.dim() >= 2, "fp4_quantize_kf: input must have at least two dimensions");
    TORCH_CHECK(globalScale.is_cuda() && globalScale.device() == input.device(),
        "fp4_quantize_kf: globalScale must be on the input's device");
    TORCH_CHECK(globalScale.scalar_type() == at::ScalarType::Float && globalScale.numel() >= 1,
        "fp4_quantize_kf: globalScale must hold an FP32 value");

    auto const& shape = input.sizes();
    int64_t const k = shape.back();
    TORCH_CHECK(k > 0 && k % 16 == 0, "fp4_quantize_kf: the last dimension must be a positive multiple of 16");
    int64_t m = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i)
    {
        m *= shape[i];
    }

    c10::cuda::CUDAGuard const guard(input.device());
    std::vector<int64_t> outputShape(shape.begin(), shape.end());
    outputShape.back() = k / 2;
    at::Tensor output = at::detail::empty_cuda(outputShape, at::ScalarType::Byte, input.device(), std::nullopt);
    at::Tensor scales = at::detail::empty_cuda({m * (k / 16)}, at::ScalarType::Byte, input.device(), std::nullopt);
    if (m == 0)
    {
        return {output, scales};
    }
    TORCH_CHECK(tensorrt_llm::kernels::isNvfp4QuantKfSupported(input.data_ptr(), output.data_ptr(), m, k),
        "fp4_quantize_kf needs an SM10x GPU, m * k / 16 within int32 and a 32-byte aligned input");
    tensorrt_llm::kernels::invokeNvfp4QuantKf(reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr()),
        globalScale.data_ptr<float>(), reinterpret_cast<uint8_t*>(output.data_ptr()),
        reinterpret_cast<uint8_t*>(scales.data_ptr()), m, k, at::cuda::getCurrentCUDAStream(input.get_device()));
    return {output, scales};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fp4_quantize_kf(Tensor input, Tensor globalScale) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp4_quantize_kf", &tensorrt_llm::torch_ext::fp4_quantize_kf);
}
