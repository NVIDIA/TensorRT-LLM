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

#include "tensorrt_llm/kernels/internal_cutlass_kernels/include/fp8_blockscale_gemm.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

namespace torch_ext
{

using Fp8BlockScaleGemmRunnerPtr
    = std::unique_ptr<tensorrt_llm::kernels::small_m_gemm::CutlassFp8BlockScaleGemmRunnerInterface>;

std::tuple<at::Tensor, at::Tensor> fp8_quantize(at::Tensor const& self)
{
    CHECK_TH_CUDA(self);
    CHECK_CONTIGUOUS(self);

    TORCH_CHECK(self.scalar_type() == at::ScalarType::BFloat16, "Input matrix dtype must be BF16.");
    TORCH_CHECK(self.dim() == 2, "input must be a matrix");

    auto const m = self.sizes()[0];
    auto const n = self.sizes()[1];

    TORCH_CHECK(m <= std::numeric_limits<int32_t>::max(), "M must be within int32");
    TORCH_CHECK(n <= std::numeric_limits<int32_t>::max(), "N must be within int32");

    Fp8BlockScaleGemmRunnerPtr mGemmRunner
        = std::make_unique<tensorrt_llm::kernels::small_m_gemm::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16,
            __nv_fp8_e4m3, __nv_bfloat16>>();

    at::Tensor valueE4M3
        = at::detail::empty_cuda({m, n}, at::ScalarType::Float8_e4m3fn, self.device(), /* stride */ std::nullopt);
    int64_t scaleSize = mGemmRunner->getActScaleSize(m, n);
    at::Tensor scaleFP8SF = at::detail::empty_cuda(
        {scaleSize}, FP8_BLOCK_SCALING_SF_DTYPE, self.device(), /* stride */ std::nullopt); // 1D tensor

    __nv_fp8_e4m3* act_buffer = reinterpret_cast<__nv_fp8_e4m3*>(valueE4M3.data_ptr());
    float* act_scale_buffer = reinterpret_cast<float*>(scaleFP8SF.data_ptr());

    auto stream = at::cuda::getCurrentCUDAStream(self.get_device());

    mGemmRunner->fp8CS1x128(
        act_buffer, act_scale_buffer, reinterpret_cast<__nv_bfloat16 const*>(self.data_ptr()), n, m, stream);

    return {valueE4M3, scaleFP8SF};
}
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fp8_quantize(Tensor input) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp8_quantize", &torch_ext::fp8_quantize);
}
