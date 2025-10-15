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
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

namespace torch_ext
{

using Fp8BlockScaleGemmRunnerPtr
    = std::unique_ptr<tensorrt_llm::kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunnerInterface>;

std::tuple<at::Tensor, at::Tensor> fp8_quantize_1x128(at::Tensor const& self)
{
    CHECK_TH_CUDA(self);
    CHECK_CONTIGUOUS(self);

    TORCH_CHECK(self.scalar_type() == at::ScalarType::BFloat16, "Input matrix dtype must be BF16.");
    TORCH_CHECK(self.dim() == 2, "input must be a matrix");

    auto const m = self.sizes()[0];
    auto const n = self.sizes()[1];

    TORCH_CHECK(m <= std::numeric_limits<int32_t>::max(), "M must be within int32");
    TORCH_CHECK(n <= std::numeric_limits<int32_t>::max(), "N must be within int32");

    // required by the sm90 fp8_block_scaling gemm kernel
    TORCH_CHECK(n % 16 == 0, "self.sizes()[1] must be a multiple of 16, but got ", n);

    auto mGemmRunner = tensorrt_llm::kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16,
        __nv_fp8_e4m3, __nv_bfloat16>();

    auto const m_padded = (m + 4 - 1) / 4 * 4;

    // row major, add padding required by the sm90 fp8_block_scaling gemm kernel
    at::Tensor valueE4M3 = at::detail::empty_cuda(
        {m_padded, n}, at::ScalarType::Float8_e4m3fn, self.device(), /* stride */ std::nullopt);
    int64_t scaleSizeInBytes = mGemmRunner.getActScaleSize(m, n); // 128-byte aligned
    int64_t elementSize = scaleSizeInBytes / torch::elementSize(FP8_BLOCK_SCALING_SF_DTYPE);

    // col major
    at::Tensor scaleFP8SF = at::detail::empty_cuda(
        {elementSize}, FP8_BLOCK_SCALING_SF_DTYPE, self.device(), /* stride */ std::nullopt); // 1D tensor

    __nv_fp8_e4m3* act_buffer = reinterpret_cast<__nv_fp8_e4m3*>(valueE4M3.data_ptr());
    float* act_scale_buffer = reinterpret_cast<float*>(scaleFP8SF.data_ptr());

    auto stream = at::cuda::getCurrentCUDAStream(self.get_device());

    mGemmRunner.fp8CS1x128(
        act_buffer, act_scale_buffer, reinterpret_cast<__nv_bfloat16 const*>(self.data_ptr()), n, m, stream);

    // Post-process the scale tensor for sm100 gemm/moe kernel
    if (tensorrt_llm::common::isSM100Family())
    {
        auto const num_n_blocks = (n + 127) / 128;
        auto const act_scal_elesize = num_n_blocks * m_padded;
        TORCH_CHECK(act_scal_elesize <= scaleFP8SF.numel(), "Scale tensor size mismatch. Expected at least ",
            act_scal_elesize, " elements, got ", scaleFP8SF.numel());

        // scaleFP8SF = scaleFP8SF[0:num_n_blocks, 0:m] // no 4-element alignment in blackwell
        // TODO: This is a hack to use sm90 quantize kernel for sm100; ideally we should have a separate quantize kernel
        // for sm100.
        scaleFP8SF
            = scaleFP8SF.slice(0, 0, act_scal_elesize).view({num_n_blocks, m_padded}).slice(1, 0, m).contiguous();
    }
    return {valueE4M3.slice(0, 0, m), scaleFP8SF};
}

std::tuple<at::Tensor, at::Tensor> fp8_batched_quantize_1x128_permute102(at::Tensor const& self)
{
    CHECK_TH_CUDA(self);

    TORCH_CHECK(self.scalar_type() == at::ScalarType::BFloat16, "Input matrix dtype must be BF16.");
    TORCH_CHECK(self.dim() == 3, "input must be a matrix");

    // [seq, num_heads, qk_nope_head_dim]
    // [m, b, n]
    auto const m = self.sizes()[0];
    auto const b = self.sizes()[1];
    auto const n = self.sizes()[2];

    auto const lda = self.strides()[1];
    TORCH_CHECK(self.strides()[2] == 1, "Last stride of self must be 1, but got ", self.strides()[2]);
    TORCH_CHECK(self.strides()[0] == lda * b, "First stride of self is expected to be ", lda * b, ", but got ",
        self.strides()[0]);

    TORCH_CHECK(b <= std::numeric_limits<int32_t>::max(), "B must be within int32");
    TORCH_CHECK(m <= std::numeric_limits<int32_t>::max(), "M must be within int32");
    TORCH_CHECK(n <= std::numeric_limits<int32_t>::max(), "N must be within int32");
    // required by the sm90 fp8_block_scaling gemm/bmm kernel
    TORCH_CHECK(n % 16 == 0, "self.sizes()[2] must be a multiple of 16, but got ", n);

    auto mGemmRunner = tensorrt_llm::kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16,
        __nv_fp8_e4m3, __nv_bfloat16>();

    auto const m_padded = (m + 4 - 1) / 4 * 4;

    // input: [b, m, n]
    // apply 102 permute
    at::Tensor valueE4M3 = at::detail::empty_cuda(
        {b * m_padded * n}, at::ScalarType::Float8_e4m3fn, self.device(), /* stride */ std::nullopt);

    int64_t scaleSizeInBytes = mGemmRunner.getActScaleSize(m, b * n);
    int64_t elementSize = scaleSizeInBytes / torch::elementSize(FP8_BLOCK_SCALING_SF_DTYPE);
    int m_4_align = (m + 3) / 4 * 4;
    at::Tensor scaleFP8SF = at::detail::empty_cuda({b, m_4_align, elementSize / b / m_4_align},
        FP8_BLOCK_SCALING_SF_DTYPE, self.device(), /* stride */ std::nullopt);

    __nv_fp8_e4m3* act_buffer = reinterpret_cast<__nv_fp8_e4m3*>(valueE4M3.data_ptr());
    float* act_scale_buffer = reinterpret_cast<float*>(scaleFP8SF.data_ptr());

    auto stream = at::cuda::getCurrentCUDAStream(self.get_device());

    auto* output_buffer = reinterpret_cast<__nv_bfloat16 const*>(self.data_ptr());
    mGemmRunner.fp8CS1x128Reshape(act_buffer, act_scale_buffer, output_buffer, n, b, m, lda, stream);

    return {valueE4M3.slice(0, 0, b * m * n).view({b, m, n}), scaleFP8SF};
}
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fp8_quantize_1x128(Tensor input) -> (Tensor, Tensor)");
    m.def("fp8_batched_quantize_1x128_permute102(Tensor input) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp8_quantize_1x128", &torch_ext::fp8_quantize_1x128);
    m.impl("fp8_batched_quantize_1x128_permute102", &torch_ext::fp8_batched_quantize_1x128_permute102);
}
