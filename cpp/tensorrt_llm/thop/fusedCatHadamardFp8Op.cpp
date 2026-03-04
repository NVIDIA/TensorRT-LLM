/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/fusedCatHadamardFp8.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

std::tuple<at::Tensor, at::Tensor> fused_cat_hadamard_fp8(at::Tensor const& pe, at::Tensor const& nope, bool use_ue8m0)
{
    CHECK_TH_CUDA(pe);
    CHECK_TH_CUDA(nope);
    CHECK_CONTIGUOUS(pe);
    CHECK_CONTIGUOUS(nope);

    TORCH_CHECK(pe.scalar_type() == at::ScalarType::BFloat16, "pe must be BF16, got ", pe.scalar_type());
    TORCH_CHECK(nope.scalar_type() == at::ScalarType::BFloat16, "nope must be BF16, got ", nope.scalar_type());
    TORCH_CHECK(pe.dim() == 2, "pe must be 2D [M, pe_dim], got ", pe.dim(), "D");
    TORCH_CHECK(nope.dim() == 2, "nope must be 2D [M, nope_dim], got ", nope.dim(), "D");
    TORCH_CHECK(pe.size(0) == nope.size(0), "pe and nope must have same M dimension. pe: ", pe.size(0),
        ", nope: ", nope.size(0));

    auto const M = static_cast<int32_t>(pe.size(0));
    auto const pe_dim = static_cast<int32_t>(pe.size(1));
    auto const nope_dim = static_cast<int32_t>(nope.size(1));
    auto const head_dim = pe_dim + nope_dim;

    TORCH_CHECK(head_dim == 128, "head_dim (pe_dim + nope_dim) must be 128, got ", head_dim);

    // Allocate output tensors
    at::Tensor fp8_out
        = at::detail::empty_cuda({M, head_dim}, at::ScalarType::Float8_e4m3fn, pe.device(), /* stride */ std::nullopt);
    at::Tensor scale_out
        = at::detail::empty_cuda({M, 1}, at::ScalarType::Float, pe.device(), /* stride */ std::nullopt);

    auto stream = at::cuda::getCurrentCUDAStream(pe.get_device());

    tensorrt_llm::kernels::invokeFusedCatHadamardFp8(reinterpret_cast<__nv_fp8_e4m3*>(fp8_out.data_ptr()),
        reinterpret_cast<float*>(scale_out.data_ptr()), reinterpret_cast<__nv_bfloat16 const*>(pe.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(nope.data_ptr()), M, pe_dim, nope_dim, head_dim, use_ue8m0, stream);

    return {fp8_out, scale_out};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fused_cat_hadamard_fp8(Tensor pe, Tensor nope, bool use_ue8m0=False) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_cat_hadamard_fp8", &tensorrt_llm::torch_ext::fused_cat_hadamard_fp8);
}
