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

#include "tensorrt_llm/kernels/fusedCatFp4.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

std::tuple<at::Tensor, at::Tensor> fused_cat_fp4(at::Tensor const& pe, at::Tensor const& nope)
{
    CHECK_TH_CUDA(pe);
    CHECK_TH_CUDA(nope);
    TORCH_CHECK(pe.device() == nope.device(), "pe and nope must be on the same CUDA device; got pe=", pe.device(),
        ", nope=", nope.device());
    // Scope subsequent CUDA calls (empty_cuda, getCurrentCUDAStream, kernel
    // launch) to pe.device() so a caller whose current device differs cannot
    // cross-dereference the raw data pointers.
    c10::cuda::CUDAGuard deviceGuard{pe.device()};

    TORCH_CHECK(pe.scalar_type() == at::ScalarType::BFloat16, "pe must be BF16, got ", pe.scalar_type());
    TORCH_CHECK(nope.scalar_type() == at::ScalarType::BFloat16, "nope must be BF16, got ", nope.scalar_type());
    TORCH_CHECK(pe.dim() >= 2, "pe must be >= 2D, got ", pe.dim(), "D");
    TORCH_CHECK(nope.dim() >= 2, "nope must be >= 2D, got ", nope.dim(), "D");

    // Innermost dimension must be contiguous for vectorized loads.
    TORCH_CHECK(pe.stride(-1) == 1, "pe must have contiguous innermost dim (stride(-1)==1), got ", pe.stride(-1));
    TORCH_CHECK(nope.stride(-1) == 1, "nope must have contiguous innermost dim (stride(-1)==1), got ", nope.stride(-1));

    // The kernel issues 8-byte vectorized loads (int2 reinterpret of 4x BF16).
    // PyTorch CUDA storage is 256-byte aligned and split() preserves BF16
    // alignment when the split boundary is a multiple of 4 elements (all
    // current call sites satisfy this), but check explicitly so an
    // out-of-tree caller can't silently misalign the load.
    TORCH_CHECK(reinterpret_cast<uintptr_t>(pe.data_ptr()) % 8 == 0,
        "pe.data_ptr() must be 8-byte aligned for vectorized BF16 loads (ptr=",
        reinterpret_cast<void const*>(pe.data_ptr()), ")");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(nope.data_ptr()) % 8 == 0,
        "nope.data_ptr() must be 8-byte aligned for vectorized BF16 loads (ptr=",
        reinterpret_cast<void const*>(nope.data_ptr()), ")");

    auto const pe_dim = static_cast<int32_t>(pe.size(-1));
    auto const nope_dim = static_cast<int32_t>(nope.size(-1));
    auto const head_dim = pe_dim + nope_dim;

    TORCH_CHECK(head_dim == 128, "head_dim (pe_dim + nope_dim) must be 128, got ", head_dim);

    // M = product of all dimensions except the last (handles 2D, 3D, 4D, etc.)
    auto const pe_M = pe.numel() / pe_dim;
    auto const nope_M = nope.numel() / nope_dim;
    TORCH_CHECK(pe_M == nope_M, "pe and nope must have same number of rows. pe: ", pe_M, ", nope: ", nope_M);
    auto const M = static_cast<int32_t>(pe_M);

    // Extract row strides — stride of the second-to-last dimension.
    // For contiguous [..., D], stride(-2) == D. For non-contiguous views from
    // split(), stride(-2) may be larger (e.g. head_dim).
    auto const pe_row_stride = static_cast<int32_t>(pe.stride(-2));
    auto const nope_row_stride = static_cast<int32_t>(nope.stride(-2));

    // Allocate output tensors.
    //   packed: [M, head_dim / 2] int8 (two E2M1 codes per byte)
    //   scale:  [M, 1] int32 (four UE8M0 exponents packed little-endian)
    at::Tensor packed_out
        = at::detail::empty_cuda({M, head_dim / 2}, at::ScalarType::Char, pe.device(), /* stride */ std::nullopt);
    at::Tensor scale_out = at::detail::empty_cuda({M, 1}, at::ScalarType::Int, pe.device(), /* stride */ std::nullopt);

    auto stream = at::cuda::getCurrentCUDAStream(pe.get_device());

    tensorrt_llm::kernels::invokeFusedCatFp4(reinterpret_cast<int8_t*>(packed_out.data_ptr()),
        reinterpret_cast<int32_t*>(scale_out.data_ptr()), reinterpret_cast<__nv_bfloat16 const*>(pe.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(nope.data_ptr()), M, pe_dim, nope_dim, head_dim, pe_row_stride,
        nope_row_stride, stream);

    return {packed_out, scale_out};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fused_cat_fp4(Tensor pe, Tensor nope) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_cat_fp4", &tensorrt_llm::torch_ext::fused_cat_fp4);
}
