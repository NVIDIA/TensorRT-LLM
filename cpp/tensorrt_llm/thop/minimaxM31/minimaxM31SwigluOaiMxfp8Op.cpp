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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/minimaxM31/swigluOaiMxfp8Sm103.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! SWIGLU_MXFP8 (see kernels/minimaxM31/swigluOaiMxfp8Sm103.h): x bf16 [rows, 2d] -> q E4M3 [rows, d] and sf uint8
//! [pad128(rows) * d / 32] (swizzled K32 UE8M0), bitwise mxfp8_quantize(swiglu_oai(x), True); `h` (bf16 [rows, d]),
//! when given, also receives the activation. PDL as quantize_with_block_size (TRTLLM_ENABLE_PDL).
void minimaxM31SwigluOaiMxfp8(
    torch::Tensor const& x, torch::Tensor& q, torch::Tensor& sf, std::optional<torch::Tensor> const& h)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(),
        "minimax_m31_swiglu_oai_mxfp8: x must be a contiguous 2-D bfloat16 CUDA tensor");
    int64_t const rows = x.size(0);
    int64_t const d = x.size(1) / 2;
    TORCH_CHECK(x.size(1) % 2 == 0 && d % 256 == 0 && d <= 8192 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
        "minimax_m31_swiglu_oai_mxfp8: the half width must be a multiple of 256 (at most 8192), x 16-byte aligned");
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kFloat8_e4m3fn && q.dim() == 2 && q.is_contiguous()
            && q.size(0) == rows && q.size(1) == d && reinterpret_cast<uintptr_t>(q.data_ptr()) % 8 == 0,
        "minimax_m31_swiglu_oai_mxfp8: q must be a contiguous float8_e4m3fn [rows, d]");
    int64_t const paddedRows = (rows + 127) / 128 * 128;
    TORCH_CHECK(sf.is_cuda() && sf.scalar_type() == torch::kUInt8 && sf.is_contiguous()
            && sf.numel() == paddedRows * d / 32 && reinterpret_cast<uintptr_t>(sf.data_ptr()) % 4 == 0,
        "minimax_m31_swiglu_oai_mxfp8: sf must be a contiguous uint8 tensor of pad128(rows) * d / 32 bytes");
    if (h.has_value())
    {
        TORCH_CHECK(h->is_cuda() && h->scalar_type() == torch::kBFloat16 && h->dim() == 2 && h->is_contiguous()
                && h->size(0) == rows && h->size(1) == d && reinterpret_cast<uintptr_t>(h->data_ptr()) % 16 == 0,
            "minimax_m31_swiglu_oai_mxfp8: h must be a contiguous, 16-byte aligned bfloat16 [rows, d]");
    }
    if (rows == 0)
    {
        return;
    }
    c10::cuda::CUDAGuard const deviceGuard{x.device()};
    thread_local int const multiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
    auto const err
        = tensorrt_llm::kernels::minimaxM31::invokeSwigluOaiMxfp8(reinterpret_cast<uint4 const*>(x.data_ptr()),
            reinterpret_cast<uint64_t*>(q.data_ptr()), reinterpret_cast<uint32_t*>(sf.data_ptr()),
            h.has_value() ? reinterpret_cast<uint4*>(h->data_ptr()) : nullptr, rows, d, multiProcessorCount,
            tensorrt_llm::common::getEnvEnablePDL(), at::cuda::getCurrentCUDAStream(x.get_device()));
    TORCH_CHECK(err == cudaSuccess, "minimax_m31_swiglu_oai_mxfp8 launch failed: ", cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("minimax_m31_swiglu_oai_mxfp8(Tensor x, Tensor(a!) q, Tensor(b!) sf, Tensor(c!)? h) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_swiglu_oai_mxfp8", &tensorrt_llm::torch_ext::minimaxM31SwigluOaiMxfp8);
}
