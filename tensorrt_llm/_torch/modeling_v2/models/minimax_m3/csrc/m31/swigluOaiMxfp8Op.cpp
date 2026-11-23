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

#include "minimax_m3/csrc/m31/swigluOaiMxfp8DecodeSm107.h"
#include "minimax_m3/csrc/m31/swigluOaiMxfp8Sm103.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

//! SWIGLU_MXFP8 (see csrc/m31/swigluOaiMxfp8Sm103.h): x bf16 [rows, 2d] -> q E4M3 [rows, d] and sf uint8
//! [pad128(rows) * d / 32] (swizzled K32 UE8M0), bitwise mxfp8_quantize(swiglu_oai(x), True); `h` (bf16 [rows, d]),
//! when given, also receives the activation. PDL as quantize_with_block_size (TRTLLM_ENABLE_PDL).
void minimaxM31SwigluOaiMxfp8(torch::Tensor const& x, torch::Tensor& q, torch::Tensor& sf,
    std::optional<torch::Tensor> const& h, bool rowsForm, int64_t kRows)
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
    auto const err = TRTLLM_MV2_NS::kernels::minimaxM31::invokeSwigluOaiMxfp8(
        reinterpret_cast<uint4 const*>(x.data_ptr()), reinterpret_cast<uint64_t*>(q.data_ptr()),
        reinterpret_cast<uint32_t*>(sf.data_ptr()), h.has_value() ? reinterpret_cast<uint4*>(h->data_ptr()) : nullptr,
        rows, d, multiProcessorCount, tensorrt_llm::common::getEnvEnablePDL(),
        at::cuda::getCurrentCUDAStream(x.get_device()), rowsForm, static_cast<int>(kRows));
    TORCH_CHECK(err == cudaSuccess, "minimax_m31_swiglu_oai_mxfp8 launch failed: ", cudaGetErrorString(err));
}

//! SWIGLU_MXFP8 decode (see csrc/m31/swigluOaiMxfp8DecodeSm107.h): x bf16 [rows, 2d] (rows may be strided,
//! columns contiguous) -> q E4M3 [rows, d] and sf uint8 [pad128(rows) * pad4(d / 32)] (swizzled K32 UE8M0).
//! finite_code0 (default): bitwise the Triton m3_swq (a scale-code-0 block scaled by 2^127); false: bitwise the
//! in-tree silu_and_mul + mxfp8_quantize(h, True) pair (a scale-code-0 block is zero data). d a multiple of 32, at most
//! 8192. PDL as TRTLLM_ENABLE_PDL.
void minimaxM31SwigluOaiMxfp8Decode(torch::Tensor const& x, torch::Tensor& q, torch::Tensor& sf, bool finiteCode0)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.stride(1) == 1,
        "minimax_m31_swiglu_oai_mxfp8_decode: x must be a 2-D bfloat16 CUDA tensor with contiguous rows");
    int64_t const rows = x.size(0);
    int64_t const d = x.size(1) / 2;
    TORCH_CHECK(x.size(1) % 2 == 0 && d > 0 && d % 32 == 0 && d <= 8192,
        "minimax_m31_swiglu_oai_mxfp8_decode: the half width must be a positive multiple of 32, at most 8192");
    TORCH_CHECK(rows <= 1 || (x.stride(0) >= 2 * d && x.stride(0) % 8 == 0),
        "minimax_m31_swiglu_oai_mxfp8_decode: x's row pitch must be a multiple of 8 elements, at least 2 * d");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
        "minimax_m31_swiglu_oai_mxfp8_decode: x must be 16-byte aligned");
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kFloat8_e4m3fn && q.dim() == 2 && q.is_contiguous()
            && q.size(0) == rows && q.size(1) == d && reinterpret_cast<uintptr_t>(q.data_ptr()) % 8 == 0,
        "minimax_m31_swiglu_oai_mxfp8_decode: q must be a contiguous, 8-byte aligned float8_e4m3fn [rows, d]");
    TORCH_CHECK(q.device() == x.device() && sf.device() == x.device(),
        "minimax_m31_swiglu_oai_mxfp8_decode: x, q and sf must be on the same device");
    int64_t const paddedRows = (rows + 127) / 128 * 128;
    int64_t const nbPad = (d / 32 + 3) / 4 * 4;
    TORCH_CHECK(
        sf.is_cuda() && sf.scalar_type() == torch::kUInt8 && sf.is_contiguous() && sf.numel() == paddedRows * nbPad,
        "minimax_m31_swiglu_oai_mxfp8_decode: sf must be a contiguous uint8 tensor of pad128(rows) * pad4(d / 32) "
        "bytes");
    if (rows == 0)
    {
        return;
    }
    c10::cuda::CUDAGuard const deviceGuard{x.device()};
    TORCH_CHECK(at::cuda::getDeviceProperties(x.get_device())->major >= 10,
        "minimax_m31_swiglu_oai_mxfp8_decode: requires an SM100-family or newer GPU");
    int64_t const pitch = rows > 1 ? x.stride(0) : 2 * d;
    auto const err = TRTLLM_MV2_NS::kernels::minimaxM31::invokeSwigluOaiMxfp8Decode(x.data_ptr(), pitch, q.data_ptr(),
        sf.data_ptr<uint8_t>(), rows, d, finiteCode0, tensorrt_llm::common::getEnvEnablePDL(),
        at::cuda::getCurrentCUDAStream(x.get_device()));
    TORCH_CHECK(err == cudaSuccess, "minimax_m31_swiglu_oai_mxfp8_decode launch failed: ", cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_swiglu_oai_mxfp8(Tensor x, Tensor(a!) q, Tensor(b!) sf, Tensor(c!)? h, bool rows_form=True, "
        "int k_rows=4) -> ()");
    m.def("minimax_m31_swiglu_oai_mxfp8_decode(Tensor x, Tensor(a!) q, Tensor(b!) sf, bool finite_code0=True) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_swiglu_oai_mxfp8", &TRTLLM_MV2_NS::torch_ext::minimaxM31SwigluOaiMxfp8);
    m.impl("minimax_m31_swiglu_oai_mxfp8_decode", &TRTLLM_MV2_NS::torch_ext::minimaxM31SwigluOaiMxfp8Decode);
}
