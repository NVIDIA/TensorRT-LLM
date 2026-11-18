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
#include "tensorrt_llm/kernels/minimaxM3/mxfp8LowMGemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! MiniMax-M3 fused MXFP8 activation quantization + low-M MXFP8 GEMM (decode).
//!
//! @param x BF16 [M, K] activation, unit column stride, 16-byte aligned rows (row stride a multiple of 8).
//! @param w E4M3 [N, K] weight, contiguous.
//! @param wsf UE8M0 weight scales in the 128x4 swizzled layout of `mxfp8_quantize(w, True)`.
//! @param warps, clusterK, rowTiles The kernel configuration; see `isMinimaxM3Mxfp8LowMGemmSupported`.
//! @return BF16 [M, N].
torch::Tensor minimaxM3Mxfp8LowMGemm(torch::Tensor const& x, torch::Tensor const& w, torch::Tensor const& wsf,
    int64_t warps, int64_t clusterK, int64_t rowTiles, bool pdl)
{
    int const sm = tensorrt_llm::common::getSMVersion();
    TORCH_CHECK(sm == 100 || sm == 103, "minimax_m3_mxfp8_low_m_gemm requires SM100 or SM103, got SM", sm);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1,
        "x must be a 2-D bf16 CUDA tensor with unit column stride");
    TORCH_CHECK(w.scalar_type() == at::kFloat8_e4m3fn && w.is_contiguous() && w.dim() == 2 && w.size(1) == x.size(1),
        "w must be a contiguous e4m3 [N, K] tensor");
    TORCH_CHECK(wsf.scalar_type() == at::kByte, "wsf must be uint8 UE8M0 scales");
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && x.stride(0) % 8 == 0, "x rows must be 16-byte aligned");
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(wsf.data_ptr()) % 16 == 0,
        "w and wsf must be 16-byte aligned");
    TORCH_CHECK(tensorrt_llm::kernels::isMinimaxM3Mxfp8LowMGemmSupported(static_cast<int32_t>(warps),
                    static_cast<int32_t>(clusterK), static_cast<int32_t>(rowTiles), static_cast<int32_t>(x.size(0))),
        "minimax_m3_mxfp8_low_m_gemm: unsupported configuration (warps ", warps, ", cluster_k ", clusterK,
        ", row_tiles ", rowTiles, ") at M=", x.size(0));
    c10::cuda::CUDAGuard const deviceGuard{x.device()};
    auto out = torch::empty({x.size(0), w.size(0)}, x.options());
    tensorrt_llm::kernels::invokeMinimaxM3Mxfp8LowMGemm(reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()),
        static_cast<int32_t>(x.stride(0)), static_cast<int32_t>(x.size(0)), static_cast<int32_t>(w.size(0)),
        static_cast<int32_t>(x.size(1)), reinterpret_cast<uint8_t const*>(w.data_ptr()),
        reinterpret_cast<uint8_t const*>(wsf.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        static_cast<int32_t>(out.stride(0)), static_cast<int32_t>(warps), static_cast<int32_t>(clusterK),
        static_cast<int32_t>(rowTiles), pdl, at::cuda::getCurrentCUDAStream(x.get_device()));
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_mxfp8_low_m_gemm(Tensor x, Tensor w, Tensor wsf, int warps, int cluster_k, int row_tiles, "
        "bool pdl) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_mxfp8_low_m_gemm", &tensorrt_llm::torch_ext::minimaxM3Mxfp8LowMGemm);
}
