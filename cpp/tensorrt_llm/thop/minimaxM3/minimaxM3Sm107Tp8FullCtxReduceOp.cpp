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
#include "tensorrt_llm/kernels/minimaxM3Sm107Tp8/fullCtxReduce.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace fcr = tensorrt_llm::kernels::minimaxM3Sm107Tp8::fullCtxReduce;

//! Merges the PrimTS split-KV decode scratch (partial_o [B, Hkv, S, R, 128] 16-bit, partial_stats [B, Hkv, S, R]
//! fp32 log2-LSE) of the MiniMax-M3 TP8 full-context decode into out (BF16 [B * SQ, Hkv * hR, 128]); with q8 / sf
//! also its MXFP8 copy (E4M3 [>= B * SQ, Hkv * hR * 128] + 128x4-swizzled UE8M0 scales, as mxfp8_quantize(out, True)).
void minimaxM3Sm107Tp8FullCtxReduce(torch::Tensor const& partialO, torch::Tensor const& partialStats,
    torch::Tensor const& seqLensKv, torch::Tensor& out, int64_t seqLenQ, int64_t qTokensPerCta, int64_t tileKv,
    int64_t instsKv, bool causal, bool pdl, std::optional<torch::Tensor> const& q8,
    std::optional<torch::Tensor> const& sf)
{
    TORCH_CHECK(partialO.is_cuda() && partialO.is_contiguous() && partialO.dim() == 5);
    TORCH_CHECK(partialO.scalar_type() == at::kHalf || partialO.scalar_type() == at::kBFloat16);
    TORCH_CHECK(partialO.size(4) == fcr::kHeadDim && partialO.size(2) <= fcr::kMaxSplits);
    TORCH_CHECK(partialStats.scalar_type() == at::kFloat && partialStats.is_contiguous()
        && partialStats.sizes() == partialO.sizes().slice(0, 4));
    TORCH_CHECK(seqLensKv.scalar_type() == at::kInt && seqLensKv.numel() >= partialO.size(0));
    int const rows = static_cast<int>(partialO.size(3));
    TORCH_CHECK(seqLenQ >= 1 && rows % seqLenQ == 0);
    int const headsPerKv = rows / static_cast<int>(seqLenQ);
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous()
        && out.numel() == partialO.size(0) * seqLenQ * partialO.size(1) * headsPerKv * fcr::kHeadDim);
    c10::cuda::CUDAGuard const deviceGuard{partialO.device()};
    fcr::Params params{partialO.data_ptr(), partialStats.data_ptr<float>(), seqLensKv.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(partialO.size(0)),
        static_cast<int>(partialO.size(1)), static_cast<int>(partialO.size(2)), rows, static_cast<int>(seqLenQ),
        headsPerKv, static_cast<int>(qTokensPerCta), static_cast<int>(tileKv), static_cast<int>(instsKv), causal,
        partialO.scalar_type() == at::kBFloat16};
    TORCH_CHECK(q8.has_value() == sf.has_value());
    if (q8.has_value())
    {
        int64_t const width = partialO.size(1) * headsPerKv * fcr::kHeadDim;
        int64_t const tokens = partialO.size(0) * seqLenQ;
        int64_t const tileRows = (tokens + 127) / 128 * 128;
        TORCH_CHECK(q8->is_cuda() && q8->is_contiguous() && q8->scalar_type() == at::kFloat8_e4m3fn && q8->dim() == 2
            && q8->size(1) == width && q8->size(0) >= tokens);
        TORCH_CHECK(sf->is_cuda() && sf->is_contiguous() && sf->scalar_type() == at::kByte
            && sf->numel() >= tileRows * (width / 32));
        params.q8 = reinterpret_cast<uint8_t*>(q8->data_ptr());
        params.sf = sf->data_ptr<uint8_t>();
        params.numSfKTiles = static_cast<int>(width / 128);
    }
    cudaError_t const err
        = fcr::invokeFullCtxReduce(params, pdl, at::cuda::getCurrentCUDAStream(partialO.get_device()));
    TORCH_CHECK(err == cudaSuccess, "minimax_m3_sm107tp8_fullctx_reduce: ", cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_sm107tp8_fullctx_reduce(Tensor partial_o, Tensor partial_stats, Tensor seq_lens_kv, "
        "Tensor(a!) out, int seq_len_q, int q_tokens_per_cta, int tile_kv, int insts_kv, bool causal, bool pdl, "
        "Tensor(b!)? q8=None, Tensor(c!)? sf=None) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_sm107tp8_fullctx_reduce", &tensorrt_llm::torch_ext::minimaxM3Sm107Tp8FullCtxReduce);
}
