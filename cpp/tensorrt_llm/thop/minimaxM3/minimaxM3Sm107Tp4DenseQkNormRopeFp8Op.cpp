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

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp4/denseQkNormRopeFp8.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace dqk = tensorrt_llm::kernels::minimaxM3Sm107Tp4::denseQkNormRopeFp8;

//! QK RMSNorm + partial RoPE of a packed BF16 [tokens, (q + k + v) * 128] row of the MiniMax-M3 TP4 dense layers,
//! returned as a new FP8 E4M3 tensor of the same shape (V copy-cast; Q/K rounded to BF16 before E4M3, as stock).
torch::Tensor minimaxM3Sm107Tp4DenseQkNormRopeFp8(torch::Tensor const& qkv, int64_t numHeadsQ, int64_t numHeadsK,
    int64_t numHeadsV, int64_t rotaryDim, double eps, torch::Tensor const& qWeight, torch::Tensor const& kWeight,
    double base, bool isNeox, torch::Tensor const& positionIds, bool useGemma)
{
    TORCH_CHECK(qkv.is_cuda() && qkv.is_contiguous() && qkv.dim() == 2 && qkv.scalar_type() == at::kBFloat16);
    TORCH_CHECK(qkv.size(1) == (numHeadsQ + numHeadsK + numHeadsV) * dqk::kHeadDim);
    TORCH_CHECK(qWeight.scalar_type() == at::kBFloat16 && kWeight.scalar_type() == at::kBFloat16);
    TORCH_CHECK(qWeight.numel() == dqk::kHeadDim && kWeight.numel() == dqk::kHeadDim);
    TORCH_CHECK(positionIds.scalar_type() == at::kInt && positionIds.numel() == qkv.size(0));
    TORCH_CHECK(rotaryDim > 0 && rotaryDim <= dqk::kHeadDim && rotaryDim % 2 == 0, "unsupported rotary_dim");
    TORCH_CHECK(!isNeox || (rotaryDim * 16) % dqk::kHeadDim == 0, "unsupported rotary_dim for the NeoX pairing");
    at::cuda::CUDAGuard guard(qkv.device());
    auto out = torch::empty_like(qkv, qkv.options().dtype(torch::kFloat8_e4m3fn));
    cudaError_t const err
        = dqk::launch(qkv.data_ptr(), out.data_ptr(), static_cast<int>(qkv.size(0)), static_cast<int>(numHeadsQ),
            static_cast<int>(numHeadsK), static_cast<int>(numHeadsV), static_cast<int>(rotaryDim),
            static_cast<float>(eps), qWeight.data_ptr(), kWeight.data_ptr(), static_cast<float>(base), !isNeox,
            positionIds.data_ptr<int>(), useGemma, nullptr, at::cuda::getCurrentCUDAStream(qkv.get_device()));
    TORCH_CHECK(err == cudaSuccess, "dense_qk_norm_rope_fp8 launch: ", cudaGetErrorString(err));
    return out;
}

//! As minimaxM3Sm107Tp4DenseQkNormRopeFp8, and the first numLiveTokens rows' K and V are also stored into the layer's
//! FP8 pool kvBuffers (the HND view [pages, 2, kv heads, tokens per block, 128]) at their int32 cache slots; with
//! qCompact, every row's Q is also stored there compact ([tokens, q heads, 128]).
torch::Tensor minimaxM3Sm107Tp4DenseQkNormRopeFp8Kv(torch::Tensor const& qkv, int64_t numHeadsQ, int64_t numHeadsK,
    int64_t numHeadsV, int64_t rotaryDim, double eps, torch::Tensor const& qWeight, torch::Tensor const& kWeight,
    double base, bool isNeox, torch::Tensor const& positionIds, bool useGemma, torch::Tensor const& kvBuffers,
    torch::Tensor const& outCacheLoc, int64_t numLiveTokens, std::optional<torch::Tensor> const& qCompact)
{
    TORCH_CHECK(qkv.is_cuda() && qkv.is_contiguous() && qkv.dim() == 2 && qkv.scalar_type() == at::kBFloat16);
    TORCH_CHECK(qkv.size(1) == (numHeadsQ + numHeadsK + numHeadsV) * dqk::kHeadDim);
    TORCH_CHECK(numHeadsK == numHeadsV, "the pool keeps one head count for K and V");
    TORCH_CHECK(qWeight.scalar_type() == at::kBFloat16 && kWeight.scalar_type() == at::kBFloat16);
    TORCH_CHECK(qWeight.numel() == dqk::kHeadDim && kWeight.numel() == dqk::kHeadDim);
    TORCH_CHECK(positionIds.scalar_type() == at::kInt && positionIds.numel() == qkv.size(0));
    TORCH_CHECK(rotaryDim > 0 && rotaryDim <= dqk::kHeadDim && rotaryDim % 2 == 0, "unsupported rotary_dim");
    TORCH_CHECK(!isNeox || (rotaryDim * 16) % dqk::kHeadDim == 0, "unsupported rotary_dim for the NeoX pairing");
    TORCH_CHECK(kvBuffers.is_cuda() && kvBuffers.scalar_type() == at::kFloat8_e4m3fn && kvBuffers.dim() == 5);
    TORCH_CHECK(kvBuffers.size(1) == 2 && kvBuffers.size(2) == numHeadsK && kvBuffers.size(4) == dqk::kHeadDim
            && kvBuffers.stride(4) == 1,
        "kv_buffers must be the HND pool view [pages, 2, kv heads, tokens per block, 128]");
    TORCH_CHECK(outCacheLoc.is_cuda() && outCacheLoc.scalar_type() == at::kInt && outCacheLoc.is_contiguous());
    TORCH_CHECK(numLiveTokens >= 0 && numLiveTokens <= qkv.size(0) && numLiveTokens <= outCacheLoc.numel());
    at::cuda::CUDAGuard guard(qkv.device());
    auto out = torch::empty_like(qkv, qkv.options().dtype(torch::kFloat8_e4m3fn));
    dqk::KvCacheView view;
    view.base = static_cast<__nv_fp8_e4m3*>(kvBuffers.data_ptr());
    view.outCacheLoc = outCacheLoc.data_ptr<int>();
    view.pageStride = kvBuffers.stride(0);
    view.kvStride = kvBuffers.stride(1);
    view.headStride = kvBuffers.stride(2);
    view.tokenStride = kvBuffers.stride(3);
    view.tokensPerBlock = static_cast<int>(kvBuffers.size(3));
    view.numLiveTokens = static_cast<int>(numLiveTokens);
    if (qCompact.has_value())
    {
        TORCH_CHECK(qCompact->is_cuda() && qCompact->is_contiguous() && qCompact->scalar_type() == at::kFloat8_e4m3fn
                && qCompact->numel() == qkv.size(0) * numHeadsQ * dqk::kHeadDim,
            "q_compact must be a contiguous FP8 [tokens, q heads, 128] tensor");
        view.qCompact = static_cast<__nv_fp8_e4m3*>(qCompact->data_ptr());
    }
    cudaError_t const err
        = dqk::launch(qkv.data_ptr(), out.data_ptr(), static_cast<int>(qkv.size(0)), static_cast<int>(numHeadsQ),
            static_cast<int>(numHeadsK), static_cast<int>(numHeadsV), static_cast<int>(rotaryDim),
            static_cast<float>(eps), qWeight.data_ptr(), kWeight.data_ptr(), static_cast<float>(base), !isNeox,
            positionIds.data_ptr<int>(), useGemma, &view, at::cuda::getCurrentCUDAStream(qkv.get_device()));
    TORCH_CHECK(err == cudaSuccess, "dense_qk_norm_rope_fp8_kv launch: ", cudaGetErrorString(err));
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_sm107tp4_dense_qk_norm_rope_fp8(Tensor qkv, int num_heads_q, int num_heads_k, int num_heads_v, "
        "int rotary_dim, float eps, Tensor q_weight, Tensor k_weight, float base, bool is_neox, Tensor position_ids, "
        "bool use_gemma) -> Tensor");
    m.def(
        "minimax_m3_sm107tp4_dense_qk_norm_rope_fp8_kv(Tensor qkv, int num_heads_q, int num_heads_k, int num_heads_v, "
        "int rotary_dim, float eps, Tensor q_weight, Tensor k_weight, float base, bool is_neox, Tensor position_ids, "
        "bool use_gemma, Tensor(a!) kv_buffers, Tensor out_cache_loc, int num_live_tokens, Tensor(b!)? q_compact=None) "
        "-> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_sm107tp4_dense_qk_norm_rope_fp8", &tensorrt_llm::torch_ext::minimaxM3Sm107Tp4DenseQkNormRopeFp8);
    m.impl("minimax_m3_sm107tp4_dense_qk_norm_rope_fp8_kv",
        &tensorrt_llm::torch_ext::minimaxM3Sm107Tp4DenseQkNormRopeFp8Kv);
}
