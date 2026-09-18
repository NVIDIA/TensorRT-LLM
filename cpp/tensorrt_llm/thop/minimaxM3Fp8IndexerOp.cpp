/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/minimaxM3Fp8IndexerKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{
constexpr int64_t kMinimaxM3IndexKHeads = 1;
constexpr int64_t kMinimaxM3HeadDim = 128;
constexpr int64_t kMinimaxM3RotaryDim = 64;
} // namespace

//! Normalize and rotate MiniMax-M3 index Q/K, returning Q and caching K as E4M3.
//!
//! `qk` is contiguous CUDA BF16 `[num_tokens, (numHeadsQ + 1) * headDim]`.
//! `indexKCache` is a mutable CUDA E4M3 HND cache
//! `[num_pages, 1, page_size, headDim]`; valid `outCacheLoc` entries address
//! `page * page_size + token`, while malformed negative or out-of-range entries
//! are defensively skipped. `qWeight` and `kWeight` are contiguous CUDA BF16
//! vectors of `headDim` elements, and `positionIds` is contiguous CUDA int32
//! with one entry per token.
//!
//! `numHeadsQ` must be positive, `headDim` must be 128, `rotaryDim` must be 64,
//! and both `eps` and `base` must be finite and positive. The function mutates
//! `indexKCache` in place and returns contiguous E4M3
//! `[num_tokens, numHeadsQ, headDim]` index Q.
torch::Tensor minimaxM3Fp8IndexerQKNormRope(torch::Tensor const& qk, torch::Tensor& indexKCache,
    torch::Tensor const& outCacheLoc, int64_t numHeadsQ, int64_t headDim, int64_t rotaryDim, double eps,
    torch::Tensor const& qWeight, torch::Tensor const& kWeight, double base, torch::Tensor const& positionIds)
{
    TORCH_CHECK(qk.dim() == 2, "Index QK must be [num_tokens, (num_heads_q + 1) * head_dim]");
    TORCH_CHECK(indexKCache.dim() == 4 && indexKCache.size(1) == kMinimaxM3IndexKHeads,
        "Index-K cache must be HND [num_pages, 1, page_size, head_dim]");
    TORCH_CHECK(outCacheLoc.dim() == 1, "out_cache_loc must be one-dimensional");
    TORCH_CHECK(positionIds.dim() == 1, "position_ids must be one-dimensional");
    TORCH_CHECK(qWeight.dim() == 1 && kWeight.dim() == 1, "Q/K norm weights must be one-dimensional");

    CHECK_INPUT(qk, torch::kBFloat16);
    CHECK_INPUT(outCacheLoc, torch::kInt32);
    CHECK_INPUT(positionIds, torch::kInt32);
    CHECK_INPUT(qWeight, torch::kBFloat16);
    CHECK_INPUT(kWeight, torch::kBFloat16);
    TORCH_CHECK(indexKCache.is_cuda(), "Index-K cache must be on CUDA");
    TORCH_CHECK(
        indexKCache.scalar_type() == at::ScalarType::Float8_e4m3fn, "Index-K cache must use torch.float8_e4m3fn");

    int64_t const numTokens = qk.size(0);
    TORCH_CHECK(numHeadsQ > 0, "num_heads_q must be greater than zero");
    TORCH_CHECK(headDim == kMinimaxM3HeadDim, "MiniMax-M3 FP8 indexer requires head_dim=128");
    TORCH_CHECK(rotaryDim == kMinimaxM3RotaryDim, "MiniMax-M3 FP8 indexer requires rotary_dim=64");
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and greater than zero");
    TORCH_CHECK(std::isfinite(base) && base > 0.0, "RoPE base must be finite and greater than zero");
    auto const epsFloat = static_cast<float>(eps);
    auto const baseFloat = static_cast<float>(base);
    TORCH_CHECK(std::isfinite(epsFloat) && epsFloat > 0.0F, "eps must remain finite and positive in float32");
    TORCH_CHECK(std::isfinite(baseFloat) && baseFloat > 0.0F, "RoPE base must remain finite and positive in float32");
    TORCH_CHECK(indexKCache.size(0) > 0, "Index-K cache must contain at least one page");
    TORCH_CHECK(indexKCache.size(2) > 0, "Index-K cache page_size must be greater than zero");
    TORCH_CHECK(numTokens <= std::numeric_limits<int>::max(), "num_tokens exceeds the CUDA kernel's int range");
    TORCH_CHECK(numHeadsQ <= std::numeric_limits<int>::max(), "num_heads_q exceeds the CUDA kernel's int range");
    TORCH_CHECK(indexKCache.size(2) <= std::numeric_limits<int>::max(),
        "Index-K cache page_size exceeds the CUDA kernel's int range");
    TORCH_CHECK(qk.size(1) == (numHeadsQ + 1) * headDim, "Index QK width must equal (num_heads_q + 1) * head_dim");
    TORCH_CHECK(indexKCache.size(3) == headDim, "Index-K cache head dimension mismatch");
    TORCH_CHECK(indexKCache.stride(3) == 1 && indexKCache.stride(2) == headDim,
        "Index-K cache must have contiguous token rows in HND layout");
    TORCH_CHECK(outCacheLoc.numel() >= numTokens, "out_cache_loc is shorter than num_tokens");
    TORCH_CHECK(positionIds.numel() == numTokens, "position_ids length must equal num_tokens");
    TORCH_CHECK(qWeight.numel() == headDim && kWeight.numel() == headDim, "Q/K norm weight width must equal head_dim");
    constexpr uintptr_t kQkAlignment = 8;
    constexpr uintptr_t kCacheAlignment = 4;
    TORCH_CHECK(reinterpret_cast<uintptr_t>(qk.data_ptr()) % kQkAlignment == 0,
        "Index QK must start at an 8-byte-aligned address for vectorized BF16 loads");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(indexKCache.data_ptr()) % kCacheAlignment == 0,
        "Index-K cache must start at a 4-byte-aligned address for packed E4M3 stores");
    TORCH_CHECK(indexKCache.stride(0) % static_cast<int64_t>(kCacheAlignment) == 0,
        "Index-K cache page stride must be a multiple of 4 E4M3 elements for packed stores");
    TORCH_CHECK(qk.get_device() == indexKCache.get_device() && qk.get_device() == outCacheLoc.get_device()
            && qk.get_device() == positionIds.get_device() && qk.get_device() == qWeight.get_device()
            && qk.get_device() == kWeight.get_device(),
        "All MiniMax-M3 FP8 indexer tensors must be on the same CUDA device");

    auto const qOut = torch::empty({numTokens, numHeadsQ, headDim}, qk.options().dtype(at::ScalarType::Float8_e4m3fn));
    if (numTokens == 0)
    {
        return qOut;
    }
    auto const stream = at::cuda::getCurrentCUDAStream(qk.get_device());
    tensorrt_llm::kernels::launchMinimaxM3Fp8IndexerQKNormRope(qk.data_ptr(), qOut.data_ptr(), indexKCache.data_ptr(),
        outCacheLoc.data_ptr<int>(), indexKCache.stride(0), indexKCache.stride(2), indexKCache.size(2),
        indexKCache.size(0), numTokens, numHeadsQ, headDim, rotaryDim, epsFloat, qWeight.data_ptr(), kWeight.data_ptr(),
        baseFloat, positionIds.data_ptr<int>(), stream);
    return qOut;
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_fp8_indexer_qk_norm_rope(Tensor qk, Tensor(a!) index_k_cache, Tensor out_cache_loc, int "
        "num_heads_q, int head_dim, int rotary_dim, float eps, Tensor q_weight, Tensor k_weight, float base, Tensor "
        "position_ids) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_fp8_indexer_qk_norm_rope", &minimaxM3Fp8IndexerQKNormRope);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
