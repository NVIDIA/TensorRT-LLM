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

#include "minimax_m3/csrc/m31/dsparkMarkovBias.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

bool isIndex(torch::Tensor const& t)
{
    return t.scalar_type() == torch::kInt32 || t.scalar_type() == torch::kInt64;
}

} // namespace

//! DSPARK_MARKOV_BIAS_CPP (csrc/m31/dsparkMarkovBias.h): the DSpark Markov head's previous-token
//! embeddings. first_prev: int32 / int64 [B]; tokens: int32 / int64 [B, K]; w1: bf16 [V, R] (R a power of two in
//! [16, 1024]); out: bf16 [B * K, R] (written), row b * K + k = w1[prev] or +0.0 for prev outside [0, V).
void minimaxM31DsparkPrevEmbeddings(
    torch::Tensor const& firstPrev, torch::Tensor const& tokens, torch::Tensor const& w1, torch::Tensor& out)
{
    TORCH_CHECK(w1.is_cuda() && w1.scalar_type() == torch::kBFloat16 && w1.dim() == 2 && w1.is_contiguous(),
        "minimax_m31_dspark_prev_embeddings: w1 must be a contiguous bf16 [V, R] CUDA tensor");
    TORCH_CHECK(tokens.device() == w1.device() && isIndex(tokens) && tokens.dim() == 2 && tokens.is_contiguous(),
        "minimax_m31_dspark_prev_embeddings: tokens must be a contiguous int32 / int64 [B, K] tensor on w1's device");
    int64_t const B = tokens.size(0);
    int64_t const K = tokens.size(1);
    int64_t const V = w1.size(0);
    int64_t const R = w1.size(1);
    TORCH_CHECK(firstPrev.device() == w1.device() && isIndex(firstPrev) && firstPrev.dim() == 1
            && firstPrev.size(0) == B && firstPrev.is_contiguous(),
        "minimax_m31_dspark_prev_embeddings: first_prev must be a contiguous int32 / int64 [B] tensor on w1's device");
    TORCH_CHECK(B >= 1 && K >= 1 && V >= 1 && R >= 16 && (R & (R - 1)) == 0 && R <= 1024
            && B * K <= std::numeric_limits<int32_t>::max(),
        "minimax_m31_dspark_prev_embeddings: invalid sizes");
    TORCH_CHECK(out.device() == w1.device() && out.scalar_type() == torch::kBFloat16 && out.is_contiguous()
            && out.numel() == B * K * R,
        "minimax_m31_dspark_prev_embeddings: out must be a contiguous bf16 [B * K, R] tensor on w1's device");
    at::cuda::CUDAGuard const guard(w1.device());
    TRTLLM_MV2_NS::kernels::minimaxM31::invokeDsparkPrevEmbeddings(firstPrev.data_ptr(),
        firstPrev.scalar_type() == torch::kInt64, tokens.data_ptr(), tokens.scalar_type() == torch::kInt64,
        w1.data_ptr(), out.data_ptr(), static_cast<int32_t>(B), static_cast<int32_t>(K), V, static_cast<int32_t>(R),
        at::cuda::getCurrentCUDAStream(w1.get_device()));
}

//! DSPARK_MARKOV_BIAS_CPP: out = logits.float() + bias.float() (fp32 IEEE add of two exactly widened bf16).
//! logits: bf16 [B, K, N] whose B * K rows start logits_row_stride elements apart with unit column stride (a
//! contiguous tensor, or the first N columns of a padded row buffer); bias: contiguous bf16 [B, K, N];
//! out: contiguous fp32 [B, K, N] (written).
void minimaxM31DsparkWidenAdd(
    torch::Tensor const& logits, int64_t logitsRowStride, torch::Tensor const& bias, torch::Tensor& out)
{
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kBFloat16 && logits.dim() == 3,
        "minimax_m31_dspark_widen_add: logits must be a bf16 [B, K, N] CUDA tensor");
    int64_t const B = logits.size(0);
    int64_t const K = logits.size(1);
    int64_t const N = logits.size(2);
    int64_t const M = B * K;
    TORCH_CHECK(M >= 1 && N >= 1 && M <= std::numeric_limits<int32_t>::max() && (N + 1023) / 1024 <= 65535,
        "minimax_m31_dspark_widen_add: invalid sizes");
    TORCH_CHECK(logitsRowStride >= N && (logits.stride(2) == 1 || N == 1)
            && (K == 1 || logits.stride(1) == logitsRowStride) && (B == 1 || logits.stride(0) == K * logitsRowStride),
        "minimax_m31_dspark_widen_add: the logits rows must be logits_row_stride apart with unit column stride");
    TORCH_CHECK(bias.device() == logits.device() && bias.scalar_type() == torch::kBFloat16 && bias.is_contiguous()
            && bias.numel() == M * N,
        "minimax_m31_dspark_widen_add: bias must be a contiguous bf16 tensor with the logits' element count");
    TORCH_CHECK(out.device() == logits.device() && out.scalar_type() == torch::kFloat32 && out.is_contiguous()
            && out.numel() == M * N,
        "minimax_m31_dspark_widen_add: out must be a contiguous fp32 tensor with the logits' element count");
    at::cuda::CUDAGuard const guard(logits.device());
    TRTLLM_MV2_NS::kernels::minimaxM31::invokeDsparkWidenAdd(logits.data_ptr(), logitsRowStride, bias.data_ptr(),
        out.data_ptr<float>(), M, N, at::cuda::getCurrentCUDAStream(logits.get_device()));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def("minimax_m31_dspark_prev_embeddings(Tensor first_prev, Tensor tokens, Tensor w1, Tensor(a!) out) -> ()");
    m.def("minimax_m31_dspark_widen_add(Tensor logits, int logits_row_stride, Tensor bias, Tensor(a!) out) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_dspark_prev_embeddings", &TRTLLM_MV2_NS::torch_ext::minimaxM31DsparkPrevEmbeddings);
    m.impl("minimax_m31_dspark_widen_add", &TRTLLM_MV2_NS::torch_ext::minimaxM31DsparkWidenAdd);
}
