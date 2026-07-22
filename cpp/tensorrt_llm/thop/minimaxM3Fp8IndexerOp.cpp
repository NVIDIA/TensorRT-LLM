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

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

torch::Tensor minimax_m3_fp8_indexer_qk_norm_rope(torch::Tensor const& qk, torch::Tensor& index_k_cache,
    torch::Tensor const& out_cache_loc, int64_t num_heads_q, int64_t head_dim, int64_t rotary_dim, double eps,
    torch::Tensor const& q_weight, torch::Tensor const& k_weight, double base, torch::Tensor const& position_ids)
{
    TORCH_CHECK(qk.dim() == 2, "Index QK must be [num_tokens, (num_heads_q + 1) * head_dim]");
    TORCH_CHECK(index_k_cache.dim() == 4 && index_k_cache.size(1) == 1,
        "Index-K cache must be HND [num_pages, 1, page_size, head_dim]");
    TORCH_CHECK(out_cache_loc.dim() == 1, "out_cache_loc must be one-dimensional");
    TORCH_CHECK(position_ids.dim() == 1, "position_ids must be one-dimensional");
    TORCH_CHECK(q_weight.dim() == 1 && k_weight.dim() == 1, "Q/K norm weights must be one-dimensional");

    CHECK_INPUT(qk, torch::kBFloat16);
    CHECK_INPUT(out_cache_loc, torch::kInt32);
    CHECK_INPUT(position_ids, torch::kInt32);
    CHECK_INPUT(q_weight, torch::kBFloat16);
    CHECK_INPUT(k_weight, torch::kBFloat16);
    TORCH_CHECK(index_k_cache.is_cuda(), "Index-K cache must be on CUDA");
    TORCH_CHECK(
        index_k_cache.scalar_type() == at::ScalarType::Float8_e4m3fn, "Index-K cache must use torch.float8_e4m3fn");

    int64_t const num_tokens = qk.size(0);
    TORCH_CHECK(qk.size(1) == (num_heads_q + 1) * head_dim, "Index QK width must equal (num_heads_q + 1) * head_dim");
    TORCH_CHECK(index_k_cache.size(3) == head_dim, "Index-K cache head dimension mismatch");
    TORCH_CHECK(index_k_cache.stride(3) == 1 && index_k_cache.stride(2) == head_dim,
        "Index-K cache must have contiguous token rows in HND layout");
    TORCH_CHECK(out_cache_loc.numel() >= num_tokens, "out_cache_loc is shorter than num_tokens");
    TORCH_CHECK(position_ids.numel() == num_tokens, "position_ids length must equal num_tokens");
    TORCH_CHECK(
        q_weight.numel() == head_dim && k_weight.numel() == head_dim, "Q/K norm weight width must equal head_dim");
    TORCH_CHECK(qk.get_device() == index_k_cache.get_device() && qk.get_device() == out_cache_loc.get_device()
            && qk.get_device() == position_ids.get_device() && qk.get_device() == q_weight.get_device()
            && qk.get_device() == k_weight.get_device(),
        "All MiniMax-M3 FP8 indexer tensors must be on the same CUDA device");

    auto q_out = torch::empty({num_tokens, num_heads_q, head_dim}, qk.options().dtype(at::ScalarType::Float8_e4m3fn));
    if (num_tokens == 0)
    {
        return q_out;
    }
    auto stream = at::cuda::getCurrentCUDAStream(qk.get_device());
    tensorrt_llm::kernels::launchMinimaxM3Fp8IndexerQKNormRope(qk.data_ptr(), q_out.data_ptr(),
        index_k_cache.data_ptr(), out_cache_loc.data_ptr<int>(), index_k_cache.stride(0), index_k_cache.stride(2),
        index_k_cache.size(2), num_tokens, num_heads_q, head_dim, rotary_dim, static_cast<float>(eps),
        q_weight.data_ptr(), k_weight.data_ptr(), static_cast<float>(base), position_ids.data_ptr<int>(), stream);
    return q_out;
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
    m.impl("minimax_m3_fp8_indexer_qk_norm_rope", &minimax_m3_fp8_indexer_qk_norm_rope);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
