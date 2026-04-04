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

#include "tensorrt_llm/kernels/fusedDiTQKNormRopeKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Fused per-head QK Norm + RoPE for Diffusion Transformers.
// Only per-head norm is supported (FLUX, Cosmos3, UniVideo).
// Full-dim norm (WAN, LTX2) will be added in a follow-up PR.
void fused_dit_qk_norm_rope(torch::Tensor& qkv, // [num_tokens, (Hq+Hk+Hv)*head_dim]
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_dim, double eps,
    torch::Tensor& q_weight,                    // [head_dim]
    torch::Tensor& k_weight,                    // [head_dim]
    c10::optional<torch::Tensor> q_add_weight,  // [head_dim] or nullopt (dual-stream)
    c10::optional<torch::Tensor> k_add_weight,  // [head_dim] or nullopt
    torch::Tensor& cos_emb,                     // [num_tokens, head_dim], float32
    torch::Tensor& sin_emb,                     // [num_tokens, head_dim], float32
    c10::SymInt num_txt_tokens_sym,             // -1 = no dual-stream
    bool interleave,                            // true = interleaved, false = rotate_half
    c10::SymInt tokens_per_batch_sym)           // seq_len per batch element for dual-stream; 0 = flat
{
    int64_t num_txt_tokens = num_txt_tokens_sym.guard_int(__FILE__, __LINE__);
    int64_t tokens_per_batch = tokens_per_batch_sym.guard_int(__FILE__, __LINE__);
    // Validation
    TORCH_CHECK(qkv.dim() == 2, "QKV tensor must be 2D: [num_tokens, total_heads*head_dim]");
    TORCH_CHECK(q_weight.dim() == 1, "q_weight must be 1D");
    TORCH_CHECK(k_weight.dim() == 1, "k_weight must be 1D");
    TORCH_CHECK(cos_emb.dim() == 2, "cos_emb must be 2D: [num_tokens, head_dim]");
    TORCH_CHECK(sin_emb.dim() == 2, "sin_emb must be 2D: [num_tokens, head_dim]");

    CHECK_INPUT(qkv, torch::kBFloat16);
    CHECK_INPUT(q_weight, torch::kBFloat16);
    CHECK_INPUT(k_weight, torch::kBFloat16);
    CHECK_INPUT(cos_emb, torch::kFloat32);
    CHECK_INPUT(sin_emb, torch::kFloat32);

    int64_t num_tokens = qkv.size(0);
    int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
    TORCH_CHECK(qkv.size(1) == total_heads * head_dim, "QKV tensor size must match total_heads * head_dim");
    TORCH_CHECK(cos_emb.size(0) == num_tokens && cos_emb.size(1) == head_dim, "cos_emb must be [num_tokens, head_dim]");
    TORCH_CHECK(sin_emb.size(0) == num_tokens && sin_emb.size(1) == head_dim, "sin_emb must be [num_tokens, head_dim]");

    // Only per-head norm supported
    TORCH_CHECK(q_weight.size(0) == head_dim,
        "fused_dit_qk_norm_rope only supports per-head norm (q_weight must be [head_dim]). "
        "Full-dim norm (q_weight [num_heads * head_dim]) is not yet supported. Got q_weight size: ",
        q_weight.size(0), ", head_dim: ", head_dim);
    TORCH_CHECK(k_weight.size(0) == head_dim,
        "fused_dit_qk_norm_rope only supports per-head norm (k_weight must be [head_dim]). Got k_weight size: ",
        k_weight.size(0));

    // Validate optional add_weights (dual-stream)
    void const* q_add_ptr = nullptr;
    void const* k_add_ptr = nullptr;
    if (q_add_weight.has_value())
    {
        CHECK_INPUT(q_add_weight.value(), torch::kBFloat16);
        TORCH_CHECK(q_add_weight.value().dim() == 1 && q_add_weight.value().size(0) == head_dim,
            "q_add_weight must be 1D [head_dim]");
        q_add_ptr = q_add_weight.value().data_ptr();
    }
    if (k_add_weight.has_value())
    {
        CHECK_INPUT(k_add_weight.value(), torch::kBFloat16);
        TORCH_CHECK(k_add_weight.value().dim() == 1 && k_add_weight.value().size(0) == head_dim,
            "k_add_weight must be 1D [head_dim]");
        k_add_ptr = k_add_weight.value().data_ptr();
    }

    auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

    tensorrt_llm::kernels::launchFusedDiTQKNormRope(qkv.data_ptr(), static_cast<int>(num_tokens),
        static_cast<int>(num_heads_q), static_cast<int>(num_heads_k), static_cast<int>(num_heads_v),
        static_cast<int>(head_dim), static_cast<float>(eps), q_weight.data_ptr(), k_weight.data_ptr(), q_add_ptr,
        k_add_ptr, reinterpret_cast<float const*>(cos_emb.data_ptr()),
        reinterpret_cast<float const*>(sin_emb.data_ptr()), static_cast<int>(num_txt_tokens), interleave,
        static_cast<int>(tokens_per_batch), stream);
}

// Register the PyTorch operator schema
TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_dit_qk_norm_rope(Tensor(a!) qkv, int num_heads_q, int num_heads_k, int num_heads_v, "
        "int head_dim, float eps, Tensor q_weight, Tensor k_weight, "
        "Tensor? q_add_weight, Tensor? k_add_weight, "
        "Tensor cos_emb, Tensor sin_emb, SymInt num_txt_tokens, "
        "bool interleave, SymInt tokens_per_batch) -> ()");
}

// Register the CUDA implementation
TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_dit_qk_norm_rope", &fused_dit_qk_norm_rope);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
