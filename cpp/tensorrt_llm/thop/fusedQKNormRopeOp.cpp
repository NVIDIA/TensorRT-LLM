/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/fusedQKNormRopeKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace torch_ext
{

// Function for fused QK Norm and RoPE
// This operator applies RMS normalization and RoPE to Q and K tensors in a single CUDA kernel.
// The OP performs operations in-place on the input qkv tensor.
void fused_qk_norm_rope(
    torch::Tensor& qkv,          // Combined QKV tensor [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int64_t num_heads_q,         // Number of query heads
    int64_t num_heads_k,         // Number of key heads
    int64_t num_heads_v,         // Number of value heads
    int64_t head_dim,            // Dimension per head
    double eps,                  // Epsilon for RMS normalization
    torch::Tensor& q_weight,     // RMSNorm weights for query [head_dim]
    torch::Tensor& k_weight,     // RMSNorm weights for key [head_dim]
    double base,                 // Base for RoPE computation
    bool is_neox,                // Whether RoPE is applied in Neox style
    torch::Tensor& position_ids, // Position IDs for RoPE [num_tokens]
    // parameters for yarn
    double factor, // factor in rope_scaling in config.json. When it is not 1.0, it means the model is using yarn.
    double low,    // threshold for high frequency
    double high,   // threshold for low frequency
    double attention_factor // attention_factor applied on cos and sin
)
{
    // Input validation
    TORCH_CHECK(qkv.dim() == 2, "QKV tensor must be 2D: [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]");
    TORCH_CHECK(position_ids.dim() == 1, "Position IDs must be 1D: [num_tokens]");
    TORCH_CHECK(q_weight.dim() == 1, "Query weights must be 1D: [head_dim]");
    TORCH_CHECK(k_weight.dim() == 1, "Key weights must be 1D: [head_dim]");
    TORCH_CHECK(q_weight.size(0) == head_dim, "Query weights size must match head dimension");
    TORCH_CHECK(k_weight.size(0) == head_dim, "Key weights size must match head dimension");

    CHECK_INPUT(qkv, torch::kBFloat16);
    CHECK_INPUT(position_ids, torch::kInt32);
    CHECK_INPUT(q_weight, torch::kBFloat16);
    CHECK_INPUT(k_weight, torch::kBFloat16);

    int64_t num_tokens = qkv.size(0);
    TORCH_CHECK(position_ids.size(0) == num_tokens, "Number of tokens in position_ids must match QKV");

    int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
    TORCH_CHECK(
        qkv.size(1) == total_heads * head_dim, "QKV tensor size must match total number of heads and head dimension");

    auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

    tensorrt_llm::kernels::launchFusedQKNormRope(reinterpret_cast<__nv_bfloat16*>(qkv.data_ptr()),
        static_cast<int>(num_tokens), static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
        static_cast<int>(num_heads_v), static_cast<int>(head_dim), static_cast<float>(eps),
        reinterpret_cast<__nv_bfloat16*>(q_weight.data_ptr()), reinterpret_cast<__nv_bfloat16*>(k_weight.data_ptr()),
        static_cast<float>(base),
        !is_neox, // interleave
        reinterpret_cast<int const*>(position_ids.data_ptr()), static_cast<float>(factor), static_cast<float>(low),
        static_cast<float>(high), static_cast<float>(attention_factor), stream);
}

// Register the PyTorch operators
TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_qk_norm_rope(Tensor(a!) qkv, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim, float "
        "eps, Tensor q_weight, Tensor k_weight, float base, bool is_neox, Tensor position_ids, float factor, float "
        "low, float high, float attention_factor) -> ()");
}

// Register the CUDA implementation
TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_qk_norm_rope", &fused_qk_norm_rope);
}

} // namespace torch_ext
