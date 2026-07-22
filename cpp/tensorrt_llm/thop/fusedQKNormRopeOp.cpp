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

TRTLLM_NAMESPACE_BEGIN

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
    int64_t rotary_dim,          // Dimension for RoPE
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
    double attention_factor, // attention_factor applied on cos and sin
    bool is_qk_norm,         // Whether to apply QK norm
    bool use_gemma,          // Whether QK norm uses Gemma-style RMSNorm (scale by (1 + weight))
    bool use_mrope,          // Whether to use interleaved mRoPE position selection
    int64_t mrope_section1,  // mrope_section[1] (height); ignored when use_mrope is false
    int64_t mrope_section2   // mrope_section[2] (width)
)
{
    // Input validation
    TORCH_CHECK(qkv.dim() == 2, "QKV tensor must be 2D: [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]");
    // Plain RoPE: position_ids is 1D [num_tokens]. Interleaved mRoPE: 2D [3, num_tokens].
    TORCH_CHECK(position_ids.dim() == 1 || (position_ids.dim() == 2 && position_ids.size(0) == 3),
        "Position IDs must be 1D [num_tokens] (plain RoPE) or 2D [3, num_tokens] (mRoPE)");
    TORCH_CHECK(!use_mrope || position_ids.dim() == 2, "use_mrope requires 2D [3, num_tokens] position_ids");
    TORCH_CHECK(q_weight.dim() == 1, "Query weights must be 1D: [head_dim]");
    TORCH_CHECK(k_weight.dim() == 1, "Key weights must be 1D: [head_dim]");
    TORCH_CHECK(q_weight.size(0) == head_dim, "Query weights size must match head dimension");
    TORCH_CHECK(k_weight.size(0) == head_dim, "Key weights size must match head dimension");

    CHECK_INPUT(qkv, torch::kBFloat16);
    CHECK_INPUT(position_ids, torch::kInt32);
    CHECK_INPUT(q_weight, torch::kBFloat16);
    CHECK_INPUT(k_weight, torch::kBFloat16);

    int64_t num_tokens = qkv.size(0);
    TORCH_CHECK(position_ids.size(-1) == num_tokens, "Number of tokens in position_ids must match QKV");

    int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
    TORCH_CHECK(
        qkv.size(1) == total_heads * head_dim, "QKV tensor size must match total number of heads and head dimension");

    auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

    tensorrt_llm::kernels::launchFusedQKNormRope(reinterpret_cast<__nv_bfloat16*>(qkv.data_ptr()),
        static_cast<int>(num_tokens), static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
        static_cast<int>(num_heads_v), static_cast<int>(head_dim), static_cast<int>(rotary_dim),
        static_cast<float>(eps), reinterpret_cast<__nv_bfloat16*>(q_weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(k_weight.data_ptr()), static_cast<float>(base),
        !is_neox, // interleave
        reinterpret_cast<int const*>(position_ids.data_ptr()), static_cast<float>(factor), static_cast<float>(low),
        static_cast<float>(high), static_cast<float>(attention_factor), stream, is_qk_norm, use_gemma, use_mrope,
        static_cast<int>(mrope_section1), static_cast<int>(mrope_section2));
}

// Out-of-place FP8 variant of fused_qk_norm_rope.
//
// Reads a BF16 qkv tensor, applies RMSNorm + RoPE to Q/K and copy-casts V, and
// returns a new FP8 (E4M3) tensor of the same shape. This folds the FP8
// activation-quant into the norm+RoPE epilogue so callers (e.g. the MiniMax-M3
// MSA path with an FP8 KV cache) do not need separate q/k/v cast kernels.
torch::Tensor fused_qk_norm_rope_to_fp8(torch::Tensor const& qkv, // [num_tokens, (num_q+num_k+num_v)*head_dim] BF16
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_dim, int64_t rotary_dim, double eps,
    torch::Tensor const& q_weight, torch::Tensor const& k_weight, double base, bool is_neox,
    torch::Tensor const& position_ids, double factor, double low, double high, double attention_factor, bool is_qk_norm,
    bool use_gemma, bool use_mrope, int64_t mrope_section1, int64_t mrope_section2)
{
    TORCH_CHECK(qkv.dim() == 2, "QKV tensor must be 2D: [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]");
    TORCH_CHECK(position_ids.dim() == 1 || (position_ids.dim() == 2 && position_ids.size(0) == 3),
        "Position IDs must be 1D [num_tokens] (plain RoPE) or 2D [3, num_tokens] (mRoPE)");
    TORCH_CHECK(!use_mrope || position_ids.dim() == 2, "use_mrope requires 2D [3, num_tokens] position_ids");
    TORCH_CHECK(q_weight.dim() == 1, "Query weights must be 1D: [head_dim]");
    TORCH_CHECK(k_weight.dim() == 1, "Key weights must be 1D: [head_dim]");
    TORCH_CHECK(q_weight.size(0) == head_dim, "Query weights size must match head dimension");
    TORCH_CHECK(k_weight.size(0) == head_dim, "Key weights size must match head dimension");

    CHECK_INPUT(qkv, torch::kBFloat16);
    CHECK_INPUT(position_ids, torch::kInt32);
    CHECK_INPUT(q_weight, torch::kBFloat16);
    CHECK_INPUT(k_weight, torch::kBFloat16);

    int64_t num_tokens = qkv.size(0);
    TORCH_CHECK(position_ids.size(-1) == num_tokens, "Number of tokens in position_ids must match QKV");

    int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
    TORCH_CHECK(
        qkv.size(1) == total_heads * head_dim, "QKV tensor size must match total number of heads and head dimension");

    auto out = torch::empty({num_tokens, total_heads * head_dim}, qkv.options().dtype(torch::kFloat8_e4m3fn));

    auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

    tensorrt_llm::kernels::launchFusedQKNormRopeOut(qkv.data_ptr(), out.data_ptr(), /*out_fp8=*/true,
        /*process_v=*/true, static_cast<int>(num_tokens), static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
        static_cast<int>(num_heads_v), static_cast<int>(head_dim), static_cast<int>(rotary_dim),
        static_cast<float>(eps), q_weight.data_ptr(), k_weight.data_ptr(), static_cast<float>(base), !is_neox,
        reinterpret_cast<int const*>(position_ids.data_ptr()), static_cast<float>(factor), static_cast<float>(low),
        static_cast<float>(high), static_cast<float>(attention_factor), stream, is_qk_norm, use_gemma, use_mrope,
        static_cast<int>(mrope_section1), static_cast<int>(mrope_section2));

    return out;
}

// Meta (fake) implementation for torch.compile / tracing: only shape+dtype.
torch::Tensor fused_qk_norm_rope_to_fp8_meta(torch::Tensor const& qkv, int64_t num_heads_q, int64_t num_heads_k,
    int64_t num_heads_v, int64_t head_dim, int64_t /*rotary_dim*/, double /*eps*/, torch::Tensor const& /*q_weight*/,
    torch::Tensor const& /*k_weight*/, double /*base*/, bool /*is_neox*/, torch::Tensor const& /*position_ids*/,
    double /*factor*/, double /*low*/, double /*high*/, double /*attention_factor*/, bool /*is_qk_norm*/,
    bool /*use_gemma*/, bool /*use_mrope*/, int64_t /*mrope_section1*/, int64_t /*mrope_section2*/)
{
    int64_t num_tokens = qkv.size(0);
    int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
    return torch::empty({num_tokens, total_heads * head_dim}, qkv.options().dtype(torch::kFloat8_e4m3fn));
}

// Register the PyTorch operators
TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_qk_norm_rope(Tensor(a!) qkv, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim, int "
        "rotary_dim, float "
        "eps, Tensor q_weight, Tensor k_weight, float base, bool is_neox, Tensor position_ids, float factor, float "
        "low, float high, float attention_factor, bool is_qk_norm, bool use_gemma, bool use_mrope, int "
        "mrope_section1, int mrope_section2) -> ()");
    m.def(
        "fused_qk_norm_rope_to_fp8(Tensor qkv, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim, int "
        "rotary_dim, float eps, Tensor q_weight, Tensor k_weight, float base, bool is_neox, Tensor position_ids, float "
        "factor, float low, float high, float attention_factor, bool is_qk_norm, bool use_gemma, bool use_mrope, int "
        "mrope_section1, int mrope_section2) -> Tensor");
}

// Register the CUDA implementation
TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_qk_norm_rope", &fused_qk_norm_rope);
    m.impl("fused_qk_norm_rope_to_fp8", &fused_qk_norm_rope_to_fp8);
}

// Register the Meta implementation (shape/dtype inference for torch.compile).
TORCH_LIBRARY_IMPL(trtllm, Meta, m)
{
    m.impl("fused_qk_norm_rope_to_fp8", &fused_qk_norm_rope_to_fp8_meta);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
