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
#include <limits>
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
    TORCH_CHECK(cos_emb.dim() >= 2 && cos_emb.dim() <= 4, "cos_emb must have rank in [2, 4]; got ", cos_emb.dim());
    TORCH_CHECK(sin_emb.sizes() == cos_emb.sizes(),
        "sin_emb shape must match cos_emb exactly (raw, pre-flatten); got cos=", cos_emb.sizes(),
        " sin=", sin_emb.sizes());

    // Flatten cos/sin to 2D internally. Two supported layouts:
    //   shape (..., num_heads_q, head_dim) → per-head cos, last 2 dims fold together
    //   shape (...,             head_dim)  → shared cos, all leading dims fold
    int64_t const cos_last_raw = cos_emb.size(-1);
    bool const fold_last_two = (cos_emb.dim() >= 3 && cos_last_raw == head_dim && cos_emb.size(-2) == num_heads_q);
    int64_t const cos_new_last = fold_last_two ? num_heads_q * head_dim : cos_last_raw;
    torch::Tensor cos_2d = cos_emb.reshape({-1, cos_new_last}).contiguous();
    torch::Tensor sin_2d = sin_emb.reshape({-1, cos_new_last}).contiguous();

    CHECK_INPUT(qkv, torch::kBFloat16);
    CHECK_INPUT(q_weight, torch::kBFloat16);
    CHECK_INPUT(k_weight, torch::kBFloat16);
    // Cos/sin may be fp32 (per-head FLUX path) or bf16 (B-2 full-dim LTX-2 path).
    // Per-head path requires fp32 (kernel has no bf16 branch); enforced below.
    auto const cos_dtype = cos_2d.scalar_type();
    TORCH_CHECK(cos_dtype == torch::kFloat32 || cos_dtype == torch::kBFloat16,
        "cos_emb dtype must be float32 or bfloat16, got ", cos_dtype);
    TORCH_CHECK(sin_2d.scalar_type() == cos_dtype, "sin_emb dtype must match cos_emb");
    bool const cos_is_bf16 = (cos_dtype == torch::kBFloat16);
    if (cos_is_bf16)
    {
        CHECK_INPUT(cos_2d, torch::kBFloat16);
        CHECK_INPUT(sin_2d, torch::kBFloat16);
    }
    else
    {
        CHECK_INPUT(cos_2d, torch::kFloat32);
        CHECK_INPUT(sin_2d, torch::kFloat32);
    }

    int64_t num_tokens = qkv.size(0);
    int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
    TORCH_CHECK(qkv.size(1) == total_heads * head_dim, "QKV tensor size must match total_heads * head_dim");
    // Auto-detect broadcast: cos rows == num_tokens (flat) or num_tokens / B (broadcast over B).
    int64_t const cos_rows = cos_2d.size(0);
    int cos_seq_per_batch = 0;
    if (cos_rows != num_tokens)
    {
        TORCH_CHECK(cos_rows > 0 && num_tokens % cos_rows == 0, "cos_emb.size(0) (", cos_rows,
            ") must equal num_tokens (", num_tokens, ") or evenly divide it (broadcast); got non-divisor count");
        cos_seq_per_batch = static_cast<int>(cos_rows);
    }
    bool const per_head_cos = (cos_2d.size(1) == num_heads_q * head_dim);
    TORCH_CHECK(per_head_cos || cos_2d.size(1) == head_dim, "cos_emb last dim must be head_dim (", head_dim,
        ") or num_heads_q*head_dim (", num_heads_q * head_dim, "); got ", cos_2d.size(1));
    TORCH_CHECK(sin_2d.size(0) == cos_rows && sin_2d.size(1) == cos_2d.size(1), "sin_emb shape must match cos_emb");

    // Auto-dispatch by weight shape:
    //   weight.size(0) == head_dim                   → per-head norm (FLUX/Cosmos3, original kernel)
    //   weight.size(0) == num_heads_per_side*head_dim → full-dim norm (LTX-2)
    bool const is_full_dim_q = (q_weight.size(0) == num_heads_q * head_dim);
    bool const is_full_dim_k = (k_weight.size(0) == num_heads_k * head_dim);
    bool const is_per_head_q = (q_weight.size(0) == head_dim);
    bool const is_per_head_k = (k_weight.size(0) == head_dim);
    TORCH_CHECK(is_full_dim_q == is_full_dim_k && is_per_head_q == is_per_head_k,
        "q_weight and k_weight must use the same norm mode (both per-head or both full-dim).");
    TORCH_CHECK(is_per_head_q || is_full_dim_q,
        "q_weight size must be [head_dim] (per-head) or [num_heads*head_dim] (full-dim); got ", q_weight.size(0),
        " head_dim=", head_dim, " num_heads_q=", num_heads_q);

    if (is_full_dim_q)
    {
        TORCH_CHECK(!q_add_weight.has_value() && !k_add_weight.has_value(),
            "Full-dim norm does not support dual-stream add_weights");
        TORCH_CHECK(num_txt_tokens <= 0, "Full-dim norm does not support dual-stream (num_txt_tokens must be -1)");
        auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
        tensorrt_llm::kernels::launchFusedDiTQKNormRopeFullDim(qkv.data_ptr(), static_cast<int>(num_tokens),
            static_cast<int>(num_heads_q), static_cast<int>(num_heads_k), static_cast<int>(num_heads_v),
            static_cast<int>(head_dim), static_cast<float>(eps), q_weight.data_ptr(), k_weight.data_ptr(),
            cos_2d.data_ptr(), sin_2d.data_ptr(), interleave, per_head_cos, cos_is_bf16, cos_seq_per_batch, stream);
        return;
    }

    // Per-head path (original FLUX/Cosmos3 kernel) — only fp32 cos supported here.
    // Broadcast over B is now supported by the kernel via cos_seq_per_batch.
    TORCH_CHECK(!cos_is_bf16,
        "Per-head fused_dit_qk_norm_rope (FLUX/Cosmos) requires fp32 cos/sin; bf16 cos is only supported "
        "by the full-dim path (LTX-2)");
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
        k_add_ptr, reinterpret_cast<float const*>(cos_2d.data_ptr()), reinterpret_cast<float const*>(sin_2d.data_ptr()),
        static_cast<int>(num_txt_tokens), interleave, static_cast<int>(tokens_per_batch), cos_seq_per_batch, stream);
}

// Out-of-place FLUX variant that preserves the fused DiT QK norm + RoPE
// numerics and folds three calibrated static E4M3 conversions into the same
// launch. The returned Q/K/V tensors are dense and directly consumable by the
// CUTEDSL attention backend.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_dit_qk_norm_rope_quant_fp8(torch::Tensor const& qkv,
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_dim, double eps,
    torch::Tensor const& q_weight, torch::Tensor const& k_weight, c10::optional<torch::Tensor> q_add_weight,
    c10::optional<torch::Tensor> k_add_weight, torch::Tensor const& q_dequant_scale,
    torch::Tensor const& k_dequant_scale, torch::Tensor const& v_dequant_scale, torch::Tensor const& cos_emb,
    torch::Tensor const& sin_emb, c10::SymInt num_txt_tokens_sym, bool interleave, c10::SymInt tokens_per_batch_sym)
{
    int64_t const num_txt_tokens = num_txt_tokens_sym.guard_int(__FILE__, __LINE__);
    int64_t const tokens_per_batch = tokens_per_batch_sym.guard_int(__FILE__, __LINE__);

    TORCH_CHECK(qkv.dim() == 2, "QKV tensor must be 2D: [num_tokens, total_heads*head_dim]");
    TORCH_CHECK(q_weight.dim() == 1 && q_weight.size(0) == head_dim, "q_weight must be 1D [head_dim]");
    TORCH_CHECK(k_weight.dim() == 1 && k_weight.size(0) == head_dim, "k_weight must be 1D [head_dim]");
    TORCH_CHECK(cos_emb.dim() >= 2 && cos_emb.dim() <= 4, "cos_emb must have rank in [2, 4]; got ", cos_emb.dim());
    TORCH_CHECK(sin_emb.sizes() == cos_emb.sizes(), "sin_emb shape must match cos_emb exactly");
    CHECK_INPUT(qkv, torch::kBFloat16);
    CHECK_INPUT(q_weight, torch::kBFloat16);
    CHECK_INPUT(k_weight, torch::kBFloat16);
    CHECK_INPUT(q_dequant_scale, torch::kFloat32);
    CHECK_INPUT(k_dequant_scale, torch::kFloat32);
    CHECK_INPUT(v_dequant_scale, torch::kFloat32);
    TORCH_CHECK(q_dequant_scale.numel() == 1, "q_dequant_scale must contain one FP32 value");
    TORCH_CHECK(k_dequant_scale.numel() == 1, "k_dequant_scale must contain one FP32 value");
    TORCH_CHECK(v_dequant_scale.numel() == 1, "v_dequant_scale must contain one FP32 value");

    int64_t const num_tokens = qkv.size(0);
    int64_t const total_heads = num_heads_q + num_heads_k + num_heads_v;
    TORCH_CHECK(num_heads_q > 0 && num_heads_k > 0 && num_heads_v > 0, "Q/K/V head counts must be positive");
    TORCH_CHECK(qkv.size(1) == total_heads * head_dim, "QKV tensor size must match total_heads * head_dim");
    TORCH_CHECK(num_tokens <= std::numeric_limits<int>::max(), "num_tokens exceeds the kernel's supported range");
    TORCH_CHECK(
        total_heads * head_dim <= std::numeric_limits<int>::max(), "QKV width exceeds the kernel's supported range");

    int64_t const cos_last_raw = cos_emb.size(-1);
    bool const fold_last_two = cos_emb.dim() >= 3 && cos_last_raw == head_dim && cos_emb.size(-2) == num_heads_q;
    int64_t const cos_new_last = fold_last_two ? num_heads_q * head_dim : cos_last_raw;
    torch::Tensor cos_2d = cos_emb.reshape({-1, cos_new_last}).contiguous();
    torch::Tensor sin_2d = sin_emb.reshape({-1, cos_new_last}).contiguous();
    CHECK_INPUT(cos_2d, torch::kFloat32);
    CHECK_INPUT(sin_2d, torch::kFloat32);
    TORCH_CHECK(cos_2d.size(1) == head_dim,
        "Static FP8 fused DiT path requires per-token, head-shared cos/sin with last dimension head_dim");

    int64_t const cos_rows = cos_2d.size(0);
    int cos_seq_per_batch = 0;
    if (cos_rows != num_tokens)
    {
        TORCH_CHECK(cos_rows > 0 && num_tokens % cos_rows == 0, "cos rows must equal or evenly divide num_tokens");
        cos_seq_per_batch = static_cast<int>(cos_rows);
    }

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

    auto options = qkv.options().dtype(torch::kFloat8_e4m3fn);
    auto q_out = torch::empty({num_tokens, num_heads_q * head_dim}, options);
    auto k_out = torch::empty({num_tokens, num_heads_k * head_dim}, options);
    auto v_out = torch::empty({num_tokens, num_heads_v * head_dim}, options);
    auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
    tensorrt_llm::kernels::launchFusedDiTQKNormRopeQuantFp8(qkv.data_ptr(), q_out.data_ptr(), k_out.data_ptr(),
        v_out.data_ptr(), static_cast<int>(num_tokens), static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
        static_cast<int>(num_heads_v), static_cast<int>(head_dim), static_cast<float>(eps), q_weight.data_ptr(),
        k_weight.data_ptr(), q_add_ptr, k_add_ptr, q_dequant_scale.data_ptr<float>(), k_dequant_scale.data_ptr<float>(),
        v_dequant_scale.data_ptr<float>(), cos_2d.data_ptr<float>(), sin_2d.data_ptr<float>(),
        static_cast<int>(num_txt_tokens), interleave, static_cast<int>(tokens_per_batch), cos_seq_per_batch, stream);
    return {q_out, k_out, v_out};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_dit_qk_norm_rope_quant_fp8_meta(torch::Tensor const& qkv,
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_dim, double /*eps*/,
    torch::Tensor const& /*q_weight*/, torch::Tensor const& /*k_weight*/, c10::optional<torch::Tensor> /*q_add_weight*/,
    c10::optional<torch::Tensor> /*k_add_weight*/, torch::Tensor const& /*q_dequant_scale*/,
    torch::Tensor const& /*k_dequant_scale*/, torch::Tensor const& /*v_dequant_scale*/,
    torch::Tensor const& /*cos_emb*/, torch::Tensor const& /*sin_emb*/, c10::SymInt /*num_txt_tokens*/,
    bool /*interleave*/, c10::SymInt /*tokens_per_batch*/)
{
    auto options = qkv.options().dtype(torch::kFloat8_e4m3fn);
    int64_t const num_tokens = qkv.size(0);
    return {torch::empty({num_tokens, num_heads_q * head_dim}, options),
        torch::empty({num_tokens, num_heads_k * head_dim}, options),
        torch::empty({num_tokens, num_heads_v * head_dim}, options)};
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
    m.def(
        "fused_dit_qk_norm_rope_quant_fp8(Tensor qkv, int num_heads_q, int num_heads_k, int num_heads_v, "
        "int head_dim, float eps, Tensor q_weight, Tensor k_weight, Tensor? q_add_weight, Tensor? k_add_weight, "
        "Tensor q_dequant_scale, Tensor k_dequant_scale, Tensor v_dequant_scale, Tensor cos_emb, Tensor sin_emb, "
        "SymInt num_txt_tokens, bool interleave, SymInt tokens_per_batch) -> (Tensor, Tensor, Tensor)");
}

// Register the CUDA implementation
TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_dit_qk_norm_rope", &fused_dit_qk_norm_rope);
    m.impl("fused_dit_qk_norm_rope_quant_fp8", &fused_dit_qk_norm_rope_quant_fp8);
}

TORCH_LIBRARY_IMPL(trtllm, Meta, m)
{
    m.impl("fused_dit_qk_norm_rope_quant_fp8", &fused_dit_qk_norm_rope_quant_fp8_meta);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
