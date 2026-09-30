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

#include <algorithm>
#include <cmath>
#include <limits>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Fused QK Norm + RoPE for Diffusion Transformers.
// Supports per-head and single-rank full-dim normalization; TP full-dim
// normalization uses the prepare/all-reduce/apply operations below.
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

namespace
{

void checkTpPackedQkv(
    torch::Tensor const& qkv, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_dim)
{
    CHECK_INPUT(qkv, torch::kBFloat16);
    TORCH_CHECK(qkv.dim() == 2, "qkv must be 2D [num_tokens, (Hq+Hk+Hv)*head_dim]");
    TORCH_CHECK(num_heads_q > 0, "num_heads_q must be positive, got ", num_heads_q);
    TORCH_CHECK(num_heads_k > 0, "num_heads_k must be positive, got ", num_heads_k);
    TORCH_CHECK(num_heads_v >= 0, "num_heads_v must be non-negative, got ", num_heads_v);
    TORCH_CHECK(num_heads_q <= std::numeric_limits<int>::max() && num_heads_k <= std::numeric_limits<int>::max()
            && num_heads_v <= std::numeric_limits<int>::max(),
        "local head counts exceed the CUDA kernel limit");
    TORCH_CHECK(head_dim == 64 || head_dim == 128, "head_dim must be 64 or 128, got ", head_dim);
    TORCH_CHECK(num_heads_q + num_heads_k <= 65535,
        "num_heads_q + num_heads_k exceeds the CUDA grid-y limit: ", num_heads_q + num_heads_k);
    int64_t const total_heads = num_heads_q + num_heads_k + num_heads_v;
    TORCH_CHECK(
        total_heads > 0 && total_heads <= std::numeric_limits<int>::max() / head_dim, "packed QKV row is too large");
    TORCH_CHECK(qkv.size(1) == total_heads * head_dim,
        "qkv.size(1) must be (num_heads_q + num_heads_k + "
        "num_heads_v) * head_dim; expected ",
        total_heads * head_dim, ", got ", qkv.size(1));
    TORCH_CHECK(qkv.size(0) <= std::numeric_limits<int>::max(), "num_tokens exceeds the CUDA kernel limit");
}

void checkSameCudaDevice(torch::Tensor const& tensor, torch::Tensor const& qkv, char const* name)
{
    TORCH_CHECK(tensor.get_device() == qkv.get_device(), name, " must be on the same CUDA device as qkv");
}

} // namespace

// Stage one of TP full-dim Q/K RMSNorm. The returned FP32 tensor is intended
// to be SUM-all-reduced by Python before being passed to the apply operation.
torch::Tensor fused_dit_qk_norm_rope_tp_prepare(
    torch::Tensor const& qkv, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_dim)
{
    checkTpPackedQkv(qkv, num_heads_q, num_heads_k, num_heads_v, head_dim);
    auto local_sums = torch::empty({qkv.size(0), 2}, qkv.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
    tensorrt_llm::kernels::launchDiTQKNormRopeFullDimTpPrepare(qkv.data_ptr(), local_sums.data_ptr<float>(),
        static_cast<int>(qkv.size(0)), static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
        static_cast<int>(num_heads_v), static_cast<int>(head_dim), stream);
    return local_sums;
}

// Shape/dtype inference for torch.compile. The CUDA implementation performs
// the actual local reduction.
torch::Tensor fused_dit_qk_norm_rope_tp_prepare_meta(torch::Tensor const& qkv, int64_t /*num_heads_q*/,
    int64_t /*num_heads_k*/, int64_t /*num_heads_v*/, int64_t /*head_dim*/)
{
    return torch::empty({qkv.size(0), 2}, qkv.options().dtype(torch::kFloat32));
}

// Stage two of TP full-dim Q/K RMSNorm. global_sums must contain the result of
// a SUM all-reduce over the tensors returned by the prepare operation.
void fused_dit_qk_norm_rope_tp_apply(torch::Tensor& qkv, torch::Tensor const& global_sums, int64_t num_heads_q,
    int64_t num_heads_k, int64_t num_heads_v, int64_t head_dim, int64_t global_q_hidden_size,
    int64_t global_k_hidden_size, double eps, torch::Tensor const& q_weight, torch::Tensor const& k_weight,
    torch::Tensor const& cos_emb, torch::Tensor const& sin_emb, bool interleave)
{
    checkTpPackedQkv(qkv, num_heads_q, num_heads_k, num_heads_v, head_dim);
    CHECK_INPUT(global_sums, torch::kFloat32);
    CHECK_INPUT(q_weight, torch::kBFloat16);
    CHECK_INPUT(k_weight, torch::kBFloat16);
    checkSameCudaDevice(global_sums, qkv, "global_sums");
    checkSameCudaDevice(q_weight, qkv, "q_weight");
    checkSameCudaDevice(k_weight, qkv, "k_weight");
    TORCH_CHECK(global_sums.dim() == 2 && global_sums.size(0) == qkv.size(0) && global_sums.size(1) == 2,
        "global_sums must have shape [num_tokens, 2]; expected [", qkv.size(0), ", 2], got ", global_sums.sizes());
    int64_t const local_q_hidden = num_heads_q * head_dim;
    int64_t const local_k_hidden = num_heads_k * head_dim;
    TORCH_CHECK(q_weight.dim() == 1 && q_weight.numel() == local_q_hidden,
        "q_weight must be 1D with local Q hidden size ", local_q_hidden, "; got ", q_weight.sizes());
    TORCH_CHECK(k_weight.dim() == 1 && k_weight.numel() == local_k_hidden,
        "k_weight must be 1D with local K hidden size ", local_k_hidden, "; got ", k_weight.sizes());
    TORCH_CHECK(global_q_hidden_size >= local_q_hidden && global_q_hidden_size % head_dim == 0,
        "global_q_hidden_size must be a multiple of head_dim and at least the local Q hidden size; got ",
        global_q_hidden_size);
    TORCH_CHECK(global_k_hidden_size >= local_k_hidden && global_k_hidden_size % head_dim == 0,
        "global_k_hidden_size must be a multiple of head_dim and at least the local K hidden size; got ",
        global_k_hidden_size);
    TORCH_CHECK(global_q_hidden_size <= std::numeric_limits<int>::max()
            && global_k_hidden_size <= std::numeric_limits<int>::max(),
        "global hidden sizes exceed the CUDA kernel limit");
    TORCH_CHECK(std::isfinite(eps) && eps >= 0.0 && eps <= std::numeric_limits<float>::max(),
        "eps must be finite, non-negative, and representable as float, got ", eps);

    TORCH_CHECK(cos_emb.dim() >= 2 && cos_emb.dim() <= 4, "cos_emb must have rank in [2, 4], got ", cos_emb.dim());
    TORCH_CHECK(sin_emb.sizes() == cos_emb.sizes(),
        "sin_emb shape must exactly match cos_emb; got cos=", cos_emb.sizes(), " sin=", sin_emb.sizes());
    auto const cos_dtype = cos_emb.scalar_type();
    TORCH_CHECK(cos_dtype == torch::kFloat32 || cos_dtype == torch::kBFloat16,
        "cos_emb must be float32 or bfloat16, got ", cos_dtype);
    TORCH_CHECK(sin_emb.scalar_type() == cos_dtype, "sin_emb dtype must match cos_emb dtype");

    // Shared layouts end in D. Per-head layouts may be flattened (..., H*D)
    // or explicit (..., H, D). H is max(local Hq, local Hk), allowing either
    // side to have more local heads while using one frequency table.
    int64_t const cos_heads = std::max(num_heads_q, num_heads_k);
    bool const explicit_per_head = cos_emb.dim() >= 3 && cos_emb.size(-1) == head_dim && cos_emb.size(-2) == cos_heads;
    int64_t const flattened_width = explicit_per_head ? cos_heads * head_dim : cos_emb.size(-1);
    TORCH_CHECK(flattened_width == head_dim || flattened_width == cos_heads * head_dim,
        "cos_emb must be shared (..., head_dim) or per-head (..., ", cos_heads, ", head_dim)/(..., ",
        cos_heads * head_dim, "); got ", cos_emb.sizes());
    bool const per_head_cos = flattened_width != head_dim;
    torch::Tensor cos_2d = cos_emb.reshape({-1, flattened_width}).contiguous();
    torch::Tensor sin_2d = sin_emb.reshape({-1, flattened_width}).contiguous();
    CHECK_INPUT(cos_2d, cos_dtype);
    CHECK_INPUT(sin_2d, cos_dtype);
    checkSameCudaDevice(cos_2d, qkv, "cos_emb");
    checkSameCudaDevice(sin_2d, qkv, "sin_emb");
    int64_t const cos_rows = cos_2d.size(0);
    TORCH_CHECK(cos_rows > 0, "cos_emb must contain at least one row");
    TORCH_CHECK(cos_rows <= std::numeric_limits<int>::max(), "cos_emb row count exceeds the CUDA kernel limit");
    int cos_seq_per_batch = 0;
    if (cos_rows != qkv.size(0))
    {
        TORCH_CHECK(qkv.size(0) % cos_rows == 0,
            "cos_emb row count must equal num_tokens or evenly divide it "
            "for batched broadcasting; got rows=",
            cos_rows, " num_tokens=", qkv.size(0));
        cos_seq_per_batch = static_cast<int>(cos_rows);
    }

    auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
    tensorrt_llm::kernels::launchDiTQKNormRopeFullDimTpApply(qkv.data_ptr(), global_sums.data_ptr<float>(),
        static_cast<int>(qkv.size(0)), static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
        static_cast<int>(num_heads_v), static_cast<int>(head_dim), static_cast<int>(global_q_hidden_size),
        static_cast<int>(global_k_hidden_size), static_cast<float>(eps), q_weight.data_ptr(), k_weight.data_ptr(),
        cos_2d.data_ptr(), sin_2d.data_ptr(), interleave, per_head_cos, cos_dtype == torch::kBFloat16,
        static_cast<int>(cos_heads), cos_seq_per_batch, stream);
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
        "fused_dit_qk_norm_rope_tp_prepare(Tensor qkv, int num_heads_q, int num_heads_k, int num_heads_v, "
        "int head_dim) -> Tensor");
    m.def(
        "fused_dit_qk_norm_rope_tp_apply(Tensor(a!) qkv, Tensor global_sums, int num_heads_q, int num_heads_k, "
        "int num_heads_v, int head_dim, int global_q_hidden_size, int global_k_hidden_size, float eps, "
        "Tensor q_weight, Tensor k_weight, Tensor cos_emb, Tensor sin_emb, bool interleave) -> ()");
}

// Register the CUDA implementation
TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_dit_qk_norm_rope", &fused_dit_qk_norm_rope);
    m.impl("fused_dit_qk_norm_rope_tp_prepare", &fused_dit_qk_norm_rope_tp_prepare);
    m.impl("fused_dit_qk_norm_rope_tp_apply", &fused_dit_qk_norm_rope_tp_apply);
}

TORCH_LIBRARY_IMPL(trtllm, Meta, m)
{
    m.impl("fused_dit_qk_norm_rope_tp_prepare", &fused_dit_qk_norm_rope_tp_prepare_meta);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
