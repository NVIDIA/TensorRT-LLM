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

#include "tensorrt_llm/kernels/fusedDiTSplitQKNormRopeKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Fused full-dim RMSNorm + RoPE for a single Q or K tensor (DiT SEPARATE_QKV
// layout, e.g. LTX-2 cross-attn). Input must be a contiguous 2D tensor
// [num_tokens, num_heads * head_dim]. For FUSE_QKV (packed buffer) use
// fused_dit_qk_norm_rope instead.
void fused_dit_split_norm_rope(torch::Tensor& tensor, int64_t num_heads, int64_t head_dim, double eps,
    torch::Tensor& weight, torch::Tensor& cos_emb, torch::Tensor& sin_emb, bool interleave)
{
    TORCH_CHECK(tensor.dim() == 2, "tensor must be 2D: [num_tokens, num_heads*head_dim]");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(cos_emb.dim() == 2, "cos_emb must be 2D");
    TORCH_CHECK(sin_emb.dim() == 2, "sin_emb must be 2D");

    CHECK_INPUT(tensor, torch::kBFloat16);
    CHECK_INPUT(weight, torch::kBFloat16);
    // Cos/sin may be fp32 (legacy) or bf16 (B-2: kernel upcasts in registers).
    auto const cos_dtype = cos_emb.scalar_type();
    TORCH_CHECK(cos_dtype == torch::kFloat32 || cos_dtype == torch::kBFloat16,
        "cos_emb dtype must be float32 or bfloat16, got ", cos_dtype);
    TORCH_CHECK(sin_emb.scalar_type() == cos_dtype, "sin_emb dtype must match cos_emb (", sin_emb.scalar_type(), " vs ",
        cos_dtype, ")");
    bool const cos_is_bf16 = (cos_dtype == torch::kBFloat16);
    if (cos_is_bf16)
    {
        CHECK_INPUT(cos_emb, torch::kBFloat16);
        CHECK_INPUT(sin_emb, torch::kBFloat16);
    }
    else
    {
        CHECK_INPUT(cos_emb, torch::kFloat32);
        CHECK_INPUT(sin_emb, torch::kFloat32);
    }

    int64_t const num_tokens = tensor.size(0);
    TORCH_CHECK(
        tensor.size(1) == num_heads * head_dim, "tensor inner dim must be num_heads*head_dim; got ", tensor.size(1));
    // Auto-detect broadcast: cos may carry one row per token (num_tokens) or one row
    // per token-in-batch (num_tokens / B); in the latter case the kernel broadcasts
    // cos across B via cos_tokenIdx = tokenIdx % cos_seq_per_batch.
    int64_t const cos_rows = cos_emb.size(0);
    int cos_seq_per_batch = 0;
    if (cos_rows != num_tokens)
    {
        TORCH_CHECK(cos_rows > 0 && num_tokens % cos_rows == 0, "cos_emb.size(0) (", cos_rows,
            ") must equal num_tokens (", num_tokens, ") or evenly divide it (broadcast); got non-divisor count");
        cos_seq_per_batch = static_cast<int>(cos_rows);
    }
    bool const per_head_cos = (cos_emb.size(1) == num_heads * head_dim);
    TORCH_CHECK(per_head_cos || cos_emb.size(1) == head_dim, "cos_emb last dim must be head_dim (", head_dim,
        ") or num_heads*head_dim (", num_heads * head_dim, "); got ", cos_emb.size(1));
    TORCH_CHECK(sin_emb.size(0) == cos_rows && sin_emb.size(1) == cos_emb.size(1), "sin_emb shape must match cos_emb");
    TORCH_CHECK(weight.size(0) == num_heads * head_dim, "weight must be [num_heads*head_dim] (full-dim norm), got ",
        weight.size(0), " expected ", num_heads * head_dim);

    auto stream = at::cuda::getCurrentCUDAStream(tensor.get_device());

    tensorrt_llm::kernels::launchFusedDiTSplitNormFullDimRope(tensor.data_ptr(), static_cast<int>(num_tokens),
        static_cast<int>(num_heads), static_cast<int>(head_dim), static_cast<float>(eps), weight.data_ptr(),
        cos_emb.data_ptr(), sin_emb.data_ptr(), interleave, per_head_cos, cos_is_bf16, cos_seq_per_batch, stream);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_dit_split_norm_rope(Tensor(a!) tensor, int num_heads, int head_dim, float eps, "
        "Tensor weight, Tensor cos_emb, Tensor sin_emb, bool interleave) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_dit_split_norm_rope", &fused_dit_split_norm_rope);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
