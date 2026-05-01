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

// Fused RMSNorm + RoPE for a single Q or K tensor (LTX-2 split design,
// SEPARATE_QKV layout). Input must be a contiguous 2D tensor
// [num_tokens, num_heads * head_dim]. For FUSE_QKV (packed buffer), use
// fused_dit_qk_norm_rope instead.
void fused_dit_split_norm_rope(torch::Tensor& tensor, int64_t num_heads, int64_t head_dim, double eps,
    torch::Tensor& weight, torch::Tensor& cos_emb, torch::Tensor& sin_emb, bool full_dim_norm, bool do_norm,
    bool interleave)
{
    TORCH_CHECK(tensor.dim() == 2, "tensor must be 2D: [num_tokens, num_heads*head_dim]");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(cos_emb.dim() == 2, "cos_emb must be 2D");
    TORCH_CHECK(sin_emb.dim() == 2, "sin_emb must be 2D");

    CHECK_INPUT(tensor, torch::kBFloat16);
    CHECK_INPUT(weight, torch::kBFloat16);
    CHECK_INPUT(cos_emb, torch::kFloat32);
    CHECK_INPUT(sin_emb, torch::kFloat32);

    int64_t const num_tokens = tensor.size(0);
    TORCH_CHECK(
        tensor.size(1) == num_heads * head_dim, "tensor inner dim must be num_heads*head_dim; got ", tensor.size(1));
    TORCH_CHECK(
        cos_emb.size(0) == num_tokens, "cos_emb token count mismatch: got ", cos_emb.size(0), " expected ", num_tokens);
    bool const per_head_cos = (cos_emb.size(1) == num_heads * head_dim);
    TORCH_CHECK(per_head_cos || cos_emb.size(1) == head_dim, "cos_emb last dim must be head_dim (", head_dim,
        ") or num_heads*head_dim (", num_heads * head_dim, "); got ", cos_emb.size(1));
    TORCH_CHECK(
        sin_emb.size(0) == num_tokens && sin_emb.size(1) == cos_emb.size(1), "sin_emb shape must match cos_emb");

    if (full_dim_norm)
    {
        TORCH_CHECK(weight.size(0) == num_heads * head_dim, "full_dim_norm: weight must be [num_heads*head_dim], got ",
            weight.size(0), " expected ", num_heads * head_dim);
    }
    else
    {
        TORCH_CHECK(weight.size(0) == head_dim, "per_head norm: weight must be [head_dim], got ", weight.size(0));
    }

    auto stream = at::cuda::getCurrentCUDAStream(tensor.get_device());

    tensorrt_llm::kernels::launchFusedDiTSplitNormRope(tensor.data_ptr(), static_cast<int>(num_tokens),
        static_cast<int>(num_heads), static_cast<int>(head_dim), static_cast<float>(eps), weight.data_ptr(),
        reinterpret_cast<float const*>(cos_emb.data_ptr()), reinterpret_cast<float const*>(sin_emb.data_ptr()),
        full_dim_norm, do_norm, interleave, per_head_cos, stream);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_dit_split_norm_rope(Tensor(a!) tensor, int num_heads, int head_dim, float eps, "
        "Tensor weight, Tensor cos_emb, Tensor sin_emb, "
        "bool full_dim_norm, bool do_norm, bool interleave) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_dit_split_norm_rope", &fused_dit_split_norm_rope);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
