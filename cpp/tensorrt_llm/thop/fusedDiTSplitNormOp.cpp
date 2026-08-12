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

#include "tensorrt_llm/kernels/fusedDiTSplitNormKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Fused full-dim RMSNorm only (no RoPE) for a single Q or K tensor.
// Mirror of fused_dit_split_norm_rope but without the RoPE step -- used by
// LTX-2 paths that need norm-only (e.g. text cross-attn Q where positional
// info is already baked into the text encoder output).
//
// Input must be a contiguous 2D tensor [num_tokens, num_heads * head_dim].
void fused_dit_split_norm(torch::Tensor& tensor, int64_t num_heads, int64_t head_dim, double eps, torch::Tensor& weight)
{
    TORCH_CHECK(tensor.dim() == 2, "tensor must be 2D: [num_tokens, num_heads*head_dim]");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");

    CHECK_INPUT(tensor, torch::kBFloat16);
    CHECK_INPUT(weight, torch::kBFloat16);

    int64_t const num_tokens = tensor.size(0);
    TORCH_CHECK(
        tensor.size(1) == num_heads * head_dim, "tensor inner dim must be num_heads*head_dim; got ", tensor.size(1));
    TORCH_CHECK(weight.size(0) == num_heads * head_dim, "weight must be [num_heads*head_dim] (full-dim norm), got ",
        weight.size(0), " expected ", num_heads * head_dim);

    auto stream = at::cuda::getCurrentCUDAStream(tensor.get_device());

    tensorrt_llm::kernels::launchFusedDiTSplitNormFullDim(tensor.data_ptr(), static_cast<int>(num_tokens),
        static_cast<int>(num_heads), static_cast<int>(head_dim), static_cast<float>(eps), weight.data_ptr(), stream);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fused_dit_split_norm(Tensor(a!) tensor, int num_heads, int head_dim, float eps, Tensor weight) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_dit_split_norm", &fused_dit_split_norm);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
