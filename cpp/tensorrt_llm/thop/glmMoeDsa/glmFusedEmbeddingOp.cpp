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

#include "tensorrt_llm/kernels/glmMoeDsa/glmFusedEmbeddingLookup.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Lookup of an unsharded BF16 embedding table in one kernel. Ids outside [0, weight.size(0)) give a zero row,
// the same guard as tensorrt_llm._torch.modules.embedding.pre_comm_embedding_ops.
at::Tensor glmFusedEmbedding(at::Tensor const& input, at::Tensor const& weight)
{
    CHECK_TH_CUDA(input);
    CHECK_TH_CUDA(weight);
    TORCH_CHECK(input.device() == weight.device(), "glm_fused_embedding: input and weight must be on the same device");
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Int || input.scalar_type() == at::ScalarType::Long,
        "glm_fused_embedding expects int32 or int64 ids, got ", input.scalar_type());
    CHECK_TYPE(weight, at::ScalarType::BFloat16);
    CHECK_CONTIGUOUS(weight);
    TORCH_CHECK(weight.dim() == 2, "glm_fused_embedding expects a 2D weight, got ", weight.dim(), "D");

    int64_t const vocabSize = weight.size(0);
    int64_t const hiddenSize = weight.size(1);
    TORCH_CHECK(hiddenSize % 8 == 0, "glm_fused_embedding needs a hidden size divisible by 8, got ", hiddenSize);
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0, "glm_fused_embedding needs a 16-byte aligned weight");

    at::Tensor const ids = input.contiguous();
    std::vector<int64_t> outShape(input.sizes().begin(), input.sizes().end());
    outShape.push_back(hiddenSize);
    at::Tensor const out = at::empty(outShape, weight.options());

    auto const stream = at::cuda::getCurrentCUDAStream(input.get_device());
    tensorrt_llm::kernels::glm_moe_dsa::invokeFusedEmbeddingLookup(static_cast<__nv_bfloat16*>(out.data_ptr()),
        ids.data_ptr(), ids.scalar_type() == at::ScalarType::Long, static_cast<__nv_bfloat16 const*>(weight.data_ptr()),
        ids.numel(), vocabSize, hiddenSize, stream);
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("glm_fused_embedding(Tensor input, Tensor weight) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm_fused_embedding", &tensorrt_llm::torch_ext::glmFusedEmbedding);
}
