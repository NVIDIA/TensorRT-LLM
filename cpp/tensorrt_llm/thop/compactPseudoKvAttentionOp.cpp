/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "tensorrt_llm/thop/compactPseudoKvAttentionOp.h"

#include "tensorrt_llm/kernels/sparseAttentionKernels.h"

#include <ATen/cuda/CUDAContext.h>
#include <NvInferRuntime.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

void runCompactPseudoKvAttention(torch::Tensor const& q, torch::Tensor& output,
    std::optional<torch::Tensor> const& compactPseudokvKey, std::optional<torch::Tensor> const& compactPseudokvValue,
    std::optional<torch::Tensor> const& compactPseudokvPositions,
    std::optional<torch::Tensor> const& compactPseudokvCausalMask,
    std::optional<int64_t> const compactPseudokvSourceSeqLen)
{
    TORCH_CHECK(compactPseudokvKey.has_value(), "compact_pseudokv_key is required.");
    TORCH_CHECK(compactPseudokvValue.has_value(), "compact_pseudokv_value is required.");
    TORCH_CHECK(compactPseudokvPositions.has_value(), "compact_pseudokv_positions is required.");
    TORCH_CHECK(compactPseudokvCausalMask.has_value(), "compact_pseudokv_causal_mask is required.");
    TORCH_CHECK(compactPseudokvSourceSeqLen.has_value(), "compact_pseudokv_source_seq_len is required.");

    auto const& compactKey = compactPseudokvKey.value();
    auto const& compactValue = compactPseudokvValue.value();
    auto const& positions = compactPseudokvPositions.value();
    auto const& causalMask = compactPseudokvCausalMask.value();
    TORCH_CHECK(q.dim() == 3, "compact pseudo-KV query must be [query_seq_len, num_heads, head_dim].");
    TORCH_CHECK(compactKey.dim() == 3, "compact_pseudokv_key must be [compact_seq_len, num_heads, head_dim].");
    TORCH_CHECK(compactValue.dim() == 3, "compact_pseudokv_value must be [compact_seq_len, num_heads, head_dim].");
    TORCH_CHECK(compactKey.sizes() == compactValue.sizes(), "compact pseudo-KV key/value shapes must match.");
    TORCH_CHECK(q.size(1) == compactKey.size(1) && q.size(2) == compactKey.size(2),
        "compact pseudo-KV query heads/head_dim must match compact key shape.");
    TORCH_CHECK(positions.dim() == 1 && positions.size(0) == compactKey.size(0),
        "compact_pseudokv_positions must have one entry per compact row.");
    TORCH_CHECK(causalMask.dim() == 2 && causalMask.size(0) == q.size(0) && causalMask.size(1) == compactKey.size(0),
        "compact_pseudokv_causal_mask must be [query_seq_len, compact_seq_len].");
    TORCH_CHECK(causalMask.scalar_type() == torch::kBool, "compact_pseudokv_causal_mask must be bool.");
    TORCH_CHECK(compactPseudokvSourceSeqLen.value() > 0, "compact_pseudokv_source_seq_len must be positive.");

    TORCH_CHECK(q.is_cuda(), "compact pseudo-KV native attention requires CUDA query.");
    TORCH_CHECK(output.is_cuda(), "compact pseudo-KV native attention requires CUDA output.");
    TORCH_CHECK(compactKey.is_cuda(), "compact pseudo-KV native attention requires CUDA key.");
    TORCH_CHECK(compactValue.is_cuda(), "compact pseudo-KV native attention requires CUDA value.");
    TORCH_CHECK(positions.is_cuda(), "compact pseudo-KV native attention requires CUDA positions.");
    TORCH_CHECK(causalMask.is_cuda(), "compact pseudo-KV native attention requires CUDA causal mask.");
    TORCH_CHECK(output.get_device() == q.get_device() && compactKey.get_device() == q.get_device()
            && compactValue.get_device() == q.get_device() && positions.get_device() == q.get_device()
            && causalMask.get_device() == q.get_device(),
        "compact pseudo-KV native attention requires all tensors on the same CUDA device.");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "compact pseudo-KV native attention currently supports FP32.");
    TORCH_CHECK(compactKey.scalar_type() == torch::kFloat32, "compact pseudo-KV key must be FP32.");
    TORCH_CHECK(compactValue.scalar_type() == torch::kFloat32, "compact pseudo-KV value must be FP32.");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "compact pseudo-KV output must be FP32.");
    TORCH_CHECK(positions.scalar_type() == torch::kInt32, "compact_pseudokv_positions must be int32.");
    bool const outputIsRank3 = output.dim() == 3 && output.size(0) == q.size(0) && output.size(1) == q.size(1)
        && output.size(2) == q.size(2);
    bool const outputIsFlattened
        = output.dim() == 2 && output.size(0) == q.size(0) && output.size(1) == q.size(1) * q.size(2);
    TORCH_CHECK(outputIsRank3 || outputIsFlattened,
        "compact pseudo-KV native output must be [query_seq_len, num_heads, head_dim] or "
        "[query_seq_len, num_heads * head_dim].");
    TORCH_CHECK(q.stride(2) == 1, "compact pseudo-KV query head dimension must be contiguous.");
    TORCH_CHECK(compactKey.stride(2) == 1, "compact pseudo-KV key head dimension must be contiguous.");
    TORCH_CHECK(compactValue.stride(2) == 1, "compact pseudo-KV value head dimension must be contiguous.");
    TORCH_CHECK(causalMask.is_contiguous(), "compact_pseudokv_causal_mask must be contiguous.");
    TORCH_CHECK(outputIsRank3 ? output.stride(2) == 1 : output.stride(1) == 1,
        "compact pseudo-KV output head dimension must be contiguous.");

    tensorrt_llm::kernels::CompactPseudoKvAttentionLaunchParams params{};
    params.query = q.data_ptr();
    params.output = output.data_ptr();
    params.query_token_count = static_cast<int32_t>(q.size(0));
    params.query_stride_token_in_bytes = q.stride(0) * q.element_size();
    params.query_stride_head_in_bytes = q.stride(1) * q.element_size();
    params.output_stride_token_in_bytes = output.stride(0) * output.element_size();
    params.output_stride_head_in_bytes
        = outputIsRank3 ? output.stride(1) * output.element_size() : q.size(2) * output.element_size();
    params.data_type = nvinfer1::DataType::kFLOAT;
    params.compact_pseudokv_params.key = compactKey.data_ptr();
    params.compact_pseudokv_params.value = compactValue.data_ptr();
    params.compact_pseudokv_params.positions = positions.data_ptr<int32_t>();
    params.compact_pseudokv_params.causal_mask = causalMask.data_ptr<bool>();
    params.compact_pseudokv_params.compact_token_count = static_cast<int32_t>(compactKey.size(0));
    params.compact_pseudokv_params.source_sequence_length = static_cast<int32_t>(compactPseudokvSourceSeqLen.value());
    params.compact_pseudokv_params.num_heads = static_cast<int32_t>(compactKey.size(1));
    params.compact_pseudokv_params.head_size = static_cast<int32_t>(compactKey.size(2));
    params.compact_pseudokv_params.key_stride_token_in_bytes = compactKey.stride(0) * compactKey.element_size();
    params.compact_pseudokv_params.key_stride_head_in_bytes = compactKey.stride(1) * compactKey.element_size();
    params.compact_pseudokv_params.value_stride_token_in_bytes = compactValue.stride(0) * compactValue.element_size();
    params.compact_pseudokv_params.value_stride_head_in_bytes = compactValue.stride(1) * compactValue.element_size();

    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    tensorrt_llm::kernels::invokeCompactPseudoKvAttention(params, stream);
}

torch::Tensor compactPseudoKvAttention(torch::Tensor const& q, torch::Tensor output,
    torch::Tensor const& compactPseudokvKey, torch::Tensor const& compactPseudokvValue,
    torch::Tensor const& compactPseudokvPositions, torch::Tensor const& compactPseudokvCausalMask,
    int64_t compactPseudokvSourceSeqLen)
{
    runCompactPseudoKvAttention(q, output, compactPseudokvKey, compactPseudokvValue, compactPseudokvPositions,
        compactPseudokvCausalMask, compactPseudokvSourceSeqLen);
    return output;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
