/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include "tensorrt_llm/kernels/IndexerKCacheScatter.h"

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

void indexer_k_cache_scatter_op(th::Tensor const& k_fp8, th::Tensor const& k_scale, th::Tensor& k_cache,
    th::Tensor const& slot_mapping_fp8, th::Tensor const& slot_mapping_scale, int64_t num_tokens)
{
    // k_fp8: [>=num_tokens, head_dim] in FP8 (1 byte/element) — reinterpreted as uint8
    // k_scale: [>=num_tokens, head_dim // quant_block_size] in float32 — reinterpreted as uint8 bytes
    // slot_mapping_fp8, slot_mapping_scale: [>=num_tokens] int64 — only first num_tokens used
    // k_cache: [num_blocks, block_size, 1, per_token_size] uint8

    TORCH_CHECK(k_fp8.is_cuda() && k_scale.is_cuda() && k_cache.is_cuda() && slot_mapping_fp8.is_cuda()
            && slot_mapping_scale.is_cuda(),
        "All tensors must be CUDA tensors");

    // Validate tensor dimensions
    TORCH_CHECK(k_fp8.dim() == 2, "k_fp8 must be 2D [num_tokens, head_dim]");
    TORCH_CHECK(k_scale.dim() == 2, "k_scale must be 2D [num_tokens, scale_elements]");
    TORCH_CHECK(slot_mapping_fp8.dim() == 1, "slot_mapping_fp8 must be 1D [num_tokens]");
    TORCH_CHECK(slot_mapping_scale.dim() == 1, "slot_mapping_scale must be 1D [num_tokens]");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be 4D [num_blocks, block_size, 1, per_token_size], got %d dims",
        static_cast<int>(k_cache.dim()));

    // Validate tensor dtypes — reinterpret_cast below assumes specific element sizes
    TORCH_CHECK(k_fp8.element_size() == 1, "k_fp8 must have 1-byte elements (e.g. FP8), got %d", k_fp8.element_size());
    TORCH_CHECK(k_scale.element_size() == 4, "k_scale must have 4-byte elements (e.g. float32), got %d",
        k_scale.element_size());
    TORCH_CHECK(slot_mapping_fp8.scalar_type() == torch::kInt64, "slot_mapping_fp8 must be int64");
    TORCH_CHECK(slot_mapping_scale.scalar_type() == torch::kInt64, "slot_mapping_scale must be int64");

    TORCH_CHECK(k_fp8.is_contiguous(), "k_fp8 must be contiguous");
    TORCH_CHECK(k_scale.is_contiguous(), "k_scale must be contiguous");
    TORCH_CHECK(slot_mapping_fp8.is_contiguous(), "slot_mapping_fp8 must be contiguous");
    TORCH_CHECK(slot_mapping_scale.is_contiguous(), "slot_mapping_scale must be contiguous");

    // FP8 is 1 byte per element, so head_dim in elements == head_dim in bytes.
    int32_t const head_dim = static_cast<int32_t>(k_fp8.size(1));
    // Scale size in bytes: num_scale_elements * bytes_per_element.
    int32_t const scale_size = static_cast<int32_t>(k_scale.size(1)) * static_cast<int32_t>(k_scale.element_size());

    int32_t const cache_dim_0 = static_cast<int32_t>(k_cache.size(0));
    int32_t const cache_dim_1 = static_cast<int32_t>(k_cache.size(1));
    int32_t const cache_dim_2 = static_cast<int32_t>(k_cache.size(2));
    int32_t const cache_dim_3 = static_cast<int32_t>(k_cache.size(3));

    TORCH_CHECK(cache_dim_2 == 1, "k_cache dimension 2 must be 1, got %d", cache_dim_2);
    TORCH_CHECK(head_dim == 128, "k_fp8 head_dim must be 128, got %d", head_dim);
    TORCH_CHECK(scale_size == 4, "k_scale scale_size must be 4 bytes, got %d", scale_size);

    int64_t const cache_stride_0 = static_cast<int64_t>(k_cache.stride(0));
    int64_t const cache_stride_1 = static_cast<int64_t>(k_cache.stride(1));
    int64_t const cache_stride_2 = static_cast<int64_t>(k_cache.stride(2));
    int64_t const cache_stride_3 = static_cast<int64_t>(k_cache.stride(3));

    auto stream = at::cuda::getCurrentCUDAStream(k_fp8.get_device());

    // Reinterpret k_fp8 as uint8 bytes and k_scale as raw bytes via data_ptr.
    // For slot mappings, use data_ptr directly — only the first num_tokens entries are read.
    tk::invokeIndexerKCacheScatter(reinterpret_cast<uint8_t const*>(k_fp8.data_ptr()),
        reinterpret_cast<uint8_t const*>(k_scale.data_ptr()), k_cache.data_ptr<uint8_t>(),
        slot_mapping_fp8.data_ptr<int64_t>(), slot_mapping_scale.data_ptr<int64_t>(), static_cast<int32_t>(num_tokens),
        head_dim, scale_size, cache_dim_0, cache_dim_1, cache_dim_2, cache_dim_3, cache_stride_0, cache_stride_1,
        cache_stride_2, cache_stride_3, stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "indexer_k_cache_scatter_op(Tensor k_fp8, Tensor k_scale, Tensor(a!) k_cache, "
        "Tensor slot_mapping_fp8, Tensor slot_mapping_scale, int num_tokens) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("indexer_k_cache_scatter_op", &tensorrt_llm::torch_ext::indexer_k_cache_scatter_op);
}
