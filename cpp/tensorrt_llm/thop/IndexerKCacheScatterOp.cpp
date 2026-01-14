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

void indexer_k_cache_scatter_op(th::Tensor const& k_fp8_bytes, th::Tensor const& k_scale_bytes, th::Tensor& k_cache,
    th::Tensor const& slot_mapping_fp8, th::Tensor const& slot_mapping_scale)
{
    // Validate all tensors are CUDA tensors
    TORCH_CHECK(k_fp8_bytes.is_cuda() && k_scale_bytes.is_cuda() && k_cache.is_cuda() && slot_mapping_fp8.is_cuda()
            && slot_mapping_scale.is_cuda(),
        "All tensors must be CUDA tensors");

    // Validate tensor dimensions
    TORCH_CHECK(k_fp8_bytes.dim() == 2, "k_fp8_bytes must be a 2D Tensor [num_tokens, head_dim]");
    TORCH_CHECK(k_scale_bytes.dim() == 2, "k_scale_bytes must be a 2D Tensor [num_tokens, scale_size]");
    TORCH_CHECK(slot_mapping_fp8.dim() == 1, "slot_mapping_fp8 must be a 1D Tensor [num_tokens]");
    TORCH_CHECK(slot_mapping_scale.dim() == 1, "slot_mapping_scale must be a 1D Tensor [num_tokens]");

    // Enforce k_cache is 4D tensor
    TORCH_CHECK(k_cache.dim() == 4,
        "k_cache must be a 4D Tensor [num_blocks, block_size, 1, per_token_size], got %d dimensions",
        static_cast<int>(k_cache.dim()));

    // Validate tensor dtypes
    TORCH_CHECK(k_fp8_bytes.scalar_type() == torch::kUInt8, "k_fp8_bytes must be uint8");
    TORCH_CHECK(k_scale_bytes.scalar_type() == torch::kUInt8, "k_scale_bytes must be uint8");
    TORCH_CHECK(slot_mapping_fp8.scalar_type() == torch::kInt64, "slot_mapping_fp8 must be int64");
    TORCH_CHECK(slot_mapping_scale.scalar_type() == torch::kInt64, "slot_mapping_scale must be int64");

    // Validate tensor shapes are consistent
    auto num_tokens = static_cast<int32_t>(k_fp8_bytes.size(0));
    TORCH_CHECK(
        k_scale_bytes.size(0) == num_tokens, "k_scale_bytes first dimension must equal k_fp8_bytes first dimension");
    TORCH_CHECK(slot_mapping_fp8.size(0) == num_tokens, "slot_mapping_fp8 length must equal num_tokens");
    TORCH_CHECK(slot_mapping_scale.size(0) == num_tokens, "slot_mapping_scale length must equal num_tokens");

    // Validate tensors are contiguous (except k_cache which may be non-contiguous)
    TORCH_CHECK(k_fp8_bytes.is_contiguous(), "k_fp8_bytes must be contiguous");
    TORCH_CHECK(k_scale_bytes.is_contiguous(), "k_scale_bytes must be contiguous");
    // k_cache can be non-contiguous - we handle this via strides
    TORCH_CHECK(slot_mapping_fp8.is_contiguous(), "slot_mapping_fp8 must be contiguous");
    TORCH_CHECK(slot_mapping_scale.is_contiguous(), "slot_mapping_scale must be contiguous");

    int32_t head_dim = static_cast<int32_t>(k_fp8_bytes.size(1));     // head_dim = quant_block_size = 128
    int32_t scale_size = static_cast<int32_t>(k_scale_bytes.size(1)); // scale_size = 4 bytes

    int32_t cache_dim_0 = static_cast<int32_t>(k_cache.size(0));      // num_blocks
    int32_t cache_dim_1 = static_cast<int32_t>(k_cache.size(1));      // block_size
    int32_t cache_dim_2 = static_cast<int32_t>(k_cache.size(2));      // num_kv_heads
    int32_t cache_dim_3 = static_cast<int32_t>(k_cache.size(3));      // per_token_size

    // Validation for indexer k cache pool for DeepSeek-V3.2 constraints
    TORCH_CHECK(cache_dim_2 == 1, "k_cache dimension 2 must be 1 for DeepSeek-V3.2, got %d", cache_dim_2);
    TORCH_CHECK(head_dim == 128, "k_fp8_bytes head_dim must be 128 for DeepSeek-V3.2, got %d", head_dim);
    TORCH_CHECK(scale_size == 4, "k_scale_bytes scale_size must be 4 bytes for DeepSeek-V3.2, got %d", scale_size);

    int64_t cache_stride_0 = static_cast<int64_t>(k_cache.stride(0));
    int64_t cache_stride_1 = static_cast<int64_t>(k_cache.stride(1));
    int64_t cache_stride_2 = static_cast<int64_t>(k_cache.stride(2));
    int64_t cache_stride_3 = static_cast<int64_t>(k_cache.stride(3));

    auto stream = at::cuda::getCurrentCUDAStream(k_fp8_bytes.get_device());

    tk::invokeIndexerKCacheScatter(k_fp8_bytes.data_ptr<uint8_t>(), k_scale_bytes.data_ptr<uint8_t>(),
        k_cache.data_ptr<uint8_t>(), slot_mapping_fp8.data_ptr<int64_t>(), slot_mapping_scale.data_ptr<int64_t>(),
        num_tokens, head_dim, scale_size, cache_dim_0, cache_dim_1, cache_dim_2, cache_dim_3, cache_stride_0,
        cache_stride_1, cache_stride_2, cache_stride_3, stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "indexer_k_cache_scatter_op(Tensor k_fp8_bytes, Tensor k_scale_bytes, Tensor(a!) k_cache, "
        "Tensor slot_mapping_fp8, Tensor slot_mapping_scale) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("indexer_k_cache_scatter_op", &tensorrt_llm::torch_ext::indexer_k_cache_scatter_op);
}
