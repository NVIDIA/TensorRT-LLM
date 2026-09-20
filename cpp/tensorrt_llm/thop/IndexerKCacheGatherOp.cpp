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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include "tensorrt_llm/kernels/IndexerKCacheGather.h"

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

std::tuple<th::Tensor, th::Tensor> indexer_k_cache_gather_op(th::Tensor const& k_cache,
    th::Tensor const& slot_mapping_fp8, th::Tensor const& slot_mapping_scale, int64_t k_token_start, int64_t num_tokens,
    int64_t head_dim)
{
    // head_dim is the number of payload bytes per token: 128 for FP8 (one byte
    // per float8 value) or 64 for FP4 (two packed E2M1 codes per byte).
    constexpr int32_t SCALE_SIZE = 4;
    TORCH_CHECK(head_dim == 128 || head_dim == 64,
        "indexer_k_cache_gather_op head_dim must be 128 (FP8) or 64 (FP4 packed), got %d", static_cast<int>(head_dim));
    auto head_dim_i32 = static_cast<int32_t>(head_dim);

    auto device = k_cache.device();

    // Early return for empty gather. Keep the FP8 dtype on the output to stay
    // compatible with the legacy caller contract; the FP4 call site reinterprets
    // the bytes before handing them to DeepGEMM.
    if (num_tokens == 0)
    {
        auto k_fp8 = th::empty({0, head_dim_i32}, th::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device));
        auto k_scale = th::empty({0, 1}, th::TensorOptions().dtype(torch::kFloat32).device(device));
        return std::make_tuple(std::move(k_fp8), std::move(k_scale));
    }

    // Validate all tensors are CUDA tensors
    TORCH_CHECK(k_cache.is_cuda() && slot_mapping_fp8.is_cuda() && slot_mapping_scale.is_cuda(),
        "All tensors must be CUDA tensors");

    // Validate tensor dimensions
    TORCH_CHECK(slot_mapping_fp8.dim() == 1, "slot_mapping_fp8 must be a 1D Tensor");
    TORCH_CHECK(slot_mapping_scale.dim() == 1, "slot_mapping_scale must be a 1D Tensor");

    // Enforce k_cache is 4D tensor
    TORCH_CHECK(k_cache.dim() == 4,
        "k_cache must be a 4D Tensor [num_blocks, block_size, 1, per_token_size], got %d dimensions",
        static_cast<int>(k_cache.dim()));

    // Validate tensor dtypes
    TORCH_CHECK(slot_mapping_fp8.scalar_type() == torch::kInt64, "slot_mapping_fp8 must be int64");
    TORCH_CHECK(slot_mapping_scale.scalar_type() == torch::kInt64, "slot_mapping_scale must be int64");

    // Validate tensors are contiguous (except k_cache which may be non-contiguous)
    // k_cache can be non-contiguous - we handle this via strides
    TORCH_CHECK(slot_mapping_fp8.is_contiguous(), "slot_mapping_fp8 must be contiguous");
    TORCH_CHECK(slot_mapping_scale.is_contiguous(), "slot_mapping_scale must be contiguous");

    // Validate slot_mapping has enough elements
    TORCH_CHECK(slot_mapping_fp8.size(0) >= k_token_start + num_tokens,
        "slot_mapping_fp8 too short for k_token_start + num_tokens");
    TORCH_CHECK(slot_mapping_scale.size(0) >= k_token_start + num_tokens,
        "slot_mapping_scale too short for k_token_start + num_tokens");

    int32_t cache_dim_0 = static_cast<int32_t>(k_cache.size(0)); // num_blocks
    int32_t cache_dim_1 = static_cast<int32_t>(k_cache.size(1)); // block_size
    int32_t cache_dim_2 = static_cast<int32_t>(k_cache.size(2)); // num_kv_heads
    int32_t cache_dim_3 = static_cast<int32_t>(k_cache.size(3)); // per_token_size

    // Validation for indexer k cache pool for DeepSeek-V3.2 constraints
    TORCH_CHECK(cache_dim_2 == 1, "k_cache dimension 2 must be 1 for DeepSeek-V3.2, got %d", cache_dim_2);

    int64_t cache_stride_0 = static_cast<int64_t>(k_cache.stride(0));
    int64_t cache_stride_1 = static_cast<int64_t>(k_cache.stride(1));
    int64_t cache_stride_2 = static_cast<int64_t>(k_cache.stride(2));
    int64_t cache_stride_3 = static_cast<int64_t>(k_cache.stride(3));

    // Allocate output buffers
    auto num_tokens_i32 = static_cast<int32_t>(num_tokens);
    auto out_fp8 = th::empty({num_tokens, head_dim_i32}, th::TensorOptions().dtype(torch::kUInt8).device(device));
    auto out_scale = th::empty({num_tokens, SCALE_SIZE}, th::TensorOptions().dtype(torch::kUInt8).device(device));

    auto stream = at::cuda::getCurrentCUDAStream(k_cache.get_device());

    tk::invokeIndexerKCacheGather(k_cache.data_ptr<uint8_t>(), slot_mapping_fp8.data_ptr<int64_t>(),
        slot_mapping_scale.data_ptr<int64_t>(), out_fp8.data_ptr<uint8_t>(), out_scale.data_ptr<uint8_t>(),
        static_cast<int32_t>(k_token_start), num_tokens_i32, head_dim_i32, SCALE_SIZE, cache_dim_0, cache_dim_1,
        cache_dim_2, cache_dim_3, cache_stride_0, cache_stride_1, cache_stride_2, cache_stride_3, stream);

    // View-cast to final dtypes (no copy). The FP8 call site keeps its existing
    // float8_e4m3fn semantics; the FP4 call site reshapes the raw bytes into
    // int8-packed nibbles after this op returns.
    auto k_fp8 = out_fp8.view(torch::kFloat8_e4m3fn);
    auto k_scale = out_scale.view(torch::kFloat32).view({num_tokens, 1});

    return std::make_tuple(std::move(k_fp8), std::move(k_scale));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "indexer_k_cache_gather_op(Tensor k_cache, Tensor slot_mapping_fp8, "
        "Tensor slot_mapping_scale, int k_token_start, int num_tokens, int head_dim) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("indexer_k_cache_gather_op", &tensorrt_llm::torch_ext::indexer_k_cache_gather_op);
}
