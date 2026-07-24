/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <limits>
#include <optional>
#include <vector>

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! Compact one uniform group of KVCacheManagerV2 HND layer pools in one batched
//! launch (per request and KV head, moves are ascending and never overtake their
//! sources: the copy runs in place).
void sparseKvCacheCompactLayers(std::vector<th::Tensor> const& pools, th::Tensor const& poolPointers,
    th::Tensor const& pageTable, th::Tensor const& sourceIndices, th::Tensor const& sourceOffsets,
    th::Tensor const& destinationBases, std::optional<th::Tensor> const& sourceLayerIndices)
{
    TORCH_CHECK(!pools.empty(), "sparse_kv_cache_compact_layers: pools must be non-empty");

    auto const& firstPool = pools.front();
    TORCH_CHECK(firstPool.is_cuda() && firstPool.dim() == 5 && firstPool.size(1) == 2 && firstPool.is_contiguous(),
        "sparse_kv_cache_compact_layers: pools must be contiguous CUDA "
        "[pages, 2, kv_heads, tokens_per_block, head_dim] tensors");
    TORCH_CHECK(pageTable.is_cuda() && pageTable.dim() == 2 && pageTable.scalar_type() == th::kInt32
            && pageTable.stride(1) == 1,
        "sparse_kv_cache_compact_layers: K block offsets must be CUDA int32 "
        "[batch, max_pages] tensors with a contiguous block dimension");

    auto const device = firstPool.get_device();
    auto const dtype = firstPool.scalar_type();
    auto const numLayers = static_cast<int32_t>(pools.size());
    auto const numKvHeads = static_cast<int32_t>(firstPool.size(2));
    auto const tokensPerBlock = static_cast<int32_t>(firstPool.size(3));
    auto const headDim = static_cast<int32_t>(firstPool.size(4));
    auto const batchSize = static_cast<int32_t>(pageTable.size(0));
    auto const pageTableRequestStride = pageTable.stride(0);

    // Layer 0 defined the reference geometry in the firstPool checks above.
    for (int32_t layer = 1; layer < numLayers; ++layer)
    {
        auto const& pool = pools[layer];
        TORCH_CHECK(pool.is_cuda() && pool.get_device() == device && pool.scalar_type() == dtype && pool.dim() == 5
                && pool.size(1) == 2 && pool.is_contiguous(),
            "sparse_kv_cache_compact_layers: all pools must have one device, dtype, layout, and contiguous storage");
        TORCH_CHECK(pool.size(2) == numKvHeads && pool.size(3) == tokensPerBlock && pool.size(4) == headDim,
            "sparse_kv_cache_compact_layers: all pools must share KV-head, block, and head-dimension geometry");
    }
    TORCH_CHECK(
        pageTable.get_device() == device, "sparse_kv_cache_compact_layers: block offsets must be on the pool device");

    TORCH_CHECK(poolPointers.is_cuda() && poolPointers.get_device() == device
            && poolPointers.scalar_type() == th::kInt64 && poolPointers.dim() == 1 && poolPointers.size(0) == numLayers
            && poolPointers.is_contiguous(),
        "sparse_kv_cache_compact_layers: pool_pointers must be contiguous CUDA int64 [num_layers]");

    TORCH_CHECK(sourceIndices.is_cuda() && sourceIndices.get_device() == device
            && sourceIndices.scalar_type() == th::kInt32 && sourceIndices.is_contiguous()
            && (sourceIndices.dim() == 2 || sourceIndices.dim() == 3),
        "sparse_kv_cache_compact_layers: source_indices must be contiguous CUDA int32 "
        "[kv_heads, total] or [source_layers, kv_heads, total]");
    int64_t sourceLayerStride = 0;
    int32_t const* sourceLayerPtr = nullptr;
    if (sourceIndices.dim() == 2)
    {
        TORCH_CHECK(sourceIndices.size(0) == numKvHeads,
            "sparse_kv_cache_compact_layers: source_indices KV-head dimension mismatch");
        TORCH_CHECK(!sourceLayerIndices.has_value(),
            "sparse_kv_cache_compact_layers: source_layer_indices require 3-D per-layer source_indices");
    }
    else
    {
        TORCH_CHECK(sourceIndices.size(0) > 0 && sourceIndices.size(1) == numKvHeads,
            "sparse_kv_cache_compact_layers: per-layer source_indices geometry mismatch");
        TORCH_CHECK(sourceLayerIndices.has_value(),
            "sparse_kv_cache_compact_layers: per-layer source_indices require source_layer_indices");
        sourceLayerStride = sourceIndices.stride(0);
    }
    if (sourceLayerIndices.has_value())
    {
        auto const& layerIndices = *sourceLayerIndices;
        TORCH_CHECK(layerIndices.is_cuda() && layerIndices.get_device() == device
                && layerIndices.scalar_type() == th::kInt32 && layerIndices.is_contiguous() && layerIndices.dim() == 1
                && layerIndices.size(0) == numLayers,
            "sparse_kv_cache_compact_layers: source_layer_indices must be contiguous CUDA int32 [num_layers]");
        sourceLayerPtr = layerIndices.data_ptr<int32_t>();
    }

    // source_offsets carve each request's move range; device-resident, the kernel trusts them.
    TORCH_CHECK(sourceOffsets.is_cuda() && sourceOffsets.get_device() == device
            && sourceOffsets.scalar_type() == th::kInt32 && sourceOffsets.is_contiguous() && sourceOffsets.dim() == 1
            && sourceOffsets.size(0) == batchSize + 1,
        "sparse_kv_cache_compact_layers: source_offsets must be contiguous CUDA int32 [batch + 1]");
    // Per-request landing positions: one launch covers a cohort with mixed
    // pinned-prompt lengths. Values live on device; the kernel trusts them.
    TORCH_CHECK(destinationBases.is_cuda() && destinationBases.get_device() == device
            && destinationBases.scalar_type() == th::kInt32 && destinationBases.dim() == 1
            && destinationBases.size(0) == batchSize && destinationBases.is_contiguous(),
        "sparse_kv_cache_compact_layers: destination_bases must be contiguous CUDA int32 [batch]");

    auto const stream = at::cuda::getCurrentCUDAStream(device);
    auto const* bases = destinationBases.data_ptr<int32_t>();
    auto const sourceHeadStride = sourceIndices.size(-1);
    if (dtype == th::kBFloat16)
    {
        tk::invokeSparseKvCacheCompactLayers<__nv_bfloat16>(poolPointers.data_ptr<int64_t>(),
            pageTable.data_ptr<int32_t>(), numLayers, pageTableRequestStride, sourceIndices.data_ptr<int32_t>(),
            sourceLayerPtr, sourceLayerStride, sourceHeadStride, sourceOffsets.data_ptr<int32_t>(), bases, batchSize,
            numKvHeads, tokensPerBlock, headDim, stream);
    }
    else
    {
        TORCH_CHECK(false,
            "sparse_kv_cache_compact_layers ships only the pipelined bf16 kernels (head size 64/128, page size "
            "32/128 tokens); got pool dtype ",
            dtype);
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "sparse_kv_cache_compact_layers(Tensor(a!)[] pools, Tensor pool_pointers, Tensor page_table, Tensor "
        "source_indices, Tensor source_offsets, Tensor destination_bases, "
        "Tensor? source_layer_indices=None) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("sparse_kv_cache_compact_layers", &tensorrt_llm::torch_ext::sparseKvCacheCompactLayers);
}
