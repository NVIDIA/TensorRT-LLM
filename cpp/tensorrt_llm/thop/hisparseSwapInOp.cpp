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

#include "tensorrt_llm/kernels/hisparseSwapIn.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

bool isInt64(torch::Tensor const& t)
{
    return t.scalar_type() == torch::kInt64;
}

void checkIntIndex(torch::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.scalar_type() == torch::kInt32 || t.scalar_type() == torch::kInt64, name,
        " must be int32 or int64, got ", t.scalar_type());
}

} // namespace

// Block/page-granular HiSparse host->device swap-in. See
// tensorrt_llm/kernels/hisparseSwapIn.h for the full contract. All buffers are mutated
// in place; the op returns nothing.
void hisparseSwapInBlocks(torch::Tensor const& topKBlocks, torch::Tensor& deviceBufferBlocks,
    torch::Tensor const& hostBlockLocs, torch::Tensor const& deviceBufferLocs, torch::Tensor const& hostCacheK,
    torch::Tensor const& hostCacheV, torch::Tensor& deviceBufferK, torch::Tensor& deviceBufferV,
    torch::Tensor& topKDeviceLocs, torch::Tensor const& reqPoolIndices, torch::Tensor const& seqLensBlocks,
    torch::Tensor& lruSlots, torch::Tensor const& numRealReqs, int64_t numTopK, int64_t hotBufferSize,
    int64_t itemSizeBytes, int64_t cudaBlockSize)
{
    TORCH_CHECK(topKBlocks.is_cuda(), "hisparse_swap_in_blocks expects CUDA tensors");
    TORCH_CHECK(topKBlocks.dim() == 2, "top_k_blocks must be [num_reqs, num_top_k]");
    TORCH_CHECK(topKBlocks.scalar_type() == torch::kInt32, "top_k_blocks must be int32");
    TORCH_CHECK(topKDeviceLocs.scalar_type() == torch::kInt32, "top_k_device_locs must be int32");
    TORCH_CHECK(deviceBufferBlocks.scalar_type() == torch::kInt32, "device_buffer_blocks must be int32");
    TORCH_CHECK(deviceBufferLocs.scalar_type() == torch::kInt32, "device_buffer_locs must be int32");
    TORCH_CHECK(hostBlockLocs.scalar_type() == torch::kInt64, "host_block_locs must be int64");
    TORCH_CHECK(lruSlots.scalar_type() == torch::kInt16, "lru_slots must be int16");
    TORCH_CHECK(numRealReqs.scalar_type() == torch::kInt32, "num_real_reqs must be int32");
    checkIntIndex(reqPoolIndices, "req_pool_indices");
    checkIntIndex(seqLensBlocks, "seq_lens_blocks");

    TORCH_CHECK(topKBlocks.size(1) == numTopK, "top_k_blocks second dim must equal num_top_k");
    TORCH_CHECK(topKDeviceLocs.size(0) == topKBlocks.size(0) && topKDeviceLocs.size(1) == numTopK,
        "top_k_device_locs must match [num_reqs, num_top_k]");
    TORCH_CHECK(cudaBlockSize % 32 == 0 && cudaBlockSize > 0 && cudaBlockSize <= 1024,
        "cuda_block_size must be a positive multiple of 32 and <= 1024 (the warp-scan carries at most 32 warps)");
    TORCH_CHECK(hotBufferSize >= numTopK, "hot_buffer_size must be >= num_top_k");
    TORCH_CHECK(itemSizeBytes > 0 && itemSizeBytes % 8 == 0, "item_size_bytes must be a positive multiple of 8");

    auto const numReqs = static_cast<int32_t>(topKBlocks.size(0));
    auto const stream = at::cuda::getCurrentCUDAStream(topKBlocks.get_device());

    tensorrt_llm::kernels::invokeHiSparseSwapInBlocks(topKBlocks.data_ptr<int32_t>(),
        deviceBufferBlocks.data_ptr<int32_t>(), hostBlockLocs.data_ptr<int64_t>(), deviceBufferLocs.data_ptr<int32_t>(),
        hostCacheK.data_ptr(), hostCacheV.data_ptr(), deviceBufferK.data_ptr(), deviceBufferV.data_ptr(),
        topKDeviceLocs.data_ptr<int32_t>(), reqPoolIndices.data_ptr(), isInt64(reqPoolIndices), seqLensBlocks.data_ptr(),
        isInt64(seqLensBlocks), lruSlots.data_ptr<int16_t>(), numRealReqs.data_ptr<int32_t>(), numReqs,
        static_cast<int32_t>(numTopK), static_cast<int32_t>(hotBufferSize), deviceBufferBlocks.stride(0),
        hostBlockLocs.stride(0), lruSlots.stride(0), topKBlocks.stride(0), topKDeviceLocs.stride(0), itemSizeBytes,
        static_cast<int32_t>(cudaBlockSize), stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "hisparse_swap_in_blocks(Tensor top_k_blocks, Tensor(a!) device_buffer_blocks, Tensor host_block_locs, "
        "Tensor device_buffer_locs, Tensor host_cache_k, Tensor host_cache_v, Tensor(b!) device_buffer_k, "
        "Tensor(c!) device_buffer_v, Tensor(d!) top_k_device_locs, Tensor req_pool_indices, Tensor seq_lens_blocks, "
        "Tensor(e!) lru_slots, Tensor num_real_reqs, int num_top_k, int hot_buffer_size, int item_size_bytes, "
        "int cuda_block_size) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("hisparse_swap_in_blocks", &tensorrt_llm::torch_ext::hisparseSwapInBlocks);
}
