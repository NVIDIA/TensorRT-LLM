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

#include "tensorrt_llm/kernels/nvfp4MlaKvCacheGather.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <limits>

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

void nvFp4MlaKvCacheGather(th::Tensor const& hostPoolPointers, th::Tensor const& hostPoolMapping,
    th::Tensor const& globalIndices, th::Tensor& output, th::Tensor& compactIndices,
    th::Tensor const& globalDequantScale, int64_t layerIdx, int64_t residualDim, int64_t numPoolTokens)
{
    TORCH_CHECK(hostPoolPointers.device().is_cpu(), "host_pool_pointers must be a CPU tensor");
    TORCH_CHECK(hostPoolPointers.scalar_type() == th::kInt64, "host_pool_pointers must be int64");
    TORCH_CHECK(hostPoolPointers.dim() == 3 && hostPoolPointers.size(2) == 2,
        "host_pool_pointers must have shape [num_pools, 2, 2] for NVFP4 cache");
    TORCH_CHECK(hostPoolMapping.device().is_cpu(), "host_pool_mapping must be a CPU tensor");
    TORCH_CHECK(hostPoolMapping.scalar_type() == th::kInt32, "host_pool_mapping must be int32");
    TORCH_CHECK(hostPoolMapping.dim() == 2 && hostPoolMapping.size(1) == 2,
        "host_pool_mapping must have shape [num_layers, 2]");
    TORCH_CHECK(layerIdx >= 0 && layerIdx < hostPoolMapping.size(0), "layer_idx is out of range");

    TORCH_CHECK(globalIndices.is_cuda() && output.is_cuda() && compactIndices.is_cuda() && globalDequantScale.is_cuda(),
        "global_indices, output, compact_indices, and global_dequant_scale must be CUDA tensors");
    TORCH_CHECK(output.device() == globalIndices.device() && compactIndices.device() == globalIndices.device()
            && globalDequantScale.device() == globalIndices.device(),
        "global_indices, output, compact_indices, and global_dequant_scale must be on the same CUDA device");
    TORCH_CHECK(globalIndices.scalar_type() == th::kInt32, "global_indices must be int32");
    TORCH_CHECK(
        globalIndices.dim() == 2 && globalIndices.is_contiguous(), "global_indices must be a contiguous 2D tensor");
    TORCH_CHECK(output.scalar_type() == th::kFloat8_e4m3fn, "output must be float8_e4m3fn");
    TORCH_CHECK(output.dim() == 3 && output.is_contiguous(), "output must be a contiguous 3D tensor");
    TORCH_CHECK(compactIndices.scalar_type() == th::kInt32, "compact_indices must be int32");
    TORCH_CHECK(
        compactIndices.dim() == 2 && compactIndices.is_contiguous(), "compact_indices must be a contiguous 2D tensor");
    TORCH_CHECK(globalDequantScale.scalar_type() == th::kFloat32 && globalDequantScale.numel() >= 1
            && globalDequantScale.is_contiguous(),
        "global_dequant_scale must contain at least one contiguous float32 value");
    TORCH_CHECK(output.size(0) == globalIndices.size(0) && output.size(1) == globalIndices.size(1),
        "output leading dimensions must match global_indices");
    TORCH_CHECK(compactIndices.sizes() == globalIndices.sizes(), "compact_indices shape must match global_indices");
    TORCH_CHECK(residualDim >= 0 && residualDim <= output.size(2) && residualDim % 16 == 0,
        "residual_dim must be a multiple of 16 in [0, head_dim]");
    TORCH_CHECK(numPoolTokens > 0, "num_pool_tokens must be positive");

    int32_t const poolIdx = hostPoolMapping.index({layerIdx, 0}).item<int32_t>();
    TORCH_CHECK(poolIdx >= 0 && poolIdx < hostPoolPointers.size(0), "mapped pool index is out of range");
    auto const dataPtr = hostPoolPointers.index({poolIdx, 0, 0}).item<int64_t>();
    auto const scalePtr = hostPoolPointers.index({poolIdx, 0, 1}).item<int64_t>();
    TORCH_CHECK(dataPtr != 0 && scalePtr != 0, "NVFP4 data and scale pool pointers must be non-null");

    auto stream = at::cuda::getCurrentCUDAStream(globalIndices.get_device()).stream();
    tk::invokeNvFp4MlaKvCacheGather(reinterpret_cast<uint8_t const*>(dataPtr),
        reinterpret_cast<__nv_fp8_e4m3 const*>(scalePtr), globalIndices.data_ptr<int32_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()), compactIndices.data_ptr<int32_t>(),
        globalDequantScale.data_ptr<float>(), static_cast<int32_t>(globalIndices.size(0)),
        static_cast<int32_t>(globalIndices.size(1)), static_cast<int32_t>(output.size(2)),
        static_cast<int32_t>(residualDim), numPoolTokens, stream);
}

void nvFp4MlaContextKvCacheGather(th::Tensor const& hostPoolPointers, th::Tensor const& hostPoolMapping,
    th::Tensor const& localTopKIndices, th::Tensor const& queryReqIndices, th::Tensor const& blockTable,
    th::Tensor const& cuKvLengths, th::Tensor& output, th::Tensor& compactIndices, th::Tensor const& globalDequantScale,
    int64_t layerIdx, int64_t totalKvTokens, int64_t tokensPerBlock, int64_t pageStride, int64_t layerId,
    int64_t residualDim, int64_t numPoolTokens)
{
    TORCH_CHECK(hostPoolPointers.device().is_cpu(), "host_pool_pointers must be a CPU tensor");
    TORCH_CHECK(hostPoolPointers.scalar_type() == th::kInt64, "host_pool_pointers must be int64");
    TORCH_CHECK(hostPoolPointers.dim() == 3 && hostPoolPointers.size(2) == 2,
        "host_pool_pointers must have shape [num_pools, 2, 2] for NVFP4 cache");
    TORCH_CHECK(hostPoolMapping.device().is_cpu(), "host_pool_mapping must be a CPU tensor");
    TORCH_CHECK(hostPoolMapping.scalar_type() == th::kInt32, "host_pool_mapping must be int32");
    TORCH_CHECK(hostPoolMapping.dim() == 2 && hostPoolMapping.size(1) == 2,
        "host_pool_mapping must have shape [num_layers, 2]");
    TORCH_CHECK(layerIdx >= 0 && layerIdx < hostPoolMapping.size(0), "layer_idx is out of range");

    TORCH_CHECK(localTopKIndices.is_cuda() && queryReqIndices.is_cuda() && blockTable.is_cuda() && cuKvLengths.is_cuda()
            && output.is_cuda() && compactIndices.is_cuda() && globalDequantScale.is_cuda(),
        "all non-host-pointer tensors must be CUDA tensors");
    auto const device = localTopKIndices.device();
    TORCH_CHECK(queryReqIndices.device() == device && blockTable.device() == device && cuKvLengths.device() == device
            && output.device() == device && compactIndices.device() == device && globalDequantScale.device() == device,
        "all CUDA tensors must be on the same device");
    TORCH_CHECK(
        localTopKIndices.scalar_type() == th::kInt32 && localTopKIndices.dim() == 2 && localTopKIndices.is_contiguous(),
        "local_topk_indices must be contiguous int32 [num_query_rows, topk]");
    TORCH_CHECK(queryReqIndices.scalar_type() == th::kInt32 && queryReqIndices.dim() == 1
            && queryReqIndices.is_contiguous() && queryReqIndices.size(0) == localTopKIndices.size(0),
        "query_req_indices must be contiguous int32 [num_query_rows]");
    TORCH_CHECK(blockTable.scalar_type() == th::kInt32 && blockTable.dim() == 2 && blockTable.is_contiguous(),
        "block_table must be a contiguous int32 2D tensor");
    TORCH_CHECK(cuKvLengths.scalar_type() == th::kInt64 && cuKvLengths.dim() == 1 && cuKvLengths.is_contiguous(),
        "cu_kv_lengths must be a contiguous int64 tensor");
    int32_t const numRequests = static_cast<int32_t>(cuKvLengths.size(0) - 1);
    TORCH_CHECK(numRequests > 0 && blockTable.size(0) >= numRequests,
        "block_table and cu_kv_lengths request dimensions must agree");
    TORCH_CHECK(totalKvTokens > 0 && totalKvTokens <= std::numeric_limits<int32_t>::max(),
        "total_kv_tokens must be a positive int32 value");
    int64_t const maxSelectedTokens = std::min(totalKvTokens, localTopKIndices.size(0) * localTopKIndices.size(1));
    TORCH_CHECK(output.scalar_type() == th::kFloat8_e4m3fn && output.dim() == 3 && output.is_contiguous()
            && output.size(0) >= maxSelectedTokens && output.size(1) == 1,
        "output must be contiguous float8_e4m3fn [capacity, 1, head_dim] with capacity >= ", maxSelectedTokens);
    TORCH_CHECK(output.size(0) <= std::numeric_limits<int32_t>::max(), "output capacity exceeds int32 range");
    TORCH_CHECK(compactIndices.scalar_type() == th::kInt32 && compactIndices.is_contiguous()
            && compactIndices.sizes() == localTopKIndices.sizes(),
        "compact_indices must be contiguous int32 with the local_topk_indices shape");
    TORCH_CHECK(residualDim >= 0 && residualDim <= output.size(2) && residualDim % 16 == 0,
        "residual_dim must be a multiple of 16 in [0, head_dim]");
    TORCH_CHECK(globalDequantScale.scalar_type() == th::kFloat32 && globalDequantScale.numel() >= 1
            && globalDequantScale.is_contiguous(),
        "global_dequant_scale must contain at least one contiguous float32 value");
    TORCH_CHECK(tokensPerBlock > 0 && pageStride > 0 && layerId >= 0 && numPoolTokens > 0,
        "invalid NVFP4 context gather geometry");

    int32_t const poolIdx = hostPoolMapping.index({layerIdx, 0}).item<int32_t>();
    TORCH_CHECK(poolIdx >= 0 && poolIdx < hostPoolPointers.size(0), "mapped pool index is out of range");
    auto const dataPtr = hostPoolPointers.index({poolIdx, 0, 0}).item<int64_t>();
    auto const scalePtr = hostPoolPointers.index({poolIdx, 0, 1}).item<int64_t>();
    TORCH_CHECK(dataPtr != 0 && scalePtr != 0, "NVFP4 data and scale pool pointers must be non-null");

    auto stream = at::cuda::getCurrentCUDAStream(localTopKIndices.get_device()).stream();
    int32_t const totalKvTokens32 = static_cast<int32_t>(totalKvTokens);
    size_t const workspaceSize = tk::getNvFp4MlaContextKvCacheGatherWorkspaceSize(totalKvTokens32, stream);
    auto workspace = th::empty(
        {static_cast<int64_t>(workspaceSize)}, th::TensorOptions().dtype(th::kUInt8).device(localTopKIndices.device()));
    tk::invokeNvFp4MlaContextKvCacheGather(reinterpret_cast<uint8_t const*>(dataPtr),
        reinterpret_cast<__nv_fp8_e4m3 const*>(scalePtr), localTopKIndices.data_ptr<int32_t>(),
        queryReqIndices.data_ptr<int32_t>(), blockTable.data_ptr<int32_t>(), cuKvLengths.data_ptr<int64_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()), compactIndices.data_ptr<int32_t>(),
        globalDequantScale.data_ptr<float>(), workspace.data_ptr(), workspaceSize,
        static_cast<int32_t>(localTopKIndices.size(0)), static_cast<int32_t>(localTopKIndices.size(1)), numRequests,
        static_cast<int32_t>(blockTable.size(1)), totalKvTokens32, static_cast<int32_t>(output.size(0)),
        static_cast<int32_t>(tokensPerBlock), static_cast<int32_t>(pageStride), static_cast<int32_t>(layerId),
        static_cast<int32_t>(output.size(2)), static_cast<int32_t>(residualDim), numPoolTokens, stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "nvfp4_mla_kv_cache_gather(Tensor host_pool_pointers, Tensor host_pool_mapping, Tensor global_indices, "
        "Tensor(a!) output, Tensor(b!) compact_indices, Tensor global_dequant_scale, int layer_idx, "
        "int residual_dim, int num_pool_tokens) -> ()");
    m.def(
        "nvfp4_mla_context_kv_cache_gather(Tensor host_pool_pointers, Tensor host_pool_mapping, "
        "Tensor local_topk_indices, Tensor query_req_indices, Tensor block_table, Tensor cu_kv_lengths, "
        "Tensor(a!) output, Tensor(b!) compact_indices, Tensor global_dequant_scale, int layer_idx, "
        "int total_kv_tokens, int tokens_per_block, int page_stride, int layer_id, int residual_dim, "
        "int num_pool_tokens) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("nvfp4_mla_kv_cache_gather", &tensorrt_llm::torch_ext::nvFp4MlaKvCacheGather);
    m.impl("nvfp4_mla_context_kv_cache_gather", &tensorrt_llm::torch_ext::nvFp4MlaContextKvCacheGather);
}
