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

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

void nvFp4MlaKvCacheGather(th::Tensor const& hostPoolPointers, th::Tensor const& hostPoolMapping,
    th::Tensor const& globalIndices, th::Tensor& output, th::Tensor& compactIndices,
    th::Tensor const& globalDequantScale, int64_t layerIdx, int64_t numPoolTokens)
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
        static_cast<int32_t>(globalIndices.size(1)), static_cast<int32_t>(output.size(2)), numPoolTokens, stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "nvfp4_mla_kv_cache_gather(Tensor host_pool_pointers, Tensor host_pool_mapping, Tensor global_indices, "
        "Tensor(a!) output, Tensor(b!) compact_indices, Tensor global_dequant_scale, int layer_idx, "
        "int num_pool_tokens) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("nvfp4_mla_kv_cache_gather", &tensorrt_llm::torch_ext::nvFp4MlaKvCacheGather);
}
