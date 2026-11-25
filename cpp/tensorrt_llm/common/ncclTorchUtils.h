/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

#include "tensorrt_llm/common/ncclUtils.h"

#if ENABLE_MULTI_DEVICE
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace common::nccl_util
{

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

// Creates a PyTorch tensor backed by an NCCL window buffer.
// The tensor will automatically release the buffer back to the pool when destroyed.
// This is analogous to torch_ext::create_userbuffers_tensor() but for NCCLWindowAllocator.
inline std::pair<torch::Tensor, NCCLWindowBuffer> createNCCLWindowTensor(
    std::shared_ptr<ncclComm_t> comm, at::IntArrayRef shape, torch::ScalarType dtype)
{
    // Calculate buffer size
    int64_t buffer_size
        = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>()) * torch::elementSize(dtype);

    // Calculate strides
    std::vector<int64_t> strides_vec(shape.size());
    if (!shape.empty())
    {
        strides_vec[shape.size() - 1] = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 1; --i)
        {
            strides_vec[i - 1] = strides_vec[i] * shape[i];
        }
    }

    // Request buffer from allocator
    auto& allocator = NCCLWindowAllocator::getInstance();
    NCCLWindowBuffer buffer;

    if (!comm || !*comm)
    {
        TLLM_LOG_DEBUG("[createNCCLWindowTensor] null comm; returning invalid buffer");
        return std::make_pair(torch::Tensor(), NCCLWindowBuffer());
    }

    // Expected resource failures, unknown capture state, and captures without a graph-pool owner
    // return an invalid buffer so callers can use their existing unregistered fallback. Other
    // programming errors must propagate instead of silently changing the collective implementation.
    buffer = allocator.requestBuffer(*comm, buffer_size, at::cuda::getCurrentCUDAStream().stream());

    // Defensive validation: ensure buffer is valid before proceeding
    if (!buffer.isValid())
    {
        TLLM_LOG_DEBUG("[createNCCLWindowTensor] invalid buffer returned from requestBuffer; returning invalid buffer");
        return std::make_pair(torch::Tensor(), NCCLWindowBuffer());
    }

    // Create custom deleter that releases the buffer
    auto deleter = [comm, ptr = buffer.ptr, leaseId = buffer.leaseId](void*)
    { NCCLWindowAllocator::getInstance().releaseBufferFromDestructor(*comm, ptr, leaseId); };

    // Create tensor from the buffer
    auto tensor = torch::from_blob(buffer.ptr, shape, strides_vec, deleter, torch::dtype(dtype).device(torch::kCUDA));
    auto const device = tensor.get_device();
    allocator.registerTensorLease(*comm, tensor.storage().unsafeGetStorageImpl(), buffer.ptr, buffer.leaseId, device);

    return std::make_pair(tensor, buffer);
}

#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

} // namespace common::nccl_util

TRTLLM_NAMESPACE_END

#endif // ENABLE_MULTI_DEVICE
