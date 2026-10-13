/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/ugpu/ugpu_utils.h"

#include <c10/cuda/CUDAGuard.h>

#include <exception>
#include <optional>

#if defined(_WIN32)
#define TLLM_UGPU_ALLOCATOR_EXPORT __declspec(dllexport)
#else
#define TLLM_UGPU_ALLOCATOR_EXPORT __attribute__((visibility("default")))
#endif

namespace torch_ext
{

void* ugpuLocalizationAlloc(size_t size, int device, void* stream, int ugpuId) noexcept
{
    try
    {
        TLLM_LOG_DEBUG("ugpuLocalizationAlloc: allocating %zu bytes memory for ugpuId=%d", size, ugpuId);
        std::optional<at::cuda::CUDAGuard> deviceGuard;
        if (device >= 0)
        {
            deviceGuard.emplace(static_cast<c10::DeviceIndex>(device));
        }
        auto handle = tensorrt_llm::ugpu::UgpuLocalizationHandle();
        void* outputPtr = nullptr;
        handle.ugpuMalloc(&outputPtr, size, ugpuId);
        return outputPtr;
    }
    catch (std::exception const& exception)
    {
        TLLM_LOG_EXCEPTION(exception);
    }
    catch (...)
    {
        TLLM_LOG_ERROR("Unknown exception thrown allocating uGPU-localized memory");
    }
    return nullptr;
}

void ugpuLocalizationFree(void* ptr, size_t size, int device, void* stream, int ugpuId) noexcept
{
    try
    {
        TLLM_LOG_DEBUG("ugpuLocalizationFree: free %zu bytes memory for ugpuId=%d", size, ugpuId);
        std::optional<at::cuda::CUDAGuard> deviceGuard;
        if (device >= 0)
        {
            deviceGuard.emplace(static_cast<c10::DeviceIndex>(device));
        }
        auto handle = tensorrt_llm::ugpu::UgpuLocalizationHandle();
        handle.ugpuFree(ptr);
    }
    catch (std::exception const& exception)
    {
        TLLM_LOG_EXCEPTION(exception);
    }
    catch (...)
    {
        TLLM_LOG_ERROR("Unknown exception thrown freeing uGPU-localized memory");
    }
}

} // namespace torch_ext

extern "C" TLLM_UGPU_ALLOCATOR_EXPORT void* trtllm_ugpu0_alloc(size_t size, int device, void* stream) noexcept
{
    return torch_ext::ugpuLocalizationAlloc(size, device, stream, 0);
}

extern "C" TLLM_UGPU_ALLOCATOR_EXPORT void* trtllm_ugpu1_alloc(size_t size, int device, void* stream) noexcept
{
    return torch_ext::ugpuLocalizationAlloc(size, device, stream, 1);
}

extern "C" TLLM_UGPU_ALLOCATOR_EXPORT void trtllm_ugpu0_free(void* ptr, size_t size, int device, void* stream) noexcept
{
    torch_ext::ugpuLocalizationFree(ptr, size, device, stream, 0);
}

extern "C" TLLM_UGPU_ALLOCATOR_EXPORT void trtllm_ugpu1_free(void* ptr, size_t size, int device, void* stream) noexcept
{
    torch_ext::ugpuLocalizationFree(ptr, size, device, stream, 1);
}
