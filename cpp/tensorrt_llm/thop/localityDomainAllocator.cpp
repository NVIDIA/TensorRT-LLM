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
#include "tensorrt_llm/runtime/locality_domain/locality_domain_utils.h"

#include <c10/cuda/CUDAGuard.h>

#include <exception>
#include <optional>

#if defined(_WIN32)
#define TLLM_LOCALITY_DOMAIN_ALLOCATOR_EXPORT __declspec(dllexport)
#else
#define TLLM_LOCALITY_DOMAIN_ALLOCATOR_EXPORT __attribute__((visibility("default")))
#endif

namespace torch_ext
{

void* localityDomainLocalizationAlloc(size_t size, int device, void* stream, int localityDomainId) noexcept
{
    try
    {
        TLLM_LOG_DEBUG("localityDomainLocalizationAlloc: allocating %zu bytes memory for localityDomainId=%d", size,
            localityDomainId);
        std::optional<at::cuda::CUDAGuard> deviceGuard;
        if (device >= 0)
        {
            deviceGuard.emplace(static_cast<c10::DeviceIndex>(device));
        }
        auto handle = tensorrt_llm::locality_domain::LocalizationHandle();
        void* outputPtr = nullptr;
        handle.localityDomainMalloc(&outputPtr, size, localityDomainId);
        return outputPtr;
    }
    catch (std::exception const& exception)
    {
        TLLM_LOG_EXCEPTION(exception);
    }
    catch (...)
    {
        TLLM_LOG_ERROR("Unknown exception thrown allocating locality domain-localized memory");
    }
    return nullptr;
}

void localityDomainLocalizationFree(void* ptr, size_t size, int device, void* stream, int localityDomainId) noexcept
{
    try
    {
        TLLM_LOG_DEBUG(
            "localityDomainLocalizationFree: free %zu bytes memory for localityDomainId=%d", size, localityDomainId);
        std::optional<at::cuda::CUDAGuard> deviceGuard;
        if (device >= 0)
        {
            deviceGuard.emplace(static_cast<c10::DeviceIndex>(device));
        }
        auto handle = tensorrt_llm::locality_domain::LocalizationHandle();
        handle.localityDomainFree(ptr);
    }
    catch (std::exception const& exception)
    {
        TLLM_LOG_EXCEPTION(exception);
    }
    catch (...)
    {
        TLLM_LOG_ERROR("Unknown exception thrown freeing locality domain-localized memory");
    }
}

} // namespace torch_ext

extern "C" TLLM_LOCALITY_DOMAIN_ALLOCATOR_EXPORT void* trtllm_locality_domain0_alloc(
    size_t size, int device, void* stream) noexcept
{
    return torch_ext::localityDomainLocalizationAlloc(size, device, stream, 0);
}

extern "C" TLLM_LOCALITY_DOMAIN_ALLOCATOR_EXPORT void* trtllm_locality_domain1_alloc(
    size_t size, int device, void* stream) noexcept
{
    return torch_ext::localityDomainLocalizationAlloc(size, device, stream, 1);
}

extern "C" TLLM_LOCALITY_DOMAIN_ALLOCATOR_EXPORT void trtllm_locality_domain0_free(
    void* ptr, size_t size, int device, void* stream) noexcept
{
    torch_ext::localityDomainLocalizationFree(ptr, size, device, stream, 0);
}

extern "C" TLLM_LOCALITY_DOMAIN_ALLOCATOR_EXPORT void trtllm_locality_domain1_free(
    void* ptr, size_t size, int device, void* stream) noexcept
{
    torch_ext::localityDomainLocalizationFree(ptr, size, device, stream, 1);
}
