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

#pragma once

#include <cuda.h>

#include <cstddef>
#include <utility>

namespace tensorrt_llm
{

namespace locality_domain
{

class Localization;

class LocalizationHandle
{
public:
    LocalizationHandle();
    ~LocalizationHandle();

    // Delete copy constructor and copy assignment
    LocalizationHandle(LocalizationHandle const&) = delete;
    LocalizationHandle& operator=(LocalizationHandle const&) = delete;

    // Allow move constructor and move assignment
    LocalizationHandle(LocalizationHandle&&) noexcept;
    LocalizationHandle& operator=(LocalizationHandle&&) noexcept;

    //! Return whether both public compute and memory locality-domain APIs are usable.
    bool supportsLocalization() const;
    //! Return whether public locality-domain VMM allocation is usable.
    bool supportsMemoryLocalization() const;
    //! Return whether public locality-domain Green Context partitioning is usable.
    bool supportsComputeLocalization() const;

    void localityDomainMalloc(void** localizedDevPtr, size_t size, int localityDomainId);
    void localityDomainFree(void* localizedDevPtr);

    CUmemGenericAllocationHandle createLocalizedAllocationHandle(
        size_t size, int localityDomainId, unsigned int requestedHandleTypes, bool gpuDirectRDMACapable);
    CUmemGenericAllocationHandle createLocalizedAllocationHandle(size_t size, int localityDomainId,
        unsigned int requestedHandleTypes, bool gpuDirectRDMACapable, unsigned int usage);
    CUresult tryCreateLocalizedAllocationHandle(CUmemGenericAllocationHandle* handle, size_t size, int localityDomainId,
        unsigned int requestedHandleTypes, bool gpuDirectRDMACapable, unsigned int usage) noexcept;

    size_t getLocalizedAllocationGranularity(
        int localityDomainId, unsigned int requestedHandleTypes, bool gpuDirectRDMACapable, unsigned int usage);
    CUresult tryGetLocalizedAllocationGranularity(size_t* granularity, int localityDomainId,
        unsigned int requestedHandleTypes, bool gpuDirectRDMACapable, unsigned int usage) noexcept;

    //! Return a process-lifetime cached stream owned by the localization singleton.
    //! Callers must not destroy the returned stream.
    CUstream createLocalizedStream(int localityDomainId);
    //! Return (localized partition SM count, full-device SM count), or (0, 0) when unavailable.
    std::pair<unsigned int, unsigned int> getLocalityDomainComputeSmCounts(int localityDomainId) const noexcept;
    //! Return a borrowed process-lifetime remainder stream, or nullptr. The caller must not destroy it.
    CUstream getReservedRemainderStream();

private:
    Localization* mImpl;
};

} // namespace locality_domain

} // namespace tensorrt_llm
