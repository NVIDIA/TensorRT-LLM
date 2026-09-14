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

#include "tensorrt_llm/runtime/locality_domain/locality_domain_utils.h"

#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/locality_domain/localityDomainResourceConfig.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace tensorrt_llm::locality_domain
{

namespace
{

#if CUDA_VERSION >= 13040
constexpr int kLocalityDomainApiVersion = 13'040;
constexpr int kLocalityDomainCount = detail::kLocalityDomainCount;
#endif

enum class LocalityDomainStreamCreateMethod
{
    kStrict,
    kBalanced,
};

LocalityDomainStreamCreateMethod parseStreamCreateMethod()
{
    char const* const value = std::getenv("TLLM_LOCALITY_DOMAIN_STREAM_CREATE_METHOD");
    if (value == nullptr)
    {
        return LocalityDomainStreamCreateMethod::kStrict;
    }

    std::string const method{value};
    if (method == "Balanced" || method == "balanced" || method == "BALANCED")
    {
        return LocalityDomainStreamCreateMethod::kBalanced;
    }

    if (method == "GreenContext" || method == "greencontext" || method == "green" || method == "locality_domain"
        || method == "LOCALITY_DOMAIN" || method == "3-part-gc" || method == "strict")
    {
        return LocalityDomainStreamCreateMethod::kStrict;
    }

    TLLM_LOG_WARNING(
        "[Localization] Unknown TLLM_LOCALITY_DOMAIN_STREAM_CREATE_METHOD=%s; using public strict Green Context split",
        value);
    return LocalityDomainStreamCreateMethod::kStrict;
}

struct InstanceKey
{
    int device{};
    CUcontext context{};
};

struct InstanceKeyLess
{
    bool operator()(InstanceKey const& lhs, InstanceKey const& rhs) const
    {
        if (lhs.device != rhs.device)
        {
            return lhs.device < rhs.device;
        }
        return std::less<CUcontext>{}(lhs.context, rhs.context);
    }
};

#if CUDA_VERSION >= 13040

struct AllocationKey
{
    int device{};
    CUcontext context{};
    CUdeviceptr pointer{};
};

struct AllocationKeyLess
{
    bool operator()(AllocationKey const& lhs, AllocationKey const& rhs) const
    {
        if (lhs.device != rhs.device)
        {
            return lhs.device < rhs.device;
        }
        if (lhs.context != rhs.context)
        {
            return std::less<CUcontext>{}(lhs.context, rhs.context);
        }
        return lhs.pointer < rhs.pointer;
    }
};

struct VmmAllocation
{
    size_t alignedSize{};
    int device{};
    CUcontext context{};
    bool mapped{};
};

std::map<AllocationKey, VmmAllocation, AllocationKeyLess>& getVmmAllocations()
{
    static auto* const allocations = new std::map<AllocationKey, VmmAllocation, AllocationKeyLess>;
    return *allocations;
}

std::mutex& getVmmAllocationMutex()
{
    static auto* const mutex = new std::mutex;
    return *mutex;
}

#endif

CUresult queryCurrentDeviceAndContext(int* device, CUcontext* context)
{
    if (device == nullptr || context == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    cudaError_t const runtimeResult = cudaFree(nullptr);
    if (runtimeResult != cudaSuccess)
    {
        TLLM_LOG_WARNING(
            "[Localization] Failed to initialize the current CUDA context: %s", cudaGetErrorString(runtimeResult));
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    int runtimeDevice{};
    cudaError_t const deviceResult = cudaGetDevice(&runtimeDevice);
    if (deviceResult != cudaSuccess)
    {
        TLLM_LOG_WARNING(
            "[Localization] Failed to query the current CUDA device: %s", cudaGetErrorString(deviceResult));
        return CUDA_ERROR_INVALID_DEVICE;
    }

    CUcontext currentContext{};
    CUresult const contextResult = cuCtxGetCurrent(&currentContext);
    if (contextResult != CUDA_SUCCESS)
    {
        return contextResult;
    }
    if (currentContext == nullptr)
    {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    *device = runtimeDevice;
    *context = currentContext;
    return CUDA_SUCCESS;
}

#if CUDA_VERSION >= 13040

template <typename Proc>
bool loadDriverProc(char const* name, Proc* proc)
{
    void* address{};
    CUdriverProcAddressQueryResult queryResult{};
    CUresult const result
        = cuGetProcAddress(name, &address, kLocalityDomainApiVersion, CU_GET_PROC_ADDRESS_DEFAULT, &queryResult);
    if (result != CUDA_SUCCESS || queryResult != CU_GET_PROC_ADDRESS_SUCCESS || address == nullptr)
    {
        TLLM_LOG_WARNING("[Localization] CUDA 13.4 driver entry point %s is unavailable (result=%d, status=%d)", name,
            static_cast<int>(result), static_cast<int>(queryResult));
        *proc = nullptr;
        return false;
    }

    *proc = reinterpret_cast<Proc>(address);
    return true;
}

class GreenContextApi
{
public:
    using DeviceGetDevResource = CUresult(CUDAAPI*)(CUdevice device, CUdevResource* resource, CUdevResourceType type);
    using DevSmResourceSplit
        = CUresult(CUDAAPI*)(CUdevResource* result, unsigned int nbGroups, CUdevResource const* input,
            CUdevResource* remainder, unsigned int flags, CU_DEV_SM_RESOURCE_GROUP_PARAMS* groupParams);
    using DevResourceGenerateDesc
        = CUresult(CUDAAPI*)(CUdevResourceDesc* desc, CUdevResource* resources, unsigned int nbResources);
    using GreenCtxCreate
        = CUresult(CUDAAPI*)(CUgreenCtx* greenContext, CUdevResourceDesc desc, CUdevice device, unsigned int flags);
    using GreenCtxDestroy = CUresult(CUDAAPI*)(CUgreenCtx greenContext);
    using GreenCtxStreamCreate
        = CUresult(CUDAAPI*)(CUstream* stream, CUgreenCtx greenContext, unsigned int flags, int priority);

    bool load()
    {
        return loadDriverProc("cuDeviceGetDevResource", &deviceGetDevResource)
            && loadDriverProc("cuDevSmResourceSplit", &devSmResourceSplit)
            && loadDriverProc("cuDevResourceGenerateDesc", &devResourceGenerateDesc)
            && loadDriverProc("cuGreenCtxCreate", &greenCtxCreate)
            && loadDriverProc("cuGreenCtxDestroy", &greenCtxDestroy)
            && loadDriverProc("cuGreenCtxStreamCreate", &greenCtxStreamCreate);
    }

    DeviceGetDevResource deviceGetDevResource{};
    DevSmResourceSplit devSmResourceSplit{};
    DevResourceGenerateDesc devResourceGenerateDesc{};
    GreenCtxCreate greenCtxCreate{};
    GreenCtxDestroy greenCtxDestroy{};
    GreenCtxStreamCreate greenCtxStreamCreate{};
};

class GreenContextPartitions
{
public:
    GreenContextPartitions() = default;
    GreenContextPartitions(GreenContextPartitions const&) = delete;
    GreenContextPartitions& operator=(GreenContextPartitions const&) = delete;

    ~GreenContextPartitions()
    {
        reset();
    }

    bool initialize(CUdevice device, LocalityDomainStreamCreateMethod method, unsigned int localityDomainSmCount)
    {
        mMethod = method;
        if (!mApi.load())
        {
            return false;
        }

        CUdevResource fullResource{};
        CUresult result = mApi.deviceGetDevResource(device, &fullResource, CU_DEV_RESOURCE_TYPE_SM);
        if (result != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING("[Localization] cuDeviceGetDevResource failed (result=%d)", static_cast<int>(result));
            return false;
        }
        if (fullResource.type != CU_DEV_RESOURCE_TYPE_SM || fullResource.sm.smCount == 0)
        {
            TLLM_LOG_WARNING("[Localization] Device returned an invalid SM resource");
            return false;
        }

        detail::SmResourceGroupParams groupParams = detail::makeStrictSmResourceGroupParams();
        if (method == LocalityDomainStreamCreateMethod::kBalanced)
        {
            if (!detail::isBalancedSmCountValid(fullResource.sm.smCount))
            {
                TLLM_LOG_WARNING(
                    "[Localization] Balanced public split requires an even per-group SM count, got "
                    "total SM count %u",
                    fullResource.sm.smCount);
                return false;
            }
            groupParams = detail::makeBalancedSmResourceGroupParams(fullResource.sm.smCount);
        }

        result = mApi.devSmResourceSplit(mLocalizedResources.data(), kLocalityDomainCount, &fullResource,
            &mRemainderResource,
            /*flags=*/0, groupParams.data());
        if (result != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING(
                "[Localization] Public locality-domain SM split failed (result=%d)", static_cast<int>(result));
            return false;
        }

        for (int localityDomainId = 0; localityDomainId < kLocalityDomainCount; ++localityDomainId)
        {
            CUdevResource const& resource = mLocalizedResources[localityDomainId];
            if (resource.type != CU_DEV_RESOURCE_TYPE_SM || resource.sm.smCount == 0
                || resource.sm.localityDomainId != static_cast<unsigned int>(localityDomainId)
                || (resource.sm.flags & CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID) == 0)
            {
                TLLM_LOG_WARNING(
                    "[Localization] Public split returned an invalid resource for locality domain%d", localityDomainId);
                reset();
                return false;
            }

            if (method == LocalityDomainStreamCreateMethod::kStrict && resource.sm.smCount != localityDomainSmCount)
            {
                TLLM_LOG_WARNING(
                    "[Localization] Strict public split returned %u SMs for locality domain%d, expected "
                    "locality-domain "
                    "attribute value %u",
                    resource.sm.smCount, localityDomainId, localityDomainSmCount);
                reset();
                return false;
            }

            if (method == LocalityDomainStreamCreateMethod::kBalanced
                && resource.sm.smCount != fullResource.sm.smCount / static_cast<unsigned int>(kLocalityDomainCount))
            {
                TLLM_LOG_WARNING(
                    "[Localization] Balanced public split returned %u SMs for locality domain%d, expected %u",
                    resource.sm.smCount, localityDomainId,
                    fullResource.sm.smCount / static_cast<unsigned int>(kLocalityDomainCount));
                reset();
                return false;
            }
        }

        if (method == LocalityDomainStreamCreateMethod::kStrict)
        {
            if (!detail::isStrictSplitCountValid(fullResource.sm.smCount, localityDomainSmCount, getRemainderSmCount()))
            {
                TLLM_LOG_WARNING(
                    "[Localization] Strict public split counts do not match two complete locality domains "
                    "(total=%u, per-domain=%u, remainder=%u)",
                    fullResource.sm.smCount, localityDomainSmCount, getRemainderSmCount());
                reset();
                return false;
            }
        }
        else if (getRemainderSmCount() != 0)
        {
            TLLM_LOG_WARNING(
                "[Localization] Balanced public split unexpectedly left %u remainder SMs", getRemainderSmCount());
            reset();
            return false;
        }

        TLLM_LOG_INFO(
            "[Localization] Public %s split: total=%u SM, locality domain 0=%u SM, locality domain 1=%u SM, "
            "remainder=%u SM",
            method == LocalityDomainStreamCreateMethod::kStrict ? "strict" : "balanced", fullResource.sm.smCount,
            mLocalizedResources[0].sm.smCount, mLocalizedResources[1].sm.smCount, getRemainderSmCount());

        for (int localityDomainId = 0; localityDomainId < kLocalityDomainCount; ++localityDomainId)
        {
            result = mApi.devResourceGenerateDesc(
                &mLocalizedDescriptors[localityDomainId], &mLocalizedResources[localityDomainId], /*nbResources=*/1);
            if (result == CUDA_SUCCESS)
            {
                result = mApi.greenCtxCreate(&mGreenContexts[localityDomainId], mLocalizedDescriptors[localityDomainId],
                    device, CU_GREEN_CTX_DEFAULT_STREAM);
            }
            if (result != CUDA_SUCCESS)
            {
                TLLM_LOG_WARNING("[Localization] Failed to create Green Context for locality domain%d (result=%d)",
                    localityDomainId, static_cast<int>(result));
                reset();
                return false;
            }
        }

        if (method == LocalityDomainStreamCreateMethod::kStrict && getRemainderSmCount() > 0)
        {
            result = mApi.devResourceGenerateDesc(&mRemainderDescriptor, &mRemainderResource, /*nbResources=*/1);
            if (result == CUDA_SUCCESS)
            {
                result = mApi.greenCtxCreate(
                    &mRemainderGreenContext, mRemainderDescriptor, device, CU_GREEN_CTX_DEFAULT_STREAM);
            }
            if (result == CUDA_SUCCESS)
            {
                result = mApi.greenCtxStreamCreate(
                    &mRemainderStream, mRemainderGreenContext, CU_STREAM_NON_BLOCKING, /*priority=*/0);
            }
            if (result != CUDA_SUCCESS)
            {
                TLLM_LOG_WARNING(
                    "[Localization] Failed to create strict split remainder Green Context/stream (result=%d)",
                    static_cast<int>(result));
                reset();
                return false;
            }
        }

        mSupported = true;
        return true;
    }

    CUresult createStream(CUstream* stream, int localityDomainId)
    {
        if (stream == nullptr || localityDomainId < 0 || localityDomainId >= kLocalityDomainCount)
        {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (!mSupported)
        {
            return CUDA_ERROR_NOT_SUPPORTED;
        }

        std::lock_guard<std::mutex> const lock{mStreamMutex};
        if (mLocalizedStreams[localityDomainId] == nullptr)
        {
            CUstream newStream{};
            CUresult const result = mApi.greenCtxStreamCreate(
                &newStream, mGreenContexts[localityDomainId], CU_STREAM_NON_BLOCKING, /*priority=*/0);
            if (result != CUDA_SUCCESS)
            {
                if (newStream != nullptr)
                {
                    static_cast<void>(cuStreamDestroy(newStream));
                }
                return result;
            }
            mLocalizedStreams[localityDomainId] = newStream;
        }
        *stream = mLocalizedStreams[localityDomainId];
        return CUDA_SUCCESS;
    }

    CUstream getRemainderStream() const
    {
        if (!mSupported || mMethod != LocalityDomainStreamCreateMethod::kStrict)
        {
            return nullptr;
        }
        return mRemainderStream;
    }

    std::pair<unsigned int, unsigned int> getSmCounts(int localityDomainId) const noexcept
    {
        if (!mSupported || localityDomainId < 0 || localityDomainId >= kLocalityDomainCount)
        {
            return {};
        }

        CUdevResource const& localizedResource = mLocalizedResources[localityDomainId];
        if (localizedResource.type != CU_DEV_RESOURCE_TYPE_SM || localizedResource.sm.smCount == 0)
        {
            return {};
        }

        unsigned int totalSmCount = getRemainderSmCount();
        for (auto const& resource : mLocalizedResources)
        {
            if (resource.type != CU_DEV_RESOURCE_TYPE_SM || resource.sm.smCount == 0)
            {
                return {};
            }
            totalSmCount += resource.sm.smCount;
        }
        return {localizedResource.sm.smCount, totalSmCount};
    }

private:
    unsigned int getRemainderSmCount() const
    {
        return mRemainderResource.type == CU_DEV_RESOURCE_TYPE_SM ? mRemainderResource.sm.smCount : 0;
    }

    void reset() noexcept
    {
        for (auto& stream : mLocalizedStreams)
        {
            if (stream != nullptr)
            {
                CUresult const result = cuStreamDestroy(stream);
                if (result != CUDA_SUCCESS)
                {
                    TLLM_LOG_WARNING(
                        "[Localization] Failed to destroy localized stream (result=%d)", static_cast<int>(result));
                }
                stream = nullptr;
            }
        }

        if (mRemainderStream != nullptr)
        {
            CUresult const result = cuStreamDestroy(mRemainderStream);
            if (result != CUDA_SUCCESS)
            {
                TLLM_LOG_WARNING(
                    "[Localization] Failed to destroy remainder stream (result=%d)", static_cast<int>(result));
            }
            mRemainderStream = nullptr;
        }

        if (mRemainderGreenContext != nullptr && mApi.greenCtxDestroy != nullptr)
        {
            CUresult const result = mApi.greenCtxDestroy(mRemainderGreenContext);
            if (result != CUDA_SUCCESS)
            {
                TLLM_LOG_WARNING(
                    "[Localization] Failed to destroy remainder Green Context (result=%d)", static_cast<int>(result));
            }
            mRemainderGreenContext = nullptr;
        }

        if (mApi.greenCtxDestroy != nullptr)
        {
            for (auto& greenContext : mGreenContexts)
            {
                if (greenContext != nullptr)
                {
                    CUresult const result = mApi.greenCtxDestroy(greenContext);
                    if (result != CUDA_SUCCESS)
                    {
                        TLLM_LOG_WARNING(
                            "[Localization] Failed to destroy Green Context (result=%d)", static_cast<int>(result));
                    }
                    greenContext = nullptr;
                }
            }
        }
        mSupported = false;
    }

    GreenContextApi mApi;
    LocalityDomainStreamCreateMethod mMethod{LocalityDomainStreamCreateMethod::kStrict};
    std::array<CUdevResource, kLocalityDomainCount> mLocalizedResources{};
    std::array<CUdevResourceDesc, kLocalityDomainCount> mLocalizedDescriptors{};
    std::array<CUgreenCtx, kLocalityDomainCount> mGreenContexts{};
    std::array<CUstream, kLocalityDomainCount> mLocalizedStreams{};
    CUdevResource mRemainderResource{};
    CUdevResourceDesc mRemainderDescriptor{};
    CUgreenCtx mRemainderGreenContext{};
    CUstream mRemainderStream{};
    bool mSupported{};
    std::mutex mStreamMutex;
};

#endif // CUDA_VERSION >= 13040

} // namespace

class Localization
{
public:
    Localization(int device, CUcontext context)
        : mDevice{device}
        , mContext{context}
        , mStreamCreateMethod{parseStreamCreateMethod()}
    {
        initialize();
    }

    bool supportsMemoryLocalization() const noexcept
    {
        return mMemoryLocalizationSupported;
    }

    bool supportsComputeLocalization() const noexcept
    {
        return mComputeLocalizationSupported;
    }

    bool supportsLocalization() const noexcept
    {
        return supportsMemoryLocalization() && supportsComputeLocalization();
    }

    CUresult localizedDeviceAlloc(void** localizedDevPtr, size_t size, int localityDomainId) noexcept
    {
        if (localizedDevPtr == nullptr || size == 0)
        {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *localizedDevPtr = nullptr;

        if (localityDomainId == -1)
        {
            return cuMemAlloc(reinterpret_cast<CUdeviceptr*>(localizedDevPtr), size);
        }

#if CUDA_VERSION >= 13040
        if (!mMemoryLocalizationSupported)
        {
            return CUDA_ERROR_NOT_SUPPORTED;
        }

        CUmemAllocationProp prop{};
        CUresult result = makeAllocationProp(&prop, localityDomainId, CU_MEM_HANDLE_TYPE_NONE,
            /*gpuDirectRDMACapable=*/false, /*usage=*/0);
        if (result != CUDA_SUCCESS)
        {
            return result;
        }

        size_t granularity{};
        result = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (result != CUDA_SUCCESS)
        {
            return result;
        }
        if (granularity == 0 || size > std::numeric_limits<size_t>::max() - (granularity - 1))
        {
            return CUDA_ERROR_INVALID_VALUE;
        }
        size_t const alignedSize = ((size + granularity - 1) / granularity) * granularity;

        CUdeviceptr address{};
        CUmemGenericAllocationHandle allocationHandle{};
        bool mapped = false;

        result = cuMemAddressReserve(&address, alignedSize, /*alignment=*/0, /*addr=*/0, /*flags=*/0);
        if (result == CUDA_SUCCESS)
        {
            result = cuMemCreate(&allocationHandle, alignedSize, &prop, /*flags=*/0);
        }
        if (result == CUDA_SUCCESS)
        {
            result = cuMemMap(address, alignedSize, /*offset=*/0, allocationHandle, /*flags=*/0);
            mapped = result == CUDA_SUCCESS;
        }
        if (result == CUDA_SUCCESS)
        {
            CUmemAccessDesc access{};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = mDevice;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            result = cuMemSetAccess(address, alignedSize, &access, /*count=*/1);
        }
        if (result == CUDA_SUCCESS)
        {
            result = cuMemRelease(allocationHandle);
            if (result == CUDA_SUCCESS)
            {
                allocationHandle = 0;
            }
        }

        if (result == CUDA_SUCCESS)
        {
            try
            {
                AllocationKey const key{mDevice, mContext, address};
                VmmAllocation const allocation{alignedSize, mDevice, mContext, mapped};
                std::lock_guard<std::mutex> const lock{getVmmAllocationMutex()};
                bool const inserted = getVmmAllocations().emplace(key, allocation).second;
                if (!inserted)
                {
                    result = CUDA_ERROR_INVALID_VALUE;
                }
            }
            catch (...)
            {
                result = CUDA_ERROR_OUT_OF_MEMORY;
            }
        }

        if (result != CUDA_SUCCESS)
        {
            if (mapped)
            {
                static_cast<void>(cuMemUnmap(address, alignedSize));
            }
            if (allocationHandle != 0)
            {
                static_cast<void>(cuMemRelease(allocationHandle));
            }
            if (address != 0)
            {
                static_cast<void>(cuMemAddressFree(address, alignedSize));
            }
            return result;
        }

        *localizedDevPtr = reinterpret_cast<void*>(address);
        return CUDA_SUCCESS;
#else
        static_cast<void>(localityDomainId);
        return CUDA_ERROR_NOT_SUPPORTED;
#endif
    }

    CUresult localizedDeviceFree(void* localizedDevPtr)
    {
        if (localizedDevPtr == nullptr)
        {
            return CUDA_SUCCESS;
        }

#if CUDA_VERSION >= 13040
        CUresult result = checkCurrentContext();
        if (result != CUDA_SUCCESS)
        {
            return result;
        }

        CUdeviceptr const address = reinterpret_cast<CUdeviceptr>(localizedDevPtr);
        AllocationKey const key{mDevice, mContext, address};
        std::lock_guard<std::mutex> const lock{getVmmAllocationMutex()};
        auto& allocations = getVmmAllocations();
        auto allocationIt = allocations.find(key);
        if (allocationIt == allocations.end())
        {
            return cuMemFree(address);
        }

        VmmAllocation& allocation = allocationIt->second;
        if (allocation.device != mDevice || allocation.context != mContext)
        {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        if (allocation.mapped)
        {
            result = cuMemUnmap(address, allocation.alignedSize);
            if (result != CUDA_SUCCESS)
            {
                return result;
            }
            allocation.mapped = false;
        }

        result = cuMemAddressFree(address, allocation.alignedSize);
        if (result == CUDA_SUCCESS)
        {
            allocations.erase(allocationIt);
        }
        return result;
#else
        return cuMemFree(reinterpret_cast<CUdeviceptr>(localizedDevPtr));
#endif
    }

    CUresult tryCreateLocalizedAllocationHandle(CUmemGenericAllocationHandle* handle, size_t size, int localityDomainId,
        unsigned int requestedHandleTypes, bool gpuDirectRDMACapable, unsigned int usage) noexcept
    {
        if (handle == nullptr || size == 0)
        {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *handle = 0;

#if CUDA_VERSION >= 13040
        CUmemAllocationProp prop{};
        CUresult result
            = makeAllocationProp(&prop, localityDomainId, requestedHandleTypes, gpuDirectRDMACapable, usage);
        if (result != CUDA_SUCCESS)
        {
            return result;
        }

        size_t granularity{};
        result = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (result != CUDA_SUCCESS)
        {
            return result;
        }
        if (granularity == 0 || (size % granularity) != 0)
        {
            return CUDA_ERROR_INVALID_VALUE;
        }
        return cuMemCreate(handle, size, &prop, /*flags=*/0);
#else
        static_cast<void>(localityDomainId);
        static_cast<void>(requestedHandleTypes);
        static_cast<void>(gpuDirectRDMACapable);
        static_cast<void>(usage);
        return CUDA_ERROR_NOT_SUPPORTED;
#endif
    }

    CUresult tryGetLocalizedAllocationGranularity(size_t* granularity, int localityDomainId,
        unsigned int requestedHandleTypes, bool gpuDirectRDMACapable, unsigned int usage) noexcept
    {
        if (granularity == nullptr)
        {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *granularity = 0;

#if CUDA_VERSION >= 13040
        CUmemAllocationProp prop{};
        CUresult const result
            = makeAllocationProp(&prop, localityDomainId, requestedHandleTypes, gpuDirectRDMACapable, usage);
        if (result != CUDA_SUCCESS)
        {
            return result;
        }
        return cuMemGetAllocationGranularity(granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
#else
        static_cast<void>(localityDomainId);
        static_cast<void>(requestedHandleTypes);
        static_cast<void>(gpuDirectRDMACapable);
        static_cast<void>(usage);
        return CUDA_ERROR_NOT_SUPPORTED;
#endif
    }

    unsigned int getAutomaticAllocationUsage(bool gpuDirectRDMACapable) const noexcept
    {
#if CUDA_VERSION >= 13040
        if (gpuDirectRDMACapable && !mDefaultRdmaSupportsLocalizedMemory)
        {
            return CU_MEM_CREATE_USAGE_GPU_DIRECT_RDMA_OVER_PCIE;
        }
#else
        static_cast<void>(gpuDirectRDMACapable);
#endif
        return 0;
    }

    CUresult createLocalizedStream(CUstream* stream, int localityDomainId)
    {
        if (stream == nullptr)
        {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *stream = nullptr;

#if CUDA_VERSION >= 13040
        CUresult const contextResult = checkCurrentContext();
        if (contextResult != CUDA_SUCCESS)
        {
            return contextResult;
        }
        return mPartitions.createStream(stream, localityDomainId);
#else
        static_cast<void>(localityDomainId);
        return CUDA_ERROR_NOT_SUPPORTED;
#endif
    }

    CUstream getReservedRemainderStream() const noexcept
    {
#if CUDA_VERSION >= 13040
        return mPartitions.getRemainderStream();
#else
        return nullptr;
#endif
    }

    std::pair<unsigned int, unsigned int> getComputeSmCounts(int localityDomainId) const noexcept
    {
#if CUDA_VERSION >= 13040
        return mPartitions.getSmCounts(localityDomainId);
#else
        static_cast<void>(localityDomainId);
        return {};
#endif
    }

    static Localization* getLocalization()
    {
        int device{};
        CUcontext context{};
        TLLM_CU_CHECK(queryCurrentDeviceAndContext(&device, &context));

        static auto* const mutex = new std::mutex;
        static auto* const localizations = new std::map<InstanceKey, Localization*, InstanceKeyLess>;

        InstanceKey const key{device, context};
        std::lock_guard<std::mutex> const lock{*mutex};
        auto const it = localizations->find(key);
        if (it != localizations->end())
        {
            return it->second;
        }

        auto* const localization = new Localization(device, context);
        localizations->emplace(key, localization);
        return localization;
    }

private:
    void initialize()
    {
#if CUDA_VERSION >= 13040
        int localityDomainCount{};
        CUresult result
            = cuDeviceGetAttribute(&localityDomainCount, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, mDevice);
        if (result != CUDA_SUCCESS || localityDomainCount < kLocalityDomainCount)
        {
            TLLM_LOG_INFO(
                "[Localization] Device %d does not expose two public locality domains (result=%d, "
                "count=%d)",
                mDevice, static_cast<int>(result), localityDomainCount);
            return;
        }
        mLocalityDomainCount = localityDomainCount;
        mMemoryLocalizationSupported = true;

        int defaultRdmaSupported{};
        result = cuDeviceGetAttribute(
            &defaultRdmaSupported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_LOCALIZED_MEMORY_SUPPORTED, mDevice);
        if (result == CUDA_SUCCESS)
        {
            mDefaultRdmaSupportsLocalizedMemory = defaultRdmaSupported != 0;
        }

        int localityDomainSmCount{};
        result = cuDeviceGetAttribute(
            &localityDomainSmCount, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_MULTIPROCESSOR_COUNT, mDevice);
        if (result != CUDA_SUCCESS || localityDomainSmCount <= 0)
        {
            TLLM_LOG_WARNING("[Localization] Failed to query locality-domain SM count (result=%d, count=%d)",
                static_cast<int>(result), localityDomainSmCount);
            return;
        }

        mComputeLocalizationSupported
            = mPartitions.initialize(mDevice, mStreamCreateMethod, static_cast<unsigned int>(localityDomainSmCount));
        if (!mComputeLocalizationSupported)
        {
            TLLM_LOG_WARNING(
                "[Localization] Public localized VMM is available, but public Green Context "
                "partitioning is unavailable");
        }
#else
        TLLM_LOG_INFO("[Localization] Built with CUDA %d; public locality-domain support requires CUDA 13.4 headers",
            CUDA_VERSION);
#endif
    }

    CUresult checkCurrentContext() const noexcept
    {
        CUcontext currentContext{};
        CUresult result = cuCtxGetCurrent(&currentContext);
        if (result != CUDA_SUCCESS)
        {
            return result;
        }
        if (currentContext != mContext)
        {
            return CUDA_ERROR_INVALID_CONTEXT;
        }

        CUdevice currentDevice{};
        result = cuCtxGetDevice(&currentDevice);
        if (result != CUDA_SUCCESS)
        {
            return result;
        }
        return currentDevice == mDevice ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT;
    }

#if CUDA_VERSION >= 13040
    CUresult makeAllocationProp(CUmemAllocationProp* prop, int localityDomainId, unsigned int requestedHandleTypes,
        bool gpuDirectRDMACapable, unsigned int usage) const noexcept
    {
        if (prop == nullptr || localityDomainId < 0 || localityDomainId >= kLocalityDomainCount || mDevice < 0
            || mDevice > std::numeric_limits<unsigned char>::max()
            || usage > std::numeric_limits<unsigned short>::max())
        {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (!mMemoryLocalizationSupported)
        {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        if (localityDomainId >= mLocalityDomainCount)
        {
            return CUDA_ERROR_INVALID_VALUE;
        }

        CUresult const contextResult = checkCurrentContext();
        if (contextResult != CUDA_SUCCESS)
        {
            return contextResult;
        }

        *prop = {};
        prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop->requestedHandleTypes = static_cast<CUmemAllocationHandleType>(requestedHandleTypes);
        prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
        prop->location.localized.deviceId = static_cast<unsigned char>(mDevice);
        prop->location.localized.localityDomainId = static_cast<unsigned char>(localityDomainId);
        prop->allocFlags.gpuDirectRDMACapable = gpuDirectRDMACapable ? 1 : 0;
        prop->allocFlags.usage = static_cast<unsigned short>(usage);
        return CUDA_SUCCESS;
    }
#endif

    int mDevice{};
    CUcontext mContext{};
    LocalityDomainStreamCreateMethod mStreamCreateMethod{LocalityDomainStreamCreateMethod::kStrict};
    bool mMemoryLocalizationSupported{};
    bool mComputeLocalizationSupported{};
#if CUDA_VERSION >= 13040
    bool mDefaultRdmaSupportsLocalizedMemory{};
    int mLocalityDomainCount{};
    GreenContextPartitions mPartitions;
#endif
};

LocalizationHandle::LocalizationHandle()
    : mImpl{Localization::getLocalization()}
{
}

LocalizationHandle::~LocalizationHandle() = default;

LocalizationHandle::LocalizationHandle(LocalizationHandle&& other) noexcept
    : mImpl{other.mImpl}
{
    other.mImpl = nullptr;
}

LocalizationHandle& LocalizationHandle::operator=(LocalizationHandle&& other) noexcept
{
    if (this != &other)
    {
        mImpl = other.mImpl;
        other.mImpl = nullptr;
    }
    return *this;
}

bool LocalizationHandle::supportsLocalization() const
{
    return mImpl != nullptr && mImpl->supportsLocalization();
}

bool LocalizationHandle::supportsMemoryLocalization() const
{
    return mImpl != nullptr && mImpl->supportsMemoryLocalization();
}

bool LocalizationHandle::supportsComputeLocalization() const
{
    return mImpl != nullptr && mImpl->supportsComputeLocalization();
}

void LocalizationHandle::localityDomainMalloc(void** localizedDevPtr, size_t size, int localityDomainId)
{
    TLLM_CHECK_WITH_INFO(mImpl != nullptr, "Cannot use a moved-from LocalizationHandle");
    TLLM_CU_CHECK(mImpl->localizedDeviceAlloc(localizedDevPtr, size, localityDomainId));
}

void LocalizationHandle::localityDomainFree(void* localizedDevPtr)
{
    TLLM_CHECK_WITH_INFO(mImpl != nullptr, "Cannot use a moved-from LocalizationHandle");
    TLLM_CU_CHECK(mImpl->localizedDeviceFree(localizedDevPtr));
}

CUmemGenericAllocationHandle LocalizationHandle::createLocalizedAllocationHandle(
    size_t size, int localityDomainId, unsigned int requestedHandleTypes, bool gpuDirectRDMACapable)
{
    TLLM_CHECK_WITH_INFO(mImpl != nullptr, "Cannot use a moved-from LocalizationHandle");
    return createLocalizedAllocationHandle(size, localityDomainId, requestedHandleTypes, gpuDirectRDMACapable,
        mImpl->getAutomaticAllocationUsage(gpuDirectRDMACapable));
}

CUmemGenericAllocationHandle LocalizationHandle::createLocalizedAllocationHandle(
    size_t size, int localityDomainId, unsigned int requestedHandleTypes, bool gpuDirectRDMACapable, unsigned int usage)
{
    CUmemGenericAllocationHandle handle{};
    TLLM_CU_CHECK(tryCreateLocalizedAllocationHandle(
        &handle, size, localityDomainId, requestedHandleTypes, gpuDirectRDMACapable, usage));
    return handle;
}

CUresult LocalizationHandle::tryCreateLocalizedAllocationHandle(CUmemGenericAllocationHandle* handle, size_t size,
    int localityDomainId, unsigned int requestedHandleTypes, bool gpuDirectRDMACapable, unsigned int usage) noexcept
{
    if (mImpl == nullptr)
    {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    return mImpl->tryCreateLocalizedAllocationHandle(
        handle, size, localityDomainId, requestedHandleTypes, gpuDirectRDMACapable, usage);
}

size_t LocalizationHandle::getLocalizedAllocationGranularity(
    int localityDomainId, unsigned int requestedHandleTypes, bool gpuDirectRDMACapable, unsigned int usage)
{
    size_t granularity{};
    TLLM_CU_CHECK(tryGetLocalizedAllocationGranularity(
        &granularity, localityDomainId, requestedHandleTypes, gpuDirectRDMACapable, usage));
    return granularity;
}

CUresult LocalizationHandle::tryGetLocalizedAllocationGranularity(size_t* granularity, int localityDomainId,
    unsigned int requestedHandleTypes, bool gpuDirectRDMACapable, unsigned int usage) noexcept
{
    if (mImpl == nullptr)
    {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    return mImpl->tryGetLocalizedAllocationGranularity(
        granularity, localityDomainId, requestedHandleTypes, gpuDirectRDMACapable, usage);
}

CUstream LocalizationHandle::createLocalizedStream(int localityDomainId)
{
    TLLM_CHECK_WITH_INFO(mImpl != nullptr, "Cannot use a moved-from LocalizationHandle");
    CUstream stream{};
    TLLM_CU_CHECK(mImpl->createLocalizedStream(&stream, localityDomainId));
    return stream;
}

std::pair<unsigned int, unsigned int> LocalizationHandle::getLocalityDomainComputeSmCounts(
    int localityDomainId) const noexcept
{
    return mImpl != nullptr ? mImpl->getComputeSmCounts(localityDomainId) : std::pair<unsigned int, unsigned int>{};
}

CUstream LocalizationHandle::getReservedRemainderStream()
{
    TLLM_CHECK_WITH_INFO(mImpl != nullptr, "Cannot use a moved-from LocalizationHandle");
    return mImpl->getReservedRemainderStream();
}

bool deviceSupportsLocalization(int device) noexcept
{
#if CUDA_VERSION >= 13040
    // cuInit() is idempotent and does not create a context.
    if (cuInit(0) != CUDA_SUCCESS)
    {
        return false;
    }

    CUdevice cuDevice{};
    if (cuDeviceGet(&cuDevice, device) != CUDA_SUCCESS)
    {
        return false;
    }

    int localityDomainCount{};
    CUresult const result
        = cuDeviceGetAttribute(&localityDomainCount, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, cuDevice);
    return result == CUDA_SUCCESS && localityDomainCount >= kLocalityDomainCount;
#else
    static_cast<void>(device);
    return false;
#endif
}

} // namespace tensorrt_llm::locality_domain
