/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
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

#include <dlfcn.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvmlWrapper.h"

#include <mutex>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

std::shared_ptr<NVMLWrapper> NVMLWrapper::getInstance()
{
    static std::mutex mutex;
    static std::weak_ptr<NVMLWrapper> instance;
    std::shared_ptr<NVMLWrapper> result = instance.lock();
    if (result)
    {
        return result;
    }

    std::lock_guard<std::mutex> const lock(mutex);
    result = instance.lock();
    if (!result)
    {
        result = std::shared_ptr<NVMLWrapper>(new NVMLWrapper());
        instance = result;
    }
    return result;
}

NVMLWrapper::NVMLWrapper()
    : mHandle(dlopen("libnvidia-ml.so.1", RTLD_LAZY))
{
    TLLM_CHECK_WITH_INFO(mHandle != nullptr, "NVML library (libnvidia-ml.so.1) could not be loaded.");

    auto loadSym = [](void* handle, char const* name) -> void* { return dlsym(handle, name); };

    auto loadRequired = [&](void* handle, char const* name) -> void*
    {
        void* sym = loadSym(handle, name);
        TLLM_CHECK_WITH_INFO(sym != nullptr, "Required NVML symbol not found: %s", name);
        return sym;
    };

    *reinterpret_cast<void**>(&_nvmlInit) = loadRequired(mHandle, "nvmlInit_v2");
    *reinterpret_cast<void**>(&_nvmlShutdown) = loadRequired(mHandle, "nvmlShutdown");
    *reinterpret_cast<void**>(&_nvmlDeviceGetHandleByIndex) = loadRequired(mHandle, "nvmlDeviceGetHandleByIndex_v2");
    *reinterpret_cast<void**>(&_nvmlDeviceGetHandleByPciBusId)
        = loadRequired(mHandle, "nvmlDeviceGetHandleByPciBusId_v2");
    *reinterpret_cast<void**>(&_nvmlDeviceGetIndex) = loadRequired(mHandle, "nvmlDeviceGetIndex");
    *reinterpret_cast<void**>(&_nvmlDeviceGetNvLinkRemotePciInfo)
        = loadRequired(mHandle, "nvmlDeviceGetNvLinkRemotePciInfo_v2");
    *reinterpret_cast<void**>(&_nvmlDeviceGetNvLinkCapability) = loadRequired(mHandle, "nvmlDeviceGetNvLinkCapability");
    *reinterpret_cast<void**>(&_nvmlDeviceGetNvLinkState) = loadRequired(mHandle, "nvmlDeviceGetNvLinkState");
    *reinterpret_cast<void**>(&_nvmlErrorString) = loadRequired(mHandle, "nvmlErrorString");
    *reinterpret_cast<void**>(&_nvmlDeviceGetComputeRunningProcesses)
        = loadRequired(mHandle, "nvmlDeviceGetComputeRunningProcesses_v3");

    // Optional symbols - nullptr is OK (older drivers may not have these)
    *reinterpret_cast<void**>(&_nvmlDeviceGetGpuFabricInfoV) = loadSym(mHandle, "nvmlDeviceGetGpuFabricInfoV");
    *reinterpret_cast<void**>(&_nvmlDeviceGetGpuFabricInfo) = loadSym(mHandle, "nvmlDeviceGetGpuFabricInfo");

    if (!_nvmlDeviceGetGpuFabricInfoV)
    {
        TLLM_LOG_INFO(
            "NVML symbol nvmlDeviceGetGpuFabricInfoV not available (older driver). MNNVL fabric detection will use "
            "legacy API or be disabled.");
    }
    if (!_nvmlDeviceGetGpuFabricInfo)
    {
        TLLM_LOG_INFO("NVML symbol nvmlDeviceGetGpuFabricInfo not available.");
    }
}

NVMLWrapper::~NVMLWrapper()
{
    dlclose(mHandle);
}

nvmlReturn_t NVMLWrapper::nvmlInit() const
{
    return (*_nvmlInit)();
}

nvmlReturn_t NVMLWrapper::nvmlShutdown() const
{
    return (*_nvmlShutdown)();
}

nvmlReturn_t NVMLWrapper::nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) const
{
    return (*_nvmlDeviceGetHandleByIndex)(index, device);
}

nvmlReturn_t NVMLWrapper::nvmlDeviceGetHandleByPciBusId(char const* pciBusId, nvmlDevice_t* device) const
{
    return (*_nvmlDeviceGetHandleByPciBusId)(pciBusId, device);
}

nvmlReturn_t NVMLWrapper::nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index) const
{
    return (*_nvmlDeviceGetIndex)(device, index);
}

nvmlReturn_t NVMLWrapper::nvmlDeviceGetNvLinkRemotePciInfo(
    nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci) const
{
    return (*_nvmlDeviceGetNvLinkRemotePciInfo)(device, link, pci);
}

nvmlReturn_t NVMLWrapper::nvmlDeviceGetNvLinkCapability(
    nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int* capResult) const
{
    return (*_nvmlDeviceGetNvLinkCapability)(device, link, capability, capResult);
}

nvmlReturn_t NVMLWrapper::nvmlDeviceGetNvLinkState(
    nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive) const
{
    return (*_nvmlDeviceGetNvLinkState)(device, link, isActive);
}

char const* NVMLWrapper::nvmlErrorString(nvmlReturn_t result) const
{
    return (*_nvmlErrorString)(result);
}

nvmlReturn_t NVMLWrapper::nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, nvmlGpuFabricInfoV_t* gpuFabricInfo) const
{
    if (!_nvmlDeviceGetGpuFabricInfoV)
    {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return (*_nvmlDeviceGetGpuFabricInfoV)(device, gpuFabricInfo);
}

nvmlReturn_t NVMLWrapper::nvmlDeviceGetGpuFabricInfo(nvmlDevice_t device, nvmlGpuFabricInfo_t* gpuFabricInfo) const
{
    if (!_nvmlDeviceGetGpuFabricInfo)
    {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return (*_nvmlDeviceGetGpuFabricInfo)(device, gpuFabricInfo);
}

nvmlReturn_t NVMLWrapper::nvmlDeviceGetComputeRunningProcesses(
    nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_v2_t* infos) const
{
    return (*_nvmlDeviceGetComputeRunningProcesses)(device, infoCount, infos);
}

bool NVMLWrapper::hasGpuFabricInfoV() const
{
    return _nvmlDeviceGetGpuFabricInfoV != nullptr;
}

bool NVMLWrapper::hasGpuFabricInfo() const
{
    return _nvmlDeviceGetGpuFabricInfo != nullptr;
}

} // namespace common

TRTLLM_NAMESPACE_END
