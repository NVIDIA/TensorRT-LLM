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

#ifndef NVML_WRAPPER_H
#define NVML_WRAPPER_H

#include "tensorrt_llm/common/config.h"

#include <nvml.h>

#include <memory>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

class NVMLWrapper
{
public:
    static std::shared_ptr<NVMLWrapper> getInstance();

    ~NVMLWrapper();
    NVMLWrapper(NVMLWrapper const&) = delete;
    NVMLWrapper& operator=(NVMLWrapper const&) = delete;
    NVMLWrapper(NVMLWrapper&&) = delete;
    NVMLWrapper& operator=(NVMLWrapper&&) = delete;

    // Required NVML functions
    nvmlReturn_t nvmlInit() const;
    nvmlReturn_t nvmlShutdown() const;
    nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) const;
    nvmlReturn_t nvmlDeviceGetHandleByPciBusId(char const* pciBusId, nvmlDevice_t* device) const;
    nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index) const;
    nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci) const;
    nvmlReturn_t nvmlDeviceGetNvLinkCapability(
        nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int* capResult) const;
    nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive) const;
    char const* nvmlErrorString(nvmlReturn_t result) const;
    nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(
        nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_v2_t* infos) const;

    // Optional NVML functions (may be nullptr on older drivers)
    nvmlReturn_t nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, nvmlGpuFabricInfoV_t* gpuFabricInfo) const;
    nvmlReturn_t nvmlDeviceGetGpuFabricInfo(nvmlDevice_t device, nvmlGpuFabricInfo_t* gpuFabricInfo) const;

    // Runtime availability checks
    bool hasGpuFabricInfoV() const;
    bool hasGpuFabricInfo() const;

private:
    void* mHandle;
    NVMLWrapper();

    // Required function pointers
    nvmlReturn_t (*_nvmlInit)();
    nvmlReturn_t (*_nvmlShutdown)();
    nvmlReturn_t (*_nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t*);
    nvmlReturn_t (*_nvmlDeviceGetHandleByPciBusId)(char const*, nvmlDevice_t*);
    nvmlReturn_t (*_nvmlDeviceGetIndex)(nvmlDevice_t, unsigned int*);
    nvmlReturn_t (*_nvmlDeviceGetNvLinkRemotePciInfo)(nvmlDevice_t, unsigned int, nvmlPciInfo_t*);
    nvmlReturn_t (*_nvmlDeviceGetNvLinkCapability)(nvmlDevice_t, unsigned int, nvmlNvLinkCapability_t, unsigned int*);
    nvmlReturn_t (*_nvmlDeviceGetNvLinkState)(nvmlDevice_t, unsigned int, nvmlEnableState_t*);
    char const* (*_nvmlErrorString)(nvmlReturn_t);
    nvmlReturn_t (*_nvmlDeviceGetComputeRunningProcesses)(nvmlDevice_t, unsigned int*, nvmlProcessInfo_v2_t*);

    // Optional function pointers (may be nullptr)
    nvmlReturn_t (*_nvmlDeviceGetGpuFabricInfoV)(nvmlDevice_t, nvmlGpuFabricInfoV_t*);
    nvmlReturn_t (*_nvmlDeviceGetGpuFabricInfo)(nvmlDevice_t, nvmlGpuFabricInfo_t*);
};

// RAII class that initializes NVML on construction and shuts down on destruction.
// Replaces duplicated NvmlManager classes in allreduceOp.cpp and allreducePlugin.cpp.
class NvmlManager
{
public:
    NvmlManager()
        : mNvml(NVMLWrapper::getInstance())
    {
        auto result = mNvml->nvmlInit();
        if (result != NVML_SUCCESS)
        {
            TLLM_THROW("Failed to initialize NVML: %s", mNvml->nvmlErrorString(result));
        }
    }

    ~NvmlManager()
    {
        mNvml->nvmlShutdown();
    }

    NVMLWrapper const& wrapper() const
    {
        return *mNvml;
    }

    std::shared_ptr<NVMLWrapper> const& sharedWrapper() const
    {
        return mNvml;
    }

private:
    std::shared_ptr<NVMLWrapper> mNvml;
};

} // namespace common

TRTLLM_NAMESPACE_END

#endif // NVML_WRAPPER_H
