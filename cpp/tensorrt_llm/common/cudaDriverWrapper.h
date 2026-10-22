/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
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

#ifndef CUDA_DRIVER_WRAPPER_H
#define CUDA_DRIVER_WRAPPER_H

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/common/tllmException.h"

#include <cuda.h>

#include <cstdio>
#include <format>
#include <memory>
#include <source_location>
#include <string>
#include <string_view>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

class CUDADriverWrapper
{
public:
    static std::shared_ptr<CUDADriverWrapper> getInstance();

    //! The live wrapper, or nullptr when none exists. Unlike getInstance(), this never loads the driver, so it is
    //! safe during process teardown.
    static std::shared_ptr<CUDADriverWrapper> getInstanceIfLoaded();

    ~CUDADriverWrapper();
    CUDADriverWrapper(CUDADriverWrapper const&) = delete;
    CUDADriverWrapper operator=(CUDADriverWrapper const&) = delete;
    CUDADriverWrapper(CUDADriverWrapper&&) = delete;
    CUDADriverWrapper operator=(CUDADriverWrapper&&) = delete;

    CUresult cuGetErrorName(CUresult error, char const** pStr) const;

    CUresult cuGetErrorString(CUresult error, char const** pStr) const;

    CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) const;

    CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) const;

    CUresult cuModuleUnload(CUmodule hmod) const;

    CUresult cuLinkDestroy(CUlinkState state) const;

    CUresult cuModuleLoadData(CUmodule* module, void const* image) const;

    CUresult cuLinkCreate(
        unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) const;

    CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, char const* name) const;

    CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, char const* name) const;

    CUresult cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, char const* name) const;

    CUresult cuLibraryLoadData(CUlibrary* library, void const* code, CUjit_option* jitOptions, void** jitOptionsValues,
        unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues,
        unsigned int numLibraryOptions) const;

    CUresult cuLibraryGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, char const* name) const;

    CUresult cuLibraryUnload(CUlibrary library) const;

    CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev) const;

    CUresult cuCtxGetDevice(CUdevice* device) const;

    CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, char const* path, unsigned int numOptions,
        CUjit_option* options, void** optionValues) const;

    CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size, char const* name,
        unsigned int numOptions, CUjit_option* options, void** optionValues) const;

    CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
        unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
        unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) const;

    CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
        CUstream hStream, void** kernelParams, void** extra) const;

    CUresult cuLaunchKernelEx(CUlaunchConfig const* config, CUfunction f, void** kernelParams, void** extra) const;

    CUresult cuTensorMapEncodeTiled(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank,
        void* globalAddress, cuuint64_t const* globalDim, cuuint64_t const* globalStrides, cuuint32_t const* boxDim,
        cuuint32_t const* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
        CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) const;

    CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) const;

    CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) const;

    CUresult cuOccupancyMaxActiveClusters(int* maxActiveClusters, CUfunction f, CUlaunchConfig const* config) const;

private:
    void* handle;
    CUDADriverWrapper();

    CUresult (*_cuGetErrorName)(CUresult, char const**);
    CUresult (*_cuGetErrorString)(CUresult, char const**);
    CUresult (*_cuFuncSetAttribute)(CUfunction, CUfunction_attribute, int);
    CUresult (*_cuLinkComplete)(CUlinkState, void**, size_t*);
    CUresult (*_cuModuleUnload)(CUmodule);
    CUresult (*_cuLinkDestroy)(CUlinkState);
    CUresult (*_cuLinkCreate)(unsigned int, CUjit_option*, void**, CUlinkState*);
    CUresult (*_cuModuleLoadData)(CUmodule*, void const*);
    CUresult (*_cuModuleGetFunction)(CUfunction*, CUmodule, char const*);
    CUresult (*_cuModuleGetGlobal)(CUdeviceptr*, size_t*, CUmodule, char const*);
    CUresult (*_cuLibraryGetKernel)(CUkernel*, CUlibrary, char const*);
    CUresult (*_cuLibraryLoadData)(
        CUlibrary*, void const*, CUjit_option*, void**, unsigned int, CUlibraryOption*, void**, unsigned int);
    CUresult (*_cuLibraryGetGlobal)(CUdeviceptr*, size_t*, CUlibrary, char const*);
    CUresult (*_cuLibraryUnload)(CUlibrary);
    CUresult (*_cuKernelSetAttribute)(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev);
    CUresult (*_cuCtxGetDevice)(CUdevice* device);
    CUresult (*_cuLinkAddFile)(CUlinkState, CUjitInputType, char const*, unsigned int, CUjit_option*, void**);
    CUresult (*_cuLinkAddData)(
        CUlinkState, CUjitInputType, void*, size_t, char const*, unsigned int, CUjit_option*, void**);
    CUresult (*_cuLaunchCooperativeKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, CUstream, void**);
    CUresult (*_cuLaunchKernel)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
        CUstream hStream, void** kernelParams, void** extra);
    CUresult (*_cuLaunchKernelEx)(CUlaunchConfig const* config, CUfunction f, void** kernelParams, void** extra);
    CUresult (*_cuTensorMapEncodeTiled)(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType,
        cuuint32_t tensorRank, void* globalAddress, cuuint64_t const* globalDim, cuuint64_t const* globalStrides,
        cuuint32_t const* boxDim, cuuint32_t const* elementStrides, CUtensorMapInterleave interleave,
        CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);
    CUresult (*_cuMemcpyDtoH)(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
    CUresult (*_cuDeviceGetAttribute)(int*, CUdevice_attribute attrib, CUdevice dev);
    CUresult (*_cuOccupancyMaxActiveClusters)(int*, CUfunction f, CUlaunchConfig const* config);
};

template <typename T>
void checkDriver(T result, char const* const func, std::source_location const& location)
{
    if (result)
    {
        // Load the driver so the CUresult formatter can name the error.
        CUDADriverWrapper::getInstance();
        throw TllmException(location, "[TensorRT-LLM][ERROR] CUDA driver error in {}: {}.", func, result);
    }
}

template <typename T>
void checkDriverExitSafe(T result, char const* const func, std::source_location const& location)
{
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED)
    {
        throw TllmException(
            location, "[TensorRT-LLM][ERROR] CUDA driver error in {}: {}.", func, static_cast<int>(result));
    }
}

} // namespace common

TRTLLM_NAMESPACE_END

//! Format CUDA driver results by name, e.g. "CUDA_ERROR_INVALID_VALUE". The name comes from the driver, so the numeric
//! code is printed when no CUDADriverWrapper is alive, as during process teardown.
template <>
struct std::formatter<CUresult> : std::formatter<std::string_view>
{
    template <typename Context>
    auto format(CUresult const result, Context& context) const
    {
        char const* name = nullptr;
        auto const driver = tensorrt_llm::common::CUDADriverWrapper::getInstanceIfLoaded();
        if (driver == nullptr || driver->cuGetErrorName(result, &name) != CUDA_SUCCESS || name == nullptr)
        {
            return std::formatter<std::string_view>::format(std::to_string(static_cast<int>(result)), context);
        }
        return std::formatter<std::string_view>::format(name, context);
    }
};

/*
 * Macros compliant with TensorRT coding conventions
 */
#define TLLM_CU_CHECK(stat)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        tensorrt_llm::common::checkDriver((stat), #stat, std::source_location::current());                             \
    } while (0)

// Avoid using CUDADriverWrapper when freeing resource, during which the global instance may already be freed.
#define TLLM_CU_CHECK_FREE_RESOURCE(stat)                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        tensorrt_llm::common::checkDriverExitSafe((stat), #stat, std::source_location::current());                     \
    } while (0)

#endif // CUDA_DRIVER_WRAPPER_H
