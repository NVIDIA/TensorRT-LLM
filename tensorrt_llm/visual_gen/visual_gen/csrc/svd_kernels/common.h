// Adapted from https://github.com/nunchaku-tech/nunchaku
// @article{
//   li2024svdquant,
//   title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
//   author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze
//   and Meng, Chenlin and Zhu, Jun-Yan and Han, Song}, journal={arXiv preprint arXiv:2411.05007}, year={2024}
// }

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <any>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <source_location>
#include <sstream>
#include <stack>
#include <unordered_map>
#include <variant>
#include <vector>

class CUDAError : public std::runtime_error
{
public:
    CUDAError(cudaError_t errorCode, std::source_location location)
        : std::runtime_error(format(errorCode, location))
        , errorCode(errorCode)
        , location(location)
    {
    }

public:
    const cudaError_t errorCode;
    const std::source_location location;

private:
    static std::string format(cudaError_t errorCode, std::source_location location)
    {
        return std::format(
            "CUDA error: {} (at {}:{})", cudaGetErrorString(errorCode), location.file_name(), location.line());
    }
};

inline cudaError_t checkCUDA(
    cudaError_t retValue, const std::source_location location = std::source_location::current())
{
    if (retValue != cudaSuccess)
    {
        (void) cudaGetLastError();
        throw CUDAError(retValue, location);
    }
    return retValue;
}

inline cublasStatus_t checkCUBLAS(
    cublasStatus_t retValue, const std::source_location location = std::source_location::current())
{
    if (retValue != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::format(
            "CUBLAS error: {} (at {}:{})", cublasGetStatusString(retValue), location.file_name(), location.line()));
    }
    return retValue;
}

inline thread_local std::stack<cudaStream_t> stackCUDAStreams;

inline cudaStream_t getCurrentCUDAStream()
{
    if (stackCUDAStreams.empty())
    {
        return 0;
    }
    return stackCUDAStreams.top();
}

struct CUDAStreamContext
{
    cudaStream_t stream;

    CUDAStreamContext(cudaStream_t stream)
        : stream(stream)
    {
        stackCUDAStreams.push(stream);
    }

    CUDAStreamContext(CUDAStreamContext const&) = delete;
    CUDAStreamContext(CUDAStreamContext&&) = delete;

    ~CUDAStreamContext()
    {
        assert(stackCUDAStreams.top() == stream);
        stackCUDAStreams.pop();
    }
};

struct CUDAStreamWrapper
{
    cudaStream_t stream;

    CUDAStreamWrapper()
    {
        checkCUDA(cudaStreamCreate(&stream));
    }

    CUDAStreamWrapper(CUDAStreamWrapper const&) = delete;
    CUDAStreamWrapper(CUDAStreamWrapper&&) = delete;

    ~CUDAStreamWrapper()
    {
        checkCUDA(cudaStreamDestroy(stream));
    }
};

struct CUDAEventWrapper
{
    cudaEvent_t event;

    CUDAEventWrapper(unsigned int flags = cudaEventDefault)
    {
        checkCUDA(cudaEventCreateWithFlags(&event, flags));
    }

    CUDAEventWrapper(CUDAEventWrapper const&) = delete;
    CUDAEventWrapper(CUDAEventWrapper&&) = delete;

    ~CUDAEventWrapper()
    {
        checkCUDA(cudaEventDestroy(event));
    }
};

/**
 * 1. hold one when entered from external code (set `device` to -1 to avoid device change)
 * 2. hold one when switching device
 * 3. hold one with `disableCache` when calling external code that may change the device
 */
class CUDADeviceContext
{
public:
    CUDADeviceContext(int device = -1, bool disableCache = false)
        : disableCache(disableCache)
    {
        if (cacheDisabled())
        {
            // no previous context => we might entered from external code, reset cache
            // previous context is reset on => external code may be executed, reset
            currentDeviceCache = -1;
        }

        ctxs.push(this);
        lastDevice = getDevice();
        if (device >= 0)
        {
            setDevice(device);
        }

        if (disableCache)
        {
            // we are about to call external code, reset cache
            currentDeviceCache = -1;
        }
    }

    CUDADeviceContext(CUDADeviceContext const&) = delete;
    CUDADeviceContext(CUDADeviceContext&&) = delete;

    ~CUDADeviceContext()
    {
        if (disableCache)
        {
            // retured from external code, cache is not reliable, reset
            currentDeviceCache = -1;
        }

        setDevice(lastDevice);
        assert(ctxs.top() == this);
        ctxs.pop();

        if (cacheDisabled())
        {
            // ctxs.empty() => we are about to return to external code, reset cache
            // otherwise => we are a nested context in a previous context with reset on, we might continue to execute
            // external code, reset
            currentDeviceCache = -1;
        }
    }

    bool const disableCache;
    int lastDevice;

public:
    static int getDevice()
    {
        int idx = -1;
        if (cacheDisabled() || currentDeviceCache < 0)
        {
            checkCUDA(cudaGetDevice(&idx));
        }
        else
        {
            idx = currentDeviceCache;
        }
        currentDeviceCache = cacheDisabled() ? -1 : idx;
        return idx;
    }

private:
    static void setDevice(int idx)
    {
        // TODO: deal with stream when switching device
        assert(idx >= 0);
        if (!cacheDisabled() && currentDeviceCache == idx)
        {
            return;
        }
        checkCUDA(cudaSetDevice(idx));
        currentDeviceCache = cacheDisabled() ? -1 : idx;
    }

private:
    static inline thread_local std::stack<CUDADeviceContext*> ctxs;
    static inline thread_local int currentDeviceCache = -1;

    static bool cacheDisabled()
    {
        return ctxs.empty() || ctxs.top()->disableCache;
    }
};

inline cudaDeviceProp* getCurrentDeviceProperties()
{
    static thread_local std::map<int, cudaDeviceProp> props;

    int deviceId = CUDADeviceContext::getDevice();
    if (!props.contains(deviceId))
    {
        cudaDeviceProp prop;
        checkCUDA(cudaGetDeviceProperties(&prop, deviceId));
        props[deviceId] = prop;
    }
    return &props.at(deviceId);
}

template <typename T>
constexpr T ceilDiv(T a, T b)
{
    return (a + b - 1) / b;
}

template <typename T>
constexpr int log2Up(T value)
{
    if (value <= 0)
        return 0;
    if (value == 1)
        return 0;
    return log2Up((value + 1) / 2) + 1;
}

struct CUBLASWrapper
{
    cublasHandle_t handle = nullptr;

    CUBLASWrapper()
    {
        checkCUBLAS(cublasCreate(&handle));
    }

    CUBLASWrapper(CUBLASWrapper&&) = delete;
    CUBLASWrapper(CUBLASWrapper const&&) = delete;

    ~CUBLASWrapper()
    {
        if (handle)
        {
            checkCUBLAS(cublasDestroy(handle));
        }
    }
};

inline std::shared_ptr<CUBLASWrapper> getCUBLAS()
{
    static thread_local std::weak_ptr<CUBLASWrapper> inst;
    std::shared_ptr<CUBLASWrapper> result = inst.lock();
    if (result)
    {
        return result;
    }
    result = std::make_shared<CUBLASWrapper>();
    inst = result;
    return result;
}
