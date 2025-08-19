/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "nccl.h"
#include "nccl_device.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include "tensorrt_llm/kernels/nccl_device/config.h"
#include <memory>
#include <map>

// Forward declarations for NCCL device communicator types
struct ncclDevComm;
struct ncclDevCommRequirements;
#if ENABLE_MULTI_DEVICE
#include "nccl.h"
#include "userbuffers.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#else
using ncclWindow_t = void*;
#endif

namespace tensorrt_llm::runtime::ub
{
static auto constexpr tensor_prefix = "allreduce_ub_";

struct UBBuffer
{
    void* addr;
    int handle;
    size_t size;
    ncclWindow_t window;

    UBBuffer(void* a = nullptr, int h = -1, size_t s = 0, ncclWindow_t w = nullptr)
        : addr(a)
        , handle(h)
        , size(s)
        , window(w)
    {
    }

    [[nodiscard]] bool invalid() const
    {
        return (addr == nullptr) || (handle == -1) || (size == 0);
    }
};

#if ENABLE_MULTI_DEVICE
class UserBufferAllocator
{
public:
    static UserBufferAllocator& Instance();

    UserBufferAllocator() = default;

    virtual void initialize(tensorrt_llm::runtime::WorldConfig const& worldConfig);
    bool isInitialized();
    UBBuffer allocate(size_t bytes);
    void deallocate(void* addr);
    UBBuffer get(int idx);
    communicator* comm();
    virtual UBBuffer registerUBBuffer(size_t bytes);

    static bool use_nccl_symmetric;

private:
    communicator* mUbComm;

protected:
    std::vector<UBBuffer> mBuffers;
    bool mIsInitialized;
    tensorrt_llm::runtime::WorldConfig mWorldConfig;
};

class NCCLHelper
{
public:
    NCCLHelper();
    ~NCCLHelper();

    // Dynamic loading function type definition
    using ncclCommWindowRegisterFunc = ncclResult_t (*)(ncclComm_t, void*, size_t, ncclWindow_t*, int);
    using ncclMemAllocFunc = ncclResult_t (*)(void**, size_t);
    using ncclCommWindowDeregisterFunc = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
    using ncclMemFreeFunc = ncclResult_t (*)(void(*));
    using ncclDevCommCreateFunc = ncclResult_t (*)(ncclComm_t, struct ncclDevCommRequirements const (*), struct ncclDevComm (*));
    using ncclDevCommDestroyFunc = ncclResult_t (*)(ncclComm_t, ncclDevComm);

    // Get function pointer for ncclCommWindowRegister
    ncclCommWindowRegisterFunc getNCCLCommWindowRegister();

    // Get function pointer for ncclMemAlloc
    ncclMemAllocFunc getNCCLMemAlloc();

    ncclCommWindowDeregisterFunc getNCCLCommWindowDeregister();
    ncclMemFreeFunc getNCCLMemFree();
    ncclDevCommCreateFunc getNCCLDevCommCreate();
    ncclDevCommDestroyFunc getNCCLDevCommDestroy();

    // Check if NCCL library is successfully loaded
    bool isLoaded() const;

private:
    void loadNCCLLibrary();
    void* loadLibraryHandle(char const* libName);
    void* getSymbolAddress(void* handle, char const* symbolName);
    
    // Robust symbol resolution methods
    ncclDevCommCreateFunc resolveNCCLDevCommCreate(void* handle);
    ncclDevCommDestroyFunc resolveNCCLDevCommDestroy(void* handle);

#ifdef _WIN32
    HMODULE mLibraryHandle;
#else
    void* mLibraryHandle;
#endif

    ncclCommWindowRegisterFunc mNCCLCommWindowRegister;
    ncclMemAllocFunc mNCCLMemAlloc;
    ncclCommWindowDeregisterFunc mNCCLCommWindowDeregister;
    ncclMemFreeFunc mNCCLMemFree;
    ncclDevCommCreateFunc mNCCLDevCommCreate;
    ncclDevCommDestroyFunc mNCCLDevCommDestroy;
    bool mIsLoaded;
};

class NCCLUserBufferAllocator : public UserBufferAllocator
{
public:
    void initialize(tensorrt_llm::runtime::WorldConfig const& world_config) override;
    UBBuffer registerUBBuffer(size_t bytes) override;

    // Get shared NCCLHelper instance
    static NCCLHelper& getNCCLHelper();

  ~NCCLUserBufferAllocator();

  ncclDevComm getNCCLDevComm(const int numLsaBarriers);
  
  // Cached NCCL device launch config functionality
  std::shared_ptr<tensorrt_llm::kernels::nccl_device::LaunchConfig> 
  getCachedNCCLDeviceLaunchConfig(nvinfer1::DataType dataType, const int hidden_dim, 
                                 const int num_tokens, const int rank, const int nRanks, 
                                 bool useResidual, bool useBias, bool unshardResidualOut);
  
private:
    std::shared_ptr<ncclComm_t> mComm;
    static std::unique_ptr<NCCLHelper> mNCCLHelper;
    std::map<int, ncclDevComm> mDevCommBlockID;
    
    // Cache for fused allreduce launch configs
    struct LaunchConfigKey {
        nvinfer1::DataType dataType;
        int hidden_dim;
        int num_tokens;
        int rank;
        int nRanks;
        bool useResidual;
        bool useBias;
        bool unshardResidualOut;
        
        bool operator<(const LaunchConfigKey& other) const {
            return std::tie(dataType, hidden_dim, num_tokens, rank, nRanks, useResidual, useBias, unshardResidualOut) <
                   std::tie(other.dataType, other.hidden_dim, other.num_tokens, other.rank, other.nRanks, other.useResidual, other.useBias, other.unshardResidualOut);
        }
    };
    
    std::map<LaunchConfigKey, std::shared_ptr<tensorrt_llm::kernels::nccl_device::LaunchConfig>> mLaunchConfigCache;
};

#else
using communicator = void;
#endif
}; // namespace tensorrt_llm::runtime::ub
