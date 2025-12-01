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
#include "ub_allocator.h"
#include "nccl.h"
#include "tensorrt_llm/common/opUtils.h"
#include <set>
#include <stdexcept>

namespace tensorrt_llm::runtime::ub
{
UserBufferAllocator& UserBufferAllocator::Instance()
{
    if (use_nccl_symmetric)
    {
        static NCCLUserBufferAllocator _;
        return _;
    }
    else
    {
        static UserBufferAllocator _;
        return _;
    }
}

void UserBufferAllocator::initialize(tensorrt_llm::runtime::WorldConfig const& worldConfig)
{
    if (!isInitialized())
    {
        mUbComm = nullptr;
        mWorldConfig = worldConfig;
        create_communicator_grouped2(&mUbComm, worldConfig);
        TLLM_CHECK(mUbComm != nullptr);
        mIsInitialized = true;
    }
}

bool UserBufferAllocator::isInitialized()
{
    return mIsInitialized;
}

UBBuffer UserBufferAllocator::registerUBBuffer(size_t bytes)
{
    TLLM_CHECK(isInitialized());
    void* addr = nullptr;
    int handle = -1;
    handle = register_user_buffer_collective((void**) &addr, bytes, mUbComm);
    return {addr, handle, bytes};
}

UBBuffer UserBufferAllocator::allocate(size_t bytes)
{
    TLLM_CHECK(isInitialized());
    auto ub_buffer = registerUBBuffer(bytes);
    TLLM_CHECK(!ub_buffer.invalid());
    mBuffers.push_back(ub_buffer);
    return ub_buffer;
}

void UserBufferAllocator::deallocate(void* addr) {}

UBBuffer UserBufferAllocator::get(int idx)
{
    TLLM_CHECK(isInitialized() && idx < mBuffers.size() && !mBuffers[idx].invalid());
    return mBuffers[idx];
}

communicator* UserBufferAllocator::comm()
{
    TLLM_CHECK(isInitialized());
    return mUbComm;
}

void NCCLUserBufferAllocator::initialize(tensorrt_llm::runtime::WorldConfig const& worldConfig)
{
    if (!isInitialized())
    {
        TLLM_LOG_INFO("Initializing NCCLUserBufferAllocator");
        std::set<int> group;
        for (int i = 0; i < worldConfig.getSize(); i++)
        {
            group.insert(i);
        }
        mComm = getComm(group);
        mIsInitialized = true;
    }
}

UBBuffer NCCLUserBufferAllocator::registerUBBuffer(size_t bytes)
{
    TLLM_CHECK(isInitialized());
    UBBuffer ub_buffer;

    auto& ncclHelper = getNCCLHelper();
    if (!ncclHelper.isLoaded())
    {
        TLLM_THROW("NCCL library could not be loaded for dynamic symbol access");
    }

    auto ncclMemAllocFunc = ncclHelper.getNCCLMemAlloc();
    auto ncclCommWindowRegisterFunc = ncclHelper.getNCCLCommWindowRegister();

    NCCLCHECK(ncclMemAllocFunc(&ub_buffer.addr, bytes));
    NCCLCHECK(ncclCommWindowRegisterFunc((*mComm), ub_buffer.addr, bytes, &ub_buffer.window, NCCL_WIN_COLL_SYMMETRIC));
    ub_buffer.handle = 5;
    ub_buffer.size = bytes;
    return ub_buffer;
}

// Static member definitions
std::unique_ptr<NCCLHelper> NCCLUserBufferAllocator::mNCCLHelper = nullptr;

NCCLHelper& NCCLUserBufferAllocator::getNCCLHelper()
{
    if (!mNCCLHelper)
    {
        mNCCLHelper = std::make_unique<NCCLHelper>();
    }
    return *mNCCLHelper;
}

NCCLUserBufferAllocator::~NCCLUserBufferAllocator()
{
    // Deallocate buffers
    auto& ncclHelper = getNCCLHelper();
    if (ncclHelper.isLoaded())
    {
        auto ncclMemFreeFunc = ncclHelper.getNCCLMemFree();
        auto ncclCommWindowDeregisterFunc = ncclHelper.getNCCLCommWindowDeregister();
        if (ncclCommWindowDeregisterFunc == nullptr)
        {
            TLLM_LOG_WARNING("NCCL buffer windows cannot be released.");
        }
        for (auto buffer : mBuffers)
        {
            buffer.size = 0;
            if (ncclCommWindowDeregisterFunc != nullptr)
            {
                NCCLCHECK(ncclCommWindowDeregisterFunc(*mComm, buffer.window));
            }
            if (ncclMemFreeFunc != nullptr)
            {
                NCCLCHECK(ncclMemFreeFunc(buffer.addr));
            }
        }
        auto ncclDevCommDestroyFunc = ncclHelper.getNCCLDevCommDestroy();
        if (ncclDevCommDestroyFunc)
        {
            for (auto x : mDevCommBlockID)
            {
                ncclDevComm devComm = x.second;
                NCCLCHECK(ncclDevCommDestroyFunc(*mComm, devComm));
            }
        }
    }
}

ncclDevComm NCCLUserBufferAllocator::getNCCLDevComm(int const numLsaBarriers)
{
    constexpr bool multimemSupport = true;
    auto commIter = mDevCommBlockID.find(numLsaBarriers); // codespell:ignore word
    if (commIter == mDevCommBlockID.end())
    {
        ncclDevComm devComm;
        ncclDevCommRequirements reqs = {0};
        memset(&reqs, 0, sizeof(ncclDevCommRequirements));
        reqs.lsaBarrierCount = numLsaBarriers;
        reqs.lsaMultimem = multimemSupport;

        auto& ncclHelper = getNCCLHelper();
        auto ncclDevCommCreateFunc = ncclHelper.getNCCLDevCommCreate();

        ncclResult_t ncclError = ncclDevCommCreateFunc(*mComm, &reqs, &devComm);
        TLLM_CHECK_WITH_INFO(
            ncclError == ncclSuccess, "Failed to create NCCL device communicator: %s", ncclGetErrorString(ncclError));
        mDevCommBlockID[numLsaBarriers] = devComm;
    }
    commIter = mDevCommBlockID.find(numLsaBarriers);
    if (commIter == mDevCommBlockID.end())
    {
        TLLM_THROW("NCCL cannot create required device communicator");
    }
    return commIter->second;
}

// NCCLHelper implementation
NCCLHelper::NCCLHelper()
    : mLibraryHandle(nullptr)
    , mNCCLCommWindowRegister(nullptr)
    , mNCCLMemAlloc(nullptr)
    , mNCCLCommWindowDeregister(nullptr)
    , mNCCLMemFree(nullptr)
    , mNCCLDevCommCreate(nullptr)
    , mNCCLDevCommDestroy(nullptr)
    , mIsLoaded(false)
{
    loadNCCLLibrary();
}

NCCLHelper::~NCCLHelper()
{
    if (mLibraryHandle)
    {
#ifdef _WIN32
        FreeLibrary(mLibraryHandle);
#else
        dlclose(mLibraryHandle);
#endif
        mLibraryHandle = nullptr;
    }
}

void NCCLHelper::loadNCCLLibrary()
{
    try
    {
#ifdef _WIN32
        char const* libraryNames[] = {"nccl.dll"};
#else
        char const* libraryNames[] = {"libnccl.so"};
#endif

        for (int i = 0; libraryNames[i] != nullptr; ++i)
        {
            TLLM_LOG_INFO("Attempting to load NCCL library: %s", libraryNames[i]);
            mLibraryHandle = loadLibraryHandle(libraryNames[i]);
            if (mLibraryHandle)
            {
                TLLM_LOG_INFO("Successfully loaded NCCL library: %s", libraryNames[i]);

                // Get the actual path of the loaded library
                Dl_info info;
                if (dladdr(mLibraryHandle, &info) && info.dli_fname)
                {
                    TLLM_LOG_INFO("NCCL library loaded from: %s", info.dli_fname);
                }
                break;
            }
            else
            {
                TLLM_LOG_WARNING("Failed to load NCCL library: %s", libraryNames[i]);
            }
        }

        if (!mLibraryHandle)
        {
            TLLM_LOG_WARNING("Failed to load NCCL library");
            return;
        }

        // Load the required symbols
        mNCCLCommWindowRegister
            = reinterpret_cast<ncclCommWindowRegisterFunc>(getSymbolAddress(mLibraryHandle, "ncclCommWindowRegister"));

        mNCCLMemAlloc = reinterpret_cast<ncclMemAllocFunc>(getSymbolAddress(mLibraryHandle, "ncclMemAlloc"));

        mNCCLCommWindowDeregister = reinterpret_cast<ncclCommWindowDeregisterFunc>(
            getSymbolAddress(mLibraryHandle, "ncclCommWindowDeregister"));

        mNCCLMemFree = reinterpret_cast<ncclMemFreeFunc>(getSymbolAddress(mLibraryHandle, "ncclMemFree"));

        // Try to resolve device communicator functions using proper symbol resolution
        mNCCLDevCommCreate = resolveNCCLDevCommCreate(mLibraryHandle);
        mNCCLDevCommDestroy = resolveNCCLDevCommDestroy(mLibraryHandle);

        if (mNCCLCommWindowRegister == nullptr or mNCCLCommWindowDeregister == nullptr)
        {
            TLLM_LOG_WARNING("Failed to load ncclCommWindowRegister symbol, NCCL symmetric will not be supported.");
        }

        if (mNCCLDevCommCreate == nullptr or mNCCLDevCommDestroy == nullptr)
        {
            TLLM_LOG_WARNING(
                "Failed to load ncclDevCommCreate/ncclDevCommDestroy symbols, NCCL fused kernels will not be "
                "supported. Ensure NCCL version >= 2.28.");
            if (mNCCLDevCommCreate == nullptr)
            {
                TLLM_LOG_WARNING("ncclDevCommCreate symbol not found (tried both C and C++ mangled names)");
            }
            if (mNCCLDevCommDestroy == nullptr)
            {
                TLLM_LOG_WARNING("ncclDevCommDestroy symbol not found (tried both C and C++ mangled names)");
            }
        }
        else
        {
            TLLM_LOG_INFO("Successfully loaded ncclDevCommCreate and ncclDevCommDestroy symbols");
        }

        if (mNCCLMemAlloc and mNCCLMemFree)
        {
            mIsLoaded = true;
        }
        else
        {
            TLLM_LOG_WARNING("Failed to load required NCCL symbols");
        }
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("Exception while loading NCCL library: %s", e.what());
    }
}

void* NCCLHelper::loadLibraryHandle(char const* libName)
{
#ifdef _WIN32
    return LoadLibraryA(libName);
#else
    return dlopen(libName, RTLD_LAZY | RTLD_GLOBAL);
#endif
}

void* NCCLHelper::getSymbolAddress(void* handle, char const* symbolName)
{
    if (!handle)
    {
        return nullptr;
    }

#ifdef _WIN32
    return GetProcAddress(static_cast<HMODULE>(handle), symbolName);
#else
    return dlsym(handle, symbolName);
#endif
}

// Robust symbol resolution for device communicator functions
NCCLHelper::ncclDevCommCreateFunc NCCLHelper::resolveNCCLDevCommCreate(void* handle)
{
    if (!handle)
        return nullptr;

    // Try C-style symbol first (preferred)
    void* symbol = getSymbolAddress(handle, "ncclDevCommCreate");
    if (symbol)
    {
        TLLM_LOG_DEBUG("Found ncclDevCommCreate with C linkage");
        return reinterpret_cast<ncclDevCommCreateFunc>(symbol);
    }

    // Try common C++ mangled variants (fallback)
    char const* mangledNames[]
        = {"_Z17ncclDevCommCreateP8ncclCommPK23ncclDevCommRequirementsP11ncclDevComm",            // GCC/Clang
            "?ncclDevCommCreate@@YAHPAUncclComm@@PBUncclDevCommRequirements@@PAUncclDevComm@@@Z", // MSVC
            nullptr};

    for (int i = 0; mangledNames[i] != nullptr; ++i)
    {
        symbol = getSymbolAddress(handle, mangledNames[i]);
        if (symbol)
        {
            TLLM_LOG_WARNING("Found ncclDevCommCreate with C++ mangled name (fragile): %s", mangledNames[i]);
            return reinterpret_cast<ncclDevCommCreateFunc>(symbol);
        }
    }

    TLLM_LOG_DEBUG("ncclDevCommCreate not found with any known symbol name");
    return nullptr;
}

NCCLHelper::ncclDevCommDestroyFunc NCCLHelper::resolveNCCLDevCommDestroy(void* handle)
{
    if (!handle)
        return nullptr;

    // Try C-style symbol first (preferred)
    void* symbol = getSymbolAddress(handle, "ncclDevCommDestroy");
    if (symbol)
    {
        TLLM_LOG_DEBUG("Found ncclDevCommDestroy with C linkage");
        return reinterpret_cast<ncclDevCommDestroyFunc>(symbol);
    }

    // Try common C++ mangled variants (fallback)
    char const* mangledNames[] = {"_Z18ncclDevCommDestroyP8ncclCommPK11ncclDevComm", // GCC/Clang
        "?ncclDevCommDestroy@@YAHPAUncclComm@@PBUncclDevComm@@@Z",                   // MSVC
        nullptr};

    for (int i = 0; mangledNames[i] != nullptr; ++i)
    {
        symbol = getSymbolAddress(handle, mangledNames[i]);
        if (symbol)
        {
            TLLM_LOG_WARNING("Found ncclDevCommDestroy with C++ mangled name (fragile): %s", mangledNames[i]);
            return reinterpret_cast<ncclDevCommDestroyFunc>(symbol);
        }
    }

    TLLM_LOG_DEBUG("ncclDevCommDestroy not found with any known symbol name");
    return nullptr;
}

NCCLHelper::ncclCommWindowRegisterFunc NCCLHelper::getNCCLCommWindowRegister()
{
    return mNCCLCommWindowRegister;
}

NCCLHelper::ncclMemAllocFunc NCCLHelper::getNCCLMemAlloc()
{
    return mNCCLMemAlloc;
}

NCCLHelper::ncclCommWindowDeregisterFunc NCCLHelper::getNCCLCommWindowDeregister()
{
    return mNCCLCommWindowDeregister;
}

NCCLHelper::ncclMemFreeFunc NCCLHelper::getNCCLMemFree()
{
    return mNCCLMemFree;
}

NCCLHelper::ncclDevCommCreateFunc NCCLHelper::getNCCLDevCommCreate()
{
    return mNCCLDevCommCreate;
}

NCCLHelper::ncclDevCommDestroyFunc NCCLHelper::getNCCLDevCommDestroy()
{
    return mNCCLDevCommDestroy;
}

bool NCCLHelper::isLoaded() const
{
    return mIsLoaded;
}

bool UserBufferAllocator::use_nccl_symmetric = false;

std::shared_ptr<tensorrt_llm::kernels::nccl_device::LaunchConfig>
NCCLUserBufferAllocator::getCachedNCCLDeviceLaunchConfig(nvinfer1::DataType dataType, int const hiddenDim,
    int const numTokens, int const rank, int const nRanks, bool useResidual, bool useBias)
{
    // Create cache key
    LaunchConfigKey key{dataType, hiddenDim, numTokens, rank, nRanks, useResidual, useBias};

    // Check if config already exists in cache
    auto it = mLaunchConfigCache.find(key);
    if (it != mLaunchConfigCache.end())
    {
        return it->second; // Return cached config
    }

    // Create new config and cache it
    auto config = tensorrt_llm::kernels::nccl_device::makeLaunchConfig(
        dataType, hiddenDim, numTokens, rank, nRanks, useResidual, useBias);

    mLaunchConfigCache[key] = config;
    return config;
}

}; // namespace tensorrt_llm::runtime::ub
