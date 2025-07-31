/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/virtualMemory.h"
#include "bufferManager.h"

#include <forward_list>
#include <shared_mutex>

namespace tensorrt_llm::runtime
{

namespace
{

template <typename T>
struct ScopeGuard
{
    bool const& ok;
    T t;

    ~ScopeGuard() noexcept(noexcept(t()))
    {
        if (!ok)
        {
            t();
        }
    }
};

template <typename T>
ScopeGuard(bool const&, T) -> ScopeGuard<T>;

} // namespace

void CUDAVirtualMemoryChunk::materialize()
{
    TLLM_CHECK_WITH_INFO(status() == RELEASED, "virtual memory not in RELEASED status, is: %d", status());
    mHandle = mCreator->create();

    // Track the number of configurators ran, so release can correctly teardown.
    for (auto const& conf : mConfigurators)
    {
        conf->setup(mHandle); // May throw
        ++mState;
    }
}

template <typename Callable, typename... Args>
static bool safe_invoke_helper(std::exception_ptr& ep, char const* msg, Callable&& f, Args&&... args) noexcept
{
    try
    {
        std::invoke(std::forward<Callable>(f), std::forward<Args>(args)...);
        return true;
    }
    catch (...)
    {
        if (ep)
        {
            try
            {
                std::rethrow_exception(ep);
            }
            catch (std::exception& e)
            {
                TLLM_LOG_ERROR(msg, e.what());
            }
        }
        ep = std::current_exception();
        return false;
    }
}

void CUDAVirtualMemoryChunk::_release(bool destructing)
{
    TLLM_CHECK_WITH_INFO(status() == MATERIALIZED || (status() == ERRORED && mState != INVALID_STATE),
        "virtual memory is in status %d which cannot be released", status());
    size_t const count = mConfigurators.size();
    size_t const start = count - mState;

    // Revert materialize(). Only configurators that ran setup() successfully
    // will have their teardown() been called.
    // Never early returns on exceptions. The last exception will be rethrown, and
    // previous ones will be logged.
    std::exception_ptr ePtr{};
    auto const* msg = "Multiple exceptions thrown during release. The previous exception is: %s";
    for (size_t i = start; i < count; ++i)
    {
        safe_invoke_helper(
            ePtr, msg, &Configurator::teardown, mConfigurators[count - i - 1].get(), mHandle, destructing);
    }
    safe_invoke_helper(ePtr, msg, &Creator::release, mCreator.get(), mHandle, destructing);
    mHandle = {};
    mState = 0;

    if (ePtr != nullptr)
    {
        mState = INVALID_STATE;
        std::rethrow_exception(ePtr);
    }
}

void OffloadConfigurator::setup(CUmemGenericAllocationHandle)
{
    if (mBackedStorage != nullptr)
    {
        if (mOndemand)
        {
            TLLM_CU_CHECK(cuMemcpyHtoD_v2(mAddress, mBackedStorage->data(), mSize));
            mBackedStorage.reset();
        }
        else
        {
            TLLM_CU_CHECK(cuMemcpyHtoDAsync_v2(mAddress, mBackedStorage->data(), mSize, mStream));
        }
    }
}

void OffloadConfigurator::teardown(CUmemGenericAllocationHandle, bool destructing)
{
    if (destructing)
    {
        return;
    }

    if (mBackedStorage == nullptr)
    {
        switch (mBackType)
        {
        case MemoryType::kCPU: mBackedStorage = BufferManager::cpu(mSize, nvinfer1::DataType::kINT8); break;
        case MemoryType::kPINNED: mBackedStorage = BufferManager::pinned(mSize, nvinfer1::DataType::kINT8); break;
        default: TLLM_THROW("Unknown memory type: %d", static_cast<int32_t>(mBackType));
        }
    }
    // We have to synchronize here, or the memory may be unmapped before the copy operation.
    TLLM_CU_CHECK_FREE_RESOURCE(cuMemcpyDtoH_v2(mBackedStorage->data(), mAddress, mSize));
}

void CudaVirtualMemoryManager::add(uintptr_t handle, std::string tag, CUDAVirtualMemoryChunk&& memory)
{
    bool success = false;

    TLLM_CHECK_WITH_INFO(
        memory.status() == CUDAVirtualMemoryChunk::RELEASED || memory.status() == CUDAVirtualMemoryChunk::MATERIALIZED,
        "CudaVirtualMemoryManager: bad virtual memory status");

    std::unique_lock lock(mMutex);
    auto [memIt, created] = mMemories.try_emplace(handle, Entry{});
    TLLM_CHECK_WITH_INFO(
        created, "CudaVirtualMemoryManager: handle 0x%016zx already being used by another memory", handle);
    ScopeGuard eraseMemIt{success, [&, memIt_ = memIt] { mMemories.erase(memIt_); }};

    auto const entryIt = mEntries.emplace(std::move(tag), memIt);
    entryIt->second->second.mEntryIt = entryIt;

    memIt->second.mMemory = std::move(memory);
    success = true;
}

void CudaVirtualMemoryManager::add(uintptr_t handle, std::string tag, CUDAVirtualMemoryChunk::CreatorPtr&& creator,
    CUDAVirtualMemoryChunk::Configurators&& configurators)
{
    std::unique_lock lock(mMutex);
    bool success = false;

    auto [memIt, created] = mMemories.try_emplace(handle,
        Entry{
            {std::move(creator), std::move(configurators)},
        });
    TLLM_CHECK_WITH_INFO(
        created, "CudaVirtualMemoryManager: handle 0x%016zx already being used by another memory", handle);
    ScopeGuard eraseMemIt{success, [&, memIt_ = memIt] { mMemories.erase(memIt_); }};

    auto const entryIt = mEntries.emplace(std::move(tag), memIt);
    memIt->second.mEntryIt = entryIt;
    ScopeGuard eraseTagIt{success, [&] { mEntries.erase(entryIt); }};

    try
    {
        // Hopefully we don't need to hold the mutex guarding mMemories and mEntries anymore.
        lock.unlock();
        memIt->second.mMemory.materialize();
        success = true;
    }
    catch (...)
    {
        // ...unless materialize() throws and we need to rollback.
        lock.lock();
        throw;
    }
}

CUDAVirtualMemoryChunk CudaVirtualMemoryManager::remove(uintptr_t handle) noexcept
{
    std::unique_lock lock(mMutex);

    return unsafeRemove(handle);
}

CUDAVirtualMemoryChunk CudaVirtualMemoryManager::unsafeRemove(uintptr_t handle) noexcept
{
    auto const nodeHandle = mMemories.extract(handle);
    if (!nodeHandle)
    {
        return {};
    }
    mEntries.erase(nodeHandle.mapped().mEntryIt);

    return std::move(nodeHandle.mapped().mMemory);
}

void CudaVirtualMemoryManager::addBadHandle(uintptr_t handle) noexcept
{
    try
    {
        mBadHandles.push_back(handle);
    }
    catch (...)
    {
    }
}

std::vector<uintptr_t> CudaVirtualMemoryManager::retrieveBadHandles() noexcept
{
    return std::move(mBadHandles);
}

size_t CudaVirtualMemoryManager::releaseWithTag(std::string const& tag)
{
    std::unique_lock lock(mMutex);

    std::exception_ptr ePtr{};
    auto [begin, end] = mEntries.equal_range(tag);
    size_t count = 0;
    for (auto it = begin; it != end;)
    {
        auto const handle = it->second->first;
        auto& memory = it->second->second.mMemory;
        ++it; // element referenced by `it` will be invalidated by unsafeRemove(handle)
        if (memory.status() == CUDAVirtualMemoryChunk::MATERIALIZED)
        {
            if (!safe_invoke_helper(ePtr,
                    "Multiple exceptions thrown during releaseWithTag. The previous exception is: %s",
                    &CUDAVirtualMemoryChunk::release, &memory))
            {
                addBadHandle(handle);
                unsafeRemove(handle);
            }
            ++count;
        }
    }

    if (ePtr != nullptr)
    {
        std::rethrow_exception(ePtr);
    }

    return count;
}

size_t CudaVirtualMemoryManager::materializeWithTag(std::string const& tag)
{
    std::unique_lock lock(mMutex);

    auto [begin, end] = mEntries.equal_range(tag);
    size_t count = 0;

    auto it = begin;

    try
    {
        for (; it != end; ++it)
        {
            auto& memory = it->second->second.mMemory;
            if (memory.status() == CUDAVirtualMemoryChunk::RELEASED)
            {
                memory.materialize();
                ++count;
            }
        }
    }
    catch (...)
    {
        for (auto itRollback = begin; itRollback != it;)
        {
            auto const handle = itRollback->second->first;
            auto& memory = itRollback->second->second.mMemory;
            ++itRollback;
            try
            {
                memory.release();
            }
            catch (std::exception& e)
            {
                addBadHandle(handle);
                unsafeRemove(handle);
                TLLM_LOG_ERROR("Additional exception thrown during rollback of materializeWithTag: %s", e.what());
            }
        }

        addBadHandle(it->second->first);
        unsafeRemove(it->second->first);

        throw;
    }
    return count;
}

static_assert(sizeof(void*) == sizeof(CUdeviceptr));

static CUdeviceptr deviceptr_cast(void* ptr)
{
    CUdeviceptr ret{};
    std::memcpy(&ret, &ptr, sizeof(CUdeviceptr));
    return ret;
}

static void* deviceptr_cast(CUdeviceptr ptr)
{
    void* ret{};
    std::memcpy(&ret, &ptr, sizeof(CUdeviceptr));
    return ret;
}

void CudaVirtualMemoryAllocator::allocate(Pointer* ptr, std::size_t n, int device) const
{
    CUdeviceptr address{};
    std::size_t const pageAlignedSize = mConfig->pageAligned(n);
    TLLM_CU_CHECK(cuMemAddressReserve(&address, pageAlignedSize, 0, {}, 0));

    CUDAVirtualMemoryChunk::Configurators configurators;
    configurators.push_back(std::make_unique<UnicastConfigurator>(address, n,
        CUmemAccessDesc{{
                            CU_MEM_LOCATION_TYPE_DEVICE,
                            device,
                        },
            CU_MEM_ACCESS_FLAGS_PROT_READWRITE}));

    switch (mConfig->mMode)
    {
    case NONE: break;
    case MEMSET:
        configurators.push_back(std::make_unique<MemsetConfigurator>(address, n, 0, mConfig->mBackStream->get()));
        break;
    case CPU:
        configurators.push_back(
            std::make_unique<OffloadConfigurator>(address, n, MemoryType::kCPU, mConfig->mBackStream->get()));
        break;
    case PINNED:
        configurators.push_back(
            std::make_unique<OffloadConfigurator>(address, n, MemoryType::kPINNED, mConfig->mBackStream->get()));
        break;
    }

    mConfig->mManager.add(address, mConfig->mTag,
        std::make_unique<LocalCreator<>>(CUmemAllocationProp{CU_MEM_ALLOCATION_TYPE_PINNED, CU_MEM_HANDLE_TYPE_NONE,
                                             {
                                                 CU_MEM_LOCATION_TYPE_DEVICE,
                                                 device,
                                             }},
            n),
        std::move(configurators));

    *ptr = deviceptr_cast(address);
}

void CudaVirtualMemoryAllocator::deallocate(Pointer ptr, std::size_t n) const
{
    auto const address = deviceptr_cast(ptr);
    mConfig->mManager.remove(address);

    std::size_t const pageAlignedSize = mConfig->pageAligned(n);
    TLLM_CU_CHECK_FREE_RESOURCE(cuMemAddressFree(address, pageAlignedSize));
}

} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::runtime
{

CudaVirtualMemoryManager& getVirtualMemoryManager()
{
    static CudaVirtualMemoryManager manager;
    return manager;
}

using AllocConf = CudaVirtualMemoryAllocator::Configuration;

AllocConf AllocConf::backgroundConfiguration{getVirtualMemoryManager(), "", NONE, nullptr, true};

static const std::shared_ptr<AllocConf> bgConf{std::shared_ptr<AllocConf>{}, &AllocConf::backgroundConfiguration};

static std::shared_mutex currentConfMutex;
static std::shared_ptr<AllocConf> currentConf = bgConf;

CudaVirtualMemoryAllocator getVirtualMemoryAllocator()
{
    std::shared_lock lock(currentConfMutex);
    return CudaVirtualMemoryAllocator{currentConf};
}

void setVirtualMemoryAllocator(
    std::string const& tag, CudaVirtualMemoryAllocator::RestoreMode mode, std::shared_ptr<CudaStream> backStream)
{
    std::unique_lock lock(currentConfMutex);

    TLLM_CHECK_WITH_INFO(currentConf == bgConf,
        "An active virtual memory allocator (tag: %s, mode: %d, stream: %p) is already present",
        currentConf->mTag.c_str(), currentConf->mMode, currentConf->mBackStream.get());
    currentConf = std::make_shared<AllocConf>(getVirtualMemoryManager(), tag, mode, backStream);
}

void clearVirtualMemoryAllocator()
{
    std::unique_lock lock(currentConfMutex);
    currentConf = bgConf;
}

} // namespace tensorrt_llm::runtime
