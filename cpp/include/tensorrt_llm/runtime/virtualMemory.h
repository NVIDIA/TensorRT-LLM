/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/memoryCounters.h"

#include <cuda.h>
#include <mutex>
#include <unistd.h>

class VirtualMemoryManagerTest;

namespace tensorrt_llm::runtime
{

/**
 * CUDAVirtualMemory is a handle to a piece of CUDA memory allocation,
 * providing the ability to release and rematerialize the allocation.
 */
class CUDAVirtualMemory
{
public:
    /**
     * CUDAVirtualMemory::Creator is the interface to obtain a CUmemGenericAllocationHandle,
     * either by creating one locally, or importing one from remote.
     */
    struct Creator
    {
        Creator() = default;
        virtual ~Creator() = default;
        Creator(Creator const&) = default;
        Creator& operator=(Creator const&) = default;
        Creator(Creator&&) = default;
        Creator& operator=(Creator&&) = default;

        // Note: create() shall not leak resources when throwing exceptions.
        // release() will only, and will always be called if create() success.
        virtual CUmemGenericAllocationHandle create() = 0;
        virtual void release(CUmemGenericAllocationHandle) = 0;
    };

    using CreatorPtr = std::unique_ptr<Creator>;

    /**
     * CUDAVirtualMemory::Configurator is the interface to configure a CUmemGenericAllocationHandle:
     * - Map into virtual address
     * - Bind to multicast object
     * - Backup and restore memory content
     */
    struct Configurator
    {
        Configurator() = default;
        virtual ~Configurator() = default;
        Configurator(Configurator const&) = default;
        Configurator& operator=(Configurator const&) = default;
        Configurator(Configurator&&) = default;
        Configurator& operator=(Configurator&&) = default;

        // Note: setup() shall not leak resources when throwing exceptions.
        // teardown() will only, and will always be called if setup() success.
        virtual void setup(CUmemGenericAllocationHandle) = 0;
        virtual void teardown(CUmemGenericAllocationHandle) = 0;
    };

    using ConfiguratorPtr = std::unique_ptr<Configurator>;
    using Configurators = std::vector<ConfiguratorPtr>;

    enum Status
    {
        INVALID,      // This is a default constructed invalid CUDAVirtualMemory.
        RELEASED,     // The memory represented by this CUDAVirtualMemory is not allocated.
        MATERIALIZED, // The memory represented by this CUDAVirtualMemory is allocated.
        ERRORED,      // Error happened during materialize() or release().
                      // This CUDAVirtualMemory cannot be used anymore.
    };

    [[nodiscard]] Status status() const noexcept
    {
        if (mCreator == nullptr)
        {
            return INVALID;
        }

        if (mState == 0 && mHandle == 0)
        {
            return RELEASED;
        }

        if (mState == mConfigurators.size() && mHandle != 0)
        {
            return MATERIALIZED;
        }

        return ERRORED;
    }

    /**
     * Materialize this CUDAVirtualMemory.
     * Shall be called only when status() == RELEASED.
     *
     * Calls creator.create(), and then configurator.setup() for each configurator in order.
     *
     * Stop at the first thrown exception and propagates it.
     */
    void materialize();

    /**
     * Release this CUDAVirtualMemory.
     * Shall be called only when status() == MATERIALIZED, or materialize() throws.
     * Will be called automatically by destructor if necessary.
     *
     * Calls configurator.teardown() for each configurator that setup() succeed in materialize() in reversed order,
     * and then creator.release().
     *
     * Never stops early upon exception. The last thrown exception will be propagated, and others logged.
     */
    void release();

    CUDAVirtualMemory(CUDAVirtualMemory const&) = delete;
    CUDAVirtualMemory& operator=(CUDAVirtualMemory const&) = delete;

    CUDAVirtualMemory(CUDAVirtualMemory&& other) noexcept
    {
        mCreator = std::move(other.mCreator);
        mConfigurators = std::move(other.mConfigurators);
        mHandle = other.mHandle;
        mState = other.mState;
        new (&other) CUDAVirtualMemory; // Put other into default constructed state
    }

    CUDAVirtualMemory& operator=(CUDAVirtualMemory&& other)
    {
        this->~CUDAVirtualMemory(); // May throw if current virtual memory need release
        new (this) CUDAVirtualMemory(std::move(other));
        return *this;
    }

    CUDAVirtualMemory() noexcept = default;

    CUDAVirtualMemory(CreatorPtr creator, Configurators configurators)
        : mCreator(std::move(creator))
        , mConfigurators(std::move(configurators))
    {
    }

    ~CUDAVirtualMemory()
    {
        // Calling release() is necessary if materialize() succeed or threw an exception.
        // If release() is already called by the user, whether succeed or threw an exception,
        // we shouldn't call release() again.
        if (mHandle != 0 && mState != INVALID_STATE)
        {
            release();
        }
    }

    explicit operator bool() const noexcept
    {
        return mCreator != nullptr;
    }

private:
    constexpr static size_t INVALID_STATE = static_cast<size_t>(-1);
    size_t mState = 0;
    CUmemGenericAllocationHandle mHandle{};
    std::unique_ptr<Creator> mCreator;
    std::vector<std::unique_ptr<Configurator>> mConfigurators;
};

/**
 * LocalCreator creates memory allocation locally through cuMemCreate.
 */
template <bool count = true>
struct LocalCreator : CUDAVirtualMemory::Creator
{
    LocalCreator(CUmemAllocationProp const& prop, size_t size)
        : mProp(prop)
        , mSize(size)
    {
    }

    CUmemGenericAllocationHandle create() override
    {
        CUmemGenericAllocationHandle handle{};
        TLLM_CU_CHECK(cuMemCreate(&handle, mSize, &mProp, 0));
        if constexpr (count)
        {
            MemoryCounters::getInstance().allocate(
                mProp.location.type == CU_MEM_LOCATION_TYPE_DEVICE ? MemoryType::kGPU : MemoryType::kPINNED, mSize);
        }
        return handle;
    }

    void release(CUmemGenericAllocationHandle handle) override
    {
        TLLM_CU_CHECK_FREE_RESOURCE(cuMemRelease(handle));
        if constexpr (count)
        {
            MemoryCounters::getInstance().deallocate(
                mProp.location.type == CU_MEM_LOCATION_TYPE_DEVICE ? MemoryType::kGPU : MemoryType::kPINNED, mSize);
        }
    }

    CUmemAllocationProp mProp{};
    size_t mSize{};
};

/**
 * UnicastConfigurator maps the allocation handle into the specified unicast address range.
 */
struct UnicastConfigurator : CUDAVirtualMemory::Configurator
{
    UnicastConfigurator(CUdeviceptr address, size_t size, CUmemAccessDesc const& desc)
        : mAddress(address)
        , mSize(size)
        , mDesc(desc)
    {
    }

    void setup(CUmemGenericAllocationHandle handle) override
    {
        TLLM_CU_CHECK(cuMemMap(mAddress, mSize, 0, handle, 0));
        TLLM_CU_CHECK(cuMemSetAccess(mAddress, mSize, &mDesc, 1));
    }

    void teardown(CUmemGenericAllocationHandle) override
    {
        TLLM_CU_CHECK_FREE_RESOURCE(cuMemUnmap(mAddress, mSize));
    }

    CUdeviceptr mAddress;
    size_t mSize;
    CUmemAccessDesc mDesc;
};

/**
 * MulticastConfigurator binds the allocation handle to the given multicast object and offset.
 */
struct MulticastConfigurator : CUDAVirtualMemory::Configurator
{
    void setup(CUmemGenericAllocationHandle handle) override
    {
        TLLM_CU_CHECK(cuMulticastBindMem(mMulticast, 0, handle, mBindOffset, mSize, 0));
    }

    void teardown(CUmemGenericAllocationHandle) override
    {
        TLLM_CU_CHECK_FREE_RESOURCE(cuMulticastUnbind(mMulticast, mDevice, 0, mSize));
    }

    CUmemGenericAllocationHandle mMulticast;
    size_t mBindOffset;
    CUdevice mDevice;
    size_t mSize;
};

/**
 * MemsetConfigurator fills the memory with given value.
 */
struct MemsetConfigurator : CUDAVirtualMemory::Configurator
{
    MemsetConfigurator(CUdeviceptr address, size_t size, uint8_t value, CUstream stream)
        : mAddress(address)
        , mSize(size)
        , mStream(stream)
        , mValue(value)
    {
    }

    void setup(CUmemGenericAllocationHandle) override
    {
        if (!mFirstTime)
        {
            TLLM_CU_CHECK(cuMemsetD8Async(mAddress, mValue, mSize, mStream));
        }
    }

    void teardown(CUmemGenericAllocationHandle) noexcept override
    {
        mFirstTime = false;
    }

    CUdeviceptr mAddress;
    size_t mSize;
    CUstream mStream{};
    uint8_t mValue;
    bool mFirstTime = true;
};

/**
 * BackedConfigurator backup the content of the allocation when teardown,
 * and restore the backup on the following setup.
 */

struct BackedConfigurator : CUDAVirtualMemory::Configurator
{
    BackedConfigurator(CUdeviceptr address, size_t size, MemoryType backType, CUstream stream, bool ondemand = false)
        : mAddress(address)
        , mSize(size)
        , mBackType(backType)
        , mStream(stream)
        , mOndemand(ondemand)
    {
    }

    void setup(CUmemGenericAllocationHandle) override;
    void teardown(CUmemGenericAllocationHandle) override;

    CUdeviceptr mAddress;
    size_t mSize;
    MemoryType mBackType;
    CUstream mStream;
    bool mOndemand;

    IBuffer::UniquePtr mBackedStorage;
    CudaEvent mEvent;
};

class CudaVirtualMemoryManager
{
public:
    /**
     * Add memory to be managed by this manager.
     * @param handle  Unique handle provided to reference this memory in `remove`.
     * @param mark    Mark the memory, so this memory can be targeted in `releaseWithMark` and `materializeWithMark`.
     * @param memory  The CUDAVirtualMemory object.
     *
     * The memory and internal state will remain valid if any exception is thrown.
     */
    void add(uintptr_t handle, std::string mark, CUDAVirtualMemory&& memory);

    /**
     * Creates and adds memory to be managed by this manager. The created memory is automatically materialized.
     * @param handle         Unique handle provided to reference this memory in `remove`.
     * @param mark           Mark the memory, so this memory can be targeted in `releaseWithMark` and
     * `materializeWithMark`.
     * @param creator        The creator for the memory.
     * @param configurators  The configurators for the memory.
     *
     * The internal state will remain valid if any exception is thrown.
     */
    void add(uintptr_t handle, std::string mark, CUDAVirtualMemory::CreatorPtr creator,
        CUDAVirtualMemory::Configurators configurators);

    template <typename... Configurators>
    void add(
        uintptr_t handle, std::string mark, CUDAVirtualMemory::CreatorPtr creator, Configurators&&... configurators)
    {
        add(handle, mark, std::move(creator), {std::forward<Configurators>(configurators)...});
    }

    /**
     * Remove the memory from the manager.
     * @param handle The handle provided to `add`.
     * @return The CUDAVirtualMemory object. If the handle is unknown, an empty CUDAVirtualMemory will be returned.
     */
    CUDAVirtualMemory remove(uintptr_t handle) noexcept;

    /**
     * Call release for CUDAVirtualMemory objects with a given mark.
     * @param mark the mark to select target memories.
     * @return Number of objects selected.
     *
     * This function will always call `CUDAVirtualMemory::release` on all selected objects.
     * The last exception thrown by `CUDAVirtualMemory::release` will be rethrown, and others will be logged.
     *
     * If any CUDAVirtualMemory threw an exception during `release`, it will be removed from the manager.
     * Call `retrieveBadHandles` to retrieve handles of all CUDAVirtualMemory that got removed due to exception.
     */
    size_t releaseWithMark(std::string const& mark);

    /**
     * Call materialize for CUDAVirtualMemory objects with a given mark.
     * @param mark the mark to select target memories.
     * @return Number of objects selected.
     *
     * This function will stop at the first `CUDAVirtualMemory::materialize` that throws exception,
     * and attempt to roll back previous successful `materialize` by calling `release`.
     * The exception thrown by `CUDAVirtualMemory::materialize` will be rethrown,
     * and any exception thrown by `release` will be logged.
     *
     * If any CUDAVirtualMemory threw an exception during `materialize` or `release`, it will be removed from the
     * manager. Successfully roll backed CUDAVirtualMemory will not be removed.
     * Call `retrieveBadHandles` to retrieve handles of all CUDAVirtualMemory that got removed due to exception.
     */
    size_t materializeWithMark(std::string const& mark);

    /**
     * Retrieve handles of all CUDAVirtualMemory that got removed due to exception.
     * The returned list may not include all removed CUDAVirtualMemory handles if OOM happened.
     * @return The handle list.
     */
    std::vector<uintptr_t> retrieveBadHandles() noexcept;

private:
    CUDAVirtualMemory unsafeRemove(uintptr_t handle) noexcept;
    void addBadHandle(uintptr_t handle) noexcept;

    struct Entry;
    using PointerMemoryMap = std::unordered_map<uintptr_t, Entry>;
    using MarkEntryMap = std::unordered_multimap<std::string, PointerMemoryMap::iterator>;

    struct Entry
    {
        CUDAVirtualMemory mMemory;
        MarkEntryMap::iterator mEntryIt;
    };

    std::mutex mMutex;
    PointerMemoryMap mMemories;
    MarkEntryMap mEntries;
    std::vector<uintptr_t> mBadHandles;

    friend VirtualMemoryManagerTest;
};

// Update to MemoryCounters is done in Creator to more precisely reflect the memory usage.
class CudaVirtualAddressAllocator
{
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using Pointer = void*;

public:
    enum BackedMode
    {
        NONE,   // The memory is not backed. Upon rematerialize, memory has uninitialized content.
        MEMSET, // The memory is memset to zero upon rematerialize.
        CPU,    // The memory is backed by normal CPU memory. The content is restored upon rematerialize.
        PINNED  // The memory is backed by pinned CPU memory. The content is restored upon rematerialize.
    };

    class Configuration
    {
        CudaVirtualMemoryManager* mManager;
        std::string mMark;
        CudaStreamPtr mBackStream;
        std::size_t mPageSize;
        BackedMode mMode;

        friend class CudaVirtualAddressAllocator;

    public:
        /**
         * CudaVirtualAddressAllocator::Configuration
         * @param manager    Manager used to track and manage virtual memories
         * @param mark       The mark for allocated memories
         * @param mode       Backed storage mode
         * @param backStream The CUDA stream used for restoring memory content
         *                   Note: Virtual Address Allocation is not async. The stream is not used in allocation.
         */
        Configuration(
            CudaVirtualMemoryManager* manager, std::string const& mark, BackedMode mode, CudaStreamPtr backStream)
            : mManager(manager)
            , mMark(mark)
            , mBackStream(std::move(backStream))
            , mPageSize(getpagesize())
            , mMode(mode)
        {
        }
    };

    explicit CudaVirtualAddressAllocator(std::shared_ptr<Configuration> config)
        : mConfig(std::move(config))
    {
    }

    explicit operator bool() const noexcept
    {
        return mConfig != nullptr;
    }

    void allocate(Pointer* ptr, std::size_t n, int device) const;
    void deallocate(Pointer ptr, std::size_t n) const;

private:
    std::shared_ptr<Configuration> mConfig;
};

} // namespace tensorrt_llm::runtime

// Experimental: Global instance
namespace tensorrt_llm::runtime
{

CudaVirtualMemoryManager* getVirtualMemoryManager();

// TODO(ytong): torch does not track allocator. This is temporary WAR.
void cudaVirtualAddressAllocatorDeallocate(void* ptr, std::size_t n);

CudaVirtualAddressAllocator const& getVirtualAddressAllocator();
void pushVirtualAddressAllocator(
    std::string const& mark, CudaVirtualAddressAllocator::BackedMode mode, std::shared_ptr<CudaStream> backStream);
void popVirtualAddressAllocator();

} // namespace tensorrt_llm::runtime
