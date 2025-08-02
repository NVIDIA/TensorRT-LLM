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

#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/memoryCounters.h"

#include <cuda.h>
#include <map>
#include <mutex>
#include <unistd.h>
#include <utility>

class VirtualMemoryManagerTest;

namespace tensorrt_llm::runtime
{

/**
 * CUDAVirtualMemoryChunk is a handle to a piece of CUDA memory allocation,
 * providing the ability to release and rematerialize the allocation.
 */
class CUDAVirtualMemoryChunk
{
public:
    /**
     * CUDAVirtualMemoryChunk::Creator is the interface to obtain a CUmemGenericAllocationHandle,
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
        // release() will be called with destructing=true when the CUDAVirtualMemoryChunk
        // is being destructed.
        virtual CUmemGenericAllocationHandle create() = 0;
        virtual void release(CUmemGenericAllocationHandle handle, bool destructing) = 0;
    };

    using CreatorPtr = std::unique_ptr<Creator>;

    /**
     * CUDAVirtualMemoryChunk::Configurator is the interface to configure a CUmemGenericAllocationHandle:
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
        // teardown() will be called with destructing=true when the CUDAVirtualMemoryChunk
        // is being destructed.
        virtual void setup(CUmemGenericAllocationHandle handle) = 0;
        virtual void teardown(CUmemGenericAllocationHandle handle, bool destructing) = 0;
    };

    using ConfiguratorPtr = std::unique_ptr<Configurator>;
    using Configurators = std::vector<ConfiguratorPtr>;

    enum Status
    {
        INVALID,      // This is a default constructed invalid CUDAVirtualMemoryChunk.
        RELEASED,     // The memory represented by this CUDAVirtualMemoryChunk is not allocated.
        MATERIALIZED, // The memory represented by this CUDAVirtualMemoryChunk is allocated.
        ERRORED,      // Error happened during materialize() or release().
                      // This CUDAVirtualMemoryChunk cannot be used anymore.
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
     * Materialize this CUDAVirtualMemoryChunk.
     * Shall be called only when status() == RELEASED.
     *
     * Calls creator.create(), and then configurator.setup() for each configurator in order.
     *
     * Stop at the first thrown exception and propagates it.
     */
    void materialize();

    /**
     * Release this CUDAVirtualMemoryChunk.
     * Shall be called only when status() == MATERIALIZED, or materialize() throws.
     * Will be called automatically by destructor if necessary.
     *
     * Calls configurator.teardown() for each configurator that setup() succeed in materialize() in reversed order,
     * and then creator.release().
     *
     * Never stops early upon exception. The last thrown exception will be propagated, and others logged.
     */
    void release()
    {
        _release(false);
    }

    CUDAVirtualMemoryChunk(CUDAVirtualMemoryChunk const&) = delete;
    CUDAVirtualMemoryChunk& operator=(CUDAVirtualMemoryChunk const&) = delete;

    CUDAVirtualMemoryChunk(CUDAVirtualMemoryChunk&& other) noexcept
    {
        mCreator = std::move(other.mCreator);
        mConfigurators = std::move(other.mConfigurators);
        mHandle = other.mHandle;
        mState = other.mState;
        new (&other) CUDAVirtualMemoryChunk; // Put other into default constructed state
    }

    CUDAVirtualMemoryChunk& operator=(CUDAVirtualMemoryChunk&& other)
    {
        this->~CUDAVirtualMemoryChunk(); // May throw if current virtual memory need release
        new (this) CUDAVirtualMemoryChunk(std::move(other));
        return *this;
    }

    CUDAVirtualMemoryChunk() noexcept = default;

    CUDAVirtualMemoryChunk(CreatorPtr&& creator, Configurators&& configurators)
        : mCreator(std::move(creator))
        , mConfigurators(std::move(configurators))
    {
    }

    virtual ~CUDAVirtualMemoryChunk()
    {
        // Calling release() is necessary if materialize() succeed or threw an exception.
        // If release() is already called by the user, whether succeed or threw an exception,
        // we shouldn't call release() again.
        if (mHandle != 0 && mState != INVALID_STATE)
        {
            _release(true);
        }
    }

    /**
     * Test if this CUDAVirtualMemoryChunk is managing a memory block.
     */
    explicit operator bool() const noexcept
    {
        return mCreator != nullptr;
    }

private:
    void _release(bool destructing);

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
struct LocalCreator : CUDAVirtualMemoryChunk::Creator
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

    void release(CUmemGenericAllocationHandle handle, bool destructing) override
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
struct UnicastConfigurator : CUDAVirtualMemoryChunk::Configurator
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

    void teardown(CUmemGenericAllocationHandle, bool) override
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
struct MulticastConfigurator : CUDAVirtualMemoryChunk::Configurator
{
    void setup(CUmemGenericAllocationHandle handle) override
    {
        TLLM_CU_CHECK(cuMulticastBindMem(mMulticast, 0, handle, mBindOffset, mSize, 0));
    }

    void teardown(CUmemGenericAllocationHandle, bool) override
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
struct MemsetConfigurator : CUDAVirtualMemoryChunk::Configurator
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
        if (mFirstTime)
        {
            mFirstTime = false;
        }
        else
        {
            TLLM_CU_CHECK(cuMemsetD8Async(mAddress, mValue, mSize, mStream));
        }
    }

    void teardown(CUmemGenericAllocationHandle, bool) noexcept override {}

    CUdeviceptr mAddress;
    size_t mSize;
    CUstream mStream{};
    uint8_t mValue;
    bool mFirstTime = true;
};

/**
 * OffloadConfigurator offload the content of the allocation to the backup storage when teardown,
 * and restore the content on the following setup.
 */
struct OffloadConfigurator : CUDAVirtualMemoryChunk::Configurator
{
    OffloadConfigurator(CUdeviceptr address, size_t size, MemoryType backType, CUstream stream, bool ondemand = false)
        : mAddress(address)
        , mSize(size)
        , mBackType(backType)
        , mStream(stream)
        , mOndemand(ondemand)
    {
    }

    void setup(CUmemGenericAllocationHandle handle) override;
    void teardown(CUmemGenericAllocationHandle handle, bool destructing) override;

    CUdeviceptr mAddress;
    size_t mSize;
    MemoryType mBackType;
    CUstream mStream;
    bool mOndemand;

    IBuffer::UniquePtr mBackedStorage;
};

class CudaVirtualMemoryManager
{
public:
    /**
     * Add memory to be managed by this manager.
     * @param handle  Unique handle provided to reference this memory in `remove`.
     * @param tag     Tag the memory, so this memory can be targeted in `releaseWithTag` and `materializeWithTag`.
     * @param memory  The CUDAVirtualMemory object.
     *
     * The memory and internal state will remain valid if any exception is thrown.
     */
    void add(uintptr_t handle, std::string tag, CUDAVirtualMemoryChunk&& memory);

    /**
     * Creates and adds memory to be managed by this manager. The created memory is automatically materialized.
     * @param handle         Unique handle provided to reference this memory in `remove`.
     * @param tag            Tag the memory, so this memory can be targeted in `releaseWithTag` and
     * `materializeWithTag`.
     * @param creator        The creator for the memory.
     * @param configurators  The configurators for the memory.
     *
     * The internal state will remain valid if any exception is thrown.
     */
    void add(uintptr_t handle, std::string tag, CUDAVirtualMemoryChunk::CreatorPtr&& creator,
        CUDAVirtualMemoryChunk::Configurators&& configurators);

    template <typename... Configurators>
    void add(uintptr_t handle, std::string tag, CUDAVirtualMemoryChunk::CreatorPtr&& creator,
        Configurators&&... configurators)
    {
        add(handle, tag, std::move(creator), {std::forward<Configurators>(configurators)...});
    }

    /**
     * Remove the memory from the manager.
     * @param handle The handle provided to `add`.
     * @return The CUDAVirtualMemory object. If the handle is unknown, an empty CUDAVirtualMemory will be returned.
     */
    CUDAVirtualMemoryChunk remove(uintptr_t handle) noexcept;

    /**
     * Call release for CUDAVirtualMemoryChunk objects with a given tag.
     * @param tag the tag to select target memories.
     * @return Number of objects selected.
     *
     * This function will always call `CUDAVirtualMemoryChunk::release` on all selected objects.
     * The last exception thrown by `CUDAVirtualMemoryChunk::release` will be rethrown, and others will be logged.
     *
     * If any CUDAVirtualMemoryChunk threw an exception during `release`, it will be removed from the manager.
     * Call `retrieveBadHandles` to retrieve handles of all CUDAVirtualMemoryChunk that got removed due to exception.
     */
    size_t releaseWithTag(std::string const& tag);

    /**
     * Call materialize for CUDAVirtualMemoryChunk objects with a given tag.
     * @param tag the tag to select target memories.
     * @return Number of objects selected.
     *
     * This function will stop at the first `CUDAVirtualMemoryChunk::materialize` that throws exception,
     * and attempt to roll back previous successful `materialize` by calling `release`.
     * The exception thrown by `CUDAVirtualMemoryChunk::materialize` will be rethrown,
     * and any exception thrown by `release` will be logged.
     *
     * If any CUDAVirtualMemoryChunk threw an exception during `materialize` or `release`, it will be removed from the
     * manager. Successfully roll backed CUDAVirtualMemoryChunk will not be removed.
     * Call `retrieveBadHandles` to retrieve handles of all CUDAVirtualMemoryChunk that got removed due to exception.
     */
    size_t materializeWithTag(std::string const& tag);

    /**
     * Retrieve handles of all CUDAVirtualMemoryChunk that got removed due to exception and reset the list.
     * The returned list may not include all removed CUDAVirtualMemoryChunk handles if OOM happened.
     * This method is only for diagnostic purpose, and should not be called concurrently with other methods.
     * @return The handle list.
     */
    std::vector<uintptr_t> retrieveBadHandles() noexcept;

private:
    CUDAVirtualMemoryChunk unsafeRemove(uintptr_t handle) noexcept;
    void addBadHandle(uintptr_t handle) noexcept;

    struct Entry;
    // Unordered map invalidates iterator upon rehash, so we can only use the ordered map.
    using PointerMemoryMap = std::map<uintptr_t, Entry>;
    using TagEntryMap = std::multimap<std::string, PointerMemoryMap::iterator>;

    struct Entry
    {
        CUDAVirtualMemoryChunk mMemory;
        TagEntryMap::iterator mEntryIt;
    };

    std::mutex mMutex;
    PointerMemoryMap mMemories;
    TagEntryMap mEntries;
    std::vector<uintptr_t> mBadHandles;

    friend VirtualMemoryManagerTest;
};

class CudaVirtualMemoryAllocator
{
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using Pointer = void*;

public:
    enum RestoreMode
    {
        NONE,   // The memory is not backed. Upon rematerialize, memory has uninitialized content.
        MEMSET, // The memory is memset to zero upon rematerialize.
        CPU,    // The memory is backed by normal CPU memory. The content is restored upon rematerialize.
        PINNED  // The memory is backed by pinned CPU memory. The content is restored upon rematerialize.
    };

    class Configuration
    {
        CudaVirtualMemoryManager& mManager;
        std::string mTag;
        CudaStreamPtr mBackStream;
        std::size_t mPageSize;
        RestoreMode mMode;
        bool mBackground{};

        friend class CudaVirtualMemoryAllocator;
        friend void setVirtualMemoryAllocator(
            std::string const& tag, RestoreMode mode, std::shared_ptr<CudaStream> backStream);

    public:
        /**
         * CudaVirtualMemoryAllocator::Configuration
         * @param manager    Manager used to track and manage virtual memories
         * @param tag        The tag for allocated memories
         * @param mode       Backed storage mode
         * @param backStream The CUDA stream used for restoring memory content
         *                   Note: Virtual Address Allocation is not async. The stream is not used in allocation.
         */
        Configuration(CudaVirtualMemoryManager& manager, std::string tag, RestoreMode mode, CudaStreamPtr backStream)
            : mManager(manager)
            , mTag(std::move(tag))
            , mBackStream(std::move(backStream))
            , mPageSize(getpagesize())
            , mMode(mode)
        {
        }

        [[nodiscard]] std::size_t pageAligned(std::size_t n) const noexcept
        {
            return (n + mPageSize - 1) & ~(mPageSize - 1);
        }

        // Background configuration, used to indicate no virtual memory allocator is explicitly configured by the user.
        static Configuration backgroundConfiguration;

    private:
        Configuration(CudaVirtualMemoryManager& manager, std::string tag, RestoreMode mode, CudaStreamPtr backStream,
            bool background)
            : Configuration(manager, std::move(tag), mode, std::move(backStream))
        {
            mBackground = background;
        }
    };

    explicit CudaVirtualMemoryAllocator(std::shared_ptr<Configuration> config)
        : mConfig(std::move(config))
    {
    }

    // Tells if this is the background allocator.
    explicit operator bool() const noexcept
    {
        return !mConfig->mBackground;
    }

    void allocate(Pointer* ptr, std::size_t n, int device) const;
    void deallocate(Pointer ptr, std::size_t n) const;

private:
    std::shared_ptr<Configuration> mConfig;
};

} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::runtime
{
CudaVirtualMemoryManager& getVirtualMemoryManager();
CudaVirtualMemoryAllocator getVirtualMemoryAllocator();
void setVirtualMemoryAllocator(
    std::string const& tag, CudaVirtualMemoryAllocator::RestoreMode mode, std::shared_ptr<CudaStream> backStream);
void clearVirtualMemoryAllocator();

} // namespace tensorrt_llm::runtime
