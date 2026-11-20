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

#include "unbackedSlotMemory.h"
#include "hostAccessibleDeviceAllocator.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/memoryCounters.h"

#include <algorithm>
#include <cuda_runtime_api.h>
#include <exception>
#include <utility>

namespace tensorrt_llm::runtime
{

namespace
{

CUmemAllocationProp slotMemoryProp(int device, bool hostAccessible = false)
{
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.allocFlags.gpuDirectRDMACapable = hostAccessible ? 1 : 0;
    return prop;
}

// Registers the GDRCopy host mapping of a backed chunk with HostAccessibleDeviceAllocator, so EPLB's CPU copies reach
// the chunk by its device address.
struct HostMappingConfigurator : CUDAVirtualMemoryChunk::Configurator
{
    HostMappingConfigurator(CUdeviceptr address, size_t size)
        : mAddress(address)
        , mSize(size)
    {
    }

    void setup(CUmemGenericAllocationHandle) override
    {
        HostAccessibleDeviceAllocator::getInstance().mapDeviceMemory(reinterpret_cast<void*>(mAddress), mSize);
    }

    void teardown(CUmemGenericAllocationHandle, bool) override
    {
        HostAccessibleDeviceAllocator::getInstance().unmapDeviceMemory(reinterpret_cast<void*>(mAddress));
    }

    CUdeviceptr mAddress;
    size_t mSize;
};

CUmemAccessDesc slotMemoryAccess(int device)
{
    CUmemAccessDesc desc{};
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = device;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return desc;
}

// cuMemMap maps whole allocations only, so the one-chunk allocation is mapped at each chunk of the range.
struct RepeatedMappingConfigurator : CUDAVirtualMemoryChunk::Configurator
{
    RepeatedMappingConfigurator(CUdeviceptr address, size_t size, size_t chunkSize, CUmemAccessDesc const& desc)
        : mAddress(address)
        , mSize(size)
        , mChunkSize(chunkSize)
        , mDesc(desc)
    {
    }

    void setup(CUmemGenericAllocationHandle handle) override
    {
        // Unmap on failure, because teardown runs only after a successful setup.
        for (size_t offset = 0; offset < mSize; offset += mChunkSize)
        {
            CUresult const result = cuMemMap(mAddress + offset, mChunkSize, 0, handle, 0);
            if (result != CUDA_SUCCESS)
            {
                if (offset > 0)
                {
                    cuMemUnmap(mAddress, offset);
                }
                TLLM_CU_CHECK(result);
            }
        }
        CUresult const result = cuMemSetAccess(mAddress, mSize, &mDesc, 1);
        if (result != CUDA_SUCCESS)
        {
            cuMemUnmap(mAddress, mSize);
            TLLM_CU_CHECK(result);
        }
    }

    void teardown(CUmemGenericAllocationHandle, bool) override
    {
        TLLM_CU_CHECK_FREE_RESOURCE(cuMemUnmap(mAddress, mSize));
    }

    CUdeviceptr mAddress;
    size_t mSize;
    size_t mChunkSize;
    CUmemAccessDesc mDesc;
};

// Every chunk maps the same allocation, which mOwner keeps alive.
struct SharedCreator : CUDAVirtualMemoryChunk::Creator
{
    SharedCreator(std::shared_ptr<void const> owner, CUmemGenericAllocationHandle handle)
        : mOwner(std::move(owner))
        , mHandle(handle)
    {
    }

    CUmemGenericAllocationHandle create() override
    {
        return mHandle;
    }

    void release(CUmemGenericAllocationHandle, bool) override {}

    std::shared_ptr<void const> mOwner;
    CUmemGenericAllocationHandle mHandle;
};

} // namespace

struct UnbackedSlotMemory::Scratch
{
    Scratch(CUmemAllocationProp const& prop, size_t bytes)
        : size(bytes)
    {
        TLLM_CU_CHECK(cuMemCreate(&handle, size, &prop, 0));
        MemoryCounters::getInstance().allocate(MemoryType::kGPU, size);
    }

    Scratch(Scratch const&) = delete;
    Scratch& operator=(Scratch const&) = delete;

    ~Scratch()
    {
        if (cuMemRelease(handle) != CUDA_SUCCESS)
        {
            TLLM_LOG_ERROR("Releasing the scratch of unbacked slot memory failed");
        }
        MemoryCounters::getInstance().deallocate(MemoryType::kGPU, size);
    }

    CUmemGenericAllocationHandle handle{};
    size_t size;
};

UnbackedSlotMemory& UnbackedSlotMemory::getInstance()
{
    // Never destroyed: CUDA may be torn down before static destructors run.
    static auto* instance = new UnbackedSlotMemory;
    return *instance;
}

CUDAVirtualMemoryChunk UnbackedSlotMemory::mapChunk(
    CUdeviceptr address, size_t size, bool backed, bool hostAccessible) const
{
    CUDAVirtualMemoryChunk::CreatorPtr creator;
    CUDAVirtualMemoryChunk::Configurators configurators;
    if (backed)
    {
        creator = std::make_unique<LocalCreator<>>(slotMemoryProp(mDevice, hostAccessible), size);
        configurators.push_back(std::make_unique<UnicastConfigurator>(address, size, slotMemoryAccess(mDevice)));
        if (hostAccessible)
        {
            configurators.push_back(std::make_unique<HostMappingConfigurator>(address, size));
        }
    }
    else
    {
        std::shared_ptr<Scratch const> scratch = mScratch.lock();
        if (!scratch)
        {
            scratch = std::make_shared<Scratch const>(slotMemoryProp(mDevice), mGranularity);
            mScratch = scratch;
        }
        creator = std::make_unique<SharedCreator>(scratch, scratch->handle);
        configurators.push_back(
            std::make_unique<RepeatedMappingConfigurator>(address, size, mGranularity, slotMemoryAccess(mDevice)));
    }
    CUDAVirtualMemoryChunk chunk(std::move(creator), std::move(configurators));
    // If this throws, the chunk's destructor releases what it made.
    chunk.materialize();
    return chunk;
}

void UnbackedSlotMemory::releaseChunks(Allocation& allocation) noexcept
{
    auto releaseLogged = [](CUDAVirtualMemoryChunk& chunk)
    {
        if (chunk.status() != CUDAVirtualMemoryChunk::MATERIALIZED)
        {
            return;
        }
        try
        {
            chunk.release();
        }
        catch (std::exception const& error)
        {
            TLLM_LOG_ERROR("Releasing unbacked slot memory failed: %s", error.what());
        }
    };
    for (auto& piece : allocation.pieces)
    {
        releaseLogged(piece.chunk);
    }
    for (auto& chunk : allocation.alwaysBacked)
    {
        releaseLogged(chunk);
    }
}

UnbackedSlotMemory::Layout UnbackedSlotMemory::plan(
    size_t slotBytes, int slotCount, std::vector<int> const& unbackedSlots)
{
    TLLM_CHECK_WITH_INFO(slotBytes > 0 && slotCount > 0, "No slots to allocate");
    int device = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    TLLM_CHECK_WITH_INFO(mDevice < 0 || mDevice == device, "Unbacked slot memory serves one device per process");
    if (mDevice < 0)
    {
        auto const prop = slotMemoryProp(device);
        TLLM_CU_CHECK(cuMemGetAllocationGranularity(&mGranularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        mDevice = device;
    }
    size_t const slots = static_cast<size_t>(slotCount);
    Layout layout{std::vector<uint8_t>(slots, 0), common::divUp(slotBytes * slots, mGranularity), {}};
    for (int slot : unbackedSlots)
    {
        TLLM_CHECK_WITH_INFO(slot >= 0 && slot < slotCount, "Unbacked slot %d is not in [0, %d)", slot, slotCount);
        layout.unbacked[slot] = 1;
    }
    layout.slotOf.resize(layout.chunks);
    for (size_t chunk = 0; chunk < layout.chunks; ++chunk)
    {
        size_t const first = firstSlotOfChunk(chunk, slotBytes, slots);
        size_t const last = lastSlotOfChunk(chunk, slotBytes, slots);
        bool inPiece = true;
        for (size_t slot = first; slot <= last && inPiece; ++slot)
        {
            inPiece = slot == slots || layout.unbacked[slot] != 0;
        }
        if (!inPiece)
        {
            layout.slotOf[chunk] = -1;
        }
        else if (first == last)
        {
            layout.slotOf[chunk] = static_cast<int64_t>(first);
        }
        else
        {
            layout.slotOf[chunk] = -2 - static_cast<int64_t>(chunk);
        }
    }
    return layout;
}

size_t UnbackedSlotMemory::firstSlotOfChunk(size_t chunk, size_t slotBytes, size_t slots) const
{
    return std::min(chunk * mGranularity / slotBytes, slots);
}

size_t UnbackedSlotMemory::lastSlotOfChunk(size_t chunk, size_t slotBytes, size_t slots) const
{
    return std::min(((chunk + 1) * mGranularity - 1) / slotBytes, slots);
}

bool UnbackedSlotMemory::savesMemory(size_t slotBytes, int slotCount, std::vector<int> const& unbackedSlots)
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto const slotOf = plan(slotBytes, slotCount, unbackedSlots).slotOf;
    return std::any_of(slotOf.begin(), slotOf.end(), [](int64_t slot) { return slot != -1; });
}

void* UnbackedSlotMemory::allocate(
    size_t slotBytes, int slotCount, std::vector<int> const& unbackedSlots, bool hostAccessible)
{
    TLLM_CHECK_WITH_INFO(!hostAccessible || HostAccessibleDeviceAllocator::getInstance().usesGdrCopy(),
        "Host accessible unbacked slot memory needs host accessible device memory to be GDRCopy mappings");
    std::lock_guard<std::mutex> lock(mMutex);
    auto const layout = plan(slotBytes, slotCount, unbackedSlots);
    auto const& slotOf = layout.slotOf;
    size_t const chunks = layout.chunks;
    size_t const slots = static_cast<size_t>(slotCount);

    CUdeviceptr base = 0;
    TLLM_CU_CHECK(cuMemAddressReserve(&base, chunks * mGranularity, mGranularity, 0, 0));
    Allocation allocation{
        std::shared_ptr<void>(reinterpret_cast<void*>(base),
            [size = chunks * mGranularity](void* ptr) { cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), size); }),
        layout.unbacked, std::vector<uint8_t>(slots, 0), {}, {}, hostAccessible};
    try
    {
        for (size_t chunk = 0; chunk < chunks;)
        {
            size_t run = chunk;
            while (run < chunks && slotOf[run] == slotOf[chunk])
            {
                ++run;
            }
            CUdeviceptr const address = base + chunk * mGranularity;
            size_t const size = (run - chunk) * mGranularity;
            if (slotOf[chunk] == -1)
            {
                allocation.alwaysBacked.push_back(mapChunk(address, size, /*backed=*/true, hostAccessible));
            }
            else
            {
                allocation.pieces.push_back(
                    Piece{address, size, static_cast<int>(firstSlotOfChunk(chunk, slotBytes, slots)),
                        static_cast<int>(lastSlotOfChunk(run - 1, slotBytes, slots)), /*backed=*/false,
                        mapChunk(address, size, /*backed=*/false, hostAccessible)});
            }
            chunk = run;
        }
    }
    catch (...)
    {
        releaseChunks(allocation);
        throw;
    }
    mAllocations.emplace(base, std::move(allocation));
    return reinterpret_cast<void*>(base);
}

void UnbackedSlotMemory::free(void* ptr) noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mAllocations.find(reinterpret_cast<CUdeviceptr>(ptr));
    if (it == mAllocations.end())
    {
        TLLM_LOG_ERROR("%p is not unbacked slot memory", ptr);
        return;
    }
    // Unlike cudaFree, cuMemUnmap does not wait for queued work, and work on any stream may use the
    // range. A tensor deleter may run with another device current.
    int current = -1;
    TLLM_CUDA_CHECK_WARN(cudaGetDevice(&current));
    if (current != mDevice)
    {
        TLLM_CUDA_CHECK_WARN(cudaSetDevice(mDevice));
    }
    // The synchronization fails during a CUDA graph capture that forbids it. Queued or captured work may then still
    // use the range, so the range stays mapped and leaks.
    cudaError_t const syncError = cudaDeviceSynchronize();
    if (syncError == cudaSuccess)
    {
        releaseChunks(it->second);
        mAllocations.erase(it);
    }
    else
    {
        TLLM_LOG_ERROR("Leaking unbacked slot memory %p because synchronizing device %d failed: %s", ptr, mDevice,
            cudaGetErrorString(syncError));
    }
    if (current != mDevice && current >= 0)
    {
        TLLM_CUDA_CHECK_WARN(cudaSetDevice(current));
    }
}

UnbackedSlotMemory::Allocation& UnbackedSlotMemory::findAllocation(void* ptr, std::vector<int> const& slots)
{
    auto it = mAllocations.find(reinterpret_cast<CUdeviceptr>(ptr));
    TLLM_CHECK_WITH_INFO(it != mAllocations.end(), "%p is not unbacked slot memory", ptr);
    auto& allocation = it->second;
    int const slotCount = static_cast<int>(allocation.unbacked.size());
    for (int slot : slots)
    {
        TLLM_CHECK_WITH_INFO(slot >= 0 && slot < slotCount && allocation.unbacked[slot] != 0,
            "Slot %d is not one of the unbacked slots of %d", slot, slotCount);
    }
    return allocation;
}

bool UnbackedSlotMemory::anyBacked(Piece const& piece, std::vector<uint8_t> const& backed)
{
    int const lastSlot = std::min(piece.lastSlot, static_cast<int>(backed.size()) - 1);
    return std::any_of(backed.begin() + piece.firstSlot, backed.begin() + lastSlot + 1,
        [](uint8_t slotBacked) { return slotBacked != 0; });
}

size_t UnbackedSlotMemory::bytesToBack(void* ptr, std::vector<int> const& slots)
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto const& allocation = findAllocation(ptr, slots);
    auto backed = allocation.backed;
    for (int slot : slots)
    {
        backed[slot] = 1;
    }
    size_t bytes = 0;
    for (auto const& piece : allocation.pieces)
    {
        bytes += !piece.backed && anyBacked(piece, backed) ? piece.size : 0;
    }
    return bytes;
}

bool UnbackedSlotMemory::isUnbacked(void const* address)
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto const target = reinterpret_cast<CUdeviceptr>(address);
    auto it = mAllocations.upper_bound(target);
    if (it == mAllocations.begin())
    {
        return false;
    }
    --it;
    return it->second.hostAccessible
        && std::any_of(it->second.pieces.begin(), it->second.pieces.end(),
            [target](Piece const& piece)
            { return !piece.backed && target >= piece.address && target < piece.address + piece.size; });
}

size_t UnbackedSlotMemory::setBacked(std::vector<SlotRange> const& ranges, bool backed)
{
    std::lock_guard<std::mutex> lock(mMutex);
    std::vector<Allocation*> allocations;
    allocations.reserve(ranges.size());
    for (auto const& range : ranges)
    {
        allocations.push_back(&findAllocation(range.ptr, range.slots));
    }
    int device = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    TLLM_CHECK_WITH_INFO(
        device == mDevice, "Unbacked slot memory of device %d changed with device %d current", mDevice, device);
    std::map<Allocation*, std::vector<uint8_t>> requested;
    for (size_t i = 0; i < ranges.size(); ++i)
    {
        auto& slotsBacked = requested.try_emplace(allocations[i], allocations[i]->backed).first->second;
        for (int slot : ranges[i].slots)
        {
            slotsBacked[slot] = static_cast<uint8_t>(backed);
        }
    }
    std::vector<std::pair<Piece*, bool>> changed;
    for (auto& [allocation, slotsBacked] : requested)
    {
        for (auto& piece : allocation->pieces)
        {
            if (anyBacked(piece, slotsBacked) != piece.backed)
            {
                changed.emplace_back(&piece, allocation->hostAccessible);
            }
        }
    }
    size_t bytes = 0;
    if (!changed.empty())
    {
        // Unlike cudaFree, cuMemUnmap does not wait for queued work on any stream that uses the slots.
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        size_t remapped = 0;
        try
        {
            for (; remapped < changed.size(); ++remapped)
            {
                remap(*changed[remapped].first, changed[remapped].second);
                bytes += changed[remapped].first->size;
            }
        }
        catch (...)
        {
            // Restore the pieces remapped before the failure, so the call changes nothing.
            while (remapped-- > 0)
            {
                try
                {
                    remap(*changed[remapped].first, changed[remapped].second);
                }
                catch (std::exception const& error)
                {
                    TLLM_LOG_ERROR("Mapping unbacked slot memory back failed: %s", error.what());
                }
            }
            throw;
        }
    }
    for (auto& [allocation, slotsBacked] : requested)
    {
        allocation->backed = std::move(slotsBacked);
    }
    return bytes;
}

void UnbackedSlotMemory::remap(Piece& piece, bool hostAccessible) const
{
    // An address holds one mapping at a time. If the new mapping fails, the old state is mapped again.
    piece.chunk.release();
    try
    {
        piece.chunk = mapChunk(piece.address, piece.size, !piece.backed, hostAccessible);
    }
    catch (...)
    {
        piece.chunk = mapChunk(piece.address, piece.size, piece.backed, hostAccessible);
        throw;
    }
    piece.backed = !piece.backed;
}

} // namespace tensorrt_llm::runtime
