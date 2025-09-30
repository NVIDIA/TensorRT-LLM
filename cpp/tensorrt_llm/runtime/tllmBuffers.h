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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/cudaMemPool.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/runtime/memoryCounters.h"
#include "tensorrt_llm/runtime/virtualMemory.h"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <list>
#include <memory>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <vector>

namespace tensorrt_llm::runtime
{

// CRTP base class
template <typename TDerived, MemoryType memoryType, bool count = true>
class BaseAllocator
{
public:
    using ValueType = void;
    using PointerType = ValueType*;
    static auto constexpr kMemoryType = memoryType;

    PointerType allocate(std::size_t n)
    {
        PointerType ptr{};
        static_cast<TDerived*>(this)->allocateImpl(&ptr, n);
        if constexpr (count)
        {
            MemoryCounters::getInstance().allocate<memoryType>(n);
        }
        return ptr;
    }

    void deallocate(PointerType ptr, std::size_t n)
    {
        if (ptr)
        {
            static_cast<TDerived*>(this)->deallocateImpl(ptr, n);
            if constexpr (count)
            {
                MemoryCounters::getInstance().deallocate<memoryType>(n);
            }
        }
    }

    [[nodiscard]] MemoryType constexpr getMemoryType() const
    {
        return memoryType;
    }
};

class CudaAllocator : public BaseAllocator<CudaAllocator, MemoryType::kGPU>
{
    friend class BaseAllocator<CudaAllocator, MemoryType::kGPU>;

public:
    CudaAllocator() noexcept = default;

protected:
    void allocateImpl(PointerType* ptr, std::size_t n) // NOLINT(readability-convert-member-functions-to-static)
    {
        TLLM_CUDA_CHECK(::cudaMalloc(ptr, n));
    }

    void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
        PointerType ptr, [[maybe_unused]] std::size_t n)
    {
        TLLM_CUDA_CHECK_FREE_RESOURCE(::cudaFree(ptr));
    }
};

class CudaAllocatorAsync : public BaseAllocator<CudaAllocatorAsync, MemoryType::kGPU>
{
    friend class BaseAllocator<CudaAllocatorAsync, MemoryType::kGPU>;

public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using CudaPoolPtr = std::shared_ptr<CudaMemPool>;

    explicit CudaAllocatorAsync(CudaStreamPtr stream, CudaPoolPtr memPool)
        : mCudaStream(std::move(stream))
        , mMemPool(std::move(memPool))
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(mCudaStream), "Undefined CUDA stream");
        TLLM_CHECK_WITH_INFO(static_cast<bool>(mMemPool), "Undefined CUDA mem pool");
    }

    [[nodiscard]] CudaStreamPtr getCudaStream() const
    {
        return mCudaStream;
    }

protected:
    void allocateImpl(PointerType* ptr, std::size_t n)
    {
        TLLM_CUDA_CHECK(::cudaMallocAsync(ptr, n, mMemPool->getPool(), mCudaStream->get()));
    }

    void deallocateImpl(PointerType ptr, [[maybe_unused]] std::size_t n)
    {
        TLLM_CUDA_CHECK_FREE_RESOURCE(::cudaFreeAsync(ptr, mCudaStream->get()));
    }

private:
    CudaStreamPtr mCudaStream;
    CudaPoolPtr mMemPool;
};

class UVMAllocator : public BaseAllocator<UVMAllocator, MemoryType::kUVM>
{
    friend class BaseAllocator<UVMAllocator, MemoryType::kUVM>;

public:
    using Base = BaseAllocator<UVMAllocator, MemoryType::kUVM>;
    UVMAllocator() noexcept = default;

protected:
    void allocateImpl(PointerType* ptr, std::size_t n) // NOLINT(readability-convert-member-functions-to-static)
    {
        TLLM_CUDA_CHECK(::cudaMallocManaged(ptr, n));
        // TLLM_CUDA_CHECK(::cudaMemAdvise(ptr, n, cudaMemAdviseSetPreferredLocation, 0));
    }

    void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
        PointerType ptr, [[maybe_unused]] std::size_t n)
    {
        TLLM_CUDA_CHECK_FREE_RESOURCE(::cudaFree(ptr));
    }
};

class PinnedAllocator : public BaseAllocator<PinnedAllocator, MemoryType::kPINNED>
{
    friend class BaseAllocator<PinnedAllocator, MemoryType::kPINNED>;

public:
    using Base = BaseAllocator<PinnedAllocator, MemoryType::kPINNED>;
    PinnedAllocator() noexcept = default;

protected:
    void allocateImpl(PointerType* ptr, std::size_t n) // NOLINT(readability-convert-member-functions-to-static)
    {
        TLLM_CUDA_CHECK(::cudaHostAlloc(ptr, n, cudaHostAllocDefault));
    }

    void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
        PointerType ptr, [[maybe_unused]] std::size_t n)
    {
        TLLM_CUDA_CHECK_FREE_RESOURCE(::cudaFreeHost(ptr));
    }
};

class HostAllocator : public BaseAllocator<HostAllocator, MemoryType::kCPU>
{
    friend class BaseAllocator<HostAllocator, MemoryType::kCPU>;

public:
    HostAllocator() noexcept = default;

protected:
    void allocateImpl(PointerType* ptr, std::size_t n) // NOLINT(readability-convert-member-functions-to-static)
    {
        *ptr = std::malloc(n);
        if (*ptr == nullptr)
        {
            throw std::bad_alloc();
        }
    }

    void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
        PointerType ptr, [[maybe_unused]] std::size_t n)
    {
        std::free(ptr);
    }
};

template <MemoryType memoryType>
class BorrowingAllocator : public BaseAllocator<BorrowingAllocator<memoryType>, memoryType, false>
{
    friend class BaseAllocator<BorrowingAllocator<memoryType>, memoryType, false>;

public:
    using Base = BaseAllocator<BorrowingAllocator<memoryType>, memoryType, false>;
    using PointerType = typename Base::PointerType;

    BorrowingAllocator(void* ptr, std::size_t capacity)
        : mPtr(ptr)
        , mCapacity(capacity)
    {
        TLLM_CHECK_WITH_INFO(capacity == std::size_t(0) || static_cast<bool>(mPtr), "Undefined pointer");
    }

protected:
    void allocateImpl(PointerType* ptr, std::size_t n) // NOLINT(readability-convert-member-functions-to-static)
    {
        if (n <= mCapacity)
        {
            *ptr = mPtr;
        }
        else
        {
            throw std::bad_alloc();
        }
    }

    void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
        [[maybe_unused]] PointerType ptr, [[maybe_unused]] std::size_t n)
    {
    }

private:
    PointerType mPtr;
    std::size_t mCapacity;
};

using CpuBorrowingAllocator = BorrowingAllocator<MemoryType::kCPU>;
using GpuBorrowingAllocator = BorrowingAllocator<MemoryType::kGPU>;
using PinnedBorrowingAllocator = BorrowingAllocator<MemoryType::kPINNED>;
using ManagedBorrowingAllocator = BorrowingAllocator<MemoryType::kUVM>;
using PinnedPoolBorrowingAllocator = BorrowingAllocator<MemoryType::kPINNEDPOOL>;

// using UVMBorrowingAllocator = BorrowingAllocator<MemoryType::kUVM>;

/**
 * A memory manager that acts as a memory pool, preallocating a configurable
 * amount of memory. It is able to grow in size and allocate memory chunks as required.
 */
template <typename TAllocator>
class MemoryPool : public BaseAllocator<MemoryPool<TAllocator>, TAllocator::kMemoryType, false>
{
    friend class BaseAllocator<MemoryPool<TAllocator>, TAllocator::kMemoryType, false>;

public:
    using Base = BaseAllocator<MemoryPool<TAllocator>, TAllocator::kMemoryType, false>;
    using PointerType = typename Base::PointerType;

    using Allocator = TAllocator;
    static_assert(std::is_same_v<typename Allocator::PointerType, PointerType>);

    static std::size_t constexpr kInitialChunkSize{std::size_t{1} << 29}; // 512 MB
    static std::size_t constexpr kAlignment{256};

    explicit MemoryPool(std::size_t chunkSize = kInitialChunkSize, Allocator allocator = Allocator{})
        : mChunkSize(chunkSize)
        , mAllocator{allocator}
    {
    }

    ~MemoryPool()
    {
        std::lock_guard<std::mutex> lock(mLock);
        TLLM_LOG_DEBUG("MemoryPool: Deallocating %zu chunks", mAllocatedChunks.size());
        for (auto const& [ptr, size] : mAllocatedChunks)
        {
            TLLM_LOG_DEBUG("MemoryPool: Deallocating %zu B", size);
            try
            {
                mAllocator.deallocate(ptr, size);
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_EXCEPTION(e);
            }
        }
        mAllocatedChunks.clear();
    }

    [[nodiscard]] std::size_t getChunkSize() const
    {
        std::lock_guard<std::mutex> lock(mLock);
        return mChunkSize;
    }

    void setChunkSize(std::size_t chunkSize)
    {
        std::lock_guard<std::mutex> lock(mLock);
        mChunkSize = chunkSize;
    }

    [[nodiscard]] std::size_t getUsedSize() const
    {
        std::lock_guard<std::mutex> lock(mLock);
        return std::accumulate(mMemorySegments.cbegin(), mMemorySegments.cend(), std::size_t{0},
            [](std::size_t sum, auto const& chunk) { return chunk.tag ? sum + chunk.size : sum; });
    }

    [[nodiscard]] std::size_t getReservedSize() const
    {
        std::lock_guard<std::mutex> lock(mLock);
        return std::accumulate(mAllocatedChunks.cbegin(), mAllocatedChunks.cend(), std::size_t{0},
            [](std::size_t sum, auto const& chunk) { return sum + std::get<1>(chunk); });
    }

    class MemorySegment
    {
    public:
        MemorySegment(PointerType basePointer, std::size_t size, std::size_t offset = 0, PointerType tag = nullptr)
            : basePointer{basePointer}
            , size{size}
            , offset{offset}
            , tag{tag}
        {
        }

        PointerType const basePointer;
        std::size_t size;
        std::size_t offset;
        PointerType tag;
    };

    // for debugging purposes only
    std::list<MemorySegment> const& getMemorySegments() const
    {
        std::lock_guard<std::mutex> lock(mLock);
        return mMemorySegments;
    }

    // for debugging purposes only
    void logSegments() const;

protected:
    void allocateImpl(PointerType* ptr, std::size_t requestedSize);

    void deallocateImpl(PointerType tag, std::size_t n);

private:
    std::size_t mChunkSize;
    TAllocator mAllocator;
    std::mutex mutable mLock{};

    std::list<MemorySegment> mMemorySegments = {};
    std::vector<std::tuple<PointerType, std::size_t>> mAllocatedChunks = {};

    void allocateChunk()
    {
        TLLM_LOG_DEBUG("MemoryPool: Allocating %zu B", mChunkSize);
        auto basePointer = mAllocator.allocate(mChunkSize);
        mAllocatedChunks.emplace_back(basePointer, mChunkSize);
        mMemorySegments.push_back(MemorySegment{basePointer, mChunkSize});
    }
};

template <typename TAllocator>
void MemoryPool<TAllocator>::allocateImpl(MemoryPool::PointerType* ptr, std::size_t requestedSize)
{
    std::lock_guard<std::mutex> lock(mLock);

    // Align requested size to kAlignment
    // When requesting 0 B, default to allocating 1 B (from "Effective C++", item 51)
    // See https://stackoverflow.com/questions/2660076/returning-aligned-memory-with-new
    std::size_t const alignedRequest{
        requestedSize == 0 ? kAlignment : common::ceilDiv(requestedSize, kAlignment) * kAlignment};

    TLLM_LOG_DEBUG("MemoryPool: Requested to reserve %zu B (%zu B aligned)", requestedSize, alignedRequest);

    // Finds first free segment providing sufficient space
    auto it = std::find_if(mMemorySegments.begin(), mMemorySegments.end(),
        [alignedRequest](auto const& ms) { return ms.tag == nullptr && ms.size >= alignedRequest; });

    if (it == mMemorySegments.end())
    {
        // There is no space available for this request:
        // Adapt mChunkSize to the aligned requested size in case it doesn't fit,
        // allocate a chunk of mChunkSize and fulfill this request
        TLLM_LOG_DEBUG("MemoryPool: Needs more space to accommodate request of %zu B", requestedSize);
        if (mChunkSize < alignedRequest)
        {
            mChunkSize = alignedRequest;
            TLLM_LOG_DEBUG("MemoryPool: Increasing chunk size to %zu B", mChunkSize);
        }
        allocateChunk();
        it = std::prev(mMemorySegments.end());
    }

    // Start of allocation
    auto const offset = it->offset;
    auto const basePointer = it->basePointer;

    // Update current segment
    it->offset += alignedRequest;
    it->size -= alignedRequest;
    if (it->size == 0)
    {
        it = mMemorySegments.erase(it);
    }

    // Update pointer
    *ptr = static_cast<PointerType>(static_cast<std::uint8_t*>(basePointer) + offset);

    // Insert an occupied segment
    mMemorySegments.insert(it, MemorySegment{basePointer, alignedRequest, offset, *ptr});
}

template <typename TAllocator>
void MemoryPool<TAllocator>::deallocateImpl(PointerType tag, std::size_t n)
{
    std::lock_guard<std::mutex> lock(mLock);
    auto it = std::find_if(mMemorySegments.begin(), mMemorySegments.end(),
        [&tag](MemorySegment const& segment) { return segment.tag == tag; });

    TLLM_CHECK_WITH_INFO(it != mMemorySegments.end(), "MemoryPool free: Requested tag %p could not be found", tag);

    // Free found tag
    it->tag = nullptr;

    if (it->size < n)
    {
        TLLM_LOG_WARNING("MemoryPool: Requested to free %zu B, but only %zu B available", n, it->size);
    }

    // Check if previous segment is free, in which case, join
    if (it != mMemorySegments.begin())
    {
        auto previousIt = std::prev(it);
        if (previousIt->tag == nullptr && previousIt->basePointer == it->basePointer)
        {
            previousIt->size += it->size;
            // Remove current element, and point to previous one
            it = std::prev(mMemorySegments.erase(it));
        }
    }

    // Check if next segment is free, in which case, join
    if (std::next(it) != mMemorySegments.end())
    {
        auto nextIt = std::next(it);
        if (nextIt->tag == nullptr && nextIt->basePointer == it->basePointer)
        {
            it->size += nextIt->size;
            // Remove next tag
            mMemorySegments.erase(nextIt);
        }
    }
}

template <typename TAllocator>
void MemoryPool<TAllocator>::logSegments() const
{
    std::lock_guard<std::mutex> lock(mLock);
    TLLM_LOG_DEBUG("MemoryPool segments:");
    for (auto ms : mMemorySegments)
    {
        TLLM_LOG_DEBUG("* Segment size %zu, tag %p, basePointer %p", ms.size, ms.tag, ms.basePointer);
    }
}

template <typename TAllocator>
class PoolAllocator : public BaseAllocator<PoolAllocator<TAllocator>, TAllocator::kMemoryType, false>
{
    friend class BaseAllocator<PoolAllocator<TAllocator>, TAllocator::kMemoryType, false>;

public:
    using Base = BaseAllocator<PoolAllocator<TAllocator>, TAllocator::kMemoryType, false>;
    using PointerType = typename Base::PointerType;
    using PoolType = MemoryPool<TAllocator>;

    static PoolType& getPool();

protected:
    void allocateImpl(PointerType* ptr, std::size_t n) // NOLINT(readability-convert-member-functions-to-static)
    {
        *ptr = getPool().allocate(n);
    }

    void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
        typename TAllocator::PointerType ptr, std::size_t n)
    {
        getPool().deallocate(ptr, n);
    }
};

using PinnedPoolAllocator = PoolAllocator<PinnedAllocator>;

class CudaVirtualMemoryAllocatorAdaptor
    : public BaseAllocator<CudaVirtualMemoryAllocatorAdaptor, MemoryType::kGPU, /* count */ false>,
      CudaVirtualMemoryAllocator
{
    // Update to MemoryCounters is done in Creator to more precisely reflect the memory usage.
    using Base = BaseAllocator<CudaVirtualMemoryAllocatorAdaptor, MemoryType::kGPU, false>;
    friend Base;

public:
    // No explicit, to allow implicit conversion from CudaVirtualMemoryAllocator
    CudaVirtualMemoryAllocatorAdaptor(CudaVirtualMemoryAllocator const& allocator)
        : CudaVirtualMemoryAllocator(allocator)
    {
    }

    using Base::allocate;
    using Base::deallocate;

protected:
    void allocateImpl(PointerType* ptr, std::size_t n) const
    {
        this->CudaVirtualMemoryAllocator::allocate(ptr, n, tensorrt_llm::common::getDevice());
    }

    void deallocateImpl(PointerType ptr, std::size_t n) const
    {
        this->CudaVirtualMemoryAllocator::deallocate(ptr, n);
    }
};

// Adopted from https://github.com/NVIDIA/TensorRT/blob/release/8.6/samples/common/buffers.h

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameter TAllocator must inherit from BaseAllocator.
//!
template <typename TAllocator>
class GenericBuffer : virtual public IBuffer, TAllocator // Inherit from TAllocator for EBO
{
public:
    using AllocatorType = TAllocator;

    //!
    //! \brief Construct an empty buffer.
    //!
    explicit GenericBuffer(nvinfer1::DataType type, TAllocator allocator = {}) // NOLINT(*-pro-type-member-init)
        : GenericBuffer{0, type, std::move(allocator)}
    {
    }

    //!
    //! \brief Construct a buffer with the specified allocation size in number of elements.
    //!
    explicit GenericBuffer( // NOLINT(*-pro-type-member-init)
        std::size_t size, nvinfer1::DataType type, TAllocator allocator = {})
        : GenericBuffer{size, size, type, std::move(allocator)}
    {
    }

    GenericBuffer(GenericBuffer const& other) = delete;
    GenericBuffer& operator=(GenericBuffer const& buf) = delete;

    GenericBuffer(GenericBuffer&& buf) noexcept
        : TAllocator(static_cast<TAllocator&&>(buf))
        , mSize{buf.mSize}
        , mCapacity{buf.mCapacity}
        , mType{buf.mType}
        , mBuffer{buf.mBuffer}
    {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf) noexcept
    {
        if (this != &buf)
        {
            this->TAllocator::deallocate(mBuffer, toBytes(mCapacity));
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            *static_cast<TAllocator*>(this) = static_cast<TAllocator&&>(buf);
            mBuffer = buf.mBuffer;
            // Reset buf.
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    using IBuffer::data;

    //!
    //! \brief Returns pointer to underlying array.
    //! \details Return nullptr if size == 0 so behavior is consistent with BufferView.
    //!
    void* data() override
    {
        return TLLM_LIKELY(mSize > 0) ? mBuffer : nullptr;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //! \details Return nullptr if size == 0 so behavior is consistent with BufferView.
    //!
    [[nodiscard]] void const* data() const override
    {
        return TLLM_LIKELY(mSize > 0) ? mBuffer : nullptr;
    }

    //!
    //! \brief Returns the size (in number of elements) of the buffer.
    //!
    [[nodiscard]] std::size_t getSize() const override
    {
        return mSize;
    }

    //!
    //! \brief Returns the capacity of the buffer.
    //!
    [[nodiscard]] std::size_t getCapacity() const override
    {
        return mCapacity;
    }

    //!
    //! \brief Returns the type of the buffer.
    //!
    [[nodiscard]] nvinfer1::DataType getDataType() const override
    {
        return mType;
    }

    //!
    //! \brief Returns the memory type of the buffer.
    //!
    [[nodiscard]] MemoryType getMemoryType() const override
    {
        return this->TAllocator::getMemoryType();
    }

    //!
    //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
    //!
    void resize(std::size_t newSize) override
    {
        if (mCapacity < newSize)
        {
            this->TAllocator::deallocate(mBuffer, toBytes(mCapacity));
            mBuffer = this->TAllocator::allocate(toBytes(newSize));
            mCapacity = newSize;
        }
        mSize = newSize;
    }

    //!
    //! \brief Releases the buffer.
    //!
    void release() override
    {
        this->TAllocator::deallocate(mBuffer, toBytes(mCapacity));
        mSize = 0;
        mCapacity = 0;
        mBuffer = nullptr;
    }

    ~GenericBuffer() override
    {
        try
        {
            this->TAllocator::deallocate(mBuffer, toBytes(mCapacity));
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_EXCEPTION(e);
        }
    }

protected:
    explicit GenericBuffer(std::size_t size, std::size_t capacity, nvinfer1::DataType type, TAllocator allocator = {})
        : TAllocator{std::move(allocator)}
        , mSize{size}
        , mCapacity{capacity}
        , mType{type}
        , mBuffer{capacity > 0 ? this->TAllocator::allocate(toBytes(capacity)) : nullptr}
    {
        TLLM_CHECK(size <= capacity);
        TLLM_CHECK(capacity == 0 || size > 0);
    }

private:
    std::size_t mSize{0}, mCapacity{0};
    nvinfer1::DataType mType;
    void* mBuffer;
};

class MulticastBuffer : virtual public IBuffer
{
public:
    explicit MulticastBuffer(nvinfer1::DataType type, std::set<int> const& ranks)
        : mSize(0)
        , mCapacity(0)
        , mType(type)
        , mRanks(ranks)
    {
        TLLM_CHECK(ranks.size() > 1);
    }

    explicit MulticastBuffer(size_t size, nvinfer1::DataType type, std::set<int> const& ranks)
        : mSize(0)
        , mCapacity(0)
        , mType(type)
        , mRanks(ranks)
    {
        TLLM_CHECK(size > 0);
        TLLM_CHECK(ranks.size() > 1);
        resize(size);
    }

    MulticastBuffer(MulticastBuffer& other) = delete;
    MulticastBuffer& operator=(MulticastBuffer const& other) = delete;

    MulticastBuffer(MulticastBuffer&& other) noexcept
        : mSize(other.mSize)
        , mCapacity(other.mCapacity)
        , mType(other.mType)
        , mRanks(other.mRanks)
        , mHandle(other.mHandle)
    {
        other.mSize = 0;
        other.mCapacity = 0;
        other.mHandle = nullptr;
    }

    ~MulticastBuffer() override
    {
        MulticastBuffer::release();
    }

    MulticastBuffer& operator=(MulticastBuffer&& other) noexcept
    {
        if (this != &other)
        {
            // free old memory as we are assigning new memory to it
            release();

            mSize = other.mSize;
            mCapacity = other.mCapacity;
            mType = other.mType;
            mRanks = other.mRanks;
            mHandle = other.mHandle;

            // reset other
            other.mSize = 0;
            other.mCapacity = 0;
            other.mHandle = nullptr;
        }
        return *this;
    }

    // Return list of pointers to each rank
    [[nodiscard]] void* dataIpcList()
    {
        return reinterpret_cast<void*>(mHandle->ipc_uc_ptrs.data());
    }

    [[nodiscard]] void const* dataIpcList() const
    {
        return reinterpret_cast<void const*>(mHandle->ipc_uc_ptrs.data());
    }

    [[nodiscard]] void* dataMC()
    {
        return reinterpret_cast<void*>(mHandle->mc_ptr);
    }

    [[nodiscard]] void const* dataMC() const
    {
        return reinterpret_cast<void const*>(mHandle->mc_ptr);
    }

    //////////////////////////
    // Methods from IBuffer
    //////////////////////////

    using IBuffer::data;

    // Return unicast pointer
    [[nodiscard]] void* data() override
    {
        return reinterpret_cast<void*>(mHandle->uc_ptr);
    }

    // Return unicast pointer
    [[nodiscard]] void const* data() const override
    {
        return reinterpret_cast<void const*>(mHandle->uc_ptr);
    }

    [[nodiscard]] std::size_t getSize() const override
    {
        return mSize;
    }

    [[nodiscard]] std::size_t getCapacity() const override
    {
        return mCapacity;
    }

    [[nodiscard]] nvinfer1::DataType getDataType() const override
    {
        return mType;
    }

    [[nodiscard]] MemoryType getMemoryType() const override
    {
        return MemoryType::kGPU;
    }

    void resize(std::size_t newSize) override
    {
        TLLM_CHECK(newSize > 0);
        if (mCapacity < newSize)
        {
            release();
            mHandle = ipcNvlsAllocate(toBytes(newSize), mRanks);

            TLLM_CHECK(mHandle->size % BufferDataType(mType).getSize() == 0);
            mCapacity = mHandle->size / BufferDataType(mType).getSize();
        }
        mSize = newSize;
    }

    void release() override
    {
        if (mCapacity > 0)
        {
            TLLM_CHECK(mHandle->size > 0);
            ipcNvlsFree(mHandle);
        }
    }

private:
    std::size_t mSize = 0;
    std::size_t mCapacity = 0;
    nvinfer1::DataType mType;
    std::set<int> mRanks;
    IpcNvlsHandle* mHandle;
};

using DeviceBuffer = GenericBuffer<CudaAllocatorAsync>;
using StaticDeviceBuffer = GenericBuffer<CudaAllocator>;
using HostBuffer = GenericBuffer<HostAllocator>;
using PinnedBuffer = GenericBuffer<PinnedAllocator>;
using PinnedPoolBuffer = GenericBuffer<PinnedPoolAllocator>;
using UVMBuffer = GenericBuffer<UVMAllocator>;
using VirtualAddressDeviceBuffer = GenericBuffer<CudaVirtualMemoryAllocatorAdaptor>;

template <typename T>
std::make_unsigned_t<T> nonNegative(T value)
{
    TLLM_CHECK_WITH_INFO(value >= 0, "Value must be non-negative");
    return static_cast<std::make_unsigned_t<T>>(value);
}

template <typename TAllocator>
class GenericTensor : virtual public ITensor, public GenericBuffer<TAllocator>
{
public:
    using Base = GenericBuffer<TAllocator>;

    //!
    //! \brief Construct an empty tensor.
    //!
    explicit GenericTensor(nvinfer1::DataType type, TAllocator allocator = {})
        : Base{type, std::move(allocator)}
    {
        mDims.nbDims = 0;
    }

    //!
    //! \brief Construct a tensor with the specified allocation dimensions.
    //!
    explicit GenericTensor(nvinfer1::Dims const& dims, nvinfer1::DataType type, TAllocator allocator = {})
        : Base{nonNegative(volume(dims)), type, std::move(allocator)}
        , mDims{dims}
    {
    }

    explicit GenericTensor(
        nvinfer1::Dims const& dims, std::size_t capacity, nvinfer1::DataType type, TAllocator allocator = {})
        : Base{nonNegative(volume(dims)), capacity, type, std::move(allocator)}
        , mDims{dims}
    {
    }

    GenericTensor(GenericTensor&& tensor) noexcept
        : Base{std::move(tensor)}
        , mDims{tensor.dims}
    {
        tensor.mDims.nbDims = 0;
    }

    GenericTensor& operator=(GenericTensor&& tensor) noexcept
    {
        if (this != &tensor)
        {
            Base::operator=(std::move(tensor));
            mDims = tensor.dims;
            // Reset tensor.
            tensor.mDims.nbDims = 0;
        }
        return *this;
    }

    [[nodiscard]] nvinfer1::Dims const& getShape() const override
    {
        return mDims;
    }

    void reshape(nvinfer1::Dims const& dims) override
    {
        Base::resize(nonNegative(volume(dims)));
        mDims = dims;
    }

    void resize(std::size_t newSize) override
    {
        ITensor::resize(newSize);
    }

    void release() override
    {
        Base::release();
        mDims.nbDims = 0;
    }

private:
    nvinfer1::Dims mDims{};
};

// Forward declaration
class MulticastTensor;

class MulticastTensorView : virtual public ITensor
{
public:
    enum class ViewType
    {
        kUNICAST,
        kMULTICAST,
        kIPC_LIST
    };

    explicit MulticastTensorView(std::weak_ptr<MulticastTensor> const& tensor, ViewType viewType);

    MulticastTensorView(MulticastTensorView&& other) noexcept;

    [[nodiscard]] MulticastTensorView& operator=(MulticastTensorView&& other) noexcept;

    /////////////////////
    // ITensor methods
    /////////////////////
    [[nodiscard]] nvinfer1::Dims const& getShape() const override;

    void reshape(nvinfer1::Dims const& dims) override;

    /////////////////////
    // IBuffer methods
    /////////////////////

    [[nodiscard]] std::size_t getSize() const override;

    [[nodiscard]] std::size_t getCapacity() const override;

    [[nodiscard]] nvinfer1::DataType getDataType() const override;

    [[nodiscard]] MemoryType getMemoryType() const override;

    using ITensor::data;

    [[nodiscard]] void* data() override
    {
        return _data();
    }

    [[nodiscard]] void const* data() const override
    {
        return _data();
    }

    void resize(std::size_t newSize) override
    {
        TLLM_THROW("Cannot resize() MulticastTensorView");
    }

    void release() override
    {
        TLLM_THROW("Cannot release() MulticastTensorView");
    }

private:
    [[nodiscard]] std::shared_ptr<MulticastBuffer> lock() const;

    [[nodiscard]] void* _data() const;

    std::weak_ptr<MulticastTensor> mTensor;
    ViewType mViewType;
    nvinfer1::Dims mDims{};
};

class MulticastTensor : virtual public ITensor,
                        public MulticastBuffer,
                        public std::enable_shared_from_this<MulticastTensor>
{
public:
    using Base = MulticastBuffer;

    explicit MulticastTensor(nvinfer1::DataType type, std::set<int> const& ranks)
        : Base(type, ranks)
    {
        mDims.nbDims = 0;
    }

    explicit MulticastTensor(nvinfer1::Dims const& dims, nvinfer1::DataType type, std::set<int> const& ranks)
        : Base(nonNegative(volume(dims)), type, ranks)
        , mDims(dims)
    {
    }

    MulticastTensor(MulticastTensor& other) = delete;
    MulticastTensor& operator=(MulticastTensor const& other) = delete;

    MulticastTensor(MulticastTensor&& tensor) noexcept
        : Base(std::move(tensor))
        , mDims(tensor.mDims)
    {
        tensor.mDims.nbDims = 0;
    }

    [[nodiscard]] MulticastTensor& operator=(MulticastTensor&& tensor) noexcept
    {
        if (this != &tensor)
        {
            Base::operator=(std::move(tensor));
            mDims = tensor.mDims;
            // Reset tensor.
            tensor.mDims.nbDims = 0;
        }
        return *this;
    }

    [[nodiscard]] std::shared_ptr<ITensor> getTensorView(MulticastTensorView::ViewType viewType)
    {
        return std::make_shared<MulticastTensorView>(weak_from_this(), viewType);
    }

    /////////////////////
    // ITensor methods
    /////////////////////
    [[nodiscard]] nvinfer1::Dims const& getShape() const override
    {
        return mDims;
    }

    void reshape(nvinfer1::Dims const& dims) override
    {
        Base::resize(nonNegative(volume(dims)));
        mDims = dims;
    }

    void resize(std::size_t newSize) override
    {
        ITensor::resize(newSize);
    }

    void release() override
    {
        Base::release();
        mDims.nbDims = 0;
    }

private:
    nvinfer1::Dims mDims{};
};

using DeviceTensor = GenericTensor<CudaAllocatorAsync>;
using StaticDeviceTensor = GenericTensor<CudaAllocator>;
using HostTensor = GenericTensor<HostAllocator>;
using PinnedTensor = GenericTensor<PinnedAllocator>;
using PinnedPoolTensor = GenericTensor<PinnedPoolAllocator>;
using UVMTensor = GenericTensor<UVMAllocator>;
using VirtualAddressDeviceTensor = GenericTensor<CudaVirtualMemoryAllocatorAdaptor>;

} // namespace tensorrt_llm::runtime
