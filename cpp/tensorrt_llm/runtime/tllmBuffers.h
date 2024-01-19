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
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/memoryCounters.h"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
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
    using SizeType = std::size_t;
    static auto constexpr kMemoryType = memoryType;

    PointerType allocate(SizeType n)
    {
        PointerType ptr{};
        static_cast<TDerived*>(this)->allocateImpl(&ptr, n);
        if constexpr (count)
            MemoryCounters::getInstance().allocate<memoryType>(n);
        return ptr;
    }

    void deallocate(PointerType ptr, SizeType n)
    {
        if (ptr)
        {
            static_cast<TDerived*>(this)->deallocateImpl(ptr, n);
            if constexpr (count)
                MemoryCounters::getInstance().deallocate<memoryType>(n);
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
    void allocateImpl(PointerType* ptr, SizeType n) // NOLINT(readability-convert-member-functions-to-static)
    {
        TLLM_CUDA_CHECK(::cudaMalloc(ptr, n));
    }

    void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
        PointerType ptr, [[maybe_unused]] SizeType n)
    {
        TLLM_CUDA_CHECK(::cudaFree(ptr));
    }
};

class CudaAllocatorAsync : public BaseAllocator<CudaAllocatorAsync, MemoryType::kGPU>
{
    friend class BaseAllocator<CudaAllocatorAsync, MemoryType::kGPU>;

public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;

    explicit CudaAllocatorAsync(CudaStreamPtr stream)
        : mCudaStream(std::move(stream))
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(mCudaStream), "Undefined CUDA stream");
    }

    [[nodiscard]] CudaStreamPtr getCudaStream() const
    {
        return mCudaStream;
    }

protected:
    void allocateImpl(PointerType* ptr, SizeType n)
    {
        TLLM_CUDA_CHECK(::cudaMallocAsync(ptr, n, mCudaStream->get()));
    }

    void deallocateImpl(PointerType ptr, [[maybe_unused]] SizeType n)
    {
        TLLM_CUDA_CHECK(::cudaFreeAsync(ptr, mCudaStream->get()));
    }

private:
    CudaStreamPtr mCudaStream;
};

class PinnedAllocator : public BaseAllocator<PinnedAllocator, MemoryType::kPINNED>
{
    friend class BaseAllocator<PinnedAllocator, MemoryType::kPINNED>;

public:
    using Base = BaseAllocator<PinnedAllocator, MemoryType::kPINNED>;
    PinnedAllocator() noexcept = default;

protected:
    void allocateImpl(PointerType* ptr, SizeType n) // NOLINT(readability-convert-member-functions-to-static)
    {
        TLLM_CUDA_CHECK(::cudaHostAlloc(ptr, n, cudaHostAllocDefault));
    }

    void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
        PointerType ptr, [[maybe_unused]] SizeType n)
    {
        TLLM_CUDA_CHECK(::cudaFreeHost(ptr));
    }
};

class HostAllocator : public BaseAllocator<HostAllocator, MemoryType::kCPU>
{
    friend class BaseAllocator<HostAllocator, MemoryType::kCPU>;

public:
    HostAllocator() noexcept = default;

protected:
    void allocateImpl(PointerType* ptr, SizeType n) // NOLINT(readability-convert-member-functions-to-static)
    {
        *ptr = std::malloc(n);
        if (*ptr == nullptr)
        {
            throw std::bad_alloc();
        }
    }

    void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
        PointerType ptr, [[maybe_unused]] SizeType n)
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
    using SizeType = typename Base::SizeType;

    BorrowingAllocator(void* ptr, SizeType capacity)
        : mPtr(ptr)
        , mCapacity(capacity)
    {
        TLLM_CHECK_WITH_INFO(capacity == 0 || static_cast<bool>(mPtr), "Undefined pointer");
        TLLM_CHECK_WITH_INFO(mCapacity >= 0, "Capacity must be non-negative");
    }

protected:
    void allocateImpl(PointerType* ptr, SizeType n) // NOLINT(readability-convert-member-functions-to-static)
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
        [[maybe_unused]] PointerType ptr, [[maybe_unused]] SizeType n)
    {
    }

private:
    PointerType mPtr;
    SizeType mCapacity;
};

using CpuBorrowingAllocator = BorrowingAllocator<MemoryType::kCPU>;
using GpuBorrowingAllocator = BorrowingAllocator<MemoryType::kGPU>;
using PinnedBorrowingAllocator = BorrowingAllocator<MemoryType::kPINNED>;

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
    using SizeType = typename Base::SizeType;

    using Allocator = TAllocator;
    static_assert(std::is_same_v<typename Allocator::PointerType, PointerType>);
    static_assert(std::is_same_v<typename Allocator::SizeType, SizeType>);

    static SizeType constexpr kInitialChunkSize{SizeType{1} << 30}; // 1 GB
    static SizeType constexpr kChunkResizeFactor{2};
    static SizeType constexpr kAlignment{256};

    explicit MemoryPool(SizeType chunkSize = kInitialChunkSize, Allocator allocator = Allocator{})
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

    [[nodiscard]] SizeType getChunkSize() const
    {
        std::lock_guard<std::mutex> lock(mLock);
        return mChunkSize;
    }

    void setChunkSize(SizeType chunkSize)
    {
        std::lock_guard<std::mutex> lock(mLock);
        mChunkSize = chunkSize;
    }

    [[nodiscard]] SizeType getUsedSize() const
    {
        std::lock_guard<std::mutex> lock(mLock);
        return std::accumulate(mMemorySegments.cbegin(), mMemorySegments.cend(), SizeType{0},
            [](SizeType sum, auto const& chunk) { return chunk.tag ? sum + chunk.size : sum; });
    }

    [[nodiscard]] SizeType getReservedSize() const
    {
        std::lock_guard<std::mutex> lock(mLock);
        return std::accumulate(mAllocatedChunks.cbegin(), mAllocatedChunks.cend(), SizeType{0},
            [](SizeType sum, auto const& chunk) { return sum + std::get<1>(chunk); });
    }

    class MemorySegment
    {
    public:
        MemorySegment(PointerType basePointer, SizeType size, SizeType offset = 0, PointerType tag = nullptr)
            : basePointer{basePointer}
            , size{size}
            , offset{offset}
            , tag{tag}
        {
        }

        PointerType const basePointer;
        SizeType size;
        SizeType offset;
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
    void allocateImpl(PointerType* ptr, SizeType requestedSize);

    void deallocateImpl(PointerType tag, SizeType n);

private:
    SizeType mChunkSize;
    TAllocator mAllocator;
    std::mutex mutable mLock{};

    std::list<MemorySegment> mMemorySegments = {};
    std::vector<std::tuple<PointerType, SizeType>> mAllocatedChunks = {};

    void allocateChunk()
    {
        TLLM_LOG_DEBUG("MemoryPool: Allocating %zu B", mChunkSize);
        auto basePointer = mAllocator.allocate(mChunkSize);
        mAllocatedChunks.emplace_back(basePointer, mChunkSize);
        mMemorySegments.push_back(MemorySegment{basePointer, mChunkSize});
    }
};

template <typename TAllocator>
void MemoryPool<TAllocator>::allocateImpl(MemoryPool::PointerType* ptr, MemoryPool::SizeType requestedSize)
{
    std::lock_guard<std::mutex> lock(mLock);
    // Align requested size to kAlignment
    // When requesting 0 B, default to allocating 1 B (from "Effective C++", item 51)
    // See https://stackoverflow.com/questions/2660076/returning-aligned-memory-with-new
    const std::size_t alignedRequest{
        requestedSize == 0 ? kAlignment : common::ceilDiv(requestedSize, kAlignment) * kAlignment};

    TLLM_LOG_DEBUG("MemoryPool: Requested to reserve %zu B (%zu B aligned)", requestedSize, alignedRequest);

    // Finds first free segment providing sufficient space
    auto it = std::find_if(mMemorySegments.begin(), mMemorySegments.end(),
        [alignedRequest](const auto& ms) { return ms.tag == nullptr && ms.size >= alignedRequest; });

    if (it == mMemorySegments.end())
    {
        // There is no space available for this request
        // If the request is bigger than mChunkSize / chunkResizeFactor, adapt mChunkSize to request *
        // chunkResizeFactor
        // Allocate more space in mChunkSize, and fulfill this request
        TLLM_LOG_DEBUG("MemoryPool: Needs more space to accommodate request of %zu B", requestedSize);
        auto const minChunkSize = alignedRequest * kChunkResizeFactor;
        if (mChunkSize < minChunkSize)
        {
            mChunkSize = minChunkSize;
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
void MemoryPool<TAllocator>::deallocateImpl(PointerType tag, SizeType n)
{
    std::lock_guard<std::mutex> lock(mLock);
    auto it = std::find_if(mMemorySegments.begin(), mMemorySegments.end(),
        [&tag](const MemorySegment& segment) { return segment.tag == tag; });

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
    using SizeType = typename Base::SizeType;
    using PoolType = MemoryPool<TAllocator>;

    static PoolType& getPool()
    {
        static PoolType pool;
        return pool;
    }

protected:
    void allocateImpl(PointerType* ptr, SizeType n) // NOLINT(readability-convert-member-functions-to-static)
    {
        *ptr = getPool().allocate(n);
    }

    void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
        typename TAllocator::PointerType ptr, SizeType n)
    {
        getPool().deallocate(ptr, n);
    }
};

using PinnedPoolAllocator = PoolAllocator<PinnedAllocator>;

// Adopted from https://github.com/NVIDIA/TensorRT/blob/release/8.6/samples/common/buffers.h

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
//!
template <typename TAllocator>
class GenericBuffer : virtual public IBuffer
{
public:
    using AllocatorType = TAllocator;

    //!
    //! \brief Construct an empty buffer.
    //!
    explicit GenericBuffer(nvinfer1::DataType type, TAllocator allocator = {}) // NOLINT(*-pro-type-member-init)
        : GenericBuffer{0, type, std::move(allocator)} {};

    //!
    //! \brief Construct a buffer with the specified allocation size in number of elements.
    //!
    explicit GenericBuffer( // NOLINT(*-pro-type-member-init)
        std::size_t size, nvinfer1::DataType type, TAllocator allocator = {})
        : GenericBuffer{size, size, type, std::move(allocator)} {};

    GenericBuffer(GenericBuffer&& buf) noexcept
        : mSize{buf.mSize}
        , mCapacity{buf.mCapacity}
        , mType{buf.mType}
        , mAllocator{std::move(buf.mAllocator)}
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
            mAllocator.deallocate(mBuffer, toBytes(mCapacity));
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mAllocator = std::move(buf.mAllocator);
            mBuffer = buf.mBuffer;
            // Reset buf.
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    void* data() override
    {
        return TLLM_LIKELY(mSize > 0) ? mBuffer : nullptr;
    }

    //!
    //! \brief Returns pointer to underlying array.
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
        return mAllocator.getMemoryType();
    }

    //!
    //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
    //!
    void resize(std::size_t newSize) override
    {
        if (mCapacity < newSize)
        {
            mAllocator.deallocate(mBuffer, toBytes(mCapacity));
            mBuffer = mAllocator.allocate(toBytes(newSize));
            mCapacity = newSize;
        }
        mSize = newSize;
    }

    //!
    //! \brief Releases the buffer.
    //!
    void release() override
    {
        mAllocator.deallocate(mBuffer, toBytes(mCapacity));
        mSize = 0;
        mCapacity = 0;
        mBuffer = nullptr;
    }

    ~GenericBuffer() override
    {
        try
        {
            mAllocator.deallocate(mBuffer, toBytes(mCapacity));
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_EXCEPTION(e);
        }
    }

protected:
    explicit GenericBuffer(std::size_t size, std::size_t capacity, nvinfer1::DataType type, TAllocator allocator = {})
        : mSize{size}
        , mCapacity{capacity}
        , mType{type}
        , mAllocator{std::move(allocator)}
        , mBuffer{capacity > 0 ? mAllocator.allocate(toBytes(capacity)) : nullptr}
    {
        TLLM_CHECK(size <= capacity);
        TLLM_CHECK(capacity == 0 || size > 0);
    }

private:
    std::size_t mSize{0}, mCapacity{0};
    nvinfer1::DataType mType;
    TAllocator mAllocator;
    void* mBuffer;
};

using DeviceBuffer = GenericBuffer<CudaAllocatorAsync>;
using HostBuffer = GenericBuffer<HostAllocator>;
using PinnedBuffer = GenericBuffer<PinnedAllocator>;
using PinnedPoolBuffer = GenericBuffer<PinnedPoolAllocator>;

template <typename T>
typename std::make_unsigned<T>::type nonNegative(T value)
{
    TLLM_CHECK_WITH_INFO(value >= 0, "Value must be non-negative");
    return static_cast<typename std::make_unsigned<T>::type>(value);
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

using DeviceTensor = GenericTensor<CudaAllocatorAsync>;
using HostTensor = GenericTensor<HostAllocator>;
using PinnedTensor = GenericTensor<PinnedAllocator>;
using PinnedPoolTensor = GenericTensor<PinnedPoolAllocator>;

} // namespace tensorrt_llm::runtime
