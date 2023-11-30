/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstdlib>
#include <limits>
#include <memory>
#include <type_traits>

namespace tensorrt_llm::runtime
{

// CRTP base class
template <typename TDerived, MemoryType memoryType>
class BaseAllocator
{
public:
    using ValueType = void;
    using PointerType = ValueType*;
    using SizeType = std::size_t;

    PointerType allocate(SizeType n)
    {
        PointerType ptr{};
        static_cast<TDerived*>(this)->allocateImpl(&ptr, n);
        MemoryCounters::getInstance().allocate<memoryType>(n);
        return ptr;
    }

    void deallocate(PointerType ptr, SizeType n)
    {
        if (ptr)
        {
            static_cast<TDerived*>(this)->deallocateImpl(ptr, n);
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
        PointerType ptr, [[gnu::unused]] SizeType n)
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

    void deallocateImpl(PointerType ptr, [[gnu::unused]] SizeType n)
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
    PinnedAllocator() noexcept = default;

protected:
    void allocateImpl(PointerType* ptr, SizeType n) // NOLINT(readability-convert-member-functions-to-static)
    {
        TLLM_CUDA_CHECK(::cudaHostAlloc(ptr, n, cudaHostAllocDefault));
    }

    void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
        PointerType ptr, [[gnu::unused]] SizeType n)
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
        PointerType ptr, [[gnu::unused]] SizeType n)
    {
        std::free(ptr);
    }
};

template <MemoryType memoryType>
class BorrowingAllocator : public BaseAllocator<BorrowingAllocator<memoryType>, memoryType>
{
    friend class BaseAllocator<BorrowingAllocator<memoryType>, memoryType>;

public:
    using Base = BaseAllocator<BorrowingAllocator<memoryType>, memoryType>;
    using typename Base::PointerType;
    using typename Base::SizeType;

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
        [[gnu::unused]] PointerType ptr, [[gnu::unused]] SizeType n)
    {
    }

private:
    typename Base::PointerType mPtr;
    typename Base::SizeType mCapacity;
};

using CpuBorrowingAllocator = BorrowingAllocator<MemoryType::kCPU>;
using GpuBorrowingAllocator = BorrowingAllocator<MemoryType::kGPU>;
using PinnedBorrowingAllocator = BorrowingAllocator<MemoryType::kPINNED>;

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
        catch (std::exception& e)
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

} // namespace tensorrt_llm::runtime
