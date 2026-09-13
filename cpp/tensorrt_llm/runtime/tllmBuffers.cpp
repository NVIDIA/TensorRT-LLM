/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/tllmBuffers.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/tllmDataType.h"

#include <cstdint>
#include <new>

namespace tensorrt_llm::runtime
{
namespace
{
//! \brief Rounds \p size up to a whole number of host pages, as required by ::cudaHostRegister.
std::size_t pageAlignedSize(std::size_t size)
{
    return common::ceilDiv(size, PinnedAllocator::kHostPageSize) * PinnedAllocator::kHostPageSize;
}

//! \brief Whether an allocation of \p n bytes is backed by host memory that is page-locked in chunks.
bool useChunkedPinning(std::size_t n)
{
    auto const chunkSize = PinnedAllocator::getPinChunkSize();
    return chunkSize != 0 && n > chunkSize;
}
} // namespace

void hostRegisterChunked(void* ptr, std::size_t size, std::size_t chunkSize)
{
    TLLM_CHECK_WITH_INFO(chunkSize > 0, "Page-locking chunk size must be positive");
    auto* const base = static_cast<std::uint8_t*>(ptr);
    std::size_t offset{0};
    try
    {
        for (; offset < size; offset += chunkSize)
        {
            TLLM_CUDA_CHECK(
                ::cudaHostRegister(base + offset, std::min(chunkSize, size - offset), cudaHostRegisterDefault));
        }
    }
    catch (...)
    {
        // Leave no partially registered range behind. Warn instead of throwing on the unwind so that the original
        // registration failure is the one that reaches the caller.
        for (std::size_t undone{0}; undone < offset; undone += chunkSize)
        {
            TLLM_CUDA_CHECK_WARN(::cudaHostUnregister(base + undone));
        }
        throw;
    }
}

void hostUnregisterChunked(void* ptr, std::size_t size, std::size_t chunkSize)
{
    TLLM_CHECK_WITH_INFO(chunkSize > 0, "Page-locking chunk size must be positive");
    auto* const base = static_cast<std::uint8_t*>(ptr);
    for (std::size_t offset{0}; offset < size; offset += chunkSize)
    {
        TLLM_CUDA_CHECK_FREE_RESOURCE(::cudaHostUnregister(base + offset));
    }
}

std::size_t PinnedAllocator::getPinChunkSize()
{
    static std::size_t const chunkSize
        = common::getUInt64Env("TRTLLM_HOST_PIN_CHUNK_BYTES").value_or(kDefaultPinChunkSize);
    return chunkSize;
}

PinnedAllocator::PointerType PinnedAllocator::allocateChunkPinned(std::size_t n, std::size_t chunkSize)
{
    auto const lockedBytes = pageAlignedSize(n);
    TLLM_LOG_DEBUG("PinnedAllocator: page-locking %zu B in chunks of %zu B", lockedBytes, chunkSize);

    auto* const base = std::aligned_alloc(kHostPageSize, lockedBytes);
    if (base == nullptr)
    {
        throw std::bad_alloc();
    }
    try
    {
        hostRegisterChunked(base, lockedBytes, chunkSize);
    }
    catch (...)
    {
        std::free(base);
        throw;
    }
    return base;
}

void PinnedAllocator::deallocateChunkPinned(PointerType ptr, std::size_t n, std::size_t chunkSize)
{
    hostUnregisterChunked(ptr, pageAlignedSize(n), chunkSize);
    std::free(ptr);
}

void PinnedAllocator::allocateImpl(PointerType* ptr, std::size_t n)
{
    if (!useChunkedPinning(n))
    {
        TLLM_CUDA_CHECK(::cudaHostAlloc(ptr, n, cudaHostAllocDefault));
        return;
    }
    *ptr = allocateChunkPinned(n, getPinChunkSize());
}

void PinnedAllocator::deallocateImpl(PointerType ptr, std::size_t n)
{
    if (!useChunkedPinning(n))
    {
        TLLM_CUDA_CHECK_FREE_RESOURCE(::cudaFreeHost(ptr));
        return;
    }
    deallocateChunkPinned(ptr, n, getPinChunkSize());
}

template <typename TAllocator>
typename PoolAllocator<TAllocator>::PoolType& PoolAllocator<TAllocator>::getPool()
{
    static PoolType pool;
    return pool;
}

MulticastTensorView::MulticastTensorView(std::weak_ptr<MulticastTensor> const& tensor, ViewType viewType)
    : mTensor(tensor)
    , mViewType(viewType)
    , mDims(mTensor.lock()->getShape())
{
}

MulticastTensorView::MulticastTensorView(MulticastTensorView&& other) noexcept
    : mTensor(std::move(other.mTensor))
    , mViewType(other.mViewType)
    , mDims(mTensor.lock()->getShape())
{
}

MulticastTensorView& MulticastTensorView::operator=(MulticastTensorView&& other) noexcept
{
    if (this != &other)
    {
        // Reset tensor.
        mTensor.reset();
        mTensor.swap(other.mTensor);
        mViewType = other.mViewType;
        mDims = mTensor.lock()->getShape();
    }
    return *this;
}

std::shared_ptr<MulticastBuffer> MulticastTensorView::lock() const
{
    auto sp = mTensor.lock();
    TLLM_CHECK(sp != nullptr);
    return sp;
}

///////////////////////////////////////
// MulticastTensorView ITensor methods
///////////////////////////////////////
tensorrt_llm::Dims const& MulticastTensorView::getShape() const
{
    return mDims;
}

void MulticastTensorView::reshape(tensorrt_llm::Dims const& dims)
{
    auto new_size = nonNegative(volume(dims));
    if (new_size > getCapacity())
    {
        TLLM_THROW("MulticastTensorView::reshape() cannot be larger than origin tensor.");
    }
    mDims = dims;
}

///////////////////////////////////////
// MulticastTensorView IBuffer methods
///////////////////////////////////////
void* MulticastTensorView::_data() const
{
    switch (mViewType)
    {
    case ViewType::kUNICAST: return lock()->data();
    case ViewType::kMULTICAST: return lock()->dataMC();
    case ViewType::kIPC_LIST: return lock()->dataIpcList();
    }
    TLLM_THROW("Invalid mViewType");
    return nullptr;
}

std::size_t MulticastTensorView::getSize() const
{
    return lock()->getSize();
}

std::size_t MulticastTensorView::getCapacity() const
{
    return lock()->getCapacity();
}

tensorrt_llm::DataType MulticastTensorView::getDataType() const
{
    return lock()->getDataType();
}

MemoryType MulticastTensorView::getMemoryType() const
{
    return lock()->getMemoryType();
}

// explicit instantiations
template class PoolAllocator<PinnedAllocator>;
} // namespace tensorrt_llm::runtime
