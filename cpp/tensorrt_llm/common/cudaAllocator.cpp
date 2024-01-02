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

#include "cudaAllocator.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <utility>

using namespace tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;

CudaAllocator::CudaAllocator(tr::BufferManager bufferManager)
    : mBufferManager(std::move(bufferManager))
{
}

ReallocType CudaAllocator::reallocType(void const* ptr, size_t size) const
{
    TLLM_CHECK(contains(ptr));
    auto const currentSize = mPointerMapping.at(ptr)->getSize();
    TLLM_LOG_DEBUG("current_buffer_size: %d, original buffer: %p, new buffer: %d", currentSize, ptr, size);
    if (currentSize < size)
    {
        return ReallocType::INCREASE;
    }
    else if (currentSize == size)
    {
        return ReallocType::REUSE;
    }
    else
    {
        return ReallocType::DECREASE;
    }
}

void* CudaAllocator::malloc(std::size_t size, bool const setZero)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    auto bufferPtr = mBufferManager.gpu(size);
    if (setZero)
    {
        mBufferManager.setZero(*bufferPtr);
    }
    void* ptr{bufferPtr->data()};
    TLLM_LOG_DEBUG("malloc buffer %p with size %ld", ptr, size);
    mPointerMapping.insert({ptr, std::move(bufferPtr)});
    return ptr;
}

void CudaAllocator::free(void** ptr)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    mPointerMapping.erase(*ptr);
    *ptr = nullptr;
}

void CudaAllocator::memSet(void* ptr, int const val, size_t const size)
{
    check_cuda_error(cudaMemsetAsync(ptr, val, size, mBufferManager.getStream().get()));
}
