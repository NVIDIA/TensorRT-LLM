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
#include "torchAllocator.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

using namespace tensorrt_llm::thop;
using namespace tensorrt_llm::common;

ReallocType TorchAllocator::reallocType(void const* ptr, size_t size) const

{
    TLLM_CHECK(contains(ptr));
    size_t currentSize = 1;
    at::Tensor const& tensor{mPointerMapping.at(ptr)};
    for (int i = 0; i < tensor.dim(); i++)
    {
        currentSize *= tensor.size(i);
    }
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

void* TorchAllocator::malloc(size_t size, bool const setZero)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    auto const bufSize = static_cast<int64_t>(size);
    torch::Tensor buf = torch::empty({bufSize}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
    void* ptr{buf.data_ptr()};
    if (setZero)
    {
        memSet(ptr, 0, size);
    }
    TLLM_LOG_DEBUG("malloc buffer %p with size %ld", ptr, size);
    mPointerMapping.insert({ptr, buf});
    return ptr;
}

void TorchAllocator::free(void** ptr)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    mPointerMapping.erase(*ptr);
    *ptr = nullptr;
}

void TorchAllocator::memSet(void* ptr, int const val, size_t const size)
{
    check_cuda_error(cudaMemsetAsync(ptr, val, size, mStream));
}
